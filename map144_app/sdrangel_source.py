# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""SDRAngelSource: IQ stream receiver for SDRangel via UDP Sink channel plugin.

Frequency architecture
----------------------
SDRangel is tuned lo_offset_khz below the calling frequency so the device DC
artifact falls outside the passband.  The UDP Sink channel within SDRangel
applies the inverse offset so its IQ output is centred on the calling frequency:

    Device center  = calling_freq_mhz - lo_offset_khz / 1000
    UDP Sink inputFrequencyOffset = +lo_offset_khz * 1000   (Hz)
    UDP Sink output centre = device_center + lo_offset = calling_freq ✓

MAP144 receives IQ centred on calling_freq and uses center_freq_mhz =
calling_freq_mhz.  The channelizer NCO offset is the normal 1500 Hz only.

This is the only approach that works when lo_offset >= sample_rate/2.  With a
50 kHz offset and 48 kHz IQ bandwidth (±24 kHz), centring on device_center
would place the calling frequency completely outside the passband.

Hardware center (device tuning) = calling_freq_mhz - lo_offset_khz / 1000

Dual-channel (polarization diversity)
--------------------------------------
When a B210 (or similar dual-RX SDR) is configured in SDRangel with two device
sets (set 0 = H stream, set 1 = V stream), MAP144 can receive both streams and
feed the polarization-combination engine.  Each device set needs its own UDP
Sink channel sending on a different UDP port.

IQ frame format
---------------
SDRangel UDP Sink "S16LE": interleaved signed 16-bit little-endian I/Q,
one pair per sample, no header.  Full scale = 32768 counts.  Each UDP
datagram contains a variable number of complete samples.

REST API control
----------------
The SDRangel REST API (default http://localhost:8091) is used to:
  - Query device type and current settings
  - Set device center frequency
  - Start the device
  - Configure the UDP Sink channel (destination, format, sample rate)
  - Find or create SSBDemod channel for WSJT-X use

Auto-start
----------
If the REST API is unreachable at start() time and the host is localhost,
SDRAngel is launched automatically.  start() polls until the REST API
responds or the 20-second timeout expires.

Pipeline contract
-----------------
Output: target_rate complex IQ, ±1.0 scale, timestamps from wall clock
anchored at start() + sample counting (~10 ms absolute accuracy).
"""

import json
import logging
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)


# ── Module-level helpers ──────────────────────────────────────────────────────

# Direct installs checked before PATH (shutil.which may return a snap wrapper).
_SDRANGEL_DIRECT_PATHS_LINUX = [
    '/opt/install/sdrangel/bin/sdrangel',
    '/opt/sdrangel/bin/sdrangel',
    '/usr/local/bin/sdrangel',
    '/usr/bin/sdrangel',
]
# Snap paths used as last resort — confinement requires raw-usb interface.
_SDRANGEL_SNAP_PATHS_LINUX = [
    '/snap/bin/sdrangel',
    '/snap/sdrangel/current/usr/bin/sdrangel',
    '/snap/sdrangelproject/current/usr/bin/sdrangel',
]
_SDRANGEL_FALLBACK_PATHS_WIN = [
    # installer default locations
    r'C:\Program Files\SDRangel\sdrangel.exe',
    r'C:\Program Files (x86)\SDRangel\sdrangel.exe',
    # portable / downloaded zip — common user-chosen locations
    r'C:\SDRangel\sdrangel.exe',
    r'D:\SDRangel\sdrangel.exe',
]


def _find_sdrangel_exe(exe_path: str | None = None) -> str | None:
    """Return the SDRangel executable path, checking PATH and known install locations.

    Search order:
      1. Explicit exe_path (from QSettings / UI field)
      2. PATH via shutil.which('sdrangel')
      3. Hard-coded fallback list (snap, opt, program files, portable)
      4. Windows: LOCALAPPDATA and user Downloads folder scanned for sdrangel.exe
    """
    import os
    # Explicit path takes priority.
    if exe_path:
        return exe_path if os.path.isfile(exe_path) else None

    if sys.platform == 'win32':
        # PATH lookup first on Windows (no snap issue there).
        found = shutil.which('sdrangel')
        if found:
            return found
        candidates = list(_SDRANGEL_FALLBACK_PATHS_WIN)
        local_app = os.environ.get('LOCALAPPDATA', '')
        if local_app:
            candidates.append(os.path.join(local_app, 'Programs', 'SDRangel', 'sdrangel.exe'))
            candidates.append(os.path.join(local_app, 'SDRangel', 'sdrangel.exe'))
        userprofile = os.environ.get('USERPROFILE', '')
        if userprofile:
            downloads = os.path.join(userprofile, 'Downloads')
            try:
                for entry in os.scandir(downloads):
                    if entry.is_dir() and 'sdrangel' in entry.name.lower():
                        candidate = os.path.join(entry.path, 'sdrangel.exe')
                        if os.path.isfile(candidate):
                            return candidate
            except OSError:
                pass
        for p in candidates:
            if os.path.isfile(p):
                return p
    else:
        # On Linux, check direct installs BEFORE shutil.which so a non-snap
        # binary is preferred — shutil.which may resolve to /snap/bin/sdrangel
        # which requires extra interface-connection steps.
        for p in _SDRANGEL_DIRECT_PATHS_LINUX:
            if os.path.isfile(p):
                return p
        # Fall back to PATH (may be snap wrapper).
        found = shutil.which('sdrangel')
        if found:
            return found
        # Last resort: snap paths.
        for p in _SDRANGEL_SNAP_PATHS_LINUX:
            if os.path.isfile(p):
                return p

    return None


_SNAP_REQUIRED_INTERFACES = [
    'raw-usb',          # USB SDR hardware access (USRP, RTL-SDR, etc.)
    'hardware-observe', # enumerate USB/serial devices
]


def _snap_name_from_exe(exe: str) -> str | None:
    """Extract the snap package name from a snap executable path.

    /snap/bin/sdrangel          → 'sdrangel'
    /snap/sdrangel/current/...  → 'sdrangel'
    """
    import os, re
    # /snap/bin/<name>  wrapper script
    if exe.startswith('/snap/bin/'):
        return os.path.basename(exe)
    # /snap/<name>/...  internal binary
    m = re.match(r'^/snap/([^/]+)/', exe)
    return m.group(1) if m else None


def _snap_ensure_interfaces(exe: str) -> None:
    """Connect any required snap interfaces that are not yet connected.

    Checks each interface in _SNAP_REQUIRED_INTERFACES.  If disconnected,
    attempts ``snap connect`` first without root (succeeds for auto-connectable
    interfaces), then with ``pkexec`` (GUI password prompt) for privileged ones.
    Logs a clear warning with the manual command if both fail.
    """
    snap_name = _snap_name_from_exe(exe)
    if snap_name is None:
        return

    for iface in _SNAP_REQUIRED_INTERFACES:
        plug = f"{snap_name}:{iface}"
        # Check current connection status.
        try:
            out = subprocess.check_output(
                ['snap', 'connections', snap_name],
                stderr=subprocess.DEVNULL, text=True,
            )
            # Look for a line where the plug appears and the slot column is non-empty.
            connected = False
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[1] == plug and parts[2] not in ('-', ''):
                    connected = True
                    break
            if connected:
                logger.debug("[sdrangel] snap interface already connected: %s", plug)
                continue
        except Exception:
            pass  # if snap CLI fails, try to connect anyway

        logger.info("[sdrangel] snap interface not connected: %s — attempting connect", plug)

        # Try without elevated privileges first (works for some interfaces).
        try:
            subprocess.check_call(
                ['snap', 'connect', plug],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("[sdrangel] snap connect %s succeeded", plug)
            continue
        except subprocess.CalledProcessError:
            pass

        # Try with pkexec (shows GUI password prompt on Linux desktops).
        try:
            subprocess.check_call(
                ['pkexec', 'snap', 'connect', plug],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("[sdrangel] snap connect %s succeeded via pkexec", plug)
            continue
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        logger.warning(
            "[sdrangel] Could not connect snap interface %s automatically.\n"
            "  Run once to fix:  sudo snap connect %s",
            plug, plug,
        )


def _sdrangel_conf_live_path() -> str | None:
    """Return the path to SDRangel's Qt settings file, or None if not found.

    Checks the snap-confined path first (SDRangel snap uses
    ~/snap/sdrangel/current/.config/...), then the standard XDG path.
    """
    import os
    candidates = [
        os.path.expanduser('~/snap/sdrangel/current/.config/f4exb/SDRangel.conf'),
        os.path.expanduser('~/.config/f4exb/SDRangel.conf'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _sdrangel_conf_restore_path() -> str:
    """Return the path to write the SDRangel conf on restore.

    Prefers the snap-confined path when the snap directory exists; falls back
    to the standard XDG path.
    """
    import os
    snap_dir = os.path.expanduser('~/snap/sdrangel/current/.config/f4exb')
    if os.path.isdir(snap_dir):
        return os.path.join(snap_dir, 'SDRangel.conf')
    return os.path.expanduser('~/.config/f4exb/SDRangel.conf')


def _save_sdrangel_layout() -> None:
    """Copy SDRangel's Qt settings file to MAP144's backup after SDRangel exits.

    Called automatically by _stop_sdrangel_source() so the layout is always
    preserved without the user having to click 'Save SDRangel Layout'.
    """
    import os
    src = _sdrangel_conf_live_path()
    if src is None:
        logger.warning("[sdrangel] Layout auto-save: SDRangel.conf not found")
        return
    dst_dir = os.path.expanduser('~/.config/map144')
    dst = os.path.join(dst_dir, 'sdrangel_layout.conf')
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("[sdrangel] Layout auto-saved: %s → %s", src, dst)
    except Exception as exc:
        logger.warning("[sdrangel] Layout auto-save failed: %s", exc)


def _restore_sdrangel_layout() -> None:
    """Copy MAP144's saved SDRangel layout backup to SDRangel's config path.

    Called just before launching SDRangel so it opens with the user's saved
    window arrangement.  No-op if no backup exists yet.
    """
    import os
    src = os.path.expanduser('~/.config/map144/sdrangel_layout.conf')
    if not os.path.isfile(src):
        return
    dst = _sdrangel_conf_restore_path()
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("[sdrangel] Restored layout: %s → %s", src, dst)
    except Exception as exc:
        logger.warning("[sdrangel] restore layout failed: %s", exc)


_sdrangel_launch_lock = threading.Lock()


def _ensure_sdrangel_running(host: str, rest_port: int,
                              timeout_s: float = 20.0,
                              exe_path: str | None = None) -> bool:
    """Return True if SDRangel REST API is reachable; launch it if not.

    Only attempts to launch when host is localhost.  Returns True as soon as
    the REST API responds — device set creation is handled by auto_configure_b210().
    ``exe_path`` overrides the automatic executable search.

    A module-level lock prevents concurrent H/V source threads from each
    launching a separate SDRangel instance.  If a launch is already in
    progress (or the existing process is alive but REST is still starting),
    the second caller waits for the REST API rather than launching again.
    """
    info_url = f"http://{host}:{rest_port}/sdrangel/devicesets"

    def _api_alive() -> bool:
        try:
            with urllib.request.urlopen(info_url, timeout=2.0):
                return True
        except Exception:
            return False

    if _api_alive():
        return True

    # REST unreachable — try to launch SDRangel.
    if host not in ('localhost', '127.0.0.1'):
        logger.warning("[sdrangel] Remote host %s:%d not reachable, cannot auto-start",
                       host, rest_port)
        return False

    with _sdrangel_launch_lock:
        # Re-check inside the lock: another caller may have just launched it.
        if _api_alive():
            return True

        # If our process is alive but REST isn't answering yet, wait for it
        # rather than launching a second instance.
        global _sdrangel_proc
        if _sdrangel_proc is not None and _sdrangel_proc.poll() is None:
            logger.info("[sdrangel] SDRangel already running (pid %d) — waiting for REST API",
                        _sdrangel_proc.pid)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                time.sleep(1.5)
                if _api_alive():
                    logger.info("[sdrangel] SDRangel REST API ready")
                    return True
            logger.warning("[sdrangel] Timed out (%ds) waiting for existing SDRangel (pid %d)",
                           int(timeout_s), _sdrangel_proc.pid)
            return False

        exe = _find_sdrangel_exe(exe_path)
        if exe is None:
            logger.warning("[sdrangel] sdrangel executable not found; cannot auto-start. "
                           "Set path in SDRangel window Connection settings.")
            return False

        # Snap confinement: ensure required interfaces are connected before launch.
        if sys.platform != 'win32' and '/snap/' in exe:
            _snap_ensure_interfaces(exe)

        # Restore saved layout (window geometry + channel config) before launch.
        _restore_sdrangel_layout()

        try:
            if sys.platform == 'win32':
                _sdrangel_proc = subprocess.Popen(
                    [exe], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                _sdrangel_proc = subprocess.Popen(
                    [exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("[sdrangel] Launched SDRangel (%s), waiting for REST API…", exe)
        except Exception as exc:
            logger.warning("[sdrangel] Failed to launch SDRangel: %s", exc)
            return False

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(1.5)
            if _api_alive():
                logger.info("[sdrangel] SDRangel REST API ready")
                return True

        logger.warning("[sdrangel] Timed out (%ds) waiting for SDRangel REST API", int(timeout_s))
        return False


def create_virtual_audio_sink(name: str, channels: int = 1,
                               rate: int = 48000) -> str | None:
    """Create a PipeWire/PulseAudio null-sink virtual audio device.

    Returns the sink name if successful, None if creation failed or
    the platform is not Linux/macOS.  The sink persists until the
    PulseAudio/PipeWire session ends; subsequent calls with the same
    name are idempotent (the existing sink is reused).

    WSJT-X selects the monitor source ("Monitor of <name>") as its
    sound card input.
    """
    if sys.platform not in ('linux', 'darwin'):
        logger.warning("[sdrangel] Virtual audio sinks only supported on Linux/macOS")
        return None

    # Check if the sink already exists.
    try:
        out = subprocess.check_output(
            ['pactl', 'list', 'short', 'sinks'],
            stderr=subprocess.DEVNULL, text=True,
        )
        for line in out.splitlines():
            if name in line:
                logger.info("[sdrangel] Virtual audio sink '%s' already exists", name)
                return name
    except Exception:
        pass  # pactl not available — try pw-cli below

    # Try pactl (PulseAudio / PipeWire-pulse compat layer).
    try:
        subprocess.check_call(
            ['pactl', 'load-module', 'module-null-sink',
             f'sink_name={name}',
             f'sink_properties=device.description={name}',
             f'channels={channels}',
             f'rate={rate}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("[sdrangel] Created virtual audio sink '%s' via pactl", name)
        return name
    except FileNotFoundError:
        pass  # pactl not installed
    except subprocess.CalledProcessError as exc:
        logger.warning("[sdrangel] pactl load-module failed: %s", exc)

    # Fallback: PipeWire native via pw-cli.
    try:
        subprocess.check_call(
            ['pw-cli', 'create-node', 'adapter',
             (f'{{ factory.name=support.null-audio-sink '
              f'node.name={name} '
              f'media.class=Audio/Sink '
              f'object.linger=true '
              f'audio.channels={channels} '
              f'audio.rate={rate} }}')],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("[sdrangel] Created virtual audio sink '%s' via pw-cli", name)
        return name
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        logger.warning("[sdrangel] pw-cli create-node failed: %s", exc)

    return None


# ── Tracked SDRangel process ─────────────────────────────────────────────────

_sdrangel_proc: 'subprocess.Popen | None' = None


def _terminate_sdrangel(host: str, rest_port: int) -> None:
    """Quit SDRangel gracefully so it flushes window geometry to its config file.

    Preferred path: wmctrl WM_DELETE_WINDOW → triggers Qt closeEvent → geometry
    saved.  REST DELETE (QCoreApplication::quit) skips closeEvent on some builds
    and leaves geometry stale in the config file.
    Falls back to REST DELETE, then SIGTERM, then SIGKILL.
    """
    global _sdrangel_proc

    # Preferred: send WM_DELETE_WINDOW by PID so Qt fires closeEvent and saves
    # mainWindowGeometry.  xdotool --pid is unambiguous — wmctrl -c matches by
    # title substring and would hit "map144 — SDRangel" (the MAP144 panel) first.
    proc = _sdrangel_proc
    logger.info("[sdrangel] _terminate_sdrangel: proc=%s pid=%s",
                proc, getattr(proc, 'pid', None))
    if proc is not None and proc.poll() is None:
        try:
            subprocess.run(
                ['xdotool', 'search', '--pid', str(proc.pid), 'windowclose'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
            logger.info("[sdrangel] xdotool windowclose sent to pid %d", proc.pid)
        except Exception:
            # xdotool not available — fall back to REST DELETE.
            try:
                req = urllib.request.Request(
                    f"http://{host}:{rest_port}/sdrangel",
                    method='DELETE',
                )
                urllib.request.urlopen(req, timeout=3.0)
                logger.info("[sdrangel] REST quit sent")
            except Exception:
                pass

    _sdrangel_proc = None
    if proc is None or proc.poll() is not None:
        return   # already exited

    try:
        proc.wait(timeout=3.0)
        logger.info("[sdrangel] SDRangel exited cleanly")
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.terminate()
        proc.wait(timeout=3.0)
        logger.info("[sdrangel] SDRangel terminated (SIGTERM)")
    except Exception:
        try:
            proc.kill()
            logger.info("[sdrangel] SDRangel killed (SIGKILL)")
        except Exception as exc:
            logger.warning("[sdrangel] Could not kill SDRangel process: %s", exc)


# ── Preset constants ─────────────────────────────────────────────────────────

_PRESET_GROUP = "MAP144"
_PRESET_DESC  = {0: "B210-H", 1: "B210-V"}   # keyed by stream_index


# ── Internal packet type ──────────────────────────────────────────────────────

class _SDRAngelPacket:
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


# ── Main class ────────────────────────────────────────────────────────────────

class SDRAngelSource:
    """IQ stream receiver for SDRangel.

    Usage::

        src = SDRAngelSource(calling_freq_mhz=144.150, lo_offset_khz=50.0)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``center_freq_mhz_actual`` reflects the device's actual tuned frequency
    (calling_freq - lo_offset), not the calling frequency.  The MAP144
    channelizer NCO bridges the gap.
    """

    def __init__(self,
                 host:             str        = 'localhost',
                 rest_port:        int        = 8091,
                 udp_port:         int        = 9999,
                 device_set:       int        = 0,
                 calling_freq_mhz: float      = 50.260,
                 lo_offset_khz:    float      = 50.0,
                 target_rate:      int        = 48000,
                 exe_path:         str | None = None,
                 dev_sample_rate:  int        = 192000,
                 gain_db:          float      = 40.0,
                 stream_index:     int        = 0,
                 rx_scale:         float | None = None):
        self.host             = host
        self.rest_port        = rest_port
        self.udp_port         = udp_port
        self.device_set       = device_set
        self.calling_freq_mhz = calling_freq_mhz
        self.lo_offset_khz    = lo_offset_khz
        self.target_rate      = target_rate
        self.exe_path         = exe_path
        self.dev_sample_rate  = dev_sample_rate
        self.gain_db          = gain_db
        self.stream_index     = stream_index  # 0=H, 1=V for dual-pol B210

        _dev_freq = calling_freq_mhz - lo_offset_khz / 1000.0
        self.center_freq_mhz        = _dev_freq
        self.center_freq_mhz_actual = _dev_freq

        # Ingress scale pushed to SDRangel's UDP Sink gain field via configure_udp_sink().
        # SDRangel's polyphase decimator accumulates signal amplitude proportional
        # to sqrt(decimation_ratio); this gain compensates so the int16 output
        # maps to the ±1.0 convention used by the direct B210 path.
        # If not supplied, auto-compute from the decimation ratio.
        if rx_scale is not None:
            self._ingress_scale = float(rx_scale)
        else:
            _deci = dev_sample_rate / target_rate
            self._ingress_scale = 1.0 / (_deci ** 0.5)

        self.sample_queue    = queue.Queue(maxsize=4000)
        self.device_hw_type  = '—'
        self.recv_count      = 0   # total UDP datagrams received from OS
        self.drop_count      = 0   # packets dropped because queue was full

        self._sock    = None
        self._thread  = None
        self._running = False
        self._abs_samps = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):
        """Launch SDRangel if needed, auto-configure B210, and begin receiving."""
        if _ensure_sdrangel_running(self.host, self.rest_port, exe_path=self.exe_path):
            self.auto_configure_b210()
        else:
            logger.warning("[sdrangel] REST API not reachable; starting recv loop anyway")

        self._abs_samps = 0
        self.recv_count = 0
        self.drop_count = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.settimeout(1.0)
        self._sock.bind(('', self.udp_port))

        self._running   = True
        self._thread    = threading.Thread(target=self._recv_loop, daemon=True,
                                           name=f'sdrangel-recv-{self.device_set}')
        self._thread.start()

        logger.info("[sdrangel] started: device=%s  device_center=%.4f MHz"
                    "  calling=%.4f MHz  UDP port=%d  device_set=%d",
                    self.device_hw_type, self.center_freq_mhz_actual,
                    self.calling_freq_mhz, self.udp_port, self.device_set)

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def retune(self, calling_freq_mhz: float):
        """Retune the SDRangel device for a new calling frequency."""
        self.calling_freq_mhz = calling_freq_mhz
        device_freq_hz = int((calling_freq_mhz - self.lo_offset_khz / 1000.0) * 1e6)
        try:
            self._set_device_freq_hz(device_freq_hz)
            logger.info("[sdrangel] retuned: calling=%.4f MHz  device=%.4f MHz",
                        calling_freq_mhz, self.center_freq_mhz_actual)
        except Exception as exc:
            logger.warning("[sdrangel] retune failed: %s", exc)

    def auto_configure_b210(self) -> bool:
        """Create or update a USRP B210 device set — no user interaction required.

        Idempotent: if the device set already exists with USRP assigned, only the
        center frequency is updated.  Otherwise the full sequence runs:
          1. create device set(s) up to self.device_set index
          2. assign USRP (stream self.stream_index)
          3. PATCH device settings (freq, sample rate, gain)
          4. start device
          5. find/create and configure UDP Sink channel

        Returns True if UDP Sink is ready to stream.
        """
        device_freq_hz = int(
            (self.calling_freq_mhz - self.lo_offset_khz / 1000.0) * 1e6
        )

        # ── 0. Try to load the MAP144 preset (fast path — preserves SDRangel layout) ──
        import os
        _desc = _PRESET_DESC.get(self.stream_index, f"B210-S{self.stream_index}")
        _preset_loaded = False
        if self.find_preset(_PRESET_GROUP, _desc):
            _preset_loaded = self.load_preset(_PRESET_GROUP, _desc)
        else:
            _pf = self._preset_file_path()
            if os.path.isfile(_pf):
                logger.info("[sdrangel] Preset '%s/%s' not in SDRangel — importing from %s",
                            _PRESET_GROUP, _desc, _pf)
                if self.import_preset_from_file(_pf):
                    time.sleep(0.5)
                    _preset_loaded = self.load_preset(_PRESET_GROUP, _desc)

        if _preset_loaded:
            logger.info("[sdrangel] Preset '%s/%s' applied — skipping programmatic B210 setup",
                        _PRESET_GROUP, _desc)
            time.sleep(0.5)   # let SDRangel finish applying the preset
            try:
                # Reconfigure UDP Sink so port/address always match current MAP144 settings.
                self.configure_udp_sink()
                return True
            except Exception as exc:
                logger.warning("[sdrangel] UDP Sink after preset load failed: %s "
                               "— falling through to full setup", exc)

        # ── 1. Ensure device set exists ──
        existing = self.probe_device_sets()
        n_existing = len(existing)
        has_this_set = any(ds['index'] == self.device_set for ds in existing)

        if not has_this_set:
            for _ in range(self.device_set - n_existing + 1):
                try:
                    self._rest_post("/sdrangel/deviceset", {"direction": 0})
                except Exception as exc:
                    logger.error("[sdrangel] create device set failed: %s", exc)
                    return False
            logger.info("[sdrangel] Created device set(s) up to index %d", self.device_set)

        # ── 2. Assign USRP if not already assigned ──
        already_usrp = False
        try:
            s = self._rest_get(
                f"/sdrangel/deviceset/{self.device_set}/device/settings"
            )
            already_usrp = (s.get('deviceHwType', '') == 'USRP')
            if already_usrp:
                self.device_hw_type = 'USRP'
        except Exception:
            pass

        if not already_usrp:
            try:
                self._rest_put(
                    f"/sdrangel/deviceset/{self.device_set}/device",
                    {
                        "hwType":            "USRP",
                        "sequence":          0,
                        "deviceStreamIndex": self.stream_index,
                        "direction":         0,
                    },
                )
                self.device_hw_type = 'USRP'
                logger.info("[sdrangel] Assigned USRP stream %d to device set %d",
                            self.stream_index, self.device_set)
            except Exception as exc:
                logger.warning("[sdrangel] USRP device assign failed: %s", exc)
                # Non-fatal — settings PATCH below may still work if already assigned.

        # ── 3. Configure device settings ──
        # Read-modify-write: get the existing settings dict so we only touch
        # centerFrequency, devSampleRate, and gain.  Sending unknown field names
        # in the hardware-specific sub-dict causes SDRangel to return 400.
        try:
            current = self._rest_get(
                f"/sdrangel/deviceset/{self.device_set}/device/settings"
            )
            hw_settings = None
            for val in current.values():
                if isinstance(val, dict) and 'centerFrequency' in val:
                    hw_settings = val
                    break
            if hw_settings is not None:
                cur_freq = hw_settings.get('centerFrequency', 0)
                cur_rate = hw_settings.get('devSampleRate', 0)
                cur_gain = hw_settings.get('gain', 0)
                freq_ok  = abs(cur_freq - device_freq_hz) < 1000
                rate_ok  = abs(cur_rate - self.dev_sample_rate) < 200
                gain_ok  = abs(cur_gain - self.gain_db) < 1
                logger.info("[sdrangel] skip check ds%d: cur_freq=%d want=%d freq_ok=%s  "
                            "cur_rate=%d want=%d rate_ok=%s  cur_gain=%s want=%s gain_ok=%s",
                            self.device_set, cur_freq, device_freq_hz, freq_ok,
                            cur_rate, self.dev_sample_rate, rate_ok,
                            cur_gain, self.gain_db, gain_ok)
                if freq_ok and rate_ok and gain_ok:
                    logger.info("[sdrangel] device settings already correct — skipping PATCH")
                    self.center_freq_mhz        = cur_freq / 1e6
                    self.center_freq_mhz_actual = cur_freq / 1e6
                else:
                    hw_settings['centerFrequency'] = device_freq_hz
                    if 'devSampleRate' in hw_settings:
                        hw_settings['devSampleRate'] = self.dev_sample_rate
                    if 'gain' in hw_settings:
                        hw_settings['gain'] = int(self.gain_db)
                    self._rest_patch(
                        f"/sdrangel/deviceset/{self.device_set}/device/settings",
                        current,
                    )
                    self.center_freq_mhz        = device_freq_hz / 1e6
                    self.center_freq_mhz_actual = device_freq_hz / 1e6
                    logger.info("[sdrangel] B210 configured: center=%.4f MHz  rate=%d  "
                                "stream=%d  gain=%.0f dB",
                                device_freq_hz / 1e6, self.dev_sample_rate,
                                self.stream_index, self.gain_db)
            else:
                logger.warning("[sdrangel] Could not find centerFrequency in device settings "
                               "— device may not be fully initialized yet")
        except Exception as exc:
            logger.warning("[sdrangel] B210 settings PATCH failed: %s", exc)

        # ── 4. Start device ──
        _device_patched = not (freq_ok and rate_ok and gain_ok) if hw_settings is not None else True
        try:
            self._rest_post(f"/sdrangel/deviceset/{self.device_set}/device/run")
            logger.info("[sdrangel] device set %d started", self.device_set)
        except Exception as exc:
            _detail = str(exc)
            if "already running" in _detail.lower() or "500" in _detail:
                logger.debug("[sdrangel] device set %d already running", self.device_set)
            else:
                logger.warning("[sdrangel] device set %d start failed: %s"
                               " — check device rate (%d Hz) is supported by hardware",
                               self.device_set, exc, self.dev_sample_rate)

        # If the device settings were PATCHed, SDRangel performs an async DSP chain
        # restart that reloads channel configs from its internal conf, overwriting any
        # channel PATCHes that arrive too early.  Wait for the restart to settle.
        if _device_patched:
            logger.info("[sdrangel] waiting for DSP restart to settle after device PATCH")
            time.sleep(3.0)

        # ── 5. Configure UDP Sink ──
        try:
            self.configure_udp_sink()
        except Exception as exc:
            logger.warning("[sdrangel] UDP Sink configure failed: %s", exc)
            return False

        # ── 6. Ensure SSBDemod exists for WSJT-X ──
        # Create a virtual audio sink named for this stream so WSJT-X can find it.
        # Non-fatal: MAP144 IQ ingress works fine without the SSB demodulator.
        _audio_name = "MAP144-WSJT-X" if self.stream_index == 0 else "MAP144-WSJT-X-V"
        try:
            _audio_dev = create_virtual_audio_sink(_audio_name)
            self.ensure_ssb_receiver(audio_device=_audio_dev)
        except Exception as exc:
            logger.debug("[sdrangel] SSBDemod auto-create: %s", exc)

        return True

    def configure_udp_sink(self):
        """Find, create-if-absent, and configure the UDPSink channel.

        Searches device set for an existing UDPSink.  If none found, creates
        one via POST.  Configures destination, format, sample rate, and
        inputFrequencyOffset so the output is centred on calling_freq.
        """
        try:
            devset = self._rest_get(f"/sdrangel/deviceset/{self.device_set}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not get device set {self.device_set} overview: {exc}."
            ) from exc

        ch_idx = None
        for ch in devset.get('channels', []):
            if ch.get('id', '') == 'UDPSink':
                ch_idx = ch.get('index')
                break

        if ch_idx is None:
            # Auto-create UDPSink channel.
            # Channel creation is asynchronous — the POST returns an ack message
            # without an 'index'; poll until the new channel appears.
            logger.info("[sdrangel] No UDPSink in device set %d — creating one",
                        self.device_set)
            try:
                self._rest_post(
                    f"/sdrangel/deviceset/{self.device_set}/channel",
                    {"channelType": "UDPSink", "direction": 0},
                )
                for _ in range(15):
                    time.sleep(0.3)
                    try:
                        ds2 = self._rest_get(f"/sdrangel/deviceset/{self.device_set}")
                        for c in ds2.get("channels", []):
                            if c.get("id") == "UDPSink":
                                ch_idx = c.get("index")
                                break
                    except Exception:
                        pass
                    if ch_idx is not None:
                        break
                if ch_idx is None:
                    raise RuntimeError("UDPSink not found after creation (timed out)")
                logger.info("[sdrangel] Created UDPSink at ch%d", ch_idx)
            except Exception as exc:
                raise RuntimeError(
                    f"No UDPSink channel in device set {self.device_set} and "
                    f"auto-create failed: {exc}.  "
                    "Add a UDP Sink channel manually in SDRangel."
                ) from exc

        # Read-modify-write: GET existing settings, apply desired values, PATCH back.
        # rollupState and spectrumConfig are set explicitly so both R0 and R1 are
        # identical except for udpPort.
        udp_settings: dict = {}
        try:
            existing = self._rest_get(
                f"/sdrangel/deviceset/{self.device_set}/channel/{ch_idx}/settings"
            )
            udp_settings = existing.get("UDPSinkSettings", {})
        except Exception:
            pass

        # Functional signal-path settings.
        udp_settings.update({
            "outputSampleRate":     self.target_rate,
            "rfBandwidth":          40000,
            "sampleFormat":         0,
            "inputFrequencyOffset": int(self.lo_offset_khz * 1000),
            "udpAddress":           "127.0.0.1",
            "udpPort":              self.udp_port,
            "gain":                 float(self._ingress_scale),
            "audioActive":          0,
            "audioStereo":          0,
            "volume":               1,
            "squelchEnabled":       1,
            "squelchDB":            -60,
            "squelchGate":          0,
        })

        # Spectrum display: fixed averaging over 10 frames.
        spec = udp_settings.get("spectrumConfig", {})
        spec.update({"averagingMode": 2, "averagingValue": 10})
        udp_settings["spectrumConfig"] = spec


        payload = {"channelType": "UDPSink", "UDPSinkSettings": udp_settings}
        logger.info("[sdrangel] UDP Sink ds%d ch%d PATCH: port=%d  rfBW=%d  rate=%d"
                    "  offset=%d  gain=%.4f  avgMode=%d  avgVal=%d  rollup=%s",
                    self.device_set, ch_idx,
                    udp_settings['udpPort'], udp_settings['rfBandwidth'],
                    udp_settings['outputSampleRate'],
                    udp_settings['inputFrequencyOffset'], udp_settings['gain'],
                    udp_settings.get('spectrumConfig', {}).get('averagingMode', -1),
                    udp_settings.get('spectrumConfig', {}).get('averagingValue', -1),
                    [c.get('isHidden') for c in
                     udp_settings.get('rollupState', {}).get('childrenStates', [])])
        resp = self._rest_patch(
            f"/sdrangel/deviceset/{self.device_set}/channel/{ch_idx}/settings",
            payload,
        )
        resp_s = resp.get("UDPSinkSettings", {}) if resp else {}
        logger.info("[sdrangel] UDP Sink ds%d ch%d response: port=%s  rfBW=%s  rate=%s"
                    "  avgMode=%s  avgVal=%s  rollup=%s",
                    self.device_set, ch_idx,
                    resp_s.get('udpPort'), resp_s.get('rfBandwidth'),
                    resp_s.get('outputSampleRate'),
                    resp_s.get('spectrumConfig', {}).get('averagingMode'),
                    resp_s.get('spectrumConfig', {}).get('averagingValue'),
                    [c.get('isHidden') for c in
                     resp_s.get('rollupState', {}).get('childrenStates', [])])

    def ensure_ssb_receiver(self, audio_device: str | None = None) -> int | None:
        """Find or create an SSBDemod channel tuned to the calling frequency.

        ``audio_device`` is the name of the audio output device (e.g. a
        PipeWire/PulseAudio null sink created by create_virtual_audio_sink).
        Pass None to keep whatever audio device is already configured.

        Returns the channel index, or None on failure.
        """
        try:
            devset = self._rest_get(f"/sdrangel/deviceset/{self.device_set}")
        except Exception as exc:
            logger.warning("[sdrangel] ensure_ssb_receiver: device set query failed: %s", exc)
            return None

        ch_idx = None
        for ch in devset.get('channels', []):
            if ch.get('id', '') == 'SSBDemod':
                ch_idx = ch.get('index')
                break

        if ch_idx is None:
            # Create an SSBDemod channel (async — poll until it appears).
            logger.info("[sdrangel] No SSBDemod in device set %d — creating one",
                        self.device_set)
            try:
                self._rest_post(
                    f"/sdrangel/deviceset/{self.device_set}/channel",
                    {"channelType": "SSBDemod", "direction": 0},
                )
                for _ in range(15):
                    time.sleep(0.3)
                    try:
                        ds2 = self._rest_get(f"/sdrangel/deviceset/{self.device_set}")
                        for c in ds2.get("channels", []):
                            if c.get("id") == "SSBDemod":
                                ch_idx = c.get("index")
                                break
                    except Exception:
                        pass
                    if ch_idx is not None:
                        break
                if ch_idx is None:
                    raise RuntimeError("SSBDemod not found after creation (timed out)")
                logger.info("[sdrangel] Created SSBDemod at ch%d", ch_idx)
            except Exception as exc:
                logger.warning("[sdrangel] SSBDemod create failed: %s", exc)
                return None

        # Read-modify-write: GET existing settings, apply desired values, PATCH back.
        ssb_settings: dict = {}
        try:
            existing = self._rest_get(
                f"/sdrangel/deviceset/{self.device_set}/channel/{ch_idx}/settings"
            )
            ssb_settings = existing.get("SSBDemodSettings", {})
        except Exception:
            pass

        # Signal-path and filter settings.
        ssb_settings.update({
            "inputFrequencyOffset": int(self.lo_offset_khz * 1000),
            "rfBandwidth":          2700,
            "lowCutoff":            300,
            "dsb":                  0,
            "agc":                  1,
            "audioMute":            0,
            "audioBinaural":        0,
            "audioFlipChannels":    0,
            "volume":               1,
            # spanLog2=4: waterfall span = device_rate / 2^4 = 192000/16 = 12 kHz
            # displayed range centres on offset; at 48k channel rate: 48000/16 = 3 kHz
            "spanLog2":             4,
        })
        if audio_device is not None:
            ssb_settings["audioDeviceName"] = audio_device


        payload = {
            "channelType": "SSBDemod",
            "SSBDemodSettings": ssb_settings,
        }
        try:
            logger.info("[sdrangel] SSBDemod ds%d ch%d PATCH: offset=%d  rfBW=%d"
                        "  lowCut=%d  spanLog2=%d  audio='%s'  rollup=%s",
                        self.device_set, ch_idx,
                        ssb_settings['inputFrequencyOffset'],
                        ssb_settings['rfBandwidth'], ssb_settings['lowCutoff'],
                        ssb_settings['spanLog2'],
                        ssb_settings.get('audioDeviceName', '(unchanged)'),
                        [c.get('isHidden') for c in
                         ssb_settings.get('rollupState', {}).get('childrenStates', [])])
            resp = self._rest_patch(
                f"/sdrangel/deviceset/{self.device_set}/channel/{ch_idx}/settings",
                payload,
            )
            resp_s = resp.get("SSBDemodSettings", {}) if resp else {}
            logger.info("[sdrangel] SSBDemod ds%d ch%d response: offset=%s  rfBW=%s"
                        "  lowCut=%s  spanLog2=%s  rollup=%s",
                        self.device_set, ch_idx,
                        resp_s.get('inputFrequencyOffset'), resp_s.get('rfBandwidth'),
                        resp_s.get('lowCutoff'), resp_s.get('spanLog2'),
                        [c.get('isHidden') for c in
                         resp_s.get('rollupState', {}).get('childrenStates', [])])
        except Exception as exc:
            logger.warning("[sdrangel] SSBDemod PATCH failed: %s", exc)
            return None

        return ch_idx

    def probe_full_state(self) -> str:
        """Query all device sets and channels; return a formatted state snapshot.

        Makes O(n_channels) REST calls — intended for user-triggered refresh,
        not the high-frequency display timer.
        """
        try:
            overview = self._rest_get("/sdrangel/devicesets")
        except Exception as exc:
            return f"Cannot reach REST API: {exc}"

        lines = []
        for ds in overview.get("deviceSets", []):
            idx     = ds.get("deviceSetIndex", "?")
            dev     = ds.get("samplingDevice", {})
            hw      = dev.get("hwType", "?")
            freq_hz = dev.get("centerFrequency", 0)
            sr_hz   = dev.get("bandwidth", 0)   # device sample rate / bandwidth
            state   = dev.get("state", "?")
            lines.append(
                f"R:{idx}  {hw}  {freq_hz/1e6:.4f} MHz  "
                f"{sr_hz/1e6:.3f} MS/s  [{state}]"
            )

            try:
                ds_detail = self._rest_get(f"/sdrangel/deviceset/{idx}")
                channels  = ds_detail.get("channels", [])
            except Exception:
                lines.append("  (channel list unavailable)")
                continue

            for ch in channels:
                ch_idx = ch.get("index", "?")
                ch_id  = ch.get("id", "?")
                try:
                    s_resp = self._rest_get(
                        f"/sdrangel/deviceset/{idx}/channel/{ch_idx}/settings"
                    )
                except Exception:
                    lines.append(f"  R{idx}:{ch_idx}  {ch_id}  (settings unavailable)")
                    continue

                if ch_id == "UDPSink":
                    s     = s_resp.get("UDPSinkSettings", {})
                    fc    = s.get("inputFrequencyOffset", 0)
                    rfbw  = s.get("rfBandwidth", 0)
                    srout = s.get("outputSampleRate", 0)
                    port  = s.get("udpPort", 0)
                    ok    = "✓" if abs(rfbw - srout * 0.85) < 2000 else "✗"
                    lines.append(
                        f"  R{idx}:{ch_idx}  UDPSink  "
                        f"fc={fc/1000:+.1f} kHz  "
                        f"RFBW={rfbw/1000:.1f} kHz {ok}  "
                        f"SR={srout}  port={port}"
                    )
                elif ch_id == "SSBDemod":
                    s     = s_resp.get("SSBDemodSettings", {})
                    fc    = s.get("inputFrequencyOffset", 0)
                    rfbw  = s.get("rfBandwidth", 0)
                    audio = s.get("audioDeviceName", "—")
                    lines.append(
                        f"  R{idx}:{ch_idx}  SSBDemod  "
                        f"fc={fc/1000:+.3f} kHz  "
                        f"BW={rfbw/1000:.1f} kHz  "
                        f"→ {audio}"
                    )
                else:
                    lines.append(f"  R{idx}:{ch_idx}  {ch_id}")

        return "\n".join(lines) if lines else "(no device sets configured)"

    def probe_device_sets(self) -> list[dict]:
        """Return a list of device set descriptors from the SDRangel instance.

        Each entry is a dict with keys: index, hwType, nbStreams.
        Returns an empty list if the REST call fails.
        """
        try:
            data = self._rest_get("/sdrangel/devicesets")
            sets = data.get('deviceSets', [])
            result = []
            for ds in sets:
                idx     = ds.get('deviceSetIndex', 0)
                dev     = ds.get('samplingDevice', {})
                result.append({
                    'index':     idx,
                    'hwType':    dev.get('hwType', '?'),
                    'nbStreams': dev.get('nbStreams', 1),
                })
            return result
        except Exception as exc:
            logger.warning("[sdrangel] probe_device_sets failed: %s", exc)
            return []

    # ── Preset management ─────────────────────────────────────────────────

    def _preset_file_path(self) -> str:
        """Return MAP144's local copy path for this stream's preset file."""
        from pathlib import Path
        desc = _PRESET_DESC.get(self.stream_index, f"B210-S{self.stream_index}")
        cfg_dir = Path.home() / '.config' / 'map144'
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return str(cfg_dir / f"sdrangel_{desc.lower().replace('-', '_')}.sdrangel")

    def find_preset(self, group: str, description: str) -> bool:
        """Return True if a preset with this group/description exists in SDRangel."""
        try:
            data = self._rest_get("/sdrangel/presets")
            for p in data.get("presets", []):
                if p.get("group") == group and p.get("description") == description:
                    return True
        except Exception:
            pass
        return False

    def load_preset(self, group: str, description: str) -> bool:
        """Load a named preset into self.device_set. Returns True on success."""
        try:
            self._rest_post("/sdrangel/preset", {
                "deviceSetIndex": self.device_set,
                "preset": {"group": group, "description": description, "type": "R"},
            })
            logger.info("[sdrangel] Loaded preset '%s/%s' → device set %d",
                        group, description, self.device_set)
            return True
        except Exception as exc:
            logger.warning("[sdrangel] load_preset '%s/%s' failed: %s", group, description, exc)
            return False

    @staticmethod
    def _sdrangel_conf_path() -> str | None:
        """Path to SDRangel's live Qt settings file, or None if not found."""
        return _sdrangel_conf_live_path()

    @staticmethod
    def _layout_backup_path() -> str:
        """Path where MAP144 keeps its backup of the SDRangel Qt settings file."""
        from pathlib import Path
        cfg_dir = Path.home() / '.config' / 'map144'
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return str(cfg_dir / 'sdrangel_layout.conf')

    def save_configuration(self) -> bool:
        """Copy SDRangel's Qt settings file to a MAP144-owned backup.

        SDRangel writes its full state (window geometry, channel configs, presets)
        to ~/.config/f4exb/SDRangel.conf when it exits.  We copy that file so it
        can be restored before the next launch, preserving the user's layout.
        Call this when the user clicks 'Save SDRangel Layout'.
        """
        src = self._sdrangel_conf_path()
        if src is None:
            logger.warning("[sdrangel] save_configuration: SDRangel.conf not found "
                           "(checked snap and ~/.config paths)")
            return False
        dst = self._layout_backup_path()
        try:
            shutil.copy2(src, dst)
            logger.info("[sdrangel] Configuration saved: %s → %s", src, dst)
            return True
        except Exception as exc:
            logger.warning("[sdrangel] save_configuration failed: %s", exc)
            return False

    def restore_configuration(self) -> bool:
        """Copy the MAP144 backup back to SDRangel's Qt settings path.

        The module-level _restore_sdrangel_layout() is called automatically
        before every SDRangel launch.  This method is kept for completeness.
        Returns True if the backup existed and was copied; False if no backup.
        """
        import os
        if not os.path.isfile(self._layout_backup_path()):
            return False
        _restore_sdrangel_layout()
        return True

    def export_preset_to_file(self, group: str, description: str, file_path: str) -> bool:
        """Export a named preset from SDRangel to a local file. Returns True on success."""
        try:
            self._rest_get_body("/sdrangel/preset/file", {
                "filePath": file_path,
                "preset": {"group": group, "description": description, "type": "R"},
            })
            logger.info("[sdrangel] Exported preset '%s/%s' → %s", group, description, file_path)
            return True
        except Exception as exc:
            logger.warning("[sdrangel] export_preset_to_file failed: %s", exc)
            return False

    def import_preset_from_file(self, file_path: str) -> bool:
        """Import a preset from a local file into SDRangel's preset list. Returns True on success."""
        try:
            self._rest_post("/sdrangel/preset/file", {"filePath": file_path})
            logger.info("[sdrangel] Imported preset from %s", file_path)
            return True
        except Exception as exc:
            logger.warning("[sdrangel] import_preset_from_file '%s' failed: %s", file_path, exc)
            return False

    # ── REST helpers ───────────────────────────────────────────────────────

    def _rest_url(self, path: str) -> str:
        return f"http://{self.host}:{self.rest_port}{path}"

    def _rest_get(self, path: str) -> dict:
        with urllib.request.urlopen(self._rest_url(path), timeout=5.0) as r:
            return json.loads(r.read())

    def _rest_get_body(self, path: str, data: dict) -> dict:
        """GET with a request body — non-standard but required by SDRangel preset export."""
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            self._rest_url(path), data=body, method='GET',
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return json.loads(r.read())

    @staticmethod
    def _http_error_detail(exc: Exception) -> str:
        """Extract SDRangel's response body from an HTTPError for diagnostic logging."""
        try:
            import urllib.error
            if isinstance(exc, urllib.error.HTTPError):
                return exc.read().decode(errors='replace')
        except Exception:
            pass
        return str(exc)

    def _rest_patch(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            self._rest_url(path), data=body, method='PATCH',
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                return json.loads(r.read())
        except Exception as exc:
            raise RuntimeError(f"PATCH {path}: {self._http_error_detail(exc)}") from exc

    def _rest_put(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            self._rest_url(path), data=body, method='PUT',
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                return json.loads(r.read())
        except Exception as exc:
            raise RuntimeError(f"PUT {path}: {self._http_error_detail(exc)}") from exc

    def _rest_post(self, path: str, data: dict | None = None) -> dict:
        body = json.dumps(data or {}).encode()
        req  = urllib.request.Request(
            self._rest_url(path), data=body, method='POST',
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                return json.loads(r.read())
        except Exception as exc:
            raise RuntimeError(f"POST {path}: {self._http_error_detail(exc)}") from exc

    # ── Internal ───────────────────────────────────────────────────────────

    def _set_device_freq_hz(self, freq_hz: int):
        """PATCH center frequency into the device-specific settings sub-dict."""
        path     = f"/sdrangel/deviceset/{self.device_set}/device/settings"
        settings = self._rest_get(path)
        for val in settings.values():
            if isinstance(val, dict) and 'centerFrequency' in val:
                val['centerFrequency'] = freq_hz
                break
        self._rest_patch(path, settings)

        # Read back actual frequency after hardware settles.
        actual_hz = freq_hz
        try:
            updated = self._rest_get(path)
            for val in updated.values():
                if isinstance(val, dict) and 'centerFrequency' in val:
                    actual_hz = val['centerFrequency']
                    break
        except Exception:
            pass
        self.center_freq_mhz        = actual_hz / 1e6
        self.center_freq_mhz_actual = actual_hz / 1e6

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
            except OSError:
                break
            except Exception:
                continue

            n_samps = len(data) // 4   # 2 bytes I + 2 bytes Q per sample
            if n_samps == 0:
                continue

            raw = np.frombuffer(data[:n_samps * 4], dtype='<i2').astype(np.float32)
            iq  = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 32768.0

            # Use wall-clock time per packet.  Sample-count timestamps
            # (t_start + abs_samps/rate) drift when the stream starts late or
            # OS drops packets, causing spectrogram data to land at wrong positions.
            t_now = time.time()
            self._abs_samps += n_samps
            self.recv_count += 1

            ts_int  = int(t_now)
            ts_frac = int((t_now - ts_int) * 1e12)

            pkt = _SDRAngelPacket(iq.reshape(-1, 1), ts_int, ts_frac)
            try:
                self.sample_queue.put_nowait(pkt)
            except queue.Full:
                self.drop_count += 1
