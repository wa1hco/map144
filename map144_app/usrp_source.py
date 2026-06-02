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
"""USRPSource: UHD Python wrapper for USRP B210 IQ streaming.

Presents the same ``sample_queue`` / ``start()`` / ``stop()`` interface as
``AirspyHFSource`` and ``NesdrSmartSource``.

Requires the UHD Python package (``import uhd``).  Install hints:
    Linux:    sudo apt install python3-uhd uhd-host && sudo uhd_images_downloader
    macOS:    brew install uhd && uhd_images_downloader
    Windows:  conda install -c conda-forge uhd  (in a conda env; the pip
              ``uhd`` package has no Windows wheels and Ettus' Windows
              installer omits the Python bindings)
              See docs/ALPHA_NOTES.md for the full Windows B210 setup.

Hardware / signal chain
-----------------------
The AD9361 RF front-end is a direct-conversion receiver.  Its analog LPF
has a minimum bandwidth of 200 kHz.  Requesting 192 kHz from UHD puts the
AD9361's internal HB filter chain to work: the analog LPF (200 kHz) is well
inside the ±96 kHz Nyquist of the 192 kHz output, so no aliasing occurs in
the analog domain.  The AD9361's internal decimation and DC/quadrature
correction calibrations are applied before samples reach the host.

LO offset and NCO
-----------------
Direct-conversion receivers have LO leakage that produces a DC artifact at
0 Hz IF.  Even with AD9361 calibration the artifact drifts with temperature
and can block signals tuned exactly to the center frequency.

To avoid this the hardware LO is tuned ``lo_offset_hz`` below the target so
the desired signal arrives at ``+lo_offset_hz`` in the IF.  The host NCO
then multiplies by exp(-j·2π·lo_offset·n/hw_rate) **before** the
decimation FIR, bringing the signal to DC and shifting the LO artifact to
``-lo_offset_hz``.  The 4× decimation FIR (65 taps, cutoff ≈ 21.6 kHz)
rejects the artifact at −40 kHz with 60+ dB of attenuation.

The NCO must run at the hardware rate (192 kHz) before decimation.
Running it after decimation at 48 kHz would require shifting from +40 kHz,
which exceeds the 24 kHz Nyquist of the output — the shift would alias.

Pipeline contract
-----------------
Output: 48 kHz complex IQ, ±1.0 scale, timestamps as picoseconds
    t = timestamp_int + timestamp_frac * 1e-12

Prerequisites
-------------
See the module docstring above for per-OS UHD install hints, and
docs/ALPHA_NOTES.md for the full Windows B210 walkthrough.
"""

import logging
import os
import queue
import sys
import threading
import time

import numpy as np
from scipy.signal import firwin, upfirdn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe for UHD Python package at import time — fail gracefully
# ---------------------------------------------------------------------------

# UHD logging.  Keep the CONSOLE quiet — UHD's INFO chatter ("Asking for clock
# rate", register loopback, …) is noise — but route a full info-level log to a
# per-process FILE so MAP144 can surface the *meaningful* lines (the two-phase
# firmware/FPGA load, USB mode, and USB/transfer errors) to the GUI + its own
# log, without the chatter.  Tailed in USRPSource._drain_uhd_log().  Every var
# is setdefault, so an explicit operator UHD_LOG_* env always wins.
os.environ.setdefault('UHD_LOG_LEVEL',         'info')      # allow info into the sinks
os.environ.setdefault('UHD_LOG_CONSOLE_LEVEL', 'warning')   # …but keep the console quiet
from pathlib import Path as _Path
_uhd_log_dir = _Path(__file__).resolve().parent.parent / 'MSK144' / 'logs'
try:
    _uhd_log_dir.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
os.environ.setdefault('UHD_LOG_FILE',       str(_uhd_log_dir / f'uhd_{os.getpid()}.log'))
os.environ.setdefault('UHD_LOG_FILE_LEVEL', 'info')

try:
    import uhd as _uhd
    _UHD_AVAILABLE = True
except ImportError:
    _uhd = None
    _UHD_AVAILABLE = False

# Stage 3 — a tiny SUBPROCESS (its own GIL) that heartbeats the console while
# UHD's make() holds the main process's GIL through the firmware/FPGA load, and
# tails the UHD log so the literal load lines stream live if UHD flushes them.
# ASCII-only (passed via `python -c`).  argv: log_path, start_byte_pos, t0.
_STARTUP_TICKER_SRC = r'''
import sys, os, time
path, pos, t0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
KEEP = ("firmware","fpga","operating over usb","detected device",
        "error","libusb","no device")
last = 0.0
while time.time() - t0 < 180.0:          # safety cap; parent kills us first
    try:
        if path and os.path.exists(path):
            with open(path, errors="replace") as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for ln in chunk.splitlines():
                if any(k in ln.lower() for k in KEEP):
                    print("[usrp]   . " + ln.split(",", 5)[-1].strip(), flush=True)
    except OSError:
        pass
    now = time.time()
    if now - last >= 3.0:
        last = now
        print("[usrp] still opening B210... %.0fs (firmware/FPGA over USB; please wait)"
              % (now - t0), flush=True)
    time.sleep(0.4)
'''

# ---------------------------------------------------------------------------
# Packet compatible with VitaPacket interface expected by runtime.py
# ---------------------------------------------------------------------------

class _USRPPacket:
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        # samples: complex64 shape (N, num_channels)
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


# ---------------------------------------------------------------------------
# Sample rates and decimation
# ---------------------------------------------------------------------------
# Request 192 kHz from UHD.  The AD9361 handles all internal decimation from
# its oversampled ADC; the analog LPF (min 200 kHz) fits cleanly inside the
# ±96 kHz Nyquist.  A single host 4× FIR stage then delivers 48 kHz to the
# pipeline.
#
# LO offset = 40 kHz.  After the host NCO shifts the signal to DC, the LO
# artifact sits at −40 kHz.  The 4× FIR cutoff is ≈21.6 kHz, so the
# artifact is 18 kHz into the stopband — well over 60 dB rejection.
# A smaller offset (e.g. 25 kHz) would place the artifact only 1 kHz past
# the cutoff, yielding insufficient rejection at 48 kHz output.

_HW_RATE     = 192_000   # request from UHD; AD9361 decimates to this rate internally
_TARGET_RATE =  48_000   # pipeline output rate
_DECIMATE    = _HW_RATE // _TARGET_RATE   # 4

# 65-tap anti-alias FIR for 4× decimation.
# Cutoff = 0.9 / 4 = 0.225 of input Nyquist = 0.225 × 96 kHz ≈ 21.6 kHz.
_FIR_TAPS = firwin(65, 0.9 / _DECIMATE).astype(np.float32)


def _make_decimator():
    """Return a stateful 4× decimation function (per-instance state).

    Uses scipy.signal.upfirdn which internally dispatches to an optimised
    Cython path.  Processing real and imaginary parts separately as float32
    avoids the complex FIR overhead and is ~38% faster than the old
    lfilter path (which in recent scipy falls back to
    apply_along_axis(np.convolve, ...) for FIR filters).

    Streaming state is maintained by prepending the last (ntaps-1) input
    samples to each block before calling upfirdn, then trimming the leading
    (ntaps-1)//decimate output samples that correspond to those prepended
    samples.  The net output count is always N//decimate for a block of N
    input samples.

    Skip derivation:
      - Extended input length: (ntaps-1) + N = 64 + 3840 = 3904
      - upfirdn output length: (3904-1)//4+1 = 976
      - Leading outputs from prepended state: (ntaps-1)//decimate = 64//4 = 16
      - Useful outputs: 976 - 16 = 960 = N//decimate  ✓
    """
    _ntaps_m1 = len(_FIR_TAPS) - 1                             # 64
    _skip     = _ntaps_m1 // _DECIMATE                          # 16
    _b_f32    = _FIR_TAPS                                       # already float32
    _state    = np.zeros(_ntaps_m1, dtype=np.complex64)

    def _apply(iq: np.ndarray) -> np.ndarray:
        nonlocal _state
        x     = np.ascontiguousarray(iq, dtype=np.complex64)
        x_ext = np.concatenate((_state, x))                    # (ntaps-1+N,) c64
        # upfirdn produces full convolution output including filter tail;
        # slice [skip : skip+out_n] discards both the leading state-warmup
        # outputs and the trailing filter-tail outputs.
        yr    = upfirdn(_b_f32, x_ext.real, up=1, down=_DECIMATE)
        yi    = upfirdn(_b_f32, x_ext.imag, up=1, down=_DECIMATE)
        _state = x[-_ntaps_m1:]                                 # save last 64 samples
        n_out = len(iq) // _DECIMATE
        out   = np.empty(n_out, dtype=np.complex64)
        out.real[:] = yr[_skip : _skip + n_out]
        out.imag[:] = yi[_skip : _skip + n_out]
        return out

    return _apply


# Receive buffer: 20 ms at 192 kHz = 3840 samples → 960 after 4× decimation.
_RECV_SIZE = 3840


class USRPSource:
    """UHD Python wrapper for USRP B210 IQ streaming.

    Usage::

        src = USRPSource(center_freq_mhz=50.260)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``pkt.samples`` is complex64 in ±1.0 scale at 48 kHz.
    Timestamps use picosecond encoding: t = timestamp_int + timestamp_frac * 1e-12
    """

    def __init__(self, center_freq_mhz: float = 50.260,
                 target_rate: int = _TARGET_RATE,
                 gain_db: float = 30.0,
                 antenna: str = "RX2",
                 lo_offset_hz: float = 40_000.0,
                 device_args: str = "",
                 dual_channel: bool = False):
        if not _UHD_AVAILABLE:
            if sys.platform == 'win32':
                hint = (
                    "the UHD Python package is not pip-installable on Windows "
                    "(no PyPI wheels; Ettus' Windows installer omits the "
                    "bindings).  Easiest path is a conda env:\n"
                    "    1. Install Miniconda from "
                    "https://docs.conda.io/projects/miniconda/\n"
                    "    2. conda create -n map144 python=3.11\n"
                    "    3. conda activate map144\n"
                    "    4. conda install -c conda-forge uhd\n"
                    "    5. pip install -r requirements.txt\n"
                    "    6. uhd_images_downloader\n"
                    "See docs/ALPHA_NOTES.md for the full Windows B210 setup.\n"
                    "If you have no B210 hardware on this PC, ignore this "
                    "message and use a different source (Flex / AirSpy)."
                )
            elif sys.platform == 'darwin':
                hint = "install with:  brew install uhd && uhd_images_downloader"
            else:
                hint = ("install with:  sudo apt install python3-uhd uhd-host "
                        "&& sudo uhd_images_downloader")
            raise RuntimeError(f"UHD Python package not found — {hint}")
        # UHD 4.6.0's bindings are compiled against NumPy 1.26 (the system
        # package).  NumPy 2.x changed the PyArrayObject layout and segfaults in
        # _recv_loop when uhd fills the IQ buffers.  If the venv has drifted off
        # the requirements.txt pin (numpy<2), fail loudly here with a fix rather
        # than crashing mid-capture some night.
        if int(np.__version__.split('.')[0]) >= 2:
            rebuild = '.\\install.ps1' if sys.platform == 'win32' else './install.sh'
            raise RuntimeError(
                f"NumPy {np.__version__} breaks UHD streaming (needs <2). "
                f"Rebuild the environment with:  {rebuild}"
            )

        self.center_freq_mhz        = center_freq_mhz
        self.target_rate            = target_rate
        self.gain_db                = gain_db
        self.antenna                = antenna
        self.lo_offset_hz           = lo_offset_hz
        self.device_args            = device_args
        self.dual_channel           = dual_channel   # True → enable RX channel 1 (RX2 port)
        self.sample_queue           = queue.Queue(maxsize=4000)
        self.center_freq_mhz_actual = center_freq_mhz

        self._usrp       = None
        self._streamer   = None
        self._thread     = None
        self._running    = False
        self._decimator  = None   # ch0 decimator
        self._decimator1 = None   # ch1 decimator (dual_channel only)
        self._nco_phase  = 0.0   # running NCO phase accumulator (radians)
        self._nco_phase1 = 0.0   # ch1 NCO phase
        self.ch0_rms     = None  # rolling RMS amplitude for display (ch0)
        self.ch1_rms     = None  # rolling RMS amplitude for display (ch1)
        self.recv_count  = 0    # total recv() calls that returned n > 0 (monotonic)
        # Startup observability — surfaced live in the GUI receiver label + the
        # console heartbeat so the operator isn't staring at a black box during
        # the firmware/FPGA load (make) or a slow USB stream start.
        self.startup_phase = 'idle'   # idle|opening|configuring|waiting_iq|streaming
        self.startup_t0    = 0.0      # wall time the current start() began
        self._startup_active = False  # gates the console startup-watcher thread
        # Stage 2 — tail UHD's own log file for the literal firmware/FPGA/USB lines.
        self.uhd_log_path   = os.environ.get('UHD_LOG_FILE')
        self._uhd_log_pos   = 0       # read cursor (bytes) into uhd_log_path
        self.startup_detail = ''      # latest meaningful UHD log line (GUI + heartbeat)
        self._startup_proc  = None    # Stage-3 live-console subprocess (GIL-free)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        # 'opening' covers the UHD make() — firmware + FPGA two-phase load (can be
        # several minutes over a marginal USB link); the GUI shows the elapsed.
        self.startup_phase = 'opening'
        self.startup_t0    = time.time()
        self.startup_detail = ''
        try:   # start the UHD-log cursor at EOF so we read only THIS open's lines
            self._uhd_log_pos = (os.path.getsize(self.uhd_log_path)
                                 if self.uhd_log_path and os.path.exists(self.uhd_log_path)
                                 else 0)
        except OSError:
            self._uhd_log_pos = 0
        # Console heartbeat in its own thread so it prints DURING the blocking
        # make()/stream-setup (the recv-loop heartbeat can't — it doesn't run
        # until the stream is up).  Covers the firmware/FPGA load + slow USB.
        self._startup_active = True
        threading.Thread(target=self._startup_watcher, daemon=True,
                         name='usrp-startup').start()
        # UHD's MultiUSRP() holds the GIL through the firmware+FPGA load, so the
        # in-process watcher/GUI freeze DURING it.  A GIL-free subprocess gives a
        # live console heartbeat (and streams the UHD load lines if UHD flushes).
        print("[usrp] opening B210 - loading firmware/FPGA; on a marginal USB "
              "link this can take 30-60 s and the UI may freeze.  Please wait...",
              flush=True)
        self._spawn_startup_ticker()
        try:
            self._usrp = _uhd.usrp.MultiUSRP(self.device_args)
        finally:
            self._kill_startup_ticker()
        # Skip the make-phase lines the ticker already streamed so the in-process
        # watcher doesn't re-print them.
        try:
            if self.uhd_log_path and os.path.exists(self.uhd_log_path):
                self._uhd_log_pos = os.path.getsize(self.uhd_log_path)
        except OSError:
            pass
        self.startup_phase = 'configuring'   # make() done; now rate/tune/stream setup

        channels = [0, 1] if self.dual_channel else [0]

        # B210 has one daughterboard with two front-ends (A:A → TX/RX port,
        # A:B → RX2 port).  UHD's default subdev_spec enables only A:A; if
        # we request channels=[0, 1] without expanding the subdev_spec, UHD
        # silently delivers channel 0 samples in both output slots — the H
        # and V streams look identical.  Expand the subdev_spec in dual
        # mode so channel 0 maps to A:A and channel 1 maps to A:B.
        # Log at print() level so misconfigurations are immediately visible
        # without turning up the log verbosity.
        if self.dual_channel:
            try:
                self._usrp.set_rx_subdev_spec(_uhd.usrp.SubdevSpec("A:A A:B"))
            except Exception as exc:
                print(f"[usrp] set_rx_subdev_spec('A:A A:B') FAILED: {exc}", flush=True)
                print("[usrp] dual-channel mode will deliver duplicated ch0 data", flush=True)
            try:
                current = self._usrp.get_rx_subdev_spec(0)
                print(f"[usrp] rx_subdev_spec now: '{current}' "
                      f"(num_rx_channels={self._usrp.get_rx_num_channels()})", flush=True)
            except Exception as exc:
                print(f"[usrp] get_rx_subdev_spec readback failed: {exc}", flush=True)

        for ch in channels:
            self._usrp.set_rx_rate(_HW_RATE, ch)
            lo_tune_hz = self.center_freq_mhz * 1e6 - self.lo_offset_hz
            tune_result = self._usrp.set_rx_freq(
                _uhd.types.TuneRequest(lo_tune_hz), ch
            )
            # Ch0 uses the user-configured gain; ch1 uses _gain_ch1 if set
            gain = self.gain_db if ch == 0 else getattr(self, '_gain_ch1', self.gain_db)
            self._usrp.set_rx_gain(gain, ch)
            # Ch0 uses the user-selected antenna; ch1 uses _antenna_ch1 if set
            ant = self.antenna if ch == 0 else getattr(self, '_antenna_ch1', 'RX2')
            self._usrp.set_rx_antenna(ant, ch)

        actual_rate = self._usrp.get_rx_rate(0)
        # Report the target RF frequency, not the raw LO position.
        self.center_freq_mhz_actual = (tune_result.actual_rf_freq + self.lo_offset_hz) / 1e6

        # Create RX streamer — fc32 (complex float32) from host, sc16 over wire.
        st_args = _uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = channels
        self._streamer = self._usrp.get_rx_stream(st_args)

        # Synchronise USRP internal clock to system wall time so that
        # packet timestamps match the Unix epoch expected by processing.py.
        self._usrp.set_time_now(_uhd.types.TimeSpec(time.time()), 0)

        # Start continuous streaming.
        # Multi-channel streamers require a timed start — stream_now=True causes
        # "will fail to time align" on UHD ≥4.x with >1 channel.  A future
        # time_spec aligns both channels.  UHD pads with zeros until the command
        # fires, so we record the wall-clock equivalent and skip those packets.
        stream_cmd = _uhd.types.StreamCMD(_uhd.types.StreamMode.start_cont)
        if self.dual_channel:
            _delay = 0.25   # seconds — enough for USB scheduling + UHD machinery
            stream_cmd.stream_now = False
            stream_cmd.time_spec  = self._usrp.get_time_now(0) + _uhd.types.TimeSpec(_delay)
            self._stream_start_wall = time.time() + _delay
        else:
            stream_cmd.stream_now = True
            self._stream_start_wall = 0.0
        self._streamer.issue_stream_cmd(stream_cmd)

        self._decimator  = _make_decimator()
        self._decimator1 = _make_decimator() if self.dual_channel else None
        self._nco_phase  = 0.0
        self._nco_phase1 = 0.0
        # Precompute NCO rotation table for one full recv buffer.
        if self.lo_offset_hz != 0.0:
            _nco_step = -2.0 * np.pi * self.lo_offset_hz / _HW_RATE
            self._nco_table = np.exp(
                1j * np.arange(_RECV_SIZE, dtype=np.float64) * _nco_step
            ).astype(np.complex64)
            self._nco_step = _nco_step
        else:
            self._nco_table = None
            self._nco_step = 0.0
        # Stream command issued; the device may take a while to actually deliver
        # IQ over a marginal USB link.  'waiting_iq' until the first samples land.
        self.startup_phase = 'waiting_iq'
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True,
                                        name='usrp-recv')
        self._thread.start()

        ch1_info = (f"  RF1: gain={getattr(self, '_gain_ch1', self.gain_db):.0f} dB  "
                    f"antenna={getattr(self, '_antenna_ch1', 'RX2')}"
                    if self.dual_channel else "")
        logger.debug("[usrp] started: %.4f MHz  LO offset=%+.1f kHz  "
                     "hw=%.0f Hz  decimation=%dx  out=%d Hz  "
                     "RF0: gain=%.0f dB  antenna=%s  channels=%s%s",
                     self.center_freq_mhz_actual, self.lo_offset_hz / 1e3,
                     actual_rate, _DECIMATE, _TARGET_RATE,
                     self.gain_db, self.antenna,
                     'dual' if self.dual_channel else 'single', ch1_info)

    def stop(self):
        self._running = False
        self._startup_active = False
        self._kill_startup_ticker()
        self.startup_phase = 'idle'
        if self._streamer is not None:
            try:
                stream_cmd = _uhd.types.StreamCMD(_uhd.types.StreamMode.stop_cont)
                self._streamer.issue_stream_cmd(stream_cmd)
            except Exception:
                pass
            self._streamer = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._usrp = None

    def retune(self, center_freq_mhz: float):
        """Retune to a new center frequency while streaming.

        Updates both channels when dual_channel is active.  Resets the NCO
        phase accumulators so the downconversion stays coherent after the LO
        settles.  Safe to call from the Qt main thread while the recv loop runs.
        """
        self.center_freq_mhz = center_freq_mhz
        if self._usrp is None:
            return
        channels = [0, 1] if self.dual_channel else [0]
        tune_result = None
        for ch in channels:
            lo_tune_hz = center_freq_mhz * 1e6 - self.lo_offset_hz
            try:
                tune_result = self._usrp.set_rx_freq(
                    _uhd.types.TuneRequest(lo_tune_hz), ch
                )
            except Exception as exc:
                logger.warning("[usrp] retune ch%d to %.3f MHz failed: %s", ch, center_freq_mhz, exc)
                return
        if tune_result is not None:
            self.center_freq_mhz_actual = (tune_result.actual_rf_freq + self.lo_offset_hz) / 1e6
        self._nco_phase  = 0.0
        self._nco_phase1 = 0.0
        logger.debug("[usrp] retuned to %.4f MHz", self.center_freq_mhz_actual)

    # ── Internal receive loop (runs in daemon thread) ─────────────────────

    def _nco_apply(self, chunk, phase_attr):
        """Apply NCO shift to chunk; update running phase. Returns shifted chunk."""
        if self._nco_table is None:
            return chunk
        ns  = len(chunk)
        rot = np.exp(1j * getattr(self, phase_attr)).astype(np.complex64)
        out = chunk * (rot * self._nco_table[:ns])
        setattr(self, phase_attr,
                float((getattr(self, phase_attr) + self._nco_step * ns) % (2.0 * np.pi)))
        return out

    def _spawn_startup_ticker(self):
        """Launch the Stage-3 GIL-free console ticker for the duration of make()."""
        if not self.uhd_log_path:
            return
        import subprocess
        import sys as _sys
        try:
            self._startup_proc = subprocess.Popen(
                [_sys.executable, '-c', _STARTUP_TICKER_SRC,
                 self.uhd_log_path, str(self._uhd_log_pos), str(self.startup_t0)])
        except Exception:
            self._startup_proc = None

    def _kill_startup_ticker(self):
        p, self._startup_proc = getattr(self, '_startup_proc', None), None
        if p is not None and p.poll() is None:
            try:
                p.terminate(); p.wait(timeout=1.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    _UHD_KEEP = ('firmware', 'fpga', 'operating over usb', 'detected device',
                 'error', 'libusb', 'no device', 'overflow', 'timeout')

    def _drain_uhd_log(self):
        """Read new UHD_LOG_FILE lines since the last drain and surface the
        meaningful ones (firmware/FPGA two-phase load, USB mode, transfer errors)
        to the Python log + startup_detail.  Filters UHD's per-call chatter."""
        p = self.uhd_log_path
        if not p:
            return
        try:
            with open(p, 'r', errors='replace') as f:
                f.seek(self._uhd_log_pos)
                chunk = f.read()
                self._uhd_log_pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if line and any(k in line.lower() for k in self._UHD_KEEP):
                logger.info("[uhd] %s", line)
                # UHD's FILE log is CSV: ts,thread,file:line,level,component,msg
                # — the message is the field after the 5th comma (keeps commas).
                self.startup_detail = line.split(',', 5)[-1].strip()[:64]

    def _startup_watcher(self):
        """Console heartbeat during startup, in its own thread so it prints while
        the blocking make()/stream-setup runs, and tails UHD's own log for the
        literal firmware/FPGA/USB lines.  Exits once streaming, on stop, or after
        a safety cap so a failed start() can't loop forever."""
        _names = {'opening':     'opening device (firmware/FPGA)',
                  'configuring': 'configuring',
                  'waiting_iq':  'waiting for IQ'}
        _last = 0.0
        while self._startup_active:
            self._drain_uhd_log()
            ph = self.startup_phase
            if ph in ('streaming', 'idle'):
                break
            _now = time.time()
            el = _now - self.startup_t0
            if el > 300.0:                 # safety: never loop forever
                break
            if _now - _last >= 5.0:
                _last = _now
                hint = ("  — USB link may be marginal (check hub/power)"
                        if el > 20.0 else "")
                _det = f"  · {self.startup_detail}" if self.startup_detail else ""
                print(f"[usrp] {_names.get(ph, ph)}… {el:.0f}s{_det}{hint}", flush=True)
            time.sleep(0.5)
        self._drain_uhd_log()   # final drain to catch the last lines (e.g. errors)

    def _recv_loop(self):
        # For a timed start (dual_channel), sleep until the commanded time arrives
        # before calling recv().  UHD over USB pads with zeros until the time_spec
        # fires; calling recv() too early also risks the streamer delivering an
        # internal burst of garbage.  Sleeping past the start time is safe because
        # UHD buffers samples internally.
        start_wall = getattr(self, '_stream_start_wall', 0.0)
        if start_wall > 0.0:
            remaining = start_wall - time.time()
            if remaining > 0:
                time.sleep(remaining + 0.05)   # +50 ms margin

        # UHD Python recv() writes in-place into a 2D numpy array (channels × samples).
        # A Python list of 1D arrays does NOT work — the binding copies list elements
        # rather than writing back, leaving the originals at zero.
        num_ch = 2 if self.dual_channel else 1
        recv_bufs = np.zeros((num_ch, _RECV_SIZE), dtype=np.complex64)
        metadata = _uhd.types.RXMetadata()

        while self._running:
            try:
                n = self._streamer.recv(recv_bufs, metadata, timeout=1.0)
            except Exception as exc:
                logger.warning("[usrp] recv error: %s", exc)
                self._drain_uhd_log()   # surface UHD's USB/transfer detail
                time.sleep(0.1)
                continue

            if metadata.error_code not in (
                _uhd.types.RXMetadataErrorCode.none,
                _uhd.types.RXMetadataErrorCode.overflow,
            ):
                logger.warning("[usrp] metadata error: %s", metadata.strerror())
                continue

            if n <= 0:
                continue

            if self.recv_count == 0:
                # First IQ — startup done; stop the watcher heartbeat.
                self.startup_phase   = 'streaming'
                self._startup_active = False
                print(f"[usrp] streaming — first IQ after "
                      f"{time.time() - self.startup_t0:.0f}s", flush=True)
            self.recv_count += 1
            ts_int  = metadata.time_spec.get_full_secs()
            ts_frac = int(metadata.time_spec.get_frac_secs() * 1e12)

            if self.dual_channel:
                # Apply NCO and decimate each channel independently.
                # NCO must run at _HW_RATE before decimation FIR.
                ch0 = self._nco_apply(recv_bufs[0, :n].copy(), '_nco_phase')
                ch1 = self._nco_apply(recv_bufs[1, :n].copy(), '_nco_phase1')
                ch0 = self._decimator(ch0)
                ch1 = self._decimator1(ch1)
                # Update rolling RMS for display (exponential moving average).
                _alpha = 0.05
                _rms0 = float(np.sqrt(np.mean(np.abs(ch0) ** 2)))
                _rms1 = float(np.sqrt(np.mean(np.abs(ch1) ** 2)))
                self.ch0_rms = _rms0 if self.ch0_rms is None else (1 - _alpha) * self.ch0_rms + _alpha * _rms0
                self.ch1_rms = _rms1 if self.ch1_rms is None else (1 - _alpha) * self.ch1_rms + _alpha * _rms1
                # Stack into (N, 2) — zero-copy column view from np.column_stack
                samples = np.column_stack((ch0, ch1))
            else:
                # NCO: shift signal from +lo_offset_hz to DC at the hardware rate
                # (192 kHz) BEFORE the decimation FIR.  The FIR then rejects the
                # LO artifact at -lo_offset_hz.  The NCO must run at _HW_RATE, not
                # the decimated target_rate — shifting by 40 kHz after decimation
                # to 48 kHz would exceed the 24 kHz Nyquist and alias.
                chunk = self._nco_apply(recv_bufs[0, :n].copy(), '_nco_phase')
                chunk = self._decimator(chunk)
                _alpha = 0.05
                _rms0 = float(np.sqrt(np.mean(np.abs(chunk) ** 2)))
                self.ch0_rms = _rms0 if self.ch0_rms is None else (1 - _alpha) * self.ch0_rms + _alpha * _rms0
                samples = chunk.reshape(-1, 1)

            pkt = _USRPPacket(samples, ts_int, ts_frac)
            try:
                self.sample_queue.put_nowait(pkt)
            except queue.Full:
                pass  # drop — same as other sources
