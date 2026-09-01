"""MAP144 -> WSJT-X audio bridge: live baseband → virtual soundcard.

Taps complex baseband (DC-centred, typically 12 kHz), upconverts to real audio
at 1500 Hz (phase-continuous across chunks), and streams signed-16 PCM into a
platform transport:

  * Linux: PipeWire/Pulse null sink + paplay (auto-created; WSJT-X records the
    sink monitor).  PipeWire resamples 12→48 kHz and absorbs clock drift.
  * Windows / macOS: PortAudio via ``sounddevice`` into a user-installed virtual
    cable (VB-CABLE / VoiceMeeter / BlackHole).  Device chosen by config env
    override, else auto-matched by common cable names.

DSP (``feed``) is platform-identical.  Only the Transport differs.

Deployment: for the B210 this runs by default -- runtime
``_setup_wsjtx_audio_export`` creates one exporter per RF port at source start,
with sink names ``map144.RF0.rx`` / ``map144.RF1.rx`` (RF0 <- RX0, RF1 <- RX1).
Disable with ``MAP144_WSJTX_AUDIO=0``.

Env tuning:
  MAP144_WSJTX_TARGET_RMS   int16 RMS (default 26 → ~30 dB on WSJT-X meter)
  MAP144_WSJTX_DEVICE       output device name (substring OK); comma-list for
                            RF0,RF1,… / dial index 0,1,…
  MAP144_WSJTX_DEVICE_RF0   per-port override (also _RF1)
"""
from __future__ import annotations

import atexit
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

_FS = 12000          # calling-channel sample rate (Hz)
_FC = 1500.0         # audio carrier WSJT-X expects (Hz)
_TAU_S = 8.0         # auto-level time constant (>> ping duration, so pings survive)
_QUEUE_MAX = 128     # ~chunks buffered before drop-oldest (bounds latency + memory)

# Auto-match candidates for PortAudio output devices (case-insensitive substring).
# Ordered: prefer first match for index 0; later entries / Cable B for index ≥1.
_WIN_DEVICE_CANDIDATES = (
    "cable a input",         # VB-CABLE A+B → index 0
    "cable input",           # single VB-CABLE → index 0 when A absent
    "cable b input",         # VB-CABLE A+B → index 1
    "voicemeeter input",
    "voicemeeter aux input",
    "voicemeeter vaio",
)
_MAC_DEVICE_CANDIDATES = (
    "blackhole 2ch",
    "blackhole 16ch",
    "blackhole",
)


# ── Transports ──────────────────────────────────────────────────────────────

class AudioTransport:
    """Byte sink for 12 kHz mono s16le PCM.  ``write`` may block briefly."""

    wsjtx_input_hint: str = ""   # what the operator should pick in WSJT-X

    def write(self, pcm: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PipeWireTransport(AudioTransport):
    """Linux: create a null sink and paplay raw PCM into it."""

    def __init__(self, sink_name: str, description: str, stream_name: str, fs: int):
        self.sink = sink_name
        self.description = description
        self.stream_name = stream_name
        self.fs = int(fs)
        self._module_id = None
        self._proc = None
        self.wsjtx_input_hint = f"Monitor of {description}"
        self._ensure_sink()
        self._open_player()

    def _sh(self, cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def _unload_existing(self):
        out = self._sh("pactl list short modules").stdout
        for ln in out.splitlines():
            if "module-null-sink" in ln and f"sink_name={self.sink}" in ln:
                self._sh(f"pactl unload-module {ln.split()[0]}")

    def _ensure_sink(self):
        self._unload_existing()
        # Nested quoting: outer dq for sink_properties value, inner sq for the
        # spaced description (pactl re-tokenises on spaces after joining argv).
        cmd = ["pactl", "load-module", "module-null-sink",
               f"sink_name={self.sink}", "media.class=Audio/Sink",
               f"sink_properties=\"device.description='{self.description}'\""]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"could not create null sink '{self.sink}': "
                               f"{r.stderr.strip()}")
        self._module_id = r.stdout.strip() or None

    def _open_player(self):
        if shutil.which("paplay"):
            cmd = ["paplay", f"--device={self.sink}", "--raw",
                   "--format=s16le", f"--rate={self.fs}", "--channels=1",
                   "--client-name=MAP144", f"--stream-name={self.stream_name}"]
        elif shutil.which("pw-play"):
            cmd = ["pw-play", f"--target={self.sink}", "--format=s16",
                   f"--rate={self.fs}", "--channels=1", "-"]
        else:
            raise RuntimeError("WSJT-X audio export needs 'paplay' or 'pw-play' "
                               "(install pulseaudio-utils or pipewire-bin)")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, pcm: bytes) -> None:
        proc = self._proc
        if proc is None:
            raise BrokenPipeError("paplay closed")
        proc.stdin.write(pcm)

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
        mod, self._module_id = self._module_id, None
        if mod:
            try:
                self._sh(f"pactl unload-module {mod}")
            except Exception:
                pass


class PortAudioTransport(AudioTransport):
    """Windows/macOS: write PCM to a PortAudio output device (virtual cable)."""

    def __init__(self, device_name: str, fs: int, device_index: int | None = None):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "WSJT-X audio on Windows/macOS needs the 'sounddevice' package "
                "(pip install sounddevice). Also install VB-CABLE (Windows) or "
                "BlackHole (macOS)."
            ) from exc
        self._sd = sd
        self.fs = int(fs)
        self.device_name = device_name
        if device_index is None:
            device_index = _find_output_device_index(sd, device_name)
        if device_index is None:
            raise RuntimeError(
                f"PortAudio output device matching {device_name!r} not found. "
                f"Install VB-CABLE / BlackHole, or set MAP144_WSJTX_DEVICE. "
                f"Available outputs: {_list_output_names(sd)}"
            )
        self.device_index = int(device_index)
        info = sd.query_devices(self.device_index)
        self.wsjtx_input_hint = (
            f"matching playback of '{info['name']}' "
            f"(e.g. CABLE Output / Cable A Output / BlackHole)"
        )
        # RawOutputStream: we push int16 mono frames from the writer thread.
        self._stream = sd.RawOutputStream(
            samplerate=self.fs,
            channels=1,
            dtype="int16",
            device=self.device_index,
            blocksize=0,
        )
        self._stream.start()
        log.info("PortAudioTransport: writing %d Hz s16 -> [%d] %s",
                 self.fs, self.device_index, info["name"])

    def write(self, pcm: bytes) -> None:
        # sounddevice expects a C-contiguous buffer; bytes is fine for Raw*.
        self._stream.write(pcm)

    def close(self) -> None:
        stream, self._stream = getattr(self, "_stream", None), None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass


# ── Device resolution (PortAudio) ───────────────────────────────────────────

def _list_output_names(sd) -> list[str]:
    names = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            names.append(f"[{i}] {d['name']}")
    return names


def _find_output_device_index(sd, name_substr: str) -> int | None:
    """First output device whose name contains name_substr (case-insensitive)."""
    needle = name_substr.lower().strip()
    if not needle:
        return None
    # Exact-ish: prefer devices where needle is in the name
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) <= 0:
            continue
        if needle in d["name"].lower():
            return i
    return None


def _platform_candidates() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return _MAC_DEVICE_CANDIDATES
    return _WIN_DEVICE_CANDIDATES


def list_matched_output_devices(sd=None) -> list[tuple[int, str]]:
    """Return [(index, name), ...] for outputs matching platform cable patterns."""
    if sd is None:
        import sounddevice as sd
    found: list[tuple[int, str]] = []
    seen = set()
    for cand in _platform_candidates():
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) <= 0:
                continue
            if i in seen:
                continue
            if cand in d["name"].lower():
                found.append((i, d["name"]))
                seen.add(i)
    return found


def resolve_output_device(
    *,
    device_name: str | None = None,
    index: int = 0,
    rf: int | None = None,
) -> str:
    """Resolve a PortAudio output device name for this exporter slot.

    Priority:
      1. Explicit ``device_name`` argument
      2. ``MAP144_WSJTX_DEVICE_RF{n}`` when ``rf`` is set
      3. ``MAP144_WSJTX_DEVICE`` (comma-separated list indexed by ``index``)
      4. Auto-match VB-CABLE / BlackHole / VoiceMeeter names (by ``index``)
    """
    if device_name:
        return device_name.strip()

    if rf is not None:
        env_rf = os.environ.get(f"MAP144_WSJTX_DEVICE_RF{int(rf)}")
        if env_rf:
            return env_rf.strip()

    env = os.environ.get("MAP144_WSJTX_DEVICE", "").strip()
    if env:
        parts = [p.strip() for p in env.split(",") if p.strip()]
        if not parts:
            pass
        elif len(parts) == 1:
            return parts[0]
        else:
            return parts[min(int(index), len(parts) - 1)]

    # Auto-match
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice not installed; pip install sounddevice, or set "
            "MAP144_WSJTX_DEVICE to a PortAudio output name"
        ) from exc

    matched = list_matched_output_devices(sd)
    if not matched:
        raise RuntimeError(
            "No VB-CABLE / BlackHole / VoiceMeeter output device found. "
            "Install a virtual cable, or set MAP144_WSJTX_DEVICE. "
            f"Available outputs: {_list_output_names(sd)}"
        )
    idx = min(int(index), len(matched) - 1)
    return matched[idx][1]


def make_transport(
    *,
    sink_name: str,
    description: str,
    stream_name: str,
    fs: int,
    device_name: str | None = None,
    index: int = 0,
    rf: int | None = None,
    transport: AudioTransport | None = None,
) -> AudioTransport:
    """Build the platform transport (or return an injected one for tests)."""
    if transport is not None:
        return transport
    if sys.platform.startswith("linux"):
        return PipeWireTransport(sink_name, description, stream_name, fs)
    # Windows / macOS / other: PortAudio into a virtual cable
    name = resolve_output_device(device_name=device_name, index=index, rf=rf)
    return PortAudioTransport(device_name=name, fs=fs)


# ── Exporter ────────────────────────────────────────────────────────────────

class WsjtxAudioExporter:
    """Streams one channel's baseband as 12 kHz audio via a platform Transport."""

    def __init__(self, sink_name=None, target_rms=None, fs=_FS, fc=_FC, label=None,
                 description=None, stream_name=None, device_name=None,
                 index: int = 0, rf: int | None = None, transport=None):
        self.sink = sink_name or os.environ.get("MAP144_WSJTX_SINK", "map144_out")
        self.label = label or self.sink
        self.description = description or self.sink
        self.stream_name = stream_name or f"{self.sink} feed"
        self.fs = int(fs)
        self.fc = float(fc)
        self.target_rms = float(target_rms
                                or os.environ.get("MAP144_WSJTX_TARGET_RMS", "26"))
        self.device_name = device_name
        self.index = int(index)
        self.rf = rf
        self._n = 0
        self._rms = None
        self._dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop = threading.Event()
        self._transport: AudioTransport | None = None

        self._transport = make_transport(
            sink_name=self.sink,
            description=self.description,
            stream_name=self.stream_name,
            fs=self.fs,
            device_name=device_name,
            index=self.index,
            rf=rf,
            transport=transport,
        )
        self._thread = threading.Thread(target=self._writer, name="wsjtx-audio",
                                        daemon=True)
        self._thread.start()
        atexit.register(self.close)
        log.info("WsjtxAudioExporter[%s]: streaming -> %s  "
                 "(WSJT-X Input ≈ %s)",
                 self.label, self.sink, self._transport.wsjtx_input_hint)

    # -- hot path (called from the DSP thread; must never block) ----------
    def feed(self, ch_iq):
        """Convert one calling-channel baseband chunk to PCM and enqueue it.

        ch_iq : complex array (n,) at self.fs, signal at DC.
        """
        if self._transport is None:
            return
        n = len(ch_iq)
        if n == 0:
            return
        # phase-continuous upconvert DC -> fc (real audio)
        k = np.arange(self._n, self._n + n, dtype=np.float64)
        audio = np.real(np.asarray(ch_iq) * np.exp(1j * 2 * np.pi * self.fc * k / self.fs))
        self._n += n

        # slow auto-level: gain tracks noise RMS, not the ping, so pings survive
        rms = float(np.sqrt(np.mean(audio * audio)) + 1e-12)
        alpha = min(1.0, n / (_TAU_S * self.fs))
        self._rms = rms if self._rms is None else (1 - alpha) * self._rms + alpha * rms
        gain = min(self.target_rms / max(self._rms, 1e-9), 1.0e7)
        pcm = np.clip(audio * gain, -32767, 32767).astype("<i2").tobytes()

        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(pcm)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.warning("WsjtxAudioExporter: audio pipe backed up, "
                            "dropped %d chunks (WSJT-X may glitch)", self._dropped)

    def _writer(self):
        while not self._stop.is_set():
            try:
                pcm = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if pcm is None:
                break
            transport = self._transport
            if transport is None:
                break
            try:
                transport.write(pcm)
            except (BrokenPipeError, ValueError, OSError, AttributeError) as exc:
                log.warning("WsjtxAudioExporter: transport closed (%s); stopping",
                            exc)
                self._transport = None
                break

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        t, self._transport = self._transport, None
        if t is not None:
            try:
                t.close()
            except Exception:
                pass
