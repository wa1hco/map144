"""Phase 1 of the MAP144 -> WSJT-X audio bridge: live calling-channel export.

Taps the calling channel's complex baseband (``ch_out[0]``, DC-centred, 12 kHz),
upconverts it to real audio at 1500 Hz (phase-continuous across chunks), and
streams 12 kHz signed-16 PCM into a PipeWire/Pulse null sink.  PipeWire resamples
12 -> 48 kHz and adaptively syncs to the graph clock, absorbing the B210-vs-system
clock drift for free.  WSJT-X records the sink's monitor and decodes as if from a
normal soundcard.

Why this is a *sink*, not a DSP stage:
  * It only reads ``ch_out[0]``; it never modifies the pipeline (read-only tap,
    a proto- ``WsjtxAudioExportBlock`` for the block/stream migration).
  * The PCM is handed to a background writer thread through a bounded
    drop-oldest queue, so a stalled audio pipe drops audio rather than blocking
    the DSP ingest loop (which would corrupt MAP144's own decode and the A/B).

Enable with ``MAP144_WSJTX_AUDIO=1``.  Optional env tuning:
  MAP144_WSJTX_SINK        sink name        (default map144_out)
  MAP144_WSJTX_TARGET_RMS  int16 RMS level  (default 26 -> ~30 dB on the WSJT-X
                                             meter, calibrated on a B210 H feed)
  MAP144_WSJTX_SOURCE      'h' or 'v'       (default h; which B210 channel)
"""
import os
import queue
import shutil
import atexit
import logging
import threading
import subprocess
import numpy as np

log = logging.getLogger(__name__)

_FS = 12000          # calling-channel sample rate (Hz)
_FC = 1500.0         # audio carrier WSJT-X expects (Hz)
_TAU_S = 8.0         # auto-level time constant (>> ping duration, so pings survive)
_QUEUE_MAX = 128     # ~chunks buffered before drop-oldest (bounds latency + memory)


class WsjtxAudioExporter:
    """Streams one channel's baseband to a PipeWire null sink as 12 kHz audio."""

    def __init__(self, sink_name=None, target_rms=None, fs=_FS, fc=_FC):
        self.sink = sink_name or os.environ.get("MAP144_WSJTX_SINK", "map144_out")
        self.fs = int(fs)
        self.fc = float(fc)
        # 26 int16-RMS lands the WSJT-X meter at ~30 dB (the recommended noise
        # level) -- calibrated live on a B210 H feed.  Auto-level pins output RMS
        # to this regardless of input, so the meter reading is deterministic.
        self.target_rms = float(target_rms
                                or os.environ.get("MAP144_WSJTX_TARGET_RMS", "26"))
        self._n = 0                  # running sample index -> phase continuity
        self._rms = None             # EMA of pre-gain audio RMS (auto-level)
        self._dropped = 0
        self._q = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop = threading.Event()
        self._proc = None

        self._ensure_sink()
        self._open_player()
        self._thread = threading.Thread(target=self._writer, name="wsjtx-audio",
                                        daemon=True)
        self._thread.start()
        atexit.register(self.close)
        log.info("WsjtxAudioExporter: streaming calling channel -> '%s' "
                 "(record 'Monitor of %s' in WSJT-X)", self.sink, self.sink)

    # -- setup ------------------------------------------------------------
    def _sh(self, cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def _ensure_sink(self):
        out = self._sh("pactl list short sinks").stdout
        if any(self.sink in ln for ln in out.splitlines()):
            return
        r = self._sh(f"pactl load-module module-null-sink sink_name={self.sink} "
                     f"object.linger=1 media.class=Audio/Sink "
                     f"sink_properties=device.description={self.sink}")
        if r.returncode != 0:
            raise RuntimeError(f"could not create null sink '{self.sink}': "
                               f"{r.stderr.strip()}")

    def _open_player(self):
        if shutil.which("paplay"):
            cmd = ["paplay", f"--device={self.sink}", "--raw",
                   "--format=s16le", f"--rate={self.fs}", "--channels=1"]
        elif shutil.which("pw-play"):
            cmd = ["pw-play", f"--target={self.sink}", "--format=s16",
                   f"--rate={self.fs}", "--channels=1", "-"]
        else:
            raise RuntimeError("WSJT-X audio export needs 'paplay' or 'pw-play' "
                               "(install pulseaudio-utils or pipewire-bin)")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # -- hot path (called from the DSP thread; must never block) ----------
    def feed(self, ch_iq):
        """Convert one calling-channel baseband chunk to PCM and enqueue it.

        ch_iq : complex array (n,) at self.fs, signal at DC.
        """
        if self._proc is None:
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
            try:                       # drop oldest, keep latency bounded
                self._q.get_nowait()
                self._q.put_nowait(pcm)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.warning("WsjtxAudioExporter: audio pipe backed up, "
                            "dropped %d chunks (WSJT-X may glitch)", self._dropped)

    # -- background writer (may block on the pipe; off the DSP thread) ----
    def _writer(self):
        while not self._stop.is_set():
            try:
                pcm = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if pcm is None:
                break
            try:
                self._proc.stdin.write(pcm)
            except (BrokenPipeError, ValueError, OSError):
                log.warning("WsjtxAudioExporter: audio player closed; stopping export")
                self._proc = None
                break

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.stdin.close()
            except Exception:
                pass
            p.terminate()
