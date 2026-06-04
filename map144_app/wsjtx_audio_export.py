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

Deployment: for the B210 on Linux this runs by default -- runtime
``_setup_wsjtx_audio_export`` creates one exporter per RF port at source start,
with sink names ``map144_RF0`` / ``map144_RF1`` (RF0 <- RX0, RF1 <- RX1).  The
sinks are named by hardware port (RF0/RF1), NOT polarization, since the antenna
on each port is operator-dependent.  Disable the whole bridge with
``MAP144_WSJTX_AUDIO=0``.

Optional env tuning (apply to every port):
  MAP144_WSJTX_TARGET_RMS  int16 RMS level  (default 26 -> ~30 dB on the WSJT-X
                                             meter, calibrated on a B210 feed)
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

    def __init__(self, sink_name=None, target_rms=None, fs=_FS, fc=_FC, label=None,
                 description=None, stream_name=None):
        self.sink = sink_name or os.environ.get("MAP144_WSJTX_SINK", "map144_out")
        self.label = label or self.sink     # short tag for logs (e.g. 'RF0')
        # Human-readable strings.  device.description is the ONLY label WSJT-X /
        # pavucontrol show the operator (the programmatic `sink` name with its
        # dots/underscores never reaches a GUI).  The monitor source WSJT-X
        # records is auto-named "Monitor of <description>".  stream_name labels
        # the paplay playback stream itself (visible in pavucontrol's Playback
        # tab) so the input->process->output chain is self-documenting there.
        self.description = description or self.sink
        self.stream_name = stream_name or f"{self.sink} feed"
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
        self._module_id = None        # pactl module index → unload on close

        self._ensure_sink()
        self._open_player()
        self._thread = threading.Thread(target=self._writer, name="wsjtx-audio",
                                        daemon=True)
        self._thread.start()
        atexit.register(self.close)
        log.info("WsjtxAudioExporter[%s]: streaming -> sink '%s'  "
                 "(WSJT-X Input = 'Monitor of %s')",
                 self.label, self.sink, self.description)

    # -- setup ------------------------------------------------------------
    def _sh(self, cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def _ensure_sink(self):
        # Remove any stale sink with our name first (e.g. left by a prior crash
        # where atexit didn't run) so we own a fresh module we can cleanly
        # unload on close.  The sink's lifetime now matches the bridge's: no
        # object.linger, so unload-module removes it and a dead bridge
        # *disappears* from WSJT-X instead of going silently mute.
        self._unload_existing()
        # List-form (shell=False) so the description may contain spaces and the
        # "->" arrow without the shell splitting args or treating ">" as a
        # redirect.  pactl joins these argv into one module-arg string and
        # re-tokenises on spaces, so a spaced value needs *nested* quoting:
        # outer double-quotes make the whole proplist the value of
        # sink_properties; inner single-quotes keep the spaced description as one
        # property.  (Verified: the single-level form truncates at the first
        # space.)  Assumes the description carries no single-quote, which holds
        # for our "MAP144 RFn RX -> WSJT-X" labels.
        cmd = ["pactl", "load-module", "module-null-sink",
               f"sink_name={self.sink}", "media.class=Audio/Sink",
               f"sink_properties=\"device.description='{self.description}'\""]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"could not create null sink '{self.sink}': "
                               f"{r.stderr.strip()}")
        self._module_id = r.stdout.strip() or None

    def _unload_existing(self):
        """Unload any module-null-sink already owning our sink name."""
        out = self._sh("pactl list short modules").stdout
        for ln in out.splitlines():
            if "module-null-sink" in ln and f"sink_name={self.sink}" in ln:
                mod = ln.split()[0]
                self._sh(f"pactl unload-module {mod}")

    def _open_player(self):
        if shutil.which("paplay"):
            # --client-name/--stream-name label this playback stream in
            # pavucontrol's Playback tab (and `pactl list sink-inputs`), so the
            # producer side of the chain is named, not just the sink.
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
            # Snapshot: close() runs on the main thread at shutdown and sets
            # self._proc = None; without the local ref a non-None pcm pulled
            # just before that nulling would hit None.stdin (AttributeError,
            # uncaught) and crash this daemon thread during teardown.
            proc = self._proc
            if proc is None:
                break
            try:
                proc.stdin.write(pcm)
            except (BrokenPipeError, ValueError, OSError, AttributeError):
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
        # Unload the null sink so the bridge's device disappears from WSJT-X
        # (rather than lingering as a silent monitor) the moment the B210 is no
        # longer the active source.
        mod, self._module_id = self._module_id, None
        if mod:
            try:
                self._sh(f"pactl unload-module {mod}")
            except Exception:
                pass
