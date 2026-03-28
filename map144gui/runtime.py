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
"""Radio client setup, background thread runtime loop, and shutdown handling.

This module owns the boundary between raw IQ sources (FlexRadio DAXIQ or WAV
file) and the DSP pipeline in ``processing.py``.  Its functions are mixed into
``RadioIQVisualizer`` and cover three concerns: source lifecycle management,
sample ingress normalisation, and clean shutdown.

Signal-level conventions
------------------------
The internal dBFS reference is ±1.0 (0 dBFS = ±1.0 float).  Every source is
normalised to this convention at ingress, close to the hardware driver:

    Flex DAXIQ
        FlexRadio VITA-49 float32 payloads use an ADC full scale of
        approximately ±32768 (16-bit-style values expressed as float32).
        ``FLEX_DAXIQ_FULL_SCALE = 32768.0`` is divided out at ingress in
        ``run_radio_source`` before samples reach the pipeline.

    WAV files
        ``_load_wav_complex`` already normalises to [-1, 1] (int16 ÷ 32768,
        float32 clipped to ±1, etc.).  No further scaling is needed.

    Other SDR sources (AirSpy HF+, RTL-SDR, etc.)
        These deliver float32 in [-1, 1] natively.  Add a new ingress branch
        in ``run_radio_source`` without any additional scaling.

Functions
---------
setup_radio_client(self)
    Imports the ``flexclient`` module at runtime (avoids a hard dependency when
    running in WAV-only mode), instantiates ``FlexDAXIQ`` with the configured
    centre frequency, sample rate, and DAX channel, then starts a
    ``QtCore.QThread`` running ``run_radio_source``.

run_radio_source(self)
    Background thread main loop.  Polls ``source_mode`` on every iteration:

    * **"wav"** mode — stops the radio client if running, then calls
      ``_process_wav_source_step`` which loads (or reloads) the WAV file,
      feeds one chunk per real-time sleep, and wraps the file when exhausted.
    * **"radio"** mode — starts the radio client (once), drains packets from
      ``radio_client.sample_queue``, divides by ``FLEX_DAXIQ_FULL_SCALE``
      to normalise FlexRadio's ±32768 ADC range to the ±1.0 internal
      convention, then calls ``process_iq_data``.

    The loop is tolerant of transient errors (prints traceback, sleeps briefly,
    retries) and exits cleanly when ``self.running`` is set to False by
    ``closeEvent``.

_load_wav_complex(path, target_rate)
    Reads a WAV file of any common format (8-bit unsigned, 16-bit signed,
    32-bit int or 32-bit float) and returns complex64 IQ:

    * **Mono** files — the real (mono) channel is converted to analytic signal
      via ``scipy.signal.hilbert`` to create a single-sideband complex
      representation.  Without this, setting Q=0 produces a symmetric spectrum
      whose squared-domain version yields mirror-image tone pairs at negative
      frequencies, causing the MSK144 detector to recover a negative fc.
    * **Stereo** files — channel 0 is I, channel 1 is Q.

    Sample rate conversion to ``target_rate`` is done by
    ``_resample_linear`` (linear interpolation; adequate for display and
    detection but not for high-fidelity decode — the decimation in
    ``detection.extract_and_decode`` handles that path separately).

_resample_linear(samples, src_rate, dst_rate)
    Resample a complex array by linear interpolation of I and Q independently.
    Fast and allocation-light for moderate rate ratios; passband ripple is
    acceptable for the 48 kHz display pipeline.

_process_wav_source_step(self)
    Called repeatedly by ``run_radio_source`` in WAV mode.  Loads or reloads
    the file when ``selected_wav_path`` changes, feeds ``fft_size × 4``-sample
    chunks (enough for two full FFT blocks with 50 % overlap), wraps the index
    at end-of-file for continuous replay, and sleeps for the real-time
    equivalent of the chunk duration to avoid spinning the CPU.

_reset_wav_timeline(self)
    Clears all spectrogram staging buffers, noise-floor estimates, energy
    buffers, ring buffer, LP-filter state, and boundary counters back to their
    initial sentinel values (−130 dBFS / NaN / 0).  Called whenever a new WAV
    file is loaded so stale data from a previous file does not bleed into the
    new display.

_start_radio_source(self) / _stop_radio_source(self)
    Thin wrappers around ``radio_client.start()`` / ``radio_client.stop()`` with
    error handling and the ``_radio_started`` guard to prevent double-starts.

_get_tuned_frequency_mhz(self)
    Reads the current centre frequency from the radio client's ``_dax_setup``
    object (prefers Slice frequency over Pan centre) and returns a 3-tuple
    ``(freq_mhz, source_label, bandwidth_hz)``.  In WAV mode returns
    ``(center_freq_mhz, "WAV", None)``.  Called by ``displays.py`` on every
    refresh to keep frequency axis labels and the window title current.

closeEvent(self, event)
    Qt close handler: persists window geometry and dB-scale slider values to
    ``QSettings``, sets ``self.running = False`` to stop the background thread,
    stops the display timer, stops the radio client, waits up to 2 s for the
    thread to exit (terminates forcibly if it does not), then accepts the event.
"""

import queue
import time
import wave
import importlib

import numpy as np
from scipy.signal import hilbert

from PyQt5 import QtCore

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

# FlexRadio DAXIQ float32 payloads use ADC full scale ≈ ±32768.
# Dividing by this constant at ingress normalises samples to the ±1.0 internal standard.
FLEX_DAXIQ_FULL_SCALE = 32768.0


def setup_radio_client(self):
    """Start the runtime thread.  The DAXIQ connection is deferred until
    the user selects 'Flex Radio' from the File menu."""
    self.radio_client = None

    self.client_thread = QtCore.QThread()
    self.client_thread.run = self.run_radio_source
    self.client_thread.setTerminationEnabled(True)
    self.client_thread.start()


def _connect_radio_client(self):
    """Instantiate FlexDAXIQ on first use (called from on_select_source_radio)."""
    if self.radio_client is not None:
        return   # already connected
    try:
        flex_client_module = importlib.import_module('flexclient')
        flex_client_class = flex_client_module.FlexDAXIQ
        self.radio_client = flex_client_class(
            center_freq_mhz=self.center_freq_mhz,
            sample_rate=self.sample_rate,
            dax_channel=1,
            bind_client_id=getattr(self, 'bind_client_id', None),
        )
        print("[radio] FlexDAXIQ connected", flush=True)
    except Exception as exc:
        print(f"[radio] FlexDAXIQ connection failed: {exc}", flush=True)
        self.radio_client = None


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.complex64)
    if samples.size == 0:
        return np.array([], dtype=np.complex64)

    out_len = int(round(samples.size * (dst_rate / src_rate)))
    old_x = np.arange(samples.size, dtype=np.float64)
    new_x = np.linspace(0, samples.size - 1, out_len, dtype=np.float64)

    re = np.interp(new_x, old_x, np.real(samples).astype(np.float64))
    im = np.interp(new_x, old_x, np.imag(samples).astype(np.float64))
    return (re + 1j * im).astype(np.complex64)


def _load_wav_complex(path: str, target_rate: int) -> tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        src_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        float_try = np.frombuffer(raw, dtype=np.float32)
        if np.all(np.isfinite(float_try)) and np.max(np.abs(float_try)) <= 10:
            data = float_try.astype(np.float32)
            max_abs = float(np.max(np.abs(data))) if data.size else 0.0
            if max_abs > 1.0:
                data /= max_abs
        else:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width}")

    if channels == 1:
        # Convert real mono to analytic (single-sideband) IQ.
        # Setting Q=0 leaves a symmetric spectrum whose squared version produces
        # mirror-image tone pairs at negative frequencies, causing the detector to
        # recover a negative fc and mix the signal to the wrong frequency.
        iq = hilbert(data.astype(np.float64)).astype(np.complex64)
    else:
        frames = data.reshape(-1, channels)
        i = frames[:, 0]
        q = frames[:, 1]
        iq = (i + 1j * q).astype(np.complex64)

    iq = _resample_linear(iq, src_rate, target_rate)
    return iq.astype(np.complex64), target_rate


def _start_radio_source(self) -> bool:
    if self._radio_started:
        return True
    if self.radio_client is None:
        return False
    try:
        self.radio_client.start()
        self._radio_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        print(f"Radio client start error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            self.radio_client.stop()
        except Exception:
            pass
        return False


def _stop_radio_source(self):
    if not self._radio_started:
        return
    try:
        self.radio_client.stop()
    except Exception:
        pass
    self._radio_started = False


def _reset_wav_timeline(self):
    self._wav_time_cursor = 0.0
    self._sbuf_end = 0
    self._ch_buf_end = 0
    self._metric_hist_idx = 0
    self._metric_hist_cnt = 0
    self.raw_buffer    = np.array([], dtype=np.complex64)

    self.spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
    self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)
    self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)

    self.spec_staging_filled = False
    self.realtime_filled = False
    self.accumulated_noise_floor = np.full(self.fft_size, -125.0)
    self.realtime_noise_floor = np.full(self.fft_size, -125.0)

    self.spec_boundary = 0
    self._realtime_boundary = 0

    self.spec_write_index = 0
    self.realtime_write_index = 0

    self.sq_spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_spec_staging = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_spec_staging_filled = False
    self.sq_realtime_data = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_realtime_filled = False
    self.sq_spec_boundary = 0
    self._sq_realtime_boundary = 0
    self.sq_spec_write_index = 0
    self.sq_realtime_write_index = 0

    self.time_in_window = 0.0
    self.next_boundary = self.history_secs

    # Reset ring buffer, detection state, and SNR heatmap history
    self._ch_snr_history[:] = 0.0
    self._ch_snr_write_idx  = 0
    self._ch_snr_boundary   = 0
    self._iq_ring[:] = 0
    self._iq_ring_pos = 0
    self._iq_abs_sample = 0
    self._detect_cooldowns = {}
    self._iq_ring_gen = getattr(self, '_iq_ring_gen', 0) + 1
    if hasattr(self, '_jt9_markers'):
        self._jt9_markers.clear()
    if hasattr(self, 'decode_panel'):
        self.decode_panel.clear()


def _process_wav_source_step(self):
    wav_path = self.selected_wav_path
    if not wav_path:
        time.sleep(0.1)
        return

    nonce = getattr(self, '_wav_load_nonce', 0)
    nonce_loaded = getattr(self, '_wav_nonce_loaded', -1)

    if self._wav_samples is None or self._wav_path_loaded != wav_path or nonce_loaded != nonce:
        try:
            samples, _ = _load_wav_complex(wav_path, self.sample_rate)
            self._wav_samples = samples
            self._wav_path_loaded = wav_path
            self._wav_index = 0
            self._wav_done = False
            self._wav_nonce_loaded = nonce
            self._wav_run_start_time = datetime.now(timezone.utc)
            _reset_wav_timeline(self)
            print(f"Loaded WAV source: {wav_path} ({len(samples)} samples @ {self.sample_rate} Hz)", flush=True)
        except Exception as exc:
            print(f"WAV load error: {exc}", flush=True)
            time.sleep(0.5)
            return

    if self._wav_samples is None or len(self._wav_samples) == 0:
        time.sleep(0.1)
        return

    # Pause when playback has finished (play once, no wrap)
    if getattr(self, '_wav_done', False):
        time.sleep(0.1)
        return

    chunk_size = self.fft_size * 4
    start = self._wav_index
    end = start + chunk_size

    if end > len(self._wav_samples):
        # Final partial chunk
        chunk = self._wav_samples[start:]
        self._wav_index = len(self._wav_samples)
        if chunk.size > 0:
            chunk = chunk.astype(np.complex64)
            wav_seconds = float(self._wav_time_cursor)
            ts_int = int(wav_seconds)
            ts_frac = int((wav_seconds - ts_int) * 1e12)
            self.process_iq_data(chunk, ts_int, ts_frac)
            time.sleep(chunk.size / self.sample_rate)
        # Mark done and trigger comparison
        self._wav_done = True
        print(f"WAV playback complete: {wav_path}", flush=True)
        threading.Thread(
            target=_run_wav_comparison,
            args=(wav_path, getattr(self, '_wav_run_start_time', None),
                  list(getattr(self, '_jt9_threads', []))),
            daemon=True,
        ).start()
        return

    chunk = self._wav_samples[start:end].astype(np.complex64)
    self._wav_index = end

    # WAV samples are already in [-1, 1] from _load_wav_complex — no ingress scaling needed.
    wav_seconds = float(self._wav_time_cursor)
    ts_int = int(wav_seconds)
    ts_frac = int((wav_seconds - ts_int) * 1e12)
    self.process_iq_data(chunk, ts_int, ts_frac)
    time.sleep(chunk_size / self.sample_rate)


def _run_wav_comparison(wav_path: str, run_start: datetime | None,
                        jt9_threads: list | None = None) -> None:
    """Read manifest alongside wav_path, compare against decode log, write report.

    Runs in a daemon thread after WAV playback completes.  Writes a timestamped
    report to MSK144/detections/ with a per-simulation-file sequence number.
    """
    # Wait for all jt9 decode threads that were running when WAV finished.
    # jt9 has a 20-second timeout; cap our wait at 25 s to avoid hanging forever.
    if jt9_threads:
        print(f"[compare] Waiting for {len(jt9_threads)} pending jt9 thread(s)...", flush=True)
        for t in jt9_threads:
            t.join(timeout=25.0)

    wav_p = Path(wav_path)
    manifest_p = wav_p.with_suffix('.json')
    if not manifest_p.exists():
        print(f"[compare] No manifest found at {manifest_p}, skipping comparison", flush=True)
        return

    try:
        manifest = json.loads(manifest_p.read_text())
    except Exception as exc:
        print(f"[compare] Failed to read manifest: {exc}", flush=True)
        return

    placements = manifest.get("placements", [])

    # Read decode log and launch log, filtered to entries since run_start
    detections_dir = Path(__file__).parent.parent / 'MSK144' / 'detections'

    def _read_jsonl(path):
        entries = []
        if not path.exists():
            return entries
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_start is not None:
                    ts_str = entry.get("timestamp", "")
                    try:
                        entry_dt = datetime.fromisoformat(ts_str)
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                        if entry_dt < run_start:
                            continue
                    except ValueError:
                        pass
                entries.append(entry)
        return entries

    log_path     = detections_dir / 'decodes.jsonl'
    launch_path  = detections_dir / 'launches.jsonl'
    decodes      = _read_jsonl(log_path)
    launches     = _read_jsonl(launch_path)

    # Build message lookup — index by full line and by each whitespace token
    # so that 13-char free-text messages match regardless of jt9 field prefix.
    decode_by_msg: dict[str, list[dict]] = {}
    for dec in decodes:
        raw = dec.get("message", "").strip()
        decode_by_msg.setdefault(raw, []).append(dec)
        for token in raw.split():
            if token != raw:
                decode_by_msg.setdefault(token, []).append(dec)

    freq_tol_khz = 2.0
    time_tol_s   = 1.5

    def _pos_match(pl):
        centre_khz = pl.get("center_hz", 0.0) / 1000.0
        delay_s    = pl.get("delay_s", 0.0)
        best, best_dist = None, float("inf")
        for dec in decodes:
            df = abs(dec.get("fc_khz", float("inf")) - centre_khz)
            dt = abs(dec.get("t_sec",  float("inf")) - delay_s)
            if df <= freq_tol_khz and dt <= time_tol_s:
                dist = (df / freq_tol_khz) ** 2 + (dt / time_tol_s) ** 2
                if dist < best_dist:
                    best_dist, best = dist, dec
        return best

    lines = []
    lines.append(f"MSK144 Comparison Report")
    lines.append(f"Simulation : {wav_p.name}")
    lines.append(f"Manifest   : {manifest_p.name}")
    lines.append(f"Generated  : {manifest.get('generated_at', 'unknown')}")
    lines.append(f"Run start  : {run_start.isoformat() if run_start else 'unknown'}")
    if 'noise_floor_dbfs' in manifest:
        lines.append(f"Noise floor: {manifest['noise_floor_dbfs']} dBFS   "
                     f"Atten: {manifest.get('atten_db', '?')} dB")
    else:
        lines.append(f"AWGN SNR   : {manifest.get('awgn_snr_db', '?')} dB   "
                     f"Atten: {manifest.get('atten_db', '?')} dB")
    n_launched   = len(launches)
    n_no_decode  = sum(1 for l in launches if l.get("outcome") == "no_decode")
    n_timeout    = sum(1 for l in launches if l.get("outcome") == "timeout")
    lines.append(f"Decode log : {log_path}  ({len(decodes)} entries in this run)")
    lines.append(f"jt9 launches: {n_launched} total  "
                 f"({len(decodes)} decoded  {n_no_decode} no_decode  {n_timeout} timeout)")
    lines.append("")
    lines.append("Message: A [P/N] ff ttt [P/N] dd www  — freq±kHz  time-ds  SNR±dB  width-ms")
    lines.append("")

    col_w = 13
    lines.append(f"{'#':>3}  {'Message':<{col_w}}  {'t (s)':>6}  {'fc (kHz)':>8}  "
                 f"{'gen SNR':>7}  {'wid ms':>7}  Status")
    lines.append("-" * 70)

    n_decoded = n_missed = n_garbled = 0
    matched_ids: set[int] = set()

    def _snr_str(v):
        return f"{v:>+4d} dB" if v is not None else "      ?"

    placements = sorted(placements, key=lambda p: p.get("delay_s", 0.0))

    for i, pl in enumerate(placements):
        msg       = pl.get("msg", "")
        c_khz     = pl.get("center_hz", 0.0) / 1000.0
        delay_s   = pl.get("delay_s", 0.0)
        snr_db    = pl.get("snr_db")
        width_ms  = pl.get("width_ms")
        gen_snr   = _snr_str(snr_db)
        width_str = f"{width_ms:>4d} ms" if width_ms is not None else "     ?"
        prefix    = f"{i+1:>3}  {msg:<{col_w}}  {delay_s:>6.2f}  {c_khz:>8.2f}  {gen_snr}  {width_str}"

        hits = decode_by_msg.get(msg, [])
        if hits:
            dec = hits[0]
            # Mark every decode of this message as matched so repeated detections
            # of the same signal don't appear as false alarms.
            for h in hits:
                matched_ids.add(id(h))
            det_t = dec.get('t_sec')
            extra = f"  ×{len(hits)}" if len(hits) > 1 else ""
            lines.append(f"{prefix}  DECODED{extra}   "
                         f"(t={det_t:.2f}s  fc={dec.get('fc_khz',float('nan')):.2f} kHz)")
            n_decoded += 1
        else:
            near = _pos_match(pl)
            if near and id(near) not in matched_ids:
                matched_ids.add(id(near))
                det_t = near.get('t_sec')
                lines.append(f"{prefix}  GARBLED   "
                             f"(jt9: '{near.get('message','')}'"
                             f"  t={det_t:.2f}s  fc={near.get('fc_khz',float('nan')):.2f} kHz)")
                n_garbled += 1
            else:
                lines.append(f"{prefix}  MISSED")
                n_missed += 1

    false_alarms = [d for d in decodes if id(d) not in matched_ids]

    lines.append("")
    lines.append("─" * 70)
    total = len(placements)
    pct   = 100 * n_decoded / total if total else 0
    lines.append(f"Total signals  : {total}")
    lines.append(f"Decoded        : {n_decoded:>3}  ({pct:.0f}%)")
    lines.append(f"Missed         : {n_missed:>3}")
    if n_garbled:
        lines.append(f"Garbled        : {n_garbled:>3}  (detected but message wrong)")
    lines.append(f"jt9 launches   : {n_launched:>3}  "
                 f"({len(decodes)} decoded  {n_no_decode} stray  {n_timeout} timeout)")
    if false_alarms:
        lines.append(f"False alarms   : {len(false_alarms):>3}  (decoded with no matching signal)")
        for dec in false_alarms:
            jsnr  = dec.get('jt9_snr_db')
            snr_s = f"  snr={jsnr:+d} dB" if jsnr is not None else ""
            lines.append(f"    t={dec.get('t_sec',0.0):.3f}s  "
                         f"fc={dec.get('fc_khz',0.0):.3f} kHz"
                         f"{snr_s}  '{dec.get('message','')}'")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text, flush=True)

    # Write report file: {wav_stem}_run{seq:03d}.txt in detections dir
    detections_dir.mkdir(parents=True, exist_ok=True)
    stem = wav_p.stem
    existing = sorted(detections_dir.glob(f"{stem}_run*.txt"))
    seq = len(existing) + 1
    report_path = detections_dir / f"{stem}_run{seq:03d}.txt"
    report_path.write_text(report_text)
    print(f"[compare] Report written: {report_path}", flush=True)


def run_radio_source(self):
    """Run selected source (radio or WAV file) and feed processing pipeline."""
    while self.running:
        try:
            if self.source_mode == "idle":
                if self._radio_started:
                    _stop_radio_source(self)
                time.sleep(0.1)
                continue

            if self.source_mode == "wav":
                if self._radio_started:
                    _stop_radio_source(self)
                _process_wav_source_step(self)
                continue

            if not _start_radio_source(self):
                time.sleep(1.0)
                continue

            try:
                packet = self.radio_client.sample_queue.get(timeout=1.0)
                # Normalise FlexRadio DAXIQ samples from ±32768 ADC scale to ±1.0.
                chunk = (np.asarray(packet.samples, dtype=np.complex64)
                         / FLEX_DAXIQ_FULL_SCALE)
                self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"Queue get/process error: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        except Exception as exc:
            print(f"Source loop error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(0.25)

    _stop_radio_source(self)


def _get_tuned_frequency_mhz(self):
    """Return current tuned frequency and source label from radio client status."""
    if self.source_mode == "wav":
        return self.center_freq_mhz, "WAV", None

    tuned_freq_mhz = None
    tuned_source = None
    tuned_bandwidth_hz = None
    if hasattr(self, 'radio_client') and self.radio_client:
        dax_setup = getattr(self.radio_client, '_dax_setup', None)
        if dax_setup:
            slice_freq = getattr(dax_setup, 'slice_frequency_mhz', None)
            pan_freq = getattr(dax_setup, 'pan_frequency_mhz', None)
            pan_bw = getattr(dax_setup, 'pan_bandwidth_hz', None)
            if slice_freq is not None:
                tuned_freq_mhz = slice_freq
                tuned_source = 'Slice'
            elif pan_freq is not None:
                tuned_freq_mhz = pan_freq
                tuned_source = 'Pan'
                tuned_bandwidth_hz = pan_bw
    return tuned_freq_mhz, tuned_source, tuned_bandwidth_hz


def closeEvent(self, event):
    """Clean up on window close."""
    from .visualizer import _SETTINGS

    # Signal panel windows that this is a real shutdown so their closeEvent
    # does not ignore the event and block teardown.
    self._app_closing = True

    _SETTINGS.setValue('window_geometry',      self.saveGeometry())
    _SETTINGS.setValue('min_level',            self.min_level)
    _SETTINGS.setValue('max_level',            self.max_level)
    _SETTINGS.setValue('detect_min_level',     self.detect_min_level)
    _SETTINGS.setValue('detect_max_level',     self.detect_max_level)
    _SETTINGS.setValue('nb_factor',            self.nb_factor)
    _td_sl = getattr(self, 'td_scale_slider', None)
    _SETTINGS.setValue('td_scale',             _td_sl.value() if _td_sl else 10)
    _SETTINGS.setValue('td_span',              int(getattr(self, 'td_span_ms', 200)))

    for win, geo_key, vis_key in [
        (getattr(self, '_fast_graph_win',    None), 'fast_graph_geometry',  'fast_graph_visible'),
        (getattr(self, '_detect_win',        None), 'detect_geometry',      'detect_visible'),
        (getattr(self, '_radio_iface_win',   None), 'radio_iface_geometry', 'radio_iface_visible'),
    ]:
        if win is not None:
            _SETTINGS.setValue(geo_key, win.saveGeometry())
            _SETTINGS.setValue(vis_key, win.isVisible())

    print("Shutting down...")
    self.running = False

    if hasattr(self, 'update_timer'):
        self.update_timer.stop()

    if hasattr(self, 'radio_client'):
        _stop_radio_source(self)

    if hasattr(self, 'client_thread') and self.client_thread.isRunning():
        self.client_thread.quit()
        if not self.client_thread.wait(2000):
            print("Thread did not exit cleanly, terminating...")
            self.client_thread.terminate()
            self.client_thread.wait()

    print("Shutdown complete")
    event.accept()
