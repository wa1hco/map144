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
"""Offline IQ replay engine for the Analysis Window.

AnalysisEngine(Engine) replays a 15-second IQ capture through the exact same
DSP pipeline (noise blanker → channeliser → per-channel detection → waterfall
FFT) that the live MAP144Visualizer uses, but:

  * No jt9 decode threads are spawned (_analysis_no_decode = True).
  * Blanked NB-block timestamps are logged to _nb_blank_log for overlay display.
  * The replay is driven synchronously at sample-accurate timestamps so the
    spectrogram rows are placed correctly, just like WAV source mode.

AnalysisWorker(QThread) wraps run_replay() for non-blocking use in the Qt GUI.
It emits finished(results) when done or error(message) on failure.
"""

import queue

import numpy as np
from PyQt5 import QtCore

from .engine import Engine
from .processing import NB_FFT_SIZE


# ── Helpers for the analysis-window freq-vs-time plot ────────────────────────


def fc_mhz_calling(engine) -> float:
    """Calling-freq estimate for ``engine``.  AnalysisEngine sets
    ``center_freq_mhz`` to the calling freq during __init__."""
    return float(getattr(engine, 'center_freq_mhz', 50.260))


def _analysis_freq_vs_time(iq: np.ndarray, sample_rate_hz: int,
                            win_s: float = 0.064,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squared-FFT instantaneous carrier estimator on complex IQ.

    Same algorithm as ``tools/measure_ping_freq_vs_time.py::_instantaneous_freq``
    but operates on the 48 kHz wideband IQ directly.  For each sliding
    window: square the IQ (multiplies all freqs by 2), FFT, find the two
    MSK tone images at 2·(f_calling ± 500 Hz), parabolic-interpolate each
    bin, recover carrier as midpoint/2.

    Returns
    -------
    f_carrier_hz : (n_windows,) — instantaneous carrier freq estimate,
                   in Hz absolute (i.e. centred near calling freq, not 0).
    t_centre_s   : (n_windows,) — window-centre time, seconds from start of IQ.
    env_db       : (n_windows,) — magnitude envelope in dB above per-window median.
    """
    if iq.ndim != 1:
        iq = iq.reshape(-1)
    fs = float(sample_rate_hz)
    n_win = max(256, int(win_s * fs) // 2 * 2)
    step  = n_win // 4   # 25 % overlap
    n_samp = iq.shape[0]
    n_steps = max(0, (n_samp - n_win) // step + 1)
    if n_steps == 0:
        return (np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32))

    sq = (iq.astype(np.complex64)) ** 2   # complex squared → tones at 2·f_carrier ± 1000 Hz
    hann = np.hanning(n_win).astype(np.float32)
    df = fs / n_win

    # Target tones in the squared spectrum (complex FFT covers ±fs/2).
    # Reference is "audio offset relative to calling freq" — here calling freq
    # is the pan-centre + channel_offset.  AnalysisEngine sets center_freq_mhz
    # to the calling freq, so for the offline path we treat the IQ as already
    # centred on calling freq and look for tones at ±1000 Hz of DC.  (The 48 kHz
    # capture IS centred on the pan; the calling freq is offset.  We compensate
    # by mixing the carrier-offset out, see below.)
    # Compensate for the channel_offset_hz between pan and calling:
    # AnalysisEngine.center_freq_mhz IS the calling freq, so a mix is needed.
    # For simplicity we estimate tones at ±1000 Hz of fc=0 in the de-mixed IQ.
    # (For the operator's Flex pan=50.259 + calling=50.260, the offset is
    # +1000 Hz, which we don't have access to here; passing fc_mhz tells us
    # nothing about pan.)  Instead: search across the natural complex
    # spectrum for two strong peaks 2 kHz apart and average them.
    n_freq = n_win
    bin_search_half = int(round(3000.0 / df))   # ±3 kHz search either side

    f_carr  = np.zeros(n_steps, dtype=np.float32)
    t_c     = np.zeros(n_steps, dtype=np.float32)
    env_lin = np.zeros(n_steps, dtype=np.float32)

    for k in range(n_steps):
        start = k * step
        seg = sq[start:start + n_win] * hann
        spec = np.abs(np.fft.fftshift(np.fft.fft(seg)))
        env_lin[k] = float(np.mean(np.abs(iq[start:start + n_win])))
        # Find the two tallest peaks within the search range.
        centre = n_freq // 2
        lo_idx = max(0, centre - bin_search_half)
        hi_idx = min(n_freq, centre + bin_search_half)
        sub = spec[lo_idx:hi_idx]
        if sub.size < 4:
            continue
        peak1 = int(np.argmax(sub))
        # Zero out a window around peak1, find peak2.
        guard = max(2, int(round(500.0 / df)))
        sub2 = sub.copy()
        sub2[max(0, peak1 - guard):peak1 + guard + 1] = 0
        if not np.any(sub2):
            continue
        peak2 = int(np.argmax(sub2))

        def _parabolic(arr, i):
            if 1 <= i < arr.size - 1:
                a, b, c = float(arr[i-1]), float(arr[i]), float(arr[i+1])
                den = (a - 2*b + c)
                if den != 0:
                    return i + 0.5 * (a - c) / den
            return float(i)

        f1 = (_parabolic(sub, peak1) + lo_idx - centre) * df
        f2 = (_parabolic(sub, peak2) + lo_idx - centre) * df
        # In the squared-IQ spectrum, MSK tones appear at 2·(fc ± 500).
        # The carrier estimate is the midpoint / 2.
        f_carr[k] = float((f1 + f2) / 4.0)   # /2 for squaring, /2 for midpoint
        t_c[k]    = (start + n_win / 2) / fs

    # Envelope (|IQ|) in dB above the per-burst 25th-percentile noise floor.
    # pct25 (vs median) is more robust to signal contamination — burst windows
    # don't pull the reference up when the signal occupies > 50 % of the
    # capture (a common case for short single-ping bursts).  Matches the
    # baseline convention sq_det + TFMF use elsewhere in the project.
    env_lin = np.maximum(env_lin, 1e-12)
    _sorted = np.sort(env_lin)
    _k_pct25 = max(0, _sorted.size // 4)
    _ref = float(max(_sorted[_k_pct25], 1e-12))
    env_db = 20.0 * np.log10(env_lin / _ref)
    return f_carr.astype(np.float32), t_c.astype(np.float32), env_db.astype(np.float32)


class AnalysisEngine(Engine):
    """DSP replay engine for offline IQ capture analysis.

    Instantiate with the IQ array and metadata dict from capture.load_capture().
    Call run_replay() to process all samples and return display results.
    """

    def __init__(self, iq: np.ndarray, meta: dict):
        rate    = int(meta.get('sample_rate', 48000))
        fc_mhz  = float(meta.get('center_freq_mhz', 50.260))

        super().__init__(center_freq_mhz=fc_mhz, sample_rate=rate)

        self._iq_data = iq.astype(np.complex64)
        self._meta    = meta

        # Enable dual-pol processing when the capture was recorded in dual-pol mode.
        self.dual_pol = bool(meta.get('dual_pol',
                                      iq.ndim == 2 and iq.shape[1] == 2))

        # Use WAV timeline mode: timestamp_int is cursor-seconds, not UTC epoch.
        self.source_mode = 'wav'

        # Anchor all boundary-tracking counters to zero so the 15-second replay
        # window [0, 15) is handled as a single coherent window with no mid-run
        # boundary flips that would wipe spectrogram or heatmap data.
        self.spec_boundary       = 0
        self._realtime_boundary  = 0
        self._ch_snr_boundary    = 0

        # Pre-set display_center_freq_mhz so detection markers carry correct coords.
        self.display_center_freq_mhz = fc_mhz

        # Blanked-block time log populated by processing.py hook.
        self._nb_blank_log = []

        # Decode queue — results from jt9 threads spawned during replay.
        self._decode_queue = queue.SimpleQueue()

    # ------------------------------------------------------------------
    def run_replay(self, nb_factor: float = 6.0, nb_enabled: bool = True,
                   progress_cb=None) -> dict:
        """Feed all IQ samples through the DSP pipeline and return display data.

        Parameters
        ----------
        nb_factor  : NB threshold multiplier (same scale as the live blanker).
        nb_enabled : False → set nb_factor to 1e9 (effectively disabled).

        Returns
        -------
        dict with keys:
            spectrogram_data  (max_history, fft_size)  accumulated 15-s waterfall
            realtime_data     (max_history, fft_size)  same data (single window)
            ch_snr_history    (N_SNR_HIST, N_CHANNELS) per-channel detection metric
            jt9_markers       list of marker dicts (outcome may still be 'launched')
            blanked_times_s   float array — start-time (s) of each blanked NB block
            duration_secs     float — actual replay duration
            meta              dict — capture metadata
            n_launched        int — number of jt9 threads spawned (pending results)
        """
        # Reset DSP state for a clean replay.
        self.nb_factor      = nb_factor if nb_enabled else 1e9
        self._nb_spec_avg   = None
        self._nb_blank_log  = []
        self._pct25_ctr     = 0
        if hasattr(self, '_pct25'):
            del self._pct25

        # Drain any stale results from a previous run.
        while not self._decode_queue.empty():
            try:
                self._decode_queue.get_nowait()
            except Exception:
                break

        iq   = self._iq_data
        n    = len(iq)
        rate = self.sample_rate

        # Feed in blocks of 8 × NB_FFT_SIZE (2048 samples ≈ 43 ms).
        # Cursor advances from 0.0 to (n/rate) seconds, staying below 15 s so
        # no boundary flip occurs mid-replay — the flip is triggered manually
        # at the end to populate spectrogram_data from spec_staging.
        block_size = NB_FFT_SIZE * 8
        pos = 0
        _progress_stride = block_size * 20   # emit progress every ~20 blocks
        _next_progress   = _progress_stride
        while pos < n:
            chunk = iq[pos:pos + block_size]
            t = pos / rate
            # Record chunk start for the NB blank-log hook.
            self._nb_log_t0 = t
            ts_int  = int(t)
            ts_frac = int((t - ts_int) * 1_000_000_000_000)   # picoseconds
            self.process_iq_data(chunk, ts_int, ts_frac)
            pos += len(chunk)
            if progress_cb is not None and pos >= _next_progress:
                progress_cb(pos / n)
                _next_progress = pos + _progress_stride

        # Return immediately — jt9 threads may still be running.
        # The caller (AnalysisWindow) drains _decode_queue via a QTimer and
        # updates marker outcomes as results arrive.
        n_launched = sum(1 for m in self._jt9_markers if m.get('outcome') == 'launched')

        # Manually trigger the spectrogram boundary flip so spec_staging is
        # copied to spectrogram_data (normally happens when the 15-s window ends).
        self.spectrogram_data = self.spec_staging.copy()
        if self.dual_pol:
            self.spectrogram_data_v = self.spec_staging_v.copy()

        # ── TFMF surface for the whole capture ────────────────────────────
        # Run synchronously on the loaded IQ (the background-thread periodic
        # dispatcher may still have work in flight; bypass it for analysis).
        # Uses the same _build_tfmf_display_surface as the live runtime, so
        # what the operator sees here matches the live sync-det display.
        from .processing import _build_tfmf_display_surface, _TFMF_PERIOD_SAMPLES
        tfmf_surface_h = None; tfmf_candidates_h = []
        tfmf_surface_v = None; tfmf_candidates_v = []
        # H and V channels.  `iq` is 1D for single-pol, 2D (N,2) for dual.
        if iq.ndim == 2 and iq.shape[1] >= 1:
            iq_h_arr = iq[:, 0]
            iq_v_arr = iq[:, 1] if iq.shape[1] >= 2 else None
        else:
            iq_h_arr = iq
            iq_v_arr = None
        try:
            wb_h = (iq_h_arr[-_TFMF_PERIOD_SAMPLES:]
                    if iq_h_arr.shape[0] >= _TFMF_PERIOD_SAMPLES else iq_h_arr)
            tfmf_surface_h, tfmf_candidates_h = _build_tfmf_display_surface(wb_h)
        except Exception as _e:
            import logging as _lg
            _lg.getLogger(__name__).warning("Analysis TFMF H compute failed: %s", _e)
        if self.dual_pol and iq_v_arr is not None:
            try:
                wb_v = (iq_v_arr[-_TFMF_PERIOD_SAMPLES:]
                        if iq_v_arr.shape[0] >= _TFMF_PERIOD_SAMPLES else iq_v_arr)
                tfmf_surface_v, tfmf_candidates_v = _build_tfmf_display_surface(wb_v)
            except Exception as _e:
                import logging as _lg
                _lg.getLogger(__name__).warning("Analysis TFMF V compute failed: %s", _e)
        # For audio-burst sources, the native signal had content only in
        # ~0–3 kHz (and after Hilbert+upsample, ZERO in negative freqs and
        # above ~6 kHz aside from numerical noise).  The per-bin pct25
        # noise floor in those quiet bins is dominated by float-roundoff
        # leakage; dividing by it produces "high SNR" candidates at freqs
        # where no signal can physically exist.  Mask candidates outside
        # the audio band so the display only shows freqs that the source
        # could have carried.
        if self._meta.get('source_was_audio'):
            _AUDIO_F_MIN_HZ = -200.0       # tiny margin for Hilbert ringing
            _AUDIO_F_MAX_HZ = 6500.0       # 6 kHz audio Nyquist + margin
            tfmf_candidates_h = [c for c in tfmf_candidates_h
                                 if _AUDIO_F_MIN_HZ <= c.freq_hz <= _AUDIO_F_MAX_HZ]
            tfmf_candidates_v = [c for c in tfmf_candidates_v
                                 if _AUDIO_F_MIN_HZ <= c.freq_hz <= _AUDIO_F_MAX_HZ]

        # ── Instantaneous carrier-freq trace (audio frame) ────────────────
        # Squared-FFT carrier estimator on the channelizer-mixed-down audio
        # at the calling-freq channel.  Same algorithm as
        # tools/measure_ping_freq_vs_time.py — produces the kind of
        # freq-vs-time plot in docs/figures/freq_vs_time_corpus/.
        inst_freq_audio = None; inst_freq_t = None; env_db = None
        try:
            inst_freq_audio, inst_freq_t, env_db = _analysis_freq_vs_time(
                iq_h_arr, rate)
        except Exception as _e:
            import logging as _lg
            _lg.getLogger(__name__).warning("Analysis freq-vs-time failed: %s", _e)

        return {
            'spectrogram_data':   self.spectrogram_data,
            'spectrogram_data_v': self.spectrogram_data_v if self.dual_pol else None,
            'realtime_data':      self.realtime_data,
            'realtime_data_v':    self.realtime_data_v   if self.dual_pol else None,
            'ch_snr_history':     self._ch_snr_history_h.copy(),
            'ch_snr_history_v':   self._ch_snr_history_v.copy() if self.dual_pol else None,
            'dual_pol':           self.dual_pol,
            'jt9_markers':        self._jt9_markers,   # live list — drainable by caller
            'blanked_times_s':    np.array(self._nb_blank_log, dtype=np.float32),
            'duration_secs':      n / rate,
            'meta':               self._meta,
            'n_launched':         n_launched,
            'tfmf_surface_h':     tfmf_surface_h,
            'tfmf_candidates_h':  tfmf_candidates_h,
            'tfmf_surface_v':     tfmf_surface_v,
            'tfmf_candidates_v':  tfmf_candidates_v,
            'inst_freq_audio':    inst_freq_audio,
            'inst_freq_t':        inst_freq_t,
            'env_db':             env_db,
            'source_was_audio':   bool(self._meta.get('source_was_audio')),
            'native_audio':       self._meta.get('native_audio'),
            'native_audio_sr':    self._meta.get('native_audio_sr'),
        }

    # ------------------------------------------------------------------
    def manual_decode(self, fc_hz: float, t_click_s: float,
                      jt9_args: list | None = None) -> dict:
        """Run jt9 synchronously on the IQ around t_click_s at carrier fc_hz.

        The ring buffer is frozen after run_replay() — this reads directly
        from it without waiting for new samples.

        Parameters
        ----------
        fc_hz      : carrier offset from center frequency in Hz (DSP domain).
        t_click_s  : click time within the 15-s window (0–15 s). Used to
                     position the extraction window: 500 ms before, 1200 ms
                     after.

        Returns
        -------
        dict with keys:
            outcome   : 'decoded' | 'no_decode' | 'timeout' | 'error'
            message   : decoded callsign string, or ''
            jt9_snr   : estimated SNR in dB, or None
            fc_hz     : as passed
            t_s       : as passed
            radio_khz : USB dial frequency in kHz
        """
        from .detection import extract_and_decode
        from pathlib import Path
        from datetime import datetime, timezone
        import queue as _queue

        # Treat the click time as the detection moment.
        detect_sample = int(t_click_s * self.sample_rate)

        # Ring state is frozen — return current position.
        ring_state_fn = lambda: (self._iq_ring_pos, self._iq_abs_sample)

        result_q = _queue.SimpleQueue()
        output_dir = str(Path(__file__).parent.parent / 'MSK144' / 'analysis')
        detect_ts  = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H:%M:%S.%f')[:21]

        # Run synchronously in this thread (called from AnalysisWorker or test).
        extract_and_decode(
            self._iq_ring,
            ring_state_fn,
            detect_sample,
            self.sample_rate,
            fc_hz,
            output_dir,
            t_click_s,
            result_q,
            0,                # marker_id
            0,                # ring_gen (frozen — no reset possible)
            lambda: 0,        # ring_gen_fn always returns 0
            self.display_center_freq_mhz,
            detect_ts,
            jt9_args,
        )

        # Collect result from the queue (extract_and_decode puts one item in).
        try:
            result = result_q.get_nowait()
        except Exception:
            result = {'outcome': 'error', 'message': '', 'jt9_snr': None}

        result.setdefault('fc_hz',     fc_hz)
        result.setdefault('t_s',       t_click_s)
        result.setdefault('radio_khz',
                          self.display_center_freq_mhz * 1000.0
                          + (fc_hz - 1500.0) / 1000.0)
        return result


# ── Qt worker thread ──────────────────────────────────────────────────────────

class AnalysisWorker(QtCore.QThread):
    """Run AnalysisEngine.run_replay() in a background thread.

    Signals
    -------
    finished(dict)   — replay complete; payload is the results dict.
    error(str)       — exception message if replay failed.
    """

    finished = QtCore.pyqtSignal(object)   # dict
    error    = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float)    # 0.0 – 1.0

    def __init__(self, engine: AnalysisEngine,
                 nb_factor: float = 6.0, nb_enabled: bool = True,
                 parent=None):
        super().__init__(parent)
        self._engine     = engine
        self._nb_factor  = nb_factor
        self._nb_enabled = nb_enabled

    def run(self):
        try:
            results = self._engine.run_replay(
                nb_factor=self._nb_factor,
                nb_enabled=self._nb_enabled,
                progress_cb=lambda f: self.progress.emit(f),
            )
            self.finished.emit(results)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class ManualDecodeWorker(QtCore.QThread):
    """Run AnalysisEngine.manual_decode() in a background thread.

    Signals
    -------
    finished(dict)  — result dict from manual_decode().
    error(str)      — exception message on failure.
    """

    finished = QtCore.pyqtSignal(object)
    error    = QtCore.pyqtSignal(str)

    def __init__(self, engine: AnalysisEngine,
                 fc_hz: float, t_click_s: float,
                 jt9_args: list | None = None,
                 parent=None):
        super().__init__(parent)
        self._engine    = engine
        self._fc_hz     = fc_hz
        self._t_click_s = t_click_s
        self._jt9_args  = jt9_args

    def run(self):
        try:
            result = self._engine.manual_decode(self._fc_hz, self._t_click_s,
                                                jt9_args=self._jt9_args)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
