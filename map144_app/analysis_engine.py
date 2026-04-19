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
    def run_replay(self, nb_factor: float = 6.0, nb_enabled: bool = True) -> dict:
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
        while pos < n:
            chunk = iq[pos:pos + block_size]
            t = pos / rate
            # Record chunk start for the NB blank-log hook.
            self._nb_log_t0 = t
            ts_int  = int(t)
            ts_frac = int((t - ts_int) * 1_000_000_000_000)   # picoseconds
            self.process_iq_data(chunk, ts_int, ts_frac)
            pos += len(chunk)

        # Return immediately — jt9 threads may still be running.
        # The caller (AnalysisWindow) drains _decode_queue via a QTimer and
        # updates marker outcomes as results arrive.
        n_launched = sum(1 for m in self._jt9_markers if m.get('outcome') == 'launched')

        # Manually trigger the spectrogram boundary flip so spec_staging is
        # copied to spectrogram_data (normally happens when the 15-s window ends).
        self.spectrogram_data = self.spec_staging.copy()
        if self.dual_pol:
            self.spectrogram_data_v = self.spec_staging_v.copy()

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
