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
"""Headless DSP engine — no Qt imports.

Can be used standalone (headless mode) or as a base class for
MAP144Visualizer(Engine, QMainWindow).
"""

import datetime
import queue
import signal

import numpy as np

from .channelizer import design_channelizer_filter, make_channelizer_state, N_CHANNELS
from .processing import (
    process_iq_data as _process_iq_data,
    N_SNR_HIST, CH_DETECT_SIZE, _METRIC_HIST_DEPTH,
)


class Engine:
    """DSP engine: holds all numpy state and the IQ processing pipeline.

    No PyQt5 imports.  In GUI mode MAP144Visualizer inherits from this class.
    In headless mode instantiate directly and call run_headless().
    """

    process_iq_data = _process_iq_data

    def __init__(self, center_freq_mhz=50.260, sample_rate=48000, fft_size=2048,
                 bind_client_id=None, nb_factor=6.0):
        self.center_freq_mhz = center_freq_mhz
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bind_client_id = bind_client_id
        self.nb_factor = nb_factor
        self.history_secs = 15
        self.blocks_per_sec = self.sample_rate / (self.fft_size // 2)
        self.max_history = int(round(self.history_secs * self.blocks_per_sec))
        self.max_time = self.history_secs
        self.running = True
        self.source_mode = "idle"
        self.selected_wav_path = None
        self._radio_started = False
        self._wav_samples = None
        self._wav_path_loaded = None
        self._wav_index = 0
        self._wav_time_cursor = 0.0
        self._wav_done = False
        self._wav_load_nonce = 0
        self._wav_nonce_loaded = -1
        self._wav_run_start_time = None
        self.radio_client = None
        self.airspy_client = None
        self._airspy_started = False
        self.rtlsdr_client = None
        self._rtlsdr_started = False

        # Overlap-add buffer for waterfall FFT
        _sbuf_cap = self.fft_size * 6
        self._sbuf = np.zeros(_sbuf_cap, dtype=np.complex64)
        self._sbuf_end = 0
        self._sbuf_t0  = 0.0   # wall-clock time of _sbuf[0], anchored to VITA timestamps

        # Noise blanker state
        self._nb_env      = None   # running mean magnitude (display)
        self._nb_floor    = None   # noise floor derived from per-bin averages (display)
        self._nb_spec_avg = None   # per-bin running average power, shape (NB_FFT_SIZE,)
        self._nb_last_P   = None   # per-bin power of the most recent block (display)
        self._nb_last_hot = 0      # hot-bin count of the most recent block (display)
        self._nb_blanked_count = 0
        self._nb_total_count = 0

        # Time-domain magnitude display buffer (200 ms circular)
        _td_n = int(0.200 * self.sample_rate)
        self._td_mag_buf = np.zeros(_td_n, dtype=np.float32)
        self._td_mag_pos = 0

        current_time = datetime.datetime.now().timestamp()
        self.time_in_window = current_time % self.history_secs
        self.next_boundary = current_time + (self.history_secs - self.time_in_window)

        # Accumulated spectrogram buffers
        self.spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_boundary = int(current_time / self.history_secs)
        self.spec_staging_filled = False

        # Real-time spectrogram buffers
        self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)
        self.realtime_time = self.history_secs
        self.realtime_filled = False
        self._realtime_boundary = self.spec_boundary

        # Noise floor arrays
        self.accumulated_noise_floor = np.full(self.fft_size, -125.0)
        self.realtime_noise_floor = np.full(self.fft_size, -125.0)

        # Channelizer state
        self._ch_taps = design_channelizer_filter(self.sample_rate)
        self._ch_state = make_channelizer_state(N_CHANNELS, self._ch_taps)
        _ch_buf_cap = CH_DETECT_SIZE + 2048
        self._ch_buf = np.zeros((N_CHANNELS, _ch_buf_cap), dtype=np.complex64)
        self._ch_buf_end = 0

        # Metric history (rolling 25th-percentile baseline)
        self._metric_hist_buf = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
        self._metric_hist_idx = 0
        self._metric_hist_cnt = 0

        # Pre-computed FFT window
        self._fft_window = np.hanning(self.fft_size).astype(np.float32)
        self._window_gain = float(np.sqrt(np.mean(self._fft_window ** 2)))

        # Per-channel SNR history for detection heatmap
        self._ch_snr_history = np.zeros((N_SNR_HIST, N_CHANNELS), dtype=np.float32)
        self._ch_snr_write_idx = 0
        self._ch_snr_boundary = int(current_time / self.history_secs)

        # IQ ring buffer (5 seconds)
        _ring_n = int(5 * self.sample_rate)
        self._iq_ring = np.zeros(_ring_n, dtype=np.complex64)
        self._iq_ring_pos = 0
        self._iq_abs_sample = 0
        self._detect_cooldowns = {}
        self._iq_ring_gen = 0
        self._jt9_threads = []
        self._jt9_markers = []
        self._jt9_marker_next_id = 0
        self._decode_queue = queue.SimpleQueue()

        # Frequency axis
        self.fft_bin_axis_mhz = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1 / self.sample_rate)
        ) / 1e6
        self.freq_axis = self.fft_bin_axis_mhz + self.center_freq_mhz
        self.display_center_freq_mhz = -1.0  # force report_freq() on first display update

    def setup_radio_client(self):
        """No-op in headless mode — overridden by MAP144Visualizer for Qt thread."""
        self.radio_client = None

    def run_headless(self):
        """Block and process IQ data with no GUI.  SIGINT/SIGTERM trigger clean shutdown."""
        from .runtime import _connect_radio_client, run_radio_source, _stop_radio_source

        def _shutdown(_sig, _frame):
            print("[map144] headless shutdown requested", flush=True)
            self.running = False
            _stop_radio_source(self)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        if self.source_mode == 'radio':
            _connect_radio_client(self)

        run_radio_source(self)
