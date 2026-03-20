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
"""Main visualizer class and shared state for the modular radio IQ GUI."""

import datetime
import numpy as np
from PyQt5 import QtCore, QtWidgets

_SETTINGS = QtCore.QSettings('RadioIQ', 'RadioIQVisualizer')

from .ui import (
    setup_ui,
    on_min_level_changed,
    on_max_level_changed,
    on_sq_min_level_changed,
    on_sq_max_level_changed,
    on_select_source_radio,
    on_select_source_wav,
    on_sq_realtime_mouse_moved,
)
from .runtime import setup_radio_client, _connect_radio_client, run_radio_source, _get_tuned_frequency_mhz, closeEvent
from .processing import process_iq_data
from .displays import update_displays
from .detection import design_lp_filter


class RadioIQVisualizer(QtWidgets.QMainWindow):
    """Main window with five synchronized displays for radio IQ data."""

    setup_ui = setup_ui
    on_min_level_changed = on_min_level_changed
    on_max_level_changed = on_max_level_changed
    on_sq_min_level_changed = on_sq_min_level_changed
    on_sq_max_level_changed = on_sq_max_level_changed
    on_select_source_radio = on_select_source_radio
    on_select_source_wav = on_select_source_wav
    on_sq_realtime_mouse_moved = on_sq_realtime_mouse_moved
    setup_radio_client = setup_radio_client
    _connect_radio_client = _connect_radio_client
    run_radio_source = run_radio_source
    _get_tuned_frequency_mhz = _get_tuned_frequency_mhz
    process_iq_data = process_iq_data
    update_displays = update_displays
    closeEvent = closeEvent

    def __init__(self, center_freq_mhz=50.260, sample_rate=48000, fft_size=2048,
                 bind_client_id=None, bind_client_handle=None):
        super().__init__()
        self.center_freq_mhz = center_freq_mhz
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bind_client_id = bind_client_id or bind_client_handle
        self.history_secs = 15
        self.blocks_per_sec = self.sample_rate / (self.fft_size // 2)  # 50% overlap hop
        self.max_history = int(round(self.history_secs * self.blocks_per_sec))
        self.running = True
        self.source_mode = "idle"
        self.selected_wav_path = None
        self._radio_started = False
        self._wav_samples = None
        self._wav_path_loaded = None
        self._wav_index = 0
        self._wav_time_cursor = 0.0
        self._wav_done = False
        self._wav_load_nonce = 0        # incremented each time user selects a file
        self._wav_nonce_loaded = -1     # nonce value at last load (forces replay on re-select)
        self._wav_run_start_time = None # UTC datetime when current WAV run started

        self.sample_buffer = np.array([], dtype=np.complex64)
        self.raw_buffer    = np.array([], dtype=np.complex64)

        current_time = datetime.datetime.now().timestamp()
        self.time_in_window = current_time % self.history_secs
        self.next_boundary = current_time + (self.history_secs - self.time_in_window)

        self.spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)

        self.spec_boundary = int(current_time / self.history_secs)
        self.spec_staging_filled = False
        initial_index = int(self.time_in_window * self.blocks_per_sec)
        self.spec_write_index = min(max(initial_index, 0), self.max_history - 1)

        self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)
        self.realtime_time = self.history_secs
        self.realtime_filled = False
        self._realtime_boundary = self.spec_boundary
        self.realtime_write_index = min(max(initial_index, 0), self.max_history - 1)

        self.accumulated_noise_floor = np.full(self.fft_size, -125.0)
        self.realtime_noise_floor = np.full(self.fft_size, -125.0)

        self.max_time = self.history_secs

        self.min_level    = int(_SETTINGS.value('min_level',    -90))
        self.max_level    = int(_SETTINGS.value('max_level',    -30))

        # Squared signal spectrogram buffers (for MSK144 tone-pair detection).
        # Squaring the IQ doubles all spectral component frequencies; MSK144 tones
        # at fc±500 Hz produce a ±1000 Hz symmetric pair in this spectrum.
        self.sq_spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
        self.sq_spec_staging = np.full((self.max_history, self.fft_size), -130.0)
        self.sq_spec_boundary = int(current_time / self.history_secs)
        self.sq_spec_staging_filled = False
        self.sq_spec_write_index = min(max(initial_index, 0), self.max_history - 1)

        self.sq_realtime_data = np.full((self.max_history, self.fft_size), -130.0)
        self.sq_realtime_filled = False
        self._sq_realtime_boundary = self.sq_spec_boundary
        self.sq_realtime_write_index = min(max(initial_index, 0), self.max_history - 1)

        # Relative frequency axis in kHz for the squared-signal plots.
        # Labels represent FFT bin offsets from center; actual spectral content
        # appears at 2× these offsets due to squaring.
        self.sq_freq_axis_khz = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate)
        ) / 1e3

        self.sq_min_level = self.min_level
        self.sq_max_level = self.max_level

        # LP filter state (10 kHz cutoff, streaming FIR)
        self._lp_taps = design_lp_filter(self.sample_rate)
        self._lp_zi_re = np.zeros(len(self._lp_taps) - 1, dtype=np.float64)
        self._lp_zi_im = np.zeros(len(self._lp_taps) - 1, dtype=np.float64)

        # Circular IQ ring buffer (5 seconds of LP-filtered samples)
        _ring_n = int(5 * self.sample_rate)
        self._iq_ring = np.zeros(_ring_n, dtype=np.complex64)
        self._iq_ring_pos = 0
        self._iq_abs_sample = 0
        self._detect_cooldowns = {}   # {freq_bin_hz: hops_remaining}

        self.fft_bin_axis_mhz = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1 / self.sample_rate)
        ) / 1e6
        self.freq_axis = self.fft_bin_axis_mhz + self.center_freq_mhz
        self.display_center_freq_mhz = self.center_freq_mhz

        print(f"Center requested: {self.center_freq_mhz:.6f} MHz", flush=True)

        self.setup_ui()
        geometry = _SETTINGS.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        self.setup_radio_client()

