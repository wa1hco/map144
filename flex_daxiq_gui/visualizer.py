"""Main visualizer class and shared state for the modular DAXIQ GUI."""

import datetime
import numpy as np
from PyQt5 import QtCore, QtWidgets

_SETTINGS = QtCore.QSettings('FlexDAXIQ', 'DAXIQVisualizer')

from .ui import (
    setup_ui,
    on_min_level_changed,
    on_max_level_changed,
    on_sq_min_level_changed,
    on_sq_max_level_changed,
    on_select_source_flex,
    on_select_source_wav,
)
from .runtime import setup_flex_client, run_flex_client, _get_tuned_frequency_mhz, closeEvent
from .processing import process_iq_data
from .displays import update_displays
from .detection import design_lp_filter


class DAXIQVisualizer(QtWidgets.QMainWindow):
    """Main window with three synchronized displays for DAXIQ data."""

    setup_ui = setup_ui
    on_min_level_changed = on_min_level_changed
    on_max_level_changed = on_max_level_changed
    on_sq_min_level_changed = on_sq_min_level_changed
    on_sq_max_level_changed = on_sq_max_level_changed
    on_select_source_flex = on_select_source_flex
    on_select_source_wav = on_select_source_wav
    setup_flex_client = setup_flex_client
    run_flex_client = run_flex_client
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
        self.source_mode = "flex"
        self.selected_wav_path = None
        self._flex_started = False
        self._wav_samples = None
        self._wav_path_loaded = None
        self._wav_index = 0
        self._wav_time_cursor = 0.0

        self.sample_buffer = np.array([], dtype=np.complex64)

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

        self.realtime_energy_buffer = np.full(self.max_history, np.nan)
        self.accumulated_energy_buffer = np.full(self.max_history, np.nan)
        self.energy_time_axis = np.arange(self.max_history) / self.blocks_per_sec
        self.energy_boundary = self.spec_boundary
        self.accumulated_energy_filled = False
        self.energy_write_index = min(max(initial_index, 0), self.max_history - 1)
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
        self._detect_cooldown = 0

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
        self.setup_flex_client()

    def _map_energy_to_freq_band(self, energy_vals, freq_min, freq_max):
        """Map energy values into the bottom 10% of the periodogram height."""
        if len(energy_vals) == 0:
            return np.array([], dtype=np.float64)

        energy_min_db = float(self.min_level)
        energy_max_db = float(self.max_level)
        if energy_max_db <= energy_min_db:
            energy_max_db = energy_min_db + 1.0
        norm = (energy_vals - energy_min_db) / (energy_max_db - energy_min_db)
        norm = np.clip(norm, 0.0, 1.0)

        periodogram_height = freq_max - freq_min
        overlay_height = 0.10 * periodogram_height
        return freq_min + norm * overlay_height
