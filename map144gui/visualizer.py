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
"""MAP144Visualizer — main window for the map144 MSK144 meteor scatter decoder."""

from PyQt5 import QtCore, QtWidgets

_SETTINGS = QtCore.QSettings('RadioIQ', 'RadioIQVisualizer')

from .engine import Engine
from .ui import (
    setup_ui,
    on_min_level_changed,
    on_max_level_changed,
    on_detect_min_level_changed,
    on_detect_max_level_changed,
    on_nb_factor_changed,
    on_td_scale_changed,
    on_td_span_changed,
    on_select_source_radio,
    on_select_source_wav,
    on_select_source_airspy,
)
from .runtime import setup_radio_client, _connect_radio_client, run_radio_source, _get_tuned_frequency_mhz, closeEvent
from .processing import N_SNR_HIST, CH_DETECT_SIZE, _METRIC_HIST_DEPTH
from .displays import update_displays


class MAP144Visualizer(Engine, QtWidgets.QMainWindow):
    """Main window for the map144 MSK144 meteor scatter decoder."""

    setup_ui = setup_ui
    on_min_level_changed = on_min_level_changed
    on_max_level_changed = on_max_level_changed
    on_detect_min_level_changed = on_detect_min_level_changed
    on_detect_max_level_changed = on_detect_max_level_changed
    on_nb_factor_changed = on_nb_factor_changed
    on_td_scale_changed = on_td_scale_changed
    on_td_span_changed  = on_td_span_changed
    on_select_source_radio = on_select_source_radio
    on_select_source_wav = on_select_source_wav
    on_select_source_airspy = on_select_source_airspy
    setup_radio_client = setup_radio_client
    _connect_radio_client = _connect_radio_client
    run_radio_source = run_radio_source
    _get_tuned_frequency_mhz = _get_tuned_frequency_mhz
    update_displays = update_displays
    closeEvent = closeEvent

    def __init__(self, center_freq_mhz=50.260, sample_rate=48000, fft_size=2048,
                 bind_client_id=None, bind_client_handle=None):
        QtWidgets.QMainWindow.__init__(self)
        Engine.__init__(self, center_freq_mhz=center_freq_mhz, sample_rate=sample_rate,
                        fft_size=fft_size,
                        bind_client_id=bind_client_id or bind_client_handle)

        self.min_level = int(_SETTINGS.value('min_level', -90))
        self.max_level = int(_SETTINGS.value('max_level', -30))
        self.detect_min_level = int(_SETTINGS.value('detect_min_level', 0))
        self.detect_max_level = int(_SETTINGS.value('detect_max_level', 15))
        try:
            self.nb_factor = float(_SETTINGS.value('nb_factor', 6.0))
        except (ValueError, TypeError):
            self.nb_factor = 6.0

        print(f"Center requested: {self.center_freq_mhz:.6f} MHz", flush=True)

        self.setup_ui()
        self.setup_radio_client()
