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
"""UI layout and slider handlers for the radio IQ visualizer.

Window structure
----------------
Main window (QMainWindow)
    Menu bar:
        File — mutually exclusive source selection
        View — show/hide each panel window
    Central widget:
        Callsign decode list
    Status bar:
        Live power / packet stats  |  tuned-frequency label  |  UTC clock

Free-floating panel windows (QWidget with Qt.Window flag)
    Fast Graph          Accumulated + real-time IQ spectrograms; IQ colour-scale sliders.
    Detection Heatmap   Per-channel SNR heatmap with threshold markers.
    IQ / Noise Blanker  IQ magnitude time-domain plot + noise blanker controls.
    Flex Radio          Flex Radio status and DAXIQ stream info.
    USRP B210           USRP B210 gain/antenna controls and IF stream info.
    Airspy HF+          Airspy HF+ IF stream info.
    RTL-SDR             RTL-SDR IF stream info.

Each panel window:
  - can be moved and resized independently on any monitor
  - hides (rather than closes) when the user clicks its X button
  - has its position, size, and visibility persisted in QSettings
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class _PanelWindow(QtWidgets.QWidget):
    """Free-floating display panel that hides instead of closing."""

    def __init__(self, title, view_action, parent, geo_key=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self._view_action = view_action
        self._geo_key = geo_key

    def closeEvent(self, event):
        parent = self.parent()
        if parent is not None and getattr(parent, '_app_closing', False):
            event.accept()
            return
        self._save_geometry()
        event.ignore()
        self.hide()
        if self._view_action is not None:
            self._view_action.setChecked(False)

    def _save_geometry(self):
        if self._geo_key is not None:
            from .visualizer import _SETTINGS
            _SETTINGS.setValue(self._geo_key, self.saveGeometry())


def setup_ui(self):
    """Build the main window and all free-floating panel windows."""
    from PyQt5 import QtGui as _QtGui
    from .processing import DETECT_THRESH_DB
    from .channelizer import N_CHANNELS
    from .visualizer import _SETTINGS
    from .source_windows import (
        setup_iq_nb_window, setup_flex_window, setup_usrp_window,
        setup_airspy_window, setup_rtlsdr_window,
    )

    self.setWindowTitle(f'map144 - {self.center_freq_mhz:.3f} MHz')
    self.setGeometry(50, 50, 420, 800)

    # ── Menu bar ──────────────────────────────────────────────────────────────
    menu_bar = self.menuBar()

    file_menu = menu_bar.addMenu("&File")
    self.source_action_group = QtWidgets.QActionGroup(self)
    self.source_action_group.setExclusive(True)

    self.source_radio_action = QtWidgets.QAction("Flex Radio", self)
    self.source_radio_action.setCheckable(True)
    self.source_radio_action.setChecked(False)
    self.source_radio_action.triggered.connect(self.on_select_source_radio)
    self.source_action_group.addAction(self.source_radio_action)
    file_menu.addAction(self.source_radio_action)

    self.source_airspy_action = QtWidgets.QAction("Airspy HF+", self)
    self.source_airspy_action.setCheckable(True)
    self.source_airspy_action.triggered.connect(self.on_select_source_airspy)
    self.source_action_group.addAction(self.source_airspy_action)
    file_menu.addAction(self.source_airspy_action)

    self.source_rtlsdr_action = QtWidgets.QAction("NESDR Smart (RTL-SDR)", self)
    self.source_rtlsdr_action.setCheckable(True)
    self.source_rtlsdr_action.triggered.connect(self.on_select_source_rtlsdr)
    self.source_action_group.addAction(self.source_rtlsdr_action)
    file_menu.addAction(self.source_rtlsdr_action)

    self.source_usrp_action = QtWidgets.QAction("USRP B210", self)
    self.source_usrp_action.setCheckable(True)
    self.source_usrp_action.triggered.connect(self.on_select_source_usrp)
    self.source_action_group.addAction(self.source_usrp_action)
    file_menu.addAction(self.source_usrp_action)

    self.source_wav_action = QtWidgets.QAction("WAV File", self)
    self.source_wav_action.setCheckable(True)
    self.source_wav_action.triggered.connect(self.on_select_source_wav)
    self.source_action_group.addAction(self.source_wav_action)
    file_menu.addAction(self.source_wav_action)

    view_menu = menu_bar.addMenu("&View")

    fg_action      = QtWidgets.QAction("Fast Graph",          self)
    det_action     = QtWidgets.QAction("Detection Heatmap",   self)
    iq_nb_action   = QtWidgets.QAction("IQ / Noise Blanker",  self)
    flex_action    = QtWidgets.QAction("Flex Radio",           self)
    usrp_action    = QtWidgets.QAction("USRP B210",            self)
    airspy_action  = QtWidgets.QAction("Airspy HF+",           self)
    rtlsdr_action  = QtWidgets.QAction("RTL-SDR",              self)
    for act in (fg_action, det_action, iq_nb_action,
                flex_action, usrp_action, airspy_action, rtlsdr_action):
        act.setCheckable(True)
        act.setChecked(True)
        view_menu.addAction(act)

    # ── Colour map (shared by all image items) ────────────────────────────────
    pg.setConfigOptions(antialias=True)
    colors = [
        (0, 0, 0), (0, 0, 64), (0, 0, 128), (0, 64, 192),
        (0, 128, 255), (64, 192, 255), (128, 255, 255),
        (255, 255, 128), (255, 255, 255),
    ]
    positions = [i / 8.0 for i in range(9)]
    colormap = pg.ColorMap(positions, colors)

    # ── Plot widgets ──────────────────────────────────────────────────────────
    self.realtime_plot = pg.PlotWidget(title="Current (15 sec)")
    self.realtime_plot.setLabel('left', 'Frequency', units='MHz')
    self.realtime_plot.setLabel('bottom', 'Time', units='s')
    self.realtime_img = pg.ImageItem(axisOrder='col-major')
    self.realtime_plot.addItem(self.realtime_img)
    self.realtime_plot.setAspectLocked(False)
    self.realtime_img.setColorMap(colormap)
    self.realtime_plot.setXRange(0, float(self.history_secs), padding=0)

    self.spectrogram_plot = pg.PlotWidget(title="Previous (15 sec snapshot)")
    self.spectrogram_plot.setLabel('left', 'Frequency', units='MHz')
    self.spectrogram_plot.setLabel('bottom', 'Time', units='s')
    self.spectrogram_img = pg.ImageItem(axisOrder='col-major')
    self.spectrogram_plot.addItem(self.spectrogram_img)
    self.spectrogram_plot.setAspectLocked(False)
    self.spectrogram_img.setColorMap(colormap)

    _half_ch = N_CHANNELS // 2
    self._detect_freq_min_khz  = -float(_half_ch)
    self._detect_freq_span_khz =  float(N_CHANNELS)

    self.ch_detect_plot = pg.PlotWidget(
        title=f"Channel Detection SNR  (threshold {DETECT_THRESH_DB:.0f} dB above noise)"
    )
    self.ch_detect_plot.setLabel('left', 'Frequency offset', units='kHz')
    self.ch_detect_plot.setLabel('bottom', 'Time', units='s')
    self.ch_detect_plot.setAspectLocked(False)
    self.ch_detect_img = pg.ImageItem(axisOrder='col-major')
    self.ch_detect_plot.addItem(self.ch_detect_img)
    self.ch_detect_img.setColorMap(colormap)
    self.ch_detect_curve_red   = pg.PlotCurveItem(pen=pg.mkPen('r', width=1.5))
    self.ch_detect_curve_green = pg.PlotCurveItem(pen=pg.mkPen('g', width=1.5))
    self.ch_detect_plot.addItem(self.ch_detect_curve_red)
    self.ch_detect_plot.addItem(self.ch_detect_curve_green)
    self.ch_detect_plot.setXRange(0, 15.0, padding=0)
    self.ch_detect_plot.setYRange(
        self._detect_freq_min_khz - 0.5,
        self._detect_freq_min_khz + self._detect_freq_span_khz + 0.5,
        padding=0,
    )
    # Do NOT setXLink here — PyQtGraph's link is bidirectional and would
    # override explicit setXRange calls on realtime_plot.
    self.realtime_plot.getViewBox().disableAutoRange()
    self.spectrogram_plot.getViewBox().disableAutoRange()
    self.ch_detect_plot.getViewBox().disableAutoRange()

    # ── Slider bar helper ─────────────────────────────────────────────────────
    def _make_slider_bar(title, min_label_ref, min_range, min_default,
                         max_label_ref, max_range, max_default,
                         min_slot, max_slot, tick_interval):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        min_lbl = QtWidgets.QLabel(f"Min: {min_default} dB")
        setattr(self, min_label_ref, min_lbl)
        row.addWidget(min_lbl)
        min_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        min_sl.setMinimum(min_range[0]); min_sl.setMaximum(min_range[1])
        min_sl.setValue(min_default)
        min_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        min_sl.setTickInterval(tick_interval)
        min_sl.valueChanged.connect(min_slot)
        row.addWidget(min_sl, stretch=1)
        max_lbl = QtWidgets.QLabel(f"Max: {max_default} dB")
        setattr(self, max_label_ref, max_lbl)
        row.addWidget(max_lbl)
        max_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        max_sl.setMinimum(max_range[0]); max_sl.setMaximum(max_range[1])
        max_sl.setValue(max_default)
        max_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        max_sl.setTickInterval(tick_interval)
        max_sl.valueChanged.connect(max_slot)
        row.addWidget(max_sl, stretch=1)
        bar.setFixedHeight(54)
        return bar

    # ── Central widget: callsign decode list ──────────────────────────────────
    _mono9 = _QtGui.QFont("Monospace", 9)

    _decode_header = QtWidgets.QLabel("UTC       Freq        SNR   Message")
    _decode_header.setFont(_mono9)
    _decode_header.setStyleSheet(
        "QLabel { background: #2a2a2a; color: #aaaaaa; "
        "border: 1px solid #555; border-bottom: none; padding: 2px 4px; }"
    )
    self.decode_panel = QtWidgets.QListWidget()
    self.decode_panel.setFont(_mono9)
    self.decode_panel.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    self.decode_panel.setStyleSheet(
        "QListWidget { background: #1a1a1a; color: #e0e0e0; border: 1px solid #555; }"
        "QListWidget::item { padding: 2px 4px; }"
    )
    central = QtWidgets.QWidget()
    self.setCentralWidget(central)
    central_vbox = QtWidgets.QVBoxLayout(central)
    central_vbox.setContentsMargins(0, 0, 0, 0)
    central_vbox.setSpacing(0)
    central_vbox.addWidget(_decode_header)
    central_vbox.addWidget(self.decode_panel, stretch=1)

    # ── Panel window: Fast Graph ──────────────────────────────────────────────
    iq_sliders = _make_slider_bar(
        "IQ Color Scale",
        "min_level_label",  (-150, -20),  self.min_level,
        "max_level_label",  (-100,   0),  self.max_level,
        self.on_min_level_changed, self.on_max_level_changed, 10,
    )
    self._fast_graph_win = _PanelWindow("MapMSK144 — Fast Graph", fg_action, self, 'fast_graph_geometry')
    self._fast_graph_win.setMinimumSize(500, 350)
    fg_layout = QtWidgets.QVBoxLayout(self._fast_graph_win)
    fg_layout.setContentsMargins(0, 0, 0, 0)
    fg_layout.setSpacing(0)
    fg_layout.addWidget(self.realtime_plot,    stretch=1)
    fg_layout.addWidget(self.spectrogram_plot, stretch=1)
    fg_layout.addWidget(iq_sliders)

    # ── Panel window: Detection Heatmap ───────────────────────────────────────
    detect_sliders = _make_slider_bar(
        "Detection Color Scale",
        "detect_min_level_label",  (0, 20),  self.detect_min_level,
        "detect_max_level_label",  (1, 50),  self.detect_max_level,
        self.on_detect_min_level_changed, self.on_detect_max_level_changed, 5,
    )
    self._detect_win = _PanelWindow("MapMSK144 — Detection Heatmap", det_action, self, 'detect_geometry')
    self._detect_win.setMinimumSize(400, 250)
    det_layout = QtWidgets.QVBoxLayout(self._detect_win)
    det_layout.setContentsMargins(0, 0, 0, 0)
    det_layout.setSpacing(0)
    det_layout.addWidget(self.ch_detect_plot, stretch=1)
    det_layout.addWidget(detect_sliders)

    # ── Panel windows: source-specific ───────────────────────────────────────
    setup_iq_nb_window(self,  iq_nb_action)
    setup_flex_window(self,   flex_action)
    setup_usrp_window(self,   usrp_action)
    setup_airspy_window(self, airspy_action)
    setup_rtlsdr_window(self, rtlsdr_action)

    # ── Wire View menu actions ────────────────────────────────────────────────
    fg_action.triggered.connect(
        lambda checked: self._fast_graph_win.show() if checked else self._fast_graph_win.hide()
    )
    det_action.triggered.connect(
        lambda checked: self._detect_win.show() if checked else self._detect_win.hide()
    )
    iq_nb_action.triggered.connect(
        lambda checked: self._iq_nb_win.show() if checked else self._iq_nb_win.hide()
    )
    flex_action.triggered.connect(
        lambda checked: self._flex_win.show() if checked else self._flex_win.hide()
    )
    usrp_action.triggered.connect(
        lambda checked: self._usrp_win.show() if checked else self._usrp_win.hide()
    )
    airspy_action.triggered.connect(
        lambda checked: self._airspy_win.show() if checked else self._airspy_win.hide()
    )
    rtlsdr_action.triggered.connect(
        lambda checked: self._rtlsdr_win.show() if checked else self._rtlsdr_win.hide()
    )

    # ── Restore saved geometry and visibility ─────────────────────────────────
    _win_settings = [
        (self,                 'window_geometry',     None),
        (self._fast_graph_win, 'fast_graph_geometry', QtCore.QRect(480, 50,  850, 650)),
        (self._detect_win,     'detect_geometry',     QtCore.QRect(480, 710, 850, 350)),
        (self._iq_nb_win,      'iq_nb_geometry',      QtCore.QRect(50,  870, 380, 420)),
        (self._flex_win,       'flex_geometry',       QtCore.QRect(450, 870, 360, 500)),
        (self._usrp_win,       'usrp_geometry',       QtCore.QRect(450, 870, 360, 440)),
        (self._airspy_win,     'airspy_geometry',     QtCore.QRect(450, 870, 360, 340)),
        (self._rtlsdr_win,     'rtlsdr_geometry',     QtCore.QRect(450, 870, 360, 340)),
    ]
    for win, key, default_rect in _win_settings:
        geo = _SETTINGS.value(key)
        if geo:
            win.restoreGeometry(geo)
        elif default_rect is not None:
            win.setGeometry(default_rect)

    # Fast Graph and Detection always restore to saved visibility.
    # IQ/NB window always shows.  Radio windows start hidden; they show
    # when a source is selected.
    fg_visible  = _SETTINGS.value('fast_graph_visible', True,  type=bool)
    det_visible = _SETTINGS.value('detect_visible',     True,  type=bool)
    iq_nb_visible = _SETTINGS.value('iq_nb_visible',    True,  type=bool)
    fg_action.setChecked(fg_visible)
    det_action.setChecked(det_visible)
    iq_nb_action.setChecked(iq_nb_visible)
    if fg_visible:    self._fast_graph_win.show()
    if det_visible:   self._detect_win.show()
    if iq_nb_visible: self._iq_nb_win.show()

    # Radio windows start hidden
    for win in (self._flex_win, self._usrp_win, self._airspy_win, self._rtlsdr_win):
        win.hide()
        if win._view_action is not None:
            win._view_action.setChecked(False)

    # ── Status bar ────────────────────────────────────────────────────────────
    self.statusBar().showMessage('Initializing...')
    self.tuned_freq_label = QtWidgets.QLabel("Tuned: --")
    self.tuned_freq_label.setStyleSheet("QLabel { font-weight: bold; padding: 0 10px; }")
    self.statusBar().addPermanentWidget(self.tuned_freq_label)
    self.utc_clock_label = QtWidgets.QLabel()
    self.utc_clock_label.setStyleSheet("QLabel { font-weight: bold; padding: 0 10px; }")
    self.statusBar().addPermanentWidget(self.utc_clock_label)

    self.update_timer = QtCore.QTimer()
    self.update_timer.timeout.connect(self.update_displays)
    self.update_timer.start(100)


def on_min_level_changed(self, value):
    self.min_level = value
    self.min_level_label.setText(f"Min: {value} dB")


def on_max_level_changed(self, value):
    self.max_level = value
    self.max_level_label.setText(f"Max: {value} dB")


def on_detect_min_level_changed(self, value):
    self.detect_min_level = value
    self.detect_min_level_label.setText(f"Min: {value} dB")


def on_detect_max_level_changed(self, value):
    self.detect_max_level = value
    self.detect_max_level_label.setText(f"Max: {value} dB")


def on_nb_factor_changed(self, value):
    self.nb_factor = value * 0.1
    self.nb_factor_label.setText(f"{self.nb_factor:.1f}")


def on_td_scale_changed(self, value):
    self.td_scale = value * 0.01   # slider 1–100 → y-max 0.01–1.0
    self.td_plot.setYRange(0.0, self.td_scale, padding=0)


def on_td_span_changed(self, value):
    self.td_span_ms = float(value)
    self.td_span_val_label.setText(f"{value} ms")
    self.td_plot.setTitle(f'IQ Magnitude — {value} ms')


def on_select_source_airspy(self):
    from .runtime import _connect_airspy_client
    from .source_windows import show_source_window
    _connect_airspy_client(self)
    self.source_mode = "airspy"
    self.selected_wav_path = None
    self.source_airspy_action.setChecked(True)
    show_source_window(self, "airspy")
    self.statusBar().showMessage("Source: Airspy HF+")


def on_select_source_rtlsdr(self):
    from .runtime import _connect_rtlsdr_client
    from .source_windows import show_source_window
    _connect_rtlsdr_client(self)
    self.source_mode = "rtlsdr"
    self.selected_wav_path = None
    self.source_rtlsdr_action.setChecked(True)
    show_source_window(self, "rtlsdr")
    self.statusBar().showMessage("Source: NESDR Smart (RTL-SDR)")


def on_select_source_usrp(self):
    from .runtime import _connect_usrp_client
    from .source_windows import show_source_window
    _connect_usrp_client(self)
    self.source_mode = "usrp"
    self.selected_wav_path = None
    self.source_usrp_action.setChecked(True)
    show_source_window(self, "usrp")
    self.statusBar().showMessage("Source: USRP B210")


def on_select_source_radio(self):
    from .source_windows import show_source_window
    self._connect_radio_client()
    self.source_mode = "radio"
    self.selected_wav_path = None
    self.source_radio_action.setChecked(True)
    show_source_window(self, "radio")
    self.statusBar().showMessage("Source: Flex Radio")


def on_select_source_wav(self):
    from pathlib import Path as _Path
    from .source_windows import show_source_window
    _default_dir = str(_Path('MSK144/simulations').resolve()) if _Path('MSK144/simulations').exists() else ""
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select Simulation WAV File",
        _default_dir,
        "WAV Files (*.wav);;All Files (*)",
    )

    if not file_path:
        return

    self.source_mode = "wav"
    self.selected_wav_path = file_path
    self._wav_load_nonce = getattr(self, '_wav_load_nonce', 0) + 1
    self._wav_done = False
    self.source_wav_action.setChecked(True)
    show_source_window(self, "wav")   # hides all radio windows
    self.statusBar().showMessage(f"Source: WAV File ({_Path(file_path).name})")
