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
    Tone Detection SNR  Per-channel squared-domain SNR image with threshold markers.
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

import subprocess

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from . import __version__
from .widgets import ClickableImageItem

# MSK144 calling frequencies (US convention) and compatible source modes.
# All frequencies are North American (US) convention per WSJT-X defaults.
# Airspy HF+ tops out at ~31 MHz so it appears in no entry here.
# Flex Radio follows the panadapter and is limited to its hardware band plan
# (used exclusively for 6m in this installation).
MSK144_BANDS = [
    # (display label,    freq MHz,  compatible source_mode values)
    ("6m   50.260 MHz",   50.260, {'usrp', 'rtlsdr', 'radio', 'sdrangel'}),
    ("2m  144.150 MHz",  144.150, {'usrp', 'rtlsdr', 'sdrangel'}),
    ("1¼m 222.090 MHz",  222.090, {'usrp', 'rtlsdr', 'sdrangel'}),
    ("70cm 432.360 MHz", 432.360, {'usrp', 'rtlsdr', 'sdrangel'}),
]


def _get_version_string():
    """Return a version string from git: hash, date, and subject line."""
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%h  %cd  %s", "--date=short"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return commit + ("  (modified)" if dirty else "")
    except Exception:
        return "unknown"


class _PanelWindow(QtWidgets.QWidget):
    """Free-floating display panel that hides instead of closing.

    Constructed with Qt parent=None so the window manager does NOT mark it as
    WM_TRANSIENT_FOR the main window.  On X11/Linux some WMs send
    WM_DELETE_WINDOW to transient children during a move, which Qt converts to
    a closeEvent and causes the panel to hide spontaneously when dragged.
    The visualizer reference is kept in _parent_win for _app_closing only.
    """

    def __init__(self, title, view_action, parent, geo_key=None):
        super().__init__(None, QtCore.Qt.Window)   # top-level — no WM transient
        self._parent_win  = parent                 # for _app_closing check only
        self.setWindowTitle(title)
        self._view_action = view_action
        self._geo_key = geo_key

    def closeEvent(self, event):
        if getattr(self._parent_win, '_app_closing', False):
            event.accept()
            return
        try:
            self._save_geometry()
        except Exception:
            pass  # never block the hide on a settings-write failure
        event.ignore()
        self.hide()
        # Menu sync handled by hideEvent below — fires on every path
        # (X button, programmatic hide(), WM events) so the View-menu
        # checkbox cannot drift out of sync with actual visibility.

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._view_action is not None and self._view_action.isChecked():
            self._view_action.setChecked(False)

    def showEvent(self, event):
        super().showEvent(event)
        if self._view_action is not None and not self._view_action.isChecked():
            self._view_action.setChecked(True)

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
        setup_airspy_window, setup_rtlsdr_window, setup_sdrangel_window,
    )
    from .reporting_window import setup_reporting_window
    from .screenshot_window import setup_screenshot_window

    self.setWindowTitle(f'map144 v{__version__}  —  {self.center_freq_mhz:.3f} MHz')
    self.setGeometry(50, 50, 420, 800)

    # ── Menu bar ──────────────────────────────────────────────────────────────
    menu_bar = self.menuBar()

    file_menu = menu_bar.addMenu("&File")

    captures_menu = file_menu.addMenu("Captures")
    open_capture_action = QtWidgets.QAction("Open Capture...", self)
    open_capture_action.triggered.connect(self._on_open_capture)
    captures_menu.addAction(open_capture_action)
    browse_captures_action = QtWidgets.QAction("Browse Captures Folder", self)
    browse_captures_action.triggered.connect(self._on_browse_captures)
    captures_menu.addAction(browse_captures_action)
    file_menu.addSeparator()

    self.source_action_group = QtWidgets.QActionGroup(self)
    self.source_action_group.setExclusive(True)

    self.source_radio_action = QtWidgets.QAction("Flex Radio", self)
    self.source_radio_action.setCheckable(True)
    self.source_radio_action.setChecked(False)
    self.source_radio_action.triggered.connect(self.on_select_source_radio)
    self.source_action_group.addAction(self.source_radio_action)
    file_menu.addAction(self.source_radio_action)

    self.source_usrp_action = QtWidgets.QAction("USRP B210", self)
    self.source_usrp_action.setCheckable(True)
    self.source_usrp_action.triggered.connect(self.on_select_source_usrp)
    self.source_action_group.addAction(self.source_usrp_action)
    file_menu.addAction(self.source_usrp_action)

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

    self.source_sdrangel_action = QtWidgets.QAction("SDRangel", self)
    self.source_sdrangel_action.setCheckable(True)
    self.source_sdrangel_action.triggered.connect(self.on_select_source_sdrangel)
    self.source_action_group.addAction(self.source_sdrangel_action)
    file_menu.addAction(self.source_sdrangel_action)

    self.source_wav_action = QtWidgets.QAction("WAV File", self)
    self.source_wav_action.setCheckable(True)
    self.source_wav_action.triggered.connect(self.on_select_source_wav)
    self.source_action_group.addAction(self.source_wav_action)
    file_menu.addAction(self.source_wav_action)
    file_menu.addSeparator()
    screenshot_action  = QtWidgets.QAction("Screenshots",          self)
    screenshot_action.setCheckable(True)
    screenshot_action.setChecked(False)
    file_menu.addAction(screenshot_action)

    view_menu = menu_bar.addMenu("&View")

    help_menu    = menu_bar.addMenu("&Help")
    about_action = QtWidgets.QAction("About map144", self)
    about_action.triggered.connect(self._on_about)
    help_menu.addAction(about_action)

    fg_action          = QtWidgets.QAction("Fast Graph",          self)
    det_action         = QtWidgets.QAction("Tone Detection SNR",   self)
    sync_det_action    = QtWidgets.QAction("Sync Detection",       self)
    iq_nb_action       = QtWidgets.QAction("Noise Blanker",  self)
    reporting_action   = QtWidgets.QAction("Reporting",            self)
    flex_action        = QtWidgets.QAction("Flex Radio",           self)
    usrp_action        = QtWidgets.QAction("USRP B210",            self)
    airspy_action      = QtWidgets.QAction("Airspy HF+",           self)
    rtlsdr_action      = QtWidgets.QAction("RTL-SDR",              self)
    sdrangel_action    = QtWidgets.QAction("SDRangel",             self)
    # Toggle-style View entries — checked state mirrors window visibility.
    for act in (fg_action, det_action, sync_det_action, iq_nb_action, reporting_action,
                flex_action, usrp_action, airspy_action, rtlsdr_action, sdrangel_action):
        act.setCheckable(True)
        act.setChecked(True)
        view_menu.addAction(act)
    # Analysis is action-style, not toggle-style: each click opens a
    # NEW analysis window for a captured WAV (multiple may be open at
    # once).  Making it checkable would mean "checked = window open"
    # but with N windows that semantic doesn't fit; non-checkable keeps
    # the menu honest.
    analysis_action    = QtWidgets.QAction("Open Analysis Window…", self)
    view_menu.addAction(analysis_action)

    # ── Diagnostics menu ──────────────────────────────────────────────────────
    # Diagnostic toggles that affect *display only* (detection logic remains
    # unchanged).  Not persisted across runs — accidentally leaving a
    # diagnostic mode on between sessions would confuse the live operator.
    diagnostics_menu = menu_bar.addMenu("&Diagnostics")
    self.show_raw_power_action = QtWidgets.QAction(
        "Show raw power (bypass pct25 normalisation)", self)
    self.show_raw_power_action.setCheckable(True)
    self.show_raw_power_action.setChecked(False)
    self.show_raw_power_action.setToolTip(
        "Detection heatmap shows raw squared-FFT power (dB) instead of "
        "dB-above-pct25.  Reveals persistent spurs and asymmetric noise-"
        "floor rolloffs that the rolling baseline normally absorbs.\n\n"
        "Detection-trigger logic is unaffected; only the heatmap display "
        "changes.  Adjust the detection colour-scale slider when toggling "
        "— raw dB has a different dynamic range from pair-metric dB."
    )
    self.show_raw_power_action.toggled.connect(self._on_show_raw_power_toggled)
    diagnostics_menu.addAction(self.show_raw_power_action)

    # Save a snapshot of every internal array that feeds the detection /
    # sync displays.  Operator-triggered so we can compare live-engine
    # state against what offline tests produce.  Used to chase
    # phenomena (e.g. heatmap banding) that only manifest in
    # long-running live sessions.
    self.dump_state_action = QtWidgets.QAction(
        "Save state snapshot…", self)
    self.dump_state_action.setToolTip(
        "Save a .npz file containing the rolling pct25 history, current "
        "pct25 baseline, noise-blanker state, and the detection-heatmap "
        "buffer.  Used to diagnose live-engine behaviour offline."
    )
    self.dump_state_action.triggered.connect(
        lambda: _save_state_snapshot(self)
    )
    diagnostics_menu.addAction(self.dump_state_action)

    # ── Band selector toolbar ─────────────────────────────────────────────────
    from .visualizer import _SETTINGS as _VS
    band_toolbar = self.addToolBar("Band")
    band_toolbar.setMovable(False)
    band_toolbar.setFloatable(False)
    band_toolbar.addWidget(QtWidgets.QLabel("  Band: "))
    self._band_combo = QtWidgets.QComboBox()
    for _bl, _, _ in MSK144_BANDS:
        self._band_combo.addItem(_bl)
    # Restore saved selection (match by frequency)
    try:
        _saved_band_freq = float(_VS.value('calling_freq_mhz', 50.260))
    except (ValueError, TypeError):
        _saved_band_freq = 50.260
    _saved_band_idx = next(
        (i for i, (_, f, _s) in enumerate(MSK144_BANDS) if abs(f - _saved_band_freq) < 0.001),
        0,
    )
    self._band_combo.blockSignals(True)
    self._band_combo.setCurrentIndex(_saved_band_idx)
    self._band_combo.blockSignals(False)
    self._band_combo.currentIndexChanged.connect(self.on_band_changed)
    band_toolbar.addWidget(self._band_combo)

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
    # No titles — saves vertical space.  H/V identity shown via Y-axis label.
    # Time axis: carried by a dedicated ruler widget (viewbox collapsed to 0 px)
    # placed below all data plots.  All data plots permanently hide their bottom
    # axis so every data viewbox gets identical height (stretch=1 with no axis).
    _time_ticks = [[(i, str(i)) for i in range(self.history_secs + 1)]]

    def _make_time_ruler():
        """PlotWidget with viewbox collapsed to zero — just a time axis bar."""
        _r = pg.PlotWidget()
        _r.getAxis('left').hide()
        _r.getAxis('right').hide()
        _r.getAxis('top').hide()
        _r.getAxis('bottom').setTicks(_time_ticks)
        _r.setXRange(0, float(self.history_secs) + 0.5, padding=0)
        _r.getViewBox().disableAutoRange()
        # Collapse the viewbox row (row 2) in PlotItem's internal GraphicsLayout
        _pi = _r.getPlotItem()
        for _row in (0, 1, 2):   # title, top-axis, viewbox
            _pi.layout.setRowMaximumHeight(_row, 0)
            _pi.layout.setRowMinimumHeight(_row, 0)
        _r.setMinimumHeight(44)
        _r.setMaximumHeight(50)
        return _r

    self.realtime_plot = pg.PlotWidget()
    self.realtime_plot.setLabel('left', 'H  Freq (MHz)')
    self.realtime_plot.getAxis('bottom').hide()
    self.realtime_plot.getViewBox().disableAutoRange()
    self.realtime_img = ClickableImageItem(axisOrder='col-major')
    self.realtime_plot.addItem(self.realtime_img)
    self.realtime_plot.setAspectLocked(False)
    self.realtime_img.setColorMap(colormap)
    self.realtime_plot.setXRange(0, float(self.history_secs) + 0.5, padding=0)
    self.realtime_plot.getAxis('bottom').setTicks(_time_ticks)
    self.realtime_img.sigClicked.connect(
        lambda x, y, mod: self._on_fast_graph_click(x, y, mod, window='current')
    )

    self.spectrogram_plot = pg.PlotWidget()
    self.spectrogram_plot.setLabel('left', 'H  Freq (MHz)')
    self.spectrogram_plot.getAxis('bottom').hide()
    self.spectrogram_plot.getViewBox().disableAutoRange()
    self.spectrogram_plot.getAxis('bottom').setTicks(_time_ticks)
    self.spectrogram_img = ClickableImageItem(axisOrder='col-major')
    self.spectrogram_plot.addItem(self.spectrogram_img)
    self.spectrogram_plot.setAspectLocked(False)
    self.spectrogram_img.setColorMap(colormap)
    self.spectrogram_img.sigClicked.connect(
        lambda x, y, mod: self._on_fast_graph_click(x, y, mod, window='previous')
    )

    # V-channel panes (dual-pol only — hidden in single-pol mode).
    # Bottom axis always hidden — the shared time ruler widget carries the scale.
    self.realtime_plot_v = pg.PlotWidget()
    self.realtime_plot_v.setLabel('left', 'V  Freq (MHz)')
    self.realtime_plot_v.getAxis('bottom').hide()
    self.realtime_plot_v.getAxis('bottom').setTicks(_time_ticks)
    self.realtime_plot_v.getViewBox().disableAutoRange()
    self.realtime_img_v = ClickableImageItem(axisOrder='col-major')
    self.realtime_plot_v.addItem(self.realtime_img_v)
    self.realtime_plot_v.setAspectLocked(False)
    self.realtime_img_v.setColorMap(colormap)
    self.realtime_plot_v.setXRange(0, float(self.history_secs) + 0.5, padding=0)
    self.realtime_img_v.sigClicked.connect(
        lambda x, y, mod: self._on_fast_graph_click(x, y, mod, window='current')
    )

    self.spectrogram_plot_v = pg.PlotWidget()
    self.spectrogram_plot_v.setLabel('left', 'V  Freq (MHz)')
    self.spectrogram_plot_v.getAxis('bottom').hide()
    self.spectrogram_plot_v.getAxis('bottom').setTicks(_time_ticks)
    self.spectrogram_plot_v.getViewBox().disableAutoRange()
    self.spectrogram_img_v = ClickableImageItem(axisOrder='col-major')
    self.spectrogram_plot_v.addItem(self.spectrogram_img_v)
    self.spectrogram_plot_v.setAspectLocked(False)
    self.spectrogram_img_v.setColorMap(colormap)
    self.spectrogram_plot_v.setXRange(0, float(self.history_secs) + 0.5, padding=0)
    self.spectrogram_img_v.sigClicked.connect(
        lambda x, y, mod: self._on_fast_graph_click(x, y, mod, window='previous')
    )
    self.realtime_plot_v.hide()
    self.spectrogram_plot.hide()    # shown only for live sources
    self.spectrogram_plot_v.hide()

    # Heatmap Y: logging dial offset kHz (fc_hz−1500)/1000 for markers; coarse IF = s×1 kHz + offset.
    _half_ch = N_CHANNELS // 2
    self._detect_freq_min_khz  = -(float(_half_ch) + 0.5)  # −24.5 kHz: channel k center aligns to y=k kHz
    self._detect_freq_span_khz =  float(N_CHANNELS)        # 48 kHz span unchanged

    _detect_title = (f"Channel Detection SNR  (threshold {DETECT_THRESH_DB:.0f} dB above noise)"
                     f"   circles: green=decoded  red=running  orange=no_decode")
    self.ch_detect_plot = pg.PlotWidget(title="H — " + _detect_title)
    self.ch_detect_plot.setLabel('left', 'Dial offset (kHz)')
    self.ch_detect_plot.setAspectLocked(False)
    self.ch_detect_img = pg.ImageItem(axisOrder='col-major')
    self.ch_detect_plot.addItem(self.ch_detect_img)
    self.ch_detect_img.setColorMap(colormap)
    self.ch_detect_curve_cyan   = pg.PlotCurveItem(pen=pg.mkPen((0,  220, 220),   width=1.5))  # SPD decoded
    self.ch_detect_curve_green  = pg.PlotCurveItem(pen=pg.mkPen('g',             width=1.5))  # jt9 decoded
    self.ch_detect_curve_red    = pg.PlotCurveItem(pen=pg.mkPen('r',             width=1.5))  # running
    self.ch_detect_curve_orange = pg.PlotCurveItem(pen=pg.mkPen((255, 140, 0),   width=1.5))  # no decode
    self.ch_detect_plot.addItem(self.ch_detect_curve_cyan)
    self.ch_detect_plot.addItem(self.ch_detect_curve_green)
    self.ch_detect_plot.addItem(self.ch_detect_curve_red)
    self.ch_detect_plot.addItem(self.ch_detect_curve_orange)
    self.ch_detect_plot.setXRange(0, 15.5, padding=0)
    self.ch_detect_plot.getAxis('bottom').hide()
    self.ch_detect_plot.setYRange(
        self._detect_freq_min_khz - 0.5,
        self._detect_freq_min_khz + self._detect_freq_span_khz + 0.5,
        padding=0,
    )

    # V-channel detection heatmap (hidden in single-pol mode)
    self.ch_detect_plot_v = pg.PlotWidget(title="V — " + _detect_title)
    self.ch_detect_plot_v.setLabel('left', 'Dial offset (kHz)')
    self.ch_detect_plot_v.getAxis('bottom').hide()
    self.ch_detect_plot_v.setAspectLocked(False)
    self.ch_detect_img_v = pg.ImageItem(axisOrder='col-major')
    self.ch_detect_plot_v.addItem(self.ch_detect_img_v)
    self.ch_detect_img_v.setColorMap(colormap)
    self.ch_detect_curve_cyan_v   = pg.PlotCurveItem(pen=pg.mkPen((0,  220, 220), width=1.5))  # SPD decoded
    self.ch_detect_curve_green_v  = pg.PlotCurveItem(pen=pg.mkPen('g',           width=1.5))  # jt9 decoded
    self.ch_detect_curve_red_v    = pg.PlotCurveItem(pen=pg.mkPen('r',           width=1.5))  # running
    self.ch_detect_curve_orange_v = pg.PlotCurveItem(pen=pg.mkPen((255, 140, 0), width=1.5))  # no decode
    self.ch_detect_plot_v.addItem(self.ch_detect_curve_cyan_v)
    self.ch_detect_plot_v.addItem(self.ch_detect_curve_green_v)
    self.ch_detect_plot_v.addItem(self.ch_detect_curve_red_v)
    self.ch_detect_plot_v.addItem(self.ch_detect_curve_orange_v)
    self.ch_detect_plot_v.setXRange(0, 15.5, padding=0)
    self.ch_detect_plot_v.setYRange(
        self._detect_freq_min_khz - 0.5,
        self._detect_freq_min_khz + self._detect_freq_span_khz + 0.5,
        padding=0,
    )
    self.ch_detect_plot_v.hide()

    # ── TFMF time-frequency surface (replaces channels×time sync heatmap) ───────
    # Shows the 2D matched-filter SNR surface (time on X, audio freq on Y) for
    # the most recently triggered channel.  Computed once per 15-s period at the
    # boundary; rendered by displays.py.  Peak-picked candidates appear as yellow
    # scatter dots.  Decode-lifecycle circles (cyan/green/red/orange) remain for
    # cross-referencing decodes, but are not redrawn on this panel (they live on
    # the tone-detection heatmap which retains the full-bandwidth channel view).
    _sync_title = ("TFMF Peak Detection  (dual-sync template, ±24 kHz wideband)"
                   "   yellow dots = peak-picked candidates")
    self.sync_detect_plot = pg.PlotWidget(title="H — " + _sync_title)
    self.sync_detect_plot.setLabel('left', 'Audio freq (kHz)')
    self.sync_detect_plot.setLabel('bottom', 'Time in period (s)')
    self.sync_detect_plot.setAspectLocked(False)
    self.sync_detect_img = pg.ImageItem(axisOrder='col-major')
    self.sync_detect_plot.addItem(self.sync_detect_img)
    self.sync_detect_img.setColorMap(colormap)
    self.sync_detect_scatter = pg.ScatterPlotItem(
        size=8, pen=pg.mkPen('y', width=1.5), brush=pg.mkBrush(None),
        symbol='o',
    )
    self.sync_detect_plot.addItem(self.sync_detect_scatter)
    # Keep curve items so existing code that addresses them by name doesn't fail,
    # but do not add them to the plot — they're no longer drawn here.
    self.sync_detect_curve_cyan   = pg.PlotCurveItem()
    self.sync_detect_curve_green  = pg.PlotCurveItem()
    self.sync_detect_curve_red    = pg.PlotCurveItem()
    self.sync_detect_curve_orange = pg.PlotCurveItem()
    self.sync_detect_plot.setXRange(0, 15.5, padding=0)
    # Hide this plot's own bottom axis — the shared time_ruler at the bottom
    # of the sync_detect window already carries the time scale (same
    # convention as realtime_plot in the fast-graph window).  Without this
    # hide, both axes render and the panel shows two redundant time scales.
    self.sync_detect_plot.getAxis('bottom').hide()
    self.sync_detect_plot.setYRange(-24.5, 24.5, padding=0)

    self.sync_detect_plot_v = pg.PlotWidget(title="V — " + _sync_title)
    self.sync_detect_plot_v.setLabel('left', 'Audio freq (kHz)')
    # Bottom axis hidden — shared time_ruler at the bottom of the window
    # carries the scale (same convention as H, see comment above).
    self.sync_detect_plot_v.getAxis('bottom').hide()
    self.sync_detect_plot_v.setAspectLocked(False)
    self.sync_detect_img_v = pg.ImageItem(axisOrder='col-major')
    self.sync_detect_plot_v.addItem(self.sync_detect_img_v)
    self.sync_detect_img_v.setColorMap(colormap)
    self.sync_detect_scatter_v = pg.ScatterPlotItem(
        size=8, pen=pg.mkPen('y', width=1.5), brush=pg.mkBrush(None),
        symbol='o',
    )
    self.sync_detect_plot_v.addItem(self.sync_detect_scatter_v)
    self.sync_detect_curve_cyan_v   = pg.PlotCurveItem()
    self.sync_detect_curve_green_v  = pg.PlotCurveItem()
    self.sync_detect_curve_red_v    = pg.PlotCurveItem()
    self.sync_detect_curve_orange_v = pg.PlotCurveItem()
    self.sync_detect_plot_v.setXRange(0, 15.5, padding=0)
    self.sync_detect_plot_v.setYRange(-24.5, 24.5, padding=0)
    self.sync_detect_plot_v.hide()
    self.sync_detect_plot.getViewBox().disableAutoRange()
    # Lock V's range too — without this, when the V panel becomes visible
    # (after a resize / dual-pol switch) its viewbox auto-fits and the
    # scatter dots end up at the right data coords but the image at a
    # different visible region.
    self.sync_detect_plot_v.getViewBox().disableAutoRange()

    # Do NOT setXLink here — PyQtGraph's link is bidirectional and would
    # override explicit setXRange calls on realtime_plot.
    self.realtime_plot.getViewBox().disableAutoRange()
    self.spectrogram_plot.getViewBox().disableAutoRange()
    self.ch_detect_plot.getViewBox().disableAutoRange()

    # ── Slider bar helper ─────────────────────────────────────────────────────
    def _make_slider_bar(title, min_label_ref, min_range, min_default,
                         max_label_ref, max_range, max_default,
                         min_slot, max_slot, tick_interval, scale=1):
        """Build a labelled min/max slider bar.

        *scale* divides slider integer positions to produce the displayed dB
        value.  scale=2 gives 0.5 dB resolution; scale=1 (default) gives 1 dB.
        *min_default* and *max_default* are the actual dB values (floats);
        the slider is initialised to round(value * scale).
        """
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        _fmt = (lambda v: f"{v:.1f}") if scale > 1 else (lambda v: f"{int(v)}")
        min_lbl = QtWidgets.QLabel(f"Min: {_fmt(min_default)} dB")
        setattr(self, min_label_ref, min_lbl)
        row.addWidget(min_lbl)
        min_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        min_sl.setMinimum(min_range[0]); min_sl.setMaximum(min_range[1])
        min_sl.setValue(round(min_default * scale))
        min_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        min_sl.setTickInterval(tick_interval)
        min_sl.valueChanged.connect(min_slot)
        row.addWidget(min_sl, stretch=1)
        max_lbl = QtWidgets.QLabel(f"Max: {_fmt(max_default)} dB")
        setattr(self, max_label_ref, max_lbl)
        row.addWidget(max_lbl)
        max_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        max_sl.setMinimum(max_range[0]); max_sl.setMaximum(max_range[1])
        max_sl.setValue(round(max_default * scale))
        max_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        max_sl.setTickInterval(tick_interval)
        max_sl.valueChanged.connect(max_slot)
        row.addWidget(max_sl, stretch=1)
        bar.setFixedHeight(54)
        return bar

    # ── Central widget: callsign decode list ──────────────────────────────────
    _mono9 = _QtGui.QFont("Monospace", 9)

    _decode_header = QtWidgets.QLabel("UTC       Freq(kHz)      DF   Pol    SNR    Message")
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

    # Thin separator between the two 15-second pairs (current / previous).
    # Shown only when the previous pair is visible (live source).
    self._fg_pair_sep = QtWidgets.QFrame()
    self._fg_pair_sep.setFixedHeight(6)
    self._fg_pair_sep.setStyleSheet("QFrame { background-color: #444; border: none; }")
    self._fg_pair_sep.hide()

    fg_layout = QtWidgets.QVBoxLayout(self._fast_graph_win)
    fg_layout.setContentsMargins(0, 0, 0, 0)
    fg_layout.setSpacing(0)
    # Pair 1: current 15 sec
    fg_layout.addWidget(self.realtime_plot,      stretch=1)
    fg_layout.addWidget(self.realtime_plot_v,    stretch=1)
    # Visual gap + pair 2: previous 15 sec
    fg_layout.addWidget(self._fg_pair_sep)
    fg_layout.addWidget(self.spectrogram_plot,   stretch=1)
    fg_layout.addWidget(self.spectrogram_plot_v, stretch=1)
    # Shared time axis ruler — all data plots have their bottom axis hidden
    fg_layout.addWidget(_make_time_ruler())
    fg_layout.addWidget(iq_sliders)

    # ── Panel window: Detection Heatmap ───────────────────────────────────────
    detect_sliders = _make_slider_bar(
        "Detection Color Scale",
        "detect_min_level_label",  (-20, 60),  self.detect_min_level,
        "detect_max_level_label",  (2,  100),  self.detect_max_level,
        self.on_detect_min_level_changed, self.on_detect_max_level_changed, 10,
        scale=2,
    )
    self._detect_win = _PanelWindow("MapMSK144 — Tone Detection SNR", det_action, self, 'detect_geometry')
    self._detect_win.setMinimumSize(400, 250)
    det_layout = QtWidgets.QVBoxLayout(self._detect_win)
    det_layout.setContentsMargins(0, 0, 0, 0)
    det_layout.setSpacing(0)
    det_layout.addWidget(self.ch_detect_plot,   stretch=1)
    det_layout.addWidget(self.ch_detect_plot_v, stretch=1)
    det_layout.addWidget(_make_time_ruler())
    det_layout.addWidget(detect_sliders)

    # ── Panel window: Sync-correlator Detection (parallel to tone detection) ──
    sync_detect_sliders = _make_slider_bar(
        "Sync Detection Color Scale",
        "sync_detect_min_level_label",  (-20, 60),  self.sync_detect_min_level,
        "sync_detect_max_level_label",  (2,  100),  self.sync_detect_max_level,
        self.on_sync_detect_min_level_changed, self.on_sync_detect_max_level_changed, 10,
        scale=2,
    )
    self._sync_detect_win = _PanelWindow(
        "MapMSK144 — Coherent Sync Detection", sync_det_action, self, 'sync_detect_geometry')
    self._sync_detect_win.setMinimumSize(400, 250)
    sync_det_layout = QtWidgets.QVBoxLayout(self._sync_detect_win)
    sync_det_layout.setContentsMargins(0, 0, 0, 0)
    sync_det_layout.setSpacing(0)
    sync_det_layout.addWidget(self.sync_detect_plot,   stretch=1)
    sync_det_layout.addWidget(self.sync_detect_plot_v, stretch=1)
    sync_det_layout.addWidget(_make_time_ruler())
    sync_det_layout.addWidget(sync_detect_sliders)

    # ── Panel windows: source-specific ───────────────────────────────────────
    setup_iq_nb_window(self,      iq_nb_action)
    setup_reporting_window(self,  reporting_action)
    setup_flex_window(self,       flex_action)
    setup_usrp_window(self,       usrp_action)
    setup_airspy_window(self,     airspy_action)
    setup_rtlsdr_window(self,     rtlsdr_action)
    setup_sdrangel_window(self,   sdrangel_action)
    setup_screenshot_window(self, screenshot_action)

    # ── Wire View menu actions ────────────────────────────────────────────────
    fg_action.triggered.connect(
        lambda checked: self._fast_graph_win.show() if checked else self._fast_graph_win.hide()
    )
    det_action.triggered.connect(
        lambda checked: self._detect_win.show() if checked else self._detect_win.hide()
    )
    sync_det_action.triggered.connect(
        lambda checked: self._sync_detect_win.show() if checked else self._sync_detect_win.hide()
    )
    iq_nb_action.triggered.connect(
        lambda checked: self._iq_nb_win.show() if checked else self._iq_nb_win.hide()
    )
    reporting_action.triggered.connect(
        lambda checked: self._reporting_win.show() if checked else self._reporting_win.hide()
    )
    # Action-style: open a fresh AnalysisWindow on each click.
    analysis_action.triggered.connect(
        lambda: _open_analysis_window(self, None)
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
    sdrangel_action.triggered.connect(
        lambda checked: self._sdrangel_win.show() if checked else self._sdrangel_win.hide()
    )
    screenshot_action.triggered.connect(
        lambda checked: self._screenshot_win.show() if checked else self._screenshot_win.hide()
    )

    # ── Restore saved geometry and visibility ─────────────────────────────────
    _win_settings = [
        (self,                 'window_geometry',     None),
        (self._fast_graph_win, 'fast_graph_geometry', QtCore.QRect(480, 50,  850, 650)),
        (self._detect_win,     'detect_geometry',     QtCore.QRect(480, 710, 850, 350)),
        (self._sync_detect_win,'sync_detect_geometry',QtCore.QRect(480, 1075,850, 350)),
        (self._iq_nb_win,      'iq_nb_geometry',      QtCore.QRect(50,  870, 380, 420)),
        (self._reporting_win,  'reporting_geometry',  QtCore.QRect(820, 870, 380, 500)),
        (self._flex_win,       'flex_geometry',       QtCore.QRect(450, 870, 360, 500)),
        (self._usrp_win,       'usrp_geometry',       QtCore.QRect(450, 870, 360, 440)),
        (self._airspy_win,     'airspy_geometry',     QtCore.QRect(450, 870, 360, 340)),
        (self._rtlsdr_win,     'rtlsdr_geometry',     QtCore.QRect(450, 870, 360, 340)),
        (self._sdrangel_win,   'sdrangel_geometry',   QtCore.QRect(450, 100, 360, 400)),
        (self._screenshot_win, 'screenshot_geometry', QtCore.QRect(50,  50,  340, 440)),
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
    sync_det_visible = _SETTINGS.value('sync_detect_visible', True, type=bool)
    iq_nb_visible      = _SETTINGS.value('iq_nb_visible',       True,  type=bool)
    reporting_visible  = _SETTINGS.value('reporting_visible',    True,  type=bool)
    fg_action.setChecked(fg_visible)
    det_action.setChecked(det_visible)
    sync_det_action.setChecked(sync_det_visible)
    iq_nb_action.setChecked(iq_nb_visible)
    reporting_action.setChecked(reporting_visible)
    if fg_visible:         self._fast_graph_win.show()
    if det_visible:        self._detect_win.show()
    if sync_det_visible:   self._sync_detect_win.show()
    if iq_nb_visible:      self._iq_nb_win.show()
    if reporting_visible:  self._reporting_win.show()

    # Radio windows start hidden
    for win in (self._flex_win, self._usrp_win, self._airspy_win, self._rtlsdr_win, self._sdrangel_win):
        win.hide()
        if win._view_action is not None:
            win._view_action.setChecked(False)

    # ── Info panel — vertical stack below decode list ─────────────────────────
    # Four compact rows: receiver | centre freq | channel range | UTC
    # Same font as the decode list so the bottom of the window is visually uniform.
    _info_ss = (
        "QLabel { background: #2a2a2a; color: #aaaaaa; padding: 1px 4px; }"
    )
    _info_ss_bold = (
        "QLabel { background: #2a2a2a; color: #cccccc; padding: 1px 4px; }"
    )

    self._receiver_label    = QtWidgets.QLabel("—")
    self.tuned_freq_label   = QtWidgets.QLabel("—")
    self._msk_monitor_label = QtWidgets.QLabel("Channels: —")
    self.utc_clock_label    = QtWidgets.QLabel("UTC: —")

    for _lbl in (self._receiver_label, self.tuned_freq_label,
                 self._msk_monitor_label, self.utc_clock_label):
        _lbl.setFont(_mono9)

    self._receiver_label.setStyleSheet(_info_ss_bold)
    self.tuned_freq_label.setStyleSheet(_info_ss_bold)
    self._msk_monitor_label.setStyleSheet(_info_ss)
    self._msk_monitor_label.setToolTip(
        "Coarse IF channel offsets (kHz) from pan centre; see channel_plan.py"
    )
    self.utc_clock_label.setStyleSheet(_info_ss_bold)

    _info_sep = QtWidgets.QFrame()
    _info_sep.setFrameShape(QtWidgets.QFrame.HLine)
    _info_sep.setStyleSheet("QFrame { color: #555; }")

    central_vbox.addWidget(_info_sep)
    central_vbox.addWidget(self._receiver_label)
    central_vbox.addWidget(self.tuned_freq_label)
    central_vbox.addWidget(self._msk_monitor_label)
    central_vbox.addWidget(self.utc_clock_label)

    # Keep status bar for transient messages only (capture saves, errors, etc.)
    self.statusBar().showMessage('Initializing...')

    # Apply initial band state: enable/disable source menu actions for saved band.
    self.on_band_changed(_saved_band_idx)

    # ── Restore last source mode ──────────────────────────────────────────────
    # Defer to a singleShot(0) so the event loop runs first: windows paint,
    # the SIGINT-tickling update_timer ticks, and a slow source connect
    # (e.g. Flex ~37 s) doesn't freeze the UI or block Ctrl-C.
    _last_mode = _SETTINGS.value('source_mode', 'idle', type=str)
    _last_wav  = _SETTINGS.value('selected_wav_path', '', type=str)
    if _last_mode == 'radio':
        QtCore.QTimer.singleShot(0, self.on_select_source_radio)
    elif _last_mode == 'usrp':
        QtCore.QTimer.singleShot(0, self.on_select_source_usrp)
    elif _last_mode == 'airspy':
        QtCore.QTimer.singleShot(0, self.on_select_source_airspy)
    elif _last_mode == 'rtlsdr':
        QtCore.QTimer.singleShot(0, self.on_select_source_rtlsdr)
    elif _last_mode == 'sdrangel':
        QtCore.QTimer.singleShot(0, self.on_select_source_sdrangel)
    elif _last_mode == 'wav':
        # Do not auto-start WAV playback on restart — user must explicitly
        # choose a file.  Auto-playing can cause a crash when a previous
        # run's threads are still cleaning up, and it's surprising behaviour.
        pass

    self.update_timer = QtCore.QTimer()
    self.update_timer.timeout.connect(self.update_displays)
    self.update_timer.start(100)

    from .displays import recolor_decode_panel
    self._age_timer = QtCore.QTimer()
    self._age_timer.timeout.connect(lambda: recolor_decode_panel(self.decode_panel))
    self._age_timer.start(15_000)   # recolor every 15 s


def on_min_level_changed(self, value):
    self.min_level = value
    self.min_level_label.setText(f"Min: {value} dB")
    self.spectrogram_img.setLevels([value, self.max_level])
    self.realtime_img.setLevels([value, self.max_level])


def on_max_level_changed(self, value):
    self.max_level = value
    self.max_level_label.setText(f"Max: {value} dB")
    self.spectrogram_img.setLevels([self.min_level, value])
    self.realtime_img.setLevels([self.min_level, value])


def on_detect_min_level_changed(self, value):
    self.detect_min_level = value * 0.5
    self.detect_min_level_label.setText(f"Min: {self.detect_min_level:.1f} dB")
    self.ch_detect_img.setLevels([self.detect_min_level, self.detect_max_level])


def on_detect_max_level_changed(self, value):
    self.detect_max_level = value * 0.5
    self.detect_max_level_label.setText(f"Max: {self.detect_max_level:.1f} dB")
    self.ch_detect_img.setLevels([self.detect_min_level, self.detect_max_level])


def on_sync_detect_min_level_changed(self, value):
    self.sync_detect_min_level = value * 0.5
    self.sync_detect_min_level_label.setText(f"Min: {self.sync_detect_min_level:.1f} dB")
    if hasattr(self, 'sync_detect_img'):
        self.sync_detect_img.setLevels([self.sync_detect_min_level, self.sync_detect_max_level])
    if hasattr(self, 'sync_detect_img_v'):
        self.sync_detect_img_v.setLevels([self.sync_detect_min_level, self.sync_detect_max_level])


def on_sync_detect_max_level_changed(self, value):
    self.sync_detect_max_level = value * 0.5
    self.sync_detect_max_level_label.setText(f"Max: {self.sync_detect_max_level:.1f} dB")
    if hasattr(self, 'sync_detect_img'):
        self.sync_detect_img.setLevels([self.sync_detect_min_level, self.sync_detect_max_level])
    if hasattr(self, 'sync_detect_img_v'):
        self.sync_detect_img_v.setLevels([self.sync_detect_min_level, self.sync_detect_max_level])


def on_nb_factor_changed(self, value):
    self.nb_factor = value * 0.1
    self.nb_factor_label.setText(f"{self.nb_factor:.1f}")
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
    _SETTINGS.setValue('nb_factor', self.nb_factor)
    # Large K steps change blanker throughput materially → detection
    # baseline can lag and falsely trigger the sustained-signal lockout.
    reset_detection_baseline(self)


def on_nb_factor_v_changed(self, value):
    self.nb_factor_v = value * 0.1
    if hasattr(self, 'nb_factor_v_label'):
        self.nb_factor_v_label.setText(f"{self.nb_factor_v:.1f}")
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
    _SETTINGS.setValue('nb_factor_v', self.nb_factor_v)
    reset_detection_baseline(self)


def _on_show_raw_power_toggled(self, checked):
    """Diagnostic: toggle the detection heatmap between pair_metric (dB
    above pct25) and raw squared-FFT power.  Detection-trigger logic is
    unaffected; only the displayed values change.

    See processing.process_iq_data — the heatmap-write site checks
    ``self._show_raw_power`` and writes the appropriate values to
    ``_ch_snr_history_h/v`` (squared-FFT detection panel) and
    ``_sync_snr_history_h/v`` (sync-correlator detection panel).

    Implementation note on display scaling: raw squared-FFT power sits
    around −90..−50 dBFS, well below the existing colour-scale slider's
    reach; raw sync-correlator output sits already in the slider window.
    Rather than fight the slider range, processing.py applies path-
    specific offsets (``_RAW_POWER_DISPLAY_OFFSET_DB_SQ`` = +85 dB,
    ``_RAW_POWER_DISPLAY_OFFSET_DB_SYNC`` = 0 dB) so both modes use the
    same slider calibration.  Channel-to-channel differences remain
    accurate; only the absolute dBFS axis is shifted.

    Not persisted across sessions: a diagnostic mode left enabled
    between runs would mislead the live operator into thinking the
    colour scale is broken.
    """
    self._show_raw_power = bool(checked)
    # Clear the history buffers so the freshly-toggled mode starts clean
    # at the current write position — without this, half the heatmap
    # would show old-mode values until the ring fills.
    if hasattr(self, '_ch_snr_history_h'):
        self._ch_snr_history_h[:] = -999.0
    if hasattr(self, '_ch_snr_history_v'):
        self._ch_snr_history_v[:] = -999.0
    if hasattr(self, '_sync_snr_history_h'):
        self._sync_snr_history_h[:] = -999.0
    if hasattr(self, '_sync_snr_history_v'):
        self._sync_snr_history_v[:] = -999.0


def on_nb_backend_changed(self, name):
    """User picked a blanker backend — apply and persist."""
    from .visualizer import _SETTINGS
    self.set_blanker(name)
    _SETTINGS.setValue('nb_backend', name)


def on_td_scale_changed(self, value):
    self.td_scale = value * 0.01   # slider 1–100 → y-max 0.01–1.0
    self.td_plot.setYRange(0.0, self.td_scale, padding=0)
    if hasattr(self, 'td_plot_v'):
        self.td_plot_v.setYRange(0.0, self.td_scale, padding=0)


def on_td_span_changed(self, value):
    self.td_span_ms = float(value)
    self.td_span_val_label.setText(f"{value} ms")
    self.td_plot.setTitle(f'IQ Magnitude — {value} ms')
    _td_ruler = getattr(self, 'td_ruler', None)
    if _td_ruler is not None:
        _td_ruler.setXRange(0.0, float(value), padding=0)


def on_select_source_airspy(self):
    from .runtime import _connect_airspy_client
    from .source_windows import show_source_window
    _connect_airspy_client(self)
    self.source_mode = "airspy"
    self.selected_wav_path = None
    self.source_airspy_action.setChecked(True)
    show_source_window(self, "airspy")
    self._receiver_label.setText("Airspy HF+")


def on_select_source_rtlsdr(self):
    from .runtime import _connect_rtlsdr_client
    from .source_windows import show_source_window
    _connect_rtlsdr_client(self)
    self.source_mode = "rtlsdr"
    self.selected_wav_path = None
    self.source_rtlsdr_action.setChecked(True)
    show_source_window(self, "rtlsdr")
    self._receiver_label.setText("RTL-SDR")


def on_select_source_usrp(self):
    from .runtime import _connect_usrp_client
    from .source_windows import show_source_window
    _connect_usrp_client(self)
    self.source_mode = "usrp"
    self.selected_wav_path = None
    self.source_usrp_action.setChecked(True)
    show_source_window(self, "usrp")
    self._receiver_label.setText("USRP B210")


def on_select_source_radio(self):
    from .source_windows import show_source_window
    self._connect_radio_client()
    self.source_mode = "radio"
    self.selected_wav_path = None
    self.source_radio_action.setChecked(True)
    show_source_window(self, "radio")
    self._receiver_label.setText("Flex Radio")


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
    self._receiver_label.setText("Simulation")
    # WAV has embedded freq metadata — band combo doesn't apply
    if hasattr(self, '_band_combo'):
        self._band_combo.setEnabled(False)


def on_select_source_sdrangel(self):
    from .runtime import _connect_sdrangel_client
    from .source_windows import show_source_window
    _connect_sdrangel_client(self)
    self.source_mode = "sdrangel"
    self.selected_wav_path = None
    self.source_sdrangel_action.setChecked(True)
    show_source_window(self, "sdrangel")
    self._receiver_label.setText("SDRangel")


def _retune_active_source(self, freq_mhz: float):
    """Retune the currently streaming source to freq_mhz, if it supports it."""
    mode = getattr(self, 'source_mode', 'idle')
    if mode == 'usrp':
        uc = getattr(self, 'usrp_client', None)
        if uc is not None and getattr(self, '_usrp_started', False):
            uc.retune(freq_mhz)
    elif mode == 'rtlsdr':
        rc = getattr(self, 'rtlsdr_client', None)
        if rc is not None and getattr(self, '_rtlsdr_started', False):
            rc.retune(freq_mhz)
    elif mode == 'airspy':
        ac = getattr(self, 'airspy_client', None)
        if ac is not None and getattr(self, '_airspy_started', False):
            ac.retune(freq_mhz)
    elif mode == 'sdrangel':
        sc = getattr(self, 'sdrangel_client', None)
        if sc is not None and getattr(self, '_sdrangel_started', False):
            # UDP Sink output is centered on calling_freq; no center_freq override needed.
            sc.retune(freq_mhz)
    # Flex (radio): center is controlled by the radio hardware — no retune


def on_band_changed(self, idx: int):
    """Handle band combo selection: update freq, filter source menu, retune if active."""
    from .visualizer import _SETTINGS as _VS
    _label, freq, supported = MSK144_BANDS[idx]

    # Update engine state
    self.calling_freq_mhz = freq
    self.center_freq_mhz  = freq
    self._rebuild_channelizer_state()
    self.display_center_freq_mhz = -1.0   # force freq-axis refresh on next display update

    # Enable only sources that can reach this band; disable the rest.
    # WAV is always available (file may contain any band).
    self.source_radio_action.setEnabled('radio'      in supported)
    self.source_usrp_action.setEnabled('usrp'        in supported)
    self.source_airspy_action.setEnabled('airspy'    in supported)
    self.source_rtlsdr_action.setEnabled('rtlsdr'    in supported)
    self.source_sdrangel_action.setEnabled('sdrangel' in supported)

    # If the active source is now unsupported, uncheck it visually
    # (user must explicitly switch; we don't auto-stop the hardware).
    mode = getattr(self, 'source_mode', 'idle')
    _mode_action = {
        'radio':    getattr(self, 'source_radio_action',    None),
        'usrp':     getattr(self, 'source_usrp_action',     None),
        'airspy':   getattr(self, 'source_airspy_action',   None),
        'rtlsdr':   getattr(self, 'source_rtlsdr_action',   None),
        'sdrangel': getattr(self, 'source_sdrangel_action', None),
    }
    if mode in _mode_action and mode not in supported:
        act = _mode_action[mode]
        if act is not None:
            act.setChecked(False)

    # Retune the active source if it supports the new band
    if mode in supported:
        _retune_active_source(self, freq)

    # Re-enable band combo (may have been disabled by WAV mode)
    if hasattr(self, '_band_combo') and mode != 'wav':
        self._band_combo.setEnabled(True)

    # Persist
    _VS.setValue('calling_freq_mhz', freq)


def _on_about(self):
    version = _get_version_string()
    QtWidgets.QMessageBox.about(
        self,
        "About map144",
        f"<b>map144</b> — MSK144 Meteor Scatter Decoder<br><br>"
        f"<b>Version:</b> {version}<br><br>"
        f"Copyright &copy; 2026 Jeff Millar, WA1HCO<br>"
        f"GNU General Public License v3",
    )


def _save_state_snapshot(self):
    """Diagnostics → Save State Snapshot — dumps the engine's internal
    arrays (rolling pct25 history, current pct25, blanker state, the
    detection-heatmap buffer) to a timestamped ``.npz`` file.  Used to
    diagnose live-engine behaviour that doesn't reproduce in offline
    tests.
    """
    from datetime import datetime
    from pathlib import Path
    out_dir = Path(__file__).parent.parent / "MSK144" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"state_snapshot_{stamp}.npz"
    try:
        self.dump_diagnostic_state(str(path))
        self.statusBar().showMessage(f"State snapshot saved: {path.name}", 4000)
    except Exception as exc:
        self.statusBar().showMessage(
            f"State snapshot failed: {exc}", 6000,
        )


def _on_fast_graph_click(self, x_sec, y_mhz, modifiers, window='current'):
    """Handle click on realtime or spectrogram image — save capture, optionally open analysis."""
    from PyQt5.QtCore import Qt
    from .capture import collect_capture_iq, save_capture, CAPTURES_DIR
    from pathlib import Path

    iq = collect_capture_iq(self, window=window)
    if iq.size == 0:
        self.statusBar().showMessage("Capture failed: ring buffer not yet full", 3000)
        return

    mode = getattr(self, 'source_mode', 'unknown')
    src_wav = getattr(self, 'selected_wav_path', None)
    is_test = mode == 'wav'
    _source_labels = {
        'radio':  'Flex Radio',
        'airspy': 'Airspy HF+',
        'rtlsdr': 'RTL-SDR',
        'usrp':   'USRP B210',
    }
    if mode == 'wav' and src_wav:
        from pathlib import Path as _Path
        source_label = f"WAV: {_Path(src_wav).name}"
    else:
        source_label = _source_labels.get(mode, mode)

    wav_path, json_path = save_capture(
        iq,
        center_freq_mhz = float(self.display_center_freq_mhz),
        source          = source_label,
        nb_factor       = float(getattr(self, 'nb_factor', 6.0)),
        nb_enabled      = True,
        is_test_wav     = is_test,
        source_wav_path = src_wav,
    )

    msg = f"Capture saved: {wav_path.name}"
    self.statusBar().showMessage(msg, 5000)
    print(f"[capture] {msg}", flush=True)

    if modifiers & Qt.ShiftModifier:
        _open_analysis_window(self, wav_path)


def _open_analysis_window(self, wav_path):
    """Launch the analysis window with the given capture WAV.

    Each call opens a fresh independent window (multiple captures can be
    analysed side-by-side).  If wav_path is None, opens a file dialog first.
    """
    from pathlib import Path
    from PyQt5.QtCore import Qt
    from .capture import browse_captures
    from .analysis_window import AnalysisWindow

    if wav_path is None:
        wav_path = browse_captures(self)
        if wav_path is None:
            return

    reporter = getattr(self, 'reporter', None)
    win = AnalysisWindow(Path(wav_path), reporter=reporter, parent=self)
    # Keep a reference so Python/Qt don't garbage-collect it.
    if not hasattr(self, '_analysis_windows'):
        self._analysis_windows = []
    self._analysis_windows.append(win)
    win.setAttribute(Qt.WA_DeleteOnClose)
    win.destroyed.connect(lambda: self._analysis_windows.remove(win)
                          if win in self._analysis_windows else None)
    win.show()


def _on_open_capture(self):
    """File → Captures → Open Capture..."""
    from .capture import browse_captures
    wav_path = browse_captures(self)
    if wav_path:
        _open_analysis_window(self, wav_path)


def _on_browse_captures(self):
    """File → Captures → Browse Captures Folder."""
    import subprocess as _sp
    from .capture import CAPTURES_DIR
    from pathlib import Path
    d = Path(CAPTURES_DIR).resolve()
    d.mkdir(parents=True, exist_ok=True)
    try:
        _sp.Popen(['xdg-open', str(d)])
    except Exception:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.information(self, "Captures Folder", str(d))
