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
        File — mutually exclusive source selection (Flex Radio / WAV File)
        View — show/hide each panel window
    Central widget:
        Callsign decode list (column header + QListWidget)
    Status bar:
        Live power / packet stats  |  tuned-frequency label  |  UTC clock

Free-floating panel windows (QWidget with Qt.Window flag)
    Fast Graph        Accumulated + real-time IQ spectrograms; IQ colour-scale sliders.
                      Matches the WSJTX "Fast Graph" panel name for MSK144 mode.
    Detection Heatmap Per-channel SNR heatmap with threshold markers;
                      detection colour-scale sliders.
    Radio Interface   Source selection, live signal-flow statistics, and
                      wideband noise-blanker control.

Each panel window:
  - can be moved and resized independently on any monitor
  - hides (rather than closes) when the user clicks its X button, keeping
    the View menu action in sync
  - has its position, size, and visibility persisted in QSettings so it
    reopens exactly where it was left
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class _PanelWindow(QtWidgets.QWidget):
    """Free-floating display panel that hides instead of closing.

    Clicking the window's X button hides the window and unchecks the
    corresponding View menu action.  When the application is shutting
    down (``_app_closing`` is set on the parent main window) the window
    accepts the close event normally so Qt can tear everything down.
    """

    def __init__(self, title, view_action, parent):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self._view_action = view_action

    def closeEvent(self, event):
        parent = self.parent()
        if parent is not None and getattr(parent, '_app_closing', False):
            event.accept()
            return
        event.ignore()
        self.hide()
        if self._view_action is not None:
            self._view_action.setChecked(False)


def setup_ui(self):
    """Build the main window and the three free-floating panel windows."""
    from PyQt5 import QtGui as _QtGui
    from .processing import DETECT_THRESH_DB
    from .channelizer import N_CHANNELS
    from .visualizer import _SETTINGS

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

    self.source_wav_action = QtWidgets.QAction("WAV File", self)
    self.source_wav_action.setCheckable(True)
    self.source_wav_action.triggered.connect(self.on_select_source_wav)
    self.source_action_group.addAction(self.source_wav_action)
    file_menu.addAction(self.source_wav_action)

    view_menu = menu_bar.addMenu("&View")

    # Create View actions first so panel windows can reference them
    fg_action  = QtWidgets.QAction("Fast Graph",        self)
    det_action = QtWidgets.QAction("Detection Heatmap", self)
    ri_action  = QtWidgets.QAction("Radio Interface",   self)
    for act in (fg_action, det_action, ri_action):
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
    # Do NOT setXLink here.  PyQtGraph's link is bidirectional: any bounds-change
    # signal on ch_detect_plot (from setImage) calls linkedViewChanged on
    # realtime_plot and forces its range, overriding explicit setXRange calls.
    # Both plots are kept at [0, 15] by explicit setXRange each frame instead.
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
    self._fast_graph_win = _PanelWindow("MapMSK144 — Fast Graph", fg_action, self)
    self._fast_graph_win.setMinimumSize(500, 350)
    fg_layout = QtWidgets.QVBoxLayout(self._fast_graph_win)
    fg_layout.setContentsMargins(0, 0, 0, 0)
    fg_layout.setSpacing(0)
    fg_layout.addWidget(self.realtime_plot,    stretch=1)   # upper: current window (WSJTX style)
    fg_layout.addWidget(self.spectrogram_plot, stretch=1)   # lower: previous window
    fg_layout.addWidget(iq_sliders)

    # ── Panel window: Detection Heatmap ───────────────────────────────────────
    detect_sliders = _make_slider_bar(
        "Detection Color Scale",
        "detect_min_level_label",  (0, 20),  self.detect_min_level,
        "detect_max_level_label",  (1, 50),  self.detect_max_level,
        self.on_detect_min_level_changed, self.on_detect_max_level_changed, 5,
    )
    self._detect_win = _PanelWindow("MapMSK144 — Detection Heatmap", det_action, self)
    self._detect_win.setMinimumSize(400, 250)
    det_layout = QtWidgets.QVBoxLayout(self._detect_win)
    det_layout.setContentsMargins(0, 0, 0, 0)
    det_layout.setSpacing(0)
    det_layout.addWidget(self.ch_detect_plot, stretch=1)
    det_layout.addWidget(detect_sliders)

    # ── Panel window: Radio Interface ─────────────────────────────────────────
    def _stat_label(attr_name, init="—", color="#1a6b1a"):
        lbl = QtWidgets.QLabel(init)
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")
        setattr(self, attr_name, lbl)
        return lbl

    def _form_group(title):
        """Return (QGroupBox, QFormLayout) pair with compact row spacing."""
        grp  = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(grp)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(8)
        return grp, form

    self._radio_iface_win = _PanelWindow("MapMSK144 — Radio Interface", ri_action, self)
    self._radio_iface_win.setMinimumSize(300, 400)
    ri_layout = QtWidgets.QVBoxLayout(self._radio_iface_win)
    ri_layout.setContentsMargins(8, 8, 8, 8)
    ri_layout.setSpacing(6)

    # ── Source selection ──────────────────────────────────────────────────────
    src_group = QtWidgets.QGroupBox("Source")
    src_vbox  = QtWidgets.QVBoxLayout(src_group)
    src_vbox.setSpacing(3)
    self._ri_radio_btn = QtWidgets.QRadioButton("Flex Radio")
    self._ri_wav_btn   = QtWidgets.QRadioButton("WAV File")
    self._ri_radio_btn.setChecked(True)   # set before connecting signals

    wav_row = QtWidgets.QHBoxLayout()
    wav_row.addWidget(self._ri_wav_btn)
    self._ri_browse_btn = QtWidgets.QPushButton("Browse...")
    self._ri_browse_btn.setEnabled(False)
    wav_row.addWidget(self._ri_browse_btn)
    wav_row.addStretch()

    self._ri_wav_path_label = QtWidgets.QLabel("")
    self._ri_wav_path_label.setStyleSheet("QLabel { color: #888; font-size: 8pt; }")
    self._ri_wav_path_label.setWordWrap(True)

    src_vbox.addWidget(self._ri_radio_btn)
    src_vbox.addLayout(wav_row)
    src_vbox.addWidget(self._ri_wav_path_label)
    ri_layout.addWidget(src_group)

    self._ri_radio_btn.toggled.connect(
        lambda checked: self.on_select_source_radio() if checked else None
    )
    self._ri_wav_btn.toggled.connect(
        lambda checked: self._ri_browse_btn.setEnabled(checked)
    )
    self._ri_wav_btn.toggled.connect(
        lambda checked: self.on_select_source_wav() if checked else None
    )
    self._ri_browse_btn.clicked.connect(self.on_select_source_wav)

    # ── Radio identity & connection status ────────────────────────────────────
    radio_grp, radio_form = _form_group("Radio")
    radio_form.addRow("Status:",   _stat_label("_ri_radio_status_val", "Idle", "#c76000"))
    radio_form.addRow("Model:",    _stat_label("_ri_model_val"))
    radio_form.addRow("Serial:",   _stat_label("_ri_serial_val"))
    radio_form.addRow("Firmware:", _stat_label("_ri_firmware_val"))
    radio_form.addRow("IP:",       _stat_label("_ri_radio_ip_val"))
    ri_layout.addWidget(radio_grp)

    # ── DAXIQ stream ──────────────────────────────────────────────────────────
    dax_grp, dax_form = _form_group("DAXIQ Stream")
    dax_form.addRow("DAX Channel:", _stat_label("_ri_dax_ch_val"))
    dax_form.addRow("Stream ID:",   _stat_label("_ri_stream_id_val"))
    dax_form.addRow("UDP Port:",    _stat_label("_ri_udp_port_val"))
    dax_form.addRow("Mode:",        _stat_label("_ri_mode_val"))
    ri_layout.addWidget(dax_grp)

    # ── IF parameters ─────────────────────────────────────────────────────────
    if_grp, if_form = _form_group("IF Parameters")
    if_form.addRow("Center Freq:", _stat_label("_ri_freq_val"))
    if_form.addRow("Sample Rate:", _stat_label("_ri_rate_val"))
    ri_layout.addWidget(if_grp)

    # ── Stream health ─────────────────────────────────────────────────────────
    health_grp, health_form = _form_group("Stream Health")
    health_form.addRow("Pkt Rate:", _stat_label("_ri_packets_val"))
    health_form.addRow("Loss:",    _stat_label("_ri_loss_val"))
    health_form.addRow("Drops:",   _stat_label("_ri_drops_val", color="#c76000"))
    ri_layout.addWidget(health_grp)

    # ── IQ magnitude plot (time domain, 200 ms) ──────────────────────────────
    self.td_plot = pg.PlotWidget()
    self.td_plot.setLabel('left',   'Magnitude')
    self.td_plot.setLabel('bottom', 'Time', units='ms')
    self.td_span_ms = float(int(_SETTINGS.value('td_span', 200)))
    self.td_plot.setTitle(f'IQ Magnitude — {int(self.td_span_ms)} ms')
    self.td_plot.setBackground('#111111')
    self.td_plot.getAxis('left').setWidth(45)
    self.td_plot.setMinimumHeight(130)
    self.td_plot.setMaximumHeight(180)

    _td_n  = int(0.200 * self.sample_rate)   # buffer always holds 200 ms max

    self.td_curve = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#4fc3f7', width=1),
    )
    self.td_plot.addItem(self.td_curve)
    self.td_plot.setXRange(0.0, 200.0, padding=0)

    # Horizontal reference line at K × (median noise floor) — updated each frame
    self.td_thresh_line = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen('#ff7043', width=1, style=QtCore.Qt.DashLine),
    )
    self.td_plot.addItem(self.td_thresh_line)

    # Vertical scale slider — controls y-axis max.
    # Integer ticks 1–20 map to actual y-max 0.1–2.0 (multiplied by 0.1).
    self.td_scale_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    self.td_scale_slider.setMinimum(1)
    self.td_scale_slider.setMaximum(100)
    try:
        _td_scale_saved = int(float(_SETTINGS.value('td_scale', 10)))
    except (ValueError, TypeError):
        _td_scale_saved = 10
    self.td_scale_slider.setValue(max(1, min(100, _td_scale_saved)))
    self.td_scale_slider.setTickPosition(QtWidgets.QSlider.TicksRight)
    self.td_scale_slider.setTickInterval(10)
    self.td_scale_slider.setInvertedAppearance(True)   # top of slider = most sensitive (smallest y-max)
    self.td_scale_slider.valueChanged.connect(self.on_td_scale_changed)

    self.td_scale = self.td_scale_slider.value() * 0.001   # slider 1–100 → y-max 0.001–0.1
    self.td_plot.setYRange(0.0, self.td_scale, padding=0)

    td_row = QtWidgets.QHBoxLayout()
    td_row.setSpacing(2)
    td_row.addWidget(self.td_plot, stretch=1)
    td_row.addWidget(self.td_scale_slider)
    ri_layout.addLayout(td_row)

    # ── TD span slider (20–200 ms) ────────────────────────────────────────────
    td_span_row = QtWidgets.QHBoxLayout()
    td_span_row.setSpacing(4)
    td_span_row.addWidget(QtWidgets.QLabel("Span:"))
    self.td_span_val_label = QtWidgets.QLabel(f"{int(self.td_span_ms)} ms")
    self.td_span_val_label.setStyleSheet(
        "QLabel { color: #c8e6c9; font-family: monospace; min-width: 52px; }"
    )
    td_span_row.addWidget(self.td_span_val_label)
    self.td_span_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.td_span_slider.setMinimum(20)
    self.td_span_slider.setMaximum(200)
    self.td_span_slider.setSingleStep(10)
    self.td_span_slider.setValue(int(self.td_span_ms))
    self.td_span_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.td_span_slider.setTickInterval(20)
    self.td_span_slider.valueChanged.connect(self.on_td_span_changed)
    td_span_row.addWidget(self.td_span_slider, stretch=1)
    ri_layout.addLayout(td_span_row)

    # ── Wideband noise blanker ────────────────────────────────────────────────
    nb_group = QtWidgets.QGroupBox("Spectral Noise Blanker")
    nb_vbox  = QtWidgets.QVBoxLayout(nb_group)
    nb_vbox.setSpacing(3)

    nb_factor_row = QtWidgets.QHBoxLayout()
    nb_factor_row.addWidget(QtWidgets.QLabel("K (× noise floor):"))
    self.nb_factor_label = QtWidgets.QLabel(f"{self.nb_factor:.1f}")
    self.nb_factor_label.setStyleSheet(
        "QLabel { color: #c8e6c9; font-family: monospace; min-width: 32px; }"
    )
    nb_factor_row.addWidget(self.nb_factor_label)
    nb_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    nb_sl.setMinimum(10); nb_sl.setMaximum(100)
    nb_sl.setValue(int(round(self.nb_factor * 10)))
    nb_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
    nb_sl.setTickInterval(10)
    nb_sl.valueChanged.connect(self.on_nb_factor_changed)
    nb_factor_row.addWidget(nb_sl, stretch=1)
    nb_vbox.addLayout(nb_factor_row)

    nb_event_row = QtWidgets.QHBoxLayout()
    nb_event_row.addWidget(QtWidgets.QLabel("Blanked blocks:"))
    self._ri_nb_count_val = QtWidgets.QLabel("0")
    self._ri_nb_count_val.setStyleSheet(
        "QLabel { color: #c76000; font-family: monospace; }"
    )
    nb_event_row.addWidget(self._ri_nb_count_val)
    nb_event_row.addStretch()
    nb_vbox.addLayout(nb_event_row)

    ri_layout.addWidget(nb_group)
    ri_layout.addStretch()

    # ── Wire View menu actions to show/hide each panel window ─────────────────
    fg_action.triggered.connect(
        lambda checked: self._fast_graph_win.show()    if checked else self._fast_graph_win.hide()
    )
    det_action.triggered.connect(
        lambda checked: self._detect_win.show()        if checked else self._detect_win.hide()
    )
    ri_action.triggered.connect(
        lambda checked: self._radio_iface_win.show()   if checked else self._radio_iface_win.hide()
    )

    # ── Restore saved geometry and visibility ─────────────────────────────────
    # Default positions: stagger panels so they don't completely overlap on first run
    _defaults = [
        (self,                  'window_geometry',      None),
        (self._fast_graph_win,  'fast_graph_geometry',  QtCore.QRect(480, 50,  850, 650)),
        (self._detect_win,      'detect_geometry',      QtCore.QRect(480, 710, 850, 350)),
        (self._radio_iface_win, 'radio_iface_geometry', QtCore.QRect(50,  870, 400, 760)),
    ]
    for win, key, default_rect in _defaults:
        geo = _SETTINGS.value(key)
        if geo:
            win.restoreGeometry(geo)
        elif default_rect is not None:
            win.setGeometry(default_rect)

    fg_visible  = _SETTINGS.value('fast_graph_visible',   True,  type=bool)
    det_visible = _SETTINGS.value('detect_visible',        True,  type=bool)
    ri_visible  = _SETTINGS.value('radio_iface_visible',   True,  type=bool)
    fg_action.setChecked(fg_visible)
    det_action.setChecked(det_visible)
    ri_action.setChecked(ri_visible)
    if fg_visible:  self._fast_graph_win.show()
    if det_visible: self._detect_win.show()
    if ri_visible:  self._radio_iface_win.show()

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
    self.nb_factor = value * 0.1    # slider 10–100 → K 1.0–10.0
    self.nb_factor_label.setText(f"{self.nb_factor:.1f}")


def on_td_scale_changed(self, value):
    self.td_scale = value * 0.001    # slider 1–100 → y-max 0.001–0.1
    self.td_plot.setYRange(0.0, self.td_scale, padding=0)


def on_td_span_changed(self, value):
    self.td_span_ms = float(value)
    self.td_span_val_label.setText(f"{value} ms")
    self.td_plot.setTitle(f'IQ Magnitude — {value} ms')


def on_select_source_airspy(self):
    from .runtime import _connect_airspy_client
    _connect_airspy_client(self)
    self.source_mode = "airspy"
    self.selected_wav_path = None
    self.source_airspy_action.setChecked(True)
    self.statusBar().showMessage("Source: Airspy HF+")


def on_select_source_radio(self):
    self._connect_radio_client()
    self.source_mode = "radio"
    self.selected_wav_path = None
    self.source_radio_action.setChecked(True)
    if hasattr(self, '_ri_radio_btn') and not self._ri_radio_btn.isChecked():
        for btn in (self._ri_radio_btn, self._ri_wav_btn):
            btn.blockSignals(True)
        self._ri_radio_btn.setChecked(True)
        for btn in (self._ri_radio_btn, self._ri_wav_btn):
            btn.blockSignals(False)
    self.statusBar().showMessage("Source: Flex Radio")


def on_select_source_wav(self):
    from pathlib import Path as _Path
    _default_dir = str(_Path('MSK144/simulations').resolve()) if _Path('MSK144/simulations').exists() else ""
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select Simulation WAV File",
        _default_dir,
        "WAV Files (*.wav);;All Files (*)",
    )

    if not file_path:
        # Revert dock radio button if the dialog was cancelled while on radio
        if self.source_mode != "wav" and hasattr(self, '_ri_radio_btn'):
            for btn in (self._ri_radio_btn, self._ri_wav_btn):
                btn.blockSignals(True)
            self._ri_radio_btn.setChecked(True)
            for btn in (self._ri_radio_btn, self._ri_wav_btn):
                btn.blockSignals(False)
        return

    self.source_mode = "wav"
    self.selected_wav_path = file_path
    self._wav_load_nonce = getattr(self, '_wav_load_nonce', 0) + 1
    self._wav_done = False
    self.source_wav_action.setChecked(True)
    if hasattr(self, '_ri_wav_btn') and not self._ri_wav_btn.isChecked():
        for btn in (self._ri_radio_btn, self._ri_wav_btn):
            btn.blockSignals(True)
        self._ri_wav_btn.setChecked(True)
        for btn in (self._ri_radio_btn, self._ri_wav_btn):
            btn.blockSignals(False)
    if hasattr(self, '_ri_wav_path_label'):
        self._ri_wav_path_label.setText(_Path(file_path).name)
    self.statusBar().showMessage(f"Source selected: WAV File ({file_path})")
