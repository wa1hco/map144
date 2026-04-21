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
"""Per-source radio interface windows and IQ/Noise Blanker window.

Five free-floating panel windows, all created during setup_ui:

    _iq_nb_win    IQ magnitude plot + noise blanker controls.
                  Always available; not tied to any source.

    _flex_win     Flex Radio status, DAXIQ stream info.
    _usrp_win     USRP B210 gain/antenna controls and IF stream info.
    _airspy_win   Airspy HF+ IF stream info.
    _rtlsdr_win   RTL-SDR IF stream info.

Radio windows auto-show when their source is selected and auto-hide
when another source is chosen.  All window geometries and
source-specific settings (e.g. USRP gain) are persisted in QSettings.

Common IF stream block (sample rate, signal dBFS, noise dBFS, drops)
appears in every radio window.  dBFS is computed from the running IQ
magnitude buffer at ~5 Hz.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


# ── Internal helpers ──────────────────────────────────────────────────────────

def _stat_label(parent, attr_name, init="—", color="#1a6b1a", font_size=None):
    size_rule = f" font-size: {font_size};" if font_size else ""
    lbl = QtWidgets.QLabel(init)
    lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace;{size_rule} }}")
    setattr(parent, attr_name, lbl)
    return lbl


def _form_group(title, compact=False):
    grp  = QtWidgets.QGroupBox(title)
    form = QtWidgets.QFormLayout(grp)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
    form.setVerticalSpacing(1 if compact else 3)
    form.setHorizontalSpacing(8)
    if compact:
        form.setContentsMargins(6, 2, 6, 2)
    return grp, form


def _setlbl(parent, attr, text):
    lbl = getattr(parent, attr, None)
    if lbl is not None:
        lbl.setText(text)


def _add_stream_rows(form, parent, prefix):
    """Add the common IF stream rows to a QFormLayout."""
    form.addRow("Sample Rate:",  _stat_label(parent, f"{prefix}_rate_val"))
    form.addRow("Signal:",       _stat_label(parent, f"{prefix}_sig_dbfs_val"))
    form.addRow("Noise Floor:",  _stat_label(parent, f"{prefix}_noise_dbfs_val"))
    form.addRow("Buffer Drops:", _stat_label(parent, f"{prefix}_drops_val", color="#c76000"))
    form.addRow("Queue:",        _stat_label(parent, f"{prefix}_queue_val"))


def _set_stream_vals(self, prefix, rate_str, sig_str, noise_str, drops_str):
    _setlbl(self, f"{prefix}_rate_val",       rate_str)
    _setlbl(self, f"{prefix}_sig_dbfs_val",   sig_str)
    _setlbl(self, f"{prefix}_noise_dbfs_val", noise_str)
    _setlbl(self, f"{prefix}_drops_val",      drops_str)


def _set_queue_label(parent, attr, q):
    """Update a queue-fullness label from a queue.Queue object."""
    lbl = getattr(parent, attr, None)
    if lbl is None or q is None:
        return
    used    = q.qsize()
    maxsize = q.maxsize or 1
    pct     = used / maxsize * 100
    text    = f"{used} / {maxsize}  ({pct:.0f}%)"
    if pct >= 90:
        color = "#b71c1c"   # red — nearly full
    elif pct >= 50:
        color = "#c76000"   # orange — getting full
    else:
        color = "#1a6b1a"   # green — healthy
    lbl.setText(text)
    lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")


def _compute_peak_db(self, buf_attr):
    """Return peak amplitude as a float dBFS value, or None if buffer is empty."""
    buf = getattr(self, buf_attr, None)
    if buf is None or len(buf) == 0:
        return None
    peak = float(np.max(buf))
    return 20.0 * np.log10(peak) if peak > 1e-10 else None


def _noise_floor_db(self):
    """Return noise floor as a float dBFS value from the wideband or per-bin estimator."""
    from .processing import NB_FFT_SIZE
    wb = getattr(self, '_nb_wideband_floor', None)
    if wb is not None and float(wb) > 1e-30:
        rms = float(wb / NB_FFT_SIZE) ** 0.5
        return 20.0 * np.log10(rms)
    nb = getattr(self, '_nb_floor', None)
    if nb is not None and float(nb) > 1e-10:
        return 20.0 * np.log10(float(nb))
    return None


class _BlankerBar(QtWidgets.QWidget):
    """Horizontal stacked bar showing noise floor / peak-clean / peak-raw levels.

    Left edge = -110 dBFS, right edge = 0 dBFS (ADC clip).

    Segments:
      dark grey  — below noise floor (the noise baseline)
      blue       — noise floor → peak clean (signal passing through blanker)
      red        — peak clean → peak raw (impulse content the blanker is catching)
      light grey — peak raw → 0 dBFS (available ADC headroom)
    """
    _DB_MIN = -110.0
    _DB_MAX =    0.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._noise   = None
        self._p_clean = None
        self._p_raw   = None
        self.setMinimumHeight(20)
        self.setMaximumHeight(28)
        self.setMinimumWidth(80)
        self.setToolTip(
            "Left = −110 dBFS  |  Right = 0 dBFS (clip)\n"
            "Dark grey: below noise floor\n"
            "Blue: clean peak above noise\n"
            "Red: impulse content caught by blanker\n"
            "Light grey: ADC headroom"
        )

    def update_values(self, noise_db, peak_clean_db, peak_raw_db):
        self._noise   = noise_db
        self._p_clean = peak_clean_db
        self._p_raw   = peak_raw_db
        self.update()

    def _x(self, db):
        t = max(0.0, min(1.0, (db - self._DB_MIN) / (self._DB_MAX - self._DB_MIN)))
        return int(t * self.width())

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen
        p = QPainter(self)
        w, h = self.width(), self.height()

        # Background: ADC headroom (light grey)
        p.fillRect(0, 0, w, h, QColor("#e0e0e0"))

        if all(v is not None for v in (self._noise, self._p_clean, self._p_raw)):
            x_noise = self._x(self._noise)
            x_clean = self._x(self._p_clean)
            x_raw   = self._x(self._p_raw)

            # Below noise floor
            p.fillRect(0, 0, x_noise, h, QColor("#9e9e9e"))
            # Clean peak above noise floor
            if x_clean > x_noise:
                p.fillRect(x_noise, 0, x_clean - x_noise, h, QColor("#1565c0"))
            # Impulse content caught by blanker
            if x_raw > x_clean:
                p.fillRect(x_clean, 0, x_raw - x_clean, h, QColor("#c62828"))

        # Tick marks every 10 dB
        p.setPen(QPen(QColor("#ffffff"), 1))
        for db in range(-100, 0, 10):
            x = self._x(float(db))
            p.drawLine(x, 0, x, h)

        # Clip marker at 0 dBFS
        p.fillRect(w - 3, 0, 3, h, QColor("#b71c1c"))
        p.end()


def _compute_peak_dbfs(self, buf_attr='_td_mag_buf'):
    """Return (peak_dbfs_str, color) from an IQ magnitude buffer.

    Color reflects headroom: green > 10 dB below clip, orange 3–10 dB, red < 3 dB.
    """
    buf = getattr(self, buf_attr, None)
    if buf is None or len(buf) == 0:
        return "—", "#555555"
    peak = float(np.max(buf))
    if peak < 1e-10:
        return "—", "#555555"
    peak_db = 20.0 * np.log10(peak)
    s = f"{peak_db:.1f} dBFS"
    if peak_db >= -3.0:
        color = "#b71c1c"   # red — near clip
    elif peak_db >= -10.0:
        color = "#8a3a00"   # orange — getting close
    else:
        color = "#1a6b1a"   # green — healthy headroom
    return s, color


def _compute_dbfs(self):
    """Return (signal_dbfs, noise_dbfs) strings from the IQ magnitude buffer.

    noise_dbfs uses the wideband median noise floor estimator when available,
    falling back to the per-bin-mean _nb_floor.  The median estimator is robust
    to narrowband spurs; the fallback is not, but is always available.
    """
    from .processing import NB_FFT_SIZE
    buf = getattr(self, '_td_mag_buf', None)
    if buf is None or len(buf) == 0:
        return "—", "—"
    rms = float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))
    sig_str = "—" if rms < 1e-10 else f"{20.0 * np.log10(rms):.1f} dBFS"

    wb_floor = getattr(self, '_nb_wideband_floor', None)
    if wb_floor is not None and float(wb_floor) > 1e-30:
        # Convert median bin power to RMS amplitude: rms = sqrt(median_P / N)
        rms_floor = float(wb_floor / NB_FFT_SIZE) ** 0.5
        noise_str = f"{20.0 * np.log10(rms_floor):.1f} dBFS"
    else:
        nb_floor = getattr(self, '_nb_floor', None)
        if nb_floor is not None and float(nb_floor) > 1e-10:
            noise_str = f"{20.0 * np.log10(float(nb_floor)):.1f} dBFS"
        else:
            noise_str = "—"
    return sig_str, noise_str


# ── Window 1: IQ / Noise Blanker ─────────────────────────────────────────────

def setup_iq_nb_window(self, view_action):
    """Build the IQ magnitude + noise blanker panel window."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS

    _FONT   = "monospace"
    _GRP_SS = (
        f"QGroupBox {{ font-family: {_FONT}; }}"
        f"QGroupBox::title {{ font-family: {_FONT}; }}"
        f"QLabel {{ font-family: {_FONT}; }}"
    )

    win = _PanelWindow("map144 — IQ / Noise Blanker", view_action, self, 'iq_nb_geometry')
    win.setMinimumSize(350, 370)
    win.setStyleSheet(f"QLabel {{ font-family: {_FONT}; }}")
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._iq_nb_win = win

    # IQ magnitude time-domain plots — both permanently hide their bottom axis;
    # a shared time ruler widget below them carries the scale.
    self.td_span_ms = float(int(_SETTINGS.value('td_span', 200)))
    _td_n = int(0.200 * self.sample_rate)

    def _make_td_plot(label):
        p = pg.PlotWidget()
        p.setLabel('left', label, color='k')
        p.setBackground('w')
        p.getAxis('left').setWidth(55)
        p.getAxis('left').setTextPen('k')
        p.getAxis('left').setPen('k')
        p.setMinimumHeight(100)
        p.setMaximumHeight(150)
        p.getViewBox().disableAutoRange()
        p.getAxis('bottom').hide()
        return p

    def _make_td_ruler(span_ms):
        """Collapsed-viewbox time ruler widget shared by H and V td plots."""
        r = pg.PlotWidget()
        r.setBackground('w')
        r.getAxis('left').setWidth(55)   # match data plot left-axis width
        r.getAxis('left').hide()
        r.getAxis('right').hide()
        r.getAxis('top').hide()
        r.setLabel('bottom', '', color='k')
        r.getAxis('bottom').setTextPen('k')
        r.getAxis('bottom').setPen('k')
        r.getAxis('bottom').show()
        r.setXRange(0.0, span_ms, padding=0)
        r.getViewBox().disableAutoRange()
        _pi = r.getPlotItem()
        for _row in (0, 1, 2):   # title, top-axis, viewbox
            _pi.layout.setRowMaximumHeight(_row, 0)
            _pi.layout.setRowMinimumHeight(_row, 0)
        r.setMinimumHeight(44)
        r.setMaximumHeight(50)
        return r

    self.td_plot = _make_td_plot('H')
    self.td_curve = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#1565c0', width=1),
    )
    self.td_plot.addItem(self.td_curve)
    self.td_plot.setXRange(0.0, 200.0, padding=0)
    self.td_thresh_line = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine),
    )
    self.td_plot.addItem(self.td_thresh_line)

    self.td_plot_v = _make_td_plot('V')
    self.td_curve_v = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#6a1b9a', width=1),
    )
    self.td_plot_v.addItem(self.td_curve_v)
    self.td_plot_v.setXRange(0.0, 200.0, padding=0)
    self.td_thresh_line_v = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine),
    )
    self.td_plot_v.addItem(self.td_thresh_line_v)
    self.td_plot_v.hide()

    # Vertical scale slider — shared by H and V
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
    self.td_scale_slider.setInvertedAppearance(True)
    self.td_scale_slider.valueChanged.connect(self.on_td_scale_changed)
    self.td_scale = self.td_scale_slider.value() * 0.01
    self.td_plot.setYRange(0.0, self.td_scale, padding=0)
    self.td_plot_v.setYRange(0.0, self.td_scale, padding=0)

    self.td_ruler = _make_td_ruler(self.td_span_ms)

    td_plots_col = QtWidgets.QVBoxLayout()
    td_plots_col.setSpacing(0)
    td_plots_col.addWidget(self.td_plot)
    td_plots_col.addWidget(self.td_plot_v)
    td_plots_col.addWidget(self.td_ruler)
    td_row = QtWidgets.QHBoxLayout()
    td_row.setSpacing(2)
    td_row.addLayout(td_plots_col, stretch=1)
    td_row.addWidget(self.td_scale_slider)
    layout.addLayout(td_row)

    # Span slider
    td_span_row = QtWidgets.QHBoxLayout()
    td_span_row.setSpacing(4)
    td_span_row.addWidget(QtWidgets.QLabel("Span (ms):"))
    self.td_span_val_label = QtWidgets.QLabel(f"{int(self.td_span_ms)} ms")
    self.td_span_val_label.setStyleSheet(
        "QLabel { color: #1a6b1a; font-family: monospace; min-width: 52px; }"
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
    layout.addLayout(td_span_row)

    # Blanker spectrum plots — H (title + freq axis) and V (no title, no axis)
    from .processing import NB_FFT_SIZE
    _nb_freqs = np.fft.fftshift(np.fft.fftfreq(NB_FFT_SIZE, 1.0 / 48000)) / 1000.0
    _nb_zeros = np.full(NB_FFT_SIZE, -120.0, dtype=np.float32)

    def _make_nb_plot(show_title):
        p = pg.PlotWidget()
        p.setBackground('w')
        p.setLabel('left', '', color='k')
        p.getAxis('left').setWidth(45)
        p.getAxis('left').setTextPen('k')
        p.getAxis('left').setPen('k')
        p.setMinimumHeight(100)
        p.setMaximumHeight(140)
        p.setXRange(_nb_freqs[0], _nb_freqs[-1], padding=0)
        p.setYRange(-120.0, 0.0, padding=0)
        p.getViewBox().disableAutoRange()
        p.getAxis('bottom').hide()
        if show_title:
            p.setTitle('Blanker Spectrum — floor / block', color='k')
        return p

    def _make_nb_ruler():
        """Collapsed-viewbox frequency ruler shared by H and V nb spectrum plots."""
        r = pg.PlotWidget()
        r.setBackground('w')
        r.getAxis('left').setWidth(45)   # match data plot left-axis width
        r.getAxis('left').hide()
        r.getAxis('right').hide()
        r.getAxis('top').hide()
        r.setLabel('bottom', 'Frequency', units='kHz', color='k')
        r.getAxis('bottom').setTextPen('k')
        r.getAxis('bottom').setPen('k')
        r.getAxis('bottom').show()
        r.setXRange(_nb_freqs[0], _nb_freqs[-1], padding=0)
        r.getViewBox().disableAutoRange()
        _pi = r.getPlotItem()
        for _row in (0, 1, 2):   # title, top-axis, viewbox
            _pi.layout.setRowMaximumHeight(_row, 0)
            _pi.layout.setRowMinimumHeight(_row, 0)
        r.setMinimumHeight(44)
        r.setMaximumHeight(50)
        return r

    self.nb_spec_plot = _make_nb_plot(show_title=True)
    self.nb_floor_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1b5e20', width=1))
    self.nb_block_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1565c0', width=1, style=QtCore.Qt.DotLine))
    self.nb_thresh_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine))
    self.nb_spec_plot.addItem(self.nb_floor_curve)
    self.nb_spec_plot.addItem(self.nb_block_curve)
    self.nb_spec_plot.addItem(self.nb_thresh_curve)
    layout.addWidget(self.nb_spec_plot)

    self.nb_spec_plot_v = _make_nb_plot(show_title=False)
    self.nb_floor_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1b5e20', width=1))
    self.nb_block_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#6a1b9a', width=1, style=QtCore.Qt.DotLine))
    self.nb_thresh_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine))
    self.nb_spec_plot_v.addItem(self.nb_floor_curve_v)
    self.nb_spec_plot_v.addItem(self.nb_block_curve_v)
    self.nb_spec_plot_v.addItem(self.nb_thresh_curve_v)
    self.nb_spec_plot_v.hide()
    layout.addWidget(self.nb_spec_plot_v)
    layout.addWidget(_make_nb_ruler())

    # Noise blanker controls
    nb_group = QtWidgets.QGroupBox("Spectral Noise Blanker")
    nb_group.setStyleSheet(_GRP_SS)
    nb_vbox  = QtWidgets.QVBoxLayout(nb_group)
    nb_vbox.setSpacing(3)

    nb_factor_row = QtWidgets.QHBoxLayout()
    nb_factor_row.addWidget(QtWidgets.QLabel("K (amplitude, K²=power ratio):"))
    self.nb_factor_label = QtWidgets.QLabel(f"{self.nb_factor:.1f}")
    self.nb_factor_label.setStyleSheet(
        "QLabel { color: #1a6b1a; font-family: monospace; min-width: 32px; }"
    )
    nb_factor_row.addWidget(self.nb_factor_label)
    nb_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    nb_sl.setMinimum(20); nb_sl.setMaximum(100)
    nb_sl.setValue(int(round(self.nb_factor * 10)))
    nb_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
    nb_sl.setTickInterval(10)
    nb_sl.valueChanged.connect(self.on_nb_factor_changed)
    nb_factor_row.addWidget(nb_sl, stretch=1)
    nb_vbox.addLayout(nb_factor_row)

    nb_status_row = QtWidgets.QHBoxLayout()
    nb_status_row.addWidget(QtWidgets.QLabel("Blanked:"))
    self._nb_count_val = QtWidgets.QLabel("0.00%")
    self._nb_count_val.setStyleSheet(
        "QLabel { color: #8a3a00; font-family: monospace; min-width: 52px; }"
    )
    nb_status_row.addWidget(self._nb_count_val)
    nb_status_row.addSpacing(12)
    nb_status_row.addWidget(QtWidgets.QLabel("Hot bins:"))
    self._nb_hot_val = QtWidgets.QLabel(f"0/{NB_FFT_SIZE}")
    self._nb_hot_val.setStyleSheet(
        "QLabel { color: #b85c00; font-family: monospace; min-width: 52px; }"
    )
    nb_status_row.addWidget(self._nb_hot_val)
    nb_status_row.addStretch()
    nb_vbox.addLayout(nb_status_row)

    layout.addWidget(nb_group)
    layout.addStretch()


# ── Window 2: Flex Radio ──────────────────────────────────────────────────────

def setup_flex_window(self, view_action):
    """Build the Flex Radio interface panel window."""
    from .ui import _PanelWindow

    # All text in the Flex Radio window uses this font and size.
    _FSZ  = "10pt"
    _FONT = "monospace"
    # Stylesheet applied to each group box so that Qt's internally-created
    # form-row labels (addRow("Label:", widget)) also get the right size.
    _GRP_SS = (
        f"QGroupBox {{ font-family: {_FONT}; font-size: {_FSZ}; }}"
        f"QGroupBox::title {{ font-family: {_FONT}; font-size: {_FSZ}; }}"
        f"QLabel {{ font-family: {_FONT}; font-size: {_FSZ}; }}"
    )

    def _sl(attr, init="—", color="#1a6b1a"):
        return _stat_label(self, attr, init, color, font_size=_FSZ)

    win = _PanelWindow("map144 — Flex Radio", view_action, self, 'flex_geometry')
    win.setMinimumSize(280, 380)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(3)
    self._flex_win = win

    # ── Radio ──
    radio_grp, radio_form = _form_group("Radio", compact=True)
    radio_grp.setStyleSheet(_GRP_SS)
    radio_form.addRow("Status:",   _sl("_flex_status_val", "Idle", "#c76000"))
    radio_form.addRow("Model:",    _sl("_flex_model_val"))
    radio_form.addRow("Serial:",   _sl("_flex_serial_val"))
    radio_form.addRow("Firmware:", _sl("_flex_firmware_val"))
    radio_form.addRow("IP:",       _sl("_flex_ip_val"))
    layout.addWidget(radio_grp)

    # ── Panadapters + Slices ──
    _TBL_FONT = _FONT
    _TBL_SZ   = _FSZ

    state_grp = QtWidgets.QGroupBox("Radio State")
    state_grp.setStyleSheet(
        f"QGroupBox {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; }}"
        f"QGroupBox::title {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; }}"
    )
    state_layout = QtWidgets.QVBoxLayout(state_grp)
    state_layout.setContentsMargins(4, 4, 4, 4)
    state_layout.setSpacing(2)

    def _make_table(col_headers, show_vheader=False):
        tbl = QtWidgets.QTableWidget(4, len(col_headers))
        tbl.setHorizontalHeaderLabels(col_headers)
        tbl.verticalHeader().setVisible(show_vheader)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        tbl.setFocusPolicy(QtCore.Qt.NoFocus)
        tbl.setStyleSheet(
            f"QTableWidget {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; "
            f"gridline-color: #cccccc; background: #ffffff; color: #000000; }}"
            f"QHeaderView::section {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; "
            f"background: #f0f0f0; color: #333333; "
            f"padding: 1px; border: 1px solid #cccccc; }}"
            f"QTableWidget::item {{ padding: 1px; }}"
        )
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.horizontalHeader().setDefaultSectionSize(70)
        tbl.verticalHeader().setDefaultSectionSize(20)
        tbl.setShowGrid(True)
        row_h = tbl.verticalHeader().defaultSectionSize()
        hdr_h = tbl.horizontalHeader().sizeHint().height()
        tbl.setFixedHeight(hdr_h + 4 * row_h + 2)
        return tbl

    pan_lbl = QtWidgets.QLabel("Panadapters")
    pan_lbl.setStyleSheet(f"QLabel {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; color: #555555; }}")
    state_layout.addWidget(pan_lbl)
    self._flex_pan_table = _make_table(["Fc (kHz)", "Span", "Ant", "DAXIQ"], show_vheader=True)
    for row in range(4):
        self._flex_pan_table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(str(row)))
        for col in range(4):
            item = QtWidgets.QTableWidgetItem("—")
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self._flex_pan_table.setItem(row, col, item)
    state_layout.addWidget(self._flex_pan_table)

    slice_lbl = QtWidgets.QLabel("Slices")
    slice_lbl.setStyleSheet(f"QLabel {{ font-family: {_TBL_FONT}; font-size: {_TBL_SZ}; color: #555555; }}")
    state_layout.addWidget(slice_lbl)
    self._flex_slice_table = _make_table(["Freq (kHz)", "Mode", "Pan", "DAX", "Owner"], show_vheader=True)
    for row in range(4):
        ltr_item = QtWidgets.QTableWidgetItem(str(row))
        ltr_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self._flex_slice_table.setVerticalHeaderItem(row, ltr_item)
        for col in range(5):
            item = QtWidgets.QTableWidgetItem("—")
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self._flex_slice_table.setItem(row, col, item)
    state_layout.addWidget(self._flex_slice_table)

    layout.addWidget(state_grp)

    # ── DAXIQ Stream ──
    dax_grp, dax_form = _form_group("DAXIQ Stream", compact=True)
    dax_grp.setStyleSheet(_GRP_SS)
    dax_form.addRow("DAX Channel:", _sl("_flex_dax_ch_val"))
    dax_form.addRow("Stream ID:",   _sl("_flex_stream_id_val"))
    dax_form.addRow("UDP Port:",    _sl("_flex_udp_port_val"))
    dax_form.addRow("Mode:",        _sl("_flex_mode_val"))
    dax_form.addRow("Pkt Rate:",    _sl("_flex_pkt_rate_val"))
    dax_form.addRow("Queue:",       _sl("_flex_vita_queue_val"))
    dax_form.addRow("Loss:",        _sl("_flex_loss_val"))
    layout.addWidget(dax_grp)

    # ── IF Stream ──
    stream_grp, stream_form = _form_group("IF Stream", compact=True)
    stream_grp.setStyleSheet(_GRP_SS)
    _if_lbl = QtWidgets.QLabel("IF centre:")
    _if_lbl.setToolTip(
        "Frequency at DC in the DAX IQ stream (digital IF). "
        "Matches the SmartSDR slice or panadapter centre tied to this stream."
    )
    stream_form.addRow(_if_lbl, _sl("_flex_freq_val"))
    _msk_lbl = QtWidgets.QLabel("Decodable dial range:")
    _msk_lbl.setToolTip(
        "USB dial frequency range a station can be tuned to and still be decoded. "
        "Derived from the channelizer audio passband (300–2700 Hz above each channel "
        "centre), converted to dial frequency by subtracting the 1500 Hz MSK144 "
        "audio reference.  Covers all auto-detect channels within the IF passband."
    )
    _msk_val = _sl("_flex_msk_span_val", "—")
    _msk_val.setToolTip(_msk_lbl.toolTip())
    stream_form.addRow(_msk_lbl, _msk_val)
    _add_stream_rows(stream_form, self, "_flex")
    layout.addWidget(stream_grp)

    layout.addStretch()


# ── Window 3: USRP B210 ───────────────────────────────────────────────────────

def setup_usrp_window(self, view_action):
    """Build the USRP B210 interface panel window."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS

    win = _PanelWindow("map144 — USRP B210", view_action, self, 'usrp_geometry')
    win.setMinimumSize(320, 360)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._usrp_win = win

    # ── Radio (hardware identity) ──
    radio_grp, radio_form = _form_group("Radio", compact=True)
    radio_form.addRow("Serial:",   _stat_label(self, "_usrp_serial_val"))
    radio_form.addRow("Firmware:", _stat_label(self, "_usrp_firmware_val"))
    radio_form.addRow("FPGA:",     _stat_label(self, "_usrp_fpga_val"))
    layout.addWidget(radio_grp)

    # ── IF Stream (shared / per-source) ──
    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Center Freq:",   _stat_label(self, "_usrp_freq_val"))
    stream_form.addRow("Sample Rate:",   _stat_label(self, "_usrp_rate_val"))
    stream_form.addRow("Data Rate:",     _stat_label(self, "_usrp_data_rate_val"))
    stream_form.addRow("Buffer Drops:",  _stat_label(self, "_usrp_drops_val", color="#c76000"))
    stream_form.addRow("Queue:",         _stat_label(self, "_usrp_queue_val"))
    layout.addWidget(stream_grp)

    # ── RF 0 ──
    rxa_grp = QtWidgets.QGroupBox("RF 0")
    rxa_form = QtWidgets.QFormLayout(rxa_grp)
    rxa_form.setVerticalSpacing(4)
    rxa_form.setHorizontalSpacing(8)

    rxa_form.addRow("Status:", _stat_label(self, "_usrp_rx0_status_val"))

    try:
        _gain_saved = float(_SETTINGS.value('usrp_gain_db', 50.0))
    except (ValueError, TypeError):
        _gain_saved = 50.0
    self._usrp_gain_db = _gain_saved
    self._usrp_gain_label = QtWidgets.QLabel(f"{_gain_saved:.0f} dB")
    self._usrp_gain_label.setStyleSheet(
        "QLabel { color: #1a6b1a; font-family: monospace; min-width: 48px; }"
    )
    self._usrp_gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._usrp_gain_slider.setMinimum(0)
    self._usrp_gain_slider.setMaximum(76)
    self._usrp_gain_slider.setValue(int(_gain_saved))
    self._usrp_gain_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self._usrp_gain_slider.setTickInterval(10)
    self._usrp_gain_slider.valueChanged.connect(self.on_usrp_gain_changed)
    gain_row = QtWidgets.QHBoxLayout()
    gain_row.addWidget(self._usrp_gain_slider, stretch=1)
    gain_row.addWidget(self._usrp_gain_label)
    rxa_form.addRow("Gain:", gain_row)

    try:
        _ant_saved = str(_SETTINGS.value('usrp_antenna', 'RX2'))
    except Exception:
        _ant_saved = 'RX2'
    self._usrp_antenna_combo = QtWidgets.QComboBox()
    self._usrp_antenna_combo.addItems(["RX2", "TX/RX"])
    self._usrp_antenna_combo.setCurrentText(_ant_saved)
    self._usrp_antenna_combo.currentTextChanged.connect(self.on_usrp_antenna_changed)
    rxa_form.addRow("Antenna:", self._usrp_antenna_combo)
    rxa_form.addRow("Noise Floor:", _stat_label(self, "_usrp_noise_dbfs_val"))
    rxa_form.addRow("Peak raw:",    _stat_label(self, "_usrp_peak_raw_dbfs_val"))
    rxa_form.addRow("Peak clean:",  _stat_label(self, "_usrp_peak_dbfs_val"))
    self._usrp_blanker_bar = _BlankerBar()
    rxa_form.addRow(self._usrp_blanker_bar)
    layout.addWidget(rxa_grp)

    # ── RF 1 (second polarization channel, always RX2 port) ──
    rxb_grp = QtWidgets.QGroupBox("RF 1")
    rxb_form = QtWidgets.QFormLayout(rxb_grp)
    rxb_form.setVerticalSpacing(4)
    rxb_form.setHorizontalSpacing(8)

    rxb_form.addRow("Status:", _stat_label(self, "_usrp_rx1_status_val"))

    try:
        _ant1_saved = str(_SETTINGS.value('usrp_antenna_ch1', 'RX2'))
    except Exception:
        _ant1_saved = 'RX2'
    self._usrp_antenna_combo_ch1 = QtWidgets.QComboBox()
    self._usrp_antenna_combo_ch1.addItems(["RX2", "TX/RX"])
    self._usrp_antenna_combo_ch1.setCurrentText(_ant1_saved)
    self._usrp_antenna_combo_ch1.currentTextChanged.connect(self.on_usrp_antenna_ch1_changed)
    rxb_form.addRow("Antenna:", self._usrp_antenna_combo_ch1)

    try:
        _gain1_saved = float(_SETTINGS.value('usrp_gain_db_ch1', _gain_saved))
    except (ValueError, TypeError):
        _gain1_saved = _gain_saved
    self._usrp_gain_db_ch1 = _gain1_saved
    self._usrp_gain_label_ch1 = QtWidgets.QLabel(f"{_gain1_saved:.0f} dB")
    self._usrp_gain_label_ch1.setStyleSheet(
        "QLabel { color: #1a6b1a; font-family: monospace; min-width: 48px; }"
    )
    self._usrp_gain_slider_ch1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._usrp_gain_slider_ch1.setMinimum(0)
    self._usrp_gain_slider_ch1.setMaximum(76)
    self._usrp_gain_slider_ch1.setValue(int(_gain1_saved))
    self._usrp_gain_slider_ch1.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self._usrp_gain_slider_ch1.setTickInterval(10)
    self._usrp_gain_slider_ch1.valueChanged.connect(self.on_usrp_gain_ch1_changed)
    gain1_row = QtWidgets.QHBoxLayout()
    gain1_row.addWidget(self._usrp_gain_slider_ch1, stretch=1)
    gain1_row.addWidget(self._usrp_gain_label_ch1)
    rxb_form.addRow("Gain:", gain1_row)
    rxb_form.addRow("Noise Floor:", _stat_label(self, "_usrp_noise_dbfs_ch1_val"))
    rxb_form.addRow("Peak raw:",    _stat_label(self, "_usrp_peak_raw_dbfs_ch1_val"))
    rxb_form.addRow("Peak clean:",  _stat_label(self, "_usrp_peak_dbfs_ch1_val"))
    self._usrp_blanker_bar_ch1 = _BlankerBar()
    rxb_form.addRow(self._usrp_blanker_bar_ch1)
    layout.addWidget(rxb_grp)

    layout.addStretch()


# ── Window 4: Airspy HF+ ─────────────────────────────────────────────────────

def setup_airspy_window(self, view_action):
    """Build the Airspy HF+ interface panel window."""
    from .ui import _PanelWindow

    win = _PanelWindow("map144 — Airspy HF+", view_action, self, 'airspy_geometry')
    win.setMinimumSize(320, 280)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._airspy_win = win

    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Center Freq:",   _stat_label(self, "_airspy_freq_val"))
    _add_stream_rows(stream_form, self, "_airspy")
    stream_form.addRow("HW Rate:",       _stat_label(self, "_airspy_hw_rate_val"))
    stream_form.addRow("Decimation:",    _stat_label(self, "_airspy_decim_val"))
    stream_form.addRow("IQ Correction:", _stat_label(self, "_airspy_iq_corr_val"))
    layout.addWidget(stream_grp)
    layout.addStretch()


# ── Window 5: RTL-SDR ─────────────────────────────────────────────────────────

def setup_rtlsdr_window(self, view_action):
    """Build the RTL-SDR interface panel window."""
    from .ui import _PanelWindow

    win = _PanelWindow("map144 — NESDR Smart (RTL-SDR)", view_action, self, 'rtlsdr_geometry')
    win.setMinimumSize(320, 280)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._rtlsdr_win = win

    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Center Freq:",  _stat_label(self, "_rtlsdr_freq_val"))
    _add_stream_rows(stream_form, self, "_rtlsdr")
    stream_form.addRow("Device Index:", _stat_label(self, "_rtlsdr_index_val"))
    stream_form.addRow("Gain:",         _stat_label(self, "_rtlsdr_gain_val"))
    stream_form.addRow("HW Rate:",      _stat_label(self, "_rtlsdr_hw_rate_val"))
    stream_form.addRow("Decimation:",   _stat_label(self, "_rtlsdr_decim_val"))
    layout.addWidget(stream_grp)
    layout.addStretch()


# ── USRP live control handlers ────────────────────────────────────────────────

def on_usrp_gain_changed(self, value):
    from .visualizer import _SETTINGS
    self._usrp_gain_db = float(value)
    if hasattr(self, '_usrp_gain_label'):
        self._usrp_gain_label.setText(f"{value:.0f} dB")
    _SETTINGS.setValue('usrp_gain_db', float(value))
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None:
        try:
            uc._usrp.set_rx_gain(float(value), 0)
        except Exception:
            pass


def on_usrp_antenna_changed(self, text):
    from .visualizer import _SETTINGS
    _SETTINGS.setValue('usrp_antenna', text)
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None:
        try:
            uc._usrp.set_rx_antenna(text, 0)
            uc.antenna = text
        except Exception:
            pass



def on_usrp_antenna_ch1_changed(self, text):
    from .visualizer import _SETTINGS
    _SETTINGS.setValue('usrp_antenna_ch1', text)
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None and getattr(uc, 'dual_channel', False):
        try:
            uc._usrp.set_rx_antenna(text, 1)
        except Exception:
            pass


def on_usrp_gain_ch1_changed(self, value):
    from .visualizer import _SETTINGS
    self._usrp_gain_db_ch1 = float(value)
    if hasattr(self, '_usrp_gain_label_ch1'):
        self._usrp_gain_label_ch1.setText(f"{value:.0f} dB")
    _SETTINGS.setValue('usrp_gain_db_ch1', float(value))
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None and getattr(uc, 'dual_channel', False):
        try:
            uc._usrp.set_rx_gain(float(value), 1)
        except Exception:
            pass


# ── Auto show/hide radio windows on source change ─────────────────────────────

_RADIO_WINDOWS = ('_flex_win', '_usrp_win', '_airspy_win', '_rtlsdr_win')
_SOURCE_WINDOW  = {
    'radio':  '_flex_win',
    'usrp':   '_usrp_win',
    'airspy': '_airspy_win',
    'rtlsdr': '_rtlsdr_win',
}


def show_source_window(self, source_mode):
    """Show the window for *source_mode*, hide all other radio windows."""
    target_attr = _SOURCE_WINDOW.get(source_mode)
    for attr in _RADIO_WINDOWS:
        win = getattr(self, attr, None)
        if win is None:
            continue
        if attr == target_attr:
            win.show()
            win.raise_()
        else:
            if win.isVisible():
                win._save_geometry()
                win.hide()
            # Keep the View menu action in sync
            va = getattr(win, '_view_action', None)
            if va is not None:
                va.setChecked(False)
    if target_attr is not None:
        win = getattr(self, target_attr, None)
        if win is not None:
            va = getattr(win, '_view_action', None)
            if va is not None:
                va.setChecked(True)


# ── Display update ────────────────────────────────────────────────────────────

def update_source_windows(self):
    """Update all visible source windows. Called from update_displays at 10 Hz."""
    sig_str, noise_str = _compute_dbfs(self)
    rate_str  = f"{self.sample_rate / 1000:.0f} kHz"
    drops_str = "—"   # per-source drop tracking not yet implemented for SDR sources

    # IQ / Noise Blanker (always)
    iq_nb_win = getattr(self, '_iq_nb_win', None)
    if iq_nb_win is not None and iq_nb_win.isVisible():
        _update_iq_nb_window(self)

    # Reporting (always — stats update even when window hidden)
    from .reporting_window import update_reporting_window
    update_reporting_window(self)

    # Radio windows — only update if visible
    if getattr(self, '_flex_win',   None) is not None and self._flex_win.isVisible():
        _update_flex_window(self, sig_str, noise_str, rate_str, drops_str)
    if getattr(self, '_usrp_win',   None) is not None and self._usrp_win.isVisible():
        _update_usrp_window(self, sig_str, noise_str, rate_str, drops_str)
    if getattr(self, '_airspy_win', None) is not None and self._airspy_win.isVisible():
        _update_airspy_window(self, sig_str, noise_str, rate_str, drops_str)
    if getattr(self, '_rtlsdr_win', None) is not None and self._rtlsdr_win.isVisible():
        _update_rtlsdr_window(self, sig_str, noise_str, rate_str, drops_str)


def _update_iq_nb_window(self):
    from .processing import NB_FFT_SIZE

    _dual = getattr(self, 'dual_pol', False)

    # Show/hide V plots based on dual_pol
    td_v   = getattr(self, 'td_plot_v',      None)
    nb_v   = getattr(self, 'nb_spec_plot_v', None)
    if td_v is not None:
        if _dual and not td_v.isVisible():
            td_v.show()
        elif not _dual and td_v.isVisible():
            td_v.hide()
    if nb_v is not None:
        if _dual and not nb_v.isVisible():
            nb_v.show()
        elif not _dual and nb_v.isVisible():
            nb_v.hide()

    # IQ magnitude plot — update at 5 Hz (every other 10 Hz tick)
    if hasattr(self, 'td_curve') and self._noise_floor_ctr % 2 == 0:
        span_ms = float(getattr(self, 'td_span_ms', 200.0))
        x_axis  = None

        def _td_display(buf, pos):
            n = len(buf)
            if pos == 0:
                full = buf
            else:
                full = np.empty(n, dtype=np.float32)
                full[:n - pos] = buf[pos:]
                full[n - pos:] = buf[:pos]
            span_n = min(int(span_ms * self.sample_rate / 1000.0), n)
            return full[-span_n:], span_n

        display, span_n = _td_display(self._td_mag_buf, self._td_mag_pos)
        x_axis = np.linspace(0.0, span_ms, span_n, endpoint=False)
        self.td_plot.setXRange(0.0, span_ms, padding=0)
        self.td_curve.setData(x_axis, display)

        nb_floor = getattr(self, '_nb_floor', None)
        if nb_floor is not None:
            nb_k = float(getattr(self, 'nb_factor', 6))
            thresh = nb_k * float(nb_floor)
            self.td_thresh_line.setValue(thresh)
            if _dual and hasattr(self, 'td_thresh_line_v'):
                self.td_thresh_line_v.setValue(thresh)

        if _dual and hasattr(self, 'td_curve_v'):
            display_v, span_n_v = _td_display(self._td_mag_buf_v, self._td_mag_pos_v)
            if x_axis is None or len(x_axis) != span_n_v:
                x_axis_v = np.linspace(0.0, span_ms, span_n_v, endpoint=False)
            else:
                x_axis_v = x_axis
            self.td_plot_v.setXRange(0.0, span_ms, padding=0)
            self.td_curve_v.setData(x_axis_v, display_v)

        _td_ruler = getattr(self, 'td_ruler', None)
        if _td_ruler is not None:
            _td_ruler.setXRange(0.0, span_ms, padding=0)

    # Blanker spectrum — floor, current block, threshold curves.
    # Gate to 2 Hz: the floor and threshold are slowly-adapting averages that
    # do not benefit from more frequent redraws, and these three paint() calls
    # showed up as 32% of wall time in the profiler during noise pulse events.
    if hasattr(self, 'nb_floor_curve') and self._noise_floor_ctr % 5 == 0:
        nb_k  = float(getattr(self, 'nb_factor', 6.0))
        ref   = 20.0 * np.log10(NB_FFT_SIZE)
        _freqs = np.fft.fftshift(np.fft.fftfreq(NB_FFT_SIZE, 1.0 / 48000)) / 1000.0

        def _update_nb_curves(spec_avg, last_P, floor_c, block_c, thresh_c):
            if spec_avg is not None:
                floor_db = np.fft.fftshift(
                    10.0 * np.log10(np.maximum(spec_avg, 1e-30)) - ref
                ).astype(np.float32)
                floor_c.setData(_freqs, floor_db)
                thresh_c.setData(_freqs, floor_db + 20.0 * np.log10(max(nb_k, 1.0)))
            if last_P is not None:
                block_db = np.fft.fftshift(
                    10.0 * np.log10(np.maximum(last_P, 1e-30)) - ref
                ).astype(np.float32)
                block_c.setData(_freqs, block_db)

        _update_nb_curves(
            getattr(self, '_nb_spec_avg', None), getattr(self, '_nb_last_P', None),
            self.nb_floor_curve, self.nb_block_curve, self.nb_thresh_curve,
        )
        if _dual and hasattr(self, 'nb_floor_curve_v'):
            _update_nb_curves(
                getattr(self, '_nb_spec_avg_v', None), getattr(self, '_nb_last_P_v', None),
                self.nb_floor_curve_v, self.nb_block_curve_v, self.nb_thresh_curve_v,
            )

    # Status row
    if hasattr(self, '_nb_count_val'):
        _nb_t = getattr(self, '_nb_total_count', 0)
        _nb_b = getattr(self, '_nb_blanked_count', 0)
        _nb_pct = (100.0 * _nb_b / _nb_t) if _nb_t > 0 else 0.0
        self._nb_count_val.setText(f"{_nb_pct:.2f}%")

    if hasattr(self, '_nb_hot_val'):
        hot = getattr(self, '_nb_last_hot', 0)
        self._nb_hot_val.setText(f"{hot}/{NB_FFT_SIZE}")



def _update_flex_window(self, sig_str, noise_str, rate_str, drops_str):
    rc   = getattr(self, 'radio_client', None)
    tcp  = rc._tcp       if rc else None
    dax  = rc._dax_setup if rc else None
    vita = rc._vita      if rc else None

    # Status
    if rc is None:
        status, color = "Idle", "#555555"
    elif tcp is None:
        status, color = "Discovering...", "#c76000"
    elif not getattr(tcp, '_running', False):
        status, color = "TCP disconnected", "#b71c1c"
    elif vita is None:
        status, color = "Configuring DAXIQ...", "#c76000"
    elif getattr(vita, '_running', False):
        if getattr(rc, 'transmitting', False):
            status, color = "Transmitting", "#cc4400"
        else:
            status, color = "Receiving", "#1a6b1a"
    else:
        status, color = "VITA stopped", "#b71c1c"
    lbl = getattr(self, '_flex_status_val', None)
    if lbl is not None:
        lbl.setText(status)
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")

    # IF stream (DAX IQ DC = digital IF reference frequency)
    if hasattr(self, '_flex_freq_val'):
        freq = None
        if dax is not None:
            freq = dax.pan_frequency_mhz or dax.slice_frequency_mhz
        if freq:
            self._flex_freq_val.setText(f"{freq * 1000:.3f} kHz")
            offset_hz = abs(freq - self.center_freq_mhz) * 1e6
            freq_color = "#b71c1c" if offset_hz > 2000 else "#1a6b1a"
        else:
            self._flex_freq_val.setText(f"{self.center_freq_mhz * 1000:.3f} kHz (req)")
            freq_color = "#1a6b1a"
        self._flex_freq_val.setStyleSheet(
            f"QLabel {{ color: {freq_color}; font-family: monospace; }}"
        )
        pan_mhz = float(freq) if freq else float(self.center_freq_mhz)
        if hasattr(self, '_flex_msk_span_val'):
            from .channel_plan import msk144_rf_usb_span_edges_mhz, JT9_AUDIO_CENTER_HZ

            ch_offset = getattr(getattr(self, '_ch_state', None), 'channel_offset_hz', 0.0)
            rf_lo, rf_hi, k_lo, k_hi = msk144_rf_usb_span_edges_mhz(
                pan_mhz, channel_offset_hz=ch_offset
            )
            if k_lo < 0 or k_hi < 0:
                self._flex_msk_span_val.setText("—")
            else:
                dial_lo = rf_lo - JT9_AUDIO_CENTER_HZ / 1e6
                dial_hi = rf_hi - JT9_AUDIO_CENTER_HZ / 1e6
                # Round to integer kHz: low side up, high side down
                dial_lo_khz = int(dial_lo * 1000 + 0.9999)   # ceil
                dial_hi_khz = int(dial_hi * 1000)             # floor
                self._flex_msk_span_val.setText(
                    f"{dial_lo_khz} – {dial_hi_khz} kHz"
                )
    _set_stream_vals(self, "_flex", rate_str, sig_str, noise_str, drops_str)

    # Radio identity
    if tcp is not None:
        radio = getattr(tcp, 'radio', None)
        if radio is not None:
            _setlbl(self, '_flex_model_val',    radio.model or "—")
            _setlbl(self, '_flex_serial_val',   radio.serial or "—")
            _setlbl(self, '_flex_firmware_val', radio.version or "—")
            _setlbl(self, '_flex_ip_val',       f"{radio.ip}:{radio.port}")

    # DAXIQ stream
    if dax is not None:
        _setlbl(self, '_flex_dax_ch_val', str(dax.dax_channel))
        sid = dax.stream_id
        _setlbl(self, '_flex_stream_id_val', f"0x{sid:08x}" if sid else "—")
        _setlbl(self, '_flex_udp_port_val',  str(dax.listen_port))
        slice_id = dax.slice_id
        if slice_id is None:
            mode = "—"
        elif slice_id == 0:
            mode = "Panadapter"
        else:
            mode = f"Slice {slice_id}"
        _setlbl(self, '_flex_mode_val', mode)

    if vita is not None:
        total    = vita.packet_count
        missed   = vita.missed_count
        loss_pct = (missed / total * 100) if total > 0 else 0.0
        prev     = getattr(self, '_flex_prev_pkt_count', total)
        rate     = (total - prev) * 10   # update_displays runs at 10 Hz
        self._flex_prev_pkt_count = total
        _setlbl(self, '_flex_pkt_rate_val', f"{rate:,} /s")
        _setlbl(self, '_flex_loss_val',     f"{loss_pct:.3f}%")
        _set_queue_label(self, '_flex_vita_queue_val', vita.out_q)
    else:
        _setlbl(self, '_flex_pkt_rate_val',   "—")
        _setlbl(self, '_flex_vita_queue_val', "—")
        _setlbl(self, '_flex_loss_val',       "—")

    # IF Stream queue (same underlying queue)
    q = rc.sample_queue if rc is not None else None
    _set_queue_label(self, '_flex_queue_val', q)

    # Radio State: panadapter table (4 fixed rows, indexed by pan number 0-3)
    if hasattr(self, '_flex_pan_table') and dax is not None:
        from PyQt5.QtGui import QColor
        tbl_active   = QColor("#1a6b1a")
        tbl_inactive = QColor("#aaaaaa")
        for row in range(4):
            # Identify pan by its low byte (0x40000000 → 0, 0x40000001 → 1, etc.)
            pid = next((p for p in dax.known_pans if (p & 0xFF) == row), None)
            if pid is not None:
                info   = dax.known_pans[pid]
                center = info.get("center")
                bw_hz  = info.get("bandwidth")
                daxiq  = info.get("daxiq_channel") or ("1" if pid == dax.pan_id else "")
                ant    = info.get("rxant") or info.get("ant") or "?"
                # Fc: integer kHz rounded to nearest
                fc_str = f"{int(round(center * 1000))}" if center is not None else "?"
                bw_str = f"{int(round(bw_hz / 1e3))} kHz" if bw_hz is not None else "?"
                color  = tbl_active
            else:
                fc_str = bw_str = ant = daxiq = "—"
                color  = tbl_inactive
            for col, text in enumerate((fc_str, bw_str, ant, str(daxiq))):
                item = self._flex_pan_table.item(row, col)
                if item:
                    item.setText(text)
                    item.setForeground(color)

    # Radio State: slice table (4 fixed rows A-D, indexed by slice number 0-3)
    if hasattr(self, '_flex_slice_table') and dax is not None:
        from PyQt5.QtGui import QColor
        tbl_active   = QColor("#1a6b1a")
        tbl_daxiq    = QColor("#0044cc")
        tbl_inactive = QColor("#aaaaaa")

        # Build handle -> short owner name from known GUI clients
        gui_clients = tcp.get_gui_clients() if tcp else []
        handle_to_owner = {}
        for c in gui_clients:
            handle = (c.get("handle") or "").lower()
            # Prefer station name, fall back to program, then IP
            name = c.get("station") or c.get("program") or c.get("ip") or handle
            if name.startswith("SmartSDR"):
                name = name[len("SmartSDR"):].lstrip("-").strip() or "SmartSDR"
            handle_to_owner[handle] = name

        for row in range(4):
            info = dax.known_slices.get(row)
            if info is not None:
                freq   = info.get("RF_frequency_mhz")
                mode   = info.get("mode", "?")
                pan_id = info.get("pan_id")
                dax_ch = info.get("dax", 0)
                handle = (info.get("client_handle") or "").lower()
                freq_str  = f"{int(round(freq * 1000))}" if freq is not None else "?"
                pan_str   = str(pan_id & 0xFF) if pan_id else "?"
                dax_str   = str(dax_ch) if dax_ch else "—"
                owner_str = handle_to_owner.get(handle, handle[:10] if handle else "?")
                color = tbl_daxiq if row == dax.slice_id else tbl_active
            else:
                freq_str = mode = pan_str = dax_str = owner_str = "—"
                color = tbl_inactive
            for col, text in enumerate((freq_str, mode, pan_str, dax_str, owner_str)):
                item = self._flex_slice_table.item(row, col)
                if item:
                    item.setText(text)
                    item.setForeground(color)


def _update_usrp_window(self, sig_str, noise_str, rate_str, drops_str):
    import time as _time
    uc = getattr(self, 'usrp_client', None)
    if hasattr(self, '_usrp_freq_val'):
        freq = uc.center_freq_mhz_actual if uc is not None else self.center_freq_mhz
        self._usrp_freq_val.setText(f"{freq:.6f} MHz")
    _setlbl(self, '_usrp_drops_val', drops_str)
    _set_queue_label(self, '_usrp_queue_val', uc.sample_queue if uc is not None else None)

    # Data rate: recv() calls per second computed from monotonic counter delta.
    now = _time.monotonic()
    cur_count = getattr(uc, 'recv_count', 0) if uc is not None else 0
    prev_count = getattr(self, '_usrp_prev_recv_count', cur_count)
    prev_time  = getattr(self, '_usrp_prev_recv_time',  now)
    dt = now - prev_time
    if dt >= 0.5:   # update at most every 0.5 s to smooth the display
        pps = (cur_count - prev_count) / dt if dt > 0 else 0.0
        self._usrp_prev_recv_count = cur_count
        self._usrp_prev_recv_time  = now
        _setlbl(self, '_usrp_data_rate_val',
                f"{pps:.0f} buf/s" if pps > 0 else ("— buf/s" if uc is not None else "—"))
    elif not hasattr(self, '_usrp_prev_recv_count'):
        self._usrp_prev_recv_count = cur_count
        self._usrp_prev_recv_time  = now
        _setlbl(self, '_usrp_data_rate_val', "—")

    # RF 0 / RF 1 status LEDs
    running = (uc is not None and getattr(uc, '_running', False))
    dual    = (uc is not None and getattr(uc, 'dual_channel', False))
    if hasattr(self, '_usrp_rx0_status_val'):
        lbl = self._usrp_rx0_status_val
        if running:
            lbl.setText("● Active")
            lbl.setStyleSheet("QLabel { color: #66bb6a; font-family: monospace; }")
        else:
            lbl.setText("● Idle")
            lbl.setStyleSheet("QLabel { color: #555555; font-family: monospace; }")
    if hasattr(self, '_usrp_rx1_status_val'):
        lbl = self._usrp_rx1_status_val
        if running and dual:
            lbl.setText("● Active")
            lbl.setStyleSheet("QLabel { color: #66bb6a; font-family: monospace; }")
        else:
            lbl.setText("● Idle")
            lbl.setStyleSheet("QLabel { color: #555555; font-family: monospace; }")

    _setlbl(self, '_usrp_rate_val', rate_str)

    def _rms_to_dbfs_str(rms):
        if rms is None or rms < 1e-10:
            return "—"
        return f"{20.0 * np.log10(rms):.1f} dBFS"

    # Signal: use recv-loop RMS directly — reliable regardless of whether
    # process_iq_data is consuming the queue.
    # Noise floor: from noise blanker envelope on ch0 (ch1 stub pending).
    _setlbl(self, '_usrp_noise_dbfs_val',      noise_str)
    _setlbl(self, '_usrp_noise_dbfs_ch1_val', noise_str)
    for _attr, _buf in (
        ('_usrp_peak_raw_dbfs_val',     '_td_mag_buf_raw'),    # pre-blanker H
        ('_usrp_peak_dbfs_val',         '_td_mag_buf'),         # post-blanker H
        ('_usrp_peak_raw_dbfs_ch1_val', '_td_mag_buf_raw_v'),  # pre-blanker V
        ('_usrp_peak_dbfs_ch1_val',     '_td_mag_buf_v'),       # post-blanker V
    ):
        _ps, _pc = _compute_peak_dbfs(self, _buf)
        _setlbl(self, _attr, _ps)
        _lbl = getattr(self, _attr, None)
        if _lbl is not None:
            _lbl.setStyleSheet(f"QLabel {{ color: {_pc}; font-family: monospace; }}")

    # Stacked bar: noise floor / peak-clean / peak-raw
    _ndb = _noise_floor_db(self)
    for _bar_attr, _buf_clean, _buf_raw in (
        ('_usrp_blanker_bar',     '_td_mag_buf',   '_td_mag_buf_raw'),
        ('_usrp_blanker_bar_ch1', '_td_mag_buf_v', '_td_mag_buf_raw_v'),
    ):
        _bar = getattr(self, _bar_attr, None)
        if _bar is not None:
            _bar.update_values(_ndb, _compute_peak_db(self, _buf_clean),
                               _compute_peak_db(self, _buf_raw))

    # Actual gain readback from hardware (once per second — every 10 update cycles)
    if (uc is not None and getattr(uc, '_usrp', None) is not None
            and self._noise_floor_ctr % 10 == 0):
        try:
            actual0 = uc._usrp.get_rx_gain(0)
            if hasattr(self, '_usrp_gain_label'):
                self._usrp_gain_label.setText(f"{actual0:.1f} dB")
        except Exception:
            pass
        try:
            if getattr(uc, 'dual_channel', False):
                actual1 = uc._usrp.get_rx_gain(1)
                if hasattr(self, '_usrp_gain_label_ch1'):
                    self._usrp_gain_label_ch1.setText(f"{actual1:.1f} dB")
        except Exception:
            pass

    # Query hardware info once after hardware opens
    if (uc is not None and getattr(uc, '_usrp', None) is not None
            and not getattr(self, '_usrp_serial_queried', False)):
        # Serial number from get_usrp_rx_info dict
        try:
            info = uc._usrp.get_usrp_rx_info(0)
            _setlbl(self, '_usrp_serial_val', info.get('mboard_serial', '—') or '—')
        except Exception:
            _setlbl(self, '_usrp_serial_val', '—')
        # Firmware and FPGA versions from the UHD property tree
        try:
            tree = uc._usrp.get_tree()
            fw = tree.access_str('/mboards/0/fw_version').get()
            _setlbl(self, '_usrp_firmware_val', fw or '—')
        except Exception:
            _setlbl(self, '_usrp_firmware_val', '—')
        try:
            tree = uc._usrp.get_tree()
            fpga = tree.access_str('/mboards/0/fpga_version').get()
            _setlbl(self, '_usrp_fpga_val', fpga or '—')
        except Exception:
            _setlbl(self, '_usrp_fpga_val', '—')
        self._usrp_serial_queried = True


def _update_airspy_window(self, sig_str, noise_str, rate_str, drops_str):
    ac = getattr(self, 'airspy_client', None)
    if hasattr(self, '_airspy_freq_val'):
        freq = ac.center_freq_mhz_actual if ac is not None else self.center_freq_mhz
        self._airspy_freq_val.setText(f"{freq:.6f} MHz")
    _set_stream_vals(self, "_airspy", rate_str, sig_str, noise_str, drops_str)
    _set_queue_label(self, '_airspy_queue_val', ac.sample_queue if ac is not None else None)
    if ac is not None:
        hw_rate = getattr(ac, '_hw_rate', None)
        _setlbl(self, '_airspy_hw_rate_val', f"{hw_rate/1000:.0f} kHz" if hw_rate else "—")
        factor = (hw_rate // ac.target_rate) if hw_rate else None
        _setlbl(self, '_airspy_decim_val', f"{factor}×" if factor else "—")
        _setlbl(self, '_airspy_iq_corr_val', "On")
    else:
        for attr in ('_airspy_hw_rate_val', '_airspy_decim_val', '_airspy_iq_corr_val'):
            _setlbl(self, attr, "—")


def _update_rtlsdr_window(self, sig_str, noise_str, rate_str, drops_str):
    rc = getattr(self, 'rtlsdr_client', None)
    if hasattr(self, '_rtlsdr_freq_val'):
        freq = rc.center_freq_mhz_actual if rc is not None else self.center_freq_mhz
        self._rtlsdr_freq_val.setText(f"{freq:.6f} MHz")
    _set_stream_vals(self, "_rtlsdr", rate_str, sig_str, noise_str, drops_str)
    _set_queue_label(self, '_rtlsdr_queue_val', rc.sample_queue if rc is not None else None)
    if rc is not None:
        from .rtlsdr_source import _HW_RATE, _DECIMATE
        _setlbl(self, '_rtlsdr_index_val',   str(getattr(rc, 'device_index', 0)))
        _setlbl(self, '_rtlsdr_gain_val',    f"{getattr(rc, 'gain_db', '?')} dB")
        _setlbl(self, '_rtlsdr_hw_rate_val', f"{_HW_RATE/1000:.0f} kHz")
        _setlbl(self, '_rtlsdr_decim_val',   f"{_DECIMATE}×")
    else:
        for attr in ('_rtlsdr_index_val', '_rtlsdr_gain_val',
                     '_rtlsdr_hw_rate_val', '_rtlsdr_decim_val'):
            _setlbl(self, attr, "—")
