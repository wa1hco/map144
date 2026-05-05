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


def _noise_floor_db(self, channel='h'):
    """Return noise floor (dBFS) for the specified channel, 'h' or 'v'.

    Falls back to the wideband estimator when available, else the
    per-bin-mean floor.  Channel-specific attributes have a ``_v`` suffix.
    """
    from .processing import NB_FFT_SIZE
    suffix = '_v' if channel == 'v' else ''
    wb = getattr(self, f'_nb_wideband_floor{suffix}', None)
    if wb is not None and float(wb) > 1e-30:
        rms = float(wb / NB_FFT_SIZE) ** 0.5
        return 20.0 * np.log10(rms)
    nb = getattr(self, f'_nb_floor{suffix}', None)
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


def _compute_dbfs(self, channel='h'):
    """Return (signal_dbfs, noise_dbfs) strings for the specified channel.

    noise_dbfs uses the wideband median noise floor estimator when available,
    falling back to the per-bin-mean _nb_floor.  ``channel`` selects H or V
    state (``_nb_floor`` / ``_nb_floor_v``).
    """
    from .processing import NB_FFT_SIZE
    suffix  = '_v' if channel == 'v' else ''
    buf_name = '_td_mag_buf_v' if channel == 'v' else '_td_mag_buf'
    buf = getattr(self, buf_name, None)
    if buf is None or len(buf) == 0:
        return "—", "—"
    rms = float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))
    sig_str = "—" if rms < 1e-10 else f"{20.0 * np.log10(rms):.1f} dBFS"

    wb_floor = getattr(self, f'_nb_wideband_floor{suffix}', None)
    if wb_floor is not None and float(wb_floor) > 1e-30:
        rms_floor = float(wb_floor / NB_FFT_SIZE) ** 0.5
        noise_str = f"{20.0 * np.log10(rms_floor):.1f} dBFS"
    else:
        nb_floor = getattr(self, f'_nb_floor{suffix}', None)
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

    win = _PanelWindow("map144 — Noise Blanker", view_action, self, 'iq_nb_geometry')
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
    # Raw (pre-blanker) trace added first so it renders behind the cleaned trace.
    self.td_curve_raw = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#e53935', width=1),   # red — pre-blanker; impulse spikes stand out
    )
    self.td_plot.addItem(self.td_curve_raw)
    self.td_curve = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#1565c0', width=1),   # solid blue — post-blanker (holes show blanking)
    )
    self.td_plot.addItem(self.td_curve)
    self.td_plot.setXRange(0.0, 200.0, padding=0)
    self.td_thresh_line = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine),
    )
    self.td_plot.addItem(self.td_thresh_line)

    self.td_plot_v = _make_td_plot('V')
    self.td_curve_raw_v = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#e53935', width=1),   # red — pre-blanker V (match H)
    )
    self.td_plot_v.addItem(self.td_curve_raw_v)
    self.td_curve_v = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#1565c0', width=1),   # blue — post-blanker V (match H)
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
        p.setYRange(-90.0, -20.0, padding=0)   # typical activity range; scroll wheel zooms Y
        p.getViewBox().disableAutoRange()
        p.setMouseEnabled(x=False, y=True)     # Y scroll/drag to pan; X stays locked to freq axis
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
    self.nb_spec_plot.setTitle('Blanker Spectrum — floor / last blanked / threshold', color='k')
    self.nb_floor_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1b5e20', width=1))    # green: floor avg
    self.nb_blanked_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#e65100', width=2))    # orange: last blanked block (frozen)
    self.nb_block_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1565c0', width=1, style=QtCore.Qt.DotLine))  # blue: current block
    self.nb_thresh_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine)) # red: threshold
    self.nb_spec_plot.addItem(self.nb_floor_curve)
    self.nb_spec_plot.addItem(self.nb_blanked_curve)
    self.nb_spec_plot.addItem(self.nb_block_curve)
    self.nb_spec_plot.addItem(self.nb_thresh_curve)
    layout.addWidget(self.nb_spec_plot)

    self.nb_spec_plot_v = _make_nb_plot(show_title=False)
    self.nb_floor_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#1b5e20', width=1))
    self.nb_blanked_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#e65100', width=2))
    self.nb_block_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#6a1b9a', width=1, style=QtCore.Qt.DotLine))
    self.nb_thresh_curve_v = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(), pen=pg.mkPen('#c62828', width=1, style=QtCore.Qt.DashLine))
    self.nb_spec_plot_v.addItem(self.nb_floor_curve_v)
    self.nb_spec_plot_v.addItem(self.nb_blanked_curve_v)
    self.nb_spec_plot_v.addItem(self.nb_block_curve_v)
    self.nb_spec_plot_v.addItem(self.nb_thresh_curve_v)
    self.nb_spec_plot_v.hide()
    layout.addWidget(self.nb_spec_plot_v)
    layout.addWidget(_make_nb_ruler())

    # Noise blanker controls
    nb_group = QtWidgets.QGroupBox("Noise Blanker")
    nb_group.setStyleSheet(_GRP_SS)
    nb_vbox  = QtWidgets.QVBoxLayout(nb_group)
    nb_vbox.setSpacing(3)

    # Backend selector — switches the active blanker at runtime.  Saved to
    # QSettings under ``nb_backend`` so the choice persists across restarts.
    from .noise_blanker import available as _nb_available
    nb_backend_row = QtWidgets.QHBoxLayout()
    nb_backend_row.addWidget(QtWidgets.QLabel("Backend:"))
    self.nb_backend_combo = QtWidgets.QComboBox()
    for _name in _nb_available():
        self.nb_backend_combo.addItem(_name)
    _saved_backend = str(_SETTINGS.value('nb_backend', 'Linrad'))
    _idx = self.nb_backend_combo.findText(_saved_backend)
    if _idx >= 0:
        self.nb_backend_combo.setCurrentIndex(_idx)
    # Apply saved choice to the engine now so the first chunk uses it.
    self.set_blanker(self.nb_backend_combo.currentText())
    self.nb_backend_combo.currentTextChanged.connect(self.on_nb_backend_changed)
    nb_backend_row.addWidget(self.nb_backend_combo, stretch=1)
    nb_vbox.addLayout(nb_backend_row)

    nb_factor_row = QtWidgets.QHBoxLayout()
    nb_factor_row.addWidget(QtWidgets.QLabel("K H:"))
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

    # V-channel K slider — independent from H for polarisation-specific
    # tuning.  Shown always, useful only when dual-pol is active; the
    # backend silently ignores it in single-pol mode.
    nb_factor_v_row = QtWidgets.QHBoxLayout()
    nb_factor_v_row.addWidget(QtWidgets.QLabel("K V:"))
    self.nb_factor_v_label = QtWidgets.QLabel(
        f"{getattr(self, 'nb_factor_v', self.nb_factor):.1f}")
    self.nb_factor_v_label.setStyleSheet(
        "QLabel { color: #1a6b1a; font-family: monospace; min-width: 32px; }"
    )
    nb_factor_v_row.addWidget(self.nb_factor_v_label)
    nb_sl_v = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    nb_sl_v.setMinimum(20); nb_sl_v.setMaximum(100)
    nb_sl_v.setValue(int(round(getattr(self, 'nb_factor_v', self.nb_factor) * 10)))
    nb_sl_v.setTickPosition(QtWidgets.QSlider.TicksBelow)
    nb_sl_v.setTickInterval(10)
    nb_sl_v.valueChanged.connect(self.on_nb_factor_v_changed)
    nb_factor_v_row.addWidget(nb_sl_v, stretch=1)
    nb_vbox.addLayout(nb_factor_v_row)

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


# ── Window 6: SDRangel ───────────────────────────────────────────────────────

def setup_sdrangel_window(self, view_action):
    """Build the SDRangel interface panel window."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS

    win = _PanelWindow("map144 — SDRangel", view_action, self, 'sdrangel_geometry')
    win.setMinimumSize(340, 480)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._sdrangel_win = win

    # ── Connection settings ──
    conn_grp, conn_form = _form_group("Connection")

    try:
        _host_saved = str(_SETTINGS.value('sdrangel_host', 'localhost'))
    except Exception:
        _host_saved = 'localhost'
    self._sdrangel_host_edit = QtWidgets.QLineEdit(_host_saved)
    conn_form.addRow("Host:", self._sdrangel_host_edit)

    try:
        _rest_port_saved = int(_SETTINGS.value('sdrangel_rest_port', 8091))
    except (ValueError, TypeError):
        _rest_port_saved = 8091
    self._sdrangel_rest_port_spin = QtWidgets.QSpinBox()
    self._sdrangel_rest_port_spin.setRange(1024, 65535)
    self._sdrangel_rest_port_spin.setValue(_rest_port_saved)
    conn_form.addRow("REST Port:", self._sdrangel_rest_port_spin)

    try:
        _udp_port_saved = int(_SETTINGS.value('sdrangel_udp_port', 9999))
    except (ValueError, TypeError):
        _udp_port_saved = 9999
    self._sdrangel_udp_port_spin = QtWidgets.QSpinBox()
    self._sdrangel_udp_port_spin.setRange(1024, 65535)
    self._sdrangel_udp_port_spin.setValue(_udp_port_saved)
    conn_form.addRow("UDP Port (H):", self._sdrangel_udp_port_spin)

    try:
        _dev_set_saved = int(_SETTINGS.value('sdrangel_device_set', 0))
    except (ValueError, TypeError):
        _dev_set_saved = 0
    self._sdrangel_dev_set_spin = QtWidgets.QSpinBox()
    self._sdrangel_dev_set_spin.setRange(0, 7)
    self._sdrangel_dev_set_spin.setValue(_dev_set_saved)
    conn_form.addRow("Device Set (H):", self._sdrangel_dev_set_spin)

    try:
        _lo_offset_saved = float(_SETTINGS.value('sdrangel_lo_offset_khz', 50.0))
    except (ValueError, TypeError):
        _lo_offset_saved = 50.0
    self._sdrangel_lo_offset_spin = QtWidgets.QDoubleSpinBox()
    self._sdrangel_lo_offset_spin.setRange(0.0, 500.0)
    self._sdrangel_lo_offset_spin.setDecimals(1)
    self._sdrangel_lo_offset_spin.setSingleStep(10.0)
    self._sdrangel_lo_offset_spin.setValue(_lo_offset_saved)
    conn_form.addRow("LO Offset (kHz):", self._sdrangel_lo_offset_spin)

    try:
        _exe_saved = str(_SETTINGS.value('sdrangel_exe', ''))
    except Exception:
        _exe_saved = ''
    exe_row = QtWidgets.QHBoxLayout()
    self._sdrangel_exe_edit = QtWidgets.QLineEdit(_exe_saved)
    self._sdrangel_exe_edit.setPlaceholderText("(auto-detect)")
    self._sdrangel_exe_edit.setToolTip(
        "Full path to the SDRangel executable for auto-start.\n"
        "Leave blank to search PATH and common install locations."
    )
    exe_browse_btn = QtWidgets.QPushButton("…")
    exe_browse_btn.setMaximumWidth(28)
    exe_browse_btn.clicked.connect(lambda: _sdrangel_browse_exe(self))
    exe_row.addWidget(self._sdrangel_exe_edit, stretch=1)
    exe_row.addWidget(exe_browse_btn)
    conn_form.addRow("SDRangel Path:", exe_row)

    # B210 hardware settings (used by auto_configure_b210).
    try:
        _dev_rate_saved = int(_SETTINGS.value('sdrangel_dev_sample_rate', 192000))
    except (ValueError, TypeError):
        _dev_rate_saved = 192000
    self._sdrangel_dev_rate_spin = QtWidgets.QDoubleSpinBox()
    self._sdrangel_dev_rate_spin.setRange(0.048, 61.44)
    self._sdrangel_dev_rate_spin.setSingleStep(1.0)
    self._sdrangel_dev_rate_spin.setDecimals(3)
    self._sdrangel_dev_rate_spin.setValue(_dev_rate_saved / 1e6)
    self._sdrangel_dev_rate_spin.setSuffix(" MHz")
    conn_form.addRow("Device Rate:", self._sdrangel_dev_rate_spin)

    try:
        _gain_saved = float(_SETTINGS.value('sdrangel_gain_db', 40.0))
    except (ValueError, TypeError):
        _gain_saved = 40.0
    self._sdrangel_gain_spin = QtWidgets.QDoubleSpinBox()
    self._sdrangel_gain_spin.setRange(0.0, 76.0)
    self._sdrangel_gain_spin.setDecimals(0)
    self._sdrangel_gain_spin.setSingleStep(1.0)
    self._sdrangel_gain_spin.setValue(_gain_saved)
    self._sdrangel_gain_spin.setSuffix(" dB")
    conn_form.addRow("Gain:", self._sdrangel_gain_spin)

    # rx_scale: ingress normalization to match ±1.0 convention.
    # SDRangel's decimator adds sqrt(dev_rate/out_rate) amplitude; this corrects it.
    # Auto-computed default at client creation time; tune until levels match B210.
    try:
        _rx_scale_saved = float(_SETTINGS.value('sdrangel_rx_scale', 0.0))
    except (ValueError, TypeError):
        _rx_scale_saved = 0.0
    self._sdrangel_rx_scale_spin = QtWidgets.QDoubleSpinBox()
    self._sdrangel_rx_scale_spin.setRange(0.01, 2.0)
    self._sdrangel_rx_scale_spin.setDecimals(3)
    self._sdrangel_rx_scale_spin.setSingleStep(0.005)
    self._sdrangel_rx_scale_spin.setValue(_rx_scale_saved if _rx_scale_saved > 0 else 0.125)
    self._sdrangel_rx_scale_spin.setToolTip(
        "Ingress amplitude correction. SDRangel's decimator adds gain proportional\n"
        "to sqrt(device_rate / output_rate). Tune until peak levels match the\n"
        "direct B210 path. 0 = auto-compute from sample rates (first run)."
    )
    conn_form.addRow("RX Scale:", self._sdrangel_rx_scale_spin)

    apply_btn = QtWidgets.QPushButton("Apply")
    apply_btn.clicked.connect(lambda: on_sdrangel_apply_clicked(self))
    conn_form.addRow(apply_btn)
    layout.addWidget(conn_grp)

    # ── IF Stream ──
    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Device Type:",   _stat_label(self, "_sdrangel_hw_type_val"))
    stream_form.addRow("Device Center:", _stat_label(self, "_sdrangel_freq_val"))
    stream_form.addRow("Dual-pol:",      _stat_label(self, "_sdrangel_dualpol_val"))
    _add_stream_rows(stream_form, self, "_sdrangel")
    stream_form.addRow("Pkt Rate:",      _stat_label(self, "_sdrangel_pkt_rate_val"))
    layout.addWidget(stream_grp)

    # ── RF 0 (H channel) signal levels ──
    rfa_grp, rfa_form = _form_group("RF 0 (H)", compact=True)
    rfa_form.addRow("Noise Floor:", _stat_label(self, "_sdrangel_noise_dbfs_h_val"))
    rfa_form.addRow("Peak raw:",    _stat_label(self, "_sdrangel_peak_raw_h_val"))
    rfa_form.addRow("Peak clean:",  _stat_label(self, "_sdrangel_peak_clean_h_val"))
    self._sdrangel_blanker_bar = _BlankerBar()
    rfa_form.addRow(self._sdrangel_blanker_bar)
    layout.addWidget(rfa_grp)

    # ── RF 1 (V channel) signal levels — hidden until dual-pol active ──
    rfb_grp, rfb_form = _form_group("RF 1 (V)", compact=True)
    rfb_form.addRow("Noise Floor:", _stat_label(self, "_sdrangel_noise_dbfs_v_val"))
    rfb_form.addRow("Peak raw:",    _stat_label(self, "_sdrangel_peak_raw_v_val"))
    rfb_form.addRow("Peak clean:",  _stat_label(self, "_sdrangel_peak_clean_v_val"))
    self._sdrangel_blanker_bar_v = _BlankerBar()
    rfb_form.addRow(self._sdrangel_blanker_bar_v)
    self._sdrangel_rf1_grp = rfb_grp
    rfb_grp.hide()
    layout.addWidget(rfb_grp)

    # ── Service buttons ──
    udp_btn = QtWidgets.QPushButton("Configure UDP Sink")
    udp_btn.setToolTip(
        "Find (or auto-create) and configure the UDPSink channel(s).\n"
        "Sets destination, sample rate, and LO frequency offset."
    )
    udp_btn.clicked.connect(lambda: _sdrangel_configure_udp_sink(self))
    layout.addWidget(udp_btn)

    save_layout_btn = QtWidgets.QPushButton("Save SDRangel Layout")
    save_layout_btn.setToolTip(
        "Save SDRangel's current device set layout as the MAP144 preset.\n"
        "Next time MAP144 starts, SDRangel will be restored to this exact\n"
        "layout including sub-window positions and channel configuration.\n"
        "Also exports a backup file to ~/.config/map144/."
    )
    save_layout_btn.clicked.connect(lambda: _sdrangel_save_layout(self))
    layout.addWidget(save_layout_btn)
    _stat_label(self, "_sdrangel_layout_status_val", init="—", color="#555555")
    layout.addWidget(self._sdrangel_layout_status_val)

    ssb_btn = QtWidgets.QPushButton("Setup SSB / WSJT-X Service")
    ssb_btn.setToolTip(
        "Find or create SSBDemod channel(s) and a virtual audio device\n"
        "so WSJT-X can tune to the calling frequency.\n"
        "Linux: creates a PipeWire/PulseAudio null sink 'MAP144-WSJT-X'.\n"
        "WSJT-X → select 'Monitor of MAP144-WSJT-X' as sound card."
    )
    ssb_btn.clicked.connect(lambda: _sdrangel_setup_ssb_service(self))
    layout.addWidget(ssb_btn)
    _stat_label(self, "_sdrangel_ssb_status_val", init="—", color="#555555")
    layout.addWidget(self._sdrangel_ssb_status_val)

    # ── SDRangel configuration snapshot ──
    snap_grp  = QtWidgets.QGroupBox("SDRangel Configuration")
    snap_vlay = QtWidgets.QVBoxLayout(snap_grp)
    snap_vlay.setContentsMargins(6, 4, 6, 4)
    snap_vlay.setSpacing(4)

    self._sdrangel_state_text = QtWidgets.QPlainTextEdit()
    self._sdrangel_state_text.setReadOnly(True)
    self._sdrangel_state_text.setPlaceholderText("Click Refresh to query SDRangel state…")
    mono = self._sdrangel_state_text.font()
    mono.setFamily("monospace")
    mono.setPointSize(9)
    self._sdrangel_state_text.setFont(mono)
    self._sdrangel_state_text.setMinimumHeight(130)
    self._sdrangel_state_text.setMaximumHeight(240)
    snap_vlay.addWidget(self._sdrangel_state_text)

    refresh_btn = QtWidgets.QPushButton("Refresh")
    refresh_btn.setToolTip("Query SDRangel REST API and update the configuration snapshot.")
    refresh_btn.clicked.connect(lambda: _sdrangel_probe_state(self))
    snap_vlay.addWidget(refresh_btn)
    layout.addWidget(snap_grp)

    layout.addStretch()


def _sdrangel_browse_exe(self):
    """Open a file browser to select the SDRangel executable."""
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select SDRangel Executable", "/opt",
        "Executables (sdrangel sdrangel.exe *);;All files (*)"
    )
    if path:
        self._sdrangel_exe_edit.setText(path)


def on_sdrangel_apply_clicked(self):
    """Apply SDRangel connection settings: save to QSettings and recreate the clients."""
    from .visualizer import _SETTINGS

    host       = self._sdrangel_host_edit.text().strip() or 'localhost'
    rest_port  = self._sdrangel_rest_port_spin.value()
    udp_port   = self._sdrangel_udp_port_spin.value()
    device_set = self._sdrangel_dev_set_spin.value()
    lo_offset  = self._sdrangel_lo_offset_spin.value()
    exe_path   = self._sdrangel_exe_edit.text().strip()
    dev_rate   = int(self._sdrangel_dev_rate_spin.value() * 1e6)
    gain_db    = self._sdrangel_gain_spin.value()
    rx_scale   = self._sdrangel_rx_scale_spin.value()

    _SETTINGS.setValue('sdrangel_host',              host)
    _SETTINGS.setValue('sdrangel_rest_port',         rest_port)
    _SETTINGS.setValue('sdrangel_udp_port',          udp_port)
    _SETTINGS.setValue('sdrangel_device_set',        device_set)
    _SETTINGS.setValue('sdrangel_lo_offset_khz',     lo_offset)
    _SETTINGS.setValue('sdrangel_exe',               exe_path)
    _SETTINGS.setValue('sdrangel_dev_sample_rate',   dev_rate)
    _SETTINGS.setValue('sdrangel_gain_db',           gain_db)
    _SETTINGS.setValue('sdrangel_rx_scale',          rx_scale)

    # Stop only the UDP recv threads — do NOT terminate SDRangel itself.
    # _stop_sdrangel_source (which kills the process) is only for switching
    # to a different source type.  Here we just release the UDP sockets so
    # the new clients can bind to the same ports, then let the drain loop
    # call _start_sdrangel_source to reconnect with the updated settings.
    for _sc in (getattr(self, 'sdrangel_client', None),
                getattr(self, 'sdrangel_client_v', None)):
        if _sc is not None:
            try:
                _sc.stop()
            except Exception:
                pass
    self.sdrangel_client   = None
    self.sdrangel_client_v = None
    self._sdrangel_started = False   # force drain loop to call _start_sdrangel_source

    try:
        from .sdrangel_source import SDRAngelSource
        self.sdrangel_client = SDRAngelSource(
            host             = host,
            rest_port        = rest_port,
            udp_port         = udp_port,
            device_set       = device_set,
            calling_freq_mhz = self.calling_freq_mhz,
            lo_offset_khz    = lo_offset,
            target_rate      = self.sample_rate,
            exe_path         = exe_path or None,
            dev_sample_rate  = dev_rate,
            gain_db          = gain_db,
            stream_index     = 0,
            rx_scale         = rx_scale if rx_scale > 0 else None,
        )
        _udp_port_v = udp_port - 1
        _dev_set_v  = device_set + 1
        self.sdrangel_client_v = SDRAngelSource(
            host             = host,
            rest_port        = rest_port,
            udp_port         = _udp_port_v,
            device_set       = _dev_set_v,
            calling_freq_mhz = self.calling_freq_mhz,
            lo_offset_khz    = lo_offset,
            target_rate      = self.sample_rate,
            exe_path         = exe_path or None,
            dev_sample_rate  = dev_rate,
            gain_db          = gain_db,
            stream_index     = 1,
            rx_scale         = rx_scale if rx_scale > 0 else None,
        )
        print("[sdrangel] H+V clients recreated with new settings", flush=True)
    except Exception as exc:
        print(f"[sdrangel] client recreation failed: {exc}", flush=True)
        self.sdrangel_client   = None
        self.sdrangel_client_v = None


def _sdrangel_probe_state(self):
    """Query SDRangel REST API and update the configuration snapshot text panel."""
    sc = getattr(self, 'sdrangel_client', None)
    if sc is None:
        txt = getattr(self, '_sdrangel_state_text', None)
        if txt:
            txt.setPlainText("No client — select SDRangel source first.")
        return
    # Runs on the Qt thread; probe_full_state makes a few REST calls (~100 ms).
    text = sc.probe_full_state()
    txt  = getattr(self, '_sdrangel_state_text', None)
    if txt:
        txt.setPlainText(text)


def _sdrangel_save_layout(self):
    """Save SDRangel's current window layout for use on next restart.

    SDRangel only writes window positions to its config file when it exits.
    This function quits SDRangel gracefully (so it flushes geometry), copies
    the config file, then sets the flag so MAP144 relaunches SDRangel with the
    saved layout.  Data flow resumes automatically after relaunch.
    """
    import os
    from .sdrangel_source import _terminate_sdrangel, _save_sdrangel_layout

    _setlbl(self, "_sdrangel_layout_status_val", "Saving — quitting SDRangel…")

    import map144_app.sdrangel_source as _ss
    _proc = _ss._sdrangel_proc
    print(f"[sdrangel] save_layout: _sdrangel_proc={_proc}  pid={getattr(_proc,'pid',None)}",
          flush=True)

    # Stop UDP recv threads so sockets are free for the relaunch.
    for _sc in (getattr(self, 'sdrangel_client', None),
                getattr(self, 'sdrangel_client_v', None)):
        if _sc is not None:
            try:
                _sc.stop()
            except Exception:
                pass

    # Gracefully quit SDRangel — it writes geometry to its conf on exit.
    sc = getattr(self, 'sdrangel_client', None)
    print(f"[sdrangel] save_layout: sdrangel_client={sc}", flush=True)
    if sc is not None:
        _terminate_sdrangel(sc.host, sc.rest_port)
    else:
        if _proc is not None:
            _terminate_sdrangel('localhost', 8091)
        else:
            print("[sdrangel] save_layout: no proc and no client — cannot quit SDRangel",
                  flush=True)

    import time as _time
    _time.sleep(0.5)  # brief pause to let SDRangel finish writing conf

    # Check conf file timestamp before copy.
    _conf = _ss._sdrangel_conf_live_path()
    print(f"[sdrangel] save_layout: conf={_conf}  exists={os.path.isfile(_conf or '')}",
          flush=True)
    if _conf:
        import time as _t
        print(f"[sdrangel] save_layout: conf mtime={_t.ctime(os.path.getmtime(_conf))}",
              flush=True)

    # Copy the now-flushed conf file.
    _save_sdrangel_layout()
    dst = os.path.expanduser('~/.config/map144/sdrangel_layout.conf')
    if os.path.isfile(dst):
        print(f"[sdrangel] Layout saved → {dst}", flush=True)
        _setlbl(self, "_sdrangel_layout_status_val", "Saved — SDRangel relaunching…")
    else:
        _setlbl(self, "_sdrangel_layout_status_val", "Save failed — check log")
        return

    # Recreate clients and let the drain loop relaunch SDRangel.
    self.sdrangel_client   = None
    self.sdrangel_client_v = None
    self._sdrangel_started = False
    from .runtime import _connect_sdrangel_client
    _connect_sdrangel_client(self)


def _sdrangel_configure_udp_sink(self):
    """Call configure_udp_sink on the active SDRangel client(s)."""
    sc   = getattr(self, 'sdrangel_client',   None)
    sc_v = getattr(self, 'sdrangel_client_v', None)
    if sc is None:
        print("[sdrangel] No client — cannot configure UDP Sink", flush=True)
        return
    try:
        sc.configure_udp_sink()
        print("[sdrangel] UDP Sink (H) configured", flush=True)
    except Exception as exc:
        print(f"[sdrangel] configure_udp_sink (H) error: {exc}", flush=True)
    if sc_v is not None:
        try:
            sc_v.configure_udp_sink()
            print("[sdrangel] UDP Sink (V) configured", flush=True)
        except Exception as exc:
            print(f"[sdrangel] configure_udp_sink (V) error: {exc}", flush=True)


def _sdrangel_setup_ssb_service(self):
    """Find/create SSBDemod + virtual audio devices for WSJT-X."""
    from .runtime import _sdrangel_ensure_ssb_service
    result = _sdrangel_ensure_ssb_service(self)
    parts = []
    if result.get('audio_h'):
        parts.append(f"H→'{result['audio_h']}'")
    if result.get('audio_v'):
        parts.append(f"V→'{result['audio_v']}'")
    msg = "SSB service: " + (", ".join(parts) if parts else "configured (no virtual audio)")
    _setlbl(self, '_sdrangel_ssb_status_val', msg)
    # Colour the status green if at least one sink was created.
    lbl = getattr(self, '_sdrangel_ssb_status_val', None)
    if lbl is not None:
        color = "#1a6b1a" if parts else "#8a3a00"
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")


def _update_sdrangel_window(self, sig_str, noise_str, rate_str, drops_str):
    import time as _time
    sc   = getattr(self, 'sdrangel_client',   None)
    sc_v = getattr(self, 'sdrangel_client_v', None)
    is_dual = sc_v is not None and getattr(self, 'dual_pol', False)

    hw = sc.device_hw_type if sc is not None else "—"
    _setlbl(self, '_sdrangel_hw_type_val', hw)

    if hasattr(self, '_sdrangel_freq_val'):
        if sc is not None:
            self._sdrangel_freq_val.setText(f"{sc.center_freq_mhz_actual:.6f} MHz")
        else:
            self._sdrangel_freq_val.setText("—")

    _setlbl(self, '_sdrangel_dualpol_val',
            "Yes (H+V)" if is_dual else ("Yes (V offline)" if sc_v is not None else "No"))

    # Use source-level drop count rather than the generic pipeline drop counter.
    _drop_str = f"{sc.drop_count}" if sc is not None else "—"
    _set_stream_vals(self, "_sdrangel", rate_str, sig_str, noise_str, _drop_str)
    _set_queue_label(self, '_sdrangel_queue_val', sc.sample_queue if sc is not None else None)

    now = _time.monotonic()
    cur_count  = getattr(sc, 'recv_count', 0) if sc is not None else 0
    prev_count = getattr(self, '_sdrangel_prev_recv_count', cur_count)
    prev_time  = getattr(self, '_sdrangel_prev_recv_time',  now)
    dt = now - prev_time
    if dt >= 0.5:
        pps = (cur_count - prev_count) / dt if dt > 0 else 0.0
        self._sdrangel_prev_recv_count = cur_count
        self._sdrangel_prev_recv_time  = now
        _setlbl(self, '_sdrangel_pkt_rate_val',
                f"{pps:.0f} pkt/s" if pps > 0 else ("— pkt/s" if sc is not None else "—"))
    elif not hasattr(self, '_sdrangel_prev_recv_count'):
        self._sdrangel_prev_recv_count = cur_count
        self._sdrangel_prev_recv_time  = now
        _setlbl(self, '_sdrangel_pkt_rate_val', "—")

    # RF 0 (H) signal levels — use the shared IQ magnitude buffer.
    _ndb = _noise_floor_db(self)
    _peak_raw_db   = _compute_peak_db(self, '_td_mag_buf_raw')
    _peak_clean_db = _compute_peak_db(self, '_td_mag_buf')
    _peak_raw_str, _pr_color = _compute_peak_dbfs(self, '_td_mag_buf_raw')
    _peak_clean_str, _pc_color = _compute_peak_dbfs(self, '_td_mag_buf')
    _, noise_str_h = _compute_dbfs(self)
    _setlbl(self, '_sdrangel_noise_dbfs_h_val', noise_str_h)
    lbl = getattr(self, '_sdrangel_peak_raw_h_val', None)
    if lbl is not None:
        lbl.setText(_peak_raw_str); lbl.setStyleSheet(
            f"QLabel {{ color: {_pr_color}; font-family: monospace; }}")
    lbl = getattr(self, '_sdrangel_peak_clean_h_val', None)
    if lbl is not None:
        lbl.setText(_peak_clean_str); lbl.setStyleSheet(
            f"QLabel {{ color: {_pc_color}; font-family: monospace; }}")
    bar = getattr(self, '_sdrangel_blanker_bar', None)
    if bar is not None:
        bar.update_values(_ndb, _peak_clean_db, _peak_raw_db)

    # RF 1 (V) signal levels — show group when dual-pol active.
    rf1_grp = getattr(self, '_sdrangel_rf1_grp', None)
    if rf1_grp is not None:
        if is_dual and rf1_grp.isHidden():
            rf1_grp.show()
        elif not is_dual and not rf1_grp.isHidden():
            rf1_grp.hide()
    if is_dual:
        _peak_raw_v_db   = _compute_peak_db(self, '_td_mag_buf_raw_v')
        _peak_clean_v_db = _compute_peak_db(self, '_td_mag_buf_v')
        _peak_raw_v_str, _prv_color = _compute_peak_dbfs(self, '_td_mag_buf_raw_v')
        _peak_clean_v_str, _pcv_color = _compute_peak_dbfs(self, '_td_mag_buf_v')
        _ndb_v = _noise_floor_db(self, channel='v')
        _, noise_str_v = _compute_dbfs(self, channel='v')
        _setlbl(self, '_sdrangel_noise_dbfs_v_val', noise_str_v)
        lbl = getattr(self, '_sdrangel_peak_raw_v_val', None)
        if lbl is not None:
            lbl.setText(_peak_raw_v_str); lbl.setStyleSheet(
                f"QLabel {{ color: {_prv_color}; font-family: monospace; }}")
        lbl = getattr(self, '_sdrangel_peak_clean_v_val', None)
        if lbl is not None:
            lbl.setText(_peak_clean_v_str); lbl.setStyleSheet(
                f"QLabel {{ color: {_pcv_color}; font-family: monospace; }}")
        bar_v = getattr(self, '_sdrangel_blanker_bar_v', None)
        if bar_v is not None:
            bar_v.update_values(_ndb_v, _peak_clean_v_db, _peak_raw_v_db)


# ── USRP live control handlers ────────────────────────────────────────────────

def on_usrp_gain_changed(self, value):
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
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
    # A gain change shifts every channel's power; the percentile baseline
    # lags ~7 s and would false-trigger the sustained-signal lockout.
    reset_detection_baseline(self)


def on_usrp_antenna_changed(self, text):
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
    _SETTINGS.setValue('usrp_antenna', text)
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None:
        try:
            uc._usrp.set_rx_antenna(text, 0)
            uc.antenna = text
        except Exception:
            pass
    reset_detection_baseline(self)



def on_usrp_antenna_ch1_changed(self, text):
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
    _SETTINGS.setValue('usrp_antenna_ch1', text)
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and getattr(uc, '_usrp', None) is not None and getattr(uc, 'dual_channel', False):
        try:
            uc._usrp.set_rx_antenna(text, 1)
        except Exception:
            pass
    reset_detection_baseline(self)


def on_usrp_gain_ch1_changed(self, value):
    from .visualizer import _SETTINGS
    from .processing  import reset_detection_baseline
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
    reset_detection_baseline(self)


# ── Auto show/hide radio windows on source change ─────────────────────────────

_RADIO_WINDOWS = ('_flex_win', '_usrp_win', '_airspy_win', '_rtlsdr_win', '_sdrangel_win')
_SOURCE_WINDOW  = {
    'radio':    '_flex_win',
    'usrp':     '_usrp_win',
    'airspy':   '_airspy_win',
    'rtlsdr':   '_rtlsdr_win',
    'sdrangel': '_sdrangel_win',
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
    sig_str, noise_str = _compute_dbfs(self, channel='h')
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
    if getattr(self, '_rtlsdr_win',    None) is not None and self._rtlsdr_win.isVisible():
        _update_rtlsdr_window(self, sig_str, noise_str, rate_str, drops_str)
    if getattr(self, '_sdrangel_win', None) is not None and self._sdrangel_win.isVisible():
        _update_sdrangel_window(self, sig_str, noise_str, rate_str, drops_str)


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
        if hasattr(self, 'td_curve_raw'):
            raw_display, _ = _td_display(self._td_mag_buf_raw, self._td_mag_pos_raw)
            self.td_curve_raw.setData(x_axis, raw_display)
        self.td_curve.setData(x_axis, display)

        nb_floor_h = getattr(self, '_nb_floor', None)
        if nb_floor_h is not None:
            nb_k_h = float(getattr(self, 'nb_factor', 6))
            self.td_thresh_line.setValue(nb_k_h * float(nb_floor_h))
        if _dual and hasattr(self, 'td_thresh_line_v'):
            nb_floor_v = getattr(self, '_nb_floor_v', None) or nb_floor_h
            if nb_floor_v is not None:
                nb_k_v = float(getattr(self, 'nb_factor_v',
                                       getattr(self, 'nb_factor', 6)))
                self.td_thresh_line_v.setValue(nb_k_v * float(nb_floor_v))

        if _dual and hasattr(self, 'td_curve_v'):
            display_v, span_n_v = _td_display(self._td_mag_buf_v, self._td_mag_pos_v)
            if x_axis is None or len(x_axis) != span_n_v:
                x_axis_v = np.linspace(0.0, span_ms, span_n_v, endpoint=False)
            else:
                x_axis_v = x_axis
            self.td_plot_v.setXRange(0.0, span_ms, padding=0)
            if hasattr(self, 'td_curve_raw_v'):
                raw_display_v, _ = _td_display(self._td_mag_buf_raw_v, self._td_mag_pos_raw_v)
                self.td_curve_raw_v.setData(x_axis_v, raw_display_v)
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
        ref   = 10.0 * np.log10(NB_FFT_SIZE)   # 10*log10(N): normalises per-bin power to dBFS
        _freqs = np.fft.fftshift(np.fft.fftfreq(NB_FFT_SIZE, 1.0 / 48000)) / 1000.0

        def _update_nb_curves(spec_avg, last_P, last_blanked_P,
                              floor_c, blanked_c, block_c, thresh_c):
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
            if last_blanked_P is not None:
                blanked_db = np.fft.fftshift(
                    10.0 * np.log10(np.maximum(last_blanked_P, 1e-30)) - ref
                ).astype(np.float32)
                blanked_c.setData(_freqs, blanked_db)

        _update_nb_curves(
            getattr(self, '_nb_spec_avg', None), getattr(self, '_nb_last_P', None),
            getattr(self, '_nb_last_blanked_P', None),
            self.nb_floor_curve, self.nb_blanked_curve,
            self.nb_block_curve, self.nb_thresh_curve,
        )
        if _dual and hasattr(self, 'nb_floor_curve_v'):
            _update_nb_curves(
                getattr(self, '_nb_spec_avg_v', None), getattr(self, '_nb_last_P_v', None),
                getattr(self, '_nb_last_blanked_P_v', None),
                self.nb_floor_curve_v, self.nb_blanked_curve_v,
                self.nb_block_curve_v, self.nb_thresh_curve_v,
            )

    # Status row.  Decay the blanker counters when they grow past ~5 s of
    # samples so the displayed "% blanked" reflects recent behaviour
    # (responds quickly to K/gain changes) instead of the session-wide
    # average.  Halving both total and blanked preserves the ratio but
    # caps the effective window — new samples then weigh ~equally with
    # past ~5 s worth, so tweaks become visible within a second or two.
    if hasattr(self, '_nb_count_val'):
        _rate = float(getattr(self, 'sample_rate', 48000)) or 48000.0
        _window_n = int(5.0 * _rate) * (2 if getattr(self, 'dual_pol', False) else 1)
        if getattr(self, '_nb_total_count', 0) > _window_n:
            self._nb_total_count   //= 2
            self._nb_blanked_count //= 2
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
    # Per-channel noise floor text — V channel has its own _nb_floor_v
    # since Tier 2.  Computed here so the caller doesn't need to know
    # about V-channel plumbing.
    _, noise_str_v = _compute_dbfs(self, channel='v')
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
    # Noise floor: per-channel, now that Linrad/NR0V maintain independent
    # _nb_floor and _nb_floor_v estimates.
    _setlbl(self, '_usrp_noise_dbfs_val',     noise_str)
    _setlbl(self, '_usrp_noise_dbfs_ch1_val', noise_str_v)
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

    # Stacked bar: noise floor / peak-clean / peak-raw — per-channel floor.
    for _bar_attr, _buf_clean, _buf_raw, _ch in (
        ('_usrp_blanker_bar',     '_td_mag_buf',   '_td_mag_buf_raw',   'h'),
        ('_usrp_blanker_bar_ch1', '_td_mag_buf_v', '_td_mag_buf_raw_v', 'v'),
    ):
        _bar = getattr(self, _bar_attr, None)
        if _bar is not None:
            _bar.update_values(_noise_floor_db(self, channel=_ch),
                               _compute_peak_db(self, _buf_clean),
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
