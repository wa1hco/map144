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

def _stat_label(parent, attr_name, init="—", color="#1a6b1a"):
    lbl = QtWidgets.QLabel(init)
    lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")
    setattr(parent, attr_name, lbl)
    return lbl


def _form_group(title):
    grp  = QtWidgets.QGroupBox(title)
    form = QtWidgets.QFormLayout(grp)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
    form.setVerticalSpacing(3)
    form.setHorizontalSpacing(8)
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


def _compute_dbfs(self):
    """Return (signal_dbfs, noise_dbfs) strings from the IQ magnitude buffer."""
    buf = getattr(self, '_td_mag_buf', None)
    if buf is None or len(buf) == 0:
        return "—", "—"
    rms = float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))
    sig_dbfs = 20.0 * np.log10(rms + 1e-12)
    sig_str  = f"{sig_dbfs:.1f} dBFS"
    nb_env = getattr(self, '_nb_env', None)
    if nb_env is not None and float(nb_env) > 0:
        noise_dbfs = 20.0 * np.log10(float(nb_env) + 1e-12)
        noise_str  = f"{noise_dbfs:.1f} dBFS"
    else:
        noise_str = "—"
    return sig_str, noise_str


# ── Window 1: IQ / Noise Blanker ─────────────────────────────────────────────

def setup_iq_nb_window(self, view_action):
    """Build the IQ magnitude + noise blanker panel window."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS

    win = _PanelWindow("map144 — IQ / Noise Blanker", view_action, self, 'iq_nb_geometry')
    win.setMinimumSize(350, 370)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._iq_nb_win = win

    # IQ magnitude time-domain plot
    self.td_plot = pg.PlotWidget()
    self.td_plot.setLabel('left', '')
    self.td_plot.setLabel('bottom', 'Time', units='ms')
    self.td_span_ms = float(int(_SETTINGS.value('td_span', 200)))
    self.td_plot.setTitle(f'IQ Magnitude — {int(self.td_span_ms)} ms')
    self.td_plot.setBackground('#111111')
    self.td_plot.getAxis('left').setWidth(55)
    self.td_plot.setMinimumHeight(130)
    self.td_plot.setMaximumHeight(180)
    self.td_plot.getViewBox().disableAutoRange()

    _td_n = int(0.200 * self.sample_rate)
    self.td_curve = pg.PlotCurveItem(
        np.linspace(0.0, 200.0, _td_n, endpoint=False),
        np.zeros(_td_n, dtype=np.float32),
        pen=pg.mkPen('#4fc3f7', width=1),
    )
    self.td_plot.addItem(self.td_curve)
    self.td_plot.setXRange(0.0, 200.0, padding=0)

    self.td_thresh_line = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen('#ff7043', width=1, style=QtCore.Qt.DashLine),
    )
    self.td_plot.addItem(self.td_thresh_line)

    # Vertical scale slider
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

    td_row = QtWidgets.QHBoxLayout()
    td_row.setSpacing(2)
    td_row.addWidget(self.td_plot, stretch=1)
    td_row.addWidget(self.td_scale_slider)
    layout.addLayout(td_row)

    # Span slider
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
    layout.addLayout(td_span_row)

    # Blanker spectrum plot — shows per-bin noise floor and current block
    from .processing import NB_FFT_SIZE
    _nb_freqs = np.fft.fftshift(np.fft.fftfreq(NB_FFT_SIZE, 1.0 / 48000)) / 1000.0
    _nb_zeros = np.full(NB_FFT_SIZE, -120.0, dtype=np.float32)

    self.nb_spec_plot = pg.PlotWidget()
    self.nb_spec_plot.setBackground('#111111')
    self.nb_spec_plot.setLabel('bottom', 'Frequency', units='kHz')
    self.nb_spec_plot.setLabel('left', '')
    self.nb_spec_plot.setTitle('Blanker Spectrum — floor / block')
    self.nb_spec_plot.getAxis('left').setWidth(45)
    self.nb_spec_plot.setMinimumHeight(120)
    self.nb_spec_plot.setMaximumHeight(160)
    self.nb_spec_plot.setXRange(_nb_freqs[0], _nb_freqs[-1], padding=0)
    self.nb_spec_plot.setYRange(-120.0, 0.0, padding=0)
    self.nb_spec_plot.getViewBox().disableAutoRange()

    # Green: slowly-adapting per-bin noise floor
    self.nb_floor_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(),
        pen=pg.mkPen('#66bb6a', width=1),
    )
    self.nb_spec_plot.addItem(self.nb_floor_curve)
    # Cyan: most recent block spectrum
    self.nb_block_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(),
        pen=pg.mkPen('#4fc3f7', width=1, style=QtCore.Qt.DotLine),
    )
    self.nb_spec_plot.addItem(self.nb_block_curve)
    # Red dashed: threshold = floor + 20*log10(nb_factor) dB
    self.nb_thresh_curve = pg.PlotCurveItem(
        _nb_freqs, _nb_zeros.copy(),
        pen=pg.mkPen('#ff7043', width=1, style=QtCore.Qt.DashLine),
    )
    self.nb_spec_plot.addItem(self.nb_thresh_curve)

    layout.addWidget(self.nb_spec_plot)

    # Noise blanker controls
    nb_group = QtWidgets.QGroupBox("Spectral Noise Blanker")
    nb_vbox  = QtWidgets.QVBoxLayout(nb_group)
    nb_vbox.setSpacing(3)

    nb_factor_row = QtWidgets.QHBoxLayout()
    nb_factor_row.addWidget(QtWidgets.QLabel("K (amplitude, K²=power ratio):"))
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

    nb_status_row = QtWidgets.QHBoxLayout()
    nb_status_row.addWidget(QtWidgets.QLabel("Blanked:"))
    self._nb_count_val = QtWidgets.QLabel("0.00%")
    self._nb_count_val.setStyleSheet(
        "QLabel { color: #c76000; font-family: monospace; min-width: 52px; }"
    )
    nb_status_row.addWidget(self._nb_count_val)
    nb_status_row.addSpacing(12)
    nb_status_row.addWidget(QtWidgets.QLabel("Hot bins:"))
    self._nb_hot_val = QtWidgets.QLabel(f"0/{NB_FFT_SIZE}")
    self._nb_hot_val.setStyleSheet(
        "QLabel { color: #ffb74d; font-family: monospace; min-width: 52px; }"
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

    win = _PanelWindow("map144 — Flex Radio", view_action, self, 'flex_geometry')
    win.setMinimumSize(320, 400)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._flex_win = win

    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Center Freq:", _stat_label(self, "_flex_freq_val"))
    _add_stream_rows(stream_form, self, "_flex")
    layout.addWidget(stream_grp)

    radio_grp, radio_form = _form_group("Radio")
    radio_form.addRow("Status:",   _stat_label(self, "_flex_status_val", "Idle", "#c76000"))
    radio_form.addRow("Model:",    _stat_label(self, "_flex_model_val"))
    radio_form.addRow("Serial:",   _stat_label(self, "_flex_serial_val"))
    radio_form.addRow("Firmware:", _stat_label(self, "_flex_firmware_val"))
    radio_form.addRow("IP:",       _stat_label(self, "_flex_ip_val"))
    layout.addWidget(radio_grp)

    dax_grp, dax_form = _form_group("DAXIQ Stream")
    dax_form.addRow("DAX Channel:", _stat_label(self, "_flex_dax_ch_val"))
    dax_form.addRow("Stream ID:",   _stat_label(self, "_flex_stream_id_val"))
    dax_form.addRow("UDP Port:",    _stat_label(self, "_flex_udp_port_val"))
    dax_form.addRow("Mode:",        _stat_label(self, "_flex_mode_val"))
    dax_form.addRow("Pkt Rate:",    _stat_label(self, "_flex_pkt_rate_val"))
    dax_form.addRow("Queue:",       _stat_label(self, "_flex_vita_queue_val"))
    dax_form.addRow("Loss:",        _stat_label(self, "_flex_loss_val"))
    layout.addWidget(dax_grp)
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

    stream_grp, stream_form = _form_group("IF Stream")
    stream_form.addRow("Center Freq:", _stat_label(self, "_usrp_freq_val"))
    _add_stream_rows(stream_form, self, "_usrp")
    layout.addWidget(stream_grp)

    # RX A controls
    rxa_grp = QtWidgets.QGroupBox("RX A")
    rxa_form = QtWidgets.QFormLayout(rxa_grp)
    rxa_form.setVerticalSpacing(4)
    rxa_form.setHorizontalSpacing(8)

    # Gain slider
    try:
        _gain_saved = float(_SETTINGS.value('usrp_gain_db', 50.0))
    except (ValueError, TypeError):
        _gain_saved = 50.0
    self._usrp_gain_db = _gain_saved
    self._usrp_gain_label = QtWidgets.QLabel(f"{_gain_saved:.0f} dB")
    self._usrp_gain_label.setStyleSheet(
        "QLabel { color: #c8e6c9; font-family: monospace; min-width: 48px; }"
    )
    self._usrp_gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._usrp_gain_slider.setMinimum(0)
    self._usrp_gain_slider.setMaximum(76)
    self._usrp_gain_slider.setValue(int(_gain_saved))
    self._usrp_gain_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self._usrp_gain_slider.setTickInterval(10)
    self._usrp_gain_slider.valueChanged.connect(self.on_usrp_gain_changed)
    gain_row = QtWidgets.QHBoxLayout()
    gain_row.addWidget(self._usrp_gain_label)
    gain_row.addWidget(self._usrp_gain_slider, stretch=1)
    rxa_form.addRow("Gain:", gain_row)

    # Antenna selector
    try:
        _ant_saved = str(_SETTINGS.value('usrp_antenna', 'RX2'))
    except Exception:
        _ant_saved = 'RX2'
    self._usrp_antenna_combo = QtWidgets.QComboBox()
    self._usrp_antenna_combo.addItems(["RX2", "TX/RX"])
    self._usrp_antenna_combo.setCurrentText(_ant_saved)
    self._usrp_antenna_combo.currentTextChanged.connect(self.on_usrp_antenna_changed)
    rxa_form.addRow("Antenna:", self._usrp_antenna_combo)

    rxa_form.addRow("Serial:", _stat_label(self, "_usrp_serial_val"))
    layout.addWidget(rxa_grp)

    # RX B stub
    rxb_grp = QtWidgets.QGroupBox("RX B  (future)")
    rxb_grp.setEnabled(False)
    QtWidgets.QVBoxLayout(rxb_grp).addWidget(QtWidgets.QLabel("Not yet implemented"))
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

    # IQ magnitude plot — update at 5 Hz (every other 10 Hz tick)
    if hasattr(self, 'td_curve') and self._noise_floor_ctr % 2 == 0:
        buf = self._td_mag_buf
        pos = self._td_mag_pos
        n   = len(buf)
        if pos == 0:
            full = buf
        else:
            full = np.empty(n, dtype=np.float32)
            full[:n - pos] = buf[pos:]
            full[n - pos:] = buf[:pos]
        span_ms = float(getattr(self, 'td_span_ms', 200.0))
        span_n  = min(int(span_ms * self.sample_rate / 1000.0), n)
        display = full[-span_n:]
        x_axis  = np.linspace(0.0, span_ms, span_n, endpoint=False)
        self.td_plot.setXRange(0.0, span_ms, padding=0)
        self.td_curve.setData(x_axis, display)
        nb_floor = getattr(self, '_nb_floor', None)
        if nb_floor is not None:
            nb_k = float(getattr(self, 'nb_factor', 6))
            self.td_thresh_line.setValue(nb_k * float(nb_floor))

    # Blanker spectrum — floor, current block, threshold curves.
    # Gate to 2 Hz: the floor and threshold are slowly-adapting averages that
    # do not benefit from more frequent redraws, and these three paint() calls
    # showed up as 32% of wall time in the profiler during noise pulse events.
    if hasattr(self, 'nb_floor_curve') and self._noise_floor_ctr % 5 == 0:
        spec_avg = getattr(self, '_nb_spec_avg', None)
        last_P   = getattr(self, '_nb_last_P',   None)
        nb_k     = float(getattr(self, 'nb_factor', 6.0))
        ref      = 20.0 * np.log10(NB_FFT_SIZE)   # 0 dBFS reference for full-scale complex tone

        _freqs = np.fft.fftshift(np.fft.fftfreq(NB_FFT_SIZE, 1.0 / 48000)) / 1000.0

        if spec_avg is not None:
            floor_db = np.fft.fftshift(
                10.0 * np.log10(np.maximum(spec_avg, 1e-30)) - ref
            ).astype(np.float32)
            self.nb_floor_curve.setData(_freqs, floor_db)
            # Threshold = floor + 20*log10(K) dB  (since metric threshold = K²)
            thresh_db = floor_db + 20.0 * np.log10(max(nb_k, 1.0))
            self.nb_thresh_curve.setData(_freqs, thresh_db)

        if last_P is not None:
            block_db = np.fft.fftshift(
                10.0 * np.log10(np.maximum(last_P, 1e-30)) - ref
            ).astype(np.float32)
            self.nb_block_curve.setData(_freqs, block_db)

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
        status, color = "Running", "#1a6b1a"
    else:
        status, color = "VITA stopped", "#b71c1c"
    lbl = getattr(self, '_flex_status_val', None)
    if lbl is not None:
        lbl.setText(status)
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-family: monospace; }}")

    # IF stream
    if hasattr(self, '_flex_freq_val'):
        freq = None
        if dax is not None:
            freq = dax.slice_frequency_mhz or dax.pan_frequency_mhz
        if freq:
            self._flex_freq_val.setText(f"{freq:.6f} MHz")
        else:
            self._flex_freq_val.setText(f"{self.center_freq_mhz:.6f} MHz (req)")
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
            label = chr(ord('A') + slice_id) if 0 <= slice_id < 26 else str(slice_id)
            mode = f"Slice {label}"
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


def _update_usrp_window(self, sig_str, noise_str, rate_str, drops_str):
    uc = getattr(self, 'usrp_client', None)
    if hasattr(self, '_usrp_freq_val'):
        freq = uc.center_freq_mhz_actual if uc is not None else self.center_freq_mhz
        self._usrp_freq_val.setText(f"{freq:.6f} MHz")
    _set_stream_vals(self, "_usrp", rate_str, sig_str, noise_str, drops_str)
    _set_queue_label(self, '_usrp_queue_val', uc.sample_queue if uc is not None else None)
    # Query serial once after hardware opens
    if (uc is not None and getattr(uc, '_usrp', None) is not None
            and not getattr(self, '_usrp_serial_queried', False)):
        try:
            info   = uc._usrp.get_usrp_rx_info(0)
            serial = info.get('mboard_serial', '—')
            _setlbl(self, '_usrp_serial_val', serial)
        except Exception:
            _setlbl(self, '_usrp_serial_val', '—')
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
