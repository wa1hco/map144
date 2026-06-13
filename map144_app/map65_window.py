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
"""MAP65 Export panel — set the destination IP/port and frequency plan for the
96 kHz dual-pol timf2 I/Q stream tee'd from the B210 to MAP65.

The actual DSP/transport lives in map65_export.Map65Exporter; this window just
collects settings, owns the exporter, and attaches it to the live USRPSource.
"""

from PyQt5 import QtWidgets, QtCore


def setup_map65_window(self, view_action):
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS
    from .map65_export import Map65Exporter

    win = _PanelWindow("map144 — MAP65 Export", view_action, self, 'map65_geometry')
    win.setMinimumSize(360, 360)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._map65_win = win

    # ── Destination (the thing you asked for: IP + port) ──────────────────────
    dst_grp  = QtWidgets.QGroupBox("MAP65 Destination  (Linrad timf2 I/Q, 96 kHz)")
    dst_form = QtWidgets.QFormLayout(dst_grp)
    dst_form.setVerticalSpacing(4)
    dst_form.setHorizontalSpacing(8)

    self._map65_enable_cb = QtWidgets.QCheckBox("Enable")
    self._map65_enable_cb.setChecked(_SETTINGS.value('map65_enabled', False, type=bool))
    dst_form.addRow("", self._map65_enable_cb)

    self._map65_host = QtWidgets.QLineEdit(str(_SETTINGS.value('map65_host', '127.0.0.1')))
    self._map65_host.setPlaceholderText("MAP65 PC IP, e.g. 192.168.10.20")
    dst_form.addRow("IP address:", self._map65_host)

    self._map65_port = QtWidgets.QSpinBox()
    self._map65_port.setRange(1, 65535)
    try:
        self._map65_port.setValue(int(_SETTINGS.value('map65_port', 50002)))
    except (ValueError, TypeError):
        self._map65_port.setValue(50002)
    dst_form.addRow("Port:", self._map65_port)
    layout.addWidget(dst_grp)

    # ── Frequency plan ────────────────────────────────────────────────────────
    fp_grp  = QtWidgets.QGroupBox("Frequency plan")
    fp_form = QtWidgets.QFormLayout(fp_grp)
    fp_form.setVerticalSpacing(4)
    fp_form.setHorizontalSpacing(8)

    self._map65_center = QtWidgets.QDoubleSpinBox()
    self._map65_center.setDecimals(3)
    self._map65_center.setRange(144.000, 144.200)
    self._map65_center.setSingleStep(0.005)
    self._map65_center.setSuffix(" MHz")
    try:
        self._map65_center.setValue(float(_SETTINGS.value('map65_center_mhz', 144.125)))
    except (ValueError, TypeError):
        self._map65_center.setValue(144.125)
    fp_form.addRow("MAP65 centre:", self._map65_center)

    self._map65_pan = QtWidgets.QDoubleSpinBox()
    self._map65_pan.setDecimals(3)
    self._map65_pan.setRange(144.000, 144.200)
    self._map65_pan.setSingleStep(0.005)
    self._map65_pan.setSuffix(" MHz")
    try:
        self._map65_pan.setValue(float(_SETTINGS.value('map65_pan_center_mhz', 144.100)))
    except (ValueError, TypeError):
        self._map65_pan.setValue(144.100)
    fp_form.addRow("B210 DC centre:", self._map65_pan)

    self._map65_level = QtWidgets.QSpinBox()
    self._map65_level.setRange(1, 4_000_000)   # B210 fc32 amplitudes are tiny;
    self._map65_level.setSingleStep(1000)      # needs a large multiplier to fill int16
    try:
        self._map65_level.setValue(int(_SETTINGS.value('map65_level', 4096)))
    except (ValueError, TypeError):
        self._map65_level.setValue(4096)
    self._map65_level.setToolTip(
        "int16 full-scale per unit IQ amplitude.  Raise until MAP65's level bar "
        "turns green (orange = too low/high).  Digital scaling only — safe to "
        "raise until peaks approach full scale; watch the [map65] level log.")
    fp_form.addRow("int16 level:", self._map65_level)
    layout.addWidget(fp_grp)

    # Enter in any field applies immediately (same as the Apply button).
    self._map65_host.returnPressed.connect(lambda: _on_map65_apply(self))
    for _w in (self._map65_port, self._map65_center, self._map65_pan, self._map65_level):
        _w.lineEdit().returnPressed.connect(lambda: _on_map65_apply(self))

    note = QtWidgets.QLabel(
        "Re-centres the B210 DC (artifact) to the 'B210 DC centre' and sends a "
        "96 kHz window centred on 'MAP65 centre'.  Requires the USRP B210 source "
        "in dual-pol mode.  MSK144 detection keeps running unaffected.")
    note.setWordWrap(True)
    note.setStyleSheet("QLabel { color: #555; font-style: italic; }")
    layout.addWidget(note)

    apply_btn = QtWidgets.QPushButton("Apply")
    apply_btn.clicked.connect(lambda: _on_map65_apply(self))
    layout.addWidget(apply_btn)

    # ── Status ────────────────────────────────────────────────────────────────
    stat_grp  = QtWidgets.QGroupBox("Status")
    stat_form = QtWidgets.QFormLayout(stat_grp)
    stat_form.setVerticalSpacing(3)
    stat_form.setHorizontalSpacing(8)

    def _slbl(attr, init="—"):
        lbl = QtWidgets.QLabel(init)
        lbl.setStyleSheet("QLabel { color: #1a6b1a; font-family: monospace; }")
        setattr(self, attr, lbl)
        return lbl

    stat_form.addRow("State:",          _slbl("_map65_state_val", "disabled"))
    stat_form.addRow("Packets sent:",   _slbl("_map65_pkts_val", "0"))
    stat_form.addRow("Last error:",     _slbl("_map65_err_val", "none"))
    layout.addWidget(stat_grp)
    layout.addStretch()

    self.map65_exporter = Map65Exporter()
    QtCore.QTimer.singleShot(0, lambda: _on_map65_apply(self))


def _on_map65_apply(self):
    """Read fields, persist, push to the exporter, attach to the USRP source."""
    from .visualizer import _SETTINGS

    enabled = self._map65_enable_cb.isChecked()
    host    = self._map65_host.text().strip()
    port    = self._map65_port.value()
    center  = self._map65_center.value()
    pan     = self._map65_pan.value()
    level   = self._map65_level.value()

    _SETTINGS.setValue('map65_enabled',        enabled)
    _SETTINGS.setValue('map65_host',           host)
    _SETTINGS.setValue('map65_port',           port)
    _SETTINGS.setValue('map65_center_mhz',     center)
    _SETTINGS.setValue('map65_pan_center_mhz', pan)
    _SETTINGS.setValue('map65_level',          level)

    exp = getattr(self, 'map65_exporter', None)
    if exp is None:
        return
    exp.apply_settings(host, port, pan, center, level)

    # Attach to the live USRP source (if that's the current source) and apply
    # the re-centering.  Other sources (Flex/WAV) simply ignore the exporter.
    uc = getattr(self, 'usrp_client', None)
    if uc is not None and hasattr(uc, 'map65_exporter'):
        uc.map65_exporter = exp
        if enabled:
            uc.pan_center_mhz = pan
            if getattr(uc, '_usrp', None) is not None:
                # Already streaming: move the LO to pan_center and rebuild NCO.
                uc.retune(uc.center_freq_mhz)

    if enabled:
        if exp._sock is None:
            exp.start()
        exp.enabled = (exp._sock is not None)
    else:
        exp.stop()


def update_map65_window(self):
    """Refresh status labels — called from the periodic source-window update."""
    exp = getattr(self, 'map65_exporter', None)
    if exp is None:
        return

    def _set(attr, text):
        lbl = getattr(self, attr, None)
        if lbl is not None:
            lbl.setText(text)

    _set('_map65_state_val', "streaming" if exp.enabled else "disabled")
    _set('_map65_pkts_val',  f"{exp.packets_sent:,}")
    err = exp.last_error or "none"
    lbl = getattr(self, '_map65_err_val', None)
    if lbl is not None:
        lbl.setText(err)
        lbl.setStyleSheet(
            f"QLabel {{ color: {'#b71c1c' if exp.last_error else '#1a6b1a'}; "
            f"font-family: monospace; }}")
