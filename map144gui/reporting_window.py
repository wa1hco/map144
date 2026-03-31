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
"""Reporting panel window — My Station settings, WSJT-X UDP and PSKReporter controls."""

from PyQt5 import QtWidgets, QtCore


def setup_reporting_window(self, view_action):
    """Build the Reporting panel window and attach reporter to visualizer."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS
    from .reporting import Reporter

    win = _PanelWindow("map144 — Reporting", view_action, self, 'reporting_geometry')
    win.setMinimumSize(360, 480)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    self._reporting_win = win

    # ── My Station ────────────────────────────────────────────────────────────
    stn_grp  = QtWidgets.QGroupBox("My Station")
    stn_form = QtWidgets.QFormLayout(stn_grp)
    stn_form.setVerticalSpacing(4)
    stn_form.setHorizontalSpacing(8)

    self._rpt_mycall_edit = QtWidgets.QLineEdit(
        str(_SETTINGS.value('reporting_mycall', ''))
    )
    self._rpt_mycall_edit.setPlaceholderText("e.g. WA1HCO")
    self._rpt_mycall_edit.setMaxLength(12)
    stn_form.addRow("My Callsign:", self._rpt_mycall_edit)

    self._rpt_mygrid_edit = QtWidgets.QLineEdit(
        str(_SETTINGS.value('reporting_mygrid', ''))
    )
    self._rpt_mygrid_edit.setPlaceholderText("e.g. FN42")
    self._rpt_mygrid_edit.setMaxLength(8)
    stn_form.addRow("My Grid:", self._rpt_mygrid_edit)

    layout.addWidget(stn_grp)

    # ── WSJT-X UDP ────────────────────────────────────────────────────────────
    udp_grp  = QtWidgets.QGroupBox("WSJT-X UDP  (GridTracker / N1MM / JTAlert)")
    udp_form = QtWidgets.QFormLayout(udp_grp)
    udp_form.setVerticalSpacing(4)
    udp_form.setHorizontalSpacing(8)

    self._rpt_wsjtx_cb = QtWidgets.QCheckBox("Enable")
    self._rpt_wsjtx_cb.setChecked(bool(_SETTINGS.value('reporting_wsjtx_enabled', False)))
    udp_form.addRow("", self._rpt_wsjtx_cb)

    self._rpt_wsjtx_host = QtWidgets.QLineEdit(
        str(_SETTINGS.value('reporting_wsjtx_host', '127.0.0.1'))
    )
    udp_form.addRow("Host:", self._rpt_wsjtx_host)

    self._rpt_wsjtx_port = QtWidgets.QSpinBox()
    self._rpt_wsjtx_port.setRange(1024, 65535)
    try:
        self._rpt_wsjtx_port.setValue(int(_SETTINGS.value('reporting_wsjtx_port', 2237)))
    except (ValueError, TypeError):
        self._rpt_wsjtx_port.setValue(2237)
    udp_form.addRow("Port:", self._rpt_wsjtx_port)

    layout.addWidget(udp_grp)

    # ── PSKReporter ───────────────────────────────────────────────────────────
    psk_grp  = QtWidgets.QGroupBox("PSKReporter")
    psk_form = QtWidgets.QFormLayout(psk_grp)
    psk_form.setVerticalSpacing(4)
    psk_form.setHorizontalSpacing(8)

    self._rpt_psk_cb = QtWidgets.QCheckBox("Enable")
    self._rpt_psk_cb.setChecked(bool(_SETTINGS.value('reporting_psk_enabled', False)))
    psk_form.addRow("", self._rpt_psk_cb)
    psk_form.addRow("Upload interval:", QtWidgets.QLabel("5 min (fixed)"))

    layout.addWidget(psk_grp)

    # ── Apply button ──────────────────────────────────────────────────────────
    apply_btn = QtWidgets.QPushButton("Apply")
    apply_btn.clicked.connect(lambda: _on_reporting_apply(self))
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

    stat_form.addRow("UDP packets sent:", _slbl("_rpt_udp_sent_val"))
    stat_form.addRow("PSK spots queued:", _slbl("_rpt_psk_queued_val"))
    stat_form.addRow("PSK spots uploaded:", _slbl("_rpt_psk_uploaded_val"))
    stat_form.addRow("Last PSK upload:", _slbl("_rpt_psk_time_val"))
    stat_form.addRow("Last error:", _slbl("_rpt_error_val", "none"))
    layout.addWidget(stat_grp)

    layout.addStretch()

    # ── Instantiate Reporter and apply saved settings immediately ─────────────
    self.reporter = Reporter()
    _on_reporting_apply(self)
    self.reporter.start()


def _on_reporting_apply(self):
    """Read UI fields, persist to QSettings, push to Reporter."""
    from .visualizer import _SETTINGS

    my_call  = self._rpt_mycall_edit.text().strip().upper()
    my_grid  = self._rpt_mygrid_edit.text().strip().upper()
    wsjtx_en = self._rpt_wsjtx_cb.isChecked()
    wsjtx_h  = self._rpt_wsjtx_host.text().strip()
    wsjtx_p  = self._rpt_wsjtx_port.value()
    psk_en   = self._rpt_psk_cb.isChecked()

    _SETTINGS.setValue('reporting_mycall',          my_call)
    _SETTINGS.setValue('reporting_mygrid',           my_grid)
    _SETTINGS.setValue('reporting_wsjtx_enabled',    wsjtx_en)
    _SETTINGS.setValue('reporting_wsjtx_host',       wsjtx_h)
    _SETTINGS.setValue('reporting_wsjtx_port',       wsjtx_p)
    _SETTINGS.setValue('reporting_psk_enabled',      psk_en)

    rpt = getattr(self, 'reporter', None)
    if rpt is not None:
        rpt.apply_settings(my_call, my_grid, wsjtx_en, wsjtx_h, wsjtx_p, psk_en)


def update_reporting_window(self):
    """Refresh status labels — called from update_source_windows at 10 Hz."""
    rpt = getattr(self, 'reporter', None)
    if rpt is None:
        return

    def _set(attr, text):
        lbl = getattr(self, attr, None)
        if lbl is not None:
            lbl.setText(text)

    _set('_rpt_udp_sent_val',    f"{rpt.stat_udp_sent:,}")
    _set('_rpt_psk_queued_val',  str(rpt.stat_psk_queued))
    _set('_rpt_psk_uploaded_val',f"{rpt.stat_psk_uploaded:,}")
    _set('_rpt_psk_time_val',    rpt.stat_last_psk_time or "—")
    err = rpt.stat_last_error or "none"
    lbl = getattr(self, '_rpt_error_val', None)
    if lbl is not None:
        lbl.setText(err)
        lbl.setStyleSheet(
            f"QLabel {{ color: {'#b71c1c' if rpt.stat_last_error else '#1a6b1a'}; "
            f"font-family: monospace; }}"
        )
