# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Thin PyQt5 GUI for the B210 multi-mode audio / IQ router."""

from __future__ import annotations

import logging
import sys
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets

from .band_plan import DialChannel, bands, channels_for_band
from .engine import RouterConfig, RouterEngine
from .lo_planner import LoPlanError
from .wideband_iq import MAP65_DEFAULT_PORT, QMAP_DEFAULT_PORT, WidebandDest

log = logging.getLogger(__name__)

_FONT_PT = 12


class RouterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAP144 — B210 Audio / IQ Router")
        self.resize(720, 640)
        self.engine = RouterEngine()
        self._channel_checks: list[tuple[DialChannel, QtWidgets.QCheckBox]] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Band ──────────────────────────────────────────────────────────
        band_row = QtWidgets.QHBoxLayout()
        band_row.addWidget(QtWidgets.QLabel("Band:"))
        self.band_combo = QtWidgets.QComboBox()
        for b in bands():
            self.band_combo.addItem(b)
        self.band_combo.currentTextChanged.connect(self._rebuild_channel_list)
        band_row.addWidget(self.band_combo, 1)
        btn_common = QtWidgets.QPushButton("Select common")
        btn_common.setToolTip("MSK144 + FT8 + FT4 on this band")
        btn_common.clicked.connect(self._select_common)
        band_row.addWidget(btn_common)
        btn_none = QtWidgets.QPushButton("Clear")
        btn_none.clicked.connect(self._clear_channels)
        band_row.addWidget(btn_none)
        layout.addLayout(band_row)

        # ── Channel checklist ─────────────────────────────────────────────
        grp = QtWidgets.QGroupBox("WSJT-X dial channels")
        grp_l = QtWidgets.QVBoxLayout(grp)
        self.channel_host = QtWidgets.QWidget()
        self.channel_layout = QtWidgets.QVBoxLayout(self.channel_host)
        self.channel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_layout.setSpacing(2)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.channel_host)
        scroll.setMinimumHeight(220)
        grp_l.addWidget(scroll)
        layout.addWidget(grp, 1)

        # ── Wideband IQ ───────────────────────────────────────────────────
        iq_grp = QtWidgets.QGroupBox("Wideband IQ outputs (Linrad TIMF2)")
        iq_form = QtWidgets.QFormLayout(iq_grp)

        self.map65_cb = QtWidgets.QCheckBox("MAP65")
        self.map65_host = QtWidgets.QLineEdit("127.0.0.1")
        self.map65_port = QtWidgets.QSpinBox()
        self.map65_port.setRange(1, 65535)
        self.map65_port.setValue(MAP65_DEFAULT_PORT)
        self.map65_center = QtWidgets.QDoubleSpinBox()
        self.map65_center.setDecimals(3)
        self.map65_center.setRange(50.0, 6000.0)
        self.map65_center.setSingleStep(0.005)
        self.map65_center.setSuffix(" MHz")
        self.map65_center.setValue(144.125)
        map65_row = QtWidgets.QHBoxLayout()
        map65_row.addWidget(self.map65_cb)
        map65_row.addWidget(QtWidgets.QLabel("host"))
        map65_row.addWidget(self.map65_host)
        map65_row.addWidget(QtWidgets.QLabel("port"))
        map65_row.addWidget(self.map65_port)
        map65_row.addWidget(QtWidgets.QLabel("centre"))
        map65_row.addWidget(self.map65_center)
        iq_form.addRow(map65_row)

        self.qmap_cb = QtWidgets.QCheckBox("QMAP")
        self.qmap_host = QtWidgets.QLineEdit("127.0.0.1")
        self.qmap_port = QtWidgets.QSpinBox()
        self.qmap_port.setRange(1, 65535)
        self.qmap_port.setValue(QMAP_DEFAULT_PORT)
        self.qmap_center = QtWidgets.QDoubleSpinBox()
        self.qmap_center.setDecimals(3)
        self.qmap_center.setRange(50.0, 6000.0)
        self.qmap_center.setSingleStep(0.005)
        self.qmap_center.setSuffix(" MHz")
        self.qmap_center.setValue(144.125)
        qmap_row = QtWidgets.QHBoxLayout()
        qmap_row.addWidget(self.qmap_cb)
        qmap_row.addWidget(QtWidgets.QLabel("host"))
        qmap_row.addWidget(self.qmap_host)
        qmap_row.addWidget(QtWidgets.QLabel("port"))
        qmap_row.addWidget(self.qmap_port)
        qmap_row.addWidget(QtWidgets.QLabel("centre"))
        qmap_row.addWidget(self.qmap_center)
        iq_form.addRow(qmap_row)
        layout.addWidget(iq_grp)

        # ── Options ───────────────────────────────────────────────────────
        opt_row = QtWidgets.QHBoxLayout()
        self.dual_cb = QtWidgets.QCheckBox("Dual RF ports (RF0+RF1)")
        self.dual_audio_cb = QtWidgets.QCheckBox("Export audio from both RF ports")
        self.dual_audio_cb.setEnabled(False)
        self.dual_cb.toggled.connect(self.dual_audio_cb.setEnabled)
        opt_row.addWidget(self.dual_cb)
        opt_row.addWidget(self.dual_audio_cb)
        opt_row.addStretch(1)
        layout.addLayout(opt_row)

        # ── Plan readout ──────────────────────────────────────────────────
        self.plan_label = QtWidgets.QLabel("Plan: (select channels)")
        self.plan_label.setWordWrap(True)
        layout.addWidget(self.plan_label)

        # ── Apply / Stop ──────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.status = QtWidgets.QLabel("Idle — this process owns the B210 while running.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Live plan preview
        for w in (self.map65_cb, self.qmap_cb, self.map65_center, self.qmap_center):
            if hasattr(w, "toggled"):
                w.toggled.connect(self._update_plan_preview)
            else:
                w.valueChanged.connect(self._update_plan_preview)

        self._rebuild_channel_list(self.band_combo.currentText())
        self._set_font()

    def _set_font(self):
        f = self.font()
        f.setPointSize(_FONT_PT)
        self.setFont(f)

    def _rebuild_channel_list(self, band: str):
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._channel_checks.clear()
        for ch in channels_for_band(band):
            cb = QtWidgets.QCheckBox(f"{ch.mode:<7}  {ch.dial_mhz:8.3f} MHz  ({ch.label})")
            cb.setProperty("dial_channel", ch)
            cb.toggled.connect(self._update_plan_preview)
            # Sensible defaults for MAP65 centre when switching to 2m/6m
            self.channel_layout.addWidget(cb)
            self._channel_checks.append((ch, cb))
        self.channel_layout.addStretch(1)
        if band == "2m":
            self.map65_center.setValue(144.125)
            self.qmap_center.setValue(144.125)
        elif band == "6m":
            self.map65_center.setValue(50.280)
            self.qmap_center.setValue(50.280)
        self._update_plan_preview()

    def _selected_dials(self) -> list[DialChannel]:
        return [ch for ch, cb in self._channel_checks if cb.isChecked()]

    def _select_common(self):
        want = {"MSK144", "FT8", "FT4"}
        for ch, cb in self._channel_checks:
            # Prefer primary labels (skip "alt" / "DX" / "low" extras for one-click)
            primary = ch.label in (ch.mode, "MSK144", "FT8", "FT4")
            cb.setChecked(ch.mode in want and primary)

    def _clear_channels(self):
        for _ch, cb in self._channel_checks:
            cb.setChecked(False)

    def _build_config(self) -> RouterConfig:
        map65 = None
        if self.map65_cb.isChecked():
            map65 = WidebandDest(
                "MAP65",
                self.map65_host.text().strip() or "127.0.0.1",
                int(self.map65_port.value()),
                float(self.map65_center.value()),
            )
        qmap = None
        if self.qmap_cb.isChecked():
            qmap = WidebandDest(
                "QMAP",
                self.qmap_host.text().strip() or "127.0.0.1",
                int(self.qmap_port.value()),
                float(self.qmap_center.value()),
            )
        return RouterConfig(
            dials=self._selected_dials(),
            map65=map65,
            qmap=qmap,
            dual_channel=self.dual_cb.isChecked(),
            dual_rf_audio=self.dual_audio_cb.isChecked(),
        )

    def _update_plan_preview(self, *_args):
        try:
            cfg = self._build_config()
            if not cfg.dials and cfg.map65 is None and cfg.qmap is None:
                self.plan_label.setText("Plan: (select channels or enable MAP65/QMAP)")
                return
            plan = self.engine.plan_only(cfg)
            self.plan_label.setText(
                f"Plan: pan={plan.pan_center_mhz:.6f} MHz   "
                f"hw_rate={plan.hw_rate_hz/1000:.0f} kHz   "
                f"span={plan.span_hz/1000:.1f} kHz   "
                f"dials={plan.dial_count}"
            )
        except LoPlanError as exc:
            self.plan_label.setText(f"Plan: ERROR — {exc}")
        except Exception as exc:
            self.plan_label.setText(f"Plan: {exc}")

    def _on_apply(self):
        if self.engine.running:
            self._on_stop()
        try:
            cfg = self._build_config()
            plan = self.engine.start(cfg)
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Router start failed", str(exc))
            self.status.setText(f"Error: {exc}")
            return
        sinks = self.engine.sink_names
        sink_txt = "\n".join(f"  • Monitor of {s}" for s in sinks) if sinks else "  (none)"
        self.status.setText(
            f"Running — pan {plan.pan_center_mhz:.6f} MHz @ {plan.hw_rate_hz/1000:.0f} kHz\n"
            f"WSJT-X Audio Input =\n{sink_txt}\n"
            f"(descriptions appear as 'MAP144 … -> WSJT-X' in the WSJT-X dropdown)"
        )
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_controls_enabled(False)

    def _on_stop(self):
        try:
            self.engine.stop()
        except Exception as exc:
            log.warning("stop: %s", exc)
        self.status.setText("Stopped.")
        self.apply_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_controls_enabled(True)
        self._update_plan_preview()

    def _set_controls_enabled(self, enabled: bool):
        self.band_combo.setEnabled(enabled)
        for _ch, cb in self._channel_checks:
            cb.setEnabled(enabled)
        for w in (self.map65_cb, self.map65_host, self.map65_port, self.map65_center,
                  self.qmap_cb, self.qmap_host, self.qmap_port, self.qmap_center,
                  self.dual_cb):
            w.setEnabled(enabled)
        self.dual_audio_cb.setEnabled(enabled and self.dual_cb.isChecked())

    def closeEvent(self, event):
        if self.engine.running:
            self.engine.stop()
        super().closeEvent(event)


def run_app():
    # Light palette
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(245, 245, 245))
    pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.black)
    pal.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
    app.setPalette(pal)
    f = app.font()
    f.setPointSize(_FONT_PT)
    app.setFont(f)
    win = RouterWindow()
    win.show()
    return app.exec_()
