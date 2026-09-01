# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""PyQt5 GUI for the B210 multi-mode audio / IQ router.

Merges channel selection with MAP144-style B210 gain/status and noise-blanker
controls on one window.
"""

from __future__ import annotations

import logging
import sys
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets

from ..noise_blanker import NB_FACTOR, available as blanker_available
from .band_plan import DialChannel, bands, channels_for_band
from .engine import RouterConfig, RouterEngine
from .lo_planner import LoPlanError
from .wideband_iq import MAP65_DEFAULT_PORT, QMAP_DEFAULT_PORT, WidebandDest

log = logging.getLogger(__name__)

_FONT_PT = 12
_MONO = "font-family: monospace;"


def _settings():
    return QtCore.QSettings("WA1HCO", "map144")


class RouterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAP144 — B210 Audio / IQ Router")
        self.resize(960, 720)
        self.engine = RouterEngine()
        self._channel_checks: list[tuple[DialChannel, QtWidgets.QCheckBox]] = []
        self._hw_info_shown = False

        s = _settings()
        try:
            self._gain0 = float(s.value("usrp_gain_db", 50.0))
        except (TypeError, ValueError):
            self._gain0 = 50.0
        try:
            self._gain1 = float(s.value("usrp_gain_db_ch1", self._gain0))
        except (TypeError, ValueError):
            self._gain1 = self._gain0
        self._ant0 = str(s.value("usrp_antenna", "RX2"))
        self._ant1 = str(s.value("usrp_antenna_ch1", "RX2"))
        self._nb_name = str(s.value("nb_backend", "Linrad"))
        try:
            self._nb_k = float(s.value("nb_factor", NB_FACTOR))
        except (TypeError, ValueError):
            self._nb_k = float(NB_FACTOR)
        try:
            self._nb_k_v = float(s.value("nb_factor_v", self._nb_k))
        except (TypeError, ValueError):
            self._nb_k_v = self._nb_k

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

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
        root.addLayout(band_row)

        # ── Mid: channels | B210 + blanker ────────────────────────────────
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(10)

        ch_grp = QtWidgets.QGroupBox("WSJT-X dial channels")
        ch_l = QtWidgets.QVBoxLayout(ch_grp)
        self.channel_host = QtWidgets.QWidget()
        self.channel_layout = QtWidgets.QVBoxLayout(self.channel_host)
        self.channel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_layout.setSpacing(2)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.channel_host)
        scroll.setMinimumWidth(320)
        ch_l.addWidget(scroll)
        mid.addWidget(ch_grp, 2)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self._build_b210_panel())
        right.addWidget(self._build_blanker_panel())
        right.addStretch(1)
        mid.addLayout(right, 3)
        root.addLayout(mid, 1)

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
        root.addWidget(iq_grp)

        # ── Options ───────────────────────────────────────────────────────
        opt_row = QtWidgets.QHBoxLayout()
        self.dual_cb = QtWidgets.QCheckBox("Dual RF ports (RF0+RF1)")
        self.dual_audio_cb = QtWidgets.QCheckBox("Export audio from both RF ports")
        self.dual_audio_cb.setEnabled(False)
        self.dual_cb.toggled.connect(self._on_dual_toggled)
        opt_row.addWidget(self.dual_cb)
        opt_row.addWidget(self.dual_audio_cb)
        opt_row.addStretch(1)
        root.addLayout(opt_row)

        self.plan_label = QtWidgets.QLabel("Plan: (select channels)")
        self.plan_label.setWordWrap(True)
        root.addWidget(self.plan_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.status = QtWidgets.QLabel("Idle — this process owns the B210 while running.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        for w in (self.map65_cb, self.qmap_cb, self.map65_center, self.qmap_center):
            if hasattr(w, "toggled"):
                w.toggled.connect(self._update_plan_preview)
            else:
                w.valueChanged.connect(self._update_plan_preview)

        self._rebuild_channel_list(self.band_combo.currentText())
        self._set_font()
        self._on_dual_toggled(self.dual_cb.isChecked())

        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # ── B210 / blanker panels ─────────────────────────────────────────────

    def _build_b210_panel(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("USRP B210")
        form = QtWidgets.QFormLayout(grp)
        form.setVerticalSpacing(4)

        self._lbl_serial = QtWidgets.QLabel("—")
        self._lbl_serial.setStyleSheet(_MONO)
        self._lbl_fw = QtWidgets.QLabel("—")
        self._lbl_fw.setStyleSheet(_MONO)
        self._lbl_fpga = QtWidgets.QLabel("—")
        self._lbl_fpga.setStyleSheet(_MONO)
        self._lbl_pan = QtWidgets.QLabel("—")
        self._lbl_pan.setStyleSheet(_MONO)
        self._lbl_rate = QtWidgets.QLabel("—")
        self._lbl_rate.setStyleSheet(_MONO)
        self._lbl_bufs = QtWidgets.QLabel("—")
        self._lbl_bufs.setStyleSheet(_MONO)
        self._lbl_phase = QtWidgets.QLabel("idle")
        self._lbl_phase.setStyleSheet(_MONO)
        form.addRow("Serial:", self._lbl_serial)
        form.addRow("Firmware:", self._lbl_fw)
        form.addRow("FPGA:", self._lbl_fpga)
        form.addRow("Pan / LO:", self._lbl_pan)
        form.addRow("Sample rate:", self._lbl_rate)
        form.addRow("Buffer rate:", self._lbl_bufs)
        form.addRow("Startup:", self._lbl_phase)

        # RF0
        self._lbl_rx0 = QtWidgets.QLabel("● Idle")
        self._lbl_rx0.setStyleSheet(_MONO + " color: #555;")
        self._gain0_label = QtWidgets.QLabel(f"{self._gain0:.0f} dB")
        self._gain0_label.setStyleSheet(_MONO + " color: #1a6b1a; min-width: 48px;")
        self._gain0_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._gain0_slider.setRange(0, 76)
        self._gain0_slider.setValue(int(self._gain0))
        self._gain0_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._gain0_slider.setTickInterval(10)
        self._gain0_slider.valueChanged.connect(self._on_gain0)
        g0 = QtWidgets.QHBoxLayout()
        g0.addWidget(self._gain0_slider, 1)
        g0.addWidget(self._gain0_label)
        self._ant0_combo = QtWidgets.QComboBox()
        self._ant0_combo.addItems(["RX2", "TX/RX"])
        if self._ant0 in ("RX2", "TX/RX"):
            self._ant0_combo.setCurrentText(self._ant0)
        self._ant0_combo.currentTextChanged.connect(self._on_ant0)
        self._lbl_rms0 = QtWidgets.QLabel("—")
        self._lbl_rms0.setStyleSheet(_MONO)
        form.addRow("RF0 status:", self._lbl_rx0)
        form.addRow("RF0 gain:", g0)
        form.addRow("RF0 antenna:", self._ant0_combo)
        form.addRow("RF0 level:", self._lbl_rms0)

        # RF1
        self._rf1_widgets = []
        self._lbl_rx1 = QtWidgets.QLabel("● Idle")
        self._lbl_rx1.setStyleSheet(_MONO + " color: #555;")
        self._gain1_label = QtWidgets.QLabel(f"{self._gain1:.0f} dB")
        self._gain1_label.setStyleSheet(_MONO + " color: #1a6b1a; min-width: 48px;")
        self._gain1_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._gain1_slider.setRange(0, 76)
        self._gain1_slider.setValue(int(self._gain1))
        self._gain1_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._gain1_slider.setTickInterval(10)
        self._gain1_slider.valueChanged.connect(self._on_gain1)
        g1 = QtWidgets.QHBoxLayout()
        g1.addWidget(self._gain1_slider, 1)
        g1.addWidget(self._gain1_label)
        self._ant1_combo = QtWidgets.QComboBox()
        self._ant1_combo.addItems(["RX2", "TX/RX"])
        if self._ant1 in ("RX2", "TX/RX"):
            self._ant1_combo.setCurrentText(self._ant1)
        self._ant1_combo.currentTextChanged.connect(self._on_ant1)
        self._lbl_rms1 = QtWidgets.QLabel("—")
        self._lbl_rms1.setStyleSheet(_MONO)
        form.addRow("RF1 status:", self._lbl_rx1)
        form.addRow("RF1 gain:", g1)
        form.addRow("RF1 antenna:", self._ant1_combo)
        form.addRow("RF1 level:", self._lbl_rms1)
        return grp

    def _build_blanker_panel(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Noise Blanker")
        v = QtWidgets.QVBoxLayout(grp)
        v.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Backend:"))
        self.nb_backend_combo = QtWidgets.QComboBox()
        for name in blanker_available():
            self.nb_backend_combo.addItem(name)
        idx = self.nb_backend_combo.findText(self._nb_name)
        if idx >= 0:
            self.nb_backend_combo.setCurrentIndex(idx)
        self.nb_backend_combo.currentTextChanged.connect(self._on_nb_backend)
        row.addWidget(self.nb_backend_combo, 1)
        v.addLayout(row)

        k_row = QtWidgets.QHBoxLayout()
        k_row.addWidget(QtWidgets.QLabel("K H:"))
        self.nb_k_label = QtWidgets.QLabel(f"{self._nb_k:.1f}")
        self.nb_k_label.setStyleSheet(_MONO + " color: #1a6b1a; min-width: 32px;")
        k_row.addWidget(self.nb_k_label)
        self.nb_k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nb_k_slider.setRange(20, 100)  # 2.0 .. 10.0
        self.nb_k_slider.setValue(int(round(self._nb_k * 10)))
        self.nb_k_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.nb_k_slider.setTickInterval(10)
        self.nb_k_slider.valueChanged.connect(self._on_nb_k)
        k_row.addWidget(self.nb_k_slider, 1)
        v.addLayout(k_row)

        self._nb_k_v_widget = QtWidgets.QWidget()
        kv = QtWidgets.QHBoxLayout(self._nb_k_v_widget)
        kv.setContentsMargins(0, 0, 0, 0)
        kv.addWidget(QtWidgets.QLabel("K V:"))
        self.nb_k_v_label = QtWidgets.QLabel(f"{self._nb_k_v:.1f}")
        self.nb_k_v_label.setStyleSheet(_MONO + " color: #1a6b1a; min-width: 32px;")
        kv.addWidget(self.nb_k_v_label)
        self.nb_k_v_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nb_k_v_slider.setRange(20, 100)
        self.nb_k_v_slider.setValue(int(round(self._nb_k_v * 10)))
        self.nb_k_v_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.nb_k_v_slider.setTickInterval(10)
        self.nb_k_v_slider.valueChanged.connect(self._on_nb_k_v)
        kv.addWidget(self.nb_k_v_slider, 1)
        v.addWidget(self._nb_k_v_widget)

        st = QtWidgets.QHBoxLayout()
        st.addWidget(QtWidgets.QLabel("Blanked:"))
        self._lbl_blanked = QtWidgets.QLabel("0.00%")
        self._lbl_blanked.setStyleSheet(_MONO + " color: #8a3a00; min-width: 52px;")
        st.addWidget(self._lbl_blanked)
        st.addStretch(1)
        v.addLayout(st)
        return grp

    # ── Control handlers ──────────────────────────────────────────────────

    def _on_gain0(self, value: int):
        self._gain0 = float(value)
        self._gain0_label.setText(f"{value:.0f} dB")
        _settings().setValue("usrp_gain_db", self._gain0)
        self.engine.set_gain_db(self._gain0, 0)

    def _on_gain1(self, value: int):
        self._gain1 = float(value)
        self._gain1_label.setText(f"{value:.0f} dB")
        _settings().setValue("usrp_gain_db_ch1", self._gain1)
        self.engine.set_gain_db(self._gain1, 1)

    def _on_ant0(self, text: str):
        self._ant0 = text
        _settings().setValue("usrp_antenna", text)
        self.engine.set_antenna(text, 0)

    def _on_ant1(self, text: str):
        self._ant1 = text
        _settings().setValue("usrp_antenna_ch1", text)
        self.engine.set_antenna(text, 1)

    def _on_nb_backend(self, name: str):
        self._nb_name = name
        _settings().setValue("nb_backend", name)
        self.engine.set_blanker(name)

    def _on_nb_k(self, value: int):
        self._nb_k = value * 0.1
        self.nb_k_label.setText(f"{self._nb_k:.1f}")
        _settings().setValue("nb_factor", self._nb_k)
        self.engine.set_nb_factor(self._nb_k, "h")

    def _on_nb_k_v(self, value: int):
        self._nb_k_v = value * 0.1
        self.nb_k_v_label.setText(f"{self._nb_k_v:.1f}")
        _settings().setValue("nb_factor_v", self._nb_k_v)
        self.engine.set_nb_factor(self._nb_k_v, "v")

    def _on_dual_toggled(self, checked: bool):
        self.dual_audio_cb.setEnabled(checked)
        self._nb_k_v_widget.setVisible(checked)
        for w in (self._lbl_rx1, self._gain1_slider, self._gain1_label,
                  self._ant1_combo, self._lbl_rms1):
            w.setEnabled(checked)

    # ── Status refresh ────────────────────────────────────────────────────

    def _refresh_status(self):
        st = self.engine.status()
        self._lbl_serial.setText(st.serial)
        self._lbl_fw.setText(st.firmware)
        self._lbl_fpga.setText(st.fpga)
        self._lbl_phase.setText(st.startup_phase)
        if st.pan_mhz is not None:
            self._lbl_pan.setText(f"{st.pan_mhz:.6f} MHz")
        else:
            self._lbl_pan.setText("—")
        if st.hw_rate_hz is not None:
            self._lbl_rate.setText(f"{st.hw_rate_hz/1000:.0f} kHz")
        else:
            self._lbl_rate.setText("—")
        if st.running:
            self._lbl_bufs.setText(f"{st.buf_per_s:.0f} buf/s")
            self._lbl_rx0.setText("● Active")
            self._lbl_rx0.setStyleSheet(_MONO + " color: #66bb6a;")
            if st.dual:
                self._lbl_rx1.setText("● Active")
                self._lbl_rx1.setStyleSheet(_MONO + " color: #66bb6a;")
            else:
                self._lbl_rx1.setText("● Idle")
                self._lbl_rx1.setStyleSheet(_MONO + " color: #555;")
        else:
            self._lbl_bufs.setText("—")
            self._lbl_rx0.setText("● Idle")
            self._lbl_rx0.setStyleSheet(_MONO + " color: #555;")
            self._lbl_rx1.setText("● Idle")
            self._lbl_rx1.setStyleSheet(_MONO + " color: #555;")

        def _lvl(db):
            return f"{db:.1f} dBFS" if db is not None else "—"

        self._lbl_rms0.setText(_lvl(st.rms_h_dbfs))
        self._lbl_rms1.setText(_lvl(st.rms_v_dbfs))
        self._lbl_blanked.setText(f"{st.blanked_pct:.2f}%")

        # Reflect hardware gain readback without fighting the slider while dragged
        if st.running and st.gain_db is not None and not self._gain0_slider.isSliderDown():
            self._gain0_label.setText(f"{st.gain_db:.1f} dB")
        if (st.running and st.gain_db_ch1 is not None
                and not self._gain1_slider.isSliderDown()):
            self._gain1_label.setText(f"{st.gain_db_ch1:.1f} dB")

    # ── Channel list / config ─────────────────────────────────────────────

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
            cb = QtWidgets.QCheckBox(
                f"{ch.mode:<7}  {ch.dial_mhz:8.3f} MHz  ({ch.label})")
            cb.toggled.connect(self._update_plan_preview)
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
            gain_db=self._gain0,
            gain_db_ch1=self._gain1,
            antenna=self._ant0_combo.currentText(),
            antenna_ch1=self._ant1_combo.currentText(),
            blanker_name=self.nb_backend_combo.currentText(),
            nb_factor=self._nb_k,
            nb_factor_v=self._nb_k_v,
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
        sink_txt = "\n".join(f"  • {s}" for s in sinks) if sinks else "  (none)"
        self.status.setText(
            f"Running — pan {plan.pan_center_mhz:.6f} MHz @ {plan.hw_rate_hz/1000:.0f} kHz\n"
            f"blanker={cfg.blanker_name}  K={cfg.nb_factor:.1f}  "
            f"gain RF0={cfg.gain_db:.0f} dB\n"
            f"Audio sinks:\n{sink_txt}\n"
            f"Linux: WSJT-X Input = Monitor of …; "
            f"Windows: CABLE Output / Cable A·B Output"
        )
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_channel_controls_enabled(False)

    def _on_stop(self):
        try:
            self.engine.stop()
        except Exception as exc:
            log.warning("stop: %s", exc)
        self.status.setText("Stopped.")
        self.apply_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_channel_controls_enabled(True)
        self._update_plan_preview()

    def _set_channel_controls_enabled(self, enabled: bool):
        """Freeze band/channel/IQ routing while running; gain/blanker stay live."""
        self.band_combo.setEnabled(enabled)
        for _ch, cb in self._channel_checks:
            cb.setEnabled(enabled)
        for w in (self.map65_cb, self.map65_host, self.map65_port, self.map65_center,
                  self.qmap_cb, self.qmap_host, self.qmap_port, self.qmap_center,
                  self.dual_cb):
            w.setEnabled(enabled)
        self.dual_audio_cb.setEnabled(enabled and self.dual_cb.isChecked())

    def closeEvent(self, event):
        self._status_timer.stop()
        if self.engine.running:
            self.engine.stop()
        super().closeEvent(event)


def run_app():
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
