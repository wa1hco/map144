#!/usr/bin/env python3
"""Viewer for MAP144 decode results from MSK144/detections/decodes.jsonl."""

import json
import sys
from pathlib import Path

# QApplication must exist before pyqtgraph imports any widgets
from PyQt5 import QtCore, QtWidgets
_app = QtWidgets.QApplication(sys.argv)

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

DECODES_FILE = Path(__file__).parent / "MSK144" / "detections" / "decodes.jsonl"


def load_decodes(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_audio_hz(jt9_line: str) -> int | None:
    """Extract audio offset Hz from jt9 output line: '000000  10  1.7 1547 & ...'"""
    parts = jt9_line.split()
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            pass
    return None


def decode_date(record: dict) -> str:
    """Return YYYY-MM-DD from record timestamp."""
    return record["timestamp"][:10]


def decode_hour(record: dict) -> float:
    """Return fractional UTC hour from record timestamp."""
    ts = record["timestamp"]  # YYYY-MM-DD_HH:MM:SS.f
    try:
        h, m, s = ts[11:13], ts[14:16], ts[17:]
        return int(h) + int(m) / 60 + float(s) / 3600
    except (ValueError, IndexError):
        return 0.0


class DecodesModel(QtCore.QAbstractTableModel):
    COLUMNS = ["Time (UTC)", "Message", "SNR (dB)", "Est SNR", "Audio Hz"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []

    def set_records(self, records: list[dict]):
        self.beginResetModel()
        self._rows = sorted(records, key=lambda r: r["timestamp"])
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        r = self._rows[index.row()]
        col = index.column()
        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return r["timestamp"][11:19]  # HH:MM:SS
            if col == 1:
                return r.get("message", "")
            if col == 2:
                return r.get("jt9_snr_db", "")
            if col == 3:
                return r.get("est_snr_db", "")
            if col == 4:
                hz = parse_audio_hz(r.get("jt9_line", ""))
                return hz if hz is not None else ""
        if role == QtCore.Qt.TextAlignmentRole:
            if col in (2, 3, 4):
                return QtCore.Qt.AlignCenter
        return None

    def record_at(self, row: int) -> dict | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, all_records: list[dict]):
        super().__init__()
        self.setWindowTitle("MAP144 Decode Viewer")
        self.resize(1100, 750)

        self._all = all_records
        self._dates = sorted({decode_date(r) for r in all_records}, reverse=True)

        # --- toolbar row ---
        toolbar = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(toolbar)
        hbox.setContentsMargins(6, 4, 6, 4)

        hbox.addWidget(QtWidgets.QLabel("Date:"))
        self._date_combo = QtWidgets.QComboBox()
        self._date_combo.addItems(self._dates)
        self._date_combo.currentIndexChanged.connect(self._refresh)
        hbox.addWidget(self._date_combo)

        hbox.addSpacing(20)
        self._stats_label = QtWidgets.QLabel()
        hbox.addWidget(self._stats_label)
        hbox.addStretch()

        # --- plot ---
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "UTC Hour")
        self._plot_widget.setLabel("left", "SNR (dB)")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMinimumHeight(280)
        self._scatter = pg.ScatterPlotItem(size=8, pen=None)
        self._plot_widget.addItem(self._scatter)

        # tooltip support
        self._scatter.sigClicked.connect(self._on_point_clicked)
        self._tooltip_text = pg.TextItem("", anchor=(0, 1), color=(180, 80, 0))
        self._plot_widget.addItem(self._tooltip_text)
        self._tooltip_text.hide()

        # --- table ---
        self._model = DecodesModel()
        self._proxy = QtCore.QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self._table = QtWidgets.QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.selectionModel().currentRowChanged.connect(self._on_table_row_changed)

        # --- layout ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self._plot_widget)
        splitter.addWidget(self._table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(toolbar)
        vbox.addWidget(splitter)
        self.setCentralWidget(central)

        self._current_records: list[dict] = []
        self._refresh()

    def _refresh(self):
        date = self._date_combo.currentText()
        records = [r for r in self._all if decode_date(r) == date]
        self._current_records = records
        self._model.set_records(records)
        self._table.sortByColumn(0, QtCore.Qt.AscendingOrder)

        # stats
        callsigns = set()
        for r in records:
            parts = r.get("message", "").split()
            callsigns.update(parts[:2])
        callsigns.discard("CQ")
        self._stats_label.setText(
            f"{len(records)} decodes   {len(callsigns)} callsigns"
        )

        # scatter
        if records:
            hours = np.array([decode_hour(r) for r in records])
            snrs = np.array([r.get("jt9_snr_db", 0) for r in records], dtype=float)
            # color by SNR: blue (low) → red (high)
            snr_min, snr_max = snrs.min(), snrs.max()
            span = max(snr_max - snr_min, 1)
            brushes = []
            for s in snrs:
                t = (s - snr_min) / span
                r_c = int(255 * t)
                b_c = int(255 * (1 - t))
                brushes.append(pg.mkBrush(r_c, 60, b_c, 210))
            spots = [
                {"pos": (h, s), "brush": b, "data": rec}
                for h, s, b, rec in zip(hours, snrs, brushes, records)
            ]
            self._scatter.setData(spots)
        else:
            self._scatter.setData([])

        self._tooltip_text.hide()

    def _on_point_clicked(self, _plot, points):
        if not points:
            return
        pt = points[0]
        rec = pt.data()
        if rec:
            x, y = pt.pos()
            label = f"{rec['timestamp'][11:19]}  {rec.get('message','')}  {y:+.0f} dB"
            self._tooltip_text.setText(label)
            self._tooltip_text.setPos(x, y)
            self._tooltip_text.show()

    def _on_table_row_changed(self, current, _previous):
        src_index = self._proxy.mapToSource(current)
        rec = self._model.record_at(src_index.row())
        if rec is None:
            return
        h = decode_hour(rec)
        snr = rec.get("jt9_snr_db", 0)
        label = f"{rec['timestamp'][11:19]}  {rec.get('message','')}  {snr:+d} dB"
        self._tooltip_text.setText(label)
        self._tooltip_text.setPos(h, snr)
        self._tooltip_text.show()


def main():
    if not DECODES_FILE.exists():
        print(f"No decodes file found at {DECODES_FILE}")
        sys.exit(1)

    records = load_decodes(DECODES_FILE)
    if not records:
        print("No records found.")
        sys.exit(1)

    win = MainWindow(records)
    win.show()
    sys.exit(_app.exec_())


if __name__ == "__main__":
    main()
