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
"""Screenshot capture window — save any combination of MAP144 windows to PNG.

Accessible via View → Screenshots.  Each window has a checkbox; pressing
"Capture Selected" or "Capture All" writes PNG files to docs/.

Filename mode (radio buttons):
  Standard name  — overwrites the canonical name used by the slide deck,
                   e.g. map144_main.png
  Add suffix     — appends a user-supplied tag before .png so the standard
                   file is not touched, e.g. map144_main_contest_run.png

Only visible (shown) windows can be captured.  Hidden windows are skipped
with a status note.
"""

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .visualizer import _SETTINGS


# Output directory relative to the project root (parent of map144_app/).
_DOCS_DIR = Path(__file__).parent.parent / "docs"


def setup_screenshot_window(self, view_action):
    """Build the Screenshot panel window and attach it to *self* (MAP144Visualizer)."""
    from .ui import _PanelWindow

    win = _PanelWindow("map144 — Screenshots", view_action, self, 'screenshot_geometry')
    win.setMinimumSize(340, 500)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    self._screenshot_win = win

    # ── Title label ──────────────────────────────────────────────────────────
    title = QtWidgets.QLabel("Capture windows to PNG files in docs/")
    title.setStyleSheet("QLabel { font-weight: bold; font-size: 10pt; }")
    layout.addWidget(title)

    out_lbl = QtWidgets.QLabel(f"Output: {_DOCS_DIR}")
    out_lbl.setStyleSheet("QLabel { font-size: 8pt; color: #555555; }")
    out_lbl.setWordWrap(True)
    layout.addWidget(out_lbl)

    layout.addSpacing(4)

    # ── Checkboxes — one per capturable window ────────────────────────────────
    # (attr_name, label, filename_stem)
    _WINDOWS = [
        ('_main_win_self',    "Controls and Decodes",  "map144_main"),
        ('_fast_graph_win',   "Fast Graph",            "map144_fast_graph"),
        ('_detect_win',       "Tone Detection SNR",    "map144_detection_heatmap"),
        ('_iq_nb_win',        "Noise Blanker",         "map144_noise_blanker"),
        ('_reporting_win',    "Reporting",             "map144_reporting"),
        ('_flex_win',         "Flex Radio",            "map144_flex_radio"),
        ('_usrp_win',         "USRP B210",             "map144_usrp_b210"),
        ('_airspy_win',       "Airspy HF+",            "map144_airspy"),
        ('_rtlsdr_win',       "RTL-SDR",               "map144_rtlsdr"),
    ]
    self._screenshot_win._window_entries = _WINDOWS

    checks_grp = QtWidgets.QGroupBox("Windows to capture")
    checks_layout = QtWidgets.QVBoxLayout(checks_grp)
    checks_layout.setSpacing(3)

    self._screenshot_checkboxes = {}
    for attr, label, stem in _WINDOWS:
        cb = QtWidgets.QCheckBox(label)
        saved = _SETTINGS.value(f'screenshot_cb_{attr}', True, type=bool)
        cb.setChecked(saved)
        checks_layout.addWidget(cb)
        self._screenshot_checkboxes[attr] = cb

    layout.addWidget(checks_grp)

    # ── Select all / none ─────────────────────────────────────────────────────
    sel_row = QtWidgets.QHBoxLayout()
    sel_all_btn  = QtWidgets.QPushButton("Select All")
    sel_none_btn = QtWidgets.QPushButton("Select None")
    sel_all_btn.clicked.connect(
        lambda: [cb.setChecked(True)  for cb in self._screenshot_checkboxes.values()]
    )
    sel_none_btn.clicked.connect(
        lambda: [cb.setChecked(False) for cb in self._screenshot_checkboxes.values()]
    )
    sel_row.addWidget(sel_all_btn)
    sel_row.addWidget(sel_none_btn)
    layout.addLayout(sel_row)

    layout.addSpacing(4)

    # ── Filename mode ─────────────────────────────────────────────────────────
    mode_grp = QtWidgets.QGroupBox("Filename")
    mode_layout = QtWidgets.QVBoxLayout(mode_grp)
    mode_layout.setSpacing(4)

    rb_standard = QtWidgets.QRadioButton("Standard name  (overwrites presentation file)")
    rb_suffix   = QtWidgets.QRadioButton("Add suffix:")
    suffix_edit = QtWidgets.QLineEdit()
    suffix_edit.setPlaceholderText("e.g.  contest_run  or  no_signal")
    suffix_edit.setEnabled(False)

    saved_mode = _SETTINGS.value('screenshot_mode', 'standard', type=str)
    if saved_mode == 'suffix':
        rb_suffix.setChecked(True)
        suffix_edit.setEnabled(True)
    else:
        rb_standard.setChecked(True)

    saved_suffix = _SETTINGS.value('screenshot_suffix', '', type=str)
    suffix_edit.setText(saved_suffix)

    rb_standard.toggled.connect(lambda checked: suffix_edit.setEnabled(not checked))

    suffix_row = QtWidgets.QHBoxLayout()
    suffix_row.addWidget(rb_suffix)
    suffix_row.addWidget(suffix_edit, 1)

    mode_layout.addWidget(rb_standard)
    mode_layout.addLayout(suffix_row)
    layout.addWidget(mode_grp)

    self._screenshot_rb_standard = rb_standard
    self._screenshot_suffix_edit = suffix_edit

    # ── Capture buttons ───────────────────────────────────────────────────────
    btn_row = QtWidgets.QHBoxLayout()

    all_btn = QtWidgets.QPushButton("Capture All")
    all_btn.setToolTip("Capture all visible windows regardless of checkboxes")
    sel_btn = QtWidgets.QPushButton("Capture Selected")
    sel_btn.setToolTip("Capture only the checked windows above")

    btn_row.addWidget(all_btn)
    btn_row.addWidget(sel_btn)
    layout.addLayout(btn_row)

    # ── Status log ────────────────────────────────────────────────────────────
    self._screenshot_status = QtWidgets.QPlainTextEdit()
    self._screenshot_status.setReadOnly(True)
    self._screenshot_status.setMaximumHeight(120)
    self._screenshot_status.setStyleSheet(
        "QPlainTextEdit { font-family: monospace; font-size: 8pt; "
        "background: #f8f8f8; color: #222222; }"
    )
    layout.addWidget(self._screenshot_status)
    layout.addStretch()

    # ── Wire buttons ──────────────────────────────────────────────────────────
    all_btn.clicked.connect(lambda: _do_capture(self, capture_all=True))
    sel_btn.clicked.connect(lambda: _do_capture(self, capture_all=False))

    # ── Persist state on close ────────────────────────────────────────────────
    _orig_close = win.closeEvent

    def _save_and_close(event, _orig=_orig_close):
        for _attr, _cb in self._screenshot_checkboxes.items():
            _SETTINGS.setValue(f'screenshot_cb_{_attr}', _cb.isChecked())
        _SETTINGS.setValue('screenshot_mode',
                           'standard' if self._screenshot_rb_standard.isChecked() else 'suffix')
        _SETTINGS.setValue('screenshot_suffix', self._screenshot_suffix_edit.text().strip())
        _orig(event)

    win.closeEvent = _save_and_close


def _do_capture(self, capture_all: bool):
    """Capture selected (or all) windows and write PNG files to docs/."""
    _WINDOWS = self._screenshot_win._window_entries
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)

    use_standard = self._screenshot_rb_standard.isChecked()
    suffix = self._screenshot_suffix_edit.text().strip()
    # Normalise suffix: strip leading underscores/spaces, prepend one underscore
    if not use_standard and suffix:
        suffix = '_' + suffix.lstrip('_ ')
    else:
        suffix = ''

    status_lines = []
    captured = 0

    for attr, label, stem in _WINDOWS:
        # Resolve widget: main window is self itself
        if attr == '_main_win_self':
            widget = self
        else:
            widget = getattr(self, attr, None)

        # Skip if checkbox unchecked (unless capture_all)
        cb = self._screenshot_checkboxes.get(attr)
        if not capture_all and cb is not None and not cb.isChecked():
            continue

        if widget is None:
            status_lines.append(f"  skip  {label}  (not available)")
            continue

        if not widget.isVisible():
            status_lines.append(f"  skip  {label}  (window hidden)")
            continue

        # Grab and save
        filename = f"{stem}{suffix}.png"
        out_path = _DOCS_DIR / filename
        pixmap = widget.grab()
        if pixmap.save(str(out_path), "PNG"):
            status_lines.append(f"  OK    {filename}")
            captured += 1
            print(f"[screenshot] saved {out_path}", flush=True)
        else:
            status_lines.append(f"  FAIL  {filename}  (save error)")
            print(f"[screenshot] failed to save {out_path}", flush=True)

    summary = f"Captured {captured} window(s) → docs/"
    status_lines.insert(0, summary)
    self._screenshot_status.setPlainText("\n".join(status_lines))
    print(f"[screenshot] {summary}", flush=True)
