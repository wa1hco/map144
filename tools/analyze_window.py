#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Standalone launcher for the improved Qt AnalysisWindow.

The AnalysisWindow lives in ``map144_app.analysis_window`` and is normally
opened from inside the main map144 app (View → Open Analysis Window…), or
from a right-click in the main-window decode list.  This launcher lets you
open it directly on a WAV without booting the full app — useful for
iterating on UI changes or replaying a single capture.

Usage:
    python tools/analyze_window.py /path/to/capture.wav
    python tools/analyze_window.py                # opens file picker

Run from the project root so the ``map144_app`` package resolves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``map144_app`` importable when this script is launched directly.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from PyQt5 import QtCore, QtWidgets   # noqa: E402

from map144_app.analysis_window import AnalysisWindow   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wav', nargs='?', type=Path, default=None,
                    help='Capture WAV to open.  If omitted, opens a file picker '
                         'rooted at MSK144/detections/.')
    args = ap.parse_args()

    # Qt 5.15 quirk: high-DPI scaling needs to be set before QApplication.
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    if args.wav is None:
        # No CLI arg — show a file picker at MSK144/detections/.
        start_dir = str(REPO / 'MSK144' / 'detections')
        wav_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open MSK144 capture WAV", start_dir, "WAV files (*.wav)")
        if not wav_str:
            return 0
        wav_path = Path(wav_str)
    else:
        wav_path = args.wav.resolve()
        if not wav_path.exists():
            print(f"WAV not found: {wav_path}", file=sys.stderr)
            return 2

    win = AnalysisWindow(wav_path, reporter=None, parent=None)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
