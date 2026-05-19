#!/usr/bin/env python3
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
"""map144 — MSK144 meteor scatter decoder for FlexRadio 6000 series.

map144 monitors the MSK144 calling frequency and 20 KHz either side.
If an MSK144 meteor-scatter ping occurs within 20 KHz of the calling
frequency map144 decodes it using the jt9 engine from WSJT-X.
Decoded contacts are logged to ``launches.jsonl`` and the audio surrounding
each detection is saved as a timestamped WAV file for offline review.

IQ samples are streamed from a radio with at least 48 kHz sample rate IQ data.
Radios include Flex, USRP, Airspy, and RTL-SDR.

A 48-channel polyphase channelizer resolves the band into 1 kHz sub-channels;
each sub-channel is monitored independently for the paired-tone signature
of an MSK144 burst. The assumption is that operators will choose 1 KHz increments
for their operation +/- tolerance.

A GUI (PyQt5) provides several options for viewing program status, including
a live spectrogram, detection heatmap, SNR history, and decode log.

SNR reporting
-------------
map144 reports two SNR estimates for each decoded ping:

``est_snr_db`` (MAP144's own estimate)
    Non-coherent energy detection: median PSD of the 500 ms pre-burst noise
    segment normalised to the WSJT-X 2500 Hz bandwidth convention.  This is
    computed before jt9 is called and is available even when jt9 cannot decode.

``jt9_snr_db`` (jt9's estimate, preferred)
    jt9's internal SNR from the 12 kHz WAV clip.  Used when available;
    falls back to ``est_snr_db`` when jt9 reports no value.

Both estimates read approximately **3–4 dB lower than WSJT-X** for the same
ping.  Calibration (``snr_calibrate.py``) confirmed this is expected and
correct: WSJT-X uses a coherent matched filter tuned to the exact MSK144
symbol structure, which has an inherent ~3–4 dB processing gain over
map144's non-coherent energy detector.  The IQ mix→decimate chain
contributes negligible loss (≤0.5 dB).  Neither the signal chain nor the
estimator algorithm is at fault — the offset is a fundamental consequence
of coherent vs. incoherent detection.

Usage
-----
::

    python map144.py [--bind-client-id UUID] [--log-level LEVEL]

Command-line arguments
----------------------
--bind-client-id UUID
    Client UUID string used for the FlexRadio ``client bind client_id=<uuid>``
    command, required when the radio firmware enforces client binding.  When
    omitted the client connects without binding.

--bind-client UUID
    Deprecated alias for ``--bind-client-id``; accepted for backwards
    compatibility.

--log-level LEVEL
    Verbosity for the root logger and the ``flexclient`` logger.
    Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL.  Default: INFO.

Bootstrap sequence
------------------
1. Parse arguments.
2. Configure logging (root logger + ``flexclient`` namespace).
3. Verify ``jt9`` (WSJT-X) is on ``PATH``.
4. Create ``QApplication``; set ``quitOnLastWindowClosed = True``.
5. Install ``SIGINT`` / ``SIGTERM`` handlers that close the main window via
   ``QTimer.singleShot(0, ...)`` so ``closeEvent`` runs (stops radio / sources
   cleanly).  Posting through the Qt event queue keeps shutdown on the main
   thread even though Python delivers signals asynchronously.
6. Instantiate and show ``MAP144Visualizer``.
7. Start a 500 ms Qt timer with a no-op slot.  Qt's C++ event loop blocks
   Python's GIL-based signal delivery; this timer forces the interpreter back
   into Python code periodically so that ``SIGINT`` (Ctrl-C) is noticed
   promptly rather than waiting for the next Qt event.
8. Keep a reference to the window on ``app._window`` to prevent premature
   garbage collection by the Python runtime.
9. Enter the Qt event loop via ``app.exec_()``.
"""

import argparse
import faulthandler
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

faulthandler.enable()   # print C-level stack trace to stderr on SIGSEGV

# Limit numpy/BLAS/OpenMP to 1 thread — must be set before numpy is imported.
# The processing loop is single-threaded; multi-threaded BLAS creates contention
# across concurrent calls (channelizer matmul vs waterfall FFT) that causes
# unpredictable 10-100x slowdowns under high signal levels.
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from map144_app.visualizer import MAP144Visualizer


def _configure_logging(level_name: str):
    level      = getattr(logging, level_name.upper(), logging.INFO)
    log_dir    = Path(__file__).parent / 'MSK144' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path   = log_dir / f"map144_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt        = '%(asctime)s %(levelname)s %(message)s'
    formatter  = logging.Formatter(fmt)

    # File handler — captures everything including print() via stdout redirect
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler — same output to terminal
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger('flexclient').setLevel(level)

    # Redirect stdout (print statements) to the log file as well.
    # Most diagnostic output in detection.py / processing.py uses print().
    sys.stdout = _TeeStream(sys.stdout, log_path)

    # Version banner first so every log file carries the version that
    # produced it — even when the user doesn't think to mention it.
    from map144_app import __version__ as _map144_version
    print(f"[map144] starting MAP144 v{_map144_version}", flush=True)
    print(f"[map144] log file: {log_path}", flush=True)


class _TeeStream:
    """Write to both the original stdout and a log file."""
    def __init__(self, original, log_path: Path):
        self._orig     = original
        self._log_file = open(log_path, 'a', buffering=1, encoding='utf-8')

    def write(self, data):
        self._orig.write(data)
        self._log_file.write(data)

    def flush(self):
        self._orig.flush()
        self._log_file.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return self._orig.isatty()


def main():
    """Launch the Radio IQ visualizer GUI."""
    parser = argparse.ArgumentParser(description='map144 MSK144 meteor scatter decoder')
    parser.add_argument('--bind-client-id', type=str, default=None,
                        help='GUI client UUID for `client bind client_id=<uuid>`')
    parser.add_argument('--bind-client', type=str, default=None,
                        help='Deprecated alias of --bind-client-id')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity (default: INFO)')
    args = parser.parse_args()

    _configure_logging(args.log_level)

    from map144_app.detection import find_jt9
    jt9_path = find_jt9()
    if jt9_path is None:
        print(
            "error: jt9 (WSJT-X) not found.\n"
            "  Install WSJT-X from https://wsjt.sourceforge.io/,\n"
            "  or set MAP144_JT9 to the full path of jt9 / jt9.exe\n"
            "  (e.g. MAP144_JT9='C:\\\\WSJT\\\\wsjtx\\\\bin\\\\jt9.exe').",
            file=sys.stderr,
        )
        sys.exit(1)
    logging.getLogger(__name__).info("Using jt9 binary: %s", jt9_path)

    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    def _graceful_shutdown(_signum, _frame):
        # Close the main window rather than calling app.quit() directly.
        # app.quit() exits the event loop without firing closeEvent, so the
        # source-stop logic in closeEvent (UHD stop, FlexDAX stop, etc.) is
        # skipped and UHD's C++ destructor calls std::terminate.
        # window.close() fires closeEvent → sources stop → event.accept() →
        # main window is destroyed → quitOnLastWindowClosed triggers app.quit.
        def _do_shutdown():
            w = getattr(app, '_window', None)
            if w is not None:
                w.close()
            else:
                app.quit()

        QtCore.QTimer.singleShot(0, _do_shutdown)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    window = MAP144Visualizer(bind_client_id=args.bind_client_id or args.bind_client)
    window.show()

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    app._window = window

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
