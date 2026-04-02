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
3. If --headless: instantiate Engine, set source_mode, call run_headless().
4. Create ``QApplication``; set ``quitOnLastWindowClosed = True``.
5. Install ``SIGINT`` / ``SIGTERM`` handlers that call ``app.quit()`` via
   ``QTimer.singleShot(0, ...)`` — posting the quit request through the Qt
   event queue ensures it fires safely from the main thread even though Python
   delivers signals asynchronously.
6. Instantiate and show ``MAP144Visualizer``.
7. Start a 500 ms Qt timer with a no-op slot.  Qt's C++ event loop blocks
   Python's GIL-based signal delivery; this timer forces the interpreter back
   into Python code periodically so that ``SIGINT`` (Ctrl-C) is noticed
   promptly rather than waiting for the next Qt event.
8. Keep a reference to the window on ``app._window`` to prevent premature
   garbage collection by the Python runtime.
9. Enter the Qt event loop via ``app.exec_()``.
"""

import logging
import signal
import sys

from map144gui.visualizer import MAP144Visualizer


def _configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')
    else:
        root_logger.setLevel(level)

    logging.getLogger('flexclient').setLevel(level)


def main():
    """Launch the Radio IQ visualizer GUI."""
    import argparse

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

    import shutil
    if shutil.which('jt9') is None:
        print("error: jt9 not found on PATH", file=sys.stderr)
        sys.exit(1)

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
        QtCore.QTimer.singleShot(0, app._window.close)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    window = MAP144Visualizer(
        bind_client_id=args.bind_client_id or args.bind_client,
    )
    window.show()

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    app._window = window

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
