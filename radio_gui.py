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
"""Radio IQ GUI launcher.

This is the top-level entry point for the radio IQ visualiser.  It is a thin
bootstrap script whose only jobs are argument parsing, logging configuration,
Qt application lifecycle management, and OS-signal handling.  All substantive
functionality lives in the ``radio_iq_gui`` package.

Usage
-----
::

    python radio_gui.py [--rate RATE] [--bind-client-id UUID] [--log-level LEVEL]

Command-line arguments
----------------------
--rate RATE
    IQ sample rate in Hz passed to ``RadioIQVisualizer`` and forwarded to the
    radio client.  Must match the rate configured on the radio source.
    Default: 48000.

--bind-client-id UUID
    Client UUID string used for the FlexRadio ``client bind client_id=<uuid>``
    command, required when the radio firmware enforces client binding.  When
    omitted the client connects without binding.

--bind-client UUID
    Deprecated alias for ``--bind-client-id``; accepted for backwards
    compatibility and merged before constructing ``RadioIQVisualizer``.

--log-level LEVEL
    Verbosity for the root logger and the ``flexclient`` logger.
    Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL.  Default: INFO.
    Applies ``logging.basicConfig`` if no handlers are already installed;
    otherwise adjusts the root handler level in-place.

Bootstrap sequence
------------------
1. Parse arguments.
2. Configure logging (root logger + ``flexclient`` namespace).
3. Create ``QApplication``; set ``quitOnLastWindowClosed = True``.
4. Install ``SIGINT`` / ``SIGTERM`` handlers that call ``app.quit()`` via
   ``QTimer.singleShot(0, ...)`` — posting the quit request through the Qt
   event queue ensures it fires safely from the main thread even though Python
   delivers signals asynchronously.
5. Instantiate and show ``RadioIQVisualizer``.
6. Start a 500 ms Qt timer with a no-op slot.  Qt's C++ event loop blocks
   Python's GIL-based signal delivery; this timer forces the interpreter back
   into Python code periodically so that ``SIGINT`` (Ctrl-C) is noticed
   promptly rather than waiting for the next Qt event.
7. Keep a reference to the window on ``app._window`` to prevent premature
   garbage collection by the Python runtime.
8. Enter the Qt event loop via ``app.exec_()``.
"""

import logging
import signal
import sys

from PyQt5 import QtCore, QtWidgets

from radio_iq_gui.visualizer import RadioIQVisualizer


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
    import shutil
    if shutil.which('jt9') is None:
        print("error: jt9 not found on PATH", file=sys.stderr)
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    def _graceful_shutdown(_signum, _frame):
        QtCore.QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    import argparse

    parser = argparse.ArgumentParser(description='Radio IQ Visualizer')
    parser.add_argument('--rate', type=int, default=48000,
                        help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--bind-client-id', type=str, default=None,
                        help='GUI client UUID for `client bind client_id=<uuid>`')
    parser.add_argument('--bind-client', type=str, default=None,
                        help='Deprecated alias of --bind-client-id')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity (default: INFO)')
    args = parser.parse_args()

    _configure_logging(args.log_level)

    window = RadioIQVisualizer(
        sample_rate=args.rate,
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
