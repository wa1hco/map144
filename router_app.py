#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""B210 multi-mode audio / IQ router — standalone entry point.

Owns the B210 while running (do not start MAP144 against the same radio).

Usage:
    ./run-router.sh
    python3 router_app.py          # re-execs into .venv when present
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _reexec_into_venv() -> None:
    """If .venv exists and we are not already in it, re-exec with that Python.

    Bare ``python router_app.py`` often hits the system interpreter, which
    lacks the project's numba/uhd/PyQt stack.  Same idea as ``run.sh``.
    """
    root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        venv_py = root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = root / ".venv" / "bin" / "python"
    if not venv_py.is_file():
        return
    try:
        if Path(sys.executable).resolve() == venv_py.resolve():
            return
    except OSError:
        return
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


def main(argv=None):
    _reexec_into_venv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from map144_app.router.gui import run_app
    return run_app()


if __name__ == "__main__":
    sys.exit(main() or 0)
