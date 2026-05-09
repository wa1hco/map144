#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Tests for ``_PanelWindow`` View-menu sync.

The bug this guards against (reported 2026-05-10):

    "If I close a window with the X in the upper right, it goes away.
    But the View pulldown still shows the window view checked.  To get
    the window back I have to uncheck it and then check it again."

Root cause was that ``_PanelWindow.closeEvent`` *did* call
``setChecked(False)`` but the path was fragile under some window
managers — and for any path that hides the window WITHOUT going
through ``closeEvent`` (programmatic ``hide()``, WM events, etc.) the
menu would silently desync.

Fix: piggy-back menu sync onto Qt's ``hideEvent`` / ``showEvent``,
which fire on EVERY visibility transition regardless of cause.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_panel_window_menu_sync -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Headless Qt — required so the test runs in CI without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402


# Single QApplication for the whole module — Qt forbids re-creating it.
_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


class _StubParent:
    """Minimal stand-in for MAP144Visualizer — _PanelWindow only reads
    the ``_app_closing`` attribute."""

    _app_closing = False


def _make_panel():
    """Construct a _PanelWindow + view_action pair.  Imports
    ``_PanelWindow`` lazily so this test can opt out cleanly if
    ``map144_app.ui`` has any import-time side effect we don't want
    to drag in."""
    from map144_app.ui import _PanelWindow  # noqa: PLC0415

    parent = _StubParent()
    action = QtWidgets.QAction("test", _APP)
    action.setCheckable(True)
    action.setChecked(True)
    win = _PanelWindow("test panel", action, parent, geo_key=None)
    win.show()
    # Pump the Qt event loop briefly so showEvent fires.
    _APP.processEvents()
    return win, action, parent


class TestPanelWindowMenuSync(unittest.TestCase):
    """``_view_action.isChecked()`` must mirror window visibility on
    every transition, not just menu-driven ones."""

    def test_show_sets_checked(self):
        win, action, _ = _make_panel()
        self.assertTrue(win.isVisible())
        self.assertTrue(action.isChecked())
        win.close()

    def test_close_via_close_call_unchecks(self):
        """Programmatic .close() — same path as Qt processes the X
        button — must uncheck the action."""
        win, action, _ = _make_panel()
        win.close()
        _APP.processEvents()
        self.assertFalse(win.isVisible())
        self.assertFalse(
            action.isChecked(),
            "View-menu action should uncheck when window is X-closed",
        )

    def test_close_event_directly_unchecks(self):
        """Synthesise the close event the way the WM would — the X
        button on most platforms ultimately fires this event path."""
        win, action, _ = _make_panel()
        evt = QtGui.QCloseEvent()
        win.closeEvent(evt)
        _APP.processEvents()
        # Hide-on-close convention: window stays in widget tree but
        # not visible.
        self.assertFalse(win.isVisible())
        self.assertFalse(action.isChecked())

    def test_programmatic_hide_unchecks(self):
        """Even paths that don't go through closeEvent (e.g. menu
        toggle handler calls .hide() directly) must keep the menu in
        sync."""
        win, action, _ = _make_panel()
        win.hide()
        _APP.processEvents()
        self.assertFalse(win.isVisible())
        self.assertFalse(action.isChecked())

    def test_programmatic_show_checks(self):
        """And the symmetric path — re-showing a hidden window
        re-checks the menu."""
        win, action, _ = _make_panel()
        win.hide();   _APP.processEvents()
        action.setChecked(False)   # simulate menu reflecting prior hide
        win.show();   _APP.processEvents()
        self.assertTrue(win.isVisible())
        self.assertTrue(action.isChecked())

    def test_app_closing_path_does_not_re_check(self):
        """When ``_app_closing`` is True the close should accept
        without flapping the menu state — the window object is going
        away anyway."""
        win, action, parent = _make_panel()
        parent._app_closing = True
        evt = QtGui.QCloseEvent()
        win.closeEvent(evt)
        _APP.processEvents()
        # When app is closing we accept the event; menu state at this
        # point is irrelevant (whole UI is going down) but must not
        # raise.

    def test_geometry_save_failure_does_not_block_hide(self):
        """If QSettings.setValue raises (disk full, permission),
        the window must still hide and the menu must still uncheck —
        the geometry-save is observability, not correctness."""
        from map144_app.ui import _PanelWindow  # noqa: PLC0415
        action = QtWidgets.QAction("boom", _APP)
        action.setCheckable(True)
        action.setChecked(True)
        win = _PanelWindow("boom panel", action, _StubParent(), geo_key="boom_key")
        win.show(); _APP.processEvents()
        # Make _save_geometry blow up without breaking the hide path.
        win._save_geometry = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("synthetic settings-write failure"),
        )
        evt = QtGui.QCloseEvent()
        win.closeEvent(evt)
        _APP.processEvents()
        self.assertFalse(win.isVisible())
        self.assertFalse(action.isChecked())


if __name__ == "__main__":
    unittest.main(verbosity=2)
