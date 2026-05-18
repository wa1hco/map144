# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Tests for ``map144_app.detection.find_jt9``.

The helper must find jt9 across Linux, macOS, and Windows install
layouts, honour an explicit env-var override, and return ``None`` when
nothing matches so the application can show a friendly install hint.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from map144_app.detection import find_jt9


class TestFindJt9(unittest.TestCase):
    def setUp(self):
        # Always clear the env-var override at the start; tests that
        # want it set re-set it explicitly via mock.patch.dict.
        self._saved_env = os.environ.pop("MAP144_JT9", None)

    def tearDown(self):
        if self._saved_env is not None:
            os.environ["MAP144_JT9"] = self._saved_env

    def test_env_var_override_wins_when_file_exists(self):
        with mock.patch.dict(os.environ, {"MAP144_JT9": "/some/custom/jt9"}):
            with mock.patch("map144_app.detection.Path") as MockPath:
                MockPath.return_value.is_file.return_value = True
                self.assertEqual(find_jt9(), "/some/custom/jt9")

    def test_env_var_ignored_when_file_missing(self):
        # Override set, but the file doesn't exist → fall through to
        # PATH lookup, which we stub to return a found binary.
        with mock.patch.dict(os.environ, {"MAP144_JT9": "/nope/jt9"}):
            with mock.patch("map144_app.detection.Path") as MockPath:
                MockPath.return_value.is_file.return_value = False
                with mock.patch("map144_app.detection.shutil.which",
                                side_effect=["/usr/bin/jt9", None]):
                    self.assertEqual(find_jt9(), "/usr/bin/jt9")

    def test_path_lookup_via_shutil_which(self):
        with mock.patch("map144_app.detection.shutil.which",
                        side_effect=["/usr/local/bin/jt9", None]):
            with mock.patch("map144_app.detection.Path") as MockPath:
                MockPath.return_value.is_file.return_value = False
                self.assertEqual(find_jt9(), "/usr/local/bin/jt9")

    def test_path_lookup_finds_exe_on_windows(self):
        # shutil.which('jt9') returns None; shutil.which('jt9.exe') hits.
        with mock.patch("map144_app.detection.shutil.which",
                        side_effect=[None, r"C:\WSJT\wsjtx\bin\jt9.exe"]):
            with mock.patch("map144_app.detection.Path") as MockPath:
                MockPath.return_value.is_file.return_value = False
                self.assertEqual(find_jt9(), r"C:\WSJT\wsjtx\bin\jt9.exe")

    def test_windows_standard_install_locations_probed(self):
        # PATH lookup empty; the per-OS candidate list is the third tier.
        # We force sys.platform = 'win32' and report the operator-default
        # path as the only existing file.
        operator_default = r"C:\WSJT\wsjtx\bin\jt9.exe"
        def _isfile_only_operator_default(self_path):
            return str(self_path) == operator_default
        with mock.patch("map144_app.detection.sys.platform", "win32"):
            with mock.patch("map144_app.detection.shutil.which",
                            return_value=None):
                with mock.patch(
                    "map144_app.detection.Path.is_file",
                    autospec=True,
                    side_effect=_isfile_only_operator_default,
                ):
                    self.assertEqual(find_jt9(), operator_default)

    def test_linux_standard_install_locations_probed(self):
        # /usr/bin/jt9 is what apt installs; force that to be the only hit.
        target = "/usr/bin/jt9"
        def _isfile(self_path):
            return str(self_path) == target
        with mock.patch("map144_app.detection.sys.platform", "linux"):
            with mock.patch("map144_app.detection.shutil.which",
                            return_value=None):
                with mock.patch(
                    "map144_app.detection.Path.is_file",
                    autospec=True,
                    side_effect=_isfile,
                ):
                    self.assertEqual(find_jt9(), target)

    def test_returns_none_when_nothing_matches(self):
        with mock.patch("map144_app.detection.shutil.which",
                        return_value=None):
            with mock.patch(
                "map144_app.detection.Path.is_file",
                autospec=True,
                return_value=False,
            ):
                self.assertIsNone(find_jt9())


if __name__ == "__main__":
    unittest.main()
