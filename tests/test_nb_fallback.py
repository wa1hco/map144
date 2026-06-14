# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise-blanker backend availability / fallback (regression).

A backend whose native library is missing (e.g. NR0V's libnob on Windows) must
fall back to Linrad in make() — never crash the radio loop at first chunk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map144_app import noise_blanker as nb


def test_base_is_available_default_true():
    assert nb.LinradBlanker.is_available() is True
    assert nb.BypassBlanker.is_available() is True


def test_make_falls_back_when_backend_unavailable():
    class _Unavail(nb.Blanker):
        name = "TestUnavail"

        @classmethod
        def is_available(cls):
            return False

        def process(self, engine, raw, raw_v, dual_pol):
            raise AssertionError("unavailable backend must never run")

    nb._REGISTRY["__test_unavail"] = _Unavail
    try:
        b = nb.make("__test_unavail")
        assert isinstance(b, nb.LinradBlanker)   # fell back, did not instantiate _Unavail
    finally:
        del nb._REGISTRY["__test_unavail"]


def test_make_uses_backend_when_available():
    b = nb.make("Bypass")            # available=True by default
    assert isinstance(b, nb.BypassBlanker)


if __name__ == "__main__":
    import traceback
    fns = [n for n in dir() if n.startswith("test_")]
    fails = 0
    for n in fns:
        try:
            globals()[n](); print(f"PASS {n}")
        except Exception:
            fails += 1; print(f"FAIL {n}"); traceback.print_exc()
    print(f"{len(fns)-fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
