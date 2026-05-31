# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""R1 acceptance test — TimeBase wired into process_iq_data.

Reproduces the 2026-05-31 bug's *shape* through the real engine: run WAV-mode
processing (which anchors the time base), then switch to a live source, and
assert the live phase re-anchors to wall-clock (2026, not the WAV-relative ≈0
that produced 1970 TFMF timestamps).  The structural guarantee is that every
source open arms ``_timebase_anchor_pending``.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np                                          # noqa: E402
from map144_app.engine import Engine                        # noqa: E402
from map144_app.runtime import (                            # noqa: E402
    _reset_wav_timeline,
    _clear_decode_panel_on_source_start,
)

RATE = 48000
YEAR_2020_UTC = 1_577_836_800.0   # 2020-01-01; anything past this is "real time"


def _feed(eng: Engine, mode: str, seconds: float, ts_int: int = 0) -> None:
    """Push `seconds` of weak complex noise through process_iq_data in `mode`."""
    eng.source_mode = mode
    chunk = eng.fft_size * 4
    n = int(seconds * RATE)
    rng = np.random.default_rng(1)
    pos = 0
    while pos < n:
        m = min(chunk, n - pos)
        c = ((rng.standard_normal(m) + 1j * rng.standard_normal(m))
             .astype(np.complex64) * 0.01)
        eng.process_iq_data(c, ts_int, 0)
        pos += chunk


class TestTimeBaseWiring(unittest.TestCase):
    def setUp(self):
        self.eng = Engine(center_freq_mhz=50.260, calling_freq_mhz=50.260,
                          sample_rate=RATE, fft_size=2048)

    def test_wav_then_live_re_anchors_to_wall_clock(self):
        # --- WAV phase: _reset_wav_timeline arms the re-anchor; first chunk
        #     anchors the time base to time.time() (replay-now). ---
        _reset_wav_timeline(self.eng)
        self.assertTrue(self.eng._timebase_anchor_pending)
        _feed(self.eng, "wav", seconds=1.0)
        self.assertTrue(self.eng.timebase.is_anchored())
        wav_utc = self.eng.timebase.utc_at(self.eng._iq_abs_sample)
        self.assertGreater(wav_utc, YEAR_2020_UTC,
                           "WAV phase must anchor to time.time(), not ≈0 (1970)")

        # --- Switch to a live source.  The source-open hook MUST re-arm the
        #     anchor so the WAV anchor cannot leak into the live run. ---
        _clear_decode_panel_on_source_start(self.eng)
        self.assertTrue(self.eng._timebase_anchor_pending,
                        "live source open must re-arm _timebase_anchor_pending")
        _feed(self.eng, "radio", seconds=1.0)
        live_utc = self.eng.timebase.utc_at(self.eng._iq_abs_sample)
        self.assertGreater(live_utc, YEAR_2020_UTC,
                           "live phase must be wall-clock (2026), never 1970")

    def test_every_source_open_arms_reanchor(self):
        # Both source-open paths must arm the flag — that's the leak fix.
        self.eng._timebase_anchor_pending = False
        _reset_wav_timeline(self.eng)
        self.assertTrue(self.eng._timebase_anchor_pending)

        self.eng._timebase_anchor_pending = False
        _clear_decode_panel_on_source_start(self.eng)
        self.assertTrue(self.eng._timebase_anchor_pending)

    def test_live_marks_periods(self):
        # After enough live processing the time base should have marked at least
        # the initial period (utc_at usable) and produce sane period indices.
        _clear_decode_panel_on_source_start(self.eng)
        _feed(self.eng, "radio", seconds=1.0)
        idx = self.eng.timebase.current_period_index()
        # 2026 / 15 s ≈ 1.18e8 — a large, positive, sane period index.
        self.assertGreater(idx, 100_000_000)


if __name__ == "__main__":
    unittest.main()
