# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Tests for map144_app.timebase.TimeBase — the dual-clock owner.

Covers the block-stream-design §3.5 contract: intra-period sample precision,
fresh-per-period UTC, no cross-period drift accumulation, and the headline
acceptance test — the 2026-05-31 "1970 timestamp" bug cannot recur because
every source open re-anchors (so a WAV-relative anchor can't leak into a live
run).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.timebase import TimeBase, TimeBaseError   # noqa: E402

RATE = 48000.0
# A representative live-radio anchor: 2026-05-31 12:00:00 UTC ≈ 1.78e9 s.
LIVE_WALL = 1_780_000_000.0


class TestAnchoringAndUtc(unittest.TestCase):
    def test_reset_required_before_utc(self):
        tb = TimeBase(RATE)
        with self.assertRaises(TimeBaseError):
            tb.utc_at(0)

    def test_reset_anchors_utc(self):
        tb = TimeBase(RATE)
        tb.reset(sample=1000, wall=LIVE_WALL)
        # The anchor sample maps to the anchor wall exactly.
        self.assertAlmostEqual(tb.utc_at(1000), LIVE_WALL, places=6)
        # One second of samples later = one second of UTC later.
        self.assertAlmostEqual(tb.utc_at(1000 + int(RATE)), LIVE_WALL + 1.0, places=6)

    def test_intra_period_sample_precision(self):
        tb = TimeBase(RATE)
        tb.reset(sample=0, wall=LIVE_WALL)
        # 0.963 s into the period (a real t_s_in_period value from the logs).
        n = int(round(0.963 * RATE))
        self.assertAlmostEqual(tb.utc_at(n), LIVE_WALL + 0.963, places=6)

    def test_period_index_is_wall_only(self):
        tb = TimeBase(RATE, period_secs=15.0)
        # 12:00:07 and 12:00:14 are the same 15-s period; 12:00:15 is the next.
        base = 1_780_000_000.0  # divisible-ish; use explicit grid below
        self.assertEqual(tb.period_index(15.0 * 100 + 0.0), 100)
        self.assertEqual(tb.period_index(15.0 * 100 + 14.999), 100)
        self.assertEqual(tb.period_index(15.0 * 101 + 0.0), 101)


class TestPeriodAnchoring(unittest.TestCase):
    def test_should_mark_period_on_boundary_cross(self):
        tb = TimeBase(RATE, period_secs=15.0)
        tb.reset(sample=0, wall=15.0 * 100 + 1.0)   # mid-period 100
        self.assertFalse(tb.should_mark_period(15.0 * 100 + 9.0))   # same period
        self.assertTrue(tb.should_mark_period(15.0 * 101 + 0.1))    # next period

    def test_re_pair_keeps_stamps_fresh_no_cross_period_drift(self):
        # Simulate a sample clock running 100 ppm FAST: after N seconds of
        # wall time, the sample counter has advanced by N*(1+100e-6)*RATE.
        # Without per-period re-pair, sample-derived UTC would drift; with it,
        # each period's stamp is anchored to a fresh wall read, so the error is
        # bounded to one period's worth (≈15 µs), not the whole run.
        ppm = 100e-6
        tb = TimeBase(RATE, period_secs=15.0)
        wall0 = 15.0 * 1000
        tb.reset(sample=0, wall=wall0)
        max_err = 0.0
        for p in range(1, 40):           # 40 periods = 10 minutes
            wall = wall0 + p * 15.0      # true wall at this period boundary
            sample = int(p * 15.0 * (1 + ppm) * RATE)   # drifted sample counter
            tb.mark_period_start(sample=sample, wall=wall)
            # An event 5 s into the period, at the (drifted) sample rate.
            ev_sample = sample + int(5.0 * (1 + ppm) * RATE)
            err = abs(tb.utc_at(ev_sample) - (wall + 5.0))
            max_err = max(max_err, err)
        # Bounded to ~one period of ppm drift, NOT growing with run length.
        self.assertLess(max_err, 0.002, f"cross-period drift leaked: {max_err*1e3:.3f} ms")

    def test_current_period_index_tracks_marks(self):
        tb = TimeBase(RATE, period_secs=15.0)
        tb.reset(sample=0, wall=15.0 * 200 + 3.0)
        self.assertEqual(tb.current_period_index(), 200)
        tb.mark_period_start(sample=int(15.0 * RATE), wall=15.0 * 201 + 0.0)
        self.assertEqual(tb.current_period_index(), 201)


class Test1970Regression(unittest.TestCase):
    """The headline acceptance test for R1: a WAV-relative anchor must not be
    able to leak into a subsequent live run.  The structural fix is that every
    source open calls reset(), which overwrites any stale anchor."""

    def test_wav_anchor_then_live_open_re_anchors(self):
        tb = TimeBase(RATE, period_secs=15.0)

        # 1) WAV replay anchors near epoch (relative timeline, the legacy ≈0).
        tb.reset(sample=0, wall=0.5)
        wav_utc = tb.utc_at(int(7.0 * RATE))         # 7 s into the WAV
        self.assertLess(wav_utc, 1_000_000, "WAV stamp should be near-epoch")  # ~1970

        # 2) Operator switches to live radio.  Source open re-anchors with a
        #    fresh time.time().  THIS is the step the legacy _start_radio_source
        #    never did — and the whole bug.
        tb.reset(sample=10**9, wall=LIVE_WALL)        # live first-packet anchor
        live_utc = tb.utc_at(10**9 + int(7.0 * RATE))
        self.assertGreater(live_utc, 1_700_000_000, "live stamp must be 2026, not 1970")
        self.assertAlmostEqual(live_utc, LIVE_WALL + 7.0, places=6)

    def test_no_derivation_from_session_start_over_long_run(self):
        # Even without any re-pair, a single anchor must not drift the *period*
        # stamp catastrophically the way `period_idx*15` from a relative
        # _loop_wall did.  Here the per-period re-pair keeps UTC sane after
        # an 11-hour run (the bug's last 1970 stamp was ~11 h since start).
        tb = TimeBase(RATE, period_secs=15.0)
        tb.reset(sample=0, wall=LIVE_WALL)
        eleven_h_samples = int(11 * 3600 * RATE)
        wall_11h = LIVE_WALL + 11 * 3600
        tb.mark_period_start(sample=eleven_h_samples, wall=wall_11h)
        self.assertGreater(tb.utc_at(eleven_h_samples), 1_700_000_000)


class TestPolicyAndValidation(unittest.TestCase):
    def test_wav_vs_radio_anchor_policy(self):
        tb = TimeBase(RATE)
        # WAV: explicit recording-time policy (not ≈0).
        rec_utc = 1_777_000_000.0
        tb.reset(sample=0, wall=rec_utc)
        self.assertAlmostEqual(tb.utc_at(0), rec_utc, places=6)
        # Radio: live policy.
        tb.reset(sample=0, wall=LIVE_WALL)
        self.assertAlmostEqual(tb.utc_at(0), LIVE_WALL, places=6)

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            TimeBase(0)
        with self.assertRaises(ValueError):
            TimeBase(RATE, period_secs=0)


if __name__ == "__main__":
    unittest.main()
