# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Tests for TFMFDetectorBlock (TODO R4, first slice).

Verifies the block mechanics — accumulate one period of wideband IQ, run the
matched-filter compute, emit one Event per candidate with §3.5 period-anchored
dual-clock stamps.  The TFMF compute itself is stubbed (its sensitivity is
covered by experiments/gpu_tfmf); these tests own the block wiring.
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

import numpy as np                                              # noqa: E402
import map144_app.processing as proc                            # noqa: E402
from map144_app.blocks.stream import Stream, make_event_stream  # noqa: E402
from map144_app.blocks.types import Record                      # noqa: E402
from map144_app.blocks.tfmf_detector_block import (             # noqa: E402
    TFMFDetectorBlock, TFMFDetectorBlockConfig,
)
from experiments.gpu_tfmf.tfmf import Candidate                 # noqa: E402

RATE = 48000
PERIOD_S = 0.1                       # tiny period so tests stay fast
PERIOD_N = int(PERIOD_S * RATE)      # 4800 samples


def _fake_candidate(time_sample=1200, time_s=0.025, freq_hz=1500.0):
    return Candidate(
        time_sample=time_sample, time_s=time_s, freq_hz=freq_hz,
        snr_db=17.0, raw_magnitude=0.3, sample_epoch_offset_samples=2.5,
    )


class TestTFMFDetectorBlock(unittest.TestCase):
    def setUp(self):
        self.blk = TFMFDetectorBlock("tfmf")
        self.blk.configure(TFMFDetectorBlockConfig(period_secs=PERIOD_S))
        self.inp = Stream("iq", capacity=64, kind=Stream.KIND_SAMPLES)
        self.out = make_event_stream("cand", capacity=64, durable=True)
        self.blk.inputs["iq"] = self.inp
        self.blk.outputs["candidates"] = self.out
        self._orig = proc._build_tfmf_display_surface

    def tearDown(self):
        proc._build_tfmf_display_surface = self._orig

    def _record(self, start_sample, n, wall):
        data = (np.random.default_rng(0).standard_normal(n)
                + 1j * np.random.default_rng(1).standard_normal(n)).astype(np.complex64)
        return Record(data=data, sample_rate_hz=RATE,
                      start_sample=start_sample, wall_clock_at_production=wall)

    def _drain(self):
        out = []
        try:
            while True:
                out.append(self.out.get(timeout=0.01))
        except Exception:
            pass
        return out

    def test_emits_one_event_per_candidate_with_anchored_stamps(self):
        cand = _fake_candidate(time_sample=1200, time_s=0.025)
        proc._build_tfmf_display_surface = lambda iq: (None, [cand])

        start_sample, wall0 = 10**9, 1_780_000_000.0
        # Three records of PERIOD_N/3 fill exactly one period (tests accumulation).
        third = PERIOD_N // 3
        self.inp.put(self._record(start_sample, third, wall0))
        self.inp.put(self._record(start_sample + third, third, wall0 + 0.01))
        self.inp.put(self._record(start_sample + 2 * third, PERIOD_N - 2 * third, wall0 + 0.02))
        for _ in range(3):
            self.blk.tick()

        events = self._drain()
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.kind, "tfmf_candidate")
        # occurred_at_sample = period_start_sample + candidate.time_sample.
        self.assertEqual(ev.occurred_at_sample, start_sample + 1200)
        # §3.5 period anchoring: signal_wall_clock = period_start_wall + time_s.
        self.assertAlmostEqual(ev.payload["signal_wall_clock"], wall0 + 0.025, places=6)
        self.assertAlmostEqual(ev.payload["freq_hz"], 1500.0, places=6)
        self.assertEqual(ev.sample_rate_hz, RATE)

    def test_no_emit_until_period_full(self):
        proc._build_tfmf_display_surface = lambda iq: (None, [_fake_candidate()])
        # Only half a period — must not fire.
        self.inp.put(self._record(0, PERIOD_N // 2, 1_780_000_000.0))
        self.blk.tick()
        self.assertEqual(len(self._drain()), 0)

    def test_multiple_candidates_multiple_events(self):
        cands = [_fake_candidate(time_sample=t, freq_hz=f)
                 for t, f in ((600, 1000.0), (2400, 1500.0), (4000, 2000.0))]
        proc._build_tfmf_display_surface = lambda iq: (None, cands)
        self.inp.put(self._record(0, PERIOD_N, 1_780_000_000.0))
        self.blk.tick()
        events = self._drain()
        self.assertEqual(len(events), 3)
        self.assertEqual({round(e.payload["freq_hz"]) for e in events},
                         {1000, 1500, 2000})

    def test_compute_error_does_not_crash_block(self):
        def _boom(iq):
            raise RuntimeError("gpu kaboom")
        proc._build_tfmf_display_surface = _boom
        self.inp.put(self._record(0, PERIOD_N, 1_780_000_000.0))
        self.blk.tick()                       # must not raise
        self.assertEqual(len(self._drain()), 0)


if __name__ == "__main__":
    unittest.main()
