#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""TODO #41c — verify that pct25 history is NOT updated when the
noise blanker fires.

Bug context (``project_post_burst_rebound`` memory):
A wideband-burst end produces a 100× baseline trigger spike in the
first 100 ms post-burst, decaying to 3–10× over seconds.  All zero
decodes.  Asymmetric (pre-burst BELOW baseline; post-burst above);
daytime-only.  Mechanism: the noise blanker zeroes the burst samples
→ the squared-FFT measures low power → the rolling 25-percentile
``_pct25`` baseline ratchets DOWN.  When the burst ends, normal noise
now reads as elevated against the depressed baseline, triggering
false detections for the next ~30 s of operation.

Fix: when the blanker fires on a chunk, skip the
``_metric_hist_buf`` history APPEND for hops derived from that chunk.
The pct25 recompute still runs on the existing history so detection
thresholding stays current; only the *update* is gated.  pct25 stays
at its pre-burst value; post-burst normal noise reads as normal.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_pct25_blanker_gate -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Engine import — no Qt subclass — fine for headless test.
from map144_app.engine import Engine     # noqa: E402
from map144_app.noise_blanker import (   # noqa: E402
    BypassBlanker, BlankerResult,
)


class _FakeBlanker(BypassBlanker):
    """Bypass blanker that lets us force a non-empty ``blank_mask`` —
    so we can verify the pct25 gate fires on the engine's
    ``_skip_pct25_update_h`` / ``_v`` flags without having to
    synthesise an impulse strong enough to trip LinradBlanker."""

    def __init__(self, force_blanking: bool = False):
        self.force_blanking = force_blanking

    def process(self, engine, raw, raw_v, dual_pol):
        result = super().process(engine, raw, raw_v, dual_pol)
        if self.force_blanking:
            # Mark a single sample as blanked.  The pct25 gate is
            # boolean (any-sample-blanked → skip-the-chunk's-history-
            # update), so one is plenty.
            result.blank_mask[0] = 1.0
            if result.blank_mask_v is not None:
                result.blank_mask_v[0] = 1.0
        return result


def _build_iq_chunk(n: int, seed: int = 0) -> np.ndarray:
    """Synthesise n samples of weak complex noise (low std).  The
    actual amplitude doesn't matter for this test — we're verifying
    the gate logic, not the DSP outputs."""
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(n).astype(np.float32) * 0.001
    im = rng.standard_normal(n).astype(np.float32) * 0.001
    return (re + 1j * im).astype(np.complex64)


class TestPct25BlankerGate(unittest.TestCase):

    def setUp(self):
        # Fresh Engine per test — no shared state.  ``Engine()`` works
        # without Qt because process_iq_data is on the base class.
        self.engine = Engine()
        # Use the controllable fake blanker — no Qt, no actual NB DSP.
        self.engine.blanker = _FakeBlanker(force_blanking=False)
        # Warm up _ch_buf so detection hops fire on the next chunk.
        # _ch_buf needs CH_DETECT_SIZE channelized samples before the
        # detect loop runs once.  CH_DETECT_SIZE = 512 channelized
        # samples = 512 × 4 = 2048 input samples at 48 kHz.  Bias
        # high to guarantee at least one hop per chunk.
        self._chunk_n = 4096

    def _push(self) -> None:
        """One process_iq_data call with weak noise IQ."""
        iq = _build_iq_chunk(self._chunk_n)
        self.engine.process_iq_data(iq, 0, 0)

    def test_clean_chunks_grow_history(self):
        """Sanity: with no blanker activity, the squared-FFT pct25
        history grows on every detection hop."""
        # First push warms up the channelizer; subsequent pushes
        # trigger actual detection hops.
        cnt_before = self.engine._metric_hist_cnt
        for _ in range(10):
            self._push()
        cnt_after = self.engine._metric_hist_cnt
        self.assertGreater(
            cnt_after, cnt_before,
            "With no blanking, every detection hop should append to "
            "_metric_hist_buf and grow _metric_hist_cnt.",
        )

    def test_blanker_firing_freezes_history(self):
        """When ``_FakeBlanker`` reports a blanked sample, all hops
        derived from that chunk must skip the history append —
        ``_metric_hist_cnt`` stays put for that chunk."""
        # First push warms the channelizer + establishes a baseline.
        for _ in range(5):
            self._push()
        cnt_baseline = self.engine._metric_hist_cnt
        # Flip the blanker into "blanking" mode for ONE chunk.
        self.engine.blanker.force_blanking = True
        self._push()
        cnt_after_blanked = self.engine._metric_hist_cnt
        self.assertEqual(
            cnt_after_blanked, cnt_baseline,
            f"After a blanked chunk, _metric_hist_cnt should NOT grow.  "
            f"Got baseline={cnt_baseline}, post-blanked={cnt_after_blanked}.  "
            f"_skip_pct25_update_h must propagate through the per-hop loop.",
        )
        # Restore — next chunk should grow history again.
        self.engine.blanker.force_blanking = False
        self._push()
        self.assertGreater(
            self.engine._metric_hist_cnt, cnt_after_blanked,
            "Once blanker stops firing, history should resume growing.",
        )

    def test_skip_flag_propagated_to_sync_correlator(self):
        """The same flag must also gate the sync correlator's history
        (TODO #29's ``_sync_metric_hist_buf``) — otherwise the sync
        detector's pct25 ratchets down during blanking the same way
        the squared-FFT path used to.

        We assert by directly calling ``_compute_sync_metric_db`` with
        ``skip_history_update=True`` and verifying its history counter
        doesn't advance.
        """
        from map144_app.processing import _compute_sync_metric_db
        from map144_app.channelizer import N_CHANNELS
        from map144_app.msk144_spd import NSPM

        # Synthesise a sync_window of weak noise.
        rng = np.random.default_rng(42)
        win = (rng.standard_normal((N_CHANNELS, NSPM)).astype(np.float32) +
               1j * rng.standard_normal((N_CHANNELS, NSPM)).astype(np.float32)
               ).astype(np.complex64) * 0.01

        # Baseline call: history advances.
        cnt_before = self.engine._sync_metric_hist_cnt
        _compute_sync_metric_db(
            win,
            self.engine._sync_metric_hist_buf,
            '_sync_metric_hist_idx', '_sync_metric_hist_cnt',
            '_sync_pct25', '_sync_pct25_ctr',
            self.engine,
            skip_history_update=False,
        )
        self.assertEqual(self.engine._sync_metric_hist_cnt, cnt_before + 1)

        # Now with the skip flag: history should NOT advance.
        cnt_mid = self.engine._sync_metric_hist_cnt
        _compute_sync_metric_db(
            win,
            self.engine._sync_metric_hist_buf,
            '_sync_metric_hist_idx', '_sync_metric_hist_cnt',
            '_sync_pct25', '_sync_pct25_ctr',
            self.engine,
            skip_history_update=True,
        )
        self.assertEqual(
            self.engine._sync_metric_hist_cnt, cnt_mid,
            "_compute_sync_metric_db with skip_history_update=True "
            "must NOT advance the sync history counter.",
        )

    def test_blanker_on_very_first_chunk_does_not_crash(self):
        """REGRESSION: if the VERY FIRST chunk after engine start
        happens to trip the blanker, the #41c gate would prevent
        ``_metric_hist_cnt`` from growing — leaving the history at
        size 0.  The pct25 recompute then did ``hist_s[k]`` with
        ``k = 0`` on a zero-length array → IndexError.  Reported
        2026-05-10 from a real Flex source where chunk-0 had a
        spike strong enough to be blanked.

        Fix: skip the recompute when history is empty AND fall back
        to ones() so detection thresholds still compute.
        """
        # Fresh engine; force blanker on first chunk.
        self.engine.blanker.force_blanking = True
        # This should NOT raise, even though history will never grow
        # during this chunk's hops.
        for _ in range(3):
            self._push()
        # And once we stop forcing blanker, normal operation resumes.
        self.engine.blanker.force_blanking = False
        for _ in range(5):
            self._push()
        self.assertGreater(
            self.engine._metric_hist_cnt, 0,
            "After blanker stops firing, history should resume "
            "growing normally.",
        )

    def test_default_blanker_no_blanking_no_skip(self):
        """The Engine's default real blanker (LinradBlanker) on weak
        noise input should NOT fire — the gate flag should be False
        and history should grow normally.  This is the regression
        guard: blanker doesn't false-positive on quiet operation."""
        # Replace the fake blanker with the real default.
        from map144_app.noise_blanker import LinradBlanker
        self.engine.blanker = LinradBlanker()
        cnt_before = self.engine._metric_hist_cnt
        for _ in range(10):
            self._push()
        # Gate flag should be False (or absent) after the run if NB
        # didn't fire.
        self.assertFalse(
            getattr(self.engine, '_skip_pct25_update_h', False),
            "LinradBlanker on weak noise should not flag blanking; "
            "_skip_pct25_update_h should be False.",
        )
        # And history grew.
        self.assertGreater(self.engine._metric_hist_cnt, cnt_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
