#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Engine shadow-Block-runtime feature-flag tests — Phase 4 cut-over option 1.

Verifies the ``MAP144_USE_BLOCKS`` env var / ``Engine.use_blocks`` flag
controls whether a parallel Block runtime is started alongside the
legacy DSP path:

- **Flag OFF (default):** ``_block_runtime`` stays None; no Block
  imports are pulled in beyond the static ones; ``process_iq_data``
  delegates straight to the legacy implementation.
- **Flag ON:** the Block runtime is lazy-initialised on the first IQ
  tick, runs alongside legacy, and writes its decode events to
  ``MSK144/detections/blocks/decode_events.jsonl``.  A failure on the
  Block side is logged but does not propagate into the legacy path.

We do NOT validate decode-output equivalence here — that's the §10
Stage 2 gate (``test_block_pipeline_replay``).  This file's scope is
just the integration plumbing: the flag toggles the runtime; the
runtime starts and stops cleanly; ``process_iq_data`` keeps working
both ways.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_engine_block_shadow -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _silent_iq_chunk(n: int = 4096) -> np.ndarray:
    """A small zero-IQ chunk — just exercises the path; no decodes
    expected (no signal in pure zeros)."""
    return np.zeros(n, dtype=np.complex64)


class TestFlagOffByDefault(unittest.TestCase):
    """Without the env var the flag is False and no Block runtime
    starts.  This is the rollback case for the cut-over commit:
    untouching the env / attribute keeps the engine identical to its
    pre-cut-over behaviour.
    """

    def test_default_flag_is_false(self):
        # Make sure no inherited env from parent shell taints the test.
        prev = os.environ.pop("MAP144_USE_BLOCKS", None)
        try:
            from map144_app.engine import Engine
            e = Engine()
            self.assertFalse(e.use_blocks)
            self.assertIsNone(e._block_runtime)
        finally:
            if prev is not None:
                os.environ["MAP144_USE_BLOCKS"] = prev

    def test_legacy_path_runs_when_flag_off(self):
        prev = os.environ.pop("MAP144_USE_BLOCKS", None)
        try:
            from map144_app.engine import Engine
            e = Engine()
            # process_iq_data dispatch falls through to legacy.  Pure
            # zero IQ is harmless; the legacy DSP path will run
            # noise-blanker + spectrogram on it without raising.
            chunk = _silent_iq_chunk(4096)
            # ts_int / ts_frac unused on the WAV / synthetic path the
            # legacy code tolerates with 0/0.
            e.process_iq_data(chunk, 0, 0)
            # Block runtime never started.
            self.assertIsNone(e._block_runtime)
        finally:
            if prev is not None:
                os.environ["MAP144_USE_BLOCKS"] = prev


class TestFlagOnFromEnv(unittest.TestCase):
    """``MAP144_USE_BLOCKS=1`` enables shadow mode at construction
    time."""

    def test_env_var_enables_flag(self):
        os.environ["MAP144_USE_BLOCKS"] = "1"
        try:
            from map144_app.engine import Engine
            e = Engine()
            self.assertTrue(e.use_blocks)
            # Lazy-init: runtime not started until first IQ tick.
            self.assertIsNone(e._block_runtime)
        finally:
            os.environ.pop("MAP144_USE_BLOCKS", None)

    def test_invalid_env_value_treated_as_off(self):
        # "0" / "false" / "no" / unset → off.  Anything truthy → on.
        for off_val in ("0", "false", "no", "off", ""):
            os.environ["MAP144_USE_BLOCKS"] = off_val
            from importlib import reload
            from map144_app import engine as _eng_mod
            reload(_eng_mod)   # _env_use_blocks reads at import + ctor
            e = _eng_mod.Engine()
            self.assertFalse(
                e.use_blocks,
                f"expected use_blocks=False for env value {off_val!r}",
            )
        os.environ.pop("MAP144_USE_BLOCKS", None)


class TestShadowRuntimeLifecycle(unittest.TestCase):
    """End-to-end: with the flag on, IQ ingress lazy-starts the Block
    runtime; an explicit ``_stop_block_runtime`` cleans up.
    """

    def setUp(self):
        os.environ["MAP144_USE_BLOCKS"] = "1"

    def tearDown(self):
        os.environ.pop("MAP144_USE_BLOCKS", None)

    def test_first_iq_tick_starts_runtime(self):
        from map144_app.engine import Engine
        e = Engine()
        try:
            self.assertIsNone(e._block_runtime)   # not yet
            chunk = _silent_iq_chunk(4096)
            e.process_iq_data(chunk, 0, 0)
            self.assertIsNotNone(
                e._block_runtime,
                "shadow Block runtime should be lazy-started on first IQ tick",
            )
            # Push another chunk — the runtime keeps going, no
            # double-init.
            chunk2 = _silent_iq_chunk(4096)
            e.process_iq_data(chunk2, 0, 0)
        finally:
            e._stop_block_runtime(timeout_s=2.0)
        self.assertIsNone(e._block_runtime)

    def test_block_path_failure_disables_flag_not_legacy(self):
        """If the Block runtime raises during pump, the flag flips off
        and legacy continues.  Simulated by replacing the IQ-input
        stream with a sentinel that always raises on put.
        """
        from map144_app.engine import Engine
        from map144_app.blocks.types import StreamClosed
        e = Engine()
        try:
            chunk = _silent_iq_chunk(1024)
            e.process_iq_data(chunk, 0, 0)
            self.assertIsNotNone(e._block_runtime)
            # Sabotage the IQ input stream.
            class _Boom:
                def put(self, *a, **k):
                    raise RuntimeError("synthetic Block-side failure")
            e._block_iq_in_stream = _Boom()
            # process_iq_data should still complete without raising;
            # legacy path runs first, so no GUI state lost.
            e.process_iq_data(chunk, 0, 0)
            # Flag has been turned off so subsequent calls don't keep
            # poking the broken stream.
            self.assertFalse(e.use_blocks)
        finally:
            e._stop_block_runtime(timeout_s=2.0)


class TestShadowTxStateHandling(unittest.TestCase):
    """Shadow runtime must mirror legacy's TX-aware state handling.

    Legacy (runtime.py:1356-1406, processing.py:503-516, 571) does THREE
    things on the TX→RX boundary:

      1. During TX: drain loop discards packets; process_iq_data not called.
      2. On TX→RX: purge ``_pct25``, ``_nb_spec_avg``, ``_ch_buf``;
         set ``_tx_settle_remaining = 0.5 s`` of samples.
      3. During the 500 ms post-TX settle window: detection skipped
         (channelizer accumulation discarded); NB still learns.

    Without these, AGC transients on TX→RX would contaminate the
    rolling pct25 baseline for ~30 s.

    The shadow runtime's pre-fix behaviour on TX:
      - silent during TX (gated for free, since process_iq_data isn't
        called)
      - DOES NOT purge state on TX→RX
      - DOES NOT skip the 500 ms settle window

    Engine commits the fix in process_iq_data:
      - if previous-tick settle was 0 and current-tick settle is > 0
        (the TX→RX transition signature), tear shadow down
      - if current-tick settle > 0 (we're inside the settle window),
        skip the IQ pump
      - the next IQ tick after settle ends lazy-rebuilds shadow with
        a fresh detector baseline
    """

    def setUp(self):
        os.environ["MAP144_USE_BLOCKS"] = "1"

    def tearDown(self):
        os.environ.pop("MAP144_USE_BLOCKS", None)

    def test_pump_gated_during_tx_settle_window(self):
        """While ``_tx_settle_remaining > 0``, no IQ should reach
        the shadow runtime — even if the runtime is up and the flag
        is on."""
        from map144_app.engine import Engine
        e = Engine()
        try:
            # Boot the runtime by sending one normal tick.
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            self.assertIsNotNone(e._block_runtime)
            iq_samples_before = e._block_iq_sample_counter

            # Simulate the engine being in a post-TX settle window
            # (legacy's runtime drain loop sets this on TX→RX).
            # We've already consumed one tick so prev_settle is 0 and
            # the next call's settle_at_entry will be 24000 — that
            # matches the TX→RX transition signature, so the runtime
            # gets torn down.
            e._tx_settle_remaining = 24_000   # 0.5 s @ 48 kHz
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)

            # Tear-down on TX→RX: shadow runtime should be None now.
            self.assertIsNone(
                e._block_runtime,
                "TX→RX transition should have torn down the shadow runtime "
                "to mirror legacy's _pct25 / _nb / _ch_state purge",
            )

            # Subsequent ticks while still in settle should NOT
            # restart the runtime (settle gate skips the pump
            # entirely).  legacy decremented _tx_settle_remaining by
            # 4096 on the first call, so it's now 19904.
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            self.assertIsNone(
                e._block_runtime,
                "Shadow runtime should stay torn down during settle",
            )

            # Force settle to complete — next tick should lazy-rebuild
            # shadow with a fresh detector baseline.
            e._tx_settle_remaining = 0
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            self.assertIsNotNone(
                e._block_runtime,
                "After settle ends, shadow runtime should lazy-rebuild "
                "on next IQ tick",
            )
        finally:
            e._stop_block_runtime(timeout_s=1.0)

    def test_pump_runs_when_settle_zero(self):
        """Sanity: with no TX-settle in progress, the pump runs as
        normal (no behaviour change to the non-TX path)."""
        from map144_app.engine import Engine
        e = Engine()
        try:
            self.assertEqual(getattr(e, '_tx_settle_remaining', 0), 0)
            sample_counter_before = e._block_iq_sample_counter
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            # Pump fired — sample counter advanced.
            self.assertGreater(e._block_iq_sample_counter, sample_counter_before)
        finally:
            e._stop_block_runtime(timeout_s=1.0)


class TestShadowChannelizerRebuild(unittest.TestCase):
    """When the operator moves the panadapter centre, legacy calls
    ``_rebuild_channelizer_state`` which rebuilds its NCO grid for
    the new channel_offset_hz.  Shadow's ChannelizerBlock + DetectorBlock
    were configured at lazy-init with the THEN-current offset; without
    a parallel rebuild, shadow's NCO stays anchored to the old offset
    and emits detection events on stale channels — bug 2 from the
    2026-05-09 bake-in (50263 vs 50260 divergence on the NA6MG QSO).
    """

    def setUp(self):
        os.environ["MAP144_USE_BLOCKS"] = "1"

    def tearDown(self):
        os.environ.pop("MAP144_USE_BLOCKS", None)

    def test_rebuild_tears_down_shadow_runtime(self):
        from map144_app.engine import Engine
        e = Engine()
        try:
            # Boot shadow with one IQ tick.
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            self.assertIsNotNone(e._block_runtime)

            # Operator moves the pan: legacy _rebuild_channelizer_state
            # is called.  Shadow should tear down so the next IQ tick
            # lazy-rebuilds with the new offset.
            e._rebuild_channelizer_state()
            self.assertIsNone(
                e._block_runtime,
                "_rebuild_channelizer_state should tear down shadow so "
                "the new channel_offset_hz takes effect on next IQ tick",
            )

            # Next IQ tick brings shadow back, lazy-rebuilt with the
            # NEW channel_offset_hz computed inside
            # _rebuild_channelizer_state.
            e.process_iq_data(_silent_iq_chunk(4096), 0, 0)
            self.assertIsNotNone(e._block_runtime)
        finally:
            e._stop_block_runtime(timeout_s=1.0)

    def test_rebuild_when_shadow_not_started_is_noop(self):
        """If shadow hasn't been lazy-init'd yet, _rebuild_channelizer_state
        must not raise — it's the same path the GUI invokes whenever
        the user retunes, regardless of bake-in flag state."""
        prev = os.environ.pop("MAP144_USE_BLOCKS", None)
        try:
            from map144_app.engine import Engine
            e = Engine()
            self.assertFalse(e.use_blocks)
            self.assertIsNone(e._block_runtime)
            # No raise.
            e._rebuild_channelizer_state()
            self.assertIsNone(e._block_runtime)
        finally:
            if prev is not None:
                os.environ["MAP144_USE_BLOCKS"] = prev


if __name__ == "__main__":
    unittest.main(verbosity=2)
