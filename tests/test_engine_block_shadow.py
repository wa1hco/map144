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


if __name__ == "__main__":
    unittest.main(verbosity=2)
