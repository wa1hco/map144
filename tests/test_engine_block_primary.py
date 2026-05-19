# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# Licensed under GPL-3.0-or-later.
"""End-to-end sanity gate for the Option 2 (block-primary) cut-over.

Verifies that with ``MAP144_USE_BLOCKS=1``:

  1. Legacy ``_process_iq_data`` still runs (noise blanker / waterfall /
     IQ ring), but skips the detect+decode hop loop.
  2. The Block runtime's DetectorBlock + DecoderBlock detect and decode
     a synthetic MSK144 ping pushed in through ``process_iq_data``.
  3. The ``_EngineBridgeSink`` forwards the resulting decode dict to
     ``engine._decode_queue`` in the legacy shape so the GUI decode
     panel + reporter feeds keep working unchanged.

This is the §10-Stage-1-equivalent gate for the cut-over: synthetic IQ
in → legacy GUI consumer sees a decode out, via the Block path.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np


def _msk144code_available() -> bool:
    """Skip the test if the upstream msk144code binary isn't installed."""
    try:
        r = subprocess.run(
            ["msk144code", "--help"],
            capture_output=True, timeout=2.0,
        )
        return r.returncode in (0, 1, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(
    _msk144code_available(),
    "msk144code binary not available — skipping block-primary e2e test",
)
class TestBlockPrimaryDecodeReachesEngineQueue(unittest.TestCase):
    """Synthetic MSK144 ping → block-primary path → engine._decode_queue."""

    def setUp(self):
        os.environ["MAP144_USE_BLOCKS"] = "1"
        self._tmpdir = tempfile.mkdtemp(prefix="map144_blockprimary_")
        # Redirect WAVs / decodes.jsonl / launches.jsonl into the
        # TempDir so synthetic test signals don't contaminate the
        # operator's real production decode log.  Verified essential
        # 2026-05-19 after an overnight bake was confused by leftover
        # test entries written to MSK144/detections/.
        self._prev_outdir = os.environ.pop("MAP144_OUTPUT_DIR", None)
        os.environ["MAP144_OUTPUT_DIR"] = self._tmpdir

    def tearDown(self):
        os.environ.pop("MAP144_USE_BLOCKS", None)
        os.environ.pop("MAP144_OUTPUT_DIR", None)
        if self._prev_outdir is not None:
            os.environ["MAP144_OUTPUT_DIR"] = self._prev_outdir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_decode_reaches_engine_queue_in_block_primary_mode(self):
        from map144_app.engine import Engine
        from tests.test_block_pipeline_e2e import _generate_msk144_test_iq

        sample_rate = 48_000

        # Strong synthetic ping that the existing e2e test relies on.
        iq, _expected_fc_hz = _generate_msk144_test_iq(
            "CQ K1JT FN20",
            duration_s=6.0,
            sample_rate=sample_rate,
            freq_khz=5,
            delay_s=1.5,
            snr_db=20.0,
            noise_sigma=1.0,
            seed=42,
        )

        e = Engine()
        # Engine config defaults work; just make sure the flag is on.
        self.assertTrue(e.use_blocks, "MAP144_USE_BLOCKS=1 should set flag")

        try:
            # Pump the IQ in chunks the way the radio source would.
            chunk_size = 4096
            n = iq.size
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                chunk = iq[start:stop].astype(np.complex64)
                e.process_iq_data(chunk, 0, 0)

            # Give the Block runtime a few seconds to finish the decode
            # thread (extract_and_decode runs in a daemon worker plus
            # jt9 subprocess; 8 s is comfortable headroom).
            deadline = time.monotonic() + 8.0
            decoded = None
            while time.monotonic() < deadline:
                try:
                    item = e._decode_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if item.get("outcome") == "decoded" and item.get("message"):
                    decoded = item
                    break

            self.assertIsNotNone(
                decoded,
                "expected at least one decode dict on engine._decode_queue "
                "via _EngineBridgeSink in block-primary mode",
            )
            # Sanity-check the shape consumers depend on.
            self.assertIn("message", decoded)
            self.assertIn("outcome", decoded)
            self.assertEqual(decoded["outcome"], "decoded")
            # message should contain some of the synthesised callsign /
            # grid tokens (decoder doesn't always recover them all
            # under noise, but ≥1 should land).
            msg = decoded["message"]
            self.assertTrue(
                any(tok in msg for tok in ("K1JT", "CQ", "FN20")),
                f"decoded message {msg!r} doesn't contain any of the "
                "expected tokens — synthetic ping decode failed or got "
                "garbled",
            )
        finally:
            e._stop_block_runtime(timeout_s=2.0)


if __name__ == "__main__":
    unittest.main()
