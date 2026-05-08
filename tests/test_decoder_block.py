#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Tests for ``map144_app.blocks.DecoderBlock`` — Stage 1 (skeleton).

Validates the structural concerns introduced before any decode logic
is wired in:

- IQ ring buffer fill (single record, multiple records, wrap-around)
- Wall-clock anchor (``_iq_t0_wall``) captured on first IQ record
- Detection events drain from input → trigger decode-worker dispatch
- Decode workers emit one DecodeEvent each (Stage-1 stub outcome)
- Semaphore caps concurrent workers at ``max_concurrent_decodes``
- Pool-full drops are accounted in stats, not queued
- Ring generation increments on ``reset_ring()``; in-flight workers
  whose dispatch generation no longer matches exit silently
- Stats fields populated on shutdown

Stage 2 will add:
- Real SPD + jt9 decode end-to-end (with synthetic IQ + known signal)
- Bit-exact-replay equivalence vs legacy ``extract_and_decode``
- ``signal_wall_clock`` derivation (verified against legacy
  PERIOD_SLIP-resistant period epoch)

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_decoder_block -v
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    DECODE_EVENT_KIND,
    DecoderBlock,
    DecoderBlockConfig,
    Event,
    Record,
    Runtime,
    Stream,
    StreamClosed,
    make_event_stream,
    make_sample_stream,
)


SR_IQ = 48_000
RING_SECONDS = 1.0   # small ring for fast wrap-around tests


# --- helpers ---------------------------------------------------------


def _iq_record(start_sample: int, n: int, fill: complex = 0.0,
               wall_clock: float = 100.0) -> Record:
    return Record(
        data=np.full(n, fill, dtype=np.complex64),
        sample_rate_hz=SR_IQ,
        start_sample=start_sample,
        wall_clock_at_production=wall_clock,
    )


def _detection_event(occurred_at_sample: int = 0,
                     channel_index: int = 5,
                     wall_clock: float = 100.0) -> Event:
    return Event(
        kind="detection",
        occurred_at_sample=occurred_at_sample,
        sample_rate_hz=SR_IQ,
        wall_clock_at_production=wall_clock,
        payload={
            "channel_index": channel_index,
            "fc_hz": 1500.0,
            "pair_metric_db": 5.0,
            "sync_metric_db": 2.5,
            "sq_metric_db":  3.5,
            "source": "sync",
        },
    )


def _drain(stream: Stream, max_items: int = 20, timeout: float = 0.05):
    items = []
    for _ in range(max_items):
        try:
            items.append(stream.get(timeout=timeout))
        except (TimeoutError, StreamClosed):
            break
    return items


class _Harness:
    """Build a Runtime with one DecoderBlock + wired streams."""

    def __init__(self, max_concurrent: int = 4):
        self.rt = Runtime()
        self.blk = self.rt.add(
            DecoderBlock("decoder"),
            DecoderBlockConfig(
                sample_rate_hz=SR_IQ,
                ring_seconds=RING_SECONDS,
                max_concurrent_decodes=max_concurrent,
                get_timeout_s=0.05,
            ),
        )
        self.iq = make_sample_stream(
            "iq", queue_depth_seconds=1.0,
            sample_rate_hz=SR_IQ, n_samples_per_record=1024,
        )
        self.det = make_event_stream("det", capacity=64)
        self.dec = make_event_stream("dec", capacity=64)
        self.blk.inputs["iq"] = self.iq
        self.blk.inputs["detections"] = self.det
        self.blk.outputs["decodes"] = self.dec

    def start_and_wait_ready(self, timeout: float = 1.0) -> None:
        """Start the runtime and block briefly until ``on_start()`` has
        allocated the ring + semaphore.  ``rt.start()`` returns before
        the block thread runs ``on_start()``, so tests that immediately
        touch internal state need this barrier."""
        self.rt.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.blk._iq_ring is not None and self.blk._decode_sem is not None:
                return
            time.sleep(0.005)
        raise TimeoutError(f"DecoderBlock not ready within {timeout}s")


# --- tests -----------------------------------------------------------


class TestRingIngest(unittest.TestCase):

    def test_single_iq_record_fills_ring_head(self):
        h = _Harness()
        h.rt.start()
        try:
            h.iq.put(_iq_record(start_sample=0, n=1024, fill=1+0j, wall_clock=42.0))
            time.sleep(0.15)
            ring_pos, abs_sample = h.blk._ring_state()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(ring_pos, 1024)
        self.assertEqual(abs_sample, 1024)
        # Ring head holds the data we wrote.
        np.testing.assert_array_equal(
            h.blk._iq_ring[:1024], np.full(1024, 1+0j, dtype=np.complex64)
        )

    def test_iq_t0_wall_anchored_at_first_record(self):
        """`_iq_t0_wall` = wall_clock for sample 0 of the source's stream."""
        h = _Harness()
        h.rt.start()
        try:
            # Record reports start_sample=10000 with wall_clock=42.0.
            # → t0_wall should be 42.0 - 10000/SR_IQ ≈ 41.7917
            h.iq.put(_iq_record(start_sample=10_000, n=1024, wall_clock=42.0))
            time.sleep(0.15)
        finally:
            h.rt.stop(timeout_s=1.0)

        expected = 42.0 - 10_000 / SR_IQ
        self.assertAlmostEqual(h.blk._iq_t0_wall, expected, places=6)

    def test_ring_wraps_correctly(self):
        h = _Harness()           # ring = 1.0 s × 48 kHz = 48000 samples
        h.rt.start()
        try:
            # Push 50 × 1024 = 51200 samples, more than ring capacity (48000).
            for k in range(50):
                h.iq.put(_iq_record(
                    start_sample=k * 1024, n=1024,
                    fill=complex(k, 0), wall_clock=100.0 + k * 0.001,
                ))
            time.sleep(0.4)
            ring_pos, abs_sample = h.blk._ring_state()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(abs_sample, 50 * 1024)
        self.assertEqual(ring_pos, (50 * 1024) % h.blk._ring_n)


class TestDetectionDispatch(unittest.TestCase):

    def test_detection_event_dispatches_one_decode_event(self):
        h = _Harness()
        h.rt.start()
        try:
            h.iq.put(_iq_record(start_sample=0, n=1024))
            h.det.put(_detection_event(occurred_at_sample=512,
                                       channel_index=10))
            time.sleep(0.2)
            decodes = _drain(h.dec)
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(len(decodes), 1)
        e = decodes[0]
        self.assertEqual(e.kind, DECODE_EVENT_KIND)
        self.assertEqual(e.occurred_at_sample, 512)
        self.assertEqual(e.payload.get("channel_index"), 10)
        self.assertEqual(e.payload.get("outcome"), "stub_pending")
        # signal_wall_clock forwarded from detection event
        self.assertAlmostEqual(e.payload.get("signal_wall_clock"), 100.0,
                               places=6)

    def test_pool_full_drops_extra_detections(self):
        # max_concurrent=1 so the second simultaneous dispatch loses.
        # Block stub workers to test pool-full path.
        h = _Harness(max_concurrent=1)

        # Stage-1 stub worker is fast; to exercise pool-full we need to
        # hold the semaphore.  Acquire it once on the test side so the
        # block can't acquire a slot — it'll drop everything.
        h.start_and_wait_ready()
        try:
            # Drain the semaphore so dispatch always fails.
            h.blk._decode_sem.acquire()
            h.iq.put(_iq_record(start_sample=0, n=1024))
            for k in range(5):
                h.det.put(_detection_event(occurred_at_sample=k * 100))
            time.sleep(0.2)
            decodes = _drain(h.dec)
            stats_dropped = h.blk._n_decode_dropped_full
            stats_attempts = h.blk._n_decode_attempts
        finally:
            try:
                h.blk._decode_sem.release()
            except Exception:
                pass
            h.rt.stop(timeout_s=1.0)

        # No decode events reached the output.
        self.assertEqual(len(decodes), 0)
        # All 5 events counted as "dropped pool full".
        self.assertEqual(stats_dropped, 5)
        # No worker was even spawned.
        self.assertEqual(stats_attempts, 0)


class TestRingGeneration(unittest.TestCase):

    def test_reset_ring_increments_generation(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            gen0 = h.blk._ring_gen_get()
            h.blk.reset_ring()
            gen1 = h.blk._ring_gen_get()
            h.blk.reset_ring()
            gen2 = h.blk._ring_gen_get()
        finally:
            h.rt.stop(timeout_s=1.0)
        self.assertEqual(gen1, gen0 + 1)
        self.assertEqual(gen2, gen1 + 1)

    def test_worker_with_stale_generation_does_not_emit(self):
        """A worker dispatched at generation N skips emission if reset
        happens before the worker actually runs."""
        # We can simulate this deterministically by patching the worker
        # entry to wait on an event before doing anything, then triggering
        # reset_ring before releasing it.
        h = _Harness(max_concurrent=4)
        worker_can_proceed = threading.Event()
        original_run = h.blk._worker_run

        def _gated_worker_run(det_evt, gen_at_dispatch):
            worker_can_proceed.wait(timeout=2.0)
            return original_run(det_evt, gen_at_dispatch)

        h.blk._worker_run = _gated_worker_run  # type: ignore[assignment]

        h.start_and_wait_ready()
        try:
            h.iq.put(_iq_record(start_sample=0, n=1024))
            h.det.put(_detection_event(occurred_at_sample=512))
            # Wait briefly so the dispatch happens but the worker is gated.
            time.sleep(0.10)
            # Now reset the ring, which invalidates the dispatched gen.
            h.blk.reset_ring()
            # Release the worker; it should detect the gen mismatch and exit.
            worker_can_proceed.set()
            time.sleep(0.15)
            decodes = _drain(h.dec)
        finally:
            h.rt.stop(timeout_s=1.0)

        # No decode event because the worker noticed the gen change.
        self.assertEqual(len(decodes), 0)


class TestStats(unittest.TestCase):

    def test_stats_populated_on_shutdown(self):
        h = _Harness()
        h.rt.start()
        try:
            for k in range(3):
                h.iq.put(_iq_record(start_sample=k * 1024, n=1024))
            for k in range(2):
                h.det.put(_detection_event(occurred_at_sample=k * 256))
            time.sleep(0.2)
            stats = h.blk.stats()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(stats["iq_records"], 3)
        self.assertEqual(stats["iq_samples"], 3 * 1024)
        self.assertEqual(stats["detection_events"], 2)
        self.assertEqual(stats["decode_attempts"], 2)
        # Workers run quickly with the Stage-1 stub.
        self.assertEqual(stats["decode_completed"], 2)
        self.assertEqual(stats["decode_dropped_full"], 0)


if __name__ == "__main__":
    unittest.main()
