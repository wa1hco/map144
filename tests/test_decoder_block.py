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
from map144_app.blocks import decoder_block as _decoder_block_mod  # noqa: E402


def _stub_extract_and_decode(decode_queue, marker_id, **kwargs):
    """Test stub for ``extract_and_decode``.  Emits a deterministic
    no_decode outcome via the decode_queue so tests can verify the
    block's queue-shim wiring without invoking real DSP / SPD / jt9.
    """
    if decode_queue is not None:
        decode_queue.put({
            "marker_id": marker_id,
            "outcome":   "no_decode",
            "decoder":   "test_stub",
        })


def _stub_decoded_extract_and_decode(decode_queue, marker_id, **kwargs):
    """Test stub that emits a 'decoded' outcome (success path)."""
    if decode_queue is not None:
        decode_queue.put({
            "marker_id": marker_id,
            "outcome":   "decoded",
            "decoder":   "test_stub",
            "message":   "TEST DE WA1HCO",
            "jt9_snr":   12,
            "t_sec":     7.5,
            "radio_khz": 50260.5,
        })


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


class TestRingGapHandling(unittest.TestCase):
    """When upstream drops records (engine ``iq_in`` POLICY_DROP_OLDEST under
    sustained load), the surviving records carry their original engine-
    counter ``start_sample``.  The decoder's ``_abs_sample`` must follow
    those engine-counter positions — not just the count of received
    samples — so detection events (which the channelizer's gap-recovery
    logic also re-anchors to engine-counter coordinates) map to the
    correct ring offset.

    Without this fix, the bake-in observed: shadow runtime decoded fine
    for the first ~73 minutes (no drops yet), then once the band got
    busy and queue overflows accumulated, decoded zero of the next ~187
    minutes' worth of detection events (1576 launches → 0 decodes).
    With this fix the engine-position ↔ ring-offset alignment survives
    drops; samples in the gap range are lost (correct — they were
    dropped) but every other burst's window still reads cleanly.
    """

    def test_gap_advances_abs_sample_to_engine_counter(self):
        """Drop one record in a sequence: ``_abs_sample`` should jump
        to ``last_rec.start_sample + last_rec.n``, not just the
        running sum of received-record sizes."""
        h = _Harness()
        h.rt.start()
        try:
            # Records 0..3 contiguous, then SKIP record 4 (drop), then 5..7.
            # Each record is 1024 samples.  start_samples emitted: 0, 1024,
            # 2048, 3072, [4096 dropped], 5120, 6144, 7168.
            sent_starts = [0, 1024, 2048, 3072, 5120, 6144, 7168]
            for ss in sent_starts:
                h.iq.put(_iq_record(start_sample=ss, n=1024,
                                    fill=complex(ss, 0)))
            time.sleep(0.20)
            ring_pos, abs_sample = h.blk._ring_state()
            stats = h.blk.stats()
        finally:
            h.rt.stop(timeout_s=1.0)

        # ``_abs_sample`` reflects engine-counter position right after
        # the last received record's tail: 7168 + 1024 = 8192.
        # Pre-fix behaviour was 7 × 1024 = 7168 (off by one record's
        # worth = the dropped 1024 samples).
        self.assertEqual(abs_sample, 8192)
        # Ring pos preserves the engine-position ↔ ring-offset
        # relationship: 8192 % ring_n.
        self.assertEqual(ring_pos, 8192 % h.blk._ring_n)
        # Gap accounting was tracked.
        self.assertEqual(stats["iq_gaps"], 1)
        self.assertEqual(stats["iq_gap_samples"], 1024)
        self.assertEqual(stats["iq_records"], len(sent_starts))
        self.assertEqual(stats["iq_samples"], len(sent_starts) * 1024)

    def test_late_first_record_anchors_correctly(self):
        """If the source has been emitting for a while before the
        decoder attaches (e.g. shadow runtime started lazily after
        the engine's first IQ tick), the first record's start_sample
        is non-zero.  ``_abs_sample`` should anchor to that, so
        subsequent gap-checks work from the correct baseline."""
        h = _Harness()
        h.rt.start()
        try:
            # First record arrives at engine-position 100_000 (e.g. ~2 s
            # of source emission before shadow attached).
            h.iq.put(_iq_record(start_sample=100_000, n=1024))
            h.iq.put(_iq_record(start_sample=101_024, n=1024))
            time.sleep(0.15)
            _, abs_sample = h.blk._ring_state()
            stats = h.blk.stats()
        finally:
            h.rt.stop(timeout_s=1.0)
        # 100_000 + 1024 + 1024 = 102_048
        self.assertEqual(abs_sample, 102_048)
        # No gaps — these two records are contiguous.
        self.assertEqual(stats["iq_gaps"], 0)

    def test_out_of_order_record_dropped(self):
        """A record with start_sample < ``_abs_sample`` is discarded
        silently — real radios don't reorder, so this only fires on
        test injection / corrupted upstream."""
        h = _Harness()
        h.rt.start()
        try:
            h.iq.put(_iq_record(start_sample=0, n=1024))
            h.iq.put(_iq_record(start_sample=1024, n=1024))
            # Now an out-of-order record (engine-position before head).
            h.iq.put(_iq_record(start_sample=512, n=1024))
            # And a valid forward record.
            h.iq.put(_iq_record(start_sample=2048, n=1024))
            time.sleep(0.15)
            _, abs_sample = h.blk._ring_state()
            stats = h.blk.stats()
        finally:
            h.rt.stop(timeout_s=1.0)
        # Out-of-order one is dropped; abs_sample reflects the 3 forward
        # records: 2048 + 1024 = 3072.
        self.assertEqual(abs_sample, 3072)
        self.assertEqual(stats["iq_records"], 3)   # only valid ones counted

    def test_decode_window_after_gap_aligned_to_engine_counter(self):
        """End-to-end: a detection event after a gap drops should
        target the correct engine-counter position in the ring, not
        a position offset by the gap size."""
        from map144_app.blocks import decoder_block as _dmod  # noqa: PLC0415

        # Stub extract_and_decode to record (detect_sample, ring_pos,
        # abs_sample) so we can verify the worker sees the right
        # engine-counter coordinates.
        captured = {}

        def _stub(iq_ring, ring_state_fn, detect_sample, sample_rate,
                  fc_hz, output_dir, decode_queue, marker_id, **_):
            ring_pos, abs_sample = ring_state_fn()
            captured["detect_sample"] = detect_sample
            captured["ring_pos_at_dispatch"] = ring_pos
            captured["abs_sample_at_dispatch"] = abs_sample
            decode_queue.put({"outcome": "no_decode", "marker_id": marker_id})

        original = _dmod.extract_and_decode
        _dmod.extract_and_decode = _stub

        h = _Harness()
        h.rt.start()
        try:
            # Push 3 contiguous records, drop one, push 2 more.
            # All records 4096 samples (one tick at 48k = 85ms).
            for ss in [0, 4096, 8192, 16384, 20480]:   # gap at 12288
                h.iq.put(_iq_record(start_sample=ss, n=4096))
            time.sleep(0.10)

            # Detection event at engine-position 19000 — well after the gap.
            # Note the event uses the full sample rate (48 kHz) here so the
            # rebase logic in _worker_run is a 1:1 pass-through.
            from map144_app.blocks import Event
            evt = Event(
                kind="detection",
                occurred_at_sample=19000,
                sample_rate_hz=SR_IQ,
                wall_clock_at_production=100.0,
                payload={"channel_index": 5, "fc_hz": 6500.0,
                         "source": "sync"},
            )
            h.det.put(evt)
            time.sleep(0.20)
        finally:
            _dmod.extract_and_decode = original
            h.rt.stop(timeout_s=1.0)

        # Decode worker should have been dispatched, and it should
        # receive the detect_sample value as engine-counter position.
        self.assertIn("detect_sample", captured)
        self.assertEqual(captured["detect_sample"], 19000)
        # The decoder's abs_sample at dispatch reflects engine-counter
        # position after the 5 received records:
        #   last_rec.start_sample + last_rec.n = 20480 + 4096 = 24576
        self.assertEqual(captured["abs_sample_at_dispatch"], 24576)


class TestDetectionDispatch(unittest.TestCase):

    def setUp(self):
        # Replace the real decoder with a deterministic stub for these
        # structural tests.  Restored in tearDown.
        self._real_decode_fn = _decoder_block_mod.extract_and_decode
        _decoder_block_mod.extract_and_decode = _stub_extract_and_decode

    def tearDown(self):
        _decoder_block_mod.extract_and_decode = self._real_decode_fn

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
        self.assertEqual(e.payload.get("outcome"), "no_decode")
        self.assertEqual(e.payload.get("decoder"), "test_stub")
        # signal_wall_clock forwarded from detection event
        self.assertAlmostEqual(e.payload.get("signal_wall_clock"), 100.0,
                               places=6)

    def test_decoded_outcome_carries_message_and_snr(self):
        # Override the stub for this test to emit a 'decoded' outcome
        # with metadata, verifying _DecodeQueueShim forwards extras.
        _decoder_block_mod.extract_and_decode = _stub_decoded_extract_and_decode

        h = _Harness()
        h.rt.start()
        try:
            h.iq.put(_iq_record(start_sample=0, n=1024))
            h.det.put(_detection_event(occurred_at_sample=512,
                                       channel_index=3))
            time.sleep(0.2)
            decodes = _drain(h.dec)
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(len(decodes), 1)
        e = decodes[0]
        self.assertEqual(e.payload.get("outcome"), "decoded")
        self.assertEqual(e.payload.get("message"), "TEST DE WA1HCO")
        self.assertEqual(e.payload.get("jt9_snr"), 12)
        self.assertEqual(e.payload.get("decoder"), "test_stub")
        # marker_id forwarded so legacy GUI overlay correlation still works
        self.assertIn("marker_id", e.payload)

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

    def setUp(self):
        self._real_decode_fn = _decoder_block_mod.extract_and_decode
        _decoder_block_mod.extract_and_decode = _stub_extract_and_decode

    def tearDown(self):
        _decoder_block_mod.extract_and_decode = self._real_decode_fn

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
