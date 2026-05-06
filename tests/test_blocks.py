#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Contract tests for ``map144_app.blocks``.

Validates the invariants laid out in ``docs/block-stream-design.md``:

- §3.1   Record fields, sample-clock anchoring, gap-free property
- §3.2   CoherentPair sample-alignment + lockstep backpressure
- §3.3   Event fields, drop-oldest behaviour
- §3.5   Dual-clock independence — wall_clock_at_production NOT derived
         from start_sample
- §4     Block lifecycle (configure / start / tick / stop)
- §5     Per-policy queue behaviour (BLOCK / DROP_OLDEST / DROP_NEWEST)
- §7     Runtime start/stop ordering, stream closure propagation

Run as::

    cd ~/ham/map144
    python3 -m unittest tests.test_blocks -v
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow ``python3 -m unittest tests.test_blocks`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402  (path injection above)
    Block,
    BlockConfig,
    CoherentPair,
    Event,
    Record,
    Runtime,
    Stream,
    StreamClosed,
    make_event_stream,
    make_sample_stream,
)


def _mk_record(start_sample: int, n: int = 1024, sr: int = 48_000) -> Record:
    """Helper: minimal Record with deterministic fields."""
    return Record(
        data=np.zeros(n, dtype=np.complex64),
        sample_rate_hz=sr,
        start_sample=start_sample,
        wall_clock_at_production=time.time(),
    )


# ---------------------------------------------------------------------
# §3.1, §3.5  Record / Event basics
# ---------------------------------------------------------------------


class TestRecord(unittest.TestCase):
    """Sample-bearing record contract."""

    def test_fields_and_derived(self):
        r = _mk_record(start_sample=10_000, n=512, sr=12_000)
        self.assertEqual(r.start_sample, 10_000)
        self.assertEqual(r.n_samples, 512)
        self.assertEqual(r.end_sample, 10_512)
        self.assertEqual(r.sample_rate_hz, 12_000)
        self.assertGreater(r.wall_clock_at_production, 0)

    def test_frozen(self):
        """Records are immutable — §3.1 requires no in-place mutation of stream payloads."""
        r = _mk_record(0)
        with self.assertRaises(Exception):  # FrozenInstanceError
            r.start_sample = 1  # type: ignore[misc]

    def test_dual_clock_independence(self):
        """§3.5: wall_clock is NOT derived from start_sample.

        Two records with the same start_sample / sample_rate may legitimately
        carry different wall_clock_at_production values (e.g. the same Source
        replayed twice).
        """
        r1 = Record(
            data=np.zeros(8, dtype=np.complex64),
            sample_rate_hz=48_000,
            start_sample=1000,
            wall_clock_at_production=1_000_000_000.0,
        )
        r2 = Record(
            data=np.zeros(8, dtype=np.complex64),
            sample_rate_hz=48_000,
            start_sample=1000,
            wall_clock_at_production=2_000_000_000.0,
        )
        self.assertEqual(r1.start_sample, r2.start_sample)
        self.assertNotEqual(r1.wall_clock_at_production, r2.wall_clock_at_production)


class TestEvent(unittest.TestCase):

    def test_fields(self):
        e = Event(
            kind="detection",
            occurred_at_sample=12_345,
            sample_rate_hz=12_000,
            wall_clock_at_production=time.time(),
            payload={"channel": 7, "fc_hz": 50_400_000},
        )
        self.assertEqual(e.kind, "detection")
        self.assertEqual(e.payload["channel"], 7)


# ---------------------------------------------------------------------
# §5  Stream queue policies
# ---------------------------------------------------------------------


class TestStreamPolicies(unittest.TestCase):

    def test_block_policy_blocks_when_full(self):
        """POLICY_BLOCK: producer waits for consumer."""
        s = Stream("test-block", capacity=2, on_full=Stream.POLICY_BLOCK)
        s.put(_mk_record(0))
        s.put(_mk_record(1024))
        # Third put should block; verify by spawning a thread and timing.
        producer_done = threading.Event()

        def producer():
            s.put(_mk_record(2048))
            producer_done.set()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        # Producer should still be blocked.
        self.assertFalse(producer_done.wait(timeout=0.05))
        # Drain one — producer should unblock.
        _ = s.get()
        self.assertTrue(producer_done.wait(timeout=1.0))

    def test_drop_oldest_evicts(self):
        s = Stream("test-drop-oldest", capacity=2, on_full=Stream.POLICY_DROP_OLDEST)
        for i in range(5):
            s.put(_mk_record(i * 1024))
        # Should hold the 2 most recent (start_samples 3072 and 4096).
        first = s.get()
        second = s.get()
        self.assertEqual(first.start_sample, 3 * 1024)
        self.assertEqual(second.start_sample, 4 * 1024)
        st = s.stats()
        self.assertEqual(st["n_dropped"], 3)

    def test_drop_newest_drops_incoming(self):
        s = Stream("test-drop-newest", capacity=2, on_full=Stream.POLICY_DROP_NEWEST)
        s.put(_mk_record(0))
        s.put(_mk_record(1024))
        s.put(_mk_record(2048))  # dropped
        s.put(_mk_record(3072))  # dropped
        first = s.get()
        second = s.get()
        self.assertEqual(first.start_sample, 0)
        self.assertEqual(second.start_sample, 1024)
        st = s.stats()
        self.assertEqual(st["n_dropped"], 2)

    def test_close_drains_then_raises(self):
        s = Stream("test-close", capacity=4, on_full=Stream.POLICY_BLOCK)
        s.put(_mk_record(0))
        s.put(_mk_record(1024))
        s.close()
        # Drain remaining items first.
        self.assertEqual(s.get().start_sample, 0)
        self.assertEqual(s.get().start_sample, 1024)
        with self.assertRaises(StreamClosed):
            s.get()

    def test_close_idempotent(self):
        s = Stream("test-close2", capacity=2, on_full=Stream.POLICY_BLOCK)
        s.close()
        s.close()  # must not raise
        self.assertTrue(s.closed)

    def test_factories_set_kind(self):
        ss = make_sample_stream("ss", queue_depth_seconds=1.0,
                                sample_rate_hz=48_000, n_samples_per_record=1024)
        es = make_event_stream("es", capacity=8)
        es_durable = make_event_stream("es-durable", capacity=8, durable=True)
        self.assertEqual(ss.kind, Stream.KIND_SAMPLES)
        self.assertEqual(ss.on_full, Stream.POLICY_BLOCK)
        self.assertEqual(es.kind, Stream.KIND_EVENTS)
        self.assertEqual(es.on_full, Stream.POLICY_DROP_OLDEST)
        self.assertEqual(es_durable.on_full, Stream.POLICY_BLOCK)


# ---------------------------------------------------------------------
# §3.2  Coherent pair
# ---------------------------------------------------------------------


class TestCoherentPair(unittest.TestCase):

    def _mk_pair(self, capacity: int = 4) -> CoherentPair:
        sh = Stream("h", capacity=capacity, on_full=Stream.POLICY_BLOCK)
        sv = Stream("v", capacity=capacity, on_full=Stream.POLICY_BLOCK)
        return CoherentPair("pair", sh, sv)

    def test_aligned_put_admits_both(self):
        p = self._mk_pair()
        p.put(_mk_record(0), _mk_record(0))
        self.assertEqual(p.stream_h.stats()["depth"], 1)
        self.assertEqual(p.stream_v.stats()["depth"], 1)

    def test_misaligned_start_sample_raises(self):
        p = self._mk_pair()
        with self.assertRaises(ValueError):
            p.put(_mk_record(0), _mk_record(1024))

    def test_misaligned_sample_rate_raises(self):
        p = self._mk_pair()
        rh = Record(data=np.zeros(8, dtype=np.complex64),
                    sample_rate_hz=48_000, start_sample=0,
                    wall_clock_at_production=time.time())
        rv = Record(data=np.zeros(8, dtype=np.complex64),
                    sample_rate_hz=12_000, start_sample=0,  # mismatched rate
                    wall_clock_at_production=time.time())
        with self.assertRaises(ValueError):
            p.put(rh, rv)

    def test_lockstep_backpressure(self):
        """Draining only one half of the pair is not enough to release the
        producer.  ``CoherentPair.put`` places H, then V; the producer must
        clear *both* sides to make a full step.

        This is the lockstep guarantee: a single record cannot be admitted
        on H without also being admitted on V, so a slow V consumer
        backpressures the producer regardless of how fast H drains.
        """
        p = self._mk_pair(capacity=2)
        # Fill both queues.
        p.put(_mk_record(0), _mk_record(0))
        p.put(_mk_record(1024), _mk_record(1024))
        producer_done = threading.Event()

        def producer():
            p.put(_mk_record(2048), _mk_record(2048))
            producer_done.set()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        # Producer blocked on ``put_h`` because H is full.
        self.assertFalse(producer_done.wait(timeout=0.05))
        # Drain H — ``put_h`` completes, but ``put_v`` blocks because V is
        # still full.  Lockstep: one consumer's progress is not enough.
        _ = p.stream_h.get()
        self.assertFalse(producer_done.wait(timeout=0.05))
        # Drain V — now ``put_v`` completes and the producer finishes.
        _ = p.stream_v.get()
        self.assertTrue(producer_done.wait(timeout=1.0))
        # H queue depth is at most capacity (no run-away on the H side).
        self.assertLessEqual(p.stream_h.stats()["depth"], p.stream_h.capacity)

    def test_requires_block_policy(self):
        sh = Stream("h", capacity=2, on_full=Stream.POLICY_DROP_OLDEST)
        sv = Stream("v", capacity=2, on_full=Stream.POLICY_BLOCK)
        with self.assertRaises(ValueError):
            CoherentPair("bad", sh, sv)

    def test_requires_equal_capacity(self):
        sh = Stream("h", capacity=2, on_full=Stream.POLICY_BLOCK)
        sv = Stream("v", capacity=4, on_full=Stream.POLICY_BLOCK)
        with self.assertRaises(ValueError):
            CoherentPair("bad", sh, sv)


# ---------------------------------------------------------------------
# §4, §7  Block lifecycle
# ---------------------------------------------------------------------


@dataclass
class _CountingConfig(BlockConfig):
    target_ticks: int = 5
    sleep_per_tick_s: float = 0.005


class _CountingBlock(Block):
    """Test block that ticks ``target_ticks`` times then exits cleanly."""
    config_type = _CountingConfig

    def __init__(self, name: str):
        super().__init__(name)
        self.tick_count = 0
        self.on_start_called = False
        self.on_stop_called = False

    def on_start(self) -> None:
        self.on_start_called = True

    def on_stop(self) -> None:
        self.on_stop_called = True

    def tick(self) -> None:
        cfg: _CountingConfig = self.config  # type: ignore[assignment]
        self.tick_count += 1
        if self.tick_count >= cfg.target_ticks:
            raise StopIteration
        time.sleep(cfg.sleep_per_tick_s)


@dataclass
class _RaisingConfig(BlockConfig):
    raise_after: int = 2


class _RaisingBlock(Block):
    config_type = _RaisingConfig

    def __init__(self, name: str):
        super().__init__(name)
        self.tick_count = 0

    def tick(self) -> None:
        cfg: _RaisingConfig = self.config  # type: ignore[assignment]
        self.tick_count += 1
        if self.tick_count > cfg.raise_after:
            raise RuntimeError("synthetic failure")
        time.sleep(0.005)


class TestBlockLifecycle(unittest.TestCase):

    def test_must_configure_before_start(self):
        b = _CountingBlock("nocfg")
        with self.assertRaises(RuntimeError):
            b.start()

    def test_configure_type_check(self):
        b = _CountingBlock("typed")
        with self.assertRaises(TypeError):
            b.configure(BlockConfig(name="wrong-type"))  # base, not CountingConfig

    def test_clean_lifecycle(self):
        b = _CountingBlock("clean")
        b.configure(_CountingConfig(name="clean", target_ticks=3, sleep_per_tick_s=0.001))
        b.start()
        # Wait for natural completion (StopIteration after target_ticks).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if b._thread is not None and not b._thread.is_alive():
                break
            time.sleep(0.005)
        b.join(timeout_s=0.5)
        self.assertTrue(b.on_start_called)
        self.assertTrue(b.on_stop_called)
        self.assertEqual(b.tick_count, 3)
        stats = b.stats()
        self.assertEqual(stats["ticks_executed"], 2)  # 2 successful, 3rd raised StopIteration
        self.assertEqual(stats["errors_in_tick"], 0)

    def test_tick_exception_stops_block(self):
        b = _RaisingBlock("raises")
        b.configure(_RaisingConfig(name="raises", raise_after=2))
        b.start()
        b.join(timeout_s=2.0)
        stats = b.stats()
        self.assertEqual(stats["errors_in_tick"], 1)
        self.assertGreaterEqual(stats["ticks_executed"], 2)

    def test_double_start_raises(self):
        b = _CountingBlock("twice")
        b.configure(_CountingConfig(name="twice", target_ticks=1000, sleep_per_tick_s=0.05))
        b.start()
        with self.assertRaises(RuntimeError):
            b.start()
        b.stop(timeout_s=1.0)


# ---------------------------------------------------------------------
# §7  Runtime: end-to-end producer/consumer
# ---------------------------------------------------------------------


@dataclass
class _ProducerConfig(BlockConfig):
    n_records: int = 10
    sample_rate_hz: int = 48_000
    n_samples: int = 1024


class _Producer(Block):
    config_type = _ProducerConfig

    def __init__(self, name: str):
        super().__init__(name)
        self.produced = 0

    def tick(self) -> None:
        cfg: _ProducerConfig = self.config  # type: ignore[assignment]
        if self.produced >= cfg.n_records:
            raise StopIteration
        rec = Record(
            data=np.full(cfg.n_samples, self.produced, dtype=np.complex64),
            sample_rate_hz=cfg.sample_rate_hz,
            start_sample=self.produced * cfg.n_samples,
            wall_clock_at_production=time.time(),
        )
        self.outputs["out"].put(rec)
        self.produced += 1


@dataclass
class _ConsumerConfig(BlockConfig):
    pass


class _Consumer(Block):
    config_type = _ConsumerConfig

    def __init__(self, name: str):
        super().__init__(name)
        self.received: list[Record] = []

    def tick(self) -> None:
        rec = self.inputs["in"].get(timeout=0.5)
        self.received.append(rec)


class TestRuntimeEndToEnd(unittest.TestCase):

    def test_producer_consumer_round_trip(self):
        rt = Runtime()
        p = rt.add(_Producer("prod"), _ProducerConfig(n_records=8))
        c = rt.add(_Consumer("cons"), _ConsumerConfig())
        rt.connect(p, "out", c, "in",
                   queue_depth_seconds=1.0,
                   sample_rate_hz=48_000, n_samples_per_record=1024)
        rt.start()
        # Producer runs to completion; consumer waits for stream close to exit.
        # Give producer time to drain its output queue, then stop.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(c.received) >= 8:
                break
            time.sleep(0.01)
        rt.stop(timeout_s=2.0)

        self.assertEqual(p.produced, 8)
        self.assertEqual(len(c.received), 8)
        # Records arrive in order; gap-free in start_sample.
        for i, rec in enumerate(c.received):
            self.assertEqual(rec.start_sample, i * 1024)

    def test_connect_after_start_raises(self):
        rt = Runtime()
        p = rt.add(_Producer("prod"), _ProducerConfig(n_records=1000))
        c = rt.add(_Consumer("cons"), _ConsumerConfig())
        rt.connect(p, "out", c, "in", queue_depth_seconds=0.5,
                   sample_rate_hz=48_000, n_samples_per_record=1024)
        rt.start()
        try:
            with self.assertRaises(RuntimeError):
                rt.connect(p, "out2", c, "in2")
        finally:
            rt.stop(timeout_s=1.0)

    def test_double_connect_same_port_raises(self):
        rt = Runtime()
        p = rt.add(_Producer("prod"), _ProducerConfig())
        c1 = rt.add(_Consumer("c1"), _ConsumerConfig())
        c2 = rt.add(_Consumer("c2"), _ConsumerConfig())
        rt.connect(p, "out", c1, "in", queue_depth_seconds=0.5,
                   sample_rate_hz=48_000, n_samples_per_record=1024)
        with self.assertRaises(ValueError):
            rt.connect(p, "out", c2, "in",  # output port "out" reused
                       queue_depth_seconds=0.5,
                       sample_rate_hz=48_000, n_samples_per_record=1024)

    def test_stats_jsonl_one_line_per_block(self):
        rt = Runtime()
        rt.add(_Producer("prod"), _ProducerConfig(n_records=2))
        rt.add(_Consumer("cons"), _ConsumerConfig())
        rt.connect(rt.blocks[0], "out", rt.blocks[1], "in",
                   queue_depth_seconds=0.5,
                   sample_rate_hz=48_000, n_samples_per_record=1024)
        rt.start()
        time.sleep(0.1)
        jsonl = rt.stats_jsonl()
        rt.stop(timeout_s=1.0)
        lines = jsonl.strip().split("\n")
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
