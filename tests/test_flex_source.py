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
"""Hardware-free tests for ``map144_app.blocks.FlexSource``.

Uses a small fake that emulates the ``FlexDAXIQ``-shaped protocol
(``sample_queue`` + ``start()`` + ``stop()``).  The protocol is small
enough that the fake is one screenful — that itself validates the
"swap-in for tests" property the design aims for.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    FLEX_DAXIQ_FULL_SCALE,
    FlexSource,
    FlexSourceConfig,
    Record,
    Runtime,
    make_sample_stream,
)


# ---------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------


@dataclass
class _FakePacket:
    """Stand-in for ``flexclient.models.VitaPacket``.  FlexSource only
    looks at ``.samples``; other fields are present for completeness.
    """

    samples: np.ndarray
    timestamp_int: int = 0
    timestamp_frac: int = 0
    stream_id: int = 0
    sequence: int = 0
    tsf: int = 1


class _FakeFlexClient:
    """Emulates the slice of ``FlexDAXIQ`` that ``FlexSource`` depends on.

    Tests can ``push()`` packets to be drained by the source.
    ``start()`` and ``stop()`` flip a flag so tests can verify
    lifecycle handling.
    """

    def __init__(self, queue_size: int = 4000):
        self.sample_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.started = False
        self.stopped = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.started = True
        self.stopped = False
        self.start_calls += 1

    def stop(self) -> None:
        self.started = False
        self.stopped = True
        self.stop_calls += 1

    def push(self, samples: np.ndarray) -> None:
        """Helper: enqueue a packet with the given samples (in
        ``±FLEX_DAXIQ_FULL_SCALE`` units, matching real Flex output).
        """
        self.sample_queue.put_nowait(_FakePacket(samples=samples))


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestFlexSourceConstructor(unittest.TestCase):

    def test_requires_client(self):
        with self.assertRaises(ValueError):
            FlexSource("flex", client=None)  # type: ignore[arg-type]

    def test_requires_sample_queue_attribute(self):
        class _NotAClient:
            pass
        with self.assertRaises(TypeError):
            FlexSource("flex", client=_NotAClient())  # type: ignore[arg-type]


class TestFlexSourceLifecycle(unittest.TestCase):

    def test_start_stop_calls_client(self):
        client = _FakeFlexClient()
        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(sample_rate_hz=48_000),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=256,
        )
        src.outputs["iq"] = out

        rt.start()
        # Give Source.on_start a moment to call client.start().
        time.sleep(0.05)
        self.assertEqual(client.start_calls, 1)
        self.assertTrue(client.started)
        rt.stop(timeout_s=1.0)
        self.assertEqual(client.stop_calls, 1)
        self.assertTrue(client.stopped)

    def test_lifecycle_disabled(self):
        """``own_client_lifecycle=False`` lets the caller manage start/stop."""
        client = _FakeFlexClient()
        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(
                sample_rate_hz=48_000,
                own_client_lifecycle=False,
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=256,
        )
        src.outputs["iq"] = out

        rt.start()
        time.sleep(0.05)
        self.assertEqual(client.start_calls, 0)
        rt.stop(timeout_s=1.0)
        self.assertEqual(client.stop_calls, 0)


class TestFlexSourceIngestion(unittest.TestCase):

    def test_packets_become_records(self):
        client = _FakeFlexClient()
        # Build three packets of distinct sample values, each at full scale.
        # After FlexSource normalises, expected payloads are ±1.0.
        packets = [
            np.full(256, 32768.0 + 0j, dtype=np.complex64),    # → 1.0
            np.full(256, -32768.0 + 0j, dtype=np.complex64),   # → -1.0
            np.full(128, 16384.0 + 16384.0j, dtype=np.complex64),  # → 0.5+0.5j, shorter
        ]
        for pkt in packets:
            client.push(pkt)

        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(
                sample_rate_hz=48_000,
                queue_get_timeout_s=0.05,
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=256,
        )
        src.outputs["iq"] = out

        rt.start()
        try:
            # Drain three records.
            recs: list[Record] = []
            deadline = time.monotonic() + 2.0
            while len(recs) < 3 and time.monotonic() < deadline:
                try:
                    recs.append(out.get(timeout=0.1))
                except Exception:
                    continue
        finally:
            rt.stop(timeout_s=1.0)

        self.assertEqual(len(recs), 3)
        # Sample-value scaling.
        np.testing.assert_allclose(recs[0].data.real, 1.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recs[1].data.real, -1.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recs[2].data.real, 0.5, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recs[2].data.imag, 0.5, rtol=0, atol=1e-6)
        # Variable record length is accepted (§3.1).
        self.assertEqual(recs[0].n_samples, 256)
        self.assertEqual(recs[2].n_samples, 128)
        # Gap-free, monotonic start_sample.
        self.assertEqual(recs[0].start_sample, 0)
        self.assertEqual(recs[1].start_sample, 256)
        self.assertEqual(recs[2].start_sample, 512)
        # Both clock anchors present and reasonable (§3.5).
        for r in recs:
            self.assertEqual(r.sample_rate_hz, 48_000)
            self.assertGreater(r.wall_clock_at_production, time.time() - 60)

    def test_empty_queue_returns_no_record(self):
        """No packets → no records emitted; tick returns cleanly so
        ``stop()`` is responsive even when the radio is silent.
        """
        client = _FakeFlexClient()
        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(
                sample_rate_hz=48_000,
                queue_get_timeout_s=0.05,
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=256,
        )
        src.outputs["iq"] = out

        t0 = time.monotonic()
        rt.start()
        time.sleep(0.20)
        rt.stop(timeout_s=1.0)
        elapsed = time.monotonic() - t0

        # No records emitted.
        items = []
        try:
            while True:
                items.append(out.get(timeout=0.01))
        except Exception:
            pass
        self.assertEqual(len([x for x in items if isinstance(x, Record)]), 0)
        # Stop was prompt (well under the implicit 5 s join timeout).
        self.assertLess(elapsed, 1.5)

    def test_packet_without_samples_attribute_is_skipped(self):
        client = _FakeFlexClient()
        # Push a malformed packet (no .samples attr) followed by a good one.
        client.sample_queue.put_nowait(object())  # bogus packet
        client.push(np.full(64, 32768.0 + 0j, dtype=np.complex64))

        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(
                sample_rate_hz=48_000,
                queue_get_timeout_s=0.05,
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=64,
        )
        src.outputs["iq"] = out

        rt.start()
        try:
            rec = out.get(timeout=1.0)
        finally:
            rt.stop(timeout_s=1.0)
        self.assertEqual(rec.n_samples, 64)
        np.testing.assert_allclose(rec.data.real, 1.0, atol=1e-6)


class TestFlexSourceUnderLoad(unittest.TestCase):

    def test_continuous_packets(self):
        """Producer thread pushes packets continuously; FlexSource keeps up."""
        client = _FakeFlexClient(queue_size=200)
        rt = Runtime()
        src = rt.add(
            FlexSource("flex", client=client),
            FlexSourceConfig(
                sample_rate_hz=48_000,
                queue_get_timeout_s=0.05,
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=256,
        )
        src.outputs["iq"] = out

        # Fixed-pattern samples so we can verify ordering downstream.
        n_packets = 50
        sample_block = np.full(256, 32768.0 + 0j, dtype=np.complex64)

        producer_done = threading.Event()

        def push_loop():
            for i in range(n_packets):
                client.push(sample_block.copy())
                time.sleep(0.002)  # ~2 ms per push (~125 packets/s)
            producer_done.set()

        rt.start()
        try:
            t = threading.Thread(target=push_loop, daemon=True)
            t.start()
            producer_done.wait(timeout=5.0)
            self.assertTrue(producer_done.is_set(), "producer didn't finish")
            time.sleep(0.2)  # let FlexSource drain remaining queue
        finally:
            rt.stop(timeout_s=2.0)

        # Drain whatever made it to the output stream.
        recs = []
        try:
            while True:
                recs.append(out.get(timeout=0.05))
        except Exception:
            pass
        self.assertEqual(len(recs), n_packets,
                         f"expected {n_packets} records, got {len(recs)}")
        # Gap-free, monotonic.
        for i, r in enumerate(recs):
            self.assertEqual(r.start_sample, i * 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
