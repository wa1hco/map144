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
"""Stream — bounded queue between blocks.

See ``docs/block-stream-design.md`` §3 (stream protocol) and §5
(backpressure model).

Two flavours, distinguished by ``Stream.kind``:

- ``KIND_SAMPLES``: carries :class:`Record` payloads.  Default policy
  ``POLICY_BLOCK`` — silent record drops are the bug class we are
  trying to make impossible.
- ``KIND_EVENTS``: carries :class:`Event` payloads.  Default policy
  ``POLICY_DROP_OLDEST`` — events are commentary that can be coalesced.
  Durable event streams (e.g. ``DecodeEventStream`` consumed by Reporter)
  override to ``POLICY_BLOCK``.

A :class:`CoherentPair` wraps two POLICY_BLOCK sample streams and
enforces the §3.2 dual-pol invariants: sample alignment, lockstep
backpressure.  Phase coherence and lockstep filter/decimation alignment
are responsibilities of the producer and the per-polarization processing
blocks; this class only enforces what the queue layer can.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .types import Record, StreamClosed

log = logging.getLogger(__name__)


# Sentinel placed on the queue by ``close()`` so consumers can drain
# remaining items before seeing end-of-stream.  Distinct from None so
# valid None payloads (if anyone ever wanted them) are not confused with
# closure.
_CLOSE_SENTINEL = object()


class Stream:
    """A typed, bounded, single-producer-many-consumers queue."""

    POLICY_BLOCK = "block"
    POLICY_DROP_OLDEST = "drop_oldest"
    POLICY_DROP_NEWEST = "drop_newest"

    KIND_SAMPLES = "samples"
    KIND_EVENTS = "events"

    _ALL_POLICIES = (POLICY_BLOCK, POLICY_DROP_OLDEST, POLICY_DROP_NEWEST)
    _ALL_KINDS = (KIND_SAMPLES, KIND_EVENTS)

    def __init__(
        self,
        name: str,
        capacity: int,
        on_full: str = POLICY_BLOCK,
        kind: str = KIND_SAMPLES,
    ):
        if on_full not in self._ALL_POLICIES:
            raise ValueError(f"unknown on_full policy: {on_full!r}")
        if kind not in self._ALL_KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.name = name
        self.capacity = capacity
        self.on_full = on_full
        self.kind = kind
        self._q: queue.Queue = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        # Stats — protected by _lock for cross-thread reads.
        self._lock = threading.Lock()
        self._n_put = 0
        self._n_get = 0
        self._n_dropped = 0
        self._queue_full_seconds = 0.0

    # --- Producer side -----------------------------------------------

    def put(self, item: Any) -> None:
        """Place ``item`` on the stream, applying ``on_full`` policy.

        Behaviour by policy:

        - ``POLICY_BLOCK``: blocks until space is available.  Time spent
          blocked is accumulated into ``queue_full_seconds`` for
          observability of slow consumers.
        - ``POLICY_DROP_OLDEST``: if full, evicts the oldest item to make
          room (counted as a drop), then enqueues ``item``.
        - ``POLICY_DROP_NEWEST``: if full, drops ``item`` itself.
        """
        if self._closed.is_set():
            raise StreamClosed(f"stream {self.name!r} is closed")

        if self.on_full == self.POLICY_BLOCK:
            t0 = time.monotonic()
            self._q.put(item)
            elapsed = time.monotonic() - t0
            with self._lock:
                self._n_put += 1
                # Only count meaningful waits.  ``Queue.put`` returns
                # promptly when there's space; sub-ms times are noise.
                if elapsed > 0.001:
                    self._queue_full_seconds += elapsed
            return

        if self.on_full == self.POLICY_DROP_OLDEST:
            try:
                self._q.put_nowait(item)
                with self._lock:
                    self._n_put += 1
            except queue.Full:
                # Make room by evicting the oldest.  Race-tolerant: if
                # someone else drained first, get_nowait raises Empty
                # and we fall through to put.
                try:
                    self._q.get_nowait()
                    with self._lock:
                        self._n_dropped += 1
                except queue.Empty:
                    pass
                self._q.put_nowait(item)
                with self._lock:
                    self._n_put += 1
            return

        if self.on_full == self.POLICY_DROP_NEWEST:
            try:
                self._q.put_nowait(item)
                with self._lock:
                    self._n_put += 1
            except queue.Full:
                with self._lock:
                    self._n_dropped += 1
            return

    # --- Consumer side -----------------------------------------------

    def get(self, timeout: float | None = None) -> Any:
        """Block on the queue.  Returns the next item, or raises
        :class:`StreamClosed` once the producer closed *and* the queue
        is drained.  Raises :class:`TimeoutError` if ``timeout`` elapses
        without an item arriving.
        """
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"stream {self.name!r} get timed out")
        if item is _CLOSE_SENTINEL:
            # Re-arm the sentinel so additional consumers also see closure.
            try:
                self._q.put_nowait(_CLOSE_SENTINEL)
            except queue.Full:
                # Should not happen — sentinel slot was just freed by our get.
                pass
            raise StreamClosed(f"stream {self.name!r} is closed")
        with self._lock:
            self._n_get += 1
        return item

    # --- Producer signals ---------------------------------------------

    def close(self) -> None:
        """Producer signals end-of-stream.  Idempotent.  Consumers will
        drain queued items and then see :class:`StreamClosed`.
        """
        if self._closed.is_set():
            return
        self._closed.set()
        # Place sentinel.  If queue is full, evict one to make room — the
        # consumer needs to see EOS even if the policy was lossy.
        try:
            self._q.put_nowait(_CLOSE_SENTINEL)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(_CLOSE_SENTINEL)
            except queue.Full:
                # Extremely unlikely race; leave the close flag set so a
                # subsequent get() at least sees the queue depth shrink.
                log.warning("%s: could not place close sentinel", self.name)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # --- Observability ------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of stream stats — see §9 #8."""
        with self._lock:
            return {
                "name": self.name,
                "kind": self.kind,
                "capacity": self.capacity,
                "on_full": self.on_full,
                "n_put": self._n_put,
                "n_get": self._n_get,
                "n_dropped": self._n_dropped,
                "queue_full_seconds": self._queue_full_seconds,
                "depth": self._q.qsize(),
                "closed": self._closed.is_set(),
            }


# --- Factories --------------------------------------------------------


def make_sample_stream(
    name: str,
    queue_depth_seconds: float,
    sample_rate_hz: int,
    n_samples_per_record: int,
    on_full: str = Stream.POLICY_BLOCK,
) -> Stream:
    """Construct a sample stream sized in seconds of buffered IQ.

    ``capacity`` (in records) is computed from the expected record rate.
    Default policy is ``POLICY_BLOCK`` — silent IQ loss is the bug class
    this design exists to prevent.
    """
    if sample_rate_hz <= 0 or n_samples_per_record <= 0:
        raise ValueError("sample_rate_hz and n_samples_per_record must be positive")
    records_per_second = max(1, sample_rate_hz // n_samples_per_record)
    capacity = max(1, int(queue_depth_seconds * records_per_second))
    return Stream(name, capacity=capacity, on_full=on_full, kind=Stream.KIND_SAMPLES)


def make_event_stream(
    name: str,
    capacity: int = 16,
    durable: bool = False,
) -> Stream:
    """Construct an event stream.

    Default policy is ``POLICY_DROP_OLDEST`` (events are coalescable
    commentary).  Pass ``durable=True`` to use ``POLICY_BLOCK`` — for
    streams whose events must reach their sink, e.g. ``DecodeEventStream``
    consumed by ``Reporter``.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    on_full = Stream.POLICY_BLOCK if durable else Stream.POLICY_DROP_OLDEST
    return Stream(name, capacity=capacity, on_full=on_full, kind=Stream.KIND_EVENTS)


# --- Coherent pair ----------------------------------------------------


class CoherentPair:
    """Two streams that share a sample-clock anchor — see §3.2.

    Use for dual-polarization H/V streams that originate from the same
    Source's same producer tick.  Invariants enforced:

    - **Sample alignment** asserted on each ``put``: paired records must
      have the same ``start_sample``.  Mismatch raises ``ValueError``.
    - **Lockstep backpressure** by construction: both underlying streams
      use ``POLICY_BLOCK`` with equal capacity.  ``put((rec_h, rec_v))``
      places H, then V.  If V's queue is full, the producer thread is
      held — the next producer tick cannot run until both records have
      been admitted, so neither queue grows past the other by more than
      one record.

    Phase coherence and lockstep filter / decimation alignment are
    *producer responsibilities*; the stream layer cannot enforce them.
    Document and assert at the producing block.
    """

    def __init__(self, name: str, stream_h: Stream, stream_v: Stream):
        if stream_h.on_full != Stream.POLICY_BLOCK or stream_v.on_full != Stream.POLICY_BLOCK:
            raise ValueError(
                f"CoherentPair {name!r}: both streams must use POLICY_BLOCK"
            )
        if stream_h.capacity != stream_v.capacity:
            raise ValueError(
                f"CoherentPair {name!r}: stream capacities must match "
                f"(H={stream_h.capacity}, V={stream_v.capacity})"
            )
        if stream_h.kind != Stream.KIND_SAMPLES or stream_v.kind != Stream.KIND_SAMPLES:
            raise ValueError(
                f"CoherentPair {name!r}: both streams must be KIND_SAMPLES"
            )
        self.name = name
        self.stream_h = stream_h
        self.stream_v = stream_v

    def put(self, item_h: Record, item_v: Record) -> None:
        """Lockstep put.  Asserts sample alignment.  Blocks if either
        underlying queue is full.
        """
        if isinstance(item_h, Record) and isinstance(item_v, Record):
            if item_h.start_sample != item_v.start_sample:
                raise ValueError(
                    f"CoherentPair {self.name!r}: start_sample mismatch "
                    f"(H={item_h.start_sample}, V={item_v.start_sample})"
                )
            if item_h.sample_rate_hz != item_v.sample_rate_hz:
                raise ValueError(
                    f"CoherentPair {self.name!r}: sample_rate_hz mismatch "
                    f"(H={item_h.sample_rate_hz}, V={item_v.sample_rate_hz})"
                )
        self.stream_h.put(item_h)
        self.stream_v.put(item_v)

    def close(self) -> None:
        self.stream_h.close()
        self.stream_v.close()
