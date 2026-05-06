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
"""Block base class — see ``docs/block-stream-design.md`` §4 / §7.

Subclass to write a pipeline stage.  Override ``tick()`` (required) and
optionally ``on_start()`` / ``on_stop()`` for setup / teardown that runs
inside the block's thread.  Declare ``config_type`` to point at the
block's typed config dataclass.

Lifecycle (called by the :class:`Runtime`, not by other blocks):

    block.configure(cfg)   # validate, store
    block.start()          # spawn thread; on_start; loop tick()
    block.stop(timeout_s)  # set _stopping; close outputs; join

The run loop is the simplest thing that works:

    on_start()
    while not _stopping.is_set():
        tick()
    on_stop()

``tick()`` may block on its inputs — the runtime has nothing to schedule.
For graceful shutdown, ``tick()`` should either use bounded ``get`` calls
that return periodically, or rely on the producer's ``close()`` to wake
the consumer with :class:`StreamClosed`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .stream import Stream
from .types import BlockConfig, StreamClosed

log = logging.getLogger(__name__)


class Block:
    """Base class for a unit of computation.

    One OS thread per block.  Subclasses override :meth:`tick`.  Wiring
    (assigning streams to ``inputs`` / ``outputs``) is performed by the
    :class:`Runtime` via :meth:`Runtime.connect`.
    """

    #: Subclasses point this at their own ``BlockConfig`` subclass.
    config_type: type[BlockConfig] = BlockConfig

    # Cap rolling tick-latency history; p50 / p99 are computed over this.
    _LATENCY_HISTORY = 1024

    def __init__(self, name: str):
        self.name = name
        self.config: BlockConfig | None = None
        self.inputs: dict[str, Stream] = {}
        self.outputs: dict[str, Stream] = {}
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._started_at: float | None = None
        # Stats — protected by _stats_lock.
        self._stats_lock = threading.Lock()
        self._tick_count = 0
        self._error_count = 0
        self._tick_latencies_ms: list[float] = []

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: BlockConfig) -> None:
        """Validate and store the per-block config.

        Subclasses that override should call ``super().configure(config)``
        first to get the type check, then validate their own fields.
        """
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{self.name!r}: configure expected {self.config_type.__name__}, "
                f"got {type(config).__name__}"
            )
        self.config = config

    def start(self) -> None:
        """Spawn the run thread.  Idempotent-error: raises if already started."""
        if self._thread is not None:
            raise RuntimeError(f"{self.name!r}: already started")
        if self.config is None:
            raise RuntimeError(f"{self.name!r}: must configure() before start()")
        self._stopping.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=f"block-{self.name}", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Cooperative shutdown.  Sets ``_stopping``, closes outputs (so
        downstream consumers wake from blocked ``get``), then joins.
        """
        self._stopping.set()
        for s in list(self.outputs.values()):
            try:
                s.close()
            except Exception:
                log.exception("%s: error closing output %s", self.name, s.name)
        self.join(timeout_s)

    def join(self, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            log.warning("%s: did not stop within %.1fs", self.name, timeout_s)

    # --- Hooks (override) ---------------------------------------------

    def on_start(self) -> None:
        """Called inside the block thread, once, before the first tick.
        Override for thread-local setup.  Errors abort the block.
        """

    def on_stop(self) -> None:
        """Called inside the block thread, once, after the last tick.
        Override for thread-local teardown.  Errors are logged but do not
        re-propagate (the thread is exiting anyway).
        """

    def tick(self) -> None:
        """One iteration of the run loop.  Subclasses MUST override.

        Conventions:

        - Read from ``self.inputs[port].get(timeout=...)`` for cancelable
          waits, or rely on the producer's ``close()`` to surface
          :class:`StreamClosed` for shutdown.
        - Catching :class:`StreamClosed` and re-raising as
          :class:`StopIteration` is the clean-exit signal — the run loop
          treats ``StopIteration`` as "no more work, stop normally."
        - Other exceptions are logged once and the block stops.
        """
        raise NotImplementedError(f"{type(self).__name__} must override tick()")

    # --- Run loop -----------------------------------------------------

    def _run(self) -> None:
        try:
            self.on_start()
        except Exception:
            log.exception("%s: on_start raised; block will not run", self.name)
            return

        try:
            while not self._stopping.is_set():
                t0 = time.monotonic()
                try:
                    self.tick()
                except StopIteration:
                    # Clean exit signalled by tick() (e.g. WavSource at EOF).
                    break
                except StreamClosed:
                    # An input closed; that's a clean shutdown signal too.
                    break
                except Exception:
                    log.exception(
                        "%s: tick raised; stopping block", self.name,
                    )
                    with self._stats_lock:
                        self._error_count += 1
                    break
                else:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    with self._stats_lock:
                        self._tick_count += 1
                        self._tick_latencies_ms.append(elapsed_ms)
                        if len(self._tick_latencies_ms) > self._LATENCY_HISTORY:
                            # Trim from the front; cheap O(n) but only on overflow.
                            self._tick_latencies_ms = (
                                self._tick_latencies_ms[-self._LATENCY_HISTORY :]
                            )
        finally:
            try:
                self.on_stop()
            except Exception:
                log.exception("%s: on_stop raised", self.name)

    # --- Observability ------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return current runtime stats — see §9 #8 for the field set.

        Per-port counters (samples_in, samples_out, events_in, events_out)
        are derived from the connected ``Stream``'s own counters and are
        labelled by the stream's ``kind``.
        """
        now = time.monotonic()
        with self._stats_lock:
            runtime_seconds = now - (self._started_at or now)
            ticks = self._tick_count
            errors = self._error_count
            latencies = sorted(self._tick_latencies_ms)
        if latencies:
            p50 = latencies[len(latencies) // 2]
            p99_idx = max(0, min(len(latencies) - 1, int(len(latencies) * 0.99)))
            p99 = latencies[p99_idx]
        else:
            p50 = 0.0
            p99 = 0.0

        samples_in: dict[str, int] = {}
        events_in: dict[str, int] = {}
        for port, s in self.inputs.items():
            stats = s.stats()
            target = samples_in if stats["kind"] == Stream.KIND_SAMPLES else events_in
            target[port] = stats["n_get"]

        samples_out: dict[str, int] = {}
        events_out: dict[str, int] = {}
        queue_full_seconds: dict[str, float] = {}
        for port, s in self.outputs.items():
            stats = s.stats()
            target = samples_out if stats["kind"] == Stream.KIND_SAMPLES else events_out
            target[port] = stats["n_put"]
            queue_full_seconds[port] = stats["queue_full_seconds"]

        return {
            "name": self.name,
            "runtime_seconds": runtime_seconds,
            "ticks_executed": ticks,
            "samples_in": samples_in,
            "samples_out": samples_out,
            "events_in": events_in,
            "events_out": events_out,
            "queue_full_seconds": queue_full_seconds,
            "tick_latency_ms_p50": p50,
            "tick_latency_ms_p99": p99,
            "errors_in_tick": errors,
        }
