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
"""DecoderBlock — Phase 4 #11 (skeleton, Stage 1).

See ``docs/block-stream-design.md`` §6 (block contract) and §6.2 (Tap).
This block consumes both:

  - an :class:`IQStream` (sample stream, continuous) — drained into a
    per-block IQ ring buffer ("Tap" — Decoder-internal detail per the
    §9 #1 ratified decision)
  - a :class:`DetectionEventStream` (event stream, sparse, from
    :class:`DetectorBlock`)

…and emits a :class:`DecodeEventStream` for downstream consumers
(:class:`ReporterBlock` for PSKReporter / DXcluster, the future
Decode-panel sub-feature of :class:`DisplayBlock`).

Stage 1 (this commit): structural scaffold.
  - IQ ring buffer fill from the sample stream
  - Detection-event drain
  - Decode-worker dispatch (semaphore-limited, daemon threads)
  - Worker stub that emits a placeholder DecodeEvent
  - All four CLAUDE.md-relevant invariants documented

Stage 2 (next commit): wire workers to ``detection.extract_and_decode``
for real SPD + jt9 decode work.  The extract_and_decode function stays
unchanged; we provide it with a per-block ring + ring_state callbacks
so it sees the same interface it sees today from the legacy engine.
This minimises risk: the DSP / SPD / jt9 path is bit-for-bit identical
between legacy and Block-form runs, validated by the §10 bit-exact
replay gate.

Threading model (§4 — one thread per block, plus pooled decode workers)
-----------------------------------------------------------------------
- The block's own run thread (``self._thread``) does ALL stream I/O:
  drains IQStream into the ring; drains DetectionEventStream; dispatches
  decode work.  It does NOT do any DSP itself.
- Decode workers are short-lived daemon threads, semaphore-limited to
  ``DecoderBlockConfig.max_concurrent_decodes`` (default 4 — matches
  the legacy ``_jt9_threads`` semaphore width).  Each worker handles
  exactly one DetectionEvent.
- Workers emit DecodeEvents directly to the output stream.
  :class:`Stream` is thread-safe (queue.Queue under the hood); no
  additional locking is needed.

IQ ring buffer
--------------
- Sized at ``ring_seconds`` × ``sample_rate_hz`` complex samples.
  Default 5 s × 48 kHz = 240,000 samples (matches legacy ``_iq_ring``).
- Ring is owned by the block; readers (decode workers) see a *shared
  reference* to the underlying numpy array plus a ``ring_state_fn()``
  callback that returns ``(ring_pos, abs_sample)`` — the same interface
  that ``extract_and_decode`` accepts today.
- ``_ring_gen`` is incremented on every reset (e.g. source switch); a
  worker that started before a reset will see a generation mismatch and
  exit silently rather than read corrupt samples.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .block import Block
from .types import BlockConfig, Event, Record, StreamClosed

log = logging.getLogger(__name__)


#: Decode-event ``kind`` field convention.  Single value to keep the
#: protocol surface narrow; payload differentiates outcomes.
DECODE_EVENT_KIND = "decode"

#: Stage-1 placeholder marker for events emitted by the stub decode
#: worker.  Will be removed in Stage 2 once real decode work is wired.
_STAGE1_STUB_OUTCOME = "stub_pending"


@dataclass
class DecoderBlockConfig(BlockConfig):
    """Per-DecoderBlock typed config."""

    # Ports.
    iq_port_name:        str = "iq"
    detection_port_name: str = "detections"
    decode_port_name:    str = "decodes"

    # IQ ring sizing.
    sample_rate_hz: int   = 48_000
    ring_seconds:   float = 5.0          # holds ~5 s of recent IQ
                                          # (max read window is 1.7 s for
                                          # SPD; 5 s gives 3 s of slack
                                          # for late detection events)

    # Decode-worker pool.  Default 4 matches legacy ``_jt9_threads``
    # semaphore width — jt9 subprocess is the bottleneck so this also
    # caps CPU for the dense-burst case.
    max_concurrent_decodes: int = 4

    # Output directory for saved decode WAVs (decoded events only).
    # Empty string disables WAV save (useful for tests).
    output_dir: str = ""

    # Get timeout for blocking stream reads in tick() — short enough
    # for responsive shutdown.
    get_timeout_s: float = 0.250

    # Stage 2 will add: jt9_args, dual_pol, center_freq_mhz, etc.


class DecoderBlock(Block):
    """SPD + jt9 fallback decoder, as a Phase 3/4 Block.

    Inputs
    ------
    ``iq``         : :class:`Record` — sample stream (continuous).
                     Samples are appended to an internal ring buffer.
    ``detections`` : :class:`Event` (kind="detection") — one event per
                     surviving cluster gate from :class:`DetectorBlock`.

    Outputs
    -------
    ``decodes``    : :class:`Event` (kind="decode") — one event per
                     completed decode attempt (success OR failure).
                     Payload includes outcome ("decoded" / "no_decode" /
                     "timeout" / "error") and decode metadata.

    State
    -----
    ``self._iq_ring``       : np.ndarray complex64, shape (ring_n,)
    ``self._ring_pos``      : int — write position (samples mod ring_n)
    ``self._abs_sample``    : int — monotonic absolute sample counter
    ``self._ring_gen``      : int — increments on reset (source switch)
    ``self._ring_lock``     : threading.Lock — protects ring writes from
                              concurrent worker reads.  Workers take a
                              consistent snapshot of (ring, ring_pos,
                              abs_sample, gen) under this lock, then read
                              without holding it (samples in the ring are
                              immutable once written, so this is safe).
    ``self._decode_sem``    : threading.Semaphore — caps concurrent
                              workers at ``max_concurrent_decodes``.
    ``self._workers``       : set[threading.Thread] — live workers, for
                              graceful shutdown drain in ``on_stop()``.
    """

    config_type = DecoderBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Allocated in on_start once we know the configured sample rate.
        self._iq_ring: np.ndarray = None  # type: ignore[assignment]
        self._ring_n: int = 0
        self._ring_pos: int = 0
        self._abs_sample: int = 0
        self._ring_gen: int = 0

        self._ring_lock: threading.Lock = threading.Lock()

        # Worker pool.  Both initialized in on_start once config is set.
        self._decode_sem: threading.Semaphore = None  # type: ignore[assignment]
        self._workers: set[threading.Thread] = set()
        self._workers_lock: threading.Lock = threading.Lock()

        # Sample-clock anchor for the ring's wall-clock conversion (used
        # by extract_and_decode for period-end calculations).  Set from
        # the first IQStream record's wall_clock_at_production.
        self._iq_t0_wall: float | None = None

        # Stats.
        self._n_iq_records:         int = 0
        self._n_iq_samples:         int = 0
        self._n_detection_events:   int = 0
        self._n_decode_attempts:    int = 0
        self._n_decode_completed:   int = 0
        self._n_decode_dropped_full:int = 0   # workers refused: pool full

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: DecoderBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if config.sample_rate_hz <= 0:
            raise ValueError(f"{self.name}: sample_rate_hz must be > 0")
        if config.ring_seconds <= 0:
            raise ValueError(f"{self.name}: ring_seconds must be > 0")
        if config.max_concurrent_decodes < 1:
            raise ValueError(
                f"{self.name}: max_concurrent_decodes must be >= 1"
            )

    def on_start(self) -> None:
        super().on_start()
        cfg: DecoderBlockConfig = self.config  # type: ignore[assignment]

        self._ring_n = int(cfg.sample_rate_hz * cfg.ring_seconds)
        self._iq_ring = np.zeros(self._ring_n, dtype=np.complex64)
        self._ring_pos = 0
        self._abs_sample = 0
        self._ring_gen += 1   # invalidates any worker that reads stale state

        self._decode_sem = threading.Semaphore(cfg.max_concurrent_decodes)
        self._workers = set()

        log.info(
            "%s: ring=%d samples (%.1f s @ %d Hz), max_concurrent=%d",
            self.name, self._ring_n, cfg.ring_seconds,
            cfg.sample_rate_hz, cfg.max_concurrent_decodes,
        )

    def on_stop(self) -> None:
        super().on_stop()
        # Wait briefly for in-flight workers to finish.  Don't wait
        # forever — jt9 subprocesses have a 20 s timeout of their own.
        with self._workers_lock:
            workers = list(self._workers)
        for t in workers:
            t.join(timeout=2.0)

        log.info(
            "%s: iq_records=%d, iq_samples=%d, detections=%d, "
            "decode_attempts=%d, completed=%d, dropped_pool_full=%d",
            self.name,
            self._n_iq_records, self._n_iq_samples,
            self._n_detection_events,
            self._n_decode_attempts, self._n_decode_completed,
            self._n_decode_dropped_full,
        )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        """Drain IQ + detection streams.  IQ is the heartbeat.

        Block on the IQ stream (fast path — IQ records arrive continuously
        at the configured packet rate).  After each IQ record is appended
        to the ring, drain any queued DetectionEvents non-blocking and
        dispatch decode workers.

        Detection events are SPARSE compared to IQ; binding the loop to
        IQ ensures the ring stays up-to-date even when no detections fire.
        """
        cfg: DecoderBlockConfig = self.config  # type: ignore[assignment]
        iq_port = cfg.iq_port_name
        det_port = cfg.detection_port_name

        if iq_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {iq_port!r} not connected"
            )

        try:
            rec = self.inputs[iq_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            # No IQ but maybe detections queued — handle them too so the
            # detection drain doesn't starve.
            self._drain_detection_events(det_port)
            return
        # StreamClosed propagates and exits cleanly.

        if not isinstance(rec, Record):
            log.warning(
                "%s: dropped non-Record %r on iq port",
                self.name, type(rec).__name__,
            )
            return

        if rec.sample_rate_hz != cfg.sample_rate_hz:
            raise RuntimeError(
                f"{self.name}: iq sample_rate_hz={rec.sample_rate_hz} "
                f"!= configured {cfg.sample_rate_hz}"
            )

        self._ingest_iq(rec)
        self._drain_detection_events(det_port)

    # --- IQ ring management -------------------------------------------

    def _ingest_iq(self, rec: Record) -> None:
        """Append samples to the ring buffer.  Wraps around at ring_n.

        Holds ``self._ring_lock`` only while updating the ring's write
        head — sample data itself is immutable once written, so a worker
        that has captured a (ring_pos, abs_sample) snapshot can read
        without further locking as long as the ring hasn't wrapped past
        the requested window.
        """
        data = np.ascontiguousarray(rec.data, dtype=np.complex64)
        # Multi-channel records (e.g. CombinerBlock's combined output is
        # mono complex; ChannelizerBlock output is per-channel) aren't
        # the right input for the Decoder — it needs the original IQ
        # stream.  Validate.
        if data.ndim != 1:
            log.warning(
                "%s: iq record shape %s is not 1-D; skipping",
                self.name, data.shape,
            )
            return
        n = data.size
        if n == 0:
            return

        with self._ring_lock:
            # Capture the wall-clock anchor at the *first* IQ record.
            # This is the t0 for sample-counter -> wall-clock conversion;
            # used by decode workers for period-end stamps.
            if self._iq_t0_wall is None:
                # rec.start_sample is the source's monotonic counter.
                # wall_clock_at_production is the source's wall-clock
                # for the same instant.  t0_wall = the wall clock for
                # sample 0 of THIS source's stream:
                self._iq_t0_wall = (
                    rec.wall_clock_at_production
                    - rec.start_sample / float(rec.sample_rate_hz)
                )

            # Append, wrapping.  Two-segment copy if needed.
            ring = self._iq_ring
            pos = self._ring_pos
            tail = self._ring_n - pos
            if n <= tail:
                ring[pos:pos + n] = data
                self._ring_pos = (pos + n) % self._ring_n
            else:
                ring[pos:] = data[:tail]
                ring[:n - tail] = data[tail:]
                self._ring_pos = n - tail
            self._abs_sample += n

        self._n_iq_records += 1
        self._n_iq_samples += n

    def _ring_state(self) -> tuple[int, int]:
        """Thread-safe snapshot of (ring_pos, abs_sample).

        Decode workers use this as ``ring_state_fn`` so they can observe
        the writer's progress while polling for "is enough post-detection
        IQ in the ring yet".
        """
        with self._ring_lock:
            return (self._ring_pos, self._abs_sample)

    def _ring_gen_get(self) -> int:
        """Generation counter — workers compare against the value
        captured at dispatch and exit silently if it changed (source
        switch / reset between dispatch and read)."""
        return self._ring_gen

    def reset_ring(self) -> None:
        """Public: reset the ring (used on source switch).

        Increments generation; any worker holding a stale state will
        notice and exit without reading garbage.  Safe to call from
        any thread, including before ``on_start()`` has allocated the
        ring (defensive: in that case only the generation counter is
        bumped — no ring data to clear yet).
        """
        with self._ring_lock:
            if self._iq_ring is not None:
                self._iq_ring[:] = 0
            self._ring_pos = 0
            self._abs_sample = 0
            self._ring_gen += 1
            self._iq_t0_wall = None

    # --- Detection-event drain + decode dispatch ----------------------

    def _drain_detection_events(self, det_port: str) -> None:
        """Pull queued DetectionEvents and dispatch decode workers."""
        det_stream = self.inputs.get(det_port)
        if det_stream is None:
            # Detection input optional in tests / synthetic graphs.
            return
        while True:
            try:
                evt = det_stream.get(timeout=0.0)
            except TimeoutError:
                return
            except StreamClosed:
                return
            if not isinstance(evt, Event) or evt.kind != "detection":
                continue
            self._n_detection_events += 1
            self._dispatch_decode(evt)

    def _dispatch_decode(self, det_evt: Event) -> None:
        """Spawn a daemon worker thread to handle one DetectionEvent.

        Non-blocking — if the semaphore is exhausted (already
        ``max_concurrent_decodes`` workers running), the event is
        DROPPED rather than queued.  This matches legacy behaviour:
        a detection that can't get a decoder slot in real time isn't
        worth catching up on later (the IQ window will have rotated
        out of the ring before a queued event reaches it).
        """
        if not self._decode_sem.acquire(blocking=False):
            self._n_decode_dropped_full += 1
            log.debug(
                "%s: decode pool full, dropping detection at sample %d",
                self.name, det_evt.occurred_at_sample,
            )
            return

        # Capture ring state at dispatch time.  The worker uses this
        # generation to detect stale-IQ-buffer reads after a source
        # switch.
        gen_at_dispatch = self._ring_gen
        worker = threading.Thread(
            target=self._worker_run,
            args=(det_evt, gen_at_dispatch),
            name=f"{self.name}-decode-{self._n_decode_attempts}",
            daemon=True,
        )
        with self._workers_lock:
            self._workers.add(worker)
        self._n_decode_attempts += 1
        worker.start()

    def _worker_run(self, det_evt: Event, gen_at_dispatch: int) -> None:
        """Decode-worker entry point.  Runs in its own daemon thread.

        Stage 1: stub — emits a placeholder DecodeEvent immediately and
        returns.  Stage 2 will replace the body with a call into
        ``map144_app.detection.extract_and_decode`` (or a slimmed-down
        equivalent), passing it the block's ring + ``_ring_state`` /
        ``_ring_gen_get`` callbacks so the function sees the same
        interface it consumes from the legacy engine today.
        """
        try:
            # Quick generation check before wasting work — if the source
            # was switched between dispatch and worker start, exit.
            if self._ring_gen_get() != gen_at_dispatch:
                return

            # Stage-1 placeholder emission.  Carries the source detection
            # event's payload forward so the contract shape can be
            # exercised in tests; outcome marks it as the stub.
            self._emit_decode_event(
                det_evt=det_evt,
                outcome=_STAGE1_STUB_OUTCOME,
                message="",
                payload_extra={},
            )
        finally:
            self._decode_sem.release()
            with self._workers_lock:
                self._workers.discard(threading.current_thread())
            self._n_decode_completed += 1

    def _emit_decode_event(
        self,
        det_evt: Event,
        outcome: str,
        message: str,
        payload_extra: dict,
    ) -> None:
        """Construct and emit one :class:`Event` (kind="decode")."""
        cfg: DecoderBlockConfig = self.config  # type: ignore[assignment]
        port = cfg.decode_port_name
        if port not in self.outputs:
            return  # decode-event sink optional (e.g. headless tests)

        from datetime import datetime, timezone
        wc_now = datetime.now(timezone.utc).timestamp()
        payload = {
            "outcome":      outcome,
            "message":      message,
            "channel_index": det_evt.payload.get("channel_index"),
            "fc_hz":        det_evt.payload.get("fc_hz"),
            "det_source":   det_evt.payload.get("source"),
            # Forward the underlying signal's wall clock from the
            # detection event so reporters can stamp WSJT-X-style period
            # ends accurately even if the decode took a while:
            "signal_wall_clock": det_evt.wall_clock_at_production,
        }
        payload.update(payload_extra)

        out = Event(
            kind=DECODE_EVENT_KIND,
            occurred_at_sample=det_evt.occurred_at_sample,
            sample_rate_hz=det_evt.sample_rate_hz,
            wall_clock_at_production=wc_now,
            payload=payload,
        )
        try:
            self.outputs[port].put(out)
        except StreamClosed:
            pass  # downstream consumer already shut down

    # --- Stats hook ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "iq_records":           self._n_iq_records,
            "iq_samples":           self._n_iq_samples,
            "detection_events":     self._n_detection_events,
            "decode_attempts":      self._n_decode_attempts,
            "decode_completed":     self._n_decode_completed,
            "decode_dropped_full":  self._n_decode_dropped_full,
            "ring_pos":             self._ring_pos,
            "abs_sample":           self._abs_sample,
            "ring_gen":             self._ring_gen,
        })
        return base
