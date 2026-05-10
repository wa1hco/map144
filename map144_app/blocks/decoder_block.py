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
"""DecoderBlock — Phase 4 #11.

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

Build history
-------------
- **Stage 1 (commit 23a55d2)**: structural scaffold — IQ ring fill,
  detection-event drain, worker dispatch with semaphore + daemon
  threads, stub worker, 8 unit tests.
- **Stage 2 (this commit)**: workers call ``detection.extract_and_decode``
  for real SPD + jt9 decode work.  The function is unchanged; we
  provide it with a per-block ring + ring_state callbacks so it sees
  the same interface it sees today from the legacy engine.  This
  minimises risk: the DSP / SPD / jt9 path is bit-for-bit identical
  between legacy and Block-form runs, validated by the §10 bit-exact
  replay gate.  ``_DecodeQueueShim`` adapts the function's
  ``decode_queue.put({…})`` calls into ``DecodeEvent`` emissions on
  the block's output stream.

Out of scope (separate TODOs)
-----------------------------
- **Period-mode decode** for weak-signal propagation modes (forward
  scatter, tropo, airplane, sporadic-E).  See TODO ``#42`` and
  ``project_propagation_modes`` memory.  IQ ring is sized to 15 s
  (a full WSJT period) to ready it for that work; burst mode here
  only reads ±1.7 s.
- **Period-end attempt-tracker / signature-aware claim** for
  per-(channel, period) dedup.  Lands with ``#42`` and ``#36``.

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

from ..detection import extract_and_decode
from .block import Block
from .types import BlockConfig, Event, Record, StreamClosed

log = logging.getLogger(__name__)


#: Decode-event ``kind`` field convention.  Single value to keep the
#: protocol surface narrow; payload differentiates outcomes.
DECODE_EVENT_KIND = "decode"


@dataclass
class DecoderBlockConfig(BlockConfig):
    """Per-DecoderBlock typed config."""

    # Ports.
    iq_port_name:        str = "iq"
    detection_port_name: str = "detections"
    decode_port_name:    str = "decodes"

    # IQ ring sizing.
    sample_rate_hz: int   = 48_000
    ring_seconds:   float = 15.0         # holds ~15 s of recent IQ — one
                                          # full WSJT MSK144 period.  Burst
                                          # mode (Stage 2) only reads ±1.7 s
                                          # around each detection event, so
                                          # the extra ~13 s is overhead for
                                          # current behaviour.  The full-
                                          # period reach is required by
                                          # period-mode decoding (TODO #42)
                                          # which integrates the entire
                                          # 15-s WSJT period to recover
                                          # weak-signal-mode reception
                                          # (forward scatter / tropo /
                                          # airplane / sporadic-E).  Memory
                                          # cost ~5.8 MB per polarisation,
                                          # negligible.

    # Decode-worker pool.  Default 4 matches legacy ``_jt9_threads``
    # semaphore width — jt9 subprocess is the bottleneck so this also
    # caps CPU for the dense-burst case.
    max_concurrent_decodes: int = 4

    # Output directory for saved decode WAVs (decoded events only).
    # Empty string disables WAV save (useful for tests).
    output_dir: str = ""

    # Receiver tuned-centre frequency (MHz) for log/dial reporting.
    # Combined with the detection event's fc_hz to compute the on-air
    # carrier frequency for decode logging and PSKReporter spotting.
    center_freq_mhz: float = 50.260

    # Dual-polarisation flag.  When True the ring is expected to carry
    # an interleaved H/V signal (legacy convention); CombinerBlock
    # already merges to mono before this block in the new pipeline,
    # so default False.
    dual_pol: bool = False

    # Optional override for jt9 CLI args; None → JT9_BASE_ARGS from
    # msk144_spd.  Useful for test runs (-d 0 / -p 15 etc.).
    jt9_args: list | None = None

    # Get timeout for blocking stream reads in tick() — short enough
    # for responsive shutdown.
    get_timeout_s: float = 0.250


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
        self._n_iq_gaps:            int = 0   # times rec.start_sample > _abs_sample
        self._n_iq_gap_samples:     int = 0   # total samples lost to gaps
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
            rec_start = int(rec.start_sample)
            if self._iq_t0_wall is None:
                # First record sets the anchors:
                # - ``_iq_t0_wall``  : source wall-clock at sample 0
                # - ``_abs_sample``  : engine-counter at the start of
                #                      this record (so the +=n below
                #                      lands at rec_start + n)
                # - ``_ring_pos``    : engine-counter modulo ring size
                #                      (preserves abs↔ring alignment)
                self._iq_t0_wall = (
                    rec.wall_clock_at_production
                    - rec_start / float(rec.sample_rate_hz)
                )
                self._abs_sample = rec_start
                self._ring_pos = rec_start % self._ring_n
            else:
                # Gap detection.  Under sustained load the engine's
                # ``iq_in`` stream uses POLICY_DROP_OLDEST and may evict
                # records — the ones that DO arrive carry their original
                # ``rec.start_sample`` (the engine's monotonic
                # sample-counter).  ``_abs_sample`` was the engine
                # position immediately AFTER the previous record; if
                # ``rec_start`` is higher, that delta is the dropped-
                # samples count.  Advance ``_ring_pos`` by the gap so
                # the engine-position ↔ ring-offset relationship stays
                # consistent — detection events are anchored to the
                # engine counter (via the channelizer's gap-recovery
                # logic) and decode workers convert them to ring
                # offsets via ``ring_state_fn`` + ``_read_ring``.  Stale
                # ring data in the gap range will be returned for any
                # read that spans the gap; that's acceptable (the
                # data was lost), and strictly better than the
                # pre-fix behaviour where decode workers read good
                # samples at the WRONG ring offsets and SPD/jt9
                # silently failed on every burst after the first
                # drop.
                gap = rec_start - self._abs_sample
                if gap > 0:
                    self._ring_pos = (self._ring_pos + gap) % self._ring_n
                    self._abs_sample = rec_start
                    self._n_iq_gap_samples += gap
                    self._n_iq_gaps += 1
                elif gap < 0:
                    # Out-of-order or duplicate record — drop silently.
                    # Real radios don't reorder, so this only fires on
                    # test injection; logging would be noise on live
                    # traffic.
                    return

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

        Calls into ``map144_app.detection.extract_and_decode`` — the
        same battle-tested SPD + jt9 fallback path the legacy engine
        uses.  We pass the block's ring + ``_ring_state`` / ``_ring_gen_get``
        callbacks so the function sees the same interface it consumes
        from the legacy engine today; the DSP / SPD / jt9 path is thus
        bit-for-bit identical between legacy and Block-form runs
        (required for the §10 bit-exact replay validation gate).

        Outcomes from extract_and_decode arrive through a queue shim
        (``_DecodeQueueShim``) that converts ``decode_queue.put({…})``
        calls into ``_emit_decode_event`` calls on this block's
        ``decodes`` output stream.

        Burst-mode only for now (1.7-s window).  Period-mode decoding
        for weak-signal propagation modes (forward scatter / tropo /
        airplane / Es) is TODO ``#42`` — see ``project_propagation_modes``
        memory.  Per-(channel, period) attempt-tracking added there;
        plug-in point is here, after this burst-mode call returns
        no_decode and CPU budget allows.
        """
        try:
            # Quick generation check before wasting work — if the source
            # was switched between dispatch and worker start, exit.
            if self._ring_gen_get() != gen_at_dispatch:
                return

            cfg: DecoderBlockConfig = self.config  # type: ignore[assignment]

            # Construct a per-launch detect_ts string in the same shape
            # the legacy engine uses (UTC '%Y-%m-%d_%H:%M:%S.%f' truncated
            # to ms).  Use the detection event's wall clock so the stamp
            # tracks signal time, not worker-thread time.
            from datetime import datetime, timezone
            det_dt = datetime.fromtimestamp(
                det_evt.wall_clock_at_production, tz=timezone.utc,
            )
            detect_ts = det_dt.strftime('%Y-%m-%d_%H:%M:%S.%f')[:21]

            # marker_id — used by the legacy GUI overlay to correlate a
            # heatmap circle with its decode outcome.  In the Block path
            # the equivalent role is played by the DecodeEvent → marker
            # join in DisplayBlock; we still pass a unique int so the
            # extract_and_decode internals work unchanged.
            marker_id = id(det_evt) & 0x7FFF_FFFF

            # Queue shim: converts decode_queue.put({...}) calls into
            # DecodeEvent emissions on this block's output stream.
            shim = _DecodeQueueShim(self, det_evt)

            # Forward any per-launch metric fields the detection event
            # carries (n_chans_300ms, sync_phase_coherence_h, etc.) to
            # extract_and_decode so they end up in launches.jsonl.
            launch_metrics = {
                k: v for k, v in (det_evt.payload or {}).items()
                if k not in {
                    "channel_index", "fc_hz", "source", "cluster_channels",
                    "cluster_span_ch", "pair_metric_db",
                    "sync_metric_db", "sq_metric_db",
                }
            } or None

            # Lookup via module attribute (not the local name binding) so
            # tests can monkey-patch ``decoder_block.extract_and_decode``
            # to a stub.  Production runs see the real function.
            import sys as _sys
            _decode_fn = getattr(
                _sys.modules[__name__], "extract_and_decode",
            )

            # Detection events arrive with ``occurred_at_sample`` indexed
            # at the DETECTOR's sample rate (typically 12 kHz, the
            # channelized rate), but ``extract_and_decode`` reads from
            # the IQ ring at the IQ sample rate (typically 48 kHz) and
            # interprets ``detect_sample`` in that rate.  Convert so the
            # decoder's ring-window read targets the actual position of
            # the signal in IQ samples.  This is the contract documented
            # in §6 of docs/block-stream-design.md: each event carries
            # its own sample_rate_hz; consumers must rebase to their
            # local clock when correlating with their own stream.
            det_sr = int(det_evt.sample_rate_hz) or int(cfg.sample_rate_hz)
            iq_sr  = int(cfg.sample_rate_hz)
            detect_sample_iq = int(
                det_evt.occurred_at_sample * iq_sr // det_sr
            )

            _decode_fn(
                iq_ring=self._iq_ring,
                ring_state_fn=self._ring_state,
                detect_sample=detect_sample_iq,
                sample_rate=int(cfg.sample_rate_hz),
                fc_hz=float(det_evt.payload.get('fc_hz', 0.0)),
                output_dir=cfg.output_dir,
                t_in_window=0.0,
                decode_queue=shim,
                marker_id=marker_id,
                ring_gen=gen_at_dispatch,
                ring_gen_fn=self._ring_gen_get,
                center_freq_mhz=cfg.center_freq_mhz,
                detect_ts=detect_ts,
                jt9_args=cfg.jt9_args,
                dual_pol=cfg.dual_pol,
                iq_t0_wall=self._iq_t0_wall,
                det_source=str(det_evt.payload.get('source', 'sq')),
                launch_metrics=launch_metrics,
            )
        except Exception:
            log.exception(
                "%s: decode worker failed for detection at sample %d",
                self.name, det_evt.occurred_at_sample,
            )
            # Emit an error event so downstream consumers see something.
            try:
                self._emit_decode_event(
                    det_evt=det_evt,
                    outcome="error",
                    message="",
                    payload_extra={},
                )
            except Exception:
                pass
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
            "iq_gaps":              self._n_iq_gaps,
            "iq_gap_samples":       self._n_iq_gap_samples,
            "detection_events":     self._n_detection_events,
            "decode_attempts":      self._n_decode_attempts,
            "decode_completed":     self._n_decode_completed,
            "decode_dropped_full":  self._n_decode_dropped_full,
            "ring_pos":             self._ring_pos,
            "abs_sample":           self._abs_sample,
            "ring_gen":             self._ring_gen,
        })
        return base


class _DecodeQueueShim:
    """Adapter that lets ``extract_and_decode`` (legacy ``decode_queue.put``
    interface) emit DecodeEvents on a :class:`DecoderBlock`'s output
    stream without modification.

    Every dict that ``extract_and_decode`` puts on the queue contains an
    ``outcome`` key ("decoded" / "no_decode" / "timeout" / "error") plus
    optional fields (``message``, ``jt9_snr``, ``decoder``, etc.).  We
    forward all of them as ``payload_extra`` so downstream consumers
    (``ReporterBlock``, the future Decode-panel sub-feature of
    DisplayBlock) see the full information.
    """

    __slots__ = ("_block", "_det_evt")

    def __init__(self, block: "DecoderBlock", det_evt: Event):
        self._block = block
        self._det_evt = det_evt

    def put(self, item: dict) -> None:
        outcome = str(item.get("outcome", "unknown"))
        message = str(item.get("message", ""))
        # Strip fields we already place at top level / on the Event itself
        # so payload_extra carries only the genuinely new info.
        extra = {
            k: v for k, v in item.items()
            if k not in {"outcome", "message", "marker_id"}
        }
        # Drop large in-memory blobs — DecodeEventStream readers don't
        # want raw audio attached to every event (the WAV file already
        # carries the audio); keep marker_id for legacy correlation
        # though, by hoisting it in if present.
        extra.pop("audio", None)
        if "marker_id" in item:
            extra["marker_id"] = item["marker_id"]
        self._block._emit_decode_event(
            det_evt=self._det_evt,
            outcome=outcome,
            message=message,
            payload_extra=extra,
        )
