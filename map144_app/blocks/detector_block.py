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
"""DetectorBlock — Block-form sync correlator + squared-FFT detector + cluster gate.

Wraps the per-channel detection logic from
``map144_app/processing.py:516-889`` as a Phase 3 :class:`Block`.
**No DSP changes.**  Same sync correlator, same squared-FFT tone-pair
detector, same rolling 25th-percentile baseline, same cluster gate with
all four invariants from CLAUDE.md.

Detection invariants from CLAUDE.md (preserved here, marked in code):

1. **Cluster definition** — triggered channels with no gap > 3 channels
   form one cluster.  Two clusters with a gap > 3 can both launch
   independently.
2. **Only-best cooldown on launch** — only the cluster's best channel
   goes into cooldown when a launch fires.  Suppressing ±1 around the
   best blocks legitimate adjacent-channel signals.
3. **Fresh-channel too-many gate** — the broadband gate
   (``_COINCIDENCE_MAX_CLUSTERS``) counts only *fresh* (non-cooldown)
   clusters.  Sidelobe remnants of a prior real detection must not
   inflate the gate count.
4. **No proximity guard** — the ±2-channel ``fc_hz`` proximity guard
   on active threads was deliberately removed; it incorrectly blocked
   legitimate adjacent-channel signals and same-frequency
   re-triggers.  Do not re-add.

Plus the sustained-signal lockout (``_SUSTAINED_HOPS``,
``_SUSTAINED_LOCKOUT``) and the per-cluster span gate
(``_COINCIDENCE_SPAN_CH``).

Out of scope for v1
-------------------
- Dual-pol path.  ``DualPolDetectorBlock`` (using :class:`CoherentPair`)
  is a follow-up.
- Display arrays (``_ch_snr_history`` etc.).  Under the Block model
  those move to the Display block, fed by an internal
  ``ChannelMetricStream``.  v1 does not emit channel metrics; the
  metrics are computed and consumed entirely inside the Detector for
  the launch decision.
- TX gating (``_tx_settle_remaining`` in the legacy code).  In the
  Block model this belongs upstream in the Source or in a TXGate block.

Threading
---------
The DetectorBlock owns its run thread per §4.  It reads ``ChannelStream``
records and emits ``DetectionEvent``\\s on its event-stream output.
Independence from the radio source thread is the whole architectural
point of Phase 3 — back-to-back pings can no longer stall ingest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

# Pull constants from the existing modules so the new path is bit-exact
# with the legacy detector.  These are part of the algorithm contract;
# any future tuning happens in one place.
from ..channelizer import CH_SAMPLE_RATE, N_CHANNELS
from ..msk144_spd import NSPM as _SYNC_NSPM
from ..msk144_spd import _sync_correlate_batch as _msk144_sync_correlate_batch
from ..processing import (
    CH_DETECT_SIZE,
    DETECT_MERGE_GAP_S,
    DETECT_THRESH_DB,
    _COINCIDENCE_MAX_CLUSTERS,
    _COINCIDENCE_SPAN_CH,
    _DC_CH_SKIP,
    _DETECT_WINDOW,
    _EDGE_CH_SKIP,
    _HI_MASK,
    _LO_MASK,
    _METRIC_HIST_DEPTH,
    _METRIC_HIST_STRIDE,
    _SUSTAINED_HOPS,
    _SUSTAINED_LOCKOUT,
)
from .block import Block
from .types import BlockConfig, ChannelMetricRecord, Event, Record, StreamClosed

# WSJT-X-faithful sq_det shadow logging (2026-05-16) — mirrors the same
# additive observation layer in processing.py.  Computes the new sq_det
# metric on the same channel window without affecting triggering.
# Disable: MAP144_SHADOW_SQDET=0
import os as _os
from ..sq_det import sq_det as _sq_det_new, to_db as _sq_to_db
_SHADOW_SQDET_ENABLED = _os.environ.get('MAP144_SHADOW_SQDET', '1') != '0'

log = logging.getLogger(__name__)


_CH_DETECT_HOP = CH_DETECT_SIZE // 2  # 256 — 50 % overlap stride

#: Default sources for ``DetectionEvent.payload['source']``.  Identifies
#: which detector(s) drove the launch — useful for downstream analysis
#: and for the eventual squared-FFT-detector deletion in Phase 5 #13.
_SOURCE_SYNC = "sync"
_SOURCE_SQ = "sq"
_SOURCE_BOTH = "both"


@dataclass
class DetectorBlockConfig(BlockConfig):
    """Per-DetectorBlock typed config."""

    # Ports.
    input_port_name: str = "channels"
    detection_port_name: str = "detections"
    #: Optional per-cycle metric stream, consumed by :class:`DisplayBlock`
    #: for the detection-metric heatmap and (later) by cluster-classifier
    #: blocks.  When the port is not connected, no metric records are
    #: produced — keeps the legacy bit-exact-replay validation path
    #: independent of Display state (§10).
    metric_port_name: str = "metrics"

    # DSP knobs — defaults match the legacy detector exactly.
    n_channels: int = N_CHANNELS
    sample_rate_hz: int = CH_SAMPLE_RATE                    # 12 000
    fft_size: int = CH_DETECT_SIZE                          # 512
    stride_samples: int = _CH_DETECT_HOP                    # 256 (50 % overlap)
    detect_thresh_db: float = DETECT_THRESH_DB              # 3.0
    detect_merge_gap_s: float = DETECT_MERGE_GAP_S          # 0.5
    enable_sync_detect: bool = True
    enable_sq_detect: bool = True
    edge_ch_skip: int = _EDGE_CH_SKIP
    # MAP144's detection invariant: the 48 kHz IQ at the engine input
    # always has clean DC, so ch_signed = 0, ±1, ±2 are LEGITIMATE
    # operating frequencies (real QSOs at calling±1/±2 kHz happen
    # routinely) and must NOT be suppressed.  DC handling lives at the
    # radio-config layer:
    #   - Flex (digital IQ): pan center IS calling freq, DC genuinely 0
    #   - B210 (analog IQ):   tune-offset + oversample + decimate moves
    #                          DC contamination out of band BEFORE MAP144
    #                          sees the IQ
    # Either way, the 48 kHz stream MAP144 processes has clean DC.
    # ``dc_ch_skip`` stays configurable as a last-resort hardware
    # escape hatch, but the default is 0.  See memory entry
    # ``project_dc_lo_offset`` and processing.py comment on _DC_CH_SKIP.
    dc_ch_skip: int = 0

    # Cluster gate parameters — see CLAUDE.md detection invariants.
    coincidence_max_clusters: int = _COINCIDENCE_MAX_CLUSTERS  # 6
    coincidence_span_ch: int = _COINCIDENCE_SPAN_CH            # 8
    sustained_hops: int = _SUSTAINED_HOPS                      # 2 s of hops
    sustained_lockout: int = _SUSTAINED_LOCKOUT                # 30 s of hops

    # Channel offset (Hz) used to compute fc for each channel index.
    # ``fc_hz_for_channel = k * channel_spacing_hz + channel_offset_hz``.
    channel_spacing_hz: float = 1000.0
    channel_offset_hz: float = 0.0

    # Get timeout — short enough for responsive shutdown.
    get_timeout_s: float = 0.250


class DetectorBlock(Block):
    """Sync + squared-FFT detector + cluster gate as a Block.

    The block accumulates ``ChannelStream`` records into per-channel
    rolling buffers, then runs detection every ``stride_samples``
    (``_CH_DETECT_HOP`` = 256).  Each detection cycle produces 0 or
    more :class:`Event`\\s on the output stream — one per cluster
    that survives the gates and launches a decode.

    Rolling buffers
    ---------------
    ``_ch_buf``    : ``(N_CHANNELS, fft_size)`` — squared-FFT window.
                     Kept exactly ``fft_size`` long; advanced by
                     ``stride_samples`` per detection cycle.
    ``_sync_buf``  : ``(N_CHANNELS, NSPM)`` — sync correlator window.
                     Always holds the most recent NSPM samples.

    Both are filled FIFO from incoming records and consumed at the
    same stride.  Cross-record continuity is preserved by retaining
    the rolling state.
    """

    config_type = DetectorBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Input-side rolling buffers.  Allocated in on_start once we
        # know the configured sizes.
        self._ch_buf: np.ndarray = None  # type: ignore[assignment]
        self._ch_buf_end: int = 0        # samples currently held
        self._sync_buf: np.ndarray = None  # type: ignore[assignment]
        self._sync_buf_filled: int = 0   # samples ever fed in (saturates at NSPM)

        # Rolling 25th-percentile state for the squared-FFT path.
        self._metric_hist_buf: np.ndarray = None  # type: ignore[assignment]
        self._metric_hist_idx: int = 0
        self._metric_hist_cnt: int = 0
        self._pct25: np.ndarray | None = None
        self._pct25_ctr: int = 0

        # Same for the sync path.
        self._sync_metric_hist_buf: np.ndarray = None  # type: ignore[assignment]
        self._sync_metric_hist_idx: int = 0
        self._sync_metric_hist_cnt: int = 0
        self._sync_pct25: np.ndarray | None = None
        self._sync_pct25_ctr: int = 0

        # Cluster-gate state.
        self._detect_cooldowns: dict[int, int] = {}   # ch_idx -> hops_remaining
        self._ch_above_thresh: dict[int, int] = {}    # ch_idx -> consecutive hops above threshold

        # Sample-counter tracking for input-side gap detection.
        self._n_in_consumed: int = 0   # input samples we've processed (drained from rolling buffers)

        # Sample-counter for the *output* event stream — anchored to the
        # input ``start_sample``.  ``occurred_at_sample`` on emitted
        # events references this counter.
        self._n_in_seen: int = 0

        # Stats.
        self._n_clusters_launched: int = 0
        self._n_clusters_too_wide: int = 0
        self._n_too_many_clusters_gates: int = 0
        self._n_sustained_lockouts: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: DetectorBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if config.fft_size != CH_DETECT_SIZE:
            raise ValueError(
                f"{self.name}: fft_size={config.fft_size} != CH_DETECT_SIZE={CH_DETECT_SIZE} "
                "— squared-FFT masks are pre-computed for CH_DETECT_SIZE only"
            )
        if config.stride_samples != _CH_DETECT_HOP:
            raise ValueError(
                f"{self.name}: stride_samples={config.stride_samples} != "
                f"_CH_DETECT_HOP={_CH_DETECT_HOP} — sustained_hops / cooldown_hops "
                "are derived from this stride"
            )

    def on_start(self) -> None:
        super().on_start()
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        nch = cfg.n_channels
        # Per-channel FFT window — held exactly fft_size samples wide.
        self._ch_buf = np.zeros((nch, cfg.fft_size), dtype=np.complex64)
        self._ch_buf_end = 0
        # Sync correlator window — NSPM samples.
        self._sync_buf = np.zeros((nch, _SYNC_NSPM), dtype=np.complex64)
        self._sync_buf_filled = 0

        # Rolling pct25 history buffers.
        self._metric_hist_buf = np.zeros(
            (_METRIC_HIST_DEPTH, nch), dtype=np.float32,
        )
        self._sync_metric_hist_buf = np.zeros(
            (_METRIC_HIST_DEPTH, nch), dtype=np.float32,
        )
        self._metric_hist_idx = 0
        self._metric_hist_cnt = 0
        self._sync_metric_hist_idx = 0
        self._sync_metric_hist_cnt = 0
        self._pct25 = None
        self._sync_pct25 = None
        self._pct25_ctr = 0
        self._sync_pct25_ctr = 0

        self._detect_cooldowns = {}
        self._ch_above_thresh = {}
        self._n_in_consumed = 0
        self._n_in_seen = 0

        log.info(
            "%s: %d ch x %d Hz, fft=%d, stride=%d, thresh=%.1f dB, "
            "sync=%s sq=%s",
            self.name, nch, cfg.sample_rate_hz, cfg.fft_size,
            cfg.stride_samples, cfg.detect_thresh_db,
            cfg.enable_sync_detect, cfg.enable_sq_detect,
        )

    def on_stop(self) -> None:
        super().on_stop()
        log.info(
            "%s: launches=%d, wide_clusters=%d, too_many_clusters_gates=%d, "
            "sustained_lockouts=%d",
            self.name,
            self._n_clusters_launched,
            self._n_clusters_too_wide,
            self._n_too_many_clusters_gates,
            self._n_sustained_lockouts,
        )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.input_port_name
        out_port = cfg.detection_port_name
        if in_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {in_port!r} not connected"
            )
        # detection port is optional — if not connected, events are dropped silently.

        try:
            rec: Record = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates and exits cleanly.

        if not isinstance(rec, Record):
            log.warning(
                "%s: dropped non-Record %r on input port",
                self.name, type(rec).__name__,
            )
            return

        if rec.sample_rate_hz != cfg.sample_rate_hz:
            raise RuntimeError(
                f"{self.name}: upstream record sample_rate_hz={rec.sample_rate_hz} "
                f"!= configured {cfg.sample_rate_hz}"
            )

        if rec.data.ndim != 2 or rec.data.shape[0] != cfg.n_channels:
            raise RuntimeError(
                f"{self.name}: upstream record shape {rec.data.shape} "
                f"!= ({cfg.n_channels}, n)"
            )

        # Gap-free check on input start_sample.
        expected_start = self._n_in_seen
        if rec.start_sample != expected_start:
            log.warning(
                "%s: input gap — got start_sample=%d, expected %d; "
                "resetting buffers",
                self.name, rec.start_sample, expected_start,
            )
            self._reset_after_gap(rec.start_sample)

        # Update sync buffer once with the whole record (it's a true FIFO).
        self._sync_ingest(rec.data)
        self._n_in_seen = rec.start_sample + rec.n_samples

        # For ``_ch_buf`` we interleave append with detection drain, because
        # a record may carry more samples than the remaining buffer space —
        # in which case we run detection cycles to make room as we go.
        # This loop is the bit-exact equivalent of the legacy "while
        # _ch_buf_end >= fft_size: detect, drain" combined with append.
        wall_clock_for_events = rec.wall_clock_at_production
        data = np.ascontiguousarray(rec.data, dtype=np.complex64)
        n_total = data.shape[1]
        cursor = 0
        while cursor < n_total:
            # Drain any complete window before appending more.
            while self._ch_buf_end >= cfg.fft_size:
                self._detect_one_cycle(wall_clock_for_events)
                self._advance_one_stride()
            space = cfg.fft_size - self._ch_buf_end
            take = min(space, n_total - cursor)
            self._ch_buf[:, self._ch_buf_end:self._ch_buf_end + take] = (
                data[:, cursor:cursor + take]
            )
            self._ch_buf_end += take
            cursor += take

        # Final drain pass after the record is fully ingested.
        while self._ch_buf_end >= cfg.fft_size:
            self._detect_one_cycle(wall_clock_for_events)
            self._advance_one_stride()

    # --- Helpers: buffer maintenance ---------------------------------

    def _reset_after_gap(self, new_start_sample: int) -> None:
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        nch = cfg.n_channels
        self._ch_buf[:] = 0
        self._ch_buf_end = 0
        self._sync_buf[:] = 0
        self._sync_buf_filled = 0
        # Don't clear pct25 — those are slow-rolling baselines and a
        # short input gap should not invalidate them.
        self._n_in_consumed = new_start_sample
        self._n_in_seen = new_start_sample

    def _sync_ingest(self, data: np.ndarray) -> None:
        """Update ``_sync_buf`` (true FIFO of the latest NSPM samples).

        ``_sync_buf`` is a pure rolling FIFO; ``_ch_buf`` is handled in
        :meth:`tick` because it must interleave with detection drains.
        """
        data = np.ascontiguousarray(data, dtype=np.complex64)
        new_n = data.shape[1]
        if new_n == 0:
            return
        if new_n >= _SYNC_NSPM:
            self._sync_buf[:] = data[:, -_SYNC_NSPM:]
            self._sync_buf_filled = _SYNC_NSPM
        else:
            self._sync_buf[:, :_SYNC_NSPM - new_n] = self._sync_buf[:, new_n:]
            self._sync_buf[:, _SYNC_NSPM - new_n:] = data
            self._sync_buf_filled = min(_SYNC_NSPM, self._sync_buf_filled + new_n)

    def _advance_one_stride(self) -> None:
        """Drain ``stride_samples`` from the front of ``_ch_buf``."""
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        s = cfg.stride_samples
        # Shift contents left by s.
        self._ch_buf[:, :cfg.fft_size - s] = self._ch_buf[:, s:cfg.fft_size]
        self._ch_buf[:, cfg.fft_size - s:] = 0
        # Clamp to non-negative.  ``_detect_one_cycle``'s cluster-gate
        # too-many branch resets ``_ch_buf_end`` to 0 inline (it wants a
        # clean restart after broadband suppression), and the caller
        # then runs us anyway since the outer ``while _ch_buf_end >=
        # fft_size`` loop entered before the reset.  Without the clamp
        # ``_ch_buf_end`` goes negative; the NEXT outer-loop iteration
        # mixes a negative start with a positive end in the slice
        # arithmetic — ``arr[:, -256:-256+512]`` becomes
        # ``arr[:, 256:256]`` (empty) and we crash trying to write a
        # full-size record into a zero-width slot.  Caught live during
        # the 2026-05-11 shadow-runtime run, ~17 s after start, when
        # the cluster gate fired its first too_many on a burst.
        self._ch_buf_end = max(0, self._ch_buf_end - s)
        self._n_in_consumed += s

    # --- Helpers: per-detector metrics --------------------------------

    def _compute_sq_metric_db(self, ch_block: np.ndarray) -> np.ndarray:
        """Squared-FFT tone-pair detector — mirrors the legacy
        ``processing.py:516-558`` math.  Returns dB above the rolling
        25th-percentile baseline.

        Also (when ``_SHADOW_SQDET_ENABLED``) maintains a parallel rolling
        history of MAX-combine ``new_raw_lin`` values per channel, for
        use by ``_emit_detection_event``'s WSJT-X-shadow logging.  Same
        shape as ``_metric_hist_buf``, same update rule.
        """
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        sq = ch_block ** 2  # complex squaring → tones at ±2*offset Hz
        x_sq = np.fft.fft(sq * _DETECT_WINDOW[np.newaxis, :], axis=1)
        power_lin = np.abs(x_sq) / cfg.fft_size

        lo_peak = power_lin[:, _LO_MASK].max(axis=1)
        hi_peak = power_lin[:, _HI_MASK].max(axis=1)
        raw_lin = (lo_peak + hi_peak) / 2.0

        # Shadow: WSJT-X-style MAX combine + parallel rolling history.
        # Stored on self so _emit_detection_event can compute detmet_norm
        # and the three noise-reference variants at launch time.
        if _SHADOW_SQDET_ENABLED:
            self._new_raw_lin = np.maximum(lo_peak, hi_peak)  # (48,) latest hop
            if not hasattr(self, '_metric_hist_buf_new'):
                self._metric_hist_buf_new = np.zeros_like(self._metric_hist_buf)
                self._metric_hist_idx_new = 0
                self._metric_hist_cnt_new = 0
            self._metric_hist_buf_new[self._metric_hist_idx_new] = self._new_raw_lin
            self._metric_hist_idx_new = (self._metric_hist_idx_new + 1) % _METRIC_HIST_DEPTH
            if self._metric_hist_cnt_new < _METRIC_HIST_DEPTH:
                self._metric_hist_cnt_new += 1

        # Rolling baseline.
        self._metric_hist_buf[self._metric_hist_idx] = raw_lin
        self._metric_hist_idx = (self._metric_hist_idx + 1) % _METRIC_HIST_DEPTH
        if self._metric_hist_cnt < _METRIC_HIST_DEPTH:
            self._metric_hist_cnt += 1
        hist = (
            self._metric_hist_buf[:self._metric_hist_cnt]
            if self._metric_hist_cnt < _METRIC_HIST_DEPTH
            else self._metric_hist_buf
        )
        self._pct25_ctr += 1
        if self._pct25_ctr % 10 == 0 or self._pct25 is None:
            hist_s = hist[::_METRIC_HIST_STRIDE]
            k = max(0, int(len(hist_s) * 0.25))
            tmp = hist_s.copy()
            tmp.partition(k, axis=0)
            self._pct25 = np.maximum(tmp[k], 1e-30)

        return (10.0 * np.log10(np.maximum(raw_lin / self._pct25, 1e-30))).astype(np.float32)

    def _compute_sync_metric_db(self) -> np.ndarray:
        """Coherent sync correlator — mirrors
        ``processing.py:_compute_sync_metric_db`` with state on
        ``self`` instead of an ``engine`` instance.  Uses the batched
        ``_sync_correlate_batch`` for all 48 channels in one BLAS call.
        """
        sync_window = np.ascontiguousarray(self._sync_buf, dtype=np.complex64)
        _, peaks, _ = _msk144_sync_correlate_batch(sync_window)

        self._sync_metric_hist_buf[self._sync_metric_hist_idx] = peaks
        self._sync_metric_hist_idx = (
            (self._sync_metric_hist_idx + 1) % _METRIC_HIST_DEPTH
        )
        if self._sync_metric_hist_cnt < _METRIC_HIST_DEPTH:
            self._sync_metric_hist_cnt += 1
        hist = (
            self._sync_metric_hist_buf[:self._sync_metric_hist_cnt]
            if self._sync_metric_hist_cnt < _METRIC_HIST_DEPTH
            else self._sync_metric_hist_buf
        )
        self._sync_pct25_ctr += 1
        if self._sync_pct25_ctr % 10 == 0 or self._sync_pct25 is None:
            hist_s = hist[::_METRIC_HIST_STRIDE]
            k = max(0, int(len(hist_s) * 0.25))
            tmp = hist_s.copy()
            tmp.partition(k, axis=0)
            self._sync_pct25 = np.maximum(tmp[k], 1e-30)

        return (10.0 * np.log10(np.maximum(peaks / self._sync_pct25, 1e-30))).astype(np.float32)

    # --- Per-cycle detection ------------------------------------------

    def _detect_one_cycle(self, wall_clock_for_events: float) -> None:
        """Run one detection cycle on the current fft-window and sync window."""
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        ch_block = self._ch_buf[:, :cfg.fft_size]

        sq_metric = (
            self._compute_sq_metric_db(ch_block)
            if cfg.enable_sq_detect
            else np.full(cfg.n_channels, -999.0, dtype=np.float32)
        )
        sync_metric = (
            self._compute_sync_metric_db()
            if cfg.enable_sync_detect
            else np.full(cfg.n_channels, -999.0, dtype=np.float32)
        )
        # Combined pair_metric per channel.
        pair_metric = np.maximum(sq_metric, sync_metric)

        # Emit a ChannelMetricRecord for the Display heatmap (and future
        # cluster-classifier blocks).  Unconditional on triggers — the
        # heatmap needs every frame.  Skipped if the metrics output port
        # is not connected, so legacy bit-exact-replay tests that wire
        # only the detection port are unaffected.
        self._emit_metric_record(
            sq_metric=sq_metric,
            sync_metric=sync_metric,
            pair_metric=pair_metric,
            wall_clock_at_production=wall_clock_for_events,
        )

        # Tick down cooldowns.
        self._detect_cooldowns = {
            k: v - 1 for k, v in self._detect_cooldowns.items() if v > 1
        }

        # Suppress edge / DC channels.
        #
        # Nyquist edge: the legacy "_nyquist_ch = N_CHANNELS // 2" only
        # works when channel_offset_hz is exactly half the spacing.  In
        # the USB-convention case (channel_offset_hz=1500, +1.5 channels
        # off-centre), the actual IF Nyquist sits at
        #     n_channels/2 - channel_offset_hz/channel_spacing_hz
        # ≈ 22.5 for the typical 48-channel × 1 kHz bank.  Suppressing
        # ch 0..3 and 44..47 instead of [22.5 ± edge_skip] leaves the
        # rolloff zone (ch_k ≈ 19 in IF, ch_signed +19 on-air) unmasked
        # — observed in launch-pattern analysis as a ~6× false-trigger
        # rate at 50279 kHz (rule "PERSISTENT_NODECODE", memory entry
        # ``project_launch_pattern_findings``).  Fix landed in
        # processing.py commit b318072; we propagate the same wrap-aware
        # Nyquist computation here so the Block path matches.
        #
        # DC: channel 0 is DC plus LO leakage; legacy suppresses ch 0
        # through dc_skip inclusive.
        edge_skip = cfg.edge_ch_skip
        dc_skip = cfg.dc_ch_skip
        nch = cfg.n_channels
        suppress_mask = np.zeros(nch, dtype=bool)

        if edge_skip > 0:
            ch_offset_in_channels = (
                cfg.channel_offset_hz / float(cfg.channel_spacing_hz)
            )
            nyquist_ch_real = (nch / 2.0) - ch_offset_in_channels
            ch_idx_arr = np.arange(nch, dtype=np.float64)
            # Wrap-aware distance (handles channels near both ends of
            # the FFT folding to the same IF location).
            dist_to_nyquist = np.minimum(
                np.minimum(
                    np.abs(ch_idx_arr - nyquist_ch_real),
                    np.abs(ch_idx_arr - nyquist_ch_real - nch),
                ),
                np.abs(ch_idx_arr - nyquist_ch_real + nch),
            )
            suppress_mask |= (dist_to_nyquist <= edge_skip)
        if dc_skip > 0:
            suppress_mask[:dc_skip + 1] = True   # ch 0 is DC; skip 0..dc_skip inclusive
        eligible = pair_metric.copy()
        eligible[suppress_mask] = -999.0

        # Gather triggered channels, sorted strongest-first so the
        # cluster gate operates on a sensible order.
        triggered = np.where(eligible > cfg.detect_thresh_db)[0]
        if triggered.size > 1:
            triggered = triggered[eligible[triggered].argsort()[::-1]]
        if triggered.size == 0:
            self._update_sustained_state(triggered)
            return

        # Run the cluster gate (preserves CLAUDE.md invariants 1..4).
        launches = self._cluster_gate(triggered, eligible)

        # Emit events for any surviving launches.
        for launch_ch, cluster_chs in launches:
            self._emit_detection_event(
                launch_ch=launch_ch,
                cluster_chs=cluster_chs,
                sync_metric=sync_metric,
                sq_metric=sq_metric,
                pair_metric=pair_metric,
                wall_clock_at_production=wall_clock_for_events,
            )
            self._n_clusters_launched += 1

        self._update_sustained_state(triggered)

    def _cluster_gate(
        self,
        triggered: np.ndarray,
        pair_metric: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """Faithful port of ``processing.py:700-789``.

        Splits triggered channels into clusters (gap > 3 = new cluster).
        Applies the broadband-noise gate (count of fresh clusters), the
        per-cluster span gate, and selects the best channel from each
        surviving cluster.  Returns ``[(launch_ch, cluster_chs), ...]``.

        Invariants from CLAUDE.md (numbered to match the module
        docstring):

        1. Cluster definition — gap > 3 channels = new cluster.
        2. Only-best cooldown on launch — the launched channel is
           cooled down; neighbours are NOT cooled.
        3. Fresh-channel too-many gate — only fresh clusters count
           toward the broadband gate threshold.
        4. No proximity guard — no ``±N`` suppression on active
           threads.
        """
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]

        cooldown_hops = max(
            1, int(cfg.detect_merge_gap_s * cfg.sample_rate_hz / cfg.stride_samples),
        )

        sorted_idx = triggered.astype(int)
        sorted_idx.sort()

        # ── Invariant 1: cluster definition ──────────────────────────
        clusters: list[np.ndarray] = []
        if sorted_idx.size:
            cur = [int(sorted_idx[0])]
            for ci in sorted_idx[1:]:
                if int(ci) - cur[-1] <= 3:
                    cur.append(int(ci))
                else:
                    clusters.append(np.array(cur, dtype=int))
                    cur = [int(ci)]
            clusters.append(np.array(cur, dtype=int))

        # ── Invariant 3: fresh-channel too-many gate ─────────────────
        fresh_clusters = [
            cl for cl in clusters
            if int(cl[np.argmax(pair_metric[cl])]) not in self._detect_cooldowns
        ]
        too_many = len(fresh_clusters) >= cfg.coincidence_max_clusters
        if too_many:
            # Broadband noise — suppress all triggered, clear FFT
            # windows so the next cycle restarts from a known state.
            for ch in triggered:
                self._detect_cooldowns[int(ch)] = cooldown_hops
            self._ch_buf[:] = 0
            self._ch_buf_end = 0
            self._n_too_many_clusters_gates += 1
            log.info(
                "%s: too_many_clusters gate fired (%d fresh ≥ %d) — broadband suppress",
                self.name, len(fresh_clusters), cfg.coincidence_max_clusters,
            )
            return []

        # ── Per-cluster span gate + Invariant 2 (only-best cooldown) ──
        launches: list[tuple[int, np.ndarray]] = []
        for cl in clusters:
            cl_span = int(cl[-1]) - int(cl[0])
            if cl_span > cfg.coincidence_span_ch:
                # Wide cluster = broadband noise; suppress all its channels.
                for ch in cl:
                    self._detect_cooldowns[int(ch)] = cooldown_hops
                self._n_clusters_too_wide += 1
                continue

            best = int(cl[np.argmax(pair_metric[cl])])
            if best not in self._detect_cooldowns:
                # Launch: only the best channel goes into cooldown.
                # Suppressing neighbours would block legitimate adjacent
                # signals (Invariant 2).
                launches.append((best, cl))
                self._detect_cooldowns[best] = cooldown_hops
            else:
                # Best is already cooled (decode in progress / just
                # launched).  Refresh the cooldown for any *already-cooled*
                # cluster members so they don't suddenly become triggers
                # on the next cycle, but DO NOT add fresh members — a
                # fresh member may be the real signal centre that becomes
                # the cluster's best on a subsequent cycle.
                for ch in cl:
                    if int(ch) in self._detect_cooldowns:
                        self._detect_cooldowns[int(ch)] = cooldown_hops

        return launches

    def _update_sustained_state(self, triggered: np.ndarray) -> None:
        """Maintain the sustained-signal counter and apply the long
        lockout when a channel stays above threshold for too long.
        Mirrors ``processing.py:776-789``.
        """
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        triggered_set = {int(c) for c in triggered}
        for ch in range(cfg.n_channels):
            if ch in triggered_set:
                self._ch_above_thresh[ch] = self._ch_above_thresh.get(ch, 0) + 1
                if self._ch_above_thresh[ch] == cfg.sustained_hops:
                    log.info(
                        "%s: ch %d sustained signal — locking out for ~%d s",
                        self.name, ch,
                        int(cfg.sustained_lockout * cfg.stride_samples / cfg.sample_rate_hz),
                    )
                    self._detect_cooldowns[ch] = cfg.sustained_lockout
                    self._n_sustained_lockouts += 1
            else:
                self._ch_above_thresh.pop(ch, None)

    # --- Event emission -----------------------------------------------

    def _fc_hz_for_channel(self, ch_idx: int) -> float:
        """Convert unsigned FFT-bin channel index to baseband fc_hz.

        Channel indices 0..N/2 represent positive-frequency offsets from
        IQ DC; indices N/2+1..N-1 wrap to negative offsets.  The legacy
        path does the same mapping (processing.py:952), so a signal at
        ch_idx=42 in a 48-channel bank actually sits at -6 kHz from DC,
        not +42 kHz.  Without the wrap, fc_hz for negative-side channels
        comes out 48 kHz too high and extract_and_decode mixes the
        signal to the wrong audio frequency — no decode possible.
        """
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        n = int(cfg.n_channels)
        ch_signed = int(ch_idx) if int(ch_idx) <= n // 2 else int(ch_idx) - n
        return float(ch_signed) * cfg.channel_spacing_hz + cfg.channel_offset_hz

    def _emit_metric_record(
        self,
        sq_metric: np.ndarray,
        sync_metric: np.ndarray,
        pair_metric: np.ndarray,
        wall_clock_at_production: float,
    ) -> None:
        """Emit a per-cycle :class:`ChannelMetricRecord` for Display.

        Mono path: ``pair_metric_db_v == pair_metric_db_h``.  The dual-pol
        DetectorBlock equivalent will emit distinct H / V arrays.

        ``start_sample`` is the position of the FFT window front
        (``self._n_in_consumed``) at the moment the cycle ran.  The next
        cycle's ``start_sample`` advances by ``stride_samples`` — gap-free
        by construction with the legacy hop-counter row index.
        """
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        port = cfg.metric_port_name
        if port not in self.outputs:
            return  # metric sink is optional

        sq_h = sq_metric if cfg.enable_sq_detect else None
        sync_h = sync_metric if cfg.enable_sync_detect else None

        # ChannelMetricRecord inherits ``data`` from Record (the primary
        # heatmap metric) plus per-polarisation sub-metrics.  Mono
        # detector populates _v == _h so consumers don't branch.
        rec = ChannelMetricRecord(
            data=pair_metric.astype(np.float32, copy=False),
            sample_rate_hz=cfg.sample_rate_hz,
            start_sample=self._n_in_consumed,
            wall_clock_at_production=wall_clock_at_production,
            metadata={
                "stride_samples": cfg.stride_samples,
                "fft_size": cfg.fft_size,
            },
            pair_metric_db_h=pair_metric.astype(np.float32, copy=False),
            pair_metric_db_v=pair_metric.astype(np.float32, copy=False),
            sq_metric_db_h=sq_h.astype(np.float32, copy=False) if sq_h is not None else None,
            sq_metric_db_v=sq_h.astype(np.float32, copy=False) if sq_h is not None else None,
            sync_metric_db_h=sync_h.astype(np.float32, copy=False) if sync_h is not None else None,
            sync_metric_db_v=sync_h.astype(np.float32, copy=False) if sync_h is not None else None,
        )
        try:
            self.outputs[port].put(rec)
        except StreamClosed:
            raise StopIteration

    def _emit_detection_event(
        self,
        launch_ch: int,
        cluster_chs: np.ndarray,
        sync_metric: np.ndarray,
        sq_metric: np.ndarray,
        pair_metric: np.ndarray,
        wall_clock_at_production: float,
    ) -> None:
        cfg: DetectorBlockConfig = self.config  # type: ignore[assignment]
        port = cfg.detection_port_name
        if port not in self.outputs:
            return  # detection-event sink is optional

        # Determine which detector(s) drove this launch.
        sync_db = float(sync_metric[launch_ch])
        sq_db = float(sq_metric[launch_ch])
        thresh = cfg.detect_thresh_db
        if sync_db > thresh and sq_db > thresh:
            source = _SOURCE_BOTH
        elif sync_db > thresh:
            source = _SOURCE_SYNC
        else:
            source = _SOURCE_SQ

        payload = {
            "channel_index": int(launch_ch),
            "fc_hz": self._fc_hz_for_channel(launch_ch),
            "pair_metric_db": float(pair_metric[launch_ch]),
            "sync_metric_db": sync_db,
            "sq_metric_db": sq_db,
            "source": source,
            "cluster_channels": [int(c) for c in cluster_chs],
            "cluster_span_ch": int(cluster_chs[-1] - cluster_chs[0]),
        }

        # ── SHADOW: WSJT-X-faithful sq_det metrics on the same window ────
        # Mirrors processing.py's per-launch shadow logging so both pipelines
        # generate the same comparison data.  Disable: MAP144_SHADOW_SQDET=0.
        # The sync window for the new sq_det() is the most recent NSPM samples
        # of the channel buffer.  detector_block keeps its own _sync_buf via
        # the parent class — same shape and update rule as legacy.
        if _SHADOW_SQDET_ENABLED and hasattr(self, '_sync_buf'):
            try:
                _new_h = _sq_det_new(
                    np.ascontiguousarray(self._sync_buf[launch_ch], dtype=np.complex128),
                    fc=0.0,
                )
                _new_h_fc_off = (_new_h.ferr_hi if _new_h.ah >= _new_h.al
                                 else _new_h.ferr_lo)
                payload['sq_new_detmet_h']       = round(float(_new_h.detmet),  4)
                payload['sq_new_detmet2_lin_h']  = round(float(_new_h.detmet2), 3)
                payload['sq_new_detmet2_db_h']   = round(float(_sq_to_db(_new_h.detmet2)), 2)
                payload['sq_new_ah_h']           = round(float(_new_h.ah),      4)
                payload['sq_new_al_h']           = round(float(_new_h.al),      4)
                payload['sq_new_fc_offset_hz_h'] = round(float(_new_h_fc_off),  2)

                # Three noise references on the same hop (cf processing.py)
                if hasattr(self, '_new_raw_lin'):
                    _now = float(self._new_raw_lin[launch_ch])
                    payload['sq_new_raw_lin_h'] = round(_now, 4)

                    if getattr(self, '_metric_hist_cnt_new', 0) > 0:
                        _hist = self._metric_hist_buf_new[:self._metric_hist_cnt_new, launch_ch]
                        _hist_s = _hist[::_METRIC_HIST_STRIDE]
                        _pct25_chan_rolling = float(np.percentile(_hist_s, 25)) if len(_hist_s) else 0.0
                    else:
                        _pct25_chan_rolling = 0.0
                    payload['sq_new_pct25_chan_rolling_h'] = round(_pct25_chan_rolling, 6)

                    payload['sq_new_pct25_cross_chan_h'] = round(
                        float(np.percentile(self._new_raw_lin, 25)), 6
                    )

                    _denom = max(_pct25_chan_rolling, 1e-30)
                    _detmet_norm = _now / _denom
                    payload['sq_new_detmet_norm_h']    = round(_detmet_norm, 3)
                    payload['sq_new_detmet_norm_db_h'] = round(float(_sq_to_db(_detmet_norm)), 2)

                if self._pct25 is not None:
                    payload['sq_new_pct25_chan_legacy_h'] = round(
                        float(self._pct25[launch_ch]), 6
                    )
                # Wideband floor lives on the engine; block path doesn't have
                # direct access.  Left absent — engine reference can be wired
                # later when the engine/block bridge is formalised.
            except Exception as _exc:    # noqa: BLE001
                payload['sq_new_error_h'] = str(_exc)[:80]

        event = Event(
            kind="detection",
            occurred_at_sample=self._n_in_consumed,
            sample_rate_hz=cfg.sample_rate_hz,
            wall_clock_at_production=wall_clock_at_production,
            payload=payload,
        )
        try:
            self.outputs[port].put(event)
        except StreamClosed:
            raise StopIteration
