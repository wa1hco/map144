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
"""DisplayBlock — channel-metric heatmap + detection-marker state.

See ``docs/block-stream-design.md`` §6 (Display row) and §9 #6 (display
tick rate).  This is the v1 / "#10a" scope: the channel-metric heatmap
plus the detection-marker overlay, fed by the new
:class:`ChannelMetricRecord` stream from :class:`DetectorBlock` and the
existing :class:`Event`-based ``DetectionEventStream``.

Out of scope for v1 (deferred to follow-on tasks):

- IQ waterfall (sub-feature c) — needs the ``IQStream`` Tap, lands with
  ``#11`` Decoder which also taps IQStream.
- Squared-domain spectrogram (sub-feature d) — additional stream from
  Detector; defer.
- Decode panel (sub-feature f) — needs ``DecodeEventStream`` from
  :class:`DecoderBlock` (``#11``).
- Status bar (sub-feature g) — already covered by ``Block.stats()``
  (§9 #8); separate concern.

Boundary handling
-----------------
The 15-second period boundary is **wall-clock-defined** by protocol —
all MSK144 / WSJT-X stations use UTC seconds divisible by 15 as the
period grid.  The legacy code at ``processing.py:628-647`` deliberately
uses ``int(wall_clock / history_secs)`` to detect boundary cross, then
resets the hop counter to 0 so the first frame of every period lands at
row 0.

Within a period, the heatmap row index is ``hop_count % N_SNR_HIST``,
where ``hop_count`` advances by 1 per :class:`ChannelMetricRecord`
arriving from the Detector.  Each metric record corresponds to one
detector stride; consecutive records always fill consecutive rows.  This
mirrors the legacy `processing.py` logic exactly and avoids the rare
"sample-clock-to-row map" edge case that left row 0 dark for an entire
period (per ``project_sample_clock_anchoring`` memory).

The dual-clock §3.5 contract is respected: ``wall_clock_at_production``
on a ``ChannelMetricRecord`` is itself sample-clock-derived at the
Source, so the boundary detection does not drift.

Threading
---------
DisplayBlock owns its run thread per §4.  It blocks on the metrics
input port; on each metric record it also drains pending detection
events non-blocking before writing the new row.  The Qt 100 ms render
timer reads :meth:`state_snapshot` from the GUI thread.

Snapshot is lockless single-writer / single-reader: ``state_snapshot``
copies the relevant arrays under a short ``threading.Lock`` so the
reader sees a consistent ``(history, write_idx, period_index)`` triple.
The copy is ~270 KB at default sizes and runs at 10 Hz — well under the
cost of the Qt ``setImage`` it feeds.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..channelizer import N_CHANNELS
from .block import Block
from .types import BlockConfig, ChannelMetricRecord, Event, Record, StreamClosed

log = logging.getLogger(__name__)


#: Default heatmap history depth — ``N_SNR_HIST`` from the legacy code
#: corresponds to one 15-s period of detector hops.  At 12 kHz channel
#: rate with 256-sample stride that's ~ 705 hops per period.
_DEFAULT_HISTORY_ROWS = 705

#: Default sentinel filling unused / cleared rows.  Matches the legacy
#: ``-999.0`` initial value, which the Qt colour map clamps to the bottom
#: of its scale.
_HISTORY_SENTINEL = -999.0

#: Waterfall sentinel — matches the legacy ``-130.0`` dBFS init value
#: that ``Engine`` uses for cleared spectrogram rows.
_WATERFALL_SENTINEL = -130.0

#: Maximum sbuf capacity (samples).  Mirrors legacy ``self.fft_size * 6``
#: — enough headroom for 6 FFT blocks of arrival before the run loop
#: drains them.  Beyond this we drop the oldest, matching legacy.
_SBUF_FFT_BLOCKS = 6


@dataclass
class DisplayBlockConfig(BlockConfig):
    """Per-DisplayBlock typed config."""

    # Ports.
    metric_port_name: str = "metrics"
    detection_port_name: str = "detections"
    #: Optional IQ input port for the waterfall (#10b).  When the input
    #: is connected, ``DisplayBlock`` runs an overlap-add FFT spectrogram
    #: on the IQ stream and exposes it via ``state_snapshot()``.  When
    #: the port is *not* connected, no waterfall work is done and the
    #: snapshot's ``spectrogram_data`` array stays at the sentinel.
    iq_port_name: str = "iq"

    # Heatmap geometry.
    n_channels: int = N_CHANNELS
    history_rows: int = _DEFAULT_HISTORY_ROWS
    history_period_s: float = 15.0   # MSK144 protocol period — boundary
                                     # is at UTC seconds divisible by this.

    # Waterfall geometry (#10b).  Defaults match the legacy engine
    # constants so the post-cut-over GUI sees the same spectrogram shape
    # as today.  fft_size=2048, sample_rate=48000 → blocks_per_sec=46.875,
    # so a 15-s window holds 703 rows.  spec_staging / spectrogram_data
    # arrays are (waterfall_rows, fft_size) float32 dB.
    iq_sample_rate_hz: int = 48_000
    iq_fft_size: int = 2048
    iq_full_scale: float = 1.0   # IQ peak-normalised to ±1.0 (CLAUDE.md
                                  # signal-level contract); use for the
                                  # dBFS = 20·log10(magnitude/full_scale)
                                  # conversion.

    # Get timeout — short enough for responsive shutdown.
    get_timeout_s: float = 0.250


@dataclass
class DisplayState:
    """Frozen snapshot of the heatmap + waterfall state — returned by
    :meth:`DisplayBlock.state_snapshot`.

    Consumers read this on the Qt GUI thread; copies of the arrays are
    safe to retain (the producer never mutates them in place after
    snapshot).
    """

    ch_snr_history_h: np.ndarray  # (history_rows, n_channels) float32
    ch_snr_history_v: np.ndarray  # (history_rows, n_channels) float32
    ch_snr_write_idx: int         # row that was last written
    ch_snr_hop_count: int         # monotonic count within the period
    ch_snr_period_index: int      # = int(wall_clock / history_period_s)
    jt9_markers: tuple[dict[str, Any], ...]  # immutable tuple snapshot

    # Waterfall (#10b).  spectrogram_data holds the most recently
    # *completed* 15-s spectrogram (snapshotted at each period boundary);
    # spec_staging is the in-progress accumulation.  Both are
    # ``(waterfall_rows, fft_size)`` float32 in dBFS.
    spectrogram_data: np.ndarray
    spec_staging:     np.ndarray
    spec_period_index: int        # period index of spec_staging
    spec_filled:      bool        # True once at least one period has
                                  # been completed (spectrogram_data
                                  # holds real data, not the sentinel)


class DisplayBlock(Block):
    """Channel-metric heatmap + detection-marker state, as a Block.

    Inputs
    ------
    ``metrics`` : :class:`ChannelMetricRecord` stream — one record per
        detector cycle, carrying per-channel ``pair_metric_db_h/v`` and
        sub-metrics.  This is the new contract introduced for ``#10``.

    ``detections`` : :class:`Event` stream of ``kind="detection"`` — the
        same events the Decoder consumes, used here only to render
        marker overlays on the heatmap.  Marker events are non-load-
        bearing (drop-oldest is fine) and read non-blocking.

    Outputs
    -------
    None on the data plane (sink).  GUI consumers call
    :meth:`state_snapshot` from the Qt thread.

    State machine
    -------------
    Each metric record:

    1. Compute ``new_period_index = int(wall_clock_at_production /
       history_period_s)``.
    2. If ``new_period_index != ch_snr_period_index``: clear the history
       buffers (set to ``_HISTORY_SENTINEL``), reset ``hop_count = 0``,
       prune all markers from the previous period, set
       ``ch_snr_period_index = new_period_index``.
    3. Drain pending detection events (non-blocking) and append to the
       marker list, tagged with the current ``period_index``.
    4. Write the metric arrays into ring row ``hop_count % history_rows``.
    5. Increment ``hop_count``.

    The boundary detection uses the *upstream* wall clock from the
    record, NOT ``time.time()`` — see "Boundary handling" in the module
    docstring.
    """

    config_type = DisplayBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Heatmap state.  Allocated in on_start once we know sizes.
        self._ch_snr_history_h: np.ndarray = None  # type: ignore[assignment]
        self._ch_snr_history_v: np.ndarray = None  # type: ignore[assignment]
        self._ch_snr_write_idx: int = 0
        self._ch_snr_hop_count: int = 0
        self._ch_snr_period_index: int = -1   # forces clear on first record
        self._jt9_markers: list[dict[str, Any]] = []

        # Waterfall state (#10b).  All arrays allocated in on_start.
        # _sbuf is a sliding accumulation buffer — IQ records append to
        # its tail; FFT blocks are consumed from its head, then samples
        # drain by one hop (fft_size // 2) per block.
        self._sbuf: np.ndarray = None  # type: ignore[assignment]
        self._sbuf_end: int = 0          # current fill level in _sbuf
        self._sbuf_t0: float | None = None  # wall_clock for _sbuf[0]
        self._spec_staging: np.ndarray = None  # type: ignore[assignment]
        self._spectrogram_data: np.ndarray = None  # type: ignore[assignment]
        self._spec_period_index: int = -1
        self._spec_filled: bool = False
        # FFT window + gain — computed once in on_start.
        self._fft_window: np.ndarray = None  # type: ignore[assignment]
        self._fft_window_gain: float = 1.0

        # Snapshot lock — held only briefly while copying arrays.
        self._snap_lock = threading.Lock()

        # Stats.
        self._n_metric_records: int = 0
        self._n_detection_events: int = 0
        self._n_period_boundaries: int = 0
        self._n_iq_records: int = 0
        self._n_iq_samples: int = 0
        self._n_spec_blocks_written: int = 0
        self._n_spec_period_snapshots: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: DisplayBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if config.history_rows < 1:
            raise ValueError(f"{self.name}: history_rows must be >= 1")
        if config.history_period_s <= 0:
            raise ValueError(f"{self.name}: history_period_s must be > 0")
        if config.n_channels < 1:
            raise ValueError(f"{self.name}: n_channels must be >= 1")
        if config.iq_fft_size < 16 or (config.iq_fft_size & (config.iq_fft_size - 1)) != 0:
            raise ValueError(
                f"{self.name}: iq_fft_size must be a power of 2 >= 16"
            )
        if config.iq_sample_rate_hz <= 0:
            raise ValueError(f"{self.name}: iq_sample_rate_hz must be > 0")

    def on_start(self) -> None:
        super().on_start()
        cfg: DisplayBlockConfig = self.config  # type: ignore[assignment]
        nch = cfg.n_channels
        self._ch_snr_history_h = np.full(
            (cfg.history_rows, nch), _HISTORY_SENTINEL, dtype=np.float32,
        )
        self._ch_snr_history_v = np.full(
            (cfg.history_rows, nch), _HISTORY_SENTINEL, dtype=np.float32,
        )
        self._ch_snr_write_idx = 0
        self._ch_snr_hop_count = 0
        self._ch_snr_period_index = -1
        self._jt9_markers = []

        # Waterfall (#10b).  Sized at legacy parity:
        #   blocks_per_sec = sample_rate / (fft_size // 2)
        #   waterfall_rows = round(history_period_s * blocks_per_sec)
        # Default 48000 / 1024 = 46.875 blocks/s × 15 s = ~703 rows.
        fft_size = cfg.iq_fft_size
        hop = fft_size // 2
        blocks_per_sec = cfg.iq_sample_rate_hz / hop
        wf_rows = max(1, int(round(cfg.history_period_s * blocks_per_sec)))
        self._spec_staging = np.full(
            (wf_rows, fft_size), _WATERFALL_SENTINEL, dtype=np.float32,
        )
        self._spectrogram_data = np.full(
            (wf_rows, fft_size), _WATERFALL_SENTINEL, dtype=np.float32,
        )
        self._spec_period_index = -1
        self._spec_filled = False

        # Sliding accumulation buffer for FFT input.
        self._sbuf = np.zeros(
            fft_size * _SBUF_FFT_BLOCKS, dtype=np.complex64,
        )
        self._sbuf_end = 0
        self._sbuf_t0 = None

        # FFT window — Hanning, matched to legacy.  window_gain is
        # the coherent gain (sum of window samples / N) used to
        # normalise magnitude back to a 0-dBFS-ish absolute scale.
        self._fft_window = np.hanning(fft_size).astype(np.float32)
        self._fft_window_gain = float(self._fft_window.mean())

        log.info(
            "%s: heatmap=%d ch × %d rows; waterfall=%d rows × %d bins "
            "(fft=%d @ %d Hz, %.2f blocks/s); period=%.1f s",
            self.name, nch, cfg.history_rows,
            wf_rows, fft_size, fft_size, cfg.iq_sample_rate_hz,
            blocks_per_sec, cfg.history_period_s,
        )

    def on_stop(self) -> None:
        super().on_stop()
        log.info(
            "%s: metric_records=%d, detection_events=%d, "
            "period_boundaries=%d",
            self.name,
            self._n_metric_records,
            self._n_detection_events,
            self._n_period_boundaries,
        )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: DisplayBlockConfig = self.config  # type: ignore[assignment]
        metric_port = cfg.metric_port_name
        if metric_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {metric_port!r} not connected"
            )

        # Block on the metric stream — that's the heartbeat.  Detection
        # events are read non-blocking on each metric tick.
        try:
            rec = self.inputs[metric_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates and exits cleanly.

        if not isinstance(rec, ChannelMetricRecord):
            log.warning(
                "%s: dropped non-ChannelMetricRecord %r on metrics port",
                self.name, type(rec).__name__,
            )
            return

        if rec.pair_metric_db_h.shape != (cfg.n_channels,):
            log.warning(
                "%s: metric record shape %s != (%d,) — dropped",
                self.name, rec.pair_metric_db_h.shape, cfg.n_channels,
            )
            return

        # 1. Boundary detection — wall-clock-grid, sample-clock-anchored
        # via the upstream record's wall_clock_at_production (§3.5).
        new_period = int(rec.wall_clock_at_production / cfg.history_period_s)
        boundary_crossed = (new_period != self._ch_snr_period_index)

        # 2. Drain pending detection events non-blocking BEFORE writing
        # the new row.  This way markers belonging to the just-finished
        # period are pruned cleanly when the boundary fires.
        det_port = cfg.detection_port_name
        det_stream = self.inputs.get(det_port)

        with self._snap_lock:
            if boundary_crossed:
                self._on_boundary_cross(new_period)

            if det_stream is not None:
                self._drain_detection_events(det_stream)

            # 4. Write into the ring at the current hop position.
            write_idx = self._ch_snr_hop_count % cfg.history_rows
            self._ch_snr_history_h[write_idx, :] = rec.pair_metric_db_h
            self._ch_snr_history_v[write_idx, :] = rec.pair_metric_db_v
            self._ch_snr_write_idx = write_idx

            # 5. Advance the hop counter.
            self._ch_snr_hop_count += 1

        self._n_metric_records += 1

        # 6. Drain IQ stream non-blocking — waterfall accumulation runs
        # on the same thread but doesn't gate the heartbeat (metrics
        # arrive at ~21 Hz; IQ at ~46 blocks/s, so per-metric we'll
        # typically drain a few records).
        iq_port = cfg.iq_port_name
        iq_stream = self.inputs.get(iq_port)
        if iq_stream is not None:
            self._drain_iq_records(iq_stream)

    # --- Helpers ------------------------------------------------------

    def _on_boundary_cross(self, new_period_index: int) -> None:
        """Reset history + prune markers at the 15-s period boundary."""
        self._ch_snr_history_h[:] = _HISTORY_SENTINEL
        self._ch_snr_history_v[:] = _HISTORY_SENTINEL
        self._ch_snr_hop_count = 0
        self._ch_snr_write_idx = 0
        # Drop markers that belong to the prior period.
        self._jt9_markers = [
            m for m in self._jt9_markers
            if m.get("period_index") == new_period_index
        ]
        self._ch_snr_period_index = new_period_index
        self._n_period_boundaries += 1

    def _drain_detection_events(self, det_stream: Any) -> None:
        """Pull all currently queued detection events into the marker list.

        ``Stream.get(timeout=0)`` is the non-blocking idiom — raises
        :class:`TimeoutError` immediately if nothing is queued.
        """
        while True:
            try:
                evt = det_stream.get(timeout=0.0)
            except TimeoutError:
                return
            except StreamClosed:
                # Upstream Detector closed its detection port; stop
                # draining but keep processing metrics until our own
                # input closes.
                return
            if not isinstance(evt, Event) or evt.kind != "detection":
                continue
            payload = evt.payload or {}
            marker = {
                "period_index": self._ch_snr_period_index,
                "occurred_at_sample": evt.occurred_at_sample,
                "wall_clock": evt.wall_clock_at_production,
                "channel_index": payload.get("channel_index"),
                "fc_hz": payload.get("fc_hz"),
                "pair_metric_db": payload.get("pair_metric_db"),
                "source": payload.get("source"),
                # Row index at which this detection landed — tells the
                # renderer where to place the marker on the heatmap.
                "row_at_event": (
                    self._ch_snr_hop_count % self.config.history_rows
                ),
            }
            self._jt9_markers.append(marker)
            self._n_detection_events += 1

    # --- IQ waterfall (#10b) -----------------------------------------

    def _drain_iq_records(self, iq_stream: Any) -> None:
        """Pull queued IQ records non-blocking and process FFT blocks.

        IQStream records arrive faster than metric records (continuous
        ~46 blocks/s vs sparse ~21 metric records/s).  Each call drains
        until empty or a small per-tick budget is exhausted, then runs
        the FFT loop on whatever was added.  Holds ``_snap_lock`` during
        the spectrogram-row writes so the Qt reader sees consistent
        arrays.
        """
        cfg: DisplayBlockConfig = self.config  # type: ignore[assignment]
        budget = 32   # bound the per-tick work; matches IQ rate at the
                      # ~21 Hz metric heartbeat with comfortable slack
        for _ in range(budget):
            try:
                rec = iq_stream.get(timeout=0.0)
            except (TimeoutError, StreamClosed):
                break
            if not isinstance(rec, Record):
                continue
            data = np.ascontiguousarray(rec.data, dtype=np.complex64)
            # IQ stream from the Source is 1-D; channelizer output is
            # multi-channel and not the right input here.
            if data.ndim != 1:
                continue
            n = data.size
            if n == 0:
                continue
            with self._snap_lock:
                # Re-anchor _sbuf_t0 from each record's wall clock,
                # mirroring legacy processing.py:1160:
                #   _sbuf_t0 = pkt_time − (samples already in sbuf) / sr
                # This handles wall-clock discontinuities (source switch,
                # packet gaps where the source increments wall_clock
                # past the buffered samples) without drifting.  For a
                # gap-free stream the result is identical to a one-shot
                # anchor + per-hop advancement.
                self._sbuf_t0 = (
                    rec.wall_clock_at_production
                    - self._sbuf_end / float(rec.sample_rate_hz)
                )

                # If incoming would overflow _sbuf, drop the oldest
                # samples to make room — same policy as legacy when the
                # FFT loop falls behind.  Tail-drop preserves recent IQ
                # which is what the FFT will actually consume.
                cap = self._sbuf.size
                if n >= cap:
                    # Incoming alone exceeds capacity — keep just the
                    # last cap samples and reset.
                    samples_dropped = self._sbuf_end + (n - cap)
                    self._sbuf[:] = data[-cap:]
                    self._sbuf_end = cap
                    self._sbuf_t0 = (
                        self._sbuf_t0 + samples_dropped / float(cfg.iq_sample_rate_hz)
                    )
                else:
                    if self._sbuf_end + n > cap:
                        # Slide _sbuf left to make room for n more.
                        overflow = self._sbuf_end + n - cap
                        keep = self._sbuf_end - overflow
                        self._sbuf[:keep] = self._sbuf[overflow : self._sbuf_end]
                        self._sbuf_end = keep
                        self._sbuf_t0 = (
                            self._sbuf_t0 + overflow / float(cfg.iq_sample_rate_hz)
                        )
                    self._sbuf[self._sbuf_end : self._sbuf_end + n] = data
                    self._sbuf_end += n

                self._n_iq_records += 1
                self._n_iq_samples += n
                self._process_fft_blocks()

    def _process_fft_blocks(self) -> None:
        """Run overlap-add FFT on the accumulated _sbuf samples.

        Called only with ``_snap_lock`` held.  Each block consumes one
        FFT window's worth of samples from the head of _sbuf, then
        slides the buffer left by one hop (fft_size // 2).  Power
        spectrum in dBFS goes into the row of ``_spec_staging``
        corresponding to the block's wall-clock time within the
        current 15-s period.

        Period boundary handling: when a block's wall-clock crosses
        into a new period, snapshot _spec_staging into _spectrogram_data
        and reset the staging buffer.
        """
        cfg: DisplayBlockConfig = self.config  # type: ignore[assignment]
        fft_size = cfg.iq_fft_size
        hop = fft_size // 2
        sr = cfg.iq_sample_rate_hz
        period_s = cfg.history_period_s
        wf_rows = self._spec_staging.shape[0]
        full_scale = max(cfg.iq_full_scale, 1e-30)
        win = self._fft_window
        win_gain = self._fft_window_gain

        while self._sbuf_end >= fft_size:
            block = self._sbuf[:fft_size]
            X = np.fft.fftshift(np.fft.fft(block * win))
            mag = np.abs(X) / (fft_size * win_gain)
            power_db = (
                20.0 * np.log10(mag / full_scale + 1e-12)
            ).astype(np.float32)

            # Sample-accurate time for this block: _sbuf_t0 is the time
            # of _sbuf[0], which is always the first sample of the
            # current block (the buffer slides by exactly one hop after
            # each block is processed).
            t_block = self._sbuf_t0
            new_period = int(t_block / period_s)
            if new_period != self._spec_period_index:
                # Snapshot completed period and reset staging.  First
                # transition seeds spectrogram_data with the (mostly
                # sentinel) staging — that's fine; spec_filled stays
                # False until the staging has had time to fill.
                if self._spec_period_index >= 0:
                    self._spectrogram_data[:] = self._spec_staging
                    self._spec_filled = True
                    self._n_spec_period_snapshots += 1
                self._spec_staging[:] = _WATERFALL_SENTINEL
                self._spec_period_index = new_period

            # Row index = position within the period scaled to the
            # waterfall row count.  Use round() not int() per the legacy
            # comment about FP reconstruction biasing systematic-low.
            t_in_period = t_block - new_period * period_s
            blocks_per_sec = sr / hop
            row_idx = round(t_in_period * blocks_per_sec)
            if 0 <= row_idx < wf_rows:
                self._spec_staging[row_idx, :] = power_db
                self._n_spec_blocks_written += 1

            # Slide buffer left by one hop (50 % overlap).
            tail = self._sbuf_end - hop
            self._sbuf[:tail] = self._sbuf[hop : self._sbuf_end]
            self._sbuf_end = tail
            self._sbuf_t0 = self._sbuf_t0 + hop / float(sr)

    # --- Public read-only API ----------------------------------------

    def state_snapshot(self) -> DisplayState:
        """Return a thread-safe immutable view of the current display state.

        Called by the Qt 100 ms timer from the GUI thread.  Holds the
        snapshot lock briefly while copying the arrays (~270 KB heatmap
        + ~5.6 MB waterfall at default sizes).
        """
        with self._snap_lock:
            return DisplayState(
                ch_snr_history_h=self._ch_snr_history_h.copy(),
                ch_snr_history_v=self._ch_snr_history_v.copy(),
                ch_snr_write_idx=self._ch_snr_write_idx,
                ch_snr_hop_count=self._ch_snr_hop_count,
                ch_snr_period_index=self._ch_snr_period_index,
                jt9_markers=tuple(dict(m) for m in self._jt9_markers),
                spectrogram_data=self._spectrogram_data.copy(),
                spec_staging=self._spec_staging.copy(),
                spec_period_index=self._spec_period_index,
                spec_filled=self._spec_filled,
            )

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "metric_records":     self._n_metric_records,
            "detection_events":   self._n_detection_events,
            "period_boundaries":  self._n_period_boundaries,
            "iq_records":         self._n_iq_records,
            "iq_samples":         self._n_iq_samples,
            "spec_blocks_written":  self._n_spec_blocks_written,
            "spec_period_snapshots": self._n_spec_period_snapshots,
            "current_period_index": self._ch_snr_period_index,
            "current_hop_count":  self._ch_snr_hop_count,
            "spec_filled":        self._spec_filled,
        })
        return base
