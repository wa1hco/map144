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
from .types import BlockConfig, ChannelMetricRecord, Event, StreamClosed

log = logging.getLogger(__name__)


#: Default heatmap history depth — ``N_SNR_HIST`` from the legacy code
#: corresponds to one 15-s period of detector hops.  At 12 kHz channel
#: rate with 256-sample stride that's ~ 705 hops per period.
_DEFAULT_HISTORY_ROWS = 705

#: Default sentinel filling unused / cleared rows.  Matches the legacy
#: ``-999.0`` initial value, which the Qt colour map clamps to the bottom
#: of its scale.
_HISTORY_SENTINEL = -999.0


@dataclass
class DisplayBlockConfig(BlockConfig):
    """Per-DisplayBlock typed config."""

    # Ports.
    metric_port_name: str = "metrics"
    detection_port_name: str = "detections"

    # Geometry.
    n_channels: int = N_CHANNELS
    history_rows: int = _DEFAULT_HISTORY_ROWS
    history_period_s: float = 15.0   # MSK144 protocol period — boundary
                                     # is at UTC seconds divisible by this.

    # Get timeout — short enough for responsive shutdown.
    get_timeout_s: float = 0.250


@dataclass
class DisplayState:
    """Frozen snapshot of the heatmap state — returned by
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

        # Snapshot lock — held only briefly while copying arrays.
        self._snap_lock = threading.Lock()

        # Stats.
        self._n_metric_records: int = 0
        self._n_detection_events: int = 0
        self._n_period_boundaries: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: DisplayBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if config.history_rows < 1:
            raise ValueError(f"{self.name}: history_rows must be >= 1")
        if config.history_period_s <= 0:
            raise ValueError(f"{self.name}: history_period_s must be > 0")
        if config.n_channels < 1:
            raise ValueError(f"{self.name}: n_channels must be >= 1")

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
        log.info(
            "%s: %d ch x %d rows, period=%.1f s",
            self.name, nch, cfg.history_rows, cfg.history_period_s,
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

    # --- Public read-only API ----------------------------------------

    def state_snapshot(self) -> DisplayState:
        """Return a thread-safe immutable view of the current display state.

        Called by the Qt 100 ms timer from the GUI thread.  Holds the
        snapshot lock briefly while copying the arrays (~270 KB at
        default sizes).
        """
        with self._snap_lock:
            return DisplayState(
                ch_snr_history_h=self._ch_snr_history_h.copy(),
                ch_snr_history_v=self._ch_snr_history_v.copy(),
                ch_snr_write_idx=self._ch_snr_write_idx,
                ch_snr_hop_count=self._ch_snr_hop_count,
                ch_snr_period_index=self._ch_snr_period_index,
                jt9_markers=tuple(dict(m) for m in self._jt9_markers),
            )

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "metric_records": self._n_metric_records,
            "detection_events": self._n_detection_events,
            "period_boundaries": self._n_period_boundaries,
            "current_period_index": self._ch_snr_period_index,
            "current_hop_count": self._ch_snr_hop_count,
        })
        return base
