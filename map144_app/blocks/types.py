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
"""Block / Stream payload types — Records and Events.

See ``docs/block-stream-design.md`` §3.1 (sample-bearing records) and §3.3
(event records).  The §3.5 dual-clock contract is encoded by both types
carrying ``start_sample``-anchored time *and* ``wall_clock_at_production``
as independent fields — neither is derived from the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Record:
    """A sample-bearing record on a stream — see §3.1.

    The unit of transfer.  Carries a contiguous run of samples (``data``)
    plus both clock anchors (§3.5):

    - ``start_sample``: canonical sample-clock time, monotonic within a
      stream connection.  Used by DSP, display indexing, decimation
      alignment, NCO phase, FIR state, coherent averaging, and H/V
      coherent-pair alignment.
    - ``wall_clock_at_production``: canonical wall-clock time, captured
      at the Source by reading ``time.time()`` independently of the sample
      counter.  Used by reporting, logging, period-edge tests, file naming,
      and cross-station correlation.

    Records are immutable (``frozen=True``).  ``data`` itself is a numpy
    array; by convention downstream consumers do not mutate it in place.

    Records are **contiguous and gap-free** within a single stream
    connection — see §3.1 invariants.
    """

    data: np.ndarray
    sample_rate_hz: int
    start_sample: int
    wall_clock_at_production: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples along the time axis."""
        return int(self.data.shape[-1])

    @property
    def end_sample(self) -> int:
        """Exclusive end index — the next gap-free record's ``start_sample``."""
        return self.start_sample + self.n_samples


@dataclass(frozen=True)
class ChannelMetricRecord(Record):
    """Per-channel detector metrics — one record per detector cycle.

    Emitted by :class:`DetectorBlock` (and the dual-pol equivalent) on its
    ``metrics`` output port; consumed by :class:`DisplayBlock` for the
    detection-metric heatmap and by future cluster-classifier blocks.

    Inherits the dual-clock contract from :class:`Record` (§3.5):
    ``start_sample`` and ``wall_clock_at_production`` are both first-class.
    The 15-second period boundary used by the heatmap is wall-clock-defined
    by protocol (UTC-aligned for cross-station synchronisation), but
    ``wall_clock_at_production`` is itself sample-clock-derived at the
    Source so it does not drift.

    ``data`` carries the primary metric used for triggering — the
    per-channel ``max(pair_metric_db_h, pair_metric_db_v)`` in dB above
    the rolling pct25 baseline.  Shape ``(n_channels,)`` float32.

    Sub-metrics are named:

    - ``pair_metric_db_h`` / ``pair_metric_db_v`` — combined
      ``max(sq, sync)`` dB above pct25, per polarisation.  For mono
      sources both arrays are equal (== ``data``).
    - ``sq_metric_db_h`` / ``sq_metric_db_v`` — squared-FFT-only metric.
    - ``sync_metric_db_h`` / ``sync_metric_db_v`` — sync-correlator-only
      metric.

    Sub-metrics may be ``None`` when the corresponding detector path is
    disabled (e.g. ``enable_sq_detect=False``).  ``data`` and the two
    ``pair_metric_db_*`` arrays are always populated.
    """

    pair_metric_db_h: np.ndarray = None  # type: ignore[assignment]
    pair_metric_db_v: np.ndarray = None  # type: ignore[assignment]
    sq_metric_db_h: np.ndarray | None = None
    sq_metric_db_v: np.ndarray | None = None
    sync_metric_db_h: np.ndarray | None = None
    sync_metric_db_v: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.pair_metric_db_h is None or self.pair_metric_db_v is None:
            raise ValueError(
                "ChannelMetricRecord requires pair_metric_db_h "
                "and pair_metric_db_v"
            )

    @property
    def n_channels(self) -> int:
        """Number of channels along the channel axis."""
        return int(self.data.shape[0])


@dataclass(frozen=True)
class Event:
    """A discrete event on a sparse stream — see §3.3.

    Events are commentary about sample data: a detection firing, a decode
    completing, a heartbeat.  They are anchored to the upstream sample
    stream via ``occurred_at_sample`` so any event can be correlated back
    to IQ post-hoc.

    Events also carry their own ``wall_clock_at_production`` (when the
    *event* was emitted by the producing block, which is generally later
    than the underlying signal's wall clock by the block's processing
    latency).  Where the underlying signal's wall clock matters
    (e.g. "when did the ping happen?"), use ``signal_wall_clock`` from
    the payload — the producing block carries it forward from the upstream
    record.  See §3.5.
    """

    kind: str
    occurred_at_sample: int
    sample_rate_hz: int
    wall_clock_at_production: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockConfig:
    """Base class for per-block typed configuration — see §9 #2.

    Each ``Block`` subclass declares ``config_type`` pointing at its
    specific config dataclass.  The runtime composes a ``RuntimeConfig``
    from per-block sections at graph build time; ``configure(cfg)`` then
    receives the typed instance.

    The base class only carries ``name`` — populated by ``Runtime`` when
    the block is added so per-block log output is identifiable.
    """

    name: str = ""


class StreamClosed(Exception):
    """Raised by ``Stream.get`` once the producer has closed the stream
    *and* its queue has been drained.  Consumer's ``tick()`` should catch
    this and exit the run loop cleanly.
    """
