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
"""Source — abstract base for radio / file / synthetic IQ ingress.

See ``docs/block-stream-design.md`` §3.5 (two clocks) and §6 (graph).
The Source is the data-flow boundary that owns the sample-clock anchor:

- Maintains the monotonic sample counter that becomes ``Record.start_sample``.
- Captures ``wall_clock_at_production`` by reading ``time.time()`` at
  production time.  These are independent observations of the same instant.
- Optionally injects synthetic drift via ``test_clock_drift_ppm`` so the
  drift-handling code paths are reachable in seconds in unit tests
  (§3.5 test-injection hook).
- Emits ``ClockHealth`` events on a separate output port at
  ``clock_health_period_s`` cadence; logs CHARACTERIZE / WARN tier
  threshold crossings rate-limited per §3.5.

Subclasses override :meth:`produce` to fetch raw IQ from their backend
(radio queue, WAV file, synthetic generator).  The base class wraps each
chunk into a :class:`Record` with both clock anchors and pushes it to the
``iq`` output stream.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from .block import Block
from .types import BlockConfig, Event, Record, StreamClosed

log = logging.getLogger(__name__)


# Environment-variable / CLI escape hatch.  Code reads the env var; the
# CLI parser (when wired into runtime startup) sets it.  Picking up from
# env keeps the test-injection contract usable from unit tests without
# plumbing CLI flags through every constructor.
_TEST_CLOCK_DRIFT_PPM_ENV = "MAP144_TEST_CLOCK_DRIFT_PPM"


def _env_drift_ppm() -> float:
    raw = os.environ.get(_TEST_CLOCK_DRIFT_PPM_ENV, "")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        log.warning("ignoring invalid %s=%r", _TEST_CLOCK_DRIFT_PPM_ENV, raw)
        return 0.0


@dataclass
class SourceConfig(BlockConfig):
    """Per-Source typed config.  Subclasses extend with their own fields."""

    sample_rate_hz: int = 48_000
    iq_port_name: str = "iq"

    # Clock-health monitoring (§3.5)
    clock_health_port_name: str = "clock_health"
    clock_health_period_s: float = 1.0

    # Threshold tiers (§3.5).  Names match the doc:
    #   CHARACTERIZE  = INFO log; expected on healthy hardware
    #   WARN          = WARNING log; only on actual misbehaviour
    drift_characterize_threshold_s: float = 0.002   # 2 ms
    drift_warn_threshold_s: float = 0.100           # 100 ms
    drift_log_rate_limit_s: float = 60.0

    # Test-injection hook (§3.5).  Non-zero injects synthetic drift into
    # ``wall_clock_at_production`` at ``ppm`` relative to the sample clock.
    # Falls back to the ``MAP144_TEST_CLOCK_DRIFT_PPM`` env var when unset
    # by the caller.
    test_clock_drift_ppm: float | None = None


class Source(Block):
    """Abstract Source block.  See module docstring."""

    config_type = SourceConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Sample-counter state — canonical sample-clock time, set in on_start.
        self._sample_counter: int = 0
        # Wall-clock anchor: real time.time() captured at on_start.  We
        # report this to consumers as the wall clock at sample 0; subsequent
        # records' wall_clock_at_production are sampled at production time.
        self._real_anchor_wall: float = 0.0
        # Resolved drift-injection rate (config or env), captured at on_start.
        self._drift_ppm: float = 0.0
        # Throttling state for periodic clock-health emit and tier logs.
        self._last_clock_health_at: float = 0.0
        self._last_characterize_log_at: float = 0.0
        self._last_warn_log_at: float = 0.0
        # Last computed wall_clock_at_production — handy for tests / stats.
        self._last_wall_clock_at_production: float = 0.0

    # --- Subclass API -------------------------------------------------

    def produce(self) -> np.ndarray | None:
        """Subclass implements: fetch one chunk of IQ samples.

        Return:

        - a 1D ``np.ndarray`` of complex samples (typically ``complex64``),
          shape ``(n_samples,)`` — the base class stamps it with both
          clock anchors and emits a :class:`Record`;
        - ``None`` or a zero-length array — "no data this tick", base class
          skips emission and continues (e.g. transient empty queue);
        - raise :class:`StopIteration` — end-of-source (e.g. WAV EOF), base
          class exits the run loop cleanly.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.produce() must be implemented"
        )

    # --- Lifecycle hooks ---------------------------------------------

    def on_start(self) -> None:
        super().on_start()
        cfg: SourceConfig = self.config  # type: ignore[assignment]
        self._sample_counter = 0
        self._real_anchor_wall = time.time()
        self._last_clock_health_at = self._real_anchor_wall
        self._last_characterize_log_at = 0.0
        self._last_warn_log_at = 0.0
        # Resolve drift injection: explicit config wins, else env var.
        if cfg.test_clock_drift_ppm is not None:
            self._drift_ppm = float(cfg.test_clock_drift_ppm)
        else:
            self._drift_ppm = _env_drift_ppm()
        if self._drift_ppm != 0.0:
            log.info(
                "%s: TEST drift injection active at %.1f ppm "
                "(this is not a production feature)",
                self.name, self._drift_ppm,
            )

    # --- Run loop body -----------------------------------------------

    def tick(self) -> None:
        cfg: SourceConfig = self.config  # type: ignore[assignment]
        data = self.produce()
        if data is None:
            return
        if not isinstance(data, np.ndarray) or data.size == 0:
            return
        wall_now = self._compute_wall_clock_at_production()
        record = Record(
            data=data,
            sample_rate_hz=cfg.sample_rate_hz,
            start_sample=self._sample_counter,
            wall_clock_at_production=wall_now,
            metadata={},
        )
        n = record.n_samples
        # Put on iq output (block-on-full per §5; back-pressure surfaces
        # here exactly the way the doc says).
        try:
            self.outputs[cfg.iq_port_name].put(record)
        except StreamClosed:
            # Downstream closed; treat as graceful shutdown.
            raise StopIteration
        self._sample_counter += n
        self._last_wall_clock_at_production = wall_now
        self._maybe_emit_clock_health(wall_now)

    # --- Dual-clock implementation -----------------------------------

    def _compute_wall_clock_at_production(self) -> float:
        """Return wall_clock_at_production for the next record.

        On healthy hardware, this is just ``time.time()``.  When drift
        injection is active, the wall clock drifts at ``_drift_ppm``
        relative to the sample clock — implemented by stretching the
        elapsed wall time from the start anchor.  See §3.5.
        """
        real_now = time.time()
        if self._drift_ppm == 0.0:
            return real_now
        elapsed_real = real_now - self._real_anchor_wall
        # injected = anchor + elapsed_real * (1 + ppm * 1e-6)
        # Equivalent to: real_now + elapsed_real * ppm * 1e-6
        injected_offset = elapsed_real * self._drift_ppm * 1e-6
        return real_now + injected_offset

    def _implied_drift_seconds(self, wall_now: float) -> float:
        """How far ``wall_clock_at_production`` has drifted from the
        sample-clock-projected wall time.

        On healthy hardware (and no test injection) this is bounded by
        the radio TCXO's drift relative to the OS clock.  Under test
        injection it grows at ``_drift_ppm``.
        """
        cfg: SourceConfig = self.config  # type: ignore[assignment]
        sample_clock_seconds = self._sample_counter / cfg.sample_rate_hz
        wall_elapsed = wall_now - self._real_anchor_wall
        return wall_elapsed - sample_clock_seconds

    def _maybe_emit_clock_health(self, wall_now: float) -> None:
        """Periodically emit a ``ClockHealth`` event and tier-appropriate
        log line — see §3.5.
        """
        cfg: SourceConfig = self.config  # type: ignore[assignment]
        if wall_now - self._last_clock_health_at < cfg.clock_health_period_s:
            return
        self._last_clock_health_at = wall_now

        drift_seconds = self._implied_drift_seconds(wall_now)
        elapsed_real = wall_now - self._real_anchor_wall
        drift_ppm = (
            (drift_seconds / elapsed_real) * 1e6 if elapsed_real > 0 else 0.0
        )
        abs_drift = abs(drift_seconds)

        # Tier-appropriate logging — §3.5.
        # CHARACTERIZE = INFO; WARN = WARNING.  Rate-limited per tier.
        if abs_drift > cfg.drift_warn_threshold_s:
            if wall_now - self._last_warn_log_at >= cfg.drift_log_rate_limit_s:
                self._last_warn_log_at = wall_now
                log.warning(
                    "%s: clock drift %.3f s (%.1f ppm) exceeds WARN threshold "
                    "%.3f s — sample / wall clocks meaningfully out of sync",
                    self.name, drift_seconds, drift_ppm,
                    cfg.drift_warn_threshold_s,
                )
        elif abs_drift > cfg.drift_characterize_threshold_s:
            if wall_now - self._last_characterize_log_at >= cfg.drift_log_rate_limit_s:
                self._last_characterize_log_at = wall_now
                log.info(
                    "%s: clock drift %.3f s (%.1f ppm) — system "
                    "characterization (TCXO behaviour, expected on healthy hardware)",
                    self.name, drift_seconds, drift_ppm,
                )

        # ClockHealth event (always emitted, regardless of tier).  Drop
        # silently if no consumer is wired up — many test setups don't
        # connect this port.
        clock_port = cfg.clock_health_port_name
        if clock_port in self.outputs:
            event = Event(
                kind="clock_health",
                occurred_at_sample=self._sample_counter,
                sample_rate_hz=cfg.sample_rate_hz,
                wall_clock_at_production=wall_now,
                payload={
                    "implied_drift_seconds": drift_seconds,
                    "drift_ppm": drift_ppm,
                    "real_wall_clock_now": time.time(),
                    "elapsed_seconds": elapsed_real,
                    "test_drift_injection_ppm": self._drift_ppm,
                },
            )
            try:
                self.outputs[clock_port].put(event)
            except StreamClosed:
                pass

    # --- Inspection --------------------------------------------------

    @property
    def sample_counter(self) -> int:
        """Current sample-clock position (next ``start_sample`` to emit)."""
        return self._sample_counter

    @property
    def wall_clock_anchor(self) -> float:
        """Wall clock captured at ``on_start`` (real time, no injection)."""
        return self._real_anchor_wall
