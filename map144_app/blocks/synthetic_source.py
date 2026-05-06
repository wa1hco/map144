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
"""SyntheticSource — deterministic IQ generator for tests and diagnostics.

Generates either a complex tone or zero-mean Gaussian noise at the
configured sample rate, paced to (close to) real time so the sample
clock and wall clock advance at the same rate.  Real-time pacing is what
makes drift-injection tests meaningful: without it,
``sample_counter / sample_rate_hz`` would race ahead of wall-clock
elapsed and the drift metric would be dominated by produce-rate skew
rather than the synthetic offset.

Tests use this as a controlled stand-in for FlexSource so they can run
without hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .source import Source, SourceConfig


@dataclass
class SyntheticSourceConfig(SourceConfig):
    samples_per_record: int = 1024

    # Signal mode.
    #   "tone"  — a complex exponential at ``tone_hz`` (relative to baseband)
    #   "noise" — zero-mean unit-variance complex Gaussian noise
    #   "zero"  — all zeros (handy for clock-only tests)
    signal_mode: str = "tone"
    tone_hz: float = 1500.0
    tone_amplitude: float = 0.5
    noise_seed: int = 1234

    # If True, ``produce()`` sleeps until real wall clock has caught up
    # to the sample-clock-implied production time.  Off for free-running
    # tests; on for drift / cadence tests.
    realtime_pacing: bool = True

    # Stop after producing this many samples; 0 = never stop.
    stop_after_samples: int = 0


class SyntheticSource(Source):
    """Generates IQ samples deterministically — see module docstring."""

    config_type = SyntheticSourceConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._rng: np.random.Generator | None = None
        self._phase: float = 0.0
        self._produce_anchor_wall: float = 0.0

    def on_start(self) -> None:
        super().on_start()
        cfg: SyntheticSourceConfig = self.config  # type: ignore[assignment]
        self._rng = np.random.default_rng(cfg.noise_seed)
        self._phase = 0.0
        self._produce_anchor_wall = self._real_anchor_wall

    def produce(self) -> np.ndarray | None:
        cfg: SyntheticSourceConfig = self.config  # type: ignore[assignment]
        n = cfg.samples_per_record

        # Stop condition.
        if cfg.stop_after_samples and self._sample_counter >= cfg.stop_after_samples:
            raise StopIteration

        # Real-time pacing: sleep until wall clock catches the sample clock.
        # Bounded sleep — if we're already behind, just produce immediately.
        if cfg.realtime_pacing:
            target_wall = (
                self._produce_anchor_wall
                + (self._sample_counter + n) / cfg.sample_rate_hz
            )
            slack = target_wall - time.time()
            if slack > 0:
                # Bounded sleep so a stop() request can interrupt the loop
                # promptly; 50 ms granularity is plenty for the 48 kHz / 1024
                # samples/record case (~21 ms per record).
                end_at = time.time() + slack
                while not self._stopping.is_set():
                    remaining = end_at - time.time()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.050))
                if self._stopping.is_set():
                    return None  # graceful exit on next tick

        if cfg.signal_mode == "zero":
            return np.zeros(n, dtype=np.complex64)

        if cfg.signal_mode == "noise":
            assert self._rng is not None
            re = self._rng.standard_normal(n).astype(np.float32)
            im = self._rng.standard_normal(n).astype(np.float32)
            return (re + 1j * im).astype(np.complex64) * np.complex64(0.5)

        if cfg.signal_mode == "tone":
            # Continuous-phase tone — keep ``self._phase`` so successive
            # records join without a phase jump.
            two_pi_over_fs = 2.0 * np.pi * cfg.tone_hz / cfg.sample_rate_hz
            phases = self._phase + two_pi_over_fs * np.arange(n, dtype=np.float64)
            self._phase = (phases[-1] + two_pi_over_fs) % (2.0 * np.pi)
            return (cfg.tone_amplitude * np.exp(1j * phases)).astype(np.complex64)

        raise ValueError(f"unknown signal_mode: {cfg.signal_mode!r}")
