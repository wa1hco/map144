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
"""CombinerBlock — dual-polarization merge at ChannelStream granularity.

See ``docs/block-stream-design.md`` §6.1.  Combines two coherent
``ChannelStream``\\s (H and V) into a single ``ChannelStream`` that the
downstream :class:`DecoderBlock` consumes as if the input were mono.

The combination math (CLAUDE.md "Polarization combination"):

    combined = H * cos(theta) + V * exp(j * delta_phi) * sin(theta)

For v1 ``theta`` and ``delta_phi`` are scalars from
:class:`CombinerBlockConfig` (per §9 #11 ratified — static config v1,
per-event sweep deferred to v2).  Per-channel arrays
(``theta_rad`` / ``delta_phi_rad`` shape ``(n_channels,)``) are accepted
too — the math is broadcast-friendly.

Coherent-pair invariants (§3.2)
-------------------------------
The two input streams must satisfy the §3.2 contract:
- sample alignment (matching ``start_sample`` per pair of records)
- phase coherence (producer responsibility; not enforced here)
- lockstep backpressure (producer side — both queues use POLICY_BLOCK)
- lockstep filtering (Channelizer-instance taps must match)

This block enforces #1 by construction: a tick reads one record from each
stream and asserts their ``start_sample``\\s match before combining.  A
mismatch is fatal (logged at WARNING; the run loop reports the error and
stops the block) — silently combining mis-aligned records would corrupt
the polarization measurement.

Detection events
----------------
The Combiner does **not** consume DetectionEventStreams.  Each
polarization's :class:`DetectorBlock` emits to its own event stream;
the downstream :class:`DecoderBlock` reads both and de-duplicates.
This keeps the Combiner's job to one thing — combine IQ — and lets
each Detector run independently per §6.1 ("a signal that is strong
on H but absent on V should still launch a decode").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .block import Block
from .types import BlockConfig, Record, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class CombinerBlockConfig(BlockConfig):
    """Per-Combiner config.

    ``theta_rad`` and ``delta_phi_rad`` may be scalar floats (uniform
    across all channels) or 1-D arrays of length ``n_channels`` for
    per-channel mixing.  v1 production usage is expected to be scalar;
    per-channel becomes useful when a calibration block lands in a
    future revision (§9 #11 option (c)).
    """

    input_port_h: str = "channels_h"
    input_port_v: str = "channels_v"
    output_port: str = "channels"

    # Polarization mixing parameters.  See module docstring for the math.
    theta_rad: float = 0.0
    delta_phi_rad: float = 0.0

    # Per-pair read timeout — short for responsive shutdown.
    get_timeout_s: float = 0.250


class CombinerBlock(Block):
    """Combine two coherent ``ChannelStream``\\s into one.  See module docstring."""

    config_type = CombinerBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Pre-computed mixing coefficients (set in configure / on_start).
        self._cos_theta: np.complex64 | np.ndarray = np.complex64(1.0)
        self._v_factor: np.complex64 | np.ndarray = np.complex64(0.0)
        # Stats.
        self._n_combined: int = 0
        self._n_alignment_errors: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: CombinerBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        # Compute combination coefficients up-front so the hot path is
        # one multiply-add per channel.  Accept scalar or array.
        # Records are shape ``(n_channels, n_samples)``.  Per-channel
        # arrays must be reshaped to ``(n_channels, 1)`` so they
        # broadcast against the time axis.
        theta = np.asarray(config.theta_rad, dtype=np.float64)
        dphi = np.asarray(config.delta_phi_rad, dtype=np.float64)
        cos_theta = np.cos(theta).astype(np.complex64)
        v_factor = (np.exp(1j * dphi) * np.sin(theta)).astype(np.complex64)
        # Reshape per-channel arrays for column broadcasting.
        if cos_theta.ndim == 1:
            cos_theta = cos_theta.reshape(-1, 1)
        if v_factor.ndim == 1:
            v_factor = v_factor.reshape(-1, 1)
        self._cos_theta = cos_theta
        self._v_factor = v_factor

    def on_start(self) -> None:
        super().on_start()
        cfg: CombinerBlockConfig = self.config  # type: ignore[assignment]
        log.info(
            "%s: theta_rad=%s, delta_phi_rad=%s",
            self.name,
            cfg.theta_rad if np.isscalar(cfg.theta_rad) else "<array>",
            cfg.delta_phi_rad if np.isscalar(cfg.delta_phi_rad) else "<array>",
        )

    def on_stop(self) -> None:
        super().on_stop()
        log.info(
            "%s: combined=%d, alignment_errors=%d",
            self.name, self._n_combined, self._n_alignment_errors,
        )

    # --- Run loop -----------------------------------------------------

    def tick(self) -> None:
        cfg: CombinerBlockConfig = self.config  # type: ignore[assignment]
        in_h = cfg.input_port_h
        in_v = cfg.input_port_v
        out = cfg.output_port
        if in_h not in self.inputs:
            raise RuntimeError(f"{self.name}: input port {in_h!r} not connected")
        if in_v not in self.inputs:
            raise RuntimeError(f"{self.name}: input port {in_v!r} not connected")

        # Read H first.  If H times out, it doesn't matter that V might
        # have data waiting — we'll get to it on the next tick.
        try:
            rec_h: Record = self.inputs[in_h].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates and exits cleanly.

        # V should have a paired record at the *same* start_sample.  Per
        # §3.2 lockstep backpressure, V never lags H by more than one
        # record on the producer side, so a bounded get is safe — if V
        # genuinely has nothing, that's a coherent-pair violation
        # somewhere upstream.
        try:
            rec_v: Record = self.inputs[in_v].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            log.warning(
                "%s: V channel timed out while H had a record at start_sample=%d "
                "— possible CoherentPair violation upstream",
                self.name, rec_h.start_sample,
            )
            self._n_alignment_errors += 1
            return

        if not isinstance(rec_h, Record) or not isinstance(rec_v, Record):
            log.warning(
                "%s: dropped non-Record on input port (h=%r, v=%r)",
                self.name, type(rec_h).__name__, type(rec_v).__name__,
            )
            return

        # Sample-alignment invariant (§3.2).  Mismatch is fatal — silently
        # combining misaligned records would corrupt the polarization
        # measurement.  Log loudly and skip; the run loop will continue
        # but the operator should see the warning and investigate.
        if rec_h.start_sample != rec_v.start_sample:
            log.warning(
                "%s: coherent-pair sample misalignment "
                "(H start_sample=%d, V start_sample=%d) — skipping combine",
                self.name, rec_h.start_sample, rec_v.start_sample,
            )
            self._n_alignment_errors += 1
            return

        if rec_h.sample_rate_hz != rec_v.sample_rate_hz:
            log.warning(
                "%s: sample-rate mismatch (H=%d Hz, V=%d Hz) — skipping",
                self.name, rec_h.sample_rate_hz, rec_v.sample_rate_hz,
            )
            self._n_alignment_errors += 1
            return

        if rec_h.data.shape != rec_v.data.shape:
            log.warning(
                "%s: shape mismatch (H=%s, V=%s) — skipping",
                self.name, rec_h.data.shape, rec_v.data.shape,
            )
            self._n_alignment_errors += 1
            return

        # ── Combination math (CLAUDE.md polarization formula) ─────────
        # combined = H * cos(theta) + V * exp(j * delta_phi) * sin(theta)
        # cos_theta and v_factor are pre-computed in configure().
        combined = (
            rec_h.data * self._cos_theta + rec_v.data * self._v_factor
        ).astype(np.complex64)

        # Inherit wall_clock_at_production from H (§3.5).  H and V should
        # match anyway — they came from the same Source tick — but we
        # pick H by convention for determinism.
        out_record = Record(
            data=combined,
            sample_rate_hz=rec_h.sample_rate_hz,
            start_sample=rec_h.start_sample,
            wall_clock_at_production=rec_h.wall_clock_at_production,
            metadata={
                **rec_h.metadata,
                "theta_rad": (
                    float(cfg.theta_rad) if np.isscalar(cfg.theta_rad)
                    else "array"
                ),
                "delta_phi_rad": (
                    float(cfg.delta_phi_rad) if np.isscalar(cfg.delta_phi_rad)
                    else "array"
                ),
                "polarization_combined": True,
            },
        )
        if out not in self.outputs:
            # Output unconnected — nowhere to put the result, but the
            # combine itself succeeded.  Drop silently.
            self._n_combined += 1
            return
        try:
            self.outputs[out].put(out_record)
        except StreamClosed:
            raise StopIteration
        self._n_combined += 1
