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
"""ChannelizerBlock — Block-form polyphase channelizer.

Wraps :func:`map144_app.channelizer.apply_channelizer` as a Phase 3
:class:`Block`.  No DSP changes — same NCO grid, same LP / HP taps, same
4× decimation, same per-channel state evolution.  This block is purely a
packaging change.

Data flow
---------
Input : ``IQStream``      complex64, shape ``(N,)`` at ``sample_rate_in_hz``
Output: ``ChannelStream`` complex64, shape ``(N_CHANNELS, N//decimate_factor)``
        at ``sample_rate_in_hz / decimate_factor``

Sample-clock and wall-clock anchoring
-------------------------------------
Per §3.5 the Channelizer **inherits** ``wall_clock_at_production`` from
the upstream record — it is a deterministic function of upstream data
and the Source remains the authoritative wall-clock anchor.  The
output's ``start_sample`` advances at the decimated rate (one output
sample per ``decimate_factor`` input samples).

Carry buffering
---------------
``apply_channelizer`` consumes ``N`` input samples and emits
``N // decimate_factor`` output samples — but only when ``N`` is exactly
divisible by ``decimate_factor``.  Sources that are slightly misaligned
(WavSource at EOF; future variable-rate sources) would otherwise lose
or misposition trailing samples.

The block keeps a small ``_carry`` of up to ``decimate_factor - 1``
input samples between ticks.  Each tick concatenates ``_carry`` with the
new record, channelizes the aligned prefix, and saves any leftover as
the next ``_carry``.  This preserves both the gap-free invariant on
``start_sample`` and the bit-exact equivalence with the legacy code path
when records are aligned (the carry is always empty in that case).

Out of scope for v1
-------------------
- Dual-pol path.  ``apply_channelizer`` already supports it; a
  follow-up ``DualPolChannelizer`` will use the :class:`CoherentPair`
  scaffolding to consume H + V records together and emit two
  ``ChannelStream`` records per tick.  v1 is mono.
- Configurable ``channel_offset_hz`` injection from the Source's
  metadata.  For now it's a static config value (matches the legacy
  pattern where ``channel_offset_hz`` is set at engine build time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..channelizer import (
    CH_SAMPLE_RATE,
    DECIMATE_FACTOR,
    N_CHANNELS,
    apply_channelizer,
    design_channelizer_filter,
    make_channelizer_state,
)
from .block import Block
from .types import BlockConfig, Record, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class ChannelizerBlockConfig(BlockConfig):
    """Per-Channelizer config."""

    # Input / output ports.
    input_port_name: str = "iq"
    output_port_name: str = "channels"

    # DSP parameters — defaults match the existing channelizer module.
    sample_rate_in_hz: int = 48_000
    n_channels: int = N_CHANNELS                     # 48
    decimate_factor: int = DECIMATE_FACTOR            # 4
    channel_offset_hz: float = 0.0                    # NCO offset; (calling-pan)+1500
    sample_rate_out_hz: int = CH_SAMPLE_RATE          # 12_000 (informational; computed from in/decim)

    # ``Stream.get`` timeout — short enough that ``stop()`` is responsive.
    get_timeout_s: float = 0.250


class ChannelizerBlock(Block):
    """Polyphase channelizer as a Block.  See module docstring."""

    config_type = ChannelizerBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._state = None
        self._lp_taps = None
        # Carry buffer for misaligned trailing samples.
        self._carry: np.ndarray = np.empty(0, dtype=np.complex64)
        # Output sample-counter — advances at the decimated rate.
        self._n_out_samples: int = 0
        # Input sample-counter (sanity-checks gap-free invariant).
        self._n_in_samples_consumed: int = 0
        # Last upstream wall-clock; inherited onto output records.
        self._last_upstream_wall_clock: float = 0.0
        # Stats: how many trailing samples have been carried; useful for
        # observing misalignment in the upstream stream.
        self._carry_carried_total: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: ChannelizerBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if config.decimate_factor < 1:
            raise ValueError(
                f"{self.name}: decimate_factor must be >= 1 (got {config.decimate_factor})"
            )
        if config.n_channels < 1:
            raise ValueError(
                f"{self.name}: n_channels must be >= 1 (got {config.n_channels})"
            )
        expected_rate_out = config.sample_rate_in_hz // config.decimate_factor
        if config.sample_rate_out_hz != expected_rate_out:
            log.warning(
                "%s: sample_rate_out_hz=%d disagrees with sample_rate_in_hz/%d=%d "
                "— using the computed value",
                self.name, config.sample_rate_out_hz,
                config.decimate_factor, expected_rate_out,
            )

    def on_start(self) -> None:
        super().on_start()
        cfg: ChannelizerBlockConfig = self.config  # type: ignore[assignment]
        self._lp_taps = design_channelizer_filter(sample_rate=cfg.sample_rate_in_hz)
        self._state = make_channelizer_state(
            n_channels=cfg.n_channels,
            lp_taps=self._lp_taps,
            sample_rate=cfg.sample_rate_in_hz,
            channel_offset_hz=cfg.channel_offset_hz,
        )
        self._carry = np.empty(0, dtype=np.complex64)
        self._n_out_samples = 0
        self._n_in_samples_consumed = 0
        self._carry_carried_total = 0
        log.info(
            "%s: %d ch x %d Hz → %d Hz (decim %d), channel_offset %.1f Hz",
            self.name,
            cfg.n_channels,
            cfg.sample_rate_in_hz,
            cfg.sample_rate_in_hz // cfg.decimate_factor,
            cfg.decimate_factor,
            cfg.channel_offset_hz,
        )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: ChannelizerBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.input_port_name
        out_port = cfg.output_port_name
        if in_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {in_port!r} not connected"
            )
        if out_port not in self.outputs:
            raise RuntimeError(
                f"{self.name}: output port {out_port!r} not connected"
            )

        try:
            rec: Record = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates to the run loop and exits cleanly.

        if not isinstance(rec, Record):
            log.warning(
                "%s: dropped non-Record %r on input port",
                self.name, type(rec).__name__,
            )
            return

        # Sanity: input rate must match config (channelizer state was built
        # for sample_rate_in_hz; mid-stream rate change isn't supported).
        if rec.sample_rate_hz != cfg.sample_rate_in_hz:
            raise RuntimeError(
                f"{self.name}: upstream record sample_rate_hz={rec.sample_rate_hz} "
                f"!= configured {cfg.sample_rate_in_hz}"
            )

        # Sanity: gap-free start_sample.  ``_n_in_samples_consumed`` counts
        # only fully-channelized input samples; ``_carry`` holds the
        # leftover tail.  So the next expected ``start_sample`` is the
        # sum of the two.  Mismatch means a gap or a misordered record.
        expected_start = self._n_in_samples_consumed + len(self._carry)
        if rec.start_sample != expected_start:
            log.warning(
                "%s: input gap detected — got start_sample=%d, expected %d",
                self.name, rec.start_sample, expected_start,
            )
            # Reset state on detected gap.  Channelizer state would be
            # stale anyway because the NCO phase no longer matches the
            # actual sample-clock position.  Easier to rebuild than to
            # try to splice; gaps should be exceptional.
            self._state = make_channelizer_state(
                n_channels=cfg.n_channels,
                lp_taps=self._lp_taps,
                sample_rate=cfg.sample_rate_in_hz,
                channel_offset_hz=cfg.channel_offset_hz,
            )
            self._carry = np.empty(0, dtype=np.complex64)
            self._n_in_samples_consumed = rec.start_sample
            self._n_out_samples = rec.start_sample // cfg.decimate_factor

        # Concatenate carry + this record's data, channelize the aligned
        # prefix, save any trailing samples as the new carry.
        self._last_upstream_wall_clock = rec.wall_clock_at_production
        if len(self._carry) > 0:
            buf = np.concatenate([self._carry, rec.data]).astype(np.complex64, copy=False)
        else:
            buf = np.ascontiguousarray(rec.data, dtype=np.complex64)

        n_total = buf.shape[0]
        n_aligned = (n_total // cfg.decimate_factor) * cfg.decimate_factor
        n_remainder = n_total - n_aligned

        if n_aligned == 0:
            # Not enough samples even with carry to make one decimated
            # output sample.  Hold them all as carry.
            self._carry = buf
            self._carry_carried_total += rec.n_samples
            return

        aligned = buf[:n_aligned]
        ch_out = apply_channelizer(
            aligned,
            self._state,
            n_channels=cfg.n_channels,
            sample_rate=cfg.sample_rate_in_hz,
            decimate_factor=cfg.decimate_factor,
        )
        # ch_out shape: (n_channels, n_aligned // decim).
        n_out = ch_out.shape[-1]

        out_record = Record(
            data=ch_out,
            sample_rate_hz=cfg.sample_rate_in_hz // cfg.decimate_factor,
            start_sample=self._n_out_samples,
            wall_clock_at_production=rec.wall_clock_at_production,  # inherit (§3.5)
            metadata={
                "n_channels": cfg.n_channels,
                "channel_offset_hz": cfg.channel_offset_hz,
            },
        )
        try:
            self.outputs[out_port].put(out_record)
        except StreamClosed:
            raise StopIteration

        self._n_in_samples_consumed += n_aligned
        self._n_out_samples += n_out
        # Save trailing samples for the next tick.
        if n_remainder > 0:
            self._carry = buf[n_aligned:].astype(np.complex64, copy=True)
            self._carry_carried_total += n_remainder
        else:
            self._carry = np.empty(0, dtype=np.complex64)
