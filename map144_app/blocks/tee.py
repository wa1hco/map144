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
"""TeeBlock — fan-out a single sample/event stream to N consumers.

Required by the Phase 3 graph because the IQ stream from the Source
feeds two independent paths in the canonical graph (§6 of
``docs/block-stream-design.md``):

    Source ── IQ ──► Tee ──► Channelizer ──► Detector ──► Decoder ──► Reporter
                       └──────────────────────────────────► Decoder

Per design §6 the Channelizer is conceptually a pass-through for IQ on
its way to the Decoder window; in the actual block graph the cleaner
factoring is to keep the Channelizer single-output (it already emits
``ChannelStream``) and put the fan-out in a dedicated Tee.  Decoders,
display waterfalls, and JSONL sinks can subscribe independently.

Records are immutable once emitted (``Record.data`` is a numpy view that
no consumer should mutate), so Tee shares the same ``Record`` instance
across all output streams without copying.  The numpy array itself is
zero-copy.

For event streams the same applies — :class:`Event` is a frozen
dataclass.

Stream policy interaction
-------------------------
If any output stream has ``POLICY_BLOCK`` and is full, the Tee will
block on the slowest consumer.  This matches the broader §5 contract:
backpressure is honoured.  For a Tee whose downstream paths have very
different consumption rates (e.g. a fast Decoder vs a slow JSONL sink),
configure the JSONL sink's stream with ``POLICY_DROP_OLDEST`` so the
fast path is not dragged down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .block import Block
from .types import BlockConfig, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class TeeBlockConfig(BlockConfig):
    """Per-TeeBlock typed config.

    ``input_port_name`` names the single input port; ``output_port_names``
    is the list of output port names the block exposes.  Output ports
    are connected via :meth:`Runtime.connect` like any other block;
    leaving an output port unconnected is allowed (the Tee will skip
    that destination silently).
    """

    input_port_name: str = "in"
    output_port_names: list[str] = field(default_factory=lambda: ["out_a", "out_b"])

    # Get timeout — short enough for responsive shutdown; long enough to
    # avoid spinning when the input is idle.
    get_timeout_s: float = 0.250


class TeeBlock(Block):
    """Fan-out one input port to N output ports.

    Inputs
    ------
    ``in`` (configurable name): any :class:`Record` or :class:`Event`.

    Outputs
    -------
    Each name in ``output_port_names`` receives a reference to the same
    record/event the Tee read from its input.  Order is the order in
    ``output_port_names`` — earlier ports get the put first, so they may
    drain marginally sooner under contention.

    Stats
    -----
    - ``items_in``  — items pulled from the input port.
    - ``items_out`` — items put on output ports (= ``items_in`` ×
      number of *connected* outputs, since unconnected outputs are
      silently skipped).
    """

    config_type = TeeBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._n_in: int = 0
        self._n_out: int = 0

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: TeeBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if not config.output_port_names:
            raise ValueError(
                f"{self.name}: TeeBlockConfig.output_port_names must be non-empty"
            )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: TeeBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.input_port_name

        if in_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {in_port!r} not connected"
            )

        try:
            item = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates and exits cleanly.

        self._n_in += 1
        for out_name in cfg.output_port_names:
            stream = self.outputs.get(out_name)
            if stream is None:
                continue
            try:
                stream.put(item)
                self._n_out += 1
            except StreamClosed:
                # One downstream closed — log once at info, continue
                # delivering to the others.  The Tee's own run loop will
                # exit on the next StreamClosed from its own input.
                log.info(
                    "%s: output %r closed; continuing to deliver to other ports",
                    self.name, out_name,
                )

    # --- Stats hook ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "items_in":  self._n_in,
            "items_out": self._n_out,
        })
        return base
