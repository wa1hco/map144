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
"""Runtime — graph wiring and lifecycle.

See ``docs/block-stream-design.md`` §7 (lifecycle) and §9 #3 (connection
API).  The :class:`Runtime` owns the block graph and orchestrates start /
stop in topological order.

Usage sketch::

    rt = Runtime()
    src = rt.add(MySource("source"), MySourceConfig(...))
    chn = rt.add(Channelizer("chan"), ChannelizerConfig(...))
    rt.connect(src, "iq", chn, "iq", queue_depth_seconds=1.0)
    rt.start()
    ...
    rt.stop()
"""

from __future__ import annotations

import json
import logging

from .block import Block
from .stream import (
    CoherentPair,
    Stream,
    make_event_stream,
    make_sample_stream,
)
from .types import BlockConfig

log = logging.getLogger(__name__)


class Runtime:
    """Owns the block graph and its lifecycle.

    Blocks are added with :meth:`add`, wired with :meth:`connect`, started
    with :meth:`start`, stopped with :meth:`stop`.  The graph is immutable
    after :meth:`start` — no block / connection changes during operation.

    Start order is **sinks-to-sources** (blocks added later run their
    consumers first); stop order is the reverse.  The simple convention
    is: add blocks in data-flow order (Source first, Reporter last) and
    the runtime handles the rest.
    """

    def __init__(self):
        self._blocks: list[Block] = []
        self._configs: dict[Block, BlockConfig] = {}
        self._streams: list[Stream] = []
        # (src, src_port, dst, dst_port, stream)
        self._connections: list[tuple[Block, str, Block, str, Stream]] = []
        self._started = False

    # --- Graph construction -------------------------------------------

    def add(self, block: Block, config: BlockConfig) -> Block:
        """Register a block with its config.  Returns the block for
        fluent chaining.  Must be called before :meth:`connect`.
        """
        if self._started:
            raise RuntimeError("cannot add blocks after start()")
        if block in self._blocks:
            raise ValueError(f"block {block.name!r} already added")
        # Auto-populate the config's `name` so log prefixes match the block.
        if isinstance(config, BlockConfig) and not config.name:
            config.name = block.name
        self._blocks.append(block)
        self._configs[block] = config
        return block

    def connect(
        self,
        src: Block,
        src_port: str,
        dst: Block,
        dst_port: str,
        *,
        stream: Stream | None = None,
        queue_depth_seconds: float | None = None,
        capacity: int | None = None,
        on_full: str | None = None,
        kind: str = Stream.KIND_SAMPLES,
        sample_rate_hz: int = 12_000,
        n_samples_per_record: int = 256,
        durable: bool = False,
    ) -> Stream:
        """Wire ``src.outputs[src_port]`` -> ``dst.inputs[dst_port]``.

        Either pass an existing ``stream`` (e.g. one half of a coherent
        pair built externally), or let the runtime construct one from
        the policy parameters.  For sample streams pass
        ``queue_depth_seconds`` (and ``sample_rate_hz`` /
        ``n_samples_per_record`` for sizing); for event streams pass
        ``capacity`` and optionally ``durable=True``.
        """
        if self._started:
            raise RuntimeError("cannot connect after start()")
        if src not in self._blocks:
            raise ValueError(f"src block {src.name!r} not added to runtime")
        if dst not in self._blocks:
            raise ValueError(f"dst block {dst.name!r} not added to runtime")
        if src_port in src.outputs:
            raise ValueError(
                f"{src.name!r} output port {src_port!r} already connected"
            )
        if dst_port in dst.inputs:
            raise ValueError(
                f"{dst.name!r} input port {dst_port!r} already connected"
            )

        if stream is None:
            stream_name = f"{src.name}.{src_port}->{dst.name}.{dst_port}"
            if kind == Stream.KIND_SAMPLES:
                if queue_depth_seconds is None:
                    queue_depth_seconds = 1.0
                stream = make_sample_stream(
                    stream_name,
                    queue_depth_seconds=queue_depth_seconds,
                    sample_rate_hz=sample_rate_hz,
                    n_samples_per_record=n_samples_per_record,
                    on_full=on_full or Stream.POLICY_BLOCK,
                )
            elif kind == Stream.KIND_EVENTS:
                if capacity is None:
                    capacity = 16
                stream = make_event_stream(
                    stream_name, capacity=capacity, durable=durable,
                )
                if on_full is not None:
                    # Caller asked for an explicit override.
                    if on_full not in Stream._ALL_POLICIES:
                        raise ValueError(f"unknown on_full policy: {on_full!r}")
                    stream.on_full = on_full
            else:
                raise ValueError(f"unknown stream kind: {kind!r}")

        src.outputs[src_port] = stream
        dst.inputs[dst_port] = stream
        self._streams.append(stream)
        self._connections.append((src, src_port, dst, dst_port, stream))
        log.debug(
            "connect %s.%s -> %s.%s (capacity=%d, on_full=%s, kind=%s)",
            src.name, src_port, dst.name, dst_port,
            stream.capacity, stream.on_full, stream.kind,
        )
        return stream

    def connect_pair(
        self,
        src: Block,
        src_port_h: str,
        src_port_v: str,
        dst: Block,
        dst_port_h: str,
        dst_port_v: str,
        *,
        queue_depth_seconds: float = 1.0,
        sample_rate_hz: int = 48_000,
        n_samples_per_record: int = 1024,
    ) -> CoherentPair:
        """Wire a coherent H/V pair end-to-end.  See §3.2 / §6.1.

        The two streams are constructed with equal capacity and
        ``POLICY_BLOCK`` as required by :class:`CoherentPair`.  The
        producing block is responsible for using
        :meth:`CoherentPair.put` to admit records (so the sample-alignment
        check fires); the consuming block reads each underlying stream
        independently.
        """
        s_h = self.connect(
            src, src_port_h, dst, dst_port_h,
            queue_depth_seconds=queue_depth_seconds,
            sample_rate_hz=sample_rate_hz,
            n_samples_per_record=n_samples_per_record,
            on_full=Stream.POLICY_BLOCK,
        )
        s_v = self.connect(
            src, src_port_v, dst, dst_port_v,
            queue_depth_seconds=queue_depth_seconds,
            sample_rate_hz=sample_rate_hz,
            n_samples_per_record=n_samples_per_record,
            on_full=Stream.POLICY_BLOCK,
        )
        pair_name = f"pair:{src.name}->{dst.name}"
        return CoherentPair(name=pair_name, stream_h=s_h, stream_v=s_v)

    # --- Lifecycle ----------------------------------------------------

    def start(self) -> None:
        """Configure all blocks, then start them in sinks-to-sources order."""
        if self._started:
            raise RuntimeError("runtime already started")
        # Configure all first; configure errors abort before any thread runs.
        for b in self._blocks:
            cfg = self._configs[b]
            try:
                b.configure(cfg)
            except Exception:
                log.exception("configure %s failed", b.name)
                raise
        # Start in reverse list order so consumers are ready before producers.
        # Caller convention: add blocks in data-flow order (Source first).
        for b in reversed(self._blocks):
            b.start()
        self._started = True

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop in sources-to-sinks order.  Idempotent."""
        if not self._started:
            return
        for b in self._blocks:
            try:
                b.stop(timeout_s)
            except Exception:
                log.exception("stop %s failed", b.name)
        self._started = False

    # --- Observability ------------------------------------------------

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Read-only view of registered blocks (in add-order)."""
        return tuple(self._blocks)

    @property
    def streams(self) -> tuple[Stream, ...]:
        """Read-only view of registered streams."""
        return tuple(self._streams)

    def stats_jsonl(self) -> str:
        """Emit per-block stats as JSONL — one line per block, in add-order.

        Used by ``--stats-jsonl-path`` for periodic dumps and by tests
        that want a machine-readable snapshot.  See §9 #8.
        """
        lines = []
        for b in self._blocks:
            try:
                lines.append(json.dumps(b.stats(), default=str))
            except Exception:
                log.exception("stats for %s failed", b.name)
                lines.append(json.dumps({"name": b.name, "error": "stats() raised"}))
        return "\n".join(lines)
