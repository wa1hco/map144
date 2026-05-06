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
"""JsonlSink — records an event stream to a JSONL file.

See ``docs/block-stream-design.md`` §9 #10.  Together with WavSource,
this is the minimum tooling for the §10 bit-exact-replay validation
gate: reproduce a captured IQ session, record every detection /
decode / clock-health event, and diff against the reference run.

One event per line (JSON-encoded).  ``data`` fields that aren't
JSON-serialisable are stringified via ``json.dumps(default=str)``.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .block import Block
from .types import BlockConfig, Event, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class JsonlSinkConfig(BlockConfig):
    """Per-JsonlSink config."""

    output_path: str = ""
    input_port_name: str = "events"

    # Bounded ``get`` timeout.  Short enough to react to ``stop()`` quickly.
    get_timeout_s: float = 0.250

    # Flush every N events; 1 = flush after every line (good for crash
    # safety, mild perf cost).
    flush_every_n: int = 1

    # Optional list of event ``kind`` strings to record; empty = record
    # everything.  Useful when one input stream multiplexes several kinds
    # (e.g. detection + clock_health + decode all on one debug stream).
    kind_filter: list[str] = field(default_factory=list)


class JsonlSink(Block):
    """Consumer block that writes incoming :class:`Event` records to JSONL.

    Each event becomes one JSON object on its own line:

        {"kind": "detection", "occurred_at_sample": 12345,
         "sample_rate_hz": 12000, "wall_clock_at_production": 1700000000.5,
         "payload": {...}}

    Stops when the upstream stream is closed (``StreamClosed``) or when
    ``Runtime.stop()`` is called.
    """

    config_type = JsonlSinkConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._fp: io.TextIOBase | None = None
        self._n_written: int = 0
        self._n_filtered: int = 0
        self._kind_filter: set[str] = set()

    def configure(self, config: JsonlSinkConfig) -> None:  # type: ignore[override]
        super().configure(config)
        if not config.output_path:
            raise ValueError(f"{self.name}: JsonlSinkConfig.output_path is required")
        self._kind_filter = set(config.kind_filter)

    def on_start(self) -> None:
        super().on_start()
        cfg: JsonlSinkConfig = self.config  # type: ignore[assignment]
        path = Path(cfg.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", encoding="utf-8")
        log.info("%s: writing JSONL to %s", self.name, path)

    def on_stop(self) -> None:
        super().on_stop()
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                log.exception("%s: error closing output file", self.name)
            self._fp = None
        log.info(
            "%s: closed (wrote %d events, filtered %d)",
            self.name, self._n_written, self._n_filtered,
        )

    def tick(self) -> None:
        cfg: JsonlSinkConfig = self.config  # type: ignore[assignment]
        port = cfg.input_port_name
        if port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {port!r} not connected"
            )
        try:
            item = self.inputs[port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return  # nothing this tick; loop continues
        # ``StreamClosed`` propagates to the run loop and exits cleanly.

        if not isinstance(item, Event):
            log.warning(
                "%s: dropped non-Event %r on input port", self.name, type(item).__name__,
            )
            return

        if self._kind_filter and item.kind not in self._kind_filter:
            self._n_filtered += 1
            return

        record = {
            "kind": item.kind,
            "occurred_at_sample": item.occurred_at_sample,
            "sample_rate_hz": item.sample_rate_hz,
            "wall_clock_at_production": item.wall_clock_at_production,
            "payload": item.payload,
        }
        line = json.dumps(record, default=str)
        assert self._fp is not None
        self._fp.write(line + "\n")
        self._n_written += 1
        if cfg.flush_every_n > 0 and self._n_written % cfg.flush_every_n == 0:
            self._fp.flush()
