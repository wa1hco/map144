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
"""Block / Stream / Record scaffolding for the Phase 3 pipeline refactor.

This subpackage implements the contracts laid out in
``docs/block-stream-design.md`` (TODO #6).  It is intentionally separate
from the existing DSP code in ``map144_app/`` — the goal of Phase 3 is to
build the new architecture *alongside* the working pipeline, then migrate
producers / consumers into blocks one at a time (TODOs #7–#12).

Public API
----------
``Record``, ``Event``       Typed payloads on streams (§3.1, §3.3 in the doc).
``BlockConfig``             Base class for per-block typed config (§9 #2).
``Stream``                  Bounded queue between blocks (§3, §5).
``CoherentPair``            Two streams with sample-aligned lockstep
                            backpressure (§3.2).
``Block``                   Block base class — one OS thread per block (§4).
``Runtime``                 Graph wiring + lifecycle (§7, §9 #3).
``StreamClosed``            Raised by ``Stream.get`` after producer closed.
"""

from __future__ import annotations

from .block import Block
from .graph import Runtime
from .stream import (
    CoherentPair,
    Stream,
    make_event_stream,
    make_sample_stream,
)
from .types import BlockConfig, Event, Record, StreamClosed

__all__ = [
    "Block",
    "BlockConfig",
    "CoherentPair",
    "Event",
    "Record",
    "Runtime",
    "Stream",
    "StreamClosed",
    "make_event_stream",
    "make_sample_stream",
]
