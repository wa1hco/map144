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
"""FlexSource — Block-form ingress for FlexRadio DAX-IQ.

See ``docs/block-stream-design.md`` §6 (graph) and §3.5 (two clocks).
This block is the Phase 3 wrapper around :class:`flexclient.FlexDAXIQ`.
The legacy ingest loop in :mod:`map144_app.runtime` is **not** touched;
FlexSource runs in parallel inside the new graph for replay /
test / Phase-3 production wiring.

Client protocol
---------------
FlexSource depends on a small protocol that the production ``FlexDAXIQ``
satisfies and that test fakes can emulate trivially:

- ``client.sample_queue`` — ``queue.Queue`` of objects with a ``.samples``
  attribute (numpy complex array, in ``±FLEX_DAXIQ_FULL_SCALE`` units).
- ``client.start()`` — start UDP / VITA receive.  May be a no-op if the
  caller already started the client.
- ``client.stop()``  — stop receive / clean up.  Must be idempotent.

The packet's optional ``timestamp_int`` / ``timestamp_frac`` fields are
*ignored* here — per the ``project_flex_timing`` memory the project
already runs Flex with ``timestamp_int=0`` and recovers wall clock from
``time.time()``.  The §3.5 dual-clock contract is implemented by the
:class:`Source` base class regardless.

Out of scope for v1
-------------------
- TX gating (the legacy code drains the queue while transmitting and
  resets channelizer state on TX→RX).  Under the Block model that
  belongs in a downstream ``TXGate`` block, not in the Source.
- Polarization handling (Flex is single-channel; a future
  ``DualPolFlexSource`` will use the H/V coherent-pair pattern when a
  multi-slice configuration is in play).
- Auto-construct from config.  The caller passes in a constructed
  client so test fakes can be injected directly.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .source import Source, SourceConfig

log = logging.getLogger(__name__)


# Matches map144_app.runtime.FLEX_DAXIQ_FULL_SCALE.  Duplicated rather
# than imported because runtime.py drags in Qt and a lot of DSP code at
# module-load time; this is just a constant.
FLEX_DAXIQ_FULL_SCALE = 32768.0


class _FlexClientProtocol(Protocol):
    """Structural protocol satisfied by ``flexclient.FlexDAXIQ`` and by
    test fakes.  Documented for clarity; not enforced at runtime.
    """

    sample_queue: queue.Queue

    def start(self) -> None: ...
    def stop(self) -> None: ...


@dataclass
class FlexSourceConfig(SourceConfig):
    """Per-FlexSource config.  Inherits clock / drift settings from SourceConfig."""

    sample_rate_hz: int = 48_000

    # Bounded queue ``get`` timeout — short enough that ``stop()`` is responsive,
    # long enough that the run loop doesn't spin when the radio is silent.
    queue_get_timeout_s: float = 0.250

    # If True, FlexSource calls ``client.start()`` in ``on_start`` and
    # ``client.stop()`` in ``on_stop``.  When False, the caller is
    # responsible for the client lifecycle (useful when sharing a single
    # FlexDAXIQ between FlexSource and other code during migration).
    own_client_lifecycle: bool = True


class FlexSource(Source):
    """Source block that wraps a FlexDAXIQ-shaped client.

    The client is injected at construction (rather than built from
    config) so tests can substitute a fake without going through
    discovery / TCP / VITA.  The legacy production code in
    :mod:`map144_app.runtime` continues to instantiate :class:`FlexDAXIQ`
    directly; Phase 3 production wiring will switch to constructing one
    here as part of the graph.
    """

    config_type = FlexSourceConfig

    def __init__(self, name: str, client: _FlexClientProtocol):
        super().__init__(name)
        if client is None:
            raise ValueError(f"{name}: FlexSource requires a non-None client")
        if not hasattr(client, "sample_queue"):
            raise TypeError(
                f"{name}: client must have a 'sample_queue' attribute "
                f"(got {type(client).__name__})"
            )
        self._client: _FlexClientProtocol = client

    def on_start(self) -> None:
        super().on_start()
        cfg: FlexSourceConfig = self.config  # type: ignore[assignment]
        if cfg.own_client_lifecycle:
            try:
                self._client.start()
                log.info("%s: client.start() OK", self.name)
            except Exception:
                log.exception("%s: client.start() failed", self.name)
                raise

    def on_stop(self) -> None:
        super().on_stop()
        cfg: FlexSourceConfig = self.config  # type: ignore[assignment]
        if cfg.own_client_lifecycle:
            try:
                self._client.stop()
                log.info("%s: client.stop() OK", self.name)
            except Exception:
                log.exception("%s: client.stop() failed", self.name)
                # do not re-raise during stop

    def produce(self) -> np.ndarray | None:
        cfg: FlexSourceConfig = self.config  # type: ignore[assignment]

        # Fast path: drain whatever is already enqueued.  Falling through to
        # the bounded blocking get keeps the run loop responsive when the
        # radio is silent (e.g. during TX or fade) without spinning.
        try:
            packet = self._client.sample_queue.get_nowait()
        except queue.Empty:
            try:
                packet = self._client.sample_queue.get(
                    timeout=cfg.queue_get_timeout_s,
                )
            except queue.Empty:
                return None  # no data this tick; run loop checks _stopping

        # Pull the IQ array out of the packet and normalise.  We accept
        # any object whose ``.samples`` is array-like and complex-valued
        # so test fakes can pass numpy arrays directly.
        raw = getattr(packet, "samples", None)
        if raw is None:
            log.warning(
                "%s: packet had no .samples attribute (got %r); skipping",
                self.name, type(packet).__name__,
            )
            return None
        samples = np.asarray(raw, dtype=np.complex64) / FLEX_DAXIQ_FULL_SCALE
        if samples.size == 0:
            return None
        # Flex is single-channel; legacy code passes through
        # _polarization_combine which is a no-op for mono.  Future
        # DualPolFlexSource will emit on a CoherentPair instead.
        return samples
