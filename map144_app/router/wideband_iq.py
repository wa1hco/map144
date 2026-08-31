# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Optional MAP65 / QMAP TIMF2 destinations for the router.

Both consumers speak the same Linrad timf2 IQ format.  Defaults:
  MAP65  → 127.0.0.1:50002
  QMAP   → 127.0.0.1:50004
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..map65_export import Map65Exporter

log = logging.getLogger(__name__)

MAP65_DEFAULT_PORT = 50002
QMAP_DEFAULT_PORT = 50004


@dataclass
class WidebandDest:
    name: str          # "MAP65" or "QMAP"
    host: str
    port: int
    center_mhz: float


class WidebandIqBank:
    """One or two Map65Exporter instances (MAP65 and/or QMAP)."""

    def __init__(
        self,
        *,
        pan_center_mhz: float,
        n_channels: int = 1,
        map65: WidebandDest | None = None,
        qmap: WidebandDest | None = None,
        exporter_factory=None,
    ):
        factory = exporter_factory or Map65Exporter
        self.exporters: list[tuple[str, Map65Exporter]] = []
        for dest in (map65, qmap):
            if dest is None:
                continue
            exp = factory(
                host=dest.host,
                port=dest.port,
                pan_center_mhz=pan_center_mhz,
                map65_center_mhz=dest.center_mhz,
                n_channels=n_channels,
            )
            exp.start()
            self.exporters.append((dest.name, exp))
            log.info("WidebandIqBank: %s -> %s:%d centre=%.6f pan=%.6f",
                     dest.name, dest.host, dest.port, dest.center_mhz,
                     pan_center_mhz)

    def process(self, raw_h: np.ndarray, raw_v: np.ndarray | None,
                ts_epoch: float) -> None:
        if not self.exporters:
            return
        if raw_v is None:
            raw_v = raw_h
        for _name, exp in self.exporters:
            exp.process(raw_h, raw_v, ts_epoch)

    def close(self) -> None:
        for _name, exp in self.exporters:
            try:
                exp.stop()
            except Exception:
                pass
        self.exporters.clear()
