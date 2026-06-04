# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Normalized spot bus for the Band Map.

One common record (:class:`Spot`) that all sources collapse to, so the map (and
later the #48 geometry engine) consume a single schema regardless of origin.
Sources for v1: PSKReporter (tail of ``pskr_spots.jsonl``).  MAP144-live and
WSJT-X ``ALL.TXT`` tailers land in step 2 — they emit the same :class:`Spot`.

Layering: this module is data only (file I/O + normalization + an aging
in-memory store).  No Qt, no projection, no drawing — the panel does that.  The
geometry lives in :mod:`map144_app.geo`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import geo


@dataclass
class Spot:
    """One reception report, source-normalized."""
    source: str            # 'pskr' | 'map144' | 'wsjtx'
    t: float | None        # report epoch seconds (integer for PSKReporter)
    freq_hz: float | None
    mode: str | None
    tx_call: str | None
    tx_grid: str | None
    rx_call: str | None
    rx_grid: str | None
    snr_db: float | None

    def tx_latlon(self):
        return _safe_latlon(self.tx_grid)

    def rx_latlon(self):
        return _safe_latlon(self.rx_grid)

    def resolvable(self) -> bool:
        """Both endpoints have a usable grid -> the path can be drawn."""
        return self.tx_latlon() is not None and self.rx_latlon() is not None


def _safe_latlon(grid):
    if not grid:
        return None
    try:
        return geo.grid_to_latlon(grid)
    except (ValueError, TypeError):
        return None


def _clean_grid(g):
    g = (g or "").strip()
    return g or None


def normalize_pskr(raw: dict) -> Spot:
    """Map a raw PSKReporter MQTT spot (sc/sl/rc/rl/f/md/rp/t) to a Spot."""
    return Spot(
        source="pskr",
        t=raw.get("t", raw.get("t_tx")),
        freq_hz=raw.get("f"),
        mode=raw.get("md"),
        tx_call=raw.get("sc"),
        tx_grid=_clean_grid(raw.get("sl")),
        rx_call=raw.get("rc"),
        rx_grid=_clean_grid(raw.get("rl")),
        snr_db=raw.get("rp"),
    )


class PskrTailer:
    """Incremental, byte-offset tail of the PSKReporter JSONL capture file.

    Only complete (newline-terminated) lines are consumed, so a half-written
    final line from the logger is never parsed.  Truncation / rotation (size <
    last offset) resets to the start.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0

    def poll(self) -> list:
        """Return normalized Spots appended since the last poll (possibly [])."""
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size < self._offset:            # truncated / rotated
            self._offset = 0
        if size == self._offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        cut = data.rfind(b"\n")
        if cut == -1:                      # no complete line yet
            return []
        chunk = data[: cut + 1]
        self._offset += len(chunk)
        out = []
        for line in chunk.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            out.append(normalize_pskr(raw))
        return out


class SpotStore:
    """In-memory aging store of recent Spots feeding the live map."""

    def __init__(self, max_age_s: float = 900.0):
        self.max_age_s = max_age_s
        self._spots: list[Spot] = []

    def add(self, spots) -> None:
        self._spots.extend(spots)

    def prune(self, now: float) -> None:
        cutoff = now - self.max_age_s
        self._spots = [s for s in self._spots if (s.t or 0) >= cutoff]

    def spots(self) -> list:
        return list(self._spots)

    def drawable(self) -> list:
        """Spots whose both endpoints geocode (ready to draw as an arc)."""
        return [s for s in self._spots if s.resolvable()]

    def endpoints_latlon(self) -> list:
        """All resolvable endpoints (tx and rx) for :func:`geo.auto_frame`."""
        pts = []
        for s in self._spots:
            for p in (s.tx_latlon(), s.rx_latlon()):
                if p is not None:
                    pts.append(p)
        return pts

    def copath_groups(self, window_s: float = 1.0) -> list:
        """Same-tx, multi-rx, within-window groups (multipath candidates),
        returned as lists of :class:`Spot`."""
        wrapped = [
            {"tx_call": s.tx_call, "rx_call": s.rx_call, "t": s.t, "_spot": s}
            for s in self._spots
        ]
        return [[w["_spot"] for w in grp]
                for grp in geo.cluster_copaths(wrapped, window_s=window_s)]
