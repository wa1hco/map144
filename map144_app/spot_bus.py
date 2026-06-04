# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Normalized spot bus for the Band Map.

One common record (:class:`Spot`) that all sources collapse to, so the map (and
later the #48 geometry engine) consume a single schema regardless of origin.
Sources: PSKReporter (tail of ``pskr_spots.jsonl``) and MAP144's own decodes
(tail of ``decodes.jsonl``).  WSJT-X ``ALL.TXT`` lands next — same :class:`Spot`.

Layering: this module is data only (file I/O + normalization + an aging
in-memory store).  No Qt, no projection, no drawing — the panel does that.  The
geometry lives in :mod:`map144_app.geo`.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from . import geo

# Maidenhead grid token (4 or 6 char) vs a report/other token.
_GRID_RE = re.compile(r"^[A-R]{2}[0-9]{2}([a-x]{2})?$", re.IGNORECASE)
# MSK144 acknowledgment / report tokens that *look* like grids or numbers but
# encode no location.  'RR73' is the sign-off (also a valid-looking square);
# reports are 73 / -01 / R-05 / +00 / 599.
_NONGRID_TOKENS = {"RR73", "RRR", "RR", "R", "73"}
_REPORT_RE = re.compile(r"^R?[+-]?\d+$")
# Synthetic simulator messages (generate_msk144 encoding) — never plot these.
# See feedback_filter_test_signals_from_realdata_stats memory.
_TESTMSG_RE = re.compile(r"^A[PN]\d{2}\d{3}[PN]\d{2}$")


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
    grid_src: str | None = None   # how tx_grid was set: net/msg/live/static

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
        grid_src="net" if _clean_grid(raw.get("sl")) else None,
    )


def parse_msk144_message(message):
    """Extract (tx_call, tx_grid) from an MSK144 message.

    MSK144 messages are '<TO> <FROM> <report/grid>'; the FROM (2nd token) is the
    transmitter (see msk144_callsign_order_tx_is_second memory).  Returns
    (None, None) for empty/unusable/synthetic-test messages so they're dropped.
    """
    msg = (message or "").strip()
    if not msg or _TESTMSG_RE.match(msg.replace(" ", "")):
        return None, None
    toks = msg.split()
    if len(toks) < 2:
        return None, None
    tx = toks[1]
    grid = None
    for tok in toks[2:]:                    # a grid token = location; reports aren't
        if tok.upper() in _NONGRID_TOKENS or _REPORT_RE.match(tok):
            continue
        if _GRID_RE.match(tok):
            grid = tok[:4].upper() + tok[4:].lower()
            break
    return tx, grid


def _parse_map144_ts(ts):
    """MAP144 'YYYY-MM-DD_HH:MM:SS(.f)' (UTC) -> epoch seconds, or None."""
    ts = (ts or "").strip()
    for fmt in ("%Y-%m-%d_%H:%M:%S.%f", "%Y-%m-%d_%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def normalize_map144(raw, my_call, my_grid):
    """Map a MAP144 decode record to a Spot (rx = home).  None if unusable."""
    tx, tx_grid = parse_msk144_message(raw.get("message"))
    if tx is None:
        return None
    khz = raw.get("radio_khz")
    snr = raw.get("jt9_snr_db")
    if snr is None:
        snr = raw.get("est_snr_db")
    return Spot(
        source="map144",
        t=_parse_map144_ts(raw.get("timestamp")),
        freq_hz=(khz * 1000.0 if khz else None),
        mode="MSK144",
        tx_call=tx,
        tx_grid=tx_grid,
        rx_call=my_call or None,
        rx_grid=_clean_grid(my_grid),
        snr_db=snr,
        grid_src="msg" if tx_grid else None,
    )


class JsonlTailer:
    """Incremental, byte-offset tail of a JSONL file, normalized via `normalize`.

    Only complete (newline-terminated) lines are consumed, so a half-written
    final line from the producer is never parsed.  Truncation / rotation (size <
    last offset) resets to the start.  `normalize(raw)` may return None to drop a
    record (e.g. a MAP144 error row or synthetic-test decode)."""

    def __init__(self, path, normalize):
        self.path = path
        self._normalize = normalize
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
            spot = self._normalize(raw)
            if spot is not None:
                out.append(spot)
        return out


class PskrTailer(JsonlTailer):
    """PSKReporter capture tailer (normalize_pskr)."""

    def __init__(self, path):
        super().__init__(path, normalize_pskr)


def grid_index(spots, extra=None):
    """call (upper) -> grid, harvested from spots that carry both endpoints'
    grids, plus an optional `extra` dict (e.g. the QRZ cache).  Used to locate
    MAP144 decodes whose message had no grid, from PSKReporter's richer data."""
    idx = {}
    for call, grid in (extra or {}).items():
        if call and grid:
            idx[call.upper()] = grid
    for s in spots:
        if s.tx_call and s.tx_grid:
            idx.setdefault(s.tx_call.upper(), s.tx_grid)
        if s.rx_call and s.rx_grid:
            idx.setdefault(s.rx_call.upper(), s.rx_grid)
    return idx


def load_qrz_cache_grids(path):
    """Read-only call->grid from the qrz_lookup cache file (no network)."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k.upper(): v["grid"] for k, v in d.items()
            if isinstance(v, dict) and v.get("grid")}


_ADIF_CALL_RE = re.compile(r"<call:\d+(?::[^>]*)?>\s*([A-Z0-9/]+)", re.IGNORECASE)
_ADIF_GRID_RE = re.compile(
    r"<gridsquare:\d+(?::[^>]*)?>\s*([A-R]{2}\d{2}(?:[a-x]{2})?)", re.IGNORECASE)
_ADIF_DATE_RE = re.compile(r"<qso_date:\d+(?::[^>]*)?>\s*(\d{8})", re.IGNORECASE)


def load_adif_grids(globs):
    """Build call(upper) -> grid from WSJT-X/LoTW ADIF logs (worked stations).

    `globs` is a list of glob patterns; each matched file is parsed per <eor>
    record for <CALL>/<GRIDSQUARE>/<QSO_DATE>.  The grid from the **most recent**
    QSO wins (ties broken toward the more precise / longer grid) — so a permanent
    move, or a callsign reassigned to a new holder in a different grid, self-
    corrects once the newer QSO is logged, instead of being pinned to an old
    (possibly Silent-Key) location.  Still only a *best guess*; live sources
    override it.  Local, instant, network-free.  Bad files are skipped.
    """
    best = {}  # call -> (qso_date_int, grid)
    for pattern in globs:
        for path in glob.glob(os.path.expanduser(pattern)):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    data = f.read()
            except OSError:
                continue
            for rec in re.split(r"<eor>", data, flags=re.IGNORECASE):
                m = _ADIF_CALL_RE.search(rec)
                g = _ADIF_GRID_RE.search(rec)
                if not (m and g):
                    continue
                call = m.group(1).upper()
                grid = g.group(1)
                d = _ADIF_DATE_RE.search(rec)
                date = int(d.group(1)) if d else 0
                prev = best.get(call)
                if (prev is None or date > prev[0]
                        or (date == prev[0] and len(grid) > len(prev[1]))):
                    best[call] = (date, grid)
    return {call: dg[1] for call, dg in best.items()}


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
