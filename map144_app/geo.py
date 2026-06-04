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
"""Pure geographic helpers for the Band Map panel.

No Qt, no I/O, no heavy geo dependency — just numpy + math so the math is
unit-testable in isolation and reusable by #26 (range) and the #48 geometry
engine.  Conventions:
  * lat/lon in decimal degrees, lon positive east, WGS84 sphere (mean radius).
  * Maidenhead grids: 4-char (field+square) or 6-char (+subsquare); converters
    return the *center* of the square at the precision supplied.
  * great circles are computed on the unit sphere (slerp), independent of the
    map projection used to draw them.
"""
from __future__ import annotations

import json
import math

import numpy as np

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- #
# Maidenhead <-> lat/lon
# --------------------------------------------------------------------------- #
def grid_to_latlon(grid: str) -> tuple:
    """Maidenhead locator -> (lat, lon) at the *center* of the square.

    Accepts 4-char (FN42) or 6-char (FN42mm); longer is truncated to 6.
    Raises ValueError on malformed input.
    """
    g = (grid or "").strip()
    if len(g) < 4:
        raise ValueError(f"grid too short: {grid!r}")
    g = g[:6]
    lon = (ord(g[0].upper()) - ord("A")) * 20.0 - 180.0
    lat = (ord(g[1].upper()) - ord("A")) * 10.0 - 90.0
    if not (g[2].isdigit() and g[3].isdigit()):
        raise ValueError(f"bad grid squares: {grid!r}")
    lon += int(g[2]) * 2.0
    lat += int(g[3]) * 1.0
    if len(g) >= 6:
        lon += (ord(g[4].lower()) - ord("a")) * (2.0 / 24.0)
        lat += (ord(g[5].lower()) - ord("a")) * (1.0 / 24.0)
        lon += (2.0 / 24.0) / 2.0   # center of subsquare cell
        lat += (1.0 / 24.0) / 2.0
    else:
        lon += 2.0 / 2.0            # center of square cell (2° lon x 1° lat)
        lat += 1.0 / 2.0
    return lat, lon


def latlon_to_grid(lat: float, lon: float, precision: int = 6) -> str:
    """Decimal lat/lon -> Maidenhead locator (mirror of tools/qrz_lookup)."""
    lon = float(lon) + 180.0
    lat = float(lat) + 90.0
    A = ord("A")
    g = (chr(A + int(lon // 20)) + chr(A + int(lat // 10))
         + str(int((lon % 20) // 2)) + str(int((lat % 10) // 1)))
    if precision >= 6:
        g += chr(ord("a") + int((lon % 2) / (2.0 / 24)))
        g += chr(ord("a") + int((lat % 1) / (1.0 / 24)))
    return g


# --------------------------------------------------------------------------- #
# Great-circle geometry (unit sphere)
# --------------------------------------------------------------------------- #
def _to_unit(lat: float, lon: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo),
                     math.sin(la)])


def _to_latlon(v: np.ndarray) -> tuple:
    x, y, z = v
    return math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))


def great_circle_points(lat1, lon1, lat2, lon2, n: int = 32) -> list:
    """Sample n points along the great circle from (lat1,lon1) to (lat2,lon2),
    inclusive of both endpoints.  Used to draw a path as a projected polyline."""
    n = max(2, int(n))
    a, b = _to_unit(lat1, lon1), _to_unit(lat2, lon2)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-9:                       # coincident endpoints
        return [(lat1, lon1), (lat2, lon2)]
    so = math.sin(omega)
    out = []
    for i in range(n):
        t = i / (n - 1)
        s1 = math.sin((1 - t) * omega) / so
        s2 = math.sin(t * omega) / so
        out.append(_to_latlon(s1 * a + s2 * b))
    return out


def great_circle_midpoint(lat1, lon1, lat2, lon2) -> tuple:
    """Midpoint of the great-circle path (the #48 first-order reflection point)."""
    a, b = _to_unit(lat1, lon1), _to_unit(lat2, lon2)
    m = a + b
    nrm = np.linalg.norm(m)
    if nrm < 1e-12:                        # antipodal — midpoint undefined
        return _to_latlon(a)
    return _to_latlon(m / nrm)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km (also serves #26 range)."""
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = la2 - la1
    dlo = math.radians(lon2 - lon1)
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


# --------------------------------------------------------------------------- #
# Projection + auto-framing
# --------------------------------------------------------------------------- #
def project_equirect(lat, lon, lat0: float):
    """Equirectangular projection with cos(lat0) longitude aspect correction.

    Returns (x, y) in degree-ish units suitable for a pyqtgraph plot.  Vectorised:
    lat/lon may be scalars or numpy arrays.  lat0 is the frame's reference
    latitude (use the frame center) so east-west scale ~ matches north-south.
    """
    k = math.cos(math.radians(lat0))
    return np.asarray(lon) * k, np.asarray(lat)


def auto_frame(points, home, pad_frac: float = 0.12, pct: float = 2.0,
               min_span_deg: float = 6.0) -> tuple:
    """Outlier-robust bounding box of activity, always including `home`.

    points : iterable of (lat, lon) to enclose (path endpoints / midpoints).
    home   : (lat, lon) always inside the frame.
    pct    : trim this percentile from each end of lat/lon before bounding,
             so one freak DX spot doesn't zoom the whole continent out.
    Returns (lon_min, lon_max, lat_min, lat_max), padded, with a floor span so
    a single nearby spot doesn't produce a degenerate (zero-area) frame.
    """
    lats = [home[0]]
    lons = [home[1]]
    for la, lo in points:
        lats.append(la)
        lons.append(lo)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    if len(lats) > 4:                      # enough to trim outliers meaningfully
        lo_la, hi_la = np.percentile(lats, [pct, 100 - pct])
        lo_lo, hi_lo = np.percentile(lons, [pct, 100 - pct])
    else:
        lo_la, hi_la = lats.min(), lats.max()
        lo_lo, hi_lo = lons.min(), lons.max()

    # never trim home out of the frame
    lo_la, hi_la = min(lo_la, home[0]), max(hi_la, home[0])
    lo_lo, hi_lo = min(lo_lo, home[1]), max(hi_lo, home[1])

    # enforce a minimum span so a lone nearby spot isn't a pinpoint
    def _floor(lo, hi):
        span = hi - lo
        if span < min_span_deg:
            c = (lo + hi) / 2.0
            lo, hi = c - min_span_deg / 2.0, c + min_span_deg / 2.0
        return lo, hi
    lo_la, hi_la = _floor(lo_la, hi_la)
    lo_lo, hi_lo = _floor(lo_lo, hi_lo)

    pad_la = (hi_la - lo_la) * pad_frac
    pad_lo = (hi_lo - lo_lo) * pad_frac
    return (lo_lo - pad_lo, hi_lo + pad_lo, lo_la - pad_la, hi_la + pad_la)


# --------------------------------------------------------------------------- #
# Basemap: GeoJSON -> projectable polylines
# --------------------------------------------------------------------------- #
def geometry_rings(geom: dict):
    """Yield each ring (list of [lon, lat]) of a GeoJSON geometry."""
    t = geom.get("type")
    c = geom.get("coordinates")
    if c is None:
        return
    if t == "Polygon":
        yield from c
    elif t == "MultiPolygon":
        for poly in c:
            yield from poly
    elif t == "LineString":
        yield c
    elif t == "MultiLineString":
        yield from c


def geojson_geometries(obj):
    """Yield geometry dicts from any of the shapes we encounter: a
    FeatureCollection, a bare geometry, a Feature, or a dict keyed by region
    name -> {'geometry': ...} (GridTracker's shapes.json layout)."""
    if not isinstance(obj, dict):
        return
    t = obj.get("type")
    if t == "FeatureCollection":
        for f in obj.get("features", []):
            g = f.get("geometry")
            if g:
                yield g
    elif t in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
        yield obj
    elif isinstance(obj.get("geometry"), dict):
        yield obj["geometry"]
    else:
        for v in obj.values():
            if isinstance(v, dict) and isinstance(v.get("geometry"), dict):
                yield v["geometry"]


def polylines_from_geojson(obj):
    """Flatten a GeoJSON container to (lons, lats) float arrays with NaN ring
    separators — ready to project and draw as one ``connect='finite'`` curve."""
    lons, lats = [], []
    for geom in geojson_geometries(obj):
        for ring in geometry_rings(geom):
            for pt in ring:
                lons.append(pt[0])
                lats.append(pt[1])
            lons.append(math.nan)
            lats.append(math.nan)
    return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)


def load_basemap(path: str):
    """Load a coastline/border basemap GeoJSON -> (lons, lats); empty if the
    file is missing or unparseable (the map just falls back to a graticule)."""
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return polylines_from_geojson(obj)


# --------------------------------------------------------------------------- #
# Simultaneity clustering — "one ping, multiple paths" candidates
# --------------------------------------------------------------------------- #
def cluster_copaths(spots, window_s: float = 1.0) -> list:
    """Group spots that share a transmitter and fall within `window_s`.

    spots: iterable of dicts with 'tx_call', 'rx_call', 't' (epoch seconds).
    Returns a list of groups, each a list of >=2 spots with the SAME tx heard
    by >=2 distinct receivers inside the window — the multipath candidates.

    Caveat baked into the default: PSKReporter MSK144 timestamps are integer
    seconds (verified), so window_s=1 is the finest honest resolution — NOT the
    300 ms physical criterion.  Sub-second confirmation needs a precise-timing
    endpoint (MAP144 t_sec).  Callers should label these 'candidate', not
    'confirmed', unless a precise endpoint is present.
    """
    by_tx = {}
    for s in spots:
        tx = s.get("tx_call")
        if tx is None or s.get("t") is None:
            continue
        by_tx.setdefault(tx, []).append(s)

    groups = []
    for tx, ss in by_tx.items():
        ss.sort(key=lambda r: r["t"])
        i = 0
        n = len(ss)
        while i < n:
            j = i + 1
            while j < n and ss[j]["t"] - ss[i]["t"] <= window_s:
                j += 1
            window = ss[i:j]
            rxs = {w.get("rx_call") for w in window}
            rxs.discard(None)
            if len(rxs) >= 2:
                groups.append(window)
            i = j
    return groups
