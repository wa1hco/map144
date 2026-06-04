# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Band Map panel — live great-circle activity map (v1: PSKReporter layer).

Geographic North-America map (not home-centered — 99.9 % of activity is SSW
through WNW of FN42).  Each reception report is drawn as a great-circle arc
between its endpoints with a small tick at the path midpoint (the #48 first-order
reflection point).  Same-second co-paths (one tx heard by multiple rx within the
±1 s PSKReporter timestamp resolution) are highlighted — the "one ping, multiple
paths" candidates.  Frame auto-fits to recent activity (damped, home always in).

Layering: this panel is pure presentation.  All geometry is in
:mod:`map144_app.geo`; ingestion/normalization/aging in :mod:`map144_app.spot_bus`.
A 1 Hz QTimer polls the spot bus and redraws; it never touches the DSP pipeline.

v2 (not here yet): MAP144-live + WSJT-X ALL.TXT layers, QRZ grid fill for
callsigns whose message carried no locator, and precise-timing (t_sec)
confirmation of the highlighted candidates.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from . import geo
from .spot_bus import (
    JsonlTailer, PskrTailer, SpotStore, grid_index, load_adif_grids,
    load_qrz_cache_grids, normalize_map144,
)

# Local call->grid sources (no network): WSJT-X / JTDX logs + GridTracker's
# LoTW ADIF.  Override with MAP144_ADIF_GLOBS (':'-separated glob patterns).
_ADIF_GLOBS = os.environ.get("MAP144_ADIF_GLOBS", ":".join([
    "~/.local/share/WSJT-X*/wsjtx_log.adi",
    "~/.local/share/JTDX*/wsjtx_log.adi",
    "~/.config/GridTracker2/Ginternal/LoTW_QSL.adif",
])).split(":")

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DET_DIR = os.path.join(_PROJ_ROOT, "MSK144", "detections")
_PSKR_PATH = os.path.join(_DET_DIR, "pskr_spots.jsonl")
_DECODES_PATH = os.path.join(_DET_DIR, "decodes.jsonl")
_QRZ_CACHE_PATH = os.path.join(_DET_DIR, "qrz_grid_cache.json")

# Coastline / state-border basemap.  Loaded at runtime from GridTracker's data
# (US states + Canadian provinces, GPL, already installed) — not copied into the
# repo.  Override with MAP144_BASEMAP; if absent the map falls back to graticule.
_BASEMAP_PATH = os.environ.get(
    "MAP144_BASEMAP", "/usr/share/gridtracker/data/shapes.json")

_MAX_AGE_S = 900.0          # 15-minute trailing window
_FRAME_DAMP_S = 20.0        # re-fit the frame at most this often
_REFRAME_MARGIN = 1.5       # also re-fit if activity drifts this far (deg) outside


def _project(latlons, lat0):
    """Project a list of (lat, lon) to (xs, ys) arrays with cos(lat0) aspect."""
    if not latlons:
        return np.array([]), np.array([])
    lats = np.array([p[0] for p in latlons], dtype=float)
    lons = np.array([p[1] for p in latlons], dtype=float)
    xs, ys = geo.project_equirect(lats, lons, lat0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def setup_band_map_window(self, view_action):
    """Build the Band Map panel window and start its update timer."""
    from .ui import _PanelWindow
    from .visualizer import _SETTINGS

    win = _PanelWindow("map144 — Band Map", view_action, self, "band_map_geometry")
    win.setMinimumSize(640, 480)
    layout = QtWidgets.QVBoxLayout(win)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)
    self._band_map_win = win

    # ── home location ─────────────────────────────────────────────────────
    home_grid = str(_SETTINGS.value("reporting_mygrid", "") or "").strip() or "FN42"
    try:
        home_ll = geo.grid_to_latlon(home_grid)
    except (ValueError, TypeError):
        home_ll = geo.grid_to_latlon("FN42")
    win._bm_home_ll = home_ll
    win._bm_home_grid = home_grid
    win._bm_my_call = str(_SETTINGS.value("reporting_mycall", "") or "").strip()

    # ── controls ──────────────────────────────────────────────────────────
    ctl = QtWidgets.QHBoxLayout()
    win._bm_autoframe = QtWidgets.QCheckBox("Auto-frame")
    win._bm_autoframe.setChecked(True)
    win._bm_highlight = QtWidgets.QCheckBox("Highlight simultaneous")
    win._bm_highlight.setChecked(True)
    win._bm_mine = QtWidgets.QCheckBox("My decodes")   # MAP144's own, in green
    win._bm_mine.setChecked(True)
    # in-range filter: 2 m/6 m single-hop MS/tropo tops out ~2300 km, so spots
    # with neither endpoint near home aren't workable from here — hide them so
    # the frame stays on relevant activity instead of spanning an ocean.
    win._bm_inrange = QtWidgets.QCheckBox("In range ≤")
    win._bm_inrange.setChecked(_SETTINGS.value("band_map_inrange", True, type=bool))
    win._bm_range_km = QtWidgets.QSpinBox()
    win._bm_range_km.setRange(250, 20000)
    win._bm_range_km.setSingleStep(250)
    win._bm_range_km.setSuffix(" km")
    win._bm_range_km.setValue(int(_SETTINGS.value("band_map_range_km", 2500, type=int)))
    win._bm_inrange.stateChanged.connect(
        lambda _v: _SETTINGS.setValue("band_map_inrange", win._bm_inrange.isChecked()))
    win._bm_range_km.valueChanged.connect(
        lambda v: _SETTINGS.setValue("band_map_range_km", int(v)))
    win._bm_status = QtWidgets.QLabel("starting…")
    ctl.addWidget(win._bm_autoframe)
    ctl.addWidget(win._bm_highlight)
    ctl.addWidget(win._bm_mine)
    ctl.addWidget(win._bm_inrange)
    ctl.addWidget(win._bm_range_km)
    ctl.addStretch(1)
    ctl.addWidget(win._bm_status)
    layout.addLayout(ctl)

    # ── plot ──────────────────────────────────────────────────────────────
    plot = pg.PlotWidget(background="w")    # light mode (CLAUDE.md GUI preference)
    # Plot in REAL lon/lat (so axis labels read true degrees) and lock the view
    # aspect to cos(lat0) in _band_map_redraw, so 1° lon renders as cos(lat0) of
    # 1° lat — geography is preserved in any window shape (no manual 4:1).
    plot.setAspectLocked(True, ratio=1.0)
    plot.showGrid(x=False, y=False)
    plot.getViewBox().setMouseEnabled(x=True, y=True)
    plot.setLabel("bottom", "")
    plot.setLabel("left", "")
    for _axname in ("bottom", "left"):
        _ax = plot.getAxis(_axname)
        _ax.setPen("#555555")
        _ax.setTextPen("#555555")
    layout.addWidget(plot, 1)
    win._bm_plot = plot
    vb = plot.getViewBox()
    vb.invertX(False)                      # lon increases east -> +x

    # drawn items (created once; updated via setData each tick)
    # basemap first so it sits behind arcs; reprojected only when lat0 changes
    win._bm_base_lon, win._bm_base_lat = geo.load_basemap(_BASEMAP_PATH)
    win._bm_basemap = plot.plot([], [], pen=pg.mkPen("#8a96a6", width=1),
                                connect="finite")
    win._bm_base_drawn = False
    win._bm_aspect_k = None

    win._bm_graticule = plot.plot([], [], pen=pg.mkPen("#dde2e8", width=1),
                                  connect="finite")
    # three age bands on a light background: fresh = strong/dark -> old = faint
    win._bm_arcs = [
        plot.plot([], [], pen=pg.mkPen(c, width=w), connect="finite")
        for c, w in (("#0a48a8", 2.0), ("#5b86c4", 1.6), ("#a9c0de", 1.2))
    ]
    win._bm_arc_hi = plot.plot([], [], pen=pg.mkPen("#e8500a", width=2.6),
                               connect="finite")
    win._bm_mid = pg.ScatterPlotItem(size=4, brush=pg.mkBrush("#c00000"),
                                     pen=None)
    plot.addItem(win._bm_mid)
    win._bm_mid_hi = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#ff9000"),
                                        pen=pg.mkPen("#a04000", width=1))
    plot.addItem(win._bm_mid_hi)
    # MAP144's own decodes — green, above the PSKReporter layer.  Confident
    # (message/live grid) = solid + filled; best-guess (static home grid) =
    # dashed + hollow, so a possibly-stale location doesn't look authoritative.
    win._bm_mine_arcs = plot.plot([], [], pen=pg.mkPen("#108a2e", width=2.4),
                                  connect="finite")
    win._bm_mine_pts = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#108a2e"),
                                          pen=pg.mkPen("#0a5018", width=1))
    plot.addItem(win._bm_mine_pts)
    win._bm_mine_arcs_bg = plot.plot(
        [], [], pen=pg.mkPen("#86b896", width=1.6, style=QtCore.Qt.DashLine),
        connect="finite")
    win._bm_mine_pts_bg = pg.ScatterPlotItem(   # hollow = best guess
        size=8, brush=None, pen=pg.mkPen("#108a2e", width=1.3))
    plot.addItem(win._bm_mine_pts_bg)
    win._bm_home = pg.ScatterPlotItem(
        size=15, symbol="star", brush=pg.mkBrush("#e00000"),
        pen=pg.mkPen("#600000", width=1))
    plot.addItem(win._bm_home)

    # ── data plumbing ─────────────────────────────────────────────────────
    win._bm_tailer = PskrTailer(_PSKR_PATH)
    win._bm_map144_tailer = JsonlTailer(
        _DECODES_PATH,
        lambda r: normalize_map144(r, win._bm_my_call, win._bm_home_grid))
    # static call->grid: ADIF logs (worked stations) + QRZ cache, network-free
    win._bm_static_grids = load_adif_grids(_ADIF_GLOBS)
    win._bm_static_grids.update(load_qrz_cache_grids(_QRZ_CACHE_PATH))
    win._bm_store = SpotStore(max_age_s=_MAX_AGE_S)
    win._bm_last_frame_t = 0.0
    win._bm_frame = None                   # (lon_min, lon_max, lat_min, lat_max)
    win._bm_lat0 = home_ll[0]

    timer = QtCore.QTimer(win)
    timer.setInterval(1000)
    timer.timeout.connect(lambda: _band_map_tick(win))
    timer.start()
    win._bm_timer = timer

    _band_map_tick(win)                    # first paint
    return win


def _band_map_tick(win):
    """Poll the spot bus, age out old spots, and redraw (if visible)."""
    try:
        now = time.time()
        win._bm_store.add(win._bm_tailer.poll())
        win._bm_store.add(win._bm_map144_tailer.poll())
        win._bm_store.prune(now)
        if not win.isVisible():
            return
        _band_map_redraw(win, now)
    except Exception as e:                  # never let a bad spot kill the panel
        try:
            win._bm_status.setText(f"redraw error: {e}")
        except Exception:
            pass


def _within_range(spot, home_ll, max_km):
    """True if either endpoint is within max_km of home (unresolvable -> out)."""
    for p in (spot.tx_latlon(), spot.rx_latlon()):
        if p is not None and geo.haversine_km(
                home_ll[0], home_ll[1], p[0], p[1]) <= max_km:
            return True
    return False


def _band_map_redraw(win, now):
    store = win._bm_store
    home_ll = win._bm_home_ll

    # locate MAP144 decodes whose message carried no grid, using PSKReporter's
    # richer data (sl/rl) + the QRZ cache — so your own decodes can be placed
    # Grid resolution priority for grid-less decodes: message grid (already on
    # the Spot) > LIVE grid (current window, PSKReporter sl/rl) > static
    # ADIF/QRZ home grid.  Static is HISTORICAL, so it is used only for calls
    # with no '/' — any suffix (/P /M /MM) or prefix (DL/, /4) means the station
    # is NOT at its logged grid, so trust only live/self-reported sources and
    # leave it unplaced rather than mislocate it.
    live_idx = grid_index(store.spots())
    n_mine = n_mine_located = 0
    for s in store.spots():
        if s.source == "map144":
            n_mine += 1
            if not s.tx_grid and s.tx_call:
                call = s.tx_call.upper()
                g = live_idx.get(call)
                if g:
                    s.tx_grid = g
                    s.grid_src = "live"            # current -> confident
                elif "/" not in call:
                    g = win._bm_static_grids.get(call)
                    if g:
                        s.tx_grid = g
                        s.grid_src = "static"      # home/historical -> best guess
            if s.resolvable():
                n_mine_located += 1

    # build the visible set; "My decodes" off hides the MAP144 layer entirely
    spots = store.spots()
    if not win._bm_mine.isChecked():
        spots = [s for s in spots if s.source != "map144"]
    # in-range filter (MAP144 decodes have rx=home -> always pass)
    if win._bm_inrange.isChecked():
        max_km = float(win._bm_range_km.value())
        spots = [s for s in spots if _within_range(s, home_ll, max_km)]
    drawable = [s for s in spots if s.resolvable()]

    # which spots are part of a same-second co-path group?
    hi_ids = set()
    n_groups = 0
    if win._bm_highlight.isChecked():
        wrapped = [{"tx_call": s.tx_call, "rx_call": s.rx_call, "t": s.t, "_spot": s}
                   for s in spots]
        groups = geo.cluster_copaths(wrapped, window_s=1.0)
        n_groups = len(groups)
        for g in groups:
            for w in g:
                hi_ids.add(id(w["_spot"]))

    # ── auto-frame (damped) ───────────────────────────────────────────────
    if win._bm_autoframe.isChecked():
        endpoints = [p for s in spots
                     for p in (s.tx_latlon(), s.rx_latlon()) if p is not None]
        target = geo.auto_frame(endpoints, home_ll)
        need = win._bm_frame is None or (now - win._bm_last_frame_t) > _FRAME_DAMP_S
        if not need and endpoints:
            lo_lo, hi_lo, lo_la, hi_la = win._bm_frame
            for la, lo in endpoints:       # re-fit early if activity left the frame
                if (lo < lo_lo - _REFRAME_MARGIN or lo > hi_lo + _REFRAME_MARGIN or
                        la < lo_la - _REFRAME_MARGIN or la > hi_la + _REFRAME_MARGIN):
                    need = True
                    break
        if need:
            win._bm_frame = target
            win._bm_last_frame_t = now

    frame = win._bm_frame or geo.auto_frame([], home_ll)
    lon_min, lon_max, lat_min, lat_max = frame
    lat0 = (lat_min + lat_max) / 2.0
    win._bm_lat0 = lat0
    k = math.cos(math.radians(lat0))

    # lock the view aspect to cos(lat0): 1° lon shown as cos(lat0) of 1° lat
    if k != win._bm_aspect_k:
        win._bm_plot.setAspectLocked(True, ratio=k)
        win._bm_aspect_k = k

    # ── coastline / border basemap (real lon/lat — drawn once) ────────────
    if win._bm_base_lon.size and not win._bm_base_drawn:
        win._bm_basemap.setData(win._bm_base_lon, win._bm_base_lat)
        win._bm_base_drawn = True

    # ── graticule (lat/lon lines every 10°) ───────────────────────────────
    gx, gy = [], []
    lo0 = int(math.floor(lon_min / 10.0) * 10)
    lo1 = int(math.ceil(lon_max / 10.0) * 10)
    la0 = int(math.floor(lat_min / 10.0) * 10)
    la1 = int(math.ceil(lat_max / 10.0) * 10)
    for lon in range(lo0, lo1 + 1, 10):
        gx += [lon, lon, math.nan]
        gy += [la0, la1, math.nan]
    for lat in range(la0, la1 + 1, 10):
        gx += [lo0, lo1, math.nan]
        gy += [lat, lat, math.nan]
    win._bm_graticule.setData(gx, gy)

    # ── arcs + midpoints ──────────────────────────────────────────────────
    band_x = [[], [], []]
    band_y = [[], [], []]
    hi_x, hi_y = [], []
    mid_pts, mid_hi_pts = [], []
    mine_x, mine_y, mine_pts = [], [], []
    mine_bx, mine_by, mine_bpts = [], [], []   # best-guess (static home grid)
    n_guess = 0
    for s in drawable:
        tla, tlo = s.tx_latlon()
        rla, rlo = s.rx_latlon()
        arc = geo.great_circle_points(tla, tlo, rla, rlo, n=24)
        ax = [p[1] for p in arc]               # real longitude
        ay = [p[0] for p in arc]               # real latitude
        mla, mlo = geo.great_circle_midpoint(tla, tlo, rla, rlo)
        if s.source == "map144":               # your own decode -> green, DX marked
            if s.grid_src == "static":         # possibly-stale home grid
                n_guess += 1
                mine_bx += ax + [math.nan]
                mine_by += ay + [math.nan]
                mine_bpts.append({"pos": (tlo, tla)})
                continue
            mine_x += ax + [math.nan]
            mine_y += ay + [math.nan]
            mine_pts.append({"pos": (tlo, tla)})
        elif id(s) in hi_ids:
            hi_x += ax + [math.nan]
            hi_y += ay + [math.nan]
            mid_hi_pts.append({"pos": (mlo, mla)})
        else:
            age = now - (s.t or now)
            b = 0 if age < 120 else (1 if age < 420 else 2)
            band_x[b] += ax + [math.nan]
            band_y[b] += ay + [math.nan]
            mid_pts.append({"pos": (mlo, mla)})

    for i in range(3):
        win._bm_arcs[i].setData(band_x[i], band_y[i])
    win._bm_arc_hi.setData(hi_x, hi_y)
    win._bm_mid.setData(mid_pts)
    win._bm_mid_hi.setData(mid_hi_pts)
    win._bm_mine_arcs.setData(mine_x, mine_y)
    win._bm_mine_pts.setData(mine_pts)
    win._bm_mine_arcs_bg.setData(mine_bx, mine_by)
    win._bm_mine_pts_bg.setData(mine_bpts)
    win._bm_home.setData([{"pos": (home_ll[1], home_ll[0])}])

    # ── apply frame + status ──────────────────────────────────────────────
    # aspect is locked, so this contains the frame and letterboxes the rest
    win._bm_plot.setRange(xRange=(lon_min, lon_max), yRange=(lat_min, lat_max),
                          padding=0)
    rng = (f"≤{int(win._bm_range_km.value())}km"
           if win._bm_inrange.isChecked() else "all")
    mine_txt = (f"mine {n_mine_located}/{n_mine}"
                + (f" ({n_guess} guess)" if n_guess else "")) if n_mine else "mine 0"
    win._bm_status.setText(
        f"{len(drawable)} paths · {mine_txt} · {n_groups} simultaneous · {rng} · "
        f"{win._bm_home_grid} · {len(store.spots())} spots/{int(_MAX_AGE_S/60)}m")
