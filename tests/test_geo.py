"""Unit tests for map144_app/geo.py — pure geographic helpers."""
import math

import numpy as np
import pytest

from map144_app import geo


# --- Maidenhead ------------------------------------------------------------
def test_grid_to_latlon_fn42_center():
    lat, lon = geo.grid_to_latlon("FN42")
    assert lat == pytest.approx(42.5, abs=1e-6)
    assert lon == pytest.approx(-71.0, abs=1e-6)


def test_grid_to_latlon_6char_inside_4char_cell():
    lat, lon = geo.grid_to_latlon("FN42mm")
    assert 42.0 <= lat <= 43.0
    assert -72.0 <= lon <= -70.0


def test_grid_roundtrip():
    for g in ("FN42", "EN53", "DM43", "JO22"):
        lat, lon = geo.grid_to_latlon(g)
        assert geo.latlon_to_grid(lat, lon).startswith(g)


def test_grid_to_latlon_bad():
    with pytest.raises(ValueError):
        geo.grid_to_latlon("FN")
    with pytest.raises(ValueError):
        geo.grid_to_latlon("FNXY")


# --- great circle ----------------------------------------------------------
def test_gc_points_endpoints_and_count():
    pts = geo.great_circle_points(0, 0, 0, 90, n=3)
    assert len(pts) == 3
    assert pts[0] == pytest.approx((0, 0), abs=1e-6)
    assert pts[-1] == pytest.approx((0, 90), abs=1e-6)
    assert pts[1][0] == pytest.approx(0, abs=1e-6)   # stays on equator
    assert pts[1][1] == pytest.approx(45, abs=1e-6)  # midpoint longitude


def test_gc_points_coincident():
    pts = geo.great_circle_points(42.5, -71.0, 42.5, -71.0, n=8)
    assert all(p == pytest.approx((42.5, -71.0), abs=1e-6) for p in pts)


def test_gc_midpoint_equator():
    assert geo.great_circle_midpoint(0, -10, 0, 10) == pytest.approx((0, 0), abs=1e-6)


def test_gc_midpoint_meridian():
    lat, lon = geo.great_circle_midpoint(0, 0, 0, 90)
    assert (lat, lon) == pytest.approx((0, 45), abs=1e-6)


def test_haversine_one_degree_lat():
    # 1 deg of latitude ~ 111.2 km
    assert geo.haversine_km(42, 0, 43, 0) == pytest.approx(111.2, abs=0.5)


def test_haversine_zero():
    assert geo.haversine_km(42.5, -71, 42.5, -71) == pytest.approx(0.0, abs=1e-6)


# --- projection ------------------------------------------------------------
def test_project_equirect_aspect():
    x, y = geo.project_equirect(42.0, -71.0, lat0=42.0)
    assert float(y) == pytest.approx(42.0)
    assert float(x) == pytest.approx(-71.0 * math.cos(math.radians(42.0)), abs=1e-6)


# --- auto-frame ------------------------------------------------------------
HOME = (42.5, -71.0)  # FN42


def test_auto_frame_includes_home_empty():
    lon_min, lon_max, lat_min, lat_max = geo.auto_frame([], HOME)
    assert lon_min <= HOME[1] <= lon_max
    assert lat_min <= HOME[0] <= lat_max


def test_auto_frame_encloses_points():
    pts = [(40.0, -90.0), (35.0, -85.0), (44.0, -80.0)]
    lon_min, lon_max, lat_min, lat_max = geo.auto_frame(pts, HOME, pad_frac=0.0)
    for la, lo in pts + [HOME]:
        assert lon_min <= lo <= lon_max
        assert lat_min <= la <= lat_max


def test_auto_frame_trims_outlier():
    # cluster of nearby points + one far DX outlier; with enough points the
    # 2nd/98th-pct trim should drop the outlier from the frame
    cluster = [(40 + i * 0.1, -85 + i * 0.1) for i in range(20)]
    outlier = (-33.0, 151.0)  # Sydney
    framed = geo.auto_frame(cluster + [outlier], HOME, pad_frac=0.0)
    lon_min, lon_max, lat_min, lat_max = framed
    assert lat_min > -30.0  # Sydney trimmed out
    assert lon_max < 100.0


def test_auto_frame_min_span():
    framed = geo.auto_frame([(42.5, -71.0)], HOME, pad_frac=0.0, min_span_deg=6.0)
    lon_min, lon_max, lat_min, lat_max = framed
    assert (lon_max - lon_min) >= 6.0 - 1e-9
    assert (lat_max - lat_min) >= 6.0 - 1e-9


# --- basemap geojson parsing ----------------------------------------------
def test_geometry_rings_polygon():
    g = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    rings = list(geo.geometry_rings(g))
    assert len(rings) == 1 and len(rings[0]) == 4


def test_geometry_rings_multipolygon():
    g = {"type": "MultiPolygon",
         "coordinates": [[[[0, 0], [1, 1]]], [[[2, 2], [3, 3]]]]}
    assert len(list(geo.geometry_rings(g))) == 2


def test_geojson_geometries_region_dict():
    # GridTracker shapes.json layout: keyed by region -> {geometry: ...}
    obj = {"MA": {"geometry": {"type": "Polygon",
                               "coordinates": [[[-73, 42], [-70, 42], [-71, 43]]]}}}
    geoms = list(geo.geojson_geometries(obj))
    assert len(geoms) == 1 and geoms[0]["type"] == "Polygon"


def test_geojson_geometries_featurecollection():
    obj = {"type": "FeatureCollection",
           "features": [{"geometry": {"type": "LineString",
                                      "coordinates": [[0, 0], [1, 1]]}}]}
    assert len(list(geo.geojson_geometries(obj))) == 1


def test_polylines_from_geojson_nan_separated():
    obj = {"A": {"geometry": {"type": "Polygon",
                              "coordinates": [[[0, 0], [1, 0], [1, 1]]]}},
           "B": {"geometry": {"type": "Polygon",
                              "coordinates": [[[5, 5], [6, 6]]]}}}
    lons, lats = geo.polylines_from_geojson(obj)
    # two rings -> two NaN separators
    assert int(np.isnan(lons).sum()) == 2
    assert lons[0] == 0 and lats[0] == 0


def test_load_basemap_missing_file_is_empty():
    lons, lats = geo.load_basemap("/nonexistent/shapes.json")
    assert lons.size == 0 and lats.size == 0


# --- simultaneity clustering ----------------------------------------------
def _spot(tx, rx, t):
    return {"tx_call": tx, "rx_call": rx, "t": t}


def test_cluster_copaths_same_ping():
    spots = [_spot("WB5JJJ", "N4SIX", 100), _spot("WB5JJJ", "W3IP", 100)]
    groups = geo.cluster_copaths(spots, window_s=1.0)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_cluster_copaths_outside_window():
    spots = [_spot("WB5JJJ", "N4SIX", 100), _spot("WB5JJJ", "W3IP", 105)]
    assert geo.cluster_copaths(spots, window_s=1.0) == []


def test_cluster_copaths_different_tx_not_grouped():
    spots = [_spot("WB5JJJ", "N4SIX", 100), _spot("W9XX", "W3IP", 100)]
    assert geo.cluster_copaths(spots, window_s=1.0) == []


def test_cluster_copaths_same_rx_not_a_copath():
    # same tx, same rx twice in window = duplicate, not two distinct paths
    spots = [_spot("WB5JJJ", "N4SIX", 100), _spot("WB5JJJ", "N4SIX", 100)]
    assert geo.cluster_copaths(spots, window_s=1.0) == []
