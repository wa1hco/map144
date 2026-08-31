"""Tests for map144_app.router.lo_planner."""
import pytest

from map144_app.router.band_plan import DialChannel
from map144_app.router.lo_planner import LoPlanError, plan_lo


def _ch(band, mode, dial):
    return DialChannel(band=band, mode=mode, dial_mhz=dial, label=mode)


def test_6m_msk_plus_ft8_fits_192k():
    dials = [_ch("6m", "MSK144", 50.260), _ch("6m", "FT8", 50.313)]
    plan = plan_lo(dials, max_hw_rate_hz=192_000)
    assert plan.hw_rate_hz == 192_000
    assert plan.dial_count == 2
    # Midpoint near mid of the two USB centres
    assert 50.26 < plan.pan_center_mhz < 50.32


def test_2m_with_map65_window():
    dials = [_ch("2m", "MSK144", 144.150), _ch("2m", "FT8", 144.174)]
    plan = plan_lo(dials, wideband_center_mhz=144.125, max_hw_rate_hz=192_000)
    assert plan.hw_rate_hz == 192_000
    # MAP65 ±48 kHz dominates; pan near 144.125
    assert abs(plan.pan_center_mhz - 144.125) < 0.05


def test_oversized_span_raises():
    dials = [_ch("2m", "MSK144", 144.100), _ch("2m", "WSPR", 144.489)]
    with pytest.raises(LoPlanError, match="spans"):
        plan_lo(dials, max_hw_rate_hz=192_000)


def test_empty_selection_raises():
    with pytest.raises(LoPlanError):
        plan_lo([])


def test_wideband_only():
    plan = plan_lo([], wideband_center_mhz=144.125, max_hw_rate_hz=192_000)
    assert plan.dial_count == 0
    assert abs(plan.pan_center_mhz - 144.125) < 1e-9


def test_pan_override_rejects_bad_centre():
    dials = [_ch("6m", "MSK144", 50.260), _ch("6m", "FT8", 50.313)]
    with pytest.raises(LoPlanError, match="pan_center"):
        plan_lo(dials, pan_center_mhz=50.000, max_hw_rate_hz=192_000)
