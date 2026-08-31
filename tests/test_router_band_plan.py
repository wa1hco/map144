"""Tests for map144_app.router.band_plan."""
from map144_app.router.band_plan import (
    DialChannel,
    all_channels,
    bands,
    channels_for_band,
)


def test_bands_include_vhf_exclude_hf_notionally():
    b = bands()
    assert "2m" in b
    assert "6m" in b
    assert "70cm" in b
    # No HF band names in the shipped B210-tunable table
    assert "20m" not in b
    assert "40m" not in b


def test_2m_has_ft8_ft4_msk144():
    rows = channels_for_band("2m")
    modes = {r.mode for r in rows}
    assert {"FT8", "FT4", "MSK144"} <= modes
    ft8 = [r for r in rows if r.mode == "FT8"]
    assert any(abs(r.dial_mhz - 144.174) < 1e-6 for r in ft8)


def test_6m_msk_and_ft8_present():
    rows = channels_for_band("6m")
    dials = {(r.mode, round(r.dial_mhz, 3)) for r in rows}
    assert ("MSK144", 50.260) in dials
    assert ("FT8", 50.313) in dials


def test_all_channels_above_50_mhz():
    for c in all_channels():
        assert c.dial_mhz >= 50.0


def test_sink_name_and_description():
    ch = DialChannel(band="2m", mode="MSK144", dial_mhz=144.150, label="MSK144")
    assert ch.sink_name() == "map144.2m.msk144.144150.rx"
    assert "144.150" in ch.description()
    assert "WSJT-X" in ch.description()
    assert ch.sink_name(rf=0).startswith("map144.RF0.")
    assert "RF0" in ch.description(rf=0)


def test_usb_center_is_dial_plus_1500():
    ch = DialChannel(band="2m", mode="FT8", dial_mhz=144.174, label="FT8")
    assert abs(ch.usb_center_hz - (144.174e6 + 1500.0)) < 1e-6
