"""Unit tests for map144_app/spot_bus.py — normalization, tailing, aging store."""
import json

import pytest

from map144_app import spot_bus


def _raw(sc="W9XX", sl="EN63bl", rc="N4SIX", rl="FM04lv", t=100, f=50260000):
    return {"sq": 1, "f": f, "md": "MSK144", "rp": 2, "t": t, "t_tx": t,
            "sc": sc, "sl": sl, "rc": rc, "rl": rl, "b": "6m"}


# --- normalize -------------------------------------------------------------
def test_normalize_fields():
    s = spot_bus.normalize_pskr(_raw())
    assert s.source == "pskr"
    assert s.tx_call == "W9XX" and s.tx_grid == "EN63bl"
    assert s.rx_call == "N4SIX" and s.rx_grid == "FM04lv"
    assert s.mode == "MSK144" and s.snr_db == 2 and s.t == 100


def test_normalize_blank_grid_to_none():
    s = spot_bus.normalize_pskr(_raw(sl=""))
    assert s.tx_grid is None


def test_resolvable_and_latlon():
    s = spot_bus.normalize_pskr(_raw())
    assert s.resolvable()
    assert s.tx_latlon() is not None
    s2 = spot_bus.normalize_pskr(_raw(sl=""))
    assert not s2.resolvable()


# --- MAP144 message + decode normalization --------------------------------
def test_parse_message_from_is_second():
    assert spot_bus.parse_msk144_message("ND0B N0AN RR73") == ("N0AN", None)


def test_parse_message_grid_when_present():
    assert spot_bus.parse_msk144_message("CQ N0AN EN42") == ("N0AN", "EN42")
    assert spot_bus.parse_msk144_message("WA1HCO K8XX FN20") == ("K8XX", "FN20")


def test_parse_message_report_is_not_grid():
    # -01 / R-05 / RRR / 73 are reports, not grids
    assert spot_bus.parse_msk144_message("WA1HCO K8XX -01") == ("K8XX", None)


def test_parse_message_synthetic_dropped():
    assert spot_bus.parse_msk144_message("AP00113N03") == (None, None)


def test_parse_message_unusable():
    assert spot_bus.parse_msk144_message("") == (None, None)
    assert spot_bus.parse_msk144_message("CQ") == (None, None)


def test_parse_ts_with_and_without_fraction():
    t1 = spot_bus._parse_map144_ts("2026-06-02_14:28:43.5")
    t2 = spot_bus._parse_map144_ts("2026-06-02_14:28:43")
    assert t1 == pytest.approx(t2 + 0.5)
    assert spot_bus._parse_map144_ts("garbage") is None


def test_normalize_map144_decode():
    raw = {"message": "WA1HCO K8XX FN20", "timestamp": "2026-06-02_14:28:43.5",
           "radio_khz": 50260, "jt9_snr_db": 3, "theta_deg": 45}
    s = spot_bus.normalize_map144(raw, "WA1HCO", "FN42")
    assert s.source == "map144"
    assert s.tx_call == "K8XX" and s.tx_grid == "FN20"
    assert s.rx_call == "WA1HCO" and s.rx_grid == "FN42"
    assert s.freq_hz == 50260000 and s.snr_db == 3


def test_normalize_map144_error_row_dropped():
    assert spot_bus.normalize_map144(
        {"message": "", "outcome": "error"}, "WA1HCO", "FN42") is None


def test_jsonl_tailer_drops_none(tmp_path):
    p = str(tmp_path / "decodes.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"message": "", "outcome": "error"}) + "\n")
        f.write(json.dumps({"message": "ND0B N0AN RR73",
                            "timestamp": "2026-06-02_14:28:43"}) + "\n")
    t = spot_bus.JsonlTailer(p, lambda r: spot_bus.normalize_map144(r, "WA1HCO", "FN42"))
    got = t.poll()
    assert len(got) == 1 and got[0].tx_call == "N0AN"


# --- grid index ------------------------------------------------------------
def test_grid_index_harvest_and_extra():
    spots = [spot_bus.normalize_pskr(_raw(sc="W9XX", sl="EN63bl",
                                          rc="N4SIX", rl="FM04lv"))]
    idx = spot_bus.grid_index(spots, extra={"k8xx": "FN20"})
    assert idx["W9XX"] == "EN63bl"
    assert idx["N4SIX"] == "FM04lv"
    assert idx["K8XX"] == "FN20"


def test_load_qrz_cache_grids(tmp_path):
    p = tmp_path / "qrz.json"
    p.write_text(json.dumps({"W9XX": {"grid": "EN63bl"},
                             "NOPE": {"grid": None}}))
    grids = spot_bus.load_qrz_cache_grids(str(p))
    assert grids == {"W9XX": "EN63bl"}


# --- tailer ----------------------------------------------------------------
def _append(path, *raws):
    with open(path, "a") as f:
        for r in raws:
            f.write(json.dumps(r) + "\n")


def test_tailer_incremental(tmp_path):
    p = str(tmp_path / "spots.jsonl")
    _append(p, _raw(t=1), _raw(t=2))
    t = spot_bus.PskrTailer(p)
    first = t.poll()
    assert len(first) == 2
    assert t.poll() == []                 # nothing new
    _append(p, _raw(t=3))
    second = t.poll()
    assert len(second) == 1 and second[0].t == 3


def test_tailer_ignores_partial_last_line(tmp_path):
    p = str(tmp_path / "spots.jsonl")
    _append(p, _raw(t=1))
    with open(p, "a") as f:
        f.write('{"partial": ')           # half a line, no newline
    t = spot_bus.PskrTailer(p)
    got = t.poll()
    assert len(got) == 1                   # only the complete line
    # complete the partial line; next poll picks it up
    with open(p, "a") as f:
        f.write('"f", "sc":"X","sl":"FN42","rc":"Y","rl":"FN42","t":2,"md":"MSK144"}\n')
    got2 = t.poll()
    assert len(got2) == 1 and got2[0].tx_call == "X"


def test_tailer_missing_file(tmp_path):
    t = spot_bus.PskrTailer(str(tmp_path / "nope.jsonl"))
    assert t.poll() == []


def test_tailer_truncation_resets(tmp_path):
    p = str(tmp_path / "spots.jsonl")
    _append(p, _raw(t=1), _raw(t=2))
    t = spot_bus.PskrTailer(p)
    t.poll()
    open(p, "w").close()                   # truncate
    _append(p, _raw(t=9))
    got = t.poll()
    assert len(got) == 1 and got[0].t == 9


# --- store -----------------------------------------------------------------
def test_store_aging():
    st = spot_bus.SpotStore(max_age_s=100)
    st.add([spot_bus.normalize_pskr(_raw(t=1000))])
    st.add([spot_bus.normalize_pskr(_raw(t=1090))])
    st.prune(now=1101)                     # cutoff 1001: t=1000 is older -> dropped
    assert len(st.spots()) == 1
    assert st.spots()[0].t == 1090


def test_store_endpoints_and_drawable():
    st = spot_bus.SpotStore()
    st.add([spot_bus.normalize_pskr(_raw())])
    st.add([spot_bus.normalize_pskr(_raw(sl=""))])   # unresolvable tx grid
    assert len(st.drawable()) == 1
    assert len(st.endpoints_latlon()) == 3           # 2 from full + 1 rx from partial


def test_store_copath_groups():
    st = spot_bus.SpotStore()
    st.add([
        spot_bus.normalize_pskr(_raw(sc="WB5JJJ", rc="N4SIX", t=100)),
        spot_bus.normalize_pskr(_raw(sc="WB5JJJ", rc="W3IP", t=100)),
        spot_bus.normalize_pskr(_raw(sc="OTHER", rc="ZZ", t=100)),
    ])
    groups = st.copath_groups(window_s=1.0)
    assert len(groups) == 1
    assert {s.rx_call for s in groups[0]} == {"N4SIX", "W3IP"}
    assert all(isinstance(s, spot_bus.Spot) for s in groups[0])
