"""Offline tests for tools/qrz_lookup.py — pure helpers, XML parsing, cache,
and the client path with the network mocked (real QRZ needs a subscription)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import qrz_lookup as q  # noqa: E402

NS = 'xmlns="http://xmldata.qrz.com"'
AUTH_OK = f'<QRZDatabase {NS}><Session><Key>KEY123</Key><Count>9</Count></Session></QRZDatabase>'
AUTH_BAD = f'<QRZDatabase {NS}><Session><Error>Username/password incorrect</Error></Session></QRZDatabase>'
CALL_OK = (f'<QRZDatabase {NS}><Callsign><call>W9XX</call><grid>EN63bl</grid>'
           '<lat>43.1</lat><lon>-89.3</lon><country>United States</country></Callsign>'
           '<Session><Key>KEY123</Key></Session></QRZDatabase>')
CALL_LATLON_ONLY = (f'<QRZDatabase {NS}><Callsign><call>ZZ</call>'
                    '<lat>42.5</lat><lon>-71.0</lon></Callsign>'
                    '<Session><Key>KEY123</Key></Session></QRZDatabase>')
CALL_NOTFOUND = (f'<QRZDatabase {NS}><Session><Key>KEY123</Key>'
                 '<Error>Not found: NOPE</Error></Session></QRZDatabase>')
SESSION_TIMEOUT = f'<QRZDatabase {NS}><Session><Error>Session Timeout</Error></Session></QRZDatabase>'


# --- pure: latlon_to_grid --------------------------------------------------
def test_grid_known_fn42():
    # Boston area; FN42 SW corner = (42N, -72), center ~ (42.5, -71)
    assert q.latlon_to_grid(42.5, -71.0).startswith("FN42")
    assert q.latlon_to_grid(42.5, -71.0) == "FN42mm"


def test_grid_precision4():
    assert q.latlon_to_grid(42.5, -71.0, precision=4) == "FN42"


def test_grid_roundtrip():
    # SW-corner-of-square + small epsilon must map back into that square
    for lat, lon, expect in [(43.05, -89.96, "EN53"), (0.01, 0.01, "JJ00"),
                             (-33.86, 151.21, "QF56")]:  # Madison EN53; Sydney QF56
        assert q.latlon_to_grid(lat, lon).startswith(expect)


# --- pure: XML parsing -----------------------------------------------------
def test_parse_session_ok():
    assert q.parse_session_xml(AUTH_OK).get("key") == "KEY123"


def test_parse_session_error():
    assert "incorrect" in q.parse_session_xml(AUTH_BAD).get("error", "")


def test_parse_callsign_ok():
    fields, sess = q.parse_callsign_xml(CALL_OK)
    assert fields["call"] == "W9XX"
    assert fields["grid"] == "EN63bl"
    assert sess.get("key") == "KEY123"


def test_grid_from_fields_prefers_grid():
    fields, _ = q.parse_callsign_xml(CALL_OK)
    assert q.grid_from_fields(fields) == "EN63bl"


def test_grid_from_fields_falls_back_to_latlon():
    fields, _ = q.parse_callsign_xml(CALL_LATLON_ONLY)
    # no <grid>, derive from lat/lon (42.5,-71.0) -> FN42mm
    assert q.grid_from_fields(fields) == "FN42mm"


# --- cache I/O -------------------------------------------------------------
def test_cache_roundtrip(tmp_path):
    p = str(tmp_path / "cache.json")
    q.save_cache(p, {"W9XX": {"grid": "EN63bl"}})
    assert q.load_cache(p)["W9XX"]["grid"] == "EN63bl"


def test_load_cache_missing(tmp_path):
    assert q.load_cache(str(tmp_path / "nope.json")) == {}


# --- credential resolution -------------------------------------------------
@pytest.fixture
def clear_qrz_env(monkeypatch):
    for k in ("QRZ_USERNAME", "QRZ_PASSWORD", "QRZ_CRED_FILE"):
        monkeypatch.delenv(k, raising=False)
    # point every file location at a guaranteed-absent path by default
    monkeypatch.setattr(q, "CONFIG_CRED_FILE", "/nonexistent/config.json")
    monkeypatch.setattr(q, "NETRC_FILE", "/nonexistent/.netrc")
    monkeypatch.setattr(q, "SCRATCH_CRED_FILE", "/nonexistent/scratch.json")


def _write_json_cred(tmp_path, name="c.json", user="wa1hco", pw="secret"):
    p = tmp_path / name
    p.write_text(json.dumps({"username": user, "password": pw}))
    return str(p)


def test_creds_env_wins(clear_qrz_env, monkeypatch, tmp_path):
    monkeypatch.setenv("QRZ_USERNAME", "ENVUSER")
    monkeypatch.setenv("QRZ_PASSWORD", "ENVPASS")
    monkeypatch.setattr(q, "CONFIG_CRED_FILE", _write_json_cred(tmp_path))
    assert q.load_credentials() == ("ENVUSER", "ENVPASS")


def test_creds_cred_file_env(clear_qrz_env, monkeypatch, tmp_path):
    monkeypatch.setenv("QRZ_CRED_FILE", _write_json_cred(tmp_path, user="FILEU"))
    assert q.load_credentials()[0] == "FILEU"


def test_creds_config_file(clear_qrz_env, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "CONFIG_CRED_FILE", _write_json_cred(tmp_path, user="CFGU"))
    assert q.load_credentials() == ("CFGU", "secret")


def test_creds_netrc(clear_qrz_env, monkeypatch, tmp_path):
    nrc = tmp_path / "netrc"
    nrc.write_text(f"machine {q.QRZ_HOST}\n  login NETU\n  password NETP\n")
    nrc.chmod(0o600)  # netrc refuses world-readable files
    monkeypatch.setattr(q, "NETRC_FILE", str(nrc))
    assert q.load_credentials() == ("NETU", "NETP")


def test_creds_scratch_fallback(clear_qrz_env, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "SCRATCH_CRED_FILE", _write_json_cred(tmp_path, user="SCR"))
    assert q.load_credentials()[0] == "SCR"


def test_creds_config_beats_scratch(clear_qrz_env, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "CONFIG_CRED_FILE",
                        _write_json_cred(tmp_path, "cfg.json", user="CFG"))
    monkeypatch.setattr(q, "SCRATCH_CRED_FILE",
                        _write_json_cred(tmp_path, "scr.json", user="SCR"))
    assert q.load_credentials()[0] == "CFG"


def test_creds_none_configured(clear_qrz_env):
    assert q.load_credentials() is None


# --- client with mocked network -------------------------------------------
class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _fake_get_factory(script):
    """script: list of response texts returned in order per call."""
    calls = {"n": 0, "params": []}

    def fake_get(url, params=None, timeout=None, headers=None):
        calls["params"].append(params)
        text = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return _FakeResp(text)

    return fake_get, calls


def test_lookup_success(monkeypatch, tmp_path):
    fake_get, calls = _fake_get_factory([AUTH_OK, CALL_OK])
    monkeypatch.setattr(q.requests, "get", fake_get)
    c = q.QrzClient(username="u", password="p",
                    cache_path=str(tmp_path / "c.json"), min_interval_s=0)
    rec = c.lookup("w9xx")
    assert rec["grid"] == "EN63bl"
    assert rec["source"] == "qrz"
    assert c.net_lookups == 1


def test_lookup_cached_no_second_network(monkeypatch, tmp_path):
    fake_get, calls = _fake_get_factory([AUTH_OK, CALL_OK])
    monkeypatch.setattr(q.requests, "get", fake_get)
    c = q.QrzClient(username="u", password="p",
                    cache_path=str(tmp_path / "c.json"), min_interval_s=0)
    c.lookup("W9XX")
    n_after_first = calls["n"]
    c.lookup("W9XX")  # must be served from cache
    assert calls["n"] == n_after_first
    assert c.cache_hits == 1


def test_lookup_notfound_caches_negative(monkeypatch, tmp_path):
    fake_get, _ = _fake_get_factory([AUTH_OK, CALL_NOTFOUND])
    monkeypatch.setattr(q.requests, "get", fake_get)
    c = q.QrzClient(username="u", password="p",
                    cache_path=str(tmp_path / "c.json"), min_interval_s=0)
    rec = c.lookup("NOPE")
    assert rec["grid"] is None
    assert "NOPE" in c.cache  # negative cached


def test_session_timeout_triggers_reauth(monkeypatch, tmp_path):
    # auth, then a timeout on lookup, then re-auth, then the real record
    fake_get, calls = _fake_get_factory([AUTH_OK, SESSION_TIMEOUT, AUTH_OK, CALL_OK])
    monkeypatch.setattr(q.requests, "get", fake_get)
    c = q.QrzClient(username="u", password="p",
                    cache_path=str(tmp_path / "c.json"), min_interval_s=0)
    rec = c.lookup("W9XX")
    assert rec["grid"] == "EN63bl"
    assert calls["n"] == 4  # auth, timeout, reauth, success


def test_no_credentials_raises(monkeypatch, tmp_path):
    fake_get, _ = _fake_get_factory([AUTH_OK])
    monkeypatch.setattr(q.requests, "get", fake_get)
    c = q.QrzClient(cache_path=str(tmp_path / "c.json"), min_interval_s=0)
    with pytest.raises(q.QrzError):
        c.lookup("W9XX")


def test_no_net_returns_none_when_uncached(tmp_path):
    c = q.QrzClient(cache_path=str(tmp_path / "c.json"), allow_network=False)
    assert c.lookup("W9XX") is None


def test_no_net_serves_cache(tmp_path):
    p = str(tmp_path / "c.json")
    q.save_cache(p, {"W9XX": {"call": "W9XX", "grid": "EN63bl", "source": "cache"}})
    c = q.QrzClient(cache_path=p, allow_network=False)
    assert c.lookup("W9XX")["grid"] == "EN63bl"
