#!/usr/bin/env python3
"""QRZ.com callsign -> Maidenhead grid lookup for MAP144.

Shared dependency for #26 (great-circle range), #28 (grid fallback for messages
that carry no locator), and the #48 propagation-geometry engine.  MAP144's own
MSK144 messages usually omit the grid (`RR73`/report messages carry none), and
PSKReporter spots only cover stations someone happened to hear; QRZ fills the
gap with the authoritative registered grid for any callsign.

Layering: this module is pure lookup + cache, no MAP144 imports.  It returns a
plain dict; callers (the offline join, range computation, geometry engine) decide
what to do with it.

Credentials are searched in this order (first hit wins; none live in the repo):
  1. env  QRZ_USERNAME / QRZ_PASSWORD
  2. env  QRZ_CRED_FILE  -> a JSON {"username": "...", "password": "..."}
  3. ~/.config/map144/qrz_cred.json   (same JSON; preferred home-dir location)
  4. ~/.netrc            machine xmldata.qrz.com / login / password (classic Unix)
  5. scratch/qrz_cred.json            (project-local fallback, gitignored)
The home-dir locations (3,4) keep the secret outside the project tree entirely.

QRZ XML API: https://xmldata.qrz.com/  (requires a QRZ XML-data subscription).
  auth:    GET ...?username=U;password=P;agent=A          -> <Session><Key>...
  lookup:  GET ...?s=KEY;callsign=CALL                     -> <Callsign><grid>...
  session keys expire; we re-authenticate once on "Session Timeout".

Cache (public reference data, not secret): MSK144/detections/qrz_grid_cache.json
  callsign(upper) -> {grid, lat, lon, name, country, ts, source}
  Negative results are cached too (grid=None) so we don't re-hammer QRZ for
  callsigns it doesn't know; clear the cache file to retry them.

CLI:
    python tools/qrz_lookup.py W9XX N4SIX WB5JJJ      # look up + cache
    python tools/qrz_lookup.py --no-net W9XX          # cache only, no network
    python tools/qrz_lookup.py --grid 42.5 -71.0      # lat/lon -> grid (offline)
"""
from __future__ import annotations

import argparse
import json
import netrc
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

QRZ_URL = "https://xmldata.qrz.com/xml/current/"
QRZ_HOST = "xmldata.qrz.com"  # the machine name to look up in ~/.netrc
DEFAULT_AGENT = "map144-qrz/1.0"
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(_PROJ, "MSK144", "detections", "qrz_grid_cache.json")
CONFIG_CRED_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "map144", "qrz_cred.json")
NETRC_FILE = os.path.join(os.path.expanduser("~"), ".netrc")
SCRATCH_CRED_FILE = os.path.join(_PROJ, "scratch", "qrz_cred.json")


# --------------------------------------------------------------------------- #
# Pure helpers (no network, no state) — independently testable
# --------------------------------------------------------------------------- #
def latlon_to_grid(lat, lon, precision=6):
    """Convert decimal lat/lon to a Maidenhead locator.

    precision: 4 -> field+square (e.g. FN42), 6 -> +subsquare (e.g. FN42mm).
    Used as a fallback when QRZ returns lat/lon but an empty <grid>.
    """
    lon = float(lon) + 180.0
    lat = float(lat) + 90.0
    A = ord("A")
    g = (
        chr(A + int(lon // 20))
        + chr(A + int(lat // 10))
        + str(int((lon % 20) // 2))
        + str(int((lat % 10) // 1))
    )
    if precision >= 6:
        g += chr(ord("a") + int((lon % 2) / (2.0 / 24)))
        g += chr(ord("a") + int((lat % 1) / (1.0 / 24)))
    return g


def _localname(tag):
    """Strip the {namespace} prefix ElementTree prepends (QRZ uses one)."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children_dict(elem):
    """Map child tag (local, lowercased) -> text, for a QRZ XML element."""
    out = {}
    for child in elem:
        out[_localname(child.tag).lower()] = (child.text or "").strip()
    return out


def parse_session_xml(text):
    """Parse a QRZ response, returning the <Session> fields as a dict.

    Keys of interest: 'key' (session key on success), 'error' (message on
    failure, e.g. 'Username/password incorrect', 'Session Timeout').
    """
    root = ET.fromstring(text)
    for elem in root.iter():
        if _localname(elem.tag) == "Session":
            return _children_dict(elem)
    return {}


def parse_callsign_xml(text):
    """Parse a QRZ lookup response.

    Returns (callsign_fields, session_fields).  callsign_fields is the
    <Callsign> children as a lowercased dict (includes 'grid','lat','lon',
    'call','name','country' when present); empty if no <Callsign> element.
    """
    root = ET.fromstring(text)
    call_fields, sess_fields = {}, {}
    for elem in root.iter():
        name = _localname(elem.tag)
        if name == "Callsign":
            call_fields = _children_dict(elem)
        elif name == "Session":
            sess_fields = _children_dict(elem)
    return call_fields, sess_fields


def grid_from_fields(fields):
    """Best grid from a parsed <Callsign> dict: prefer QRZ's <grid>, else
    derive from <lat>/<lon>.  Returns a grid string or None."""
    grid = fields.get("grid", "").strip()
    if grid:
        return grid
    lat, lon = fields.get("lat", "").strip(), fields.get("lon", "").strip()
    if lat and lon:
        try:
            return latlon_to_grid(float(lat), float(lon))
        except (ValueError, TypeError):
            return None
    return None


# --------------------------------------------------------------------------- #
# Credentials & cache I/O
# --------------------------------------------------------------------------- #
def _creds_from_json(path):
    """(username, password) from a JSON cred file, or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("username") and d.get("password"):
            return d["username"], d["password"]
    except (ValueError, OSError):
        pass
    return None


def _creds_from_netrc(path, host=QRZ_HOST):
    """(login, password) for `host` from a .netrc file, or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        auth = netrc.netrc(path).authenticators(host)
    except (netrc.NetrcParseError, OSError):
        return None
    if auth and auth[0] and auth[2]:
        return auth[0], auth[2]  # (login, account, password)
    return None


def load_credentials():
    """First (username, password) found across the documented search order,
    or None if no credentials are configured anywhere."""
    user = os.environ.get("QRZ_USERNAME")
    pw = os.environ.get("QRZ_PASSWORD")
    if user and pw:
        return user, pw
    return (
        _creds_from_json(os.environ.get("QRZ_CRED_FILE"))
        or _creds_from_json(CONFIG_CRED_FILE)
        or _creds_from_netrc(NETRC_FILE)
        or _creds_from_json(SCRATCH_CRED_FILE)
    )


def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, OSError):
            return {}
    return {}


def save_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    os.replace(tmp, path)  # atomic; never leave a half-written cache


# --------------------------------------------------------------------------- #
# QRZ client
# --------------------------------------------------------------------------- #
class QrzError(Exception):
    pass


class QrzClient:
    """Caching QRZ grid lookup.  One persistent session key, re-auth on timeout,
    polite minimum interval between network calls, local cache to avoid repeats."""

    def __init__(self, username=None, password=None, agent=DEFAULT_AGENT,
                 cache_path=DEFAULT_CACHE, min_interval_s=0.5, timeout_s=10.0,
                 allow_network=True):
        self.username = username
        self.password = password
        self.agent = agent
        self.cache_path = cache_path
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.allow_network = allow_network
        self.session_key = None
        self._last_call = 0.0
        self._dirty = False
        self.cache = load_cache(cache_path)
        self.net_lookups = 0
        self.cache_hits = 0

    # -- network plumbing --
    def _throttle(self):
        dt = time.time() - self._last_call
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last_call = time.time()

    def _get(self, params):
        self._throttle()
        resp = requests.get(QRZ_URL, params=params, timeout=self.timeout_s,
                            headers={"User-Agent": self.agent})
        resp.raise_for_status()
        return resp.text

    def authenticate(self):
        if not (self.username and self.password):
            raise QrzError(
                "no QRZ credentials (set QRZ_USERNAME/QRZ_PASSWORD, or write "
                f"{CONFIG_CRED_FILE}, or a ~/.netrc entry for {QRZ_HOST})")
        sess = parse_session_xml(
            self._get({"username": self.username, "password": self.password,
                       "agent": self.agent}))
        if sess.get("error"):
            raise QrzError(f"QRZ auth failed: {sess['error']}")
        key = sess.get("key")
        if not key:
            raise QrzError("QRZ auth returned no session key")
        self.session_key = key
        return key

    # -- the one method callers want --
    def lookup(self, callsign, force=False):
        """Return {'call','grid','lat','lon','name','country','source'} or None.

        Cache-first.  source is 'cache', 'qrz', or 'qrz:latlon'.  Returns the
        cached negative (grid=None still as a dict) for known-unknown calls
        unless force=True.
        """
        call = callsign.strip().upper()
        if not call:
            return None
        if not force and call in self.cache:
            self.cache_hits += 1
            return dict(self.cache[call])
        if not self.allow_network:
            return None

        rec = self._lookup_network(call)
        self.cache[call] = rec
        self._dirty = True
        return dict(rec)

    def _lookup_network(self, call):
        if not self.session_key:
            self.authenticate()
        text = self._get({"s": self.session_key, "callsign": call})
        fields, sess = parse_callsign_xml(text)

        # session expired -> re-auth once and retry
        if sess.get("error", "").lower().startswith("session"):
            self.authenticate()
            text = self._get({"s": self.session_key, "callsign": call})
            fields, sess = parse_callsign_xml(text)

        self.net_lookups += 1
        if not fields:
            # not found / other error: cache a negative so we don't retry it
            return {"call": call, "grid": None, "lat": None, "lon": None,
                    "name": None, "country": None,
                    "source": "qrz", "error": sess.get("error", "not found"),
                    "ts": round(time.time(), 0)}
        grid = grid_from_fields(fields)
        source = "qrz" if fields.get("grid", "").strip() else (
            "qrz:latlon" if grid else "qrz")
        return {
            "call": call,
            "grid": grid,
            "lat": fields.get("lat") or None,
            "lon": fields.get("lon") or None,
            "name": fields.get("name") or fields.get("fname") or None,
            "country": fields.get("country") or None,
            "source": source,
            "ts": round(time.time(), 0),
        }

    def flush(self):
        if self._dirty:
            save_cache(self.cache_path, self.cache)
            self._dirty = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="QRZ callsign -> grid lookup")
    ap.add_argument("calls", nargs="*", help="callsigns to look up")
    ap.add_argument("--no-net", action="store_true",
                    help="cache only; never hit the network")
    ap.add_argument("--force", action="store_true",
                    help="re-query even if cached")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--grid", nargs=2, metavar=("LAT", "LON"), type=float,
                    help="offline: convert lat lon -> grid and exit")
    args = ap.parse_args(argv)

    if args.grid:
        print(latlon_to_grid(args.grid[0], args.grid[1]))
        return 0

    if not args.calls:
        ap.error("give one or more callsigns (or --grid LAT LON)")

    creds = load_credentials()
    client = QrzClient(
        username=creds[0] if creds else None,
        password=creds[1] if creds else None,
        cache_path=args.cache,
        allow_network=not args.no_net,
    )
    rc = 0
    try:
        for call in args.calls:
            try:
                rec = client.lookup(call, force=args.force)
            except QrzError as e:
                print(f"{call:10} ERROR  {e}", file=sys.stderr)
                rc = 1
                break
            except requests.RequestException as e:
                print(f"{call:10} NETERR {e}", file=sys.stderr)
                rc = 1
                break
            if rec is None:
                print(f"{call:10} (not cached; --no-net)")
            elif rec.get("grid"):
                src = rec.get("source", "?")
                print(f"{call:10} {rec['grid']:8} [{src}] "
                      f"{rec.get('country') or ''}")
            else:
                print(f"{call:10} no grid ({rec.get('error','?')})")
    finally:
        client.flush()
    if client.net_lookups or client.cache_hits:
        print(f"# {client.net_lookups} network, {client.cache_hits} cached",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
