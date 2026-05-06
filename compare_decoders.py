#!/usr/bin/env python3
"""Compare MAP144 vs WSJT-X MSK144 decode logs — interactive GUI.

GUI (default):
    python compare_decoders.py [--date YYYYMMDD]

Save PNGs and exit without GUI:
    python compare_decoders.py --no-gui --plot detect.png --date 20260421
"""
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import json
import re
import signal
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

# ── Paths ─────────────────────────────────────────────────────────────────────
WSJTX_ALL       = Path.home() / ".local/share/WSJT-X - flex/ALL.TXT"
MAP144_LOG      = Path(__file__).parent / "MSK144/detections/decodes.jsonl"
MAP144_LAUNCHES = Path(__file__).parent / "MSK144/detections/launches.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def period_start(ts: datetime) -> datetime:
    """Floor a UTC datetime to the nearest 15-second MSK144 period boundary."""
    epoch = int(ts.timestamp() // 15) * 15
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_wsjtx(path: Path):
    """Parse WSJT-X ALL.TXT MSK144 lines into list of dicts."""
    pat = re.compile(
        r'^(\d{6}_\d{6})\s+'
        r'([\d.]+)\s+Rx\s+MSK144\s+'
        r'([+-]?\d+)\s+'
        r'([\d.]+)\s+'
        r'(\d+)\s+'
        r'(.+)$'
    )
    results = []
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue
            ts_str, freq, snr, dt_s, af, msg = m.groups()
            try:
                ts_period = datetime.strptime("20" + ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts = period_start(ts_period + timedelta(seconds=float(dt_s)))
            results.append({
                'ts':        ts,
                'ts_period': ts_period,
                'freq_mhz':  float(freq),
                'snr':       int(snr),
                'dt':        float(dt_s),
                'af_hz':     int(af),
                'message':   msg.strip(),
                'source':    'wsjtx',
            })
    return results


def parse_map144(path: Path):
    """Parse MAP144 decodes.jsonl into list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_ts_str = d.get('period_ts') or d.get('timestamp', '')
            fmt = "%Y-%m-%d_%H:%M:%S"
            try:
                ts = datetime.strptime(raw_ts_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    ts = datetime.strptime(raw_ts_str, "%Y-%m-%d_%H:%M:%S.%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            ts = period_start(ts)
            est = d.get('est_snr_db')
            jt9 = d.get('jt9_snr_db')
            snr = jt9 if jt9 is not None else est
            results.append({
                'ts':       ts,
                'freq_mhz': d.get('radio_khz', 0) / 1000.0,
                'snr':      snr,
                'jt9_snr':  jt9,
                'af_hz':    0,
                'message':  d.get('message', '').strip(),
                'source':   'map144',
                # Raw fields preserved for capture-WAV lookup — the
                # ``ts`` above is period-floored (15-s boundary), but the
                # capture filename uses the actual second-precision wall
                # clock, so we need the raw string here.
                'raw_timestamp': d.get('timestamp', ''),
                'raw_radio_khz': int(d.get('radio_khz', 0)),
            })
    return results


_TEST_MSG_RE = re.compile(r'^A[NP]\d{5}[PN]\d{2}(\d{3})?$')

def is_test_message(msg: str) -> bool:
    return bool(_TEST_MSG_RE.match(msg.strip()))

def normalise_msg(msg: str) -> str:
    return ' '.join(msg.upper().split())


def parse_launches(path: Path) -> list[dict]:
    launches = []
    if not path.exists():
        return launches
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_str = d.get('timestamp', '')
            try:
                wall = datetime.strptime(ts_str, "%Y-%m-%d_%H:%M:%S.%f").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    wall = datetime.strptime(ts_str, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            t_sec = d.get('t_sec', 0.0) or 0.0
            approx_period_epoch = (wall.timestamp() - t_sec) // 15 * 15
            d['period'] = datetime.fromtimestamp(approx_period_epoch, tz=timezone.utc)
            launches.append(d)
    return launches


# ── Analysis ──────────────────────────────────────────────────────────────────

def _find_launch_for_wsjtx(wsjtx_decode: dict, launches: list[dict],
                           freq_tol_khz: float = 2.5) -> dict | None:
    sig_khz = wsjtx_decode['freq_mhz'] * 1000.0 + wsjtx_decode.get('af_hz', 0) / 1000.0
    period  = wsjtx_decode['ts']
    outcome_rank = {'decoded': 0, 'no_decode': 1, 'timeout': 2, 'error': 3}
    best, best_rank, best_df = None, 99, float('inf')
    for launch in launches:
        if launch['period'] != period:
            continue
        df = abs(launch.get('radio_khz', 0) - sig_khz)
        if df > freq_tol_khz:
            continue
        rank = outcome_rank.get(launch.get('outcome', ''), 9)
        if (rank < best_rank) or (rank == best_rank and df < best_df):
            best, best_rank, best_df = launch, rank, df
    return best


def _dedup(decodes):
    def _snr_key(d):
        s = d.get('snr')
        return s if s is not None else 0
    best: dict[tuple, dict] = {}
    for d in decodes:
        key = (normalise_msg(d['message']), d['ts'])
        prev = best.get(key)
        if prev is None or _snr_key(d) < _snr_key(prev):
            best[key] = d
    return list(best.values())


def match_decodes(wsjtx, map144, window_sec=2):
    wsjtx  = _dedup(wsjtx)
    map144 = _dedup(map144)
    m144_by_msg = defaultdict(list)
    for d in map144:
        m144_by_msg[normalise_msg(d['message'])].append(d)
    matched, wsjtx_only, used_ids = [], [], set()
    for w in wsjtx:
        key = normalise_msg(w['message'])
        candidates = m144_by_msg.get(key, [])
        best, best_dt = None, None
        for m in candidates:
            if id(m) in used_ids:
                continue
            dt = abs((w['ts'] - m['ts']).total_seconds())
            if dt <= window_sec:
                if best_dt is None or dt < best_dt:
                    best, best_dt = m, dt
        if best is not None:
            used_ids.add(id(best))
            matched.append((w, best))
        else:
            wsjtx_only.append(w)
    map144_only = [d for d in map144 if id(d) not in used_ids]
    return matched, wsjtx_only, map144_only


def _build_wsjtx_tags(wsjtx_only, launches):
    tags = {}
    counts = defaultdict(int)
    for d in wsjtx_only:
        launch = _find_launch_for_wsjtx(d, launches) if launches else None
        if launch is None:
            tag = 'NOT_DETECTED'
        elif launch.get('outcome') in ('no_decode', 'error'):
            tag = 'NO_DECODE'
        elif launch.get('outcome') == 'timeout':
            tag = 'TIMEOUT'
        elif launch.get('outcome') == 'decoded':
            tag = 'PERIOD_SLIP'
        else:
            tag = launch.get('outcome', '?').upper()
        tags[id(d)] = tag
        counts[tag] += 1
    return tags, counts


_SKIP_WORDS = {'CQ', 'DE', 'QRZ', 'RR73', '73', 'RRR'}
_CALL_RE    = re.compile(r'^[A-Z0-9]{2,}(/[A-Z0-9]+)?$')
_GRID_RE    = re.compile(r'^[A-R]{2}[0-9]{2}([A-X]{2})?$', re.IGNORECASE)
#: Tokens that introduce a CQ-form message.  The transmitting station is
#: then the first extracted callsign; any 2-letter region/state filter
#: between CQ and the call is filtered by ``_calls_in_message`` (no digit).
_CQ_PREFIXES = {'CQ', 'DE', 'QRZ'}


def _calls_in_message(msg: str) -> list[str]:
    """Return tokens that look like real amateur callsigns.

    Real callsigns contain at least one digit (e.g. ``K1ABC``,
    ``W2DEF``, ``BT1NFE/R``).  Without the digit requirement, CQ
    qualifiers like ``NA`` / ``DX`` / ``EU`` (and US-state abbreviations
    in CQ-by-region messages) sneak through as "callsigns" and get
    counted in per-callsign statistics.
    """
    out = []
    for w in msg.split():
        if w in _SKIP_WORDS:
            continue
        if _GRID_RE.match(w):
            continue
        if not _CALL_RE.match(w):
            continue
        # Require ≥1 digit in the base call (before any /SUFFIX).
        base = w.split('/', 1)[0]
        if not any(c.isdigit() for c in base):
            continue
        out.append(w)
    return out


def sender_in_message(msg: str) -> str | None:
    """Return the *transmitting* station of an MSK144 message — None if
    no clear sender can be identified.

    MSK144 message-form conventions:

    +----------------------------------+--------------------+
    | Message form                     | Sender             |
    +==================================+====================+
    | ``CQ <SENDER> [grid]``           | first extracted    |
    | ``CQ DX <SENDER> [grid]``        | first extracted    |
    | ``CQ <region> <SENDER> [grid]``  | first extracted    |
    | ``<addressed> <SENDER> <grid>``  | second extracted   |
    | ``<addressed> <SENDER> <report>``| second extracted   |
    | ``<addressed> <SENDER> RR73``    | second extracted   |
    +----------------------------------+--------------------+

    The first callsign in a non-CQ message is the *addressed* station
    (the operator the SENDER is calling) and contributes nothing to
    SENDER statistics: their grid, SNR, or transmission count cannot be
    inferred from this message.
    """
    tokens = msg.split()
    if not tokens:
        return None
    calls = _calls_in_message(msg)
    if not calls:
        return None
    if tokens[0] in _CQ_PREFIXES:
        # CQ form: the first extracted callsign is the sender.  Any
        # 2-letter regional filter ('DX', 'NA', 'MA' …) was already
        # excluded by the digit requirement above.
        return calls[0]
    # Standard QSO form: first call is addressed, second is sender.
    if len(calls) >= 2:
        return calls[1]
    # Degenerate (single callsign, non-CQ) — best guess: that's the sender.
    return calls[0]


def _local_callsigns(all_decodes: list[dict], min_periods: int = 5) -> dict[str, int]:
    """Return callsigns appearing in >= min_periods distinct 15-second periods.

    A station seen in many periods is transmitting continuously (local terrestrial
    path) rather than producing random meteor pings.  The returned dict maps
    callsign -> period count.
    """
    call_periods: dict[str, set] = defaultdict(set)
    for d in all_decodes:
        for call in _calls_in_message(d['message']):
            call_periods[call].add(d['ts'])
    return {c: len(ps) for c, ps in call_periods.items() if len(ps) >= min_periods}


def _is_local(d: dict, local_calls: dict) -> bool:
    return any(w in local_calls for w in _calls_in_message(d['message']))


def unique_calls(decodes, word_index):
    calls = set()
    for d in decodes:
        words = d['message'].split()
        if len(words) > word_index:
            c = words[word_index]
            if re.match(r'^[A-Z0-9]{2,}(/[A-Z0-9]+)?$', c) and c not in (
                    'CQ', 'DE', 'QRZ', 'RR73', '73', 'RRR'):
                calls.add(c)
    return calls


def _format_summary(matched, wsjtx_only, map144_only, wsjtx_all, map144_all,
                    wsjtx_tags=None, tag_counts=None, local_calls=None) -> str:
    lines = []

    def _ts_range(lst):
        if not lst:
            return '—', '—'
        return (min(d['ts'] for d in lst).strftime('%H:%M'),
                max(d['ts'] for d in lst).strftime('%H:%M'))

    w_lo, w_hi = _ts_range(wsjtx_all)
    m_lo, m_hi = _ts_range(map144_all)
    lines.append(f"WSJT-X:  {len(wsjtx_all):4d} MSK144 decodes  ({w_lo}–{w_hi} UTC)")
    lines.append(f"MAP144:  {len(map144_all):4d} MSK144 decodes  ({m_lo}–{m_hi} UTC)")
    lines.append(f"\nMatch window ±1 s, same message:")
    lines.append(f"  Both decoded:  {len(matched):4d}")
    lines.append(f"  WSJT-X only:   {len(wsjtx_only):4d}")
    lines.append(f"  MAP144 only:   {len(map144_only):4d}")

    total = len(matched) + len(wsjtx_only) + len(map144_only)
    if total:
        lines.append(f"\n  WSJT-X coverage: {(len(matched)+len(wsjtx_only))/total*100:.1f}%")
        lines.append(f"  MAP144 coverage: {(len(matched)+len(map144_only))/total*100:.1f}%")

    if tag_counts:
        parts = '  '.join(f"{k}={v}" for k, v in sorted(tag_counts.items()))
        lines.append(f"\nWSJT-X missed breakdown:  {parts}")

    if matched:
        snr_pairs = [(w['snr'], m['snr']) for w, m in matched if m.get('snr') is not None]
        if snr_pairs:
            diffs = [wx - mx for wx, mx in snr_pairs]
            n_est = sum(1 for _, m in matched
                        if m.get('jt9_snr') != m.get('snr') and m.get('snr') is not None)
            lines.append(f"\nSNR bias (WSJT-X − MAP144)  n={len(snr_pairs)}"
                         f"  mean={sum(diffs)/len(diffs):+.1f} dB"
                         f"  min={min(diffs):+d}  max={max(diffs):+d}")
            if n_est:
                lines[-1] += f"  ({n_est} using est_snr)"

    def all_calls(lst):
        s = set()
        for d in lst:
            s |= unique_calls([d], 0)
            s |= unique_calls([d], 1)
        return s

    w_calls = all_calls([w for w, _ in matched] + wsjtx_only)
    m_calls = all_calls([m for _, m in matched] + map144_only)
    both_c   = w_calls & m_calls
    w_only_c = w_calls - m_calls
    m_only_c = m_calls - w_calls
    lines.append(f"\nCallsigns — both: {len(both_c)}"
                 f"  WSJT-X only: {len(w_only_c)}"
                 f"  MAP144 only: {len(m_only_c)}")
    if w_only_c:
        lines.append(f"  WSJT-X only: {sorted(w_only_c)}")
    if m_only_c:
        lines.append(f"  MAP144 only: {sorted(m_only_c)}")

    if wsjtx_tags:
        lines.append(f"\nWSJT-X decoded / MAP144 missed ({len(wsjtx_only)}):")
        for d in sorted(wsjtx_only, key=lambda x: x['snr'])[:40]:
            sig_khz = d['freq_mhz'] * 1000.0 + d.get('af_hz', 0) / 1000.0
            tag = wsjtx_tags.get(id(d), '')
            lines.append(f"  {d['ts'].strftime('%H:%M:%S')}  SNR{d['snr']:+3d}"
                         f"  {sig_khz:8.1f} kHz  {d['message']}"
                         + (f"  [{tag}]" if tag else ''))
        if len(wsjtx_only) > 40:
            lines.append(f"  … and {len(wsjtx_only)-40} more")

    if local_calls:
        lines.append(f"\nLocal / terrestrial stations (≥ threshold periods, highlighted orange):")
        all_d = ([w for w, _ in matched] + [m for _, m in matched]
                 + wsjtx_only + map144_only)
        for call, n_periods in sorted(local_calls.items(), key=lambda x: -x[1]):
            w_n  = sum(1 for d in wsjtx_all  if call in _calls_in_message(d['message']))
            m_n  = sum(1 for d in map144_all if call in _calls_in_message(d['message']))
            lines.append(f"  {call:<12}  {n_periods:3d} periods   "
                         f"WSJT-X: {w_n}  MAP144: {m_n}")

    return '\n'.join(lines)


# ── Classification (density + distance) ──────────────────────────────────────
#
# Four-class scheme combining period-density (continuous-TX vs sparse) and
# great-circle range from the operator's grid:
#
#   TRUE_PING     ≤ max_ping_periods periods                      (mission target)
#   WEAK_FADING   between max_ping_periods and min_local_periods   (intermittent)
#   STRONG_LOCAL  ≥ min_local_periods AND range < local_max_km     (LOS / ground-wave)
#   CONT_PROP     ≥ min_local_periods AND range ≥ local_max_km     (continuous propagation
#                                                                    — Es / F2 / beacon)
#
# When a callsign has no known grid we default high-density to STRONG_LOCAL
# (conservative: assume local until proven otherwise).
#
# CLASS_RANK doubles as the "most-noise wins" tie-breaker for messages
# carrying multiple callsigns and as the sort order in the callsign table
# (mission-first when reversed).

CLASS_RANK = {
    "STRONG_LOCAL": 0,
    "CONT_PROP":    1,
    "WEAK_FADING":  2,
    "TRUE_PING":    3,
    "UNKNOWN":      4,
}

_GRID_TOKEN_RE = re.compile(r'^[A-R]{2}[0-9]{2}([A-X]{2})?$', re.IGNORECASE)


def grid_to_lat_lon(grid: str) -> tuple[float, float]:
    """Maidenhead grid (4 or 6 char) → (lat, lon) of cell center."""
    g = grid.upper()
    if len(g) not in (4, 6):
        raise ValueError(f"unsupported grid length: {grid!r}")
    lon = (ord(g[0]) - ord('A')) * 20.0 - 180.0
    lat = (ord(g[1]) - ord('A')) * 10.0 - 90.0
    lon += int(g[2]) * 2.0
    lat += int(g[3]) * 1.0
    if len(g) == 6:
        lon += (ord(g[4]) - ord('A')) * (5.0 / 60.0) + (2.5 / 60.0)
        lat += (ord(g[5]) - ord('A')) * (2.5 / 60.0) + (1.25 / 60.0)
    else:
        lon += 1.0
        lat += 0.5
    return lat, lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    import math
    R_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R_km * math.asin(math.sqrt(a))


def range_km_for_grid(grid: str | None,
                       my_lat: float | None,
                       my_lon: float | None) -> float | None:
    if not grid or my_lat is None or my_lon is None:
        return None
    try:
        lat, lon = grid_to_lat_lon(grid)
    except ValueError:
        return None
    return haversine_km(my_lat, my_lon, lat, lon)


def extract_grid_assignments(decodes: list[dict]) -> dict[str, str]:
    """For each callsign appearing in a message with a grid token, return
    its most common observed grid.  The grid is associated with the
    *last callsign before the grid token* in the message (the sender).

    Examples
    --------
    "CQ K1ABC FN42"            → grid for K1ABC
    "K1ABC W2DEF FN42"         → grid for W2DEF
    "K1ABC W2DEF -01"          → no grid
    "K1ABC W2DEF RR73"         → no grid — RR73 is a sign-off, not a grid

    Note that ``RR73`` matches the Maidenhead pattern (RR is letters in
    [A-R], 73 is digits) but is the WSJT sign-off — must be skipped
    explicitly via ``_SKIP_WORDS``.  Without this skip, every QSO ending
    "RR73" would falsely assign grid ``RR73`` (point in the Arctic
    Ocean) to the receiving station.
    """
    grid_obs: dict[str, Counter] = defaultdict(Counter)
    for d in decodes:
        tokens = d['message'].split()
        grid_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            t = tokens[i]
            if t in _SKIP_WORDS:           # RR73 / 73 / CQ / DE / QRZ / RRR
                continue
            if _GRID_TOKEN_RE.match(t):
                grid_idx = i
                break
        if grid_idx is None:
            continue
        grid = tokens[grid_idx].upper()
        for i in range(grid_idx - 1, -1, -1):
            t = tokens[i]
            if t in _SKIP_WORDS:
                continue
            if _GRID_TOKEN_RE.match(t):
                continue
            if _CALL_RE.match(t):
                grid_obs[t][grid] += 1
                break
    return {cs: ctr.most_common(1)[0][0] for cs, ctr in grid_obs.items()}


def classify_callsigns(
    map144_all: list[dict],
    wsjtx_all: list[dict],
    cs2grid: dict[str, str] | None = None,
    my_lat: float | None = None,
    my_lon: float | None = None,
    *,
    min_local_periods: int = 30,
    max_ping_periods: int = 5,
    local_max_km: float = 300.0,
) -> dict[str, str]:
    """Return ``{callsign: classification}`` over the union of decodes.

    Period count is attributed to the *transmitting* station only — the
    addressed station in a QSO message ("X Y FN20" → Y is sender,
    X is just whom they're calling) contributes nothing.  See
    :func:`sender_in_message`.
    """
    cs2grid = cs2grid or {}
    periods_by_call: dict[str, set] = defaultdict(set)
    for src in (map144_all, wsjtx_all):
        for d in src:
            sender = sender_in_message(d['message'])
            if sender is not None:
                periods_by_call[sender].add(d['ts'])

    out: dict[str, str] = {}
    for cs, periods in periods_by_call.items():
        n = len(periods)
        if n <= max_ping_periods:
            out[cs] = "TRUE_PING"
        elif n < min_local_periods:
            out[cs] = "WEAK_FADING"
        else:
            grid = cs2grid.get(cs)
            rng = range_km_for_grid(grid, my_lat, my_lon)
            if rng is None:
                out[cs] = "STRONG_LOCAL"      # unknown distance → conservative
            elif rng < local_max_km:
                out[cs] = "STRONG_LOCAL"
            else:
                out[cs] = "CONT_PROP"
    return out


def classify_decode(d: dict, cs2cls: dict[str, str]) -> str:
    """Return the classification of the decode's *transmitting* station."""
    sender = sender_in_message(d['message'])
    if sender is None:
        return "UNKNOWN"
    return cs2cls.get(sender, "UNKNOWN")


def build_callsign_rows(
    decodes: list[dict],
    cs2cls: dict[str, str],
    all_time_decodes: list[dict],
    cs2grid: dict[str, str] | None = None,
    my_lat: float | None = None,
    my_lon: float | None = None,
) -> list[dict]:
    """One row per *transmitting* callsign in ``decodes``.

    Each statistic (SNR percentiles, decode count, period count) is
    attributed to the message's SENDER only.  Addressed stations are
    not credited — they didn't transmit anything we received in the
    relevant message.  See :func:`sender_in_message`.
    """
    cs2grid = cs2grid or {}
    by_call_snrs: dict[str, list[float]] = defaultdict(list)
    by_call_periods: dict[str, set] = defaultdict(set)
    by_call_count: Counter = Counter()
    for d in decodes:
        sender = sender_in_message(d['message'])
        if sender is None:
            continue
        snr = d.get('snr')
        if snr is not None:
            by_call_snrs[sender].append(snr)
        by_call_periods[sender].add(d['ts'])
        by_call_count[sender] += 1

    alltime_periods: dict[str, set] = defaultdict(set)
    for d in all_time_decodes:
        sender = sender_in_message(d['message'])
        if sender is not None:
            alltime_periods[sender].add(d['ts'])

    rows = []
    for cs, n in by_call_count.items():
        snrs = by_call_snrs[cs]
        if snrs:
            arr = np.array(snrs, dtype=float)
            row = {
                'callsign': cs,
                'n_decodes': n,
                'n_periods': len(by_call_periods[cs]),
                'snr_min': float(arr.min()),
                'snr_p25': float(np.percentile(arr, 25)),
                'snr_med': float(np.percentile(arr, 50)),
                'snr_p75': float(np.percentile(arr, 75)),
                'snr_max': float(arr.max()),
            }
        else:
            row = {
                'callsign': cs, 'n_decodes': n,
                'n_periods': len(by_call_periods[cs]),
                'snr_min': None, 'snr_p25': None, 'snr_med': None,
                'snr_p75': None, 'snr_max': None,
            }
        row['classification'] = cs2cls.get(cs, 'UNKNOWN')
        row['n_periods_alltime'] = len(alltime_periods[cs])
        grid = cs2grid.get(cs)
        row['grid'] = grid
        row['range_km'] = range_km_for_grid(grid, my_lat, my_lon)
        rows.append(row)
    return rows


# ── Capture-WAV lookup (per-decode audio inspection) ─────────────────────────


def decode_to_capture_path(d: dict, captures_dir: Path) -> Path | None:
    """Map a decode entry to its capture WAV file, if one exists.

    The capture filename convention is::

        YYYYMMDD_HHMMSSZ_<radio_khz>kHz_<message_with_underscores>.wav

    where the time component is the second-precision UTC timestamp
    (fractional second dropped) and the message tokens are joined by
    underscores (with ``+`` and ``-`` also collapsed to underscores).
    We glob on the timestamp + frequency prefix because the message
    suffix is lossy after the +/- substitutions and we don't need it
    to find the file.

    Reads ``raw_timestamp`` / ``raw_radio_khz`` if present (set by
    :func:`parse_map144`); falls back to ``ts`` (which is
    period-floored — works only if the capture happened exactly at the
    period boundary, which it doesn't, so the fallback is best-effort).
    """
    ts_str = d.get('raw_timestamp') or ''
    radio_khz = d.get('raw_radio_khz')
    if not ts_str or radio_khz is None:
        # Fallback: derive from parsed ``ts`` and ``freq_mhz``.  This is
        # period-floored and will only match if the capture happened
        # right on the period boundary; usually we want the raw fields.
        ts = d.get('ts')
        if ts is None:
            return None
        ts_str = ts.strftime('%Y-%m-%d_%H:%M:%S')
        if radio_khz is None and 'freq_mhz' in d:
            radio_khz = int(round(d['freq_mhz'] * 1000))
    try:
        date_part, time_part = ts_str.split('_', 1)
        time_part = time_part.split('.')[0]
        h, m, s = time_part.split(':')
        ts_compact = f"{date_part.replace('-', '')}_{h}{m}{s}Z"
    except ValueError:
        return None
    pattern = f"{ts_compact}_{radio_khz}kHz_*.wav"
    matches = list(captures_dir.glob(pattern))
    return matches[0] if matches else None


def wavs_for_sender(
    sender: str,
    decodes: list[dict],
    captures_dir: Path,
) -> list[tuple[dict, Path]]:
    """Return ``[(decode, wav_path), ...]`` for every decode where
    ``sender`` is the transmitting station and the corresponding WAV
    capture file exists.  Sorted newest-first.
    """
    out: list[tuple[dict, Path]] = []
    for d in decodes:
        if sender_in_message(d['message']) != sender:
            continue
        wav = decode_to_capture_path(d, captures_dir)
        if wav is not None and wav.is_file():
            out.append((d, wav))
    out.sort(key=lambda x: x[0]['ts'], reverse=True)
    return out


# ── Figure builders ───────────────────────────────────────────────────────────

_TAG_STYLE = {
    'NOT_DETECTED': dict(color='steelblue',  marker='^', label='Not detected'),
    'NO_DECODE':    dict(color='#ff7f0e',    marker='D', label='Detected / no decode'),
    'PERIOD_SLIP':  dict(color='#9467bd',    marker='v', label='Period slip (decoded ±15 s)'),
    'TIMEOUT':      dict(color='gray',       marker='^', label='Timeout'),
    '_UNKNOWN':     dict(color='gray',       marker='^', label='WSJT-X only'),
}


def _make_timeline_fig(matched, wsjtx_only, map144_only,
                       wsjtx_tags=None, date_label='', local_calls=None) -> Figure:
    """Time (Y, inverted) vs SNR (X).

    Text labels are NOT drawn at render time — that stalls with 500+ points.
    Labels are added lazily via fig._update_labels(show), which only annotates
    points currently visible in the axes.  _PlotPane wires this to a checkbox
    and auto-refreshes it on zoom/pan.
    """
    fig = Figure(figsize=(11, 10), facecolor='#f8f8f8')
    ax  = fig.add_subplot(111)
    ax.set_facecolor('#f8f8f8')

    def _yn(ts):
        return mdates.date2num(ts)

    # _point_data stores every point as (x, y_datenum, message, color)
    # for lazy annotation when the user enables labels.
    _point_data: list[tuple] = []

    def _collect(xs, ys, msgs, color):
        for x, y, msg in zip(xs, ys, msgs):
            _point_data.append((x, y, msg, color))

    # ── Both decoded ──────────────────────────────────────────────────────────
    if matched:
        xs = [w['snr'] for w, _ in matched]
        ys = [_yn(m['ts']) for _, m in matched]
        ax.scatter(xs, ys, marker='o', color='#2ca02c', s=50, zorder=6,
                   label=f'Both decoded  ({len(matched)})')
        _collect(xs, ys, [m['message'] for _, m in matched], '#2ca02c')

    # ── WSJT-X only, split by tag ─────────────────────────────────────────────
    if wsjtx_tags:
        tag_groups: dict[str, list] = defaultdict(list)
        for d in wsjtx_only:
            tag_groups[wsjtx_tags.get(id(d), '_UNKNOWN')].append(d)
        for tag, group in tag_groups.items():
            style = _TAG_STYLE.get(tag, _TAG_STYLE['_UNKNOWN'])
            xs = [d['snr'] for d in group]
            ys = [_yn(d['ts']) for d in group]
            ax.scatter(xs, ys, marker=style['marker'], color=style['color'],
                       s=45, zorder=5, alpha=0.85,
                       label=f"{style['label']}  ({len(group)})")
            _collect(xs, ys, [d['message'] for d in group], style['color'])
    elif wsjtx_only:
        xs = [d['snr'] for d in wsjtx_only]
        ys = [_yn(d['ts']) for d in wsjtx_only]
        ax.scatter(xs, ys, marker='^', color='steelblue', s=45, zorder=5,
                   label=f'WSJT-X only  ({len(wsjtx_only)})')
        _collect(xs, ys, [d['message'] for d in wsjtx_only], 'steelblue')

    # ── MAP144 only ───────────────────────────────────────────────────────────
    m144_with_snr = [d for d in map144_only if d.get('snr') is not None]
    if m144_with_snr:
        xs = [d['snr'] for d in m144_with_snr]
        ys = [_yn(d['ts']) for d in m144_with_snr]
        ax.scatter(xs, ys, marker='>', color='#d62728', s=45, zorder=5,
                   label=f'MAP144 only  ({len(map144_only)})')
        _collect(xs, ys, [d['message'] for d in m144_with_snr], '#d62728')

    # ── Local / terrestrial overlay (orange ring) ─────────────────────────────
    if local_calls:
        all_pts = [(x, y, msg, color) for x, y, msg, color in _point_data]
        lx = [x for x, y, msg, c in all_pts
              if any(w in local_calls for w in _calls_in_message(msg))]
        ly = [y for x, y, msg, c in all_pts
              if any(w in local_calls for w in _calls_in_message(msg))]
        if lx:
            ax.scatter(lx, ly, marker='o', s=120, zorder=7,
                       facecolors='none', edgecolors='orange', linewidths=1.8,
                       label=f'Local / terrestrial path  ({len(lx)})')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=18))
    ax.invert_yaxis()   # earlier at top

    ax.set_xlabel('SNR  (dB)', fontsize=11)
    ax.set_ylabel('UTC Time', fontsize=11)
    ax.grid(linestyle=':', alpha=0.35)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)

    title = 'MAP144 vs WSJT-X  —  Timeline'
    if date_label:
        title += f'\n{date_label}'
    ax.set_title(title, fontsize=12, pad=8)
    fig.tight_layout()

    # ── Lazy label machinery ──────────────────────────────────────────────────
    _active_annotations: list = []

    def _update_labels(show: bool) -> None:
        """Add/remove message annotations for points visible in current view."""
        for ann in _active_annotations:
            try:
                ann.remove()
            except Exception:
                pass
        _active_annotations.clear()

        if not show:
            return

        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())   # inverted axis: ylo > yhi in display
        for x, y, msg, color in _point_data:
            if xlo <= x <= xhi and ylo <= y <= yhi:
                ann = ax.annotate(
                    msg, (x, y), xytext=(5, 0),
                    textcoords='offset points', fontsize=6.5,
                    va='center', color=color, clip_on=True, alpha=0.85)
                _active_annotations.append(ann)

    fig._update_labels = _update_labels   # type: ignore[attr-defined]
    return fig


def _make_detection_rate_fig(matched, wsjtx_only, date_label='') -> Figure:
    BIN = 2
    all_w_snrs   = [w['snr'] for w, _ in matched] + [w['snr'] for w in wsjtx_only]
    match_w_snrs = [w['snr'] for w, _ in matched]
    if not all_w_snrs:
        fig = Figure(figsize=(9, 5))
        fig.add_subplot(111).set_title('No WSJT-X data')
        return fig

    lo = (min(all_w_snrs) // BIN) * BIN
    hi = (max(all_w_snrs) // BIN) * BIN + BIN
    bins = list(range(lo, hi + BIN, BIN))

    total_counts = defaultdict(int)
    match_counts = defaultdict(int)
    for s in all_w_snrs:
        total_counts[(s // BIN) * BIN] += 1
    for s in match_w_snrs:
        match_counts[(s // BIN) * BIN] += 1

    xs     = [b for b in bins if total_counts[b] > 0]
    totals = [total_counts[b] for b in xs]
    rates  = [match_counts[b] / total_counts[b] * 100 for b in xs]

    fig = Figure(figsize=(9, 5), facecolor='#f8f8f8')
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('#f8f8f8')

    ax2 = ax1.twinx()
    ax2.bar(xs, totals, width=BIN * 0.8, color='steelblue', alpha=0.25,
            label='WSJT-X decodes (count)')
    ax2.set_ylabel('WSJT-X decode count per bin', color='steelblue', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(0, max(totals) * 3.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    ax1.plot(xs, rates, color='#d62728', linewidth=2.5, marker='o',
             markersize=7, zorder=5, label='MAP144 detection rate')
    for x, r, t in zip(xs, rates, totals):
        if t >= 3:
            ax1.annotate(f'{r:.0f}%', (x, r),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8.5, color='#d62728')

    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax1.text(xs[-1] + 0.3, 51, '50 %', va='bottom', ha='right',
             fontsize=8, color='gray')

    ax1.set_xlabel('SNR reported by WSJT-X  (dB)', fontsize=12)
    ax1.set_ylabel('MAP144 detection rate  (%)', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 115)
    ax1.set_xlim(lo - BIN, hi + BIN)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(BIN))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):+d}'))
    ax1.grid(axis='x', linestyle=':', alpha=0.4)

    title = 'MAP144 vs WSJT-X  —  MSK144 detection rate by SNR'
    if date_label:
        title += f'\n{date_label}'
    ax1.set_title(title, fontsize=13, pad=10)

    handles = [
        Line2D([0], [0], color='#d62728', lw=2.5, marker='o', ms=7,
               label='MAP144 detection rate'),
        Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.4,
                  label='WSJT-X decode count'),
    ]
    ax1.legend(handles=handles, loc='upper left', fontsize=10, framealpha=0.8)
    fig.tight_layout()
    return fig


def _dot_stack(values, step=0.30, max_height=4.5):
    bin_idx: dict[int, int] = {}
    offsets = np.zeros(len(values))
    for i, v in enumerate(values):
        b = round(float(v))
        idx = bin_idx.get(b, 0)
        bin_idx[b] = idx + 1
        offsets[i] = min(idx * step, max_height)
    return offsets


def _make_snr_scatter_fig(matched, wsjtx_only, map144_only,
                          date_label='', wsjtx_tags=None, local_calls=None) -> Figure:
    matched_with_snr = [(w, m) for w, m in matched if m.get('snr') is not None]
    mx = [w['snr'] for w, _ in matched_with_snr]
    my = [m['snr'] for _, m in matched_with_snr]

    _NODEC_TAGS = ('NO_DECODE', 'TIMEOUT', 'ERROR')
    if wsjtx_tags:
        wx_nd   = [w['snr'] for w in wsjtx_only if wsjtx_tags.get(id(w)) == 'NOT_DETECTED']
        wx_dec  = [w['snr'] for w in wsjtx_only if wsjtx_tags.get(id(w)) in _NODEC_TAGS]
        wx_slip = [w['snr'] for w in wsjtx_only if wsjtx_tags.get(id(w)) == 'PERIOD_SLIP']
        wx_unk  = [w['snr'] for w in wsjtx_only
                   if wsjtx_tags.get(id(w)) not in ('NOT_DETECTED', 'PERIOD_SLIP') + _NODEC_TAGS]
    else:
        wx_nd, wx_dec, wx_slip = [], [], []
        wx_unk = [w['snr'] for w in wsjtx_only]

    py = [m['snr'] for m in map144_only if m.get('snr') is not None]
    all_x = mx + [w['snr'] for w in wsjtx_only]
    all_y = my + py
    if not all_x and not all_y:
        fig = Figure(figsize=(8, 7))
        fig.add_subplot(111).set_title('No data')
        return fig

    data_lo = min(min(all_x, default=0), min(all_y, default=0))
    data_hi = max(max(all_x, default=0), max(all_y, default=0))
    SENTINEL = data_lo - 7

    fig = Figure(figsize=(8, 7), facecolor='#f8f8f8')
    ax  = fig.add_subplot(111)
    ax.set_facecolor('#f8f8f8')

    n_est = sum(1 for _, m in matched_with_snr if m.get('jt9_snr') != m.get('snr'))
    snr_label = (f'Both decoded  (n={len(mx)}'
                 + (f', {n_est} using est_snr' if n_est else '') + ')')
    if mx:
        ax.scatter(mx, my, s=50, color='#2ca02c', alpha=0.75, zorder=4,
                   label=snr_label)

    for vals, color, marker, label_prefix in (
        (wx_nd,   'steelblue',  '^', 'Not detected'),
        (wx_dec,  '#ff7f0e',   'D', 'Detected / no decode'),
        (wx_slip, '#9467bd',   'v', 'Period slip (decoded ±15 s)'),
        (wx_unk,  'gray',      '^', 'WSJT-X only (no launch data)'),
    ):
        if vals:
            offs = _dot_stack(vals)
            ax.scatter(vals, SENTINEL + offs, s=45,
                       color=color, alpha=0.80, marker=marker, zorder=3,
                       label=f'{label_prefix}  (n={len(vals)})')

    if py:
        offs = _dot_stack(py)
        ax.scatter(SENTINEL + offs, py, s=40,
                   color='#d62728', alpha=0.70, marker='>', zorder=3,
                   label=f'MAP144 only  (n={len(py)})')

    ax.axhline(SENTINEL, color='steelblue', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(SENTINEL, color='#d62728',   linewidth=0.8, linestyle='--', alpha=0.5)

    lim_lo = SENTINEL - 1
    lim_hi = data_hi + 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color='gray',
            linestyle=':', linewidth=1, alpha=0.6, label='y = x  (perfect agreement)')

    tick_lo = (data_lo // 2) * 2
    tick_hi = (data_hi // 2) * 2 + 2
    real_ticks = list(range(tick_lo, tick_hi + 1, 2))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator(real_ticks + [SENTINEL]))
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _, s=SENTINEL: ('missed' if abs(v - s) < 0.5 else f'{int(v):+d}')))

    ax.set_xlim(SENTINEL - 1.5, lim_hi)
    ax.set_ylim(SENTINEL - 1.5, lim_hi)
    ax.set_xlabel('WSJT-X SNR  (dB)\n[coherent matched filter, 15 s frame]', fontsize=11)
    ax.set_ylabel('MAP144 SNR  (dB)\n[est_snr_db: pre-burst PSD, 1500 ms window]', fontsize=11)
    ax.grid(linestyle=':', alpha=0.35)

    # ── Local / terrestrial overlay ───────────────────────────────────────────
    if local_calls:
        # Both-decoded locals: ring on scatter (WSJT-X SNR on x, MAP144 SNR on y)
        lx_m = [w['snr'] for w, m in matched_with_snr if _is_local(w, local_calls)]
        ly_m = [m['snr'] for w, m in matched_with_snr if _is_local(w, local_calls)]
        if lx_m:
            ax.scatter(lx_m, ly_m, s=120, facecolors='none',
                       edgecolors='orange', linewidths=1.8, zorder=6,
                       label=f'Local / terrestrial  ({len(lx_m)} matched)')
        # WSJT-X-only locals: ring on bottom rail
        lx_w = [w['snr'] for w in wsjtx_only if _is_local(w, local_calls)]
        if lx_w:
            ax.scatter(lx_w, [SENTINEL] * len(lx_w), s=120, facecolors='none',
                       edgecolors='orange', linewidths=1.8, zorder=6)

    title = 'MAP144 vs WSJT-X  —  per-ping SNR comparison'
    if date_label:
        title += f'\n{date_label}'
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=10, framealpha=0.85, loc='upper left')
    fig.tight_layout()
    return fig


# ── Figure builders, by-class variants ───────────────────────────────────────

#: Class colour map shared between the by-class plots and the GUI table.
CLASS_COLORS = {
    "TRUE_PING":    "#1f77b4",   # blue   — primary mission
    "WEAK_FADING":  "#2ca02c",   # green  — secondary mission
    "CONT_PROP":    "#ff7f0e",   # orange — distant-but-frequent (Es / F2 / beacon)
    "STRONG_LOCAL": "#d62728",   # red    — local interference
}


def _make_detection_rate_by_class_fig(
    matched, wsjtx_only, period_slip_ids: set, cs2cls: dict[str, str],
    date_label: str = '',
) -> Figure:
    """MAP144 detection rate (hits / WSJT-X total) vs WSJT-X SNR, one
    line per classification.  Period-slipped decodes count as hits —
    MAP144 *did* decode them, just landed in a different period.
    """
    snr_bins = np.arange(-8.5, 9.5, 1.0)
    bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    classes = ("TRUE_PING", "WEAK_FADING", "CONT_PROP", "STRONG_LOCAL")
    hits = {c: np.zeros(len(bin_centers)) for c in classes}
    total = {c: np.zeros(len(bin_centers)) for c in classes}

    def bin_idx(snr):
        if snr is None:
            return None
        i = int(np.searchsorted(snr_bins, snr) - 1)
        return i if 0 <= i < len(bin_centers) else None

    for w, _m in matched:
        cls = classify_decode(w, cs2cls)
        if cls not in classes:
            continue
        i = bin_idx(w['snr'])
        if i is None:
            continue
        hits[cls][i] += 1
        total[cls][i] += 1
    for w in wsjtx_only:
        cls = classify_decode(w, cs2cls)
        if cls not in classes:
            continue
        i = bin_idx(w['snr'])
        if i is None:
            continue
        if id(w) in period_slip_ids:
            hits[cls][i] += 1
        total[cls][i] += 1

    fig = Figure(figsize=(8.5, 6.5))
    ax_rate = fig.add_axes([0.10, 0.32, 0.85, 0.55])
    ax_count = fig.add_axes([0.10, 0.08, 0.85, 0.18], sharex=ax_rate)
    for cls in classes:
        t = total[cls]
        rate = np.divide(hits[cls], t,
                         out=np.full_like(t, np.nan), where=t > 0)
        ax_rate.plot(bin_centers, rate, 'o-',
                     color=CLASS_COLORS[cls], label=cls, lw=2, ms=6)
        ax_count.bar(bin_centers, t, width=0.8,
                     color=CLASS_COLORS[cls], alpha=0.4, label=cls)
    ax_rate.set_ylim(-0.05, 1.05)
    ax_rate.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax_rate.set_ylabel('MAP144 detection rate\n(hits / WSJT-X total)')
    title = 'MAP144 detection rate by class — period-slipped decodes count as hits'
    if date_label:
        title += f'\n{date_label}'
    ax_rate.set_title(title, fontsize=12)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(loc='lower right')
    ax_count.set_ylabel('WSJT-X count\n(per bin)')
    ax_count.set_xlabel('WSJT-X SNR (dB)')
    ax_count.grid(True, alpha=0.3)
    return fig


def _make_snr_scatter_by_class_fig(
    matched, wsjtx_only, map144_only, cs2cls: dict[str, str],
    date_label: str = '',
) -> Figure:
    """SNR vs UTC time, one panel per classification.

    Panels stacked vertically with shared x-axis: TRUE_PING, WEAK_FADING,
    CONT_PROP, STRONG_LOCAL.  WSJT-X-only blue dots, MAP144-only red x,
    matched green o.
    """
    classes = ("TRUE_PING", "WEAK_FADING", "CONT_PROP", "STRONG_LOCAL")
    fig = Figure(figsize=(11, 9))
    axes = []
    n = len(classes)
    for k, cls in enumerate(classes):
        ax = fig.add_subplot(n, 1, k + 1, sharex=axes[0] if axes else None)
        axes.append(ax)

    by_class = {c: {'matched': [], 'wsjtx_only': [], 'map144_only': []}
                for c in classes}
    for w, _m in matched:
        cls = classify_decode(w, cs2cls)
        if cls in by_class:
            by_class[cls]['matched'].append((w['ts'], w['snr']))
    for w in wsjtx_only:
        cls = classify_decode(w, cs2cls)
        if cls in by_class:
            by_class[cls]['wsjtx_only'].append((w['ts'], w['snr']))
    for d in map144_only:
        cls = classify_decode(d, cs2cls)
        if cls in by_class:
            by_class[cls]['map144_only'].append((d['ts'], d['snr']))

    for ax, cls in zip(axes, classes):
        bucket = by_class[cls]

        def split(pairs):
            if not pairs:
                return [], []
            ts, snr = zip(*[(t, s) for t, s in pairs if s is not None])
            return list(ts), list(snr)

        m_t, m_s = split(bucket['matched'])
        w_t, w_s = split(bucket['wsjtx_only'])
        x_t, x_s = split(bucket['map144_only'])
        ax.scatter(w_t, w_s, color='#1f77b4', marker='.', s=18, alpha=0.55,
                   label=f"WSJT-X only (n={len(w_t)})")
        ax.scatter(x_t, x_s, color='#d62728', marker='x', s=22, alpha=0.55,
                   label=f"MAP144 only (n={len(x_t)})")
        ax.scatter(m_t, m_s, color='#2ca02c', marker='o', s=22, alpha=0.65,
                   label=f"matched (n={len(m_t)})")
        ax.set_ylabel('SNR (dB)')
        n_total = len(m_t) + len(w_t) + len(x_t)
        ax.set_title(f"{cls}  —  {n_total} decodes", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-12, 25)
    axes[-1].set_xlabel('UTC time')
    suptitle = 'SNR scatter by class'
    if date_label:
        suptitle += f' — {date_label}'
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ── Legacy CLI plot wrappers (kept for --no-gui mode) ─────────────────────────

def plot_detection_rate(matched, wsjtx_only, out_path, date_label=''):
    fig = _make_detection_rate_fig(matched, wsjtx_only, date_label)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")
    p = Path(out_path)
    scatter_path = p.with_stem(p.stem + '_scatter')
    return scatter_path


def plot_snr_scatter(matched, wsjtx_only, map144_only, out_path,
                     date_label='', wsjtx_tags=None):
    fig = _make_snr_scatter_fig(matched, wsjtx_only, map144_only, date_label, wsjtx_tags)
    fig.savefig(out_path, dpi=150)
    print(f"Scatter plot saved → {out_path}")


# ── GUI ────────────────────────────────────────────────────────────────────────

class _PlotPane(QtWidgets.QWidget):
    """Matplotlib figure embedded in a widget with toolbar, Labels toggle, and Save button."""

    def __init__(self, fig: Figure, default_stem: str, parent=None):
        super().__init__(parent)
        self._fig   = fig
        self._stem  = default_stem
        self._labels_cb: QtWidgets.QCheckBox | None = None
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._do_refresh_labels)

        self._canvas  = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self._canvas, self)

        hbar = QtWidgets.QHBoxLayout()
        hbar.addWidget(toolbar)

        # Labels checkbox — only for figures that support lazy annotations
        if hasattr(fig, '_update_labels'):
            cb = QtWidgets.QCheckBox("Labels")
            cb.setToolTip("Show message labels for visible points.\n"
                          "Zoom in first, then enable for best readability.")
            cb.stateChanged.connect(self._on_labels_changed)
            hbar.addWidget(cb)
            self._labels_cb = cb
            # Auto-refresh labels whenever the axes limits change (zoom/pan)
            for ax in fig.axes:
                ax.callbacks.connect('xlim_changed', self._on_axis_changed)
                ax.callbacks.connect('ylim_changed', self._on_axis_changed)

        save_btn = QtWidgets.QPushButton("Save PNG…")
        save_btn.setFixedWidth(100)
        save_btn.clicked.connect(self._on_save)
        hbar.addWidget(save_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(hbar)
        layout.addWidget(self._canvas)

    # ── Label callbacks ───────────────────────────────────────────────────────

    def _on_labels_changed(self, state):
        show = (state == QtCore.Qt.Checked)
        self._fig._update_labels(show)
        self._canvas.draw_idle()

    def _on_axis_changed(self, _ax):
        """Debounced refresh: only re-annotate if labels are currently enabled."""
        if self._labels_cb is not None and self._labels_cb.isChecked():
            self._refresh_timer.start(120)   # ms — coalesce rapid zoom events

    def _do_refresh_labels(self):
        if self._labels_cb is not None and self._labels_cb.isChecked():
            self._fig._update_labels(True)
            self._canvas.draw_idle()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _on_save(self):
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        default = f"{self._stem}_{ts}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot", default, "PNG (*.png);;All files (*)")
        if path:
            self._fig.savefig(path, dpi=150)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to\n{path}")


class CompareGUI(QtWidgets.QMainWindow):
    def __init__(self, initial_date: str | None = None):
        super().__init__()
        self.setWindowTitle("MAP144 vs WSJT-X Compare")
        self.resize(1400, 900)

        self._figs: list[Figure] = []

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = QtWidgets.QWidget()
        cbox = QtWidgets.QHBoxLayout(ctrl)
        cbox.setContentsMargins(6, 4, 6, 4)

        cbox.addWidget(QtWidgets.QLabel("Date (YYYYMMDD):"))
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        self._date_edit = QtWidgets.QLineEdit(initial_date or today)
        self._date_edit.setFixedWidth(90)
        cbox.addWidget(self._date_edit)

        cbox.addSpacing(12)
        cbox.addWidget(QtWidgets.QLabel("UTC window:"))
        self._utc_edit = QtWidgets.QLineEdit()
        self._utc_edit.setPlaceholderText("HH:MM-HH:MM  (blank = all)")
        self._utc_edit.setFixedWidth(160)
        cbox.addWidget(self._utc_edit)

        cbox.addSpacing(12)
        cbox.addWidget(QtWidgets.QLabel("Match window (s):"))
        self._win_spin = QtWidgets.QDoubleSpinBox()
        self._win_spin.setRange(0.5, 30.0)
        self._win_spin.setSingleStep(0.5)
        self._win_spin.setValue(1.0)
        self._win_spin.setFixedWidth(70)
        cbox.addWidget(self._win_spin)

        cbox.addSpacing(12)
        cbox.addWidget(QtWidgets.QLabel("Local ≥"))
        self._local_spin = QtWidgets.QSpinBox()
        self._local_spin.setRange(1, 100)
        self._local_spin.setValue(20)
        self._local_spin.setFixedWidth(55)
        self._local_spin.setToolTip(
            "Callsigns appearing in this many periods or more are flagged as\n"
            "local / terrestrial paths and highlighted orange in the plots.")
        cbox.addWidget(self._local_spin)
        cbox.addWidget(QtWidgets.QLabel("periods"))

        # New: operator's grid (drives range_km column + density-distance class)
        cbox.addSpacing(12)
        cbox.addWidget(QtWidgets.QLabel("My grid:"))
        self._mygrid_edit = QtWidgets.QLineEdit("FN42EV")
        self._mygrid_edit.setFixedWidth(72)
        self._mygrid_edit.setToolTip(
            "Operator's Maidenhead grid (4 or 6 char) — used for great-circle\n"
            "range column and the local-vs-distant classifier split.")
        cbox.addWidget(self._mygrid_edit)

        cbox.addSpacing(12)
        cbox.addWidget(QtWidgets.QLabel("Local max km:"))
        self._localkm_spin = QtWidgets.QDoubleSpinBox()
        self._localkm_spin.setRange(50.0, 5000.0)
        self._localkm_spin.setSingleStep(50.0)
        self._localkm_spin.setDecimals(0)
        self._localkm_spin.setValue(300.0)
        self._localkm_spin.setFixedWidth(80)
        self._localkm_spin.setToolTip(
            "High-density callsign within this range = STRONG_LOCAL;\n"
            "beyond this range = CONT_PROP (continuous propagation,\n"
            "e.g. sporadic-E season distant station).")
        cbox.addWidget(self._localkm_spin)

        cbox.addSpacing(12)
        self._no_launches_cb = QtWidgets.QCheckBox("No launches data")
        cbox.addWidget(self._no_launches_cb)

        cbox.addSpacing(20)
        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setFixedWidth(70)
        self._run_btn.clicked.connect(self._run)
        cbox.addWidget(self._run_btn)

        self._save_all_btn = QtWidgets.QPushButton("Save all PNGs")
        self._save_all_btn.setEnabled(False)
        self._save_all_btn.clicked.connect(self._save_all)
        cbox.addWidget(self._save_all_btn)

        cbox.addStretch()

        # ── Summary text ──────────────────────────────────────────────────────
        self._summary = QtWidgets.QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(200)
        self._summary.setFont(QtWidgets.QApplication.font())

        # ── Plot tabs ─────────────────────────────────────────────────────────
        self._tabs = QtWidgets.QTabWidget()

        # ── Callsign table (left pane — populated in _do_run) ─────────────────
        self._callsign_summary = QtWidgets.QLabel("Run to populate.")
        self._callsign_summary.setStyleSheet("font-weight: 600; padding: 2px;")
        self._callsign_table = QtWidgets.QTableWidget(0, 12)
        self._callsign_table.setHorizontalHeaderLabels([
            "callsign", "grid", "rng_km", "class",
            "n_dec", "n_per", "n_alltime",
            "SNR min", "p25", "med", "p75", "max",
        ])
        self._callsign_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self._callsign_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._callsign_table.setAlternatingRowColors(True)
        self._callsign_table.setSortingEnabled(True)
        # Right-click → submenu of capture WAVs for the selected sender,
        # each of which launches analyze_msk144.py as a subprocess.
        self._callsign_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._callsign_table.customContextMenuRequested.connect(
            self._on_callsign_context_menu)
        # Cache populated each Run; empty until then.
        self._all_decodes_for_lookup: list[dict] = []
        left_pane = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left_pane)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(self._callsign_summary)
        lv.addWidget(self._callsign_table, 1)

        # ── Layout (horizontal splitter: callsign table | tabs) ───────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_pane)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 1100])
        self._splitter = splitter

        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(ctrl)
        vbox.addWidget(self._summary)
        vbox.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready — press Run to load data.")

        # Auto-run if date provided
        QtCore.QTimer.singleShot(0, self._run)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_utc_window(self):
        txt = self._utc_edit.text().strip()
        if not txt:
            return None, None
        try:
            t0s, t1s = txt.split('-')
            t0h, t0m = int(t0s.split(':')[0]), int(t0s.split(':')[1])
            t1h, t1m = int(t1s.split(':')[0]), int(t1s.split(':')[1])
            return t0h * 60 + t0m, t1h * 60 + t1m
        except Exception:
            return None, None

    def _date_label(self):
        d = self._date_edit.text().strip()
        if len(d) == 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return d

    def _run(self):
        self._run_btn.setEnabled(False)
        self._save_all_btn.setEnabled(False)
        self.statusBar().showMessage("Loading…")
        QtWidgets.QApplication.processEvents()

        try:
            self._do_run()
        except Exception as exc:
            import traceback
            self._summary.setPlainText(f"Error:\n{exc}\n\n{traceback.format_exc()}")
            self.statusBar().showMessage(f"Error: {exc}")
        finally:
            self._run_btn.setEnabled(True)

    def _do_run(self):
        # ── Load data ─────────────────────────────────────────────────────────
        wsjtx_all  = parse_wsjtx(WSJTX_ALL)
        map144_all = parse_map144(MAP144_LOG)

        n_test = sum(1 for d in map144_all if is_test_message(d['message']))
        map144_all = [d for d in map144_all if not is_test_message(d['message'])]

        # ── Date filter ───────────────────────────────────────────────────────
        date_str = self._date_edit.text().strip()
        if date_str:
            try:
                day_start = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                day_end   = day_start + timedelta(days=1)
                wsjtx_all  = [d for d in wsjtx_all  if day_start <= d['ts'] < day_end]
                map144_all = [d for d in map144_all if day_start <= d['ts'] < day_end]
            except ValueError:
                pass

        # ── UTC window filter ─────────────────────────────────────────────────
        lo_min, hi_min = self._parse_utc_window()
        if lo_min is not None:
            def _in_win(ts):
                m = ts.hour * 60 + ts.minute
                return lo_min <= m < hi_min
            wsjtx_all  = [d for d in wsjtx_all  if _in_win(d['ts'])]
            map144_all = [d for d in map144_all if _in_win(d['ts'])]

        # ── Launches ──────────────────────────────────────────────────────────
        launches: list[dict] = []
        if not self._no_launches_cb.isChecked():
            launches = parse_launches(MAP144_LAUNCHES)

        # ── Match ─────────────────────────────────────────────────────────────
        win = self._win_spin.value()
        matched, wsjtx_only, map144_only = match_decodes(wsjtx_all, map144_all, win)

        wsjtx_tags, tag_counts = {}, {}
        if launches:
            wsjtx_tags, tag_counts = _build_wsjtx_tags(wsjtx_only, launches)

        all_decodes = ([w for w, _ in matched] + [m for _, m in matched]
                       + wsjtx_only + map144_only)
        local_calls = _local_callsigns(all_decodes, self._local_spin.value())

        # ── Summary text ──────────────────────────────────────────────────────
        summary = _format_summary(
            matched, wsjtx_only, map144_only,
            wsjtx_all, map144_all,
            wsjtx_tags=wsjtx_tags, tag_counts=tag_counts,
            local_calls=local_calls,
        )
        if n_test:
            summary = f"(filtered {n_test} MAP144 test messages)\n\n" + summary
        self._summary.setPlainText(summary)

        # ── Classification (density + range) ──────────────────────────────────
        my_grid = self._mygrid_edit.text().strip().upper() or "FN42EV"
        try:
            my_lat, my_lon = grid_to_lat_lon(my_grid)
        except ValueError:
            self._summary.setPlainText(
                f"{self._summary.toPlainText()}\n\n"
                f"Warning: invalid My grid {my_grid!r}; using FN42EV"
            )
            my_grid = "FN42EV"
            my_lat, my_lon = grid_to_lat_lon(my_grid)
        local_max_km = float(self._localkm_spin.value())
        cs2grid = extract_grid_assignments(map144_all + wsjtx_all)
        cs2cls = classify_callsigns(
            map144_all, wsjtx_all,
            cs2grid=cs2grid, my_lat=my_lat, my_lon=my_lon,
            min_local_periods=self._local_spin.value(),
            max_ping_periods=5,
            local_max_km=local_max_km,
        )

        # Period-slip set for the per-class detection-rate plot (period-
        # slipped wsjtx_only entries count as MAP144 hits — MAP144 *did*
        # decode them, just into a different period).
        period_slip_ids = {
            d_id for d_id, tag in wsjtx_tags.items()
            if tag == 'PERIOD_SLIP'
        }

        # ── Callsign table (left pane) ────────────────────────────────────────
        rows = build_callsign_rows(
            map144_all + wsjtx_all, cs2cls, map144_all + wsjtx_all,
            cs2grid=cs2grid, my_lat=my_lat, my_lon=my_lon,
        )
        self._populate_callsign_table(rows, my_grid)
        # Cache the merged decode list so the right-click "Analyze captures…"
        # menu can look up WAV paths without re-parsing the logs.
        self._all_decodes_for_lookup = map144_all + wsjtx_all

        # ── Build figures ─────────────────────────────────────────────────────
        date_label = self._date_label()
        timeline_fig = _make_timeline_fig(
            matched, wsjtx_only, map144_only,
            wsjtx_tags=wsjtx_tags or None,
            date_label=date_label,
            local_calls=local_calls or None,
        )
        detrate_fig = _make_detection_rate_fig(matched, wsjtx_only, date_label)
        scatter_fig = _make_snr_scatter_fig(
            matched, wsjtx_only, map144_only, date_label,
            wsjtx_tags=wsjtx_tags or None,
            local_calls=local_calls or None,
        )
        detrate_byclass_fig = _make_detection_rate_by_class_fig(
            matched, wsjtx_only, period_slip_ids, cs2cls,
            date_label=f"{date_label} @{my_grid}",
        )
        scatter_byclass_fig = _make_snr_scatter_by_class_fig(
            matched, wsjtx_only, map144_only, cs2cls,
            date_label=f"{date_label} @{my_grid}",
        )

        # Close old figures to release memory
        for f in self._figs:
            f.clf()
        self._figs = [
            timeline_fig, detrate_fig, scatter_fig,
            detrate_byclass_fig, scatter_byclass_fig,
        ]

        # ── Populate tabs ─────────────────────────────────────────────────────
        self._tabs.clear()
        stem = f"compare_{date_str}" if date_str else "compare"
        self._tabs.addTab(_PlotPane(timeline_fig, f"{stem}_timeline"), "Timeline")
        self._tabs.addTab(_PlotPane(detrate_fig,  f"{stem}_detect"),   "Detection Rate")
        self._tabs.addTab(_PlotPane(scatter_fig,  f"{stem}_scatter"),  "SNR Scatter")
        self._tabs.addTab(_PlotPane(detrate_byclass_fig,
                                    f"{stem}_detect_byclass"),
                          "Detection Rate (by class)")
        self._tabs.addTab(_PlotPane(scatter_byclass_fig,
                                    f"{stem}_scatter_byclass"),
                          "SNR Scatter (by class)")

        self._save_all_btn.setEnabled(True)
        n_total = len(matched) + len(wsjtx_only) + len(map144_only)
        self.statusBar().showMessage(
            f"Loaded — {n_total} unique pings  |  "
            f"WSJT-X: {len(wsjtx_all)}  MAP144: {len(map144_all)}  "
            f"matched: {len(matched)}")

    def _populate_callsign_table(self, rows: list[dict], my_grid: str) -> None:
        """Fill the left-pane callsign table from ``build_callsign_rows`` output.

        Sorted by classification (mission first) then n_decodes desc.
        Class column is colour-coded per ``CLASS_COLORS`` shifted to Qt
        named colours for legibility against the alternating-row backdrop.
        """
        sort_rank = {"TRUE_PING": 0, "WEAK_FADING": 1,
                     "CONT_PROP": 2, "STRONG_LOCAL": 3, "UNKNOWN": 4}
        rows = sorted(
            rows, key=lambda r: (sort_rank[r["classification"]], -r["n_decodes"]),
        )
        cls_color = {
            "TRUE_PING":    QtCore.Qt.darkBlue,
            "WEAK_FADING":  QtCore.Qt.darkGreen,
            "CONT_PROP":    QtCore.Qt.darkYellow,
            "STRONG_LOCAL": QtCore.Qt.darkRed,
            "UNKNOWN":      QtCore.Qt.gray,
        }

        # Disable sorting while populating to avoid visual flicker.
        self._callsign_table.setSortingEnabled(False)
        self._callsign_table.setRowCount(len(rows))

        def cell_int(v):
            it = QtWidgets.QTableWidgetItem()
            it.setData(QtCore.Qt.DisplayRole, int(v))
            it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return it

        def cell_float_db(v):
            it = QtWidgets.QTableWidgetItem()
            if v is None:
                it.setData(QtCore.Qt.DisplayRole, "—")
            else:
                it.setData(QtCore.Qt.DisplayRole, f"{v:+.0f}")
            it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return it

        def cell_text(v, align_right=False):
            it = QtWidgets.QTableWidgetItem(v if v else "—")
            if align_right:
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return it

        def cell_range(km):
            it = QtWidgets.QTableWidgetItem()
            if km is None:
                it.setData(QtCore.Qt.DisplayRole, "—")
            else:
                it.setData(QtCore.Qt.DisplayRole, int(round(km)))
            it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return it

        for i, r in enumerate(rows):
            self._callsign_table.setItem(i, 0,
                QtWidgets.QTableWidgetItem(r["callsign"]))
            self._callsign_table.setItem(i, 1, cell_text(r["grid"]))
            self._callsign_table.setItem(i, 2, cell_range(r["range_km"]))
            it_cls = QtWidgets.QTableWidgetItem(r["classification"])
            it_cls.setForeground(cls_color.get(r["classification"], QtCore.Qt.black))
            self._callsign_table.setItem(i, 3, it_cls)
            self._callsign_table.setItem(i, 4, cell_int(r["n_decodes"]))
            self._callsign_table.setItem(i, 5, cell_int(r["n_periods"]))
            self._callsign_table.setItem(i, 6, cell_int(r["n_periods_alltime"]))
            self._callsign_table.setItem(i, 7, cell_float_db(r["snr_min"]))
            self._callsign_table.setItem(i, 8, cell_float_db(r["snr_p25"]))
            self._callsign_table.setItem(i, 9, cell_float_db(r["snr_med"]))
            self._callsign_table.setItem(i, 10, cell_float_db(r["snr_p75"]))
            self._callsign_table.setItem(i, 11, cell_float_db(r["snr_max"]))

        self._callsign_table.resizeColumnsToContents()
        self._callsign_table.setSortingEnabled(True)

        # Header summary: total + per-class counts.
        cls_counts = Counter(r["classification"] for r in rows)
        summary_parts = []
        for c in ("TRUE_PING", "WEAK_FADING", "CONT_PROP", "STRONG_LOCAL", "UNKNOWN"):
            n = cls_counts.get(c, 0)
            if n:
                summary_parts.append(f"{c}={n}")
        self._callsign_summary.setText(
            f"{len(rows)} callsigns @ {my_grid} — " + ", ".join(summary_parts)
        )

    # ── Right-click integration with analyze_msk144.py ───────────────────────

    _CAPTURES_DIR = Path(__file__).parent / "MSK144" / "detections"
    _ANALYZE_MSK144 = Path(__file__).parent / "analyze_msk144.py"

    def _on_callsign_context_menu(self, pos):
        """Show "Analyze captures…" submenu for the right-clicked sender.

        Each entry is one capture WAV where this callsign was the
        transmitting station (per ``sender_in_message``).  Clicking an
        entry launches :file:`analyze_msk144.py` on that WAV in a
        separate process so the GUI stays responsive.
        """
        item = self._callsign_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        cs_item = self._callsign_table.item(row, 0)
        if cs_item is None:
            return
        callsign = cs_item.text()

        wavs = wavs_for_sender(
            callsign, self._all_decodes_for_lookup, self._CAPTURES_DIR,
        )
        menu = QtWidgets.QMenu(self)
        title = menu.addAction(f"{callsign} — captures (sender)")
        title.setEnabled(False)
        menu.addSeparator()
        if not wavs:
            none_action = menu.addAction("(no capture WAVs found)")
            none_action.setEnabled(False)
        else:
            # Cap the menu — anything more than ~30 entries is unwieldy
            # and the table-row sender is the same across all of them.
            for d, wav in wavs[:30]:
                ts = d['ts'].strftime('%Y-%m-%d %H:%M:%S')
                snr = d.get('snr')
                snr_str = f"{snr:+d} dB" if snr is not None else "—"
                label = f"{ts}   {snr_str}   {d['message']}"
                act = menu.addAction(label)
                act.setData(str(wav))
            if len(wavs) > 30:
                more = menu.addAction(f"… and {len(wavs) - 30} more (use the "
                                       "filter or a tighter window)")
                more.setEnabled(False)

        chosen = menu.exec_(self._callsign_table.viewport().mapToGlobal(pos))
        if chosen is not None and chosen.data():
            self._launch_analyze_msk144(chosen.data())

    def _launch_analyze_msk144(self, wav_path: str) -> None:
        """Spawn analyze_msk144.py in a subprocess so the GUI doesn't block."""
        import subprocess
        if not self._ANALYZE_MSK144.is_file():
            QtWidgets.QMessageBox.warning(
                self, "Tool missing",
                f"analyze_msk144.py not found at {self._ANALYZE_MSK144}",
            )
            return
        try:
            subprocess.Popen([sys.executable, str(self._ANALYZE_MSK144), wav_path])
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Launch failed",
                f"Could not launch analyze_msk144.py:\n{exc}",
            )
            return
        self.statusBar().showMessage(
            f"Launched analyze_msk144.py {Path(wav_path).name}", 5000,
        )

    def _save_all(self):
        """Auto-save all current figures to timestamped PNGs."""
        date_str = self._date_edit.text().strip() or datetime.now().strftime('%Y%m%d')
        stem = Path(__file__).parent / f"compare_{date_str}"
        names = ['timeline', 'detect', 'scatter',
                 'detect_byclass', 'scatter_byclass']
        saved = []
        for fig, name in zip(self._figs, names):
            path = str(stem) + f'_{name}.png'
            fig.savefig(path, dpi=150)
            saved.append(path)
        QtWidgets.QMessageBox.information(
            self, "Saved", "Saved:\n" + "\n".join(saved))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date',   default=None, help='UTC date YYYYMMDD')
    ap.add_argument('--no-gui', action='store_true',
                    help='CLI mode: save PNGs and exit (requires --plot)')
    ap.add_argument('--plot',   metavar='FILE', default=None,
                    help='(--no-gui) base path for detection-rate PNG')
    ap.add_argument('--window', type=float, default=1.0,
                    help='Match window in seconds (default: 1)')
    ap.add_argument('--since',  default=None, metavar='YYYYMMDD')
    ap.add_argument('--utc',    default=None, metavar='HH:MM-HH:MM')
    ap.add_argument('--launches', metavar='FILE', default=str(MAP144_LAUNCHES))
    ap.add_argument('--no-launches', action='store_true')
    ap.add_argument('--my-grid', default='FN42EV',
                    help="Operator's Maidenhead grid (4 or 6 char) for "
                         "great-circle range column and the local-vs-distant "
                         "classifier split (default FN42EV — WA1HCO)")
    ap.add_argument('--local-max-km', type=float, default=300.0,
                    help='High-density callsign within this range = STRONG_LOCAL; '
                         'beyond = CONT_PROP (default 300)')
    args = ap.parse_args()

    if args.no_gui:
        # ── Headless CLI path ─────────────────────────────────────────────────
        wsjtx_all  = parse_wsjtx(WSJTX_ALL)
        map144_all = parse_map144(MAP144_LOG)
        map144_all = [d for d in map144_all if not is_test_message(d['message'])]

        if args.date:
            day_start = datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
            day_end   = day_start + timedelta(days=1)
            wsjtx_all  = [d for d in wsjtx_all  if day_start <= d['ts'] < day_end]
            map144_all = [d for d in map144_all if day_start <= d['ts'] < day_end]

        if args.since:
            since_dt = datetime.strptime(args.since, "%Y%m%d").replace(tzinfo=timezone.utc)
            wsjtx_all  = [d for d in wsjtx_all  if d['ts'] >= since_dt]
            map144_all = [d for d in map144_all if d['ts'] >= since_dt]

        if args.utc:
            t0s, t1s = args.utc.split('-')
            t0h, t0m = int(t0s.split(':')[0]), int(t0s.split(':')[1])
            t1h, t1m = int(t1s.split(':')[0]), int(t1s.split(':')[1])
            lo_min = t0h * 60 + t0m
            hi_min = t1h * 60 + t1m
            wsjtx_all  = [d for d in wsjtx_all  if lo_min <= d['ts'].hour*60+d['ts'].minute < hi_min]
            map144_all = [d for d in map144_all if lo_min <= d['ts'].hour*60+d['ts'].minute < hi_min]

        launches: list[dict] = []
        if not args.no_launches:
            launches = parse_launches(Path(args.launches))

        matched, wsjtx_only, map144_only = match_decodes(wsjtx_all, map144_all, args.window)
        wsjtx_tags, tag_counts = _build_wsjtx_tags(wsjtx_only, launches) if launches else ({}, {})

        date_label = args.date or ''
        print(_format_summary(matched, wsjtx_only, map144_only, wsjtx_all, map144_all,
                               wsjtx_tags, tag_counts))

        if args.plot:
            plot_detection_rate(matched, wsjtx_only, args.plot, date_label)
            p = Path(args.plot)
            plot_snr_scatter(matched, wsjtx_only, map144_only,
                             str(p.with_stem(p.stem + '_scatter')),
                             date_label, wsjtx_tags or None)
        return

    # ── GUI path ──────────────────────────────────────────────────────────────
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Ctrl-C from the terminal: Qt's event loop ignores Python signals while it
    # is running C++ code, so we restore the default SIGINT handler (which raises
    # KeyboardInterrupt → exit) and tick a no-op QTimer every 200 ms to give
    # control back to the Python interpreter often enough for the signal to fire.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _sigint_timer = QtCore.QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    win = CompareGUI(initial_date=args.date)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
