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
from collections import defaultdict
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

def _calls_in_message(msg: str) -> list[str]:
    return [w for w in msg.split()
            if _CALL_RE.match(w) and w not in _SKIP_WORDS and not _GRID_RE.match(w)]


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

        # ── Layout ────────────────────────────────────────────────────────────
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(ctrl)
        vbox.addWidget(self._summary)
        vbox.addWidget(self._tabs, stretch=1)
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

        # Close old figures to release memory
        for f in self._figs:
            f.clf()
        self._figs = [timeline_fig, detrate_fig, scatter_fig]

        # ── Populate tabs ─────────────────────────────────────────────────────
        self._tabs.clear()
        stem = f"compare_{date_str}" if date_str else "compare"
        self._tabs.addTab(_PlotPane(timeline_fig, f"{stem}_timeline"), "Timeline")
        self._tabs.addTab(_PlotPane(detrate_fig,  f"{stem}_detect"),   "Detection Rate")
        self._tabs.addTab(_PlotPane(scatter_fig,  f"{stem}_scatter"),  "SNR Scatter")

        self._save_all_btn.setEnabled(True)
        n_total = len(matched) + len(wsjtx_only) + len(map144_only)
        self.statusBar().showMessage(
            f"Loaded — {n_total} unique pings  |  "
            f"WSJT-X: {len(wsjtx_all)}  MAP144: {len(map144_all)}  "
            f"matched: {len(matched)}")

    def _save_all(self):
        """Auto-save all three figures to timestamped PNGs."""
        date_str = self._date_edit.text().strip() or datetime.now().strftime('%Y%m%d')
        stem = Path(__file__).parent / f"compare_{date_str}"
        names = ['timeline', 'detect', 'scatter']
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
