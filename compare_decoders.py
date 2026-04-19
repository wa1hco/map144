#!/usr/bin/env python3
"""Compare MAP144 vs WSJT-X MSK144 decode logs.

Usage:
    python compare_decoders.py [--date YYYYMMDD] [--hours N] [--window SEC]

    --date    UTC date to analyse (default: today)
    --hours   how many hours of that day to include (default: all 24)
    --window  seconds within which two decodes are considered the same ping (default: 2)
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
WSJTX_ALL     = Path.home() / ".local/share/WSJT-X - Local/ALL.TXT"
MAP144_LOG    = Path(__file__).parent / "MSK144/detections/decodes.jsonl"
MAP144_LAUNCHES = Path(__file__).parent / "MSK144/detections/launches.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def period_start(ts: datetime) -> datetime:
    """Floor a UTC datetime to the nearest 15-second MSK144 period boundary."""
    epoch = int(ts.timestamp() // 15) * 15
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_wsjtx(path: Path):
    """Parse WSJT-X ALL.TXT MSK144 lines into list of dicts."""
    # Format: 260415_125018    50.260 Rx MSK144   0  2.6 1612 VA3TO N4JP R+03
    pat = re.compile(
        r'^(\d{6}_\d{6})\s+'          # timestamp  YYMMDD_HHMMSS
        r'([\d.]+)\s+Rx\s+MSK144\s+'  # freq MHz
        r'([+-]?\d+)\s+'               # SNR dB
        r'([\d.]+)\s+'                 # dt
        r'(\d+)\s+'                    # audio freq Hz
        r'(.+)$'                       # message
    )
    results = []
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue
            ts_str, freq, snr, dt_s, af, msg = m.groups()
            # ts_str: YYMMDD_HHMMSS  → parse as 20YY...
            # WSJT-X timestamps the START of the Rx period; dt is seconds into
            # the period where the ping was detected.  Add dt to get ping time.
            try:
                ts_period = datetime.strptime("20" + ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts = period_start(ts_period + timedelta(seconds=float(dt_s)))
            results.append({
                'ts': ts,
                'ts_period': ts_period,
                'freq_mhz': float(freq),
                'snr': int(snr),
                'dt': float(dt_s),
                'af_hz': int(af),
                'message': msg.strip(),
                'source': 'wsjtx',
            })
    return results


def parse_map144(path: Path):
    """Parse MAP144 decodes.jsonl into list of dicts."""
    # {"timestamp": "2026-04-15_11:50:15.6", "period_ts": "2026-04-15_11:50:15",
    #  "t_sec": ..., "radio_khz": 50246, "message": "CQ K8TS EN82", ...}
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
            # Prefer period_ts if present (new format); fall back to computing it
            raw_ts_str = d.get('period_ts') or d.get('timestamp', '')
            fmt = "%Y-%m-%d_%H:%M:%S"
            try:
                ts = datetime.strptime(raw_ts_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    ts = datetime.strptime(raw_ts_str, "%Y-%m-%d_%H:%M:%S.%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            # Ensure we always store the period-start, not the raw ping time
            ts = period_start(ts)
            results.append({
                'ts': ts,
                'freq_mhz': d.get('radio_khz', 0) / 1000.0,
                'snr':     int(d.get('jt9_snr_db', 0) or 0),
                'est_snr': d.get('est_snr_db'),   # MAP144 pre-burst estimate; None if old log
                'af_hz': 0,
                'message': d.get('message', '').strip(),
                'source': 'map144',
            })
    return results


_TEST_MSG_RE = re.compile(r'^A[NP]\d{5}[PN]\d{2}(\d{3})?$')  # MAP144 test: 10-char AN05099P04 or 13-char AN09062P03128

def is_test_message(msg: str) -> bool:
    return bool(_TEST_MSG_RE.match(msg.strip()))

def normalise_msg(msg: str) -> str:
    """Normalise a message for matching: uppercase, collapse spaces."""
    return ' '.join(msg.upper().split())


def parse_launches(path: Path) -> list[dict]:
    """Parse MAP144 launches.jsonl.

    Each entry:  {"timestamp": "2026-04-19_12:36:10.3", "t_sec": 10.3,
                  "radio_khz": 50277, "outcome": "no_decode", ...}

    Returns a list with an added 'period' field (datetime at 15-s boundary).
    """
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
            # period_start: wall time minus t_sec offset gives approximate period start
            t_sec = d.get('t_sec', 0.0) or 0.0
            approx_period_epoch = (wall.timestamp() - t_sec) // 15 * 15
            d['period'] = datetime.fromtimestamp(approx_period_epoch, tz=timezone.utc)
            launches.append(d)
    return launches


def _find_launch_for_wsjtx(wsjtx_decode: dict, launches: list[dict],
                           freq_tol_khz: float = 2.5) -> dict | None:
    """Return the best MAP144 launch matching a WSJT-X decode, or None.

    Matches on:
      - Same 15-second period (exact period_start equality)
      - radio_khz within freq_tol_khz of the WSJT-X signal frequency
        (WSJT-X signal freq = dial_khz + af_hz/1000)

    Returns the closest-frequency launch, preferring 'decoded' > 'no_decode'
    > 'timeout' outcomes.
    """
    # WSJT-X actual signal frequency: dial (MHz → kHz) + audio offset
    sig_khz = wsjtx_decode['freq_mhz'] * 1000.0 + wsjtx_decode.get('af_hz', 0) / 1000.0
    period  = wsjtx_decode['ts']   # already period_start from parse_wsjtx

    outcome_rank = {'decoded': 0, 'no_decode': 1, 'timeout': 2, 'error': 3}

    best      = None
    best_rank = 99
    best_df   = float('inf')

    for launch in launches:
        if launch['period'] != period:
            continue
        df = abs(launch.get('radio_khz', 0) - sig_khz)
        if df > freq_tol_khz:
            continue
        rank = outcome_rank.get(launch.get('outcome', ''), 9)
        if (rank < best_rank) or (rank == best_rank and df < best_df):
            best      = launch
            best_rank = rank
            best_df   = df

    return best


def _dedup(decodes):
    """Deduplicate decode list to one entry per (normalised_message, period_start).

    A strong station calling CQ produces 3–6 identical decodes per 15-second
    window in WSJT-X and 1–2 in MAP144.  Counting each repetition separately
    inflates the WSJT-X-only count and understates MAP144 coverage.  The
    meaningful question is: did the decoder hear this station in this window?
    Keep the entry with the best (lowest) SNR to represent the hardest copy.
    """
    best: dict[tuple, dict] = {}
    for d in decodes:
        key = (normalise_msg(d['message']), d['ts'])
        prev = best.get(key)
        if prev is None or d.get('snr', 0) < prev.get('snr', 0):
            best[key] = d
    return list(best.values())


def match_decodes(wsjtx, map144, window_sec=2):
    """
    Match decodes from both sources that refer to the same ping.
    Two decodes match if:
      - normalised messages are identical
      - timestamps are within window_sec of each other

    Both lists are first deduplicated to one entry per
    (message, 15-second period) so repeated decodes of the same CQ from
    a strong station don't inflate the WSJT-X-only or MAP144-only counts.

    Returns (matched, wsjtx_only, map144_only).
    """
    wsjtx  = _dedup(wsjtx)
    map144 = _dedup(map144)

    # Index map144 by normalised message for fast lookup
    m144_by_msg = defaultdict(list)
    for d in map144:
        m144_by_msg[normalise_msg(d['message'])].append(d)

    matched      = []   # list of (wsjtx_decode, map144_decode)
    wsjtx_only   = []
    used_ids     = set()   # id(map144_decode) for already-matched items

    for w in wsjtx:
        key = normalise_msg(w['message'])
        candidates = m144_by_msg.get(key, [])
        best = None
        best_dt = None
        for m in candidates:
            if id(m) in used_ids:
                continue
            dt = abs((w['ts'] - m['ts']).total_seconds())
            if dt <= window_sec:
                if best_dt is None or dt < best_dt:
                    best = m
                    best_dt = dt
        if best is not None:
            used_ids.add(id(best))
            matched.append((w, best))
        else:
            wsjtx_only.append(w)

    map144_only = [d for d in map144 if id(d) not in used_ids]

    return matched, wsjtx_only, map144_only


def snr_histogram(decodes, label, bins=None):
    if not decodes:
        return
    snrs = [d['snr'] for d in decodes]
    lo, hi = min(snrs), max(snrs)
    if bins is None:
        bins = range(max(-30, lo - 1), min(30, hi + 2), 2)
    counts = defaultdict(int)
    for s in snrs:
        bucket = (s // 2) * 2
        counts[bucket] += 1
    print(f"\n  SNR histogram for {label} ({len(snrs)} decodes, "
          f"min={lo} max={hi} median={sorted(snrs)[len(snrs)//2]}):")
    for b in sorted(counts):
        bar = '█' * min(counts[b], 60)
        print(f"    {b:+4d} dB : {bar} {counts[b]}")


def unique_calls(decodes, word_index):
    """Extract unique callsigns from word at position word_index (0-based) of message."""
    calls = set()
    for d in decodes:
        words = d['message'].split()
        if len(words) > word_index:
            c = words[word_index]
            if re.match(r'^[A-Z0-9]{2,}(/[A-Z0-9]+)?$', c) and c not in ('CQ', 'DE', 'QRZ', 'RR73', '73', 'RRR'):
                calls.add(c)
    return calls


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_detection_rate(matched, wsjtx_only, out_path, date_label=''):
    """
    Detection-rate curve: MAP144 decode probability vs SNR, using WSJT-X SNR
    as ground truth.  Background bars show the WSJT-X decode count per bin.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    BIN = 2   # dB per bin

    # Build per-SNR-bin counts using WSJT-X SNR as reference
    all_w_snrs   = [w['snr'] for w, _ in matched] + [w['snr'] for w in wsjtx_only]
    match_w_snrs = [w['snr'] for w, _ in matched]

    if not all_w_snrs:
        print("No WSJT-X data to plot.")
        return

    lo = (min(all_w_snrs) // BIN) * BIN
    hi = (max(all_w_snrs) // BIN) * BIN + BIN
    bins = list(range(lo, hi + BIN, BIN))

    total_counts = defaultdict(int)
    match_counts = defaultdict(int)
    for s in all_w_snrs:
        total_counts[(s // BIN) * BIN] += 1
    for s in match_w_snrs:
        match_counts[(s // BIN) * BIN] += 1

    xs      = [b for b in bins if total_counts[b] > 0]
    totals  = [total_counts[b] for b in xs]
    matched_ = [match_counts[b] for b in xs]
    rates   = [m / t * 100 for m, t in zip(matched_, totals)]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#f8f8f8')
    ax1.set_facecolor('#f8f8f8')

    # Background bars: WSJT-X decode count
    ax2 = ax1.twinx()
    ax2.bar(xs, totals, width=BIN * 0.8, color='steelblue', alpha=0.25,
            label='WSJT-X decodes (count)')
    ax2.set_ylabel('WSJT-X decode count per bin', color='steelblue', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(0, max(totals) * 3.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    # Detection-rate line
    ax1.plot(xs, rates, color='#d62728', linewidth=2.5, marker='o',
             markersize=7, zorder=5, label='MAP144 detection rate')
    # Annotate points that have enough data (≥3 WSJT-X decodes)
    for x, r, t in zip(xs, rates, totals):
        if t >= 3:
            ax1.annotate(f'{r:.0f}%', (x, r),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8.5, color='#d62728')

    # 50 % reference line
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax1.text(xs[-1] + 0.3, 51, '50 %', va='bottom', ha='right',
             fontsize=8, color='gray')

    ax1.set_xlabel('SNR reported by WSJT-X  (dB)', fontsize=12)
    ax1.set_ylabel('MAP144 detection rate  (%)', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 115)
    ax1.set_xlim(lo - BIN, hi + BIN)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(BIN))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{int(v):+d}'))
    ax1.grid(axis='x', linestyle=':', alpha=0.4)

    title = 'MAP144 vs WSJT-X  —  MSK144 detection rate by SNR'
    if date_label:
        title += f'\n{date_label}'
    ax1.set_title(title, fontsize=13, pad=10)

    # Combined legend
    handles = (
        [plt.Line2D([0], [0], color='#d62728', lw=2.5, marker='o', ms=7,
                    label='MAP144 detection rate')]
        + [plt.Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.4,
                          label='WSJT-X decode count')]
    )
    ax1.legend(handles=handles, loc='upper left', fontsize=10,
               framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


def plot_snr_scatter(matched, wsjtx_only, map144_only, out_path, date_label=''):
    """
    2-D SNR scatter plot.
      • Both decoded  → mark at (wsjtx_snr, map144_snr)
      • WSJT-X only   → mark projected onto Y = sentinel (bottom rail)
      • MAP144 only   → mark projected onto X = sentinel (left rail)
    The rail marks form natural histograms along each axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    def _dot_stack(values, step=0.30, max_height=4.5):
        """Return perpendicular offsets for a stacked dot plot.

        Marks at the same integer-rounded SNR bin are stacked upward from
        the rail.  Bars are capped at max_height dB so they stay below the
        main scatter data regardless of how many marks fall in one bin.
        """
        bin_idx: dict[int, int] = {}
        offsets = np.zeros(len(values))
        for i, v in enumerate(values):
            b = round(float(v))
            idx = bin_idx.get(b, 0)
            bin_idx[b] = idx + 1
            offsets[i] = min(idx * step, max_height)
        return offsets

    # Collect SNR values
    mx   = [w['snr'] for w, _ in matched]          # WSJT-X SNR for matched
    my   = [m['snr'] for _, m in matched]          # MAP144 jt9 SNR for matched
    # MAP144 pre-burst estimate — only where logged (new runs); paired with WSJT-X SNR
    est_pairs = [(w['snr'], m['est_snr'])
                 for w, m in matched if m.get('est_snr') is not None]
    ex = [p[0] for p in est_pairs]
    ey = [p[1] for p in est_pairs]

    wx = [w['snr'] for w in wsjtx_only]   # WSJT-X only
    py = [m['snr'] for m in map144_only]  # MAP144 only

    all_x = mx + wx
    all_y = my + py + ey
    if not all_x and not all_y:
        print("No data to scatter-plot.")
        return

    data_lo = min(min(all_x, default=0), min(all_y, default=0))
    data_hi = max(max(all_x, default=0), max(all_y, default=0))

    SENTINEL  = data_lo - 7   # rail position — gap keeps capped bars below data

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#f8f8f8')
    ax.set_facecolor('#f8f8f8')

    # Both decoded — jt9 SNR
    if mx:
        ax.scatter(mx, my, s=50, color='#2ca02c', alpha=0.75, zorder=4,
                   label=f'Both: MAP144 jt9 SNR  (n={len(mx)})')

    # Both decoded — MAP144 pre-burst est_snr (only if present in log)
    if ex:
        ax.scatter(ex, ey, s=50, color='#ff7f0e', alpha=0.75, zorder=4,
                   marker='s', label=f'Both: MAP144 est_snr  (n={len(ex)})')

    # WSJT-X only — bottom rail, stacked upward
    if wx:
        offs = _dot_stack(wx)
        ax.scatter(wx, SENTINEL + offs, s=40,
                   color='steelblue', alpha=0.70, marker='^', zorder=3,
                   label=f'WSJT-X only   (n={len(wx)})')

    # MAP144 only — left rail, stacked rightward
    if py:
        offs = _dot_stack(py)
        ax.scatter(SENTINEL + offs, py, s=40,
                   color='#d62728', alpha=0.70, marker='>', zorder=3,
                   label=f'MAP144 only   (n={len(py)})')

    # Dashed rail lines at the sentinel value
    ax.axhline(SENTINEL, color='steelblue', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(SENTINEL, color='#d62728',   linewidth=0.8, linestyle='--', alpha=0.5)

    # y = x reference line (perfect agreement)
    lim_lo = SENTINEL - 1
    lim_hi = data_hi + 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color='gray',
            linestyle=':', linewidth=1, alpha=0.6, label='y = x  (perfect agreement)')

    # Axis ticks: real data range only, then a gap to the rail
    tick_lo = (data_lo // 2) * 2
    tick_hi = (data_hi // 2) * 2 + 2
    real_ticks = list(range(tick_lo, tick_hi + 1, 2))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator(real_ticks + [SENTINEL]))
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _, s=SENTINEL: ('missed' if abs(v - s) < 0.5
                                      else f'{int(v):+d}')))

    ax.set_xlim(SENTINEL - 1.5, lim_hi)
    ax.set_ylim(SENTINEL - 1.5, lim_hi)
    ax.set_xlabel('WSJT-X SNR  (dB)\n[coherent matched filter]',
                  fontsize=11)
    ax.set_ylabel('MAP144 SNR  (dB)\n[non-coherent energy — ~3–4 dB lower than WSJT-X]',
                  fontsize=11)
    ax.grid(linestyle=':', alpha=0.35)

    title = 'MAP144 vs WSJT-X  —  per-ping SNR comparison'
    if date_label:
        title += f'\n{date_label}'
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=10, framealpha=0.85, loc='upper left')

    fig.tight_layout()
    # Save alongside the first plot, inserting _scatter before the suffix
    p = Path(out_path)
    scatter_path = p.with_stem(p.stem + '_scatter')
    fig.savefig(scatter_path, dpi=150)
    print(f"Scatter plot saved → {scatter_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date',   default=None,
                    help='UTC date YYYYMMDD to analyse (default: all data)')
    ap.add_argument('--since',  default=None, metavar='YYYYMMDD',
                    help='Include data from this UTC date onwards')
    ap.add_argument('--hours',  type=float, default=24.0,
                    help='Hours of that date to include (default: 24)')
    ap.add_argument('--utc',    default=None, metavar='HH:MM-HH:MM',
                    help='UTC time-of-day window e.g. 09:00-13:00 (default: all day)')
    ap.add_argument('--window', type=float, default=1.0,
                    help='Match window in seconds (default: 1; both sources now use period-start timestamps)')
    ap.add_argument('--plot',   metavar='FILE', default=None,
                    help='Save detection-rate plot to FILE (e.g. detect.png)')
    ap.add_argument('--launches', metavar='FILE', default=str(MAP144_LAUNCHES),
                    help='MAP144 launches.jsonl (default: MSK144/detections/launches.jsonl)')
    ap.add_argument('--no-launches', action='store_true',
                    help='Skip launch cross-reference diagnostic')
    args = ap.parse_args()

    # Load
    print(f"Loading WSJT-X:  {WSJTX_ALL}")
    wsjtx_all  = parse_wsjtx(WSJTX_ALL)
    print(f"Loading MAP144:  {MAP144_LOG}")
    map144_all = parse_map144(MAP144_LOG)

    # Drop MAP144 development test messages (AN.../AP... 13-char synthetic)
    n_test = sum(1 for d in map144_all if is_test_message(d['message']))
    map144_all = [d for d in map144_all if not is_test_message(d['message'])]
    if n_test:
        print(f"  (filtered {n_test} MAP144 test messages)")

    # Filter by date/time window
    if args.date:
        try:
            day_start = datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Bad date: {args.date}"); return
        day_end = day_start + timedelta(hours=args.hours)
        wsjtx_all  = [d for d in wsjtx_all  if day_start <= d['ts'] < day_end]
        map144_all = [d for d in map144_all if day_start <= d['ts'] < day_end]

    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Bad --since date: {args.since}"); return
        wsjtx_all  = [d for d in wsjtx_all  if d['ts'] >= since_dt]
        map144_all = [d for d in map144_all if d['ts'] >= since_dt]

    # Filter by UTC time-of-day window (applies across all dates)
    if args.utc:
        try:
            t0_str, t1_str = args.utc.split('-')
            t0_h, t0_m = int(t0_str.split(':')[0]), int(t0_str.split(':')[1])
            t1_h, t1_m = int(t1_str.split(':')[0]), int(t1_str.split(':')[1])
        except Exception:
            print(f"Bad --utc value '{args.utc}', expected HH:MM-HH:MM"); return
        def _in_window(ts):
            mins = ts.hour * 60 + ts.minute
            lo   = t0_h * 60 + t0_m
            hi   = t1_h * 60 + t1_m
            return lo <= mins < hi
        before_w = len(wsjtx_all) + len(map144_all)
        wsjtx_all  = [d for d in wsjtx_all  if _in_window(d['ts'])]
        map144_all = [d for d in map144_all if _in_window(d['ts'])]
        dropped = before_w - len(wsjtx_all) - len(map144_all)
        print(f"  (UTC window {args.utc}: dropped {dropped} outside-window decodes)")

    # Load launches for diagnostic
    launches_all: list[dict] = []
    if not args.no_launches:
        launches_path = Path(args.launches)
        if launches_path.exists():
            launches_all = parse_launches(launches_path)
            print(f"  MAP144 launches: {len(launches_all)} entries")
        else:
            print(f"  MAP144 launches: not found ({launches_path})")

    if not wsjtx_all and not map144_all:
        print("No data in the specified time window.")
        return

    # Frequency diagnostic — helps spot if the two decoders are on different bands
    if wsjtx_all:
        wf = sorted(set(d['freq_mhz'] for d in wsjtx_all))
        print(f"  WSJT-X freqs (MHz): {wf[:8]}")
    if map144_all:
        mf = sorted(set(round(d['freq_mhz'], 3) for d in map144_all))
        print(f"  MAP144 freqs (MHz): {mf[:8]}")

    ts_range = lambda lst: (
        (min(d['ts'] for d in lst).strftime('%H:%M'),
         max(d['ts'] for d in lst).strftime('%H:%M'))
        if lst else ('—', '—')
    )
    w_lo, w_hi = ts_range(wsjtx_all)
    m_lo, m_hi = ts_range(map144_all)

    print(f"\nWSJT-X:  {len(wsjtx_all):4d} MSK144 decodes  ({w_lo}–{w_hi} UTC)")
    print(f"MAP144:  {len(map144_all):4d} MSK144 decodes  ({m_lo}–{m_hi} UTC)")

    matched, wsjtx_only, map144_only = match_decodes(wsjtx_all, map144_all, args.window)

    print(f"\nMatch window: ±{args.window} s, same message")
    print(f"  Both decoded:      {len(matched):4d}")
    print(f"  WSJT-X only:       {len(wsjtx_only):4d}")
    print(f"  MAP144 only:       {len(map144_only):4d}")

    total_unique = len(matched) + len(wsjtx_only) + len(map144_only)
    if total_unique:
        print(f"\n  WSJT-X coverage:   {(len(matched)+len(wsjtx_only))/total_unique*100:.1f}%  "
              f"of unique pings")
        print(f"  MAP144 coverage:   {(len(matched)+len(map144_only))/total_unique*100:.1f}%  "
              f"of unique pings")

    # SNR comparison on matched pairs
    if matched:
        jt9_diffs = [w['snr'] - m['snr'] for w, m in matched]
        print(f"\nSNR bias on matched decodes (WSJT-X − MAP144 jt9):")
        print(f"  mean={sum(jt9_diffs)/len(jt9_diffs):+.1f} dB  "
              f"min={min(jt9_diffs):+d}  max={max(jt9_diffs):+d}")

        est_pairs = [(w['snr'], m['est_snr']) for w, m in matched
                     if m.get('est_snr') is not None]
        if est_pairs:
            est_diffs = [wx - ey for wx, ey in est_pairs]
            print(f"  WSJT-X − MAP144 est_snr (pre-burst, {len(est_pairs)} pairs):")
            print(f"  mean={sum(est_diffs)/len(est_diffs):+.1f} dB  "
                  f"min={min(est_diffs):+d}  max={max(est_diffs):+d}")
            if abs(sum(jt9_diffs)/len(jt9_diffs) - sum(est_diffs)/len(est_diffs)) > 1.0:
                print(f"  → jt9 and est_snr biases differ: likely jt9 short-clip effect")
            else:
                print(f"  → jt9 and est_snr biases agree: likely signal-chain SNR loss")

    # SNR histograms
    snr_histogram([w for w, _ in matched] + wsjtx_only, "WSJT-X all decoded")
    snr_histogram([m for _, m in matched] + map144_only, "MAP144 all decoded")

    # Unique callsigns heard by each
    def all_calls(lst):
        s = set()
        for d in lst:
            s |= unique_calls([d], 0)
            s |= unique_calls([d], 1)
        return s

    w_calls  = all_calls([w for w, _ in matched] + wsjtx_only)
    m_calls  = all_calls([m for _, m in matched] + map144_only)
    both_c   = w_calls & m_calls
    w_only_c = w_calls - m_calls
    m_only_c = m_calls - w_calls

    print(f"\nUnique callsigns heard:")
    print(f"  Both:        {len(both_c)}")
    print(f"  WSJT-X only: {len(w_only_c)}  {sorted(w_only_c)[:10]}")
    print(f"  MAP144 only: {len(m_only_c)}  {sorted(m_only_c)[:10]}")

    # Show WSJT-X-only decodes (MAP144 missed) with launch cross-reference
    if wsjtx_only:
        # Cross-reference each WSJT-X-only ping against MAP144 launches
        n_detected_nodec = 0
        n_detected_to    = 0
        n_not_detected   = 0
        rows = []
        for d in sorted(wsjtx_only, key=lambda x: x['snr']):
            if launches_all:
                launch = _find_launch_for_wsjtx(d, launches_all)
                if launch is None:
                    tag = 'NOT_DETECTED'
                    n_not_detected += 1
                elif launch.get('outcome') in ('no_decode', 'error'):
                    tag = 'NO_DECODE'
                    n_detected_nodec += 1
                elif launch.get('outcome') == 'timeout':
                    tag = 'TIMEOUT'
                    n_detected_to += 1
                else:
                    tag = launch.get('outcome', '?').upper()
            else:
                tag = ''
            rows.append((d, tag))

        print(f"\nWSJT-X decoded, MAP144 missed ({len(wsjtx_only)}):")
        if launches_all:
            print(f"  NOT_DETECTED={n_not_detected}  NO_DECODE={n_detected_nodec}"
                  + (f"  TIMEOUT={n_detected_to}" if n_detected_to else ""))
        for d, tag in rows[:30]:
            sig_khz = d['freq_mhz'] * 1000.0 + d.get('af_hz', 0) / 1000.0
            tag_str = f'  [{tag}]' if tag else ''
            print(f"  {d['ts'].strftime('%H:%M:%S')}  SNR{d['snr']:+3d}  "
                  f"{sig_khz:8.1f} kHz  {d['message']}{tag_str}")
        if len(wsjtx_only) > 30:
            print(f"  ... and {len(wsjtx_only)-30} more")

    # Show MAP144-only decodes (WSJT-X missed)
    if map144_only:
        print(f"\nMAP144 decoded, WSJT-X missed ({len(map144_only)}):")
        for d in sorted(map144_only, key=lambda x: x['snr'])[:20]:
            print(f"  {d['ts'].strftime('%H:%M:%S')}  SNR{d['snr']:+3d}  {d['message']}")
        if len(map144_only) > 20:
            print(f"  ... and {len(map144_only)-20} more")

    if args.plot:
        lbl_parts = []
        if args.date:
            lbl_parts.append(args.date[:4] + '-' + args.date[4:6] + '-' + args.date[6:])
        elif args.since:
            s = args.since
            lbl_parts.append(f'since {s[:4]}-{s[4:6]}-{s[6:]}')
        if args.utc:
            lbl_parts.append(f'{args.utc} UTC')
        lbl = '  '.join(lbl_parts)
        plot_detection_rate(matched, wsjtx_only, args.plot, date_label=lbl)
        plot_snr_scatter(matched, wsjtx_only, map144_only, args.plot, date_label=lbl)


if __name__ == '__main__':
    main()
