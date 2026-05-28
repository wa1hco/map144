#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Compare TFMF candidate log against sq_det's launches/decodes for an
overnight-style live run.

Aligns events by period and freq:
    - TFMF candidate: (period_idx, t_s_in_period, freq_hz, snr_db, pol)
    - sq_det launch : (timestamp, t_sec, radio_khz, ch_signed, outcome, message)
    - decoded launch: launch with matching entry in decodes.jsonl

Matching tolerance: ±1 s in time AND ±1 kHz in audio freq.  These are the
peak-pick footprint scales (7.5 ms × 1 kHz) loosened to absorb sq_det's
channel quantisation (channels are 1 kHz wide).

Cross-tab printed:
                          TFMF candidate nearby   No TFMF candidate
    sq_det launch + decode      A                       B
    sq_det launch, no decode    C                       D
    no sq_det launch (TFMF-only) — counted separately per period

Plus per-hour rates, freq distribution of disagreements, and (optional) a
scatter plot of both detectors' events in time × freq for visual review.

Usage:
    python tools/compare_tfmf_vs_sqdet.py
    python tools/compare_tfmf_vs_sqdet.py --plot /tmp/tfmf_vs_sqdet.png
    python tools/compare_tfmf_vs_sqdet.py --tol-time 1.0 --tol-freq-hz 1000
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.detection_compare._filters import is_test_message  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

TS_FMT = "%Y-%m-%d_%H:%M:%S.%f"
TS_FMT_NOFRAC = "%Y-%m-%d_%H:%M:%S"
PERIOD_S = 15.0
PAN_CENTER_MHZ = 50.259   # default Flex pan centre on 6 m (≠ calling freq)


def parse_ts(s: str) -> float | None:
    """Parse a log timestamp to epoch seconds (UTC)."""
    for fmt in (TS_FMT, TS_FMT_NOFRAC):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def epoch_to_period(e: float) -> int:
    return int(e // PERIOD_S)


# ── Loaders ────────────────────────────────────────────────────────────────


def load_tfmf(path: Path) -> tuple[dict[int, list], tuple[float, float]]:
    """Group TFMF candidates by period_idx.  Returns (cands_by_period,
    (min_epoch, max_epoch)) covering the log."""
    cands = defaultdict(list)
    epochs = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = int(r['period_idx'])
            cands[p].append({
                't_s': float(r['t_s_in_period']),
                'freq_hz': float(r['freq_hz']),
                'snr_db': float(r['snr_db']),
                'pol': r.get('pol', 'h'),
            })
            ts = parse_ts(r['timestamp'])
            if ts is not None:
                epochs.append(ts)
    if not epochs:
        return cands, (0.0, 0.0)
    return cands, (min(epochs), max(epochs))


def load_launches(path: Path, min_epoch: float, max_epoch: float,
                  pan_center_mhz: float) -> dict[int, list]:
    """Stream-read launches.jsonl, keep only those in the TFMF window and
    non-test-pollution messages.  Group by period_idx, project to audio
    freq for cross-detector matching."""
    out = defaultdict(list)
    n_total = n_test = n_window = 0
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            ts = parse_ts(r.get('timestamp', ''))
            if ts is None or ts < min_epoch - PERIOD_S or ts > max_epoch + PERIOD_S:
                continue
            n_window += 1
            msg = r.get('message') or ''
            if msg and is_test_message(msg):
                n_test += 1
                continue
            # Audio freq (relative to pan centre) — matches TFMF's freq_hz
            # convention.  Use radio_khz directly (not ch_signed × 1000)
            # because the channelizer grid is offset by channel_offset_hz
            # to align ch_0 with the calling freq, NOT with the pan centre.
            # On the Flex pan=50.259, calling=50.260: a signal at ch_signed=0
            # (= calling freq) is at radio_khz=50260, audio_hz = +1000.  The
            # naive ch_signed × 1000 mapping would put it at 0 and miss the
            # match against TFMF candidates that are correctly at +1 kHz.
            # Add sq_new_fc_offset_hz for sub-channel precision when present.
            radio_khz = r.get('radio_khz')
            if radio_khz is None:
                continue
            sub_hz = r.get('sq_new_fc_offset_hz_h') or 0
            audio_hz = (float(radio_khz) - pan_center_mhz * 1000.0) * 1000.0 + float(sub_hz)
            p = epoch_to_period(ts)
            out[p].append({
                'epoch': ts,
                't_sec': float(r.get('t_sec', 0.0)),
                'audio_hz': audio_hz,
                'outcome': r.get('outcome', 'launched'),
                'message': msg,
                'jt9_snr_db': r.get('jt9_snr_db'),
            })
    print(f'  launches.jsonl: {n_total} total; {n_window} in window; '
          f'{n_test} test-message lines filtered', file=sys.stderr)
    return out


def load_decodes(path: Path, min_epoch: float, max_epoch: float) -> set[tuple[int, str]]:
    """Return set of (period_idx, message) that successfully decoded."""
    out = set()
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(r.get('timestamp', ''))
            if ts is None or ts < min_epoch - PERIOD_S or ts > max_epoch + PERIOD_S:
                continue
            msg = r.get('message') or ''
            if not msg or is_test_message(msg):
                continue
            out.add((epoch_to_period(ts), msg))
    return out


# ── Cross-tab ─────────────────────────────────────────────────────────────


def matches(launch: dict, cands: list, tol_time_s: float, tol_freq_hz: float) -> bool:
    """True if any TFMF cand is within (±tol_time, ±tol_freq) of the launch."""
    t0 = launch['t_sec']
    f0 = launch['audio_hz']
    for c in cands:
        if abs(c['t_s'] - t0) <= tol_time_s and abs(c['freq_hz'] - f0) <= tol_freq_hz:
            return True
    return False


def analyze(args) -> None:
    repo = Path(__file__).resolve().parent.parent
    tfmf_path = repo / 'MSK144' / 'detections' / 'tfmf_candidates.jsonl'
    launches_path = repo / 'MSK144' / 'detections' / 'launches.jsonl'
    decodes_path = repo / 'MSK144' / 'detections' / 'decodes.jsonl'

    print('Loading TFMF candidates…', file=sys.stderr)
    tfmf, (e_min, e_max) = load_tfmf(tfmf_path)
    print(f'  {sum(len(v) for v in tfmf.values())} candidates across '
          f'{len(tfmf)} periods', file=sys.stderr)
    print(f'  Window: {datetime.fromtimestamp(e_min, tz=timezone.utc)} → '
          f'{datetime.fromtimestamp(e_max, tz=timezone.utc)} UTC', file=sys.stderr)

    print('Streaming launches…', file=sys.stderr)
    launches = load_launches(launches_path, e_min, e_max, args.pan_center_mhz)
    print(f'  {sum(len(v) for v in launches.values())} launches in window after filtering',
          file=sys.stderr)

    print('Streaming decodes…', file=sys.stderr)
    decoded = load_decodes(decodes_path, e_min, e_max)
    print(f'  {len(decoded)} decoded (period, message) pairs in window', file=sys.stderr)

    # ── Cross-tabulate per launch ─────────────────────────────────────────
    a = b = c = d = 0
    tfmf_only_counts = Counter()       # tfmf cands per period not near a launch
    disagreement_records = []          # for plot/freq histograms
    decoded_periods = {p for p, _ in decoded}

    for p, lst in launches.items():
        cands_in_p = tfmf.get(p, [])
        for launch in lst:
            # Decoded if (period, message) matches a decoded entry.
            is_decoded = (p, launch['message']) in decoded if launch['message'] else False
            has_tfmf = matches(launch, cands_in_p, args.tol_time, args.tol_freq_hz)
            if is_decoded and has_tfmf:
                a += 1
            elif is_decoded and not has_tfmf:
                b += 1
                disagreement_records.append(('sq_decoded_no_tfmf', launch, p))
            elif not is_decoded and has_tfmf:
                c += 1
            else:
                d += 1

    # ── TFMF-only count: candidates with no launch nearby ──────────────────
    tfmf_only_total = 0
    tfmf_only_periods = Counter()
    for p, lst in tfmf.items():
        launches_in_p = launches.get(p, [])
        for cand in lst:
            near_any = False
            for launch in launches_in_p:
                if (abs(cand['t_s'] - launch['t_sec']) <= args.tol_time and
                        abs(cand['freq_hz'] - launch['audio_hz']) <= args.tol_freq_hz):
                    near_any = True
                    break
            if not near_any:
                tfmf_only_total += 1
                tfmf_only_periods[p] += 1

    # ── Report ───────────────────────────────────────────────────────────
    print()
    print('═' * 72)
    print('TFMF candidates vs sq_det launches — overnight comparison')
    print('═' * 72)
    print(f'Window: {datetime.fromtimestamp(e_min, tz=timezone.utc):%Y-%m-%d %H:%M} → '
          f'{datetime.fromtimestamp(e_max, tz=timezone.utc):%Y-%m-%d %H:%M} UTC '
          f'({(e_max - e_min) / 3600:.1f} h)')
    print(f'Periods with any event:  {len(set(launches) | set(tfmf))}')
    print(f'Match tolerance:         ±{args.tol_time:.2f} s, ±{args.tol_freq_hz:.0f} Hz')
    print(f'Pan centre:              {args.pan_center_mhz} MHz')
    print()
    print('  Cross-tab (per sq_det launch):')
    print(f'                              TFMF nearby   no TFMF   total')
    print(f'    sq launched, decoded      {a:>10}   {b:>8}   {a+b:>5}')
    print(f'    sq launched, no decode    {c:>10}   {d:>8}   {c+d:>5}')
    print(f'    total                     {a+c:>10}   {b+d:>8}   {a+b+c+d:>5}')
    print()
    n_launches = a + b + c + d
    n_decoded = a + b
    print(f'  sq_det decode rate: {n_decoded}/{n_launches} = '
          f'{100*n_decoded/max(n_launches,1):.2f}%')
    print(f'  TFMF agreement on decoded launches:    '
          f'{a}/{max(n_decoded,1)} = {100*a/max(n_decoded,1):.1f}%')
    print(f'  TFMF caught a decoded launch sq missed:  N/A (TFMF doesn'"'"'t '
          f'launch decodes itself yet)')
    print()
    print(f'  TFMF candidates with NO sq_det launch nearby: {tfmf_only_total}')
    print(f'    Across {len(tfmf_only_periods)} periods (median '
          f'{sorted(tfmf_only_periods.values())[len(tfmf_only_periods)//2] if tfmf_only_periods else 0} per period)')
    print(f'    → TFMF "extra" detections sq_det missed; mix of real-but-marginal,')
    print(f'      FAs sq_det filtered, and signals at freqs sq_det doesn'"'"'t cover.')
    print()

    # ── Per-hour breakdown ────────────────────────────────────────────────
    hour_l = Counter(); hour_d = Counter(); hour_t = Counter()
    for p, lst in launches.items():
        h = datetime.fromtimestamp(p * PERIOD_S, tz=timezone.utc).strftime('%Y-%m-%d_%H')
        hour_l[h] += len(lst)
        hour_d[h] += sum(1 for l in lst if (p, l['message']) in decoded if l['message'])
    for p, lst in tfmf.items():
        h = datetime.fromtimestamp(p * PERIOD_S, tz=timezone.utc).strftime('%Y-%m-%d_%H')
        hour_t[h] += len(lst)
    print('  Per-hour rates (UTC):')
    print(f'    {"hour":<16} {"sq_launch":>10} {"sq_decoded":>11} {"tfmf_cand":>10}')
    for h in sorted(set(hour_l) | set(hour_t)):
        print(f'    {h:<16} {hour_l.get(h, 0):>10} {hour_d.get(h, 0):>11} {hour_t.get(h, 0):>10}')
    print()

    # ── Frequency distribution of TFMF-only candidates ────────────────────
    bin_edges = list(range(-24, 25, 2))   # 2-kHz bins from -24 to +24 kHz
    bins = Counter()
    for p, lst in tfmf.items():
        launches_in_p = launches.get(p, [])
        for cand in lst:
            near_any = False
            for launch in launches_in_p:
                if (abs(cand['t_s'] - launch['t_sec']) <= args.tol_time and
                        abs(cand['freq_hz'] - launch['audio_hz']) <= args.tol_freq_hz):
                    near_any = True
                    break
            if not near_any:
                khz = cand['freq_hz'] / 1000.0
                idx = bisect.bisect_right(bin_edges, khz) - 1
                if 0 <= idx < len(bin_edges) - 1:
                    bins[bin_edges[idx]] += 1
    print('  TFMF-only candidates by freq band (2-kHz bins, audio):')
    max_n = max(bins.values()) if bins else 1
    for lo in bin_edges[:-1]:
        n = bins.get(lo, 0)
        bar = '#' * int(60 * n / max_n) if max_n else ''
        print(f'    {lo:+4d}..{lo+2:+4d} kHz   {n:>6}   {bar}')
    print()

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available; skipping plot', file=sys.stderr)
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        # All TFMF cands: small grey dots
        tx, ty = [], []
        for p, lst in tfmf.items():
            t0 = p * PERIOD_S
            for c in lst:
                tx.append(datetime.fromtimestamp(t0 + c['t_s'], tz=timezone.utc))
                ty.append(c['freq_hz'] / 1000.0)
        ax.scatter(tx, ty, s=2, c='#808080', alpha=0.4, label=f'TFMF cands ({len(tx)})')
        # sq launches not decoded: orange
        ox, oy = [], []
        # sq launches decoded: green
        gx, gy = [], []
        for p, lst in launches.items():
            t0 = p * PERIOD_S
            for L in lst:
                is_dec = bool(L['message']) and (p, L['message']) in decoded
                x = datetime.fromtimestamp(t0 + L['t_sec'], tz=timezone.utc)
                y = L['audio_hz'] / 1000.0
                if is_dec:
                    gx.append(x); gy.append(y)
                else:
                    ox.append(x); oy.append(y)
        ax.scatter(ox, oy, s=8, c='orange', alpha=0.4, label=f'sq launches no-decode ({len(ox)})')
        ax.scatter(gx, gy, s=24, c='green', edgecolors='black', linewidths=0.5,
                   label=f'sq decoded ({len(gx)})')
        ax.set_xlabel('UTC time')
        ax.set_ylabel('Audio freq (kHz)')
        ax.set_ylim(-24, 24)
        ax.set_title(f'TFMF candidates vs sq_det events — '
                     f'{datetime.fromtimestamp(e_min, tz=timezone.utc):%Y-%m-%d %H:%M}'
                     f' → {datetime.fromtimestamp(e_max, tz=timezone.utc):%H:%M} UTC')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f'  plot: {args.plot}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol-time', type=float, default=2.0,
                    help='Matching tolerance in seconds (default 2.0; covers '
                         'MSK144 ping duration plus sq_det/TFMF time-peak offset)')
    ap.add_argument('--tol-freq-hz', type=float, default=2000.0,
                    help='Matching tolerance in Hz (default 2000; covers '
                         'sq_det 1-kHz channel quantisation plus per-ping drift)')
    ap.add_argument('--pan-center-mhz', type=float, default=PAN_CENTER_MHZ,
                    help=f'Pan centre in MHz (default {PAN_CENTER_MHZ})')
    ap.add_argument('--plot', type=str, default=None,
                    help='Save scatter plot to this path (e.g. /tmp/x.png)')
    args = ap.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
