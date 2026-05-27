#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantify the period-mode decoding opportunity for MAP144.

For each corpus case where MAP144 produced no decode but WSJT-X did
(``weak ∩ wsjtx_only`` by default — overridable via --tag/--any-tag),
run ``jt9 --msk144`` on the full WSJT-X save WAV and compare the
decoded messages to the WSJT-X ALL.TXT entries for that period.

Reports:

  * how many cases decoded with full-WAV jt9 (the gap closure ceiling
    for adding a period-mode decode path to MAP144)
  * coverage: per-decode-message agreement with WSJT-X
  * SNR distribution of the cases that DID and DID NOT close
  * a per-case CSV for follow-up review

Defaults match MAP144's internal jt9 settings (``-d 2``, ftol=200,
af-centre=1500) so the test isolates the period-mode-vs-burst-mode
question cleanly — same decoder parameters, different window.

Usage::

    python3 tools/test_period_mode_decode.py
    python3 tools/test_period_mode_decode.py --tag wsjtx_only --depth 3
    python3 tools/test_period_mode_decode.py --workers 8 \
        --out-csv scratch/results/period_mode_d2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = _REPO_ROOT / "tests" / "corpus" / "index.csv"

# jt9 ALL.TXT-style line regex (snr dt af mode-glyph msg)
_JT9_RE = re.compile(
    r'^\s*\d{6}\s+(?P<snr>[-+]?\d+)\s+(?P<dt>[-+]?[\d.]+)\s+'
    r'(?P<af>\d+)\s+\S+\s+(?P<msg>.+?)\s*$'
)

DEFAULT_FTOL = 200          # MAP144 internal default
DEFAULT_AF_C = 1500
DEFAULT_DEPTH = 2           # matches MAP144's current ndepth (project_jt9_decode_depth)


def _norm(msg: str) -> str:
    return ' '.join(msg.upper().split())


def _row_tags(row: dict) -> set[str]:
    return set(row['tags'].split(';')) if row['tags'] else set()


def run_jt9(wav_path: Path, ftol: int, depth: int,
            af_c: int = DEFAULT_AF_C, timeout_s: float = 30.0) -> list[dict]:
    cmd = [
        'jt9', '--msk144',
        '-L', str(af_c - ftol),
        '-H', str(af_c + ftol),
        '-F', str(ftol),
        '-d', str(depth),
        str(wav_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return [{'error': 'timeout'}]
    out: list[dict] = []
    for ln in r.stdout.splitlines():
        m = _JT9_RE.match(ln)
        if m:
            out.append({
                'snr': int(m['snr']),
                'dt':  float(m['dt']),
                'af':  int(m['af']),
                'msg': _norm(m['msg']),
            })
    return out


def _process_one(case_id: str, wav_path_str: str, wsjtx_msgs_norm: list[str],
                 ftol: int, depth: int) -> dict:
    """Worker — must be top-level so ProcessPoolExecutor can pickle it."""
    decodes = run_jt9(Path(wav_path_str), ftol, depth)
    error = next((d.get('error') for d in decodes if 'error' in d), None)
    decoded_msgs = {d['msg'] for d in decodes if 'error' not in d}
    wsjtx_set    = set(wsjtx_msgs_norm)
    return {
        'case_id':       case_id,
        'wav_path':      wav_path_str,
        'jt9_n_decoded': len(decoded_msgs),
        'jt9_msgs':      ';'.join(sorted(decoded_msgs)),
        'wsjtx_msgs':    ';'.join(sorted(wsjtx_set)),
        'overlap':       len(wsjtx_set & decoded_msgs),
        'wsjtx_only':    len(wsjtx_set - decoded_msgs),   # WSJT-X got, jt9-full missed
        'jt9_only':      len(decoded_msgs - wsjtx_set),   # bonus jt9-full found
        'error':         error or '',
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index',   type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--tag',     action='append', default=['weak', 'wsjtx_only'],
                    help='Require this tag (repeat for AND).  Default: '
                         "['weak','wsjtx_only'] — the mission-gap bucket.  "
                         'Override to test a different population.')
    ap.add_argument('--any-tag', action='append', default=[])
    ap.add_argument('--ftol',    type=int, default=DEFAULT_FTOL)
    ap.add_argument('--depth',   type=int, default=DEFAULT_DEPTH)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max',     type=int, default=None,
                    help='Cap on number of cases (for quick smoke tests)')
    ap.add_argument('--out-csv', type=Path, default=None,
                    help='Write per-case results to CSV')
    ap.add_argument('--noise-mode', action='store_true',
                    help='Phantom-rate test: filter to periods with a WSJT-X WAV '
                         'but ZERO WSJT-X decodes (i.e. noise-only).  Any jt9 '
                         'decode is counted as a phantom.  Overrides --tag.')
    args = ap.parse_args(argv)

    if not args.index.is_file():
        print(f'error: index not found at {args.index}', file=sys.stderr)
        return 1

    # ── Filter rows.  Two modes:                                ─────────────
    #   default: tag filter (e.g. weak+wsjtx_only — periods WSJT-X decoded)
    #   --noise-mode: zero WSJT-X decodes (phantom-rate test)
    # Both require an actual WAV to feed to jt9.
    req     = set(args.tag)
    any_set = set(args.any_tag)
    cases: list[tuple[str, str, list[str], int]] = []   # (id, wav_path, wsjtx_msgs, snr_min)
    with args.index.open() as fh:
        for r in csv.DictReader(fh):
            wav = r.get('wsjtx_wav')
            if not wav:
                continue
            if args.noise_mode:
                if int(r.get('n_wsjtx_decodes') or 0) != 0:
                    continue
                wsjtx_msgs = []   # any jt9 decode is a phantom
                snr_min = None
            else:
                tags = _row_tags(r)
                if req and not req.issubset(tags):
                    continue
                if any_set and not (any_set & tags):
                    continue
                try:
                    decodes = json.loads(r['wsjtx_decodes_json'] or '[]')
                except Exception:
                    decodes = []
                wsjtx_msgs = [_norm(d['msg']) for d in decodes]
                try:    snr_min = int(r['wsjtx_snr_min'])
                except: snr_min = None
            cases.append((r['case_id'], wav, wsjtx_msgs, snr_min))
    if args.max:
        cases = cases[:args.max]

    print(f'period-mode decode test', file=sys.stderr)
    print(f'  filter   : require={args.tag}  any={args.any_tag}', file=sys.stderr)
    print(f'  cases    : {len(cases)} with WSJT-X WAVs', file=sys.stderr)
    print(f'  jt9      : -L {DEFAULT_AF_C-args.ftol} -H {DEFAULT_AF_C+args.ftol} '
          f'-F {args.ftol} -d {args.depth}', file=sys.stderr)
    print(f'  workers  : {args.workers}', file=sys.stderr)
    print(file=sys.stderr)

    # ── Run jt9 in parallel ───────────────────────────────────────────────────
    results: list[dict] = []
    snr_by_id = {cid: snr for cid, _, _, snr in cases}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_process_one, cid, wav, msgs, args.ftol, args.depth): cid
            for cid, wav, msgs, _ in cases
        }
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(cases):
                print(f'  [{i}/{len(cases)}]', file=sys.stderr, flush=True)

    # ── Categorise per case ──────────────────────────────────────────────────
    n_total = len(results)
    n_error = 0
    if args.noise_mode:
        # Phantom-rate test: any decode is a phantom.  WSJT-X reference is
        # empty by construction; jt9_only equals total jt9 decodes.
        from collections import Counter
        per_case_counts: list[int] = []
        unique_phantoms: Counter = Counter()
        for r in results:
            if r['error']:
                n_error += 1; continue
            n_phantoms_here = r['jt9_only']
            per_case_counts.append(n_phantoms_here)
            for m in (r['jt9_msgs'].split(';') if r['jt9_msgs'] else []):
                if m: unique_phantoms[m] += 1
        n_clean    = sum(1 for n in per_case_counts if n == 0)
        n_one      = sum(1 for n in per_case_counts if n == 1)
        n_multi    = sum(1 for n in per_case_counts if n >= 2)
        n_phantoms = sum(per_case_counts)

        print()
        print('=' * 72)
        print(f'PHANTOM-RATE TEST  (n={n_total} noise-only periods, '
              f'jt9 -d {args.depth} on full WAV)')
        print('=' * 72)
        pct = lambda n: f'{100*n/n_total:>4.1f}%' if n_total else '   -'
        print(f'  clean (0 phantoms)        : {n_clean:>5}  ({pct(n_clean)})')
        print(f'  1 phantom                 : {n_one:>5}  ({pct(n_one)})')
        print(f'  2+ phantoms               : {n_multi:>5}  ({pct(n_multi)})')
        if n_error:
            print(f'  jt9 errors                : {n_error:>5}')
        print()
        print(f'  total phantom messages    : {n_phantoms}')
        print(f'  per-period phantom rate   : {n_phantoms/max(1,n_total):.3f}  '
              f'(phantoms / period processed)')
        if unique_phantoms:
            print()
            print(f'  most-frequent phantom messages (top 8):')
            for m, c in unique_phantoms.most_common(8):
                print(f'    ×{c:<3}  {m[:64]}')
        print()
        print('Interpretation: in production at this depth, period-mode would')
        print(f'add ~{n_phantoms/max(1,n_total):.3f} phantoms per period processed')
        print('(scale by your daily period count for nightly phantom rate).')
    else:
        n_full = n_partial = n_none = 0
        snr_full: list[int] = []
        snr_none: list[int] = []
        n_bonus = 0
        for r in results:
            if r['error']:
                n_error += 1; continue
            # WSJT-X reference set non-empty by construction (filter requires
            # at least one WSJT-X decode); wsjtx_only > 0 means jt9-full missed
            # at least one of WSJT-X's decodes.
            if r['wsjtx_only'] == 0 and r['overlap'] >= 1:
                n_full += 1
                if snr_by_id.get(r['case_id']) is not None:
                    snr_full.append(snr_by_id[r['case_id']])
            elif r['overlap'] >= 1:
                n_partial += 1
            else:
                n_none += 1
                if snr_by_id.get(r['case_id']) is not None:
                    snr_none.append(snr_by_id[r['case_id']])
            if r['jt9_only'] >= 1:
                n_bonus += 1

        print()
        print('=' * 72)
        print(f'PERIOD-MODE DECODE OPPORTUNITY  (n={n_total} cases, MAP144 got 0 each)')
        print('=' * 72)
        pct = lambda n: f'{100*n/n_total:>4.1f}%' if n_total else '   -'
        print(f'  full closure (jt9 caught EVERY WSJT-X decode)  : '
              f'{n_full:>4}  ({pct(n_full)})')
        print(f'  partial closure (jt9 caught SOME of them)      : '
              f'{n_partial:>4}  ({pct(n_partial)})')
        print(f'  no closure (jt9 caught nothing WSJT-X had)     : '
              f'{n_none:>4}  ({pct(n_none)})')
        if n_error:
            print(f'  jt9 timeout / error                            : {n_error:>4}')
        print()
        print(f'  bonus: jt9 found decodes WSJT-X did NOT have    : {n_bonus:>4}  '
              f'({pct(n_bonus)})')
        print()
        if snr_full:
            print(f'  SNR distribution of FULL-CLOSURE cases  '
                  f'(min snr in period): n={len(snr_full)}')
            print(f'    median {statistics.median(snr_full):+.1f}  '
                  f'min {min(snr_full):+d}  max {max(snr_full):+d}')
        if snr_none:
            print(f'  SNR distribution of NO-CLOSURE cases     '
                  f'(min snr in period): n={len(snr_none)}')
            print(f'    median {statistics.median(snr_none):+.1f}  '
                  f'min {min(snr_none):+d}  max {max(snr_none):+d}')
        print()
        print('Bottom line: adding period-mode jt9 to MAP144 (run -d 2 on the full')
        print('15 s WAV in parallel with burst-mode) would close at MOST')
        print(f'{n_full + n_partial} of {n_total} cases ({pct(n_full + n_partial)}) '
              'in this bucket.')

    # ── Per-case CSV ─────────────────────────────────────────────────────────
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Sort by case_id for stability
        results.sort(key=lambda r: r['case_id'])
        with args.out_csv.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()) if results else [])
            w.writeheader()
            for r in results: w.writerow(r)
        print(f'\nwrote per-case CSV: {args.out_csv}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
