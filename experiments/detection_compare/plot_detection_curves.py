#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Plot detection probability vs SNR for the detectors in detection_curves.csv.

Usage examples:
    # Default: all algorithms, Pd vs SNR, save PNG.
    python plot_detection_curves.py

    # Choose a subset of algorithms:
    python plot_detection_curves.py --algos sq_det,tfmf_2block

    # Different metric:
    python plot_detection_curves.py --metric candidates  # avg candidate count per trial

    # Save to a specific path:
    python plot_detection_curves.py --out /tmp/foo.png

The CSV format (one row per trial × algorithm):
    algorithm, snr_db, trial, detected (0/1), n_candidates, threshold
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: Path):
    """Load CSV → dict[algorithm] -> list of rows, each row a dict of values."""
    by_algo = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['snr_db']       = int(row['snr_db'])
            row['trial']        = int(row['trial'])
            row['detected']     = int(row['detected'])
            row['n_candidates'] = int(row['n_candidates'])
            row['threshold']    = float(row['threshold'])
            by_algo[row['algorithm']].append(row)
    return by_algo


def aggregate(rows, metric: str):
    """For one algorithm's rows, return (snr_array, value_array, ci_lo, ci_hi).

    metric = 'pd'         : detection probability per SNR bucket
    metric = 'candidates' : mean candidate count per trial per SNR bucket
    """
    by_snr = defaultdict(list)
    for r in rows:
        by_snr[r['snr_db']].append(r)
    snrs = sorted(by_snr.keys())
    vals, lo, hi = [], [], []
    for s in snrs:
        bucket = by_snr[s]
        n = len(bucket)
        if metric == 'pd':
            k = sum(r['detected'] for r in bucket)
            p = k / n if n else 0.0
            # Wilson 95% CI
            z = 1.96
            denom = 1 + z*z/n
            centre = (p + z*z/(2*n)) / denom
            half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
            vals.append(p)
            lo.append(max(0.0, centre - half))
            hi.append(min(1.0, centre + half))
        elif metric == 'candidates':
            mc = np.mean([r['n_candidates'] for r in bucket])
            sd = np.std([r['n_candidates'] for r in bucket])
            vals.append(mc)
            lo.append(max(0.0, mc - sd))
            hi.append(mc + sd)
        else:
            raise ValueError(f"unknown metric {metric}")
    return np.array(snrs), np.array(vals), np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', type=Path,
                    default=Path(__file__).parent / 'detection_curves.csv',
                    help='Input CSV (default: ./detection_curves.csv)')
    ap.add_argument('--algos', type=str, default=None,
                    help='Comma-separated list of algorithms to plot '
                         '(default: all present in CSV)')
    ap.add_argument('--metric', choices=['pd', 'candidates'], default='pd',
                    help='What to plot: pd (detection probability) or '
                         'candidates (mean candidate count)')
    ap.add_argument('--out', type=Path, default=None,
                    help='Output PNG path (default: derive from csv + metric)')
    ap.add_argument('--decode-floor', type=str, default='0,-8.5',
                    help='Comma-separated decode floor SNRs to mark with '
                         'vertical lines (default: 0,-8.5 for K=1 and K=7).')
    ap.add_argument('--no-show', action='store_true',
                    help='Save without popping a window')
    args = ap.parse_args()

    by_algo = load_csv(args.csv)
    if args.algos:
        wanted = args.algos.split(',')
        by_algo = {k: v for k, v in by_algo.items() if k in wanted}

    if not by_algo:
        raise SystemExit("No algorithms to plot — check --algos or CSV contents.")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, (algo, rows) in enumerate(sorted(by_algo.items())):
        snrs, vals, lo, hi = aggregate(rows, args.metric)
        col = colors[idx % len(colors)]
        ax.plot(snrs, vals, marker='o', label=algo, color=col, linewidth=2)
        ax.fill_between(snrs, lo, hi, color=col, alpha=0.15)

    # Decode-floor reference lines
    if args.metric == 'pd' and args.decode_floor:
        for spec in args.decode_floor.split(','):
            spec = spec.strip()
            if not spec:
                continue
            try:
                x = float(spec)
            except ValueError:
                continue
            ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.text(x, 0.02, f' decode floor {x:g} dB ', rotation=90,
                    fontsize=8, va='bottom', ha='right', color='gray')

    ax.set_xlabel('SNR_2.5k (dB)')
    if args.metric == 'pd':
        ax.set_ylabel('Detection probability')
        ax.set_ylim(-0.02, 1.05)
        ax.set_title('Detection probability vs SNR at matched FA rate')
    else:
        ax.set_ylabel('Mean candidate count per 15-s trial')
        ax.set_title('Candidate count vs SNR')
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    out = args.out
    if out is None:
        stem = args.csv.stem
        out = args.csv.parent / f'{stem}_{args.metric}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
