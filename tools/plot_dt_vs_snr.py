#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""2-D plot of WSJT-X ping time (dt) vs SNR for the corpus.

Reads ``tests/corpus/index.csv``, applies the same ``--tag`` / ``--any-tag``
filters as ``gallery_corpus.py``, parses the ``wsjtx_decodes_json`` blob
per row, and renders a scatter of (dt, SNR) with marginal histograms.

Use for spotting biases of the form "weak signals cluster near dt=0" or
"high-SNR signals favour the middle of the window".  The marginal
histograms make non-uniformity visible at a glance; the scatter makes
single-QSO clusters visible (same-callsign tail-of-fading shows up as a
vertical streak at fixed dt).

Usage
-----
::

    # All WSJT-X decodes in the corpus
    python3 tools/plot_dt_vs_snr.py --out ~/gallery/dt_vs_snr_all.png

    # Restrict to the mission-gap bucket
    python3 tools/plot_dt_vs_snr.py --tag weak --tag wsjtx_only \\
        --out ~/gallery/dt_vs_snr_gap.png

    # Or any filter the gallery supports
    python3 tools/plot_dt_vs_snr.py --any-tag canonical_strong \\
        --any-tag both_decoded --out ~/gallery/dt_vs_snr_decoded.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_REPO_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = _REPO_ROOT / "tests" / "corpus" / "index.csv"


def _row_tags(row: dict) -> set[str]:
    return set(row['tags'].split(';')) if row['tags'] else set()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index',   type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--tag',     action='append', default=[],
                    help='Require this tag (repeat for AND)')
    ap.add_argument('--any-tag', action='append', default=[],
                    help='Match if any of these tags present (OR)')
    ap.add_argument('--out',     type=Path,
                    default=_REPO_ROOT / 'scratch' / 'plots' / 'dt_vs_snr.png',
                    help='Output PNG path.  Default writes to scratch/plots/ '
                         '(in-tree, gitignored).')
    args = ap.parse_args(argv)

    if not args.index.is_file():
        print(f'error: {args.index} not found; run select_stress_corpus.py first',
              file=sys.stderr)
        return 1

    # ── Load + filter ───────────────────────────────────────────────────────
    req     = set(args.tag)
    any_set = set(args.any_tag)
    dts: list[float] = []
    snrs: list[int]  = []
    callsign_clusters: dict[tuple[str, str], list[int]] = {}
    with args.index.open() as fh:
        for r in csv.DictReader(fh):
            rt = _row_tags(r)
            if req and not req.issubset(rt):
                continue
            if any_set and not (any_set & rt):
                continue
            try:
                decodes = json.loads(r.get('wsjtx_decodes_json') or '[]')
            except Exception:
                continue
            for d in decodes:
                dts.append(float(d['dt']))
                snrs.append(int(d['snr']))
                # Pull the responding callsign (second token, approximate) so
                # we can highlight same-QSO clusters.
                toks = d['msg'].split()
                if len(toks) >= 2:
                    key = (toks[1], r['period_ts'][:10])  # callsign + date
                    callsign_clusters.setdefault(key, []).append(len(dts) - 1)

    if not dts:
        print('error: no decodes matched the filter', file=sys.stderr)
        return 1

    dts_arr  = np.asarray(dts, dtype=np.float32)
    snrs_arr = np.asarray(snrs, dtype=np.float32)

    filter_desc = (
        f"all of [{', '.join(args.tag)}]" if args.tag else 'all rows'
    ) + (
        f" & any of [{', '.join(args.any_tag)}]" if args.any_tag else ''
    )

    print(f'scatter: n={len(dts)}  filter: {filter_desc}', file=sys.stderr)
    print(f'  dt   mean={dts_arr.mean():.2f}  median={np.median(dts_arr):.2f}  '
          f'min={dts_arr.min():.2f}  max={dts_arr.max():.2f}', file=sys.stderr)
    print(f'  snr  mean={snrs_arr.mean():.2f}  median={np.median(snrs_arr):.1f}  '
          f'min={int(snrs_arr.min())}  max={int(snrs_arr.max())}', file=sys.stderr)

    # ── Plot: scatter + marginal histograms ─────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           wspace=0.04, hspace=0.04)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # SNR-coloured scatter: alpha for density, color encodes SNR so weak
    # vs strong is visible spatially.  cool→warm range (viridis_r) — light
    # for strong, dark for weak.
    sc = ax_main.scatter(dts_arr, snrs_arr,
                         c=snrs_arr, cmap='viridis',
                         s=18, alpha=0.55, edgecolors='none')
    ax_main.set_xlabel('WSJT-X dt (s)', fontsize=11)
    ax_main.set_ylabel('WSJT-X SNR (dB)', fontsize=11)
    ax_main.grid(True, alpha=0.25, linestyle='--')
    # Period-boundary reference lines
    ax_main.axvline(0,  color='#888', linewidth=0.7, linestyle=':')
    ax_main.axvline(15, color='#888', linewidth=0.7, linestyle=':')
    ax_main.set_xlim(-1.0, 16.0)

    # Top marginal: dt histogram with uniform-baseline reference
    n_dt_bins = 30
    counts_top, edges_top, _ = ax_top.hist(
        dts_arr, bins=n_dt_bins, range=(0, 15),
        color='#5577aa', edgecolor='white', linewidth=0.5)
    expected = len(dts) / n_dt_bins
    ax_top.axhline(expected, color='#c44', linewidth=0.8, linestyle='--',
                   alpha=0.8, label=f'uniform = {expected:.1f}')
    ax_top.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_top.set_ylabel('count', fontsize=9)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, alpha=0.2, axis='y')

    # Right marginal: SNR histogram (1-dB bins)
    snr_lo = int(np.floor(snrs_arr.min())) - 1
    snr_hi = int(np.ceil(snrs_arr.max()))  + 1
    ax_right.hist(snrs_arr, bins=range(snr_lo, snr_hi + 1),
                  orientation='horizontal',
                  color='#5577aa', edgecolor='white', linewidth=0.5)
    ax_right.set_xlabel('count', fontsize=9)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(True, alpha=0.2, axis='x')

    # Title (top-left)
    fig.suptitle(
        f'WSJT-X ping time vs SNR — {filter_desc}   '
        f'(n={len(dts)} decodes)',
        fontsize=11, y=0.965,
    )

    # Stats annotation in main panel (top-left corner)
    stats = (
        f'n           = {len(dts)}\n'
        f'dt  mean    = {dts_arr.mean():.2f} s\n'
        f'dt  median  = {np.median(dts_arr):.2f} s\n'
        f'snr mean    = {snrs_arr.mean():.2f} dB\n'
        f'snr median  = {np.median(snrs_arr):.1f} dB\n'
        f'|dt|<1 frac = {(np.abs(dts_arr) < 1).mean():.0%}\n'
        f'(uniform    = {1/15:.0%})'
    )
    ax_main.text(0.012, 0.985, stats,
                 transform=ax_main.transAxes, fontsize=9, family='monospace',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffce8',
                           edgecolor='#ddd5a0', alpha=0.92))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
