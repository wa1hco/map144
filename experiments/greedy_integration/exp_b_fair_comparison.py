#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment B' — TGI vs single-frame at MATCHED false-positive rate.

The raw Experiment B (exp_b_tgi_value.py) showed TGI achieving "+11 dB
SNR50 gain" but ALSO 74-96% false-positive rate on pure noise.  That
gain is illusory — TGI fires on noise as readily as on signal because
the phase-search inflation pushes its metric over the threshold-3
detection bar even when no signal is present.

This experiment performs the standard signal-detection-theory comparison:
calibrate each algorithm's threshold so they share the SAME false-positive
rate on noise, then measure detection rates on signal at that fair
threshold.  Apples-to-apples.

Procedure
---------
1. Compute baseline xmed (5 s of noise).

2. **Calibrate**: run each algorithm on N_noise=500 pure-noise buffers,
   record the metric distribution per algorithm, find the threshold T*
   that gives the target false-positive rate (default 1%).

3. **Measure**: SNR sweep on signal bursts.  Each algorithm DETECTS if
   its metric >= its OWN T* (NOT the universal 3.0).  Report detection
   rate vs SNR per algorithm.

4. Output: per-trial CSV, summary CSV, plot showing fair detection rates
   AND the calibrated thresholds.

Headline result
---------------
Reading SNR50 with the calibrated thresholds tells whether TGI's
apparent advantage survives matched FP-rate normalization.  If yes,
TGI gives real sensitivity gain X dB.  If no, the algorithm needs a
better greedy criterion before it can be considered useful.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root for map144_app
from map144_app.sq_det import sq_det, slide_sq_det, NSPM, FS, DEFAULT_FC
from algos.tgi import tgi_integrate, TGIConfig
from exp_a_formula_comparison import make_audio_noise, audio_to_analytic
from exp_b_tgi_value import (
    make_audio_buffer_with_burst, find_strongest_anchor,
    algo_single_frame, algo_tgi,
)


def calibrate_threshold(metrics: np.ndarray, target_fp_rate: float) -> float:
    """Find the threshold T such that fraction of metrics >= T equals
    target_fp_rate.  Threshold = (1 - target_fp_rate)-percentile of metrics.
    """
    return float(np.percentile(metrics, 100.0 * (1.0 - target_fp_rate)))


def calibrate_all(xmed: float, n_noise_trials: int = 500,
                  fc: float = DEFAULT_FC) -> dict[str, np.ndarray]:
    """Run all three algorithms on n_noise_trials pure-noise buffers; return
    the metric distributions for threshold calibration."""
    pre_n   = int(0.4 * FS)
    burst_n = 5 * NSPM
    post_n  = int(0.4 * FS)
    metrics = {'single': [], 'naive': [], 'search': []}
    for k in range(n_noise_trials):
        rng = np.random.default_rng(50000 + k)
        audio = rng.normal(0, 1.0, pre_n + burst_n + post_n).astype(np.float64)
        t0, t1 = pre_n / FS, (pre_n + burst_n) / FS
        anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
        s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
        n = algo_tgi(audio, anchor_t, xmed, n_phase_search=1, fc=fc)
        p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
        metrics['single'].append(s.detmet_norm)
        metrics['naive'].append(n.detmet_norm)
        metrics['search'].append(p.detmet_norm)
    return {k: np.array(v) for k, v in metrics.items()}


@dataclass
class TrialResult:
    snr_db: float; seed: int
    single_metric: float; single_pass: bool
    naive_metric:  float; naive_pass:  bool;  naive_n_kept:  int
    search_metric: float; search_pass: bool;  search_n_kept: int


def run_signal_trial(snr_db: float, seed: int, xmed: float,
                     thresholds: dict[str, float],
                     n_frames: int = 5, fc: float = DEFAULT_FC) -> TrialResult:
    audio, t0, t1 = make_audio_buffer_with_burst(n_frames, snr_db,
                                                  fc=fc, seed=seed)
    anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
    s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
    n = algo_tgi(audio, anchor_t, xmed, n_phase_search=1, fc=fc)
    p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
    return TrialResult(
        snr_db=snr_db, seed=seed,
        single_metric=s.detmet_norm, single_pass=s.detmet_norm >= thresholds['single'],
        naive_metric =n.detmet_norm, naive_pass =n.detmet_norm >= thresholds['naive'],
        naive_n_kept=n.n_kept,
        search_metric=p.detmet_norm, search_pass=p.detmet_norm >= thresholds['search'],
        search_n_kept=p.n_kept,
    )


def aggregate(results: list[TrialResult]):
    snr_set = sorted({r.snr_db for r in results})
    grid = np.array(snr_set)
    s_rate = np.zeros_like(grid); n_rate = np.zeros_like(grid); p_rate = np.zeros_like(grid)
    n_kept_naive  = np.zeros_like(grid); n_kept_search = np.zeros_like(grid)
    for i, snr in enumerate(grid):
        bin_results = [r for r in results if r.snr_db == snr]
        n = len(bin_results)
        s_rate[i] = sum(r.single_pass for r in bin_results) / n
        n_rate[i] = sum(r.naive_pass  for r in bin_results) / n
        p_rate[i] = sum(r.search_pass for r in bin_results) / n
        n_kept_naive[i]  = float(np.mean([r.naive_n_kept  for r in bin_results]))
        n_kept_search[i] = float(np.mean([r.search_n_kept for r in bin_results]))
    return grid, s_rate, n_rate, p_rate, n_kept_naive, n_kept_search


def find_snr50(grid, rate):
    if rate.max() < 0.5: return float('nan')
    if rate.min() >= 0.5: return float(grid[0])
    for i in range(len(grid) - 1):
        if rate[i] < 0.5 <= rate[i + 1]:
            x0, x1, y0, y1 = grid[i], grid[i + 1], rate[i], rate[i + 1]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float('nan')


def make_plots(grid, s, n, p, kn, ks, snr50_s, snr50_n, snr50_p,
               thresholds: dict[str, float], target_fp: float,
               n_seeds: int, n_noise: int, plot_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(grid, s * 100, 'o-', color='#444', linewidth=2,
             label=f'single-frame  (T*={thresholds["single"]:.2f}, '
                   f'SNR50={snr50_s:+.2f} dB)')
    ax1.plot(grid, n * 100, 's-', color='#a40', linewidth=2,
             label=f'TGI naive      (T*={thresholds["naive"]:.2f}, '
                   f'SNR50={snr50_n:+.2f} dB)')
    ax1.plot(grid, p * 100, 'D-', color='#226', linewidth=2,
             label=f'TGI w/ search  (T*={thresholds["search"]:.2f}, '
                   f'SNR50={snr50_p:+.2f} dB)')
    ax1.axhline(50, color='#888', linestyle=':', linewidth=0.7)
    if not np.isnan(snr50_s): ax1.axvline(snr50_s, color='#444', linestyle=':', linewidth=0.6)
    if not np.isnan(snr50_n): ax1.axvline(snr50_n, color='#a40', linestyle=':', linewidth=0.6)
    if not np.isnan(snr50_p): ax1.axvline(snr50_p, color='#226', linestyle=':', linewidth=0.6)
    ax1.set_xlabel('Audio SNR (dB)')
    ax1.set_ylabel('Detection rate (%)')
    ax1.set_ylim(-5, 105)
    ax1.set_title(f'Experiment B′ — fair comparison at FP rate = {target_fp*100:.1f}%\n'
                  f'(thresholds calibrated from {n_noise} noise trials; '
                  f'detect rate = {n_seeds} signal trials/SNR)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    ax2.plot(grid, kn, 's-', color='#a40', linewidth=2,
             label='TGI naive  (mean n_kept)')
    ax2.plot(grid, ks, 'D-', color='#226', linewidth=2,
             label='TGI w/ search  (mean n_kept)')
    ax2.axhline(1, color='#888', linestyle=':', linewidth=0.7,
                label='1 (single-frame equivalent)')
    ax2.axhline(5, color='#0a0', linestyle=':', linewidth=0.7,
                label='5 (burst length)')
    ax2.set_xlabel('Audio SNR (dB)')
    ax2.set_ylabel('Mean frames kept by TGI')
    ax2.set_ylim(0, 8)
    ax2.set_title('TGI integration depth vs SNR')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-fp',    type=float, default=0.01,
                    help='Target false-positive rate for threshold calibration (default 0.01 = 1%%)')
    ap.add_argument('--n-noise',      type=int,   default=500,
                    help='Pure-noise trials for threshold calibration (default 500)')
    ap.add_argument('--snr-lo',       type=float, default=-15.0)
    ap.add_argument('--snr-hi',       type=float, default=+5.0)
    ap.add_argument('--snr-step',     type=float, default=1.0)
    ap.add_argument('--n-seeds',      type=int,   default=50)
    ap.add_argument('--n-frames',     type=int,   default=5)
    ap.add_argument('--noise-seconds', type=float, default=5.0)
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--out-trials', type=Path,
                    default=repo_root / 'scratch/results/exp_b_fair_per_trial.csv')
    ap.add_argument('--out-summary', type=Path,
                    default=repo_root / 'scratch/results/exp_b_fair_summary.csv')
    ap.add_argument('--out-plot', type=Path,
                    default=repo_root / 'scratch/plots/exp_b_fair_comparison.png')
    args = ap.parse_args(argv)

    print(f'Computing WSJT-X xmed baseline ({args.noise_seconds} s of noise)…',
          file=sys.stderr)
    n = int(args.noise_seconds * FS)
    noise = make_audio_noise(n, seed=0)
    analytic = audio_to_analytic(noise)
    xmed = float(slide_sq_det(analytic).xmed)
    print(f'  xmed = {xmed:.6f}', file=sys.stderr)

    print(f'Calibrating thresholds at FP={args.target_fp*100:.1f}% from '
          f'{args.n_noise} noise trials…', file=sys.stderr)
    noise_metrics = calibrate_all(xmed, n_noise_trials=args.n_noise)
    thresholds = {k: calibrate_threshold(v, args.target_fp)
                  for k, v in noise_metrics.items()}
    print(f'  Calibrated thresholds (metric units = detmet/xmed):', file=sys.stderr)
    for k, t in thresholds.items():
        # Sanity: report what fraction actually exceeds (should ≈ target_fp)
        actual_fp = float((noise_metrics[k] >= t).mean())
        print(f'    {k:<8}: T* = {t:>7.3f}   '
              f'(actual FP at this T: {actual_fp*100:.2f}%, target {args.target_fp*100:.1f}%)',
              file=sys.stderr)
        # Also show what the universal "3.0" threshold gives
        u_fp = float((noise_metrics[k] >= 3.0).mean())
        print(f'              (FP at universal threshold 3.0: {u_fp*100:.1f}%)',
              file=sys.stderr)

    snr_grid = np.arange(args.snr_lo, args.snr_hi + args.snr_step / 2, args.snr_step)
    print(f'\nSNR sweep at calibrated thresholds: {len(snr_grid)} steps × '
          f'{args.n_seeds} trials', file=sys.stderr)
    results: list[TrialResult] = []
    for snr_db in snr_grid:
        seed_base = int(round(3000 * (snr_db + 50)))
        for k in range(args.n_seeds):
            results.append(run_signal_trial(float(snr_db), seed_base + k, xmed,
                                             thresholds, n_frames=args.n_frames))

    grid, s, n_rate, p, kn, ks = aggregate(results)
    snr50_s = find_snr50(grid, s)
    snr50_n = find_snr50(grid, n_rate)
    snr50_p = find_snr50(grid, p)

    print(file=sys.stderr)
    print('=' * 70, file=sys.stderr)
    print(f'Experiment B′ — Fair comparison at FP={args.target_fp*100:.1f}%',
          file=sys.stderr)
    print('=' * 70, file=sys.stderr)
    print(f'  {"SNR(dB)":>7}  {"single":>7}  {"TGI_naive":>9}  {"TGI_search":>10}  '
          f'{"naive_n":>7}  {"search_n":>8}', file=sys.stderr)
    for i, snr in enumerate(grid):
        print(f'  {snr:>+7.1f}  {s[i]*100:>6.1f}%  {n_rate[i]*100:>8.1f}%  '
              f'{p[i]*100:>9.1f}%  {kn[i]:>7.2f}  {ks[i]:>8.2f}', file=sys.stderr)
    print(file=sys.stderr)
    print(f'SNR50 (50% detection rate at calibrated thresholds):', file=sys.stderr)
    print(f'  single-frame baseline : {snr50_s:+.2f} dB', file=sys.stderr)
    print(f'  TGI naive             : {snr50_n:+.2f} dB  (gain {snr50_s - snr50_n:+.2f} dB)', file=sys.stderr)
    print(f'  TGI w/ phase search   : {snr50_p:+.2f} dB  (gain {snr50_s - snr50_p:+.2f} dB)', file=sys.stderr)

    # Per-trial CSV
    args.out_trials.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trials.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_db', 'seed',
                    'single_metric', 'single_pass',
                    'naive_metric', 'naive_pass', 'naive_n_kept',
                    'search_metric', 'search_pass', 'search_n_kept'])
        for r in results:
            w.writerow([f'{r.snr_db:+.1f}', r.seed,
                        f'{r.single_metric:.4f}', int(r.single_pass),
                        f'{r.naive_metric:.4f}',  int(r.naive_pass),  r.naive_n_kept,
                        f'{r.search_metric:.4f}', int(r.search_pass), r.search_n_kept])

    # Summary CSV
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['target_fp_rate', f'{args.target_fp:.4f}'])
        w.writerow(['threshold_single', f'{thresholds["single"]:.4f}'])
        w.writerow(['threshold_naive',  f'{thresholds["naive"]:.4f}'])
        w.writerow(['threshold_search', f'{thresholds["search"]:.4f}'])
        w.writerow([])
        w.writerow(['snr_db', 'single_rate', 'naive_rate', 'search_rate',
                    'naive_mean_n_kept', 'search_mean_n_kept'])
        for i, snr in enumerate(grid):
            w.writerow([f'{snr:+.1f}', f'{s[i]:.4f}', f'{n_rate[i]:.4f}',
                        f'{p[i]:.4f}', f'{kn[i]:.2f}', f'{ks[i]:.2f}'])

    make_plots(grid, s, n_rate, p, kn, ks, snr50_s, snr50_n, snr50_p,
               thresholds, args.target_fp, args.n_seeds, args.n_noise,
               args.out_plot)
    print(file=sys.stderr)
    print(f'Wrote:', file=sys.stderr)
    print(f'  per-trial CSV : {args.out_trials}', file=sys.stderr)
    print(f'  summary CSV   : {args.out_summary}', file=sys.stderr)
    print(f'  plot          : {args.out_plot}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
