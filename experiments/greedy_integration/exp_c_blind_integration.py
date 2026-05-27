#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment C — BLIND integration (forced-add all K frames, phase-searched).

Diagnostic: does coherent integration HELP at all when we remove the greedy
decision-making?  Always-add all 5 frames with phase compensation per frame.

If blind integration shows clear gain over single-frame at matched FP rate,
the algorithm gap in TGI is the GREEDY DECISION (option A fix can recover it).

If blind integration ALSO disappoints, the integration architecture is the
problem (intermod ceiling or implementation bug).

Compares THREE algorithms at MATCHED 1% false-positive rate:
  - single-frame baseline
  - blind integration (force-add all 5 frames, phase search per frame)
  - TGI w/ phase search (greedy add, from exp_b_fair_comparison)

For brevity, reuses calibration + signal-burst generators from earlier
experiments.
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root for map144_app
from map144_app.sq_det import sq_det, slide_sq_det, NSPM, FS, DEFAULT_FC
from algos.tgi import TGIConfig
from exp_a_formula_comparison import make_audio_noise, audio_to_analytic
from exp_b_tgi_value import (make_audio_buffer_with_burst, find_strongest_anchor,
                              algo_single_frame, algo_tgi)


def algo_blind(audio: np.ndarray, anchor_t: float, xmed: float,
               n_frames_each_side: int = 2, n_phase_search: int = 8,
               fc: float = DEFAULT_FC):
    """Force-add all candidate frames within ±n_frames_each_side, with
    per-candidate phase search.  Always keep — no greedy decision.
    """
    analytic = audio_to_analytic(audio).astype(np.complex128)
    n_anchor = int(round(anchor_t * FS))
    n_anchor = max(0, min(len(analytic) - NSPM, n_anchor))
    integrated = analytic[n_anchor : n_anchor + NSPM].copy()
    n_kept = 1
    for off in [+1, -1, +2, -2, +n_frames_each_side, -n_frames_each_side][:2*n_frames_each_side]:
        n_start = n_anchor + off * NSPM
        n_end = n_start + NSPM
        if n_start < 0 or n_end > len(analytic):
            continue
        candidate = analytic[n_start:n_end]
        # Phase search: pick rotation that maximises detmet (raw amplitude — for
        # blind we use the raw-peak goodness, since we're not using detmet2 to
        # decide whether to add).
        best_d = -np.inf; best_phi = 0.0
        for k in range(n_phase_search):
            phi = 2.0 * np.pi * k / n_phase_search
            trial = integrated + candidate * np.exp(1j * phi)
            d = sq_det(trial, fc=fc).detmet
            if d > best_d:
                best_d = d; best_phi = phi
        integrated = integrated + candidate * np.exp(1j * best_phi)
        n_kept += 1
    final = sq_det(integrated, fc=fc)
    detmet_norm = final.detmet / max(xmed, 1e-30)
    return detmet_norm, n_kept


def calibrate(xmed: float, n_noise_trials: int, fc: float):
    """Run all three algos on noise; return metric arrays."""
    pre_n   = int(0.4 * FS); burst_n = 5 * NSPM; post_n  = int(0.4 * FS)
    metrics = {'single': [], 'blind': [], 'tgi': []}
    for k in range(n_noise_trials):
        rng = np.random.default_rng(70000 + k)
        audio = rng.normal(0, 1.0, pre_n + burst_n + post_n).astype(np.float64)
        t0, t1 = pre_n / FS, (pre_n + burst_n) / FS
        anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
        s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
        b_norm, _ = algo_blind(audio, anchor_t, xmed, fc=fc)
        p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
        metrics['single'].append(s.detmet_norm)
        metrics['blind'].append(b_norm)
        metrics['tgi'].append(p.detmet_norm)
    return {k: np.array(v) for k, v in metrics.items()}


def detect_rate(snr_grid, n_seeds, xmed, thresholds, fc):
    rates = {'single': np.zeros_like(snr_grid),
             'blind':  np.zeros_like(snr_grid),
             'tgi':    np.zeros_like(snr_grid)}
    for i, snr in enumerate(snr_grid):
        seed_base = int(round(4000 * (snr + 50)))
        s_hits = b_hits = t_hits = 0
        for k in range(n_seeds):
            audio, t0, t1 = make_audio_buffer_with_burst(5, float(snr), fc=fc, seed=seed_base + k)
            anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
            s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
            b_norm, _ = algo_blind(audio, anchor_t, xmed, fc=fc)
            p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
            if s.detmet_norm >= thresholds['single']: s_hits += 1
            if b_norm         >= thresholds['blind']:  b_hits += 1
            if p.detmet_norm >= thresholds['tgi']:    t_hits += 1
        rates['single'][i] = s_hits / n_seeds
        rates['blind'][i]  = b_hits / n_seeds
        rates['tgi'][i]    = t_hits / n_seeds
    return rates


def find_snr50(grid, rate):
    if rate.max() < 0.5: return float('nan')
    if rate.min() >= 0.5: return float(grid[0])
    for i in range(len(grid) - 1):
        if rate[i] < 0.5 <= rate[i + 1]:
            x0, x1, y0, y1 = grid[i], grid[i + 1], rate[i], rate[i + 1]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-fp', type=float, default=0.01)
    ap.add_argument('--n-noise',   type=int,   default=200)
    ap.add_argument('--snr-lo',    type=float, default=-15.0)
    ap.add_argument('--snr-hi',    type=float, default=+5.0)
    ap.add_argument('--snr-step',  type=float, default=1.0)
    ap.add_argument('--n-seeds',   type=int,   default=30)
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--out-plot', type=Path,
                    default=repo_root / 'scratch/plots/exp_c_blind_integration.png')
    args = ap.parse_args()

    print('Computing xmed…', file=sys.stderr)
    noise = make_audio_noise(int(5.0 * FS), seed=0)
    xmed = float(slide_sq_det(audio_to_analytic(noise)).xmed)

    print(f'Calibrating thresholds at FP={args.target_fp*100:.1f}%…', file=sys.stderr)
    nm = calibrate(xmed, args.n_noise, fc=DEFAULT_FC)
    thresholds = {k: float(np.percentile(v, 100*(1 - args.target_fp))) for k, v in nm.items()}
    print(f'  single thresh = {thresholds["single"]:.2f}  blind thresh = {thresholds["blind"]:.2f}  '
          f'tgi thresh = {thresholds["tgi"]:.2f}', file=sys.stderr)

    snr_grid = np.arange(args.snr_lo, args.snr_hi + args.snr_step / 2, args.snr_step)
    print(f'Running SNR sweep…', file=sys.stderr)
    rates = detect_rate(snr_grid, args.n_seeds, xmed, thresholds, fc=DEFAULT_FC)

    snr50_s = find_snr50(snr_grid, rates['single'])
    snr50_b = find_snr50(snr_grid, rates['blind'])
    snr50_t = find_snr50(snr_grid, rates['tgi'])
    print(f'\nSNR50 at matched {args.target_fp*100:.1f}% FP rate:', file=sys.stderr)
    print(f'  single-frame      : {snr50_s:+.2f} dB', file=sys.stderr)
    print(f'  blind integration : {snr50_b:+.2f} dB  (gain {snr50_s - snr50_b:+.2f} dB)', file=sys.stderr)
    print(f'  TGI greedy        : {snr50_t:+.2f} dB  (gain {snr50_s - snr50_t:+.2f} dB)', file=sys.stderr)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(snr_grid, rates['single']*100, 'o-', color='#444', linewidth=2,
            label=f'single-frame      (T*={thresholds["single"]:.2f}, SNR50={snr50_s:+.2f} dB)')
    ax.plot(snr_grid, rates['blind']*100,  'D-', color='#070', linewidth=2,
            label=f'blind integration (T*={thresholds["blind"]:.2f}, SNR50={snr50_b:+.2f} dB)')
    ax.plot(snr_grid, rates['tgi']*100,    's-', color='#a40', linewidth=2,
            label=f'TGI greedy        (T*={thresholds["tgi"]:.2f}, SNR50={snr50_t:+.2f} dB)')
    ax.axhline(50, color='#888', linestyle=':', linewidth=0.7)
    ax.set_xlabel('Audio SNR (dB)'); ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_title(f'Experiment C — blind vs greedy integration at matched FP={args.target_fp*100:.1f}%\n'
                 f'(5-frame burst, {args.n_seeds} signal trials/SNR, {args.n_noise} noise trials for calibration)')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right', fontsize=10)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_plot, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'\nPlot: {args.out_plot}', file=sys.stderr)


if __name__ == '__main__':
    main()
