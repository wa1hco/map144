#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment D — TGI with Option A noise-growth-aware margin.

Compares at MATCHED 1% FP rate:
  - single-frame baseline
  - blind integration (force-add all 5 frames; upper-bound for any
    coherent-add algorithm that can find phase)
  - TGI margin='none'         (the broken v1 behaviour)
  - TGI margin='fixed'  M=1.5 (require 1.5x improvement to keep)
  - TGI margin='fixed'  M=1.7 (more conservative)
  - TGI margin='kadapt' safety=1.1 (K-adaptive, 10%% safety above linear)
  - TGI margin='kadapt' safety=1.3

Question: can option A close the gap toward blind integration's +3.38 dB?
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root for map144_app
from map144_app.sq_det import sq_det, slide_sq_det, NSPM, FS, DEFAULT_FC
from algos.tgi import tgi_integrate, TGIConfig
from exp_a_formula_comparison import make_audio_noise, audio_to_analytic
from exp_b_tgi_value import (make_audio_buffer_with_burst, find_strongest_anchor,
                              algo_single_frame)
from exp_c_blind_integration import algo_blind


# ── Algorithm variants under test ───────────────────────────────────────────

VARIANTS = {
    'single':           None,
    'blind':            None,    # special-case
    'tgi_margin_none':  TGIConfig(max_span_frames=4, n_phase_search=8, margin_mode='none'),
    'tgi_fixed_1.5':    TGIConfig(max_span_frames=4, n_phase_search=8, margin_mode='fixed', margin_fixed=1.5),
    'tgi_fixed_1.7':    TGIConfig(max_span_frames=4, n_phase_search=8, margin_mode='fixed', margin_fixed=1.7),
    'tgi_kadapt_1.1':   TGIConfig(max_span_frames=4, n_phase_search=8, margin_mode='kadapt', margin_safety=1.1),
    'tgi_kadapt_1.3':   TGIConfig(max_span_frames=4, n_phase_search=8, margin_mode='kadapt', margin_safety=1.3),
}


def algo_dispatch(variant: str, audio: np.ndarray, anchor_t: float,
                  xmed: float, fc: float = DEFAULT_FC):
    """Returns (detmet_norm, n_kept) for the given variant."""
    if variant == 'single':
        r = algo_single_frame(audio, anchor_t, xmed, fc=fc)
        return r.detmet_norm, 1
    if variant == 'blind':
        return algo_blind(audio, anchor_t, xmed, fc=fc)
    cfg = VARIANTS[variant]
    analytic = audio_to_analytic(audio).astype(np.complex128)
    r = tgi_integrate(analytic, anchor_t, cfg)
    return r.detmet / max(xmed, 1e-30), r.n_kept


def calibrate(xmed: float, n_noise_trials: int, fc: float):
    pre_n = int(0.4 * FS); burst_n = 5 * NSPM; post_n = int(0.4 * FS)
    metrics = {v: [] for v in VARIANTS}
    for k in range(n_noise_trials):
        rng = np.random.default_rng(80000 + k)
        audio = rng.normal(0, 1.0, pre_n + burst_n + post_n).astype(np.float64)
        t0, t1 = pre_n / FS, (pre_n + burst_n) / FS
        anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
        for v in VARIANTS:
            m, _ = algo_dispatch(v, audio, anchor_t, xmed, fc=fc)
            metrics[v].append(m)
        if (k + 1) % 50 == 0:
            print(f'  calibration {k + 1}/{n_noise_trials}', file=sys.stderr)
    return {v: np.array(m) for v, m in metrics.items()}


def detect_rate(snr_grid, n_seeds, xmed, thresholds, fc):
    rates = {v: np.zeros_like(snr_grid) for v in VARIANTS}
    n_kept = {v: np.zeros_like(snr_grid) for v in VARIANTS}
    for i, snr in enumerate(snr_grid):
        seed_base = int(round(5000 * (snr + 50)))
        hits = {v: 0 for v in VARIANTS}
        kept = {v: 0 for v in VARIANTS}
        for k in range(n_seeds):
            audio, t0, t1 = make_audio_buffer_with_burst(5, float(snr), fc=fc, seed=seed_base + k)
            anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
            for v in VARIANTS:
                m, nk = algo_dispatch(v, audio, anchor_t, xmed, fc=fc)
                if m >= thresholds[v]: hits[v] += 1
                kept[v] += nk
        for v in VARIANTS:
            rates[v][i] = hits[v] / n_seeds
            n_kept[v][i] = kept[v] / n_seeds
    return rates, n_kept


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
                    default=repo_root / 'scratch/plots/exp_d_option_a.png')
    args = ap.parse_args()

    print('Computing xmed…', file=sys.stderr)
    noise = make_audio_noise(int(5.0 * FS), seed=0)
    xmed = float(slide_sq_det(audio_to_analytic(noise)).xmed)

    print(f'Calibrating thresholds at FP={args.target_fp*100:.1f}% across {len(VARIANTS)} variants…',
          file=sys.stderr)
    nm = calibrate(xmed, args.n_noise, fc=DEFAULT_FC)
    thresholds = {v: float(np.percentile(nm[v], 100 * (1 - args.target_fp))) for v in VARIANTS}
    print('  Thresholds:', file=sys.stderr)
    for v in VARIANTS:
        print(f'    {v:<22}: T*={thresholds[v]:>10.3f}', file=sys.stderr)

    snr_grid = np.arange(args.snr_lo, args.snr_hi + args.snr_step / 2, args.snr_step)
    print(f'\nSNR sweep…', file=sys.stderr)
    rates, n_kept = detect_rate(snr_grid, args.n_seeds, xmed, thresholds, fc=DEFAULT_FC)

    snr50 = {v: find_snr50(snr_grid, rates[v]) for v in VARIANTS}
    base = snr50['single']
    print(f'\nSNR50 vs single-frame baseline ({base:+.2f} dB):', file=sys.stderr)
    for v in VARIANTS:
        gain = (base - snr50[v]) if not (np.isnan(snr50[v]) or np.isnan(base)) else float('nan')
        n_kept_at_zero = n_kept[v][np.argmin(np.abs(snr_grid))] if v != 'single' else 1.0
        print(f'  {v:<22}: SNR50={snr50[v]:>+6.2f} dB   gain={gain:>+5.2f} dB   '
              f'<n_kept@0dB>={n_kept_at_zero:.1f}', file=sys.stderr)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    style = {
        'single':           ('o-',  '#000', 2.0),
        'blind':            ('D-',  '#070', 2.0),
        'tgi_margin_none':  ('x--', '#a40', 1.4),
        'tgi_fixed_1.5':    ('s-',  '#226', 1.6),
        'tgi_fixed_1.7':    ('s-',  '#48a', 1.6),
        'tgi_kadapt_1.1':   ('^-',  '#a08', 1.6),
        'tgi_kadapt_1.3':   ('^-',  '#d4a', 1.6),
    }
    for v in VARIANTS:
        marker, color, lw = style[v]
        ax.plot(snr_grid, rates[v] * 100, marker, color=color, linewidth=lw,
                label=f'{v}  (T*={thresholds[v]:.1f}, SNR50={snr50[v]:+.2f} dB)')
    ax.axhline(50, color='#888', linestyle=':', linewidth=0.7)
    ax.set_xlabel('Audio SNR (dB)'); ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_title(f'Experiment D — TGI margin variants vs blind & single-frame at FP={args.target_fp*100:.1f}%\n'
                 f'(5-frame burst, {args.n_seeds} signal trials/SNR)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_plot, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'\nPlot: {args.out_plot}', file=sys.stderr)


if __name__ == '__main__':
    main()
