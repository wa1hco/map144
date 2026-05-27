#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment B — TGI integration value vs single-frame baseline.

Both algorithms use the WSJT-X sq_det formula (apples-to-apples on the
detector — Experiment A already isolated the formula change).  This
experiment measures how much MORE detection sensitivity TGI's greedy
multi-frame integration adds.

Variants compared
-----------------
1. WSJTX_single   — single-frame baseline (one strongest frame, no integration)
2. TGI_naive      — n_phase_search=1, criterion='self' detmet2
                    (no phase compensation; baseline TGI to expose phase issue)
3. TGI_search     — n_phase_search=8, criterion='self' detmet2
                    (with phase search; the proposed v1 algorithm)

Procedure
---------
1. Compute WSJT-X xmed baseline from noise-only segment (5 s).
2. For each SNR step (default −15 to +5 dB):
   For each seed:
     a. Generate a MULTI-frame burst: 5 frames of MSK at requested SNR
     b. Build a long buffer of (pre-noise + burst + post-noise) ≈ 1.7 s
     c. Find the strongest single-frame trigger position in the burst region
     d. WSJTX_single: detmet/xmed at that position
     e. TGI_naive:    starts at trigger, n_phase_search=1
     f. TGI_search:   starts at trigger, n_phase_search=8
     g. Record per-trial: detection result + integrated detmet/xmed + n_kept
3. Aggregate: detection rate vs SNR per algorithm; n_kept vs SNR for TGI
4. Save: per-trial CSV, summary CSV, two plots (detection-rate, n_kept)

Output ANSWER
-------------
- SNR50 per algorithm.  SNR50_TGI - SNR50_baseline = integration gain (dB).
- n_kept distribution per SNR — TGI's adaptation to signal level.
- False positive rate of TGI on the noise-only segment (a separate scalar).
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
from map144_app.sq_det import (
    sq_det, slide_sq_det, NSPM, FS, DEFAULT_FC, DETECT_THRESHOLD_NORM,
)
from algos.tgi import tgi_integrate, TGIConfig
from exp_a_formula_comparison import (
    make_audio_msk_frame, make_audio_noise, audio_to_analytic,
)


# ── Multi-frame test signal ─────────────────────────────────────────────────

def make_audio_msk_burst(n_frames: int, snr_db: float, fc: float = DEFAULT_FC,
                         seed: int = 0) -> np.ndarray:
    """N consecutive MSK frames at the requested SNR — same data pattern
    repeated, so the carrier is continuous-phase and frames sum coherently
    (after phase compensation for any unbalanced data accumulation)."""
    rng = np.random.default_rng(seed)
    NSPS = 6
    n_total = n_frames * NSPM
    # Use SAME symbols across all frames (matches WSJT-X repeated-message TX)
    n_symbols_per_frame = NSPM // NSPS
    symbols_one = rng.choice([-1, 1], size=n_symbols_per_frame)
    symbols_full = np.tile(symbols_one, n_frames)
    freqs = np.repeat(symbols_full * 500.0, NSPS)[:n_total] + fc
    phases = 2 * np.pi * np.cumsum(freqs) / FS
    A = 10 ** (snr_db / 20.0)
    signal = A * np.cos(phases)
    noise = rng.normal(0, 1.0, n_total)
    return (signal + noise).astype(np.float64)


def make_audio_buffer_with_burst(n_frames: int, snr_db: float,
                                  pre_seconds: float = 0.4,
                                  post_seconds: float = 0.4,
                                  fc: float = DEFAULT_FC, seed: int = 0,
                                  ) -> tuple[np.ndarray, float, float]:
    """Build a (pre-noise) + (burst) + (post-noise) audio buffer.
    Returns (audio, burst_t_start, burst_t_end) — both times in seconds.
    """
    pre_n  = int(pre_seconds * FS)
    post_n = int(post_seconds * FS)
    # Use signal seed for the burst, deterministic-different seeds for noise pads
    pre  = make_audio_noise(pre_n,  seed=seed * 7 + 1)
    burst = make_audio_msk_burst(n_frames, snr_db, fc=fc, seed=seed)
    post = make_audio_noise(post_n, seed=seed * 7 + 2)
    audio = np.concatenate([pre, burst, post]).astype(np.float64)
    return audio, pre_n / FS, (pre_n + n_frames * NSPM) / FS


# ── Algorithm wrappers (uniform return) ─────────────────────────────────────

@dataclass
class AlgoResult:
    detmet:        float    # final integrated detmet
    detmet_norm:   float    # detmet / xmed (using shared xmed reference)
    detected:      bool     # detmet_norm >= 3.0
    n_kept:        int      # frames in the integration (1 = no integration)


def algo_single_frame(audio: np.ndarray, anchor_t: float,
                      xmed: float, fc: float = DEFAULT_FC) -> AlgoResult:
    """Baseline: just sq_det at the anchor frame."""
    analytic = audio_to_analytic(audio)
    n_anchor = int(anchor_t * FS)
    n_anchor = max(0, min(len(analytic) - NSPM, n_anchor))
    r = sq_det(analytic[n_anchor:n_anchor + NSPM], fc=fc)
    norm = r.detmet / max(xmed, 1e-30)
    return AlgoResult(detmet=r.detmet, detmet_norm=norm,
                      detected=norm >= DETECT_THRESHOLD_NORM, n_kept=1)


def algo_tgi(audio: np.ndarray, anchor_t: float, xmed: float,
             n_phase_search: int, fc: float = DEFAULT_FC,
             max_span_frames: int = 6) -> AlgoResult:
    """TGI variant.  Uses 'self' criterion (current best understanding)."""
    analytic = audio_to_analytic(audio)
    cfg = TGIConfig(fc=fc, max_span_frames=max_span_frames,
                    n_phase_search=n_phase_search)
    r = tgi_integrate(analytic, anchor_t, cfg)
    norm = r.detmet / max(xmed, 1e-30)
    return AlgoResult(detmet=r.detmet, detmet_norm=norm,
                      detected=norm >= DETECT_THRESHOLD_NORM, n_kept=r.n_kept)


def find_strongest_anchor(audio: np.ndarray, t_lo: float, t_hi: float,
                          fc: float = DEFAULT_FC) -> float:
    """Find the slide-position with highest single-frame detmet within
    [t_lo, t_hi] seconds.  Returns the anchor time."""
    analytic = audio_to_analytic(audio)
    trace = slide_sq_det(analytic, fc=fc)
    in_range = (trace.t_centers >= t_lo) & (trace.t_centers <= t_hi)
    if not in_range.any():
        return (t_lo + t_hi) / 2
    detmet_in_range = trace.detmet_raw.copy()
    detmet_in_range[~in_range] = -np.inf
    best_idx = int(np.argmax(detmet_in_range))
    return float(trace.t_centers[best_idx])


# ── Experiment ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    snr_db: float
    seed: int
    anchor_t: float
    single_norm:  float; single_detected:  bool
    naive_norm:   float; naive_detected:   bool;  naive_n_kept:   int
    search_norm:  float; search_detected:  bool;  search_n_kept:  int


def run_trial(snr_db: float, seed: int, xmed: float, n_frames: int = 5,
              fc: float = DEFAULT_FC) -> TrialResult:
    audio, t0, t1 = make_audio_buffer_with_burst(n_frames, snr_db,
                                                  fc=fc, seed=seed)
    anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
    s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
    n = algo_tgi(audio, anchor_t, xmed, n_phase_search=1, fc=fc)
    p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
    return TrialResult(
        snr_db=snr_db, seed=seed, anchor_t=anchor_t,
        single_norm=s.detmet_norm, single_detected=s.detected,
        naive_norm=n.detmet_norm,  naive_detected=n.detected,  naive_n_kept=n.n_kept,
        search_norm=p.detmet_norm, search_detected=p.detected, search_n_kept=p.n_kept,
    )


def measure_noise_false_positives(xmed: float, n_trials: int = 100,
                                  fc: float = DEFAULT_FC) -> dict[str, float]:
    """Run all three algorithms on pure-noise buffers; report false-positive
    rates for each."""
    pre_n = int(0.4 * FS)
    burst_n = 5 * NSPM
    post_n = int(0.4 * FS)
    fp = {'single': 0, 'naive': 0, 'search': 0}
    for k in range(n_trials):
        # Pure noise buffer of the same length as a burst-bearing buffer
        rng = np.random.default_rng(99000 + k)
        audio = rng.normal(0, 1.0, pre_n + burst_n + post_n).astype(np.float64)
        # Find strongest single-frame candidate within the "would-be burst" region
        t0 = pre_n / FS; t1 = (pre_n + burst_n) / FS
        anchor_t = find_strongest_anchor(audio, t0, t1, fc=fc)
        s = algo_single_frame(audio, anchor_t, xmed, fc=fc)
        n = algo_tgi(audio, anchor_t, xmed, n_phase_search=1, fc=fc)
        p = algo_tgi(audio, anchor_t, xmed, n_phase_search=8, fc=fc)
        if s.detected: fp['single'] += 1
        if n.detected: fp['naive']  += 1
        if p.detected: fp['search'] += 1
    return {k: v / n_trials for k, v in fp.items()}


def aggregate(results: list[TrialResult]):
    """Return (snr_grid, single_rate, naive_rate, search_rate, mean_naive_n_kept, mean_search_n_kept)."""
    snr_set = sorted({r.snr_db for r in results})
    grid = np.array(snr_set)
    s_rate = np.zeros_like(grid); n_rate = np.zeros_like(grid); p_rate = np.zeros_like(grid)
    n_kept_naive  = np.zeros_like(grid); n_kept_search = np.zeros_like(grid)
    for i, snr in enumerate(grid):
        bin_results = [r for r in results if r.snr_db == snr]
        n_in = len(bin_results)
        s_rate[i] = sum(r.single_detected for r in bin_results) / n_in
        n_rate[i] = sum(r.naive_detected  for r in bin_results) / n_in
        p_rate[i] = sum(r.search_detected for r in bin_results) / n_in
        n_kept_naive[i]  = float(np.mean([r.naive_n_kept  for r in bin_results]))
        n_kept_search[i] = float(np.mean([r.search_n_kept for r in bin_results]))
    return grid, s_rate, n_rate, p_rate, n_kept_naive, n_kept_search


def find_snr50(snr_grid, rate):
    if rate.max() < 0.5: return float('nan')
    if rate.min() >= 0.5: return float(snr_grid[0])
    for i in range(len(snr_grid) - 1):
        if rate[i] < 0.5 <= rate[i + 1]:
            x0, x1, y0, y1 = snr_grid[i], snr_grid[i+1], rate[i], rate[i+1]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float('nan')


# ── Output ──────────────────────────────────────────────────────────────────

def write_per_trial_csv(results: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_db', 'seed', 'anchor_t',
                    'single_norm', 'single_detected',
                    'naive_norm', 'naive_detected', 'naive_n_kept',
                    'search_norm', 'search_detected', 'search_n_kept'])
        for r in results:
            w.writerow([f'{r.snr_db:+.1f}', r.seed, f'{r.anchor_t:.4f}',
                        f'{r.single_norm:.4f}', int(r.single_detected),
                        f'{r.naive_norm:.4f}',  int(r.naive_detected),  r.naive_n_kept,
                        f'{r.search_norm:.4f}', int(r.search_detected), r.search_n_kept])


def write_summary_csv(grid, s, n, p, kn, ks, fp_rates: dict[str, float],
                      path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_db', 'single_rate', 'naive_rate', 'search_rate',
                    'naive_mean_n_kept', 'search_mean_n_kept'])
        for i, snr in enumerate(grid):
            w.writerow([f'{snr:+.1f}', f'{s[i]:.4f}', f'{n[i]:.4f}', f'{p[i]:.4f}',
                        f'{kn[i]:.2f}', f'{ks[i]:.2f}'])
        w.writerow([])
        w.writerow(['# false-positive rates on pure-noise buffers'])
        w.writerow(['algorithm', 'fp_rate'])
        for k, v in fp_rates.items():
            w.writerow([k, f'{v:.4f}'])


def make_plots(grid, s, n, p, kn, ks, snr50_s, snr50_n, snr50_p,
               fp_rates: dict[str, float], n_seeds: int,
               plot_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(grid, s * 100, 'o-', color='#444', linewidth=2,
             label=f'single-frame baseline  (SNR50 = {snr50_s:+.2f} dB,  FP={fp_rates["single"]*100:.0f}%)')
    ax1.plot(grid, n * 100, 's-', color='#a40', linewidth=2,
             label=f'TGI naive (no phase srch)  (SNR50 = {snr50_n:+.2f} dB,  FP={fp_rates["naive"]*100:.0f}%)')
    ax1.plot(grid, p * 100, 'D-', color='#226', linewidth=2,
             label=f'TGI w/ phase search       (SNR50 = {snr50_p:+.2f} dB,  FP={fp_rates["search"]*100:.0f}%)')
    ax1.axhline(50, color='#888', linestyle=':', linewidth=0.7)
    if not np.isnan(snr50_s): ax1.axvline(snr50_s, color='#444', linestyle=':', linewidth=0.6)
    if not np.isnan(snr50_n): ax1.axvline(snr50_n, color='#a40', linestyle=':', linewidth=0.6)
    if not np.isnan(snr50_p): ax1.axvline(snr50_p, color='#226', linestyle=':', linewidth=0.6)
    ax1.set_xlabel('Audio SNR (dB)')
    ax1.set_ylabel('Detection rate (%)')
    ax1.set_ylim(-5, 105)
    ax1.set_title(f'Experiment B — TGI vs single-frame detection rate\n'
                  f'(WSJT-X sq_det formula, 5-frame burst, {n_seeds} trials/SNR)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # Right panel: n_kept vs SNR
    ax2.plot(grid, kn, 's-', color='#a40', linewidth=2,
             label='TGI naive (mean n_kept)')
    ax2.plot(grid, ks, 'D-', color='#226', linewidth=2,
             label='TGI w/ phase search (mean n_kept)')
    ax2.axhline(1, color='#888', linestyle=':', linewidth=0.7,
                label='1 (single-frame equivalent)')
    ax2.axhline(5, color='#0a0', linestyle=':', linewidth=0.7,
                label='5 (burst length, max if all kept)')
    ax2.set_xlabel('Audio SNR (dB)')
    ax2.set_ylabel('Mean frames kept by TGI')
    ax2.set_ylim(0, 7)
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
    ap.add_argument('--snr-lo',     type=float, default=-15.0)
    ap.add_argument('--snr-hi',     type=float, default=+5.0)
    ap.add_argument('--snr-step',   type=float, default=1.0)
    ap.add_argument('--n-seeds',    type=int,   default=50)
    ap.add_argument('--n-frames',   type=int,   default=5)
    ap.add_argument('--noise-seconds', type=float, default=5.0)
    ap.add_argument('--n-fp-trials',   type=int,   default=100)
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--out-trials', type=Path,
                    default=repo_root / 'scratch/results/exp_b_per_trial.csv')
    ap.add_argument('--out-summary', type=Path,
                    default=repo_root / 'scratch/results/exp_b_summary.csv')
    ap.add_argument('--out-plot', type=Path,
                    default=repo_root / 'scratch/plots/exp_b_tgi_value.png')
    args = ap.parse_args(argv)

    print(f'Computing WSJT-X xmed baseline ({args.noise_seconds} s of noise)…',
          file=sys.stderr)
    n = int(args.noise_seconds * FS)
    noise = make_audio_noise(n, seed=0)
    analytic = audio_to_analytic(noise)
    xmed = float(slide_sq_det(analytic).xmed)
    print(f'  xmed = {xmed:.6f}', file=sys.stderr)

    snr_grid = np.arange(args.snr_lo, args.snr_hi + args.snr_step / 2, args.snr_step)
    print(f'SNR sweep: {len(snr_grid)} steps × {args.n_seeds} trials × 3 algorithms',
          file=sys.stderr)

    results: list[TrialResult] = []
    for snr_db in snr_grid:
        seed_base = int(round(2000 * (snr_db + 50)))
        for k in range(args.n_seeds):
            results.append(run_trial(float(snr_db), seed_base + k, xmed,
                                      n_frames=args.n_frames))

    grid, s, n_rate, p, kn, ks = aggregate(results)
    snr50_s = find_snr50(grid, s)
    snr50_n = find_snr50(grid, n_rate)
    snr50_p = find_snr50(grid, p)

    print('Measuring false-positive rates on noise…', file=sys.stderr)
    fp = measure_noise_false_positives(xmed, n_trials=args.n_fp_trials)

    print(file=sys.stderr)
    print('=' * 65, file=sys.stderr)
    print('Experiment B — TGI integration value (vs single-frame baseline)', file=sys.stderr)
    print('=' * 65, file=sys.stderr)
    print(f'  {"SNR(dB)":>7}  {"single":>7}  {"TGI_naive":>9}  {"TGI_search":>10}  '
          f'{"naive_n":>7}  {"search_n":>8}', file=sys.stderr)
    for i, snr in enumerate(grid):
        print(f'  {snr:>+7.1f}  {s[i]*100:>6.1f}%  {n_rate[i]*100:>8.1f}%  '
              f'{p[i]*100:>9.1f}%  {kn[i]:>7.2f}  {ks[i]:>8.2f}', file=sys.stderr)
    print(file=sys.stderr)
    print(f'SNR50 (50% detection):', file=sys.stderr)
    print(f'  single-frame baseline : {snr50_s:+.2f} dB', file=sys.stderr)
    print(f'  TGI naive             : {snr50_n:+.2f} dB  (gain {snr50_s - snr50_n:+.2f} dB)', file=sys.stderr)
    print(f'  TGI w/ phase search   : {snr50_p:+.2f} dB  (gain {snr50_s - snr50_p:+.2f} dB)', file=sys.stderr)
    print(file=sys.stderr)
    print(f'False-positive rates (pure-noise buffers, n={args.n_fp_trials}):', file=sys.stderr)
    print(f'  single-frame baseline : {fp["single"]*100:>5.1f}%', file=sys.stderr)
    print(f'  TGI naive             : {fp["naive"]*100:>5.1f}%', file=sys.stderr)
    print(f'  TGI w/ phase search   : {fp["search"]*100:>5.1f}%', file=sys.stderr)

    write_per_trial_csv(results, args.out_trials)
    write_summary_csv(grid, s, n_rate, p, kn, ks, fp, args.out_summary)
    make_plots(grid, s, n_rate, p, kn, ks, snr50_s, snr50_n, snr50_p, fp,
               args.n_seeds, args.out_plot)

    print(file=sys.stderr)
    print(f'Wrote:', file=sys.stderr)
    print(f'  per-trial CSV : {args.out_trials}', file=sys.stderr)
    print(f'  summary CSV   : {args.out_summary}', file=sys.stderr)
    print(f'  plot          : {args.out_plot}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
