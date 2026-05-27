#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment A — Detector formula comparison: MAP144 vs WSJT-X.

Single-frame, no integration.  Tests whether MAP144's current sq_det
implementation gives different detection sensitivity than WSJT-X's
implementation, ON THE SAME synthetic test bursts.

Quantifies the {FFT size, windowing, combine rule, baseline source,
threshold} differences as a single detection-rate-vs-SNR curve per
algorithm.

Procedure
---------
1. Build a noise-only reference (5 s) → compute baselines:
     WSJT-X xmed    = 25th-percentile of sliding sq_det.detmet across noise
     MAP144 pct25   = 25th-percentile of sliding map144 raw_lin across noise
   Both algorithms get a fair, fixed noise reference for the trials.

2. For each SNR step (default −15 to +5 dB in 1 dB steps):
   For each trial seed (default 50):
     a. Generate ONE MSK144 frame of audio at fc=1500 Hz with controlled SNR
     b. Convert to both algorithms' input domains:
          WSJT-X: analytic at audio carrier (Hilbert)
          MAP144: complex baseband at DC (Hilbert + mix-down)
     c. Run each algorithm's per-frame sq_det
     d. Threshold:
          WSJT-X: detmet/xmed >= 3.0 (linear ratio)
          MAP144: 10*log10(raw_lin/pct25) >= 3.5 (dB form)
     e. Record per-trial detection result + raw metric values

3. Aggregate detection rate per (SNR, algorithm) and save:
     - per-trial CSV  → scratch/results/exp_a_per_trial.csv
     - aggregate CSV  → scratch/results/exp_a_summary.csv
     - plot           → scratch/plots/exp_a_formula_comparison.png

Output ANSWER
-------------
For each algorithm: SNR50 = SNR (dB) at which detection rate crosses 50%.
The DIFFERENCE between the two SNR50 values quantifies the formula
sensitivity gap.

Notes on SNR definition
-----------------------
We use audio amplitude ratio: ``SNR_dB = 20·log10(A_signal / σ_noise)``
where A is the cosine amplitude of the audio carrier and σ is the per-
quadrature noise std.  This is NOT the WSJT-X 2500 Hz BW reference;
it's an internal SNR convention for the experiment.  Comparisons
across algorithms are valid; absolute calibration to WSJT-X's reported
SNR is not.
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
from algos.map144_sq_det import (
    map144_sq_det_per_frame, slide_map144_sq_det,
    audio_to_complex_baseband, CH_DETECT_SIZE, DETECT_THRESHOLD_DB,
)


# ── Synthetic signal generators ─────────────────────────────────────────────

def make_audio_msk_frame(snr_db: float, fc: float = DEFAULT_FC,
                        n_samples: int = NSPM, seed: int = 0,
                        ) -> np.ndarray:
    """Generate ONE NSPM-sample MSK144 audio frame at audio carrier fc,
    with additive Gaussian noise calibrated to give the requested SNR.

    SNR_dB = 20·log10(A_signal / σ_noise), where A is the cosine amplitude
    of the audio carrier and σ is the noise std.  Real-audio output.
    """
    rng = np.random.default_rng(seed)
    NSPS = 6
    n_symbols = (n_samples + NSPS - 1) // NSPS
    symbols = rng.choice([-1, 1], size=n_symbols)
    freqs = np.repeat(symbols * 500.0, NSPS)[:n_samples] + fc
    phases = 2 * np.pi * np.cumsum(freqs) / FS
    # Signal amplitude derived from requested SNR (we hold noise σ=1).
    A = 10 ** (snr_db / 20.0)
    signal = A * np.cos(phases)
    noise = rng.normal(0, 1.0, n_samples)
    return (signal + noise).astype(np.float64)


def make_audio_noise(n_samples: int, seed: int = 0) -> np.ndarray:
    """Pure noise audio (σ=1) for baseline reference."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, n_samples).astype(np.float64)


def audio_to_analytic(audio_real: np.ndarray) -> np.ndarray:
    """Hilbert transform → analytic complex signal at audio carrier."""
    from scipy.signal import hilbert
    return hilbert(audio_real.astype(np.float64))


# ── Experiment ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    snr_db: float
    seed: int
    wsjtx_detmet: float
    wsjtx_xmed: float
    wsjtx_detmet_norm: float
    wsjtx_detected: bool
    map144_raw_lin: float
    map144_pct25: float
    map144_metric_db: float
    map144_detected: bool


def compute_baselines(noise_seconds: float = 5.0, seed: int = 0,
                      fc: float = DEFAULT_FC) -> tuple[float, float]:
    """Compute WSJT-X xmed and MAP144 pct25 from a long noise-only segment.

    Returns (wsjtx_xmed, map144_pct25).  Both algorithms get a stable,
    signal-free baseline so the per-trial detection thresholds are fair.
    """
    n = int(noise_seconds * FS)
    noise = make_audio_noise(n, seed=seed)
    # WSJT-X: needs analytic input at audio carrier
    analytic = audio_to_analytic(noise)
    wsjtx_trace = slide_sq_det(analytic)
    wsjtx_xmed = float(wsjtx_trace.xmed)
    # MAP144: needs complex baseband
    bb = audio_to_complex_baseband(noise, fc_hz=fc)
    map144_trace = slide_map144_sq_det(bb)
    map144_pct25 = float(map144_trace.pct25)
    return wsjtx_xmed, map144_pct25


def run_trial(snr_db: float, seed: int, fc: float,
              wsjtx_xmed: float, map144_pct25: float) -> TrialResult:
    audio = make_audio_msk_frame(snr_db, fc=fc, n_samples=NSPM, seed=seed)
    # WSJT-X path
    analytic = audio_to_analytic(audio)
    w = sq_det(analytic, fc=fc)
    detmet_norm = w.detmet / max(wsjtx_xmed, 1e-30)
    wsjtx_detected = bool(detmet_norm >= DETECT_THRESHOLD_NORM)
    # MAP144 path: convert audio to baseband, then take CH_DETECT_SIZE samples
    bb = audio_to_complex_baseband(audio, fc_hz=fc)
    if len(bb) < CH_DETECT_SIZE:
        # Should not happen since NSPM=864 > CH_DETECT_SIZE=512; fits fine.
        raise RuntimeError(f"bb length {len(bb)} < CH_DETECT_SIZE")
    m = map144_sq_det_per_frame(bb[:CH_DETECT_SIZE])
    metric_db = 10.0 * np.log10(max(m.raw_lin, 1e-30) / max(map144_pct25, 1e-30))
    map144_detected = bool(metric_db >= DETECT_THRESHOLD_DB)
    return TrialResult(
        snr_db=snr_db, seed=seed,
        wsjtx_detmet=w.detmet, wsjtx_xmed=wsjtx_xmed,
        wsjtx_detmet_norm=detmet_norm, wsjtx_detected=wsjtx_detected,
        map144_raw_lin=m.raw_lin, map144_pct25=map144_pct25,
        map144_metric_db=metric_db, map144_detected=map144_detected,
    )


def run_experiment(snr_lo: float, snr_hi: float, snr_step: float,
                   n_seeds: int, fc: float = DEFAULT_FC,
                   noise_seconds: float = 5.0,
                   ) -> tuple[list[TrialResult], float, float]:
    """Run the SNR sweep and return (per-trial results, xmed, pct25)."""
    print('Computing baselines from {:.1f} s of noise…'.format(noise_seconds),
          file=sys.stderr)
    wsjtx_xmed, map144_pct25 = compute_baselines(noise_seconds, seed=0, fc=fc)
    print(f'  WSJT-X xmed   : {wsjtx_xmed:.6f}', file=sys.stderr)
    print(f'  MAP144 pct25  : {map144_pct25:.6f}', file=sys.stderr)

    snr_grid = np.arange(snr_lo, snr_hi + snr_step / 2, snr_step)
    print(f'SNR sweep: {len(snr_grid)} steps × {n_seeds} trials = '
          f'{len(snr_grid) * n_seeds} total trials', file=sys.stderr)

    results: list[TrialResult] = []
    for snr_db in snr_grid:
        # Use a deterministic seed offset per SNR step for reproducibility
        seed_base = int(round(1000 * (snr_db + 50)))   # avoid collisions
        for k in range(n_seeds):
            results.append(run_trial(float(snr_db), seed_base + k, fc,
                                     wsjtx_xmed, map144_pct25))
    return results, wsjtx_xmed, map144_pct25


def aggregate(results: list[TrialResult]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (snr_grid, wsjtx_rate, map144_rate)."""
    snr_set = sorted({r.snr_db for r in results})
    snr_grid = np.array(snr_set)
    wsjtx_rate = np.zeros_like(snr_grid)
    map144_rate = np.zeros_like(snr_grid)
    for i, snr in enumerate(snr_grid):
        bin_results = [r for r in results if r.snr_db == snr]
        n = len(bin_results)
        wsjtx_rate[i]  = sum(r.wsjtx_detected for r in bin_results) / n
        map144_rate[i] = sum(r.map144_detected for r in bin_results) / n
    return snr_grid, wsjtx_rate, map144_rate


def find_snr50(snr_grid: np.ndarray, rate: np.ndarray) -> float:
    """Linear interpolation: SNR at which detection rate crosses 0.5."""
    if rate.max() < 0.5:
        return float('nan')
    if rate.min() >= 0.5:
        return float(snr_grid[0])
    for i in range(len(snr_grid) - 1):
        if rate[i] < 0.5 <= rate[i + 1]:
            x0, x1 = snr_grid[i], snr_grid[i + 1]
            y0, y1 = rate[i], rate[i + 1]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float('nan')


# ── Output ──────────────────────────────────────────────────────────────────

def write_per_trial_csv(results: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_db', 'seed',
                    'wsjtx_detmet', 'wsjtx_xmed', 'wsjtx_detmet_norm', 'wsjtx_detected',
                    'map144_raw_lin', 'map144_pct25', 'map144_metric_db', 'map144_detected'])
        for r in results:
            w.writerow([f'{r.snr_db:.2f}', r.seed,
                        f'{r.wsjtx_detmet:.6e}', f'{r.wsjtx_xmed:.6e}',
                        f'{r.wsjtx_detmet_norm:.4f}', int(r.wsjtx_detected),
                        f'{r.map144_raw_lin:.6e}', f'{r.map144_pct25:.6e}',
                        f'{r.map144_metric_db:.4f}', int(r.map144_detected)])


def write_summary_csv(snr_grid: np.ndarray, wsjtx_rate: np.ndarray,
                      map144_rate: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_db', 'wsjtx_detect_rate', 'map144_detect_rate'])
        for i, snr in enumerate(snr_grid):
            w.writerow([f'{snr:.2f}', f'{wsjtx_rate[i]:.4f}', f'{map144_rate[i]:.4f}'])


def make_plot(snr_grid: np.ndarray, wsjtx_rate: np.ndarray,
              map144_rate: np.ndarray, snr50_wsjtx: float, snr50_map144: float,
              path: Path, n_seeds: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(snr_grid, wsjtx_rate * 100, 'o-', color='#226', linewidth=2,
            label=f'WSJT-X formula  (SNR50 = {snr50_wsjtx:+.2f} dB)')
    ax.plot(snr_grid, map144_rate * 100, 's-', color='#a40', linewidth=2,
            label=f'MAP144 formula  (SNR50 = {snr50_map144:+.2f} dB)')
    ax.axhline(50, color='#888', linestyle=':', linewidth=0.7)
    if not np.isnan(snr50_wsjtx):  ax.axvline(snr50_wsjtx, color='#226', linestyle=':', linewidth=0.7)
    if not np.isnan(snr50_map144): ax.axvline(snr50_map144, color='#a40', linestyle=':', linewidth=0.7)
    delta = snr50_map144 - snr50_wsjtx
    title = (f'Experiment A — single-frame detection rate vs SNR\n'
             f'(MSK144 audio at fc=1500, {n_seeds} trials per SNR step)\n'
             f'Formula sensitivity gap (MAP144 - WSJT-X) = {delta:+.2f} dB')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Audio SNR (dB)')
    ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snr-lo',     type=float, default=-15.0)
    ap.add_argument('--snr-hi',     type=float, default=+5.0)
    ap.add_argument('--snr-step',   type=float, default=1.0)
    ap.add_argument('--n-seeds',    type=int,   default=50,
                    help='Trials per SNR step (default 50)')
    ap.add_argument('--noise-seconds', type=float, default=5.0,
                    help='Length of noise-only baseline reference (default 5.0)')

    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--out-trials', type=Path,
                    default=repo_root / 'scratch/results/exp_a_per_trial.csv')
    ap.add_argument('--out-summary', type=Path,
                    default=repo_root / 'scratch/results/exp_a_summary.csv')
    ap.add_argument('--out-plot', type=Path,
                    default=repo_root / 'scratch/plots/exp_a_formula_comparison.png')
    args = ap.parse_args(argv)

    results, wsjtx_xmed, map144_pct25 = run_experiment(
        snr_lo=args.snr_lo, snr_hi=args.snr_hi, snr_step=args.snr_step,
        n_seeds=args.n_seeds, noise_seconds=args.noise_seconds,
    )
    snr_grid, w_rate, m_rate = aggregate(results)
    snr50_w = find_snr50(snr_grid, w_rate)
    snr50_m = find_snr50(snr_grid, m_rate)

    print(file=sys.stderr)
    print('=' * 55, file=sys.stderr)
    print('Experiment A — Detection rate vs SNR (single-frame)', file=sys.stderr)
    print('=' * 55, file=sys.stderr)
    print(f'  {"SNR(dB)":>7}  {"WSJT-X":>8}  {"MAP144":>8}', file=sys.stderr)
    for i, snr in enumerate(snr_grid):
        print(f'  {snr:>+7.1f}  {w_rate[i]*100:>7.1f}%  {m_rate[i]*100:>7.1f}%',
              file=sys.stderr)
    print(file=sys.stderr)
    print(f'SNR50 (50% detection rate):', file=sys.stderr)
    print(f'  WSJT-X formula : {snr50_w:+.2f} dB', file=sys.stderr)
    print(f'  MAP144 formula : {snr50_m:+.2f} dB', file=sys.stderr)
    print(f'  Gap (MAP - WSJT): {snr50_m - snr50_w:+.2f} dB '
          f'(positive = MAP144 less sensitive)', file=sys.stderr)

    write_per_trial_csv(results, args.out_trials)
    write_summary_csv(snr_grid, w_rate, m_rate, args.out_summary)
    make_plot(snr_grid, w_rate, m_rate, snr50_w, snr50_m, args.out_plot,
              n_seeds=args.n_seeds)

    print(file=sys.stderr)
    print(f'Wrote:', file=sys.stderr)
    print(f'  per-trial CSV : {args.out_trials}', file=sys.stderr)
    print(f'  summary CSV   : {args.out_summary}', file=sys.stderr)
    print(f'  plot          : {args.out_plot}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
