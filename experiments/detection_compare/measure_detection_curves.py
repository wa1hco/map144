#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Measure detection probability vs SNR for three detectors at matched FA rate.

Detectors:
    sq_det              — WSJT-X-faithful, sliding window per 18 ms
    tfmf_1block         — TFMF matched filter, single 8-sym sync sub-block
    tfmf_2block         — TFMF matched filter, two sub-blocks coherently summed

Methodology (per docs/detection_test_protocol.md §4-§6):
    1. Calibrate each detector's threshold on pure-noise IQ for a target FA rate
       (10 false alarms per 15-s period).
    2. For each SNR in the grid, generate N independent trials: one ping at the
       target (time, freq, SNR), fresh noise seed.  Run each detector and
       record whether it caught the ping within tolerance.
    3. Output CSV with per-trial detection results.

Output: experiments/detection_compare/detection_curves.csv

Plot via plot_detection_curves.py.
"""
from __future__ import annotations

import sys
import csv
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.signal import decimate

from experiments.gpu_tfmf import tfmf, sync_template, synthetic
from map144_app.sq_det import sq_det, NSPM

# Configuration
SNR_GRID_DB    = list(range(-12, 6))   # -12 to +5 dB
N_TRIALS       = 50

# FA budget — system-level 10 per 15-s.  TFMF spreads its FAs across the
# full ±24 kHz band, so it gets the full budget (10/15s).  sq_det in
# production runs on 40 channels, so per-channel budget = 10/40 = 0.25.
# We're testing sq_det on the single channel where the signal lives, so
# we calibrate it at the per-channel rate to match production fairness.
TARGET_FA_15S_TFMF  = 10.0
TARGET_FA_15S_SQDET = 0.25

TRUTH_FREQ_HZ  = 1500.0                # MSK144 audio carrier convention
TRUTH_TIME_S   = 5.0                   # fixed time within 15-s window
TIME_TOL_S     = 0.5
FREQ_TOL_HZ    = 100.0    # tightened from 500 to remove sq_det noise-baseline artifact
NOISE_SIGMA    = 1.0
DURATION_S     = 15.0
SAMPLE_RATE    = 48_000
N_WORKERS      = 8                     # parallel worker processes

OUT_DIR = REPO / 'experiments' / 'detection_compare'
OUT_CSV = OUT_DIR / 'detection_curves.csv'


# ── Detector wrappers ───────────────────────────────────────────────────────


def _run_tfmf(iq, template, threshold_db: float,
              two_subblock_sum: bool,
              k_frame: int = 1) -> list[tuple[float, float, float]]:
    """Returns [(time_s, freq_hz, snr_db), ...]"""
    cfg = tfmf.TFMFConfig(
        sample_rate_hz=SAMPLE_RATE,
        stride_samples=24,
        threshold_db=threshold_db,
        two_subblock_sum=two_subblock_sum,
        parabolic_freq_interp=False,
        k_frame_integration=k_frame,
    )
    cands = tfmf.detect(iq, template, cfg, use_gpu=False)
    return [(c.time_s, c.freq_hz, c.snr_db) for c in cands]


def _run_sq_det(iq, threshold_lin: float,
                assumed_fc_hz: float = TRUTH_FREQ_HZ
                ) -> list[tuple[float, float, float]]:
    """Sliding-window sq_det.  Threshold is detmet_norm (linear).
    Returns [(time_s, freq_hz, detmet_norm), ...]"""
    audio_12k = decimate(iq, 4, ftype='fir', zero_phase=True).astype(np.complex128)
    hop = NSPM // 4
    detmets, ferr_avgs = [], []
    for start in range(0, audio_12k.size - NSPM + 1, hop):
        r = sq_det(audio_12k[start:start + NSPM], fc=assumed_fc_hz)
        detmets.append(float(r.detmet))
        ferr_avgs.append(float((r.ferr_lo + r.ferr_hi) / 2.0))
    detmets = np.array(detmets, dtype=np.float32)
    xmed = float(np.percentile(detmets, 25))
    if xmed <= 0:
        xmed = 1e-30
    detmet_norm = detmets / xmed
    out = []
    for i, m in enumerate(detmet_norm):
        if m < threshold_lin:
            continue
        time_s = (i * hop) / 12_000.0
        freq_hz = assumed_fc_hz + ferr_avgs[i]
        out.append((time_s, freq_hz, float(m)))
    return out


# ── Threshold calibration ───────────────────────────────────────────────────


def calibrate_tfmf_threshold(template, two_subblock_sum: bool,
                              target_fa_15s: float = TARGET_FA_15S_TFMF,
                              n_runs: int = 5,
                              seed_base: int = 10_000,
                              k_frame: int = 1) -> float:
    """Find TFMF threshold (dB) giving target_fa false alarms per 15-s noise-only.

    Direct surface computation (bypasses Candidate-object construction).  Uses
    the same surface + per-bin-median + 20·log10 chain as ``tfmf.detect``, so
    the threshold value is interchangeable.
    """
    all_top_snr_db = []  # store only top per run, not all 5.76M cells
    target_total = max(int(round(target_fa_15s * n_runs)), 1)
    for run_idx in range(n_runs):
        n = synthetic.noise_only(duration_s=DURATION_S, noise_sigma=NOISE_SIGMA,
                                 seed=seed_base + run_idx)
        surface = tfmf.compute_tf_surface_cpu(
            n.iq, template, stride_samples=24, fft_length=None,
        )
        if two_subblock_sum:
            surface = tfmf._apply_two_subblock_sum(surface, stride_samples=24)
        if k_frame > 1:
            surface = tfmf._apply_k_frame_sum(surface, stride_samples=24, k=k_frame)
        magnitude = np.abs(surface)
        noise_floor = np.maximum(np.median(magnitude, axis=0), 1e-30)
        snr_db = 20.0 * np.log10(magnitude / noise_floor[None, :])
        # Keep top (target_total) per run — cheap, avoids holding 5.76M values
        flat = snr_db.ravel()
        k_top = min(target_total, flat.size)
        idx = np.argpartition(flat, -k_top)[-k_top:]
        all_top_snr_db.append(flat[idx])
    pool = np.concatenate(all_top_snr_db)
    pool.sort()
    return float(pool[-target_total])


def calibrate_sqdet_threshold(target_fa_15s: float = TARGET_FA_15S_SQDET,
                               n_runs: int = 40,
                               seed_base: int = 10_000) -> float:
    """Find sq_det detmet_norm threshold giving target_fa per 15-s noise-only.

    Default `n_runs=40`: target_fa_15s = 0.25 × 40 = ~10 FAs total over 600 s
    of noise — enough samples for a stable threshold estimate.
    """
    all_norms = []
    for k in range(n_runs):
        n = synthetic.noise_only(duration_s=DURATION_S, noise_sigma=NOISE_SIGMA,
                                 seed=seed_base + k)
        cands = _run_sq_det(n.iq, threshold_lin=0.0)
        all_norms.extend([c[2] for c in cands])
    all_norms.sort(reverse=True)
    target_total = max(int(round(target_fa_15s * n_runs)), 1)
    if len(all_norms) <= target_total:
        return 1.0
    return all_norms[target_total - 1]


# ── Matching ────────────────────────────────────────────────────────────────


def _has_match(candidates, truth_time_s: float, truth_freq_hz: float) -> bool:
    for ct, cf, _ in candidates:
        if abs(ct - truth_time_s) < TIME_TOL_S and abs(cf - truth_freq_hz) < FREQ_TOL_HZ:
            return True
    return False


# ── Worker for parallel measurement ─────────────────────────────────────────

# Module-level globals populated once per worker by _worker_init.  This
# avoids re-pickling the template across every trial.
_WORKER_TEMPLATE: np.ndarray | None = None


def _worker_init() -> None:
    """Per-process initialization: build the sync template once."""
    global _WORKER_TEMPLATE
    _WORKER_TEMPLATE = sync_template.build_sync_template(SAMPLE_RATE)


def _measure_one_trial(args: tuple) -> list[dict]:
    """Worker entry: generate one ping, run all detectors, return result rows.

    args = (snr_db, trial_idx, seed, thresholds_dict)
    where thresholds_dict maps algorithm name → calibrated threshold.
    """
    snr_db, trial, seed, ths = args
    template = _WORKER_TEMPLATE
    s = synthetic.single_ping(
        snr_db=float(snr_db),
        freq_hz=TRUTH_FREQ_HZ,
        time_s=TRUTH_TIME_S,
        duration_s=DURATION_S,
        noise_sigma=NOISE_SIGMA,
        seed=seed,
    )
    cands_sq      = _run_sq_det(s.iq, threshold_lin=ths['sq_det'])
    cands_1       = _run_tfmf(s.iq, template, ths['tfmf_1block'], two_subblock_sum=False)
    cands_2       = _run_tfmf(s.iq, template, ths['tfmf_2block'], two_subblock_sum=True)
    cands_2_k2    = _run_tfmf(s.iq, template, ths['tfmf_2block_k2'], two_subblock_sum=True, k_frame=2)
    cands_2_k4    = _run_tfmf(s.iq, template, ths['tfmf_2block_k4'], two_subblock_sum=True, k_frame=4)
    cands_2_k7    = _run_tfmf(s.iq, template, ths['tfmf_2block_k7'], two_subblock_sum=True, k_frame=7)
    out = []
    for algo, cands in [
        ('sq_det',         cands_sq),
        ('tfmf_1block',    cands_1),
        ('tfmf_2block',    cands_2),
        ('tfmf_2block_k2', cands_2_k2),
        ('tfmf_2block_k4', cands_2_k4),
        ('tfmf_2block_k7', cands_2_k7),
    ]:
        hit = _has_match(cands, TRUTH_TIME_S, TRUTH_FREQ_HZ)
        out.append({
            'algorithm':    algo,
            'snr_db':       snr_db,
            'trial':        trial,
            'detected':     int(hit),
            'n_candidates': len(cands),
            'threshold':    ths[algo],
        })
    return out


# ── Main loop ───────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    template = sync_template.build_sync_template(SAMPLE_RATE)

    print(f"Calibrating thresholds on pure-noise IQ...")
    print(f"  sq_det target: {TARGET_FA_15S_SQDET} FA / 15 s (per channel; "
          f"production = 10/15s ÷ 40 channels)")
    print(f"  TFMF target:   {TARGET_FA_15S_TFMF} FA / 15 s (wideband system budget)")
    t0 = time.time()
    th_sqdet      = calibrate_sqdet_threshold()
    th_tfmf_1blk  = calibrate_tfmf_threshold(template, two_subblock_sum=False)
    th_tfmf_2blk  = calibrate_tfmf_threshold(template, two_subblock_sum=True)
    th_tfmf_2blk_k2 = calibrate_tfmf_threshold(template, two_subblock_sum=True, k_frame=2)
    th_tfmf_2blk_k4 = calibrate_tfmf_threshold(template, two_subblock_sum=True, k_frame=4)
    th_tfmf_2blk_k7 = calibrate_tfmf_threshold(template, two_subblock_sum=True, k_frame=7)
    print(f"  sq_det threshold (detmet_norm, linear): {th_sqdet:.3f}")
    print(f"  TFMF 1-block threshold (dB):            {th_tfmf_1blk:.2f}")
    print(f"  TFMF 2-block threshold (dB):            {th_tfmf_2blk:.2f}")
    print(f"  TFMF 2-block K=2 threshold (dB):        {th_tfmf_2blk_k2:.2f}")
    print(f"  TFMF 2-block K=4 threshold (dB):        {th_tfmf_2blk_k4:.2f}")
    print(f"  TFMF 2-block K=7 threshold (dB):        {th_tfmf_2blk_k7:.2f}")
    print(f"  Calibration took {time.time()-t0:.1f}s")

    n_total = len(SNR_GRID_DB) * N_TRIALS
    print(f"\nMeasuring detection on {len(SNR_GRID_DB)} SNR × {N_TRIALS} trials "
          f"× 3 algorithms = {n_total} trials, {N_WORKERS} workers...")
    rows = []
    t_meas = time.time()

    # Bundle thresholds in a dict so future detectors can be added without
    # changing the worker signature.
    thresholds = {
        'sq_det':         th_sqdet,
        'tfmf_1block':    th_tfmf_1blk,
        'tfmf_2block':    th_tfmf_2blk,
        'tfmf_2block_k2': th_tfmf_2blk_k2,
        'tfmf_2block_k4': th_tfmf_2blk_k4,
        'tfmf_2block_k7': th_tfmf_2blk_k7,
    }

    # Build work list: one tuple per trial.  Seed deterministic per (snr, trial).
    work = []
    for snr_db in SNR_GRID_DB:
        for trial in range(N_TRIALS):
            seed = 1_000_000 + (snr_db + 20) * 1000 + trial
            work.append((snr_db, trial, seed, thresholds))

    # Dispatch trials across the worker pool.  Each worker sees the same
    # template (initialised once via _worker_init).
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_worker_init) as ex:
        for trial_rows in ex.map(_measure_one_trial, work, chunksize=4):
            rows.extend(trial_rows)
            done += 1
            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t_meas
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{n_total}] {elapsed:5.1f}s elapsed "
                      f"({rate:.1f}/s, ~{eta_s:.0f}s remaining)",
                      flush=True)

    print(f"\nTotal measurement took {time.time()-t_meas:.1f}s")

    # Per-SNR summary
    algos = ('sq_det', 'tfmf_1block', 'tfmf_2block',
             'tfmf_2block_k2', 'tfmf_2block_k4', 'tfmf_2block_k7')
    print("\nPer-SNR detection rates (algorithm = Pd/N):")
    for snr_db in SNR_GRID_DB:
        bucket = [r for r in rows if r['snr_db'] == snr_db]
        parts = [f"SNR {snr_db:+3d} dB:"]
        for algo in algos:
            algo_rows = [r for r in bucket if r['algorithm'] == algo]
            n_det = sum(r['detected'] for r in algo_rows)
            label = algo.replace('tfmf_', '').replace('block', 'b')
            parts.append(f"{label}={n_det}/{N_TRIALS}")
        print("  " + "  ".join(parts))

    with OUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['algorithm', 'snr_db', 'trial', 'detected',
                            'n_candidates', 'threshold']
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()
