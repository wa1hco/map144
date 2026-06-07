# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Sweep harness: measure detection rate vs (SNR, frequency, threshold).

Output: one row per (snr_db, freq_offset_hz, trial) with detection
outcome.  Aggregate downstream (pandas, or just grep) to compute
detection-rate curves.

Usage:
    cd <repo>
    .venv/bin/python -m experiments.gpu_wideband_sync.benchmark \\
        --out /tmp/sweep.csv --use-gpu

Honest about what this measures:
    - This is per-ping detection rate on synthetic IQ — does the
      detector say "yes, there's a sync at (time, freq)" at the right
      place?
    - It does NOT measure decode rate (no jt9 in the loop).  That comes
      later when the detector emits Candidates compatible with the
      existing DecoderBlock.
    - Threshold sweep is done post-hoc (record raw SNR, threshold off-
      line) — much cheaper than re-running detection per threshold.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root or this dir.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_wideband_sync import sync_template, synthetic, wideband_sync


def matched_against_truth(candidate, ping,
                          time_tol_s: float = 0.1,
                          freq_tol_hz: float = 500.0) -> bool:
    """Return True if ``candidate`` is a plausible detection of ``ping``.

    Tolerances are intentionally generous for the first round:
    refine once the detector is producing tight peaks.
    """
    return (
        abs(candidate.time_s - ping.time_s) < time_tol_s
        and abs(candidate.freq_hz - ping.freq_hz) < freq_tol_hz
    )


def run_one_trial(snr_db: float, freq_hz: float, seed: int,
                  cfg: wideband_sync.WidebandSyncConfig,
                  use_gpu: bool):
    """Build a synthetic signal, run the detector, return result row."""
    s = synthetic.single_ping(
        snr_db=snr_db,
        freq_hz=freq_hz,
        time_s=0.5,
        sample_rate_hz=cfg.sample_rate_hz,
        duration_s=2.0,
        seed=seed,
    )
    h = sync_template.build_sync_template(cfg.sample_rate_hz)

    t0 = time.perf_counter()
    candidates = wideband_sync.detect(s.iq, h, cfg, use_gpu=use_gpu)
    elapsed_s = time.perf_counter() - t0

    truth = s.pings[0]
    matched = [c for c in candidates if matched_against_truth(c, truth)]

    best = max(matched, key=lambda c: c.snr_db) if matched else None
    return {
        "snr_db_set":        snr_db,
        "freq_hz_set":       freq_hz,
        "seed":              seed,
        "n_candidates":      len(candidates),
        "n_matched":         len(matched),
        "detected":          1 if matched else 0,
        "best_match_snr_db": best.snr_db if best else None,
        "best_match_time_s": best.time_s if best else None,
        "best_match_freq_hz": best.freq_hz if best else None,
        "wallclock_s":       round(elapsed_s, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep.csv", help="Output CSV path")
    ap.add_argument("--snr-min", type=float, default=-5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--snr-step", type=float, default=2.5)
    ap.add_argument("--freq-min", type=float, default=-20_000.0)
    ap.add_argument("--freq-max", type=float, default=20_000.0)
    ap.add_argument("--freq-step", type=float, default=5_000.0)
    ap.add_argument("--trials", type=int, default=10,
                    help="Trials per (snr, freq) cell")
    ap.add_argument("--threshold-db", type=float, default=10.0)
    ap.add_argument("--use-gpu", action="store_true",
                    help="Use GPU (CuPy).  Required for non-trivial speed.")
    args = ap.parse_args()

    cfg = wideband_sync.WidebandSyncConfig(
        sample_rate_hz=48_000,
        threshold_db=args.threshold_db,
    )

    snrs = np.arange(args.snr_min, args.snr_max + 0.001, args.snr_step)
    freqs = np.arange(args.freq_min, args.freq_max + 0.001, args.freq_step)

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = None
        n_done = 0
        n_total = len(snrs) * len(freqs) * args.trials
        for snr in snrs:
            for freq in freqs:
                for trial in range(args.trials):
                    seed = 1000 + n_done            # deterministic but varied
                    row = run_one_trial(
                        snr_db=float(snr),
                        freq_hz=float(freq),
                        seed=seed,
                        cfg=cfg,
                        use_gpu=args.use_gpu,
                    )
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    n_done += 1
                    if n_done % 25 == 0:
                        print(f"  {n_done}/{n_total}  "
                              f"(snr={snr:.1f}, freq={freq:.0f})",
                              flush=True)

    print(f"\nDone.  {n_done} trials → {out_path}")
    print("Aggregate detection rate by SNR:")
    # Cheap summary, no pandas dep.
    by_snr = {}
    with out_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = float(row["snr_db_set"])
            by_snr.setdefault(snr, []).append(int(row["detected"]))
    for snr in sorted(by_snr):
        det = by_snr[snr]
        print(f"  snr={snr:>5.1f} dB   detected {sum(det):>4d} / {len(det):<4d}  "
              f"({100*sum(det)/len(det):>5.1f}%)")


if __name__ == "__main__":
    main()
