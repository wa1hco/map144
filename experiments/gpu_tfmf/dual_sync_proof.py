# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Prove that coherent dual-sync combining gives ~3 dB detection gain.

Injects sync1 and sync2 at the correct MSK144 frame positions into AWGN,
runs TFMF with single and dual templates, and compares peak SNR at the
true (time, freq) location.

The dual-sync template is sparse (zeros at data positions); the FFT
naturally handles the phase relationship between the two sync words.

Math:
  matched-filter SNR ∝ ‖h‖²  (template energy)
  dual template has 2× the nonzero energy of single → +3 dB expected.

Usage:
    python -m experiments.gpu_tfmf.dual_sync_proof [--snr-db 10] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_tfmf import sync_template, tfmf


def inject_two_syncs(iq: np.ndarray, h_single: np.ndarray,
                     offset_samples: int, freq_hz: float,
                     sample_rate_hz: int, snr_db: float,
                     noise_sigma: float) -> None:
    """Inject sync1 and sync2 at the correct MSK144 frame positions.

    Sync1 at offset_samples; sync2 at offset_samples + 56*SPS.
    Both at the same freq_hz and amplitude.  Data positions left as AWGN.
    """
    sps = sample_rate_hz // 2000
    delta = sync_template.SECOND_SYNC_OFFSET_SYMBOLS * sps

    noise_power = 2.0 * noise_sigma ** 2
    target_power = noise_power * 10.0 ** (snr_db / 10.0)
    rms = float(np.sqrt(np.mean(np.abs(h_single) ** 2)))
    scale = float(np.sqrt(target_power)) / (rms + 1e-30)

    for start in (offset_samples, offset_samples + delta):
        end = min(start + len(h_single), len(iq))
        length = end - start
        t = np.arange(length) / sample_rate_hz
        carrier = np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)
        iq[start:end] += (h_single[:length] * carrier * scale).astype(np.complex64)


def peak_snr_at(surface: np.ndarray, true_window: int, true_freq_bin: int,
                search_r_t: int = 10, search_r_f: int = 4) -> float:
    """Return peak SNR (dB above per-bin median) near the true location."""
    magnitude = np.abs(surface)
    noise_floor = np.median(magnitude, axis=0)
    noise_floor = np.maximum(noise_floor, 1e-30)
    snr_db = 20.0 * np.log10(magnitude / noise_floor[None, :])

    n_t, n_f = snr_db.shape
    t0 = max(0, true_window - search_r_t)
    t1 = min(n_t, true_window + search_r_t + 1)
    f0 = max(0, true_freq_bin - search_r_f)
    f1 = min(n_f, true_freq_bin + search_r_f + 1)
    return float(np.max(snr_db[t0:t1, f0:f1]))


def run_one(snr_db: float, freq_hz: float, seed: int,
            sample_rate_hz: int = 48_000, duration_s: float = 2.0,
            ping_time_s: float = 0.5, noise_sigma: float = 1.0,
            stride: int = 24) -> dict:
    """Single trial: returns peak SNR for single and dual templates."""
    h_single = sync_template.build_sync_template(sample_rate_hz,
                                                  edge_truncate=True)
    h_dual   = sync_template.build_dual_sync_template(sample_rate_hz)

    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    iq = (
        rng.standard_normal(n).astype(np.float32) +
        1j * rng.standard_normal(n).astype(np.float32)
    ).astype(np.complex64) * np.float32(noise_sigma)

    offset = int(ping_time_s * sample_rate_hz)
    inject_two_syncs(iq, h_single, offset, freq_hz,
                     sample_rate_hz, snr_db, noise_sigma)

    # Both surfaces use fft_len bins — compute true bin on that axis.
    fft_len       = len(h_dual)
    true_window   = offset // stride
    freq_axis     = np.fft.fftfreq(fft_len, 1.0 / sample_rate_hz)
    true_bin      = int(np.argmin(np.abs(freq_axis - freq_hz)))
    # Aliases for clarity in the calls below
    true_window_single = true_window_dual = true_window
    true_bin_single    = true_bin_dual    = true_bin

    # Both surfaces use the same FFT length so the frequency axis is
    # identical — this isolates the coherent-combining gain from any
    # frequency-resolution difference between the two templates.
    fft_len = len(h_dual)
    surf_single = tfmf.compute_tf_surface_cpu(iq, h_single,
                                               stride_samples=stride,
                                               fft_length=fft_len)
    surf_dual   = tfmf.compute_tf_surface_cpu(iq, h_dual,
                                               stride_samples=stride,
                                               fft_length=fft_len)

    snr_single = peak_snr_at(surf_single, true_window_single, true_bin_single)
    snr_dual   = peak_snr_at(surf_dual,   true_window_dual,   true_bin_dual)

    return {
        "snr_db_injected": snr_db,
        "freq_hz":         freq_hz,
        "seed":            seed,
        "snr_single_db":   snr_single,
        "snr_dual_db":     snr_dual,
        "gain_db":         snr_dual - snr_single,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snr-db", type=float, default=10.0,
                    help="Injected signal SNR in dB (default: 10)")
    ap.add_argument("--freq-hz", type=float, default=5000.0)
    ap.add_argument("--trials", type=int, default=20,
                    help="Noise trials to average over (default: 20)")
    ap.add_argument("--plot", action="store_true",
                    help="Save surface comparison plot to /tmp/dual_sync.png")
    args = ap.parse_args()

    FS     = 48_000
    STRIDE = 24

    print(f"Dual-sync coherent combining proof")
    print(f"  SNR = {args.snr_db:.1f} dB   freq = {args.freq_hz:.0f} Hz   "
          f"trials = {args.trials}")
    h_single = sync_template.build_sync_template(FS, edge_truncate=False)
    h_dual   = sync_template.build_dual_sync_template(FS)
    print(f"  Single template : {len(h_single)} samples  "
          f"({len(h_single)/FS*1000:.1f} ms)  "
          f"freq_res = {FS/len(h_single):.1f} Hz/bin")
    print(f"  Dual template   : {len(h_dual)} samples  "
          f"({len(h_dual)/FS*1000:.1f} ms)  "
          f"freq_res = {FS/len(h_dual):.1f} Hz/bin")
    print(f"  Expected gain   : {10*np.log10(2):.2f} dB  (2× template energy)")
    print()

    gains = []
    snrs_s = []
    snrs_d = []
    for trial in range(args.trials):
        r = run_one(args.snr_db, args.freq_hz, seed=trial,
                    stride=STRIDE)
        gains.append(r["gain_db"])
        snrs_s.append(r["snr_single_db"])
        snrs_d.append(r["snr_dual_db"])

    gains  = np.array(gains)
    snrs_s = np.array(snrs_s)
    snrs_d = np.array(snrs_d)

    print(f"Results over {args.trials} noise trials:")
    print(f"  Single-sync peak SNR : {snrs_s.mean():.2f} ± {snrs_s.std():.2f} dB")
    print(f"  Dual-sync   peak SNR : {snrs_d.mean():.2f} ± {snrs_d.std():.2f} dB")
    print(f"  Measured gain        : {gains.mean():.2f} ± {gains.std():.2f} dB")
    print(f"  Expected gain        : {10*np.log10(2):.2f} dB")
    print(f"  Residual error       : {gains.mean() - 10*np.log10(2):.2f} dB")

    # Sweep over SNRs for the plot
    if args.plot:
        snr_range = np.arange(-5, 21, 2.5)
        mean_gains = []
        for snr in snr_range:
            g = [run_one(float(snr), args.freq_hz, seed=t,
                         stride=STRIDE)["gain_db"]
                 for t in range(10)]
            mean_gains.append(float(np.mean(g)))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))

            axes[0].hist(gains, bins=15, color="steelblue", edgecolor="white")
            axes[0].axvline(10*np.log10(2), color="red", ls="--",
                            label=f"theory {10*np.log10(2):.2f} dB")
            axes[0].axvline(gains.mean(), color="orange", ls="-",
                            label=f"mean {gains.mean():.2f} dB")
            axes[0].set_xlabel("Measured gain (dB)")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"Dual-sync gain distribution\n"
                              f"SNR={args.snr_db:.0f} dB, "
                              f"n={args.trials} trials")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(snr_range, mean_gains, "o-", label="measured gain")
            axes[1].axhline(10*np.log10(2), color="red", ls="--",
                            label=f"theory {10*np.log10(2):.2f} dB")
            axes[1].set_xlabel("Injected SNR (dB)")
            axes[1].set_ylabel("Dual-sync gain (dB)")
            axes[1].set_title("Gain vs SNR  (n=10 trials each)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            fig.suptitle("Coherent dual-sync combining — proof of 3 dB gain",
                         fontsize=11)
            fig.tight_layout()
            out = "/tmp/dual_sync.png"
            fig.savefig(out, dpi=120)
            print(f"\nPlot saved to {out}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
