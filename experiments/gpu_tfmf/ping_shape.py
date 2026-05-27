# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Measure the intrinsic 2D shape of the TFMF matched-filter response.

Injects the sync template at a known (time, freq) into AWGN and measures the
correlation surface shape normalized to the peak magnitude.  This is a
property of the template autocorrelation — independent of SNR and threshold.

The peak-normalized shape is what determines the correct peak-picking
neighborhood: cells within N dB of the peak are "clearly the same ping"
and should be suppressed by the local-max filter.

Usage:
    python -m experiments.gpu_tfmf.ping_shape [--plot] [--snr-db 40]

    # Compare across SNRs to confirm shape is SNR-independent:
    for snr in 40 20 10 5; do
        python -m experiments.gpu_tfmf.ping_shape --snr-db $snr
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_tfmf import sync_template, tfmf


def inject_ping(iq: np.ndarray, template: np.ndarray,
                offset_samples: int, freq_hz: float,
                sample_rate_hz: int, snr_db: float,
                noise_sigma: float) -> None:
    N = len(template)
    n_samples = len(iq)
    end = min(offset_samples + N, n_samples)
    length = end - offset_samples

    t = np.arange(length) / sample_rate_hz
    carrier = np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)

    noise_power = 2.0 * noise_sigma ** 2
    target_signal_power = noise_power * (10.0 ** (snr_db / 10.0))
    template_rms = np.sqrt(np.mean(np.abs(template[:length]) ** 2))
    scale = np.sqrt(target_signal_power) / (template_rms + 1e-30)

    iq[offset_samples:end] += (template[:length] * carrier * scale).astype(np.complex64)


def peak_normalized_profiles(surface: np.ndarray,
                              true_window: int, true_freq_bin: int,
                              search_radius_t: int = 20,
                              search_radius_f: int = 20,
                              ) -> dict:
    """Return peak-normalized magnitude profiles around the true ping location.

    All values are in dB relative to the peak magnitude (0 dB at peak,
    negative everywhere else).  This is independent of SNR and threshold.
    """
    magnitude = np.abs(surface)                            # raw MF output
    n_t, n_f = magnitude.shape

    # Find peak near the true location
    t0 = max(0, true_window - search_radius_t)
    t1 = min(n_t, true_window + search_radius_t + 1)
    f0 = max(0, true_freq_bin - search_radius_f)
    f1 = min(n_f, true_freq_bin + search_radius_f + 1)
    region = magnitude[t0:t1, f0:f1]
    lt, lf = np.unravel_index(np.argmax(region), region.shape)
    peak_t = t0 + lt
    peak_f = f0 + lf
    peak_mag = float(magnitude[peak_t, peak_f])

    # Profiles centred at peak — use a wider window for clean display
    R_T = min(search_radius_t, peak_t, n_t - peak_t - 1)
    R_F = min(search_radius_f, peak_f, n_f - peak_f - 1)

    time_mag  = magnitude[peak_t - R_T : peak_t + R_T + 1, peak_f]
    freq_mag  = magnitude[peak_t, peak_f - R_F : peak_f + R_F + 1]
    time_offsets = np.arange(-R_T, R_T + 1)
    freq_offsets = np.arange(-R_F, R_F + 1)

    # Normalize to peak — express as dB below peak (0 at peak, negative elsewhere)
    eps = peak_mag * 1e-6
    time_db  = 20.0 * np.log10(np.maximum(time_mag,  eps) / peak_mag)
    freq_db  = 20.0 * np.log10(np.maximum(freq_mag,  eps) / peak_mag)

    def span_at(profile_db: np.ndarray, level_db: float) -> int:
        """Number of contiguous samples from peak that stay above level_db."""
        mid = len(profile_db) // 2
        left = mid
        while left > 0 and profile_db[left - 1] >= level_db:
            left -= 1
        right = mid
        while right < len(profile_db) - 1 and profile_db[right + 1] >= level_db:
            right += 1
        return right - left + 1

    return {
        "peak_t":       peak_t,
        "peak_f":       peak_f,
        "peak_mag":     peak_mag,
        "time_offsets": time_offsets,
        "freq_offsets": freq_offsets,
        "time_db":      time_db,
        "freq_db":      freq_db,
        "magnitude":    magnitude,
        # Contiguous spans at standard reference levels
        "time_span_3db":   span_at(time_db,  -3.0),
        "time_span_10db":  span_at(time_db, -10.0),
        "time_span_20db":  span_at(time_db, -20.0),
        "freq_span_3db":   span_at(freq_db,  -3.0),
        "freq_span_10db":  span_at(freq_db, -10.0),
        "freq_span_20db":  span_at(freq_db, -20.0),
    }


def print_profile(label: str, offsets: np.ndarray, values_db: np.ndarray,
                  bar_scale: float = 0.5) -> None:
    """ASCII bar chart of a peak-normalized profile (dB below peak)."""
    print(f"\n  {label}:")
    for off, val in zip(offsets, values_db):
        marker = "← peak" if off == 0 else ""
        # Bars sized by how close to peak (0 dB = full bar, -30 dB = nothing)
        bar_len = max(0, int((val + 30) * bar_scale))
        bar = "█" * bar_len
        level = ("◀3 " if val >= -3 else
                 "◀10" if val >= -10 else
                 "◀20" if val >= -20 else
                 "   ")
        print(f"    {off:+4d}  {level}  {val:7.1f} dB  {bar} {marker}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snr-db", type=float, default=40.0,
                    help="Injected ping SNR in dB (default: 40 — high for clean shape)")
    ap.add_argument("--freq-hz", type=float, default=5000.0,
                    help="Ping frequency offset in Hz (default: 5000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true",
                    help="Save a PNG of the normalized surface")
    ap.add_argument("--plot-out", default="/tmp/ping_shape.png")
    args = ap.parse_args()

    FS        = 48_000
    DURATION  = 2.0
    PING_TIME = 0.5
    NOISE_SIG = 1.0
    STRIDE    = 24

    h = sync_template.build_sync_template(FS)
    N = len(h)

    rng = np.random.default_rng(args.seed)
    n_samp = int(DURATION * FS)
    iq = (
        rng.standard_normal(n_samp).astype(np.float32) +
        1j * rng.standard_normal(n_samp).astype(np.float32)
    ).astype(np.complex64) * np.complex64(NOISE_SIG)

    offset_samp  = int(PING_TIME * FS)
    inject_ping(iq, h, offset_samp, args.freq_hz, FS, args.snr_db, NOISE_SIG)

    true_window   = offset_samp // STRIDE
    freq_axis     = np.fft.fftfreq(N, 1.0 / FS)
    true_freq_bin = int(np.argmin(np.abs(freq_axis - args.freq_hz)))

    print(f"Ping shape — peak-normalized MF response")
    print(f"  Injected SNR     : {args.snr_db:.0f} dB  (high → noise negligible in shape)")
    print(f"  Freq             : {args.freq_hz:.0f} Hz  (bin {true_freq_bin}, "
          f"{freq_axis[true_freq_bin]:.0f} Hz)")
    print(f"  Template         : {N} samples  ({N/FS*1000:.1f} ms)")
    print(f"  Stride           : {STRIDE} samples  ({STRIDE/FS*1000:.2f} ms/window)")
    print(f"  Freq resolution  : {FS/N:.1f} Hz/bin")

    surface = tfmf.compute_tf_surface_cpu(iq, h, stride_samples=STRIDE)
    p = peak_normalized_profiles(surface, true_window, true_freq_bin)

    print(f"\nPeak location:")
    print(f"  Window {p['peak_t']}  (offset from true: {p['peak_t'] - true_window:+d})")
    print(f"  Bin    {p['peak_f']}  (offset from true: {p['peak_f'] - true_freq_bin:+d}  "
          f"= {(p['peak_f'] - true_freq_bin) * FS / N:.0f} Hz)")

    print(f"\nPeak-normalized spans (contiguous from peak):")
    print(f"  {'Level':>8}  {'Time (windows)':>16}  {'Time (ms)':>10}  "
          f"{'Freq (bins)':>12}  {'Freq (Hz)':>10}")
    for lev, ts, fs in [
        ( -3, p["time_span_3db"],  p["freq_span_3db"]),
        (-10, p["time_span_10db"], p["freq_span_10db"]),
        (-20, p["time_span_20db"], p["freq_span_20db"]),
    ]:
        print(f"  {lev:>7} dB  {ts:>16d}  {ts*STRIDE/FS*1000:>10.1f}  "
              f"{fs:>12d}  {fs*FS/N:>10.1f}")

    print_profile("Time profile  (dB below peak, at peak freq bin)",
                  p["time_offsets"], p["time_db"])
    print_profile("Freq profile  (dB below peak, at peak window)",
                  p["freq_offsets"], p["freq_db"])

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mag  = p["magnitude"]
            pt, pf = p["peak_t"], p["peak_f"]
            R_T, R_F = 20, 20
            t0 = max(0, pt - R_T); t1 = min(mag.shape[0], pt + R_T + 1)
            f0 = max(0, pf - R_F); f1 = min(mag.shape[1], pf + R_F + 1)

            patch = 20.0 * np.log10(
                np.maximum(mag[t0:t1, f0:f1], mag[pt, pf] * 1e-6) / mag[pt, pf]
            )

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # Normalized surface heatmap
            im = axes[0].imshow(
                patch.T, aspect="auto", origin="lower",
                extent=[t0 - pt, t1 - pt - 1, f0 - pf, f1 - pf - 1],
                vmin=-30, vmax=0,
                cmap="inferno",
            )
            for lev, col in [(-3, "white"), (-10, "cyan"), (-20, "lime")]:
                axes[0].contour(
                    np.linspace(t0 - pt, t1 - pt - 1, patch.shape[0]),
                    np.linspace(f0 - pf, f1 - pf - 1, patch.shape[1]),
                    patch.T, levels=[lev], colors=[col], linewidths=0.8,
                )
            axes[0].set_xlabel("Window offset from peak")
            axes[0].set_ylabel("Freq-bin offset from peak")
            axes[0].set_title("Peak-normalized MF response (dB)\ncontours: −3/−10/−20 dB")
            cb = fig.colorbar(im, ax=axes[0])
            cb.set_label("dB below peak")

            # Time profile
            axes[1].plot(p["time_offsets"], p["time_db"])
            for lev, col, lab in [(-3, "green", "-3 dB"),
                                   (-10, "orange", "-10 dB"),
                                   (-20, "red", "-20 dB")]:
                axes[1].axhline(lev, color=col, ls="--", lw=0.8, label=lab)
            axes[1].axvline(0, color="gray", ls=":", lw=0.8)
            axes[1].set_xlabel("Window offset from peak")
            axes[1].set_ylabel("dB below peak")
            axes[1].set_title(
                f"Time profile\n"
                f"−3 dB: {p['time_span_3db']} win  "
                f"−10 dB: {p['time_span_10db']} win  "
                f"−20 dB: {p['time_span_20db']} win"
            )
            axes[1].set_ylim(-35, 2)
            axes[1].legend(fontsize=7)
            axes[1].grid(True, alpha=0.3)

            # Freq profile
            axes[2].plot(p["freq_offsets"], p["freq_db"])
            for lev, col, lab in [(-3, "green", "-3 dB"),
                                   (-10, "orange", "-10 dB"),
                                   (-20, "red", "-20 dB")]:
                axes[2].axhline(lev, color=col, ls="--", lw=0.8, label=lab)
            axes[2].axvline(0, color="gray", ls=":", lw=0.8)
            axes[2].set_xlabel("Freq-bin offset from peak")
            axes[2].set_ylabel("dB below peak")
            axes[2].set_title(
                f"Freq profile\n"
                f"−3 dB: {p['freq_span_3db']} bins  "
                f"−10 dB: {p['freq_span_10db']} bins  "
                f"−20 dB: {p['freq_span_20db']} bins"
            )
            axes[2].set_ylim(-35, 2)
            axes[2].legend(fontsize=7)
            axes[2].grid(True, alpha=0.3)

            fig.suptitle(
                f"TFMF ping shape (peak-normalized) — "
                f"SNR={args.snr_db:.0f} dB, f={args.freq_hz:.0f} Hz",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(args.plot_out, dpi=120)
            print(f"\nPlot saved to {args.plot_out}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
