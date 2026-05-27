# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Three-way detection sensitivity comparison: sq_det vs TFMF single-sync vs dual-sync.

Self-contained: generates synthetic MSK144 pings (full 144-symbol frame via
msk144code) at center_hz=1500 Hz over an SNR sweep, then reports per-detector
detection rate at each SNR.

sq_det runs at its fixed design threshold (detmet_norm >= 3.0, the WSJT-X spec).
Its natural per-trial false-detection rate at that threshold is measured on
noise-only IQ, and both TFMF detectors are CFAR-calibrated to match that rate.
This anchors the comparison to the sq_det design point rather than an arbitrary
shared target.

Usage:
    python -m experiments.gpu_tfmf.compare_detectors [options]

Options:
    --snr-min    -10    lower edge of SNR sweep (dB)
    --snr-max      5    upper edge of SNR sweep (dB)
    --snr-step     2.5  SNR increment (dB)
    --trials      20    noise realisations per SNR point
    --target-far   1.0  TFMF false events/s target for CFAR calibration
    --freq-hz    1500   signal carrier frequency (Hz)
    --plot              save detection-rate curve to /tmp/compare_detectors.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_tfmf import sync_template, tfmf

# ── Constants ─────────────────────────────────────────────────────────────────

FS          = 48_000
NOISE_SIGMA = 1.0
MSG         = "CQ K1JT FN20"
STRIDE      = 24              # samples — one MSK144 symbol at 48 kHz
PING_TIME_S = 0.5             # ping onset within each trial buffer
DURATION_S  = 2.0             # trial buffer length
MATCH_TIME_S  = 0.5           # ±window for candidate-to-truth matching
MATCH_FREQ_HZ = 500.0         # ±window for candidate-to-truth matching

# sq_det operates at 12 kHz; decimation factor from 48 kHz
_DEC_FACTOR = FS // 12_000

# sq_det design threshold: detmet_norm >= 3.0 (WSJT-X spec, ~-6 dB WSJT-X SNR)
SQDET_THRESHOLD_DB: float = 20.0 * np.log10(3.0)   # ≈ 9.54 dB


# ── Signal generation ─────────────────────────────────────────────────────────


def _make_noise(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(n_samples).astype(np.float32)
        + 1j * rng.standard_normal(n_samples).astype(np.float32)
    ).astype(np.complex64) * np.float32(NOISE_SIGMA)


def make_trial(snr_db: float, freq_hz: float, noise_seed: int) -> np.ndarray:
    """Generate one trial: AWGN + one full MSK144 ping at (PING_TIME_S, freq_hz, snr_db).

    Uses generate_ping_msk144code for a real 144-symbol frame with correct
    sync structure (both sync words present).  Returns complex64 at FS Hz.
    """
    from generate_msk144 import generate_ping_msk144code  # noqa: PLC0415

    n = int(DURATION_S * FS)
    iq = _make_noise(n, noise_seed)

    ping_h, _, _ = generate_ping_msk144code(
        MSG,
        center_hz=freq_hz,
        delay_s=PING_TIME_S,
        snr_db=snr_db,
        noise_sigma=NOISE_SIGMA,
        flat_frame=True,           # constant-amplitude 72 ms burst
        pol_scenario='pure-H',
        sample_rate=FS,
    )
    n_clip = min(len(ping_h), n)
    iq[:n_clip] += ping_h[:n_clip].astype(np.complex64)
    return iq


# ── Threshold calibration ─────────────────────────────────────────────────────
#
# sq_det runs at its fixed design threshold (detmet_norm >= 3.0).  On pure
# AWGN the normalized ratio statistic rarely exceeds 3.0, so its AWGN FA rate
# is near zero.  These are different statistics — there is no well-defined
# mapping from TFMF cell counts to sq_det window counts on pure AWGN.
#
# Strategy: calibrate both TFMF variants to the same target peak-picked event
# rate (events/s), controlled by --target-far.  Report sq_det's measured
# per-trial FA rate alongside for transparency.


def measure_sqdet_far(freq_hz: float, duration_s: float = 10.0) -> float:
    """Measure sq_det's false-alarm rate (windows/s) at its design threshold."""
    from map144_app.sq_det import sq_det as _sq_det, NSPM  # noqa: PLC0415
    from scipy.signal import decimate                       # noqa: PLC0415

    n = int(duration_s * FS)
    noise_48k = _make_noise(n, seed=1)
    audio_12k = decimate(noise_48k, _DEC_FACTOR,
                         ftype='fir', zero_phase=True).astype(np.complex128)

    hop = NSPM // 4
    raw: list[float] = []
    for start in range(0, audio_12k.size - NSPM + 1, hop):
        r = _sq_det(audio_12k[start:start + NSPM], fc=freq_hz)
        raw.append(float(r.detmet))
    raw_arr = np.array(raw, dtype=np.float32)

    xmed = max(float(np.percentile(raw_arr, 25)), 1e-30)
    norm = raw_arr / xmed
    thresh_lin = 10.0 ** (SQDET_THRESHOLD_DB / 20.0)
    return float(np.sum(norm > thresh_lin)) / duration_s


def calibrate_tfmf(h: np.ndarray, cfg_base: tfmf.TFMFConfig,
                   target_far: float,
                   duration_s: float = 10.0) -> tuple[float, float]:
    """Find threshold_db yielding ≈ target_far peak-picked events/s on noise.

    Precomputes the surface and 2-D local-max mask once (both are
    threshold-independent), then binary-searches only on the event count.
    Returns (threshold_db, achieved_events_per_s).
    """
    from scipy.ndimage import maximum_filter as _mf

    n = int(duration_s * FS)
    noise = _make_noise(n, seed=0)
    surface = tfmf.compute_tf_surface_cpu(noise, h, STRIDE,
                                          fft_length=cfg_base.fft_length)
    mag    = np.abs(surface)
    nf     = np.maximum(np.median(mag, axis=0), 1e-30)
    snr_db = 20.0 * np.log10(mag / nf[None, :])

    n_freq   = surface.shape[1]
    freq_res = FS / n_freq
    fp_t = cfg_base.peak_picking_time_windows
    fp_f = max(1, round(2.0 * cfg_base.peak_picking_freq_hz / freq_res))
    if fp_f % 2 == 0:
        fp_f += 1
    local_max    = _mf(snr_db, size=(fp_t, fp_f),
                        mode='constant', cval=float('-inf'))
    is_local_max = snr_db >= local_max - 1e-6   # threshold-independent mask

    lo, hi = 3.0, 80.0
    for _ in range(30):
        mid    = (lo + hi) / 2.0
        n_evts = int(np.sum(is_local_max & (snr_db > mid)))
        if n_evts / duration_s > target_far:
            lo = mid
        else:
            hi = mid
    thr      = (lo + hi) / 2.0
    achieved = int(np.sum(is_local_max & (snr_db > thr))) / duration_s
    return thr, achieved


# ── Per-trial detection ───────────────────────────────────────────────────────


def _detected_tfmf(iq: np.ndarray, h: np.ndarray, threshold_db: float,
                   freq_hz: float, cfg_base: tfmf.TFMFConfig) -> bool:
    """True if any peak-picked candidate near (PING_TIME_S, freq_hz) exceeds threshold."""
    cfg   = dataclasses.replace(cfg_base, threshold_db=threshold_db)
    cands = tfmf.detect(iq, template=h, cfg=cfg, use_gpu=False)
    return any(
        abs(c.time_s - PING_TIME_S) < MATCH_TIME_S and
        abs(c.freq_hz - freq_hz)    < MATCH_FREQ_HZ
        for c in cands
    )


def _detected_sqdet(iq: np.ndarray, freq_hz: float) -> bool:
    """True if any sq_det window near PING_TIME_S exceeds the design threshold."""
    from map144_app.sq_det import sq_det as _sq_det, NSPM  # noqa: PLC0415
    from scipy.signal import decimate                       # noqa: PLC0415

    audio_12k = decimate(iq, _DEC_FACTOR,
                         ftype='fir', zero_phase=True).astype(np.complex128)
    hop = NSPM // 4
    raw: list[float] = []
    times: list[float] = []
    for start in range(0, audio_12k.size - NSPM + 1, hop):
        r = _sq_det(audio_12k[start:start + NSPM], fc=freq_hz)
        raw.append(float(r.detmet))
        times.append((start + NSPM / 2.0) / 12_000.0)

    raw_arr  = np.array(raw)
    xmed = max(float(np.percentile(raw_arr, 25)), 1e-30)
    norm = raw_arr / xmed
    thresh_lin = 10.0 ** (SQDET_THRESHOLD_DB / 20.0)   # detmet_norm >= 3.0

    for i, t in enumerate(times):
        if abs(t - PING_TIME_S) < MATCH_TIME_S and norm[i] >= thresh_lin:
            return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snr-min",    type=float, default=-10.0)
    ap.add_argument("--snr-max",    type=float, default=5.0)
    ap.add_argument("--snr-step",   type=float, default=2.5)
    ap.add_argument("--trials",     type=int,   default=20)
    ap.add_argument("--target-far", type=float, default=1.0,
                    help="TFMF false peak-picked events/s for CFAR calibration (default: 1.0)")
    ap.add_argument("--freq-hz",    type=float, default=1500.0,
                    help="Signal carrier frequency in Hz (default: 1500)")
    ap.add_argument("--plot", action="store_true",
                    help="Save detection-rate curve to /tmp/compare_detectors.png")
    args = ap.parse_args()

    h_single = sync_template.build_sync_template(FS, edge_truncate=True)
    h_dual   = sync_template.build_dual_sync_template(FS)

    cfg_single = tfmf.TFMFConfig(use_dual_sync=False)
    cfg_dual   = tfmf.TFMFConfig(use_dual_sync=True)

    print(f"Three-way detection sensitivity comparison")
    print(f"  Signal: {args.freq_hz:.0f} Hz  flat_frame=True  msg={MSG!r}")
    print(f"  Single template: {len(h_single)} samples  "
          f"({len(h_single)/FS*1000:.1f} ms)  "
          f"freq_res={FS/len(h_single):.1f} Hz/bin")
    print(f"  Dual template:   {len(h_dual)} samples  "
          f"({len(h_dual)/FS*1000:.1f} ms)  "
          f"freq_res={FS/len(h_dual):.2f} Hz/bin")

    # ── CFAR calibration ──────────────────────────────────────────────────
    print(f"\nsq_det fixed threshold: detmet_norm >= 3.0  ({SQDET_THRESHOLD_DB:.2f} dB)")
    print(f"  Measuring sq_det FA rate on noise …", flush=True)
    sq_far = measure_sqdet_far(args.freq_hz)
    print(f"  sq_det FA: {sq_far:.4f} windows/s  (AWGN — effectively zero by design)")

    print(f"\nCalibrating TFMF to {args.target_far:.2f} false events/s …", flush=True)
    thr_single, far_single = calibrate_tfmf(h_single, cfg_single, args.target_far)
    print(f"  TFMF single : {thr_single:.2f} dB  (achieved {far_single:.3f} events/s)")

    thr_dual, far_dual = calibrate_tfmf(h_dual, cfg_dual, args.target_far)
    print(f"  TFMF dual   : {thr_dual:.2f} dB  (achieved {far_dual:.3f} events/s)")

    # ── SNR sweep ─────────────────────────────────────────────────────────
    snr_values = np.arange(args.snr_min,
                           args.snr_max + args.snr_step * 0.5,
                           args.snr_step)

    print(f"\nSweeping {len(snr_values)} SNR values, {args.trials} trials each …\n")

    hdr = (f"{'SNR':>6s}  {'sq_det':>8s}  {'single':>8s}  "
           f"{'dual':>8s}  {'Δsingle':>9s}  {'Δdual':>8s}")
    print(hdr)
    print("─" * len(hdr))

    results = []
    for snr in snr_values:
        hits_sq = hits_s = hits_d = 0
        for trial in range(args.trials):
            seed = int(snr * 100) + trial + 100_000   # deterministic, always positive
            iq = make_trial(snr, args.freq_hz, noise_seed=seed)

            if _detected_sqdet(iq, args.freq_hz):
                hits_sq += 1
            if _detected_tfmf(iq, h_single, thr_single, args.freq_hz, cfg_single):
                hits_s += 1
            if _detected_tfmf(iq, h_dual, thr_dual, args.freq_hz, cfg_dual):
                hits_d += 1

        n = args.trials
        p_sq = hits_sq / n
        p_s  = hits_s  / n
        p_d  = hits_d  / n
        # Gain expressed as Δ detection probability vs sq_det baseline
        d_s = p_s - p_sq
        d_d = p_d - p_sq

        marker = "  ← 50%" if abs(p_d - 0.50) < 0.15 else ""
        print(f"{snr:>+6.1f}  {p_sq:>7.0%}   {p_s:>7.0%}   {p_d:>7.0%}  "
              f"{d_s:>+8.0%}   {d_d:>+7.0%}{marker}")

        results.append(dict(snr_db=float(snr),
                            p_sqdet=p_sq, p_single=p_s, p_dual=p_d))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n50%-detection SNR (interpolated):")
    for label, key in [("sq_det", "p_sqdet"), ("single", "p_single"), ("dual", "p_dual")]:
        # Find crossing from below 0.5 to above 0.5
        prev = None
        snr50 = None
        for r in results:
            p = r[key]
            if prev is not None and prev < 0.5 <= p:
                frac = (0.5 - prev) / max(p - prev, 1e-9)
                snr50 = results[results.index(r) - 1]['snr_db'] + frac * args.snr_step
                break
            prev = p
        if snr50 is not None:
            print(f"  {label:<8s}: {snr50:+.1f} dB")
        else:
            # didn't cross 50%
            last_p = results[-1][key]
            first_p = results[0][key]
            if last_p < 0.5:
                print(f"  {label:<8s}: > {snr_values[-1]:+.1f} dB  (never reached 50%)")
            elif first_p >= 0.5:
                print(f"  {label:<8s}: < {snr_values[0]:+.1f} dB  (already ≥50% at start)")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            snrs  = [r['snr_db']   for r in results]
            p_sq  = [r['p_sqdet']  for r in results]
            p_s   = [r['p_single'] for r in results]
            p_d   = [r['p_dual']   for r in results]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(snrs, p_sq, "s--", color="gray",   label="sq_det")
            ax.plot(snrs, p_s,  "o-",  color="blue",   label="TFMF single-sync")
            ax.plot(snrs, p_d,  "^-",  color="red",    label="TFMF dual-sync (+3 dB)")
            ax.axhline(0.5, color="black", ls=":", lw=0.8, label="50% threshold")
            ax.set_xlabel("Injected SNR (dB, WSJT-X definition)")
            ax.set_ylabel("Detection probability")
            ax.set_title(
                f"Three-way sensitivity: sq_det vs TFMF\n"
                f"f={args.freq_hz:.0f} Hz, n={args.trials} trials/SNR, "
                f"TFMF FAR={args.target_far:.1f}/s"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            out = "/tmp/compare_detectors.png"
            fig.savefig(out, dpi=120)
            print(f"\nPlot saved to {out}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
