#!/usr/bin/env python3
"""End-to-end SNR calibration for MAP144's detection processing chain.

Generates synthetic MSK144-like tone-pair bursts at precisely known SNR,
runs them through the same mix → decimate → estimate_snr path that real pings
take in detection.py, and reports the bias at each SNR level.

This answers two questions without needing overnight radio data:
  1. Is MAP144's _estimate_snr_db algorithm correctly calibrated?
  2. Does the mix→decimate chain introduce a systematic SNR loss?

The synthetic signal is a 2-tone burst (1000 Hz + 2000 Hz, ±500 Hz around
1500 Hz) with a Gamma meteor-trail envelope, placed in exactly the same audio
layout as a real decode WAV:

    [100 ms pad] [500 ms AWGN noise] [burst region] [100 ms pad]

The AWGN amplitude sets the noise floor; the burst amplitude is then
computed from the target SNR using the WSJT-X 2500 Hz bandwidth definition:

    SNR = peak² / (2·σ²·BW/fs)    BW=2500 Hz, fs=12 kHz (audio rate)

Usage:
    python snr_calibrate.py [--plot FILE] [--snr-min N] [--snr-max N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import decimate

# ── Constants matching detection.py ───────────────────────────────────────────
_DECODE_RATE     = 12_000          # audio sample rate jt9 / est_snr expect
_DECIMATE_FACTOR = 4               # 48 kHz IQ → 12 kHz audio
_IQ_RATE         = _DECODE_RATE * _DECIMATE_FACTOR   # 48 kHz
_TARGET_FC_HZ    = 1500.0          # MSK144 audio carrier convention

# ── Import MAP144's own SNR estimator ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from map144_app.detection import _estimate_snr_db


# ── Signal generation ─────────────────────────────────────────────────────────

def make_tone_burst_audio(snr_db: float, width_ms: float = 150.0,
                          noise_sigma: float = 0.01,
                          sample_rate: int = _DECODE_RATE,
                          rng=None) -> np.ndarray:
    """
    Build a *real audio* array directly at *sample_rate* (default 12 kHz)
    matching the MAP144 detect-WAV layout:

        [pad_n] [pre_n noise] [burst+post] [pad_n]

    This tests the SNR estimator in isolation, with no decimation chain
    ambiguity.  Noise is white Gaussian with std=noise_sigma, so the SNR
    definition is exact:

        SNR = A² / (2·σ²·2500/fs)   →   A = σ·sqrt(2·10^(snr/10)·2500/fs)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pad_n  = int(0.100 * sample_rate)
    pre_n  = int(0.500 * sample_rate)
    post_n = int(1.200 * sample_rate)
    total  = pad_n + pre_n + post_n + pad_n

    audio = rng.standard_normal(total).astype(np.float32) * noise_sigma

    # Burst: Gamma envelope, two real tones at 1000 Hz and 2000 Hz
    burst_n   = post_n
    t         = np.arange(burst_n, dtype=np.float64) / sample_rate
    tau       = width_ms / 1000.0
    envelope  = (t / tau) * np.exp(-t / tau)
    envelope /= envelope.max()

    burst = ((np.cos(2 * np.pi * 1000.0 * t) +
              np.cos(2 * np.pi * 2000.0 * t)) * envelope / np.sqrt(2)).astype(np.float32)

    A = noise_sigma * np.sqrt(2.0 * (10.0 ** (snr_db / 10.0)) * 2500.0 / sample_rate)
    audio[pad_n + pre_n:pad_n + pre_n + burst_n] += burst * A

    return audio


def make_tone_burst_iq(snr_db: float, width_ms: float = 150.0,
                       noise_sigma: float = 0.01,
                       fc_hz: float = 1500.0,
                       sample_rate: int = _IQ_RATE,
                       rng=None) -> np.ndarray:
    """
    Build a complex IQ array at *sample_rate* (48 kHz) then decimate to audio.
    Used to test the full IQ→mix→decimate chain for additional SNR loss.

    Noise is complex white Gaussian at 48 kHz with per-component std=noise_sigma.
    The burst amplitude is calibrated at 48 kHz:
        SNR = A² / (2·σ_IQ²·2500/fs_IQ)
    so the stated SNR is the SNR a receiver would measure if it could process
    the full 48 kHz IQ directly — before any decimation noise reduction.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pad_n  = int(0.100 * sample_rate)
    pre_n  = int(0.500 * sample_rate)
    post_n = int(1.200 * sample_rate)
    total  = pad_n + pre_n + post_n + pad_n

    iq = (rng.standard_normal(total) +
          1j * rng.standard_normal(total)).astype(np.complex64) * noise_sigma

    burst_n   = post_n
    t         = np.arange(burst_n, dtype=np.float64) / sample_rate
    tau       = width_ms / 1000.0
    envelope  = (t / tau) * np.exp(-t / tau)
    envelope /= envelope.max()

    tone_lo  = np.exp(2j * np.pi * (fc_hz - 500.0) * t).astype(np.complex64)
    tone_hi  = np.exp(2j * np.pi * (fc_hz + 500.0) * t).astype(np.complex64)
    burst_iq = ((tone_lo + tone_hi) * envelope / np.sqrt(2)).astype(np.complex64)

    # Calibrate at IQ rate — noise_sigma is defined here
    A = noise_sigma * np.sqrt(2.0 * (10.0 ** (snr_db / 10.0)) * 2500.0 / sample_rate)
    iq[pad_n + pre_n:pad_n + pre_n + burst_n] += burst_iq * A

    return iq


def iq_to_audio(iq: np.ndarray, fc_hz: float = _TARGET_FC_HZ,
                factor: int = _DECIMATE_FACTOR,
                zero_phase: bool = False) -> np.ndarray:
    """
    Replicate detection.py's signal path:
        mix to 1500 Hz  →  Re(IQ)  →  decimate to 12 kHz
    """
    n     = len(iq)
    t     = np.arange(n, dtype=np.float64)
    shift = fc_hz - _TARGET_FC_HZ
    mixed = iq * np.exp(-2j * np.pi * shift * t / _IQ_RATE).astype(np.complex64)
    return decimate(np.real(mixed).astype(np.float64), factor,
                    zero_phase=zero_phase).astype(np.float32)


# ── Calibration sweep ─────────────────────────────────────────────────────────

def _run_one_sweep(label, gen_fn, snr_range, n_trials):
    """Run n_trials per SNR level, return list of result dicts."""
    print(f"\n{'─'*60}\n{label}\n{'─'*60}")
    results = []
    for snr_db in snr_range:
        vals = []
        for seed in range(n_trials):
            audio = gen_fn(snr_db, rng=np.random.default_rng(seed))
            peak  = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = audio / peak * 0.9
            est = _estimate_snr_db(audio)
            if est is not None:
                vals.append(est)
        if vals:
            mean_e = np.mean(vals)
            std_e  = np.std(vals)
            bias   = mean_e - snr_db
            results.append({'snr_in': snr_db, 'mean_est': mean_e,
                             'std': std_e, 'bias': bias, 'n': len(vals)})
            print(f"  {snr_db:+4.0f} dB in  →  {mean_e:+5.1f} dB est  "
                  f"bias {bias:+.1f} dB  std {std_e:.1f} dB")
    if results:
        b = [r['bias'] for r in results]
        # exclude low-SNR saturation region (estimator hits noise floor)
        hi = [r['bias'] for r in results if r['snr_in'] >= 0]
        print(f"  → Overall bias: {np.mean(b):+.2f} dB  "
              f"(≥0 dB region: {np.mean(hi):+.2f} ± {np.std(hi):.2f} dB)")
    return results


def run_calibration(snr_range, n_trials: int = 20,
                    zero_phase: bool = False) -> dict:
    """
    Two sweeps:
      1. Audio-domain test  — signal generated directly at 12 kHz, no IQ chain.
         Tests the SNR estimator in pure isolation.
      2. IQ-chain test      — signal generated at 48 kHz IQ, passed through
         mix→decimate, same as detection.py.  Reveals any additional chain loss.
    """
    # Sweep 1: estimator only
    r_audio = _run_one_sweep(
        "Test 1: estimator only (audio at 12 kHz, no IQ chain)",
        lambda snr, rng: make_tone_burst_audio(snr, rng=rng),
        snr_range, n_trials)

    # Sweep 2: full IQ chain (causal or zero-phase)
    label2 = (f"Test 2: full IQ→decimate chain "
              f"({'zero-phase' if zero_phase else 'causal'} filter)")
    r_iq = _run_one_sweep(
        label2,
        lambda snr, rng: iq_to_audio(
            make_tone_burst_iq(snr, rng=rng), zero_phase=zero_phase),
        snr_range, n_trials)

    # Chain contribution = Test2_bias − Test1_bias
    print(f"\n{'─'*60}")
    print("IQ chain additional bias (Test 2 − Test 1):")
    for r1, r2 in zip(r_audio, r_iq):
        chain = r2['bias'] - r1['bias']
        print(f"  {r1['snr_in']:+4.0f} dB : {chain:+.1f} dB")

    return {'audio': r_audio, 'iq': r_iq}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snr-min',   type=float, default=-10.0)
    ap.add_argument('--snr-max',   type=float, default=+20.0)
    ap.add_argument('--snr-step',  type=float, default=2.0)
    ap.add_argument('--trials',    type=int,   default=20,
                    help='AWGN trials per SNR level (default: 20)')
    ap.add_argument('--zero-phase', action='store_true',
                    help='Use zero-phase decimation (forward+backward filter)')
    ap.add_argument('--plot',      default=None, metavar='FILE')
    args = ap.parse_args()

    snr_range = list(np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step))

    print(f"Decimation: zero_phase={args.zero_phase}  "
          f"(causal IIR  vs  zero-phase forward+backward)")
    print(f"Running {len(snr_range)} SNR levels × {args.trials} trials each...\n")

    results = run_calibration(snr_range, n_trials=args.trials,
                              zero_phase=args.zero_phase)

    if args.plot and (results['audio'] or results['iq']):
        _plot(results, args.plot, args.zero_phase)


def _plot(results, out_path, zero_phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ra = results['audio']
    ri = results['iq']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    fig.patch.set_facecolor('#f8f8f8')

    def _plot_series(ax, res, label, color, fmt):
        if not res:
            return
        xs   = [r['snr_in']   for r in res]
        ys   = [r['mean_est'] for r in res]
        errs = [r['std']      for r in res]
        ax.errorbar(xs, ys, yerr=errs, fmt=fmt, color=color, capsize=4,
                    linewidth=2, markersize=6, label=label)

    # Top: estimated vs true
    if ra:
        _plot_series(ax1, ra, 'Estimator only (audio, no IQ chain)', 'steelblue', 'o-')
    if ri:
        _plot_series(ax1, ri,
                     f'Full IQ chain ({"zero-phase" if zero_phase else "causal"} filter)',
                     '#d62728', 's--')
    xs_all = [r['snr_in'] for r in (ra or ri)]
    ax1.plot([xs_all[0], xs_all[-1]], [xs_all[0], xs_all[-1]],
             color='gray', linestyle=':', linewidth=1, label='ideal y = x')
    ax1.set_ylabel('Estimated SNR  (dB)', fontsize=11)
    ax1.set_title('MAP144 SNR estimator calibration\n(synthetic tone-pair burst)',
                  fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(linestyle=':', alpha=0.4)

    # Bottom: bias
    w = (xs_all[1] - xs_all[0]) * 0.35 if len(xs_all) > 1 else 0.7
    xs_np = np.array(xs_all)
    if ra:
        ax2.bar(xs_np - w/2, [r['bias'] for r in ra], width=w,
                color='steelblue', alpha=0.7, label='Estimator only')
    if ri:
        ax2.bar(xs_np + w/2, [r['bias'] for r in ri], width=w,
                color='#d62728', alpha=0.7,
                label=f'Full IQ chain ({"zero-phase" if zero_phase else "causal"})')
    ax2.axhline(0, color='gray', linewidth=1)
    ax2.set_xlabel('True SNR  (dB)', fontsize=11)
    ax2.set_ylabel('Bias  (est − true, dB)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(linestyle=':', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


def args_step_for_plot(results):
    if len(results) < 2:
        return 1.5
    return (results[1]['snr_in'] - results[0]['snr_in']) * 0.7


if __name__ == '__main__':
    main()
