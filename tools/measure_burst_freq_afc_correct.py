"""Squared-FFT AFC + mix-corrected coherent integration prototype.

Architecture (per discussion with operator, 2026-05-12):

  burst audio
      │
      ▼
  [squared-FFT MSK144 tracker]   ← sees 100% of signal energy via the
      │                            two modulation tones (1000, 2000 Hz);
      │                            rejects non-MSK144 interference via
      │                            the coincident-peak validity gate.
      ▼
  f̂(t), valid(t) per window
      │
      ▼
  [smoother]                     ← brings f̂ noise below ~1 Hz so phase
      │                            integration doesn't scramble signal
      ▼
  f̂_s(t) smoothed
      │
      ▼
  [mix-correct]                  ← x'(t) = x(t)·exp(-j·2π·∫f̂_s(τ)dτ)
      │                            burst is now stationary at baseband
      ▼
  [coherent frame sum]           ← s = Σ x'[k·NSPM:(k+1)·NSPM]
      │                            one super-frame, N× signal, √N noise
      ▼
  s — single MSK144 frame with N-frame coherent gain

The decoder (downstream, not in this tool) can scan alignments either
via 2-frame duplication (linear scan compatibility) or circular scan
(if the decoder supports it).  For this prototype we just measure the
coherent-matched-filter peak magnitude on the single output frame and
compare to the existing uncorrected coherent estimator on the same
burst.  If AFC-corrected peak_mag exceeds baseline, the mix-correction
extracted extra coherent gain that the simple sum-then-correlate
loses to drift / flutter.

Key design points
-----------------
1. The squared-FFT tracker requires BOTH the 2 kHz line and the 4 kHz
   line to be present with consistent Δf.  Non-MSK144 tones produce
   only one line; this is a built-in MSK144 discriminator.

2. When valid(t) is False for >50% of the burst, AFC is skipped —
   we just report the baseline numbers.  Don't mix-correct against
   noise.

3. The squared-FFT window is 6 frames wide (432 ms) to get σ(f̂)
   below ~1 Hz at moderate SNR.  Trades fast-flutter tracking for
   the freq precision needed to actually help.

4. The smoother is intentionally simple (centred median-of-5) — if
   the prototype shows promise we can upgrade to Kalman / spline.

Outputs
-------
- 3-panel plot per WAV (envelope / f̂(t) with validity / matched-filter
  response: baseline vs AFC-corrected on the same axis).
- CSV in batch mode: per-burst before/after peak_mag, FWHM, pk/noise.

Usage
-----
    python tools/measure_burst_freq_afc_correct.py --wav FILE
    python tools/measure_burst_freq_afc_correct.py --date 20260511 --csv OUT.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import medfilt

# Reuse the existing tools' helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_ping_freq_vs_time import (
    _bandpass, _envelope_db, _find_bursts,
    _load_wsjtx_messages, _msgs_for_wav,
    BURST_THRESH_DB, MIN_BURST_S,
)
from measure_burst_freq_coherent import (
    coherent_burst_freq_estimate, _audio_to_baseband,
    SR_AUDIO, FC_AUDIO,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import _sync_correlate, _SYNC_CB_CONJ, NSPM


# Squared-FFT tracker parameters
TRACK_WIN_FRAMES        = 6        # window = 6 frames = 432 ms
TRACK_STEP_FRAC         = 0.5      # 50% overlap
LINE_SEARCH_HZ          = 300.0    # ±Hz around each squared line
LINE_COINCIDENCE_TOL_HZ = 10.0     # |Δf_2k − Δf_4k| must be < this
LINE_SNR_THRESHOLD      = 4.0      # peak / median (in-band) must exceed this
VALID_FRACTION_MIN      = 0.5      # if fewer windows valid, skip AFC

# Smoother
SMOOTHER_MEDIAN_LEN     = 5        # median-of-5 (centred)


# ── MSK144-discriminating squared-FFT tracker ─────────────────────────────

@dataclass
class TrackPoint:
    """One squared-FFT tracker output point."""
    t_center_s:     float
    f_carrier_hz:   float           # offset from 1500 Hz audio centre
    valid:          bool
    snr_2k:         float           # line-peak/local-median, for diagnostics
    snr_4k:         float
    disagreement:   float           # |Δf_2k − Δf_4k|


def squared_fft_msk_tracker(
    audio_real: np.ndarray,
    burst_start_n: int,
    burst_end_n: int,
    sr: int = SR_AUDIO,
    win_frames: int | None = None,
    step_frac: float = TRACK_STEP_FRAC,
) -> list[TrackPoint]:
    """Track f̂(t) within burst via coincident-line squared-FFT.

    win_frames defaults to min(TRACK_WIN_FRAMES, n_frames) so short
    bursts still produce at least one window (single-shot f̂ estimate
    with no within-burst drift tracking).  Returns one TrackPoint per
    window position.  valid=True only when both the 2 kHz and 4 kHz
    lines are above the SNR threshold and their inferred Δf agree
    within tolerance.
    """
    burst = audio_real[burst_start_n:burst_end_n]
    n_frames_avail = len(burst) // NSPM
    if win_frames is None:
        win_frames = min(TRACK_WIN_FRAMES, max(1, n_frames_avail))
    if len(burst) < win_frames * NSPM:
        return []

    win_n  = win_frames * NSPM
    step_n = max(1, int(win_n * step_frac))
    nfft   = win_n
    df     = sr / nfft

    bin_2k_centre = int(round((2 * FC_AUDIO - 1000) / df))   # squared lo line
    bin_4k_centre = int(round((2 * FC_AUDIO + 1000) / df))   # squared hi line
    bin_half_win  = int(round(LINE_SEARCH_HZ / df))

    hann = np.hanning(win_n).astype(np.float64)
    sq   = burst.astype(np.float64) ** 2

    def _parabolic(i: int, s: np.ndarray) -> float:
        if 1 <= i < len(s) - 1:
            a, b, c = s[i-1], s[i], s[i+1]
            den = a - 2*b + c
            if den != 0:
                return i + 0.5 * (a - c) / den
        return float(i)

    points: list[TrackPoint] = []
    for start in range(0, len(burst) - win_n + 1, step_n):
        seg  = sq[start:start + win_n] * hann
        spec = np.abs(np.fft.rfft(seg))

        # Bound search ranges
        lo_lo = max(0, bin_2k_centre - bin_half_win)
        lo_hi = min(len(spec), bin_2k_centre + bin_half_win)
        hi_lo = max(0, bin_4k_centre - bin_half_win)
        hi_hi = min(len(spec), bin_4k_centre + bin_half_win)
        if lo_hi <= lo_lo or hi_hi <= hi_lo:
            continue

        # Peak in each band
        i_lo = lo_lo + int(np.argmax(spec[lo_lo:lo_hi]))
        i_hi = hi_lo + int(np.argmax(spec[hi_lo:hi_hi]))
        peak_2k = float(spec[i_lo])
        peak_4k = float(spec[i_hi])

        # Local-median noise floor inside each search window
        med_2k = float(np.median(spec[lo_lo:lo_hi]))
        med_4k = float(np.median(spec[hi_lo:hi_hi]))
        snr_2k = peak_2k / med_2k if med_2k > 0 else float('inf')
        snr_4k = peak_4k / med_4k if med_4k > 0 else float('inf')

        # Sub-bin refinement
        f_2k = _parabolic(i_lo, spec) * df
        f_4k = _parabolic(i_hi, spec) * df

        # Inferred Δf from each line — both should agree
        dF_2k = (f_2k - (2 * FC_AUDIO - 1000)) / 2.0
        dF_4k = (f_4k - (2 * FC_AUDIO + 1000)) / 2.0
        disagreement = abs(dF_2k - dF_4k)

        # Validity = both lines above threshold AND lines agree
        valid = (snr_2k >= LINE_SNR_THRESHOLD and
                 snr_4k >= LINE_SNR_THRESHOLD and
                 disagreement <= LINE_COINCIDENCE_TOL_HZ)

        # Best Δf estimate = inverse-variance average (here just mean since
        # both lines have the same Fourier resolution)
        f_carrier = 0.5 * (dF_2k + dF_4k)
        t_center  = (burst_start_n + start + win_n / 2) / sr

        points.append(TrackPoint(
            t_center_s=t_center,
            f_carrier_hz=f_carrier,
            valid=valid,
            snr_2k=snr_2k,
            snr_4k=snr_4k,
            disagreement=disagreement,
        ))

    return points


# ── Smoother and per-sample interpolation ────────────────────────────────

def smooth_track(points: list[TrackPoint],
                 burst_start_n: int, burst_end_n: int,
                 sr: int = SR_AUDIO) -> tuple[np.ndarray, np.ndarray]:
    """Smooth f̂(t) and interpolate to per-sample rate over the burst.

    Returns (per_sample_f_hz, per_sample_valid).  Where valid is False
    along the way, we hold-over the nearest valid value (or zero if
    everything is invalid).
    """
    n_samp = burst_end_n - burst_start_n
    per_sample_f     = np.zeros(n_samp, dtype=np.float64)
    per_sample_valid = np.zeros(n_samp, dtype=bool)

    if not points:
        return per_sample_f, per_sample_valid

    # Extract arrays
    t_arr = np.array([p.t_center_s for p in points])
    f_arr = np.array([p.f_carrier_hz for p in points])
    v_arr = np.array([p.valid for p in points])

    # Smoother — adaptive to point count.  Median-of-K requires ≥K points.
    # Below that fall back to: median-of-all (≥2 points) or the single value.
    if v_arr.any():
        valid_idx = np.where(v_arr)[0]
        n_valid = len(valid_idx)
        if n_valid >= SMOOTHER_MEDIAN_LEN:
            # Linearly interp through invalid gaps first, then median-filter
            f_filled = np.interp(np.arange(len(f_arr)),
                                 valid_idx, f_arr[valid_idx])
            f_smooth = medfilt(f_filled, kernel_size=SMOOTHER_MEDIAN_LEN)
        elif n_valid >= 2:
            # Too few points for median-of-5; use the median of valid values
            # as a constant (single freq estimate, no drift tracking)
            f_smooth = np.full(len(f_arr), float(np.median(f_arr[valid_idx])))
        else:
            # Single valid window → hold its freq across the burst
            f_smooth = np.full(len(f_arr), float(f_arr[valid_idx[0]]))
    else:
        f_smooth = f_arr.copy()

    # Per-sample interpolation
    t_samp = (burst_start_n + np.arange(n_samp)) / sr
    per_sample_f = np.interp(t_samp, t_arr, f_smooth)
    # Validity propagates from the nearest window centre; treat as valid
    # if EITHER neighbour is valid (we held-over)
    if v_arr.any():
        # Build a step function of validity by finding nearest window
        nearest_idx = np.clip(np.searchsorted(t_arr, t_samp), 0, len(t_arr) - 1)
        per_sample_valid = v_arr[nearest_idx]

    return per_sample_f, per_sample_valid


# ── Mix-correct + coherent sum into one super-frame ──────────────────────

def afc_mix_correct_and_sum(
    audio_real: np.ndarray,
    burst_start_n: int, burst_end_n: int,
    per_sample_f_hz: np.ndarray,
    sr: int = SR_AUDIO,
) -> np.ndarray | None:
    """Mix-correct burst by smoothed f̂(t) then coherent-sum frames.

    Returns one super-frame (NSPM samples complex), or None if burst
    too short.
    """
    n_avail = burst_end_n - burst_start_n
    n_frames = n_avail // NSPM
    if n_frames < 1:
        return None

    # First convert burst audio to complex baseband (same as the existing
    # coherent estimator does — Hilbert + mix by -1500 Hz, RMS-normalise)
    baseband_full = _audio_to_baseband(audio_real, sr=sr)
    burst_bb      = baseband_full[burst_start_n:burst_start_n + n_frames * NSPM]

    # Integrate per-sample f̂ to get phase, then mix-correct
    f_slice = per_sample_f_hz[:n_frames * NSPM]
    # cumsum gives phase in Hz·sample; divide by sr to get cycles, ×2π to get rad
    phase = (2 * np.pi / sr) * np.cumsum(f_slice)
    corrected = (burst_bb * np.exp(-1j * phase).astype(np.complex64)).astype(np.complex64)

    # Coherent frame sum: reshape to (n_frames, NSPM) and sum along frame axis
    super_frame = corrected.reshape(n_frames, NSPM).sum(axis=0)
    return super_frame.astype(np.complex64)


# ── Per-burst comparison: baseline vs AFC ───────────────────────────────

@dataclass
class AfcComparison:
    """Side-by-side baseline (uncorrected) vs AFC-corrected results."""
    t_start_s:           float
    t_end_s:             float
    n_frames:            int
    valid_fraction:      float
    afc_applied:         bool
    # Baseline (existing coherent_burst_freq_estimate, sum-then-correlate
    # WITHOUT pre-mix-correction)
    base_f0_hz:          float
    base_peak_mag:       float
    base_fwhm_hz:        float
    base_peak_to_noise:  float
    # AFC-corrected: mix-correct → coherent sum → sync-correlate on summed frame
    afc_peak_mag:        float
    afc_sync_xmax:       float       # raw sync correlator output on the summed frame
    # Improvement
    gain_db:             float       # 10*log10(afc_peak_mag / base_peak_mag)
    # Tracker outputs (for plotting)
    track_points:        list[TrackPoint]


def compare_burst(audio_real: np.ndarray,
                   burst_start_n: int, burst_end_n: int,
                   sr: int = SR_AUDIO) -> AfcComparison:
    """Compute baseline + AFC-corrected metrics on one burst."""
    # Baseline: existing tool, no mix-correction
    baseband_full = _audio_to_baseband(audio_real, sr=sr)
    base = coherent_burst_freq_estimate(baseband_full, burst_start_n, burst_end_n, sr=sr)

    n_avail = burst_end_n - burst_start_n
    n_frames = n_avail // NSPM

    if base is None or n_frames < 1:
        return AfcComparison(
            t_start_s=burst_start_n / sr,
            t_end_s=burst_end_n / sr,
            n_frames=n_frames,
            valid_fraction=0.0,
            afc_applied=False,
            base_f0_hz=float('nan'),
            base_peak_mag=0.0,
            base_fwhm_hz=float('nan'),
            base_peak_to_noise=float('nan'),
            afc_peak_mag=0.0,
            afc_sync_xmax=0.0,
            gain_db=float('nan'),
            track_points=[],
        )

    # Run the squared-FFT tracker
    points = squared_fft_msk_tracker(audio_real, burst_start_n, burst_end_n, sr=sr)
    valid_frac = (sum(1 for p in points if p.valid) / len(points)) if points else 0.0

    if valid_frac < VALID_FRACTION_MIN or not points:
        # Don't apply AFC — fall back to baseline
        return AfcComparison(
            t_start_s=burst_start_n / sr,
            t_end_s=burst_end_n / sr,
            n_frames=n_frames,
            valid_fraction=valid_frac,
            afc_applied=False,
            base_f0_hz=base.f0_hz,
            base_peak_mag=base.peak_mag,
            base_fwhm_hz=base.peak_width_hz,
            base_peak_to_noise=base.peak_to_noise,
            afc_peak_mag=base.peak_mag,
            afc_sync_xmax=base.peak_mag,
            gain_db=0.0,
            track_points=points,
        )

    # Smooth and interpolate per-sample f̂(t)
    per_sample_f, _ = smooth_track(points, burst_start_n, burst_end_n, sr=sr)

    # Mix-correct and coherent-sum into one super-frame
    super_frame = afc_mix_correct_and_sum(audio_real, burst_start_n, burst_end_n,
                                          per_sample_f, sr=sr)
    if super_frame is None:
        afc_peak = 0.0
        sync_xmax = 0.0
    else:
        # Sync correlation on the summed frame: WHERE the message sits within
        # the frame is unknown a priori — that's the alignment freedom the
        # 2-frame duplicate (or circular scan) handles downstream.  Here we
        # just report the matched-filter peak magnitude as the SNR proxy.
        _, sync_xmax, _ = _sync_correlate(super_frame)
        afc_peak = float(sync_xmax)

    # Convert to dB.  These are MAGNITUDES (not power); ratio in dB = 20 log.
    if base.peak_mag > 0 and afc_peak > 0:
        gain_db = 20.0 * np.log10(afc_peak / base.peak_mag)
    else:
        gain_db = float('nan')

    return AfcComparison(
        t_start_s=burst_start_n / sr,
        t_end_s=burst_end_n / sr,
        n_frames=n_frames,
        valid_fraction=valid_frac,
        afc_applied=True,
        base_f0_hz=base.f0_hz,
        base_peak_mag=base.peak_mag,
        base_fwhm_hz=base.peak_width_hz,
        base_peak_to_noise=base.peak_to_noise,
        afc_peak_mag=afc_peak,
        afc_sync_xmax=sync_xmax,
        gain_db=gain_db,
        track_points=points,
    )


# ── WAV-level driver ─────────────────────────────────────────────────────

def analyse_wav(path: Path) -> tuple[np.ndarray, float, list[AfcComparison]]:
    """Find bursts in a WAV and run AFC comparison on each."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    if sr != SR_AUDIO:
        raise ValueError(f"expected {SR_AUDIO} Hz, got {sr}")

    audio_bp = _bandpass(audio, sr)
    env_db, bin_s = _envelope_db(audio_bp, sr)
    burst_idx_pairs = _find_bursts(env_db, bin_s, BURST_THRESH_DB, MIN_BURST_S)

    comparisons = []
    for i_start, i_end in burst_idx_pairs:
        burst_start_n = int(i_start * bin_s * sr)
        burst_end_n = int(i_end * bin_s * sr)
        cmp = compare_burst(audio, burst_start_n, burst_end_n, sr=sr)
        comparisons.append(cmp)
    return env_db, bin_s, comparisons


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_afc(path: Path,
              env_db: np.ndarray, env_bin_s: float,
              comparisons: list[AfcComparison],
              decoded_msgs: list[str] | None = None,
              out_path: Path | None = None) -> Path:
    """3-panel: envelope, f̂(t) with validity, baseline-vs-AFC magnitudes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wav_duration = len(env_db) * env_bin_s
    t_env = np.arange(len(env_db)) * env_bin_s
    median_db = float(np.median(env_db))

    if comparisons:
        t_lo = max(0.0, min(c.t_start_s for c in comparisons) - 0.5)
        t_hi = min(wav_duration, max(c.t_end_s for c in comparisons) + 0.5)
    else:
        t_lo, t_hi = 0.0, wav_duration

    fig, (ax_env, ax_f, ax_bar) = plt.subplots(3, 1, figsize=(13, 9),
                                                gridspec_kw={'height_ratios': [1, 1.3, 1]})

    # ── Top: envelope ─────────────────────────────────────────────────────
    ax_env.plot(t_env, env_db - median_db, color='C0', lw=1.0)
    ax_env.axhline(BURST_THRESH_DB, color='r', ls='--', lw=0.5)
    for i, cmp in enumerate(comparisons):
        c = 'tab:green' if cmp.afc_applied else 'tab:gray'
        ax_env.axvspan(cmp.t_start_s, cmp.t_end_s, color=c, alpha=0.18)
        label = (f'b{i+1}: AFC' if cmp.afc_applied else f'b{i+1}: skip')
        ax_env.annotate(
            f'{label}\n{cmp.n_frames}fr  valid={cmp.valid_fraction:.0%}',
            xy=((cmp.t_start_s + cmp.t_end_s)/2, ax_env.get_ylim()[1]*0.8),
            ha='center', fontsize=8, color=c,
        )
    ax_env.set_xlim(t_lo, t_hi)
    zoom_mask = (t_env >= t_lo) & (t_env <= t_hi)
    if zoom_mask.any():
        seg = env_db[zoom_mask] - median_db
        ax_env.set_ylim(max(-10.0, float(seg.min()) - 2.0),
                        max(float(seg.max()) + 2.0, BURST_THRESH_DB + 1.0))
    ax_env.set_ylabel('Envelope dB above median')
    title_parts = [path.name]
    if decoded_msgs:
        title_parts.append('  /  '.join(decoded_msgs[:3]))
    ax_env.set_title('  —  '.join(title_parts))
    ax_env.grid(alpha=0.3)

    # ── Middle: tracker f̂(t) with validity marker shape ───────────────────
    for i, cmp in enumerate(comparisons):
        if not cmp.track_points:
            continue
        t = np.array([p.t_center_s for p in cmp.track_points])
        f = np.array([p.f_carrier_hz for p in cmp.track_points])
        v = np.array([p.valid for p in cmp.track_points])
        # Valid points = solid green circle, invalid = open red triangle
        if v.any():
            ax_f.scatter(t[v], f[v], s=40, c='tab:green',
                          edgecolor='black', linewidth=0.5, marker='o',
                          label=f'b{i+1} valid' if i == 0 else None, zorder=3)
        if (~v).any():
            ax_f.scatter(t[~v], f[~v], s=40, c='none',
                          edgecolor='tab:red', linewidth=1.0, marker='^',
                          label=f'b{i+1} invalid' if i == 0 else None, zorder=2)
    ax_f.axhline(0.0, color='k', ls=':', lw=0.5)
    ax_f.set_xlim(t_lo, t_hi)
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('f̂_carrier (Hz from 1500 Hz)')
    ax_f.set_title('Squared-FFT MSK144 tracker  '
                    '(coincident 2 kHz + 4 kHz peaks; both required for validity)')
    ax_f.grid(alpha=0.3)
    if any(c.track_points for c in comparisons):
        ax_f.legend(loc='best', fontsize=8)

    # ── Bottom: AFC gain in dB — the headline outcome ─────────────────────
    # Absolute magnitudes look near-identical at this scale; the meaningful
    # signal is the ratio.  Plot 20·log10(afc/base) directly with a zero
    # reference and colour-code by sign.  Skipped bursts get a flat gray
    # marker at zero to distinguish them from "applied but no effect".
    indices = np.arange(len(comparisons))
    gains   = np.array([c.gain_db if not np.isnan(c.gain_db) else 0.0
                          for c in comparisons])
    colors  = []
    for c in comparisons:
        if not c.afc_applied:
            colors.append('lightgray')
        elif c.gain_db > 0:
            colors.append('tab:green')
        else:
            colors.append('tab:red')
    ax_bar.bar(indices, gains, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax_bar.axhline(0.0, color='k', lw=1.0)
    # Reference dB-line guides — eye finds ±1 dB and ±0.5 dB easily
    for ref in (-1.0, -0.5, 0.5, 1.0):
        ax_bar.axhline(ref, color='k', ls=':', lw=0.4, alpha=0.4)

    # Per-bar annotation: dB on the bar, magnitudes underneath
    for i, c in enumerate(comparisons):
        if c.afc_applied and not np.isnan(c.gain_db):
            y_text = c.gain_db + (0.10 if c.gain_db >= 0 else -0.10)
            va = 'bottom' if c.gain_db >= 0 else 'top'
            ax_bar.annotate(f'{c.gain_db:+.2f} dB',
                              xy=(i, y_text), ha='center', va=va,
                              fontsize=10, fontweight='bold',
                              color='tab:green' if c.gain_db > 0 else 'tab:red')
        else:
            ax_bar.annotate('skipped',
                              xy=(i, 0.10), ha='center', va='bottom',
                              fontsize=9, color='gray', style='italic')
        # Magnitudes shown below the x-axis label
        ax_bar.annotate(f'base={c.base_peak_mag:.0f}\nafc ={c.afc_peak_mag:.0f}',
                          xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                          ha='center', va='top', fontsize=7, color='gray',
                          family='monospace')

    ax_bar.set_xticks(indices)
    ax_bar.set_xticklabels([f'b{i+1}' for i in range(len(comparisons))])
    ax_bar.set_ylabel('AFC gain (dB) over baseline')
    ax_bar.set_title('Coherent matched-filter gain from AFC  '
                      '— 20·log₁₀(afc_mag / baseline_mag)')
    # Y-range: at least ±1.5 dB so the reference lines are visible
    y_extreme = max(1.5, float(np.abs(gains).max() * 1.2))
    ax_bar.set_ylim(-y_extreme, y_extreme)
    ax_bar.grid(alpha=0.2, axis='y')

    if out_path is None:
        default_dir = (Path(__file__).resolve().parent.parent
                        / 'docs' / 'figures' / 'burst_freq_afc_single')
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / f'burstafc_{path.stem}.png'

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None,
                     help='Single WAV path or filename in --wsjtx-dir/save/')
    ap.add_argument('--wsjtx-dir', type=Path,
                     default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None,
                     help='Batch: process all WAVs for this UTC date (YYYYMMDD)')
    ap.add_argument('--csv', type=Path, default=None,
                     help='Write per-burst before/after metrics to this CSV')
    ap.add_argument('--plot-each-burst', type=Path, default=None,
                     metavar='DIR', help='In batch mode, save per-WAV plot to DIR')
    ap.add_argument('--max-wavs', type=int, default=None)
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt) if all_txt.exists() else {}

    # ── Single-WAV mode ─────────────────────────────────────────────────────
    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        if not wav_path.exists():
            print(f"error: WAV not found: {args.wav}", file=sys.stderr)
            return 1
        env_db, bin_s, cmps = analyse_wav(wav_path)
        decoded_msgs = _msgs_for_wav(wav_path, msg_index)
        png = plot_afc(wav_path, env_db, bin_s, cmps, decoded_msgs)
        msgs_str = '  /  '.join(decoded_msgs) if decoded_msgs else '(none)'
        print(f"{wav_path.name}  ({len(cmps)} bursts)  WSJT-X: {msgs_str}")
        for i, c in enumerate(cmps):
            applied = 'AFC' if c.afc_applied else 'skip'
            print(f"  burst {i+1}: t={c.t_start_s:5.2f}-{c.t_end_s:5.2f}s  "
                  f"{c.n_frames}fr  valid={c.valid_fraction:.0%}  [{applied}]  "
                  f"base_mag={c.base_peak_mag:6.0f}  "
                  f"afc_mag={c.afc_peak_mag:6.0f}  "
                  f"gain={c.gain_db:+5.2f} dB")
        print(f"\nPlot: {png}")
        return 0

    # ── Batch mode ───────────────────────────────────────────────────────────
    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]
    if not wavs:
        print(f"error: no WAVs for date {args.date}", file=sys.stderr)
        return 1

    print(f"Processing {len(wavs)} WAVs", file=sys.stderr)
    if args.plot_each_burst:
        args.plot_each_burst.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, str, int, AfcComparison]] = []
    n_plotted = 0
    for i, p in enumerate(wavs):
        try:
            env_db, bin_s, cmps = analyse_wav(p)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        decoded_msgs = _msgs_for_wav(p, msg_index)
        msgs_str = '|'.join(decoded_msgs)
        for bi, c in enumerate(cmps):
            all_rows.append((p.name, msgs_str, bi, c))
        if args.plot_each_burst and cmps:
            out_png = args.plot_each_burst / f'burstafc_{p.stem}.png'
            try:
                plot_afc(p, env_db, bin_s, cmps, decoded_msgs, out_path=out_png)
                n_plotted += 1
            except Exception as exc:
                print(f"  {p.name}: PLOT ERROR {exc}", file=sys.stderr)
        if (i + 1) % 25 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(all_rows)} bursts, "
                  f"{n_plotted} plots", file=sys.stderr)

    if args.csv and all_rows:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'wsjtx_msgs', 'burst_idx',
                        't_start_s', 't_end_s', 'n_frames',
                        'valid_fraction', 'afc_applied',
                        'base_f0_hz', 'base_peak_mag', 'base_fwhm_hz',
                        'base_peak_to_noise',
                        'afc_peak_mag', 'gain_db'])
            for name, msgs, bi, c in all_rows:
                w.writerow([name, msgs, bi,
                            f'{c.t_start_s:.3f}', f'{c.t_end_s:.3f}',
                            c.n_frames,
                            f'{c.valid_fraction:.3f}', int(c.afc_applied),
                            f'{c.base_f0_hz:+.2f}', f'{c.base_peak_mag:.0f}',
                            f'{c.base_fwhm_hz:.2f}', f'{c.base_peak_to_noise:.2f}',
                            f'{c.afc_peak_mag:.0f}', f'{c.gain_db:+.2f}'])
        print(f"\nCSV: {args.csv}")

    # ── Summary distributions ────────────────────────────────────────────────
    if all_rows:
        applied = [c for *_, c in all_rows if c.afc_applied]
        skipped = [c for *_, c in all_rows if not c.afc_applied]
        print()
        print(f"AFC prototype — {len(all_rows)} bursts "
              f"({len(applied)} AFC-applied, {len(skipped)} skipped low-validity)")
        if applied:
            gains = np.array([c.gain_db for c in applied if not np.isnan(c.gain_db)])
            improved = np.sum(gains > 0)
            print(f"  Gain distribution (dB) over {len(gains)} AFC-applied bursts:")
            print(f"    min/p25/median/p75/max : "
                  f"{gains.min():+.2f}  "
                  f"{np.percentile(gains, 25):+.2f}  "
                  f"{np.median(gains):+.2f}  "
                  f"{np.percentile(gains, 75):+.2f}  "
                  f"{gains.max():+.2f}")
            print(f"    fraction with gain > 0  : {improved}/{len(gains)} "
                  f"({100*improved/len(gains):.0f}%)")
            print(f"    fraction with gain > 1  : "
                  f"{np.sum(gains > 1)}/{len(gains)} "
                  f"({100*np.sum(gains > 1)/len(gains):.0f}%)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
