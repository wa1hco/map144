"""Step-1 reliability measurement for greedy-frame-inclusion AFC.

Question this tool answers
--------------------------
The greedy-frame-inclusion algorithm uses the squared-line SNR of the
COHERENTLY-SUMMED accumulator as the per-step decision metric:

    accept frame k  iff  line_snr(accumulator + frame_k)
                          >  line_snr(accumulator) + Δ_min

For this rule to be reliable, two empirical numbers need to be
established:

    1. Signal slope — how much does line_snr go UP when adding a true
       in-burst signal frame to the accumulator?
    2. Metric noise  — what is the std of line_snr change when adding
       a NOISE-ONLY frame (drawn from outside any burst) to the same
       accumulator?

Δ_min must exceed the metric noise by some margin (say 2σ) to keep
false-include rate on noise frames < ~2.3%.  If signal slope is large
compared to metric noise, the greedy algorithm is decisive.  If they
overlap, the greedy approach can't reliably distinguish at marginal
SNR and we have to fall back to decoder-success-as-validity.

Method
------
For each multi-frame burst in the corpus:

  1. Identify the centre frame and the squared-FFT-derived f̂.
  2. Build a baseline accumulator from just the centre frame (mix-
     corrected by f̂).  Measure line_snr_0.
  3. Signal slope: incrementally add each adjacent in-burst frame,
     recording line_snr after each add and the delta from previous.
  4. Metric noise: for N_NOISE_TRIALS random positions OUTSIDE any
     burst, add that single noise frame to the centre-alone baseline,
     and record line_snr_noise − line_snr_0.

Outputs
-------
- CSV with per-(burst, step, type) records.
- Histogram plot comparing signal-deltas vs noise-deltas.
- Console summary: median/p25/p75 of signal slope, std of noise delta,
  signal-to-metric-noise separation in σ-units.

Usage
-----
    python tools/measure_afc_metric_reliability.py --date 20260511 \\
            --csv docs/figures/afc_metric_reliability.csv \\
            --plot docs/figures/afc_metric_reliability.png
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_ping_freq_vs_time import (
    _bandpass, _envelope_db, _find_bursts,
    _load_wsjtx_messages, _msgs_for_wav,
    BURST_THRESH_DB, MIN_BURST_S,
)
from measure_burst_freq_afc_correct import (
    squared_fft_msk_tracker,
    _audio_to_baseband,
    SR_AUDIO, FC_AUDIO,
    LINE_SEARCH_HZ,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import NSPM


N_NOISE_TRIALS_PER_BURST = 20      # noise-frame positions to sample per burst
MAX_TEST_FRAMES          = 8       # cap on adjacent signal frames to test
MIN_BURST_FRAMES         = 4       # need ≥ 4 frames for useful signal slope


# ── Line-SNR metric on a complex-baseband summed frame ────────────────────

def line_snr_of_summed_frame(summed_complex: np.ndarray,
                              sr: int = SR_AUDIO,
                              line_search_hz: float = LINE_SEARCH_HZ) -> float:
    """Line-SNR metric: min(peak/median) over the ±1000 Hz squared lines.

    For a complex-baseband signal carrying MSK144 (modulation tones at
    ±500 Hz around 0 Hz baseband), squaring produces spectral lines at
    ±1000 Hz.  We take the conservative SNR = min of the two line SNRs
    (peak / local median).
    """
    if len(summed_complex) == 0:
        return 0.0
    sq = (summed_complex.astype(np.complex128)) ** 2
    nfft = len(sq)
    df   = sr / nfft
    spec = np.abs(np.fft.fftshift(np.fft.fft(sq)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/sr))

    # Lines at ±1000 Hz in complex baseband
    half = line_search_hz
    mask_lo = (freqs > -1000 - half) & (freqs < -1000 + half)
    mask_hi = (freqs >  1000 - half) & (freqs <  1000 + half)
    if not mask_lo.any() or not mask_hi.any():
        return 0.0

    peak_lo = float(spec[mask_lo].max())
    peak_hi = float(spec[mask_hi].max())
    med_lo  = float(np.median(spec[mask_lo]))
    med_hi  = float(np.median(spec[mask_hi]))

    snr_lo = peak_lo / med_lo if med_lo > 0 else 0.0
    snr_hi = peak_hi / med_hi if med_hi > 0 else 0.0
    return float(min(snr_lo, snr_hi))


# ── Per-burst measurement ────────────────────────────────────────────────

@dataclass
class FrameAddRecord:
    """One step of the experiment: an attempted frame addition."""
    wav:                 str
    burst_idx:           int
    step_type:           str     # 'signal' | 'noise'
    step:                int     # 1, 2, 3, … (which signal frame, or trial j)
    line_snr_before:     float
    line_snr_after:      float
    delta:               float


def _frame_at(baseband: np.ndarray, start_n: int) -> np.ndarray:
    """Extract one NSPM-sample complex-baseband frame starting at start_n.

    Returns NSPM zeros if the requested range is outside the buffer.
    """
    if start_n < 0 or start_n + NSPM > len(baseband):
        return np.zeros(NSPM, dtype=np.complex64)
    return baseband[start_n:start_n + NSPM].copy()


def _frame_at_f(baseband: np.ndarray, start_n: int, f_hz: float,
                 sr: int = SR_AUDIO) -> np.ndarray:
    """Frame from baseband, mix-corrected by −f_hz.  Returns NSPM-length complex."""
    frame = _frame_at(baseband, start_n)
    if len(frame) != NSPM:
        return frame
    t = np.arange(NSPM, dtype=np.float64) / sr
    # Note: the "absolute time" within the WAV does NOT matter for line-SNR
    # purposes — what matters is that all frames are mixed by the same f̂
    # so their relative phase preserves coherent addition for the message.
    return (frame * np.exp(-2j * np.pi * f_hz * t).astype(np.complex64)).astype(np.complex64)


def measure_burst_reliability(
    audio_real: np.ndarray,
    burst_idx_pairs: list[tuple[int, int]],
    burst_idx: int,
    sr: int = SR_AUDIO,
    rng: np.random.Generator | None = None,
) -> list[FrameAddRecord]:
    """For one burst, produce signal-slope and noise-delta records.

    burst_idx_pairs is the full list (envelope bin indices) so we can
    exclude all burst windows when picking noise positions.
    """
    if rng is None:
        rng = np.random.default_rng()
    records: list[FrameAddRecord] = []
    bin_s_env = 1.0  # placeholder; we'll convert below

    # Get audio-sample-level burst bounds
    i_start_env, i_end_env = burst_idx_pairs[burst_idx]
    # Re-resolve to samples: caller is responsible for passing burst_idx_pairs
    # in audio-sample units already
    burst_start_n, burst_end_n = i_start_env, i_end_env

    n_frames = (burst_end_n - burst_start_n) // NSPM
    if n_frames < MIN_BURST_FRAMES:
        return []  # insufficient

    # f̂ from squared-FFT tracker on the burst (any valid window)
    pts = squared_fft_msk_tracker(audio_real, burst_start_n, burst_end_n, sr=sr)
    valid_pts = [p for p in pts if p.valid]
    if not valid_pts:
        return []
    f_hat = float(np.median([p.f_carrier_hz for p in valid_pts]))

    # Build complex baseband once for the whole WAV
    baseband = _audio_to_baseband(audio_real, sr=sr)

    # Centre frame = first frame inside the burst
    centre_start = burst_start_n
    centre_frame = _frame_at_f(baseband, centre_start, f_hat, sr=sr)
    accumulator  = centre_frame.copy()
    line_snr_0   = line_snr_of_summed_frame(accumulator, sr=sr)

    # SIGNAL SLOPE — add each adjacent in-burst frame
    n_add_signal = min(n_frames - 1, MAX_TEST_FRAMES)
    prev_snr = line_snr_0
    accum_signal = centre_frame.copy()
    for step in range(1, n_add_signal + 1):
        frame_start = burst_start_n + step * NSPM
        if frame_start + NSPM > burst_end_n:
            break
        f_k = _frame_at_f(baseband, frame_start, f_hat, sr=sr)
        accum_signal = (accum_signal + f_k).astype(np.complex64)
        new_snr = line_snr_of_summed_frame(accum_signal, sr=sr)
        records.append(FrameAddRecord(
            wav='', burst_idx=burst_idx, step_type='signal',
            step=step,
            line_snr_before=prev_snr,
            line_snr_after=new_snr,
            delta=new_snr - prev_snr,
        ))
        prev_snr = new_snr

    # METRIC NOISE — add a single noise-only frame to the centre-alone baseline
    # Noise frame positions: anywhere in the WAV that is OUTSIDE all burst
    # windows, with margin so we don't catch burst tails
    margin_n = 2 * NSPM
    noise_windows: list[tuple[int, int]] = []
    last_end = margin_n
    burst_bounds_sorted = sorted(burst_idx_pairs)
    for (bs, be) in burst_bounds_sorted:
        if bs - margin_n > last_end:
            noise_windows.append((last_end, bs - margin_n))
        last_end = max(last_end, be + margin_n)
    tail_end = len(audio_real) - NSPM - margin_n
    if tail_end > last_end:
        noise_windows.append((last_end, tail_end))

    if not noise_windows:
        return records  # no clean noise to sample; skip this burst's noise trials

    # Pre-compute the centre-only line_snr for the noise-delta baseline
    noise_base_snr = line_snr_0

    # Sample N noise positions
    for trial in range(1, N_NOISE_TRIALS_PER_BURST + 1):
        # Pick a random noise window proportional to its length
        lengths = np.array([w[1] - w[0] for w in noise_windows])
        if lengths.sum() <= 0:
            break
        probs = lengths / lengths.sum()
        wi = int(rng.choice(len(noise_windows), p=probs))
        start = int(rng.integers(noise_windows[wi][0], noise_windows[wi][1]))
        # Snap to frame boundary if convenient; not required
        noise_frame = _frame_at_f(baseband, start, f_hat, sr=sr)
        if len(noise_frame) != NSPM:
            continue
        accum_noise = (centre_frame + noise_frame).astype(np.complex64)
        new_snr = line_snr_of_summed_frame(accum_noise, sr=sr)
        records.append(FrameAddRecord(
            wav='', burst_idx=burst_idx, step_type='noise',
            step=trial,
            line_snr_before=noise_base_snr,
            line_snr_after=new_snr,
            delta=new_snr - noise_base_snr,
        ))

    return records


# ── WAV-level driver ─────────────────────────────────────────────────────

def analyse_wav(path: Path,
                rng: np.random.Generator) -> list[FrameAddRecord]:
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    if sr != SR_AUDIO:
        raise ValueError(f"expected {SR_AUDIO} Hz, got {sr}")

    audio_bp = _bandpass(audio, sr)
    env_db, bin_s = _envelope_db(audio_bp, sr)
    burst_idx_pairs_env = _find_bursts(env_db, bin_s, BURST_THRESH_DB, MIN_BURST_S)
    # Convert envelope-bin indices → audio sample indices
    burst_idx_pairs = [(int(s * bin_s * sr), int(e * bin_s * sr))
                       for (s, e) in burst_idx_pairs_env]

    all_records: list[FrameAddRecord] = []
    for bi, (bs, be) in enumerate(burst_idx_pairs):
        recs = measure_burst_reliability(audio, burst_idx_pairs, bi,
                                          sr=sr, rng=rng)
        for r in recs:
            r.wav = path.name
            all_records.append(r)
    return all_records


# ── Aggregate plotting ──────────────────────────────────────────────────

def plot_distributions(records: list[FrameAddRecord], out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sig_deltas   = np.array([r.delta for r in records if r.step_type == 'signal'])
    noise_deltas = np.array([r.delta for r in records if r.step_type == 'noise'])

    fig, (ax_hist, ax_slope) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: histogram of per-add line_snr deltas ────────────────────────
    if sig_deltas.size and noise_deltas.size:
        all_min = min(float(sig_deltas.min()), float(noise_deltas.min()))
        all_max = max(float(sig_deltas.max()), float(noise_deltas.max()))
        bins = np.linspace(all_min, all_max, 40)
        ax_hist.hist(noise_deltas, bins=bins, color='tab:red', alpha=0.55,
                      label=f'noise add  (n={len(noise_deltas)})',
                      edgecolor='black', linewidth=0.4)
        ax_hist.hist(sig_deltas, bins=bins, color='tab:green', alpha=0.55,
                      label=f'signal add (n={len(sig_deltas)})',
                      edgecolor='black', linewidth=0.4)
        # 2σ noise threshold (from noise std) — proposed Δ_min
        noise_std = float(noise_deltas.std())
        noise_mean = float(noise_deltas.mean())
        ax_hist.axvline(noise_mean, color='tab:red', ls=':', lw=1.2,
                         label=f'noise mean = {noise_mean:+.2f}')
        ax_hist.axvline(noise_mean + 2*noise_std, color='tab:red', ls='--', lw=1.2,
                         label=f'+2σ_noise = {noise_mean + 2*noise_std:+.2f} (Δ_min candidate)')
        ax_hist.axvline(0, color='k', ls='-', lw=0.5, alpha=0.5)
    ax_hist.set_xlabel('Δ line_snr after adding one frame')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Signal-add vs noise-add deltas — '
                       'separation = how decisive the greedy algorithm can be')
    ax_hist.legend(loc='best', fontsize=8)
    ax_hist.grid(alpha=0.3)

    # ── Right: line_snr trajectory through signal-frame additions ─────────
    # Group records by (wav, burst) and plot trajectories
    from collections import defaultdict
    sig_records = [r for r in records if r.step_type == 'signal']
    groups = defaultdict(list)
    for r in sig_records:
        groups[(r.wav, r.burst_idx)].append(r)
    for key, recs in groups.items():
        recs.sort(key=lambda r: r.step)
        steps = [0] + [r.step for r in recs]
        snrs  = [recs[0].line_snr_before] + [r.line_snr_after for r in recs]
        ax_slope.plot(steps, snrs, '-o', alpha=0.5, lw=0.8, markersize=3)
    ax_slope.set_xlabel('Frames added (0 = centre alone)')
    ax_slope.set_ylabel('line_snr of accumulator')
    ax_slope.set_title(f'Signal-slope trajectories  ({len(groups)} bursts)')
    ax_slope.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None)
    ap.add_argument('--wsjtx-dir', type=Path,
                     default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None,
                     help='Batch: process all WAVs for this UTC date (YYYYMMDD)')
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--plot', type=Path, default=None)
    ap.add_argument('--max-wavs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42,
                     help='RNG seed for noise-position sampling')
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    save_dir = args.wsjtx_dir / 'save'

    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        recs = analyse_wav(wav_path, rng)
        for r in recs:
            print(f"  {r.wav}  b{r.burst_idx}  {r.step_type:6s}  step={r.step:2d}  "
                  f"before={r.line_snr_before:6.2f}  after={r.line_snr_after:6.2f}  "
                  f"Δ={r.delta:+6.2f}")
        return 0

    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]

    print(f"Processing {len(wavs)} WAVs", file=sys.stderr)
    all_records: list[FrameAddRecord] = []
    n_qualifying_bursts = 0
    for i, p in enumerate(wavs):
        try:
            recs = analyse_wav(p, rng)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        all_records.extend(recs)
        n_qualifying_bursts += len({(r.wav, r.burst_idx) for r in recs})
        if (i + 1) % 50 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(all_records)} records  "
                  f"{n_qualifying_bursts} qualifying bursts", file=sys.stderr)

    if args.csv and all_records:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'burst_idx', 'step_type', 'step',
                        'line_snr_before', 'line_snr_after', 'delta'])
            for r in all_records:
                w.writerow([r.wav, r.burst_idx, r.step_type, r.step,
                            f'{r.line_snr_before:.3f}',
                            f'{r.line_snr_after:.3f}',
                            f'{r.delta:+.3f}'])
        print(f"\nCSV: {args.csv}")

    sig = np.array([r.delta for r in all_records if r.step_type == 'signal'])
    noise = np.array([r.delta for r in all_records if r.step_type == 'noise'])

    print()
    print(f"=== Metric reliability summary ===")
    print(f"Qualifying bursts: {n_qualifying_bursts}  (n_frames ≥ {MIN_BURST_FRAMES})")
    if sig.size:
        print(f"Signal-add deltas  (n={len(sig):3d}):  "
              f"p25={np.percentile(sig,25):+.2f}  "
              f"median={np.median(sig):+.2f}  "
              f"p75={np.percentile(sig,75):+.2f}  "
              f"mean={sig.mean():+.2f}")
    if noise.size:
        n_std = noise.std()
        n_mean = noise.mean()
        print(f"Noise-add deltas   (n={len(noise):3d}):  "
              f"p25={np.percentile(noise,25):+.2f}  "
              f"median={np.median(noise):+.2f}  "
              f"p75={np.percentile(noise,75):+.2f}  "
              f"mean={n_mean:+.2f}  std={n_std:.2f}")
        delta_min = n_mean + 2 * n_std
        if sig.size:
            sig_pass = (sig > delta_min).sum() / len(sig)
            print(f"Proposed Δ_min = noise_mean + 2σ = {delta_min:+.2f}")
            print(f"Fraction of signal-adds that would PASS Δ_min: "
                  f"{sig_pass:.1%}  ({(sig > delta_min).sum()}/{len(sig)})")
            print(f"                 (=true-positive rate at false-include ≈ 2.3%)")

    if args.plot and all_records:
        plot_distributions(all_records, args.plot)
        print(f"Plot: {args.plot}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
