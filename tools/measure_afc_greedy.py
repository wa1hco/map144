"""Step-2 prototype: blind greedy frame-inclusion AFC decoder.

Architecture (final per operator's reframing):

  burst audio
    → detect centre frame (highest single-frame line_snr in burst)
    → squared-FFT on centre frame → f̂  (centre is above detection by
       construction, so squared-FFT is reliable on it)
    → accumulator = mix_correct(centre)
    → for each other frame in the burst (distance-from-centre order):
        tentative   = accumulator + mix_correct(frame_k)
        new_snr     = line_snr(tentative)
        if new_snr > current_snr + Δ_min:
            accumulator = tentative
            current_snr = new_snr
        else:
            reject this frame; try next
    → final accumulator: one super-frame fed to decoder

The candidate frames are NOT pre-ranked by envelope or single-frame
metric — by construction they're below detection, so any such ranking
is degenerate.  The accumulator's line_snr metric is the ONLY decision
input.  The greedy algorithm accepts the metric noise as a cost.

Comparison metrics (per burst, vs single-frame centre baseline):
- single_frame: sync_corr peak on centre frame alone (AFC pre-corrected)
- all_burst   : sync_corr peak on coherent sum of ALL in-burst frames
                (the previous AFC prototype's output)
- greedy      : sync_corr peak on greedy-selected subset sum

Gains in dB relative to single-frame baseline.  If greedy reliably
beats both single-frame AND all-burst, the approach is validated.

Usage
-----
    python tools/measure_afc_greedy.py --wav 260511_111530.wav
    python tools/measure_afc_greedy.py --date 20260511 --csv OUT.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
)
from measure_afc_metric_reliability import line_snr_of_summed_frame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import _sync_correlate, NSPM


DELTA_MIN_DB = 0.0       # threshold for accepting a frame add (0 = any positive)


@dataclass
class GreedyResult:
    """Per-burst greedy decoder output + comparison baselines."""
    burst_idx:        int
    t_start_s:        float
    t_end_s:          float
    n_frames:         int
    f_hat:            float           # f̂ from squared-FFT on centre frame
    centre_idx:       int             # which in-burst frame was centre

    # Single-frame baseline: centre alone, mix-corrected, sync-correlated
    sf_peak:          float
    sf_line_snr:      float

    # All-burst-sum: ALL frames in burst summed coherently after mix-correct
    ab_peak:          float
    ab_line_snr:      float

    # Greedy: blind frame-inclusion using line_snr as validator
    greedy_peak:      float
    greedy_line_snr:  float
    included_frames:  list[int]       # which frame indices were kept

    # Per-step trace (line_snr_before, line_snr_after, accepted) for diagnostic
    trace:            list[tuple[int, float, float, bool]] = field(default_factory=list)


def _mix_correct_frame(baseband: np.ndarray, start_n: int,
                        f_hz: float, sr: int = SR_AUDIO) -> np.ndarray:
    """Extract one NSPM-sample frame and mix-correct by −f_hz."""
    if start_n < 0 or start_n + NSPM > len(baseband):
        return np.zeros(NSPM, dtype=np.complex64)
    frame = baseband[start_n:start_n + NSPM]
    t = np.arange(NSPM, dtype=np.float64) / sr
    return (frame * np.exp(-2j * np.pi * f_hz * t).astype(np.complex64)
            ).astype(np.complex64)


def greedy_decode_burst(audio_real: np.ndarray,
                          burst_start_n: int,
                          burst_end_n: int,
                          burst_idx: int = 0,
                          sr: int = SR_AUDIO,
                          delta_min_db: float = DELTA_MIN_DB,
                          ) -> GreedyResult | None:
    """Run the full single-frame / all-burst / greedy comparison on one burst."""
    n_frames = (burst_end_n - burst_start_n) // NSPM
    if n_frames < 1:
        return None

    # Build mix-corrected complex baseband once for the whole WAV
    baseband = _audio_to_baseband(audio_real, sr=sr)

    # 1. Identify centre frame: enumerate all frames in burst, mix-correct
    #    by 0 Hz (we don't have f̂ yet), measure single-frame line_snr,
    #    pick the strongest.  Frames are 72 ms; we can iterate cheaply.
    #
    #    Subtlety: line_snr of a non-mix-corrected frame is what we have
    #    before knowing the Doppler offset.  That's fine for picking the
    #    centre frame — it ranks by signal-power-at-MSK-lines which is
    #    invariant under small mix-shifts.  We just need to find the
    #    strongest frame, then estimate f̂ from it.
    frame_snrs = []
    for k in range(n_frames):
        st = burst_start_n + k * NSPM
        # Mix by 0 here = just convert to baseband, don't shift
        f_unmixed = _mix_correct_frame(baseband, st, 0.0, sr)
        s = line_snr_of_summed_frame(f_unmixed, sr=sr)
        frame_snrs.append((k, s))
    frame_snrs.sort(key=lambda x: -x[1])
    centre_idx, centre_snr = frame_snrs[0]
    centre_start = burst_start_n + centre_idx * NSPM

    # 2. Estimate f̂ from squared-FFT on the centre frame.  We use a
    #    single-frame window for the tracker because the centre is the
    #    only frame guaranteed above the squared detector's threshold.
    pts = squared_fft_msk_tracker(
        audio_real, centre_start, centre_start + NSPM, sr=sr,
        win_frames=1, step_frac=1.0,
    )
    valid = [p for p in pts if p.valid]
    if not valid:
        # Squared-FFT couldn't lock even on the strongest frame; fall back
        # to f̂ = 0 (assume signal is at 1500 Hz audio centre).  This is a
        # graceful degradation rather than a hard failure.
        f_hat = 0.0
    else:
        f_hat = float(np.median([p.f_carrier_hz for p in valid]))

    # 3. Build mix-corrected frames keyed by burst-position index
    mc_frames: dict[int, np.ndarray] = {}
    for k in range(n_frames):
        st = burst_start_n + k * NSPM
        mc_frames[k] = _mix_correct_frame(baseband, st, f_hat, sr)

    # ── Single-frame baseline ───────────────────────────────────────────
    sf_acc = mc_frames[centre_idx].copy()
    sf_line_snr = line_snr_of_summed_frame(sf_acc, sr=sr)
    _, sf_peak, _ = _sync_correlate(sf_acc)

    # ── All-burst sum baseline ──────────────────────────────────────────
    ab_acc = sum(mc_frames[k] for k in range(n_frames)).astype(np.complex64)
    ab_line_snr = line_snr_of_summed_frame(ab_acc, sr=sr)
    _, ab_peak, _ = _sync_correlate(ab_acc)

    # ── Greedy frame inclusion ──────────────────────────────────────────
    accumulator   = mc_frames[centre_idx].copy()
    current_snr   = sf_line_snr
    included      = [centre_idx]
    trace: list[tuple[int, float, float, bool]] = []

    # Candidate order: outward from centre, alternating directions.
    # This is "blind" in the sense that we don't pre-rank by quality —
    # we just try near-neighbours first, then progressively farther.
    candidates: list[int] = []
    for offset in range(1, n_frames):
        if centre_idx - offset >= 0:
            candidates.append(centre_idx - offset)
        if centre_idx + offset < n_frames:
            candidates.append(centre_idx + offset)

    for k in candidates:
        tentative = (accumulator + mc_frames[k]).astype(np.complex64)
        new_snr = line_snr_of_summed_frame(tentative, sr=sr)
        # Δ_min specified in dB on the SNR ratio.  Convert: a multiplicative
        # ratio of r = 10^(δ/20) on amplitudes ⇒ accept iff new ≥ r·current.
        ratio_threshold = 10.0 ** (delta_min_db / 20.0)
        accept = (new_snr > current_snr * ratio_threshold) if current_snr > 0 else (new_snr > 0)
        trace.append((k, current_snr, new_snr, accept))
        if accept:
            accumulator = tentative
            current_snr = new_snr
            included.append(k)

    greedy_line_snr = current_snr
    _, greedy_peak, _ = _sync_correlate(accumulator)

    return GreedyResult(
        burst_idx=burst_idx,
        t_start_s=burst_start_n / sr,
        t_end_s=burst_end_n / sr,
        n_frames=n_frames,
        f_hat=f_hat,
        centre_idx=centre_idx,
        sf_peak=float(sf_peak),
        sf_line_snr=float(sf_line_snr),
        ab_peak=float(ab_peak),
        ab_line_snr=float(ab_line_snr),
        greedy_peak=float(greedy_peak),
        greedy_line_snr=float(greedy_line_snr),
        included_frames=sorted(included),
        trace=trace,
    )


# ── WAV-level driver ─────────────────────────────────────────────────────

def analyse_wav(path: Path,
                delta_min_db: float = DELTA_MIN_DB,
                ) -> tuple[np.ndarray, float, list[GreedyResult]]:
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

    results = []
    for bi, (i_start, i_end) in enumerate(burst_idx_pairs):
        burst_start_n = int(i_start * bin_s * sr)
        burst_end_n = int(i_end * bin_s * sr)
        r = greedy_decode_burst(audio, burst_start_n, burst_end_n,
                                  burst_idx=bi, sr=sr,
                                  delta_min_db=delta_min_db)
        if r is not None:
            results.append(r)
    return env_db, bin_s, results


# ── CLI ──────────────────────────────────────────────────────────────────

def _print_result_line(name: str, r: GreedyResult) -> None:
    sf_dB = 0.0   # baseline
    ab_dB = 20.0 * np.log10(r.ab_peak / r.sf_peak) if r.sf_peak > 0 and r.ab_peak > 0 else float('nan')
    gr_dB = 20.0 * np.log10(r.greedy_peak / r.sf_peak) if r.sf_peak > 0 and r.greedy_peak > 0 else float('nan')
    print(f"  {name}  centre={r.centre_idx:>2}  {r.n_frames}fr  "
          f"sf={r.sf_peak:>5.0f}  ab={r.ab_peak:>5.0f}  gr={r.greedy_peak:>5.0f}  "
          f"| ab={ab_dB:+5.2f} dB  gr={gr_dB:+5.2f} dB  "
          f"| kept={r.included_frames}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None)
    ap.add_argument('--wsjtx-dir', type=Path,
                     default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None)
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--max-wavs', type=int, default=None)
    ap.add_argument('--delta-min-db', type=float, default=DELTA_MIN_DB,
                     help='Min Δ line_snr (dB) for greedy add; default 0 = any positive')
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt) if all_txt.exists() else {}

    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        env_db, bin_s, results = analyse_wav(wav_path, args.delta_min_db)
        msgs = _msgs_for_wav(wav_path, msg_index)
        msgs_str = '  /  '.join(msgs) if msgs else '(none)'
        print(f"{wav_path.name}  ({len(results)} bursts)  WSJT-X: {msgs_str}")
        for r in results:
            _print_result_line(f"b{r.burst_idx+1}", r)
        return 0

    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]

    print(f"Processing {len(wavs)} WAVs (Δ_min = {args.delta_min_db:+.2f} dB)",
          file=sys.stderr)
    all_results: list[tuple[str, str, GreedyResult]] = []
    for i, p in enumerate(wavs):
        try:
            env_db, bin_s, results = analyse_wav(p, args.delta_min_db)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        msgs = _msgs_for_wav(p, msg_index)
        msgs_str = '|'.join(msgs)
        for r in results:
            all_results.append((p.name, msgs_str, r))
        if (i + 1) % 50 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(all_results)} bursts",
                  file=sys.stderr)

    if args.csv and all_results:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'wsjtx_msgs', 'burst_idx', 't_start_s', 't_end_s',
                        'n_frames', 'centre_idx', 'f_hat_hz',
                        'sf_peak', 'ab_peak', 'greedy_peak',
                        'sf_line_snr', 'ab_line_snr', 'greedy_line_snr',
                        'ab_gain_db', 'greedy_gain_db', 'n_included', 'included'])
            for name, msgs, r in all_results:
                ab_dB = 20.0*np.log10(r.ab_peak/r.sf_peak) if r.sf_peak > 0 and r.ab_peak > 0 else float('nan')
                gr_dB = 20.0*np.log10(r.greedy_peak/r.sf_peak) if r.sf_peak > 0 and r.greedy_peak > 0 else float('nan')
                w.writerow([name, msgs, r.burst_idx,
                            f'{r.t_start_s:.3f}', f'{r.t_end_s:.3f}',
                            r.n_frames, r.centre_idx, f'{r.f_hat:+.2f}',
                            f'{r.sf_peak:.0f}', f'{r.ab_peak:.0f}', f'{r.greedy_peak:.0f}',
                            f'{r.sf_line_snr:.2f}', f'{r.ab_line_snr:.2f}',
                            f'{r.greedy_line_snr:.2f}',
                            f'{ab_dB:+.2f}', f'{gr_dB:+.2f}',
                            len(r.included_frames),
                            ' '.join(map(str, r.included_frames))])
        print(f"\nCSV: {args.csv}")

    # Summary
    if all_results:
        ab_gains = []
        gr_gains = []
        for _, _, r in all_results:
            if r.sf_peak > 0 and r.ab_peak > 0:
                ab_gains.append(20.0*np.log10(r.ab_peak/r.sf_peak))
            if r.sf_peak > 0 and r.greedy_peak > 0:
                gr_gains.append(20.0*np.log10(r.greedy_peak/r.sf_peak))
        ab_gains = np.array(ab_gains)
        gr_gains = np.array(gr_gains)
        print()
        print(f"=== Comparison vs single-frame baseline  ({len(all_results)} bursts) ===")
        print(f"  all-burst sum gain (dB):    "
              f"p25={np.percentile(ab_gains,25):+.2f}  "
              f"median={np.median(ab_gains):+.2f}  "
              f"p75={np.percentile(ab_gains,75):+.2f}  "
              f"max={ab_gains.max():+.2f}  "
              f">0: {100*np.sum(ab_gains>0)/len(ab_gains):.0f}%  "
              f">1: {100*np.sum(ab_gains>1)/len(ab_gains):.0f}%")
        print(f"  greedy        gain (dB):    "
              f"p25={np.percentile(gr_gains,25):+.2f}  "
              f"median={np.median(gr_gains):+.2f}  "
              f"p75={np.percentile(gr_gains,75):+.2f}  "
              f"max={gr_gains.max():+.2f}  "
              f">0: {100*np.sum(gr_gains>0)/len(gr_gains):.0f}%  "
              f">1: {100*np.sum(gr_gains>1)/len(gr_gains):.0f}%")
        # Frame-count distribution for greedy
        n_inc = [len(r.included_frames) for _, _, r in all_results]
        print(f"  greedy frames-kept: median={int(np.median(n_inc))}  "
              f"range=[{min(n_inc)}, {max(n_inc)}]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
