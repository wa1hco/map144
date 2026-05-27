"""Focused experiment: use analyze_msk144's WSJT-X-style detector to find
candidate frames in 260511_105430.wav, then test whether greedy coherent
integration can decode the 4.824 s region (which analyze_msk144 detected
but could not decode standalone).

This is a one-off test of the architectural question: does coherent
integration across multiple sub-detection-threshold frames pull out new
decodes that individual-frame attempts miss?

Method
------
1. Load WAV.
2. Run analyze_msk144._compute_squared_spectrogram to get det_norm[t]
   and frame freq offsets across the whole WAV.
3. Identify candidate frames where det_norm >= 3.0.
4. For each cluster of candidate frames close in time:
   a. Pick the strongest as the centre frame
   b. Use its f̂ for mix-correction
   c. Coherently sum the cluster's frames after mix-correction
   d. Run sync correlation + jt9 on the sum
5. Compare:
   - Per-frame baseline (centre alone)
   - Greedy sum (all cluster frames combined)
   - vs analyze_msk144's standalone results

Specifically interested in the 4.824 s region — analyze_msk144 detected
it (peak 3.42, 324 ms) but no standalone decode.  Can a sum of multiple
candidate frames in 3.5-5.0 s decode it?
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_msk144 import _compute_squared_spectrogram, DETECT_THRESH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_burst_freq_afc_correct import _audio_to_baseband, SR_AUDIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import _sync_correlate, _frame_to_wav_and_decode, NSPM


WAV_PATH = '/home/jeff/.local/share/WSJT-X - flex/save/260511_105430.wav'


def _phantom_check(msg: str | None) -> bool:
    """Phantom regex from memory note project_phantom_decode_signature."""
    if not msg:
        return False
    import re
    RE_CALL = re.compile(r'^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z]$')
    RE_GRID = re.compile(r'^[A-R]{2}\d{2}([A-X]{2})?$')
    RE_REPORT = re.compile(r'^R?[+-]?\d{1,3}$')
    PREFIX = {'CQ', 'QRZ', 'DE'}
    SUFFIX = {'73', 'RR73', 'RRR', 'TNX'}
    for tok in msg.split():
        if tok in PREFIX or tok in SUFFIX:
            continue
        if RE_CALL.match(tok) or RE_GRID.match(tok) or RE_REPORT.match(tok):
            continue
        return False
    return True


def _f_hat_from_squared_spectrum(audio_analytic: np.ndarray,
                                   frame_start_n: int,
                                   sr: int = SR_AUDIO) -> float:
    """Compute f̂ (carrier offset from 1500 Hz) by squared-FFT on one frame.

    Uses analytic signal already.  Returns offset from 1500 Hz carrier.
    """
    blk = audio_analytic[frame_start_n:frame_start_n + NSPM].astype(np.complex128)
    # Match analyze_msk144's edge fade
    edge_n = 12
    rcw = (1.0 - np.cos(np.arange(edge_n) * np.pi / edge_n)) / 2.0
    blk_sq = blk ** 2
    blk_sq[:edge_n]        *= rcw
    blk_sq[NSPM - edge_n:] *= rcw[::-1]
    fft_sq = np.fft.fft(blk_sq, n=NSPM)
    tone_pos = np.abs(fft_sq[:NSPM // 2 + 1]) ** 2
    df = sr / NSPM
    bin_2k = int(round(2000 / df))
    bin_4k = int(round(4000 / df))
    hw = int(round(2 * 50.0 / df))  # ±100 Hz

    lo_l = max(0, bin_2k - hw); lo_h = min(len(tone_pos), bin_2k + hw + 1)
    hi_l = max(0, bin_4k - hw); hi_h = min(len(tone_pos), bin_4k + hw + 1)

    peak_2k_val = float(tone_pos[lo_l:lo_h].max())
    peak_4k_val = float(tone_pos[hi_l:hi_h].max())
    if peak_4k_val > peak_2k_val:
        i = hi_l + int(np.argmax(tone_pos[hi_l:hi_h]))
        # parabolic refine
        if 1 <= i < len(tone_pos) - 1:
            a, b, c = tone_pos[i-1], tone_pos[i], tone_pos[i+1]
            den = a - 2*b + c
            if den != 0:
                i_refined = i + 0.5 * (a - c) / den
            else:
                i_refined = float(i)
        else:
            i_refined = float(i)
        f_squared = i_refined * df
        return (f_squared - 4000) / 2.0
    else:
        i = lo_l + int(np.argmax(tone_pos[lo_l:lo_h]))
        if 1 <= i < len(tone_pos) - 1:
            a, b, c = tone_pos[i-1], tone_pos[i], tone_pos[i+1]
            den = a - 2*b + c
            if den != 0:
                i_refined = i + 0.5 * (a - c) / den
            else:
                i_refined = float(i)
        else:
            i_refined = float(i)
        f_squared = i_refined * df
        return (f_squared - 2000) / 2.0


def _mix_correct_frame_at(baseband: np.ndarray, frame_start_n: int,
                            f_hz: float, sr: int = SR_AUDIO) -> np.ndarray:
    if frame_start_n < 0 or frame_start_n + NSPM > len(baseband):
        return np.zeros(NSPM, dtype=np.complex64)
    frame = baseband[frame_start_n:frame_start_n + NSPM]
    t = np.arange(NSPM, dtype=np.float64) / sr
    return (frame * np.exp(-2j * np.pi * f_hz * t).astype(np.complex64)
            ).astype(np.complex64)


def _decode_complex_frame(c: np.ndarray) -> tuple[str | None, bool]:
    if c is None or len(c) != NSPM:
        return None, False
    _, _, ish_best = _sync_correlate(c)
    aligned = np.roll(c, -ish_best).astype(np.complex64)
    msg, _ = _frame_to_wav_and_decode(aligned, fc_out=1500.0)
    return msg, _phantom_check(msg)


def main() -> int:
    # Load
    with wave.open(WAV_PATH, 'rb') as wf:
        rate = wf.getframerate()
        audio_real = np.frombuffer(wf.readframes(wf.getnframes()),
                                     dtype=np.int16).astype(np.float64) / 32768.0
    print(f"Loaded: {WAV_PATH}")
    print(f"  rate={rate}, samples={len(audio_real)}, duration={len(audio_real)/rate:.2f}s")
    print()

    # Get analyze_msk144's detection metric across whole WAV
    time_s, freq_hz, spec_db, det_norm = _compute_squared_spectrogram(
        audio_real, rate, fc_hz=1500.0, ntol_hz=50.0
    )
    print(f"Detector ran: {len(time_s)} frames over the WAV")
    print(f"  det_norm distribution: p50={np.percentile(det_norm, 50):.2f}  "
          f"p90={np.percentile(det_norm, 90):.2f}  "
          f"max={det_norm.max():.2f}")
    print()

    # Find frames with det_norm >= 3.0
    above = det_norm >= DETECT_THRESH
    above_idx = np.where(above)[0]
    print(f"Frames with det_norm >= {DETECT_THRESH}: {len(above_idx)}")
    print()

    # Convert frame indices to sample positions
    # Step in samples per detector frame = NSPM/4 (75% overlap)
    step_samples = NSPM // 4
    print(f"{'frame_idx':>9}  {'t (s)':>7}  {'det_norm':>9}  {'sample_n':>9}")
    for fi in above_idx:
        t = time_s[fi]
        sample_n = int(t * rate - NSPM / 2)  # convert centre time to start sample
        print(f"  {fi:>7}  {t:>7.3f}  {det_norm[fi]:>9.2f}  {sample_n:>9}")
    print()

    # Build analytic signal for f̂ computation
    audio_analytic = scipy.signal.hilbert(audio_real).astype(np.complex128)
    baseband = _audio_to_baseband(audio_real.astype(np.float32), rate).astype(np.complex64)

    # Test 1: Per-frame f̂ at each high-det-norm position
    print("=== Per-frame f̂ at each candidate ===")
    print(f"{'t (s)':>7}  {'det_norm':>9}  {'f_hat (Hz)':>12}  {'sample_n':>9}")
    candidates = []
    for fi in above_idx:
        t = time_s[fi]
        sample_n = max(0, int(t * rate - NSPM / 2))
        if sample_n + NSPM > len(audio_real):
            continue
        f_hat = _f_hat_from_squared_spectrum(audio_analytic, sample_n, rate)
        candidates.append((t, det_norm[fi], f_hat, sample_n))
        print(f"  {t:>7.3f}  {det_norm[fi]:>9.2f}  {f_hat:>+12.2f}  {sample_n:>9}")
    print()

    # Focus on the 3.5-5.0 region — combine those candidates
    print("=== Decode attempts on 3.5-5.0 s region (user's weak ping) ===")
    region = [(t, dn, fh, sn) for (t, dn, fh, sn) in candidates if 3.5 <= t <= 5.2]
    print(f"  Candidates in region: {len(region)}")
    for (t, dn, fh, sn) in region:
        print(f"    t={t:.3f}  det_norm={dn:.2f}  f̂={fh:+.2f} Hz")
    print()

    if not region:
        print("  No candidates in region.")
        return 0

    # Get centre as strongest in region
    region.sort(key=lambda x: -x[1])
    centre = region[0]
    print(f"  Centre frame: t={centre[0]:.3f}  det_norm={centre[1]:.2f}  f̂={centre[2]:+.2f} Hz")
    print()

    # Build accumulator starting from centre, then try adding others greedily
    # using line_snr_of_summed_frame as the greedy validator
    from measure_afc_metric_reliability import line_snr_of_summed_frame

    f_hat = float(centre[2])
    centre_sn = centre[3]
    centre_frame = _mix_correct_frame_at(baseband, centre_sn, f_hat, rate)
    accumulator = centre_frame.copy()
    line_snr_curr = line_snr_of_summed_frame(accumulator, rate)
    msg, valid = _decode_complex_frame(accumulator)
    print(f"  After centre alone:  line_snr={line_snr_curr:.2f}  "
          f"decode={msg!r}  valid={valid}")

    # Try adding each other candidate (excluding centre)
    other = [(t, dn, fh, sn) for (t, dn, fh, sn) in candidates if (t, dn, fh, sn) != centre]
    # Sort by proximity to centre time
    other.sort(key=lambda x: abs(x[0] - centre[0]))
    included = [centre]
    for (t, dn, fh, sn) in other:
        frame = _mix_correct_frame_at(baseband, sn, f_hat, rate)  # use centre's f̂
        tentative = (accumulator + frame).astype(np.complex64)
        new_snr = line_snr_of_summed_frame(tentative, rate)
        accept = new_snr > line_snr_curr
        print(f"  Try add  t={t:.3f}  det_norm={dn:.2f}  "
              f"line_snr {line_snr_curr:.2f} → {new_snr:.2f}  "
              f"{'ACCEPT' if accept else 'reject'}")
        if accept:
            accumulator = tentative
            line_snr_curr = new_snr
            included.append((t, dn, fh, sn))

    # Final decode attempt
    msg, valid = _decode_complex_frame(accumulator)
    print()
    print(f"=== Final result ===")
    print(f"  Included {len(included)} frames at times: {[f'{c[0]:.3f}' for c in included]}")
    print(f"  Final line_snr: {line_snr_curr:.2f}")
    print(f"  Decode: {msg!r}  valid={valid}")

    # Test 2: Try all in-region candidates summed at common f̂
    print()
    print("=== Alternative: sum ALL 3.5-5.0 s region candidates at centre's f̂ ===")
    region_accum = np.zeros(NSPM, dtype=np.complex64)
    for (t, dn, fh, sn) in [(t, dn, fh, sn) for (t, dn, fh, sn) in candidates if 3.5 <= t <= 5.2]:
        frame = _mix_correct_frame_at(baseband, sn, f_hat, rate)
        region_accum = (region_accum + frame).astype(np.complex64)
    region_snr = line_snr_of_summed_frame(region_accum, rate)
    msg, valid = _decode_complex_frame(region_accum)
    print(f"  Sum of all {len(region)} candidates: line_snr={region_snr:.2f}  "
          f"decode={msg!r}  valid={valid}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
