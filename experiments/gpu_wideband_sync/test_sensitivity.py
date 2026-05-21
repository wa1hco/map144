# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Measure sync-over-15s sensitivity + frequency-estimation accuracy.

CPU-only test — no GPU required.  Designed to validate the wideband
sync matched filter against a fair sq_det baseline before any GPU work.

Inputs:
    Pair of pre-generated ramp simulation files:
      * IQ WAV (15 s, 48 kHz complex IQ, 16 MSK144 pings at varying SNR)
      * Truth JSON  (placements: msg, center_hz, delay_s, snr_db per ping)

What it measures:
    1. Detection rate per SNR bucket — wideband-sync vs sliding-window sq_det
       on the same audio.  Same threshold-on-noise-floor convention for both
       so the comparison isn't rigged.
    2. Frequency-estimation error vs truth's ``center_hz`` for sync-detected
       candidates.  Mean / std / max-abs reported per SNR bucket.

Output:
    Console summary table.
    Optional CSV via ``--out FILE`` for further plotting / aggregation.

Usage:
    python -m experiments.gpu_wideband_sync.test_sensitivity \\
        MSK144/simulations/ramp_<stamp>_iq.wav \\
        MSK144/simulations/ramp_<stamp>_iq.json \\
        --out /tmp/sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running from repo root.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_wideband_sync import sync_template, wideband_sync


# ── Data loaders ────────────────────────────────────────────────────────────


def load_wav_iq(path: Path) -> tuple[np.ndarray, int]:
    """Load any ramp WAV — IQ-stereo or mono-audio — and return complex IQ.

    Handles two cases the simulator emits:

      * **48 kHz IQ stereo** (``ramp_*_iq.wav``): left=I, right=Q (float).
        Returned as native complex.
      * **12 kHz audio mono** (``ramp_*_audio.wav``): real-valued audio
        (int16 or float).  Converted to analytic (complex) signal via
        Hilbert transform so the downstream matched filter (which expects
        complex input) works without modification.

    Auto-detects sample rate from the WAV header and returns it alongside
    the IQ samples.

    Returns
    -------
    iq         : complex64 ndarray, shape (n,)
    sample_rate_hz : int  — native rate from the WAV header
    """
    import scipy.io.wavfile as wavfile           # noqa: PLC0415
    from scipy.signal import hilbert             # noqa: PLC0415

    sr, data = wavfile.read(str(path))
    # data dtype handling: int16 → normalise to [-1, 1] float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 1:
        # Real-valued audio.  Convert to analytic signal so the matched
        # filter sees a complex stream.
        iq = hilbert(data).astype(np.complex64)
    elif data.ndim == 2 and data.shape[1] == 2:
        # Stereo IQ: left=I, right=Q.
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    elif data.ndim == 2 and data.shape[1] == 1:
        iq = hilbert(data[:, 0]).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported WAV shape {data.shape}")
    return np.ascontiguousarray(iq), int(sr)


@dataclass
class TruthPing:
    """One ground-truth ping placement from the simulation."""
    msg: str
    center_hz: float
    delay_s: float
    snr_db: float
    width_ms: int


def load_truth(path: Path) -> tuple[list[TruthPing], dict]:
    """Load the truth JSON sidecar.  Returns (pings, metadata)."""
    with path.open() as f:
        j = json.load(f)
    pings = [TruthPing(
        msg=p['msg'],
        center_hz=float(p['center_hz']),
        delay_s=float(p['delay_s']),
        snr_db=float(p['snr_db']),
        width_ms=int(p.get('width_ms', 0)),
    ) for p in j.get('placements', [])]
    meta = {
        'duration_s':       float(j['duration_s']),
        'output_rate':      int(j['output_rate']),
        'iq_offset_hz':     float(j.get('iq_offset_hz', 0.0)),
        'noise_floor_dbfs': j.get('noise_floor_dbfs'),
    }
    return pings, meta


# ── sq_det baseline (sliding-window WSJT-X-faithful) ────────────────────────


def run_sq_det_baseline(iq: np.ndarray, sample_rate_hz: int,
                       assumed_fc_hz: float = 1500.0,
                       threshold_db: float = 10.0) -> list[dict]:
    """Sliding-window per-frame sq_det.

    Runs the WSJT-X-faithful sq_det implementation at every NSPM/4 step
    across the input.  Returns candidate dicts with time_s, freq_hz, detmet.

    sq_det operates at 12 kHz (WSJT-X channelized rate).  If the input
    sample_rate is already 12 kHz, no decimation; otherwise decimate by
    the integer ratio.

    ``assumed_fc_hz`` tells sq_det where in the audio band the signal's
    carrier is expected — for the ramp simulations, 1500 Hz.  Squared
    tones land at 2*(fc-500) and 2*(fc+500).
    """
    from map144_app.sq_det import sq_det, NSPM    # noqa: PLC0415
    from scipy.signal import decimate             # noqa: PLC0415

    if sample_rate_hz == 12_000:
        audio_12k = iq.astype(np.complex128)
    elif sample_rate_hz % 12_000 == 0:
        factor = sample_rate_hz // 12_000
        audio_12k = decimate(iq, factor, ftype='fir', zero_phase=True).astype(np.complex128)
    else:
        raise ValueError(f"sample_rate_hz {sample_rate_hz} not 12 kHz or a "
                         f"multiple thereof — can't decimate to sq_det's 12 kHz")

    hop = NSPM // 4    # 216 samples = 18 ms at 12 kHz
    detmets: list[float] = []
    ferr_avgs: list[float] = []   # save per-window fc-offset estimate

    n = audio_12k.size
    for start in range(0, n - NSPM + 1, hop):
        frame = audio_12k[start:start + NSPM]
        result = sq_det(frame, fc=assumed_fc_hz)
        detmets.append(float(result.detmet))
        ferr_avgs.append(float((result.ferr_lo + result.ferr_hi) / 2.0))

    detmets = np.array(detmets, dtype=np.float32)
    xmed = float(np.percentile(detmets, 25))
    if xmed <= 0:
        xmed = 1e-30
    detmet_norm = detmets / xmed

    thresh_lin = 10 ** (threshold_db / 20.0)

    candidates = []
    for i, m in enumerate(detmet_norm):
        if m < thresh_lin:
            continue
        time_s = (i * hop) / 12_000.0
        # Absolute audio freq = assumed_fc + within-channel residual
        freq_hz = assumed_fc_hz + ferr_avgs[i]
        candidates.append({
            'time_s':     time_s,
            'freq_hz':    freq_hz,
            'detmet':     float(detmets[i]),
            'detmet_norm': float(m),
            'snr_db':     float(20.0 * np.log10(max(m, 1e-30))),
        })
    return candidates


# ── Matching candidates to truth ────────────────────────────────────────────


def match_candidates(candidates: list, truth: list[TruthPing],
                      time_tol_s: float = 0.5,
                      freq_tol_hz: float = 500.0,
                      key_time='time_s', key_freq='freq_hz') -> dict:
    """Match candidates to truth pings; return dict[ping_idx] = best_candidate."""
    matched = {}
    for i, ping in enumerate(truth):
        best = None
        best_dist = None
        for c in candidates:
            ct = c[key_time] if isinstance(c, dict) else getattr(c, key_time)
            cf = c[key_freq] if isinstance(c, dict) else getattr(c, key_freq)
            dt = abs(ct - ping.delay_s)
            df = abs(cf - ping.center_hz)
            if dt < time_tol_s and df < freq_tol_hz:
                # Composite distance: prefer time-closer matches
                dist = dt + df / 1000.0
                if best is None or dist < best_dist:
                    best = c
                    best_dist = dist
        if best is not None:
            matched[i] = best
    return matched


# ── Main test driver ────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('iq_wav', type=Path, help='IQ WAV file (15 s, 48 kHz)')
    ap.add_argument('truth_json', type=Path, help='Truth JSON sidecar')
    ap.add_argument('--out', type=Path, default=None,
                    help='Optional CSV output (one row per truth ping)')
    ap.add_argument('--sync-threshold-db', type=float, default=10.0,
                    help='Wideband sync detection threshold (dB above '
                         'per-bin noise floor)  [default: 10]')
    ap.add_argument('--sqdet-threshold-db', type=float, default=9.5,
                    help='sq_det detection threshold (dB; WSJT-X primary '
                         'uses ~9.5 = detmet_norm >= 3.0)  [default: 9.5]')
    ap.add_argument('--sync-stride-samples', type=int, default=24,
                    help='Wideband sync correlation stride (samples at '
                         '48 kHz; 24 = one MSK144 symbol)  [default: 24]')
    args = ap.parse_args()

    # Load — auto-detect IQ vs audio, native sample rate
    print(f"Loading WAV:   {args.iq_wav}")
    iq, wav_sample_rate = load_wav_iq(args.iq_wav)
    print(f"  Native sample rate: {wav_sample_rate} Hz")
    print(f"  Samples: {iq.size}  ({iq.size / wav_sample_rate:.2f} s)")
    print(f"Loading truth: {args.truth_json}")
    truth, meta = load_truth(args.truth_json)
    print(f"  Duration: {meta['duration_s']} s, "
          f"truth says output_rate: {meta['output_rate']} Hz, "
          f"{len(truth)} pings")
    if wav_sample_rate != meta['output_rate']:
        print(f"  (WAV rate differs from truth metadata — using WAV rate)")

    # The sync template lives at the WAV's native sample rate so the
    # matched filter operates without resampling.
    SR = wav_sample_rate

    # ── Wideband sync detector (the "future" path) ─────────────────────
    print("\n=== Wideband sync over 15-s window ===")
    h = sync_template.build_sync_template(SR)
    cfg = wideband_sync.WidebandSyncConfig(
        sample_rate_hz=SR,
        stride_samples=args.sync_stride_samples,
        threshold_db=args.sync_threshold_db,
    )
    sync_candidates = wideband_sync.detect(iq, h, cfg, use_gpu=False)
    print(f"  template length: {len(h)} samples ({1000*len(h)/SR:.1f} ms)")
    print(f"  freq resolution: {SR/len(h):.1f} Hz/bin")
    print(f"  {len(sync_candidates)} candidates above {args.sync_threshold_db:.1f} dB")
    sync_matches = match_candidates(
        sync_candidates, truth,
        key_time='time_s', key_freq='freq_hz',
    )

    # ── sq_det baseline (the "current" path) ────────────────────────────
    print("\n=== sq_det baseline (sliding-window, WSJT-X-faithful) ===")
    sq_candidates = run_sq_det_baseline(
        iq, SR, threshold_db=args.sqdet_threshold_db,
    )
    print(f"  {len(sq_candidates)} candidates above {args.sqdet_threshold_db:.1f} dB")
    sq_matches = match_candidates(
        sq_candidates, truth,
        key_time='time_s', key_freq='freq_hz',
    )

    # ── Per-ping report ─────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'msg':<14s} {'SNR':>5s} {'truth_f':>8s} "
          f"{'sync_det':>9s} {'sync_f_err':>11s} {'sync_db':>8s}  "
          f"{'sq_det':>7s} {'sq_db':>7s}")
    print(f"{'-'*78}")
    rows = []
    for i, ping in enumerate(truth):
        sync_match = sync_matches.get(i)
        sq_match = sq_matches.get(i)
        sync_det = 'YES' if sync_match else '   '
        sq_det = 'YES' if sq_match else '   '
        if sync_match:
            sync_f_err = sync_match.freq_hz - ping.center_hz
            sync_db = sync_match.snr_db
        else:
            sync_f_err = None
            sync_db = None
        sq_db = sq_match['snr_db'] if sq_match else None
        print(f"{ping.msg:<14s} {ping.snr_db:>+5.1f} {ping.center_hz:>8.1f} "
              f"{sync_det:>9s} "
              f"{('%+.1f' % sync_f_err) if sync_f_err is not None else '       —':>11s} "
              f"{('%.1f' % sync_db) if sync_db is not None else '     —':>8s}  "
              f"{sq_det:>7s} "
              f"{('%.1f' % sq_db) if sq_db is not None else '     —':>7s}")
        rows.append({
            'msg':                ping.msg,
            'snr_truth_db':       ping.snr_db,
            'time_s_truth':       ping.delay_s,
            'freq_hz_truth':      ping.center_hz,
            'detected_by_sync':   1 if sync_match else 0,
            'sync_time_s':        sync_match.time_s if sync_match else None,
            'sync_freq_hz':       sync_match.freq_hz if sync_match else None,
            'sync_freq_err_hz':   sync_f_err,
            'sync_snr_db':        sync_db,
            'detected_by_sqdet':  1 if sq_match else 0,
            'sqdet_time_s':       sq_match['time_s'] if sq_match else None,
            'sqdet_freq_hz':      sq_match['freq_hz'] if sq_match else None,
            'sqdet_snr_db':       sq_db,
        })

    # ── Aggregate summary ────────────────────────────────────────────────
    n_total = len(truth)
    n_sync = sum(r['detected_by_sync'] for r in rows)
    n_sq   = sum(r['detected_by_sqdet'] for r in rows)
    print(f"\n=== Sensitivity summary ===")
    print(f"  Wideband sync (15s): {n_sync}/{n_total} pings detected")
    print(f"  sq_det baseline:     {n_sq}/{n_total} pings detected")
    if n_sync > n_sq:
        print(f"  → sync wins by {n_sync - n_sq} pings")
    elif n_sq > n_sync:
        print(f"  → sq_det wins by {n_sq - n_sync} pings")
    else:
        print(f"  → tied")

    # Frequency-estimation accuracy on sync-detected pings (excluding misses).
    sync_freq_errs = [r['sync_freq_err_hz'] for r in rows
                      if r['sync_freq_err_hz'] is not None]
    if sync_freq_errs:
        errs = np.array(sync_freq_errs)
        print(f"\n=== Frequency-estimation accuracy (sync, n={len(errs)}) ===")
        print(f"  mean error:     {errs.mean():+.2f} Hz")
        print(f"  std:            {errs.std():.2f} Hz")
        print(f"  max abs error:  {np.max(np.abs(errs)):.2f} Hz")
        # Bin by SNR
        print(f"  by SNR bucket:")
        for snr_lo in range(-10, 5, 5):
            snr_hi = snr_lo + 5
            bucket_errs = [r['sync_freq_err_hz'] for r in rows
                           if r['sync_freq_err_hz'] is not None
                           and snr_lo <= r['snr_truth_db'] < snr_hi]
            if bucket_errs:
                be = np.array(bucket_errs)
                print(f"    SNR {snr_lo:+d}..{snr_hi:+d} dB: "
                      f"n={len(be)}  mean={be.mean():+5.1f}  "
                      f"std={be.std():5.1f}  max-abs={np.max(np.abs(be)):5.1f}  Hz")

    # CSV
    if args.out:
        with args.out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows: writer.writerow(r)
        print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == '__main__':
    main()
