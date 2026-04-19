#!/usr/bin/env python3
"""Gap 3 diagnostic: profile SPD detection/sync overhead vs direct decode.

The SPD complex-IQ path threshold is +1.4 dB; direct msk144decodeframe on an
aligned frame is 0 dB.  This 1.4 dB overhead comes from somewhere in the SPD
pipeline.  This script locates which stage is responsible.

For each trial the signal is placed at a KNOWN position in the buffer so we can
ask three oracle questions:

  Q1 — Detection failure: was the correct istp candidate found (detmet[istp] in
       top-5 with detmet > 3.0)?  If not, no further processing is possible.

  Q2 — Sync failure (given detection): if we force ib to the oracle position,
       does the best xmax_norm across all navmasks reach the 1.3 threshold?
       This isolates the detection-window-placement uncertainty from the sync
       threshold uncertainty.

  Q3 — Decode failure (given oracle sync pass): does msk144decodeframe succeed
       on the oracle-extracted and timing-corrected frame?  This measures how
       much loss comes from timing / frequency residuals inside the SPD vs the
       decoder itself.

Failure taxonomy per trial:
  'success'         — normal SPD decoded correctly
  'det_miss'        — correct candidate not found in top-5
  'sync_miss'       — candidate found but xmax_norm < 1.3 for all navmasks/freq
  'decode_miss'     — sync passed but msk144decodeframe failed
  'oracle_det_miss' — even oracle ib fails the sync threshold
  'oracle_sync_miss'— oracle ib passes sync but decode fails despite oracle timing

Usage:
    python gap3_diagnose.py [--snr-min 0] [--snr-max 3] [--snr-step 0.5]
                            [--trials 400] [--msg 'CQ K2GV FN30']
                            [--verbose]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import decimate, firwin, filtfilt

sys.path.insert(0, str(Path(__file__).parent))
from generate_msk144 import generate_ping_msk144code
from map144_app.msk144_decode import msk144decodeframe as _msk144decodeframe
from map144_app.msk144_spd import (
    NSPM, FS, SYNC_THRESHOLD, _NAV_PATTERNS,
    _freq_search_avg, msk144_spd_decode,
)

# ── Signal layout (mirrors decode_sensitivity.py run_trial_spd) ──────────────

_IQ_RATE    = 48_000
_AUDIO_RATE = 12_000
_DECIMATE   = _IQ_RATE // _AUDIO_RATE
_FC_HZ      = 1500.0
_PRE_S      = 0.500   # 500 ms noise pre-burst (matches run_trial_spd delay_s)
_POST_S     = 1.200
_PAD_S      = 0.100
_NOISE_SIGMA = 0.01   # AWGN amplitude (same as decode_sensitivity.py)

_SQRT2 = np.float32(np.sqrt(2.0))


def _apply_audio_bpf(iq_bb: np.ndarray, lo_hz: float, hi_hz: float,
                     fc_hz: float = 1500.0, fs: float = 12000.0,
                     ntaps: int = 129) -> np.ndarray:
    """Model a receiver SSB audio bandpass filter on complex baseband IQ.

    The audio BPF (lo_hz to hi_hz in the original audio domain, centred on
    fc_hz) is equivalent to a complex bandpass in the baseband domain.  For a
    filter symmetric around fc_hz the baseband passband is centred at 0 Hz,
    reducing to a simple lowpass.  For asymmetric filters a frequency shift is
    applied to centre the passband before filtering, then undone.

    The filter is applied independently to the I and Q channels so that I/Q
    noise independence is preserved.  (An earlier implementation used a
    mix-up → real BPF → Hilbert → mix-down chain which works for real-audio
    inputs but converts independent I/Q noise into a correlated analytic pair,
    causing the BP decoder to produce wrong LLRs.)

    Parameters
    ----------
    iq_bb  : complex64 array at fs, centred at 0 Hz (baseband)
    lo_hz  : lower −6 dB edge of the audio bandpass (e.g. 300 Hz)
    hi_hz  : upper −6 dB edge of the audio bandpass (e.g. 2700 Hz)
    fc_hz  : audio carrier frequency corresponding to baseband 0 Hz
    fs     : sample rate of iq_bb (default 12 000 Hz)
    ntaps  : FIR filter length (odd, default 129)
    """
    # Map audio passband edges to complex baseband frequencies
    lo_bb = lo_hz - fc_hz          # lower baseband edge (e.g. −1200 Hz)
    hi_bb = hi_hz - fc_hz          # upper baseband edge (e.g. +1200 Hz)
    center_bb = (lo_bb + hi_bb) / 2.0   # passband centre in baseband (0 for symmetric)
    half_bw   = (hi_bb - lo_bb)   / 2.0 # half-bandwidth (e.g. 1200 Hz)

    n   = len(iq_bb)

    # If the passband is not centred at 0 Hz, shift so it is
    if abs(center_bb) > 0.5:
        t         = np.arange(n, dtype=np.float64) / fs
        iq_work   = (iq_bb * np.exp(-2j * np.pi * center_bb * t)).astype(np.complex64)
        need_shift = True
    else:
        iq_work    = iq_bb
        need_shift = False
        t          = None

    # Zero-phase lowpass applied independently to I and Q.
    # filtfilt eliminates both causal group delay and onset transient —
    # a 513-tap causal FIR at 48 kHz would corrupt the first 128/864 samples
    # of the 72 ms frame after decimation.
    nyq    = fs / 2.0
    h      = firwin(ntaps, half_bw / nyq)
    i_filt = filtfilt(h, 1.0, np.real(iq_work).astype(np.float64))
    q_filt = filtfilt(h, 1.0, np.imag(iq_work).astype(np.float64))
    iq_filt = (i_filt + 1j * q_filt).astype(np.complex64)

    # Undo the frequency shift if one was applied
    if need_shift:
        if t is None:
            t = np.arange(n, dtype=np.float64) / fs
        iq_filt = (iq_filt * np.exp(2j * np.pi * center_bb * t)).astype(np.complex64)

    return iq_filt


def _make_iq_12k(msg: str, snr_db: float, seed: int,
                  use_direct_12k: bool = False,
                  audio_bw_hz: float | None = None) -> tuple[np.ndarray, int]:
    """Generate complex baseband IQ at 12 kHz with signal at a known sample.

    Parameters
    ----------
    use_direct_12k : bool
        When False (default): generate at 48 kHz, mix to baseband, and
        decimate to 12 kHz with a causal IIR filter (zero_phase=False).
        This matches the production decode_sensitivity.py path but
        introduces ~2.4-sample group delay (40% of the 6-sample chip
        period), which reduces matched-filter efficiency.

        When True: generate the signal directly at 12 kHz with no
        decimation filter.  This removes the IIR distortion entirely and
        isolates the SPD algorithm's own overhead (navmask uncertainty,
        freq-search residual, sync threshold).

    audio_bw_hz : float or None
        When set (e.g. 2700.0), apply a receiver audio bandpass filter
        (300 Hz – audio_bw_hz) to the 12 kHz complex baseband AFTER
        decimation (or directly for the direct-12k path).  This models
        the SSB receiver IF/audio filter that limits the noise bandwidth
        to ~2.4–3.2 kHz rather than the 4.8 kHz of the IIR decimation
        output or the 6 kHz of the direct-12k path.

        The SNR label is preserved: the signal (±500 Hz deviation around
        0 Hz carrier) lies well within the 300–audio_bw_hz passband.
        Only the noise outside the audio passband is removed, reducing
        the noise floor in the squared-spectrum detector.

    Returns
    -------
    iq_padded        : complex64 ndarray — the full buffer passed to SPD
    signal_start_12k : int — sample index where the flat 72-ms frame begins
    """
    rng = np.random.default_rng(seed)
    pad_n = int(_PAD_S * _AUDIO_RATE)

    if use_direct_12k:
        # Generate directly at 12 kHz, signal already at baseband (center_hz=0)
        total_12k = int((_PRE_S + _POST_S) * _AUDIO_RATE)
        ping_h, _ping_v, _meta = generate_ping_msk144code(
            msg,
            center_hz=0.0,
            delay_s=_PRE_S,
            snr_db=snr_db,
            noise_sigma=_NOISE_SIGMA,
            width_ms=72,
            pol_scenario='pure-H',
            rng=rng,
            sample_rate=_AUDIO_RATE,
            flat_frame=True,
        )
        if len(ping_h) < total_12k:
            ping_h = np.pad(ping_h, (0, total_12k - len(ping_h)))
        else:
            ping_h = ping_h[:total_12k]

        noise_iq = (rng.standard_normal(total_12k) +
                    1j * rng.standard_normal(total_12k)).astype(np.complex64) * _NOISE_SIGMA
        iq_12k = (ping_h + noise_iq).astype(np.complex64)
    else:
        # Generate at 48 kHz, mix to baseband, decimate (production path)
        total_iq = int((_PRE_S + _POST_S) * _IQ_RATE)
        ping_h, _ping_v, _meta = generate_ping_msk144code(
            msg,
            center_hz=_FC_HZ,
            delay_s=_PRE_S,
            snr_db=snr_db,
            noise_sigma=_NOISE_SIGMA,
            width_ms=72,
            pol_scenario='pure-H',
            rng=rng,
            sample_rate=_IQ_RATE,
            flat_frame=True,
        )
        if len(ping_h) < total_iq:
            ping_h = np.pad(ping_h, (0, total_iq - len(ping_h)))
        else:
            ping_h = ping_h[:total_iq]

        noise_iq = (rng.standard_normal(total_iq) +
                    1j * rng.standard_normal(total_iq)).astype(np.complex64) * _NOISE_SIGMA
        iq_noisy = (ping_h + noise_iq).astype(np.complex64)

        # Mix to baseband and decimate to 12 kHz (causal IIR, zero_phase=False)
        t_48k = np.arange(total_iq, dtype=np.float64) / _IQ_RATE
        iq_bb = (iq_noisy * np.exp(-2j * np.pi * _FC_HZ * t_48k)).astype(np.complex64)
        i_dec = decimate(np.real(iq_bb).astype(np.float64), _DECIMATE, zero_phase=False)
        q_dec = decimate(np.imag(iq_bb).astype(np.float64), _DECIMATE, zero_phase=False)
        iq_12k = (i_dec + 1j * q_dec).astype(np.complex64)

    # Optional receiver audio bandpass filter at 12 kHz (models SSB IF/audio filter).
    # Applied post-decimation using a complex lowpass on I and Q independently
    # (see _apply_audio_bpf for the reasoning behind this approach).
    if audio_bw_hz is not None:
        iq_12k = _apply_audio_bpf(iq_12k, lo_hz=300.0, hi_hz=audio_bw_hz,
                                   fs=float(_AUDIO_RATE))

    # Add zero pads (matches run_trial_spd)
    iq_padded = np.concatenate([
        np.zeros(pad_n, dtype=np.complex64),
        iq_12k,
        np.zeros(pad_n, dtype=np.complex64),
    ])

    # Signal frame starts at: pad_n + PRE_S * AUDIO_RATE
    signal_start_12k = pad_n + int(_PRE_S * _AUDIO_RATE)

    return iq_padded, signal_start_12k


def _spd_detect_metric(cbase: np.ndarray, ntol: float = 200.0) -> np.ndarray:
    """Replicate the SPD squared-spectrum detection step; return normalised detmet."""
    n = len(cbase)
    nstep  = (n - NSPM) // 216
    detmet = np.zeros(nstep, dtype=np.float32)

    nfft     = NSPM
    df       = FS / nfft
    idx_pos  = int(round(1000.0 / df))
    idx_neg  = NSPM - idx_pos
    half_win = max(2, int(round(ntol * 2 / df)))

    rcw = ((1.0 - np.cos(np.arange(12) * np.pi / 12)) / 2).astype(np.float32)

    for istp in range(nstep):
        ns = istp * 216
        ne = ns + NSPM
        if ne > n:
            break
        seg_sq = (cbase[ns:ne] ** 2).copy()
        seg_sq[:12]      *= rcw
        seg_sq[NSPM-12:] *= rcw[::-1]
        spec     = np.fft.fft(seg_sq, n=nfft)
        tonespec = np.abs(spec) ** 2
        hi_lo = max(0, idx_pos - half_win)
        hi_hi = min(nfft // 2, idx_pos + half_win)
        lo_lo = max(nfft // 2, idx_neg - half_win)
        lo_hi = min(nfft, idx_neg + half_win)
        ah = float(np.max(tonespec[hi_lo:hi_hi])) if hi_hi > hi_lo else 0.0
        al = float(np.max(tonespec[lo_lo:lo_hi])) if lo_hi > lo_lo else 0.0
        detmet[istp] = max(ah, al)

    q_idx = max(1, nstep // 4)
    sorted_met = np.sort(detmet[:nstep])
    xmed = float(sorted_met[q_idx]) if q_idx > 0 else 1.0
    if xmed > 0:
        detmet[:nstep] /= xmed

    return detmet


def _oracle_ib(signal_start: int) -> int:
    """Compute the oracle ib (window start) for a signal starting at signal_start.

    The detection step nearest the signal: istp_oracle = signal_start // 216.
    ib = max(0, nstart_cand - NSPM - 216)  (mirrors msk144_spd_decode).
    """
    istp_oracle = signal_start // 216
    nstart_cand = istp_oracle * 216
    return max(0, nstart_cand - NSPM - 216)


def _run_oracle_sync_decode(
    cbase: np.ndarray,
    ib: int,
    n: int,
    ntol_inner: float = 8.0,
    delf_inner: float = 2.0,
    threshold: float = SYNC_THRESHOLD,
) -> tuple[bool, bool, float]:
    """Run oracle freq-search + sync + decode for a forced ib.

    Returns
    -------
    oracle_sync_pass  : bool — any navmask gave xmax_norm >= threshold
    oracle_decode_pass: bool — decode succeeded for the best navmask+timing
    best_xmax         : float — highest xmax_norm seen across all navmasks
    """
    ie_search = ib + 3 * NSPM
    if ie_search > n:
        ie_search = n
        ib = max(0, ie_search - 3 * NSPM)

    ie_ext = min(ib + 4 * NSPM, n)
    cbase_ext = cbase[ib:ie_ext]
    if len(cbase_ext) < 3 * NSPM:
        return False, False, 0.0

    best_xmax    = 0.0
    sync_pass    = False
    decode_pass  = False

    for navmask in _NAV_PATTERNS:
        c_avg, xmax_norm, ferr_best, xcc, ish_best, best_mixed = _freq_search_avg(
            cbase_ext, navmask, ntol=ntol_inner, delf=delf_inner,
        )
        if c_avg is None:
            continue
        if xmax_norm > best_xmax:
            best_xmax = xmax_norm

        if xmax_norm < threshold:
            continue
        sync_pass = True

        # Try the top-2 timing peaks with ±1 nudge
        xcc_copy  = xcc.copy()
        peak_locs = []
        for _ in range(2):
            pk = int(np.argmax(xcc_copy))
            peak_locs.append(pk)
            lo = max(0, pk - 7)
            hi = min(NSPM, pk + 8)
            xcc_copy[lo:hi] = 0.0

        mixed_len = len(best_mixed)
        for ic0 in peak_locs:
            for delta in (0, -1, 1):
                ic = (ic0 + delta) % NSPM
                c_shifted = np.zeros(NSPM, dtype=np.complex64)
                for fi, m in enumerate(navmask):
                    if m:
                        start = fi * NSPM + ic
                        end   = start + NSPM
                        if end <= mixed_len:
                            c_shifted += best_mixed[start:end]
                        else:
                            avail = max(0, mixed_len - start)
                            if avail > 0:
                                c_shifted[:avail] += best_mixed[start:start + avail]

                msg, _nhe = _msk144decodeframe(c_shifted)
                if msg is not None:
                    decode_pass = True

    return sync_pass, decode_pass, best_xmax


def run_trial_diag(msg: str, snr_db: float, seed: int,
                   use_direct_12k: bool = False,
                   audio_bw_hz: float | None = None) -> dict:
    """Run one SPD diagnostic trial.

    Returns a dict with:
      snr_db       : float
      spd_success  : bool — normal SPD decoded OK
      det_rank     : int or None — rank of correct candidate in detmet (0-based); None if not found
      det_hit      : bool — correct istp was in top-5 candidates (detmet > 3.0)
      det_metric   : float — detmet value at the oracle istp (after normalisation)
      oracle_sync_pass  : bool — oracle ib achieved xmax_norm ≥ 1.3
      oracle_decode_pass: bool — oracle ib + timing achieved decode
      oracle_xmax  : float — best xmax_norm with oracle ib
      failure_mode : str — one of: success | det_miss | sync_miss | decode_miss
                           | oracle_sync_miss | oracle_decode_miss
    """
    iq_padded, signal_start = _make_iq_12k(msg, snr_db, seed,
                                            use_direct_12k=use_direct_12k,
                                            audio_bw_hz=audio_bw_hz)

    # Normal SPD run
    try:
        decoded_msg, _snr, _navg, _fest, xmax_norm = msk144_spd_decode(
            complex_baseband=iq_padded,
        )
        spd_success = decoded_msg is not None
    except Exception:
        spd_success = False
        xmax_norm   = 0.0

    # Normalise cbase the same way SPD does (for oracle detection check)
    cb128 = iq_padded.astype(np.complex128)
    rms   = float(np.sqrt(np.mean(np.abs(cb128) ** 2)))
    if rms < 1e-12:
        return {
            'snr_db': snr_db, 'spd_success': False, 'det_rank': None,
            'det_hit': False, 'det_metric': 0.0,
            'oracle_sync_pass': False, 'oracle_decode_pass': False,
            'oracle_xmax': 0.0, 'failure_mode': 'rms_zero',
        }
    cbase = (iq_padded * (_SQRT2 / rms)).astype(np.complex64)
    n     = len(cbase)

    # Detection metric at oracle istp
    detmet = _spd_detect_metric(cbase, ntol=200.0)
    nstep  = len(detmet)
    istp_oracle = signal_start // 216
    det_metric  = float(detmet[istp_oracle]) if istp_oracle < nstep else 0.0

    # Determine rank of oracle istp among all candidates
    if istp_oracle < nstep:
        rank = int(np.sum(detmet[:nstep] > det_metric))  # how many steps are strictly better
    else:
        rank = None
    det_hit = (det_metric >= 3.0 and rank is not None and rank < 5)

    # Oracle sync + decode (forced correct ib)
    ib_oracle = _oracle_ib(signal_start)
    oracle_sync_pass, oracle_decode_pass, oracle_xmax = _run_oracle_sync_decode(
        cbase, ib_oracle, n,
    )

    # Classify failure mode
    if spd_success:
        mode = 'success'
    elif not det_hit:
        mode = 'det_miss'
    elif not oracle_sync_pass:
        mode = 'sync_miss'
    elif not oracle_decode_pass:
        mode = 'oracle_decode_miss'
    else:
        # SPD failed but oracle succeeded — detection chose wrong candidate
        # or timing/nav pattern selection issue
        mode = 'decode_miss'

    return {
        'snr_db':             snr_db,
        'spd_success':        spd_success,
        'det_rank':           rank,
        'det_hit':            det_hit,
        'det_metric':         det_metric,
        'oracle_sync_pass':   oracle_sync_pass,
        'oracle_decode_pass': oracle_decode_pass,
        'oracle_xmax':        oracle_xmax,
        'failure_mode':       mode,
    }


def run_sweep(msg: str, snr_range, n_trials: int, verbose: bool = False,
              use_direct_12k: bool = False,
              audio_bw_hz: float | None = None) -> list[dict]:
    """Sweep SNR levels; at each level profile failure modes."""
    from collections import Counter

    sig_label = 'direct 12 kHz (no IIR distortion)' if use_direct_12k else 'IIR decimated 48→12 kHz (causal)'
    if audio_bw_hz is not None:
        sig_label += f' + audio BPF 300–{audio_bw_hz:.0f} Hz'
    print(f"\nGap-3 SPD overhead diagnostic: complex-IQ path")
    print(f"  message : {msg!r}")
    print(f"  trials  : {n_trials} per SNR level")
    print(f"  signal  : {sig_label}")
    print(f"  flat frame, pure-H, single 72-ms frame\n")

    hdr = (f"  {'SNR':>6}  {'spd%':>6}  {'det_hit%':>9}  {'det_met':>8}"
           f"  {'oracle_sync%':>12}  {'oracle_dec%':>11}  "
           f"  {'det_miss':>8}  {'sync_miss':>9}  {'dec_miss':>8}")
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))

    all_results = []

    for snr_db in snr_range:
        results = [run_trial_diag(msg, snr_db, seed,
                                  use_direct_12k=use_direct_12k,
                                  audio_bw_hz=audio_bw_hz)
                   for seed in range(n_trials)]
        all_results.extend(results)

        n_spd    = sum(r['spd_success']        for r in results)
        n_dethit = sum(r['det_hit']            for r in results)
        n_osync  = sum(r['oracle_sync_pass']   for r in results)
        n_odec   = sum(r['oracle_decode_pass'] for r in results)

        det_mets  = [r['det_metric'] for r in results]
        mean_dmet = float(np.mean(det_mets))

        modes     = Counter(r['failure_mode'] for r in results)
        n_fail    = n_trials - n_spd
        n_dm      = modes.get('det_miss', 0)
        n_sm      = modes.get('sync_miss', 0)
        n_decodem = modes.get('decode_miss', 0) + modes.get('oracle_decode_miss', 0)

        print(f"  {snr_db:+6.1f}  {100*n_spd/n_trials:>5.1f}%"
              f"  {100*n_dethit/n_trials:>8.1f}%"
              f"  {mean_dmet:>8.2f}"
              f"  {100*n_osync/n_trials:>11.1f}%"
              f"  {100*n_odec/n_trials:>10.1f}%"
              f"  {n_dm:>8d}  {n_sm:>9d}  {n_decodem:>8d}")

        if verbose and n_fail > 0:
            for r in results:
                if not r['spd_success']:
                    print(f"    [FAIL] seed=? mode={r['failure_mode']}"
                          f" det_metric={r['det_metric']:.2f}"
                          f" oracle_xmax={r['oracle_xmax']:.3f}"
                          f" det_hit={r['det_hit']}"
                          f" det_rank={r['det_rank']}")

    # Summary
    print()
    print("  Failure mode breakdown (all SNR levels combined):")
    all_modes = Counter(r['failure_mode'] for r in all_results)
    total_fail = sum(1 for r in all_results if not r['spd_success'])
    for mode, count in sorted(all_modes.items()):
        frac = 100 * count / len(all_results)
        fail_frac = 100 * count / max(total_fail, 1) if mode != 'success' else 0.0
        if mode == 'success':
            print(f"    {mode:<22} {count:5d} / {len(all_results):5d}  ({frac:.1f}%)")
        else:
            print(f"    {mode:<22} {count:5d} / {total_fail:5d} failures  ({fail_frac:.1f}% of failures)")
    print()

    return all_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--msg',      default='CQ K2GV FN30')
    ap.add_argument('--snr-min',  type=float, default=0.0)
    ap.add_argument('--snr-max',  type=float, default=3.0)
    ap.add_argument('--snr-step', type=float, default=0.5)
    ap.add_argument('--trials',   type=int,   default=400)
    ap.add_argument('--verbose',  action='store_true')
    ap.add_argument('--direct-12k', dest='direct_12k', action='store_true',
                    help='Generate signal directly at 12 kHz (no IIR decimation) to '
                         'isolate SPD algorithm overhead from IIR filter distortion.')
    ap.add_argument('--audio-bw', dest='audio_bw', type=float, default=None,
                    metavar='HZ',
                    help='Apply a receiver audio bandpass filter (300 Hz – HZ) after '
                         'decimation.  Models an SSB receiver IF/audio filter (typical '
                         'values: 2700 for narrow SSB, 3000 or 3500 for wide SSB).  '
                         'Can be combined with --direct-12k.')
    args = ap.parse_args()

    snr_range = list(np.arange(args.snr_min,
                               args.snr_max + args.snr_step * 0.5,
                               args.snr_step))
    run_sweep(args.msg, snr_range, args.trials, verbose=args.verbose,
              use_direct_12k=args.direct_12k, audio_bw_hz=args.audio_bw)


if __name__ == '__main__':
    main()
