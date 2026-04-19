#!/usr/bin/env python3
"""Gap 2 diagnostic: compare Python vs Fortran bpdecode128_90 on identical LLRs.

Generates synthetic MSK144 frames at a range of SNRs around the Python decoder
threshold (−1 to +1 dB), runs Python msk144decodeframe and Fortran bpdecode128_90
on the same frame, and compares where they diverge.

Three measurement tiers:
  1. Same LLR → compare bpdecode128_90 results (decoder-only comparison)
  2. Same frame → compare softbits before/after normalization
  3. Summary: measure 50% threshold for Python and Fortran on n_trials each

Usage:
    python gap2_diagnose.py [--snr-min -2] [--snr-max 2] [--snr-step 0.5]
                            [--trials 200] [--msg 'CQ K2GV FN30']
                            [--verbose]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_msk144 import generate_ping_msk144code
from map144_app.msk144_decode import (
    NSPM, FS, _PP, _CB, Mn, Nm, nrw, K_LDPC,
    _bpdecode128_90, _check_crc13, msk144decodeframe,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BP_TEST = Path(__file__).parent / 'fortran_bp_test' / 'bp_test'
_MF_TEST = Path(__file__).parent / 'fortran_bp_test' / 'mf_test'
assert _BP_TEST.exists(), f"Build fortran_bp_test/bp_test first: {_BP_TEST}"
assert _MF_TEST.exists(), f"Build fortran_bp_test/mf_test first: {_MF_TEST}"

# ── Signal generation ─────────────────────────────────────────────────────────
_NOISE_SIGMA = 1.0
_SAMPLE_RATE = 12000


def make_frame(msg: str, snr_db: float, seed: int) -> np.ndarray:
    """Generate one flat 72-ms MSK144 frame at 12 kHz complex baseband."""
    rng = np.random.default_rng(seed)
    ping_h, _v, _ = generate_ping_msk144code(
        msg,
        center_hz=0.0,
        delay_s=0.0,
        snr_db=snr_db,
        noise_sigma=_NOISE_SIGMA,
        width_ms=72,
        pol_scenario='pure-H',
        rng=rng,
        sample_rate=_SAMPLE_RATE,
        flat_frame=True,
    )
    # Trim / pad to exactly NSPM samples
    if len(ping_h) < NSPM:
        ping_h = np.pad(ping_h, (0, NSPM - len(ping_h)))
    frame = ping_h[:NSPM].astype(np.complex64)

    noise = (rng.standard_normal(NSPM) + 1j * rng.standard_normal(NSPM)).astype(
        np.complex64) * _NOISE_SIGMA
    return (frame + noise).astype(np.complex64)


# ── Python matched filter → LLR ───────────────────────────────────────────────

def frame_to_llr(c: np.ndarray) -> np.ndarray:
    """Replicate msk144decodeframe phase-correction + matched filter + LLR.

    Returns LLR as float32 shape (128,) — same values that _bpdecode128_90 sees.
    Returns None if sync hard-error gate fails.
    """
    c = c.astype(np.complex64)

    # Phase correction (msk144decodeframe.f90 lines 52-59)
    cca = np.sum(c[0:42] * np.conj(_CB))
    ccb = np.sum(c[336:378] * np.conj(_CB))
    phase0 = float(np.angle(cca + ccb))
    cfac = np.complex64(np.cos(phase0) + 1j * np.sin(phase0))
    c = c * np.conj(cfac)

    # Matched filter → 144 softbits
    softbits = np.empty(144, dtype=np.float32)
    softbits[0] = (np.sum(np.imag(c[0:6])   * _PP[6:12]) +
                   np.sum(np.imag(c[858:864]) * _PP[0:6]))
    softbits[1] = np.sum(np.real(c[0:12]) * _PP)
    for i in range(2, 73):
        softbits[2*i - 2] = np.sum(np.imag(c[(i-1)*12 - 6 : (i-1)*12 + 6]) * _PP)
        softbits[2*i - 1] = np.sum(np.real(c[(i-1)*12     : (i-1)*12 + 12]) * _PP)

    # Sync hard-error gate
    _S8 = np.array([-1, 1, 1, 1, -1, -1, 1, -1], dtype=np.float32)
    hardbits = (softbits >= 0).astype(np.int32)
    nb1 = (8 - int(np.sum((2*hardbits[0:8]  - 1) * _S8))) // 2
    nb2 = (8 - int(np.sum((2*hardbits[56:64] - 1) * _S8))) // 2
    if nb1 + nb2 > 4:
        return None

    # Normalize
    sav  = float(np.mean(softbits))
    s2av = float(np.mean(softbits ** 2))
    ssig = float(np.sqrt(max(s2av - sav**2, 1e-10)))
    softbits = softbits / ssig

    # LLR
    sigma = 0.60
    llr = np.empty(128, dtype=np.float32)
    llr[0:48]   = softbits[8:56]
    llr[48:128] = softbits[64:144]
    llr *= 2.0 / (sigma ** 2)
    return llr


# ── Fortran bpdecode128_90 wrapper ────────────────────────────────────────────

def fortran_bpdecode(llr_batch: list[np.ndarray]) -> list[bool]:
    """Run Fortran bpdecode128_90 on a batch of LLR vectors.

    Returns list of booleans: True if decoded successfully (nsuccess==1).
    """
    lines = [str(len(llr_batch))]
    for llr in llr_batch:
        lines.append(' '.join(f'{v:.6f}' for v in llr))
    stdin_text = '\n'.join(lines) + '\n'

    result = subprocess.run(
        [str(_BP_TEST)],
        input=stdin_text,
        capture_output=True, text=True, timeout=30,
    )
    successes = []
    out_lines = result.stdout.strip().splitlines()
    i = 0
    while i < len(out_lines):
        parts = out_lines[i].split()
        if len(parts) < 3:
            i += 1
            continue
        try:
            nsuccess = int(parts[0])
        except ValueError:
            i += 1
            continue
        successes.append(nsuccess == 1)
        if nsuccess == 1:
            i += 1  # skip the 77-bit line
        i += 1
    # Pad with False if output was short (error/timeout)
    while len(successes) < len(llr_batch):
        successes.append(False)
    return successes


# ── Fortran mf_test: full matched filter + decode ────────────────────────────

def fortran_mfdecode(frames: list[np.ndarray]) -> list[bool]:
    """Run Fortran phase-correction + matched filter + bpdecode128_90 on a batch.

    Each frame is (NSPM,) complex64. Written as 1728 floats (re im re im ...).
    Returns list of booleans: True if nsuccess==1.
    """
    lines = [str(len(frames))]
    for c in frames:
        vals = []
        for samp in c:
            vals.append(f'{float(np.real(samp)):.8f}')
            vals.append(f'{float(np.imag(samp)):.8f}')
        lines.append(' '.join(vals))
    stdin_text = '\n'.join(lines) + '\n'

    result = subprocess.run(
        [str(_MF_TEST)],
        input=stdin_text,
        capture_output=True, text=True, timeout=60,
    )
    successes = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 1:
            try:
                successes.append(int(parts[0]) == 1)
            except ValueError:
                pass
    while len(successes) < len(frames):
        successes.append(False)
    return successes


# ── Per-frame comparison ──────────────────────────────────────────────────────

def compare_one_frame(c: np.ndarray) -> dict:
    """Compare Python and Fortran decoders on one complex frame.

    Returns dict with:
      llr_ok       : bool — LLR could be computed (sync gate passed)
      py_decode    : bool — Python msk144decodeframe succeeded
      ft_decode    : bool — Fortran bpdecode128_90 on same LLR succeeded
      agree        : bool — both give same result
    """
    # Python full decoder
    msg, nhe = msk144decodeframe(c)
    py_decode = (msg is not None)

    # Extract LLR from same frame
    llr = frame_to_llr(c)
    if llr is None:
        return {'llr_ok': False, 'py_decode': py_decode, 'ft_decode': False, 'agree': not py_decode}

    # Python bpdecode128_90 only (should match msk144decodeframe result)
    py_bits, py_nhe = _bpdecode128_90(llr)
    py_bp = (py_bits is not None)

    # Fortran bpdecode128_90
    ft_results = fortran_bpdecode([llr])
    ft_bp = ft_results[0]

    return {
        'llr_ok':    True,
        'py_decode': py_decode,
        'py_bp':     py_bp,
        'ft_bp':     ft_bp,
        'agree':     (py_bp == ft_bp),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(msg, snr_range, n_trials, verbose=False):
    """Sweep SNR levels; at each level compare Python and Fortran."""
    print(f"\nGap-2 decoder comparison: Python vs Fortran bpdecode128_90")
    print(f"  message : {msg!r}")
    print(f"  frames  : {n_trials} per SNR level")
    print()

    hdr = (f"  {'SNR':>6}  {'trials':>6}  {'py_mf%':>7}  {'ft_mf%':>7}"
           f"  {'ft_bp%':>7}  {'mf_agree%':>10}  {'ft>py':>6}")
    sep = '  ' + '─' * (len(hdr) - 2)
    print(hdr)
    print(sep)

    results = []
    for snr_db in snr_range:
        # Generate all frames
        frames = [make_frame(msg, snr_db, seed) for seed in range(n_trials)]

        # Python full msk144decodeframe (matched filter + bpdecode128_90)
        py_decodes = [msk144decodeframe(f)[0] is not None for f in frames]

        # Fortran full mf_test (matched filter + bpdecode128_90)
        ft_mf = fortran_mfdecode(frames)

        # Fortran bpdecode128_90 on Python-computed LLRs (previous test)
        llrs = [frame_to_llr(f) for f in frames]
        valid_llrs = [(i, llr) for i, llr in enumerate(llrs) if llr is not None]
        ft_bp_decoded = [False] * n_trials
        if valid_llrs:
            indices, llr_batch = zip(*valid_llrs)
            ft_bp_results = fortran_bpdecode(list(llr_batch))
            for idx, ft_ok in zip(indices, ft_bp_results):
                ft_bp_decoded[idx] = ft_ok

        n_py    = sum(py_decodes)
        n_ft_mf = sum(ft_mf)
        n_ft_bp = sum(ft_bp_decoded)
        n_mf_agree = sum(p == f for p, f in zip(py_decodes, ft_mf))
        n_ft_mf_only = sum(f and not p for p, f in zip(py_decodes, ft_mf))

        print(f"  {snr_db:+6.1f}  {n_trials:>6}  "
              f"{100*n_py/n_trials:>6.1f}%  {100*n_ft_mf/n_trials:>6.1f}%  "
              f"{100*n_ft_bp/n_trials:>6.1f}%  {100*n_mf_agree/n_trials:>9.1f}%"
              f"  {n_ft_mf_only:>6}")

        if verbose and n_ft_mf_only > 0:
            for i, (p, f) in enumerate(zip(py_decodes, ft_mf)):
                if f and not p:
                    print(f"    [FT_MF_ONLY] seed={i}")

        results.append({
            'snr_db': snr_db, 'n_trials': n_trials,
            'n_py': n_py, 'n_ft_mf': n_ft_mf, 'n_ft_bp': n_ft_bp,
            'n_mf_agree': n_mf_agree, 'n_ft_mf_only': n_ft_mf_only,
        })

    def threshold50(key):
        for r in results:
            if r[key] / r['n_trials'] >= 0.5:
                return r['snr_db']
        return float('inf')

    py_thr    = threshold50('n_py')
    ft_mf_thr = threshold50('n_ft_mf')
    ft_bp_thr = threshold50('n_ft_bp')
    print()
    print(f"  Python msk144decodeframe 50% threshold : {py_thr:+.1f} dB")
    print(f"  Fortran mf_test (full MF+BP) threshold : {ft_mf_thr:+.1f} dB")
    print(f"  Fortran bpdecode128_90 only threshold  : {ft_bp_thr:+.1f} dB  (same LLR as Python)")
    if abs(py_thr) < 1e3 and abs(ft_mf_thr) < 1e3:
        gap = py_thr - ft_mf_thr
        print(f"  Fortran MF gap vs Python               : {gap:+.1f} dB  "
              f"({'Fortran better' if gap > 0 else 'Python better or equal'})")
    print()

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--msg',      default='CQ K2GV FN30')
    ap.add_argument('--snr-min',  type=float, default=-2.0)
    ap.add_argument('--snr-max',  type=float, default=2.0)
    ap.add_argument('--snr-step', type=float, default=0.5)
    ap.add_argument('--trials',   type=int,   default=200)
    ap.add_argument('--verbose',  action='store_true')
    args = ap.parse_args()

    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step * 0.5,
                          args.snr_step)
    run_sweep(args.msg, snr_range, args.trials, verbose=args.verbose)


if __name__ == '__main__':
    main()
