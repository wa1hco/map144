#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Measure jt9 decode probability (Pc) vs SNR for MSK144.

Methodology:
    1. Generate synthetic MSK144 frame (known message) at SNR X with fresh noise.
    2. Convert to 12 kHz real audio WAV with signal at 1500 Hz.
    3. Invoke jt9 --msk144 at depth d ∈ {1, 2}.  Skip d=3 initially per user
       request (≈ 3× compute cost).
    4. Parse jt9 output; compare decoded message to truth.
    5. Count Pc = (correct decodes) / trials per (SNR, depth) bucket.

Output: experiments/detection_compare/decoder_curves.csv
"""
from __future__ import annotations

import sys
import csv
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.signal import decimate

from experiments.gpu_tfmf import synthetic
from map144_app.detection import find_jt9

# Configuration
SNR_GRID_DB    = list(range(-10, 4))    # -10 to +3 dB (decoder-relevant range)
N_TRIALS       = 30                     # trials per (SNR, depth) bucket
DEPTHS         = [1, 2]                 # jt9 -d levels; add 3 later
TRUTH_MESSAGE  = "CQ K1JT FN20"
TRUTH_FREQ_HZ  = 1500.0
TRUTH_TIME_S   = 5.0
NOISE_SIGMA    = 1.0
DURATION_S     = 15.0
SAMPLE_RATE    = 48_000
DECODE_RATE    = 12_000
N_WORKERS      = 6                      # physical-core count on this box

JT9_PATH = find_jt9()
if JT9_PATH is None:
    print("ERROR: jt9 binary not found.  See map144_app.detection.find_jt9 for "
          "search locations.", file=sys.stderr)
    sys.exit(1)

OUT_DIR = REPO / 'experiments' / 'detection_compare'
OUT_CSV = OUT_DIR / 'decoder_curves.csv'


# ── Audio prep + jt9 invocation ─────────────────────────────────────────────


def _iq_to_audio_wav(iq: np.ndarray, path: str) -> None:
    """Convert 48 kHz complex IQ → 12 kHz real-audio WAV with MSK144 at 1500 Hz.

    Steps: take real part (signal at +1500 Hz becomes a real cosine at 1500 Hz),
    decimate by 4, normalise, write 16-bit mono WAV.  Both signal and noise
    scale identically through real() so SNR is preserved.
    """
    audio_48k = iq.real.astype(np.float64)
    audio_12k = decimate(audio_48k, 4, ftype='fir', zero_phase=True)
    peak = float(np.max(np.abs(audio_12k)))
    if peak < 1e-30:
        peak = 1.0
    audio_int16 = (audio_12k / peak * 30000).astype(np.int16)  # headroom 0.92
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(DECODE_RATE)
        wf.writeframes(audio_int16.tobytes())


_DECODE_LINE_RE = re.compile(
    r'^\s*\d+\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s+\d+\s+&\s+(.+?)\s*$'
)


def _parse_jt9_output(stdout: str) -> str:
    """Extract the decoded message text from a jt9 stdout block.

    jt9 line format for MSK144: ``HHMMSS  SNR  DT  FREQ  &  MESSAGE``
    Returns empty string if no decode line found.
    """
    for line in stdout.splitlines():
        s = line.strip()
        if not s or s.startswith('<') or s.startswith('EOF'):
            continue
        m = _DECODE_LINE_RE.match(s)
        if m:
            return m.group(1).strip()
        # Fallback: line without the structured prefix but post-'&'.
        if '&' in s:
            return s.split('&', 1)[1].strip()
    return ""


def _is_correct(decoded: str, truth: str = TRUTH_MESSAGE) -> bool:
    """Normalize and compare.  Treats whitespace and case as insignificant."""
    norm = lambda x: re.sub(r'\s+', ' ', x.strip().upper())
    return norm(decoded) == norm(truth)


def _run_jt9(wav_path: str, depth: int, timeout_s: float = 25.0) -> tuple[str, str]:
    """Invoke jt9 on the WAV, return (decoded_message, raw_stdout).

    Returns ("", stdout) if no decode (jt9 prints nothing, or returns
    structured-empty output).
    """
    cmd = [
        JT9_PATH, "--msk144",
        "-L", "1400", "-H", "1600", "-F", "100",
        "-d", str(depth),
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ("", "<TIMEOUT>")
    return (_parse_jt9_output(result.stdout), result.stdout)


# ── Worker ──────────────────────────────────────────────────────────────────


def _worker_trial(args: tuple) -> list[dict]:
    """Worker entry: generate one ping, write WAV once, run jt9 at each depth.

    args = (snr_db, trial, seed, depths_to_run)
    """
    snr_db, trial, seed, depths = args
    s = synthetic.single_ping(
        snr_db=float(snr_db),
        freq_hz=TRUTH_FREQ_HZ,
        time_s=TRUTH_TIME_S,
        duration_s=DURATION_S,
        noise_sigma=NOISE_SIGMA,
        seed=seed,
    )
    # One WAV per trial — reused across depths.
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name
    out = []
    try:
        _iq_to_audio_wav(s.iq, wav_path)
        for d in depths:
            decoded, stdout_blob = _run_jt9(wav_path, d)
            correct = _is_correct(decoded)
            out.append({
                'depth':       d,
                'snr_db':      snr_db,
                'trial':       trial,
                'truth_msg':   TRUTH_MESSAGE,
                'decoded_msg': decoded,
                'correct':     int(correct),
            })
    finally:
        try:
            Path(wav_path).unlink()
        except FileNotFoundError:
            pass
    return out


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"jt9: {JT9_PATH}")
    print(f"SNR grid: {SNR_GRID_DB[0]} to {SNR_GRID_DB[-1]} dB, {len(SNR_GRID_DB)} points")
    print(f"Trials:   {N_TRIALS} per (SNR, depth)")
    print(f"Depths:   {DEPTHS}")
    print(f"Workers:  {N_WORKERS}")
    n_total = len(SNR_GRID_DB) * N_TRIALS
    print(f"Total trials: {n_total} (× {len(DEPTHS)} depths = "
          f"{n_total * len(DEPTHS)} jt9 calls)")

    work = []
    for snr_db in SNR_GRID_DB:
        for trial in range(N_TRIALS):
            seed = 2_000_000 + (snr_db + 20) * 1000 + trial
            work.append((snr_db, trial, seed, DEPTHS))

    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for trial_rows in ex.map(_worker_trial, work, chunksize=2):
            rows.extend(trial_rows)
            done += 1
            if done % 20 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{n_total}] {elapsed:5.1f}s "
                      f"({rate:.1f}/s, ~{eta:.0f}s remaining)",
                      flush=True)

    print(f"\nTotal measurement: {time.time()-t0:.1f}s")

    # Per-(SNR, depth) summary
    print("\nDecode probability Pc per SNR × depth:")
    for snr_db in SNR_GRID_DB:
        line = f"  SNR {snr_db:+3d} dB:"
        for d in DEPTHS:
            algo_rows = [r for r in rows if r['snr_db'] == snr_db and r['depth'] == d]
            n_correct = sum(r['correct'] for r in algo_rows)
            line += f"  -d{d}={n_correct}/{N_TRIALS}"
        print(line)

    with OUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['depth', 'snr_db', 'trial', 'truth_msg',
                            'decoded_msg', 'correct']
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()
