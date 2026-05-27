# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Controlled false-alarm rate measurement: pure noise → jt9 --msk144.

Generates N pure-AWGN 15-second WAV files at 12 kHz (the format msk144sim
and jt9 expect), runs `jt9 -k` on each, and counts:

  - ``n_detected``: periods where jt9 printed at least one decode line
    (SPD triggered AND bpdecode succeeded = full-chain false alarm)
  - ``n_decode_lines``: total decode lines across all periods
  - ``n_periods``: total periods run

False-alarm rate = n_detected / n_periods  (per 15 s period)
                 = n_detected / (n_periods * 15)  (per second)

This is the end-to-end false decode rate, which is the operationally
meaningful quantity: how often does a noise-only input produce a logged
decode?

To measure SPD detection rate (before bpdecode), you would need to instrument
the msk144spd Fortran or parse the ndet field in <DecodeFinished> output --
see note at bottom of file.

Usage:
    python -m experiments.gpu_tfmf.measure_fa [options]

    python -m experiments.gpu_tfmf.measure_fa --periods 200 --seed 1 \\
        --ftol 200 --depth 1

Notes:
    - WAV noise amplitude: 3000 RMS is consistent with WSJT-X's expected
      level (ADC range ~32767; WSJT-X normalises internally).
    - jt9 path: uses `jt9` on PATH (same binary WSJT-X calls).
    - The <DecodeFinished> line format is:
          <DecodeFinished>   ndet   ...
      where ndet is the number of SPD candidates passed to the decoder.
      Parsing this gives SPD detection rate without source modification.
"""

from __future__ import annotations

import argparse
import subprocess
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np


# ── WAV generation ──────────────────────────────────────────────────────────

_FS       = 12_000   # Hz — jt9 MSK144 input sample rate
_DURATION = 15.0     # s  — WSJT-X period length
_N        = int(_FS * _DURATION)   # 180 000 samples


def _make_noise_wav(path: Path, rng: np.random.Generator,
                    amplitude: float = 3000.0) -> None:
    """Write a 12 kHz / 16-bit / mono pure-AWGN WAV to *path*."""
    samples = rng.standard_normal(_N).astype(np.float32) * amplitude
    # Clip to int16 range — at 3000 RMS, ±6σ = ±18 000, well within ±32767.
    pcm = np.clip(np.round(samples), -32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(_FS)
        wf.writeframes(pcm.tobytes())


# ── jt9 runner ──────────────────────────────────────────────────────────────


def _run_jt9(wav_path: Path, ftol: int, depth: int,
             jt9_bin: str = "jt9") -> tuple[int, list[str]]:
    """Run `jt9 -k` on *wav_path*.

    Returns (ndet_spd, decode_lines) where:
      - ndet_spd: number of SPD candidates (from <DecodeFinished> line)
      - decode_lines: list of successfully decoded-message lines
    """
    cmd = [
        jt9_bin, "-k",
        "-d", str(depth),
        "-F", str(ftol),
        "-L", str(1500 - ftol),
        "-H", str(1500 + ftol),
        str(wav_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = result.stdout.splitlines()

    decode_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("<DecodeFinished>") or s.startswith("EOF"):
            # <DecodeFinished> fields are only populated in shmem daemon
            # mode; in file mode they are always "0 0 0" and unusable.
            pass
        elif s:
            decode_lines.append(s)

    return 0, decode_lines


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--periods", type=int, default=100,
                    help="Number of 15-s noise periods to test (default: 100)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility (default: 0)")
    ap.add_argument("--amplitude", type=float, default=3000.0,
                    help="Noise RMS amplitude in int16 units (default: 3000)")
    ap.add_argument("--ftol", type=int, default=200,
                    help="Frequency tolerance passed to jt9 -F (default: 200)")
    ap.add_argument("--depth", type=int, default=1,
                    help="Decoder depth passed to jt9 -d (default: 1)")
    ap.add_argument("--jt9", default="jt9",
                    help="Path to jt9 binary (default: jt9 on PATH)")
    ap.add_argument("--keep-wavs", action="store_true",
                    help="Keep generated WAV files in a temp dir (for debug)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_periods      = args.periods
    n_decoded      = 0      # periods with at least one false decode
    n_decode_lines = 0      # total false decode lines

    with tempfile.TemporaryDirectory(prefix="map144_fa_") as tmpdir:
        tmp = Path(tmpdir)
        wav_path = tmp / "noise.wav"

        print(f"Running {n_periods} periods "
              f"(seed={args.seed}, ftol={args.ftol}, depth={args.depth}, "
              f"amplitude={args.amplitude:.0f})")

        for i in range(n_periods):
            _make_noise_wav(wav_path, rng, amplitude=args.amplitude)
            _, dlines = _run_jt9(wav_path, args.ftol, args.depth, args.jt9)

            n_decode_lines += len(dlines)
            if dlines:
                n_decoded += 1
                for dl in dlines:
                    print(f"  [period {i:4d}] FALSE DECODE: {dl}")

            if (i + 1) % 10 == 0:
                print(f"  {i+1:4d}/{n_periods}  false_decodes_so_far={n_decoded}",
                      flush=True)

    total_s = n_periods * _DURATION

    # 95% Poisson upper bound on false decode rate: -ln(0.05) / n_periods
    import math
    ub_per_period = math.log(20) / n_periods   # -ln(0.05) = ln(20)
    ub_per_s = ub_per_period / _DURATION

    print()
    print("═" * 60)
    print("FALSE-ALARM RATE MEASUREMENT — pure AWGN → jt9 --msk144")
    print("═" * 60)
    print(f"  Periods          : {n_periods}  ({total_s:.0f} s total)")
    print(f"  Sample rate      : {_FS} Hz")
    print(f"  Noise amplitude  : {args.amplitude:.0f} RMS (int16)")
    print(f"  jt9 ftol         : ±{args.ftol} Hz")
    print(f"  jt9 depth        : {args.depth}")
    print()
    print("Note: <DecodeFinished> ndet field is always 0 in jt9 file mode")
    print("      (only populated in shmem/daemon mode).  SPD candidate rate")
    print("      is not directly observable; this script measures end-to-end")
    print("      false decode rate only.")
    print()
    print("End-to-end false decodes (SPD + bpdecode):")
    print(f"  Periods with decode   : {n_decoded} / {n_periods}  "
          f"({100*n_decoded/n_periods:.2f}%)")
    print(f"  Total decode lines    : {n_decode_lines}")
    if n_decoded > 0:
        print(f"  False decode rate     : "
              f"{n_decoded/n_periods:.5f} /period  "
              f"({n_decoded/total_s:.6f} /s)")
    else:
        print(f"  False decode rate     : 0  (95% UB: {ub_per_period:.4f} /period  "
              f"= {ub_per_s:.5f} /s)")
    print()
    print("Comparison:")
    print("  Corpus off-freq empirical : ~0.00684 false/channel/15s-period")
    print("                              = 0.000456 /s per channel (upper bound,")
    print("                                contaminated by real off-freq signals)")
    print(f"  This measurement (95% UB) : {ub_per_period:.5f} /period = {ub_per_s:.5f} /s")


if __name__ == "__main__":
    main()
