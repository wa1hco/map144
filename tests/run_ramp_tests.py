#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Automated multi-seed ramp test driver: generate, decode, score, aggregate.

For each (seed, iq_offset) combination this driver:

  1. Calls ``generate_msk144.py --ramp ...`` to produce a paired audio + IQ + truth.
  2. Calls ``jt9 --msk144 <audio.wav>`` headlessly and captures the decode log.
  3. Cross-references the decode log against the truth sidecar.
  4. Optionally cross-references a MAP144 ``decodes.jsonl`` (see ``--map144-jsonl``).
  5. Pools results across seeds and reports a per-SNR decode rate.

This covers the WSJT-X (jt9) side end-to-end without GUI interaction.  MAP144
currently still needs a separate replay step (its WAV ingest path runs through
the live Engine + channelizer); supply ``--map144-jsonl`` once you have replayed
the IQ files manually, and the driver will pool MAP144 stats the same way.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent   # tests/ → repo root
GENERATE_PY = PROJECT_DIR / 'generate_msk144.py'

_AN_AP_RE = re.compile(r'\bA[PN]\d{5}[PN]\d{2}\b')


def parse_jt9_log(text: str) -> set[str]:
    return set(_AN_AP_RE.findall(text))


def parse_map144_jsonl(path: Path, since_ts: str | None = None) -> set[str]:
    """Return AN/AP messages from MAP144 ``decodes.jsonl``.  If ``since_ts`` is
    given (ISO-ish ``YYYY-MM-DD_HH:MM:SS`` prefix), include only entries whose
    ``timestamp`` is lexicographically ≥ ``since_ts``."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since_ts and (d.get('timestamp') or '') < since_ts:
            continue
        msg = (d.get('message') or '').strip()
        if _AN_AP_RE.fullmatch(msg):
            out.add(msg)
    return out


def replay_map144_headless(iq_path: Path, decodes_jsonl: Path) -> set[str]:
    """Replay one IQ WAV through MAP144 headlessly.  Return the AN/AP messages
    decoded **during this replay** (filtered by the captured run_start)."""
    # Lazy import so this file works as a doc target without map144_app deps.
    from headless_replay import replay_iq_wav
    run_start = replay_iq_wav(str(iq_path), verbose=False)
    # decodes.jsonl entries log timestamp as "YYYY-MM-DD_HH:MM:SS.f"; convert
    # run_start to the same prefix so lexicographic compare works.
    since_prefix = run_start.strftime('%Y-%m-%d_%H:%M:%S')
    return parse_map144_jsonl(decodes_jsonl, since_ts=since_prefix)


def run_one_ramp(args, seed: int, iq_offset_hz: float, out_dir: Path) -> dict:
    """Generate one ramp file pair, decode the audio side with jt9, score it.

    Returns a dict with keys: truth_path, audio_path, iq_path, jt9_decoded (set),
    pings (truth list).
    """
    cmd = [
        sys.executable, str(GENERATE_PY),
        '--ramp',
        '--ramp-snr-min', str(args.snr_min),
        '--ramp-snr-max', str(args.snr_max),
        '--ramp-snr-step', str(args.snr_step),
        '--ramp-spacing', str(args.spacing),
        '--ramp-t0', str(args.t0),
        '--ramp-frames', str(args.frames),
        '--iq-offset-hz', str(iq_offset_hz),
        '--seed', str(seed),
        '--duration', str(args.duration),
        '--noise-floor-dbfs', str(args.noise_floor_dbfs),
        '--output-dir', str(out_dir),
        '--skip-plots',
    ]
    print(f"  generate: seed={seed} iq_offset={iq_offset_hz:+.0f}Hz", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"generate_msk144 failed: {r.stderr}")
    # Parse the printed paths back out.  Some lines have trailing parentheticals
    # ("...iq.wav  (offset +0 Hz)") so trim everything after the .wav/.json
    # extension instead of relying on a pure colon split.
    def _extract(line: str) -> str:
        rest = line.split(':', 1)[1].strip()
        for ext in ('.wav', '.json'):
            i = rest.find(ext)
            if i >= 0:
                return rest[:i + len(ext)]
        return rest
    lines = r.stdout.splitlines()
    audio_path = next((_extract(l) for l in lines if 'audio (WSJT-X' in l), None)
    iq_path    = next((_extract(l) for l in lines if 'IQ    (MAP144)' in l), None)
    truth_path = next((_extract(l) for l in lines if l.lstrip().startswith('truth')), None)
    if not (audio_path and truth_path):
        raise RuntimeError(f"could not locate generator output paths in:\n{r.stdout}")
    audio_path = (PROJECT_DIR / audio_path).resolve() if not Path(audio_path).is_absolute() else Path(audio_path)
    iq_path    = (PROJECT_DIR / iq_path).resolve()    if iq_path and not Path(iq_path).is_absolute() else (Path(iq_path) if iq_path else None)
    truth_path = (PROJECT_DIR / truth_path).resolve() if not Path(truth_path).is_absolute() else Path(truth_path)

    # Run jt9 on the audio file
    jt9 = subprocess.run(['jt9', '--msk144', str(audio_path)], capture_output=True, text=True)
    jt9_decoded = parse_jt9_log(jt9.stdout + '\n' + jt9.stderr)

    truth = json.loads(truth_path.read_text())
    result = {
        'seed':         seed,
        'iq_offset_hz': iq_offset_hz,
        'truth_path':   truth_path,
        'audio_path':   audio_path,
        'iq_path':      iq_path,
        'jt9_decoded':  jt9_decoded,
        'pings':        truth['pings'],
    }

    # Optionally run MAP144 headless replay on the same IQ file
    if args.with_map144 and iq_path is not None:
        decodes_jsonl = PROJECT_DIR / 'MSK144' / 'detections' / 'decodes.jsonl'
        result['map144_decoded'] = replay_map144_headless(iq_path, decodes_jsonl)

    return result


def aggregate_per_snr(results: list[dict], decoded_key: str) -> dict[float, tuple[int, int]]:
    """Across all results pool decode/total counts per input-SNR bin.

    Returns ``{snr_db: (n_decoded, n_trials)}``.
    """
    counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])  # [n_ok, n_total]
    for r in results:
        decoded = r[decoded_key]
        for p in r['pings']:
            counts[p['snr_db_input']][1] += 1
            if p['message'] in decoded:
                counts[p['snr_db_input']][0] += 1
    return {snr: (ok, total) for snr, (ok, total) in counts.items()}


def report(results: list[dict], label: str, decoded_key: str) -> str:
    pooled = aggregate_per_snr(results, decoded_key)
    if not pooled:
        return f"\n=== {label}: no data ==="
    snr_keys = sorted(pooled)
    lines = [f"\n=== {label}  ({len(results)} trial(s)) ==="]
    lines.append(f"{'input dB':>9}  {'rate':>8}  {'bar':<22}")
    threshold = None
    for snr in snr_keys:
        ok, total = pooled[snr]
        rate = ok / total if total else 0.0
        bar = ('#' * int(round(rate * 20))).ljust(20)
        marker = '  ✓' if rate >= 0.5 and threshold is None else ''
        if rate >= 0.5 and threshold is None:
            threshold = snr
        lines.append(f"{snr:>+9.1f}  {ok:>3d}/{total:<3d}  {bar}{marker}")
    if threshold is not None:
        lines.append(f"  50%-threshold (lowest SNR with ≥50% decode rate): {threshold:+.1f} dB")
    else:
        lines.append("  50%-threshold: not reached at any SNR in the ramp")
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=10,
                    help='Number of independent noise-seed realizations per (iq-offset) point')
    ap.add_argument('--seed-base', type=int, default=1000,
                    help='Base RNG seed; per-realization seed = seed-base + k')
    ap.add_argument('--iq-offsets-hz', default='0',
                    help='Comma-separated IQ offsets in Hz, e.g. "-12000,-6000,0,+6000,+12000"')
    ap.add_argument('--snr-min',  type=float, default=-12.0)
    ap.add_argument('--snr-max',  type=float, default=+5.0)
    ap.add_argument('--snr-step', type=float, default=1.0)
    ap.add_argument('--spacing',  type=float, default=0.8)
    ap.add_argument('--t0',       type=float, default=0.5)
    ap.add_argument('--frames',   type=int,   default=5)
    ap.add_argument('--duration', type=float, default=15.0)
    ap.add_argument('--noise-floor-dbfs', type=float, default=-96.0)
    ap.add_argument('--output-dir', default='MSK144/simulations/ramp_runs',
                    help='Directory tree for generated files (one subdir per run)')
    ap.add_argument('--with-map144', action='store_true',
                    help='Headlessly replay each IQ file through MAP144 (no GUI) and pool '
                         'its decodes per realization.  Adds MAP144 vs WSJT-X comparison to '
                         'the same one-command output.')
    ap.add_argument('--map144-jsonl', metavar='PATH',
                    help='Alternative to --with-map144: pool against an existing '
                         'decodes.jsonl from a prior manual MAP144 GUI replay.  Less '
                         'accurate (no per-run filtering) — prefer --with-map144.')
    args = ap.parse_args()

    iq_offsets = [float(x.strip()) for x in args.iq_offsets_hz.split(',') if x.strip()]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Ramp test sweep: {args.seeds} seed(s) × {len(iq_offsets)} iq-offset(s)"
          f" = {args.seeds * len(iq_offsets)} runs")
    print(f"  SNR: {args.snr_min:+.1f} → {args.snr_max:+.1f} dB / step {args.snr_step}"
          f"  frames: {args.frames}  spacing: {args.spacing}s")

    all_results: list[dict] = []
    for offset in iq_offsets:
        # One sub-output-dir per offset so each run's files don't collide on time-stamp
        sub_dir = out_root / f"offset_{int(round(offset)):+d}Hz"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for k in range(args.seeds):
            seed = args.seed_base + k
            r = run_one_ramp(args, seed, offset, sub_dir)
            all_results.append(r)

    # WSJT-X / jt9 results — always available
    print(report(all_results, "WSJT-X (jt9 --msk144)", 'jt9_decoded'))

    # MAP144 results — per-run sets if --with-map144, else a shared set if --map144-jsonl
    if args.with_map144:
        print(report(all_results, "MAP144  (headless replay)", 'map144_decoded'))
    elif args.map144_jsonl:
        map144_msgs = parse_map144_jsonl(Path(args.map144_jsonl))
        for r in all_results:
            r['map144_decoded'] = map144_msgs   # same set against every truth
        print(report(all_results, f"MAP144  ({args.map144_jsonl})", 'map144_decoded'))


if __name__ == '__main__':
    main()
