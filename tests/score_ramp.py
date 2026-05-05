#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Score a calibrated SNR-ramp test against WSJT-X (jt9) and / or MAP144 decode logs.

Usage
-----
    # Audio side (WSJT-X / jt9):
    jt9 --msk144 ramp_<stamp>_audio.wav > /tmp/wsjt_log.txt
    score_ramp.py ramp_<stamp>_truth.json --wsjt /tmp/wsjt_log.txt

    # IQ side (MAP144 — first replay the IQ WAV through the GUI, then):
    score_ramp.py ramp_<stamp>_truth.json --map144 MSK144/detections/decodes.jsonl

    # Both, side by side:
    score_ramp.py ramp_<stamp>_truth.json --wsjt /tmp/wsjt_log.txt \
                  --map144 MSK144/detections/decodes.jsonl

The truth sidecar contains one ``A...`` message per ping; each decoder reports
its decoded message string in its log.  Matching is exact-string — every ping
is uniquely identifiable, so there is no ambiguity.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_AN_AP_RE = re.compile(r'\bA[PN]\d{5}[PN]\d{2}\b')


def parse_jt9_log(path: Path) -> set[str]:
    """Parse a ``jt9 --msk144`` stdout dump.  Return the set of A... messages decoded."""
    msgs = set()
    for line in Path(path).read_text(errors='replace').splitlines():
        for m in _AN_AP_RE.findall(line):
            msgs.add(m)
    return msgs


def parse_map144_log(path: Path) -> set[str]:
    """Parse MAP144 decodes.jsonl.  Return set of A... messages found in 'message' field."""
    msgs = set()
    for line in Path(path).read_text(errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = (d.get('message') or '').strip()
        if _AN_AP_RE.fullmatch(msg):
            msgs.add(msg)
    return msgs


def score(truth_pings: list[dict], decoded: set[str]) -> list[tuple[float, bool]]:
    """For each truth ping, return ``(snr_db_input, decoded_flag)``."""
    return [(p['snr_db_input'], p['message'] in decoded) for p in truth_pings]


def summarise(rows: list[tuple[float, bool]], label: str) -> str:
    if not rows:
        return f"{label}: no truth pings"
    by_snr: dict[float, list[bool]] = defaultdict(list)
    for snr, ok in rows:
        by_snr[snr].append(ok)
    lines = [f"\n=== {label} ==="]
    lines.append(f"{'input dB':>9}  {'decoded':>8}")
    threshold = None
    snr_keys = sorted(by_snr)
    for snr in snr_keys:
        oks = by_snr[snr]
        rate = sum(oks) / len(oks)
        marker = '  ✓' if rate >= 0.5 and threshold is None else ''
        if rate >= 0.5 and threshold is None:
            threshold = snr
        bar  = ('#' * int(round(rate * 10))).ljust(10)
        lines.append(f"{snr:>+9.1f}  {sum(oks):>3d}/{len(oks):<3d}  {bar}{marker}")
    total = sum(1 for _, ok in rows if ok)
    lines.append(f"\n  total decoded: {total}/{len(rows)}")
    if threshold is not None:
        lines.append(f"  50%-threshold (lowest SNR with ≥50% decoded): {threshold:+.1f} dB")
    else:
        lines.append("  50%-threshold: not reached at any SNR in the ramp")
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('truth', help='Path to ramp truth JSON sidecar')
    ap.add_argument('--wsjt',   metavar='LOG',
                    help='jt9 --msk144 stdout log (or any file containing decoded messages)')
    ap.add_argument('--map144', metavar='JSONL',
                    help='MAP144 decodes.jsonl (any superset is fine — only A... messages match)')
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    pings = truth['pings']
    print(f"Truth: {len(pings)} pings, SNR "
          f"{min(p['snr_db_input'] for p in pings):+.1f} → "
          f"{max(p['snr_db_input'] for p in pings):+.1f} dB")
    if 'iq_offset_hz' in truth:
        print(f"  IQ offset: {truth['iq_offset_hz']:+.0f} Hz")

    if args.wsjt is None and args.map144 is None:
        sys.exit("Provide --wsjt and/or --map144 (otherwise nothing to score).")

    if args.wsjt:
        decoded = parse_jt9_log(Path(args.wsjt))
        print(summarise(score(pings, decoded), f"WSJT-X / jt9  ({args.wsjt})"))

    if args.map144:
        decoded = parse_map144_log(Path(args.map144))
        print(summarise(score(pings, decoded), f"MAP144  ({args.map144})"))


if __name__ == '__main__':
    main()
