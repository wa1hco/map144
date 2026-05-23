#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Inspect decoded messages by ch_signed to distinguish real activity from
false decodes (Pi events from CRC-collision on noise).

A real radio channel has:
  - repeated callsigns (operators send their call dozens of times)
  - CQ / grid / signal-report patterns
  - consistent vocabulary

A noise channel has:
  - mostly unique messages (random LDPC+CRC collisions)
  - no callsign repetition pattern
  - mix of garbage tokens
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from experiments.detection_compare._filters import is_test_message

INPUT = REPO / 'MSK144' / 'detections' / 'launches.jsonl'

# Channels of interest based on the prior decode-count distribution.
CHANNELS_TO_INSPECT = [0, 2, 3, 5, 7, 8, 17, 12, -1, -2, -5]
TOP_N = 8        # show top-N messages per channel
SAMPLE_LINES = 5 # plus N example "non-top" messages


def callsign_in(msg: str) -> str | None:
    """Extract the first callsign-like token (2-3 prefix letters + digit + 1-3 suffix letters)."""
    m = re.search(r'\b([A-Z0-9]{1,2}\d[A-Z]{1,3})\b', msg.upper())
    return m.group(1) if m else None


def main() -> None:
    if not INPUT.exists():
        print(f"Not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    by_ch: dict[int, list[str]] = defaultdict(list)
    with INPUT.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = d.get('ch_signed')
            outcome = d.get('outcome')
            msg = d.get('message')
            if ch is None or outcome != 'decoded' or not msg:
                continue
            if is_test_message(msg):
                continue
            ch = int(ch)
            if ch in CHANNELS_TO_INSPECT:
                by_ch[ch].append(msg.strip())

    print("Decoded-message content by channel "
          "(real signal => repeated callsigns; noise => mostly unique)\n")
    for ch in CHANNELS_TO_INSPECT:
        msgs = by_ch.get(ch, [])
        total = len(msgs)
        if total == 0:
            print(f"=== ΔF = {ch:+d} kHz : 0 decoded ===\n")
            continue
        ctr = Counter(msgs)
        unique = len(ctr)
        unique_ratio = unique / total
        # Top callsigns extracted
        calls = Counter()
        for m in msgs:
            cs = callsign_in(m)
            if cs:
                calls[cs] += 1
        # Show
        print(f"=== ΔF = {ch:+d} kHz : {total} decoded, "
              f"{unique} unique ({unique_ratio*100:.0f}%) ===")
        print(f"  Top messages:")
        for m, n in ctr.most_common(TOP_N):
            print(f"    {n:4d}× {m!r}")
        if calls:
            print(f"  Top callsigns extracted (n={sum(calls.values())} with callsign):")
            for c, n in calls.most_common(TOP_N):
                print(f"    {n:4d}× {c}")
        else:
            print(f"  No callsign-shaped tokens found.")
        # Sample of non-top messages
        non_top = [m for m, _ in ctr.most_common()[TOP_N:]][:SAMPLE_LINES]
        if non_top:
            print(f"  Sample of lower-rank messages: {non_top}")
        print()


if __name__ == '__main__':
    main()
