#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Tally launches.jsonl by (ch_signed, outcome).

Tests the hypothesis: if no-decode triggers are mostly noise FAs, they should
spread uniformly across ch_signed ∈ [-20, +20]; real signal triggers should
cluster on ch_signed=0 (calling frequency, off-contest).

Output: text histogram + CSV.
"""
from __future__ import annotations

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from experiments.detection_compare._filters import is_test_message

INPUT = REPO / 'MSK144' / 'detections' / 'launches.jsonl'
OUT_CSV = REPO / 'experiments' / 'detection_compare' / 'channel_distribution.csv'


def main() -> None:
    if not INPUT.exists():
        print(f"Not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    counts: dict[tuple[int, str], int] = defaultdict(int)
    total = 0
    skipped_no_ch = 0
    skipped_test_msg = 0
    outcomes_seen: set[str] = set()

    with INPUT.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = d.get('ch_signed')
            outcome = d.get('outcome')
            if ch is None or outcome is None:
                skipped_no_ch += 1
                continue
            msg = d.get('message') or ''
            if is_test_message(msg):
                skipped_test_msg += 1
                continue
            counts[(int(ch), str(outcome))] += 1
            outcomes_seen.add(str(outcome))
            total += 1

    print(f"Parsed {total} launches "
          f"(skipped {skipped_no_ch} pre-schema, "
          f"{skipped_test_msg} synthetic-test artifacts).")
    print(f"Outcomes observed: {sorted(outcomes_seen)}\n")

    # Build a per-channel table.
    channels = sorted({c for c, _ in counts.keys()})
    outcomes = sorted(outcomes_seen)

    # Header
    header = f"{'ch':>4s} | " + " | ".join(f"{o:>10s}" for o in outcomes) + " |  total"
    print(header)
    print("-" * len(header))

    # Per-channel rows; track totals for summary
    outcome_totals = {o: 0 for o in outcomes}
    rows = []
    for ch in channels:
        row_counts = {o: counts.get((ch, o), 0) for o in outcomes}
        row_total = sum(row_counts.values())
        rows.append({'ch_signed': ch, **row_counts, 'total': row_total})
        for o in outcomes:
            outcome_totals[o] += row_counts[o]
        parts = [f"{ch:>4d}"]
        for o in outcomes:
            parts.append(f"{row_counts[o]:>10d}")
        parts.append(f"{row_total:>8d}")
        print(" | ".join(parts))

    print("-" * len(header))
    summary = [f"{'tot':>4s}"]
    for o in outcomes:
        summary.append(f"{outcome_totals[o]:>10d}")
    summary.append(f"{sum(outcome_totals.values()):>8d}")
    print(" | ".join(summary))

    # Quick channel-uniformity statistic for no_decode outcome.
    if 'no_decode' in outcomes:
        nd_counts = [counts.get((c, 'no_decode'), 0) for c in channels]
        mean = sum(nd_counts) / max(len(nd_counts), 1)
        # Coefficient of variation: std/mean — high CV = clustered, low CV = uniform
        if mean > 0:
            var = sum((x - mean) ** 2 for x in nd_counts) / len(nd_counts)
            std = var ** 0.5
            cv = std / mean
            print(f"\nno_decode distribution stats:")
            print(f"  mean = {mean:.0f}, std = {std:.0f}, CV = {cv:.3f}")
            print(f"  (CV near 0 = uniform; CV >> 1 = clustered)")
            # Compare ch=0 vs everywhere else
            ch0 = counts.get((0, 'no_decode'), 0)
            avg_other = (sum(nd_counts) - ch0) / max(len(nd_counts) - 1, 1)
            ratio = ch0 / avg_other if avg_other > 0 else float('inf')
            print(f"  ch=0 vs avg of others: {ch0} / {avg_other:.0f} = {ratio:.2f}x")
            print(f"  (ratio near 1 = uniform; >> 1 = ch 0 spike)")

    # Compare for decoded outcomes
    decoded_outcomes = [o for o in outcomes if 'decod' in o.lower() and o != 'no_decode']
    if decoded_outcomes:
        # Combine all decoded variants
        dec_counts = [sum(counts.get((c, o), 0) for o in decoded_outcomes) for c in channels]
        mean = sum(dec_counts) / max(len(dec_counts), 1)
        if mean > 0:
            var = sum((x - mean) ** 2 for x in dec_counts) / len(dec_counts)
            std = var ** 0.5
            cv = std / mean
            print(f"\ndecoded distribution stats:")
            print(f"  mean = {mean:.0f}, std = {std:.0f}, CV = {cv:.3f}")
            ch0 = sum(counts.get((0, o), 0) for o in decoded_outcomes)
            avg_other = (sum(dec_counts) - ch0) / max(len(dec_counts) - 1, 1)
            ratio = ch0 / avg_other if avg_other > 0 else float('inf')
            print(f"  ch=0 vs avg of others: {ch0} / {avg_other:.0f} = {ratio:.2f}x")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ch_signed'] + outcomes + ['total'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote per-channel table to {OUT_CSV}")


if __name__ == '__main__':
    main()
