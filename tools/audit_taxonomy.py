#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit the corpus tag taxonomy for overlap and gaps.

Reads ``tests/corpus/index.csv`` (produced by ``select_stress_corpus.py``)
and reports:

  * per-tag counts and pass-criterion classification
  * co-occurrence Jaccard for every tag pair — high overlap (>= 0.7)
    is flagged as a merge candidate
  * tags with no-predicate (declared in :mod:`tests.corpus.tags` but
    not yet computable)
  * tags with zero matches (predicate implemented but no data fits)
  * orphan tag combinations (tag declared but never co-occurs with the
    artefact tags ``has_wsjtx_wav`` / ``has_map144_wav``)
  * tag combinations that look like they describe the same regime

Read-only.  Run it after each restructuring of ``tests/corpus/tags.py``
and after each fresh ``select_stress_corpus.py`` run.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.corpus.tags import TAGS, TAG_BY_NAME    # noqa: E402

DEFAULT_INDEX = _REPO_ROOT / "tests" / "corpus" / "index.csv"
HIGH_OVERLAP_J = 0.70   # Jaccard threshold for "merge candidate"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--top-cooccur', type=int, default=20)
    args = ap.parse_args(argv)

    if not args.index.is_file():
        print(f'error: index not found at {args.index}', file=sys.stderr)
        return 1

    # ── Load index ──────────────────────────────────────────────────────────
    tag_periods: dict[str, set[str]] = defaultdict(set)
    n_total = 0
    with args.index.open() as fh:
        for row in csv.DictReader(fh):
            n_total += 1
            if not row['tags']:
                continue
            for t in row['tags'].split(';'):
                tag_periods[t].add(row['case_id'])

    print(f"taxonomy audit  (corpus: {args.index})")
    print(f"  total periods         : {n_total}")
    print(f"  periods with any tag  : "
          f"{len(set().union(*tag_periods.values())) if tag_periods else 0}")
    print()

    # ── Per-tag distribution ────────────────────────────────────────────────
    print("=" * 72)
    print("PER-TAG COUNTS  (pass_when: 'decoded' = catch test,"
          " 'no_decode' = false-positive test, 'any' = orthogonal)")
    print("=" * 72)
    print(f"  {'tag':<28} {'count':>7}  {'pass_when':<10}  {'has predicate?'}")
    declared = {t.name for t in TAGS}
    rows = []
    for t in TAGS:
        n = len(tag_periods.get(t.name, set()))
        pred = '   yes' if t.predicate is not None else 'no (skipped)'
        rows.append((t.name, n, t.pass_when, pred))
    # sort descending count
    rows.sort(key=lambda r: -r[1])
    for name, n, pw, pr in rows:
        print(f"  {name:<28} {n:>7}  {pw:<10}  {pr}")

    # ── Tags that fired zero times (predicate but no data) ──────────────────
    print()
    print("=" * 72)
    print("ZERO-MATCH TAGS  (predicate exists but no period satisfies it)")
    print("=" * 72)
    zero = [t for t in TAGS if t.predicate is not None
            and len(tag_periods.get(t.name, set())) == 0]
    if not zero:
        print("  (none — every active predicate matched at least one period)")
    else:
        for t in zero:
            print(f"  {t.name:<28} — {t.description.splitlines()[0][:60]}")

    # ── Tags in index but not declared (predicate renamed/removed) ──────────
    leftover = set(tag_periods) - declared
    if leftover:
        print()
        print("=" * 72)
        print("UNDECLARED TAGS IN INDEX  (tag present in CSV but not in tags.py)")
        print("=" * 72)
        for t in sorted(leftover):
            print(f"  {t:<28} {len(tag_periods[t])}")

    # ── Co-occurrence Jaccard ───────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"TOP CO-OCCURRENCES  (Jaccard >= 0 sorted, capped at {args.top_cooccur})")
    print("=" * 72)
    pairs = []
    for a, b in combinations(sorted(tag_periods), 2):
        sa, sb = tag_periods[a], tag_periods[b]
        inter = len(sa & sb)
        union = len(sa | sb)
        if union == 0 or inter == 0:
            continue
        j = inter / union
        pairs.append((j, a, b, inter, len(sa), len(sb)))
    pairs.sort(reverse=True)
    print(f"  {'jaccard':>7}  {'tag_a':<26} {'tag_b':<26} "
          f"{'both':>6} {'A':>6} {'B':>6}  flag")
    for j, a, b, inter, na, nb in pairs[:args.top_cooccur]:
        flag = '  MERGE?' if j >= HIGH_OVERLAP_J else ''
        print(f"  {j:>7.3f}  {a:<26} {b:<26} {inter:>6} {na:>6} {nb:>6}{flag}")

    # ── Tag-class summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("PASS-CRITERION BUCKETS  (how many periods are usable as each test type)")
    print("=" * 72)
    by_pass = {'decoded': set(), 'no_decode': set(), 'any': set()}
    for t in TAGS:
        if t.predicate is None: continue
        by_pass.setdefault(t.pass_when, set()).update(tag_periods.get(t.name, set()))
    for pw in ('decoded', 'no_decode', 'any'):
        periods = by_pass.get(pw, set())
        print(f"  pass_when = {pw:<10} periods covered : {len(periods)}")

    # ── Mission-critical buckets ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("MISSION-CRITICAL BUCKETS  (named individually for plan reference)")
    print("=" * 72)
    interesting = [
        ('wsjtx_miss',                 'WSJT-X decoded, MAP144 did not (the 70/day gap)'),
        ('weak',                       'WSJT-X SNR <= -3'),
        ('weak ∩ wsjtx_miss',          'weak + WSJT-X miss (most actionable)'),
        ('canonical_strong',           'sanity-floor decodes'),
        ('both_decoded ∩ both_wavs_available',
                                       'correctness comparison with both WAVs available'),
        ('post_burst_rebound',         'wideband-burst-driven false triggers'),
        ('persistent_nodecode',        'noise-driven trigger periods'),
        ('map144_only',                'MAP144 decoded, WSJT-X did not (advantage / phantom)'),
    ]
    for label, desc in interesting:
        if '∩' in label:
            a, b = label.split(' ∩ ')
            n = len(tag_periods.get(a, set()) & tag_periods.get(b, set()))
        else:
            n = len(tag_periods.get(label, set()))
        print(f"  {label:<42} {n:>6}   {desc}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
