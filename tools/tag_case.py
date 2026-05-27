#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Append a per-case manual tag annotation to the corpus.

Usage
-----
    tools/tag_case.py <case_id> [+tag ...] [-tag ...] ["note text"]

Examples
--------
    # Add two tags and a note
    tools/tag_case.py period_260507_092845 +short_burst +fluttery "classic flutter"

    # Remove an auto-applied tag that's wrong
    tools/tag_case.py period_260507_095930 -wsjtx_miss "MAP144 did detect, msk144_spd path"

    # Add a brand-new tag name on the fly (the audit tool will flag it as
    # undeclared in tags.py until you promote it)
    tools/tag_case.py period_260511_105430 +below_floor

Semantics
---------
Each invocation appends ONE record to ``tests/corpus/manual_tags.jsonl``:

    {"ts": <utc ISO>, "case": <case_id>, "ops": ["+tag", "-tag", ...], "note": <text>}

When ``select_stress_corpus.py`` rebuilds the index, it replays this log in
chronological order, applying additions/removals per case.  Latest entry
wins per (case, tag); deletions are sticky until a later ``+tag`` re-adds.

This file is append-only by convention.  Hand-edit it only to remove
mistakes; never reorder lines.  It's safe to commit to git.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


_REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = _REPO_ROOT / "tests" / "corpus" / "manual_tags.jsonl"

# case_id format from select_stress_corpus.py: period_yymmdd_HHMMSS
_CASE_RE = re.compile(r'^period_\d{6}_\d{6}$')
_TAG_RE  = re.compile(r'^[a-z][a-z0-9_]*$')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case_id',
                    help='Corpus case id (e.g. period_260507_092845)')
    ap.add_argument('args', nargs='*',
                    help='Tag ops (+tag / -tag) and optional final note string')
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG_PATH,
                    help='Path to manual_tags.jsonl (default: %(default)s)')
    args = ap.parse_args(argv)

    if not _CASE_RE.match(args.case_id):
        print(f"error: case_id {args.case_id!r} not in expected form "
              f"'period_yymmdd_HHMMSS'", file=sys.stderr)
        return 2

    ops: list[str] = []
    note_parts: list[str] = []
    for tok in args.args:
        if tok.startswith('+') or tok.startswith('-'):
            tag = tok[1:]
            if not _TAG_RE.match(tag):
                print(f"error: tag name {tag!r} must match [a-z][a-z0-9_]*",
                      file=sys.stderr)
                return 2
            ops.append(tok)
        else:
            note_parts.append(tok)

    if not ops and not note_parts:
        print("error: nothing to do — supply at least one +tag/-tag or a note",
              file=sys.stderr)
        return 2

    rec = {
        "ts":   datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "case": args.case_id,
        "ops":  ops,
        "note": ' '.join(note_parts),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('a') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"appended {rec} -> {args.log}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
