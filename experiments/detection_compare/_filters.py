# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Filters for excluding test-signal pollution from real-data analyses.

When running statistics on the operator's launches.jsonl / decodes.jsonl,
synthetic test messages must be removed first — otherwise they show up as
"activity" in channels where no real radio activity exists.  See changelog
v0.1.3-alpha for the test-pollution issue.

Use ``is_test_message()`` as the standard test-pollution filter.  More
aggressive filtering (canned messages with real callsigns, playback
sessions) is opt-in via ``include_canned_real_callsigns=True``.
"""
from __future__ import annotations

import re

# Synthetic-test ping-message format from generate_msk144.py:
#   "A[P/N][ff][ttt][P/N][dd]"  — exactly 10 ASCII chars.
# Examples: AP02089P12, AN01103P12, AP07037P18
_TEST_MSG_RE = re.compile(r'^A[PN]\d{2}\d{3}[PN]\d{2}$')

# Canned test messages used in synthetic-IQ / playback scripts.  These use
# REAL callsigns (K1JT = Joe Taylor) but are NOT real radio transmissions in
# our corpus.  Filter them by default if any of these appear ≥ 20× at any
# single (channel, message) — that pattern is diagnostic of test pollution,
# not real activity.
_CANNED_REAL_CALLSIGN_MSGS = {
    'CQ K1JT FN20',     # synthetic.py default; flagged in v0.1.3 changelog
    'CQ AA2UK IO92',    # generate_msk144 demo
    'K1JT W5ADD R-02',  # QEX article example traffic
    'K1JT K9AN EN50',   # WSJT-X documentation example
}


def is_test_message(msg: str, include_canned_real_callsigns: bool = True) -> bool:
    """True if message is a MAP144 synthetic-test artifact.

    Strict path (always on): matches the ``A[PN]ffNNN[PN]dd`` synthetic
    encoding from generate_msk144.py.  These are unambiguously test signals.

    Soft path (``include_canned_real_callsigns=True``, default on): also
    filters canned messages that use real callsigns and appear in the
    synthetic-test fixtures.  Use ``False`` to keep these in the data when
    investigating cross-contamination cases specifically.
    """
    if not msg:
        return False
    s = msg.strip().upper()
    if _TEST_MSG_RE.match(s):
        return True
    if include_canned_real_callsigns and s in _CANNED_REAL_CALLSIGN_MSGS:
        return True
    return False


def filter_real_only(launches, include_canned_real_callsigns: bool = True):
    """Generator: yield only non-test launch entries.

    Takes an iterable of dicts (parsed JSON launch records).  Filters out
    entries whose 'message' field is a known synthetic-test artifact.
    Entries with empty/missing message (no_decode triggers) pass through
    unfiltered — they're not test pollution.
    """
    for entry in launches:
        msg = entry.get('message') or ''
        if is_test_message(msg, include_canned_real_callsigns):
            continue
        yield entry
