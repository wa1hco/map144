#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Compare legacy-MAP144 vs Block-shadow decodes — bake-in audit tool.

After running MAP144 with ``MAP144_USE_BLOCKS=1`` (cut-over option 1, the
shadow Block runtime alongside the legacy DSP path), this script
diff-analyses the two ``decodes.jsonl`` files to verify the Block path
saw the same decodes as legacy.

Inputs (default paths)
----------------------
- ``MSK144/detections/decodes.jsonl``        — legacy authoritative decodes
- ``MSK144/detections/blocks/decodes.jsonl`` — Block-shadow decodes

Both files share the schema written by ``detection.extract_and_decode``:

    {"timestamp": "2026-05-09_05:26:31.6",
     "period_ts": "2026-05-09_05:26:45",
     "t_sec": 1.344, "radio_khz": 50265,
     "message": "CQ K1JT FN20",
     "jt9_snr_db": 16, "est_snr_db": 16,
     "jt9_line": "[SPD navg=1 ...]",
     "theta_deg": null}

Match criteria
--------------
A legacy decode and a shadow decode are the SAME event when:

  - their ``message`` strings share at least one callsign-shaped token
    (3-6 chars, contains a digit), AND
  - their ``radio_khz`` are within ``--freq-tol`` kHz (default 1), AND
  - their ``timestamp`` are within ``--time-tol`` seconds (default 5).

Output report
-------------

  - **N_legacy / N_shadow / N_matched** — counts.
  - **Coverage** — fraction of legacy decodes that have a shadow match.
    A regression in Block sensitivity shows up as < 100 % coverage.
  - **Shadow-only** — decodes the shadow saw but legacy didn't.  Could
    be a sensitivity GAIN (shadow stronger) or a false-positive.  Worth
    eyeballing.
  - **Time-skew distribution** — median + p99 of (shadow_ts − legacy_ts)
    for matched pairs.  Expected near 0 ± seconds; a large +ve median
    means shadow runs slower; -ve means shadow fires earlier.
  - **SNR delta distribution** — typical (shadow_snr − legacy_snr) for
    matched pairs.  Expected ~0 ± 1-2 dB jitter (same SPD/jt9 on the
    same IQ window; small differences are attribution-window choice).

Run::

    python3 tools/compare_shadow.py
    python3 tools/compare_shadow.py --since 2026-05-09_00:00 --until 2026-05-09_23:59
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY_PATH = _REPO_ROOT / "MSK144" / "detections" / "decodes.jsonl"
DEFAULT_SHADOW_PATH = _REPO_ROOT / "MSK144" / "detections" / "blocks" / "decodes.jsonl"

# ``YYYY-MM-DD_HH:MM:SS.mmm`` and shorter prefixes.
_TS_FMT_FULL  = "%Y-%m-%d_%H:%M:%S.%f"
_TS_FMT_SHORT = "%Y-%m-%d_%H:%M:%S"


def _parse_ts(ts: str) -> datetime:
    """Parse the engine's UTC timestamp.  Fractional second optional;
    ``--since`` / ``--until`` accept ``YYYY-MM-DD``,
    ``YYYY-MM-DD_HH``, ``YYYY-MM-DD_HH:MM`` prefixes too — anything
    shorter pads with zeros so the comparison is well-defined.
    """
    for fmt in (_TS_FMT_FULL, _TS_FMT_SHORT,
                "%Y-%m-%d_%H:%M", "%Y-%m-%d_%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp format: {ts!r}")


def _load(path: Path, since: datetime | None, until: datetime | None) -> list[dict]:
    """Load all JSONL entries from ``path`` within the optional window."""
    if not path.is_file():
        print(f"[warn] {path} does not exist — treating as empty", file=sys.stderr)
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_str = entry.get("timestamp", "")
        try:
            ts = _parse_ts(ts_str)
        except ValueError:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        entry["_ts"] = ts
        out.append(entry)
    out.sort(key=lambda e: e["_ts"])
    return out


_TOKEN_RE = re.compile(r"\b[A-Z0-9]{3,6}\b")


def _callsign_tokens(message: str) -> set[str]:
    """Extract callsign-ish tokens from a message — at least one digit
    so we don't match common English words like 'CQ' or 'TNX' on their
    own.  ``CQ K1JT FN20`` → ``{'K1JT', 'FN20'}``.
    """
    out = set()
    for tok in _TOKEN_RE.findall(message.upper()):
        if any(c.isdigit() for c in tok):
            out.add(tok)
    return out


def _match(legacy: list[dict], shadow: list[dict],
           freq_tol_khz: int, time_tol_s: float) -> tuple[
               list[tuple[dict, dict]], list[dict], list[dict]]:
    """Greedy match each legacy entry to a shadow entry within tolerance.

    Returns (matched_pairs, legacy_only, shadow_only).
    """
    used_shadow_ix: set[int] = set()
    matched: list[tuple[dict, dict]] = []
    legacy_only: list[dict] = []

    for L in legacy:
        l_tokens = _callsign_tokens(L.get("message", ""))
        l_freq   = L.get("radio_khz")
        l_ts     = L["_ts"]

        best_ix: int | None = None
        best_skew = float("inf")
        for ix, S in enumerate(shadow):
            if ix in used_shadow_ix:
                continue
            s_tokens = _callsign_tokens(S.get("message", ""))
            if not (l_tokens & s_tokens):
                continue
            s_freq = S.get("radio_khz")
            if l_freq is None or s_freq is None:
                continue
            if abs(int(l_freq) - int(s_freq)) > freq_tol_khz:
                continue
            skew_s = abs((S["_ts"] - l_ts).total_seconds())
            if skew_s > time_tol_s:
                continue
            if skew_s < best_skew:
                best_skew = skew_s
                best_ix = ix
        if best_ix is not None:
            matched.append((L, shadow[best_ix]))
            used_shadow_ix.add(best_ix)
        else:
            legacy_only.append(L)

    shadow_only = [S for ix, S in enumerate(shadow) if ix not in used_shadow_ix]
    return matched, legacy_only, shadow_only


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _print_section_header(title: str) -> None:
    print()
    print(title)
    print("─" * len(title))


def _format_entry(e: dict) -> str:
    msg  = e.get("message", "")
    f    = e.get("radio_khz", "?")
    snr  = e.get("jt9_snr_db", "?")
    ts   = e.get("timestamp", "?")
    return f"{ts}  {f} kHz  SNR {snr:>3}  {msg!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", default=str(DEFAULT_LEGACY_PATH),
                    help="path to legacy decodes.jsonl")
    ap.add_argument("--shadow", default=str(DEFAULT_SHADOW_PATH),
                    help="path to Block-shadow decodes.jsonl")
    ap.add_argument("--since", default=None,
                    help="only consider decodes with timestamp >= this "
                         "(YYYY-MM-DD_HH:MM[:SS])")
    ap.add_argument("--until", default=None,
                    help="only consider decodes with timestamp <= this")
    ap.add_argument("--freq-tol", type=int, default=1,
                    help="frequency match tolerance in kHz (default 1)")
    ap.add_argument("--time-tol", type=float, default=5.0,
                    help="timestamp match tolerance in seconds (default 5)")
    ap.add_argument("--show-mismatches", type=int, default=20,
                    help="how many legacy-only / shadow-only entries to "
                         "print (default 20; 0 disables)")
    args = ap.parse_args()

    since = _parse_ts(args.since) if args.since else None
    until = _parse_ts(args.until) if args.until else None

    legacy = _load(Path(args.legacy), since, until)
    shadow = _load(Path(args.shadow), since, until)

    _print_section_header("Inputs")
    print(f"  legacy : {args.legacy}")
    print(f"           {len(legacy)} decode(s)")
    print(f"  shadow : {args.shadow}")
    print(f"           {len(shadow)} decode(s)")
    if since:
        print(f"  since  : {since}")
    if until:
        print(f"  until  : {until}")

    matched, legacy_only, shadow_only = _match(
        legacy, shadow,
        freq_tol_khz=args.freq_tol,
        time_tol_s=args.time_tol,
    )

    n_legacy = len(legacy)
    n_shadow = len(shadow)
    n_match  = len(matched)

    _print_section_header("Match summary")
    print(f"  matched      : {n_match}")
    print(f"  legacy-only  : {len(legacy_only)}")
    print(f"  shadow-only  : {len(shadow_only)}")
    if n_legacy > 0:
        coverage = 100.0 * n_match / n_legacy
        print(f"  coverage     : {coverage:5.1f} %  "
              f"(fraction of legacy decodes seen by shadow)")
    if n_shadow > 0:
        precision = 100.0 * n_match / n_shadow
        print(f"  precision    : {precision:5.1f} %  "
              f"(fraction of shadow decodes that legacy also saw)")

    if matched:
        # Time skew: shadow_ts − legacy_ts.  Positive = shadow lags legacy.
        skews = [(S["_ts"] - L["_ts"]).total_seconds() for L, S in matched]
        snr_deltas = []
        for L, S in matched:
            l_snr = L.get("jt9_snr_db")
            s_snr = S.get("jt9_snr_db")
            if isinstance(l_snr, (int, float)) and isinstance(s_snr, (int, float)):
                snr_deltas.append(float(s_snr) - float(l_snr))

        _print_section_header("Time skew (shadow − legacy)")
        print(f"  median       : {_percentile(skews, 0.5):+.3f} s")
        print(f"  p99          : {_percentile(skews, 0.99):+.3f} s")
        print(f"  min/max      : {min(skews):+.3f} / {max(skews):+.3f} s")

        if snr_deltas:
            _print_section_header("SNR delta (shadow − legacy)")
            print(f"  median       : {_percentile(snr_deltas, 0.5):+.2f} dB")
            print(f"  p99          : {_percentile(snr_deltas, 0.99):+.2f} dB")
            print(f"  min/max      : {min(snr_deltas):+.1f} / {max(snr_deltas):+.1f} dB")

    n_show = args.show_mismatches
    if n_show > 0 and legacy_only:
        _print_section_header(
            f"Legacy-only decodes (shadow MISSED)  ―  showing first {min(n_show, len(legacy_only))}"
        )
        for e in legacy_only[:n_show]:
            print(f"  {_format_entry(e)}")
        if len(legacy_only) > n_show:
            print(f"  ... and {len(legacy_only) - n_show} more")

    if n_show > 0 and shadow_only:
        _print_section_header(
            f"Shadow-only decodes (legacy missed; possible sensitivity gain)  ―  showing first {min(n_show, len(shadow_only))}"
        )
        for e in shadow_only[:n_show]:
            print(f"  {_format_entry(e)}")
        if len(shadow_only) > n_show:
            print(f"  ... and {len(shadow_only) - n_show} more")

    # Exit-code semantics: 0 only if shadow saw every legacy decode.
    # The morning eyeball check uses this.
    return 0 if not legacy_only else 1


if __name__ == "__main__":
    sys.exit(main())
