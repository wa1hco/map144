#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-WAV smoke test for detection-stage changes.

Replays a curated set of real-world WAVs through MAP144's full DSP pipeline
(headless_replay), collects the resulting decodes from
``MSK144/detections/decodes.jsonl``, and reports a table of
(case_id, expected, decoded, status).

The smoke set lives in ``tests/corpus/smoke_set.json`` — see that file's
``comment`` field for the rationale and update protocol.

Usage
-----
    python tools/smoke_test.py                      # run full set, print table
    python tools/smoke_test.py --strict             # exit non-zero on regression
    python tools/smoke_test.py --case period_260507_091015  # one case
    python tools/smoke_test.py --json out.json      # also emit machine-readable summary

Why this exists (project lesson 2026-05-15)
-------------------------------------------
The overnight WSJT-X-formula port broke MAP144's decode rate to ZERO because
synthetic balanced-MSK smoke tests didn't exercise CW-interferer or
asymmetric-multipath signal patterns.  This tool runs REAL WAVs from the
operator's corpus — known weakness cases — and is the standing gate before
shipping any detection-stage change.

See feedback_smoke_test_with_balanced_synth_misses_combine_rule_failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.headless_replay import replay_iq_wav  # noqa: E402

_SMOKE_SET_PATH = _REPO_ROOT / "tests" / "corpus" / "smoke_set.json"
_LEGACY_DECODES_PATH = _REPO_ROOT / "MSK144" / "detections" / "decodes.jsonl"


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    category: str
    description: str
    expected_messages: list[str]
    decoded_messages: list[str] = field(default_factory=list)
    decoded_radio_khz: list[int] = field(default_factory=list)
    n_decoded: int = 0
    n_expected: int = 0
    n_matched: int = 0
    n_extra: int = 0          # decoded but not in expected
    n_missed: int = 0         # in expected but not decoded
    status: str = ""          # "PASS" | "REGRESS" | "EXTRA" | "REPORT"
    error: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────

def _read_decodes_since(since: datetime) -> list[dict]:
    """Read decodes.jsonl entries written at or after ``since``."""
    if not _LEGACY_DECODES_PATH.is_file():
        return []
    keep: list[dict] = []
    since_str = since.strftime("%Y-%m-%d_%H:%M:%S")
    for line in _LEGACY_DECODES_PATH.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("timestamp", "") >= since_str:
            keep.append(entry)
    return keep


def _classify(case: dict, decoded: list[dict]) -> tuple[str, int, int, int]:
    """Compare decoded messages to expectations.

    Returns (status, n_matched, n_extra, n_missed).
    """
    expected = list(case.get("expected_messages") or [])
    decoded_msgs = [d.get("message", "").strip() for d in decoded]

    # For must_decode: each expected message must appear at least once
    if case.get("category") == "must_decode":
        missed = [m for m in expected if not any(m in d for d in decoded_msgs)]
        extra = [d for d in decoded_msgs if not any(m in d for m in expected)]
        if missed:
            return ("REGRESS", len(expected) - len(missed), len(extra), len(missed))
        return ("PASS", len(expected), len(extra), 0)

    # For noise_only_*: any decode is an unexpected event but not strict failure
    if case.get("category", "").startswith("noise_only"):
        if decoded_msgs:
            return ("EXTRA", 0, len(decoded_msgs), 0)
        return ("PASS", 0, 0, 0)

    # For operator_observed / asymmetric_multipath / continuous_local:
    # just report what decoded; the expected list is empty (current MAP144 misses these).
    return ("REPORT", 0, len(decoded_msgs), 0)


# ── Per-case runner ───────────────────────────────────────────────────────

def run_case(case: dict, *, verbose: bool = False) -> CaseResult:
    wav = case.get("wav", "")
    cat = case.get("category", "")
    res = CaseResult(
        case_id=case["case_id"],
        category=cat,
        description=case.get("description", ""),
        expected_messages=list(case.get("expected_messages") or []),
        n_expected=len(case.get("expected_messages") or []),
    )

    if not wav or not Path(wav).is_file():
        res.status = "ERROR"
        res.error = f"WAV not found: {wav}"
        return res

    try:
        run_start = replay_iq_wav(
            wav,
            center_freq_mhz=50.260,
            calling_freq_mhz=50.260,
            sample_rate=48_000,
            jt9_join_timeout_s=20.0,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        res.status = "ERROR"
        res.error = f"replay failed: {exc}"
        return res

    decoded = _read_decodes_since(run_start)
    res.decoded_messages = [d.get("message", "").strip() for d in decoded]
    res.decoded_radio_khz = [int(d.get("radio_khz", -1)) for d in decoded]
    res.n_decoded = len(decoded)

    status, matched, extra, missed = _classify(case, decoded)
    res.status, res.n_matched, res.n_extra, res.n_missed = status, matched, extra, missed
    return res


# ── Reporting ─────────────────────────────────────────────────────────────

_STATUS_TAG = {"PASS": "✓", "REGRESS": "✗", "EXTRA": "!", "REPORT": "·", "ERROR": "E"}

def print_table(results: list[CaseResult]) -> None:
    print()
    print(f"{'status':>6}  {'case_id':<26} {'category':<22} {'expected':<5} {'decoded':<5}  message")
    print("-" * 110)
    for r in results:
        tag = _STATUS_TAG.get(r.status, "?")
        msg_preview = "; ".join(r.decoded_messages[:2]) if r.decoded_messages else "—"
        if len(r.decoded_messages) > 2:
            msg_preview += f"  (+{len(r.decoded_messages) - 2} more)"
        if r.error:
            msg_preview = f"ERROR: {r.error}"
        print(f"  {tag} {r.status:>4}  {r.case_id:<26} {r.category:<22} "
              f"{r.n_expected:<5} {r.n_decoded:<5}  {msg_preview}")
    print("-" * 110)
    # Tallies
    by_status = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print("Tallies: " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-set", type=Path, default=_SMOKE_SET_PATH,
                    help="path to smoke_set.json")
    ap.add_argument("--case", type=str, default=None,
                    help="run only this case_id")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any REGRESS or ERROR (default: report only)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write machine-readable summary to JSON file")
    ap.add_argument("--verbose", action="store_true",
                    help="show headless_replay output per case")
    args = ap.parse_args()

    if not args.smoke_set.is_file():
        print(f"ERROR: smoke set not found at {args.smoke_set}", file=sys.stderr)
        return 2
    spec = json.loads(args.smoke_set.read_text())
    cases = spec.get("cases", [])
    if args.case:
        cases = [c for c in cases if c.get("case_id") == args.case]
        if not cases:
            print(f"ERROR: no case matching {args.case}", file=sys.stderr)
            return 2

    print(f"smoke set: {len(cases)} case(s) from {args.smoke_set}")
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] running {case['case_id']} ({case.get('category', '?')})...",
              file=sys.stderr, flush=True)
        results.append(run_case(case, verbose=args.verbose))

    print_table(results)

    if args.json:
        args.json.write_text(json.dumps(
            [r.__dict__ for r in results], indent=2, default=str
        ))
        print(f"\nJSON summary written to {args.json}")

    if args.strict:
        bad = sum(1 for r in results if r.status in ("REGRESS", "ERROR"))
        if bad:
            print(f"\nFAIL: {bad} case(s) regressed or errored.")
            return 1
        print("\nPASS: no regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
