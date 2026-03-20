#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Compare known MSK144 test signals with jt9 decode results.

This script reads the JSON manifest written by ``generate_msk144_test_signal.py``
and the JSONL decode log written by ``radio_iq_gui/detection.py``, then produces
a signal-by-signal comparison showing which signals were decoded, which were
missed, and whether any false decodes occurred.

Manifest format (``msk144_combined_iq_48k.json``)
--------------------------------------------------
Written by ``generate_msk144_test_signal.py`` alongside the output WAV file.
Contains a list of ``placements``, each with:

  ``msg``        — 13-character self-describing MSK144 free-text message
  ``center_hz``  — signal centre frequency in the 48 kHz IQ passband (Hz)
  ``delay_s``    — signal start time in the output file (seconds)
  ``freq_khz``   — decoded: frequency offset from band centre (kHz)
  ``time_ds``    — decoded: start time in deciseconds
  ``snr_db``     — decoded: per-ping SNR relative to reference level (dB)
  ``width_ms``   — decoded: ping envelope width (ms)

Decode log format (``MSK144/detections/decodes.jsonl``)
-------------------------------------------------------
Appended by ``radio_iq_gui/detection.py`` for every successful jt9 decode.
Each JSON line contains:

  ``timestamp``  — ISO-8601 UTC decode time
  ``t_sec``      — position within the 15-second display window (seconds)
  ``fc_khz``     — detected carrier frequency (kHz)
  ``message``    — decoded message string from jt9

Matching logic
--------------
A decode is considered a *hit* if its ``message`` exactly matches one of the
manifest ``msg`` values.  Message-exact matching is robust because the 13-char
self-describing format encodes the signal parameters; a decode that recovered
the right string proves the full pipeline succeeded.

When ``--freq-tol-khz`` and ``--time-tol-s`` are provided, unmatched decodes
are checked against the manifest by position (frequency and time) to detect
partial decodes where jt9 recovered a garbled version of the message.

Output
------
  * Per-signal table: DECODED / MISSED / (GARBLED) with decoded message and
    time/frequency match quality.
  * Summary line: N decoded, M missed, K false alarms (decodes with no
    corresponding manifest entry).
  * Exit code 0 if all signals decoded, 1 otherwise (for use in scripts).

Usage
-----
  compare_msk144.py [--manifest FILE] [--log FILE]
                    [--freq-tol-khz F] [--time-tol-s T]
                    [--since ISO8601]

  --manifest    Path to the manifest JSON (default: msk144_combined_iq_48k.json)
  --log         Path to the decode JSONL log (default: MSK144/detections/decodes.jsonl)
  --freq-tol-khz  Frequency tolerance for positional matching (default: 2.0 kHz)
  --time-tol-s    Time tolerance for positional matching (default: 1.5 s)
  --since       Ignore decode log entries before this ISO-8601 UTC timestamp
                (useful when the log accumulates across multiple runs)
  --clear-log   Delete the decode log after reading (resets for the next run)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_decodes(path: Path, since: datetime | None = None) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                ts = entry.get("timestamp", "")
                try:
                    entry_dt = datetime.fromisoformat(ts)
                    if entry_dt < since:
                        continue
                except ValueError:
                    pass
            entries.append(entry)
    return entries


def _match_by_position(placement: dict, decodes: list[dict],
                       freq_tol_khz: float, time_tol_s: float) -> dict | None:
    """Return the closest positional-match decode, or None."""
    center_khz = placement["center_hz"] / 1000.0
    delay_s = placement["delay_s"]
    best = None
    best_dist = float("inf")
    for dec in decodes:
        df = abs(dec.get("fc_khz", float("inf")) - center_khz)
        dt = abs(dec.get("t_sec", float("inf")) - delay_s)
        if df <= freq_tol_khz and dt <= time_tol_s:
            dist = (df / freq_tol_khz) ** 2 + (dt / time_tol_s) ** 2
            if dist < best_dist:
                best_dist = dist
                best = dec
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated MSK144 signals with jt9 decode results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default="msk144_combined_iq_48k.json",
        help="Manifest JSON from generate_msk144_test_signal.py "
             "(default: msk144_combined_iq_48k.json)",
    )
    parser.add_argument(
        "--log",
        default="MSK144/detections/decodes.jsonl",
        help="Decode log JSONL from radio_iq_gui/detection.py "
             "(default: MSK144/detections/decodes.jsonl)",
    )
    parser.add_argument(
        "--freq-tol-khz",
        type=float,
        default=2.0,
        help="Frequency tolerance for positional near-miss detection (default: 2.0 kHz)",
    )
    parser.add_argument(
        "--time-tol-s",
        type=float,
        default=1.5,
        help="Time tolerance for positional near-miss detection (default: 1.5 s)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Ignore decodes before this ISO-8601 UTC timestamp "
             "(e.g. 2026-03-19T12:00:00Z)",
    )
    parser.add_argument(
        "--clear-log",
        action="store_true",
        help="Delete the decode log after reading (resets for the next test run)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    log_path = Path(args.log)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = _load_manifest(manifest_path)
    placements = manifest.get("placements", [])

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: cannot parse --since '{args.since}'", file=sys.stderr)
            return 2

    decodes = _load_decodes(log_path, since=since_dt)

    # Build lookup: message → list of decode entries.
    # jt9 output lines contain fields before the message text
    # (e.g. "2200  -3  0.2  1500  ~  AP05075N07150"), so index by
    # every whitespace-delimited token as well as the full stripped line
    # so that manifest messages (just the 13-char text) can be found.
    decode_by_msg: dict[str, list[dict]] = {}
    for dec in decodes:
        raw = dec.get("message", "").strip()
        # Index under the full line
        decode_by_msg.setdefault(raw, []).append(dec)
        # Also index under each individual token (picks up the bare message)
        for token in raw.split():
            if token != raw:
                decode_by_msg.setdefault(token, []).append(dec)

    # ── Per-signal comparison ────────────────────────────────────────────────
    print()
    print(f"Manifest : {manifest_path}")
    print(f"Generated: {manifest.get('generated_at', 'unknown')}")
    print(f"WAV      : {manifest.get('output_wav', 'unknown')}")
    if 'noise_floor_dbfs' in manifest:
        print(f"Noise flr: {manifest['noise_floor_dbfs']} dBFS   "
              f"Atten: {manifest.get('atten_db', '?')} dB")
    else:
        print(f"AWGN SNR : {manifest.get('awgn_snr_db', '?')} dB   "
              f"Atten: {manifest.get('atten_db', '?')} dB")
    print(f"Decode log: {log_path}  ({len(decodes)} entries"
          + (f" since {args.since}" if args.since else "") + ")")
    print()

    col_w = 13
    print(f"{'#':>3}  {'Message':<{col_w}}  {'fc (kHz)':>8}  {'t (s)':>6}  "
          f"{'SNR (dB)':>8}  Status")
    print("-" * 72)

    n_decoded = 0
    n_missed = 0
    n_garbled = 0
    matched_decode_ids: set[int] = set()

    for i, pl in enumerate(placements):
        msg = pl.get("msg", "")
        center_khz = pl.get("center_hz", 0.0) / 1000.0
        delay_s = pl.get("delay_s", 0.0)
        snr_db = pl.get("snr_db", "?")

        prefix = f"{i+1:>3}  {msg:<{col_w}}  {center_khz:>8.2f}  {delay_s:>6.2f}  {str(snr_db):>8}"

        hits = decode_by_msg.get(msg, [])
        if hits:
            # Exact message match
            dec = hits[0]
            matched_decode_ids.add(id(dec))
            d_t = dec.get("t_sec", float("nan"))
            d_fc = dec.get("fc_khz", float("nan"))
            print(f"{prefix}  DECODED   "
                  f"(jt9: t={d_t:.2f}s  fc={d_fc:.2f} kHz)")
            n_decoded += 1
        else:
            # Check for positional near-miss (garbled decode)
            near = _match_by_position(pl, decodes, args.freq_tol_khz, args.time_tol_s)
            if near and id(near) not in matched_decode_ids:
                matched_decode_ids.add(id(near))
                near_msg = near.get("message", "")
                d_t = near.get("t_sec", float("nan"))
                d_fc = near.get("fc_khz", float("nan"))
                print(f"{prefix}  GARBLED   "
                      f"(jt9: '{near_msg}'  t={d_t:.2f}s  fc={d_fc:.2f} kHz)")
                n_garbled += 1
            else:
                print(f"{prefix}  MISSED")
                n_missed += 1

    # ── False alarms (decodes with no manifest match) ───────────────────────
    false_alarms = [dec for dec in decodes if id(dec) not in matched_decode_ids]

    print()
    print(f"{'─'*72}")
    total = len(placements)
    pct = 100 * n_decoded / total if total else 0
    print(f"Total signals  : {total}")
    print(f"Decoded        : {n_decoded:>3}  ({pct:.0f}%)")
    print(f"Missed         : {n_missed:>3}")
    if n_garbled:
        print(f"Garbled        : {n_garbled:>3}  (detected but message wrong)")
    if false_alarms:
        print(f"False alarms   : {len(false_alarms):>3}  (decoded with no matching signal)")
        for dec in false_alarms:
            print(f"    t={dec.get('t_sec','?'):.2f}s  "
                  f"fc={dec.get('fc_khz','?'):.2f} kHz  "
                  f"'{dec.get('message','')}'")
    print()

    if args.clear_log and log_path.exists():
        log_path.unlink()
        print(f"Cleared: {log_path}")

    return 0 if n_missed == 0 and n_garbled == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
