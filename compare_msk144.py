#!/home/jeff/ham/map144/.venv/bin/python3
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
"""Compare known MSK144 test signals with map144 decode results.

Reads:
  - the JSON manifest written by generate_msk144.py  (signal ground truth)
  - decodes.jsonl  (successful decodes, SPD or jt9)
  - launches.jsonl (every decode attempt, including no_decode outcomes)

Each manifest ping is classified as:
  DECODED     — correct message found in decodes.jsonl
  NO_DECODE   — a launch was found at the right time/freq but decode failed
  NOT DETECTED — no matching launch in the correct channel at all

The launches.jsonl outcomes (decoded / no_decode / timeout / error) are read
directly, so the summary counts are accurate rather than inferred.

Usage
-----
  compare_msk144.py [--manifest FILE] [--log FILE] [--launches FILE]
                    [--freq-tol-khz F] [--time-tol-s T]
                    [--since ISO8601] [--clear-log]

  --manifest      Manifest JSON (default: msk144_combined_iq_48k.json)
  --log           decodes.jsonl  (default: MSK144/detections/decodes.jsonl)
  --launches      launches.jsonl (default: MSK144/detections/launches.jsonl)
  --freq-tol-khz  Frequency tolerance for launch matching (default: 1.5 kHz)
  --time-tol-s    Time tolerance for launch matching (default: 1.5 s)
  --band-center-khz  IQ stream centre frequency in kHz (default: 50260.0)
  --since         Ignore log entries before this ISO-8601 UTC timestamp
  --clear-log     Delete both log files after reading
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── JSONL loaders ─────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, since: datetime | None = None) -> list[dict]:
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
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    if entry_dt < since:
                        continue
                except ValueError:
                    pass
            entries.append(entry)
    return entries


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Launch matching ────────────────────────────────────────────────────────────

def _find_launch(placement: dict, launches: list[dict],
                 band_center: float | None,
                 freq_tol_khz: float, time_tol_s: float) -> dict | None:
    """Return the closest launch entry for this manifest ping, or None.

    Matches on time (always) and frequency (when band_center is known).
    The closest match within tolerances is returned.
    """
    center_hz = placement.get("center_hz", 0.0)
    delay_s   = placement.get("delay_s", 0.0)

    expected_radio_khz = (
        band_center + (center_hz - 1500.0) / 1000.0
        if band_center is not None else None
    )

    best      = None
    best_dist = float("inf")

    for launch in launches:
        l_t = launch.get("t_sec", float("inf"))
        # A detection cannot fire more than 300 ms before the ping starts —
        # if the launch timestamp is well before delay_s it belongs to a
        # different ping or is a ghost from an earlier signal.
        if l_t < delay_s - 0.3:
            continue
        dt = abs(l_t - delay_s)
        if dt > time_tol_s:
            continue

        if expected_radio_khz is not None:
            df = abs(launch.get("radio_khz", float("inf")) - expected_radio_khz)
            if df > freq_tol_khz:
                continue
            dist = (dt / time_tol_s) ** 2 + (df / freq_tol_khz) ** 2
        else:
            dist = dt / time_tol_s

        if dist < best_dist:
            best_dist = dist
            best = launch

    return best


# ── Decoder label ─────────────────────────────────────────────────────────────

def _decoder_label(jt9_line: str) -> str:
    if jt9_line and jt9_line.startswith("[SPD"):
        return "SPD"
    return "jt9"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated MSK144 signals with map144 decode results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest",    default="msk144_combined_iq_48k.json")
    parser.add_argument("--log",         default="MSK144/detections/decodes.jsonl")
    parser.add_argument("--launches",    default="MSK144/detections/launches.jsonl")
    parser.add_argument("--freq-tol-khz", type=float, default=1.5)
    parser.add_argument("--time-tol-s",   type=float, default=1.5)
    parser.add_argument("--band-center-khz", type=float, default=50260.0,
                        help="IQ stream centre frequency in kHz (default: 50260.0)")
    parser.add_argument("--since",        default=None)
    parser.add_argument("--clear-log",    action="store_true")
    args = parser.parse_args()

    manifest_path  = Path(args.manifest)
    log_path       = Path(args.log)
    launches_path  = Path(args.launches)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest   = _load_manifest(manifest_path)
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

    decodes     = _load_jsonl(log_path,      since=since_dt)
    launches    = _load_jsonl(launches_path, since=since_dt)
    band_center = args.band_center_khz

    # ── Launch summary from launches.jsonl ────────────────────────────────────
    n_launches  = len(launches)
    n_l_decoded = sum(1 for l in launches if l.get("outcome") == "decoded")
    n_l_nodec   = sum(1 for l in launches if l.get("outcome") == "no_decode")
    n_l_timeout = sum(1 for l in launches if l.get("outcome") == "timeout")
    n_l_error   = sum(1 for l in launches if l.get("outcome") == "error")
    n_spd       = sum(1 for d in decodes if _decoder_label(d.get("jt9_line","")) == "SPD")
    n_jt9       = sum(1 for d in decodes if _decoder_label(d.get("jt9_line","")) == "jt9")

    # ── Build message → decode lookup ─────────────────────────────────────────
    decode_by_msg: dict[str, list[dict]] = {}
    for dec in decodes:
        raw = dec.get("message", "").strip()
        decode_by_msg.setdefault(raw, []).append(dec)
        for tok in raw.split():
            if tok != raw:
                decode_by_msg.setdefault(tok, []).append(dec)

    # ── Header ────────────────────────────────────────────────────────────────
    wav_name = Path(manifest.get("output_wav", "?")).name
    print()
    print(f"MSK144 Comparison Report")
    print(f"Simulation : {wav_name}")
    print(f"Manifest   : {manifest_path.name}")
    print(f"Generated  : {manifest.get('generated_at', 'unknown')}")
    if since_dt:
        print(f"Run start  : {since_dt.isoformat()}")
    if "noise_floor_dbfs" in manifest:
        print(f"Noise floor: {manifest['noise_floor_dbfs']} dBFS"
              f"   Atten: {manifest.get('atten_db', '?')} dB")
    print(f"Decode log : {log_path}  ({len(decodes)} entries"
          + (f" since {args.since}" if args.since else "") + ")")
    if launches_path.exists():
        print(f"Launches   : {launches_path}  ({n_launches} entries)")
    print(f"Launches   : {n_launches} total"
          f"  ({n_l_decoded} decoded"
          f"  {n_l_nodec} no_decode"
          f"  {n_l_timeout} timeout"
          + (f"  {n_l_error} error" if n_l_error else "")
          + ")")
    print(f"Decoders   : SPD={n_spd}  jt9={n_jt9}")
    print(f"Band center: {band_center:.1f} kHz")
    print()

    # ── Per-signal table (sorted by delay_s) ──────────────────────────────────
    col_w   = 13
    sep     = "─" * 86
    hdr_fmt = (f"{'#':>3}  {'Message':<{col_w}}  {'t (s)':>6}  {'fc (kHz)':>9}"
               f"  {'SNR':>5}  {'width':>7}  Status")
    print(hdr_fmt)
    print(sep)

    n_decoded    = 0
    n_no_decode  = 0
    n_not_det    = 0
    n_garbled    = 0
    matched_launch_ids: set[int] = set()
    matched_decode_ids: set[int] = set()

    for i, pl in enumerate(sorted(placements, key=lambda p: p.get("delay_s", 0))):
        msg       = pl.get("msg", "")
        center_hz = pl.get("center_hz", 0.0)
        delay_s   = pl.get("delay_s", 0.0)
        snr_db    = pl.get("snr_db", "?")
        width_ms  = pl.get("width_ms", None)

        snr_str   = f"{snr_db:+d} dB" if isinstance(snr_db, (int, float)) else str(snr_db)
        wid_str   = f"{width_ms} ms"  if width_ms is not None else "  ?"
        fc_str    = f"{center_hz/1000.0:+.2f}"

        prefix = (f"{i+1:>3}  {msg:<{col_w}}  {delay_s:>6.2f}  {fc_str:>9}"
                  f"  {snr_str:>5}  {wid_str:>7}")

        # 1. Did this message decode correctly?
        hits = decode_by_msg.get(msg, [])
        if hits:
            dec       = hits[0]
            matched_decode_ids.add(id(dec))
            d_t       = dec.get("t_sec", float("nan"))
            d_khz     = dec.get("radio_khz", float("nan"))
            decoder   = _decoder_label(dec.get("jt9_line", ""))
            # Mark this ping's launch as used so it won't be mistaken for
            # a no_decode match when processing a nearby ping later.
            own_launch = _find_launch(pl, launches, band_center,
                                      args.freq_tol_khz, args.time_tol_s)
            if own_launch is not None:
                matched_launch_ids.add(id(own_launch))
            print(f"{prefix}  DECODED   "
                  f"(t={d_t:.2f}s  radio={d_khz:.0f} kHz  [{decoder}])")
            n_decoded += 1
            continue

        # 2. Was there a launch for this ping (decode failed)?
        launch = _find_launch(pl, launches, band_center,
                              args.freq_tol_khz, args.time_tol_s)
        if launch is not None and id(launch) not in matched_launch_ids:
            matched_launch_ids.add(id(launch))
            l_t   = launch.get("t_sec", float("nan"))
            l_khz = launch.get("radio_khz", float("nan"))
            print(f"{prefix}  NO_DECODE "
                  f"(t={l_t:.2f}s  radio={l_khz:.0f} kHz)")
            n_no_decode += 1
            continue

        # 3. Never detected.
        print(f"{prefix}  NOT DETECTED")
        n_not_det += 1

    # ── False alarms (decodes with no manifest match) ─────────────────────────
    false_alarms = [d for d in decodes if id(d) not in matched_decode_ids]

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(placements)
    pct   = 100 * n_decoded / total if total else 0

    print()
    print(sep)
    print(f"Total signals  : {total}")
    print(f"Decoded        : {n_decoded:>3}  ({pct:.0f}%)"
          f"  — SPD={n_spd}  jt9={n_jt9}")
    if n_no_decode:
        print(f"No decode      : {n_no_decode:>3}"
              "  — launched in correct channel, both SPD and jt9 failed")
    if n_not_det:
        print(f"Not detected   : {n_not_det:>3}"
              "  — no launch in the correct channel")
    if n_garbled:
        print(f"Garbled        : {n_garbled:>3}  — message corrupted")
    if false_alarms:
        print(f"False alarms   : {len(false_alarms):>3}"
              "  — decoded message has no matching signal")
        for dec in false_alarms:
            print(f"    t={dec.get('t_sec','?'):.2f}s"
                  f"  radio={dec.get('radio_khz','?')} kHz"
                  f"  '{dec.get('message','')}'")
    print(f"Launches total : {n_launches}"
          f"  ({n_l_decoded} decoded"
          f"  {n_l_nodec} no_decode"
          f"  {n_l_timeout} timeout"
          + (f"  {n_l_error} error" if n_l_error else "")
          + ")")
    print()

    if args.clear_log:
        for p in (log_path, launches_path):
            if p.exists():
                p.unlink()
                print(f"Cleared: {p}")

    return 0 if (n_not_det == 0 and n_no_decode == 0 and n_garbled == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
