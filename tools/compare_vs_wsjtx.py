#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Use WSJT-X's decode log (ALL.TXT) as ground truth and cross-tab against
MAP144's sq_det launches/decodes and TFMF candidates.

──────────────────────────────────────────────────────────────────────────────
FREQUENCY CONVENTIONS (the source of one bug in the first draft of this tool)
──────────────────────────────────────────────────────────────────────────────

Three different frequencies show up in this comparison; they all describe the
same physical signal but in different reference frames:

  f_rf_hz       —  absolute RF carrier frequency of the signal in air, in Hz.
                   "On-air freq."  Independent of any receiver.  Reported in
                   this script as the common frame for cross-detector matching.

  f_call_off_hz —  offset from the MSK144 calling frequency (50.260 MHz on 6 m),
                   in Hz.  Useful because operators always use integer-kHz
                   calling freqs and Ftol ~ ±100-200 Hz.  =0 for a signal
                   exactly on the calling freq.

  f_audio_hz    —  in WSJT-X: the freq in WSJT-X's 12 kHz REAL-audio frame,
                   where the design centre is 1500 Hz.  ALL.TXT's "dial"
                   column is the QSO/carrier freq (= the displayed value in
                   WSJT-X), NOT the radio's actual USB dial — the operator
                   has set the radio 1500 Hz below the QSO freq so the signal
                   lands at audio = 1500 Hz.

                   in MAP144: ``freq_hz`` in tfmf_candidates.jsonl is the
                   audio freq offset relative to the engine's pan centre
                   (typically 50.259 MHz), NOT relative to the calling freq.

Conversions:

  WSJT-X ALL.TXT line:  "260528_095815  50.260  Rx  MSK144  3  5.2  1456  CQ ..."
                                          │                          │
                                          │ QSO/carrier freq (MHz)   │ audio_hz
                                          │ = dial_mhz (WSJT-X term) │ in 12 kHz frame
                                          │                          │ (1500 = QSO freq)

      f_rf_hz       = dial_mhz · 1e6 + (audio_hz − 1500)
      f_call_off_hz = (dial_mhz − calling_mhz) · 1e6 + (audio_hz − 1500)
                    = audio_hz − 1500   (when dial_mhz = calling_mhz, the common case)

  MAP144 sq_det launches.jsonl:
      f_rf_hz       = radio_khz · 1000 + sq_new_fc_offset_hz_h
      f_call_off_hz = (radio_khz − calling_khz) · 1000 + sq_new_fc_offset_hz_h

  MAP144 TFMF tfmf_candidates.jsonl:
      f_rf_hz       = pan_center_hz + freq_hz_in_log
      f_call_off_hz = freq_hz_in_log − (calling_hz − pan_center_hz)
                    = freq_hz_in_log − channel_offset_hz

Sanity check: a signal exactly on calling freq (50.260 MHz on 6 m, Flex pan
= 50.259):
  WSJT-X:        audio_hz = 1500   → f_call_off_hz = 0     ✓
  MAP144 sq_det: ch_signed = 0     → f_call_off_hz = 0     ✓
  MAP144 TFMF:   freq_hz = +1000   → f_call_off_hz = 0     ✓ (pan-center offset
                                                              is +1000 Hz for the
                                                              calling-freq channel)

This script matches by f_call_off_hz throughout — "offset from the calling
freq" is the most operator-meaningful frame and avoids carrying the pan
centre around in every comparison.

The "ground truth" framing is: WSJT-X decoded this message in this period; did
MAP144's sq_det also decode it?  Did sq_det at least launch on it?  Did TFMF
have a candidate near the (t_sec, audio_hz) location?

Usage:
    python tools/compare_vs_wsjtx.py
    python tools/compare_vs_wsjtx.py --window 2026-05-28_07:10 2026-05-28_13:54
    python tools/compare_vs_wsjtx.py --tol-time 2.0 --tol-freq-hz 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.detection_compare._filters import is_test_message  # noqa: E402


WSJTX_ALL_TXT = Path("/home/jeff/.local/share/WSJT-X - flex/ALL.TXT")
MAP144_DET    = REPO / "MSK144" / "detections"

PERIOD_S = 15.0
TS_FMT   = "%Y-%m-%d_%H:%M:%S.%f"
WX_TS_FMT = "%y%m%d_%H%M%S"


def parse_map144_ts(s: str) -> float | None:
    try:
        return datetime.strptime(s, TS_FMT).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d_%H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None


def parse_wsjtx_ts(s: str) -> float | None:
    try:
        return datetime.strptime(s, WX_TS_FMT).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


# ── Loaders ────────────────────────────────────────────────────────────────


_WX_LINE_RE = re.compile(
    r'^(?P<ts>\d{6}_\d{6})\s+'
    r'(?P<dial>\d+\.\d+)\s+'
    r'Rx\s+(?P<mode>\S+)\s+'
    r'(?P<snr>-?\d+)\s+'
    r'(?P<tsec>[-\d.]+)\s+'
    r'(?P<audio>\d+)\s+'
    r'(?P<msg>.+?)\s*$'
)


def load_wsjtx_decodes(path: Path, e_min: float, e_max: float,
                       calling_mhz: float) -> list[dict]:
    """Parse ALL.TXT and yield each decode as a dict with ``f_call_off_hz``
    = signed offset of the signal from the calling freq, in Hz."""
    out = []
    n_total = n_window = n_mode = n_test = 0
    with path.open(errors='replace') as f:
        for line in f:
            m = _WX_LINE_RE.match(line)
            if m is None:
                continue
            n_total += 1
            if m.group('mode') != 'MSK144':
                continue
            n_mode += 1
            ts = parse_wsjtx_ts(m.group('ts'))
            if ts is None or ts < e_min - PERIOD_S or ts > e_max + PERIOD_S:
                continue
            n_window += 1
            msg = m.group('msg').strip()
            if is_test_message(msg):
                n_test += 1
                continue
            dial_mhz = float(m.group('dial'))   # ALL.TXT "dial" = QSO/carrier freq
            wx_audio = float(m.group('audio'))
            # f_call_off_hz: signed offset from calling freq, in Hz.
            # See module docstring for derivation.
            f_call_off_hz = (dial_mhz - calling_mhz) * 1e6 + (wx_audio - 1500.0)
            out.append({
                'epoch':         ts,
                'period_idx':    int(ts // PERIOD_S),
                'qso_mhz':       dial_mhz,
                't_sec':         float(m.group('tsec')),
                'f_call_off_hz': f_call_off_hz,
                'wx_audio_hz':   wx_audio,
                'snr':           int(m.group('snr')),
                'message':       msg,
            })
    print(f"  ALL.TXT: {n_total} lines parsed, {n_mode} MSK144, "
          f"{n_window} in window, {n_test} test-message filtered",
          file=sys.stderr)
    return out


def load_map144_decodes(path: Path, e_min: float, e_max: float) -> set:
    out = set()
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = r.get('message') or ''
            if not msg or is_test_message(msg):
                continue
            ts = parse_map144_ts(r.get('timestamp', ''))
            if ts is None or ts < e_min - PERIOD_S or ts > e_max + PERIOD_S:
                continue
            out.add((int(ts // PERIOD_S), msg.strip()))
    return out


def load_map144_launches(path: Path, e_min: float, e_max: float,
                          calling_mhz: float) -> dict:
    """Yield each launch as a dict with ``f_call_off_hz`` = signed offset
    of the launch's reported tuning point from the calling freq, in Hz."""
    out = defaultdict(list)
    calling_khz = calling_mhz * 1000.0
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_map144_ts(r.get('timestamp', ''))
            if ts is None or ts < e_min - PERIOD_S or ts > e_max + PERIOD_S:
                continue
            radio_khz = r.get('radio_khz')
            if radio_khz is None:
                continue
            sub_hz = r.get('sq_new_fc_offset_hz_h') or 0
            # f_call_off_hz: signed offset from calling freq, in Hz.
            f_call_off_hz = (float(radio_khz) - calling_khz) * 1000.0 + float(sub_hz)
            out[int(ts // PERIOD_S)].append({
                't_sec':            float(r.get('t_sec', 0)),
                'f_call_off_hz':    f_call_off_hz,
                'outcome':          r.get('outcome', 'launched'),
                'pair_metric_db_h': r.get('pair_metric_db_h'),
                'msg':              r.get('message') or '',
            })
    return out


def load_tfmf_candidates(path: Path,
                          calling_mhz: float,
                          pan_mhz: float) -> tuple[dict, tuple[float, float]]:
    """Yield each TFMF candidate with ``f_call_off_hz`` = signed offset of
    the candidate's freq from the calling freq, in Hz.

    The log's ``freq_hz`` is relative to the pan centre; convert by
    subtracting ``channel_offset_hz = (calling - pan) · 1e6``.
    """
    out = defaultdict(list)
    e_min = float('inf'); e_max = -float('inf')
    channel_offset_hz = (calling_mhz - pan_mhz) * 1e6   # = +1000 Hz on the Flex
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = int(r['period_idx'])
            out[p].append({
                't_s':              float(r['t_s_in_period']),
                'f_call_off_hz':    float(r['freq_hz']) - channel_offset_hz,
                'snr_db':           float(r['snr_db']),
            })
            ts = parse_map144_ts(r['timestamp'])
            if ts:
                e_min = min(e_min, ts); e_max = max(e_max, ts)
    return out, (e_min, e_max)


# ── Cross-tab ─────────────────────────────────────────────────────────────


def has_nearby(target_t: float, target_f: float, items: list,
                tol_t: float, tol_f: float,
                t_key: str = 't_s', f_key: str = 'freq_hz') -> bool:
    for it in items:
        if (abs(it[t_key] - target_t) <= tol_t
                and abs(it[f_key] - target_f) <= tol_f):
            return True
    return False


def analyze(args) -> None:
    tfmf_path     = MAP144_DET / 'tfmf_candidates.jsonl'
    decodes_path  = MAP144_DET / 'decodes.jsonl'
    launches_path = MAP144_DET / 'launches.jsonl'

    print("Loading MAP144 TFMF candidates…", file=sys.stderr)
    tfmf, (e_min, e_max) = load_tfmf_candidates(
        tfmf_path, args.calling_mhz, args.pan_center_mhz)
    if args.start:
        e_min = datetime.strptime(args.start, "%Y-%m-%d_%H:%M").replace(
            tzinfo=timezone.utc).timestamp()
    if args.end:
        e_max = datetime.strptime(args.end, "%Y-%m-%d_%H:%M").replace(
            tzinfo=timezone.utc).timestamp()
    print(f"  window: {datetime.fromtimestamp(e_min, tz=timezone.utc):%Y-%m-%d %H:%M} → "
          f"{datetime.fromtimestamp(e_max, tz=timezone.utc):%H:%M} UTC", file=sys.stderr)

    print("Loading WSJT-X ALL.TXT…", file=sys.stderr)
    wx = load_wsjtx_decodes(args.wsjtx_log, e_min, e_max, args.calling_mhz)
    print(f"  {len(wx)} WSJT-X decodes in window after filters", file=sys.stderr)

    print("Loading MAP144 decodes.jsonl…", file=sys.stderr)
    map_decoded = load_map144_decodes(decodes_path, e_min, e_max)
    print(f"  {len(map_decoded)} MAP144 decoded (period, message) pairs", file=sys.stderr)

    print("Streaming MAP144 launches.jsonl…", file=sys.stderr)
    launches = load_map144_launches(launches_path, e_min, e_max, args.calling_mhz)
    print(f"  {sum(len(v) for v in launches.values())} launches in window",
          file=sys.stderr)

    # ── Cross-tab: for each WSJT-X decode, what did MAP144 do? ───────────
    sq_decoded_too = sq_launched_only = sq_silent = 0
    tfmf_saw = tfmf_missed = 0
    wsjtx_only_records = []   # WSJT-X decoded, MAP144 sq did NOT decode

    for w in wx:
        p = w['period_idx']
        same_in_map = (p, w['message']) in map_decoded
        if same_in_map:
            sq_decoded_too += 1
        else:
            # Did sq_det at least launch nearby?
            near_launch = has_nearby(
                w['t_sec'], w['f_call_off_hz'], launches.get(p, []),
                args.tol_time, args.tol_freq_hz,
                t_key='t_sec', f_key='f_call_off_hz')
            if near_launch:
                sq_launched_only += 1
            else:
                sq_silent += 1
            wsjtx_only_records.append((w, near_launch))

        # TFMF presence (independent of sq_det)
        near_tfmf = has_nearby(
            w['t_sec'], w['f_call_off_hz'], tfmf.get(p, []),
            args.tol_time, args.tol_freq_hz,
            t_key='t_s', f_key='f_call_off_hz')
        if near_tfmf:
            tfmf_saw += 1
        else:
            tfmf_missed += 1

    # ── MAP144-only decodes (decoded by sq_det but not WSJT-X) ───────────
    wx_set = {(w['period_idx'], w['message']) for w in wx}
    map_only = [pm for pm in map_decoded if pm not in wx_set]

    # ── Report ───────────────────────────────────────────────────────────
    n_wx = len(wx)
    print()
    print("═" * 72)
    print("MAP144 (sq_det + TFMF) vs WSJT-X as ground truth")
    print("═" * 72)
    print(f"Window: {datetime.fromtimestamp(e_min, tz=timezone.utc):%Y-%m-%d %H:%M} → "
          f"{datetime.fromtimestamp(e_max, tz=timezone.utc):%H:%M} UTC "
          f"({(e_max - e_min) / 3600:.1f} h)")
    print(f"Match tolerance:  ±{args.tol_time:.1f} s, ±{args.tol_freq_hz:.0f} Hz")
    print()
    print(f"WSJT-X decoded:                  {n_wx:>5}")
    print(f"  MAP144 sq_det decoded same:    {sq_decoded_too:>5}   "
          f"({100*sq_decoded_too/max(n_wx,1):.1f}%)")
    print(f"  MAP144 sq_det launched only:   {sq_launched_only:>5}   "
          f"(sq fired but jt9/SPD didn't decode)")
    print(f"  MAP144 sq_det silent:          {sq_silent:>5}   "
          f"(no sq_det event in (t_sec, audio_hz) box)")
    print()
    print(f"  MAP144 TFMF saw signal:        {tfmf_saw:>5}   "
          f"({100*tfmf_saw/max(n_wx,1):.1f}%)")
    print(f"  MAP144 TFMF missed signal:     {tfmf_missed:>5}   "
          f"({100*tfmf_missed/max(n_wx,1):.1f}%)")
    print()
    print(f"MAP144 decoded but WSJT-X didn't: {len(map_only):>5}   "
          f"(jt9-CRC collisions or different period/message)")
    print()

    # ── Per-hour ──────────────────────────────────────────────────────────
    h_wx = Counter()
    h_map = Counter()
    h_tfmf = Counter()
    for w in wx:
        h = datetime.fromtimestamp(w['epoch'], tz=timezone.utc).strftime("%H")
        h_wx[h] += 1
        if (w['period_idx'], w['message']) in map_decoded:
            h_map[h] += 1
        if has_nearby(w['t_sec'], w['f_call_off_hz'], tfmf.get(w['period_idx'], []),
                      args.tol_time, args.tol_freq_hz,
                      t_key='t_s', f_key='f_call_off_hz'):
            h_tfmf[h] += 1
    print("  Per-hour (UTC):")
    print(f'    {"hour":<6}{"WSJT-X":>9}{"MAP sq dec":>12}{"TFMF saw":>11}{"sq miss-rate":>14}')
    for h in sorted(h_wx):
        nw = h_wx[h]; nm = h_map[h]; nt = h_tfmf[h]
        print(f'    {h+":00":<6}{nw:>9}{nm:>12}{nt:>11}'
              f'{100*(nw-nm)/max(nw,1):>13.1f}%')
    print()

    # ── WSJT-X-only sample (most interesting: MAP144 misses) ──────────────
    print(f"  WSJT-X-only decodes (MAP144 sq_det did NOT decode): {len(wsjtx_only_records)}")
    if wsjtx_only_records:
        # Sample a few representative.  f_call_off_hz = Hz offset from calling freq;
        # 0 = exactly on the calling carrier.
        print(f'    {"timestamp":>16}  {"snr":>3}  {"t":>5}  {"f_call_off":>10}  '
              f'sq fired?  tfmf?     message')
        sorted_wx = sorted(wsjtx_only_records, key=lambda x: -x[0]['snr'])
        for w, near_launch in sorted_wx[:20]:
            tfmf_yes = has_nearby(
                w['t_sec'], w['f_call_off_hz'], tfmf.get(w['period_idx'], []),
                args.tol_time, args.tol_freq_hz,
                t_key='t_s', f_key='f_call_off_hz')
            ts_h = datetime.fromtimestamp(w['epoch'], tz=timezone.utc).strftime("%H:%M:%S")
            print(f'    {ts_h:>16}  {w["snr"]:>+3}  {w["t_sec"]:>5.1f}  '
                  f'{w["f_call_off_hz"]:>+9.0f}Hz  '
                  f'{("sq=Y" if near_launch else "sq=N"):>9}  '
                  f'{"tfmf=Y" if tfmf_yes else "tfmf=N":>6}    {w["message"]}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsjtx-log', type=Path, default=WSJTX_ALL_TXT,
                    help=f'Path to WSJT-X ALL.TXT (default {WSJTX_ALL_TXT})')
    ap.add_argument('--calling-mhz', type=float, default=50.260,
                    help='MSK144 calling-freq in MHz (6 m default 50.260; '
                         'all matching is done in offset-from-calling Hz)')
    ap.add_argument('--pan-center-mhz', type=float, default=50.259,
                    help='Engine pan-centre in MHz (only needed to translate '
                         'TFMF log\'s pan-centre-relative freq_hz to '
                         'calling-relative offset; default 50.259)')
    ap.add_argument('--tol-time', type=float, default=2.0,
                    help='Time-match tolerance, seconds (default 2.0)')
    ap.add_argument('--tol-freq-hz', type=float, default=2000.0,
                    help='Freq-match tolerance, Hz (default 2000; spans the '
                         '±1 kHz channel quantisation plus per-ping drift)')
    ap.add_argument('--start', type=str, default=None,
                    help='Override window start (YYYY-MM-DD_HH:MM UTC)')
    ap.add_argument('--end', type=str, default=None,
                    help='Override window end (YYYY-MM-DD_HH:MM UTC)')
    args = ap.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
