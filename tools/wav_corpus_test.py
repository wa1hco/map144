"""WAV-corpus decoder regression test.

Goal
----
Eliminate the overnight-wait debugging cycle for decoder-side changes.

Given a corpus of WSJT-X-saved WAV files plus the corresponding ALL.TXT
decode log, this tool runs MAP144's jt9 invocation against each WAV
and reports coverage vs the WSJT-X reference labels.  Any decoder
parameter change (FTOL, depth, etc.) can be validated in a few
minutes instead of waiting for the next MS opening.

Workflow
--------
1. Pick a WSJT-X save directory (one with both WAVs and ALL.TXT).
2. Run::

    python tools/wav_corpus_test.py --date 20260511

   Reports:
       - Coverage = MAP144 decoded / WSJT-X decoded
       - MAP144-only catches
       - WSJT-X-only catches

3. Sweep parameters with ``--ftol`` and ``--depth`` to A/B the
   change you just made.

Caveats
-------
- Tests only the jt9 invocation; SPD path is not exercised.  For
  decoder-side regression that's the right scope (SPD has its own
  freq search via ``ntol``, separate from jt9's ``-F``).
- Only periods where WSJT-X saved a WAV are tested.  When WSJT-X
  is in "save decoded" mode, this skews toward harder cases.
- Audio in WSJT-X WAVs is post-AF mixed (centred at audio 1500 Hz),
  so the channelizer/detector path is not exercised here.  Detector
  changes need separate validation (synthetic or IQ replay).

Output is tab-separated; pipe through ``column -t`` for readability.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Defaults match map144_app/detection.py current values ────────────────────
DEFAULT_FTOL  = 200        # post-2026-05-11 widening
DEFAULT_DEPTH = 1          # post-2026-05-11 phantom-suppression change
DEFAULT_AF_CENTRE = 1500


# ── WSJT-X ALL.TXT parser ────────────────────────────────────────────────────
_ALL_RE = re.compile(
    r'^(?P<ts>\d{6}_\d{6})\s+'
    r'(?P<mhz>[\d.]+)\s+Rx\s+(?P<mode>MSK144)\s+'
    r'(?P<snr>[-+]?\d+)\s+(?P<dt>[-+]?[\d.]+)\s+(?P<af>\d+)\s+'
    r'(?P<msg>.+)$'
)


def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime("20" + ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def _period_start(dt: datetime) -> datetime:
    """Floor to 15-s boundary (WSJT-X / post-2026-05-11 MAP144 convention)."""
    epoch = (dt - datetime(2026, 1, 1, tzinfo=timezone.utc)).total_seconds()
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(epoch // 15) * 15)


def parse_all_txt(path: Path, date_filter: str | None = None) -> dict[datetime, list[dict]]:
    """Return {period_start_ts: [decode_dict, ...]} for MSK144 lines on the given date.

    Each decode_dict contains: snr, dt, af, msg, raw_ts.
    """
    period_map: dict[datetime, list[dict]] = {}
    with path.open(errors='replace') as fh:
        for ln in fh:
            m = _ALL_RE.match(ln.strip())
            if not m:
                continue
            ts = _parse_ts(m['ts'])
            if date_filter and not m['ts'].startswith(date_filter[2:]):
                # date_filter is YYYYMMDD, ts_str is YYMMDD_HHMMSS — strip century
                continue
            # WSJT-X MSK144 ALL.TXT convention: the timestamp is when jt9
            # ran (often mid-period, NOT a 15-s boundary), and ``dt`` is
            # the offset from the START of the analysis window — which is
            # floor(timestamp/15s), not the timestamp itself.  So the
            # signal time is window_start + dt, not timestamp + dt.
            # Without this floor-first step, entries with non-boundary
            # timestamps (e.g. 092914 dt=14.0) get attributed to the
            # following period (09:29:15) instead of the correct period
            # (09:29:00).  Diagnosed 2026-05-11 against WAV 260511_092900.
            window_start = _period_start(ts)
            period = _period_start(window_start + timedelta(seconds=float(m['dt'])))
            period_map.setdefault(period, []).append({
                'snr':    int(m['snr']),
                'dt':     float(m['dt']),
                'af':     int(m['af']),
                'msg':    m['msg'].strip(),
                'raw_ts': m['ts'],
            })
    return period_map


def normalise_msg(msg: str) -> str:
    return ' '.join(msg.upper().split())


# ── jt9 driver ────────────────────────────────────────────────────────────────
_JT9_LINE_RE = re.compile(
    r'^\s*\d{6}\s+(?P<snr>[-+]?\d+)\s+(?P<dt>[-+]?[\d.]+)\s+(?P<af>\d+)\s+\S+\s+(?P<msg>.+?)\s*$'
)


def run_jt9(wav_path: Path, ftol: int, depth: int, af_centre: int = DEFAULT_AF_CENTRE,
            timeout_s: float = 30.0) -> list[dict]:
    """Run jt9 on a WAV file with the given parameters; return list of decodes."""
    cmd = [
        'jt9', '--msk144',
        '-L', str(af_centre - ftol),
        '-H', str(af_centre + ftol),
        '-F', str(ftol),
        '-d', str(depth),
        str(wav_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return [{'error': 'timeout'}]
    decodes = []
    for ln in out.stdout.splitlines():
        m = _JT9_LINE_RE.match(ln)
        if m:
            decodes.append({
                'snr': int(m['snr']),
                'dt':  float(m['dt']),
                'af':  int(m['af']),
                'msg': normalise_msg(m['msg']),
            })
    return decodes


# ── WAV → period_ts ─────────────────────────────────────────────────────────
_WAV_NAME_RE = re.compile(r'^(\d{6})_(\d{6})\.wav$')


def wav_period_ts(wav_path: Path) -> datetime | None:
    """Convert WAV filename (yymmdd_HHMMSS.wav) to a period_start datetime."""
    m = _WAV_NAME_RE.match(wav_path.name)
    if not m:
        return None
    try:
        return _parse_ts(f"{m.group(1)}_{m.group(2)}")
    except ValueError:
        return None


# ── Worker (top-level so it can be pickled by ProcessPoolExecutor) ────────────
def _process_one(wav_path: Path, ftol: int, depth: int) -> tuple[Path, list[dict]]:
    return wav_path, run_jt9(wav_path, ftol=ftol, depth=depth)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsjtx-dir', type=Path,
                    default=Path("/home/jeff/.local/share/WSJT-X - flex"),
                    help='WSJT-X user directory containing save/ and ALL.TXT')
    ap.add_argument('--date', default='20260511',
                    help='UTC date YYYYMMDD to test (default 20260511)')
    ap.add_argument('--ftol', type=int, default=DEFAULT_FTOL,
                    help=f'jt9 -F half-width Hz (default {DEFAULT_FTOL})')
    ap.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                    help=f'jt9 -d decode depth (default {DEFAULT_DEPTH})')
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel jt9 invocations (default 4)')
    ap.add_argument('--max-wavs', type=int, default=None,
                    help='Cap on number of WAVs processed (for quick tests)')
    ap.add_argument('--show-diffs', action='store_true',
                    help='Print per-WAV diff rows where MAP144 and WSJT-X disagree')
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt  = args.wsjtx_dir / 'ALL.TXT'
    if not save_dir.is_dir():
        print(f"error: save dir not found: {save_dir}", file=sys.stderr)
        return 1
    if not all_txt.is_file():
        print(f"error: ALL.TXT not found: {all_txt}", file=sys.stderr)
        return 1

    # ── Find WAVs for the requested date ─────────────────────────────────────
    date_prefix = args.date[2:]   # YYMMDD
    wavs = sorted(p for p in save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]
    if not wavs:
        print(f"error: no WAVs for date {args.date} in {save_dir}", file=sys.stderr)
        return 1

    print(f"WAV corpus regression test", file=sys.stderr)
    print(f"  source : {args.wsjtx_dir}", file=sys.stderr)
    print(f"  date   : {args.date}", file=sys.stderr)
    print(f"  wavs   : {len(wavs)}", file=sys.stderr)
    print(f"  jt9    : -F {args.ftol} -L {DEFAULT_AF_CENTRE-args.ftol} "
          f"-H {DEFAULT_AF_CENTRE+args.ftol} -d {args.depth}", file=sys.stderr)
    print(file=sys.stderr)

    # ── Parse WSJT-X reference ──────────────────────────────────────────────
    period_map = parse_all_txt(all_txt, date_filter=args.date)
    print(f"WSJT-X reference: {sum(len(v) for v in period_map.values())} raw decode lines "
          f"across {len(period_map)} periods", file=sys.stderr)

    # ── Run jt9 in parallel ──────────────────────────────────────────────────
    print(f"Running jt9 on {len(wavs)} WAVs with {args.workers} workers…", file=sys.stderr)
    results: dict[Path, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, w, args.ftol, args.depth) for w in wavs]
        for i, fut in enumerate(as_completed(futs), 1):
            wav_path, decodes = fut.result()
            results[wav_path] = decodes
            if i % 25 == 0 or i == len(wavs):
                print(f"  [{i}/{len(wavs)}]", file=sys.stderr, flush=True)

    # ── Categorise per-WAV ──────────────────────────────────────────────────
    cats = Counter()
    diffs = []
    for wav in wavs:
        period = wav_period_ts(wav)
        if period is None:
            cats['BAD_NAME'] += 1
            continue
        wsjtx_msgs = {normalise_msg(d['msg']) for d in period_map.get(period, [])}
        map_msgs   = {d['msg'] for d in results.get(wav, []) if 'error' not in d}

        if not wsjtx_msgs and not map_msgs:
            cats['NEITHER'] += 1
        elif wsjtx_msgs == map_msgs:
            cats['MATCH'] += 1
        else:
            overlap   = wsjtx_msgs & map_msgs
            wsjtx_only = wsjtx_msgs - map_msgs
            map_only   = map_msgs   - wsjtx_msgs
            if overlap and not wsjtx_only and not map_only:
                cats['MATCH'] += 1   # set equality should have caught this; redundant guard
            elif overlap:
                cats['PARTIAL'] += 1
            elif wsjtx_only and not map_only:
                cats['WSJTX_ONLY'] += 1
            elif map_only and not wsjtx_only:
                cats['MAP_ONLY'] += 1
            else:
                cats['BOTH_DIFFER'] += 1
            diffs.append((wav.name, wsjtx_msgs, map_msgs))

    # ── Per-message-level stats (across all WAVs, not per-WAV) ──────────────
    total_wsjtx_msgs = sum(len({normalise_msg(d['msg']) for d in period_map.get(wav_period_ts(w), [])})
                           for w in wavs if wav_period_ts(w))
    total_map_msgs   = sum(len({d['msg'] for d in results.get(w, []) if 'error' not in d}) for w in wavs)
    both = 0; w_only = 0; m_only = 0
    for wav in wavs:
        period = wav_period_ts(wav)
        if period is None: continue
        ws = {normalise_msg(d['msg']) for d in period_map.get(period, [])}
        ms = {d['msg'] for d in results.get(wav, []) if 'error' not in d}
        both   += len(ws & ms)
        w_only += len(ws - ms)
        m_only += len(ms - ws)

    # ── Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"PER-WAV CATEGORIES (n={len(wavs)})")
    print("=" * 70)
    for c, n in cats.most_common():
        print(f"  {c:>14}: {n:>5}  ({100*n/len(wavs):.1f}%)")

    print()
    print("=" * 70)
    print("PER-MESSAGE COVERAGE")
    print("=" * 70)
    total = both + w_only + m_only
    print(f"  WSJT-X total messages : {total_wsjtx_msgs}")
    print(f"  MAP144 total messages : {total_map_msgs}")
    print(f"  Both decoded          : {both}")
    print(f"  WSJT-X only           : {w_only}")
    print(f"  MAP144 only           : {m_only}")
    if total:
        print(f"  WSJT-X coverage       : {100*(both+w_only)/total:.1f}%")
        print(f"  MAP144 coverage       : {100*(both+m_only)/total:.1f}%")
        print(f"  MAP144 / WSJT-X ratio : {100*(both)/(both+w_only):.1f}%"
              " (fraction of WSJT-X decodes MAP144 also got)")

    if args.show_diffs and diffs:
        print()
        print("=" * 70)
        print("PER-WAV DIFFS")
        print("=" * 70)
        for fname, ws, ms in diffs[:50]:
            print(f"  {fname}")
            for m in sorted(ws - ms):
                print(f"    WSJT-X only:  {m}")
            for m in sorted(ms - ws):
                print(f"    MAP144 only:  {m}")
            for m in sorted(ws & ms):
                print(f"    both       :  {m}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
