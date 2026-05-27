#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the DSP-regression corpus index.

Reads existing real-world artefacts — WSJT-X ALL.TXT decode log, MAP144's
``launches.jsonl`` event log, WSJT-X ``save/`` audio WAVs, MAP144
``MSK144/detections/`` audio WAVs — and groups them by 15-s period.  Then
applies each tag predicate from :mod:`tests.corpus.tags` and writes one
row per period to ``tests/corpus/index.csv``.

Nothing is copied or moved.  The index references WAVs by path so the
predicates can be re-run after taxonomy changes.

Default date range is the last 7 days that have any artefacts on disk
(typically the WSJT-X save/ directory's coverage).

Output columns
--------------
``case_id``           — stable identifier ``period_yymmdd_HHMMSS``
``period_ts``         — ISO UTC datetime of period start
``freq_khz``          — radio frequency from the dominant ALL.TXT or
                        launches.jsonl entry (median across rows in period)
``wsjtx_wav``         — path or empty
``map144_wavs``       — ``;``-joined list of MAP144 detection WAVs in period
``n_wsjtx_decodes``   — count
``n_map144_events``   — count
``n_map144_decodes``  — count of launches.jsonl rows with outcome=decoded
``wsjtx_msgs``        — ``;``-joined unique normalised messages
``map144_msgs``       — ``;``-joined unique decoded messages from launches
``wsjtx_snr_min``     — min over wsjtx_decodes (or empty)
``wsjtx_snr_max``     — max (or empty)
``max_n_chans_2s``    — max broadband-burst indicator across launches
``median_coherence_h``— per-period median sync_phase_coherence_h (or empty)
``tags``              — ``;``-joined list of applied tag names

Predicates that aren't yet implementable (e.g. requires width / sigma_f
measurement) are skipped here and the corresponding tag is left out;
``tools/audit_taxonomy.py`` reports those tags as 'no-predicate'.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.corpus.tags import TAGS, Period   # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_WSJTX_DIR  = Path("/home/jeff/.local/share/WSJT-X - flex")
DEFAULT_DETECT_DIR = _REPO_ROOT / "MSK144" / "detections"
DEFAULT_LAUNCHES   = DEFAULT_DETECT_DIR / "launches.jsonl"
DEFAULT_OUT_CSV    = _REPO_ROOT / "tests" / "corpus" / "index.csv"
DEFAULT_MANUAL     = _REPO_ROOT / "tests" / "corpus" / "manual_tags.jsonl"


# ── Period helpers ────────────────────────────────────────────────────────────

EPOCH_2026 = datetime(2026, 1, 1, tzinfo=timezone.utc)

def _period_start(dt: datetime) -> datetime:
    """Floor a datetime to the 15-s WSJT-X period boundary."""
    sec = (dt - EPOCH_2026).total_seconds()
    return EPOCH_2026 + timedelta(seconds=int(sec // 15) * 15)


# ── ALL.TXT parsing ───────────────────────────────────────────────────────────

_ALL_RE = re.compile(
    r'^(?P<ts>\d{6}_\d{6})\s+'
    r'(?P<mhz>[\d.]+)\s+Rx\s+(?P<mode>MSK144)\s+'
    r'(?P<snr>[-+]?\d+)\s+(?P<dt>[-+]?[\d.]+)\s+(?P<af>\d+)\s+'
    r'(?P<msg>.+)$'
)

def _parse_ts_yy(ts_str: str) -> datetime:
    return datetime.strptime("20" + ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)

def parse_all_txt(path: Path, date_lo: datetime, date_hi: datetime,
                  ) -> dict[datetime, list[dict]]:
    """Return {period_start: [decode_dict, ...]} for MSK144 lines whose
    period falls in [date_lo, date_hi).

    See wav_corpus_test.py for the timestamp-vs-window-start subtlety: the
    ALL.TXT timestamp is when jt9 ran, ``dt`` is offset from the start of
    the analysis window which is floor(timestamp/15s).
    """
    out: dict[datetime, list[dict]] = defaultdict(list)
    with path.open(errors='replace') as fh:
        for ln in fh:
            m = _ALL_RE.match(ln.strip())
            if not m:
                continue
            ts = _parse_ts_yy(m['ts'])
            window_start = _period_start(ts)
            period = _period_start(window_start + timedelta(seconds=float(m['dt'])))
            if not (date_lo <= period < date_hi):
                continue
            out[period].append({
                'snr':    int(m['snr']),
                'dt':     float(m['dt']),
                'af':     int(m['af']),
                'mhz':    float(m['mhz']),
                'msg':    m['msg'].strip(),
                'raw_ts': m['ts'],
            })
    return out


# ── launches.jsonl ────────────────────────────────────────────────────────────

def parse_launches(path: Path, date_lo: datetime, date_hi: datetime,
                   ) -> dict[datetime, list[dict]]:
    """Return {period_start: [event, ...]} from launches.jsonl in range."""
    out: dict[datetime, list[dict]] = defaultdict(list)
    with path.open(errors='replace') as fh:
        for ln in fh:
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            try:
                ts = datetime.strptime(ev['timestamp'], '%Y-%m-%d_%H:%M:%S.%f'
                                       ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            period = _period_start(ts)
            if not (date_lo <= period < date_hi):
                continue
            out[period].append(ev)
    return out


# ── WAV scanning ──────────────────────────────────────────────────────────────

_WSJTX_WAV_RE = re.compile(r'^(\d{6})_(\d{6})\.wav$')
# MAP144 detection WAV: 20260507_111026Z_50260kHz_<...>.wav
_MAP144_WAV_RE = re.compile(r'^(\d{8})_(\d{6})Z_(\d+)kHz_(.+)\.wav$')

def scan_wsjtx_wavs(save_dir: Path, date_lo: datetime, date_hi: datetime,
                    ) -> dict[datetime, Path]:
    """One WSJT-X save WAV per period at most.  Returns {period_ts: path}."""
    out: dict[datetime, Path] = {}
    if not save_dir.is_dir():
        return out
    for p in save_dir.glob('*.wav'):
        m = _WSJTX_WAV_RE.match(p.name)
        if not m:
            continue
        try:
            ts = _parse_ts_yy(f'{m.group(1)}_{m.group(2)}')
        except ValueError:
            continue
        period = _period_start(ts)
        if not (date_lo <= period < date_hi):
            continue
        # WSJT-X audio WAV filenames ARE the period start
        out[period] = p
    return out

def scan_map144_wavs(detect_dir: Path, date_lo: datetime, date_hi: datetime,
                     ) -> dict[datetime, list[Path]]:
    """Multiple MAP144 detection WAVs per period possible."""
    out: dict[datetime, list[Path]] = defaultdict(list)
    if not detect_dir.is_dir():
        return out
    for p in detect_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != '.wav':
            continue
        m = _MAP144_WAV_RE.match(p.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(f'{m.group(1)}_{m.group(2)}',
                                   '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        period = _period_start(ts)
        if not (date_lo <= period < date_hi):
            continue
        out[period].append(p)
    return out


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm(msg: str) -> str:
    return ' '.join(msg.upper().split())


# ── Manual-tag log replay ────────────────────────────────────────────────────

def load_manual_tags(path: Path,
                     ) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Replay the manual annotation log in chronological order.

    Returns (additions, removals, notes) keyed by case_id.  Latest entry
    per (case, tag) wins; a later '+tag' un-does an earlier '-tag' and
    vice versa.  ``notes`` is the latest non-empty note per case.

    The log is append-only; we assume file order = chronological order.
    """
    add_set: dict[str, set[str]] = defaultdict(set)
    rem_set: dict[str, set[str]] = defaultdict(set)
    notes:   dict[str, str]      = {}
    if not path.is_file():
        return add_set, rem_set, notes
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            case = rec.get('case')
            if not case:
                continue
            for op in rec.get('ops', []):
                if not op or op[0] not in '+-':
                    continue
                sign, tag = op[0], op[1:]
                if sign == '+':
                    add_set[case].add(tag)
                    rem_set[case].discard(tag)
                else:
                    rem_set[case].add(tag)
                    add_set[case].discard(tag)
            note = rec.get('note') or ''
            if note:
                notes[case] = note
    return add_set, rem_set, notes


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsjtx-dir',   type=Path, default=DEFAULT_WSJTX_DIR)
    ap.add_argument('--detect-dir',  type=Path, default=DEFAULT_DETECT_DIR)
    ap.add_argument('--launches',    type=Path, default=DEFAULT_LAUNCHES)
    ap.add_argument('--out-csv',     type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument('--manual-tags', type=Path, default=DEFAULT_MANUAL,
                    help='Per-case annotation log (append-only jsonl).  '
                         'Latest +tag/-tag per case wins.')
    ap.add_argument('--date-from',   default=None,
                    help='UTC YYYYMMDD inclusive (default: 7 days ago)')
    ap.add_argument('--date-to',     default=None,
                    help='UTC YYYYMMDD exclusive (default: tomorrow)')
    args = ap.parse_args(argv)

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.date_from:
        date_lo = datetime.strptime(args.date_from, '%Y%m%d').replace(tzinfo=timezone.utc)
    else:
        date_lo = today - timedelta(days=7)
    if args.date_to:
        date_hi = datetime.strptime(args.date_to, '%Y%m%d').replace(tzinfo=timezone.utc)
    else:
        date_hi = today + timedelta(days=1)

    print(f"corpus index", file=sys.stderr)
    print(f"  range          : {date_lo.date()} .. {date_hi.date()}", file=sys.stderr)
    print(f"  wsjtx_dir      : {args.wsjtx_dir}", file=sys.stderr)
    print(f"  detect_dir     : {args.detect_dir}", file=sys.stderr)
    print(f"  launches.jsonl : {args.launches}", file=sys.stderr)
    print(f"  out_csv        : {args.out_csv}", file=sys.stderr)

    all_txt = args.wsjtx_dir / 'ALL.TXT'
    if not all_txt.is_file():
        print(f'warning: ALL.TXT not found at {all_txt}', file=sys.stderr)
        wsjtx_periods: dict[datetime, list[dict]] = {}
    else:
        wsjtx_periods = parse_all_txt(all_txt, date_lo, date_hi)
    launch_periods = parse_launches(args.launches, date_lo, date_hi) \
        if args.launches.is_file() else {}
    wsjtx_wavs   = scan_wsjtx_wavs(args.wsjtx_dir / 'save', date_lo, date_hi)
    map144_wavs  = scan_map144_wavs(args.detect_dir,        date_lo, date_hi)

    print(f"  WSJT-X decodes : {sum(len(v) for v in wsjtx_periods.values())} "
          f"in {len(wsjtx_periods)} periods", file=sys.stderr)
    print(f"  MAP144 events  : {sum(len(v) for v in launch_periods.values())} "
          f"in {len(launch_periods)} periods", file=sys.stderr)
    print(f"  WSJT-X WAVs    : {len(wsjtx_wavs)}", file=sys.stderr)
    print(f"  MAP144 WAVs    : {sum(len(v) for v in map144_wavs.values())} "
          f"in {len(map144_wavs)} periods", file=sys.stderr)

    # ── Union of periods that have any artefact ──────────────────────────────
    all_periods = set(wsjtx_periods) | set(launch_periods) | set(wsjtx_wavs) | set(map144_wavs)
    print(f"  union periods  : {len(all_periods)}", file=sys.stderr)

    # ── Resolve predicates we can actually run ───────────────────────────────
    active = [t for t in TAGS if t.predicate is not None]
    skipped = [t.name for t in TAGS if t.predicate is None]
    print(f"  active tags    : {len(active)}  (skipped {len(skipped)} no-predicate: {skipped})",
          file=sys.stderr)

    # ── Manual annotations ──────────────────────────────────────────────────
    manual_adds, manual_rems, manual_notes = load_manual_tags(args.manual_tags)
    print(f"  manual tags    : {sum(len(v) for v in manual_adds.values())} additions, "
          f"{sum(len(v) for v in manual_rems.values())} removals, "
          f"{len(manual_notes)} notes "
          f"across {len(set(manual_adds) | set(manual_rems) | set(manual_notes))} cases",
          file=sys.stderr)

    # ── Build rows ───────────────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ['case_id', 'period_ts', 'freq_khz',
              'wsjtx_wav', 'map144_wavs',
              'n_wsjtx_decodes', 'n_map144_events', 'n_map144_decodes',
              'wsjtx_msgs', 'map144_msgs',
              'wsjtx_snr_min', 'wsjtx_snr_max',
              'max_n_chans_2s', 'median_coherence_h', 'max_sq_metric_db',
              'tags', 'manual_tags', 'manual_note',
              # Detail blobs (JSON) — kept as separate columns so the
              # primary fields stay grep-friendly.  Consumers parse only
              # if they need per-decode / per-event detail.
              'wsjtx_decodes_json', 'map144_peak_json']
    n_rows = 0
    with args.out_csv.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for period in sorted(all_periods):
            wd = wsjtx_periods.get(period, [])
            me = launch_periods.get(period, [])
            ww = wsjtx_wavs.get(period)
            mw = map144_wavs.get(period, [])

            p_obj = Period(period_ts=period, wsjtx_wav=ww, map144_wavs=mw,
                           wsjtx_decodes=wd, map144_events=me)
            auto_tags = {t.name for t in active if _safe_predicate(t, p_obj)}
            case_id  = f'period_{period.strftime("%y%m%d_%H%M%S")}'
            m_add    = manual_adds.get(case_id, set())
            m_rem    = manual_rems.get(case_id, set())
            tags     = sorted((auto_tags | m_add) - m_rem)

            # Aggregate fields
            wsjtx_msgs  = sorted({_norm(d['msg']) for d in wd})
            map144_msgs = sorted({e.get('message','') for e in me
                                  if e.get('outcome') == 'decoded' and e.get('message')})
            snrs        = [d['snr'] for d in wd]
            mhz_values  = [d['mhz'] for d in wd] + [(e.get('radio_khz') or 0)/1000.0 for e in me]
            mhz_values  = [v for v in mhz_values if v]
            freq_khz    = int(round(statistics.median(mhz_values) * 1000)) if mhz_values else ''
            coh         = [e.get('sync_phase_coherence_h') for e in me]
            coh         = [v for v in coh if v is not None]
            med_coh     = f'{statistics.median(coh):.3f}' if coh else ''
            sq_max      = max((e.get('sq_metric_db_h') or -99.0 for e in me), default=None)
            sq_max_s    = f'{sq_max:.1f}' if sq_max is not None and sq_max > -99 else ''
            max_nc2s    = max((e.get('n_chans_2s') or 0 for e in me), default=0)
            n_dec_m144  = sum(1 for e in me if e.get('outcome') == 'decoded')

            # Per-decode WSJT-X blob: minimal subset for display
            wsjtx_decodes_blob = json.dumps([
                {'snr': d['snr'], 'dt': round(d['dt'], 2), 'af': d['af'],
                 'msg': _norm(d['msg'])}
                for d in wd
            ], ensure_ascii=False)

            # Peak MAP144 event by sq_metric_db_h (with pair_metric tie-break),
            # so the gallery can surface "what triggered MAP144 when it
            # missed" without re-joining launches.jsonl.
            peak_event = max(
                me,
                key=lambda e: (e.get('sq_metric_db_h') or -99.0,
                               e.get('pair_metric_db_h') or -99.0),
                default=None,
            ) if me else None
            if peak_event is not None:
                peak_blob = json.dumps({
                    't_sec':     peak_event.get('t_sec'),
                    'ch_signed': peak_event.get('ch_signed'),
                    'det':       peak_event.get('det_source'),
                    'sq':        peak_event.get('sq_metric_db_h'),
                    'pair':      peak_event.get('pair_metric_db_h'),
                    'sync':      peak_event.get('sync_metric_db_h'),
                    'coh':       peak_event.get('sync_phase_coherence_h'),
                })
            else:
                peak_blob = ''

            w.writerow([
                case_id,
                period.isoformat(),
                freq_khz,
                str(ww) if ww else '',
                ';'.join(str(p) for p in mw),
                len(wd),
                len(me),
                n_dec_m144,
                ';'.join(wsjtx_msgs),
                ';'.join(map144_msgs),
                min(snrs) if snrs else '',
                max(snrs) if snrs else '',
                max_nc2s,
                med_coh,
                sq_max_s,
                ';'.join(tags),
                ';'.join(sorted(m_add)),
                manual_notes.get(case_id, ''),
                wsjtx_decodes_blob,
                peak_blob,
            ])
            n_rows += 1
    print(f"wrote {n_rows} rows to {args.out_csv}", file=sys.stderr)
    return 0


def _safe_predicate(tag, period) -> bool:
    """Run a tag predicate and treat exceptions as non-match."""
    try:
        return bool(tag.predicate(period))
    except Exception as exc:
        print(f"  warning: tag {tag.name!r} predicate raised {exc!r}", file=sys.stderr)
        return False


if __name__ == '__main__':
    sys.exit(main())
