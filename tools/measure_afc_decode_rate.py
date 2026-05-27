"""Decode-rate comparison: single-frame / all-burst / greedy AFC vs jt9.

The peak_mag comparison in measure_afc_greedy.py is a sensitivity
proxy — it correlates with decode likelihood but isn't the ground
truth.  This tool runs each candidate accumulator through the real
jt9 decoder and counts how many produce VALID decodes (callsign /
grid passing the phantom-format regex).

Methodology
-----------
For each multi-frame burst in the corpus:

  1. Compute single-frame, all-burst, and greedy accumulators (same
     three accumulators measure_afc_greedy.py compares).
  2. For each accumulator:
     a. Find ish_best via sync correlation
     b. Cyclically align frame so message starts at sample 0
     c. Call jt9 via _frame_to_wav_and_decode
     d. Apply phantom-format filter to returned message
  3. Tally per-method:
     - n_valid_decodes
     - n_phantoms
     - Unique decodes (decoded by THIS method but not others)

Phantom filter (from memory note project_phantom_decode_signature)
----------------------------------------------------------------
A valid MSK144 message contains structurally well-formed callsigns
and (optionally) a grid square.  Phantoms — jt9 hash-table parity
hits on noise — produce malformed strings like "JSLZ62BUO1X".  We
filter by regex format check; valid decodes pass through, phantoms
are tallied separately.

Usage
-----
    python tools/measure_afc_decode_rate.py --wav 260511_111530.wav
    python tools/measure_afc_decode_rate.py --date 20260511 --csv OUT.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_ping_freq_vs_time import (
    _bandpass, _envelope_db, _find_bursts,
    _load_wsjtx_messages, _msgs_for_wav,
    BURST_THRESH_DB, MIN_BURST_S,
)
from measure_burst_freq_afc_correct import (
    _audio_to_baseband,
    SR_AUDIO, FC_AUDIO,
)
from measure_afc_greedy import greedy_decode_burst, GreedyResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import (
    _sync_correlate, _frame_to_wav_and_decode, NSPM,
)


# Phantom format regexes — from memory note
_RE_CALL = re.compile(r'^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z]$')
_RE_GRID = re.compile(r'^[A-R]{2}\d{2}([A-X]{2})?$')
# Recognised report / suffix tokens.  Reports come in formats:
#   +06   -15        signal report
#   R+06  R-15       received-acknowledged signal report (R prefix)
_RE_REPORT = re.compile(r'^R?[+-]?\d{1,3}$')
_TOK_SUFFIX = {'73', 'RR73', 'RRR', 'TNX'}
_TOK_PREFIX = {'CQ', 'QRZ', 'DE'}


def _is_valid_decode(msg: str) -> bool:
    """Phantom filter: every token must be a valid call, grid, report,
    or known marker.  Reject if any token is structurally malformed."""
    if not msg:
        return False
    tokens = msg.strip().split()
    if not tokens:
        return False
    for tok in tokens:
        if tok in _TOK_PREFIX or tok in _TOK_SUFFIX:
            continue
        if _RE_CALL.match(tok):
            continue
        if _RE_GRID.match(tok):
            continue
        if _RE_REPORT.match(tok):
            continue
        return False  # unrecognised → phantom
    return True


# ── Decode helper: align frame and call jt9 ──────────────────────────────

def _decode_accumulator(accum_complex: np.ndarray,
                         fc_out: float = 1500.0) -> tuple[str | None, bool]:
    """Sync-align an accumulator frame and run jt9.

    Returns (raw_message_or_None, is_valid_per_phantom_filter).
    """
    if accum_complex is None or len(accum_complex) != NSPM:
        return None, False
    # Find alignment
    _, _, ish_best = _sync_correlate(accum_complex)
    # Cyclically shift so message starts at sample 0
    aligned = np.roll(accum_complex, -ish_best).astype(np.complex64)
    msg, _snr = _frame_to_wav_and_decode(aligned, fc_out=fc_out)
    if msg is None:
        return None, False
    return msg, _is_valid_decode(msg)


# ── Per-burst decode comparison ──────────────────────────────────────────

@dataclass
class DecodeComparison:
    burst_idx:        int
    t_start_s:        float
    t_end_s:          float
    n_frames:         int
    centre_idx:       int
    # Each method: peak_mag, raw_msg, is_valid
    sf_peak:          float
    sf_msg:           str | None
    sf_valid:         bool
    ab_peak:          float
    ab_msg:           str | None
    ab_valid:         bool
    gr_peak:          float
    gr_msg:           str | None
    gr_valid:         bool
    included:         list[int]
    # Reference: what WSJT-X decoded in the same period (from ALL.TXT)
    wsjtx_msgs:       list[str] = field(default_factory=list)


def compare_decoders_for_burst(
    audio_real: np.ndarray,
    burst_start_n: int, burst_end_n: int,
    burst_idx: int,
    sr: int = SR_AUDIO,
) -> DecodeComparison | None:
    """Run all three accumulators through jt9 and compare decode outcomes."""
    g = greedy_decode_burst(audio_real, burst_start_n, burst_end_n,
                             burst_idx=burst_idx, sr=sr)
    if g is None:
        return None

    # Rebuild the three accumulators (greedy_decode_burst doesn't return
    # the actual complex arrays).  Re-run the mix-correction and sums
    # for each method.
    baseband = _audio_to_baseband(audio_real, sr=sr)
    n_frames = g.n_frames

    from measure_afc_greedy import _mix_correct_frame
    mc_frames = {k: _mix_correct_frame(baseband, burst_start_n + k * NSPM,
                                          g.f_hat, sr)
                 for k in range(n_frames)}

    sf_acc = mc_frames[g.centre_idx].copy()
    ab_acc = sum(mc_frames[k] for k in range(n_frames)).astype(np.complex64)
    gr_acc = sum(mc_frames[k] for k in g.included_frames).astype(np.complex64)

    sf_msg, sf_valid = _decode_accumulator(sf_acc)
    ab_msg, ab_valid = _decode_accumulator(ab_acc)
    gr_msg, gr_valid = _decode_accumulator(gr_acc)

    return DecodeComparison(
        burst_idx=burst_idx,
        t_start_s=g.t_start_s,
        t_end_s=g.t_end_s,
        n_frames=n_frames,
        centre_idx=g.centre_idx,
        sf_peak=g.sf_peak, sf_msg=sf_msg, sf_valid=sf_valid,
        ab_peak=g.ab_peak, ab_msg=ab_msg, ab_valid=ab_valid,
        gr_peak=g.greedy_peak, gr_msg=gr_msg, gr_valid=gr_valid,
        included=g.included_frames,
    )


# ── WAV-level driver ─────────────────────────────────────────────────────

def analyse_wav(path: Path,
                msg_index: dict,
                ) -> list[DecodeComparison]:
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    if sr != SR_AUDIO:
        raise ValueError(f"expected {SR_AUDIO} Hz, got {sr}")

    audio_bp = _bandpass(audio, sr)
    env_db, bin_s = _envelope_db(audio_bp, sr)
    burst_idx_pairs = _find_bursts(env_db, bin_s, BURST_THRESH_DB, MIN_BURST_S)

    wsjtx_msgs = _msgs_for_wav(path, msg_index)

    results = []
    for bi, (i_start, i_end) in enumerate(burst_idx_pairs):
        burst_start_n = int(i_start * bin_s * sr)
        burst_end_n = int(i_end * bin_s * sr)
        cmp = compare_decoders_for_burst(audio, burst_start_n, burst_end_n,
                                          bi, sr=sr)
        if cmp is not None:
            cmp.wsjtx_msgs = wsjtx_msgs
            results.append(cmp)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def _format_decode(msg: str | None, valid: bool) -> str:
    if msg is None:
        return '-'
    flag = '✓' if valid else '✗'
    return f'{flag} "{msg}"'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None)
    ap.add_argument('--wsjtx-dir', type=Path,
                     default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None)
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument('--max-wavs', type=int, default=None)
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt) if all_txt.exists() else {}

    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        results = analyse_wav(wav_path, msg_index)
        wsjtx_msgs = _msgs_for_wav(wav_path, msg_index)
        msgs_str = '  /  '.join(wsjtx_msgs) if wsjtx_msgs else '(none)'
        print(f"{wav_path.name}  ({len(results)} bursts)  WSJT-X: {msgs_str}")
        for r in results:
            print(f"\n  burst {r.burst_idx+1}: t={r.t_start_s:.2f}-{r.t_end_s:.2f}s  "
                  f"{r.n_frames}fr  centre={r.centre_idx}  kept={r.included}")
            print(f"    single-frame:  peak={r.sf_peak:>5.0f}  decode={_format_decode(r.sf_msg, r.sf_valid)}")
            print(f"    all-burst   :  peak={r.ab_peak:>5.0f}  decode={_format_decode(r.ab_msg, r.ab_valid)}")
            print(f"    greedy      :  peak={r.gr_peak:>5.0f}  decode={_format_decode(r.gr_msg, r.gr_valid)}")
        return 0

    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]

    print(f"Processing {len(wavs)} WAVs (calls real jt9 — slow)", file=sys.stderr)
    all_results: list[tuple[str, DecodeComparison]] = []
    for i, p in enumerate(wavs):
        try:
            results = analyse_wav(p, msg_index)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        for r in results:
            all_results.append((p.name, r))
        if (i + 1) % 20 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(all_results)} bursts processed",
                  file=sys.stderr)

    if args.csv and all_results:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 't_start_s', 't_end_s', 'n_frames', 'centre_idx',
                        'included', 'wsjtx_msgs',
                        'sf_peak', 'sf_msg', 'sf_valid',
                        'ab_peak', 'ab_msg', 'ab_valid',
                        'gr_peak', 'gr_msg', 'gr_valid'])
            for name, r in all_results:
                w.writerow([name,
                            f'{r.t_start_s:.3f}', f'{r.t_end_s:.3f}',
                            r.n_frames, r.centre_idx,
                            ' '.join(map(str, r.included)),
                            '|'.join(r.wsjtx_msgs),
                            f'{r.sf_peak:.0f}', r.sf_msg or '', int(r.sf_valid),
                            f'{r.ab_peak:.0f}', r.ab_msg or '', int(r.ab_valid),
                            f'{r.gr_peak:.0f}', r.gr_msg or '', int(r.gr_valid)])
        print(f"\nCSV: {args.csv}")

    # ── Summary ────────────────────────────────────────────────────────────
    n_total = len(all_results)
    sf_valid = sum(1 for _, r in all_results if r.sf_valid)
    ab_valid = sum(1 for _, r in all_results if r.ab_valid)
    gr_valid = sum(1 for _, r in all_results if r.gr_valid)
    sf_phantoms = sum(1 for _, r in all_results if r.sf_msg and not r.sf_valid)
    ab_phantoms = sum(1 for _, r in all_results if r.ab_msg and not r.ab_valid)
    gr_phantoms = sum(1 for _, r in all_results if r.gr_msg and not r.gr_valid)

    # Best-of-three: any method produces a valid decode
    best_valid = sum(1 for _, r in all_results
                     if r.sf_valid or r.ab_valid or r.gr_valid)
    # Unique contributions
    only_gr = sum(1 for _, r in all_results
                  if r.gr_valid and not r.sf_valid and not r.ab_valid)
    only_ab = sum(1 for _, r in all_results
                  if r.ab_valid and not r.sf_valid and not r.gr_valid)
    only_sf = sum(1 for _, r in all_results
                  if r.sf_valid and not r.ab_valid and not r.gr_valid)

    print()
    print(f"=== Decode rate comparison  ({n_total} bursts) ===")
    print(f"  Method             valid    phantoms   only-this")
    print(f"  single-frame    : {sf_valid:>4} ({100*sf_valid/max(n_total,1):>4.0f}%)  "
          f"{sf_phantoms:>4} ({100*sf_phantoms/max(n_total,1):>4.0f}%)  "
          f"{only_sf:>4}")
    print(f"  all-burst sum   : {ab_valid:>4} ({100*ab_valid/max(n_total,1):>4.0f}%)  "
          f"{ab_phantoms:>4} ({100*ab_phantoms/max(n_total,1):>4.0f}%)  "
          f"{only_ab:>4}")
    print(f"  greedy          : {gr_valid:>4} ({100*gr_valid/max(n_total,1):>4.0f}%)  "
          f"{gr_phantoms:>4} ({100*gr_phantoms/max(n_total,1):>4.0f}%)  "
          f"{only_gr:>4}")
    print()
    print(f"  best-of-three (any method valid): {best_valid} ({100*best_valid/max(n_total,1):.0f}%)")
    print(f"  net gain vs single-frame alone:    +{best_valid - sf_valid}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
