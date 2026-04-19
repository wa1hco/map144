#!/usr/bin/env python3
"""Audit MAP144 SNR estimates against jt9 output and optionally WSJT-X.

Reads saved decode WAVs from MSK144/detections/, recomputes MAP144's
pre-burst SNR estimate from the raw audio, and compares it to:
  • jt9_snr_db  — what jt9 reported (logged in decodes.jsonl)
  • est_snr_db  — MAP144's own estimate logged at decode time (if present)
  • wsjtx_snr   — WSJT-X SNR for the same ping (optional, via ALL.TXT)

This distinguishes two bias hypotheses:
  H1 (jt9 short-clip artifact): est_snr ≈ wsjtx_snr, jt9_snr is the outlier
  H2 (signal-chain SNR loss):   est_snr << wsjtx_snr, both MAP144 estimates agree

Usage:
    python snr_audit.py [--date YYYYMMDD] [--utc HH:MM-HH:MM] [--plot FILE]
"""

import argparse
import json
import re
import wave
import struct
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DETECTIONS_DIR = Path(__file__).parent / "MSK144/detections"
DECODES_JSONL  = DETECTIONS_DIR / "decodes.jsonl"
WSJTX_ALL      = Path.home() / ".local/share/WSJT-X - Local/ALL.TXT"

# ── SNR estimator (mirrors detection.py, but also uses post-ping noise) ───────
_DECODE_RATE  = 12_000
_SNR_N_FFT    = 1024
_SNR_WINDOW   = np.hanning(_SNR_N_FFT)
_SNR_FREQS    = np.fft.rfftfreq(_SNR_N_FFT, d=1.0 / _DECODE_RATE)
_SNR_SIG_MASK   = (_SNR_FREQS >= 900.0)  & (_SNR_FREQS <= 2100.0)
_SNR_NOISE_MASK = (_SNR_FREQS > 200.0)   & (_SNR_FREQS < 5500.0) & ~_SNR_SIG_MASK
_SNR_REF_BINS   = 2500.0 / (_DECODE_RATE / _SNR_N_FFT)


def _psd_frames(seg: np.ndarray, hop_divisor: int = 2) -> list:
    hop  = _SNR_N_FFT // hop_divisor
    psds = []
    for off in range(0, max(1, len(seg) - _SNR_N_FFT + 1), hop):
        blk = seg[off:off + _SNR_N_FFT]
        if len(blk) < _SNR_N_FFT:
            blk = np.pad(blk, (0, _SNR_N_FFT - len(blk)))
        psds.append(np.abs(np.fft.rfft(blk * _SNR_WINDOW)) ** 2)
    return psds


def estimate_snr(audio: np.ndarray, use_post_noise: bool = False,
                 post_noise_start_ms: float = 900.0) -> dict:
    """
    Compute SNR from a MAP144 decode WAV.

    WAV structure (at 12 kHz):
        100 ms pad | 500 ms pre-burst noise | ping + post (1200 ms) | 100 ms pad

    Returns dict with keys:
        snr_pre     – estimate using pre-burst noise only (current algorithm)
        snr_combined – estimate using pre + post noise (improved)
        n_pre_frames, n_post_frames
    """
    pad_n   = int(0.100 * _DECODE_RATE)   # 1200
    pre_n   = int(0.500 * _DECODE_RATE)   # 6000
    noise_end   = pad_n + pre_n           # 7200 = burst_start
    post_start  = noise_end + int(post_noise_start_ms / 1000 * _DECODE_RATE)

    result = {'snr_pre': None, 'snr_combined': None,
              'n_pre_frames': 0, 'n_post_frames': 0}

    if len(audio) <= noise_end:
        return result

    audio = audio.astype(np.float64)

    # ── Pre-burst noise frames ────────────────────────────────────────────────
    pre_psds = _psd_frames(audio[pad_n:noise_end])
    result['n_pre_frames'] = len(pre_psds)
    if not pre_psds:
        return result

    # ── Post-ping noise frames (ping has faded by post_noise_start_ms) ────────
    post_psds = []
    if use_post_noise and post_start < len(audio) - pad_n:
        post_psds = _psd_frames(audio[post_start:-pad_n or len(audio)])
    result['n_post_frames'] = len(post_psds)

    # ── Signal: peak in-band power over the burst region ─────────────────────
    burst_psds = _psd_frames(audio[noise_end:], hop_divisor=4)
    best_sig = max(
        (float(np.sum(p[_SNR_SIG_MASK])) for p in burst_psds),
        default=0.0
    )
    if best_sig <= 0.0:
        return result

    def _snr_from_frames(frames):
        if not frames:
            return None
        noise_per_bin = float(np.median(np.mean(frames, axis=0)[_SNR_NOISE_MASK]))
        if noise_per_bin <= 0.0:
            return None
        return int(round(10.0 * np.log10(best_sig / (noise_per_bin * _SNR_REF_BINS))))

    result['snr_pre']      = _snr_from_frames(pre_psds)
    result['snr_combined'] = _snr_from_frames(pre_psds + post_psds)
    return result


def read_wav_audio(path: Path) -> np.ndarray | None:
    """Read a mono 16-bit or 32-bit WAV, return float32 array normalised to ±1."""
    try:
        with wave.open(str(path), 'rb') as wf:
            n_ch     = wf.getnchannels()
            sampw    = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw      = wf.readframes(n_frames)
        if sampw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
        else:
            return None
        if n_ch > 1:
            samples = samples[::n_ch]   # take first channel
        return samples
    except Exception:
        return None


def load_decodes(path: Path, date_filter=None, utc_filter=None) -> list:
    """Load decodes.jsonl, optionally filtering by date and UTC time window."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('outcome', 'decoded') != 'decoded' and 'message' not in d:
                continue
            if not d.get('message'):
                continue
            ts_str = d.get('timestamp', '')
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d_%H:%M:%S.%f").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            if date_filter:
                if ts.date() != date_filter:
                    continue
            if utc_filter:
                lo, hi = utc_filter
                t_min = ts.hour * 60 + ts.minute
                if not (lo <= t_min < hi):
                    continue
            d['_ts'] = ts
            results.append(d)
    return results


def load_wsjtx(path: Path) -> dict:
    """Load WSJT-X ALL.TXT; return dict keyed by (period_ts, normalised_msg)."""
    pat = re.compile(
        r'^(\d{6}_\d{6})\s+([\d.]+)\s+Rx\s+MSK144\s+([+-]?\d+)\s+([\d.]+)\s+(\d+)\s+(.+)$'
    )
    index = {}
    if not path.exists():
        return index
    with open(path, errors='replace') as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            ts_str, _, snr, dt_s, _, msg = m.groups()
            try:
                ts = datetime.strptime("20" + ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts += timedelta(seconds=float(dt_s))
            epoch = int(ts.timestamp() // 15) * 15
            period = datetime.fromtimestamp(epoch, tz=timezone.utc)
            key = (period, ' '.join(msg.strip().upper().split()))
            index[key] = int(snr)
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date',  default=None, metavar='YYYYMMDD',
                    help='UTC date to audit (default: all)')
    ap.add_argument('--utc',   default=None, metavar='HH:MM-HH:MM',
                    help='UTC time-of-day window e.g. 09:00-13:00')
    ap.add_argument('--post',  type=float, default=900.0, metavar='MS',
                    help='ms after detection where post-ping noise starts (default: 900)')
    ap.add_argument('--plot',  default=None, metavar='FILE',
                    help='Save scatter plot to FILE')
    args = ap.parse_args()

    date_filter = None
    if args.date:
        try:
            date_filter = datetime.strptime(args.date, "%Y%m%d").date()
        except ValueError:
            print(f"Bad --date: {args.date}"); return

    utc_filter = None
    if args.utc:
        try:
            t0, t1 = args.utc.split('-')
            utc_filter = (int(t0[:2])*60+int(t0[3:]), int(t1[:2])*60+int(t1[3:]))
        except Exception:
            print(f"Bad --utc: {args.utc}"); return

    print(f"Loading decodes:  {DECODES_JSONL}")
    decodes = load_decodes(DECODES_JSONL, date_filter, utc_filter)
    print(f"  {len(decodes)} entries after filtering")

    print(f"Loading WSJT-X:   {WSJTX_ALL}")
    wsjtx = load_wsjtx(WSJTX_ALL)
    print(f"  {len(wsjtx)} MSK144 decodes indexed")

    # ── Match each decode entry to its WAV file and re-estimate SNR ───────────
    rows = []   # (jt9_snr, est_snr_logged, snr_pre, snr_combined, wsjtx_snr, message)

    for d in decodes:
        ts      = d['_ts']
        ts_file = (ts.strftime('%Y%m%d') + '_' +
                   ts.strftime('%H%M%S') + 'Z')
        rf_int  = int(round(d.get('radio_khz', 0)))
        msg     = d.get('message', '')
        msg_safe = re.sub(r'[^A-Za-z0-9]+', '_', msg).strip('_')
        wav_name = f"{ts_file}_{rf_int}kHz_{msg_safe}.wav"
        wav_path = DETECTIONS_DIR / wav_name

        if not wav_path.exists():
            # Try glob — timestamp might differ by a second (sub-second truncation)
            candidates = sorted(DETECTIONS_DIR.glob(
                f"*_{rf_int}kHz_{msg_safe}.wav"))
            wav_path = next(
                (p for p in candidates
                 if abs((datetime.strptime(p.stem.split('_')[0] + '_' +
                                           p.stem.split('_')[1].rstrip('Z'),
                                           '%Y%m%d_%H%M%S')
                         .replace(tzinfo=timezone.utc) - ts).total_seconds()) < 3),
                None)

        audio = read_wav_audio(wav_path) if wav_path and wav_path.exists() else None

        est = estimate_snr(audio, use_post_noise=True,
                           post_noise_start_ms=args.post) if audio is not None else {}

        # WSJT-X lookup by (period, message)
        epoch    = int(ts.timestamp() // 15) * 15
        period   = datetime.fromtimestamp(epoch, tz=timezone.utc)
        norm_msg = ' '.join(msg.upper().split())
        wsjtx_snr = wsjtx.get((period, norm_msg))

        rows.append({
            'ts':              ts,
            'message':         msg,
            'jt9_snr':         d.get('jt9_snr_db'),
            'est_snr_logged':  d.get('est_snr_db'),    # None for old log entries
            'snr_pre':         est.get('snr_pre'),
            'snr_combined':    est.get('snr_combined'),
            'n_pre_frames':    est.get('n_pre_frames', 0),
            'n_post_frames':   est.get('n_post_frames', 0),
            'wsjtx_snr':       wsjtx_snr,
            'wav_found':       audio is not None,
        })

    n_wav = sum(1 for r in rows if r['wav_found'])
    print(f"\nWAV files found: {n_wav} / {len(rows)}")

    # ── Statistics ────────────────────────────────────────────────────────────
    def _bias(label_a, key_a, label_b, key_b):
        pairs = [(r[key_a], r[key_b]) for r in rows
                 if r[key_a] is not None and r[key_b] is not None]
        if not pairs:
            return
        diffs = [a - b for a, b in pairs]
        print(f"  {label_a} − {label_b}:  n={len(pairs):3d}  "
              f"mean={sum(diffs)/len(diffs):+.1f} dB  "
              f"std={float(np.std(diffs)):.1f} dB  "
              f"min={min(diffs):+d}  max={max(diffs):+d}")

    print("\nSNR bias summary:")
    _bias("wsjtx_snr",      "wsjtx_snr",      "jt9_snr",        "jt9_snr")
    _bias("wsjtx_snr",      "wsjtx_snr",      "snr_pre (recomp)","snr_pre")
    _bias("wsjtx_snr",      "wsjtx_snr",      "snr_combined",   "snr_combined")
    _bias("snr_pre",        "snr_pre",         "jt9_snr",        "jt9_snr")
    _bias("snr_combined",   "snr_combined",    "snr_pre",        "snr_pre")

    # Noise frame counts
    frm = [r for r in rows if r['wav_found']]
    if frm:
        avg_pre  = sum(r['n_pre_frames']  for r in frm) / len(frm)
        avg_post = sum(r['n_post_frames'] for r in frm) / len(frm)
        print(f"\nAvg noise frames — pre: {avg_pre:.1f}  post: {avg_post:.1f}  "
              f"(combined improves noise floor std by "
              f"×{(avg_pre/(avg_pre+avg_post))**0.5:.2f} → "
              f"×{((avg_pre+avg_post)/avg_pre)**0.5:.2f} better)")

    if args.plot:
        _plot(rows, args.plot)


def _plot(rows, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Three scatter series: pre vs wsjtx, combined vs wsjtx, jt9 vs wsjtx
    def _pairs(key_x, key_y):
        return [(r[key_x], r[key_y]) for r in rows
                if r[key_x] is not None and r[key_y] is not None]

    p_jt9  = _pairs('wsjtx_snr', 'jt9_snr')
    p_pre  = _pairs('wsjtx_snr', 'snr_pre')
    p_comb = _pairs('wsjtx_snr', 'snr_combined')

    if not p_jt9 and not p_pre:
        print("No matched WSJT-X pairs to plot.")
        return

    all_vals = ([v for p in (p_jt9+p_pre+p_comb) for v in p])
    lo, hi   = min(all_vals) - 2, max(all_vals) + 2

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#f8f8f8')
    ax.set_facecolor('#f8f8f8')

    kw = dict(alpha=0.7, s=45)
    if p_pre:
        ax.scatter(*zip(*p_pre),  color='steelblue', marker='o',
                   label=f'MAP144 pre-noise only  (n={len(p_pre)})', **kw)
    if p_comb:
        ax.scatter(*zip(*p_comb), color='#2ca02c',   marker='s',
                   label=f'MAP144 pre+post noise  (n={len(p_comb)})', **kw)
    if p_jt9:
        ax.scatter(*zip(*p_jt9),  color='#d62728',   marker='^',
                   label=f'jt9 SNR               (n={len(p_jt9)})', **kw)

    ax.plot([lo, hi], [lo, hi], color='gray', linestyle=':', linewidth=1,
            alpha=0.6, label='y = x (perfect agreement)')

    ax.set_xlabel('SNR reported by WSJT-X  (dB)', fontsize=12)
    ax.set_ylabel('SNR reported by MAP144  (dB)', fontsize=12)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(linestyle=':', alpha=0.35)
    ax.set_title('MAP144 SNR estimator audit\n(WSJT-X as ground truth)', fontsize=13)
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
