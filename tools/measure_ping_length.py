"""Measure MS ping duration in WSJT-X WAV files, two ways.

Purpose
-------
Operator's hypothesis (2026-05-11): SPD's navg distribution (97.5% navg=1,
2.5% navg=2, 0% navg=3) is inconsistent with visual observation that most
MS pings on the fast graph are >150 ms — which would predict navg=2
dominance.  If pings are mostly long but SPD reports navg=1, that points
to a bug in SPD's frame-averaging logic.

This tool measures each WAV's ping duration two ways:

  1. Broadband audio envelope (|audio|² with smoothing) — mirrors what
     the WSJT-X fast graph displays.
  2. Audio-band squared spectrum (power in 1300–1700 Hz) — closer to
     what MAP144's squared-FFT detector sees.

For each WAV, the tool returns ping start, end, duration, peak above
median, and ping count.  Output is per-WAV TSV and a summary
distribution.  Can be cross-referenced with launches.jsonl to test the
ping-length vs navg correlation.

Usage
-----
    python tools/measure_ping_length.py --date 20260511
    python tools/measure_ping_length.py --wav 260511_104230.wav --plot

Notes
-----
- Threshold: "burst" = envelope > median + threshold_db.  Default 4 dB.
- Bin: 50 ms broadband, 100 ms narrowband (cleaner envelope).
- Min burst: 50 ms (single bin) — anything shorter is ignored as noise.
- Output is mainly used for cross-tab with the navg field in
  decodes.jsonl / launches.jsonl.
"""
from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


# ── Burst extraction ──────────────────────────────────────────────────────────

@dataclass
class Burst:
    """A single contiguous envelope excursion above threshold."""
    t_start: float      # seconds into WAV
    t_end:   float
    duration: float     # = t_end - t_start
    peak_db: float      # peak of envelope - median (dB)


def _find_bursts(env_db: np.ndarray, bin_s: float, thresh_db: float,
                 min_duration_s: float = 0.050) -> list[Burst]:
    """Find all contiguous bins where env_db > median + thresh_db."""
    median = np.median(env_db)
    above = env_db > (median + thresh_db)
    bursts: list[Burst] = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        t_start = i * bin_s
        t_end   = j * bin_s
        if t_end - t_start >= min_duration_s:
            peak_db = float(env_db[i:j].max() - median)
            bursts.append(Burst(t_start, t_end, t_end - t_start, peak_db))
        i = j + 1
    return bursts


# ── Two envelope computations ────────────────────────────────────────────────

def broadband_envelope(audio: np.ndarray, sr: int, bin_s: float = 0.050
                       ) -> tuple[np.ndarray, float]:
    """RMS envelope in dBFS, broadband, ``bin_s``-second bins.
    Returns (env_db, bin_seconds)."""
    win = max(1, int(bin_s * sr))
    n_bins = len(audio) // win
    env = np.array([np.sqrt(np.mean(audio[i*win:(i+1)*win]**2))
                    for i in range(n_bins)])
    return 20 * np.log10(np.maximum(env, 1e-10)), bin_s


def audio_band_envelope(audio: np.ndarray, sr: int,
                        f_lo: float = 1300, f_hi: float = 1700,
                        bin_s: float = 0.100) -> tuple[np.ndarray, float]:
    """RMS envelope after bandpassing into 1300–1700 Hz audio band.
    Mirrors what MAP144's squared-FFT detector sees in the calling channel."""
    b, a = butter(4, [f_lo, f_hi], btype='bandpass', fs=sr)
    y = filtfilt(b, a, audio)
    win = max(1, int(bin_s * sr))
    n_bins = len(y) // win
    env = np.array([np.sqrt(np.mean(y[i*win:(i+1)*win]**2))
                    for i in range(n_bins)])
    return 20 * np.log10(np.maximum(env, 1e-10)), bin_s


# ── Per-WAV measurement ──────────────────────────────────────────────────────

@dataclass
class WavMeasurement:
    name:       str
    duration_s: float
    bb_bursts:  list[Burst]    # broadband bursts
    ab_bursts:  list[Burst]    # audio-band bursts

    @property
    def bb_longest_ms(self) -> float:
        return max((b.duration * 1000 for b in self.bb_bursts), default=0.0)

    @property
    def ab_longest_ms(self) -> float:
        return max((b.duration * 1000 for b in self.ab_bursts), default=0.0)

    @property
    def bb_total_ms(self) -> float:
        return sum(b.duration * 1000 for b in self.bb_bursts)

    @property
    def ab_total_ms(self) -> float:
        return sum(b.duration * 1000 for b in self.ab_bursts)


def measure_wav(path: Path, thresh_db: float = 4.0) -> WavMeasurement:
    with wave.open(str(path), 'rb') as wf:
        sr  = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)

    bb_env, bb_bin = broadband_envelope(audio, sr)
    ab_env, ab_bin = audio_band_envelope(audio, sr)

    bb_bursts = _find_bursts(bb_env, bb_bin, thresh_db)
    ab_bursts = _find_bursts(ab_env, ab_bin, thresh_db)

    return WavMeasurement(
        name=path.name,
        duration_s=nfr / sr,
        bb_bursts=bb_bursts,
        ab_bursts=ab_bursts,
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot(meas: WavMeasurement, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with wave.open(str(path), 'rb') as wf:
        sr  = wf.getframerate()
        audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    bb_env, bb_bin = broadband_envelope(audio, sr)
    ab_env, ab_bin = audio_band_envelope(audio, sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t_bb = np.arange(len(bb_env)) * bb_bin
    t_ab = np.arange(len(ab_env)) * ab_bin
    ax1.plot(t_bb, bb_env - np.median(bb_env), 'b-', lw=0.8)
    ax1.axhline(4.0, color='r', ls='--', lw=0.5, label='4 dB threshold')
    ax1.set_ylabel('Broadband env (dB above median)')
    ax1.set_title(f'{path.name} — broadband audio envelope (50 ms bins)')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(t_ab, ab_env - np.median(ab_env), 'g-', lw=0.8)
    ax2.axhline(4.0, color='r', ls='--', lw=0.5, label='4 dB threshold')
    ax2.set_ylabel('Audio-band env 1300-1700 Hz (dB above median)')
    ax2.set_xlabel('Time within WAV (s)')
    ax2.set_title(f'{path.name} — audio-band envelope (100 ms bins)')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    # Annotate detected bursts
    for b in meas.bb_bursts:
        ax1.axvspan(b.t_start, b.t_end, color='b', alpha=0.15)
        ax1.text((b.t_start + b.t_end) / 2, ax1.get_ylim()[1] * 0.9,
                 f'{b.duration*1000:.0f} ms\n+{b.peak_db:.1f} dB',
                 ha='center', fontsize=8)
    for b in meas.ab_bursts:
        ax2.axvspan(b.t_start, b.t_end, color='g', alpha=0.15)
        ax2.text((b.t_start + b.t_end) / 2, ax2.get_ylim()[1] * 0.9,
                 f'{b.duration*1000:.0f} ms\n+{b.peak_db:.1f} dB',
                 ha='center', fontsize=8)

    out = Path(f'/tmp/ping_length_{path.stem}.png')
    plt.tight_layout()
    plt.savefig(str(out), dpi=110)
    plt.close()
    print(f"  plot: {out}", file=sys.stderr)


# ── Cross-reference with launches.jsonl (when navg is needed) ──────────────

def navg_for_wav_period(wav_name: str, launches_path: Path) -> int | None:
    """For a WAV named YYMMDD_HHMMSS.wav, scan launches.jsonl and return
    the SPD navg from the first 'decoded' record whose timestamp falls in
    the 15-second window starting at the WAV's filename time.

    Returns None if no decoded SPD record found in that period."""
    import json
    import re
    from datetime import datetime, timedelta, timezone

    m = re.match(r'(\d{6})_(\d{6})\.wav$', wav_name)
    if not m:
        return None
    p_start = datetime.strptime(f"20{m.group(1)}_{m.group(2)}",
                                "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    p_end = p_start + timedelta(seconds=15)
    navg_re = re.compile(r'navg=(\d+)')

    with launches_path.open() as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get('outcome') != 'decoded':
                continue
            ts_str = rec.get('timestamp', '')
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d_%H:%M:%S.%f') \
                             .replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if not (p_start <= ts < p_end):
                continue
            line = rec.get('jt9_line', '') or ''
            nm = navg_re.search(line)
            if nm:
                return int(nm.group(1))
            return 0      # decoded but no navg → jt9 fallback
    return None


# ── Main / CLI ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--wsjtx-dir', type=Path,
                    default=Path("/home/jeff/.local/share/WSJT-X - flex"),
                    help='WSJT-X user directory containing save/')
    ap.add_argument('--date', default='20260511',
                    help='UTC date YYYYMMDD to process')
    ap.add_argument('--wav', default=None,
                    help='Single WAV to measure (overrides --date)')
    ap.add_argument('--thresh-db', type=float, default=4.0,
                    help='Burst threshold above envelope median, dB (default 4)')
    ap.add_argument('--plot', action='store_true',
                    help='Plot the envelopes (writes to /tmp/)')
    ap.add_argument('--launches', type=Path,
                    default=Path("/home/jeff/ham/map144/MSK144/detections/launches.jsonl"),
                    help='Path to launches.jsonl for navg cross-reference')
    ap.add_argument('--cross-navg', action='store_true',
                    help='Cross-tabulate ping length × navg (decoded WAVs only)')
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'

    # ── Single-WAV mode ─────────────────────────────────────────────────────
    if args.wav:
        wav_path = save_dir / args.wav
        if not wav_path.exists():
            wav_path = Path(args.wav)
        if not wav_path.exists():
            print(f"error: WAV not found: {args.wav}", file=sys.stderr)
            return 1
        meas = measure_wav(wav_path, thresh_db=args.thresh_db)
        print(f"{wav_path.name}  duration {meas.duration_s:.2f} s")
        print(f"  Broadband bursts:")
        for b in meas.bb_bursts:
            print(f"    t={b.t_start:5.2f}s-{b.t_end:5.2f}s  "
                  f"dur={b.duration*1000:6.0f} ms  +{b.peak_db:.1f} dB")
        print(f"  Audio-band bursts (1300-1700 Hz):")
        for b in meas.ab_bursts:
            print(f"    t={b.t_start:5.2f}s-{b.t_end:5.2f}s  "
                  f"dur={b.duration*1000:6.0f} ms  +{b.peak_db:.1f} dB")
        if args.plot:
            _plot(meas, wav_path)
        return 0

    # ── Batch mode ──────────────────────────────────────────────────────────
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if not wavs:
        print(f"error: no WAVs for date {args.date} in {save_dir}", file=sys.stderr)
        return 1

    print(f"Measuring {len(wavs)} WAVs from {save_dir}", file=sys.stderr)
    measurements = [measure_wav(w, thresh_db=args.thresh_db) for w in wavs]

    # ── Cross-navg mode: tabulate ping length × navg ────────────────────────
    if args.cross_navg:
        from collections import defaultdict
        # buckets: navg → list of (bb_longest_ms, ab_longest_ms)
        buckets = defaultdict(list)
        n_no_decode = 0
        for w, meas in zip(wavs, measurements):
            navg = navg_for_wav_period(w.name, args.launches)
            if navg is None:
                n_no_decode += 1
                continue
            buckets[navg].append((meas.bb_longest_ms, meas.ab_longest_ms))

        print(f"\nCross-tab: ping length (longest burst) × navg")
        print(f"  WAVs with no matching decoded launch: {n_no_decode}")
        print()
        print(f"  {'navg':>6}  {'n':>4}  {'bb_med_ms':>10}  {'bb_p25':>7}  {'bb_p75':>7}  "
              f"{'ab_med_ms':>10}  {'ab_p25':>7}  {'ab_p75':>7}")
        for navg in sorted(buckets):
            bb = np.array([x[0] for x in buckets[navg]])
            ab = np.array([x[1] for x in buckets[navg]])
            label = 'jt9 fb' if navg == 0 else str(navg)
            print(f"  {label:>6}  {len(bb):>4}  "
                  f"{np.median(bb):>10.0f}  {np.percentile(bb,25):>7.0f}  {np.percentile(bb,75):>7.0f}  "
                  f"{np.median(ab):>10.0f}  {np.percentile(ab,25):>7.0f}  {np.percentile(ab,75):>7.0f}")

        # Distribution of bb_longest_ms across SPD decodes only (navg≥1)
        all_spd = [x[0] for navg, recs in buckets.items() if navg >= 1 for x in recs]
        if all_spd:
            print()
            print(f"  All SPD decodes (n={len(all_spd)}): "
                  f"<72ms={sum(1 for x in all_spd if x<72)}  "
                  f"72-150ms={sum(1 for x in all_spd if 72<=x<150)}  "
                  f"150-300ms={sum(1 for x in all_spd if 150<=x<300)}  "
                  f">300ms={sum(1 for x in all_spd if x>=300)}")
            print(f"  Predicted (per operator's hypothesis): >150ms should dominate, "
                  f"matching navg=2 wins.  Observed split above.")
        return 0

    # ── Default: bulk summary ───────────────────────────────────────────────
    bb_all = [m.bb_longest_ms for m in measurements if m.bb_longest_ms > 0]
    ab_all = [m.ab_longest_ms for m in measurements if m.ab_longest_ms > 0]
    print(f"\nDuration of longest burst per WAV (ms):")
    print(f"  Broadband (n={len(bb_all)}):  "
          f"min={min(bb_all):.0f}  p25={np.percentile(bb_all,25):.0f}  "
          f"median={np.median(bb_all):.0f}  p75={np.percentile(bb_all,75):.0f}  "
          f"max={max(bb_all):.0f}")
    print(f"  Audio band (n={len(ab_all)}):  "
          f"min={min(ab_all):.0f}  p25={np.percentile(ab_all,25):.0f}  "
          f"median={np.median(ab_all):.0f}  p75={np.percentile(ab_all,75):.0f}  "
          f"max={max(ab_all):.0f}")

    print(f"\nWAVs with no burst above {args.thresh_db} dB: "
          f"{sum(1 for m in measurements if not m.bb_bursts)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
