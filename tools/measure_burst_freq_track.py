"""Per-window coherent f₀(t) tracker — measures within-burst frequency drift.

The companion tool ``measure_burst_freq_coherent.py`` gives one tight f₀
per burst by coherently summing all frames.  This tool slides that same
coherent matched filter through the burst in 1-frame windows (50%
overlap) to get f₀(t) — the within-burst frequency trajectory.

Use it to answer:

- Is the signal stationary, drifting linearly, or fluttering?
- What's the drift rate in Hz/s (trail Doppler signature)?
- What's the residual scatter around a linear-drift model (multipath
  signature)?

Method per burst (n_frames ≥ 2)
-------------------------------
1. Slide a 1-frame window (NSPM = 864 samples = 72 ms) through the
   burst with step NSPM/2 (50% overlap).  Gives 2·n_frames − 1
   measurements.
2. At each window position, run the existing
   ``coherent_burst_freq_estimate`` (one frame ⇒ Fourier limit
   ~14 Hz, parabolic interpolation refines below that for moderate
   SNR).
3. Linear-fit f₀(t): slope = drift rate (Hz/s); residual RMS = scatter.
4. Classify:

   ===============  =================================  ====================
   Label             Criterion                          Interpretation
   ===============  =================================  ====================
   STATIONARY       |drift| < 5 Hz/s AND resid < 6 Hz   clean stationary
   LINEAR_DRIFT     |drift| ≥ 5 Hz/s AND resid < 6 Hz   trail Doppler decay
   FLUTTERY         resid ≥ 6 Hz                        multipath flutter
   INSUFFICIENT     n_frames < 2                        can't track
   ===============  =================================  ====================

The 6 Hz threshold matches a 14 Hz FWHM / 2.355 single-window 1σ
noise level.  Residual below that is within measurement noise.

Outputs
-------
- 3-panel PNG per WAV:
    1. envelope (zoomed to burst, same as coherent tool)
    2. f₀(t) scatter with FWHM error bars + linear-drift fit + label
    3. per-window matched-filter response curves (colored by time)
- CSV in batch mode: one row per WINDOW (wav, burst_idx, win_idx,
  t_center_s, f0_hz, fwhm_hz, peak_to_noise), plus a companion
  ``*_summary.csv`` with per-burst (drift_hz_per_s, resid_hz, label).

Usage
-----
    python tools/measure_burst_freq_track.py --wav FILE
    python tools/measure_burst_freq_track.py --date YYYYMMDD --csv OUT.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

# Reuse the existing tools' helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_ping_freq_vs_time import (
    _bandpass, _envelope_db, _find_bursts,
    _load_wsjtx_messages, _msgs_for_wav,
    BURST_THRESH_DB, MIN_BURST_S,
)
from measure_burst_freq_coherent import (
    coherent_burst_freq_estimate, _audio_to_baseband,
    CoherentFreqEstimate,
    SR_AUDIO, FC_AUDIO,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import NSPM


# Classification thresholds — see module docstring
DRIFT_HZ_PER_S_THRESH = 5.0
RESID_HZ_THRESH = 6.0


# ── Per-window and per-burst data structures ──────────────────────────────

@dataclass
class WindowEstimate:
    """One sliding-window coherent f₀ measurement."""
    win_idx:        int
    t_center_s:     float
    f0_hz:          float
    fwhm_hz:        float
    peak_to_noise:  float
    peak_mag:       float


@dataclass
class BurstTrack:
    """One burst's per-window track plus a linear-drift summary."""
    burst_idx:        int
    t_start_s:        float
    t_end_s:          float
    n_frames:         int
    burst_avg_f0_hz:  float       # whole-burst coherent f₀ for reference
    windows:          list[WindowEstimate]
    drift_hz_per_s:   float | None
    drift_intercept:  float | None  # f₀ at midpoint, Hz
    resid_hz:         float | None
    label:            str


# ── Sliding-window coherent tracker ───────────────────────────────────────

def track_burst_frequency(
    baseband: np.ndarray,
    burst_start_n: int,
    burst_end_n: int,
    burst_idx: int,
    sr: int = SR_AUDIO,
    window_frames: int = 1,
    step_frac: float = 0.5,
    f_search_hz: float = 250.0,
    f_step_hz: float = 1.0,
) -> BurstTrack:
    """Slide a coherent matched filter through a burst; return f₀(t) + summary."""
    n_avail = burst_end_n - burst_start_n
    n_frames = n_avail // NSPM
    win_n = window_frames * NSPM
    step_n = max(1, int(win_n * step_frac))

    # Burst-level coherent estimate (for reference; also drives plot)
    whole = coherent_burst_freq_estimate(baseband, burst_start_n, burst_end_n,
                                          sr=sr, f_search_hz=f_search_hz,
                                          f_step_hz=f_step_hz)
    burst_avg = whole.f0_hz if whole is not None else float('nan')

    if n_frames < 2:
        return BurstTrack(
            burst_idx=burst_idx,
            t_start_s=burst_start_n / sr,
            t_end_s=burst_end_n / sr,
            n_frames=n_frames,
            burst_avg_f0_hz=burst_avg,
            windows=[],
            drift_hz_per_s=None,
            drift_intercept=None,
            resid_hz=None,
            label='INSUFFICIENT',
        )

    # Slide 1-frame windows through the burst
    windows: list[WindowEstimate] = []
    win_idx = 0
    pos = burst_start_n
    while pos + win_n <= burst_end_n:
        est = coherent_burst_freq_estimate(baseband, pos, pos + win_n,
                                            sr=sr, f_search_hz=f_search_hz,
                                            f_step_hz=f_step_hz)
        if est is not None:
            t_center = (pos + win_n / 2) / sr
            windows.append(WindowEstimate(
                win_idx=win_idx,
                t_center_s=t_center,
                f0_hz=est.f0_hz,
                fwhm_hz=est.peak_width_hz,
                peak_to_noise=est.peak_to_noise,
                peak_mag=est.peak_mag,
            ))
        win_idx += 1
        pos += step_n

    if len(windows) < 2:
        return BurstTrack(
            burst_idx=burst_idx,
            t_start_s=burst_start_n / sr,
            t_end_s=burst_end_n / sr,
            n_frames=n_frames,
            burst_avg_f0_hz=burst_avg,
            windows=windows,
            drift_hz_per_s=None,
            drift_intercept=None,
            resid_hz=None,
            label='INSUFFICIENT',
        )

    # Linear fit f₀ vs t (weighted by pk/noise — gives high-SNR windows more pull)
    t = np.array([w.t_center_s for w in windows])
    f = np.array([w.f0_hz for w in windows])
    w_weight = np.array([w.peak_to_noise for w in windows])
    # Centre time on burst midpoint so intercept is interpretable
    t_mid = 0.5 * (windows[0].t_center_s + windows[-1].t_center_s)
    t_centered = t - t_mid
    # Weighted linear least squares (closed form)
    W = w_weight.sum()
    if W <= 0:
        slope = 0.0
        intercept = float(f.mean())
    else:
        tw = (t_centered * w_weight).sum() / W
        fw = (f * w_weight).sum() / W
        denom = (w_weight * (t_centered - tw) ** 2).sum()
        if denom > 0:
            slope = float((w_weight * (t_centered - tw) * (f - fw)).sum() / denom)
            intercept = float(fw - slope * tw)
        else:
            slope = 0.0
            intercept = float(fw)
    fit = intercept + slope * t_centered
    resid = float(np.sqrt(((f - fit) ** 2 * w_weight).sum() / max(W, 1e-9)))

    # Classify
    if abs(slope) < DRIFT_HZ_PER_S_THRESH and resid < RESID_HZ_THRESH:
        label = 'STATIONARY'
    elif abs(slope) >= DRIFT_HZ_PER_S_THRESH and resid < RESID_HZ_THRESH:
        label = 'LINEAR_DRIFT'
    elif resid >= RESID_HZ_THRESH:
        label = 'FLUTTERY'
    else:
        label = 'UNCLASSIFIED'

    return BurstTrack(
        burst_idx=burst_idx,
        t_start_s=burst_start_n / sr,
        t_end_s=burst_end_n / sr,
        n_frames=n_frames,
        burst_avg_f0_hz=burst_avg,
        windows=windows,
        drift_hz_per_s=slope,
        drift_intercept=intercept,
        resid_hz=resid,
        label=label,
    )


# ── WAV-level analysis ────────────────────────────────────────────────────

def analyse_wav(path: Path,
                window_frames: int = 1,
                step_frac: float = 0.5) -> tuple[np.ndarray, float, list[BurstTrack]]:
    """Find bursts in a WAV and run the f₀(t) tracker on each."""
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

    baseband = _audio_to_baseband(audio, sr)
    tracks = []
    for bi, (i_start, i_end) in enumerate(burst_idx_pairs):
        burst_start_n = int(i_start * bin_s * sr)
        burst_end_n = int(i_end * bin_s * sr)
        track = track_burst_frequency(baseband, burst_start_n, burst_end_n,
                                      burst_idx=bi, sr=sr,
                                      window_frames=window_frames,
                                      step_frac=step_frac)
        tracks.append(track)
    return env_db, bin_s, tracks


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_burst_tracks(path: Path,
                       env_db: np.ndarray, env_bin_s: float,
                       tracks: list[BurstTrack],
                       decoded_msgs: list[str] | None = None,
                       out_path: Path | None = None) -> Path:
    """3-panel plot: envelope + f₀(t) + per-window response curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wav_duration = len(env_db) * env_bin_s
    t_env = np.arange(len(env_db)) * env_bin_s
    median_db = float(np.median(env_db))

    if tracks:
        t_lo = max(0.0, min(t.t_start_s for t in tracks) - 0.5)
        t_hi = min(wav_duration, max(t.t_end_s for t in tracks) + 0.5)
    else:
        t_lo, t_hi = 0.0, wav_duration

    fig, (ax_env, ax_f, ax_resp) = plt.subplots(3, 1, figsize=(13, 9),
                                                  gridspec_kw={'height_ratios': [1, 1.3, 1.2]})

    # ── Top: envelope ─────────────────────────────────────────────────────
    ax_env.plot(t_env, env_db - median_db, color='C0', lw=1.0)
    ax_env.axhline(BURST_THRESH_DB, color='r', ls='--', lw=0.5,
                   label=f'{BURST_THRESH_DB} dB')
    label_cmap = {
        'STATIONARY':   'tab:green',
        'LINEAR_DRIFT': 'tab:orange',
        'FLUTTERY':     'tab:red',
        'INSUFFICIENT': 'tab:gray',
        'UNCLASSIFIED': 'tab:purple',
    }
    for tr in tracks:
        c = label_cmap.get(tr.label, 'tab:gray')
        ax_env.axvspan(tr.t_start_s, tr.t_end_s, color=c, alpha=0.18)
        ax_env.annotate(f'b{tr.burst_idx+1}: {tr.label}\n{tr.n_frames} frames',
                        xy=((tr.t_start_s + tr.t_end_s)/2,
                             ax_env.get_ylim()[1]*0.80),
                        ha='center', fontsize=8, color=c)
    ax_env.set_xlim(t_lo, t_hi)
    zoom_mask = (t_env >= t_lo) & (t_env <= t_hi)
    if zoom_mask.any():
        seg = env_db[zoom_mask] - median_db
        ax_env.set_ylim(max(-10.0, float(seg.min()) - 2.0),
                         max(float(seg.max()) + 2.0, BURST_THRESH_DB + 1.0))
    ax_env.set_ylabel('Envelope (dB above median)')
    title_parts = [path.name]
    if decoded_msgs:
        title_parts.append('  /  '.join(decoded_msgs[:3]))
    ax_env.set_title('  —  '.join(title_parts) + f'    [zoom {t_lo:.2f}-{t_hi:.2f} s]')
    ax_env.grid(alpha=0.3)
    ax_env.legend(loc='upper right', fontsize=8)

    # ── Middle: f₀(t) scatter with FWHM error bars + drift fit ─────────────
    for tr in tracks:
        if not tr.windows:
            continue
        c = label_cmap.get(tr.label, 'tab:gray')
        t = np.array([w.t_center_s for w in tr.windows])
        f = np.array([w.f0_hz for w in tr.windows])
        # half-FWHM as error bar (rough 1σ proxy)
        err = np.array([w.fwhm_hz / 2.0 for w in tr.windows])
        # marker size scales with pk/noise (clipped to keep readable)
        pn = np.array([w.peak_to_noise for w in tr.windows])
        sizes = np.clip(pn * 12.0, 12.0, 120.0)
        ax_f.errorbar(t, f, yerr=err, fmt='none', ecolor=c, alpha=0.4, capsize=2)
        ax_f.scatter(t, f, s=sizes, c=c, edgecolor='black', linewidth=0.5,
                      zorder=3, label=f'b{tr.burst_idx+1}: {tr.label}')
        # Drift fit line
        if tr.drift_hz_per_s is not None:
            t_mid = 0.5 * (tr.windows[0].t_center_s + tr.windows[-1].t_center_s)
            t_line = np.linspace(t.min(), t.max(), 50)
            f_line = tr.drift_intercept + tr.drift_hz_per_s * (t_line - t_mid)
            ax_f.plot(t_line, f_line, color=c, ls='--', lw=1.2, alpha=0.7)
            # Annotation: drift, residual, burst-avg
            mid_t = float(t_line.mean()); mid_f = float(f_line.mean())
            ax_f.annotate(
                f'drift = {tr.drift_hz_per_s:+.1f} Hz/s\n'
                f'resid = {tr.resid_hz:.1f} Hz\n'
                f'⟨f₀⟩ = {tr.burst_avg_f0_hz:+.1f} Hz',
                xy=(mid_t, mid_f), xytext=(8, 8), textcoords='offset points',
                fontsize=8, color=c,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, alpha=0.85),
            )
    ax_f.axhline(0.0, color='k', ls=':', lw=0.5, alpha=0.5)
    ax_f.set_xlim(t_lo, t_hi)
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('f₀ (Hz from 1500 Hz audio)')
    ax_f.set_title('Within-burst frequency track — '
                    'point = per-window coherent f₀, bar = FWHM/2, size ∝ pk/noise')
    ax_f.grid(alpha=0.3)
    if any(tr.windows for tr in tracks):
        ax_f.legend(loc='best', fontsize=7)

    # ── Bottom: per-window response curves (colored by time within burst) ──
    # Re-run estimators just for the freq-resp curves — they're already
    # stored implicitly via the per-window f₀, but the magnitude arrays
    # are only kept by coherent_burst_freq_estimate.  For ax_resp we
    # cheaply replot each window's f₀ as a vertical line, since storing
    # 15+ full response arrays per burst would balloon plot memory.
    # Useful info: where each window's peak landed, relative to burst-avg.
    for tr in tracks:
        if not tr.windows:
            continue
        n_win = len(tr.windows)
        # gradient color: early window light, late window dark
        cmap = plt.get_cmap('viridis')
        for j, w in enumerate(tr.windows):
            col = cmap(j / max(n_win - 1, 1))
            ax_resp.axvline(w.f0_hz, color=col, lw=1.5, alpha=0.7,
                             label=f'b{tr.burst_idx+1} win{j+1}' if j == 0 or j == n_win - 1 else None)
        # burst-avg as a thicker reference
        ax_resp.axvline(tr.burst_avg_f0_hz, color=label_cmap.get(tr.label, 'gray'),
                         lw=2.5, ls=':', alpha=0.8,
                         label=f'b{tr.burst_idx+1} ⟨f₀⟩')
    ax_resp.axvline(0.0, color='k', ls=':', lw=0.5)
    ax_resp.set_xlabel('Frequency offset from 1500 Hz (Hz)')
    ax_resp.set_ylabel('(per-window f₀ marker)')
    ax_resp.set_title('Per-window f₀ markers — color gradient = window position '
                       '(early light → late dark)')
    ax_resp.grid(alpha=0.3)
    # Only show legend if there are labeled artists (skip on INSUFFICIENT-only WAVs)
    handles, labels = ax_resp.get_legend_handles_labels()
    if handles:
        ax_resp.legend(loc='upper right', fontsize=7)
    # Limit x-range to a reasonable window around burst-avg
    if tracks and any(tr.windows for tr in tracks):
        all_f = np.concatenate([
            np.array([w.f0_hz for w in tr.windows])
            for tr in tracks if tr.windows
        ])
        if all_f.size:
            cx = float(np.median(all_f))
            half = max(50.0, 1.5 * float(np.percentile(np.abs(all_f - cx), 95)))
            ax_resp.set_xlim(cx - half, cx + half)

    if out_path is None:
        default_dir = (Path(__file__).resolve().parent.parent
                       / 'docs' / 'figures' / 'burst_freq_track_single')
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / f'bursttrack_{path.stem}.png'

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None,
                    help='Single WAV path or filename in --wsjtx-dir/save/')
    ap.add_argument('--wsjtx-dir', type=Path,
                    default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None,
                    help='Batch: process all WAVs for this UTC date (YYYYMMDD)')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Write per-window estimates to this CSV (batch mode)')
    ap.add_argument('--summary-csv', type=Path, default=None,
                    help='Write per-burst summary (drift/resid/label) to this CSV')
    ap.add_argument('--plot-each-burst', type=Path, default=None,
                    metavar='DIR', help='In batch mode, save per-WAV plot to DIR')
    ap.add_argument('--window-frames', type=int, default=1,
                    help='Sliding window size in frames (default 1)')
    ap.add_argument('--step-frac', type=float, default=0.5,
                    help='Window step as fraction of window length (default 0.5)')
    ap.add_argument('--max-wavs', type=int, default=None)
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt) if all_txt.exists() else {}

    # ── Single-WAV mode ─────────────────────────────────────────────────────
    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        if not wav_path.exists():
            print(f"error: WAV not found: {args.wav}", file=sys.stderr)
            return 1
        env_db, bin_s, tracks = analyse_wav(wav_path,
                                             window_frames=args.window_frames,
                                             step_frac=args.step_frac)
        decoded_msgs = _msgs_for_wav(wav_path, msg_index)
        png = plot_burst_tracks(wav_path, env_db, bin_s, tracks, decoded_msgs)
        msgs_str = '  /  '.join(decoded_msgs) if decoded_msgs else '(no decoded message)'
        print(f"{wav_path.name}  ({len(tracks)} bursts)  WSJT-X: {msgs_str}")
        for tr in tracks:
            print(f"  burst {tr.burst_idx+1}: t={tr.t_start_s:5.2f}-{tr.t_end_s:5.2f}s  "
                  f"{tr.n_frames} frames  {len(tr.windows)} windows  "
                  f"⟨f₀⟩ = {tr.burst_avg_f0_hz:+7.2f} Hz  "
                  f"drift = {tr.drift_hz_per_s if tr.drift_hz_per_s is not None else float('nan'):+7.2f} Hz/s  "
                  f"resid = {tr.resid_hz if tr.resid_hz is not None else float('nan'):5.2f} Hz  "
                  f"label = {tr.label}")
        print(f"\nPlot: {png}")
        return 0

    # ── Batch mode ───────────────────────────────────────────────────────────
    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]
    if not wavs:
        print(f"error: no WAVs for date {args.date}", file=sys.stderr)
        return 1

    print(f"Processing {len(wavs)} WAVs", file=sys.stderr)
    if args.plot_each_burst:
        args.plot_each_burst.mkdir(parents=True, exist_ok=True)

    win_rows: list[tuple[str, str, int, WindowEstimate]] = []
    summary_rows: list[tuple[str, str, BurstTrack]] = []
    n_plotted = 0
    for i, p in enumerate(wavs):
        try:
            env_db, bin_s, tracks = analyse_wav(p,
                                                 window_frames=args.window_frames,
                                                 step_frac=args.step_frac)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        decoded_msgs = _msgs_for_wav(p, msg_index)
        msgs_str = '|'.join(decoded_msgs)
        for tr in tracks:
            summary_rows.append((p.name, msgs_str, tr))
            for w in tr.windows:
                win_rows.append((p.name, msgs_str, tr.burst_idx, w))
        if args.plot_each_burst and tracks:
            out_png = args.plot_each_burst / f'bursttrack_{p.stem}.png'
            try:
                plot_burst_tracks(p, env_db, bin_s, tracks, decoded_msgs,
                                   out_path=out_png)
                n_plotted += 1
            except Exception as exc:
                print(f"  {p.name}: PLOT ERROR {exc}", file=sys.stderr)
        if (i + 1) % 25 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(summary_rows)} bursts, "
                  f"{len(win_rows)} windows, {n_plotted} plots", file=sys.stderr)

    if args.csv and win_rows:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'wsjtx_msgs', 'burst_idx', 'win_idx',
                        't_center_s', 'f0_hz', 'fwhm_hz', 'peak_to_noise'])
            for name, msgs, bi, win in win_rows:
                w.writerow([name, msgs, bi, win.win_idx,
                            f'{win.t_center_s:.4f}', f'{win.f0_hz:+.2f}',
                            f'{win.fwhm_hz:.2f}', f'{win.peak_to_noise:.2f}'])
        print(f"\nPer-window CSV: {args.csv}")
    if args.summary_csv and summary_rows:
        with args.summary_csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'wsjtx_msgs', 'burst_idx', 't_start_s', 't_end_s',
                        'n_frames', 'n_windows', 'burst_avg_f0_hz',
                        'drift_hz_per_s', 'resid_hz', 'label'])
            for name, msgs, tr in summary_rows:
                w.writerow([name, msgs, tr.burst_idx,
                            f'{tr.t_start_s:.3f}', f'{tr.t_end_s:.3f}',
                            tr.n_frames, len(tr.windows),
                            f'{tr.burst_avg_f0_hz:+.2f}',
                            f'{tr.drift_hz_per_s:+.2f}' if tr.drift_hz_per_s is not None else '',
                            f'{tr.resid_hz:.2f}' if tr.resid_hz is not None else '',
                            tr.label])
        print(f"Per-burst summary CSV: {args.summary_csv}")

    # ── Summary distributions ────────────────────────────────────────────────
    if summary_rows:
        import collections
        label_counts = collections.Counter(tr.label for _, _, tr in summary_rows)
        n_total = len(summary_rows)
        print()
        print(f"Frequency-track classifier — {n_total} bursts")
        for lbl in ['STATIONARY', 'LINEAR_DRIFT', 'FLUTTERY', 'INSUFFICIENT', 'UNCLASSIFIED']:
            c = label_counts.get(lbl, 0)
            print(f"  {lbl:14s} : {c:4d}  ({100.0*c/n_total:5.1f}%)")
        # Drift / resid distributions over the trackable subset
        trackable = [tr for _, _, tr in summary_rows
                     if tr.drift_hz_per_s is not None]
        if trackable:
            drifts = np.array([tr.drift_hz_per_s for tr in trackable])
            resids = np.array([tr.resid_hz for tr in trackable])
            print()
            print(f"  Drift (Hz/s) over {len(trackable)} trackable bursts:")
            print(f"    p10/median/p90 : {np.percentile(drifts, 10):+.2f}  "
                  f"{np.median(drifts):+.2f}  {np.percentile(drifts, 90):+.2f}")
            print(f"  Residual scatter (Hz):")
            print(f"    p10/median/p90 : {np.percentile(resids, 10):.2f}  "
                  f"{np.median(resids):.2f}  {np.percentile(resids, 90):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
