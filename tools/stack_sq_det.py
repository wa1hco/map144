#!/usr/bin/env python3
"""
stack_sq_det.py — coherent frame stacking with sq_det-derived drift correction.

Workflow:
  1. Run slide_sq_det over the full WAV → per-18ms-frame freq estimates (ferr).
  2. Fit a line to ferr(t) over the ping window → sq_det drift estimate (Hz/s).
  3. Sweep a fine grid (±refine Hz/s around the estimate) for each N=1..max_frames.
  4. Stack N frames at the best drift for each N, run sq_det.

Plots:
  A. ferr(t) trajectory with linear-drift fit + DB drift for comparison
  B. sq_det metric vs drift offset (one curve per N)
  C. Stacked squared spectrum at best (N, drift)

Usage:
  python tools/stack_sq_det.py --id 17096
  python tools/stack_sq_det.py --wav FILE --dt 0.5 --fc 1454
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent.parent))
from map144_app.sq_det import (
    NSPM, FS, DEFAULT_FC, DEFAULT_NTOL,
    DETECT_THRESHOLD_NORM_LIN as THR_NORM,
    sq_det, slide_sq_det,
)
from tools.long_ping_gallery import load_wav

FRAME_DUR = NSPM / FS      # 0.072 s
STRIDE    = NSPM // 4      # 216 samples = 18 ms


def _stack(cdat_win, n_frames, drift_hz_per_s):
    """Quadratic-phase drift correction + coherent frame stack → (NSPM,) complex."""
    n = n_frames * NSPM
    t = np.arange(n, dtype=np.float64) / FS
    corrected = cdat_win[:n] * np.exp(-1j * np.pi * drift_hz_per_s * t ** 2)
    return corrected.reshape(n_frames, NSPM).sum(axis=0).astype(np.complex128)


def _fit_drift(t, ferr, weights=None):
    """Weighted linear fit to ferr(t). Returns (slope_hz_per_s, intercept_hz)."""
    if len(t) < 2:
        return 0.0, float(np.mean(ferr)) if len(ferr) else 0.0
    if weights is None:
        weights = np.ones(len(t))
    W  = weights.sum()
    tw = (weights * t).sum() / W
    fw = (weights * ferr).sum() / W
    denom = (weights * (t - tw) ** 2).sum()
    slope = float((weights * (t - tw) * (ferr - fw)).sum() / denom) if denom > 0 else 0.0
    return slope, float(fw - slope * tw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",          default="scratch/signals.db")
    ap.add_argument("--id",          type=int,   help="Signal ID from DB")
    ap.add_argument("--wav",         help="WAV path (overrides DB)")
    ap.add_argument("--dt",          type=float, help="dt_s override")
    ap.add_argument("--fc",          type=float, help="carrier Hz override")
    ap.add_argument("--max-frames",  type=int,   default=7)
    ap.add_argument("--refine",      type=float, default=150.0,
                    help="sweep ±Hz/s around sq_det drift estimate (default 150)")
    ap.add_argument("--refine-steps",type=int,   default=61)
    ap.add_argument("--no-show",     action="store_true")
    args = ap.parse_args()

    wav_path   = args.wav
    dt_s       = args.dt
    audio_hz   = args.fc
    db_drift   = None
    sig_label  = ""

    if args.id is not None:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        row  = conn.execute("SELECT * FROM signals WHERE id=?", (args.id,)).fetchone()
        conn.close()
        if row is None:
            sys.exit(f"Signal {args.id} not found in {args.db}")
        if wav_path is None:               wav_path  = row["wav_path"]
        if dt_s     is None:               dt_s      = float(row["dt_s"] or 7.5)
        if audio_hz is None and row["audio_hz"]: audio_hz = float(row["audio_hz"])
        if row["drift_hz_per_s"] is not None:    db_drift = float(row["drift_hz_per_s"])
        sig_label = f"id={args.id}  {row['timestamp']}  {row['message']}"

    wav_path = Path(wav_path) if wav_path else None
    if wav_path is None or not wav_path.exists():
        sys.exit("Need --wav or --id with a valid wav_path in the DB")
    if dt_s     is None: dt_s     = 7.5
    if audio_hz is None: audio_hz = DEFAULT_FC

    print(f"WAV      : {wav_path}")
    print(f"dt_s     : {dt_s:.3f}s   fc: {audio_hz:.1f} Hz")
    if db_drift is not None:
        print(f"DB drift : {db_drift:+.1f} Hz/s  (from measure_burst_freq_track)")

    # ── load + full-file slide_sq_det ─────────────────────────────────────────
    audio = load_wav(wav_path)
    cdat  = hilbert(audio.astype(np.float64)).astype(np.complex128)
    tt    = slide_sq_det(cdat, fc=audio_hz)
    xmed  = max(float(tt.xmed), 1e-30)
    print(f"xmed     : {xmed:.3g}  (single-frame noise ref)")

    # ── extract window ────────────────────────────────────────────────────────
    start_ideal = int(dt_s * FS) - NSPM // 2
    start       = max(0, (start_ideal // STRIDE) * STRIDE)
    win_n       = args.max_frames * NSPM
    if start + win_n > len(cdat):
        start = max(0, len(cdat) - win_n)
    cdat_win = cdat[start: start + win_n]
    t0       = start / FS
    print(f"Window   : [{t0:.3f}, {t0 + win_n/FS:.3f}] s")

    # ── sq_det drift estimate from per-frame ferr ─────────────────────────────
    # Select frames in the ping window where the signal is above threshold.
    tc   = tt.t_centers
    ferr = tt.ferr
    dn   = tt.detmet_norm

    pad  = 0.5
    win_mask = (tc >= t0) & (tc <= t0 + win_n / FS)
    sig_mask = win_mask & (dn >= THR_NORM)

    if sig_mask.sum() >= 2:
        sq_drift, sq_intercept = _fit_drift(tc[sig_mask], ferr[sig_mask],
                                            weights=dn[sig_mask])
        print(f"sq_det drift estimate : {sq_drift:+.1f} Hz/s  "
              f"(from {sig_mask.sum()} frames above threshold)")
    else:
        # Fallback: use all frames in window weighted by metric
        sq_drift, sq_intercept = _fit_drift(tc[win_mask], ferr[win_mask],
                                            weights=np.maximum(dn[win_mask], 0.1))
        print(f"sq_det drift estimate : {sq_drift:+.1f} Hz/s  "
              f"(weak signal — used all window frames)")

    # ── refinement sweep around sq_det drift estimate ────────────────────────
    frame_ns  = list(range(1, args.max_frames + 1))
    offsets   = np.linspace(-args.refine, args.refine, args.refine_steps)
    drifts    = sq_drift + offsets   # absolute drift values

    # metrics[n_idx, drift_idx] = raw detmet from sq_det on stacked frame
    metrics = np.zeros((len(frame_ns), len(drifts)))
    print(f"Sweeping ±{args.refine:.0f} Hz/s around {sq_drift:+.1f} Hz/s …")
    for ni, n in enumerate(frame_ns):
        for di, d in enumerate(drifts):
            stacked      = _stack(cdat_win, n, d)
            r            = sq_det(stacked, fc=audio_hz, ntol=DEFAULT_NTOL)
            metrics[ni, di] = r.detmet
        bi     = np.argmax(metrics[ni])
        print(f"  N={n}: peak={metrics[ni,bi]/xmed:.1f}×  @ {drifts[bi]:+.0f} Hz/s")

    metrics_norm = metrics / xmed

    best_flat        = np.argmax(metrics_norm)
    best_ni, best_di = np.unravel_index(best_flat, metrics_norm.shape)
    best_n           = frame_ns[best_ni]
    best_d           = drifts[best_di]
    best_val         = metrics_norm[best_ni, best_di]
    print(f"\nBest: N={best_n}  drift={best_d:+.1f} Hz/s  metric={best_val:.1f}×")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    title = (f"{wav_path.stem}  fc={audio_hz:.1f} Hz  dt={dt_s:.3f}s")
    if sig_label:
        title = sig_label + "\n" + title
    title += (f"\nBest stack: N={best_n}  drift={best_d:+.1f} Hz/s  "
              f"metric={best_val:.1f}×xmed")
    fig.suptitle(title, fontsize=8)

    # Panel A: ferr(t) trajectory + drift fits
    ax = axes[0]
    disp_mask = (tc >= t0 - pad) & (tc <= t0 + win_n / FS + pad)
    ax.scatter(tc[disp_mask], ferr[disp_mask] + audio_hz,
               c=dn[disp_mask], cmap="viridis", s=12, alpha=0.8,
               norm=plt.Normalize(0, max(6.0, dn[disp_mask].max())),
               label="per-frame fc estimate  (color=metric/xmed)")
    # sq_det drift fit line
    t_line = np.array([t0, t0 + win_n / FS])
    f_line = sq_intercept + sq_drift * t_line + audio_hz
    ax.plot(t_line, f_line, "r-", lw=1.5,
            label=f"sq_det fit  {sq_drift:+.1f} Hz/s")
    # DB drift if available
    if db_drift is not None:
        f_db = sq_intercept + db_drift * t_line + audio_hz
        ax.plot(t_line, f_db, "b--", lw=1.2, alpha=0.7,
                label=f"DB drift    {db_drift:+.1f} Hz/s")
    ax.axvline(dt_s, color="grey", lw=0.8, ls=":", alpha=0.7, label="dt_s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Carrier frequency (Hz)")
    ax.set_title("Per-frame carrier frequency (ferr from slide_sq_det)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: metric vs drift offset, one curve per N
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(frame_ns)))
    for ni, n in enumerate(frame_ns):
        bi    = np.argmax(metrics_norm[ni])
        peak  = metrics_norm[ni, bi]
        ax.plot(offsets, metrics_norm[ni], color=colors[ni], lw=1.1, alpha=0.85,
                label=f"N={n}  {peak:.1f}× @ {drifts[bi]:+.0f} Hz/s")
    ax.axhline(THR_NORM, color="k", ls="--", lw=0.9,
               label=f"per-frame threshold {THR_NORM}")
    ax.axvline(0, color="red", ls=":", lw=0.8,
               label=f"sq_det estimate  {sq_drift:+.1f} Hz/s")
    ax.set_xlabel(f"Drift offset from sq_det estimate ({sq_drift:+.1f} Hz/s)  →  Hz/s")
    ax.set_ylabel("detmet / xmed  (single-frame noise ref)")
    ax.set_title("sq_det metric after coherent stacking vs drift offset")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel C: stacked squared spectrum at best (N, drift)
    ax = axes[2]
    stk  = _stack(cdat_win, best_n, best_d)
    sq   = stk ** 2
    spec = np.fft.fft(sq)
    fq   = np.fft.fftfreq(NSPM, 1.0 / FS)
    pwr  = np.abs(spec) ** 2
    pos  = (fq > 0) & (fq <= 6000)
    ax.plot(fq[pos], 10 * np.log10(pwr[pos] + 1e-30), lw=0.8, color="steelblue")
    ax.axvline(2 * audio_hz - 1000, color="lime",   ls="--", lw=0.9,
               label=f"lo tone  {2*audio_hz-1000:.0f} Hz")
    ax.axvline(2 * audio_hz + 1000, color="orange", ls="--", lw=0.9,
               label=f"hi tone  {2*audio_hz+1000:.0f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Stacked squared spectrum  N={best_n}  drift={best_d:+.1f} Hz/s")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6000)

    plt.tight_layout()
    out = Path("scratch") / f"stack_sq_{wav_path.stem}_n{args.max_frames}.png"
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
