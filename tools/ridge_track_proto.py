#!/usr/bin/env python3
"""
ridge_track_proto.py -- EXPLORATORY prototype (development mode, NOT production).

Demonstrates the data-domain frequency-vs-time tracker for an MSK144 burst:

    band-limit (analytic)  ->  square (full-energy carrier recovery)
        ->  squared spectrogram  ->  Viterbi ridge extraction (the drift track)

Why each step:
  * Band-limit BEFORE squaring. Squaring z = s+n gives s^2 + 2sn + n^2; the
    n^2 term is the autocorrelation of the noise spectrum, so wideband noise
    folds into a pedestal that buries the carrier tone. Band-limiting to ~the
    signal occupied bandwidth is the optimal-noncoherent-receiver structure and
    the reason channelization is load-bearing (squaring the full 48 kHz fails).
  * Square. For MSK (CPM h=0.5) the data phase-flips cancel under squaring,
    leaving discrete tones at 2*(carrier +/- 500 Hz). This uses ALL 144 symbols
    (full energy, ~+9.5 dB over the 16-symbol sync), at the cost of a noncoherent
    squaring penalty and loss of absolute timing/phase.
  * Squared spectrogram = the carrier-vs-time surface. A drifting signal is a
    smooth ridge through it.
  * Viterbi/DP = optimal smooth high-energy path: states = freq bins, emission =
    log power, transition = -lam*(df)^2 within +/-max_jump. Greedy peak-picking
    hops between tracks; Viterbi commits to a smooth ridge (and a second pass
    after masking finds a genuine second component if one exists).

Test case: 260511_093000.wav -- KA1EME K0TPP 73 (SNR +5, DT 12.4 s), which shows
two frequency components + non-linear drift in the first frames (audio ~1606 Hz).
The overlapping CQ KD9VV EN71 sits 3 Hz away at 1603.

Output: scratch/ridge_proto/ridge_260511_093000.png   (throwaway, gitignored)
"""
import sys
import argparse
import os
import wave
import numpy as np
from scipy.signal import hilbert  # noqa: F401  (kept for reference; FFT-mask analytic used below)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 12000.0
NSPM = 864           # samples per MSK144 frame (72 ms)

NFFT = 512
HOP = 48
ENV_THRESH_DB = 3.0                # burst gate: envelope dB above median

# Per-ping presets (carrier = decoded audio Hz; t_lo/t_hi = gate window; plot = zoom)
PRESETS = {
    "k0tpp": dict(
        wav="/home/jeff/.local/share/WSJT-X - flex/save/260511_093000.wav",
        carrier=1606, t_lo=12.10, t_hi=12.85, plot_lo=12.25, plot_hi=12.49,
        title="260511_093000  KA1EME K0TPP 73 (SNR +5)"),
    "kb8otk": dict(
        wav="/home/jeff/.local/share/WSJT-X - flex/save/260511_104230.wav",
        carrier=1623, t_lo=11.05, t_hi=11.70, plot_lo=11.28, plot_hi=11.48,
        title="260511_104230  KB8OTK N4JP FM14 (SNR +9)"),
}


def load_audio(path):
    w = wave.open(path)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    w.close()
    return a


def bandlimit_analytic(audio, f_lo, f_hi):
    """Analytic signal band-limited to [f_lo, f_hi] Hz via an FFT mask.
    Returns complex baseband-at-RF (still centered at audio freq, just
    band-limited)."""
    N = len(audio)
    Z = np.fft.fft(audio)
    f = np.fft.fftfreq(N, 1.0 / FS)
    mask = (f >= f_lo) & (f <= f_hi)          # positive freqs only -> analytic
    Zb = np.zeros_like(Z)
    Zb[mask] = 2.0 * Z[mask]                   # x2 keeps the one-sided energy
    return np.fft.ifft(Zb)


def stft_power(z, nfft, hop):
    win = np.hanning(nfft)
    T = 1 + (len(z) - nfft) // hop
    S = np.empty((T, nfft), np.float64)
    for t in range(T):
        seg = z[t * hop:t * hop + nfft] * win
        S[t] = np.abs(np.fft.fft(seg, nfft)) ** 2
    times = (np.arange(T) * hop + nfft / 2.0) / FS
    freqs = np.fft.fftfreq(nfft, 1.0 / FS)
    return times, freqs, S


def viterbi_ridge(S_region, lam, max_jump):
    """Best smooth high-energy path through a (T, F) power region.
    states = freq-bin index, emission = per-time-baselined log power,
    transition penalty = -lam * (delta_bin)^2 within +/- max_jump bins."""
    T, F = S_region.shape
    E = np.log(S_region + 1e-12)
    E -= np.median(E, axis=1, keepdims=True)      # per-time baseline (handle power swings)
    score = np.full((T, F), -1e18)
    back = np.zeros((T, F), np.int32)
    score[0] = E[0]
    for t in range(1, T):
        prev = score[t - 1]
        for f in range(F):
            lo = max(0, f - max_jump)
            hi = min(F, f + max_jump + 1)
            cand = prev[lo:hi] - lam * (np.arange(lo, hi) - f) ** 2
            j = int(np.argmax(cand))
            back[t, f] = lo + j
            score[t, f] = E[t, f] + cand[j]
    path = np.empty(T, np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    # path total emission (for comparing ridge #1 vs #2)
    tot = float(np.sum(E[np.arange(T), path]))
    return path, tot


def find_burst(times, env_db, t_lo, t_hi):
    """Return a boolean time-mask for frames inside the band-limited
    envelope's burst (env_db above its in-window median by ENV_THRESH_DB),
    restricted to [t_lo, t_hi]. The gate keeps the ridge tracker from
    wandering through noise outside the ping."""
    win = (times >= t_lo) & (times <= t_hi)
    med = np.median(env_db[win])
    burst = win & (env_db > med + ENV_THRESH_DB)
    return burst, med


def main():
    ap = argparse.ArgumentParser(description="Viterbi ridge tracker on one MSK144 ping")
    ap.add_argument("preset", nargs="?", default="k0tpp", choices=list(PRESETS),
                    help="which ping preset to track")
    args = ap.parse_args()
    p = PRESETS[args.preset]
    WAV = p["wav"]
    carrier = p["carrier"]
    T_LO, T_HI = p["t_lo"], p["t_hi"]
    PLOT_LO, PLOT_HI = p["plot_lo"], p["plot_hi"]
    TITLE = p["title"]
    OUT = f"scratch/ridge_proto/ridge_{os.path.basename(WAV).replace('.wav','')}.png"
    # carrier-relative bands: band-limit +/-700 Hz; squared window around 2*carrier +/-1300
    SIG_LO, SIG_HI = carrier - 700.0, carrier + 700.0
    SQ_LO, SQ_HI = 2 * carrier - 1300.0, 2 * carrier + 1300.0

    audio = load_audio(WAV)
    z = bandlimit_analytic(audio, SIG_LO, SIG_HI)     # analytic, band-limited
    y = z * z                                          # SQUARE (full-energy carrier recovery)

    ta, fa, Sa = stft_power(z, NFFT, HOP)              # analytic (raw signal/MSK tones)
    tq, fq, Sq = stft_power(y, NFFT, HOP)              # squared (carrier ridge)

    # band-limited envelope (in-band power vs time) -> locate the burst
    fa_band = (fa >= SIG_LO) & (fa <= SIG_HI)
    env = Sa[:, fa_band].sum(axis=1)
    env_db = 10 * np.log10(env + 1e-12)
    env_db -= np.median(env_db)
    burst, med = find_burst(ta, env_db + np.median(env_db), T_LO, T_HI)  # mask on raw scale
    # recompute simply on the normalized scale
    win = (ta >= T_LO) & (ta <= T_HI)
    burst = win & (env_db > ENV_THRESH_DB)
    bt = ta[burst]
    if bt.size:
        b_lo, b_hi = bt.min(), bt.max()
    else:
        b_lo, b_hi = T_LO, T_HI
    print("=== ridge_track_proto: 260511_093000.wav (K0TPP ping) ===")
    print(f"burst gate (> {ENV_THRESH_DB} dB above median): "
          f"{burst.sum()} frames, {b_lo:.3f}-{b_hi:.3f} s, peak {env_db[win].max():.1f} dB")

    # --- squared-domain Viterbi ridge, GATED to the burst time span ---
    fpos = fq >= 0
    order = np.argsort(fq[fpos])
    fq_pos = fq[fpos][order]
    Sq_pos = Sq[:, fpos][:, order]
    fwin = (fq_pos >= SQ_LO) & (fq_pos <= SQ_HI)
    fq_r = fq_pos[fwin]
    gate = (tq >= b_lo) & (tq <= b_hi)
    tq_g = tq[gate]
    Sq_g = Sq_pos[np.ix_(gate, fwin)]
    df = fq_r[1] - fq_r[0]
    # Physical drift limit: ~ a few Hz/ms. HOP=4 ms; squared domain doubles drift.
    # Allow ~50 Hz/hop in the squared domain, penalise quadratically for smoothness.
    max_jump = max(1, int(50.0 / df))
    lam = 0.05
    ridge1, tot1 = viterbi_ridge(Sq_g, lam, max_jump)
    # second ridge after masking the first
    Sq_g2 = Sq_g.copy()
    guard = max(2, int(150.0 / df))
    for t in range(Sq_g2.shape[0]):
        lo = max(0, ridge1[t] - guard); hi = min(Sq_g2.shape[1], ridge1[t] + guard + 1)
        Sq_g2[t, lo:hi] = 1e-12
    ridge2, tot2 = viterbi_ridge(Sq_g2, lam, max_jump)
    f_r1, f_r2 = fq_r[ridge1], fq_r[ridge2]
    carr1, carr2 = f_r1 / 2.0, f_r2 / 2.0
    print(f"squared bin width {df:.1f} Hz   max_jump {max_jump}   lam {lam}")
    print(f"ridge #1: carrier-equiv {carr1.min():.0f}-{carr1.max():.0f} Hz  "
          f"drift {carr1.max()-carr1.min():.0f} Hz  tot {tot1:.1f}")
    print(f"ridge #2: carrier-equiv {carr2.min():.0f}-{carr2.max():.0f} Hz  "
          f"drift {carr2.max()-carr2.min():.0f} Hz  tot {tot2:.1f}  ratio {tot2/max(tot1,1e-9):.2f}")

    # --- no-squaring cross-check: band-limited spectral CENTROID over the burst ---
    # centroid of the raw analytic spectrogram = carrier for balanced MSK (averages the two tones)
    fa_idx = np.where(fa_band)[0]
    fa_vals = fa[fa_idx]
    Sa_band = Sa[:, fa_idx]
    centroid = (Sa_band * fa_vals).sum(axis=1) / (Sa_band.sum(axis=1) + 1e-12)
    cen_g = centroid[gate]

    # ---- plot ----
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    fa_pos = (fa >= SIG_LO - 100) & (fa <= SIG_HI + 100)
    extent_a = [ta[0], ta[-1], fa[fa_pos].min(), fa[fa_pos].max()]
    ax0.imshow(10 * np.log10(Sa[:, fa_pos].T + 1e-12), origin="lower", aspect="auto",
               extent=extent_a, cmap="viridis")
    ax0.plot(tq_g, cen_g, color="red", lw=2.2, label="band-limited spectral centroid (no squaring)")
    ax0.axvspan(b_lo, b_hi, color="white", alpha=0.12)
    ax0.axhline(carrier, color="cyan", lw=0.8, ls=":", label=f"decode {carrier} Hz")
    ax0.set_xlim(PLOT_LO, PLOT_HI); ax0.set_ylim(carrier - 230, carrier + 230)
    ax0.set_ylabel("audio Hz")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"{TITLE}  |  analytic spectrogram + carrier centroid (burst-gated)")

    extent_q = [tq_g[0], tq_g[-1], fq_r.min(), fq_r.max()]
    ax1.imshow(10 * np.log10(Sq_g.T + 1e-12), origin="lower", aspect="auto",
               extent=extent_q, cmap="viridis")
    ax1.plot(tq_g, f_r1, color="red", lw=2.2, label="Viterbi ridge #1")
    ax1.plot(tq_g, f_r2, color="orange", lw=1.6, ls="--",
             label=f"ridge #2 ({tot2/max(tot1,1e-9):.0%} of #1)")
    ax1.axhline(2 * carrier, color="cyan", lw=0.8, ls=":", label=f"2x{carrier} (decode)")
    ax1.set_xlim(PLOT_LO, PLOT_HI); ax1.set_ylim(SQ_LO, SQ_HI)
    ax1.set_ylabel("squared-domain Hz  (carrier = /2)")
    ax1.set_xlabel("Time within WAV (s)")
    ax1.set_title("squared spectrogram (full-energy) + Viterbi drift ridge(s), burst-gated")
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
