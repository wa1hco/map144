#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Frame-offset station-signature study.

Resolves the open question: does the MSK144 *frame offset* — the sync-word
arrival time within the 72 ms frame, ``onset_sample mod 864`` at 12 kHz — work
as a per-station signature?  Two measurements, on the WSJT-X WAV corpus with
ALL.TXT as ground truth:

  within-station σ   how stable is one station's frame offset frame-to-frame
                     within a period (raw, and after coherent averaging)?
                     → sets the useful precision → N = 864/(k·σ) distinguishable
                       stations.
  between-station    in periods with ≥2 decoded transmitters at separated
                     frequencies, do the two transmitters' frame offsets differ?

Units are 12 kHz samples throughout (NSPM = 864 = 72 ms), per the reference
implementation.  Transmitter = the SECOND callsign in the message (the sender);
CQ → the callsign after CQ.  Only ``Rx MSK144`` lines are used (the corpus is
~84% FT8, which has no 72 ms frame).

Usage:
    python tools/frame_offset_study.py --period 260507_091115   # validate one
    python tools/frame_offset_study.py --list 20                # list test periods
    python tools/frame_offset_study.py --corpus 100             # aggregate study
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scipy.signal import hilbert, fftconvolve, find_peaks    # noqa: E402
from map144_app.msk144_spd import (                          # noqa: E402
    NSPM, FS, _SYNC_CB, _sync_correlate, _sync_phase_features,
)

# Matched-filter kernel: time-reversed conjugate of the 42-sample MSK144 sync
# waveform.  |fftconvolve(signal, this)| peaks at each true sync onset.
_SYNC_MF = np.conj(_SYNC_CB[::-1]).astype(np.complex64)

MAP144_BURSTS = _REPO / "MSK144" / "detections"   # 12 kHz isolated burst WAVs


# ── VALIDATED frame-offset measurement (2026-05-31 study) ────────────────────
# The reliable method: per-burst FREQUENCY SEARCH around 1500 Hz, then
# FRAME-ALIGNED non-overlapping NSPM windows through the decoder's
# _sync_correlate.  ``ish_best`` is the sync position WITHIN the frame = the
# frame offset directly (no window-position term — that was the +18 ms slide
# bug).  Per-frame ``xmax`` is the sync-correlation strength; STRONG frames
# (the pings we optimise) give sub-sample-stable offsets, weak ones scatter.
# Measured on the 11k MAP144 burst WAVs: strong (strength>=6) bursts have
# within-burst σ ≈ 0.6 samples (0.05 ms) → ~360 distinguishable stations.
def clean_burst_offsets(audio: np.ndarray, f_search=range(1450, 1551, 10)):
    """(frame_offsets, strengths, best_freq) for a single isolated burst.

    frame_offsets[i] / strengths[i] are the sync position-in-frame and
    correlation strength for the i-th NSPM frame, at the best-fit carrier."""
    z = hilbert(audio).astype(np.complex64)
    nn = np.arange(len(z), dtype=np.float64)
    best = None
    for f0 in f_search:
        zd = (z * np.exp(-2j * np.pi * f0 * nn / FS)).astype(np.complex64)
        ish, xm = [], []
        for k in range(0, len(zd) - NSPM, NSPM):     # FRAME-ALIGNED, no overlap
            _xcc, xmax, o = _sync_correlate(np.ascontiguousarray(zd[k:k + NSPM]))
            ish.append(int(o)); xm.append(float(xmax))
        xm = np.array(xm, np.float32)
        score = float(np.sort(xm)[-3:].mean()) if len(xm) >= 3 else 0.0
        if best is None or score > best[0]:
            best = (score, np.array(ish), xm, f0)
    return (best[1], best[2], best[3]) if best else (np.array([]), np.array([]), 1500)

WSJTX = Path("/home/jeff/.local/share/WSJT-X - flex")
ALL_TXT = WSJTX / "ALL.TXT"
SAVE = WSJTX / "save"
CALL_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}$")
GRID_RE = re.compile(r"^[A-R]{2}\d\d")


# ── ALL.TXT → periods ─────────────────────────────────────────────────────────
def _tx_call(msg: str) -> str | None:
    """The SENDING station: 2nd callsign in a message, or the call after CQ."""
    toks = msg.split()
    calls = [t for t in toks if CALL_RE.match(t)]
    if not calls:
        return None
    if toks and toks[0] == "CQ":
        return calls[0]
    return calls[1] if len(calls) >= 2 else calls[0]


def load_periods() -> dict:
    """period_key 'YYMMDD_HHMMSS' (15-s grid) → list of (audio_hz, tx_call)."""
    per: dict = defaultdict(list)
    with open(ALL_TXT, errors="ignore") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 7 or p[2] != "Rx" or p[3] != "MSK144":
                continue
            try:
                audio = int(p[6])
            except ValueError:
                continue
            yymmdd, hms = p[0].split("_")
            h, m, s = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
            ps = ((h * 3600 + m * 60 + s) // 15) * 15
            key = "%s_%02d%02d%02d" % (yymmdd, ps // 3600, (ps % 3600) // 60, ps % 60)
            per[key].append((audio, _tx_call(" ".join(p[7:]))))
    return per


def stations_in(lst):
    """{tx_call -> representative audio_hz} for a period (mean freq per tx)."""
    by = defaultdict(list)
    for a, t in lst:
        if t:
            by[t].append(a)
    return {t: int(np.mean(v)) for t, v in by.items()}


# ── WAV → per-frame frame offsets ─────────────────────────────────────────────
def load_wav_12k(path: Path) -> np.ndarray:
    w = wave.open(str(path))
    assert w.getframerate() == 12000 and w.getnchannels() == 1
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def analytic_dc(audio: np.ndarray, audio_hz: float) -> np.ndarray:
    """Analytic signal mixed so the carrier at ``audio_hz`` lands at DC."""
    z = hilbert(audio).astype(np.complex64)
    n = np.arange(len(z), dtype=np.float64)
    return (z * np.exp(-2j * np.pi * audio_hz * n / FS)).astype(np.complex64)


def find_bursts(z_dc: np.ndarray, k_thresh: float = 4.0,
                min_ms: float = 200.0, max_ms: float = 800.0):
    """Bursts of `min_ms`..`max_ms` duration from the signal ENVELOPE (not the
    sync correlation).  These are the overdense-ping / scatter events we
    optimise for — strong locals run continuous and fall outside the window."""
    power = (z_dc.real ** 2 + z_dc.imag ** 2).astype(np.float32)
    win = NSPM // 2
    env = np.convolve(power, np.ones(win, np.float32) / win, mode="same")
    nf = float(np.median(env)) + 1e-12
    above = env > k_thresh * nf
    bursts, i, N = [], 0, len(above)
    while i < N:
        if above[i]:
            j = i
            while j < N and above[j]:
                j += 1
            dur = (j - i) * 1000.0 / FS
            if min_ms <= dur <= max_ms:
                bursts.append((i, j, dur))
            i = j
        else:
            i += 1
    return bursts


def burst_frame_offsets(z_dc: np.ndarray, burst):
    """Per-frame sync onsets within one burst, via matched filter.

    `|fftconvolve(z_dc[burst], _SYNC_MF)|` peaks at each true sync onset (one
    per 72 ms frame).  We find those peaks INDEPENDENTLY (find_peaks, min
    spacing ~0.7 frame) and take each absolute position mod NSPM — so whether
    they cluster (one frame offset) or scatter is *measured*, not assumed.
    Returns list of (frame_offset_samples, peak_magnitude).
    """
    i, j, _ = burst
    if j - i < NSPM:
        return []
    mf = np.abs(fftconvolve(z_dc[i:j], _SYNC_MF, mode="same"))
    pk, props = find_peaks(mf, distance=int(NSPM * 0.7), height=0.4 * mf.max())
    return [(int((i + p) % NSPM), float(mf[p])) for p in pk]


def _circ_stats(fo_samples):
    """Circular mean/std (mod NSPM) of frame offsets, returned in 12 kHz samples."""
    ang = np.asarray(fo_samples) * (2 * np.pi / NSPM)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    R = np.hypot(c, s)
    mean = (np.arctan2(s, c) % (2 * np.pi)) * NSPM / (2 * np.pi)
    std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-9)))) * NSPM / (2 * np.pi)  # circular σ
    return mean, std, R


# ── modes ─────────────────────────────────────────────────────────────────────
def _burst_offset(z_dc, burst):
    """(median frame_offset, within-burst σ, R, n_frames) for one burst, or None
    if it doesn't hold enough sync peaks."""
    offs = burst_frame_offsets(z_dc, burst)
    if len(offs) < 3:
        return None
    mean, std, R = _circ_stats([o for o, _ in offs])
    return mean, std, R, len(offs)


def run_period(key: str, per: dict, coh_thresh: float):
    if key not in per:
        print(f"period {key} not in ALL.TXT MSK144 decodes"); return
    wav = SAVE / (key + ".wav")
    if not wav.exists():
        print(f"no WAV for {key}"); return
    audio = load_wav_12k(wav)
    stns = stations_in(per[key])
    print(f"=== {key}  ({wav.name})  stations: "
          + "  ".join(f"{t}@{f}Hz" for t, f in stns.items()))
    for tx, fhz in stns.items():
        z = analytic_dc(audio, fhz)
        bursts = find_bursts(z)
        if not bursts:
            print(f"  {tx:8s}@{fhz}Hz : no 200-800 ms burst")
            continue
        for b in bursts:
            r = _burst_offset(z, b)
            if r is None:
                print(f"  {tx:8s}@{fhz}Hz t={b[0]/FS:5.1f}s dur={b[2]:3.0f}ms : <3 sync peaks")
                continue
            mean, std, R, nf = r
            print(f"  {tx:8s}@{fhz}Hz t={b[0]/FS:5.1f}s dur={b[2]:3.0f}ms  {nf} frames  "
                  f"frame_offset={mean*1000/FS:6.2f}ms  σ={std*1000/FS:.3f}ms "
                  f"({std:.2f}samp)  R={R:.2f}")


def run_corpus(limit: int, per: dict, coh_thresh: float):
    # Richest periods first (most stations, most decodes) — that's where the
    # pings are, so the scan finds qualifying bursts efficiently.
    cands = sorted((k for k in per if (SAVE / (k + ".wav")).exists()),
                   key=lambda k: -(len(stations_in(per[k])) * 100 + len(per[k])))
    print(f"scanning up to {limit} station-rich periods for 200-800 ms bursts (R>0.6)…")
    within, by_period_tx = [], defaultdict(list)
    nb = 0
    for k in cands[:limit]:
        audio = load_wav_12k(SAVE / (k + ".wav"))
        for tx, fhz in stations_in(per[k]).items():
            z = analytic_dc(audio, fhz)
            for b in find_bursts(z):
                r = _burst_offset(z, b)
                if r is None:
                    continue
                mean, std, R, nf = r
                if R > 0.6 and nf >= 3:        # a real, concentrated burst
                    nb += 1
                    within.append(std)
                    by_period_tx[(k, tx)].append(mean)
    # between-station: periods with >=2 distinct tx that each produced a burst
    between = []
    by_period = defaultdict(dict)
    for (k, tx), means in by_period_tx.items():
        by_period[k][tx] = float(np.median(means))   # station's frame offset
    for k, txm in by_period.items():
        txs = list(txm)
        for a in range(len(txs)):
            for b in range(a + 1, len(txs)):
                d = abs(txm[txs[a]] - txm[txs[b]])
                between.append(min(d, NSPM - d))
    print(f"qualified bursts: {nb}")
    if within:
        ws = np.array(within)
        print(f"WITHIN-station σ:  n={len(ws)}  median={np.median(ws):.2f} samp "
              f"({np.median(ws)*1000/FS:.3f} ms)  p90={np.percentile(ws,90):.2f} samp")
    if between:
        bs = np.array(between)
        print(f"BETWEEN-station sep: n={len(bs)}  median={np.median(bs):.1f} samp "
              f"({np.median(bs)*1000/FS:.2f} ms)  p10={np.percentile(bs,10):.1f} samp")
    if within:
        ms = np.median(within)
        print(f"=> N ≈ {NSPM/(4*max(ms,1e-3)):.0f} distinguishable stations (k=4·σ_median)")


def run_map144(limit: int):
    """Within-station σ on MAP144's isolated, station-labelled burst WAVs —
    the clean per-station dataset (no overlap, pre-shifted to 1500 Hz)."""
    import glob
    import random
    files = sorted(glob.glob(str(MAP144_BURSTS / "*.wav")))
    random.seed(1)
    random.shuffle(files)
    sig = []
    for f in files[:limit]:
        try:
            w = wave.open(f)
            if w.getframerate() != 12000:
                continue
            audio = np.frombuffer(w.readframes(w.getnframes()),
                                  np.int16).astype(np.float32) / 32768.0
        except Exception:
            continue
        ish, xm, _f0 = clean_burst_offsets(audio)
        if len(ish) < 4:
            continue
        hi = ish[xm > float(np.median(xm)) * 1.8]
        if len(hi) < 4:
            continue
        _m, std, R = _circ_stats(hi)
        strength = float(np.sort(xm)[-3:].mean())
        if strength >= 6 and std < 2.0 and R > 0.5:      # clean strong ping
            sig.append(std)
    sig = np.array(sig)
    if len(sig):
        ms = float(np.median(sig))
        print(f"clean strong MAP144 bursts: n={len(sig)}  within-station σ "
              f"median={ms:.2f} samp ({ms*1000/FS:.3f} ms)  p90={np.percentile(sig,90):.2f}")
        print(f"=> sub-sample stable; N ≈ {NSPM/(4*max(ms,0.01)):.0f} "
              f"distinguishable stations (k=4·σ)")
    else:
        print("no clean strong bursts found in sample")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", help="validate one period YYMMDD_HHMMSS")
    ap.add_argument("--list", type=int, metavar="N", help="list N multi-station periods")
    ap.add_argument("--corpus", type=int, metavar="N", help="aggregate over N periods")
    ap.add_argument("--map144", type=int, metavar="N",
                    help="within-station σ on N MAP144 burst WAVs (the clean dataset)")
    ap.add_argument("--coh", type=float, default=0.40, help="coherence threshold")
    args = ap.parse_args()
    if args.map144:
        run_map144(args.map144)
        return
    per = load_periods()
    if args.period:
        run_period(args.period, per, args.coh)
    elif args.list:
        shown = 0
        for k, lst in per.items():
            stns = stations_in(lst)
            if len(stns) >= 2 and (SAVE / (k + ".wav")).exists():
                print(k, " ".join(f"{t}@{f}" for t, f in stns.items()))
                shown += 1
                if shown >= args.list:
                    break
    elif args.corpus:
        run_corpus(args.corpus, per, args.coh)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
