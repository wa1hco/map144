#!/usr/bin/env python3
"""
sync_coherence_probe.py -- EXPLORATORY (development mode).

Tests the user's design: for the 260511_093000 K0TPP ping, is the +20 Hz
(audio ~1520) component a real MSK144 sync-bearing signal (a coherent path),
or noise?

Method (project's own metrics, no homemade math):
  for each candidate carrier f:
      mix analytic signal to baseband at f
      slide an NSPM (72 ms) window across the burst
      _sync_correlate -> (xmax, ish)          [sync matched-filter peak + frame offset]
      _sync_phase_features -> coherence in [0,1]   (~1 real sync, ~0.109 noise)
  report, per f, the coherence + frame offset at the peak-xmax window.

Reference: K0TPP decoded at audio 1606 (DT 12.4 s). That carrier is the
known-good control -- coherence there should be clearly above the ~0.11 noise
floor. The +20 Hz component sits at ~1520. Same frame offset as 1606 => one
station / two paths (multipath). Different => independent. Receiver had a ~+100
Hz calibration offset, common to all signals, so it cancels in the relative test.

Output: scratch/ridge_proto/sync_coherence_260511_093000.png
"""
import sys
import wave
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from map144_app.msk144_spd import _sync_correlate, _sync_phase_features, NSPM, FS  # noqa: E402

WAV = "/home/jeff/.local/share/WSJT-X - flex/save/260511_093000.wav"
OUT = "scratch/ridge_proto/sync_coherence_260511_093000.png"

NOISE_FLOOR = 0.109          # coherence of pure noise (1/sqrt(2*42))
BURST_LO, BURST_HI = 12.28, 12.43     # burst span (s); NSPM window slides within
NOISE_LO, NOISE_HI = 11.50, 12.10     # signal-free control span (s)
F_LO, F_HI, F_STEP = 1440, 1720, 4    # candidate carriers (received audio Hz)
SLIDE_STEP = 12                       # window slide step (samples)


def load_audio(path):
    w = wave.open(path)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    w.close()
    return a


def scan_freq(z, n, f, s_lo, s_hi):
    """Best-xmax NSPM window in [s_lo, s_hi); return its (xmax, ish, coherence)."""
    zf = (z * np.exp(-2j * np.pi * f * n / FS)).astype(np.complex64)
    best = (-1.0, 0, 0.0)
    for start in range(s_lo, s_hi - NSPM, SLIDE_STEP):
        frame = np.ascontiguousarray(zf[start:start + NSPM])
        _xcc, xmax, ish = _sync_correlate(frame)
        if xmax > best[0]:
            coh, _ts, _fo = _sync_phase_features(frame, ish)
            best = (float(xmax), int(ish), float(coh))
    return best


def main():
    audio = load_audio(WAV)
    z = hilbert(audio).astype(np.complex64)
    n = np.arange(len(z), dtype=np.float64)

    b_lo, b_hi = int(BURST_LO * FS), int(BURST_HI * FS)
    nz_lo, nz_hi = int(NOISE_LO * FS), int(NOISE_HI * FS)

    freqs = np.arange(F_LO, F_HI + 1, F_STEP)
    xmax_b = np.zeros(len(freqs))
    coh_b = np.zeros(len(freqs))
    ish_b = np.zeros(len(freqs), dtype=int)
    coh_n = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        xmax_b[i], ish_b[i], coh_b[i] = scan_freq(z, n, f, b_lo, b_hi)
        _xn, _in, coh_n[i] = scan_freq(z, n, f, nz_lo, nz_hi)

    # report the two components of interest
    def at(target):
        j = int(np.argmin(np.abs(freqs - target)))
        return freqs[j], xmax_b[j], coh_b[j], ish_b[j]

    print("=== sync_coherence_probe: 260511_093000.wav (K0TPP ping) ===")
    print(f"noise-floor coherence reference: {NOISE_FLOOR:.3f}")
    f1, x1, c1, i1 = at(1606)   # K0TPP decode (control)
    f2, x2, c2, i2 = at(1520)   # +20 Hz component
    print(f"  K0TPP carrier  ~{f1} Hz : xmax {x1:.1f}  coherence {c1:.3f}  frame_offset(ish) {i1}")
    print(f"  +20 component  ~{f2} Hz : xmax {x2:.1f}  coherence {c2:.3f}  frame_offset(ish) {i2}")
    print(f"  frame-offset delta (i_1520 - i_1606): {i2 - i1} samples "
          f"({(i2 - i1) / FS * 1000:.2f} ms)")
    # peak coherence across the scan (where does sync look most real?)
    jpk = int(np.argmax(coh_b))
    print(f"  peak burst coherence {coh_b[jpk]:.3f} at {freqs[jpk]} Hz (ish {ish_b[jpk]})")
    print(f"  median noise-region coherence across freqs: {np.median(coh_n):.3f}")

    # ---- plot ----
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0.plot(freqs, coh_b, "-o", ms=3, color="C0", label="burst (sync coherence)")
    ax0.plot(freqs, coh_n, "-", color="0.6", lw=1, label="noise region (control)")
    ax0.axhline(NOISE_FLOOR, color="red", ls=":", lw=1, label=f"noise floor {NOISE_FLOOR:.3f}")
    ax0.axvline(1606, color="cyan", ls="--", lw=1, label="K0TPP decode 1606")
    ax0.axvline(1520, color="orange", ls="--", lw=1, label="+20 component 1520")
    ax0.set_ylabel("sync phase coherence  [0,1]")
    ax0.set_title("260511_093000  K0TPP ping  |  sync coherence vs candidate carrier "
                  "(burst-gated, NSPM window)")
    ax0.legend(fontsize=8, ncol=2)
    ax0.set_ylim(0, max(0.5, coh_b.max() * 1.15))

    ax1.plot(freqs, xmax_b, "-o", ms=3, color="C2", label="burst sync xmax")
    ax1.axvline(1606, color="cyan", ls="--", lw=1)
    ax1.axvline(1520, color="orange", ls="--", lw=1)
    ax1.set_ylabel("sync xmax (matched-filter peak)")
    ax1.set_xlabel("candidate carrier (received audio Hz)")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
