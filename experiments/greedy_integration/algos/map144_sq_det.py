# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAP144's current sq_det formula, faithful port for the experiment.

Mirrors map144_app/blocks/detector_block.py:_compute_sq_metric_db and
the constants in map144_app/processing.py.  This is the BASELINE
detector that Experiment A compares against the WSJT-X formula.

Differs from WSJT-X (algos/sq_det.py) in five ways recorded in
[[project_wsjtx_sq_det_formula]]:

    1. NFFT          = 512    (vs 864 in WSJT-X)
    2. Window        = Hanning (vs rectangular in WSJT-X)
    3. Input domain  = complex baseband (vs analytic at audio carrier)
    4. Combine rule  = (lo + hi) / 2  (vs max(lo, hi))
    5. Baseline      = rolling 25th-pctile of 300 prior FFTs
                       (vs per-block 25th-pctile from current slide trace)

Output is in dB (``10*log10(raw_lin / pct25)``) with detection threshold
3.5 dB.  Note: 3.5 in MAP144's dB form is NOT directly comparable to
WSJT-X's "3 in linear amplitude ratio" — they're different scales.

For the experiment, both algorithms get the SAME underlying signal but
in their natural input domain (complex baseband for MAP144, analytic at
audio carrier for WSJT-X).  Detection rate vs SNR is the apples-to-apples
comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Constants from map144_app/processing.py ──────────────────────────────────
CH_SAMPLE_RATE      = 12000      # Hz (same as WSJT-X)
CH_DETECT_SIZE      = 512        # MAP144's FFT length (different!)
DF_MAP144           = float(CH_SAMPLE_RATE) / CH_DETECT_SIZE  # 23.4375 Hz/bin
DETECT_THRESHOLD_DB = 3.5        # MAP144's threshold (in dB)

# Squared-signal target lines at ±1000 Hz from DC (signal is at 0 Hz baseband).
# Same physics as WSJT-X's 2000/4000 Hz in the audio-carrier convention.
_SQ_TONE_HZ = 1000.0
_SQ_NTOL_HZ = 200.0    # ±200 Hz search half-width around each tone

# Pre-computed: Hanning window and tone search masks.
_DETECT_WINDOW = np.hanning(CH_DETECT_SIZE).astype(np.float64)
# Use unshifted fftfreq so masks index directly into the raw FFT output
# (no fftshift in the hot loop).  Same convention as the legacy code.
_SQ_FREQ = np.fft.fftfreq(CH_DETECT_SIZE, 1.0 / CH_SAMPLE_RATE)
_LO_MASK = ((_SQ_FREQ >= -_SQ_TONE_HZ - _SQ_NTOL_HZ) &
            (_SQ_FREQ <= -_SQ_TONE_HZ + _SQ_NTOL_HZ))
_HI_MASK = ((_SQ_FREQ >=  _SQ_TONE_HZ - _SQ_NTOL_HZ) &
            (_SQ_FREQ <=  _SQ_TONE_HZ + _SQ_NTOL_HZ))

# Rolling history depth for the percentile baseline.  MAP144 uses 300
# prior FFTs (= 300 detection windows × 21 ms hop ≈ 6.4 s of history).
_METRIC_HIST_DEPTH = 300


@dataclass
class Map144SqDetResult:
    """Per-frame MAP144 sq_det output (no normalisation applied yet)."""
    raw_lin: float       # average of (lo_peak + hi_peak) / 2 amplitudes
    lo_peak: float       # peak amplitude in lo-tone (-1000 ± 200 Hz) mask
    hi_peak: float       # peak amplitude in hi-tone (+1000 ± 200 Hz) mask


def map144_sq_det_per_frame(cbb: np.ndarray) -> Map144SqDetResult:
    """Compute MAP144 per-frame sq_det metric on one CH_DETECT_SIZE=512
    complex-baseband frame.

    Parameters
    ----------
    cbb : complex array, shape (512,)
        Complex baseband frame at FS=12 kHz.  Signal already mixed to DC
        (e.g. by the channeliser).  MSK signal has tones at ±half-baud
        ≈ ±500 Hz from DC; squaring puts the squared lines at ±1000 Hz.

    Returns
    -------
    Map144SqDetResult with raw_lin / lo_peak / hi_peak (all linear amplitude).
    Use ``map144_detection_db()`` to convert to dB and compare to threshold.
    """
    if cbb.shape[0] != CH_DETECT_SIZE:
        raise ValueError(f"cbb must be CH_DETECT_SIZE={CH_DETECT_SIZE} samples; "
                         f"got {cbb.shape[0]}")

    # Square the complex baseband.  For complex z, z² has spectral content
    # at twice z's frequencies.  MSK at fc=0 ± half-baud → squared lines
    # at ±2 × half-baud = ±1000 Hz (the _SQ_TONE_HZ targets).
    sq = (cbb.astype(np.complex128)) ** 2

    # FFT with Hanning window.  Note: MAP144 uses linear amplitude (|FFT|/N),
    # not power (|FFT|²).  Different units from WSJT-X.
    spec = np.fft.fft(sq * _DETECT_WINDOW)
    power_lin = np.abs(spec) / CH_DETECT_SIZE

    lo_peak = float(power_lin[_LO_MASK].max())
    hi_peak = float(power_lin[_HI_MASK].max())
    raw_lin = (lo_peak + hi_peak) / 2.0

    return Map144SqDetResult(raw_lin=raw_lin, lo_peak=lo_peak, hi_peak=hi_peak)


def map144_detection_db(raw_lin: float, pct25_baseline: float) -> float:
    """Convert raw_lin to dB above the rolling 25th-percentile baseline.

    Returns ``10 * log10(raw_lin / pct25_baseline)``.  Detection fires
    if this >= DETECT_THRESHOLD_DB (3.5).
    """
    return float(10.0 * np.log10(max(raw_lin, 1e-30) /
                                  max(pct25_baseline, 1e-30)))


# ── Block / sweep helpers ────────────────────────────────────────────────────

@dataclass
class Map144TriggerTrace:
    """Result of running MAP144 sq_det at every slide position across a
    complex-baseband block, then computing the rolling-percentile baseline
    and thresholding."""
    raw_lin:     np.ndarray   # amplitude per slide position
    pct25:       float        # 25th-percentile baseline used for normalisation
    metric_db:   np.ndarray   # 10*log10(raw_lin / pct25)
    triggers:    np.ndarray   # boolean: metric_db >= DETECT_THRESHOLD_DB
    t_centers:   np.ndarray   # centre time (s) of each slide position


def slide_map144_sq_det(cbb: np.ndarray,
                        stride: int = CH_DETECT_SIZE // 2,
                        pct25_override: Optional[float] = None,
                        ) -> Map144TriggerTrace:
    """Slide MAP144 sq_det through ``cbb`` (complex baseband).

    Default stride is CH_DETECT_SIZE/2 = 256 samples = 21.3 ms (50% overlap),
    matching the legacy MAP144 detector hop.

    pct25_override: if supplied, use this as the baseline (e.g. computed from
    a noise-only reference segment for fair experimental comparison).
    Otherwise the 25th-percentile is computed from the slide trace itself
    (analogous to WSJT-X's xmed approach).
    """
    n = cbb.shape[0]
    if n < CH_DETECT_SIZE:
        raise ValueError(f"need at least CH_DETECT_SIZE={CH_DETECT_SIZE} "
                         f"samples; got {n}")
    n_step = max(1, (n - CH_DETECT_SIZE) // stride)

    raw_lin   = np.empty(n_step, dtype=np.float64)
    t_centers = np.empty(n_step, dtype=np.float64)
    for k in range(n_step):
        ns = k * stride
        ne = ns + CH_DETECT_SIZE
        if ne > n:
            break
        r = map144_sq_det_per_frame(cbb[ns:ne])
        raw_lin[k]   = r.raw_lin
        t_centers[k] = (ns + CH_DETECT_SIZE / 2.0) / CH_SAMPLE_RATE

    if pct25_override is not None:
        pct25 = max(float(pct25_override), 1e-30)
    else:
        pct25 = max(float(np.percentile(raw_lin, 25)), 1e-30)
    metric_db = 10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30))
    triggers = metric_db >= DETECT_THRESHOLD_DB

    return Map144TriggerTrace(
        raw_lin=raw_lin, pct25=pct25, metric_db=metric_db,
        triggers=triggers, t_centers=t_centers,
    )


def audio_to_complex_baseband(audio_real: np.ndarray, fc_hz: float = 1500.0,
                              fs_hz: float = CH_SAMPLE_RATE) -> np.ndarray:
    """Convert real audio at carrier fc to complex baseband (signal at 0 Hz).

    For Experiment A we feed each algorithm its natural input format:
    WSJT-X gets analytic-at-carrier, MAP144 gets baseband.  This function
    converts the same source audio to MAP144's expected input domain.

    Method: Hilbert transform → analytic, then mix down by fc.
    """
    from scipy.signal import hilbert
    analytic = hilbert(audio_real.astype(np.float64))
    t = np.arange(len(analytic)) / fs_hz
    bb = analytic * np.exp(-1j * 2.0 * np.pi * fc_hz * t)
    return bb.astype(np.complex128)
