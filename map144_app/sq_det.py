# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""WSJT-X sq_det formula for MSK144, ported from msk144spd.f90 lines 23-138.

The shared metric used by every algorithm in this experiment.  Reproduces
the WSJT-X reference EXACTLY so comparisons are apples-to-apples.

Two-stage flow (matches the Fortran)
------------------------------------
1. Per-frame metric (``sq_det()``):
       cdat   ← analytic-signal slice, length NSPM=864
       ctmp   = cdat ** 2                        # complex squaring
       spec   = |FFT_864(ctmp)| ** 2             # power spectrum of squared signal
       ah, al = peaks in 4000 Hz / 2000 Hz mask windows  (±2·ntol bins each)
       ahavp, alavp = peak-excluded in-band averages   (LOCAL noise floor)
       returns SqDetResult with:
         detmet  = max(ah, al)                  # raw amplitude (primary)
         detmet2 = max(ah/ahavp, al/alavp)      # per-frame peak/local ratio (fallback)

2. Block normalization + threshold (``normalize_and_threshold()``):
       given N raw detmet values across a sliding window,
       xmed = 25th-percentile of detmet across the block
       detmet_norm = detmet / xmed              # noise floor ≡ 1.0
       DETECT where detmet_norm >= 3.0          # ← THE WSJT-X THRESHOLD

The "3" is NOT 3× the in-band noise floor (~9.5 dB); it's 3× the
25th-percentile of the peak-amplitude trace across the analysis block.
Noise excursions on that trace typically reach 2-2.5; threshold 3 is
just above them — about 1-2 dB above noise peaks.  Very sensitive.

WSJT-X fallback (if too few candidates fire the detmet>=3 test):
    detect where detmet2 >= 12.0                # the in-band ratio path
This is the higher-threshold per-frame check; used as a backup.  We
expose both modes; experiments should use ``detmet_norm >= 3`` first.

Input
-----
    cdat (complex array, length NSPM=864)
        Analytic-signal slice of the audio at 12 kHz.  WSJT-X gets this
        from ``call analytic(real_audio, ..., cdat, ...)``; we accept it
        directly as the input contract.
    fc (float, default 1500)
        Audio carrier frequency in Hz.  Squared lines fall at 2·(fc-500)
        and 2·(fc+500) — 2000 and 4000 Hz for the nominal 1500 Hz center.
    ntol (float, default 100)
        Frequency tolerance.  Search window for each squared tone is
        ±2·ntol bins around its target.

Output (``SqDetResult``)
------------------------
    detmet, detmet2 — see above
    ah, al           — individual hi/lo tone peak amplitudes
    ahavp, alavp     — individual hi/lo tone local noise floors
    ferr_hi, ferr_lo — sub-bin freq error per tone (Hz at audio rate)

The bare ``.detected`` property is intentionally NOT defined on
SqDetResult, because a single-frame answer to "did this detect?" needs
the block normalization context.  Use ``normalize_and_threshold()``
across a sequence of frames to get the boolean trigger times.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


NSPM        = 864       # MSK144 frame size in samples at 12 kHz
NFFT        = NSPM      # FFT length for sq_det (matches WSJT-X)
FS          = 12000.0
DF          = FS / NFFT       # bin width = 13.888... Hz
DEFAULT_FC  = 1500.0    # default audio carrier
DEFAULT_NTOL = 100.0    # default freq tolerance (WSJT-X "ntol")

# ── UNIT CONVENTION ─────────────────────────────────────────────────────────
# Internal math is LINEAR ratios (matches WSJT-X msk144spd.f90).
# All metrics are POWER ratios in the |FFT|² domain (tonespec = |FFT(s²)|²).
# Threshold "3.0" is a power ratio of 3 in that domain.
#
# Variable names declare their units:
#   _lin   suffix (or no suffix on a known-linear value): power ratio
#   _db    suffix: 10*log10(linear power ratio) = "power dB"
#
# The dB form is a derived view of the linear value — same physical quantity,
# different presentation.  Never compute thresholds in dB to compare against
# linear metrics, or vice versa.  Use to_db()/to_lin() at the boundary only.
#
# Caveat: this "power dB" is dB of the SQUARED-SIGNAL's power spectrum, which
# scales as the 4th power of the underlying audio amplitude.  When comparing
# to other dB metrics (e.g. sync correlator's amplitude-dB), convert through
# linear domain — don't subtract dB values directly.  See
# project_detection_metric_db_scale_mismatch.md.


def to_db(x_lin: float | np.ndarray) -> float | np.ndarray:
    """Convert linear ratio to dB.  Guarded at 1e-30 to avoid -inf."""
    return 10.0 * np.log10(np.maximum(x_lin, 1e-30))


def to_lin(x_db: float | np.ndarray) -> float | np.ndarray:
    """Convert dB to linear ratio."""
    return np.power(10.0, x_db / 10.0)


# WSJT-X threshold on the BLOCK-NORMALIZED detmet (= detmet / 25th-percentile).
# Noise excursions on the normalised trace reach ~2.0-2.5; threshold 3 is
# only marginally above noise — a very sensitive detector.
DETECT_THRESHOLD_NORM_LIN    = 3.0     # primary: detmet/xmed >= 3.0  (WSJT-X canonical)
DETECT_THRESHOLD_NORM_DB     = float(to_db(DETECT_THRESHOLD_NORM_LIN))   # ≈ 4.77 dB
DETECT_THRESHOLD_DETMET2_LIN = 12.0    # fallback: per-frame peak/local-noise >= 12
DETECT_THRESHOLD_DETMET2_DB  = float(to_db(DETECT_THRESHOLD_DETMET2_LIN)) # ≈ 10.79 dB

# Back-compat aliases — existing tests + experiment scripts reference these
# bare names; keep them as aliases of the _LIN values.
DETECT_THRESHOLD_NORM    = DETECT_THRESHOLD_NORM_LIN
DETECT_THRESHOLD_DETMET2 = DETECT_THRESHOLD_DETMET2_LIN

PCT_FOR_XMED             = 25.0   # 25th percentile is the WSJT-X noise reference


@dataclass
class SqDetResult:
    """Per-frame sq_det output (no normalisation applied yet).

    Use ``normalize_and_threshold()`` across a sequence to get the
    boolean detection trace WSJT-X uses.

    Units: all *_lin fields are linear (power ratios in |FFT|² domain);
    *_db fields are 10·log10 views of the same linear values.  See the
    UNIT CONVENTION comment at the top of this file.
    """
    detmet:  float    # raw peak amplitude — max(ah, al) — linear, not normalised
    detmet2: float    # per-frame peak/local-noise ratio — max(ah/ahavp, al/alavp) — LINEAR
    ah:      float    # hi-tone (4000 Hz region) peak amplitude — linear
    al:      float    # lo-tone (2000 Hz region) peak amplitude — linear
    ahavp:   float    # hi-tone local in-band noise floor (peak-excluded) — linear
    alavp:   float    # lo-tone local in-band noise floor — linear
    ferr_hi: float    # hi-tone frequency error estimate (Hz, sub-bin)
    ferr_lo: float    # lo-tone frequency error estimate (Hz, sub-bin)

    @property
    def detmet2_db(self) -> float:
        """Per-frame metric in power-dB (= 10·log10(detmet2)).

        Threshold for detmet2 is 12.0 linear ≡ 10.79 dB.  Use this view
        for human reading or for cross-checking against dB-scale plots;
        algorithmic comparisons should use ``detmet2`` directly to avoid
        cumulative log/exp round-off.
        """
        return float(to_db(self.detmet2))


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-bin peak position via 3-point parabolic interpolation.

    Returns an offset in (-0.5, +0.5) bin-units from index i.  Matches
    the WSJT-X ``deltah`` / ``deltal`` calculation (msk144spd.f90 ~line 100).
    Returns 0 at the array boundaries.
    """
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    y_lo, y_pk, y_hi = float(y[i-1]), float(y[i]), float(y[i+1])
    denom = (y_lo - 2*y_pk + y_hi)
    if abs(denom) < 1e-30:
        return 0.0
    return 0.5 * (y_lo - y_hi) / denom


def sq_det(cdat: np.ndarray, fc: float = DEFAULT_FC,
           ntol: float = DEFAULT_NTOL) -> SqDetResult:
    """Compute WSJT-X sq_det metric on one NSPM-length analytic-signal frame.

    Parameters
    ----------
    cdat : complex64/complex128 array, shape (NSPM,)
        One frame of analytic signal at FS=12 kHz with audio carrier at fc.
    fc : float
        Audio carrier frequency in Hz (default 1500).
    ntol : float
        Frequency tolerance in Hz at the audio carrier (default 100).
        Search window for each squared tone is ±2*ntol Hz.

    Returns
    -------
    SqDetResult with detmet2 as the value to threshold at 3.

    Matches WSJT-X ``msk144spd.f90`` line 86-117 ah/al/ahavp/alavp computation.
    """
    if cdat.shape[0] != NSPM:
        raise ValueError(f"cdat must be NSPM={NSPM} samples; got {cdat.shape[0]}")

    # Square the analytic signal.  For complex z = A·exp(j·2π·fc·t), z² has
    # a strong component at 2·fc with magnitude |A|².  The MSK ±half-baud
    # tones at fc±500 Hz become squared lines at 2·(fc-500) and 2·(fc+500).
    ctmp = cdat.astype(np.complex128) ** 2

    # FFT — same length as the frame, NO WINDOW (rectangular).
    #
    # Why no window?  At NFFT=NSPM=864 with fs=12 kHz, every interesting
    # frequency falls on an exact bin centre:
    #   500 Hz → bin 36     1000 Hz → bin 72
    #   2000 Hz → bin 144   3000 Hz → bin 216    4000 Hz → bin 288
    # Integer-cycle alignment ⇒ zero spectral leakage ⇒ a window would
    # only LOSE peak amplitude (Hanning costs ~1.5-2 dB peak/noise on
    # on-bin tones with no benefit).  Rectangular is optimal here.
    spec = np.fft.fft(ctmp)
    tonespec = np.abs(spec) ** 2

    # Target frequencies in the squared spectrum
    nfhi = 2.0 * (fc + 500.0)
    nflo = 2.0 * (fc - 500.0)

    # Locate target bins via argmin of |freqs - target|.  This correctly
    # handles BOTH positive targets (e.g. WSJT-X fc=1500 → 2000/4000 Hz
    # both positive) AND signed targets (e.g. fc=0 baseband → ±1000 Hz,
    # where the −1000 Hz bin lives in the back half of the FFT output).
    freqs = np.fft.fftfreq(NFFT, 1.0 / FS)
    i_hi = int(np.argmin(np.abs(freqs - nfhi)))
    i_lo = int(np.argmin(np.abs(freqs - nflo)))
    bw = int(round(2.0 * ntol / DF))   # ±2*ntol bins around target

    hi_lo_idx = max(0, i_hi - bw)
    hi_hi_idx = min(NFFT, i_hi + bw + 1)
    lo_lo_idx = max(0, i_lo - bw)
    lo_hi_idx = min(NFFT, i_lo + bw + 1)

    hi_band = tonespec[hi_lo_idx:hi_hi_idx]
    lo_band = tonespec[lo_lo_idx:lo_hi_idx]

    # Peak in each band
    hi_pk_local = int(np.argmax(hi_band))
    lo_pk_local = int(np.argmax(lo_band))
    ah = float(hi_band[hi_pk_local])
    al = float(lo_band[lo_pk_local])

    # In-band noise floor, peak-excluded (matches WSJT-X sum-then-subtract)
    n_hi = max(1, len(hi_band) - 1)
    n_lo = max(1, len(lo_band) - 1)
    ahavp = float(np.sum(hi_band) - ah) / n_hi
    alavp = float(np.sum(lo_band) - al) / n_lo

    # Sub-bin freq error via parabolic interpolation, converted to Hz at
    # the *audio* carrier.  WSJT-X divides by 2 because the squared spectrum
    # is at 2*fc — a bin shift there equals half a bin at the audio rate.
    # Use ``freqs[bin]`` (signed) instead of ``bin * DF`` so negative-frequency
    # bins (e.g. -1000 Hz baseband at fft index 792) read out correctly.
    deltah = _parabolic_peak(hi_band, hi_pk_local)
    deltal = _parabolic_peak(lo_band, lo_pk_local)
    ihpk_abs = hi_lo_idx + hi_pk_local
    ilpk_abs = lo_lo_idx + lo_pk_local
    ferr_hi = float(freqs[ihpk_abs] + deltah * DF - nfhi) / 2.0
    ferr_lo = float(freqs[ilpk_abs] + deltal * DF - nflo) / 2.0

    detmet  = max(ah, al)
    if ahavp > 0 and alavp > 0:
        detmet2 = max(ah / ahavp, al / alavp)
    elif ahavp > 0:
        detmet2 = ah / ahavp
    elif alavp > 0:
        detmet2 = al / alavp
    else:
        detmet2 = 0.0

    return SqDetResult(
        detmet=detmet, detmet2=detmet2,
        ah=ah, al=al, ahavp=ahavp, alavp=alavp,
        ferr_hi=ferr_hi, ferr_lo=ferr_lo,
    )


# ── Block normalisation + threshold ─────────────────────────────────────────

@dataclass
class TriggerTrace:
    """Result of running sq_det at every slide position across an analytic block,
    then normalising and thresholding the way WSJT-X does in msk144spd.f90.

    Units: ``detmet_norm`` is the LINEAR power ratio (= 3.0 at WSJT-X
    threshold).  ``detmet_norm_db`` is the same in power-dB (= 4.77 dB
    at threshold).  Trigger boolean is computed from the linear values
    to match WSJT-X exactly; the dB view is for human reading / plots.
    """
    detmet_raw:  np.ndarray   # raw max(ah,al) per slide position — linear, un-normalised
    detmet2_raw: np.ndarray   # raw per-frame peak/local ratio per slide position — linear
    xmed:        float        # 25th-percentile of detmet_raw (the "noise floor")
    detmet_norm: np.ndarray   # detmet_raw / xmed (noise floor ≡ 1.0) — LINEAR, threshold 3.0
    triggers:    np.ndarray   # boolean: detmet_norm >= 3.0 (or detmet2 >= 12 fallback)
    t_centers:   np.ndarray   # centre time (s) of each slide position
    ferr:        np.ndarray   # per-position freq-error estimate (Hz)

    @property
    def detmet_norm_db(self) -> np.ndarray:
        """detmet_norm in power-dB (= 10·log10(detmet_norm)).  Threshold ≈ 4.77 dB."""
        return to_db(self.detmet_norm)

    @property
    def detmet2_db(self) -> np.ndarray:
        """detmet2_raw in power-dB.  Threshold ≈ 10.79 dB."""
        return to_db(self.detmet2_raw)


def slide_sq_det(cdat: np.ndarray, fc: float = DEFAULT_FC,
                 ntol: float = DEFAULT_NTOL,
                 stride: int = NSPM // 4,    # WSJT-X: 18 ms = quarter-frame
                 ) -> TriggerTrace:
    """Slide sq_det through ``cdat`` in WSJT-X's quarter-frame stride.

    Default stride is 216 samples (18 ms at 12 kHz) — matches msk144spd.f90:
    ``nstep = (n - NSPM) / 216``.

    Returns the TriggerTrace ready for block-normalise + threshold.
    """
    n = cdat.shape[0]
    if n < NSPM:
        raise ValueError(f"need at least NSPM={NSPM} samples; got {n}")
    n_step = (n - NSPM) // stride
    if n_step <= 0:
        n_step = 1

    detmet_raw  = np.empty(n_step, dtype=np.float64)
    detmet2_raw = np.empty(n_step, dtype=np.float64)
    ferr        = np.empty(n_step, dtype=np.float64)
    t_centers   = np.empty(n_step, dtype=np.float64)

    for k in range(n_step):
        ns = k * stride
        ne = ns + NSPM
        if ne > n:
            break
        r = sq_det(cdat[ns:ne], fc=fc, ntol=ntol)
        detmet_raw[k]  = r.detmet
        detmet2_raw[k] = r.detmet2
        # Per WSJT-X: use the freq error of WHICHEVER tone won (ah or al)
        ferr[k]        = r.ferr_hi if r.ah >= r.al else r.ferr_lo
        t_centers[k]   = (ns + NSPM / 2.0) / FS

    return _normalize_and_threshold(detmet_raw, detmet2_raw, ferr, t_centers)


def _normalize_and_threshold(detmet_raw: np.ndarray, detmet2_raw: np.ndarray,
                             ferr: np.ndarray, t_centers: np.ndarray,
                             ) -> TriggerTrace:
    """Apply WSJT-X normalisation + dual-threshold logic.

    Matches msk144spd.f90 lines 122-150:
      xmed = 25th-percentile of detmet_raw
      detmet_norm = detmet_raw / xmed
      primary trigger: detmet_norm >= 3.0
      fallback:        detmet2_raw >= 12.0  (used per-step in WSJT-X when
                       the primary pass finds < 3 candidates)
    """
    # 25th percentile.  WSJT-X picks `indices(nstep/4)` of the sorted
    # detmet — that's the n/4-th order statistic, i.e. the 25th percentile.
    xmed = float(np.percentile(detmet_raw, PCT_FOR_XMED))
    if xmed <= 0.0:
        xmed = 1e-30
    detmet_norm = detmet_raw / xmed

    primary  = detmet_norm   >= DETECT_THRESHOLD_NORM     # >= 3.0
    fallback = detmet2_raw   >= DETECT_THRESHOLD_DETMET2  # >= 12.0
    triggers = primary | fallback

    return TriggerTrace(
        detmet_raw=detmet_raw, detmet2_raw=detmet2_raw, xmed=xmed,
        detmet_norm=detmet_norm, triggers=triggers,
        t_centers=t_centers, ferr=ferr,
    )


# ── Per-channel batched variant for production hot loop ─────────────────────
#
# MAP144 runs detection on 48 channels simultaneously, every hop.  The batched
# version below does ONE FFT over the channel axis and vectorises every step
# the per-frame ``sq_det()`` does serially.  Same algorithm, same outputs,
# vectorised across channels.
#
# Input: complex baseband (channelizer output) — fc defaults to 0 Hz.  Pass
# fc=DEFAULT_FC=1500 if you have analytic-at-audio-carrier instead.


@dataclass
class SqDetBatchResult:
    """Per-channel batched sq_det output.  All fields shape (n_channels,).

    Units: linear (power ratios in |FFT|² domain).  See top-of-file
    UNIT CONVENTION comment.
    """
    detmet:       np.ndarray   # raw peak amplitude — max(ah, al) per channel
    detmet2:      np.ndarray   # per-frame peak/local-noise ratio — linear, threshold 12.0
    ah:           np.ndarray   # hi-tone peak per channel
    al:           np.ndarray   # lo-tone peak per channel
    ahavp:        np.ndarray   # hi-tone local in-band noise (peak-excluded)
    alavp:        np.ndarray   # lo-tone local in-band noise
    ferr_hi:      np.ndarray   # hi-tone freq error (Hz, sub-bin via parabolic)
    ferr_lo:      np.ndarray   # lo-tone freq error (Hz)
    fc_offset_hz: np.ndarray   # signed offset from nominal — picked from STRONG band
                               # (= ferr_hi if ah >= al else ferr_lo).  Avoids the
                               # noise-driven fc_hz pollution that bit the 2026-05-15
                               # overnight regression on max-combine.

    @property
    def detmet2_db(self) -> np.ndarray:
        return to_db(self.detmet2)


def sq_det_batch(c_channels: np.ndarray,
                 fc: float = 0.0,
                 ntol: float = DEFAULT_NTOL) -> SqDetBatchResult:
    """Batched sq_det across n_channels of NSPM-sample complex windows.

    Parameters
    ----------
    c_channels : complex array, shape (n_channels, NSPM)
        Per-channel windows of complex samples at FS=12 kHz.  Defaults to
        complex baseband (fc=0) — the format MAP144's channelizer produces.
    fc : float
        Audio carrier of the input.  0.0 for complex baseband (default),
        DEFAULT_FC=1500.0 for analytic-real-audio (matches WSJT-X / the
        per-frame ``sq_det()``).  The squared-spectrum tone targets are
        ``2*(fc ± 500)`` Hz; for fc=0 they're at ±1000 Hz.
    ntol : float
        Frequency tolerance at the audio carrier (Hz).  Search window for
        each squared tone is ``±2*ntol`` Hz, matching WSJT-X.

    Returns
    -------
    SqDetBatchResult with per-channel arrays.

    Notes
    -----
    Algorithm is bit-identical to per-channel calls of ``sq_det()`` with the
    same fc/ntol, except parabolic peak interpolation is vectorised.
    """
    if c_channels.ndim != 2 or c_channels.shape[1] != NSPM:
        raise ValueError(
            f"sq_det_batch expects (n_channels, NSPM={NSPM}); got {c_channels.shape}"
        )

    n_ch = c_channels.shape[0]

    # Square the analytic / baseband signal, then batched FFT over NSPM axis.
    sq   = c_channels.astype(np.complex128) ** 2
    spec = np.fft.fft(sq, axis=-1)
    tonespec = np.abs(spec) ** 2                    # (n_ch, NSPM) power spectrum

    # Target frequencies in the squared spectrum
    nfhi = 2.0 * (fc + 500.0)   # +1000 for fc=0 baseband; 4000 for fc=1500 analytic
    nflo = 2.0 * (fc - 500.0)   # -1000 for fc=0 baseband; 2000 for fc=1500 analytic

    # Locate target bins.  np.fft.fftfreq returns [0, df, ..., (N/2-1)·df,
    # -N/2·df, ..., -df] — negative frequencies land in the back half.
    # argmin of |freq - target| picks the closest bin regardless of sign.
    freqs = np.fft.fftfreq(NSPM, 1.0 / FS)
    i_hi  = int(np.argmin(np.abs(freqs - nfhi)))
    i_lo  = int(np.argmin(np.abs(freqs - nflo)))
    bw    = int(round(2.0 * ntol / DF))

    hi_lo_idx, hi_hi_idx = max(0, i_hi - bw), min(NSPM, i_hi + bw + 1)
    lo_lo_idx, lo_hi_idx = max(0, i_lo - bw), min(NSPM, i_lo + bw + 1)

    hi_band = tonespec[:, hi_lo_idx:hi_hi_idx]      # (n_ch, hi_w)
    lo_band = tonespec[:, lo_lo_idx:lo_hi_idx]      # (n_ch, lo_w)

    # Peak per channel
    hi_pk_local = hi_band.argmax(axis=1)            # (n_ch,)
    lo_pk_local = lo_band.argmax(axis=1)
    ch_idx      = np.arange(n_ch)
    ah          = hi_band[ch_idx, hi_pk_local]
    al          = lo_band[ch_idx, lo_pk_local]

    # In-band noise floor — peak-excluded (matches WSJT-X)
    n_hi   = max(1, hi_band.shape[1] - 1)
    n_lo   = max(1, lo_band.shape[1] - 1)
    ahavp  = (hi_band.sum(axis=1) - ah) / n_hi
    alavp  = (lo_band.sum(axis=1) - al) / n_lo

    # detmet, detmet2 (LINEAR, threshold 12.0 on detmet2; threshold 3.0 on detmet/xmed)
    detmet  = np.maximum(ah, al)
    ahavp_s = np.maximum(ahavp, 1e-30)
    alavp_s = np.maximum(alavp, 1e-30)
    detmet2 = np.maximum(ah / ahavp_s, al / alavp_s)

    # ── Parabolic peak interpolation, vectorised ────────────────────────────
    # Skip channels whose peak lies at the band edge (no neighbours for parabolic fit).
    def _vparabolic(band: np.ndarray, pk: np.ndarray) -> np.ndarray:
        n = band.shape[0]
        delta = np.zeros(n, dtype=np.float64)
        interior = (pk > 0) & (pk < band.shape[1] - 1)
        if not np.any(interior):
            return delta
        idx = np.where(interior)[0]
        pks = pk[idx]
        y_lo = band[idx, pks - 1]
        y_pk = band[idx, pks]
        y_hi = band[idx, pks + 1]
        denom = y_lo - 2.0 * y_pk + y_hi
        safe  = np.abs(denom) > 1e-30
        if np.any(safe):
            num = 0.5 * (y_lo - y_hi)
            d   = np.where(safe, num / np.where(safe, denom, 1.0), 0.0)
            delta[idx] = d
        return delta

    deltah = _vparabolic(hi_band, hi_pk_local)
    deltal = _vparabolic(lo_band, lo_pk_local)

    # Frequency at peak bin (with sub-bin offset), in the squared spectrum
    f_hi_at_peak = freqs[hi_lo_idx + hi_pk_local] + deltah * DF
    f_lo_at_peak = freqs[lo_lo_idx + lo_pk_local] + deltal * DF

    # Convert to error at the audio carrier (divide by 2 — the squared spectrum
    # is at 2·f_audio, so a bin shift there equals half a bin in audio space).
    ferr_hi = (f_hi_at_peak - nfhi) / 2.0
    ferr_lo = (f_lo_at_peak - nflo) / 2.0

    # ── fc_offset from STRONG band only ─────────────────────────────────────
    # The overnight 2026-05-15 regression came from using max-combine for the
    # trigger metric but reading fc_hz from BOTH bands' argmax — so when one
    # band fired alone, the other band's argmax was noise-driven and corrupted
    # fc_hz.  Read the freq error from whichever band fired stronger.
    fc_offset_hz = np.where(ah >= al, ferr_hi, ferr_lo)

    return SqDetBatchResult(
        detmet=detmet,
        detmet2=detmet2,
        ah=ah, al=al, ahavp=ahavp, alavp=alavp,
        ferr_hi=ferr_hi, ferr_lo=ferr_lo,
        fc_offset_hz=fc_offset_hz,
    )
