# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""GPU Time-Frequency Matched Filter (TFMF) for MSK144.

Algorithm (single window, all-frequencies-at-once via FFT):

    For each sliding window of N samples of input IQ:
        1. Multiply elementwise by conj(sync_template), length N each
        2. FFT of the product, length N (or N_padded for finer freq grid)
        3. magnitude_squared(FFT_bin_k) = squared correlation at
           frequency offset k * fs/N from the centre.

Repeated across sliding windows with stride S, this builds a (time, freq)
time-frequency correlation surface.  Peak-find above threshold → detection candidates.

The two-FFT trick is the standard "FFT-of-product-equals-correlation-at-all-freqs" identity:
the matched filter at frequency offset f and time offset t is

    y(t, f) = ∫ x(τ) · h*(τ - t) · exp(-j 2π f τ) dτ

For a fixed t (one window), varying f is one FFT.  See README math.

Implementation lives entirely in CuPy — pure GPU.  numpy fallback exists
only for CPU testing of the math (see `compute_tf_surface_cpu`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp                                 # type: ignore[import-not-found]
    _HAS_CUPY = True
except ImportError:
    cp = None                                          # type: ignore[assignment]
    _HAS_CUPY = False


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class TFMFConfig:
    """Configuration for the TFMF matched-filter detector."""

    sample_rate_hz: int = 48_000
    #: Stride (samples) between consecutive correlation windows.
    #: Smaller stride → finer time resolution at quadratic-ish cost.
    #: Default = 24 samples = one MSK144 symbol at 48 kHz.
    stride_samples: int = 24
    #: FFT length.  Set equal to template length for natural freq
    #: resolution; set > template length to zero-pad for finer freq
    #: viewing (does NOT improve underlying resolution).
    fft_length: int | None = None
    #: Detection threshold in dB above the per-bin noise-floor estimate.
    #: Default 13.5 dB → ~1 false alarm per 15-s period on pure AWGN at
    #: default stride/FFT settings (5.76 M cells).  See README "False alarm
    #: rate vs threshold" for the table.  Lower for high-sensitivity sweeps,
    #: raise for clean detection logs.
    threshold_db: float = 13.5
    #: Per-bin noise-floor estimator: rolling-median window length, in
    #: number of correlation windows.
    noise_window: int = 300
    #: If True, coherently sum the matched-filter outputs of MSK144's two
    #: 8-symbol sync sub-blocks (positions 0-7 and 56-63 within one
    #: 144-symbol frame).  Gives +3 dB sensitivity at the same threshold
    #: and FA rate — both signal and noise scale by √2, so the detector
    #: statistic |MF|/median(|MF|) is unchanged on pure noise.  Requires
    #: ``stride_samples`` to divide ``SYNC_SPACING_SAMPLES`` (= 1344 at
    #: 48 kHz).
    two_subblock_sum: bool = True
    #: If True, refine each detection's freq estimate to sub-bin precision
    #: by parabolic interpolation across the 3 bins centred on the peak.
    #: Cheap (no extra FFT).  Lowers the freq error handed to the decoder.
    parabolic_freq_interp: bool = True
    #: K-frame coherent integration.  K=1 disables; K>1 sums the matched-
    #: filter outputs of K consecutive MSK144 frames at the same (t, f).
    #: At FFT-bin frequencies, the per-frame phase rotation is an integer
    #: multiple of 2π (since 3456 = 18 × FFT length), so the coherent sum
    #: is plain elementwise complex addition.  Gain on coherent signal:
    #: +10·log10(K) dB; noise lifts by +5·log10(K) dB → SNR ratio
    #: improvement +5·log10(K) dB.  Strongest benefit is suppression of
    #: single-frame impulsive noise (which doesn't repeat across frames).
    #: Requires the input IQ to contain ≥ K consecutive MSK144 frames at
    #: a stable freq; for short meteor pings (< 144 ms) K-frame > 2 may
    #: not have enough frames to integrate.
    k_frame_integration: int = 1


# Hard-coded MSK144 geometry constants
SYNC_SPACING_SAMPLES_48K: int = 56 * 24   # = 1344, between sub-blocks in a frame
FRAME_SPACING_SAMPLES_48K: int = 144 * 24 # = 3456, MSK144 message frame length


# ── CPU reference ────────────────────────────────────────────────────────────


def compute_tf_surface_cpu(iq: np.ndarray,
                                  template: np.ndarray,
                                  stride_samples: int,
                                  fft_length: int | None = None
                                  ) -> np.ndarray:
    """CPU (numpy) reference implementation — for validating GPU math.

    Returns
    -------
    surface : complex64 ndarray, shape (n_windows, fft_length)
        Element [t, f] = matched-filter output at time-window t and
        FFT-bin f.  Take magnitude to get correlation strength;
        compare against noise floor for SNR.
    """
    N = template.shape[0]
    if fft_length is None:
        fft_length = N
    n_windows = max(0, (iq.shape[0] - N) // stride_samples + 1)
    if n_windows == 0:
        return np.empty((0, fft_length), dtype=np.complex64)

    h_conj = np.conj(template).astype(np.complex64)
    out = np.empty((n_windows, fft_length), dtype=np.complex64)
    for i in range(n_windows):
        start = i * stride_samples
        window = iq[start:start + N]
        product = window * h_conj
        if fft_length > N:
            padded = np.zeros(fft_length, dtype=np.complex64)
            padded[:N] = product
            product = padded
        out[i] = np.fft.fft(product, n=fft_length).astype(np.complex64)
    return out


# ── GPU implementation ──────────────────────────────────────────────────────


def compute_tf_surface_gpu(iq_cp, template_cp, stride_samples: int,
                                   fft_length: int | None = None):
    """GPU implementation of the TF-correlation-surface computation.

    Same math as ``compute_tf_surface_cpu`` but vectorised:
        - Build (n_windows, N) matrix of windowed views in one shot
        - Single batched elementwise multiply by conj(template)
        - Single batched FFT along axis=1

    Parameters
    ----------
    iq_cp : cupy.ndarray, complex64, shape (n_samples,)
    template_cp : cupy.ndarray, complex64, shape (N,)
    stride_samples : int
    fft_length : int or None

    Returns
    -------
    surface : cupy.ndarray, complex64, shape (n_windows, fft_length)
    """
    if not _HAS_CUPY:
        raise RuntimeError(
            "cupy not installed.  Install with: pip install cupy-cuda12x "
            "(or cupy-cuda11x, matching your CUDA major version)."
        )

    N = template_cp.shape[0]
    if fft_length is None:
        fft_length = N
    n_samples = int(iq_cp.shape[0])
    n_windows = max(0, (n_samples - N) // stride_samples + 1)
    if n_windows == 0:
        return cp.empty((0, fft_length), dtype=cp.complex64)

    # Build windowed views via stride tricks — zero-copy until we
    # multiply.  Each row of `windows` is a length-N slice of iq starting
    # stride_samples later than the previous row.
    # TODO: switch to cupyx.lib.stride_tricks if/when needed for memory
    # efficiency; for now use direct indexing.
    indices = cp.arange(N)[None, :] + (
        cp.arange(n_windows) * stride_samples
    )[:, None]
    windows = iq_cp[indices]                                  # (n_windows, N)

    # Elementwise multiply by conj(template) (broadcast).
    h_conj = cp.conj(template_cp).astype(cp.complex64)
    products = windows * h_conj[None, :]                      # (n_windows, N)

    if fft_length > N:
        padded = cp.zeros((n_windows, fft_length), dtype=cp.complex64)
        padded[:, :N] = products
        products = padded

    surface = cp.fft.fft(products, n=fft_length, axis=1)
    return surface.astype(cp.complex64)


# ── Two-sub-block coherent sum ──────────────────────────────────────────────


def _apply_two_subblock_sum(surface, stride_samples: int,
                             sync_spacing_samples: int = SYNC_SPACING_SAMPLES_48K):
    """Sum each row of ``surface`` with the row that is exactly one sync-
    sub-block-spacing later in time.

    For an MSK144 message frame placed so its sync-1 lands at row ``i``,
    sync-2 lands at row ``i + shift`` where ``shift = sync_spacing_samples
    / stride_samples``.  Summing surface[i] + surface[i + shift] gives the
    coherent two-sub-block matched-filter output: signal amplitudes add
    (+6 dB power) while independent-noise variances add (+3 dB power), a
    net +3 dB SNR gain.

    At FFT bin centres the phase between the two MF outputs is
    ``2π · k · 7`` (since 1344 = 7 × 192) — integer multiple of 2π for all
    integer ``k``, so coherent sum is a plain complex add with no phase
    correction.

    Works on either numpy or cupy arrays — uses generic slicing.

    Parameters
    ----------
    surface : ndarray, shape (n_rows, n_freq), complex64
        Matched-filter time-frequency correlation surface.
    stride_samples : int
        Same stride used to build the surface.  Must divide
        ``sync_spacing_samples`` evenly.
    sync_spacing_samples : int
        Sample-distance between sub-blocks at the surface's native rate.

    Returns
    -------
    combined : ndarray, shape (n_rows - shift, n_freq), complex64
        Truncated by ``shift`` rows at the tail — bursts whose sync-1
        falls in those trailing rows lose their sync-2 partner.
    """
    if sync_spacing_samples % stride_samples != 0:
        raise ValueError(
            f"stride_samples ({stride_samples}) must divide "
            f"sync_spacing_samples ({sync_spacing_samples}); "
            f"got remainder {sync_spacing_samples % stride_samples}"
        )
    shift = sync_spacing_samples // stride_samples
    n_rows = surface.shape[0]
    if n_rows <= shift:
        # Input too short to contain a full frame — return empty surface
        # of compatible shape.
        return surface[:0]
    return surface[:n_rows - shift] + surface[shift:]


def _apply_k_frame_sum(surface, stride_samples: int, k: int,
                        frame_spacing_samples: int = FRAME_SPACING_SAMPLES_48K):
    """Coherently sum K consecutive MSK144 frames into a single surface.

    Each cell ``out[i, f]`` becomes::

        out[i, f] = sum_{n=0}^{K-1} surface[i + n*shift, f]

    where ``shift = frame_spacing_samples / stride_samples`` (= 144 at the
    default stride=24).  At FFT bin centres the per-frame phase rotation
    is ``2π · k_bin · (3456 / 192) = 2π · k_bin · 18`` ≡ 0 mod 2π, so the
    coherent sum is a plain elementwise complex addition — no phase
    correction needed.

    For coherent multi-frame signal: SNR voltage ratio improves by √K
    (= +10·log10(K) dB power).  For single-window impulsive noise that
    fires only one row: the impulse is diluted by 1/K in voltage, so the
    detector statistic effectively rejects single-frame impulses while
    preserving signals coherent across all K frames.

    Returns an array shorter by ``(K-1) * shift`` rows.  Works on either
    numpy or cupy arrays via generic slicing.
    """
    if k <= 1:
        return surface
    if frame_spacing_samples % stride_samples != 0:
        raise ValueError(
            f"stride_samples ({stride_samples}) must divide "
            f"frame_spacing_samples ({frame_spacing_samples}); "
            f"got remainder {frame_spacing_samples % stride_samples}"
        )
    shift = frame_spacing_samples // stride_samples
    n_rows = surface.shape[0]
    needed = n_rows - (k - 1) * shift
    if needed <= 0:
        return surface[:0]
    out = surface[:needed].copy()
    for n in range(1, k):
        out = out + surface[n * shift : n * shift + needed]
    return out


# ── Candidate extraction ────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One detection candidate from the time-frequency correlation surface."""
    time_sample: int          # input-IQ sample index of the window centre
    time_s: float             # seconds from start of input
    freq_hz: float            # frequency offset from IQ DC
    snr_db: float             # ratio of magnitude to per-bin noise-floor
    raw_magnitude: float      # |matched_filter_output|


def _parabolic_freq_offset(left: float, peak: float, right: float) -> float:
    """Parabolic-interpolation sub-bin offset, range (-0.5, +0.5) bins.

    Given the magnitudes at the peak bin and its two neighbours, return
    the fractional bin shift of the true peak relative to the centre bin.
    Standard 3-point parabolic fit: δ = 0.5·(left - right) / (left - 2·peak + right).

    Returns 0.0 (no refinement) when the denominator is non-positive
    (degenerate — peak not a strict local max).
    """
    denom = left - 2.0 * peak + right
    if denom >= 0.0:
        return 0.0
    delta = 0.5 * (left - right) / denom
    # Clamp to (-0.5, +0.5) — anything outside is a bad fit.
    if delta > 0.5: delta = 0.5
    if delta < -0.5: delta = -0.5
    return float(delta)


def extract_candidates(surface_cp, cfg: TFMFConfig,
                       template_length: int) -> list[Candidate]:
    """Scan the time-frequency correlation surface for detections above threshold.

    Per-bin noise floor estimated as the median magnitude across the
    last ``cfg.noise_window`` time-windows in each freq bin.  Detections
    are local maxima above ``cfg.threshold_db`` over their per-bin noise
    floor.

    TODO: this is the simplest possible peak-picker.  Improvements:
        - 2D local-max filter (suppress same-burst multi-detection)
        - Parabolic interpolation around the peak for sub-bin freq
        - Coherent integration across multiple frames (later)
    """
    if not _HAS_CUPY:
        raise RuntimeError("cupy not installed")
    if surface_cp.shape[0] == 0:
        return []

    magnitude = cp.abs(surface_cp)                            # (n_win, n_freq)

    # Per-freq-bin noise floor: median across windows.  Cheap and
    # signal-robust (median ignores rare spike windows).
    noise_floor = cp.median(magnitude, axis=0)                # (n_freq,)
    # Avoid div-by-zero in fully-zero bins (edge cases at fft boundaries).
    noise_floor = cp.maximum(noise_floor, 1e-30)

    snr_lin = magnitude / noise_floor[None, :]
    snr_db = 20.0 * cp.log10(snr_lin)

    # Above threshold mask
    above = snr_db > cfg.threshold_db                         # bool (n_win, n_freq)
    if not bool(above.any()):
        return []

    # Pull to CPU to do the (small) post-processing.  TODO: do peak-
    # picking on GPU too once volume justifies it.
    above_np = cp.asnumpy(above)
    snr_db_np = cp.asnumpy(snr_db)
    magnitude_np = cp.asnumpy(magnitude)
    n_freq = surface_cp.shape[1]

    # Frequency axis: FFT bins map to fftfreq(n_freq, 1/fs).  Bins 0..n/2
    # are positive freqs; n/2..n-1 wrap to negative.
    freq_axis = np.fft.fftfreq(n_freq, 1.0 / cfg.sample_rate_hz)

    bin_width_hz = cfg.sample_rate_hz / n_freq
    out: list[Candidate] = []
    win_idxs, freq_idxs = np.where(above_np)
    for t, f in zip(win_idxs.tolist(), freq_idxs.tolist()):
        sample_idx = t * cfg.stride_samples + template_length // 2
        freq_hz = float(freq_axis[f])
        if cfg.parabolic_freq_interp:
            left  = float(magnitude_np[t, (f - 1) % n_freq])
            peak  = float(magnitude_np[t, f])
            right = float(magnitude_np[t, (f + 1) % n_freq])
            delta = _parabolic_freq_offset(left, peak, right)
            freq_hz = freq_hz + delta * bin_width_hz
        out.append(Candidate(
            time_sample=int(sample_idx),
            time_s=float(sample_idx / cfg.sample_rate_hz),
            freq_hz=freq_hz,
            snr_db=float(snr_db_np[t, f]),
            raw_magnitude=float(magnitude_np[t, f]),
        ))
    return out


def extract_candidates_cpu(surface: np.ndarray, cfg: TFMFConfig,
                            template_length: int) -> list[Candidate]:
    """CPU (numpy) implementation of candidate extraction.

    Same algorithm as ``extract_candidates`` (the GPU version) but
    operating on a numpy time-frequency correlation surface — for use when CuPy isn't
    installed (i.e. on hosts without a GPU).  Mostly useful for the
    sensitivity-test driver in ``test_sensitivity.py``.

    Per-freq-bin noise floor estimated as the median magnitude across
    windows.  Detections are bins above ``cfg.threshold_db`` over their
    per-bin noise floor.
    """
    if surface.shape[0] == 0:
        return []
    magnitude = np.abs(surface)
    noise_floor = np.median(magnitude, axis=0)
    noise_floor = np.maximum(noise_floor, 1e-30)
    snr_lin = magnitude / noise_floor[None, :]
    snr_db = 20.0 * np.log10(snr_lin)
    above = snr_db > cfg.threshold_db
    if not above.any():
        return []

    n_freq = surface.shape[1]
    freq_axis = np.fft.fftfreq(n_freq, 1.0 / cfg.sample_rate_hz)
    bin_width_hz = cfg.sample_rate_hz / n_freq

    out: list[Candidate] = []
    win_idxs, freq_idxs = np.where(above)
    for t, f in zip(win_idxs.tolist(), freq_idxs.tolist()):
        sample_idx = t * cfg.stride_samples + template_length // 2
        freq_hz = float(freq_axis[f])
        if cfg.parabolic_freq_interp:
            # 3-point parabolic refinement.  Use circular wrap so bin 0
            # and bin n_freq-1 work without special-casing.
            left  = float(magnitude[t, (f - 1) % n_freq])
            peak  = float(magnitude[t, f])
            right = float(magnitude[t, (f + 1) % n_freq])
            delta = _parabolic_freq_offset(left, peak, right)
            # Centre-bin frequency + delta·bin_width, signed (fftfreq
            # already handles wrap to negative half).  Note we add to the
            # *signed* centre freq, not to the raw bin index, so this
            # composes correctly with the fftfreq wrap.
            freq_hz = freq_hz + delta * bin_width_hz
        out.append(Candidate(
            time_sample=int(sample_idx),
            time_s=float(sample_idx / cfg.sample_rate_hz),
            freq_hz=freq_hz,
            snr_db=float(snr_db[t, f]),
            raw_magnitude=float(magnitude[t, f]),
        ))
    return out


# ── Convenience entry point ─────────────────────────────────────────────────


def detect(iq: np.ndarray, template: np.ndarray,
           cfg: TFMFConfig | None = None,
           use_gpu: bool = True
           ) -> list[Candidate]:
    """One-shot detection: feed IQ, get back candidate list.

    Parameters
    ----------
    iq : complex64 ndarray, shape (n_samples,)
        Input IQ at ``cfg.sample_rate_hz``.
    template : complex64 ndarray, shape (N,)
        Sync template at the same sample rate (see ``sync_template.py``).
    cfg : TFMFConfig
        Detector config.  Default config = 48 kHz, stride=24, threshold=13.5 dB.
    use_gpu : bool
        If False, run the numpy reference path (slow; for tests).

    Returns
    -------
    candidates : list of Candidate
    """
    if cfg is None:
        cfg = TFMFConfig()

    if use_gpu and _HAS_CUPY:
        iq_cp = cp.asarray(iq, dtype=cp.complex64)
        tpl_cp = cp.asarray(template, dtype=cp.complex64)
        surface = compute_tf_surface_gpu(
            iq_cp, tpl_cp,
            stride_samples=cfg.stride_samples,
            fft_length=cfg.fft_length,
        )
        if cfg.two_subblock_sum:
            surface = _apply_two_subblock_sum(surface, cfg.stride_samples)
        if cfg.k_frame_integration > 1:
            surface = _apply_k_frame_sum(surface, cfg.stride_samples,
                                          cfg.k_frame_integration)
        return extract_candidates(surface, cfg, template_length=tpl_cp.shape[0])
    else:
        # CPU end-to-end path: surface in numpy, candidate extraction in
        # numpy.  Used by the sensitivity-test driver on hosts without
        # a GPU.
        surface_np = compute_tf_surface_cpu(
            iq, template,
            stride_samples=cfg.stride_samples,
            fft_length=cfg.fft_length,
        )
        if cfg.two_subblock_sum:
            surface_np = _apply_two_subblock_sum(surface_np, cfg.stride_samples)
        if cfg.k_frame_integration > 1:
            surface_np = _apply_k_frame_sum(surface_np, cfg.stride_samples,
                                             cfg.k_frame_integration)
        return extract_candidates_cpu(
            surface_np, cfg, template_length=template.shape[0]
        )
