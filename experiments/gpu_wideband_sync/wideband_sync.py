# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""GPU wideband sync matched filter for MSK144.

Algorithm (single window, all-frequencies-at-once via FFT):

    For each sliding window of N samples of input IQ:
        1. Multiply elementwise by conj(sync_template), length N each
        2. FFT of the product, length N (or N_padded for finer freq grid)
        3. magnitude_squared(FFT_bin_k) = squared correlation at
           frequency offset k * fs/N from the centre.

Repeated across sliding windows with stride S, this builds a (time, freq)
ambiguity surface.  Peak-find above threshold → detection candidates.

The two-FFT trick is the standard ambiguity-function-via-FFT identity:
the matched filter at frequency offset f and time offset t is

    y(t, f) = ∫ x(τ) · h*(τ - t) · exp(-j 2π f τ) dτ

For a fixed t (one window), varying f is one FFT.  See README math.

Implementation lives entirely in CuPy — pure GPU.  numpy fallback exists
only for CPU testing of the math (see `compute_ambiguity_surface_cpu`).
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
class WidebandSyncConfig:
    """Configuration for the wideband sync matched-filter detector."""

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
    threshold_db: float = 10.0
    #: Per-bin noise-floor estimator: rolling-median window length, in
    #: number of correlation windows.
    noise_window: int = 300


# ── CPU reference ────────────────────────────────────────────────────────────


def compute_ambiguity_surface_cpu(iq: np.ndarray,
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


def compute_ambiguity_surface_gpu(iq_cp, template_cp, stride_samples: int,
                                   fft_length: int | None = None):
    """GPU implementation of the ambiguity-surface computation.

    Same math as ``compute_ambiguity_surface_cpu`` but vectorised:
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


# ── Candidate extraction ────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One detection candidate from the ambiguity surface."""
    time_sample: int          # input-IQ sample index of the window centre
    time_s: float             # seconds from start of input
    freq_hz: float            # frequency offset from IQ DC
    snr_db: float             # ratio of magnitude to per-bin noise-floor
    raw_magnitude: float      # |matched_filter_output|


def extract_candidates(surface_cp, cfg: WidebandSyncConfig,
                       template_length: int) -> list[Candidate]:
    """Scan the ambiguity surface for detections above threshold.

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

    out: list[Candidate] = []
    win_idxs, freq_idxs = np.where(above_np)
    for t, f in zip(win_idxs.tolist(), freq_idxs.tolist()):
        sample_idx = t * cfg.stride_samples + template_length // 2
        out.append(Candidate(
            time_sample=int(sample_idx),
            time_s=float(sample_idx / cfg.sample_rate_hz),
            freq_hz=float(freq_axis[f]),
            snr_db=float(snr_db_np[t, f]),
            raw_magnitude=float(magnitude_np[t, f]),
        ))
    return out


# ── Convenience entry point ─────────────────────────────────────────────────


def detect(iq: np.ndarray, template: np.ndarray,
           cfg: WidebandSyncConfig | None = None,
           use_gpu: bool = True
           ) -> list[Candidate]:
    """One-shot detection: feed IQ, get back candidate list.

    Parameters
    ----------
    iq : complex64 ndarray, shape (n_samples,)
        Input IQ at ``cfg.sample_rate_hz``.
    template : complex64 ndarray, shape (N,)
        Sync template at the same sample rate (see ``sync_template.py``).
    cfg : WidebandSyncConfig
        Detector config.  Default config = 48 kHz, stride=24, threshold=10 dB.
    use_gpu : bool
        If False, run the numpy reference path (slow; for tests).

    Returns
    -------
    candidates : list of Candidate
    """
    if cfg is None:
        cfg = WidebandSyncConfig()

    if use_gpu and _HAS_CUPY:
        iq_cp = cp.asarray(iq, dtype=cp.complex64)
        tpl_cp = cp.asarray(template, dtype=cp.complex64)
        surface = compute_ambiguity_surface_gpu(
            iq_cp, tpl_cp,
            stride_samples=cfg.stride_samples,
            fft_length=cfg.fft_length,
        )
        return extract_candidates(surface, cfg, template_length=tpl_cp.shape[0])
    else:
        # CPU path: build surface in numpy, then ship to GPU (or skip
        # extraction) — for now, basic numpy reference.
        surface_np = compute_ambiguity_surface_cpu(
            iq, template,
            stride_samples=cfg.stride_samples,
            fft_length=cfg.fft_length,
        )
        # TODO: implement extract_candidates_cpu so CPU path is fully
        # functional.  For now, ship surface to GPU just for extraction
        # if cupy is present.
        if _HAS_CUPY:
            return extract_candidates(
                cp.asarray(surface_np), cfg, template_length=template.shape[0]
            )
        raise NotImplementedError(
            "CPU candidate extraction not implemented yet; install cupy to "
            "use GPU end-to-end, or implement extract_candidates_cpu."
        )
