#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``map144_app.sq_det.sq_det_batch``.

The batched function is the production hot-path entry point — it runs across
all 48 channels per detection hop.  Correctness check: for each channel,
sq_det_batch output must equal a per-channel call of sq_det() within
floating-point tolerance.

Also validates:
- Default fc=0 (complex baseband) matches MAP144's channelizer format
- fc_offset_hz picks the STRONG band's argmax (the regression-fix from
  2026-05-15 — see project_detection_metric_db_scale_mismatch)
- dB views are consistent with linear values
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.sq_det import (
    NSPM, FS, DF, DEFAULT_FC, DEFAULT_NTOL,
    sq_det, sq_det_batch, SqDetBatchResult,
    to_db, to_lin,
    DETECT_THRESHOLD_NORM_LIN, DETECT_THRESHOLD_NORM_DB,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _msk_like_window(seed: int, amplitude: float = 1.0,
                    fc_audio: float = DEFAULT_FC) -> np.ndarray:
    """One-NSPM analytic-signal window with MSK-like phase modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(NSPM) / FS
    # MSK is continuous-phase ±B/4 frequency shift per chip
    chips = rng.choice([-1.0, 1.0], size=NSPM // 6)
    phase_increments = np.repeat(chips, 6)[:NSPM] * (np.pi / 12)   # ±π/12 per sample
    cumulative_phase = np.cumsum(phase_increments)
    return amplitude * np.exp(1j * (2 * np.pi * fc_audio * t + cumulative_phase)).astype(np.complex128)


def _baseband_from_analytic(analytic: np.ndarray, fc: float = DEFAULT_FC) -> np.ndarray:
    """Mix analytic signal at fc down to 0 Hz baseband."""
    t = np.arange(analytic.shape[-1]) / FS
    return (analytic * np.exp(-1j * 2 * np.pi * fc * t)).astype(np.complex128)


# ── Equivalence vs per-frame sq_det ─────────────────────────────────────────

class TestBatchedMatchesPerFrame:
    """For each channel, sq_det_batch must equal sq_det() within tolerance."""

    def test_baseband_input_matches_per_frame(self):
        """fc=0 baseband.  Per-frame sq_det() with fc=0 must match the batched."""
        n_ch = 6
        channels = np.zeros((n_ch, NSPM), dtype=np.complex128)
        for i in range(n_ch):
            analytic = _msk_like_window(seed=100 + i, amplitude=1.0 + 0.2 * i)
            channels[i] = _baseband_from_analytic(analytic, fc=DEFAULT_FC)

        batched = sq_det_batch(channels, fc=0.0, ntol=DEFAULT_NTOL)

        for i in range(n_ch):
            ref = sq_det(channels[i], fc=0.0, ntol=DEFAULT_NTOL)
            np.testing.assert_allclose(batched.detmet[i],  ref.detmet,  rtol=1e-6)
            np.testing.assert_allclose(batched.detmet2[i], ref.detmet2, rtol=1e-6)
            np.testing.assert_allclose(batched.ah[i],      ref.ah,      rtol=1e-6)
            np.testing.assert_allclose(batched.al[i],      ref.al,      rtol=1e-6)
            np.testing.assert_allclose(batched.ferr_hi[i], ref.ferr_hi, atol=0.01)
            np.testing.assert_allclose(batched.ferr_lo[i], ref.ferr_lo, atol=0.01)

    def test_analytic_input_matches_per_frame(self):
        """fc=1500 analytic input.  Same equivalence at the WSJT-X carrier."""
        n_ch = 4
        channels = np.stack([_msk_like_window(seed=200 + i) for i in range(n_ch)])
        batched = sq_det_batch(channels, fc=DEFAULT_FC, ntol=DEFAULT_NTOL)
        for i in range(n_ch):
            ref = sq_det(channels[i], fc=DEFAULT_FC, ntol=DEFAULT_NTOL)
            np.testing.assert_allclose(batched.detmet[i],  ref.detmet,  rtol=1e-6)
            np.testing.assert_allclose(batched.detmet2[i], ref.detmet2, rtol=1e-6)


# ── fc_offset from STRONG band only (regression fix) ────────────────────────

class TestFcOffsetFromStrongBand:
    """fc_offset_hz must come from the band that fired stronger.

    The 2026-05-15 overnight regression came from averaging fc_offset from
    BOTH bands' argmax, which polluted fc_hz with noise from the un-signaled
    band on single-tone events.
    """

    def test_single_tone_in_hi_band_fc_offset_from_hi(self):
        """Signal only in hi-tone region of squared spectrum → fc_offset = ferr_hi."""
        # Synthesize a signal whose squared spectrum lands at +1000 Hz baseband.
        # That means baseband signal at +500 Hz (since squaring doubles frequency).
        t = np.arange(NSPM) / FS
        signal_baseband = np.exp(1j * 2 * np.pi * (+500.0) * t).astype(np.complex128)
        channels = signal_baseband[np.newaxis, :].copy()
        r = sq_det_batch(channels, fc=0.0, ntol=DEFAULT_NTOL)
        # ah should dominate al
        assert r.ah[0] > 10.0 * r.al[0], f"hi-band should dominate; ah={r.ah[0]} al={r.al[0]}"
        # fc_offset must come from hi-band freq error, not lo-band
        np.testing.assert_allclose(r.fc_offset_hz[0], r.ferr_hi[0], rtol=1e-6)

    def test_single_tone_in_lo_band_fc_offset_from_lo(self):
        """Same, but signal in lo-tone region (−1000 Hz baseband)."""
        t = np.arange(NSPM) / FS
        signal_baseband = np.exp(1j * 2 * np.pi * (-500.0) * t).astype(np.complex128)
        channels = signal_baseband[np.newaxis, :].copy()
        r = sq_det_batch(channels, fc=0.0, ntol=DEFAULT_NTOL)
        assert r.al[0] > 10.0 * r.ah[0], f"lo-band should dominate; ah={r.ah[0]} al={r.al[0]}"
        np.testing.assert_allclose(r.fc_offset_hz[0], r.ferr_lo[0], rtol=1e-6)


# ── dB views consistency ────────────────────────────────────────────────────

class TestDbViews:
    """detmet2_db must equal 10·log10(detmet2)."""

    def test_detmet2_db_matches_linear(self):
        channels = np.stack([_msk_like_window(seed=300 + i) for i in range(3)])
        batched = sq_det_batch(channels, fc=DEFAULT_FC)
        np.testing.assert_allclose(
            batched.detmet2_db,
            10.0 * np.log10(np.maximum(batched.detmet2, 1e-30)),
            rtol=1e-10,
        )

    def test_threshold_dB_round_trip(self):
        """DETECT_THRESHOLD_NORM_DB must be 10·log10(DETECT_THRESHOLD_NORM_LIN)."""
        np.testing.assert_allclose(
            DETECT_THRESHOLD_NORM_DB,
            10.0 * np.log10(DETECT_THRESHOLD_NORM_LIN),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            to_lin(DETECT_THRESHOLD_NORM_DB),
            DETECT_THRESHOLD_NORM_LIN,
            rtol=1e-10,
        )


# ── Input validation ────────────────────────────────────────────────────────

class TestInputValidation:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="NSPM"):
            sq_det_batch(np.zeros(NSPM, dtype=np.complex128))   # 1-D, missing channel axis
        with pytest.raises(ValueError, match="NSPM"):
            sq_det_batch(np.zeros((4, NSPM - 1), dtype=np.complex128))   # short last axis
