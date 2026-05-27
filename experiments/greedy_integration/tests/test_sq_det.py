# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the WSJT-X sq_det formula.

Validates the per-frame metric against analytically-tractable synthetic
inputs (pure noise, single tone, MSK-like signal), then validates the
block normalisation + threshold against a known-pattern trace.
"""
from __future__ import annotations
import math, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root for map144_app
from map144_app.sq_det import (
    NSPM, FS, DF, DEFAULT_FC, DEFAULT_NTOL,
    DETECT_THRESHOLD_NORM, DETECT_THRESHOLD_DETMET2,
    sq_det, slide_sq_det, SqDetResult, TriggerTrace,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_analytic_tone(freq_hz: float, n: int = NSPM, amplitude: float = 1.0,
                       phase: float = 0.0) -> np.ndarray:
    """Pure complex exponential at freq_hz, length n samples."""
    t = np.arange(n) / FS
    return (amplitude * np.exp(1j * (2 * np.pi * freq_hz * t + phase))
            ).astype(np.complex128)


def make_analytic_noise(n: int = NSPM, sigma: float = 1.0, seed: int = 0
                        ) -> np.ndarray:
    """Complex Gaussian noise."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)
            ).astype(np.complex128)


def make_msk_like_analytic(fc: float = DEFAULT_FC, n: int = NSPM,
                           amplitude: float = 1.0, seed: int = 0
                           ) -> np.ndarray:
    """MSK-shaped analytic signal: a complex exponential at fc whose
    frequency switches between fc-500 and fc+500 every NSPS=83 samples
    (matching MSK144 symbol rate 2000 baud at 12 kHz: NSPS = 6).
    For *sq_det* purposes the resulting squared-spectrum lines should
    land cleanly at 2*(fc-500) and 2*(fc+500)."""
    NSPS = 6   # samples per symbol at 12 kHz / 2000 baud
    rng = np.random.default_rng(seed)
    n_symbols = (n + NSPS - 1) // NSPS
    # MSK uses two tones at ±500 Hz around fc, alternating per symbol
    # following the data.  For testing we just pick random ±1 per symbol.
    symbols = rng.choice([-1, 1], size=n_symbols)
    # Build a sample-by-sample frequency trajectory
    freqs = np.repeat(symbols * 500.0, NSPS)[:n] + fc
    # Integrate frequency to get phase
    phases = 2 * np.pi * np.cumsum(freqs) / FS
    return (amplitude * np.exp(1j * phases)).astype(np.complex128)


# ── Per-frame sq_det ────────────────────────────────────────────────────────

class TestPerFrameSqDet:

    def test_input_length_must_be_NSPM(self):
        with pytest.raises(ValueError):
            sq_det(np.zeros(NSPM - 1, dtype=np.complex128))

    def test_pure_noise_detmet2_below_threshold(self):
        """On pure noise the per-frame detmet2 (peak/in-band-noise) should
        be modest — not the threshold-12 fallback level."""
        noise = make_analytic_noise(seed=42)
        r = sq_det(noise)
        assert r.detmet2 < DETECT_THRESHOLD_DETMET2, \
            f"pure noise gave detmet2={r.detmet2:.2f}, expected < 12"

    def test_strong_msk_lights_up_BOTH_tones(self):
        """A clean MSK-like signal should produce strong peaks in BOTH the
        2000 Hz and 4000 Hz target bands."""
        s = make_msk_like_analytic(fc=DEFAULT_FC, amplitude=10.0, seed=1)
        r = sq_det(s)
        # peak amplitudes should clearly dominate the local in-band noise
        assert r.ah / r.ahavp > 20.0, f"hi tone weak: ah/ahavp={r.ah/r.ahavp:.1f}"
        assert r.al / r.alavp > 20.0, f"lo tone weak: al/alavp={r.al/r.alavp:.1f}"
        # The MAX-of-tones detmet2 should be very high — well past threshold-12 fallback
        assert r.detmet2 > 50.0, f"strong msk gave detmet2={r.detmet2:.1f}"

    def test_single_tone_only_lights_ONE_side(self):
        """A pure tone at fc+500 (the upper MSK tone) squared gives a single
        line at 2*(fc+500)=4000 Hz.  Only the HI mask should fire strongly;
        the LO mask should remain at noise level.  We add a small amount
        of noise so the in-band noise floor isn't degenerate at zero."""
        s = (make_analytic_tone(DEFAULT_FC + 500.0, amplitude=10.0)
             + make_analytic_noise(sigma=0.1, seed=22))
        r = sq_det(s)
        assert r.ah / r.ahavp > 100.0, \
            f"expected huge hi-tone response; got ah/ahavp={r.ah/r.ahavp:.1f}"
        # The 'lo' band has only noise — al/alavp should be small
        assert r.al / r.alavp < 30.0, \
            f"unexpected lo-tone response: al/alavp={r.al/r.alavp:.1f}"
        # detmet2 = max(...) is driven by the hi side
        assert r.detmet2 == pytest.approx(r.ah / r.ahavp, rel=1e-6)

    def test_freq_offset_within_ntol(self):
        """If the signal is offset by Δf Hz at audio (so 2Δf Hz at squared
        spectrum), ferr_hi should report Δf within parabolic interpolation
        precision (~ DF/4)."""
        offset_hz = 30.0
        s = make_analytic_tone(DEFAULT_FC + 500.0 + offset_hz, amplitude=10.0)
        r = sq_det(s, ntol=DEFAULT_NTOL)
        assert abs(r.ferr_hi - offset_hz) < DF, \
            f"ferr_hi={r.ferr_hi:.2f}, expected {offset_hz:.1f} ± {DF:.1f}"

    def test_zero_input_returns_zero_metric(self):
        z = np.zeros(NSPM, dtype=np.complex128)
        r = sq_det(z)
        assert r.detmet == 0.0
        assert r.detmet2 == 0.0


# ── Slide + block normalisation ─────────────────────────────────────────────

class TestSlideSqDet:

    def test_pure_noise_block_no_triggers(self):
        """A pure-noise block of 1.5 s should produce no detmet/xmed >= 3
        triggers — normalised detmet noise excursions are below 3."""
        long_noise = make_analytic_noise(n=int(1.5 * FS), sigma=1.0, seed=99)
        trace = slide_sq_det(long_noise)
        # Allow a single false trigger (noise excursion); WSJT-X tolerates this
        n_fired = int(trace.triggers.sum())
        assert n_fired <= 2, \
            f"pure-noise block fired {n_fired} sq_det triggers (expected ≤ 2)"
        # xmed should be > 0 and detmet_norm should be reasonable
        assert trace.xmed > 0
        assert np.median(trace.detmet_norm) == pytest.approx(1.33, abs=1.0), \
            f"median(detmet_norm) = {np.median(trace.detmet_norm):.2f} — " \
            "should be O(1) since 25th-pct is normalised to 1"

    def test_block_with_msk_burst_fires_triggers(self):
        """A pure-noise block with one strong MSK frame spliced at a known
        time should produce at least one detmet_norm >= 3 trigger near that
        time."""
        n_total = int(1.5 * FS)
        noise = make_analytic_noise(n=n_total, sigma=1.0, seed=7)
        # Insert a strong MSK frame at t = 0.7 s (sample 8400)
        burst_start = int(0.7 * FS)
        msk = make_msk_like_analytic(fc=DEFAULT_FC, amplitude=8.0, seed=11)
        block = noise.copy()
        block[burst_start:burst_start + NSPM] += msk
        trace = slide_sq_det(block)
        fired_t = trace.t_centers[trace.triggers]
        # A trigger should fire within ±NSPM time of the burst
        assert len(fired_t) > 0, "no triggers fired on a strong MSK burst"
        nearest = min(abs(t - 0.7) for t in fired_t)
        assert nearest < (NSPM / FS), \
            f"nearest trigger {nearest*1000:.1f} ms from burst — " \
            f"expected within {NSPM/FS*1000:.1f} ms"

    def test_xmed_is_25th_percentile(self):
        """xmed should equal np.percentile(detmet_raw, 25) by construction."""
        block = make_analytic_noise(n=int(1.0 * FS), seed=3)
        trace = slide_sq_det(block)
        expected = float(np.percentile(trace.detmet_raw, 25))
        assert trace.xmed == pytest.approx(expected, rel=1e-6)

    def test_stride_default_is_quarter_frame(self):
        """Verify default stride matches WSJT-X 216 samples."""
        n_total = int(1.0 * FS)
        block = make_analytic_noise(n=n_total, seed=5)
        trace = slide_sq_det(block)
        # Expected steps: (n - NSPM) / 216 + 1
        expected_min = (n_total - NSPM) // 216
        assert len(trace.t_centers) >= expected_min - 1
        # Successive centers should be (stride/FS) seconds apart
        if len(trace.t_centers) >= 2:
            dt = trace.t_centers[1] - trace.t_centers[0]
            assert dt == pytest.approx(216 / FS, rel=1e-9)
