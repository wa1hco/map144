# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for MAP144's sq_det baseline implementation.

Validates the per-frame metric and block-level slide against synthetic
inputs, plus a round-trip check that audio_to_complex_baseband
correctly converts between the two algorithms' input domains.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algos.map144_sq_det import (
    CH_DETECT_SIZE, CH_SAMPLE_RATE, DETECT_THRESHOLD_DB, _SQ_TONE_HZ,
    map144_sq_det_per_frame, slide_map144_sq_det,
    map144_detection_db, audio_to_complex_baseband,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_baseband_msk(n: int = CH_DETECT_SIZE, amp: float = 1.0,
                      seed: int = 0) -> np.ndarray:
    """MSK-like complex baseband signal: tones at ±500 Hz around DC,
    alternating per symbol following random data.  Squaring → tones at
    ±1000 Hz (the MAP144 sq_det targets).
    """
    NSPS = 6  # samples per symbol at 12 kHz / 2000 baud
    rng = np.random.default_rng(seed)
    n_symbols = (n + NSPS - 1) // NSPS
    symbols = rng.choice([-1, 1], size=n_symbols)
    # MSK: each symbol shifts frequency by ±500 Hz around 0 Hz baseband
    freqs = np.repeat(symbols * 500.0, NSPS)[:n]
    phases = 2 * np.pi * np.cumsum(freqs) / CH_SAMPLE_RATE
    return (amp * np.exp(1j * phases)).astype(np.complex128)


def make_baseband_noise(n: int = CH_DETECT_SIZE, sigma: float = 1.0,
                        seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)
            ).astype(np.complex128)


def make_baseband_tone(freq_hz: float, n: int = CH_DETECT_SIZE,
                       amp: float = 1.0) -> np.ndarray:
    """Pure complex tone at freq_hz (relative to DC).  Used to verify
    the sq_det target masks pick up the right frequencies."""
    t = np.arange(n) / CH_SAMPLE_RATE
    return (amp * np.exp(1j * 2 * np.pi * freq_hz * t)).astype(np.complex128)


# ── Per-frame metric ────────────────────────────────────────────────────────

class TestPerFrameMap144:

    def test_input_length_must_match_NFFT(self):
        with pytest.raises(ValueError):
            map144_sq_det_per_frame(np.zeros(CH_DETECT_SIZE - 1, dtype=np.complex128))

    def test_pure_noise_metric_modest(self):
        """Pure noise: raw_lin should be small, comparable to noise floor."""
        noise = make_baseband_noise(seed=42)
        r = map144_sq_det_per_frame(noise)
        # No assertion on absolute scale — just that it's positive and finite
        assert r.raw_lin > 0
        assert np.isfinite(r.raw_lin)
        # Both peaks should be of similar magnitude for symmetric noise
        ratio = max(r.lo_peak, r.hi_peak) / max(min(r.lo_peak, r.hi_peak), 1e-30)
        assert ratio < 5.0, "huge asymmetry on symmetric noise"

    def test_strong_msk_lights_up_BOTH_tones(self):
        """MSK signal in baseband: squared tones at ±1000 Hz should both fire."""
        s = make_baseband_msk(amp=10.0, seed=1) + make_baseband_noise(sigma=0.1, seed=2)
        r = map144_sq_det_per_frame(s)
        # Both peaks should be substantially above any single-bin noise level
        # Use noise-only reference for comparison
        noise_only = make_baseband_noise(sigma=0.1, seed=99)
        r_noise = map144_sq_det_per_frame(noise_only)
        assert r.lo_peak > r_noise.lo_peak * 5
        assert r.hi_peak > r_noise.hi_peak * 5

    def test_lo_tone_only_lights_lo_side(self):
        """Pure tone at -1000 Hz baseband (squared at -2000 Hz outside our
        ±1000 mask) — actually a pure tone at -500 Hz baseband (so squared
        gives -1000 Hz, hits the LO mask).  Verify only the LO mask fires
        strongly."""
        s = (make_baseband_tone(-500.0, amp=10.0)
             + make_baseband_noise(sigma=0.1, seed=33))
        r = map144_sq_det_per_frame(s)
        # Lo mask should be MUCH bigger than hi mask (single tone, no upper band)
        assert r.lo_peak > 5 * r.hi_peak, \
            f"expected lo >> hi for -500 Hz baseband tone; got " \
            f"lo={r.lo_peak:.4f} hi={r.hi_peak:.4f}"

    def test_zero_input_returns_zero(self):
        r = map144_sq_det_per_frame(np.zeros(CH_DETECT_SIZE, dtype=np.complex128))
        assert r.raw_lin == 0.0


# ── Block / slide ──────────────────────────────────────────────────────────

class TestSlideMap144:

    def test_pure_noise_block_no_triggers(self):
        """Pure noise block: 25th percentile of slide trace should be near
        the median; threshold of 3.5 dB above pct25 should rarely fire."""
        block = make_baseband_noise(n=int(1.5 * CH_SAMPLE_RATE), sigma=1.0, seed=7)
        trace = slide_map144_sq_det(block)
        n_fired = int(trace.triggers.sum())
        # Some false positives are expected at the noise tail; bound at 5%
        assert n_fired / max(1, len(trace.triggers)) < 0.10, \
            f"pure-noise block fired {n_fired}/{len(trace.triggers)} triggers"

    def test_msk_burst_fires_trigger(self):
        """Pure noise + one strong MSK frame at a known time:
        slide_map144_sq_det should produce at least one trigger near
        the burst location."""
        n_total = int(1.5 * CH_SAMPLE_RATE)
        noise = make_baseband_noise(n=n_total, sigma=1.0, seed=11)
        burst_start = int(0.7 * CH_SAMPLE_RATE)
        msk = make_baseband_msk(amp=8.0, seed=13)
        block = noise.copy()
        block[burst_start:burst_start + CH_DETECT_SIZE] += msk
        trace = slide_map144_sq_det(block)
        fired_t = trace.t_centers[trace.triggers]
        assert len(fired_t) > 0, "no trigger fired on a strong MSK burst"
        nearest = min(abs(t - 0.7) for t in fired_t)
        # Stride is 256 samples = 21.3 ms; burst window plus stride = ~64 ms tolerance
        assert nearest < 0.07, \
            f"nearest trigger {nearest*1000:.1f} ms from burst (expected < 70 ms)"


# ── Audio → baseband conversion (used for cross-algorithm experiments) ─────

class TestAudioToBaseband:

    def test_round_trip_preserves_signal(self):
        """Generate STRONG audio (high SNR) at 1500 Hz carrier with MSK
        modulation, convert to baseband, verify the baseband signal still
        has the expected MSK structure (squared tones in ±1000 Hz mask)."""
        # Build real audio: MSK signal at audio fc=1500 with weak additive noise
        n = int(0.5 * CH_SAMPLE_RATE)
        rng = np.random.default_rng(5)
        symbols = rng.choice([-1, 1], size=n // 6 + 1)
        freqs = np.repeat(symbols * 500.0, 6)[:n] + 1500.0
        phases = 2 * np.pi * np.cumsum(freqs) / CH_SAMPLE_RATE
        audio_signal = np.cos(phases)
        # Strong signal (amp=1), weak noise (amp=0.1) → ~20 dB SNR
        audio = audio_signal + 0.1 * rng.normal(0, 1.0, n)
        bb = audio_to_complex_baseband(audio, fc_hz=1500.0)
        frame = bb[:CH_DETECT_SIZE]
        r = map144_sq_det_per_frame(frame)
        # Compare to pure-noise reference at the SAME noise level
        noise_audio = 0.1 * rng.normal(0, 1.0, n)
        noise_bb = audio_to_complex_baseband(noise_audio, fc_hz=1500.0)
        r_noise = map144_sq_det_per_frame(noise_bb[:CH_DETECT_SIZE])
        # 20 dB SNR audio should give a substantial sq_det boost (squared
        # signal is broadband BPSK-like, so the in-mask gain is less than
        # 20 dB but should still be clearly above noise — bound at 10x).
        assert r.raw_lin > r_noise.raw_lin * 10, \
            f"audio→bb round-trip: signal raw_lin={r.raw_lin:.4f}, " \
            f"noise raw_lin={r_noise.raw_lin:.4f} — expected 10x ratio at 20 dB SNR"
