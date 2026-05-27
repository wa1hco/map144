# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for TGI (Tone-Guided Integration).

Tests the greedy frame-inclusion behaviour against synthetic inputs of
known character: stable signal, pure noise, mixed (signal + noise),
and bi-Doppler interference.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root for map144_app
from map144_app.sq_det import NSPM, FS, DEFAULT_FC, sq_det
from algos.tgi import tgi_integrate, TGIConfig
from tests.test_sq_det import make_msk_like_analytic, make_analytic_noise, make_analytic_tone


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_ramp_msk_block(n_frames: int, snr_db: float = 0.0,
                        fc: float = DEFAULT_FC, seed: int = 0,
                        ) -> np.ndarray:
    """Build n_frames consecutive MSK frames at the given audio SNR.

    SNR is the in-band signal power / noise power per Hz × 2500 Hz
    (WSJTX 2.5 kHz reference bandwidth convention).  For these tests
    we just scale signal amplitude vs unit-variance complex noise so
    the metric numbers are reasonable; absolute SNR calibration isn't
    critical for the comparative tests.
    """
    rng = np.random.default_rng(seed)
    # Signal: n_frames * NSPM samples of MSK at fc with consistent symbol pattern
    # so coherent integration adds constructively.
    NSPS = 6
    n_total = n_frames * NSPM
    # Use the SAME seed-derived symbols across frames so frames repeat
    # the modulation (matches WSJT-X's repeated-frame transmission).
    n_symbols_per_frame = NSPM // NSPS
    symbols_one_frame = rng.choice([-1, 1], size=n_symbols_per_frame)
    symbols_full = np.tile(symbols_one_frame, n_frames)
    freqs = np.repeat(symbols_full * 500.0, NSPS)[:n_total] + fc
    phases = 2 * np.pi * np.cumsum(freqs) / FS
    # Linear amplitude from SNR.  Use 10**(snr/20) as the signal multiplier
    # against unit-variance noise.  Not a calibrated SNR but monotonic.
    amp = 10 ** (snr_db / 20.0)
    signal = amp * np.exp(1j * phases)
    noise = (rng.normal(0, 1.0, n_total) + 1j * rng.normal(0, 1.0, n_total))
    return (signal + noise).astype(np.complex128)


def make_two_doppler_block(n_frames: int, fc: float = DEFAULT_FC,
                           f_a: float = 0.0, f_b: float = 50.0,
                           amp_a: float = 1.0, amp_b: float = 1.0,
                           seed: int = 0, snr_signal: float = 5.0,
                           ) -> np.ndarray:
    """Two simultaneous reflections from an expanding overdense cloud:
    ray A at carrier offset f_a, ray B at carrier offset f_b, both with
    same symbol pattern (same source).
    """
    rng = np.random.default_rng(seed)
    NSPS = 6
    n_total = n_frames * NSPM
    n_symbols = NSPM // NSPS
    symbols_one = rng.choice([-1, 1], size=n_symbols)
    symbols = np.tile(symbols_one, n_frames)
    freq_traj = np.repeat(symbols * 500.0, NSPS)[:n_total]
    base_phase = 2 * np.pi * np.cumsum(freq_traj) / FS
    # Build ray A
    fc_a = fc + f_a
    ph_a = base_phase + 2 * np.pi * np.arange(n_total) * fc_a / FS
    sig_a = amp_a * np.exp(1j * ph_a)
    # Build ray B (same symbol modulation, different carrier offset)
    fc_b = fc + f_b
    ph_b = base_phase + 2 * np.pi * np.arange(n_total) * fc_b / FS
    sig_b = amp_b * np.exp(1j * ph_b)
    # Combined (with arbitrary relative phase from the rng)
    amp = 10 ** (snr_signal / 20.0)
    signal = amp * (sig_a + sig_b)
    noise = (rng.normal(0, 1.0, n_total) + 1j * rng.normal(0, 1.0, n_total))
    return (signal + noise).astype(np.complex128)


# ── Sanity: TGI degenerates correctly ───────────────────────────────────────

class TestTGIDegenerate:

    def test_naive_no_phase_search_rejects_noise_additions(self):
        """With n_phase_search=1 (no phase search), greedy on pure noise
        should reject most candidates: detmet2 of (noise + noise) ≈
        detmet2 of single noise, so adding doesn't improve.

        This is the ONLY noise-side invariant we test in unit tests.
        The phase-search variant's false-positive behaviour is a property
        for Experiment B to measure across the full noise corpus, not
        bound in tests.
        """
        n_kept_distribution = []
        for seed in range(20):
            block = make_analytic_noise(n=10 * NSPM, sigma=1.0, seed=seed)
            anchor_t = (4 * NSPM) / FS
            r_naive = tgi_integrate(block, anchor_t,
                                    TGIConfig(max_span_frames=4, n_phase_search=1))
            n_kept_distribution.append(r_naive.n_kept)
        mean_n_kept = float(np.mean(n_kept_distribution))
        # Without phase search, mean keep on noise should be modest
        # (some random improvements still happen; assert "much less than max")
        assert mean_n_kept < 4.0, \
            f"naive mean n_kept on noise = {mean_n_kept:.1f} (max possible 9) " \
            f"distribution: {n_kept_distribution}"

    def test_clamps_anchor_at_boundaries(self):
        """Anchor at t=0 should still produce a valid trigger frame."""
        block = make_ramp_msk_block(5, snr_db=10.0, seed=1)
        r = tgi_integrate(block, 0.0)
        assert r.integrated_frame.shape[0] == NSPM
        assert r.n_kept >= 1


# ── Core: TGI accepts in-phase frames ───────────────────────────────────────

class TestTGIInPhase:

    def test_strong_repeated_msk_keeps_all_frames(self):
        """5 frames of strong repeated MSK + noise: TGI should keep all of
        them (+/-1, +/-2 from anchor at frame 2)."""
        block = make_ramp_msk_block(5, snr_db=5.0, seed=10)
        anchor_t = (2 * NSPM) / FS
        r = tgi_integrate(block, anchor_t, TGIConfig(max_span_frames=4))
        assert r.n_kept >= 4, \
            f"expected n_kept >= 4 on strong repeated MSK; got {r.n_kept}"

    def test_detmet_grows_with_kept_frames(self):
        """detmet of integrated frame should be larger than detmet of single-frame anchor."""
        block = make_ramp_msk_block(5, snr_db=3.0, seed=20)
        anchor_t = (2 * NSPM) / FS
        # Single-frame baseline
        anchor_n = int(2 * NSPM)
        single = block[anchor_n : anchor_n + NSPM]
        single_detmet = sq_det(single).detmet
        # TGI
        r = tgi_integrate(block, anchor_t, TGIConfig(max_span_frames=4))
        assert r.detmet > single_detmet, \
            f"TGI detmet={r.detmet:.1f} not > single-frame={single_detmet:.1f}"
        # The ratio should be at least sqrt(n_kept) for in-phase coherent gain
        # (allowing some slack — perfect coherent gain is N, but real signals
        # have phase noise, freq drift, etc.)
        if r.n_kept >= 2:
            ratio = r.detmet / single_detmet
            assert ratio >= 1.3, \
                f"TGI/single = {ratio:.2f} (expected ≥ 1.3 for n_kept={r.n_kept})"


# ── Core: TGI rejects bad-coherence frames ──────────────────────────────────

class TestTGIDopplerSelectivity:

    def test_two_doppler_partially_kept(self):
        """Two reflections at Δf=50 Hz (3 cycles per NSPM): each frame
        contains BOTH; coherent sum across frames should still help
        because both rays are present in every frame.  This is more of a
        sanity test that the algorithm doesn't catastrophically fail."""
        block = make_two_doppler_block(5, f_a=0.0, f_b=50.0,
                                        amp_a=1.0, amp_b=1.0,
                                        snr_signal=5.0, seed=33)
        anchor_t = (2 * NSPM) / FS
        r = tgi_integrate(block, anchor_t, TGIConfig(max_span_frames=4))
        # Should keep at least the trigger and likely some additional frames
        assert r.n_kept >= 1
        # Should not blow up
        assert np.isfinite(r.detmet) and r.detmet > 0


# ── Run-time sanity ─────────────────────────────────────────────────────────

class TestTGIRuntime:

    def test_returns_valid_result_on_short_audio(self):
        """If audio barely contains one frame, TGI should still succeed."""
        block = make_ramp_msk_block(1, snr_db=10.0, seed=5)
        r = tgi_integrate(block, 0.0)
        assert r.n_kept == 1
        # All forward/backward candidates would be out-of-range
        assert len(r.offsets_kept) == 0

    def test_max_span_respected(self):
        """If max_span_frames=2, TGI shouldn't try frames at offset > ±2."""
        block = make_ramp_msk_block(7, snr_db=10.0, seed=7)
        anchor_t = (3 * NSPM) / FS
        r = tgi_integrate(block, anchor_t, TGIConfig(max_span_frames=2))
        for off in r.offsets_kept + r.offsets_rejected:
            assert abs(off) <= 2, f"offset {off} exceeds max_span_frames=2"
