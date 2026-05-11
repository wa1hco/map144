"""Regression test: per-channel NCO phasor must not drift in magnitude.

Background
----------
The fused Numba kernels ``_nb_mix_fir_sp`` and ``_nb_mix_fir_dp`` in
``map144_app/channelizer.py`` advance the per-channel NCO phase by
``ph *= mix_step`` in complex64.  Float32 rounding asymmetries cause
``|ph|`` to drift away from 1.0 at a per-channel-specific rate; without
renormalization at the chunk boundary, that drift integrates over many
hours of operation and produces ~6 dB P-P banding across channels in
the raw squared-FFT detector output (doubled to ~12 dB after squaring).

This test pins the renormalization invariant — both kernels must
return ``|new_mix_phase[k]| ~ 1.0`` regardless of how many samples have
been processed.  See feedback_dont_punt.md and
project_channelizer_nco_drift.md for the diagnosis history.
"""

import numpy as np
import pytest

from map144_app.channelizer import (
    CHANNEL_SPACING_HZ,
    N_CHANNELS,
    design_channelizer_filter,
    make_channelizer_state,
    apply_channelizer,
)


def _flat_awgn(n_samples: int, seed: int = 0) -> np.ndarray:
    """Generate flat complex AWGN at unit variance per real/imag."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_samples).astype(np.float32) +
            1j * rng.standard_normal(n_samples).astype(np.float32))


def test_mix_phase_unit_magnitude_after_short_run():
    """After a single chunk, every channel's NCO phasor must be unit-magnitude."""
    state = make_channelizer_state(
        n_channels=N_CHANNELS,
        lp_taps=design_channelizer_filter(48000),
        sample_rate=48000,
        channel_offset_hz=1500.0,
    )
    iq = _flat_awgn(4096)
    apply_channelizer(iq, state)

    mags = np.abs(state._mix_phase)
    assert np.allclose(mags, 1.0, atol=1e-5), (
        f"mix_phase magnitudes drifted in one chunk: "
        f"min={mags.min():.6f}, max={mags.max():.6f}, P-P={mags.ptp():.6e}"
    )


def test_mix_phase_unit_magnitude_after_long_run():
    """After many chunks (≈5 minutes of 48 kHz IQ), no per-channel drift.

    Pre-fix behaviour: |mix_phase| P-P drift ~6 dB across 48 channels.
    Post-fix invariant: |mix_phase| stays within 1 ulp of 1.0.
    """
    state = make_channelizer_state(
        n_channels=N_CHANNELS,
        lp_taps=design_channelizer_filter(48000),
        sample_rate=48000,
        channel_offset_hz=1500.0,
    )

    chunk = 4096
    # 5 minutes at 48 kHz = 14_400_000 samples ≈ 3516 chunks.  Use a
    # shorter run for unit-test speed; the renormalization is per-chunk
    # so 200 chunks (~17 s) is sufficient to expose drift if present.
    n_chunks = 200
    for c in range(n_chunks):
        iq = _flat_awgn(chunk, seed=c)
        apply_channelizer(iq, state)

    mags = np.abs(state._mix_phase)
    pp_db = 20 * np.log10(mags.max() / mags.min())
    assert pp_db < 0.1, (
        f"After {n_chunks} chunks, |mix_phase| P-P = {pp_db:.3f} dB "
        f"across {N_CHANNELS} channels; expected < 0.1 dB. "
        f"min={mags.min():.6f}, max={mags.max():.6f}"
    )


def test_channel_power_flat_under_awgn():
    """End-to-end: per-channel output power must be flat under flat AWGN.

    This is the operationally meaningful invariant — the squared-FFT
    detector sees channel power, and per-channel drift in |mix_phase|
    becomes per-channel gain drift in the mixed signal which becomes
    per-channel power drift in the detector.
    """
    state = make_channelizer_state(
        n_channels=N_CHANNELS,
        lp_taps=design_channelizer_filter(48000),
        sample_rate=48000,
        channel_offset_hz=1500.0,
    )

    chunk = 4096
    n_chunks = 200
    accum_power = np.zeros(N_CHANNELS, dtype=np.float64)
    n_out_total = 0
    for c in range(n_chunks):
        iq = _flat_awgn(chunk, seed=c + 1000)
        ch_out = apply_channelizer(iq, state)
        accum_power += (ch_out.real**2 + ch_out.imag**2).sum(axis=1)
        n_out_total += ch_out.shape[1]

    mean_power = accum_power / n_out_total
    # Exclude the half-channel adjacent to Nyquist (k = N/2) where the
    # FIR rolloff legitimately suppresses signal — that's not drift.
    # Inner ±20 channels span the flat passband.
    inner = mean_power[4:-4]
    pp_db = 10 * np.log10(inner.max() / inner.min())
    assert pp_db < 1.0, (
        f"Per-channel mean power P-P = {pp_db:.3f} dB across inner "
        f"channels; expected < 1.0 dB on flat AWGN.  "
        f"inner min={inner.min():.4e}, max={inner.max():.4e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
