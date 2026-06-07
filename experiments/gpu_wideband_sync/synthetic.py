# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Synthetic-IQ test signal generator for the wideband sync detector.

Wraps the repo's ``generate_msk144.py`` so we can produce controlled
test signals with known ground truth:

    - One or many MSK144 pings
    - At specified positions in time
    - At specified frequency offsets (anywhere in ±20 kHz at 48 kHz fs)
    - At specified SNRs (vs background AWGN)
    - With deterministic noise seeds for reproducibility

The detector under test produces ``Candidate(time_s, freq_hz, snr_db)``
records.  These are compared against the ground-truth labels we
generated here to compute detection rate / false-positive rate at
varying threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroundTruthPing:
    """One MSK144 ping with its ground-truth (time, freq, SNR) labels."""
    message: str            # MSK144 message text (e.g. "CQ K1JT FN20")
    time_s: float           # start-of-ping in seconds
    freq_hz: float          # carrier offset from IQ DC
    snr_db: float           # synthesized SNR vs background noise


@dataclass
class SyntheticIQ:
    """A synthetic IQ buffer with all its ground-truth labels."""
    iq: np.ndarray            # complex64, shape (n_samples,)
    sample_rate_hz: int
    pings: list[GroundTruthPing]
    seed: int                 # noise seed (for reproducibility)


def generate_test_signal(
    pings: list[GroundTruthPing],
    duration_s: float = 15.0,
    sample_rate_hz: int = 48_000,
    noise_sigma: float = 1.0,
    seed: int = 42,
) -> SyntheticIQ:
    """Build a synthetic IQ buffer containing the requested pings + AWGN.

    Each ping is placed at its (time_s, freq_hz, snr_db) and added on top
    of a complex AWGN background with the given ``noise_sigma``.  Ping
    amplitude is scaled so its post-mix in-band power achieves the
    requested SNR relative to noise_sigma.

    TODO #1: currently relies on the repo's ``generate_msk144`` —
    requires the ``msk144code`` binary to be on PATH (provided by
    WSJT-X).  Will fail at import if unavailable; consider a fully
    in-Python fallback that builds a sync-only signal from
    ``sync_template.build_sync_template`` so the detector can be tested
    without WSJT-X installed.

    TODO #2: add freq-drift-during-ping and amplitude-fade options once
    the basic detector is solid — real MS pings often have ±10 Hz of
    drift and a Gamma envelope (already supported by
    ``generate_msk144.generate_ping_msk144code``).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate_hz)
    iq = (
        rng.standard_normal(n_samples).astype(np.float32) +
        1j * rng.standard_normal(n_samples).astype(np.float32)
    ).astype(np.complex64) * np.complex64(noise_sigma)

    # Lazy import — keeps the module importable even without msk144code.
    try:
        from generate_msk144 import generate_ping_msk144code  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "generate_msk144 module not importable.  Run synthetic.py "
            "from the repo root or add the repo to PYTHONPATH."
        ) from exc

    for ping in pings:
        ping_iq, _, _ = generate_ping_msk144code(
            ping.message,
            center_hz=ping.freq_hz,
            delay_s=ping.time_s,
            snr_db=ping.snr_db,
            noise_sigma=noise_sigma,
            sample_rate=sample_rate_hz,
            flat_frame=False,           # realistic Gamma envelope
            width_ms=60,
            pol_scenario='pure-H',
            rng=rng,
        )
        # generate_ping returns ping placed in a buffer of length
        # duration_s*sample_rate_hz; clamp into ours.
        n = min(ping_iq.size, n_samples)
        iq[:n] = iq[:n] + ping_iq[:n].astype(np.complex64)

    return SyntheticIQ(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        pings=list(pings),
        seed=seed,
    )


def single_ping(snr_db: float = 10.0,
                freq_hz: float = 0.0,
                time_s: float = 0.5,
                **kwargs) -> SyntheticIQ:
    """Convenience: one ping at the given (time, freq, SNR)."""
    return generate_test_signal(
        pings=[GroundTruthPing(
            message="CQ K1JT FN20",
            time_s=time_s,
            freq_hz=freq_hz,
            snr_db=snr_db,
        )],
        **kwargs,
    )


def noise_only(duration_s: float = 15.0,
               sample_rate_hz: int = 48_000,
               noise_sigma: float = 1.0,
               seed: int = 42) -> SyntheticIQ:
    """Noise-only buffer for false-positive-rate measurement."""
    return generate_test_signal(
        pings=[],
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        noise_sigma=noise_sigma,
        seed=seed,
    )


if __name__ == "__main__":
    # CLI smoke: generate a single 10-dB ping, print summary.
    s = single_ping(snr_db=10.0, freq_hz=5_000.0)
    print(f"Generated {s.iq.size} samples at {s.sample_rate_hz} Hz "
          f"({s.iq.size / s.sample_rate_hz:.2f} s).")
    print(f"Pings: {len(s.pings)}")
    for p in s.pings:
        print(f"  {p}")
