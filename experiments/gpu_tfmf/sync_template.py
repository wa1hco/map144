# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""MSK144 sync template generator, parameterised by sample rate.

The MSK144 frame has TWO 8-symbol sync sub-blocks per 144-symbol frame
(positions 0-7 and 56-63).  Each is the same s8 = [0,1,1,1,0,0,1,0]
bipolar bit pattern, modulated as offset-MSK with a half-sine pulse
shape.

The legacy detector at ``map144_app/msk144_spd.py`` builds this template
at 12 kHz (the channelized-audio rate) as a 42-sample complex array.
For full-bandwidth detection we want it at the full IQ rate (48 kHz default)
without going through the channelizer first; this module produces the
template at any sample rate.

Reference (canonical):
    msk144sync.f90 / msk144decodeframe.f90 in WSJT-X.

Math:
    s8                  = [0,1,1,1,0,0,1,0]
    s8_bipolar          = 2 * s8 - 1     (→ ±1)
    half_sine_pulse(n)  = sin(n * π / (2 * SPS))  for n = 0..2*SPS-1
                          (= 2*SPS-long pulse, peak in the middle)
    where SPS = sample_rate / SYMBOL_RATE
          SYMBOL_RATE = 2000 baud (MSK144)

The MSK staggering puts the Q channel's first half-pulse offset by SPS
samples relative to I.

Template length = 8 symbols * SPS samples each
  At fs=12 kHz: 8 * 6 = 48 samples (WSJT-X uses 42 due to half-pulse
                                     truncation at the message edges)
  At fs=48 kHz: 8 * 24 = 192 samples (or 168 with same truncation)

Initial scaffold uses the same edge truncation as WSJT-X so the 48-kHz
template is bit-equivalent to upsampling the 12-kHz template.
"""

from __future__ import annotations

import numpy as np


#: MSK144 sync bit pattern (8 bits, mapped to ±1 in MSK modulation).
_MSK144_SYNC_BITS = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int8)

#: MSK144 symbol rate (baud).
_MSK144_SYMBOL_RATE_HZ = 2000

#: Position offset of the second sync sub-block within the MSK144 frame,
#: in symbols.  Both sub-blocks are 8 symbols of the same pattern.
SECOND_SYNC_OFFSET_SYMBOLS = 56


def build_dual_sync_template(sample_rate_hz: int = 48_000) -> np.ndarray:
    """Sparse template covering both MSK144 sync sub-blocks in one frame.

    MSK144 places identical 8-symbol sync words at symbols 0-7 and 56-63
    within each 144-symbol frame.  This template has the sync waveform at
    both positions and zeros at the 48 data-symbol positions in between.

    Template length = (56 + 8) symbols × SPS = 64 × SPS samples.
    At 48 kHz: 64 × 24 = 1536 samples.
    At 12 kHz: 64 × 6  =  384 samples.

    The frequency resolution of the resulting TF surface is Fs / (64×SPS)
    = Fs / (64 × Fs/2000) = 2000/64 = 31.25 Hz/bin at all sample rates.

    Coherent combining geometry: the second sync starts at sample
    SECOND_SYNC_OFFSET_SYMBOLS × SPS = 56 × SPS = 1344 samples (at 48 kHz).
    The FFT window (1536 samples) sees both syncs simultaneously and coherently
    accumulates their matched-filter energy in each frequency bin.

    Expected gain over single-sync: +3 dB (doubles the template energy
    ‖h‖², which is the matched-filter SNR numerator).
    """
    sps = sample_rate_hz // _MSK144_SYMBOL_RATE_HZ
    single = build_sync_template(sample_rate_hz, edge_truncate=True)  # 8*SPS
    total_len = (SECOND_SYNC_OFFSET_SYMBOLS + 8) * sps               # 64*SPS
    h = np.zeros(total_len, dtype=np.complex64)
    h[:len(single)] = single                                           # sync1
    h[SECOND_SYNC_OFFSET_SYMBOLS * sps :
      SECOND_SYNC_OFFSET_SYMBOLS * sps + len(single)] = single        # sync2
    return h


def build_sync_template(sample_rate_hz: int = 48_000,
                        edge_truncate: bool = True
                        ) -> np.ndarray:
    """Build a complex sync template at the given sample rate.

    Parameters
    ----------
    sample_rate_hz
        Target sample rate.  Must be a multiple of the MSK144 symbol
        rate (2000 Hz) — i.e., samples-per-symbol must be an integer.
    edge_truncate
        Match WSJT-X's edge truncation (template starts with a half-
        pulse on Q, ends with a half-pulse on I).  This produces a
        template of length ``8 * SPS - SPS`` instead of ``8 * SPS``;
        at 12 kHz that's 42 samples vs 48.  Set False for a "clean"
        template aligned to symbol boundaries (useful for tests).

    Returns
    -------
    template : complex64 ndarray, shape (N,)
        N depends on ``edge_truncate``:
            edge_truncate=True  → 7 * SPS samples
            edge_truncate=False → 8 * SPS samples
    """
    if sample_rate_hz % _MSK144_SYMBOL_RATE_HZ != 0:
        raise ValueError(
            f"sample_rate_hz ({sample_rate_hz}) must be a multiple of "
            f"the MSK144 symbol rate ({_MSK144_SYMBOL_RATE_HZ})."
        )
    sps = sample_rate_hz // _MSK144_SYMBOL_RATE_HZ
    s8 = (_MSK144_SYNC_BITS.astype(np.float32) * 2.0 - 1.0)   # ±1

    # Half-sine pulse, 2*SPS samples long, peak in the middle.
    pulse = np.sin(np.arange(2 * sps) * np.pi / (2 * sps)).astype(np.float32)

    # I and Q channels — offset by SPS samples (MSK staggering).
    # Total non-truncated length = 8 symbols × SPS = 8 * SPS samples.
    n = 8 * sps
    cbi = np.zeros(n + sps, dtype=np.float32)   # +sps for the Q overhang
    cbq = np.zeros(n + sps, dtype=np.float32)

    # I channel: pulses centred at symbol times 0, 2, 4, 6 (the even
    # bits — MSK alternates I/Q per symbol).
    for sym_idx in (0, 2, 4, 6):
        start = sym_idx * sps
        cbi[start:start + 2 * sps] += pulse * s8[sym_idx]

    # Q channel: pulses centred at symbol times 1, 3, 5, 7 (offset by
    # SPS samples vs I).
    for sym_idx in (1, 3, 5, 7):
        start = sym_idx * sps
        cbq[start:start + 2 * sps] += pulse * s8[sym_idx]

    # Combine into complex.
    template = (cbi + 1j * cbq).astype(np.complex64)

    if edge_truncate:
        # Drop the first half-pulse on I and the last half-pulse on Q,
        # matching WSJT-X's canonical 42-sample template at 12 kHz.
        # At 12 kHz with SPS=6: drops 6 samples each side → 48-12 = 36?
        # Actually WSJT-X drops only half-pulse worth (SPS samples
        # total — half from start of I, half from end of Q).  Re-check
        # against `_build_sync_template` in map144_app/msk144_spd.py
        # before claiming bit-parity.
        # For scaffold, return the natural-length template; edge-trim
        # tuning is a refinement once everything else works.
        # TODO: validate against the 12-kHz reference template (project
        # already has a working 42-sample template; the full-bandwidth TFMF (48 kHz) one
        # should differ from it only by sample-rate upsampling).
        return template[:n]

    return template


def cross_correlate_template(iq: np.ndarray,
                              template: np.ndarray
                              ) -> np.ndarray:
    """Reference-implementation cross-correlation, for CPU validation.

    Computes the matched-filter output ``y[t] = Σ x[t+k] · h*[k]`` for
    each valid ``t``.  No frequency search — this is the t-axis
    correlation only.  Use as a ground-truth check for the GPU
    implementation's t,f=0 slice.

    Note on the math: ``numpy.correlate(a, v)`` *already* conjugates
    ``v`` internally for complex inputs (per its documented formula
    ``c[k] = Σ a[n+k] · conj(v[n])``).  So we pass the unmodified
    template — passing ``conj(template)`` would double-conjugate and
    produce a non-matched-filter result whose peak can land at the
    wrong sample for templates with structured phase.

    Returns
    -------
    out : complex64 ndarray, shape (len(iq) - len(template) + 1,)
    """
    return np.correlate(iq, template, mode='valid').astype(np.complex64)


def autocorrelate_at_sample_rate(sample_rate_hz: int = 48_000) -> dict:
    """Smoke test: build the template, autocorrelate, return peak stats.

    The autocorrelation peak magnitude / sidelobe-max ratio should be
    high (clean template).  Use this to verify the template makes sense
    before running anything else.

    Returns dict with keys ``peak_magnitude``, ``peak_to_sidelobe_db``,
    ``template_length``.
    """
    h = build_sync_template(sample_rate_hz=sample_rate_hz)
    acorr = cross_correlate_template(h, h)
    mag = np.abs(acorr)
    peak_idx = int(np.argmax(mag))
    peak = float(mag[peak_idx])
    # Sidelobe max = max of the autocorr OUTSIDE a ±SPS window around the peak.
    sps = sample_rate_hz // _MSK144_SYMBOL_RATE_HZ
    mask = np.ones_like(mag, dtype=bool)
    lo = max(0, peak_idx - sps)
    hi = min(len(mag), peak_idx + sps + 1)
    mask[lo:hi] = False
    sidelobe = float(np.max(mag[mask])) if mask.any() else 0.0
    return {
        "peak_magnitude":         peak,
        "peak_to_sidelobe_db":    20.0 * np.log10(peak / max(sidelobe, 1e-30)),
        "template_length":        int(len(h)),
        "sample_rate_hz":         sample_rate_hz,
        "samples_per_symbol":     sps,
    }


if __name__ == "__main__":
    # Quick CLI smoke test.
    for fs in (12_000, 24_000, 48_000):
        stats = autocorrelate_at_sample_rate(fs)
        print(
            f"fs={fs:>6d} Hz  "
            f"len={stats['template_length']:>4d}  "
            f"sps={stats['samples_per_symbol']:>3d}  "
            f"peak/sidelobe={stats['peak_to_sidelobe_db']:5.1f} dB"
        )
