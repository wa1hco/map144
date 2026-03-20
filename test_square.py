#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Development verification script for the IQ squaring operator.

Generates a synthetic complex tone and demonstrates the frequency-doubling
effect of time-domain squaring, which is the basis of the MSK144 tone-pair
detection algorithm in ``detection.py`` and ``analyze_msk144.py``.

Background
----------
MSK144 transmits two tones at fc ± 500 Hz.  Squaring a complex signal
``exp(j·2π·f·t)`` maps frequency ``f`` → ``2f``.  Squaring the two-tone
MSK144 signal therefore produces spectral lines at:

    2·(fc + 500) Hz   and   2·(fc − 500) Hz

These are separated by exactly 2000 Hz in the squared domain, regardless of
the carrier phase.  A peak-pair search at this fixed spacing allows carrier
detection without needing a phase reference.

Two test scenarios (run sequentially)
--------------------------------------
Test 1 — Basic squaring
    A 10 kHz complex tone (single frequency, 48 kHz rate, 100 ms duration)
    is squared in the time domain.  The resulting spectrum shows a single
    line at 20 kHz, confirming the 2× frequency shift.  Plots: original
    spectrum vs. squared spectrum.

Test 2 — Band-limited squaring
    The same 10 kHz tone is first low-pass filtered to ±20 kHz using a
    101-tap FIR filter (``bandlimit_iq``), then squared.  This mirrors the
    preprocessing step in the live pipeline, where an LP filter is applied
    before squaring to prevent out-of-band signals from aliasing into the
    detection band.  Plots: original / bandlimited / squared spectra.

Functions
---------
bandlimit_iq(iq_samples, fs, bandwidth_hz)
    Designs a 101-tap Kaiser-windowed LP FIR (via ``scipy.signal.firwin``)
    with cutoff at ``bandwidth_hz / (fs / 2)`` and applies it independently
    to the I and Q channels using ``lfilter``.  Returns the filtered
    complex signal.

square_iq_time_domain(iq_samples)
    Multiplies each complex sample by itself (``x * x``).  Equivalent to
    ``np.square(iq_samples)`` but written explicitly for clarity.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import firwin, lfilter

def bandlimit_iq(iq_samples, fs, bandwidth_hz):
    """Apply a lowpass filter to limit IQ bandwidth to +/- bandwidth_hz."""
    nyq = fs / 2
    taps = firwin(numtaps=101, cutoff=bandwidth_hz/nyq)
    # Apply to real and imag separately
    i_filt = lfilter(taps, 1.0, np.real(iq_samples))
    q_filt = lfilter(taps, 1.0, np.imag(iq_samples))
    return i_filt + 1j * q_filt

def square_iq_time_domain(iq_samples: np.ndarray) -> np.ndarray:
    """Square the IQ signal in the time domain."""
    return iq_samples * iq_samples

if __name__ == "__main__":
    # Generate a test IQ signal: 10 kHz tone at 48 kHz sample rate
    fs = 48000
    t = np.arange(0, 0.1, 1/fs)
    freq = 10000  # 10 kHz
    iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)

    # Square in time domain
    iq_sq = square_iq_time_domain(iq)

    # Plot spectrum before and after
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.magnitude_spectrum(iq, Fs=fs, scale='dB')
    plt.title("Original IQ Spectrum")
    plt.subplot(1,2,2)
    plt.magnitude_spectrum(iq_sq, Fs=fs, scale='dB')
    plt.title("Squared IQ Spectrum")
    plt.tight_layout()
    plt.show()

        # Bandlimit to +/-20 kHz before squaring
    iq_limited = bandlimit_iq(iq, fs, 20000)
    iq_sq = square_iq_time_domain(iq_limited)

    # Plot spectrum before and after
    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.magnitude_spectrum(iq, Fs=fs, scale='dB')
    plt.title("Original IQ Spectrum")
    plt.subplot(2,2,2)
    plt.magnitude_spectrum(iq_limited, Fs=fs, scale='dB')
    plt.title("Bandlimited IQ Spectrum")
    plt.subplot(2,2,3)
    plt.magnitude_spectrum(iq_sq, Fs=fs, scale='dB')
    plt.title("Squared IQ Spectrum")
    plt.tight_layout()
    plt.show()
