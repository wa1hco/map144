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
