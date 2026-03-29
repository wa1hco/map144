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
"""AirspyHFSource: SoapySDR wrapper for Airspy HF+ IQ streaming.

Presents the same ``sample_queue`` / ``start()`` / ``stop()`` interface as
``FlexDAXIQ`` so ``run_radio_source`` can drive it without special-casing.

Hardware notes
--------------
The Airspy HF+ (and HF+ Discovery) outputs complex float32 IQ already
normalised to ±1.0 — no ADC scale factor is applied at ingress.

Sample rate
-----------
The pipeline expects 48 kHz IQ.  ``AirspyHFSource`` first attempts to
configure the device at 48 kHz directly.  If that rate is not available
it falls back to 96 kHz and decimates each chunk by 2 using a short FIR
anti-aliasing filter before enqueueing.

Timestamps
----------
SoapyAirspyHF does not provide hardware timestamps.  Each chunk is
stamped with the wall-clock time of the *first sample* in that chunk,
derived by tracking the cumulative sample count since ``start()`` was
called:

    t_chunk = t_start + abs_samples_before_chunk / hw_sample_rate

Fractional seconds are encoded as integer picoseconds, matching the WAV
source convention so that ``processing.py`` can use the same 1e-12 scale
factor for both sources.

Dependencies
------------
    pip install SoapySDR          (or install with GNU Radio)
    SoapyAirspyHF driver module   (usually ships with Airspy tools)

The SoapySDR driver for Airspy HF+ is provided by the ``SoapyAirspyHF``
module, typically installed alongside ``airspyhf-host`` tools.
"""

import queue
import threading
import time
from typing import Optional

import numpy as np

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    _SOAPY_AVAILABLE = True
except ImportError:
    _SOAPY_AVAILABLE = False

# Chunk size fed to the processing pipeline (samples at the target 48 kHz rate)
_CHUNK_SAMPLES = 1024   # ≈ 21 ms per chunk — matches FlexRadio DAXIQ packet cadence

# Target sample rate for the pipeline
_TARGET_RATE = 48000

# Preferred hardware rates to try, in order.  96 kHz with 2× decimate is the
# reliable fallback when 48 kHz is not offered by the firmware.
_PREFERRED_HW_RATES = [48000, 96000]


def _decimate2(samples: np.ndarray) -> np.ndarray:
    """Simple 2× FIR decimate (anti-alias filter + downsample by 2).

    Uses a 15-tap half-band FIR (Kaiser-windowed sinc) to attenuate
    aliases before keeping every other sample.  The filter is pre-computed
    once at import time.
    """
    return _DECIMATE2_FILTER(samples)


def _build_decimate2_filter():
    """Return a callable that applies a 2× anti-alias FIR + downsample."""
    from scipy.signal import firwin, lfilter
    taps = firwin(15, 0.45).astype(np.float32)  # cutoff 0.45 × Nyquist
    zi   = [np.zeros(len(taps) - 1)]             # mutable state for lfilter

    def _apply(samples: np.ndarray) -> np.ndarray:
        # Apply to I and Q independently so we can keep complex dtype
        real = samples.real.astype(np.float32)
        imag = samples.imag.astype(np.float32)
        filtered_r, zi[0] = _lfilter_r(taps, 1.0, real, zi=zi[0])
        filtered_i = lfilter(taps, 1.0, imag)
        combined   = (filtered_r[::2] + 1j * filtered_i[::2]).astype(np.complex64)
        return combined

    from scipy.signal import lfilter as _lf

    def _lfilter_r(b, a, x, zi):
        y, zo = _lf(b, a, x, zi=zi)
        return y, zo

    _apply.__name__ = '_decimate2_apply'
    return _apply


_DECIMATE2_FILTER = _build_decimate2_filter()


class _AirspyPacket:
    """Minimal packet compatible with the VitaPacket interface expected by runtime.py."""
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


class AirspyHFSource:
    """SoapySDR wrapper for Airspy HF+ that mimics the FlexDAXIQ queue interface.

    Usage::

        src = AirspyHFSource(center_freq_mhz=50.260)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``pkt.samples`` is complex64 in ±1.0 scale.  ``pkt.timestamp_int`` and
    ``pkt.timestamp_frac`` use the same picosecond encoding as the WAV source:
    ``t = timestamp_int + timestamp_frac * 1e-12``.
    """

    def __init__(self, center_freq_mhz: float = 50.260, target_rate: int = _TARGET_RATE):
        self.center_freq_mhz = center_freq_mhz
        self.target_rate     = target_rate
        self.sample_queue    = queue.Queue(maxsize=4000)

        self._sdr       = None
        self._stream    = None
        self._thread    = None
        self._running   = False
        self._hw_rate   = None     # actual hardware sample rate
        self._decimate  = False    # True when hw_rate == 2 × target_rate

        # Frequency info exposed for _get_tuned_frequency_mhz
        self.center_freq_mhz_actual = center_freq_mhz

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if not _SOAPY_AVAILABLE:
            raise RuntimeError(
                "SoapySDR Python module not found.  "
                "Install it with your package manager or 'pip install SoapySDR', "
                "and ensure the SoapyAirspyHF driver module is on the SoapySDR path."
            )

        self._sdr = SoapySDR.Device({'driver': 'airspyhf'})

        # Choose the best available hardware sample rate
        available = [int(r) for r in self._sdr.listSampleRates(SOAPY_SDR_RX, 0)]
        hw_rate = None
        for preferred in _PREFERRED_HW_RATES:
            if preferred in available:
                hw_rate = preferred
                break
        if hw_rate is None:
            # Fall back to lowest available rate ≥ target; decimate later if needed
            above = sorted(r for r in available if r >= self.target_rate)
            hw_rate = above[0] if above else max(available)

        self._hw_rate  = hw_rate
        self._decimate = (hw_rate == self.target_rate * 2)

        self._sdr.setSampleRate(SOAPY_SDR_RX, 0, float(hw_rate))
        self._sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq_mhz * 1e6)
        self.center_freq_mhz_actual = self._sdr.getFrequency(SOAPY_SDR_RX, 0) / 1e6

        # Read-back the actual rate (may differ slightly from requested)
        self._hw_rate = int(self._sdr.getSampleRate(SOAPY_SDR_RX, 0))

        self._stream  = self._sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self._sdr.activateStream(self._stream)

        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True,
                                         name='airspyhf-recv')
        self._thread.start()
        print(f"[airspy] started: {self.center_freq_mhz_actual:.4f} MHz  "
              f"hw={self._hw_rate} Hz  decimate={self._decimate}", flush=True)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._stream is not None and self._sdr is not None:
            try:
                self._sdr.deactivateStream(self._stream)
                self._sdr.closeStream(self._stream)
            except Exception:
                pass
            self._stream = None
        if self._sdr is not None:
            try:
                self._sdr = None   # SoapySDR Device is closed on GC
            except Exception:
                pass

    # ── Internal receive loop ─────────────────────────────────────────────────

    def _recv_loop(self):
        """Read IQ from the hardware and put _AirspyPacket objects onto sample_queue."""
        hw_rate    = self._hw_rate
        chunk_hw   = _CHUNK_SAMPLES * (2 if self._decimate else 1)
        buff       = np.zeros(chunk_hw, dtype=np.complex64)
        t_start    = time.time()
        abs_samps  = 0   # cumulative samples at hw_rate

        while self._running:
            sr = self._sdr.readStream(self._stream, [buff], chunk_hw,
                                      timeoutUs=1_000_000)
            if sr.ret <= 0:
                # Timeout or error — readStream returns 0 on timeout
                if sr.ret < 0:
                    print(f"[airspy] readStream error {sr.ret}", flush=True)
                continue

            n_read = sr.ret
            chunk  = buff[:n_read].copy()

            # Compute wall-clock time of the first sample in this chunk.
            # We track abs_samps rather than calling time.time() each chunk
            # to avoid jitter that would cause gaps in the spectrogram.
            t_chunk = t_start + abs_samps / hw_rate
            abs_samps += n_read

            if self._decimate and n_read >= 2:
                chunk = _decimate2(chunk)

            # Encode as picoseconds (same convention as WAV source so
            # processing.py uses the same 1e-12 scale path).
            ts_int  = int(t_chunk)
            ts_frac = int((t_chunk - ts_int) * 1e12)

            pkt = _AirspyPacket(chunk.astype(np.complex64), ts_int, ts_frac)
            try:
                self.sample_queue.put_nowait(pkt)
            except queue.Full:
                pass   # Drop and continue — same behaviour as VITAReceiver
