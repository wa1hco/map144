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
"""AirspyHFSource: ctypes wrapper for Airspy HF+ IQ streaming via libairspyhf.

Presents the same ``sample_queue`` / ``start()`` / ``stop()`` interface as
``FlexDAXIQ`` so ``run_radio_source`` can drive it without special-casing.

No SoapySDR required — uses libairspyhf.so directly.

Hardware notes
--------------
The Airspy HF+ outputs complex float32 IQ already normalised to ±1.0.

Sample rate
-----------
The pipeline expects 48 kHz IQ.  ``AirspyHFSource`` queries the hardware for
available rates and prefers 48 kHz.  If only 192/384/768 kHz rates are
available it takes the lowest available rate ≥ 48 kHz and decimates down to
48 kHz using a multi-stage FIR anti-alias filter.

Timestamps
----------
libairspyhf does not provide hardware timestamps.  Each chunk is stamped with
the wall-clock time of the first sample, derived by tracking cumulative sample
count since ``start()`` was called:

    t_chunk = t_start + abs_samples_before_chunk / sample_rate

Fractional seconds are encoded as integer picoseconds, matching the WAV source
convention so that ``processing.py`` can use the same 1e-12 scale factor.
"""

import ctypes
import ctypes.util
import queue
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Load libairspyhf
# ---------------------------------------------------------------------------

def _load_lib():
    for name in ('airspyhf', 'libairspyhf.so.0', 'libairspyhf.so.1'):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    # Try absolute paths common on Linux
    for path in ('/usr/local/lib/libairspyhf.so.0',
                 '/usr/local/lib/libairspyhf.so',
                 '/usr/lib/x86_64-linux-gnu/libairspyhf.so.1'):
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    raise RuntimeError(
        "libairspyhf not found.  Install with: sudo apt install libairspyhf1"
    )

try:
    _lib = _load_lib()
    _AIRSPYHF_AVAILABLE = True
except RuntimeError:
    _lib = None
    _AIRSPYHF_AVAILABLE = False

# ---------------------------------------------------------------------------
# ctypes structures matching airspyhf.h
# ---------------------------------------------------------------------------

class _ComplexFloat(ctypes.Structure):
    _fields_ = [('re', ctypes.c_float), ('im', ctypes.c_float)]


class _Transfer(ctypes.Structure):
    pass  # forward declaration — fields set after class definition


# airspyhf_sample_block_cb_fn signature
_CB_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Transfer))

_Transfer._fields_ = [
    ('device',        ctypes.c_void_p),
    ('ctx',           ctypes.c_void_p),
    ('samples',       ctypes.POINTER(_ComplexFloat)),
    ('sample_count',  ctypes.c_int),
    ('dropped_samples', ctypes.c_uint64),
]

# ---------------------------------------------------------------------------
# Decimate-to-48kHz helper (multi-stage if needed)
# ---------------------------------------------------------------------------

def _build_decimator(factor: int):
    """Return a stateful callable that decimates complex64 by *factor*.

    Uses cascaded half-band FIR stages (factor must be a power of 2).
    Each stage anti-aliases with a 15-tap Kaiser-windowed sinc at 0.45 × Nyquist
    then keeps every other sample.
    """
    from scipy.signal import firwin, lfilter

    assert factor > 0 and (factor & (factor - 1)) == 0, "factor must be power of 2"
    n_stages = factor.bit_length() - 1  # log2(factor)

    taps = firwin(15, 0.45).astype(np.float32)
    # One zi per stage per channel (real / imag share a filter)
    zi_r = [np.zeros(len(taps) - 1) for _ in range(n_stages)]
    zi_i = [np.zeros(len(taps) - 1) for _ in range(n_stages)]

    def _apply(samples: np.ndarray) -> np.ndarray:
        r = samples.real.astype(np.float32)
        i = samples.imag.astype(np.float32)
        for s in range(n_stages):
            r, zi_r[s] = lfilter(taps, 1.0, r, zi=zi_r[s])
            i, zi_i[s] = lfilter(taps, 1.0, i, zi=zi_i[s])
            r = r[::2]
            i = i[::2]
        return (r + 1j * i).astype(np.complex64)

    return _apply


# ---------------------------------------------------------------------------
# Packet compatible with VitaPacket interface expected by runtime.py
# ---------------------------------------------------------------------------

class _AirspyPacket:
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_TARGET_RATE = 48000


class AirspyHFSource:
    """libairspyhf ctypes wrapper that mimics the FlexDAXIQ queue interface.

    Usage::

        src = AirspyHFSource(center_freq_mhz=50.260)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``pkt.samples`` is complex64 in ±1.0 scale.  ``pkt.timestamp_int`` and
    ``pkt.timestamp_frac`` use picosecond encoding:
        t = timestamp_int + timestamp_frac * 1e-12
    """

    def __init__(self, center_freq_mhz: float = 50.260, target_rate: int = _TARGET_RATE):
        self.center_freq_mhz        = center_freq_mhz
        self.target_rate            = target_rate
        self.sample_queue           = queue.Queue(maxsize=4000)
        self.center_freq_mhz_actual = center_freq_mhz

        self._dev       = ctypes.c_void_p(None)
        self._running   = False
        self._hw_rate   = target_rate
        self._decimator = None   # callable or None
        self._cb_ref    = None   # keep callback alive (prevent GC)

        # Timestamp tracking — set in start(), updated in callback
        self._t_start   = 0.0
        self._abs_samps = 0     # cumulative hw samples received

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        if not _AIRSPYHF_AVAILABLE:
            raise RuntimeError("libairspyhf not available — install libairspyhf1")

        # Open device
        ret = _lib.airspyhf_open(ctypes.byref(self._dev))
        if ret != 0:
            raise RuntimeError(f"airspyhf_open failed: {ret}")

        # Query available sample rates — two-step API:
        #   step 1: len=0  → buffer[0] receives the count of available rates
        #   step 2: len=N  → buffer filled with N rate values
        cnt_buf = (ctypes.c_uint32 * 1)()
        _lib.airspyhf_get_samplerates(self._dev, cnt_buf, 0)
        n_rates = cnt_buf[0]
        if n_rates > 0:
            rate_buf = (ctypes.c_uint32 * n_rates)()
            _lib.airspyhf_get_samplerates(self._dev, rate_buf, n_rates)
            available = sorted(rate_buf[i] for i in range(n_rates))
        else:
            available = [768000, 384000, 192000, 96000, 48000]  # safe fallback

        # Pick the lowest available rate ≥ target (prefer exact match)
        hw_rate = None
        for r in sorted(available):
            if r >= self.target_rate:
                hw_rate = r
                break
        if hw_rate is None:
            hw_rate = max(available)

        # Build decimator if needed
        factor = hw_rate // self.target_rate
        if factor > 1:
            self._decimator = _build_decimator(factor)
        else:
            self._decimator = None
        self._hw_rate = hw_rate

        ret = _lib.airspyhf_set_samplerate(self._dev, ctypes.c_uint32(hw_rate))
        if ret != 0:
            raise RuntimeError(f"airspyhf_set_samplerate({hw_rate}) failed: {ret}")

        ret = _lib.airspyhf_set_freq_double(
            self._dev, ctypes.c_double(self.center_freq_mhz * 1e6))
        if ret != 0:
            raise RuntimeError(f"airspyhf_set_freq_double failed: {ret}")

        # Enable built-in IQ correction / IF shift
        _lib.airspyhf_set_lib_dsp(self._dev, ctypes.c_uint8(1))

        self.center_freq_mhz_actual = self.center_freq_mhz  # no readback API

        self._t_start   = time.time()
        self._abs_samps = 0
        self._running   = True

        # Keep a reference so the GC doesn't collect the C callback
        self._cb_ref = _CB_FUNC_TYPE(self._callback)
        ret = _lib.airspyhf_start(self._dev, self._cb_ref, None)
        if ret != 0:
            self._running = False
            raise RuntimeError(f"airspyhf_start failed: {ret}")

        print(f"[airspy] started: {self.center_freq_mhz_actual:.4f} MHz  "
              f"hw={hw_rate} Hz  decimation={factor}x", flush=True)

    def stop(self):
        self._running = False
        if self._dev:
            try:
                _lib.airspyhf_stop(self._dev)
                _lib.airspyhf_close(self._dev)
            except Exception:
                pass
            self._dev = ctypes.c_void_p(None)
        self._cb_ref = None

    # ── Internal callback (called from libairspyhf thread) ────────────────

    def _callback(self, transfer_ptr) -> int:
        if not self._running:
            return 0

        transfer = transfer_ptr.contents
        n        = transfer.sample_count
        if n <= 0:
            return 0

        # Compute timestamp of first sample in this chunk
        t_chunk     = self._t_start + self._abs_samps / self._hw_rate
        self._abs_samps += n

        # Copy samples from C buffer to numpy array
        raw = np.ctypeslib.as_array(
            ctypes.cast(transfer.samples, ctypes.POINTER(ctypes.c_float)),
            shape=(n * 2,)
        ).copy()
        chunk = raw.view(np.complex64)

        if self._decimator is not None:
            chunk = self._decimator(chunk)

        ts_int  = int(t_chunk)
        ts_frac = int((t_chunk - ts_int) * 1e12)

        pkt = _AirspyPacket(chunk, ts_int, ts_frac)
        try:
            self.sample_queue.put_nowait(pkt)
        except queue.Full:
            pass  # drop — same behaviour as VITAReceiver

        return 0
