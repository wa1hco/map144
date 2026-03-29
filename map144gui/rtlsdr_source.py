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
"""NesdrSmartSource: ctypes wrapper for RTL-SDR (NooElec NESDR Smart) IQ streaming.

Presents the same ``sample_queue`` / ``start()`` / ``stop()`` interface as
``AirspyHFSource`` and ``FlexDAXIQ``.

No pyrtlsdr required — uses librtlsdr.so directly via ctypes.

Hardware notes
--------------
The NooElec NESDR Smart uses the R820T2 tuner + RTL2832U demodulator.
Frequency coverage: ~24 MHz – 1766 MHz (covers 6m at 50.260 MHz and
10m at 28.180 MHz).

The RTL2832U outputs raw interleaved uint8 IQ:
    I = (byte[0::2] - 127.5) / 127.5   → float32 in ±1.0
    Q = (byte[1::2] - 127.5) / 127.5

Sample rate
-----------
The pipeline expects 48 kHz IQ.  The R820T2 minimum reliable sample rate
is ~240 kHz.  This driver uses 960 kHz (a common reliable rate) and
decimates 20× to 48 kHz using a 101-tap FIR anti-alias filter applied
separately to I and Q.

Timestamps
----------
librtlsdr does not provide hardware timestamps.  Timestamps are derived
from the cumulative sample count since ``start()`` was called, encoded as
picoseconds matching the WAV and Airspy source conventions.

Prerequisites
-------------
librtlsdr must be installed:
    sudo apt install librtlsdr2

The RTL-SDR kernel driver must be blacklisted so librtlsdr can claim the
USB device.  Create /etc/modprobe.d/blacklist-rtl.conf containing:
    blacklist dvb_usb_rtl28xxu
    blacklist rtl2832
    blacklist rtl2830
Then reload: sudo rmmod dvb_usb_rtl28xxu 2>/dev/null; true
"""

import ctypes
import queue
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter

# ---------------------------------------------------------------------------
# Load librtlsdr
# ---------------------------------------------------------------------------

def _load_lib():
    for name in ('rtlsdr', 'librtlsdr.so.2', 'librtlsdr.so'):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    for path in ('/usr/lib/x86_64-linux-gnu/librtlsdr.so.2',
                 '/usr/local/lib/librtlsdr.so'):
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    raise RuntimeError(
        "librtlsdr not found.  Install with: sudo apt install librtlsdr2"
    )

try:
    _lib = _load_lib()
    _RTLSDR_AVAILABLE = True
except RuntimeError:
    _lib = None
    _RTLSDR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Decimation filter — 960 kHz → 48 kHz (factor 20)
# ---------------------------------------------------------------------------

_HW_RATE    = 960_000   # hardware sample rate
_TARGET_RATE = 48_000   # pipeline sample rate
_DECIMATE   = _HW_RATE // _TARGET_RATE   # 20

# 101-tap low-pass FIR: cutoff 0.9 × new Nyquist (0.9 × 24 kHz / 480 kHz)
_FIR_TAPS = firwin(101, 0.9 / _DECIMATE).astype(np.float32)

# Persistent filter state for I and Q channels
_zi_i = np.zeros(len(_FIR_TAPS) - 1, dtype=np.float64)
_zi_q = np.zeros(len(_FIR_TAPS) - 1, dtype=np.float64)

def _decimate(iq: np.ndarray) -> np.ndarray:
    """Apply FIR anti-alias filter then downsample by _DECIMATE.

    Maintains state across calls so the filter is continuous across chunk
    boundaries — avoids the transient artifact at the start of each chunk.
    """
    global _zi_i, _zi_q
    i_f, _zi_i = lfilter(_FIR_TAPS, 1.0, iq.real.astype(np.float64), zi=_zi_i)
    q_f, _zi_q = lfilter(_FIR_TAPS, 1.0, iq.imag.astype(np.float64), zi=_zi_q)
    return (i_f[::_DECIMATE] + 1j * q_f[::_DECIMATE]).astype(np.complex64)

# ---------------------------------------------------------------------------
# ctypes callback type
# ---------------------------------------------------------------------------

# void (*rtlsdr_read_async_cb_t)(unsigned char *buf, uint32_t len, void *ctx)
_CB_TYPE = ctypes.CFUNCTYPE(None,
                            ctypes.POINTER(ctypes.c_ubyte),
                            ctypes.c_uint32,
                            ctypes.c_void_p)

# ---------------------------------------------------------------------------
# Packet compatible with VitaPacket interface
# ---------------------------------------------------------------------------

class _RtlPacket:
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

# read_async buffer size in bytes: must be multiple of 512.
# 19200 samples × 2 bytes = 38400 bytes = 75 × 512.
# At 960 kHz: 19200 samples = 20 ms, decimates to 960 samples at 48 kHz.
_BUF_LEN = 38400


class NesdrSmartSource:
    """librtlsdr ctypes wrapper for NooElec NESDR Smart (RTL-SDR).

    Usage::

        src = NesdrSmartSource(center_freq_mhz=50.260)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``pkt.samples`` is complex64 in ±1.0 scale.
    Timestamps use picosecond encoding: t = timestamp_int + timestamp_frac * 1e-12
    """

    def __init__(self, center_freq_mhz: float = 50.260,
                 device_index: int = 0,
                 target_rate: int = _TARGET_RATE):
        self.center_freq_mhz        = center_freq_mhz
        self.device_index           = device_index
        self.target_rate            = target_rate
        self.sample_queue           = queue.Queue(maxsize=4000)
        self.center_freq_mhz_actual = center_freq_mhz

        self._dev     = ctypes.c_void_p(None)
        self._thread  = None
        self._running = False
        self._cb_ref  = None   # keep callback alive

        self._t_start   = 0.0
        self._abs_samps = 0    # cumulative 48 kHz samples delivered

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        if not _RTLSDR_AVAILABLE:
            raise RuntimeError("librtlsdr not available — install librtlsdr2")

        ret = _lib.rtlsdr_open(ctypes.byref(self._dev),
                               ctypes.c_uint32(self.device_index))
        if ret != 0:
            raise RuntimeError(
                f"rtlsdr_open(index={self.device_index}) failed: {ret}.  "
                "Check that the dvb_usb_rtl28xxu kernel module is blacklisted."
            )

        _lib.rtlsdr_set_sample_rate(self._dev, ctypes.c_uint32(_HW_RATE))
        _lib.rtlsdr_set_center_freq(
            self._dev,
            ctypes.c_uint32(int(self.center_freq_mhz * 1e6))
        )
        actual_hz = _lib.rtlsdr_get_center_freq(self._dev)
        self.center_freq_mhz_actual = actual_hz / 1e6

        # Auto gain — let the R820T2 AGC manage level
        _lib.rtlsdr_set_tuner_gain_mode(self._dev, ctypes.c_int(0))
        _lib.rtlsdr_set_agc_mode(self._dev, ctypes.c_int(1))

        _lib.rtlsdr_reset_buffer(self._dev)

        self._t_start   = time.time()
        self._abs_samps = 0
        self._running   = True

        self._cb_ref = _CB_TYPE(self._callback)
        self._thread = threading.Thread(target=self._async_loop, daemon=True,
                                        name='rtlsdr-async')
        self._thread.start()

        print(f"[rtlsdr] started: {self.center_freq_mhz_actual:.4f} MHz  "
              f"hw={_HW_RATE} Hz  decimation={_DECIMATE}x", flush=True)

    def stop(self):
        self._running = False
        if self._dev:
            try:
                _lib.rtlsdr_cancel_async(self._dev)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._dev:
            try:
                _lib.rtlsdr_close(self._dev)
            except Exception:
                pass
            self._dev = ctypes.c_void_p(None)
        self._cb_ref = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _async_loop(self):
        """Runs in a daemon thread; blocks inside rtlsdr_read_async."""
        _lib.rtlsdr_read_async(self._dev, self._cb_ref, None,
                               ctypes.c_uint32(0),        # buf_num: 0 = default
                               ctypes.c_uint32(_BUF_LEN))

    def _callback(self, buf_ptr, buf_len, ctx):
        if not self._running:
            return

        n_bytes = buf_len
        if n_bytes < 2:
            return

        # Convert uint8 interleaved IQ to complex64 ±1.0
        raw = np.ctypeslib.as_array(buf_ptr, shape=(n_bytes,)).copy()
        iq  = ((raw.astype(np.float32) - 127.5) / 127.5).view(np.complex64)

        # Timestamp of first sample in this chunk (at hw rate)
        n_hw = len(iq)
        t_chunk     = self._t_start + self._abs_samps / _HW_RATE
        self._abs_samps += n_hw

        chunk = _decimate(iq)

        ts_int  = int(t_chunk)
        ts_frac = int((t_chunk - ts_int) * 1e12)

        pkt = _RtlPacket(chunk, ts_int, ts_frac)
        try:
            self.sample_queue.put_nowait(pkt)
        except queue.Full:
            pass
