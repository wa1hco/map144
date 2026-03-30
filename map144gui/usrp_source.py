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
"""USRPSource: UHD Python wrapper for USRP B210 IQ streaming.

Presents the same ``sample_queue`` / ``start()`` / ``stop()`` interface as
``AirspyHFSource`` and ``NesdrSmartSource``.

Requires the UHD Python package (``import uhd``).  Install with:
    sudo apt install python3-uhd uhd-host
    sudo uhd_images_downloader

Hardware / signal chain
-----------------------
The AD9361 RF front-end is a direct-conversion receiver.  Its analog LPF
has a minimum bandwidth of 200 kHz.  Requesting 192 kHz from UHD puts the
AD9361's internal HB filter chain to work: the analog LPF (200 kHz) is well
inside the ±96 kHz Nyquist of the 192 kHz output, so no aliasing occurs in
the analog domain.  The AD9361's internal decimation and DC/quadrature
correction calibrations are applied before samples reach the host.

LO offset and NCO
-----------------
Direct-conversion receivers have LO leakage that produces a DC artifact at
0 Hz IF.  Even with AD9361 calibration the artifact drifts with temperature
and can block signals tuned exactly to the center frequency.

To avoid this the hardware LO is tuned ``lo_offset_hz`` below the target so
the desired signal arrives at ``+lo_offset_hz`` in the IF.  The host NCO
then multiplies by exp(-j·2π·lo_offset·n/hw_rate) **before** the
decimation FIR, bringing the signal to DC and shifting the LO artifact to
``-lo_offset_hz``.  The 4× decimation FIR (65 taps, cutoff ≈ 21.6 kHz)
rejects the artifact at −40 kHz with 60+ dB of attenuation.

The NCO must run at the hardware rate (192 kHz) before decimation.
Running it after decimation at 48 kHz would require shifting from +40 kHz,
which exceeds the 24 kHz Nyquist of the output — the shift would alias.

Pipeline contract
-----------------
Output: 48 kHz complex IQ, ±1.0 scale, timestamps as picoseconds
    t = timestamp_int + timestamp_frac * 1e-12

Prerequisites
-------------
    sudo apt install python3-uhd uhd-host
    sudo uhd_images_downloader
"""

import queue
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter

# ---------------------------------------------------------------------------
# Probe for UHD Python package at import time — fail gracefully
# ---------------------------------------------------------------------------

try:
    import uhd as _uhd
    _UHD_AVAILABLE = True
except ImportError:
    _uhd = None
    _UHD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Packet compatible with VitaPacket interface expected by runtime.py
# ---------------------------------------------------------------------------

class _USRPPacket:
    __slots__ = ('samples', 'timestamp_int', 'timestamp_frac')

    def __init__(self, samples, timestamp_int, timestamp_frac):
        self.samples        = samples
        self.timestamp_int  = timestamp_int
        self.timestamp_frac = timestamp_frac


# ---------------------------------------------------------------------------
# Sample rates and decimation
# ---------------------------------------------------------------------------
# Request 192 kHz from UHD.  The AD9361 handles all internal decimation from
# its oversampled ADC; the analog LPF (min 200 kHz) fits cleanly inside the
# ±96 kHz Nyquist.  A single host 4× FIR stage then delivers 48 kHz to the
# pipeline.
#
# LO offset = 40 kHz.  After the host NCO shifts the signal to DC, the LO
# artifact sits at −40 kHz.  The 4× FIR cutoff is ≈21.6 kHz, so the
# artifact is 18 kHz into the stopband — well over 60 dB rejection.
# A smaller offset (e.g. 25 kHz) would place the artifact only 1 kHz past
# the cutoff, yielding insufficient rejection at 48 kHz output.

_HW_RATE     = 192_000   # request from UHD; AD9361 decimates to this rate internally
_TARGET_RATE =  48_000   # pipeline output rate
_DECIMATE    = _HW_RATE // _TARGET_RATE   # 4

# 65-tap anti-alias FIR for 4× decimation.
# Cutoff = 0.9 / 4 = 0.225 of input Nyquist = 0.225 × 96 kHz ≈ 21.6 kHz.
_FIR_TAPS = firwin(65, 0.9 / _DECIMATE).astype(np.float32)


def _make_decimator():
    """Return a stateful 4× decimation function (per-instance state)."""
    zi_i = np.zeros(len(_FIR_TAPS) - 1, dtype=np.float64)
    zi_q = np.zeros(len(_FIR_TAPS) - 1, dtype=np.float64)

    def _apply(iq: np.ndarray) -> np.ndarray:
        nonlocal zi_i, zi_q
        r = iq.real.astype(np.float64)
        i = iq.imag.astype(np.float64)
        r, zi_i = lfilter(_FIR_TAPS, 1.0, r, zi=zi_i)
        i, zi_q = lfilter(_FIR_TAPS, 1.0, i, zi=zi_q)
        r = r[::_DECIMATE]
        i = i[::_DECIMATE]
        return (r + 1j * i).astype(np.complex64)

    return _apply


# Receive buffer: 20 ms at 192 kHz = 3840 samples → 960 after 4× decimation.
_RECV_SIZE = 3840


class USRPSource:
    """UHD Python wrapper for USRP B210 IQ streaming.

    Usage::

        src = USRPSource(center_freq_mhz=50.260)
        src.start()
        while running:
            pkt = src.sample_queue.get(timeout=1.0)
            process(pkt.samples, pkt.timestamp_int, pkt.timestamp_frac)
        src.stop()

    ``pkt.samples`` is complex64 in ±1.0 scale at 48 kHz.
    Timestamps use picosecond encoding: t = timestamp_int + timestamp_frac * 1e-12
    """

    def __init__(self, center_freq_mhz: float = 50.260,
                 target_rate: int = _TARGET_RATE,
                 gain_db: float = 30.0,
                 antenna: str = "RX2",
                 lo_offset_hz: float = 40_000.0,
                 device_args: str = ""):
        if not _UHD_AVAILABLE:
            raise RuntimeError(
                "UHD Python package not found — install with: sudo apt install python3-uhd"
            )

        self.center_freq_mhz        = center_freq_mhz
        self.target_rate            = target_rate
        self.gain_db                = gain_db
        self.antenna                = antenna
        self.lo_offset_hz           = lo_offset_hz
        self.device_args            = device_args
        self.sample_queue           = queue.Queue(maxsize=4000)
        self.center_freq_mhz_actual = center_freq_mhz

        self._usrp      = None
        self._streamer  = None
        self._thread    = None
        self._running   = False
        self._decimator = None
        self._nco_phase = 0.0   # running NCO phase accumulator (radians)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        self._usrp = _uhd.usrp.MultiUSRP(self.device_args)

        self._usrp.set_rx_rate(_HW_RATE, 0)
        actual_rate = self._usrp.get_rx_rate(0)

        # Tune LO below the target so the signal arrives at +lo_offset_hz in
        # the IF.  The NCO in _recv_loop then shifts it to DC while pushing
        # the LO leakage artifact to -lo_offset_hz (stopband of the FIR).
        lo_tune_hz = self.center_freq_mhz * 1e6 - self.lo_offset_hz
        tune_result = self._usrp.set_rx_freq(
            _uhd.libpyuhd.types.tune_request(lo_tune_hz), 0
        )
        # Report the target RF frequency, not the raw LO position.
        self.center_freq_mhz_actual = (tune_result.actual_rf_freq + self.lo_offset_hz) / 1e6

        self._usrp.set_rx_gain(self.gain_db, 0)
        self._usrp.set_rx_antenna(self.antenna, 0)

        # Create RX streamer — fc32 (complex float32) from host, sc16 over wire.
        st_args = _uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self._streamer = self._usrp.get_rx_stream(st_args)

        # Synchronise USRP internal clock to system wall time so that
        # packet timestamps match the Unix epoch expected by processing.py.
        self._usrp.set_time_now(_uhd.types.TimeSpec(time.time()), 0)

        # Start continuous streaming.
        stream_cmd = _uhd.types.StreamCMD(_uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self._streamer.issue_stream_cmd(stream_cmd)

        self._decimator = _make_decimator()
        self._nco_phase = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True,
                                        name='usrp-recv')
        self._thread.start()

        print(f"[usrp] started: {self.center_freq_mhz_actual:.4f} MHz  "
              f"LO offset={self.lo_offset_hz/1e3:+.1f} kHz  "
              f"hw={actual_rate:.0f} Hz  decimation={_DECIMATE}x  "
              f"out={_TARGET_RATE} Hz  "
              f"gain={self.gain_db} dB  antenna={self.antenna}", flush=True)

    def stop(self):
        self._running = False
        if self._streamer is not None:
            try:
                stream_cmd = _uhd.types.StreamCMD(_uhd.types.StreamMode.stop_cont)
                self._streamer.issue_stream_cmd(stream_cmd)
            except Exception:
                pass
            self._streamer = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._usrp = None

    # ── Internal receive loop (runs in daemon thread) ─────────────────────

    def _recv_loop(self):
        recv_buffer = np.zeros(_RECV_SIZE, dtype=np.complex64)
        metadata    = _uhd.types.RXMetadata()

        while self._running:
            try:
                n = self._streamer.recv(recv_buffer, metadata, timeout=1.0)
            except Exception as exc:
                print(f"[usrp] recv error: {exc}", flush=True)
                time.sleep(0.1)
                continue

            if metadata.error_code not in (
                _uhd.types.RXMetadataErrorCode.none,
                _uhd.types.RXMetadataErrorCode.overflow,
            ):
                print(f"[usrp] metadata error: {metadata.strerror()}", flush=True)
                continue

            if n <= 0:
                continue

            chunk = recv_buffer[:n].copy()

            # NCO: shift signal from +lo_offset_hz to DC at the hardware rate
            # (192 kHz) BEFORE the decimation FIR.  The FIR then rejects the
            # LO artifact at -lo_offset_hz.  The NCO must run at _HW_RATE, not
            # the decimated target_rate — shifting by 40 kHz after decimation
            # to 48 kHz would exceed the 24 kHz Nyquist and alias.
            if self.lo_offset_hz != 0.0:
                step  = -2.0 * np.pi * self.lo_offset_hz / _HW_RATE
                ns    = len(chunk)
                phase = self._nco_phase + step * np.arange(ns, dtype=np.float64)
                chunk = (chunk * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
                self._nco_phase = float((self._nco_phase + step * ns) % (2.0 * np.pi))

            chunk = self._decimator(chunk)

            ts_int  = metadata.time_spec.get_full_secs()
            ts_frac = int(metadata.time_spec.get_frac_secs() * 1e12)

            pkt = _USRPPacket(chunk, ts_int, ts_frac)
            try:
                self.sample_queue.put_nowait(pkt)
            except queue.Full:
                pass  # drop — same as other sources
