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
"""MAP65 timf2 I/Q export — tee a 96 kHz dual-pol stream from the B210 to MAP65.

This is a *tap* off the raw B210 stream, independent of the MSK144 path.  It is
the only consumer that needs the full 192 kHz, so it is fed the raw dual-channel
IQ from inside ``USRPSource._recv_loop`` BEFORE that path's NCO/decimate runs.

Signal flow
-----------
    raw H,V @ 192 kHz   (complex64; IQ-DC = pan_center = 144.100 MHz)
      -> NCO shift -shift_hz   (default 25 kHz: brings 144.125 MHz to DC, so
                                MAP65's window centres on 144.125)
      -> /2 anti-alias decimate (192 kHz -> 96 kHz; exact integer, no resample)
      -> fc32 -> int16  (level_scale, clipped to +-32767)
      -> interleave  I_h, Q_h, I_v, Q_v  (per frame, 2 RF channels)
      -> Linrad timf2 UDP packets -> MAP65  (host:port, default :50002)

Frequency plan
--------------
The B210 hardware LO sits at ``pan_center`` (144.100 MHz), so the DC artifact is
at 144.100 — below the 144.110-144.140 EME activity.  This exporter then shifts
by ``shift_hz = (map65_center - pan_center)`` (default +25 kHz) so the delivered
stream is centred on ``map65_center`` (144.125 MHz, MAP65's conventional QSO
centre).  The DC artifact therefore lands at ``-shift_hz`` (= 144.100 MHz) in
MAP65's 96 kHz window (144.077-144.173 MHz) — at the quiet low edge, below the
digital EME activity.

Conventions
-----------
- IQ stays complex64 through NCO + decimate; the int16 cast is the LAST step
  (transport format only — never feed int16 back into DSP here).
- ``passband_center`` in the timf2 header is the *delivered* stream centre
  (map65_center = 144.125 MHz), NOT the B210 hardware LO / DC (144.100 MHz).
- timf2 samples are native-endian (little-endian on the x86 MAP65 host) int16,
  interleaved I_h, Q_h, I_v, Q_v per frame (Linrad globdef.h NET_RX_STRUCT).
- DSP (NCO/decimate/int16) is kept separate from transport (packetize/UDP).

On-air validation note
-----------------------
The 24-byte timf2 header layout is verified against Linrad's NET_RX_STRUCT, but
the UDP framing details MAP65 relies on for stream continuity — the ring-buffer
``ptr`` accounting, ``buffer_size`` agreement, and ``block_no`` sequencing — are
best confirmed against a live MAP65 (or a second Linrad instance) since the
authoritative spec is the Linrad/MAP65 C source.  These are isolated in
``build_timf2_header`` / ``Map65Exporter._flush`` so they can be adjusted without
touching the DSP.
"""

import logging
import socket
import struct

import numpy as np
from scipy.signal import firwin, upfirdn

logger = logging.getLogger(__name__)

# ── Rates ──────────────────────────────────────────────────────────────────
HW_RATE_IN  = 192_000          # raw B210 rate fed to process()
MAP65_RATE  = 96_000           # MAP65 standard network rate (exact /2 of input)
DECIMATE    = HW_RATE_IN // MAP65_RATE   # 2

# ── timf2 header (Linrad globdef.h NET_RX_STRUCT, native/little-endian) ──────
#   double passband_center;        // 8  centre freq of the stream, MHz
#   int    time;                   // 4  time of day, milliseconds
#   float  userx_freq;             // 4
#   int    ptr;                    // 4  ring-buffer byte offset (mod buffer_size)
#   unsigned short block_no;       // 2  packet sequence number
#   signed char    userx_no;       // 1
#   signed char    passband_direction; // 1
TIMF2_HEADER_FMT  = "<difiHbb"
TIMF2_HEADER_SIZE = struct.calcsize(TIMF2_HEADER_FMT)   # 24
assert TIMF2_HEADER_SIZE == 24, TIMF2_HEADER_SIZE

BYTES_PER_CHANNEL = 4          # one channel per frame: I,Q as int16 = 2 x 2 bytes

# Linrad/MAP65 require each timf2 data packet payload to be EXACTLY this many
# bytes (Linrad globdef.h: "must be a multiple of 48").  The receiver indexes
# its ring buffer in fixed NET_MULTICAST_PAYLOAD chunks via the `ptr` field, so
# a wrong payload size mis-aligns every packet → a comb of spurious lines that
# is independent of the IQ level.  1392 / bytes_per_frame is integral for both
# 1-channel (348) and 2-channel (174) 16-bit formats.
NET_MULTICAST_PAYLOAD = 1392


def build_timf2_header(passband_center_mhz: float, time_ms: int, ptr: int,
                       block_no: int, userx_freq: float = 0.0,
                       userx_no: int = 0, passband_direction: int = 1) -> bytes:
    """Pack a 24-byte Linrad timf2 packet header (native little-endian)."""
    return struct.pack(
        TIMF2_HEADER_FMT,
        float(passband_center_mhz),
        int(time_ms) & 0x7FFFFFFF,       # ms of day fits in int32
        float(userx_freq),
        int(ptr) & 0x7FFFFFFF,
        int(block_no) & 0xFFFF,
        int(userx_no),
        int(passband_direction),
    )


# MAP65's Fortran sync reads samples as INTEGER*2 and rejects the int16
# extremes: a saturated +32767 trips its "out of range" check and crashes
# *_sync.f90, and -32768 has no positive counterpart so abs()/negate overflows.
# Clip just inside the range so an over-driven level can never feed it those
# values.  (The real cure for overload is a sane level — this is a guard.)
_SAMPLE_MAX = 32766


def _clip_int16(x: np.ndarray) -> np.ndarray:
    """Round to nearest and clip just inside the int16 range.

    Avoids the ±full-scale extremes that crash MAP65's INTEGER*2 sync routine.
    """
    return np.clip(np.rint(x), -_SAMPLE_MAX, _SAMPLE_MAX).astype(np.int16)


def _make_decimator(factor: int = DECIMATE, ntaps: int = 65):
    """Stateful integer decimator (complex64 in/out), mirroring usrp_source.

    Prepends the last (ntaps-1) input samples each block and trims the
    corresponding leading outputs, so the streaming output is exactly
    len(in)//factor with no block-boundary transients.
    """
    taps      = firwin(ntaps, 0.9 / factor).astype(np.float32)
    ntaps_m1  = len(taps) - 1
    skip      = ntaps_m1 // factor
    state     = np.zeros(ntaps_m1, dtype=np.complex64)

    def apply(iq: np.ndarray) -> np.ndarray:
        nonlocal state
        x     = np.ascontiguousarray(iq, dtype=np.complex64)
        x_ext = np.concatenate((state, x))
        yr    = upfirdn(taps, x_ext.real, up=1, down=factor)
        yi    = upfirdn(taps, x_ext.imag, up=1, down=factor)
        state = (x[-ntaps_m1:] if len(x) >= ntaps_m1
                 else np.concatenate((state, x))[-ntaps_m1:])
        n_out = len(iq) // factor
        out   = np.empty(n_out, dtype=np.complex64)
        out.real[:] = yr[skip:skip + n_out]
        out.imag[:] = yi[skip:skip + n_out]
        return out

    return apply


def _nco_shift(chunk: np.ndarray, phase: float, step: float):
    """Multiply chunk by exp(j*(phase + step*n)); return (shifted, new_phase).

    ``step`` is radians/sample at HW_RATE_IN.  A negative step shifts a positive
    IF frequency down toward DC.  Phase is carried as float64 to avoid drift.
    """
    n   = len(chunk)
    ph  = phase + step * np.arange(n, dtype=np.float64)
    rot = np.exp(1j * ph).astype(np.complex64)
    out = np.ascontiguousarray(chunk, dtype=np.complex64) * rot
    new_phase = float((phase + step * n) % (2.0 * np.pi))
    return out, new_phase


class Map65Exporter:
    """Decimate raw 192 kHz dual-pol IQ to 96 kHz timf2 and UDP it to MAP65.

    Usage::

        exp = Map65Exporter(host="127.0.0.1", port=50002)
        exp.start()
        # in the recv loop, per raw block (complex64, 192 kHz):
        exp.process(raw_h, raw_v, ts_epoch)
        exp.stop()

    ``process`` is called from USRPSource's recv thread; UDP sends are fire-and-
    forget (errors recorded in ``last_error``, never raised into the recv loop).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50002,
                 pan_center_mhz: float = 144.100,
                 map65_center_mhz: float = 144.125,
                 level_scale: float = 4096.0,
                 buffer_size: int = 1 << 18,
                 n_channels: int = 1):
        # n_channels: 1 = single pol (RF0 only), 2 = dual H+V.
        # MAP65 learns the layout from userx_no: |userx_no| = number of RF
        # channels, sign = sample format (positive = 16-bit int, negative =
        # 32-bit float).  We send 16-bit int, so userx_no = +n_channels.
        # (0 would mean "unknown" → MAP65 shows no signal level.)
        self.host              = host
        self.port              = int(port)
        self.pan_center_mhz    = pan_center_mhz
        self.map65_center_mhz  = map65_center_mhz
        # +shift_hz = how far MAP65 centre sits above the hardware DC.  We shift
        # the signal DOWN by that much (negative NCO step) to put it at baseband.
        self.shift_hz          = (map65_center_mhz - pan_center_mhz) * 1e6
        self.level_scale       = float(level_scale)
        self.buffer_size       = int(buffer_size)
        self.n_channels        = int(n_channels)
        self.userx_no          = int(n_channels)            # +ch count → int16 format
        self._bytes_per_frame  = BYTES_PER_CHANNEL * self.n_channels
        # Payload MUST be exactly NET_MULTICAST_PAYLOAD bytes — size the frame
        # count to hit it (348 for 1 ch, 174 for 2 ch).
        self.frames_per_packet = NET_MULTICAST_PAYLOAD // self._bytes_per_frame

        self.enabled = False
        self._sock   = None

        self._dec_h  = _make_decimator()
        self._dec_v  = _make_decimator()
        self._step   = -2.0 * np.pi * self.shift_hz / HW_RATE_IN
        self._ph_h   = 0.0
        self._ph_v   = 0.0

        self._pending  = bytearray()   # interleaved int16 bytes awaiting a packet
        self._ptr      = 0
        self._block_no = 0

        # Observability (read by UI)
        self.packets_sent = 0
        self.last_error   = ""
        self._logged_lvl  = False   # one-shot level log for calibration

    # ── Public API ──────────────────────────────────────────────────────────

    def apply_settings(self, host: str, port: int, pan_center_mhz: float,
                       map65_center_mhz: float, level_scale: float,
                       n_channels: int = None):
        """Update target + frequency-plan + level (+ channel count) from the UI.

        Safe while live.  Recomputes the NCO shift/step from the new centres.
        Decimator state / NCO phase are left running; the small phase transient
        from a step change is harmless for MAP65.  Changing n_channels changes
        the frame size, so the partial-packet buffer is dropped.
        """
        self.host             = host.strip()
        self.port             = int(port)
        self.pan_center_mhz   = float(pan_center_mhz)
        self.map65_center_mhz = float(map65_center_mhz)
        self.level_scale      = float(level_scale)
        self.shift_hz         = (self.map65_center_mhz - self.pan_center_mhz) * 1e6
        self._step            = -2.0 * np.pi * self.shift_hz / HW_RATE_IN
        if n_channels is not None and int(n_channels) != self.n_channels:
            self.n_channels        = int(n_channels)
            self.userx_no          = int(n_channels)
            self._bytes_per_frame  = BYTES_PER_CHANNEL * self.n_channels
            self.frames_per_packet = NET_MULTICAST_PAYLOAD // self._bytes_per_frame
            self._pending.clear()        # frame size changed — drop partial buffer
            self._logged_lvl       = False

    def start(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.enabled = True
        except OSError as e:
            self.last_error = f"socket: {e}"
            self._sock = None

    def stop(self):
        self.enabled = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._pending.clear()

    def process(self, raw_h: np.ndarray, raw_v: np.ndarray, ts_epoch: float):
        """Push one raw 192 kHz dual-pol block; emits 0+ timf2 UDP packets.

        raw_h, raw_v: complex64 (or convertible), same length, at HW_RATE_IN.
        ts_epoch:     Unix seconds for this block (used for the timf2 time field).
        """
        if not self.enabled or self._sock is None:
            return

        # NCO shift ch0 (H) down to baseband at the MAP65 centre, then /2.
        h, self._ph_h = _nco_shift(raw_h, self._ph_h, self._step)
        h = self._dec_h(h)
        n = len(h)
        if n == 0:
            return

        s = self.level_scale
        if self.n_channels == 2:
            # Dual pol: interleave I_h, Q_h, I_v, Q_v.
            v, self._ph_v = _nco_shift(raw_v, self._ph_v, self._step)
            v = self._dec_v(v)
            frames = np.empty((n, 4), dtype=np.int16)
            frames[:, 0] = _clip_int16(h.real * s)
            frames[:, 1] = _clip_int16(h.imag * s)
            frames[:, 2] = _clip_int16(v.real * s)
            frames[:, 3] = _clip_int16(v.imag * s)
        else:
            # Single pol (RF0 only): interleave I_h, Q_h.
            frames = np.empty((n, 2), dtype=np.int16)
            frames[:, 0] = _clip_int16(h.real * s)
            frames[:, 1] = _clip_int16(h.imag * s)

        if not self._logged_lvl:
            self._logged_lvl = True
            h_rms = float(np.sqrt(np.mean(frames[:, :2].astype(np.float64) ** 2)))
            vtxt  = ""
            if self.n_channels == 2:
                v_rms = float(np.sqrt(np.mean(frames[:, 2:].astype(np.float64) ** 2)))
                vtxt  = f"  V int16 RMS={v_rms:.0f}"
            print(f"[map65] level: H int16 RMS={h_rms:.0f}{vtxt}  "
                  f"peak={int(np.max(np.abs(frames)))}  (scale={self.level_scale:.0f}); "
                  f"userx_no={self.userx_no} (int16, {self.n_channels} ch)", flush=True)

        self._pending += frames.tobytes()
        self._flush(ts_epoch)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self, ts_epoch: float):
        """Emit as many full timf2 packets as the pending buffer allows."""
        pkt_bytes = self.frames_per_packet * self._bytes_per_frame
        time_ms   = int((ts_epoch % 86_400.0) * 1000.0)   # ms of UTC day
        while len(self._pending) >= pkt_bytes:
            payload = bytes(self._pending[:pkt_bytes])
            del self._pending[:pkt_bytes]
            header = build_timf2_header(
                self.map65_center_mhz, time_ms, self._ptr, self._block_no,
                userx_no=self.userx_no,
            )
            dgram = header + payload
            try:
                self._sock.sendto(dgram, (self.host, self.port))
                if self.packets_sent == 0:
                    print(f"[map65] sending {len(dgram)}-byte timf2 packets "
                          f"({TIMF2_HEADER_SIZE}B hdr + {len(payload)}B IQ) "
                          f"-> {self.host}:{self.port}", flush=True)
                self.packets_sent += 1
            except OSError as e:
                self.last_error = f"send: {e}"
                return
            self._ptr      = (self._ptr + pkt_bytes) % self.buffer_size
            self._block_no = (self._block_no + 1) & 0xFFFF
