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
"""VITAReceiver: VITA-49 UDP packet receiver and IQ sample unpacker.

This module receives UDP datagrams from the FlexRadio DAXIQ stream, decodes
the VITA-49 binary header, and delivers ``VitaPacket`` objects to a queue for
the processing pipeline.

VITA-49 packet structure (FlexRadio DAXIQ)
-------------------------------------------
All header words are big-endian (network byte order).

Word 0 — Header
    Bits 31-28  Packet type    (0x1 = IF Data with stream ID)
    Bit  27     Class ID present
    Bit  26     Trailer present
    Bits 25-24  Reserved
    Bits 23-22  TSI            (integer timestamp type: 0=none, 1=UTC, 2=GPS)
    Bits 21-20  TSF            (fractional timestamp: 0=none, 1=sample count,
                                2=real-time picoseconds, 3=free-running)
    Bits 19-16  Sequence number (4-bit, wraps at 16)
    Bits 15-0   Packet size in 32-bit words (includes header, excludes trailer)

Word 1 — Stream ID
    32-bit stream identifier assigned by SmartSDR at ``stream create`` time.
    Used to filter packets to the expected stream when ``filter_sid`` is set.

Words 2-3 — Class ID (8 bytes, present when bit 27 of Word 0 is set)
    Upper 32 bits: pad(8) + OUI(24)   — FlexRadio OUI = 0x001c2d
    Lower 32 bits: information class(16) + packet class(16)

Integer timestamp (4 bytes, present when TSI != 0)
    Seconds in the epoch specified by TSI (typically GPS or Unix).

Fractional timestamp (8 bytes, present when TSF != 0)
    TSF=1: sample count since last integer second.
    TSF=2: picoseconds since last integer second.

Payload — interleaved I/Q as IEEE-754 32-bit floats, **little-endian**
    FlexRadio DAXIQ uses little-endian float32 for the payload despite the
    big-endian VITA-49 header.  Values use a 16-bit-style ADC scale where
    ±32768 represents full scale (not ±1.0).  The consumer is responsible for
    dividing by 32768 to convert to the ±1.0 internal convention.
    ``raw[0::2]`` = I channel, ``raw[1::2]`` = Q channel → combined as
    complex64 ``I + j*Q``.

Trailer (4 bytes, present when bit 26 of Word 0 is set)
    Excluded from the payload slice before IQ unpacking.

Class: VITAReceiver
-------------------
``start()``
    Binds a UDP socket with a 4 MB receive buffer to ``listen_port``, sets a
    1 s ``SO_TIMEOUT`` so the receive loop can check ``_running`` periodically,
    and launches a daemon thread running ``_recv_loop``.

``_recv_loop()``
    Calls ``recvfrom(65536)`` in a tight loop, passes each datagram to
    ``_unpack``, filters by ``filter_sid``, and puts valid packets onto
    ``out_q`` via ``put_nowait``.  Drops the packet and increments
    ``drop_count`` when the queue is full; logs a warning.

``_unpack(data)``
    Decodes the VITA-49 header byte by byte using ``struct.unpack_from``,
    advances an offset pointer through the optional fields, and slices the
    payload.  Returns ``None`` for malformed packets (too short, no IQ pairs).

Sequence gap detection
    A per-stream dict ``_last_seq`` tracks the most recent 4-bit sequence
    number.  On each packet the expected sequence is ``(last + 1) & 0xF``;
    a mismatch logs a warning and accumulates the gap in ``missed_count``.
    ``missed_count`` and ``packet_count`` are readable by the GUI display
    layer for the VITA-49 packet-loss status-bar field.
"""

import socket
import threading
import queue
import struct
import time
from typing import Optional

import numpy as np

from .common import VITA_UDP_PORT, log
from .models import VitaPacket

class VITAReceiver:
    """
    Receives VITA-49 UDP packets from the Flex and unpacks IQ samples.
    Delivers VitaPacket objects to a queue for downstream processing.
    """

    def __init__(self, listen_port: int = VITA_UDP_PORT,
                 stream_id: Optional[int] = None,
                 output_queue: Optional[queue.Queue] = None):
        self.listen_port  = listen_port
        self.filter_sid   = stream_id      # None = accept all streams
        self.out_q        = output_queue or queue.Queue(maxsize=4000)
        self._sock        = None
        self._running     = False
        self._thread      = None
        self.packet_count  = 0
        self.drop_count    = 0
        self.missed_count  = 0
        self._last_seq     = {}   # stream_id -> last 4-bit sequence number
        self._last_drop_log = 0.0  # monotonic time of last drop log message

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.bind(("", self.listen_port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        log.info(f"VITA receiver listening on UDP:{self.listen_port}")

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(65536)
                pkt = self._unpack(data)
                if pkt is None:
                    continue
                if self.filter_sid and pkt.stream_id != self.filter_sid:
                    continue
                self.packet_count += 1
                
                # Check for dropped packets via 4-bit sequence number
                sid = pkt.stream_id
                if sid in self._last_seq:
                    expected = (self._last_seq[sid] + 1) & 0xF
                    if pkt.sequence != expected:
                        missed = (pkt.sequence - expected) & 0xF
                        self.missed_count += missed
                        log.warning(f"Sequence gap on stream 0x{sid:08x}: "
                                    f"expected {expected}, got {pkt.sequence} "
                                    f"({missed} packets missed)")
                self._last_seq[sid] = pkt.sequence

                try:
                    self.out_q.put_nowait(pkt)
                except queue.Full:
                    self.drop_count += 1
                    now = time.monotonic()
                    if now - self._last_drop_log >= 1.0:
                        self._last_drop_log = now
                        log.warning(f"VITA queue full, dropping packet "
                                    f"(total drops: {self.drop_count})")
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.error(f"VITA recv error: {e}")

    # Note: DAXIQ uses IEEE-754 32-bit floats (not fixed-point integers)

    def _unpack(self, data: bytes) -> Optional[VitaPacket]:
        if len(data) < 4:
            return None

        # ── Word 0: VITA-49 header (big-endian throughout) ──────────────────
        # Bits 31-28: packet type (0x1 = IF Data with stream id)
        # Bit  27:    class id present
        # Bit  26:    trailer present
        # Bits 25-24: reserved
        # Bits 23-22: TSI (integer timestamp type: 0=none,1=UTC,2=GPS,3=other)
        # Bits 21-20: TSF (fractional timestamp type: 0=none,1=sample count,
        #                   2=real time picoseconds, 3=free running)
        # Bits 19-16: packet sequence number (4-bit, wraps at 16)
        # Bits 15-0:  packet size in 32-bit words (including header)
        word0        = struct.unpack_from(">I", data, 0)[0]
        pkt_type     = (word0 >> 28) & 0xF
        has_class_id = (word0 >> 27) & 0x1
        has_trailer  = (word0 >> 26) & 0x1
        tsi          = (word0 >> 22) & 0x3
        tsf          = (word0 >> 20) & 0x3
        sequence     = (word0 >> 16) & 0xF
        pkt_size_words = word0 & 0xFFFF

        offset = 4

        # ── Word 1: Stream ID (always present for IF Data packets) ──────────
        if len(data) < offset + 4:
            return None
        stream_id = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        # ── Words 2-3: Class ID (OUI + packet class, 8 bytes if present) ────
        # Upper 32 bits: pad(8) + OUI(24)
        # Lower 32 bits: information class code(16) + packet class code(16)
        class_id = None
        if has_class_id:
            if len(data) < offset + 8:
                return None
            class_id = struct.unpack_from(">Q", data, offset)[0]
            offset += 8

        # ── Integer timestamp (4 bytes if TSI != 0) ──────────────────────────
        timestamp_int = 0
        if tsi != 0:
            if len(data) < offset + 4:
                return None
            timestamp_int = struct.unpack_from(">I", data, offset)[0]
            offset += 4

        # ── Fractional timestamp (8 bytes if TSF != 0) ───────────────────────
        # For Flex: TSF=1 -> sample count, TSF=2 -> picoseconds real time
        timestamp_frac = 0
        if tsf != 0:
            if len(data) < offset + 8:
                return None
            timestamp_frac = struct.unpack_from(">Q", data, offset)[0]
            offset += 8

        # ── Payload: interleaved I/Q as IEEE-754 32-bit floats, big-endian ─
        # Trim trailer if present (1 word = 4 bytes at end of packet)
        payload_end = pkt_size_words * 4
        if has_trailer:
            payload_end -= 4
        payload = data[offset:payload_end]

        n_words = len(payload) // 4
        if n_words < 2:
            return None

        # Unpack as little-endian IEEE-754 32-bit floats (DAXIQ format: payload_endian=little)
        raw = np.frombuffer(payload[:n_words * 4], dtype="<f4")
        
        # FlexRadio DAXIQ float32 payload uses ±32768 ADC scale.
        # Normalisation to ±1.0 is applied by the consumer (FLEX_DAXIQ_FULL_SCALE in runtime.py).
        
        # Interleaved I, Q pairs -> complex64
        # FlexRadio DAXIQ interleaves I then Q.  Standard orientation: a signal
        # at RF = LO + f produces I+jQ = exp(+j2πft), positive baseband frequency.
        n_samples = n_words // 2
        samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

        return VitaPacket(
            stream_id=stream_id,
            timestamp_int=timestamp_int,
            timestamp_frac=timestamp_frac,
            sequence=sequence,
            samples=samples
        )

