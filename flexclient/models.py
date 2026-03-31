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
"""Data-only dataclasses for FlexRadio discovery metadata and VITA-49 IQ packets.

FlexRadio
---------
Populated by ``discovery._parse_discovery()`` from a UDP broadcast payload.
Carries enough information to open a TCP command/control connection and to
identify which GUI client UUIDs are available for ``client bind`` negotiation.

Fields:
    ip                  str        Radio's IP address (from payload or source IP).
    port                int        TCP command port (always 4992).
    model               str        Radio model string, e.g. ``"FLEX-6600"``.
    serial              str        Radio serial number.
    version             str        SmartSDR firmware version string.
    gui_client_handles  list[str]  Handle tokens of currently connected GUI clients.
    gui_client_ids      list[str]  UUID strings of connected GUI clients; used
                                   to select a ``client_id`` for ``client bind``.

VitaPacket
----------
Produced by ``vita.VITAReceiver._unpack()`` for each received UDP datagram.
Delivered to ``FlexDAXIQ.sample_queue`` as the unit of IQ data that callers
consume.

Fields:
    stream_id       int          32-bit VITA-49 stream identifier assigned by
                                 SmartSDR when the DAXIQ stream was created.
    timestamp_int   int          Integer seconds from the VITA-49 TSI field
                                 (GPS epoch or Unix time depending on radio config).
    timestamp_frac  int          Fractional timestamp from the VITA-49 TSF field
                                 (picoseconds when TSF=2; sample count when TSF=1).
    sequence        int          4-bit packet sequence number (0–15, wraps).
                                 Used by ``VITAReceiver`` to detect dropped packets.
    samples         np.ndarray   Complex64 IQ samples: ``I + j*Q``.  Each element
                                 is one sample pair unpacked from the little-endian
                                 IEEE-754 float32 interleaved payload.  Values are
                                 in the range approximately ±1.0 (full-scale float).
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class FlexRadio:
    ip: str
    port: int
    model: str = ""
    serial: str = ""
    version: str = ""
    gui_client_handles: list[str] = field(default_factory=list)
    gui_client_ids: list[str] = field(default_factory=list)

@dataclass
class VitaPacket:
    """Unpacked VITA-49 IQ data packet."""
    stream_id:      int
    timestamp_int:  int         # integer seconds (GPS epoch or Unix)
    timestamp_frac: int         # fractional timestamp — units depend on tsf field
    tsf:            int         # TSF field from header: 0=none, 1=sample count, 2=picoseconds
    sequence:       int
    samples:        np.ndarray  # complex64 array, I+jQ

