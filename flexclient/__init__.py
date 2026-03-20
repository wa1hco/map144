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
"""flexclient — FlexRadio SmartSDR DAXIQ client package.

Provides a complete Python interface to the FlexRadio SmartSDR API for
receiving DAXIQ (Direct Access eXchangeable IQ) sample streams over a local
network.  All public symbols are re-exported from this ``__init__`` so that
callers can simply ``import flexclient`` and access everything from one
namespace.

Package modules
---------------
common.py
    Shared constants (port numbers, VITA packet type codes, valid sample
    rates, SmartSDR status code table), the package-wide ``log`` logger, and
    three small helpers: ``_format_status_detail``,
    ``_maybe_log_unmapped_status_code``, and ``_pick_udp_listen_port``.

models.py
    Lightweight ``dataclass`` definitions:
      * ``FlexRadio`` — discovery metadata (IP, model, serial, GUI client IDs).
      * ``VitaPacket`` — one unpacked VITA-49 IQ packet (stream ID, integer and
        fractional timestamps, complex64 sample array).

discovery.py
    ``discover()`` — listens on UDP port 4992 for VITA-49 formatted broadcast
    advertisements from FlexRadio hardware, parses the ``key=value`` payload
    into ``FlexRadio`` objects, and returns a list of found radios.

tcp_client.py
    ``FlexTCPClient`` — manages the long-lived TCP command/control connection
    to SmartSDR (port 4992).  Sends sequenced commands (``C<seq>|<cmd>``),
    parses responses (``R<seq>|<status>|<payload>``), routes unsolicited status
    lines to a registered callback, and tracks GUI client registrations for
    ``client bind`` negotiation.

setup.py
    ``DAXIQSetup`` — executes the SmartSDR API sequence needed to bring up a
    DAXIQ stream: subscribe to panadapter/slice status, discover existing GUI
    context, assign a DAX IQ channel to a panadapter, create the stream, and
    monitor status updates for centre-frequency and bandwidth changes.

vita.py
    ``VITAReceiver`` — binds a UDP socket, receives VITA-49 packets from the
    radio, decodes the 32-bit big-endian header (packet type, class ID, TSI/TSF
    timestamps, sequence number), extracts the little-endian IEEE-754 float32
    interleaved I/Q payload, and delivers ``VitaPacket`` objects to an output
    queue.  Tracks 4-bit sequence numbers and logs gaps.

client.py
    ``FlexDAXIQ`` — top-level orchestrator.  ``start()`` calls ``discover()``,
    connects ``FlexTCPClient``, performs ``client bind`` (if a GUI client UUID
    is available or supplied), instantiates ``DAXIQSetup`` and ``VITAReceiver``,
    and begins streaming.  ``get_samples()`` drains the output queue.
    ``stop()`` tears down the VITA receiver, DAXIQ stream, and TCP connection.

Typical usage
-------------
::

    from flexclient import FlexDAXIQ

    client = FlexDAXIQ(sample_rate=48000)
    client.start()
    while True:
        pkt = client.get_samples(timeout=1.0)
        if pkt:
            process(pkt.samples)   # complex64 NumPy array
"""

from .common import (
	DISCOVERY_PORT,
	FLEX_TCP_PORT,
	VITA_UDP_PORT,
	VITA_TYPE_IF_DATA,
	VITA_TYPE_EXT_DATA,
	SAMPLE_RATES,
	SMARTSDR_STATUS_MESSAGES,
	_format_status_detail,
	_maybe_log_unmapped_status_code,
	_pick_udp_listen_port,
)
from .models import FlexRadio, VitaPacket
from .discovery import discover, _parse_discovery, _format_discovery_summary
from .tcp_client import FlexTCPClient
from .setup import DAXIQSetup, _extract_key, _parse_freq_to_mhz
from .vita import VITAReceiver
from .client import FlexDAXIQ

__all__ = [
	"DISCOVERY_PORT",
	"FLEX_TCP_PORT",
	"VITA_UDP_PORT",
	"VITA_TYPE_IF_DATA",
	"VITA_TYPE_EXT_DATA",
	"SAMPLE_RATES",
	"SMARTSDR_STATUS_MESSAGES",
	"_format_status_detail",
	"_maybe_log_unmapped_status_code",
	"_pick_udp_listen_port",
	"FlexRadio",
	"VitaPacket",
	"discover",
	"_parse_discovery",
	"_format_discovery_summary",
	"FlexTCPClient",
	"DAXIQSetup",
	"_extract_key",
	"_parse_freq_to_mhz",
	"VITAReceiver",
	"FlexDAXIQ",
]
