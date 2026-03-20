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
"""Shared constants and diagnostic helpers for the FlexRadio SmartSDR client.

This module is imported by every other module in the ``flexclient`` package.
It defines the numeric constants, status-code table, and small utility
functions that are too general to belong to any one submodule.

Port constants
--------------
DISCOVERY_PORT  = 4992  UDP port on which FlexRadio hardware broadcasts
                         VITA-49 formatted discovery advertisements.  Also
                         used for TCP command/control (SmartSDR reuses 4992
                         for both UDP discovery and TCP C&C).
FLEX_TCP_PORT   = 4992  TCP port for the SmartSDR command/control API.
VITA_UDP_PORT   = 4991  Default UDP port to which the radio sends DAXIQ
                         VITA-49 IQ packets.  Overridden at runtime by an
                         ephemeral port chosen via ``_pick_udp_listen_port``.

VITA packet type codes
----------------------
VITA_TYPE_IF_DATA  = 0x1  IF Data packet — carries IQ sample payload.
VITA_TYPE_EXT_DATA = 0x3  Extended data packet (not used for DAXIQ IQ).

SAMPLE_RATES
    List of sample rates (Hz) supported by FlexRadio DAX IQ channels:
    [24000, 48000, 96000, 192000].

SMARTSDR_STATUS_MESSAGES
    Dict mapping known 32-bit SmartSDR status codes (returned in command
    response lines as ``R<seq>|<code>|<message>``) to human-readable strings.
    Codes in the 0x5000xxxx range are SmartSDR application-level errors.
    Code 0x00000000 is success.  The table is intentionally incomplete;
    ``_maybe_log_unmapped_status_code`` warns once per novel code so new
    codes can be identified in the field and added.

Logging
-------
A module-level ``log = logging.getLogger(__name__)`` logger is created and
configured with ``basicConfig(level=INFO)`` if no root handler has been
installed yet.  All other ``flexclient`` modules import and use this same
``log`` instance so that the caller can configure logging once at the
application level and have consistent output across the package.

Helper functions
----------------
_format_status_detail(status)
    Returns a human-readable string for a SmartSDR status code, e.g.
    ``"0x50000005 (Incorrect number or type of parameters)"``.

_maybe_log_unmapped_status_code(status)
    Logs a one-time warning when a status code is not in
    ``SMARTSDR_STATUS_MESSAGES``.  Uses a module-level set to suppress
    duplicate warnings for the same code.

_pick_udp_listen_port()
    Asks the OS to bind a UDP socket to port 0, reads back the assigned
    ephemeral port number, closes the socket, and returns the port.  Used
    to find a free local port for the VITA-49 IQ stream without hard-coding
    a port number that might already be in use.
"""

import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DISCOVERY_PORT  = 4992          # UDP broadcast port for Flex discovery

FLEX_TCP_PORT   = 4992          # TCP command/control port

VITA_UDP_PORT   = 4991          # UDP port Flex sends IQ packets to (check your radio)

VITA_TYPE_IF_DATA   = 0x1       # IF data packet (IQ samples)

VITA_TYPE_EXT_DATA  = 0x3       # Extended data

SAMPLE_RATES = [24000, 48000, 96000, 192000]

SMARTSDR_STATUS_MESSAGES = {
    0x00000000: "Success",
    0x50000001: "Unable to get foundation receiver assignment",
    0x50000003: "License check failed, cannot create slice receiver",
    0x50000005: "Incorrect number or type of parameters",
    0x50000016: "Malformed command (parse error, e.g., frequency field)",
    0x5000002C: "Incorrect number of parameters",
    0x5000002D: "Bad field",
    0x50000063: "Operation not allowed (likely)",
    0x50001000: "Command handler rejection",
}

_UNMAPPED_STATUS_CODES_LOGGED: set[int] = set()

def _format_status_detail(status: int) -> str:
    hex_code = f"0x{status:08X}"
    message = SMARTSDR_STATUS_MESSAGES.get(status)
    if message:
        return f"{hex_code} ({message})"
    return f"{hex_code} (unmapped status code)"

def _maybe_log_unmapped_status_code(status: int):
    if status in SMARTSDR_STATUS_MESSAGES:
        return
    if status in _UNMAPPED_STATUS_CODES_LOGGED:
        return
    _UNMAPPED_STATUS_CODES_LOGGED.add(status)
    log.warning(
        "Encountered unmapped SmartSDR status code 0x%08X. "
        "Add it to SMARTSDR_STATUS_MESSAGES when meaning is confirmed.",
        status,
    )

def _pick_udp_listen_port() -> int:
    """Return a free ephemeral local UDP port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()

