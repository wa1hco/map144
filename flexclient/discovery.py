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
"""FlexRadio UDP discovery: listen for VITA-49 broadcast advertisements.

FlexRadio SmartSDR hardware periodically broadcasts discovery packets on UDP
port 4992 (the same port used for TCP command/control).  Each packet is a
VITA-49 formatted datagram carrying a ``key=value`` ASCII payload that
describes the radio's identity and currently connected GUI clients.

Protocol details
----------------
VITA-49 header (Word 0, big-endian)
    Bits 31-28  Packet type (IF Data = 0x1)
    Bit  27     Class ID present flag
    Bits 23-22  TSI — integer timestamp type
    Bits 21-20  TSF — fractional timestamp type
    Bits 19-16  4-bit sequence number
    Bits 15-0   Packet size in 32-bit words

Class ID (Words 2-3, 8 bytes, present when bit 27 set)
    Upper 32 bits: pad(8) + OUI(24)
    Lower 32 bits: information class code(16) + packet class code(16)
    FlexRadio discovery packets: OUI = 0x001c2d, PacketClass = 0xffff

Payload (following Class ID)
    UTF-8 string of space-separated ``key=value`` pairs, null-terminated.
    Typical keys: ``ip``, ``model``, ``serial``, ``version``, ``nickname``,
    ``callsign``, ``status``, ``inuse_host``, ``available_clients``,
    ``gui_client_ips``, ``gui_client_handles``, ``gui_client_ids``,
    ``gui_client_programs``, ``gui_client_stations``.

Functions
---------
discover(timeout)
    Binds to UDP port 4992 with ``SO_REUSEADDR`` / ``SO_REUSEPORT`` (so
    SmartSDR can run concurrently on the same machine), receives packets
    for up to ``timeout`` seconds, and returns a list of ``FlexRadio``
    objects.  Stops after the first valid FlexRadio advertisement to minimise
    startup latency in single-radio deployments.

_parse_discovery(msg, ip)
    Parses the UTF-8 ``key=value`` payload string into a ``FlexRadio``
    dataclass.  Uses the packet source IP as a fallback if the ``ip`` key is
    absent from the payload.  Extracts GUI client handle and UUID lists for
    subsequent ``client bind`` negotiation.

_format_discovery_summary(msg, fallback_ip)
    Formats a multi-line human-readable summary of the discovery payload for
    logging, including model, nickname, callsign, status, firmware version,
    and a per-entry breakdown of all connected GUI clients.
"""

import socket
import struct
import time
from typing import Optional

from .common import DISCOVERY_PORT, FLEX_TCP_PORT, log
from .models import FlexRadio

def discover(timeout: float = 3.0) -> list[FlexRadio]:
    """
    Send UDP broadcast and listen for Flex radio responses.
    FlexRadio responds with a key=value status string.
    """
    radios = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # SO_REUSEPORT allows multiple processes to bind to the same port
    if hasattr(socket, 'SO_REUSEPORT'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.settimeout(timeout)
    sock.bind(("", DISCOVERY_PORT))  # Bind to port 4992 to receive broadcasts

    log.info("Listening for FlexRadio discovery broadcasts...")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(4096)
            log.debug(f"Received {len(data)} bytes from {addr[0]}")
            
            # FlexRadio sends VITA-49 formatted discovery packets
            # Parse VITA header to extract payload
            if len(data) < 8:
                log.debug(f"Packet too short: {len(data)} bytes")
                continue
            
            # Word 0: VITA header
            word0 = struct.unpack(">I", data[0:4])[0]
            has_class_id = (word0 >> 27) & 0x1
            log.debug(f"VITA header: 0x{word0:08x}, has_class_id={has_class_id}")
            
            # Word 1: stream ID
            offset = 8
            
            # Words 2-3: class ID (if present) - 8 bytes
            if has_class_id:
                if len(data) < offset + 8:
                    log.debug(f"Packet too short for class_id: {len(data)} bytes")
                    continue
                # Check for FlexRadio discovery packet (OUI=0x001c2d, PacketClass=0xffff)
                class_id = struct.unpack(">Q", data[offset:offset+8])[0]
                # VITA-49 Class ID structure (64 bits):
                # Bits 63-32: pad(8) + OUI(24) 
                # Bits 31-16: Information Class Code
                # Bits 15-0:  Packet Class Code
                oui = (class_id >> 32) & 0xFFFFFF
                packet_class = class_id & 0xFFFF
                log.debug(f"OUI=0x{oui:06x}, PacketClass=0x{packet_class:04x}")
                offset += 8
                
                if oui == 0x001c2d and packet_class == 0xffff:
                    # Extract payload (key=value string)
                    payload = data[offset:].decode("utf-8", errors="replace").rstrip('\x00')
                    log.info(_format_discovery_summary(payload, addr[0]))
                    radio = _parse_discovery(payload, addr[0])
                    if radio:
                        radios.append(radio)
                        # Stop after finding first radio to speed up single-radio setups
                        break
            else:
                log.debug("Packet has no class_id")
        except socket.timeout:
            break
        except Exception as e:
            log.warning(f"Discovery recv error: {e}", exc_info=True)

    sock.close()
    return radios

def _parse_discovery(msg: str, ip: str) -> Optional[FlexRadio]:
    """Parse SmartSDR discovery response into a FlexRadio object."""
    # Response format: key=value key=value ...
    # e.g. "radio ip=192.168.1.100 model=FLEX-6600 serial=... version=..."
    kv = {}
    for token in msg.split():
        if "=" in token:
            k, _, v = token.partition("=")
            kv[k.strip()] = v.strip()

    ip_addr = kv.get("ip", ip)
    model   = kv.get("model", "unknown")
    serial  = kv.get("serial", "")
    version = kv.get("version", "")
    gui_handles = [h for h in kv.get("gui_client_handles", "").split(",") if h]
    gui_ids = [cid for cid in kv.get("gui_client_ids", "").split(",") if cid]

    return FlexRadio(ip=ip_addr, port=FLEX_TCP_PORT, model=model,
                     serial=serial, version=version,
                     gui_client_handles=gui_handles,
                     gui_client_ids=gui_ids)

def _format_discovery_summary(msg: str, fallback_ip: str) -> str:
    kv = {}
    for token in msg.split():
        if "=" in token:
            key, _, value = token.partition("=")
            kv[key.strip()] = value.strip()

    ip_addr = kv.get("ip", fallback_ip)
    lines = [
        "Discovery response:",
        f"  model={kv.get('model', 'unknown')} nickname={kv.get('nickname', 'n/a')} callsign={kv.get('callsign', 'n/a')}",
        f"  ip={ip_addr}:{kv.get('port', FLEX_TCP_PORT)} status={kv.get('status', 'unknown')} version={kv.get('version', 'unknown')}",
        f"  in_use_by={kv.get('inuse_host', kv.get('inuse_ip', 'n/a'))} available_clients={kv.get('available_clients', 'n/a')}",
    ]

    gui_ips = kv.get('gui_client_ips', '').split(',') if kv.get('gui_client_ips') else []
    gui_hosts = kv.get('gui_client_hosts', '').split(',') if kv.get('gui_client_hosts') else []
    gui_programs = kv.get('gui_client_programs', '').split(',') if kv.get('gui_client_programs') else []
    gui_stations = kv.get('gui_client_stations', '').split(',') if kv.get('gui_client_stations') else []
    gui_handles = kv.get('gui_client_handles', '').split(',') if kv.get('gui_client_handles') else []
    gui_ids = kv.get('gui_client_ids', '').split(',') if kv.get('gui_client_ids') else []

    gui_count = max(len(gui_ips), len(gui_hosts), len(gui_programs), len(gui_stations), len(gui_handles), len(gui_ids))
    if gui_count > 0:
        lines.append(f"  gui_clients={gui_count}")
        for idx in range(gui_count):
            ip = gui_ips[idx] if idx < len(gui_ips) and gui_ips[idx] else 'n/a'
            host = gui_hosts[idx] if idx < len(gui_hosts) and gui_hosts[idx] else 'n/a'
            program = gui_programs[idx] if idx < len(gui_programs) and gui_programs[idx] else 'n/a'
            station = gui_stations[idx] if idx < len(gui_stations) and gui_stations[idx] else 'n/a'
            handle = gui_handles[idx] if idx < len(gui_handles) and gui_handles[idx] else 'n/a'
            client_id = gui_ids[idx] if idx < len(gui_ids) and gui_ids[idx] else None
            detail = f"    [{idx}] ip={ip} host={host} program={program} station={station} handle={handle}"
            if client_id:
                detail += f" client_id={client_id}"
            lines.append(detail)

    return "\n".join(lines)

