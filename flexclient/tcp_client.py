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
"""FlexTCPClient: SmartSDR TCP command/control connection and protocol parser.

This module implements the long-lived TCP connection to SmartSDR (port 4992)
and the line-oriented message protocol used to send commands and receive both
direct responses and unsolicited status messages.

SmartSDR TCP message protocol
------------------------------
All messages are UTF-8 text, newline-terminated (``\\n``).

Command (client → radio)
    ``C<seq>|<command>\\n``
    ``seq`` is a monotonically increasing integer; the radio echoes it back
    in the response so commands and responses can be matched asynchronously.

Response (radio → client)
    ``R<seq>|<status_hex>|<payload>\\n``
    ``status_hex`` is a 32-bit hex code; 0x00000000 = success.  Non-zero
    codes are looked up in ``SMARTSDR_STATUS_MESSAGES`` and logged.
    ``payload`` is the data returned by the command, or an error description.

Unsolicited status (radio → client)
    Lines beginning with ``S`` (status) or ``V`` (version).  Format:
    ``S<handle>|<category> <id> <key>=<value> ...\\n``
    These lines are routed to the registered ``_status_cb`` callback for
    consumption by ``DAXIQSetup._status_monitor`` and for GUI client tracking.

Class: FlexTCPClient
--------------------
Threading model
    A single daemon thread (``_recv_loop``) runs ``recv(4096)`` in a loop,
    assembles a line buffer, and dispatches complete lines to ``_handle_line``.
    ``send_command`` is called from the caller's thread; it acquires a lock,
    stores a ``threading.Event`` keyed on the sequence number, sends the
    command, and waits on the event.  ``_handle_line`` signals the event when
    the matching response arrives.  The lock protects ``_responses`` and
    ``_pending_cmds`` across threads.

send_command(cmd, timeout)
    Thread-safe blocking call.  Raises ``RuntimeError`` on:
      * Non-zero SmartSDR status code (radio rejected the command).
      * Timeout (no response within ``timeout`` seconds, default 5 s).
    Returns the response payload string on success.

GUI client tracking
    ``_capture_client_status`` / ``_capture_client_payload`` parse ``client
    <handle> gui=1 client_id=<uuid> ...`` entries from both unsolicited status
    lines and ``client list`` command responses.  Only entries with ``gui=1``
    or ``program`` starting with ``"SmartSDR"`` and a non-empty ``client_id``
    are retained.  The accumulated table is used by ``FlexDAXIQ.start()`` to
    select a UUID for ``client bind`` without requiring the user to supply one
    explicitly.

Key methods
-----------
connect()                  Open TCP socket; start ``_recv_loop`` daemon thread.
disconnect()               Stop recv loop; close socket.
send_command(cmd, timeout) Send sequenced command; block for response.
get_local_ip()             Return local IP of the TCP socket (used as the
                           ``ip=`` argument to ``stream create daxiq``).
set_status_callback(cb)    Register callable ``(line: str) -> None`` for
                           unsolicited status lines.
get_gui_clients()          Return list of GUI client metadata dicts.
get_gui_client_ids()       Return list of GUI client UUID strings.
refresh_client_list()      Send ``client list``; parse and ingest all entries;
                           return ``(line_count, new_gui_count)``.
"""

import socket
import threading
from typing import Callable

from .common import _maybe_log_unmapped_status_code, log
from .models import FlexRadio

class FlexTCPClient:
    """
    Manages the SmartSDR TCP command/control connection.
    Sends sequenced commands, receives status/response messages.
    """

    def __init__(self, radio: FlexRadio):
        self.radio      = radio
        self._sock      = None
        self._seq       = 1
        self._lock      = threading.Lock()
        self._responses = {}        # seq -> response string
        self._pending_cmds = {}     # seq -> command text
        self._status_cb = None      # callback for unsolicited status messages
        self._running   = False
        self._recv_thread = None
        self._gui_clients = {}      # client_id -> metadata from status

    def connect(self):
        log.debug(f"Connecting to {self.radio.ip}:{self.radio.port}")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.radio.ip, self.radio.port))
        self._sock.settimeout(None)
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        log.debug("TCP connected")
    
    def get_local_ip(self) -> str:
        """Get the local IP address used for this connection."""
        if self._sock:
            return self._sock.getsockname()[0]
        return "0.0.0.0"

    def disconnect(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def set_status_callback(self, cb: Callable[[str], None]):
        """Register callback for unsolicited status messages from the radio."""
        self._status_cb = cb

    def get_gui_clients(self) -> list[dict]:
        """Return GUI clients observed from status lines."""
        with self._lock:
            items = [dict(v) for v in self._gui_clients.values()]
        items.sort(key=lambda item: (item.get("station", ""), item.get("client_id", "")))
        return items

    def get_gui_client_ids(self) -> list[str]:
        """Return GUI client UUIDs observed from status lines."""
        return [item.get("client_id") for item in self.get_gui_clients() if item.get("client_id")]

    def refresh_client_list(self) -> tuple[int, int]:
        """Query radio client list and ingest any GUI client_id entries.

        Returns:
            (line_count, parsed_gui_count)
        """
        try:
            response = self.send_command("client list")
        except RuntimeError:
            return (0, 0)

        line_count = 0
        parsed_before = len(self.get_gui_clients())
        for raw_line in response.splitlines():
            payload = raw_line.strip()
            if not payload:
                continue
            line_count += 1
            self._capture_client_payload(payload)

        parsed_after = len(self.get_gui_clients())
        parsed_delta = max(parsed_after - parsed_before, 0)
        return (line_count, parsed_delta)

    def send_command(self, cmd: str, timeout: float = 5.0) -> str:
        """
        Send a sequenced command and wait for the response.
        Returns the response string (everything after the status code).
        Raises RuntimeError if command fails (non-zero status).
        """
        with self._lock:
            seq = self._seq
            self._seq += 1
            msg = f"C{seq}|{cmd}\n"
            ev = threading.Event()
            self._responses[seq] = (ev, None)
            self._pending_cmds[seq] = cmd

        log.debug(f"TX: {msg.strip()}")
        self._sock.sendall(msg.encode())

        ev.wait(timeout=timeout)
        with self._lock:
            _, result = self._responses.pop(seq, (None, None))
            self._pending_cmds.pop(seq, None)
        
        if result is None:
            raise RuntimeError(f"Command timeout: {cmd}")
        
        status, response = result
        if status != 0:
            detail = response.strip()
            _maybe_log_unmapped_status_code(status)
            if detail:
                raise RuntimeError(f"Radio rejected command: {cmd} -> {detail}")
            raise RuntimeError(f"Radio rejected command: {cmd}")
        
        return response

    def _recv_loop(self):
        buf = b""
        while self._running:
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    log.warning("TCP connection closed by radio")
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line.decode("utf-8", errors="replace"))
            except Exception as e:
                if self._running:
                    log.error(f"TCP recv error: {e}")
                break

    def _handle_line(self, line: str):
        log.debug(f"RX: {line}")
        if line.startswith("R"):
            # Response to a sequenced command: R<seq>|<status>|<message>
            parts = line[1:].split("|", 2)
            if len(parts) >= 2:
                try:
                    seq = int(parts[0])
                    status = int(parts[1], 16)  # Status code in hex
                except ValueError:
                    return
                response = parts[2] if len(parts) > 2 else ""
                
                # Log errors
                if status != 0:
                    cmd = self._pending_cmds.get(seq, "<unknown command>")
                    detail = response.strip()
                    _maybe_log_unmapped_status_code(status)
                    if detail:
                        log.warning(f"Radio rejected command: {cmd} -> {detail}")
                    else:
                        log.warning(f"Radio rejected command: {cmd}")
                
                with self._lock:
                    if seq in self._responses:
                        ev, _ = self._responses[seq]
                        # Return tuple of (status, response)
                        self._responses[seq] = (ev, (status, response))
                        ev.set()
        elif line.startswith("S") or line.startswith("V"):
            # Unsolicited status or version message
            self._capture_client_status(line)
            if self._status_cb:
                self._status_cb(line)

    def _capture_client_status(self, line: str):
        """Capture GUI client metadata from status lines like: S..|client 0x... gui=1 client_id=..."""
        try:
            _, payload = line.split("|", 1)
        except ValueError:
            return

        self._capture_client_payload(payload)

    def _capture_client_payload(self, payload: str):
        """Capture GUI client metadata from a payload that may include a `client ...` record."""
        payload = payload.strip()
        if payload.startswith("client "):
            client_payload = payload
        elif " client " in payload:
            _, _, client_payload = payload.partition(" client ")
            client_payload = f"client {client_payload.strip()}"
        else:
            return

        tokens = client_payload.split()
        if len(tokens) < 2 or tokens[0] != "client":
            return

        handle = tokens[1]
        kv = {}
        for token in tokens[2:]:
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            kv[key.strip()] = value.strip()

        gui_flag = kv.get("gui", "")
        client_id = kv.get("client_id")
        program = kv.get("program", "")
        is_gui = (gui_flag == "1") or program.startswith("SmartSDR")
        if not is_gui or not client_id:
            return

        entry = {
            "client_id": client_id,
            "handle": handle,
            "station": kv.get("station", ""),
            "program": kv.get("program", ""),
            "host": kv.get("host", ""),
            "ip": kv.get("ip", ""),
        }
        with self._lock:
            self._gui_clients[client_id] = entry

