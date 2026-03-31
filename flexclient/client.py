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
"""High-level FlexDAXIQ client: discovery, connection, stream, and sample delivery.

This module contains ``FlexDAXIQ``, the single class a caller needs to start
receiving IQ samples from a FlexRadio.  It orchestrates the lower-level
components in ``discovery``, ``tcp_client``, ``setup``, and ``vita`` and
exposes a simple start / get_samples / stop API.

Class: FlexDAXIQ
----------------
Constructor parameters
    radio_ip         : str | None   — Radio IP; auto-discovered via UDP broadcast
                                      if omitted.
    center_freq_mhz  : float        — Requested centre frequency in MHz.
                                      Only applied to a slice if one is assigned
                                      to the DAX IQ channel; panadapter frequency
                                      is controlled by SmartSDR.
    sample_rate      : int          — DAXIQ sample rate (24000 / 48000 / 96000 /
                                      192000 Hz).  SmartSDR must be configured to
                                      match; this client does not force the rate.
    dax_channel      : int          — DAX IQ channel number (1–8 depending on licence).
    listen_port      : int          — Local UDP port for VITA-49 stream; overridden
                                      at ``start()`` time by an auto-selected
                                      ephemeral port via ``_pick_udp_listen_port()``.
    bind_client_id   : str | None   — GUI client UUID for ``client bind
                                      client_id=<uuid>``.  Required when SmartSDR
                                      enforces client binding so that the DAXIQ
                                      stream is associated with an active GUI
                                      context and UDP packets actually flow.
    bind_client_handle : str | None — Deprecated alias for ``bind_client_id``.

start() sequence
    1. Discover or use provided radio IP; create ``FlexRadio`` object.
    2. Connect ``FlexTCPClient`` (TCP port 4992).
    3. Subscribe for client status (``sub client all`` / ``sub client``) and
       wait 0.5 s for the radio to emit ``client`` status lines.
    4. Collect GUI client UUIDs from status lines and discovery advertisements.
    5. Determine ``bind_client_id``: explicit argument → status-observed IDs →
       discovery-advertised IDs → none (log warning).
    6. Send ``client bind client_id=<uuid>`` if a UUID is available; run bound-
       context diagnostics (``slice list``) to verify the binding context.
    7. Pick an ephemeral UDP listen port via ``_pick_udp_listen_port()``.
    8. Instantiate ``DAXIQSetup`` and call ``setup(center_freq_mhz)`` to:
         - Subscribe to panadapter status and discover existing pan/slice context.
         - Assign the DAX IQ channel to a panadapter (``dax iq set``).
         - Create the DAXIQ stream (``stream create daxiq=``).
         - Optionally tune a slice to ``center_freq_mhz``.
    9. Instantiate ``VITAReceiver`` and start the UDP receive thread.

sample_queue
    A ``queue.Queue(maxsize=500)`` populated by ``VITAReceiver``.  Each entry
    is a ``VitaPacket`` with fields:
      - ``samples``        — complex64 NumPy array, I + jQ
      - ``timestamp_int``  — integer seconds (GPS or Unix epoch)
      - ``timestamp_frac`` — fractional timestamp (picoseconds)
      - ``stream_id``      — 32-bit VITA-49 stream identifier
      - ``sequence``       — 4-bit packet sequence number

get_samples(timeout)
    Convenience wrapper around ``sample_queue.get(timeout=timeout)``.
    Returns ``None`` on timeout; raises nothing.

stop()
    Stops the VITA receiver thread, sends ``stream remove`` to the radio,
    and closes the TCP connection.  Safe to call even if ``start()`` was
    never completed successfully.
"""

import queue
import time
from typing import Optional

from .common import FLEX_TCP_PORT, VITA_UDP_PORT, _pick_udp_listen_port, log
from .discovery import discover
from .models import FlexRadio, VitaPacket
from .setup import DAXIQSetup
from .tcp_client import FlexTCPClient
from .vita import VITAReceiver

class FlexDAXIQ:
    """
    High-level interface: discover radio, connect, start DAXIQ stream,
    deliver IQ sample blocks via a queue.
    """

    def __init__(self, radio_ip: Optional[str] = None,
                 center_freq_mhz: float = 14.0,
                 sample_rate: int = 96000,
                 dax_channel: int = 1,
                 listen_port: int = VITA_UDP_PORT,
                 bind_client_id: Optional[str] = None,
                 bind_client_handle: Optional[str] = None):
        self.radio_ip        = radio_ip
        self.center_freq_mhz = center_freq_mhz
        self.sample_rate     = sample_rate
        self.dax_channel     = dax_channel
        self.listen_port     = listen_port
        self.bind_client_id = bind_client_id or bind_client_handle
        self.sample_queue    = queue.Queue(maxsize=4000)
        self._tcp            = None
        self._dax_setup      = None
        self._vita           = None

    def _log_bound_context_diagnostics(self):
        """Log bound-context diagnostics via supported list commands."""

        try:
            slice_list_resp = self._tcp.send_command("slice list")
            log.info(f"slice list response payload: {slice_list_resp!r}")
            visible_slice_labels = []
            for raw_line in slice_list_resp.splitlines():
                line = raw_line.strip()
                if line:
                    first_token = line.split()[0]
                    try:
                        slice_num = int(first_token)
                    except ValueError:
                        continue

                    if 0 <= slice_num < 26:
                        visible_slice_labels.append(chr(ord('A') + slice_num))
                    else:
                        visible_slice_labels.append(str(slice_num))

            if visible_slice_labels:
                log.info(f"Visible slices: {', '.join(visible_slice_labels)}")
        except RuntimeError as e:
            log.warning(f"slice list failed: {e}")

    def _request_client_status(self):
        """Ask radio to publish client status lines so GUI client UUIDs can be discovered."""
        list_lines, parsed_count = self._tcp.refresh_client_list()
        if list_lines > 0:
            log.info(f"Client list lines={list_lines}, parsed_gui_client_ids={parsed_count}")
        for cmd in ("sub client all", "sub client"):
            try:
                self._tcp.send_command(cmd)
                log.debug(f"Subscribed for client status with command: {cmd}")
                return
            except RuntimeError:
                continue
        log.debug("Client status subscription command not accepted by radio")

    def start(self):
        # Discover or use provided IP
        if self.radio_ip:
            radio = FlexRadio(ip=self.radio_ip, port=FLEX_TCP_PORT)
        else:
            radios = discover()
            if not radios:
                raise RuntimeError("No FlexRadio found on network")
            radio = radios[0]
            log.debug(f"Found radio: {radio.model} at {radio.ip}")

        # Connect TCP
        self._tcp = FlexTCPClient(radio)
        self._tcp.connect()

        self._request_client_status()
        time.sleep(0.5)

        gui_clients = self._tcp.get_gui_clients()
        if gui_clients:
            log.debug("GUI clients discovered:")
            for idx, item in enumerate(gui_clients):
                log.debug(
                    "  [%d] station=%s program=%s host=%s ip=%s handle=%s client_id=%s",
                    idx,
                    item.get("station", "n/a") or "n/a",
                    item.get("program", "n/a") or "n/a",
                    item.get("host", "n/a") or "n/a",
                    item.get("ip", "n/a") or "n/a",
                    item.get("handle", "n/a") or "n/a",
                    item.get("client_id", "n/a") or "n/a",
                )
        else:
            log.debug("GUI clients discovered: none with client_id yet")

        if radio.gui_client_ids:
            log.debug(f"Discovery advertised GUI client_ids: {', '.join(radio.gui_client_ids)}")

        bind_client_id = self.bind_client_id
        if not bind_client_id:
            status_ids = self._tcp.get_gui_client_ids()
            if status_ids:
                bind_client_id = status_ids[0]
            elif radio.gui_client_ids:
                bind_client_id = radio.gui_client_ids[0]
            else:
                log.debug("No GUI client_id available for auto-bind")

        if bind_client_id:
            bind_cmd = f"client bind client_id={bind_client_id}"
            log.debug(f"Sending bind command: {bind_cmd}")
            try:
                self._tcp.send_command(bind_cmd)
                log.debug(f"Bound to GUI client_id {bind_client_id}")
                self._log_bound_context_diagnostics()
            except RuntimeError as e:
                log.warning(f"Could not bind to GUI client_id {bind_client_id}: {e}")
                log.warning(f"Bind command failed: {bind_cmd}")

        selected_port = _pick_udp_listen_port()
        log.debug(f"Using UDP:{selected_port} for DAXIQ stream")
        self.listen_port = selected_port

        self._dax_setup = DAXIQSetup(
            self._tcp,
            self.sample_rate,
            self.dax_channel,
            self.listen_port,
        )
        time.sleep(0.5)  # let status burst populate existing pan/slice state

        # Setup DAXIQ
        stream_id = self._dax_setup.setup(self.center_freq_mhz)

        # Start VITA receiver
        self._vita = VITAReceiver(
            listen_port=self.listen_port,
            stream_id=stream_id,
            output_queue=self.sample_queue
        )
        self._vita.start()
        log.debug("DAXIQ stream running")

    def stop(self):
        if self._vita:
            self._vita.stop()
        if self._dax_setup:
            self._dax_setup.teardown()
        if self._tcp:
            self._tcp.disconnect()

    def get_samples(self, timeout: float = 1.0) -> Optional[VitaPacket]:
        """Block until a packet arrives or timeout. Returns VitaPacket or None."""
        try:
            return self.sample_queue.get(timeout=timeout)
        except queue.Empty:
            return None

