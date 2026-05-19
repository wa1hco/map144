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
"""DAXIQSetup: SmartSDR API sequence to configure and start a DAXIQ IQ stream.

This module contains ``DAXIQSetup``, which handles the multi-step SmartSDR
command sequence required to get VITA-49 IQ UDP packets flowing to the client.
It also installs a persistent status callback on ``FlexTCPClient`` to track
panadapter centre frequency, bandwidth, and slice frequency in real time so
the GUI can display the correct frequency labels.

SmartSDR API sequence (``DAXIQSetup.setup``)
---------------------------------------------
1. **Pan status subscription** — send ``sub pan all`` or ``sub pan`` to ask the
   radio to emit ``display pan <id> ...`` status lines for existing panadapters.
   Wait 250 ms for the burst of status lines to arrive.

2. **Panadapter discovery** — the persistent status callback
   (``_status_monitor``) populates ``known_pans`` from incoming
   ``display pan <id> center=... bandwidth=...`` lines.  If a
   ``preferred_pan_id`` was supplied (from ``client bind`` context) it takes
   priority; otherwise the lowest-ID non-zero panadapter is selected.

3. **DAX IQ channel assignment** — send ``dax iq set <channel> pan=0x<id>``
   to associate the DAX IQ channel with the chosen panadapter.  In bound
   non-GUI contexts this is often required before the radio will route UDP
   packets; in GUI-owned contexts SmartSDR may reject the command, which is
   logged as a warning and not treated as fatal.

4. **Stream creation** — send ``stream create daxiq=<channel> ip=<local_ip>
   port=<udp_port>`` and parse the returned 32-bit hex stream ID.  This ID
   is used by ``VITAReceiver`` to filter incoming VITA-49 packets.

5. **Pan subscription** — subscribe to the chosen panadapter's ongoing status
   updates (``sub pan 0x<id>``) so frequency and bandwidth changes made in
   SmartSDR are reflected in the GUI without restarting.

6. **Frequency setting** — if ``center_freq_mhz`` is provided and the stream
   is in slice mode (``slice_id != 0``), send ``slice set <id>
   RF_frequency=<hz>``.  If the stream is panadapter-only, the frequency is
   controlled by SmartSDR and this client does not override it.

Persistent status callback (``_status_monitor``)
-------------------------------------------------
Installed via ``FlexTCPClient._status_cb`` (chained after any existing
callback) at ``__init__`` time.  Parses three families of unsolicited status
lines:

``|stream <id> dax_iq ...``
    Records the slice ID and panadapter ID assigned to our stream.

``|slice <num> RF_frequency=<hz> ...``
    Updates ``slice_frequency_mhz`` when our slice is retuned.

``|display pan <id> center=... bandwidth=... ...``
    Updates ``known_pans`` with the latest centre frequency (converting the
    SmartSDR ambiguous Hz/MHz representation via ``_parse_freq_to_mhz``) and
    bandwidth (via ``_parse_bandwidth_to_hz``).  Suppresses redundant log
    output using a content signature and a 750 ms cooldown timer.

Teardown
--------
``teardown()`` sends ``stream remove 0x<stream_id>`` to the radio.  Errors
are caught and logged; the TCP connection is closed by the caller
(``FlexDAXIQ.stop()``).

Helper functions
----------------
_extract_key(response, key)
    Scans a whitespace-separated ``key=value`` string and returns the value
    for the first matching key, or ``None`` if absent.

_parse_freq_to_mhz(freq_value)
    Handles SmartSDR's ambiguous frequency representation: values ≥ 1,000,000
    are divided by 1e6 to convert Hz → MHz; smaller values are assumed already
    in MHz.

_parse_bandwidth_to_hz(bw_value)
    Converts SmartSDR panadapter bandwidth: values < 1000 are treated as
    fractional MHz (e.g. ``0.063298`` → 63298 Hz); values ≥ 1000 are Hz.
"""

import time
from typing import Optional

from .common import VITA_UDP_PORT, log
from .tcp_client import FlexTCPClient

class DAXIQSetup:
    """
    Handles the SmartSDR API calls needed to start a DAXIQ stream.
    Sequence: discover existing pan/slice context -> request daxiq -> subscribe
    """

    def __init__(self, tcp: FlexTCPClient, sample_rate: int = 96000,
                 dax_channel: int = 1, listen_port: int = VITA_UDP_PORT,
                 preferred_pan_id: Optional[int] = None):
        self.tcp            = tcp
        self.sample_rate    = sample_rate
        self.dax_channel    = dax_channel
        self.listen_port    = listen_port
        self.pan_id         = None
        self.slice_id       = None
        self.stream_id      = None
        self._vita_ref      = None   # set by client.py after VITAReceiver is created
        self._old_status_cb = None
        self.slice_frequency_mhz = None  # Track actual slice frequency
        self.pan_frequency_mhz = None    # Track actual panadapter frequency
        self.pan_bandwidth_hz = None     # Track actual panadapter bandwidth
        self.known_pans = {}             # pan_id(int) -> dict of known settings
        self.known_slices = {}           # slice_num(int) -> {RF_frequency_mhz, mode, pan_id}
        self._known_pans_signature = None
        self._last_pan_report_ts = 0.0
        self._pan_discovery_phase = False
        self.preferred_pan_id = preferred_pan_id

        # Set up persistent status callback
        self._old_status_cb = self.tcp._status_cb
        self.tcp._status_cb = self._status_monitor

    def setup(self, center_freq_mhz: float = 14.0) -> int:
        """
        Configure radio for DAXIQ and return the stream_id to filter on.
        Uses existing GUI-owned panadapters/slices (if available) and starts DAXIQ.
        If center_freq_mhz is provided, attempts slice tuning only when a slice is assigned.
        """
        self._pan_discovery_phase = True
        self._subscribe_pan_status()
        time.sleep(0.25)
        self._maybe_report_known_panadapters(force=True)
        self._pan_discovery_phase = False

        # Use already-observed GUI-owned panadapters when present.
        if self.pan_id is None and self.preferred_pan_id not in (None, 0):
            self.pan_id = int(self.preferred_pan_id)
            log.info(f"Using bound-context panadapter 0x{self.pan_id:08x}")
        elif self.pan_id is None and self.known_pans:
            valid_pans = sorted([pan for pan in self.known_pans.keys() if pan != 0])
            if valid_pans:
                self.pan_id = valid_pans[0]
                log.info(f"Using existing panadapter 0x{self.pan_id:08x}")

        # Create DAXIQ stream - radio needs client IP to send UDP stream to
        client_ip = self.tcp.get_local_ip()
        log.debug(f"Creating DAXIQ stream to {client_ip}:{self.listen_port}")

        # If we have a panadapter, try to assign DAXIQ channel to it.
        # In bound/non-GUI contexts this is often required for UDP IQ packets to flow.
        if self.pan_id:
            try:
                resp = self.tcp.send_command(
                    f"dax iq set {self.dax_channel} pan=0x{self.pan_id:08x}"
                )
                log.debug(f"DAXIQ channel {self.dax_channel} on panadapter 0x{self.pan_id:08x}: {resp}")
                time.sleep(0.1)
            except RuntimeError as e:
                log.warning(f"Could not assign DAXIQ channel {self.dax_channel} to panadapter 0x{self.pan_id:08x} (may be controlled by SmartSDR): {e}")

        resp = self.tcp.send_command(
            f"stream create daxiq={self.dax_channel} ip={client_ip} port={self.listen_port}"
        )
        log.debug(f"DAXIQ stream create response: {resp}")
        self.stream_id = int(resp.strip(), 16) if resp.strip() else None
        if not self.stream_id:
            raise RuntimeError("Failed to create DAXIQ stream")
        # Remember the stream WE created.  The status-watch handler
        # below uses this to distinguish a radio-initiated reassignment
        # of our slice (rare — ADC-overload recovery; update filter) from
        # a foreign client creating its own DAX-IQ stream (common — other
        # MAP144 / WSJT-X; do NOT update filter).  Without ownership
        # awareness the filter silently follows the foreign stream and
        # we stop receiving IQ; teardown then tries to ``stream remove``
        # the foreign id, the radio rejects, and our own stream leaks.
        # See ``project_flex_multi_client_stream_leak`` memory.
        self._own_stream_id = self.stream_id
        log.debug(f"DAXIQ stream created: 0x{self.stream_id:08x}")

        # Wait briefly for stream status message with slice/pan assignment.
        time.sleep(0.2)

        # Subscribe to panadapter updates.
        if self.pan_id:
            try:
                resp = self.tcp.send_command(f"sub pan 0x{self.pan_id:08x}")
                log.debug(f"Subscribed to panadapter: {resp}")
            except RuntimeError as e:
                log.warning(f"Could not subscribe to panadapter: {e}")

        # Subscribe to slice status updates.
        try:
            self.tcp.send_command("sub slice all")
            log.debug("Subscribed to slice status updates")
        except RuntimeError:
            pass

        log.debug("Using SmartSDR-configured DAXIQ rate")

        # Set slice frequency if a slice is assigned; otherwise the pan center
        # is controlled by SmartSDR and we leave it alone.
        if center_freq_mhz is not None:
            if self.slice_id is not None and self.slice_id != 0:
                try:
                    freq_hz = int(center_freq_mhz * 1e6)
                    resp = self.tcp.send_command(
                        f"slice set {self.slice_id} RF_frequency={freq_hz}"
                    )
                    log.debug(f"Set slice {self.slice_id} frequency to {center_freq_mhz} MHz: {resp}")
                except RuntimeError as e:
                    log.warning(f"Could not set slice frequency: {e}")
            elif self.slice_id == 0 and self.pan_id not in (None, 0):
                log.info(
                    "Panadapter frequency is controlled by SmartSDR GUI; "
                    "using current GUI-selected center/bandwidth"
                )
            else:
                log.warning(
                    f"DAXIQ channel {self.dax_channel} mode unknown "
                    "(no slice or panadapter). Set frequency in SmartSDR."
                )

        log.debug(f"DAXIQ ready: stream_id=0x{self.stream_id:08x}")
        return self.stream_id

    def _subscribe_pan_status(self):
        """Ask radio to emit status for existing panadapters (firmware dependent)."""
        for cmd in ("sub pan all", "sub pan"):
            try:
                self.tcp.send_command(cmd)
                log.debug(f"Subscribed for pan status with command: {cmd}")
                return
            except RuntimeError:
                continue
        log.debug("Pan status subscription command not accepted by radio")

    def _report_known_panadapters(self):
        """Print panadapters discovered from status lines observed so far."""
        if not self.known_pans:
            log.debug("Existing panadapters: none observed yet (will populate as status lines arrive)")
            return

        log.debug("Existing panadapters (from status):")
        for pan_id in sorted(self.known_pans.keys()):
            info = self.known_pans[pan_id]
            parts = [f"pan=0x{pan_id:08x}"]
            center = info.get("center")
            if center is not None:
                parts.append(f"center={center:.6f} MHz")
            bandwidth = info.get("bandwidth")
            if bandwidth is not None:
                parts.append(f"bandwidth={bandwidth}")
            ant = info.get("ant") or info.get("rxant")
            if ant:
                parts.append(f"ant={ant}")
            stream_id = info.get("stream_id")
            if stream_id:
                parts.append(f"stream_id={stream_id}")
            log.debug("  " + " ".join(parts))

    def _maybe_report_known_panadapters(self, force: bool = False):
        """Emit known panadapter list only when content changes."""
        if self._pan_discovery_phase and not force:
            return
        if not self.known_pans:
            return

        signature_items = []
        for pan_id in sorted(self.known_pans.keys()):
            info = self.known_pans[pan_id]
            signature_items.append((
                pan_id,
                info.get("center"),
                info.get("bandwidth"),
                info.get("ant"),
                info.get("rxant"),
            ))

        signature = tuple(signature_items)
        if not force and signature == self._known_pans_signature:
            return

        now = time.time()
        if not force and (now - self._last_pan_report_ts) < 0.75:
            return

        self._known_pans_signature = signature
        self._last_pan_report_ts = now
        self._report_known_panadapters()

    def _status_monitor(self, line: str):
        """Monitor all status messages for stream and slice updates."""
        if self._old_status_cb:
            self._old_status_cb(line)

        # Monitor stream status for slice assignment
        if "|stream " in line and "dax_iq" in line:
            log.debug(f"Stream status: {line}")
            try:
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    tokens = parts[1].split()
                    if len(tokens) >= 2 and tokens[0] == "stream":
                        msg_stream_id = int(tokens[1], 16)
                        # Parse slice once — used both for filter-update
                        # ownership decisions and for slice/pan tracking.
                        slice_id_str = _extract_key(parts[1], "slice")
                        pan_id_str = _extract_key(parts[1], "pan")
                        msg_slice_id = (
                            int(slice_id_str, 16) if slice_id_str else None
                        )
                        _own_sid = getattr(self, '_own_stream_id', None)

                        # ── Foreign-vs-our-stream gate ─────────────────────
                        # If the broadcast is for a stream id we didn't
                        # create, decide whether it's a radio-initiated
                        # reassignment of OUR slice (rare — update filter)
                        # or another DAX client's stream (common — leave
                        # our state alone).  The slice id distinguishes
                        # them: our slice stays the same across radio
                        # reassignment, while foreign clients run on a
                        # different slice.
                        is_our_stream = (
                            _own_sid is None or msg_stream_id == _own_sid
                        )
                        if not is_our_stream:
                            if (self.slice_id is not None
                                    and msg_slice_id is not None
                                    and msg_slice_id == self.slice_id
                                    and self._vita_ref is not None):
                                # Same slice → radio reassigned our stream.
                                log.warning(
                                    f"DAXIQ stream reassigned by radio: "
                                    f"0x{_own_sid:08x} → 0x{msg_stream_id:08x} "
                                    f"(same slice 0x{msg_slice_id:x}) — updating filter"
                                )
                                self._own_stream_id = msg_stream_id
                                self.stream_id = msg_stream_id
                                self._vita_ref.filter_sid = msg_stream_id
                                self._vita_ref._last_seq = {}
                            else:
                                # Different slice (or no slice info yet) →
                                # foreign DAX client.  Do NOT touch our
                                # filter or stream_id — both are still
                                # valid on the radio.  Operator workaround
                                # if both clients must run: assign each
                                # a different slice (radio supports it).
                                _slice_str = (
                                    f"0x{msg_slice_id:x}"
                                    if msg_slice_id is not None else "?"
                                )
                                _own_slice_str = (
                                    f"0x{self.slice_id:x}"
                                    if self.slice_id is not None else "?"
                                )
                                log.info(
                                    f"DAXIQ: another client owns stream "
                                    f"0x{msg_stream_id:08x} (slice {_slice_str}); "
                                    f"our stream 0x{_own_sid:08x} on "
                                    f"slice {_own_slice_str} unaffected — "
                                    f"ignoring broadcast"
                                )
                            # Foreign / reassigned: skip the slice/pan
                            # absorption below — it'd capture foreign info.
                        else:
                            # ── Our stream → absorb slice/pan info ─────
                            if slice_id_str:
                                self.slice_id = int(slice_id_str, 16)
                                status = "ASSIGNED" if self.slice_id != 0 else "NOT ASSIGNED"
                                log.debug(
                                    f"DAXIQ stream 0x{msg_stream_id:08x} slice status: 0x{self.slice_id:x} ({status})"
                                )

                            if pan_id_str:
                                parsed_pan_id = int(pan_id_str, 16)
                                if parsed_pan_id == 0:
                                    log.debug(
                                        f"Ignoring invalid panadapter id 0x00000000 from stream 0x{msg_stream_id:08x}"
                                    )
                                else:
                                    self.pan_id = parsed_pan_id
                                    pan_info = self.known_pans.setdefault(self.pan_id, {})
                                    pan_info["stream_id"] = f"0x{msg_stream_id:08x}"
                                    self._maybe_report_known_panadapters()
                                    log.debug(f"DAXIQ stream 0x{msg_stream_id:08x} panadapter: 0x{self.pan_id:08x}")
            except (ValueError, TypeError, IndexError) as e:
                log.debug(f"Error parsing stream status: {e}")

        # Monitor slice status updates — track all slices for radio state display
        if "|slice " in line:
            try:
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    tokens = parts[1].split()
                    if len(tokens) >= 2 and tokens[0] == "slice":
                        slice_num = int(tokens[1])
                        payload = parts[1]
                        entry = self.known_slices.setdefault(slice_num, {})

                        freq_str = _extract_key(payload, "RF_frequency")
                        if freq_str:
                            freq_val = float(freq_str)
                            entry["RF_frequency_mhz"] = freq_val if freq_val < 1e6 else freq_val / 1e6

                        mode_str = _extract_key(payload, "mode")
                        if mode_str:
                            entry["mode"] = mode_str

                        pan_str = _extract_key(payload, "pan")
                        if pan_str:
                            entry["pan_id"] = int(pan_str, 16)

                        daxiq_str = _extract_key(payload, "dax_iq_channel")
                        if daxiq_str is not None:
                            entry["dax_iq_channel"] = int(daxiq_str)

                        dax_str = _extract_key(payload, "dax")
                        if dax_str is not None:
                            entry["dax"] = int(dax_str)

                        client_handle_str = _extract_key(payload, "client_handle")
                        if client_handle_str:
                            entry["client_handle"] = client_handle_str

                        in_use_str = _extract_key(payload, "in_use")
                        if in_use_str == "0":
                            self.known_slices.pop(slice_num, None)
                            log.debug(f"Slice {slice_num} removed (in_use=0)")
                            return
                        elif in_use_str is not None:
                            entry["in_use"] = True

                        if slice_num == self.slice_id and self.slice_id != 0:
                            if "RF_frequency_mhz" in entry:
                                self.slice_frequency_mhz = entry["RF_frequency_mhz"]
                                log.debug(f"Slice {self.slice_id} frequency: {self.slice_frequency_mhz:.6f} MHz")
            except (ValueError, TypeError, IndexError) as e:
                log.debug(f"Error parsing slice status: {e}")

        # Monitor panadapter frequency updates
        if "|display pan " in line:
            try:
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    tokens = parts[1].split()
                    if len(tokens) >= 3 and tokens[0] == "display" and tokens[1] == "pan":
                        pan_id = int(tokens[2], 16)
                        if pan_id == 0:
                            return
                        removed_str = _extract_key(parts[1], "removed")
                        if removed_str == "1" or (len(tokens) == 3 and removed_str is None and
                                                   _extract_key(parts[1], "center") is None):
                            if pan_id in self.known_pans:
                                self.known_pans.pop(pan_id)
                                log.debug(f"Panadapter 0x{pan_id:08x} removed")
                            return
                        pan_info = self.known_pans.setdefault(pan_id, {})

                        center_str = _extract_key(parts[1], "center")
                        if center_str:
                            center_mhz = _parse_freq_to_mhz(center_str)
                            if center_mhz is not None:
                                pan_info["center"] = center_mhz

                        bandwidth = _extract_key(parts[1], "bandwidth")
                        if bandwidth:
                            pan_info["bandwidth"] = _parse_bandwidth_to_hz(bandwidth)

                        ant = _extract_key(parts[1], "ant")
                        if ant:
                            pan_info["ant"] = ant

                        rxant = _extract_key(parts[1], "rxant")
                        if rxant:
                            pan_info["rxant"] = rxant

                        stream_id = _extract_key(parts[1], "stream_id")
                        if stream_id:
                            pan_info["stream_id"] = stream_id

                        daxiq_ch = _extract_key(parts[1], "daxiq_channel")
                        if daxiq_ch:
                            pan_info["daxiq_channel"] = daxiq_ch

                        self._maybe_report_known_panadapters()

                        if self.pan_id is None:
                            self.pan_id = pan_id
                            log.debug(f"Captured panadapter ID: 0x{self.pan_id:08x}")

                        if self.pan_id and pan_id == self.pan_id:
                            if center_str:
                                center_mhz = _parse_freq_to_mhz(center_str)
                                if center_mhz is not None:
                                    self.pan_frequency_mhz = center_mhz
                                    log.debug(
                                        f"Panadapter 0x{self.pan_id:08x} center frequency: {self.pan_frequency_mhz:.6f} MHz"
                                    )

                            bandwidth_str = _extract_key(parts[1], "bandwidth")
                            if bandwidth_str:
                                bandwidth_hz = _parse_bandwidth_to_hz(bandwidth_str)
                                if bandwidth_hz is not None:
                                    self.pan_bandwidth_hz = bandwidth_hz
            except (ValueError, TypeError, IndexError) as e:
                log.debug(f"Error parsing panadapter status: {e}")

    def teardown(self):
        """Clean up DAXIQ resources. Doesn't raise on errors during cleanup."""
        stream_result = "n/a"

        if self.stream_id:
            try:
                self.tcp.send_command(f"stream remove 0x{self.stream_id:08x}")
                stream_result = "ok"
            except Exception as e:
                stream_result = f"rejected ({e})"
                log.warning(f"Failed to remove stream: {e}")
        else:
            stream_result = "none"

        log.info(f"Shutdown cleanup: stream={stream_result}")


def _extract_key(response: str, key: str) -> Optional[str]:
    """Extract a value from a key=value response string."""
    for token in response.split():
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return None

def _parse_freq_to_mhz(freq_value: str) -> Optional[float]:
    """Parse SmartSDR frequency field that may be reported in Hz or MHz."""
    try:
        value = float(freq_value)
    except (TypeError, ValueError):
        return None
    return value / 1e6 if value >= 1e6 else value


def _parse_bandwidth_to_hz(bw_value: str) -> Optional[float]:
    """Parse SmartSDR panadapter bandwidth that may be reported in Hz or MHz."""
    try:
        value = float(bw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # Flex status commonly reports pan bandwidth as fractional MHz (e.g. 0.063298).
    if value < 1000:
        return value * 1e6
    return value
