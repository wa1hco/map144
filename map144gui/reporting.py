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
"""Reporting — WSJT-X UDP broadcast and PSKReporter upload for decoded MSK144.

Two output protocols mirror what WSJT-X produces:

WSJT-X UDP (port 2237 by default)
    Sends a Heartbeat (type 0) every 15 s and a Decode (type 2) for each
    successfully decoded message.  GridTracker, N1MM Logger+ (via its
    built-in WSJT-X integration), JTAlert, and Log4OM all subscribe to
    this stream.  The wire format follows the WSJT-X Network Message
    Protocol as documented in NetworkMessage.hpp from the WSJT-X source.

PSKReporter (https://pskreporter.info)
    Spots are accumulated and uploaded via HTTPS POST to
    https://www.pskreporter.info/cgi-bin/pskr/upload every
    PSKREPORTER_INTERVAL_S seconds (default 300 / 5 minutes).  The upload
    uses the PSKReporter simple XML format accepted by the REST endpoint.

Message parsing
    MSK144 messages follow the same free-text format as WSJT-X:
        CQ <CALL> <GRID>        — CQ with optional grid
        <DE> <TO> <GRID|report> — standard exchange
        <DE> <TO> RRR | 73 | RR73
    _parse_msk144() extracts (de_call, to_call, grid) best-effort.
    Unparseable messages are still reported via UDP with message=raw text
    and de_call='' so GridTracker can display them.

Usage
    reporter = Reporter(settings)          # settings: QSettings-like dict
    reporter.start()                       # opens UDP socket, starts threads
    reporter.report_decode(decode_dict)    # from displays.py decode drain
    reporter.stop()                        # flush PSKReporter, close socket
"""

import queue
import re
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


# ── Constants ────────────────────────────────────────────────────────────────

WSJTX_MAGIC   = 0xADBCCBDA
WSJTX_SCHEMA  = 2
WSJTX_ID      = "map144"

MSG_HEARTBEAT = 0
MSG_DECODE    = 2

HEARTBEAT_INTERVAL_S    = 15
PSKREPORTER_INTERVAL_S  = 300   # 5 minutes — pskreporter asks for ≥5 min

PSKREPORTER_URL = "https://www.pskreporter.info/cgi-bin/pskr/upload.pl"


# ── WSJT-X wire format helpers ───────────────────────────────────────────────

def _pack_utf8(s: str) -> bytes:
    """WSJT-X length-prefixed UTF-8 string: uint32 length + bytes.
    Empty string is encoded as length=0xFFFFFFFF (null QByteArray convention).
    """
    if s == "":
        return struct.pack(">I", 0xFFFFFFFF)
    b = s.encode("utf-8")
    return struct.pack(">I", len(b)) + b


def _pack_bool(v: bool) -> bytes:
    return struct.pack(">?", v)


def _pack_u32(v: int) -> bytes:
    return struct.pack(">I", v)


def _pack_i32(v: int) -> bytes:
    return struct.pack(">i", v)


def _pack_u64(v: int) -> bytes:
    return struct.pack(">Q", v)


def _pack_f64(v: float) -> bytes:
    return struct.pack(">d", v)


def _header(msg_type: int, seq: int) -> bytes:
    return (struct.pack(">I", WSJTX_MAGIC)
            + struct.pack(">I", WSJTX_SCHEMA)
            + struct.pack(">I", msg_type)
            + struct.pack(">I", seq)
            + _pack_utf8(WSJTX_ID))


def build_heartbeat(seq: int, max_schema: int = 3, version: str = "2.7.0",
                    revision: str = "") -> bytes:
    """Build a WSJT-X Heartbeat (type 0) datagram."""
    return (_header(MSG_HEARTBEAT, seq)
            + _pack_u32(max_schema)
            + _pack_utf8(version)
            + _pack_utf8(revision))


def build_decode(seq: int, decode: dict, my_call: str, dial_freq_hz: int) -> bytes:
    """Build a WSJT-X Decode (type 2) datagram.

    decode dict keys (from detection.py decode_queue):
        message   str    raw MSK144 text (e.g. "W1AW WA1HCO FN42")
        jt9_snr   int|None   SNR in dB
        t_sec     float  time offset within the 15-s period
        radio_khz float  dial frequency in kHz
        utc_time  str    "HH:MM:SS"
    """
    msg     = decode.get('message', '')
    snr     = decode.get('jt9_snr') or 0
    delta_t = float(decode.get('t_sec', 0.0))
    # dial_freq_hz from the engine's center_freq_mhz or radio_khz
    freq_hz = dial_freq_hz

    # WSJT-X Decode fields (in order from NetworkMessage.hpp):
    #   bool     new (always True for live decodes)
    #   bool     is_new (True)
    #   QTime    time — ms since midnight UTC
    #   qint32   snr
    #   double   delta_time (s)
    #   quint32  delta_frequency (Hz offset from dial — 1500 Hz convention)
    #   utf8     mode
    #   utf8     message
    #   bool     low_confidence (False)
    #   bool     off_air (False)

    utc_str = decode.get('utc_time', datetime.now(timezone.utc).strftime('%H:%M:%S'))
    try:
        h, m, s = (int(x) for x in utc_str.split(':'))
        ms_since_midnight = (h * 3600 + m * 60 + s) * 1000
    except Exception:
        ms_since_midnight = 0

    delta_freq = 1500   # MSK144 audio centre — matches JT9_BASE_ARGS -f 1500

    return (_header(MSG_DECODE, seq)
            + _pack_bool(True)           # is_new
            + _pack_u32(ms_since_midnight)
            + _pack_i32(int(snr))
            + _pack_f64(delta_t)
            + _pack_u32(delta_freq)
            + _pack_utf8("MSK144")
            + _pack_utf8(msg)
            + _pack_bool(False)          # low_confidence
            + _pack_bool(False))         # off_air


# ── MSK144 message parser ────────────────────────────────────────────────────

_CALL_RE  = r'[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z](?:/[A-Z0-9]+)?'
_GRID_RE  = r'[A-R]{2}[0-9]{2}(?:[A-X]{2})?'
_REPORT_RE = r'[+-][0-9]{2}|RRR|RR73|73|RR'

_CQ_RE  = re.compile(
    rf'^CQ\s+(?:DX\s+)?({_CALL_RE})(?:\s+({_GRID_RE}))?$', re.IGNORECASE)
_STD_RE = re.compile(
    rf'^({_CALL_RE})\s+({_CALL_RE})(?:\s+({_GRID_RE}|{_REPORT_RE}))?$',
    re.IGNORECASE)


def _parse_msk144(message: str):
    """Return (de_call, to_call, grid) from an MSK144 message string.

    de_call is the transmitting station, to_call is the addressed station.
    grid may be None.  Returns ('', '', None) if unparseable.
    """
    msg = message.strip().upper()
    m = _CQ_RE.match(msg)
    if m:
        return m.group(1), 'CQ', m.group(2)   # de=caller, to=CQ, grid=optional

    m = _STD_RE.match(msg)
    if m:
        to_call, de_call = m.group(1), m.group(2)
        grid_or_rpt = m.group(3) if m.lastindex >= 3 else None
        grid = grid_or_rpt if (grid_or_rpt and re.match(_GRID_RE, grid_or_rpt)) else None
        return de_call, to_call, grid

    return '', '', None


# ── PSKReporter XML builder ───────────────────────────────────────────────────

def _build_pskreporter_xml(spots: list, my_call: str, my_grid: str,
                            program: str = "map144") -> bytes:
    """Build PSKReporter simple XML upload body.

    spots: list of dicts with keys:
        de_call, de_grid, freq_hz, snr, mode, utc_epoch
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<recvr>')
    lines.append(f'  <recvrHeader>')
    lines.append(f'    <appContact>{my_call}</appContact>')
    lines.append(f'    <recvrCallsign>{my_call}</recvrCallsign>')
    lines.append(f'    <recvrLocator>{my_grid}</recvrLocator>')
    lines.append(f'    <programId>{program}</programId>')
    lines.append(f'  </recvrHeader>')
    lines.append(f'  <recvrs>')
    lines.append(f'    <recvr>')
    for sp in spots:
        utc_str = datetime.fromtimestamp(sp['utc_epoch'], tz=timezone.utc).strftime('%Y%m%d%H%M%S')
        lines.append(f'      <spot>')
        lines.append(f'        <senderCallsign>{sp["de_call"]}</senderCallsign>')
        if sp.get('de_grid'):
            lines.append(f'        <senderLocator>{sp["de_grid"]}</senderLocator>')
        lines.append(f'        <frequency>{sp["freq_hz"]}</frequency>')
        lines.append(f'        <mode>{sp["mode"]}</mode>')
        if sp.get('snr') is not None:
            lines.append(f'        <sNR>{sp["snr"]}</sNR>')
        lines.append(f'        <flowStartSeconds>{utc_str}</flowStartSeconds>')
        lines.append(f'      </spot>')
    lines.append(f'    </recvr>')
    lines.append(f'  </recvrs>')
    lines.append(f'</recvr>')
    return '\n'.join(lines).encode('utf-8')


# ── Reporter class ────────────────────────────────────────────────────────────

class Reporter:
    """Manages WSJT-X UDP broadcast and PSKReporter uploads.

    Instantiate once, call start(), feed report_decode() from the GUI
    decode-drain loop, call stop() on shutdown.
    """

    def __init__(self):
        # Settings — populated by apply_settings() from the UI
        self.my_call   = ''
        self.my_grid   = ''

        self.wsjtx_enabled = False
        self.wsjtx_host    = '127.0.0.1'
        self.wsjtx_port    = 2237

        self.pskreporter_enabled = False

        # Runtime state
        self._seq        = 0
        self._sock       = None
        self._running    = False
        self._decode_q   = queue.SimpleQueue()
        self._psk_spots  = []           # pending PSKReporter spots
        self._psk_lock   = threading.Lock()
        self._last_psk_upload = 0.0

        # Stats (read by the UI)
        self.stat_udp_sent      = 0
        self.stat_psk_uploaded  = 0
        self.stat_psk_queued    = 0
        self.stat_last_psk_time = ''
        self.stat_last_error    = ''

        self._hb_thread   = None
        self._work_thread = None

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_settings(self, my_call: str, my_grid: str,
                       wsjtx_enabled: bool, wsjtx_host: str, wsjtx_port: int,
                       pskreporter_enabled: bool):
        self.my_call   = my_call.strip().upper()
        self.my_grid   = my_grid.strip().upper()
        self.wsjtx_enabled = wsjtx_enabled
        self.wsjtx_host    = wsjtx_host.strip()
        self.wsjtx_port    = int(wsjtx_port)
        self.pskreporter_enabled = pskreporter_enabled

    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            self.stat_last_error = f"UDP socket: {e}"

        self._hb_thread = threading.Thread(target=self._heartbeat_loop,
                                           daemon=True, name='reporter-hb')
        self._hb_thread.start()

        self._work_thread = threading.Thread(target=self._work_loop,
                                             daemon=True, name='reporter-work')
        self._work_thread.start()

    def stop(self):
        self._running = False
        # Unblock work thread
        self._decode_q.put(None)
        if self._work_thread:
            self._work_thread.join(timeout=3.0)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        # Final PSKReporter flush
        if self.pskreporter_enabled and self.my_call:
            self._upload_pskreporter()

    def report_decode(self, decode: dict):
        """Called from the GUI decode-drain loop for every successful decode."""
        self._decode_q.put(decode)

    # ── Internal threads ──────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        while self._running:
            if self.wsjtx_enabled and self.my_call and self._sock:
                try:
                    pkt = build_heartbeat(self._next_seq())
                    self._sock.sendto(pkt, (self.wsjtx_host, self.wsjtx_port))
                    self.stat_udp_sent += 1
                except OSError as e:
                    self.stat_last_error = f"UDP send: {e}"
            for _ in range(HEARTBEAT_INTERVAL_S * 10):
                if not self._running:
                    return
                time.sleep(0.1)

    def _work_loop(self):
        while self._running:
            # PSKReporter upload timer
            now = time.time()
            if (self.pskreporter_enabled and self.my_call
                    and now - self._last_psk_upload >= PSKREPORTER_INTERVAL_S):
                self._upload_pskreporter()

            try:
                decode = self._decode_q.get(timeout=5.0)
            except queue.Empty:
                continue
            if decode is None:
                break

            self._handle_decode(decode)

    def _handle_decode(self, decode: dict):
        msg       = decode.get('message', '')
        radio_khz = float(decode.get('radio_khz', 0.0))
        freq_hz   = int(round(radio_khz * 1000))
        snr       = decode.get('jt9_snr')

        de_call, to_call, grid = _parse_msk144(msg)

        # ── WSJT-X UDP ────────────────────────────────────────────────────────
        if self.wsjtx_enabled and self.my_call and self._sock:
            try:
                pkt = build_decode(self._next_seq(), decode, self.my_call, freq_hz)
                self._sock.sendto(pkt, (self.wsjtx_host, self.wsjtx_port))
                self.stat_udp_sent += 1
            except OSError as e:
                self.stat_last_error = f"UDP send: {e}"

        # ── PSKReporter ───────────────────────────────────────────────────────
        if self.pskreporter_enabled and self.my_call and de_call:
            spot = {
                'de_call':   de_call,
                'de_grid':   grid or '',
                'freq_hz':   freq_hz,
                'snr':       snr,
                'mode':      'MSK144',
                'utc_epoch': int(time.time()),
            }
            with self._psk_lock:
                self._psk_spots.append(spot)
                self.stat_psk_queued = len(self._psk_spots)

    def _upload_pskreporter(self):
        with self._psk_lock:
            if not self._psk_spots:
                self._last_psk_upload = time.time()
                return
            spots = list(self._psk_spots)
            self._psk_spots.clear()
            self.stat_psk_queued = 0

        self._last_psk_upload = time.time()
        try:
            body = _build_pskreporter_xml(spots, self.my_call, self.my_grid)
            req  = urllib.request.Request(
                PSKREPORTER_URL,
                data=body,
                headers={'Content-Type': 'application/xml'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                _ = resp.read()
            self.stat_psk_uploaded += len(spots)
            self.stat_last_psk_time = datetime.now(timezone.utc).strftime('%H:%M:%S')
            print(f"[reporter] PSKReporter: uploaded {len(spots)} spots", flush=True)
        except Exception as e:
            self.stat_last_error = f"PSKReporter: {e}"
            print(f"[reporter] PSKReporter upload failed: {e}", flush=True)
            # Re-queue spots so they're not lost
            with self._psk_lock:
                self._psk_spots = spots + self._psk_spots
                self.stat_psk_queued = len(self._psk_spots)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
