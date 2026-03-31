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
from datetime import datetime, timezone


# ── Constants ────────────────────────────────────────────────────────────────

WSJTX_MAGIC   = 0xADBCCBDA
WSJTX_SCHEMA  = 3
WSJTX_ID      = "map144"

MSG_HEARTBEAT = 0
MSG_STATUS    = 1
MSG_DECODE    = 2

HEARTBEAT_INTERVAL_S    = 15
PSKREPORTER_INTERVAL_S  = 300   # 5 minutes — pskreporter asks for ≥5 min

PSKREPORTER_HOST = "report.pskreporter.info"
PSKREPORTER_PORT = 4739
PSKREPORTER_ENTERPRISE  = 30351
PSKREPORTER_SENDER_TMPL = 0x50e3   # template ID for spot data sets
PSKREPORTER_RECV_TMPL   = 0x50e2   # template ID for receiver options set


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


def _header(msg_type: int) -> bytes:
    return (struct.pack(">I", WSJTX_MAGIC)
            + struct.pack(">I", WSJTX_SCHEMA)
            + struct.pack(">I", msg_type)
            + _pack_utf8(WSJTX_ID))


def build_heartbeat(max_schema: int = 3, version: str = "2.7.0",
                    revision: str = "") -> bytes:
    """Build a WSJT-X Heartbeat (type 0) datagram."""
    return (_header(MSG_HEARTBEAT)
            + _pack_u32(max_schema)
            + _pack_utf8(version)
            + _pack_utf8(revision))


def build_status(dial_freq_hz: int, my_call: str, my_grid: str) -> bytes:
    """Build a WSJT-X Status (type 1) datagram.

    Sends the minimum fields GridTracker and N1MM need to place map144 on
    the band map: dial frequency, mode, DE callsign, DE grid.

    TX fields are all false/zero — map144 is receive-only.

    Field order from WSJT-X NetworkMessage.hpp:
        quint64  Dial Frequency (Hz)
        utf8     Mode
        utf8     DX Call
        utf8     Report
        utf8     TX Mode
        bool     TX Enabled
        bool     Transmitting
        bool     Decoding
        quint32  RX DF  (audio Hz offset)
        quint32  TX DF
        utf8     DE Call (my callsign)
        utf8     DE Grid (my grid)
        utf8     DX Grid
        bool     TX Watchdog
        utf8     Sub-mode
        bool     Fast mode
        quint8   Special operation mode
        quint32  Frequency tolerance (Hz)
        quint32  T/R period (s)
        utf8     Configuration Name
        utf8     TX message
    """
    return (_header(MSG_STATUS)
            + _pack_u64(dial_freq_hz)     # Dial Frequency
            + _pack_utf8("MSK144")        # Mode
            + _pack_utf8("")              # DX Call
            + _pack_utf8("")              # Report
            + _pack_utf8("MSK144")        # TX Mode
            + _pack_bool(False)           # TX Enabled
            + _pack_bool(False)           # Transmitting
            + _pack_bool(True)            # Decoding
            + _pack_u32(1500)             # RX DF (audio centre Hz)
            + _pack_u32(1500)             # TX DF
            + _pack_utf8(my_call)         # DE Call
            + _pack_utf8(my_grid)         # DE Grid
            + _pack_utf8("")              # DX Grid
            + _pack_bool(False)           # TX Watchdog
            + _pack_utf8("")              # Sub-mode
            + _pack_bool(True)            # Fast mode (MSK144 uses fast/short T/R)
            + struct.pack(">B", 0)        # Special operation mode
            + _pack_u32(100)              # Frequency tolerance (Hz)
            + _pack_u32(15)               # T/R period (s)
            + _pack_utf8("Default")       # Configuration Name
            + _pack_utf8(""))             # TX message


def build_decode(decode: dict, my_call: str, dial_freq_hz: int) -> bytes:
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

    return (_header(MSG_DECODE)
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

# ── PSKReporter IPFIX helpers ─────────────────────────────────────────────────

def _ipfix_varlen(data: bytes) -> bytes:
    """IPFIX variable-length field: 1-byte length prefix (or 3-byte if ≥255)."""
    n = len(data)
    if n < 255:
        return struct.pack('B', n) + data
    return struct.pack('>BH', 255, n) + data


def _ipfix_ef(ie_id: int, length: int) -> bytes:
    """Enterprise field specifier: IE id with enterprise bit + length + enterprise number."""
    return struct.pack('>HHI', ie_id | 0x8000, length, PSKREPORTER_ENTERPRISE)


def _ipfix_sf(ie_id: int, length: int) -> bytes:
    """Standard IANA field specifier: IE id + length (no enterprise number)."""
    return struct.pack('>HH', ie_id, length)


def _ipfix_header(total_len: int, export_time: int, seq: int, obs_id: int) -> bytes:
    return struct.pack('>HHIII', 10, total_len, export_time, seq, obs_id)


def _ipfix_pad(body: bytes) -> bytes:
    """Pad body to 4-byte boundary."""
    rem = len(body) % 4
    return body + bytes((4 - rem) % 4)


def _build_psk_datagram(spots: list, my_call: str, my_grid: str,
                         obs_id: int, seq: int,
                         send_templates: bool = True,
                         software: str = "map144") -> bytes:
    """Build a single IPFIX UDP datagram matching WSJT-X PSKReporter.cpp exactly.

    Structure (all in one datagram):
        IPFIX header
        [optional] Sender template set   (set ID 2)
        [optional] Receiver options template set (set ID 3)
        Receiver data set                (template ID 0x50e2)
        Sender data set                  (template ID 0x50e3)

    Key details from WSJT-X source:
        - informationSource = 1 (REPORTER_SOURCE_AUTOMATIC)
        - Receiver options template scope field count = 0
        - All sets in one datagram so PSKReporter can correlate templates/data
    """
    VARLEN = 0xFFFF
    payload = b''

    if send_templates:
        # Sender template set (set ID 2)
        sender_tmpl = (
            struct.pack('>HH', PSKREPORTER_SENDER_TMPL, 7)  # link_id, field_count
            + _ipfix_ef(0x0001, VARLEN)   # senderCallsign
            + _ipfix_ef(0x0005, 5)        # frequency (5 bytes)
            + _ipfix_ef(0x0006, 1)        # sNR (1 byte signed)
            + _ipfix_ef(0x000a, VARLEN)   # mode
            + _ipfix_ef(0x0003, VARLEN)   # senderLocator
            + _ipfix_ef(0x000b, 1)        # informationSource
            + _ipfix_sf(150, 4)           # dateTimeSeconds (IANA IE 150)
        )
        sender_set = struct.pack('>HH', 2, 0) + sender_tmpl
        sender_set = _ipfix_pad(sender_set)
        sender_set = struct.pack('>HH', 2, len(sender_set)) + sender_set[4:]

        # Receiver options template set (set ID 3), scope field count = 0
        recv_tmpl = (
            struct.pack('>HHH', PSKREPORTER_RECV_TMPL, 5, 0)  # link_id, field_count, scope_count=0
            + _ipfix_ef(0x0002, VARLEN)   # receiverCallsign
            + _ipfix_ef(0x0004, VARLEN)   # receiverLocator
            + _ipfix_ef(0x0008, VARLEN)   # decodingSoftware
            + _ipfix_ef(0x0009, VARLEN)   # antennaInformation
            + _ipfix_ef(0x000d, VARLEN)   # rigInformation
        )
        recv_tmpl_set = struct.pack('>HH', 3, 0) + recv_tmpl
        recv_tmpl_set = _ipfix_pad(recv_tmpl_set)
        recv_tmpl_set = struct.pack('>HH', 3, len(recv_tmpl_set)) + recv_tmpl_set[4:]

        payload += sender_set + recv_tmpl_set

    # Receiver data set (our station info)
    recv_data = (
        _ipfix_varlen(my_call.encode('ascii', errors='replace'))
        + _ipfix_varlen(my_grid.encode('ascii', errors='replace'))
        + _ipfix_varlen(software.encode('ascii'))
        + _ipfix_varlen(b'')   # antennaInformation
        + _ipfix_varlen(b'')   # rigInformation
    )
    recv_set = struct.pack('>HH', PSKREPORTER_RECV_TMPL, 0) + recv_data
    recv_set = _ipfix_pad(recv_set)
    recv_set = struct.pack('>HH', PSKREPORTER_RECV_TMPL, len(recv_set)) + recv_set[4:]
    payload += recv_set

    # Sender data set (spots)
    records = b''
    for sp in spots:
        call = sp['de_call'].encode('ascii', errors='replace')
        freq = struct.pack('>Q', int(sp['freq_hz']))[-5:]   # high 5 bytes big-endian
        snr  = max(-127, min(127, int(sp.get('snr') or 0)))
        mode = sp.get('mode', 'MSK144').encode('ascii')
        grid = (sp.get('de_grid') or '').encode('ascii', errors='replace')
        utc  = int(sp['utc_epoch'])
        records += (
            _ipfix_varlen(call)
            + freq
            + struct.pack('b', snr)
            + _ipfix_varlen(mode)
            + _ipfix_varlen(grid)
            + struct.pack('B', 1)        # informationSource = 1 (REPORTER_SOURCE_AUTOMATIC)
            + struct.pack('>I', utc)
        )
    sender_set = struct.pack('>HH', PSKREPORTER_SENDER_TMPL, 0) + records
    sender_set = _ipfix_pad(sender_set)
    sender_set = struct.pack('>HH', PSKREPORTER_SENDER_TMPL, len(sender_set)) + sender_set[4:]
    payload += sender_set

    now = int(time.time())
    return _ipfix_header(16 + len(payload), now, seq, obs_id) + payload


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
        self.dial_freq_hz        = 0    # updated by report_freq()

        self.dx_enabled = False
        self.dx_host    = 'dxc.ve7cc.net'
        self.dx_port    = 7373

        # PSKReporter IPFIX state
        import random
        self._psk_obs_id  = random.randint(1, 0xFFFFFFFF)
        self._psk_seq     = 0
        self._psk_sock    = None

        # DX cluster state
        self._dx_sock     = None
        self._dx_lock     = threading.Lock()
        self._dx_logged_in = False

        # Runtime state
        self._sock       = None
        self._running    = False
        self._decode_q   = queue.SimpleQueue()
        self._psk_spots  = []           # pending PSKReporter spots
        self._psk_lock   = threading.Lock()
        self._last_psk_upload = time.time()  # don't upload until first spots arrive

        # Stats (read by the UI)
        self.stat_udp_sent      = 0
        self.stat_psk_uploaded  = 0
        self.stat_psk_queued    = 0
        self.stat_last_psk_time = ''
        self.stat_dx_status     = 'disabled'
        self.stat_dx_sent       = 0
        self.stat_last_error    = ''

        self._hb_thread   = None
        self._work_thread = None

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_settings(self, my_call: str, my_grid: str,
                       wsjtx_enabled: bool, wsjtx_host: str, wsjtx_port: int,
                       pskreporter_enabled: bool,
                       dx_enabled: bool = False, dx_host: str = 'dxc.ve7cc.net',
                       dx_port: int = 7373):
        self.my_call   = my_call.strip().upper()
        self.my_grid   = my_grid.strip().upper()
        self.wsjtx_enabled = wsjtx_enabled
        self.wsjtx_host    = wsjtx_host.strip()
        self.wsjtx_port    = int(wsjtx_port)
        self.pskreporter_enabled = pskreporter_enabled

        # Reconnect DX cluster if settings changed
        dx_changed = (dx_enabled != self.dx_enabled or
                      dx_host.strip() != self.dx_host or
                      int(dx_port) != self.dx_port)
        self.dx_enabled = dx_enabled
        self.dx_host    = dx_host.strip()
        self.dx_port    = int(dx_port)
        if dx_changed:
            self._dx_disconnect()
        if self.dx_enabled and self.my_call:
            self._dx_connect()

        self._send_heartbeat()
        self._send_status()

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
        # Send heartbeat immediately so GridTracker registers this client
        # before the first decode arrives (don't wait up to 15 s)
        self._send_heartbeat()

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
        self._dx_disconnect()
        # Final PSKReporter flush
        if self.pskreporter_enabled and self.my_call:
            self._upload_pskreporter()

    def report_decode(self, decode: dict):
        """Called from the GUI decode-drain loop for every successful decode."""
        self._decode_q.put(decode)

    def report_freq(self, freq_hz: int):
        """Called when the tuned frequency changes — triggers a Status send."""
        if freq_hz != self.dial_freq_hz:
            self.dial_freq_hz = freq_hz
            self._send_status()

    # ── Internal threads ──────────────────────────────────────────────────────

    def _dx_connect(self):
        """Open TCP connection to DX cluster and log in with callsign."""
        with self._dx_lock:
            if self._dx_sock is not None:
                return  # already connected
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect((self.dx_host, self.dx_port))
                s.settimeout(None)
                # Read banner (may include login prompt), then send callsign
                s.recv(4096)
                s.sendall((self.my_call + '\r\n').encode('ascii'))
                # Read login response
                s.recv(4096)
                self._dx_sock = s
                self._dx_logged_in = True
                self.stat_dx_status = f"connected: {self.dx_host}:{self.dx_port}"
                print(f"[reporter] DX cluster: connected to {self.dx_host}:{self.dx_port}", flush=True)
            except Exception as e:
                self.stat_dx_status = f"error: {e}"
                self.stat_last_error = f"DX cluster: {e}"
                self._dx_sock = None
                self._dx_logged_in = False

    def _dx_disconnect(self):
        with self._dx_lock:
            if self._dx_sock is not None:
                try:
                    self._dx_sock.close()
                except Exception:
                    pass
                self._dx_sock = None
                self._dx_logged_in = False
                self.stat_dx_status = 'disabled'

    def _dx_send_spot(self, freq_khz: float, dx_call: str, comment: str):
        """Send one DX spot. Reconnects once if the connection has dropped."""
        if not (self.dx_enabled and self.my_call):
            return
        cmd = f"DX {freq_khz:.1f} {dx_call} {comment}\r\n"
        for attempt in range(2):
            with self._dx_lock:
                if self._dx_sock is None:
                    break
                try:
                    self._dx_sock.sendall(cmd.encode('ascii'))
                    self.stat_dx_sent += 1
                    return
                except OSError:
                    try:
                        self._dx_sock.close()
                    except Exception:
                        pass
                    self._dx_sock = None
                    self._dx_logged_in = False
                    self.stat_dx_status = 'reconnecting...'
            if attempt == 0:
                self._dx_connect()

    def _send_heartbeat(self):
        if not (self.wsjtx_enabled and self.my_call and self._sock):
            return
        try:
            pkt = build_heartbeat()
            self._sock.sendto(pkt, (self.wsjtx_host, self.wsjtx_port))
            self.stat_udp_sent += 1
        except OSError as e:
            self.stat_last_error = f"UDP send: {e}"

    def _send_status(self):
        """Send a WSJT-X Status (type 1) datagram if conditions are met."""
        if not (self.wsjtx_enabled and self.my_call and self._sock):
            return
        try:
            pkt = build_status(self.dial_freq_hz, self.my_call, self.my_grid)
            self._sock.sendto(pkt, (self.wsjtx_host, self.wsjtx_port))
            self.stat_udp_sent += 1
        except OSError as e:
            self.stat_last_error = f"UDP send: {e}"

    def _heartbeat_loop(self):
        while self._running:
            self._send_heartbeat()
            self._send_status()
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
                pkt = build_decode(decode, self.my_call, freq_hz)
                self._sock.sendto(pkt, (self.wsjtx_host, self.wsjtx_port))
                self.stat_udp_sent += 1
            except OSError as e:
                self.stat_last_error = f"UDP send: {e}"

        # ── PSKReporter ───────────────────────────────────────────────────────
        # Skip self-spots and spots without a grid (PSKReporter requires both)
        if (self.pskreporter_enabled and self.my_call and de_call
                and de_call.upper() != self.my_call.upper()
                and grid):
            spot = {
                'de_call':   de_call,
                'de_grid':   grid,
                'freq_hz':   freq_hz,
                'snr':       snr,
                'mode':      'MSK144',
                'utc_epoch': int(time.time()),
            }
            with self._psk_lock:
                self._psk_spots.append(spot)
                self.stat_psk_queued = len(self._psk_spots)

        # ── DX Cluster ────────────────────────────────────────────────────────
        # Send immediately on each decode; skip self-spots
        if (self.dx_enabled and self.my_call and de_call
                and de_call.upper() != self.my_call.upper()):
            freq_khz = radio_khz
            snr_str  = f"{snr:+d}dB" if snr is not None else ""
            grid_str = grid or ""
            comment  = f"MSK144 {grid_str} {snr_str}".strip()
            self._dx_send_spot(freq_khz, de_call, comment)

    def _psk_next_seq(self) -> int:
        self._psk_seq += 1
        return self._psk_seq

    def _upload_pskreporter(self):
        with self._psk_lock:
            if not self._psk_spots:
                return
            spots = list(self._psk_spots)
            self._psk_spots.clear()
            self.stat_psk_queued = 0

        self._last_psk_upload = time.time()
        try:
            if self._psk_sock is None:
                self._psk_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            host = socket.gethostbyname(PSKREPORTER_HOST)

            # Always send templates in the same datagram as data — UDP has no
            # ordering guarantee and we only upload every 5 min, so the overhead
            # is negligible and PSKReporter always gets templates before data.
            dgram = _build_psk_datagram(spots, self.my_call, self.my_grid,
                                         self._psk_obs_id, self._psk_next_seq(),
                                         send_templates=True)
            self._psk_sock.sendto(dgram, (host, PSKREPORTER_PORT))

            self.stat_psk_uploaded += len(spots)
            self.stat_last_psk_time = datetime.now(timezone.utc).strftime('%H:%M:%S')
            print(f"[reporter] PSKReporter: sent {len(spots)} spots via IPFIX UDP", flush=True)
            for sp in spots:
                print(f"  spot: {sp['de_call']:12s}  {sp['freq_hz']:12d} Hz  "
                      f"{sp.get('mode','?'):8s}  SNR {sp.get('snr','?'):>4}  "
                      f"grid {sp.get('de_grid') or '----'}  "
                      f"t={datetime.fromtimestamp(sp['utc_epoch'], tz=timezone.utc).strftime('%H:%M:%S')}",
                      flush=True)
        except Exception as e:
            self.stat_last_error = f"PSKReporter: {e}"
            print(f"[reporter] PSKReporter upload failed: {e}", flush=True)
            # Re-queue spots so they're not lost
            with self._psk_lock:
                self._psk_spots = spots + self._psk_spots
                self.stat_psk_queued = len(self._psk_spots)

