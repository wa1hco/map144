#!/usr/bin/env python3
"""Inject a synthetic WSJT-X UDP decode to test the map144 -> GridTracker -> N1MM chain.

Reuses map144's own wire-format builders (map144_app/reporting.py) so the
datagrams are byte-for-byte identical to what map144 emits on a real decode.
No radio or meteor ping required.

Sends, in order:  Heartbeat (type 0) -> Status (type 1) -> Decode (type 2).
GridTracker registers the client on the heartbeat/status and shows the decode.

Usage (Windows, from the repo root):
    env\\python.exe tools\\wsjtx_inject.py --host 192.168.10.12 --port 2237

Common options:
    --host   target IP   (the GridTracker PC's receive address)
    --port   target port (GridTracker's *receive* port; default 2237)
    --call   your callsign      (default W2SZ)
    --grid   your grid          (default FN32JP)
    --freq   dial freq MHz      (default 144.150)
    --message  fake decode text (default 'CQ K1ABC FN42')
    --snr    reported SNR dB     (default 5)
    --repeat N  send the decode N times, 2 s apart (default 1)
"""
import argparse
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make 'map144_app' importable when run from tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map144_app.reporting import build_heartbeat, build_status, build_decode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='192.168.10.12', help='target IP (GridTracker PC)')
    ap.add_argument('--port', type=int, default=2237, help='target UDP port (GridTracker receive)')
    ap.add_argument('--call', default='W2SZ', help='your (DE) callsign')
    ap.add_argument('--grid', default='FN32JP', help='your grid')
    ap.add_argument('--freq', type=float, default=144.150, help='dial frequency in MHz')
    ap.add_argument('--message', default='CQ K1ABC FN42', help='fake decode message text')
    ap.add_argument('--snr', type=int, default=5, help='reported SNR (dB)')
    ap.add_argument('--repeat', type=int, default=1, help='number of decodes to send')
    args = ap.parse_args()

    dial_hz = int(round(args.freq * 1e6))
    dest = (args.host, args.port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Heartbeat + Status so GridTracker registers "map144" as a live client.
    sock.sendto(build_heartbeat(), dest)
    sock.sendto(build_status(dial_hz, args.call.upper(), args.grid.upper()), dest)
    print(f"[inject] heartbeat + status -> {args.host}:{args.port}  "
          f"({args.call.upper()} {args.grid.upper()} @ {args.freq:.3f} MHz)")

    for i in range(max(1, args.repeat)):
        decode = {
            'message':  args.message.upper(),
            'jt9_snr':  args.snr,
            't_sec':    0.5,
            'radio_khz': args.freq * 1000.0,
            'utc_time': datetime.now(timezone.utc).strftime('%H:%M:%S'),
        }
        pkt = build_decode(decode, args.call.upper(), dial_hz)
        sock.sendto(pkt, dest)
        print(f"[inject] decode {i+1}/{args.repeat}: '{decode['message']}' "
              f"SNR {args.snr:+d} -> {args.host}:{args.port}  ({len(pkt)} bytes)")
        if i + 1 < args.repeat:
            time.sleep(2.0)

    sock.close()
    print("[inject] done. Watch GridTracker's decode/activity and N1MM's "
          "WSJT Decode List for the message above.")


if __name__ == '__main__':
    main()
