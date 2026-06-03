#!/usr/bin/env python3
"""Live PSKReporter MQTT spot logger for MAP144 propagation research.

Subscribes to the PSKReporter live MQTT feed (mqtt.pskreporter.info, hosted by
Tom M0LTE, data from N1DQ) and appends every matching reception spot to a local
JSONL file.  This is a *passive* subscriber: the broker pushes spots to us, so
the only cost to the shared resource is the bandwidth of the narrow topics we
ask for.  We never poll and never touch the rate-limited HTTP retrieve API.

Capture-first by design: log everything raw now, do the join against MAP144's
decodes.jsonl offline later.  That way we consume the shared feed exactly once.

Topic format (from https://www.mqtt.pskreporter.info/):
    pskr/filter/v2/{band}/{mode}/{tx_call}/{rx_call}/{tx_grid}/{rx_grid}/{tx_dxcc}/{rx_dxcc}
    +  = single-level wildcard, # = everything below (end only)

Payload (JSON) fields we care about:
    sc/sl = sender call / locator      rc/rl = receiver call / locator
    f     = frequency Hz               md    = mode        rp = SNR dB
    t     = report epoch               t_tx  = normalised 15-s tx-start epoch
    b     = band                       sa/ra = sender / receiver ADIF DXCC
    sq    = sequence number (dedup key)

Each logged line is the raw spot JSON plus two added fields:
    _rx    = local epoch when we received it (for clock-skew checks)
    _topic = the full topic the broker delivered it on

Run a connectivity test (FT8 is dense, spots arrive immediately):
    python tools/pskr_logger.py --modes FT8 --bands 6m

Run the research capture (sparse — expect long silences, heartbeat confirms alive):
    python tools/pskr_logger.py            # default: 2m+6m, MSK144

Stop with Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import paho.mqtt.client as mqtt

DEFAULT_HOST = "mqtt.pskreporter.info"
DEFAULT_PORT = 1883  # plain MQTT over TCP
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MSK144", "detections", "pskr_spots.jsonl",
)


def build_topics(bands, modes):
    """Cartesian product of bands x modes -> filtered v2 topic subscriptions."""
    topics = []
    for band in bands:
        for mode in modes:
            topics.append(f"pskr/filter/v2/{band}/{mode}/#")
    return topics


def make_client(client_id):
    """Construct an MQTT client that works on both paho 1.x and 2.x."""
    try:
        # paho-mqtt >= 2.0 requires an explicit callback API version.
        from paho.mqtt.client import CallbackAPIVersion
        return mqtt.Client(
            CallbackAPIVersion.VERSION1, client_id=client_id, clean_session=True
        )
    except (ImportError, TypeError):
        # paho-mqtt 1.x
        return mqtt.Client(client_id=client_id, clean_session=True)


class SpotLogger:
    def __init__(self, out_path, topics, quiet=False):
        self.out_path = out_path
        self.topics = topics
        self.quiet = quiet
        self.count = 0
        self.t_start = time.time()
        self.last_heartbeat = self.t_start
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # line-buffered append so a kill -9 never loses more than the last line
        self.fh = open(out_path, "a", buffering=1, encoding="utf-8")

    # --- MQTT callbacks (VERSION1 signatures) ---
    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[pskr] connect failed rc={rc}", file=sys.stderr)
            return
        for t in self.topics:
            client.subscribe(t, qos=0)
        print(f"[pskr] connected, subscribed to {len(self.topics)} topic(s):")
        for t in self.topics:
            print(f"         {t}")
        print(f"[pskr] logging -> {self.out_path}")
        print("[pskr] waiting for spots (sparse topics may be silent a long while; "
              "heartbeat every 60 s confirms the link is alive)")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"[pskr] unexpected disconnect rc={rc}; auto-reconnecting...",
                  file=sys.stderr)

    def on_message(self, client, userdata, msg):
        try:
            spot = json.loads(msg.payload.decode("utf-8", "replace"))
        except (ValueError, UnicodeError):
            return  # ignore malformed payloads rather than crash the logger
        spot["_rx"] = round(time.time(), 1)
        spot["_topic"] = msg.topic
        self.fh.write(json.dumps(spot, separators=(",", ":")) + "\n")
        self.count += 1
        if not self.quiet:
            sc = spot.get("sc", "?")
            sl = spot.get("sl", "")
            rc = spot.get("rc", "?")
            rl = spot.get("rl", "")
            rp = spot.get("rp", "?")
            b = spot.get("b", "?")
            md = spot.get("md", "?")
            print(f"[spot] {b:>3} {md:<6} {sc:<9}{sl:<7} -> {rc:<9}{rl:<7} {rp:>4} dB")

    def heartbeat(self):
        now = time.time()
        if now - self.last_heartbeat >= 60.0:
            mins = (now - self.t_start) / 60.0
            rate = self.count / mins if mins > 0 else 0.0
            print(f"[pskr] alive {mins:5.1f} min  spots={self.count}  "
                  f"({rate:.1f}/min)")
            self.last_heartbeat = now

    def close(self):
        try:
            self.fh.flush()
            self.fh.close()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="PSKReporter MQTT spot logger")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--bands", default="2m,6m",
                    help="comma-separated band segments (e.g. 2m,6m)")
    ap.add_argument("--modes", default="MSK144",
                    help="comma-separated mode segments (e.g. MSK144,FT8)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL path")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-spot lines (heartbeat only)")
    args = ap.parse_args(argv)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    topics = build_topics(bands, modes)

    # polite, stable-ish client id: identifies us to the broker operator
    client_id = f"map144-wa1hco-{os.getpid()}"
    logger = SpotLogger(args.out, topics, quiet=args.quiet)

    client = make_client(client_id)
    client.on_connect = logger.on_connect
    client.on_disconnect = logger.on_disconnect
    client.on_message = logger.on_message
    # be a good citizen: don't reconnect-storm if the broker bounces us
    client.reconnect_delay_set(min_delay=2, max_delay=120)

    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as e:
        print(f"[pskr] could not connect to {args.host}:{args.port}: {e}",
              file=sys.stderr)
        logger.close()
        return 1

    client.loop_start()
    try:
        while True:
            time.sleep(1.0)
            logger.heartbeat()
    except KeyboardInterrupt:
        print(f"\n[pskr] stopping; captured {logger.count} spots "
              f"in {(time.time()-logger.t_start)/60:.1f} min")
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
