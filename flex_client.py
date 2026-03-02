"""Compatibility shim for legacy imports.

All FlexRadio client implementation now lives in ``flexclient.core``.
"""

import argparse
import time

from flexclient.core import *
from flexclient.common import log


def main() -> None:
    parser = argparse.ArgumentParser(description="FlexRadio DAXIQ test receiver")
    parser.add_argument("--ip", default=None, help="Radio IP (auto-discover if omitted)")
    parser.add_argument("--freq", default=50.260, type=float, help="Center freq MHz")
    parser.add_argument("--rate", default=96000, type=int, help="Sample rate Hz")
    parser.add_argument("--bind-client-id", default=None, help="GUI client UUID for client bind client_id=<uuid>")
    parser.add_argument("--bind-client", default=None, help="Deprecated alias of --bind-client-id")
    parser.add_argument("--secs", default=5, type=int, help="Seconds to run")
    args = parser.parse_args()

    client = FlexDAXIQ(
        radio_ip=args.ip,
        center_freq_mhz=args.freq,
        sample_rate=args.rate,
        bind_client_id=args.bind_client_id or args.bind_client,
    )

    try:
        client.start()
        t_end = time.time() + args.secs
        total_samples = 0
        packet_count = 0

        while time.time() < t_end:
            pkt = client.get_samples(timeout=0.5)
            if pkt:
                packet_count += 1
                total_samples += len(pkt.samples)
                if packet_count % 50 == 0:
                    ts = pkt.timestamp_int + pkt.timestamp_frac * 1e-12
                    log.info(
                        f"Packets: {packet_count}  Samples: {total_samples}  "
                        f"Last timestamp: {ts:.3f}  "
                        f"Stream: 0x{pkt.stream_id:08x}"
                    )

        log.info(
            f"Done. {packet_count} packets, {total_samples} samples, "
            f"{client._vita.drop_count} drops"
        )

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
