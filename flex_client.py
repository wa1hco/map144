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
"""Compatibility shim and standalone DAXIQ test receiver.

This file serves two purposes:

1. **Backwards-compatibility import shim** — ``from flexclient.core import *``
   pulls every public symbol from the ``flexclient`` package into this
   module's namespace.  Any code that previously imported from the old
   top-level ``flex_client`` module continues to work without changes.

2. **Standalone command-line test receiver** — when run as a script
   (``python flex_client.py``), ``main()`` connects to a FlexRadio, receives
   DAXIQ samples for a configurable number of seconds, and reports packet
   count, sample count, and drop count.  Useful for verifying network
   connectivity and VITA-49 stream health before running the full GUI.

Command-line arguments (standalone mode)
-----------------------------------------
--ip IP            Radio IP address; auto-discovered via UDP broadcast if
                   omitted.
--freq FLOAT       Centre frequency in MHz (default: 50.260).
--rate INT         Sample rate in Hz (default: 96000).
--bind-client-id UUID   GUI client UUID for ``client bind`` (see ``flexclient``
                   documentation); omit for auto-discovery.
--bind-client UUID Deprecated alias of ``--bind-client-id``.
--secs INT         Duration in seconds (default: 5).

Output
------
Every 50 packets a log line is printed showing cumulative packet count,
sample count, the most recent VITA-49 timestamp (integer + fractional seconds),
and the stream ID.  At exit, total packets, samples, and drop count are logged.
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
