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
"""Flat public API re-export for the flexclient package.

This module exists to support the legacy import pattern used by
``flex_client.py``::

    from flexclient.core import *

It re-exports every public symbol from the package submodules into a single
flat namespace so that code written before the package was restructured
continues to work without modification.

For new code, prefer importing directly from the package::

    from flexclient import FlexDAXIQ, VitaPacket

or from the specific submodule::

    from flexclient.client import FlexDAXIQ

Exported symbols (identical to ``flexclient.__init__.__all__``)
---------------------------------------------------------------
Constants       DISCOVERY_PORT, FLEX_TCP_PORT, VITA_UDP_PORT,
                VITA_TYPE_IF_DATA, VITA_TYPE_EXT_DATA, SAMPLE_RATES,
                SMARTSDR_STATUS_MESSAGES
Helpers         _format_status_detail, _maybe_log_unmapped_status_code,
                _pick_udp_listen_port
Data classes    FlexRadio, VitaPacket
Discovery       discover, _parse_discovery, _format_discovery_summary
TCP client      FlexTCPClient
DAXIQ setup     DAXIQSetup, _extract_key, _parse_freq_to_mhz
VITA receiver   VITAReceiver
Orchestrator    FlexDAXIQ
"""

from .common import (
    DISCOVERY_PORT,
    FLEX_TCP_PORT,
    VITA_UDP_PORT,
    VITA_TYPE_IF_DATA,
    VITA_TYPE_EXT_DATA,
    SAMPLE_RATES,
    SMARTSDR_STATUS_MESSAGES,
    _format_status_detail,
    _maybe_log_unmapped_status_code,
    _pick_udp_listen_port,
    log,
)
from .models import FlexRadio, VitaPacket
from .discovery import discover, _parse_discovery, _format_discovery_summary
from .tcp_client import FlexTCPClient
from .setup import DAXIQSetup, _extract_key, _parse_freq_to_mhz
from .vita import VITAReceiver
from .client import FlexDAXIQ

__all__ = [
    "DISCOVERY_PORT",
    "FLEX_TCP_PORT",
    "VITA_UDP_PORT",
    "VITA_TYPE_IF_DATA",
    "VITA_TYPE_EXT_DATA",
    "SAMPLE_RATES",
    "SMARTSDR_STATUS_MESSAGES",
    "_format_status_detail",
    "_maybe_log_unmapped_status_code",
    "_pick_udp_listen_port",
    "FlexRadio",
    "VitaPacket",
    "discover",
    "_parse_discovery",
    "_format_discovery_summary",
    "FlexTCPClient",
    "DAXIQSetup",
    "_extract_key",
    "_parse_freq_to_mhz",
    "VITAReceiver",
    "FlexDAXIQ",
]

