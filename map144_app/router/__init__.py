# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""B210 multi-mode audio / IQ router core.

Band → WSJT-X dial channel list → user selection → PipeWire sinks (+ optional
MAP65/QMAP TIMF2).  Standalone from the MAP144 detect/decode path.
"""

from .band_plan import DialChannel, bands, channels_for_band, load_frequency_table
from .engine import RouterConfig, RouterEngine, RouterStatus
from .lo_planner import LoPlan, LoPlanError, plan_lo
from .wideband_iq import WidebandDest

__all__ = [
    "DialChannel",
    "LoPlan",
    "LoPlanError",
    "RouterConfig",
    "RouterEngine",
    "RouterStatus",
    "WidebandDest",
    "bands",
    "channels_for_band",
    "load_frequency_table",
    "plan_lo",
]
