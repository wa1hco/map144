# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Choose B210 pan_center + hardware sample rate to cover selected dials.

The router tees raw IQ at ``hw_rate`` with DC = ``pan_center_mhz``.  Each dial
audio branch then NCO-mixes ``usb_center = dial + 1500 Hz`` to baseband.
Optional MAP65/QMAP needs a ±(wideband_half_hz) window around its centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .band_plan import DialChannel

# Proven host rate today (USRPSource / Map65Exporter).  384 kHz is listed for
# future auto-widen but stays behind max_hw_rate until live-validated.
SUPPORTED_HW_RATES_HZ: tuple[int, ...] = (192_000, 384_000)
DEFAULT_HW_RATE_HZ = 192_000

# Guard band past the extreme selected USB centres / wideband edges (Hz).
DEFAULT_GUARD_HZ = 15_000.0

# MAP65/QMAP delivered half-bandwidth at 96 kHz IQ.
WIDEBAND_HALF_HZ = 48_000.0


class LoPlanError(ValueError):
    """Selection does not fit in the largest allowed IF."""


@dataclass(frozen=True)
class LoPlan:
    """LO / IF plan for one Apply."""

    pan_center_mhz: float
    hw_rate_hz: int
    span_hz: float
    usable_if_hz: float
    f_min_hz: float
    f_max_hz: float
    dial_count: int
    wideband_center_mhz: float | None = None

    @property
    def pan_center_hz(self) -> float:
        return self.pan_center_mhz * 1e6


def _required_edges(
    dials: Sequence[DialChannel],
    *,
    wideband_center_mhz: float | None,
    wideband_half_hz: float,
) -> tuple[float, float]:
    """Return (f_min_hz, f_max_hz) covering dials + optional wideband window."""
    edges: list[float] = []
    for d in dials:
        edges.append(d.usb_center_hz)
    if wideband_center_mhz is not None:
        c = float(wideband_center_mhz) * 1e6
        edges.append(c - wideband_half_hz)
        edges.append(c + wideband_half_hz)
    if not edges:
        raise LoPlanError("no dial channels or wideband centre selected")
    return min(edges), max(edges)


def usable_if_hz(hw_rate_hz: int) -> float:
    """Two-sided usable IF width at the given hardware rate (Hz).

    Uses the full Nyquist of the raw stream (±hw_rate/2).  The MSK144 host FIR
    is *not* applied on the router path, so this is wider than MAP144's ~40 kHz
    detect grid.
    """
    return float(hw_rate_hz)


def plan_lo(
    dials: Sequence[DialChannel],
    *,
    wideband_center_mhz: float | None = None,
    wideband_half_hz: float = WIDEBAND_HALF_HZ,
    guard_hz: float = DEFAULT_GUARD_HZ,
    hw_rates: Sequence[int] = SUPPORTED_HW_RATES_HZ,
    max_hw_rate_hz: int = DEFAULT_HW_RATE_HZ,
    pan_center_mhz: float | None = None,
) -> LoPlan:
    """Pick pan_center and the smallest hw_rate that covers the selection.

    Parameters
    ----------
    dials
        Selected WSJT-X dial channels (may be empty if only wideband IQ is on).
    wideband_center_mhz
        MAP65/QMAP stream centre, or None if IQ export is off.
    max_hw_rate_hz
        Cap on rate (v1 default 192 kHz).  Raise to unlock 384 kHz after validation.
    pan_center_mhz
        Optional operator override; default = midpoint of required span.
    """
    allowed = sorted(r for r in hw_rates if r <= int(max_hw_rate_hz))
    if not allowed:
        raise LoPlanError(f"no supported hw_rate ≤ {max_hw_rate_hz}")

    f_min, f_max = _required_edges(
        dials,
        wideband_center_mhz=wideband_center_mhz,
        wideband_half_hz=wideband_half_hz,
    )
    span = (f_max - f_min) + 2.0 * float(guard_hz)

    chosen = None
    for rate in allowed:
        if usable_if_hz(rate) + 1e-9 >= span:
            chosen = rate
            break
    if chosen is None:
        raise LoPlanError(
            f"selection spans {span/1e3:.1f} kHz (with {guard_hz/1e3:.1f} kHz "
            f"guard) but max IF is {usable_if_hz(allowed[-1])/1e3:.1f} kHz "
            f"at {allowed[-1]/1000} kHz; deselect channels or raise max_hw_rate"
        )

    if pan_center_mhz is None:
        mid_hz = 0.5 * (f_min + f_max)
        pan_center_mhz = mid_hz / 1e6
    else:
        pan_center_mhz = float(pan_center_mhz)
        # Verify override still covers the span inside ±Nyquist.
        half = 0.5 * usable_if_hz(chosen)
        lo = pan_center_mhz * 1e6 - half
        hi = pan_center_mhz * 1e6 + half
        need_lo = f_min - guard_hz
        need_hi = f_max + guard_hz
        if need_lo < lo - 1e-6 or need_hi > hi + 1e-6:
            raise LoPlanError(
                f"pan_center {pan_center_mhz:.6f} MHz at {chosen/1000} kHz "
                f"does not cover [{need_lo/1e6:.6f}, {need_hi/1e6:.6f}] MHz"
            )

    return LoPlan(
        pan_center_mhz=pan_center_mhz,
        hw_rate_hz=int(chosen),
        span_hz=span,
        usable_if_hz=usable_if_hz(chosen),
        f_min_hz=f_min,
        f_max_hz=f_max,
        dial_count=len(dials),
        wideband_center_mhz=wideband_center_mhz,
    )
