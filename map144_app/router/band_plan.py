# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""WSJT-X dial-frequency table for the B210 audio/IQ router.

Loads the shipped JSON table and filters to bands the B210 can usefully
tune.  Official Ettus B210 floor is 70 MHz; 6 m (50 MHz) is included because
MAP144 already operates the B210 there.  HF (< 50 MHz) is excluded in v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

_DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "data" / "wsjtx_frequencies.json"

# Practical VHF+ floor for this project (includes 6 m).  Not the Ettus datasheet
# 70 MHz floor — see module docstring.
_MIN_DIAL_MHZ = 50.0
_MAX_DIAL_MHZ = 6000.0

# Display order for band combo boxes.
_BAND_ORDER = ("4m", "6m", "2m", "1.25m", "70cm", "23cm")


@dataclass(frozen=True)
class DialChannel:
    """One WSJT-X working-frequency row."""

    band: str
    mode: str
    dial_mhz: float
    label: str = ""

    @property
    def dial_hz(self) -> float:
        return self.dial_mhz * 1e6

    @property
    def usb_center_hz(self) -> float:
        """RF Hz of USB signal energy centre (dial + 1500 Hz)."""
        return self.dial_hz + 1500.0

    @property
    def dial_khz_token(self) -> str:
        """Compact dial token for sink names, e.g. 144150 for 144.150 MHz."""
        return f"{self.dial_mhz * 1000.0:.0f}"

    def sink_name(self, rf: int | None = None) -> str:
        """PipeWire machine sink name (GUI-invisible)."""
        mode = self.mode.lower().replace(" ", "")
        band = self.band.lower().replace(".", "p")
        base = f"map144.{band}.{mode}.{self.dial_khz_token}.rx"
        if rf is None:
            return base
        return f"map144.RF{int(rf)}.{band}.{mode}.{self.dial_khz_token}.rx"

    def description(self, rf: int | None = None) -> str:
        """Human label shown in WSJT-X / pavucontrol."""
        lab = self.label or self.mode
        core = f"MAP144 {self.band} {lab} {self.dial_mhz:.3f} -> WSJT-X"
        if rf is None:
            return core
        return f"MAP144 RF{int(rf)} {self.band} {lab} {self.dial_mhz:.3f} -> WSJT-X"


def load_frequency_table(path: Path | None = None) -> dict:
    """Load and return the raw frequency-table JSON object."""
    p = Path(path) if path is not None else _DEFAULT_TABLE
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _in_b210_range(dial_mhz: float, min_mhz: float = _MIN_DIAL_MHZ,
                   max_mhz: float = _MAX_DIAL_MHZ) -> bool:
    return min_mhz <= float(dial_mhz) <= max_mhz


@lru_cache(maxsize=4)
def _cached_entries(path_str: str) -> tuple[DialChannel, ...]:
    data = load_frequency_table(Path(path_str) if path_str else _DEFAULT_TABLE)
    min_mhz = float(data.get("b210_min_mhz", _MIN_DIAL_MHZ))
    # Prefer project floor (includes 6 m) over a stricter JSON floor if present.
    min_mhz = min(min_mhz, _MIN_DIAL_MHZ)
    max_mhz = float(data.get("b210_max_mhz", _MAX_DIAL_MHZ))
    out: list[DialChannel] = []
    for row in data.get("entries", []):
        dial = float(row["dial_mhz"])
        if not _in_b210_range(dial, min_mhz, max_mhz):
            continue
        out.append(DialChannel(
            band=str(row["band"]),
            mode=str(row["mode"]),
            dial_mhz=dial,
            label=str(row.get("label") or row["mode"]),
        ))
    return tuple(out)


def all_channels(path: Path | None = None) -> tuple[DialChannel, ...]:
    """All dial channels that pass the B210-tunable filter."""
    p = str(path) if path is not None else str(_DEFAULT_TABLE)
    return _cached_entries(p)


def bands(path: Path | None = None) -> list[str]:
    """Band names that have at least one tunable dial, in display order."""
    present = {c.band for c in all_channels(path)}
    ordered = [b for b in _BAND_ORDER if b in present]
    extras = sorted(present - set(_BAND_ORDER))
    return ordered + extras


def channels_for_band(band: str, path: Path | None = None,
                      modes: Sequence[str] | None = None) -> list[DialChannel]:
    """Dial channels for one band, optionally filtered by mode names."""
    mode_set = {m.upper() for m in modes} if modes is not None else None
    rows = [c for c in all_channels(path) if c.band == band]
    if mode_set is not None:
        rows = [c for c in rows if c.mode.upper() in mode_set]
    rows.sort(key=lambda c: (c.mode, c.dial_mhz))
    return rows


def channels_by_key(keys: Iterable[tuple[str, str, float]],
                    path: Path | None = None) -> list[DialChannel]:
    """Resolve (band, mode, dial_mhz) keys to DialChannel rows.

    Matching uses 1 Hz tolerance on dial_mhz.  Unknown keys are skipped.
    """
    index = {(c.band, c.mode.upper(), round(c.dial_mhz, 6)): c
             for c in all_channels(path)}
    out: list[DialChannel] = []
    for band, mode, dial in keys:
        k = (band, mode.upper(), round(float(dial), 6))
        ch = index.get(k)
        if ch is not None:
            out.append(ch)
    return out
