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
"""IF channel grid: indices, coarse mix frequencies, and usable rows.

The channelizer places mix frequencies at k x 1 kHz + channel_offset_hz on the
48 kHz IF (see channelizer.py).  channel_offset_hz = (calling_freq_hz - pan_center_hz) + 1500,
so each channel NCO sits at the USB signal energy position for stations dialled to
calling_freq + k x 1000 Hz (USB signal energy = dial + 1500 Hz).

Which rows are usable follows the IF passband (+-24 kHz about IQ DC) and
Nyquist-edge masking: NYQUIST_EDGE_CHANNEL_SKIP excludes rows within that many
indices of the Nyquist bin (ch 24).  It must stay aligned with processing.py.

In-channel band shaping (HP/LP at 12 kHz / 48 kHz) is in channelizer.py.

Separate helpers (*_dial_*) map coarse IF offsets to logging / decode frequencies
using the decoder's fixed audio reference (1500 Hz); that layer is not part of
the channelizer itself.
"""

from __future__ import annotations

from .channelizer import (
    CHANNEL_OFFSET_HZ,
    CHANNEL_SPACING_HZ,
    HP_CUTOFF_HZ,
    LP_CUTOFF_HZ,
    N_CHANNELS,
)

# Half-width in channel index from Nyquist (ch 24) -- excluded from auto-detect.
# Must match processing.py (_EDGE_CH_SKIP aliases this).
NYQUIST_EDGE_CHANNEL_SKIP = 4

# Fixed audio reference frequency used when mapping IF offset to logging dial (Hz).
JT9_AUDIO_CENTER_HZ = 1500.0


def channel_index_to_signed(ch_k: int) -> int:
    """Map row index ch_k to signed IF step.

    Channels 0 to N_CHANNELS//2 map to non-negative signed indices.
    Channels above N_CHANNELS//2 map to negative signed indices (wrap-around).
    Same rule as processing.py.
    """
    if ch_k <= N_CHANNELS // 2:
        return int(ch_k)
    return int(ch_k) - N_CHANNELS


def coarse_if_offset_hz(ch_k: int, channel_offset_hz: float = CHANNEL_OFFSET_HZ) -> float:
    """Coarse IF offset (Hz) of channel centre: signed x 1 kHz + channel_offset_hz.

    channel_offset_hz should match the value used to build the channelizer state:
    (calling_freq_hz - pan_center_hz) + 1500.
    """
    s = channel_index_to_signed(ch_k)
    return s * CHANNEL_SPACING_HZ + channel_offset_hz


def usable_channel_indices(edge_skip: int = NYQUIST_EDGE_CHANNEL_SKIP) -> list[int]:
    """Row indices kept for auto-detect (outside Nyquist guard region)."""
    ny = N_CHANNELS // 2
    return [k for k in range(N_CHANNELS) if abs(k - ny) > edge_skip]


def monitored_if_span_khz(
    edge_skip: int = NYQUIST_EDGE_CHANNEL_SKIP,
    channel_offset_hz: float = CHANNEL_OFFSET_HZ,
) -> tuple[float, float, int]:
    """Min/max coarse IF offset (kHz) for usable rows and channel count."""
    ks = usable_channel_indices(edge_skip)
    if not ks:
        return 0.0, 0.0, 0
    offs_khz = [coarse_if_offset_hz(k, channel_offset_hz) / 1000.0 for k in ks]
    return min(offs_khz), max(offs_khz), len(ks)


def msk144_rf_usb_span_edges_mhz(
    if_centre_mhz: float,
    edge_skip: int = NYQUIST_EDGE_CHANNEL_SKIP,
    channel_offset_hz: float = CHANNEL_OFFSET_HZ,
) -> tuple[float, float, int, int]:
    """RF span (MHz) for MSK144 in-channel audio over auto-detect rows.

    The channelizer retains roughly HP_CUTOFF_HZ to LP_CUTOFF_HZ Hz above each
    channel's IF mix centre in the complex row.  Returns the lowest RF of the
    HP edge, the highest RF of the LP edge across usable_channel_indices, and
    the row indices k that attain those extrema (for display).
    """
    ks = usable_channel_indices(edge_skip)
    if not ks:
        return if_centre_mhz, if_centre_mhz, -1, -1
    best_lo = None
    best_hi = None
    k_lo = k_hi = -1
    for k in ks:
        if_hz = coarse_if_offset_hz(k, channel_offset_hz)
        rf_c_mhz = if_centre_mhz + if_hz / 1e6
        edge_lo_mhz = rf_c_mhz + HP_CUTOFF_HZ / 1e6
        edge_hi_mhz = rf_c_mhz + LP_CUTOFF_HZ / 1e6
        if best_lo is None or edge_lo_mhz < best_lo:
            best_lo = edge_lo_mhz
            k_lo = k
        if best_hi is None or edge_hi_mhz > best_hi:
            best_hi = edge_hi_mhz
            k_hi = k
    return best_lo, best_hi, k_lo, k_hi


def format_monitor_status(if_min_khz: float, if_max_khz: float, n_ch: int) -> str:
    """One-line status: IF offset range for monitored channels."""
    if n_ch <= 0:
        return "Channels: --"
    return f"Channels: {if_min_khz:+.0f} to {if_max_khz:+.0f} kHz  ({n_ch} ch)"


def channel_summary_lines(
    pan_center_mhz: float,
    edge_skip: int = NYQUIST_EDGE_CHANNEL_SKIP,
    channel_offset_hz: float = CHANNEL_OFFSET_HZ,
) -> list[str]:
    """Per usable row: k, signed index, IF centre offset kHz, absolute RF at IF centre (MHz)."""
    lines = []
    for k in usable_channel_indices(edge_skip):
        s = channel_index_to_signed(k)
        if_hz = coarse_if_offset_hz(k, channel_offset_hz)
        rf_mhz = pan_center_mhz + if_hz / 1e6
        lines.append(
            f"k={k:2d}  signed={s:+3d}  IF_centre={if_hz/1000:+.3f} kHz  RF@IF={rf_mhz:.6f} MHz"
        )
    return lines


# --- Decode / logging (IF to dial) -- not part of channelizer geometry ---


def logging_dial_offset_khz_from_if_offset(if_offset_hz: float) -> float:
    """kHz dial offset from pan centre for logging, given IF offset (decoder 1500 Hz ref)."""
    return (if_offset_hz - JT9_AUDIO_CENTER_HZ) / 1000.0


def logging_dial_mhz(
    pan_center_mhz: float,
    ch_k: int,
    channel_offset_hz: float = CHANNEL_OFFSET_HZ,
) -> float:
    """Absolute logging dial (MHz) at channel coarse centre."""
    ddk = logging_dial_offset_khz_from_if_offset(coarse_if_offset_hz(ch_k, channel_offset_hz))
    return pan_center_mhz + ddk / 1000.0


def monitored_logging_dial_span_mhz(
    pan_center_mhz: float,
    edge_skip: int = NYQUIST_EDGE_CHANNEL_SKIP,
    channel_offset_hz: float = CHANNEL_OFFSET_HZ,
) -> tuple[float, float, int]:
    """Min/max logging dial (MHz) at coarse centres (same rows as usable_channel_indices)."""
    ks = usable_channel_indices(edge_skip)
    if not ks:
        return pan_center_mhz, pan_center_mhz, 0
    dials = [logging_dial_mhz(pan_center_mhz, k, channel_offset_hz) for k in ks]
    return min(dials), max(dials), len(ks)
