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
"""Polyphase channelizer: splits 48 kHz IQ into N sub-bands at 12 kHz each.

Backend selection
-----------------
If GNURadio is importable, ``apply_channelizer`` uses
``gnuradio.filter.pfb_channelizer_ccf``, which implements the polyphase
filter bank in VOLK-accelerated C++ — typically 10-20x faster than the
scipy loop for this channel count and filter length.

If GNURadio is not installed the original scipy mix/LP-filter/decimate loop
is used automatically; no code changes are required in callers.

Filter design
-------------
Two stages:
  1. LP prototype (0 - LP_CUTOFF_HZ = 2700 Hz) passed to pfb_channelizer_ccf.
     Designed at the input rate (48 kHz); padded to a multiple of N_CHANNELS.
     With oversample_rate = CH_SAMPLE_RATE x N_CHANNELS / Fs_in = 12, the PFB
     outputs 12 kHz per channel -- matching the WSJT-X decode rate.

  2. Per-channel HP (HP_CUTOFF_HZ = 300 Hz, scipy FIR, 65 taps at 12 kHz).
     Removes DC and low-frequency interference AFTER channelization, giving a
     flat 300-2700 Hz passband in each sub-band.

Channel ordering
----------------
Channel k is centred at k x CHANNEL_SPACING_HZ Hz (k = 0 ... N_CHANNELS-1).
Channels above Nyquist (k > Fs/2 / CHANNEL_SPACING_HZ = 24) alias to the
negative-frequency half of the complex spectrum; this matches IQ convention
and is consistent with the original scipy implementation.

Streaming state
---------------
ChannelizerState holds the GNURadio top-block (created once by
make_channelizer_state) and the per-channel HP filter state.  The top-block
is persistent between apply_channelizer calls so that the internal polyphase
filter state is preserved across chunk boundaries.

Public API
----------
design_channelizer_filter(sample_rate) -> ndarray
    Prototype LP filter taps.

make_channelizer_state(n_channels, lp_taps) -> ChannelizerState
    Build the channelizer state (and GNURadio flowgraph if available).

apply_channelizer(iq_block, state, ...) -> ndarray  (n_channels, n_out)
    Process one IQ chunk; mutates *state*.

Constants
---------
N_CHANNELS          : 48    -- number of sub-bands
CHANNEL_SPACING_HZ  : 1000  -- centre frequency step per channel (Hz)
DECIMATE_FACTOR     : 4     -- 48 kHz / 4 = 12 kHz per channel
CH_SAMPLE_RATE      : 12000 -- output sample rate per channel (Hz)
LP_CUTOFF_HZ        : 2700  -- LP prototype passband edge (Hz)
HP_CUTOFF_HZ        : 300   -- per-channel HP passband edge (Hz)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin

log = logging.getLogger(__name__)

# ── Public constants ──────────────────────────────────────────────────────────
N_CHANNELS         = 48
CHANNEL_SPACING_HZ = 1000.0
CHANNEL_OFFSET_HZ  = 500.0      # shift channel centres by +500 Hz so that USB MSK144
                                  # signals at integer-kHz dial + 1500 Hz audio land exactly
                                  # at a channel centre (carrier = N×1000 + 1500 Hz → offset
                                  # channel k centred at k×1000 + 500 Hz = N×1000 + 1500 Hz
                                  # when k = N+1, giving fc_offset = 0 and perfect mixing).
DECIMATE_FACTOR    = 4
CH_SAMPLE_RATE     = 12000
LP_CUTOFF_HZ       = 2700.0
HP_CUTOFF_HZ       = 300.0
LP_NUMTAPS         = 23         # scipy fallback tap count (reduced for speed)

# oversample_rate for pfb_channelizer_ccf:
#   = CH_SAMPLE_RATE x N_CHANNELS / Fs_in = 12000 x 48 / 48000 = 12
_OVERSAMPLE_RATE   = CH_SAMPLE_RATE * N_CHANNELS // 48000   # 12

# ── GNURadio availability ─────────────────────────────────────────────────────
try:
    from gnuradio import gr, blocks            # type: ignore[import]
    from gnuradio.filter import pfb, firdes    # type: ignore[import]
    _HAS_GNURADIO = True
    log.info("GNURadio found -- using pfb_channelizer_ccf backend")
except ImportError:
    _HAS_GNURADIO = False
    log.info("GNURadio not found -- using scipy mix/filter/decimate backend")


# ── Filter design ─────────────────────────────────────────────────────────────

def design_channelizer_filter(
    sample_rate: int = 48000,
    cutoff_hz: float = LP_CUTOFF_HZ,
    numtaps: int = LP_NUMTAPS,
) -> np.ndarray:
    """Design the LP prototype filter for the channelizer.

    Returns float32 taps when GNURadio is available (pfb_channelizer_ccf
    requires float32 and a length that is a multiple of N_CHANNELS).
    Returns float64 taps for the scipy fallback.
    """
    if _HAS_GNURADIO:
        # Transition from cutoff to Nyquist of the output rate (6 kHz).
        transition = (CH_SAMPLE_RATE / 2.0) - cutoff_hz   # 3300 Hz
        taps = firdes.low_pass(
            1.0,
            sample_rate,
            cutoff_hz,
            transition,
            firdes.WIN_HAMMING,
        )
        taps = list(taps)
        rem  = len(taps) % N_CHANNELS
        if rem:
            taps += [0.0] * (N_CHANNELS - rem)
        return np.array(taps, dtype=np.float32)
    else:
        nyq = sample_rate / 2.0
        return firwin(numtaps, cutoff_hz / nyq).astype(np.float64)


def _design_hp_filter(
    ch_rate: int   = CH_SAMPLE_RATE,
    cutoff:  float = HP_CUTOFF_HZ,
    ntaps:   int   = 15,
) -> np.ndarray:
    """HP FIR at *cutoff* Hz for the per-channel 300 Hz passband edge."""
    return firwin(ntaps, cutoff / (ch_rate / 2.0), pass_zero=False).astype(np.float64)


# ── GNURadio flowgraph ────────────────────────────────────────────────────────

class _GRFlowgraph:
    """Persistent pfb_channelizer_ccf top-block, reused for all chunks."""

    def __init__(self, n_ch: int, lp_taps: np.ndarray, oversample: int):
        tb    = gr.top_block("channelizer")
        src   = blocks.vector_source_c([], repeat=False)
        chan  = pfb.channelizer_ccf(
            n_ch, list(lp_taps.astype(np.float32)), float(oversample)
        )
        sinks = [blocks.vector_sink_c() for _ in range(n_ch)]

        tb.connect(src, chan)
        for k, sink in enumerate(sinks):
            tb.connect((chan, k), sink)

        self._tb    = tb
        self._src   = src
        self._sinks = sinks
        self._n_ch  = n_ch
        log.debug(
            "GR channelizer top-block created (%d ch, oversample=%d, %d taps)",
            n_ch, oversample, len(lp_taps),
        )

    def process(self, iq_block: np.ndarray) -> np.ndarray:
        """Feed *iq_block* through the PFB; return (n_ch, n_out) complex64."""
        self._src.set_data(list(iq_block.astype(np.complex64)))
        for s in self._sinks:
            s.reset()
        self._tb.run()
        return np.array(
            [np.array(s.data(), dtype=np.complex64) for s in self._sinks],
            dtype=np.complex64,
        )


# ── ChannelizerState ──────────────────────────────────────────────────────────

@dataclass
class ChannelizerState:
    """Mutable state that persists across apply_channelizer() calls.

    GNURadio path
    -------------
    _gr        -- _GRFlowgraph instance (PFB filter state lives here)
    _hp_taps   -- scipy HP filter taps (float64, ntaps,)
    _hp_zi     -- HP filter state,      (n_channels, ntaps-1) complex128

    Scipy fallback path
    -------------------
    zi_re / zi_im  -- LP filter state, (n_channels, ntaps-1) float64
    abs_sample     -- total input samples seen (for phase continuity)
    """
    # GNURadio path
    _gr:       object     = field(default=None, repr=False)
    _hp_taps:  np.ndarray = field(default=None, repr=False)
    _hp_zi:    np.ndarray = field(default=None, repr=False)

    # Scipy fallback path
    zi_re:     np.ndarray = field(default=None, repr=False)
    zi_im:     np.ndarray = field(default=None, repr=False)
    abs_sample: int       = 0


def make_channelizer_state(n_channels: int, lp_taps: np.ndarray) -> ChannelizerState:
    """Allocate all channelizer state (and GNURadio flowgraph if available)."""
    hp_taps  = _design_hp_filter()
    hp_ntaps = len(hp_taps)

    # _hp_zi stored as complex128 (real part = re state, imag part = im state)
    # so _fir_filt_2d can split them without separate arrays in ChannelizerState.
    if _HAS_GNURADIO:
        fg    = _GRFlowgraph(n_channels, lp_taps, _OVERSAMPLE_RATE)
        hp_zi = np.zeros((n_channels, hp_ntaps - 1), dtype=np.complex128)
        return ChannelizerState(_gr=fg, _hp_taps=hp_taps, _hp_zi=hp_zi)
    else:
        lp_ntaps = len(lp_taps)
        zi_re    = np.zeros((n_channels, lp_ntaps - 1), dtype=np.float64)
        zi_im    = np.zeros((n_channels, lp_ntaps - 1), dtype=np.float64)
        hp_zi    = np.zeros((n_channels, hp_ntaps - 1), dtype=np.complex128)
        return ChannelizerState(
            _hp_taps=hp_taps, _hp_zi=hp_zi,
            zi_re=zi_re, zi_im=zi_im,
        )


# ── Vectorized FIR helper ─────────────────────────────────────────────────────

def _fir_filt_2d(b: np.ndarray, x: np.ndarray, zi: np.ndarray):
    """FIR filter *x* (n_ch, N) along axis=1, returning (y, new_zi).

    Uses stride-trick overlap to avoid scipy's apply_along_axis overhead.
    *zi* shape: (n_ch, ntaps-1); holds the last ntaps-1 input samples.
    """
    ntaps  = len(b)
    n_ch, N = x.shape
    # Prepend state samples so the window can start at sample 0
    x_ext = np.concatenate([zi, x], axis=1)          # (n_ch, ntaps-1 + N)
    # Build sliding windows: shape (n_ch, N, ntaps) via strided view
    s0, s1 = x_ext.strides
    windows = np.lib.stride_tricks.as_strided(
        x_ext,
        shape=(n_ch, N, ntaps),
        strides=(s0, s1, s1),
    )
    # Dot each window against the reversed taps — equivalent to convolution
    y      = windows @ b[::-1]                        # (n_ch, N)
    new_zi = x_ext[:, -(ntaps - 1):]                  # (n_ch, ntaps-1)
    return y, new_zi


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_channelizer(
    iq_block: np.ndarray,
    state: ChannelizerState,
    n_channels: int            = N_CHANNELS,
    sample_rate: int           = 48000,
    channel_spacing_hz: float  = CHANNEL_SPACING_HZ,
    lp_taps: np.ndarray | None = None,
    decimate_factor: int       = DECIMATE_FACTOR,
) -> np.ndarray:
    """Mix iq_block into sub-bands, LP-filter, decimate, and HP-filter.

    Parameters
    ----------
    iq_block  : complex IQ input at *sample_rate* Hz, shape (N,)
    state     : ChannelizerState -- mutated in place on return

    Returns
    -------
    complex64 array of shape (n_channels, N // decimate_factor)
    """
    raw = iq_block.astype(np.complex64)

    if state._gr is not None:
        # ── GNURadio polyphase filter bank ────────────────────────────────
        ch_out = state._gr.process(raw)          # (n_ch, N // decimate_factor)
    else:
        # ── Scipy mix / LP-filter / decimate fallback ─────────────────────
        if lp_taps is None:
            lp_taps = design_channelizer_filter(sample_rate)

        N = len(raw)
        t = state.abs_sample + np.arange(N, dtype=np.float64)

        k     = np.arange(n_channels, dtype=np.float64)[:, np.newaxis]
        phase = np.exp(
            -2j * np.pi * (k * channel_spacing_hz + CHANNEL_OFFSET_HZ) * t[np.newaxis, :] / sample_rate
        )
        mixed = phase * raw[np.newaxis, :].astype(np.complex128)   # (n_ch, N)

        b_lp = lp_taps.astype(np.float64)
        filt_re, new_zi_re = _fir_filt_2d(b_lp, mixed.real, state.zi_re)
        filt_im, new_zi_im = _fir_filt_2d(b_lp, mixed.imag, state.zi_im)
        filtered = (filt_re + 1j * filt_im).astype(np.complex64)

        state.zi_re = new_zi_re
        state.zi_im = new_zi_im

        ch_out = filtered[:, ::decimate_factor]  # (n_ch, N // decimate_factor)

    state.abs_sample += len(raw)

    # ── Per-channel HP filter (300 Hz, both paths) ────────────────────────
    ch_out_c128 = ch_out.astype(np.complex128)
    # Apply HP to real and imaginary parts separately so _fir_filt_2d (real-valued
    # taps, real input) can use the BLAS matmul path without complex arithmetic.
    hp_re, new_hp_zi_re = _fir_filt_2d(state._hp_taps, ch_out_c128.real, state._hp_zi.real)
    hp_im, new_hp_zi_im = _fir_filt_2d(state._hp_taps, ch_out_c128.imag, state._hp_zi.imag)
    state._hp_zi = new_hp_zi_re + 1j * new_hp_zi_im
    hp_out = hp_re + 1j * hp_im

    return hp_out.astype(np.complex64)
