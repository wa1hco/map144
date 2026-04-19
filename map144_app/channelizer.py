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
"""Polyphase channelizer: splits 48 kHz IQ into 48 channels at 12 kHz complex rate.

Design goal
-----------
Channel k captures signals from stations whose USB dial is at calling_freq + k × 1000 Hz.

In USB radio the dial is the suppressed carrier; signal energy is 1500 Hz above
the dial.  A station with dial at calling_freq + k × 1000 Hz transmits energy at
calling_freq + k × 1000 + 1500 Hz.  In the IF centred at pan_center, that energy
arrives at k × 1000 + (calling_freq - pan_center) + 1500 Hz from IQ DC.

The NCO for channel k is placed exactly there so the signal lands at DC (0 Hz)
in the channel output — required by the squaring detector.

NCO placement
-------------
Each channel k mixes at:

    f_nco(k) = k x CHANNEL_SPACING_HZ + channel_offset_hz   [Hz from IQ DC]

where:

    channel_offset_hz = (calling_freq_hz - pan_center_hz) + 1500

Two components:
    calling_freq_hz - pan_center_hz  :  keeps the grid on calling_freq + k kHz
                                         when the panadapter is not exactly there
    +1500 Hz                         :  USB audio offset — places each NCO on
                                         the signal energy, not the dial frequency

When pan = calling_freq, channel_offset_hz = 1500, so channel 0 NCO is at 1500 Hz
(where a station dialled to calling_freq transmits), channel 1 at 2500 Hz, etc.

Adapting wideband IF to internal DSP
--------------------------------------
1. Per-channel mix grid on the IF.  Channel k uses f_c(k) as above.

2. Per-channel band-limiting.  After mix: LP at 48 kHz (LP_CUTOFF_HZ),
   4x decimate to 12 kHz, HP at 12 kHz (HP_CUTOFF_HZ).  This retains
   300-2700 Hz of audio above each channel center.

3. Auto-detect skips rows near Nyquist per channel_plan.NYQUIST_EDGE_CHANNEL_SKIP.

Channel boundary limits
------------------------
With a panadapter offset of delta Hz, the audio passband edge of channel k
in the IF is:

    upper audio edge: k x 1000 - delta + 2700 <= 24000
                      k_max = floor((21300 + delta) / 1000)

    lower audio edge: k x 1000 - delta + 300 >= -24000
                      k_min = ceil((-24300 + delta) / 1000)

For worst-case |delta| = 500 Hz, one channel is lost at the near-Nyquist edge.
The existing NYQUIST_EDGE_CHANNEL_SKIP guard already excludes those channels.

Per-channel output
-------------------
Each row is complex64 at 12 kHz.  0 Hz in the row is the channel center after
the NCO mix.  The row is not real-only audio; squaring uses both I and Q.
Real-audio extraction for the decoder is done in detection.extract_and_decode.

Filter design
--------------
LP to LP_CUTOFF_HZ (2700 Hz) at 48 kHz after mix.  Limits bandwidth to enable
4x decimation to 12 kHz.

HP at HP_CUTOFF_HZ (300 Hz) at 12 kHz.  Passes roughly 300-2700 Hz in each
channel's complex spectrum.

Streaming state
----------------
ChannelizerState holds LP/HP FIR state and NCO phase between apply_channelizer
calls.  It also holds the channel_offset_hz used to build the NCO, which
processing.py reads back when computing fc_hz for the decoder.

Public API
-----------
design_channelizer_filter(sample_rate) -> ndarray
    Prototype LP filter taps.

make_channelizer_state(n_channels, lp_taps, sample_rate, channel_offset_hz) -> ChannelizerState
    Build the channelizer state (NCO and FIR state).

apply_channelizer(iq_block, state, ...) -> ndarray  shape (n_channels, n_out)
    Process one IQ chunk; mutates state.  Returns complex channels at 12 kHz.

Constants
----------
N_CHANNELS          : 48
CHANNEL_SPACING_HZ  : 1000 -- spacing between mix frequencies (Hz)
CHANNEL_OFFSET_HZ   : 0    -- module default; overridden at runtime via make_channelizer_state
DECIMATE_FACTOR     : 4
CH_SAMPLE_RATE      : 12000 -- complex samples per row per second
LP_CUTOFF_HZ        : 2700
HP_CUTOFF_HZ        : 300
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin

log = logging.getLogger(__name__)

# -- Public constants ----------------------------------------------------------
N_CHANNELS         = 48
CHANNEL_SPACING_HZ = 1000.0
CHANNEL_OFFSET_HZ  = 0.0       # module default; pass channel_offset_hz to make_channelizer_state
DECIMATE_FACTOR    = 4
CH_SAMPLE_RATE     = 12000
LP_CUTOFF_HZ       = 2700.0
HP_CUTOFF_HZ       = 300.0
LP_NUMTAPS         = 23         # 23 taps needed for adequate stopband rejection; fewer taps
                                 # spread weak-ping energy across adjacent channels, reducing
                                 # per-channel peak below the detection threshold

log.info("channelizer: scipy mix / LP FIR / decimate")


# -- Filter design -------------------------------------------------------------

def design_channelizer_filter(
    sample_rate: int = 48000,
    cutoff_hz: float = LP_CUTOFF_HZ,
    numtaps: int = LP_NUMTAPS,
) -> np.ndarray:
    """Design the LP prototype filter for the channelizer (float64 taps)."""
    nyq = sample_rate / 2.0
    return firwin(numtaps, cutoff_hz / nyq).astype(np.float64)


def _design_hp_filter(
    ch_rate: int   = CH_SAMPLE_RATE,
    cutoff:  float = HP_CUTOFF_HZ,
    ntaps:   int   = 15,
) -> np.ndarray:
    """HP FIR at cutoff Hz for the per-channel 300 Hz passband edge."""
    return firwin(ntaps, cutoff / (ch_rate / 2.0), pass_zero=False).astype(np.float64)


# -- ChannelizerState ----------------------------------------------------------

@dataclass
class ChannelizerState:
    """Mutable state that persists across apply_channelizer() calls.

    zi_re / zi_im      -- LP filter state, shape (n_channels, ntaps-1) float64
    abs_sample         -- total input samples seen (for phase continuity)
    _hp_taps, _hp_zi   -- per-channel HP FIR taps and state
    _mix_step          -- per-channel NCO step phasor (complex64)
    _mix_phase         -- per-channel NCO current phase phasor (complex64)
    channel_offset_hz  -- offset applied to all NCOs:
                          (calling_freq_hz - pan_center_hz) + 1500 Hz
    """
    _hp_taps:  np.ndarray = field(default=None, repr=False)
    _hp_zi:    np.ndarray = field(default=None, repr=False)

    zi_re:     np.ndarray = field(default=None, repr=False)
    zi_im:     np.ndarray = field(default=None, repr=False)
    abs_sample: int       = 0

    _mix_step:  np.ndarray = field(default=None, repr=False)
    _mix_phase: np.ndarray = field(default=None, repr=False)

    channel_offset_hz: float = 0.0


def make_channelizer_state(
    n_channels: int,
    lp_taps: np.ndarray,
    sample_rate: int = 48000,
    channel_offset_hz: float = 0.0,
) -> ChannelizerState:
    """Allocate channelizer state (LP/HP FIR state and NCO).

    channel_offset_hz = (calling_freq_hz - pan_center_hz) + 1500.  Channel k
    then captures stations with USB dial at calling_freq + k x 1000 Hz.  The
    +1500 aligns each NCO with the USB signal energy (dial + 1500 Hz).  See
    engine.py and the channelizer.py module docstring for the derivation.
    """
    hp_taps  = _design_hp_filter()
    hp_ntaps = len(hp_taps)

    k         = np.arange(n_channels, dtype=np.float64)
    fc        = k * CHANNEL_SPACING_HZ + channel_offset_hz
    mix_step  = np.exp(-2j * np.pi * fc / float(sample_rate)).astype(np.complex64)
    mix_phase = np.ones(n_channels, dtype=np.complex64)

    lp_ntaps = len(lp_taps)
    zi_re    = np.zeros((n_channels, lp_ntaps - 1), dtype=np.float64)
    zi_im    = np.zeros((n_channels, lp_ntaps - 1), dtype=np.float64)
    hp_zi    = np.zeros((n_channels, hp_ntaps - 1), dtype=np.complex128)
    return ChannelizerState(
        _hp_taps=hp_taps, _hp_zi=hp_zi,
        zi_re=zi_re, zi_im=zi_im,
        _mix_step=mix_step, _mix_phase=mix_phase,
        channel_offset_hz=channel_offset_hz,
    )


# -- Vectorized FIR helper -----------------------------------------------------

def _fir_filt_2d(b: np.ndarray, x: np.ndarray, zi: np.ndarray):
    """FIR filter x (n_ch, N) along axis=1, returning (y, new_zi).

    Stride-trick overlap + matmul.  Taps and zi must already be float64
    (callers must not pass float32 or mixed dtypes — that forces astype
    copies which dominate runtime).

    zi shape: (n_ch, ntaps-1).
    """
    ntaps   = len(b)
    n_ch, N = x.shape
    x_ext   = np.concatenate([zi, x], axis=1)          # (n_ch, ntaps-1+N)
    s0, s1  = x_ext.strides
    windows = np.lib.stride_tricks.as_strided(
        x_ext,
        shape=(n_ch, N, ntaps),
        strides=(s0, s1, s1),
    )
    y      = windows @ b[::-1]                          # (n_ch, N)
    new_zi = x_ext[:, -(ntaps - 1):].copy()            # (n_ch, ntaps-1)
    return y, new_zi


# -- Main entry point ----------------------------------------------------------

def apply_channelizer(
    iq_block: np.ndarray,
    state: ChannelizerState,
    n_channels: int            = N_CHANNELS,
    sample_rate: int           = 48000,
    channel_spacing_hz: float  = CHANNEL_SPACING_HZ,
    lp_taps: np.ndarray | None = None,
    decimate_factor: int       = DECIMATE_FACTOR,
) -> np.ndarray:
    """Mix iq_block into channels, LP-filter, decimate, and HP-filter.

    Parameters
    -----------
    iq_block  : complex IQ at sample_rate Hz (typically 48 kHz IF), shape (N,).
                DC = IF centre (panadapter center).
    state     : ChannelizerState -- mutated in place on return.

    Returns
    --------
    complex64 array of shape (n_channels, N // decimate_factor).  Each row is
    complex at 12 kHz.  0 Hz in the row is the channel center after the NCO,
    which corresponds to calling_freq + k x 1000 Hz in RF when state was built
    with the correct channel_offset_hz.
    """
    raw = iq_block.astype(np.complex64)

    if lp_taps is None:
        lp_taps = design_channelizer_filter(sample_rate)

    N = len(raw)

    angles = np.angle(state._mix_step).astype(np.float32)
    phase0 = np.angle(state._mix_phase).astype(np.float32)
    n_idx      = np.arange(N, dtype=np.float32)
    angle_mat  = phase0[:, np.newaxis] + np.outer(angles, n_idx)
    phase      = (np.cos(angle_mat) + 1j * np.sin(angle_mat)).astype(np.complex64)
    state._mix_phase = (np.cos(phase0 + angles * N) +
                        1j * np.sin(phase0 + angles * N)).astype(np.complex64)

    mixed = phase * raw[np.newaxis, :]

    b_lp = lp_taps.astype(np.float64)
    filt_re, new_zi_re = _fir_filt_2d(b_lp, mixed.real, state.zi_re)
    filt_im, new_zi_im = _fir_filt_2d(b_lp, mixed.imag, state.zi_im)
    filtered = (filt_re + 1j * filt_im).astype(np.complex64)

    state.zi_re = new_zi_re
    state.zi_im = new_zi_im

    ch_out = filtered[:, ::decimate_factor]

    state.abs_sample += len(raw)

    ch_out_c128 = ch_out.astype(np.complex128)
    hp_re, new_hp_zi_re = _fir_filt_2d(state._hp_taps, ch_out_c128.real, state._hp_zi.real)
    hp_im, new_hp_zi_im = _fir_filt_2d(state._hp_taps, ch_out_c128.imag, state._hp_zi.imag)
    state._hp_zi = new_hp_zi_re + 1j * new_hp_zi_im

    return (hp_re + 1j * hp_im).astype(np.complex64)
