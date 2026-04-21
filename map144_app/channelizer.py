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
import numba
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
    """Design the LP prototype filter for the channelizer (float32 taps)."""
    nyq = sample_rate / 2.0
    return firwin(numtaps, cutoff_hz / nyq).astype(np.float32)


def _design_hp_filter(
    ch_rate: int   = CH_SAMPLE_RATE,
    cutoff:  float = HP_CUTOFF_HZ,
    ntaps:   int   = 15,
) -> np.ndarray:
    """HP FIR at cutoff Hz for the per-channel 300 Hz passband edge."""
    return firwin(ntaps, cutoff / (ch_rate / 2.0), pass_zero=False).astype(np.float32)


# -- ChannelizerState ----------------------------------------------------------

@dataclass
class ChannelizerState:
    """Mutable state that persists across apply_channelizer() calls.

    zi_c       -- LP filter state, shape (n_channels, ntaps-1) complex64
    abs_sample -- total input samples seen (for phase continuity)
    _hp_taps, _hp_zi   -- per-channel HP FIR taps and state
    _b_lp_rev          -- pre-reversed LP taps (float32); stored once at init
    _mix_step          -- per-channel NCO step phasor (complex64)
    _mix_phase         -- per-channel NCO current phase phasor (complex64)
    channel_offset_hz  -- offset applied to all NCOs:
                          (calling_freq_hz - pan_center_hz) + 1500 Hz
    """
    _hp_taps:  np.ndarray = field(default=None, repr=False)
    _hp_zi:    np.ndarray = field(default=None, repr=False)

    zi_c:      np.ndarray = field(default=None, repr=False)
    abs_sample: int       = 0

    _mix_step:  np.ndarray = field(default=None, repr=False)
    _mix_phase: np.ndarray = field(default=None, repr=False)

    _b_lp_rev:  np.ndarray = field(default=None, repr=False)

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

    lp_taps_f32 = np.ascontiguousarray(lp_taps, dtype=np.float32)
    b_lp_rev    = np.ascontiguousarray(lp_taps_f32[::-1])     # pre-reversed once

    lp_ntaps = len(lp_taps)
    zi_c     = np.zeros((n_channels, lp_ntaps - 1), dtype=np.complex64)
    hp_zi    = np.zeros((n_channels, hp_ntaps - 1), dtype=np.complex64)
    return ChannelizerState(
        _hp_taps=hp_taps, _hp_zi=hp_zi,
        zi_c=zi_c,
        _mix_step=mix_step, _mix_phase=mix_phase,
        _b_lp_rev=b_lp_rev,
        channel_offset_hz=channel_offset_hz,
    )


# -- Helpers -------------------------------------------------------------------

def _c64(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Pack float32 re/im arrays into complex64 without a float64 intermediate.

    ``re + 1j * im`` promotes float32 to complex128 because Python's ``1j``
    is float64-class.  Assigning to pre-allocated .real/.imag views stays in
    float32 throughout.
    """
    out = np.empty(re.shape, dtype=np.complex64)
    out.real[:] = re
    out.imag[:] = im
    return out


# -- Vectorized FIR helper -----------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _nb_fir_2d(b_rev: np.ndarray, x_ext: np.ndarray, y: np.ndarray) -> None:
    """Numba JIT kernel: y[k,n] = sum_j b_rev[j] * x_ext[k, n+j].

    Parallelises over the channel axis (k).  Stays in float32 throughout.
    cache=True: compiled on first call, reused across restarts.
    """
    ntaps = b_rev.shape[0]
    n_ch  = x_ext.shape[0]
    N     = y.shape[1]
    for k in numba.prange(n_ch):
        for n in range(N):
            acc = np.float32(0.0)
            for j in range(ntaps):
                acc += b_rev[j] * x_ext[k, n + j]
            y[k, n] = acc


@numba.njit(parallel=True, cache=True)
def _nb_fir_2d_c64(b_rev: np.ndarray, x_ext: np.ndarray, y: np.ndarray) -> None:
    """Numba JIT kernel: complex64 FIR with real float32 coefficients.

    y[k,n] = sum_j b_rev[j] * x_ext[k, n+j]
    Parallel over channels (k).  Faster than splitting into re/im FIR + _c64 pack.
    """
    ntaps = b_rev.shape[0]
    n_ch  = x_ext.shape[0]
    N     = y.shape[1]
    for k in numba.prange(n_ch):
        for n in range(N):
            acc = np.complex64(0.0)
            for j in range(ntaps):
                acc += b_rev[j] * x_ext[k, n + j]
            y[k, n] = acc


def _fir_filt_2d_c64(b: np.ndarray, x: np.ndarray, zi: np.ndarray):
    """Complex64 FIR along axis=1, returning (y, new_zi).

    b   : float32 filter taps, shape (ntaps,)
    x   : complex64 input, shape (n_ch, N)
    zi  : complex64 state, shape (n_ch, ntaps-1)
    """
    ntaps   = len(b)
    n_ch, N = x.shape
    x_ext   = np.concatenate([zi, x], axis=1)
    b_rev   = np.ascontiguousarray(b[::-1])
    y       = np.empty((n_ch, N), dtype=np.complex64)
    _nb_fir_2d_c64(b_rev, x_ext, y)
    new_zi  = x_ext[:, -(ntaps - 1):].copy()
    return y, new_zi


def _fir_filt_2d(b: np.ndarray, x: np.ndarray, zi: np.ndarray):
    """FIR filter x (n_ch, N) along axis=1, returning (y, new_zi).

    Delegates to _nb_fir_2d (Numba parallel JIT, ~3× faster than the
    previous as_strided + matmul path).

      y[k, n] = sum_{j=0}^{ntaps-1}  b_rev[j] * x_ext[k, n+j]

    b, x, and zi must all be the same real float dtype (float32).

    zi shape: (n_ch, ntaps-1).
    """
    ntaps   = len(b)
    n_ch, N = x.shape
    x_ext   = np.concatenate([zi, x], axis=1)          # (n_ch, ntaps-1+N)
    b_rev   = np.ascontiguousarray(b[::-1])             # contiguous reversed taps
    y       = np.empty((n_ch, N), dtype=np.float32)
    _nb_fir_2d(b_rev, x_ext, y)
    new_zi  = x_ext[:, -(ntaps - 1):].copy()           # (n_ch, ntaps-1)
    return y, new_zi


# -- Fused NCO + LP FIR kernels ------------------------------------------------
# These kernels replace the three-step: (1) build phase matrix, (2) mix, (3) FIR.
# Each Numba thread keeps a per-channel local buffer (zi + N mixed samples) in
# L1/L2 cache, avoiding two (n_ch, N) complex64 heap allocations per hop.
# mix_phase is advanced N steps inside the kernel and returned via new_mix_phase.

@numba.njit(parallel=True, cache=True)
def _nb_mix_fir_sp(b_rev, raw, mix_phase, mix_step, zi, y, new_zi, new_mix_phase):
    """Single-pol fused NCO mix + complex64 FIR, parallel over channels.

    b_rev        : float32, (ntaps,) — pre-reversed LP taps
    raw          : complex64, (N,)   — shared IQ input block
    mix_phase    : complex64, (n_ch,) — starting NCO phase per channel
    mix_step     : complex64, (n_ch,) — per-sample NCO step per channel
    zi           : complex64, (n_ch, ntaps-1) — FIR state
    y            : complex64, (n_ch, N) — output (pre-allocated)
    new_zi       : complex64, (n_ch, ntaps-1) — FIR state on exit (pre-allocated)
    new_mix_phase: complex64, (n_ch,) — NCO phase on exit (pre-allocated)
    """
    ntaps = b_rev.shape[0]
    N     = raw.shape[0]
    n_ch  = mix_phase.shape[0]
    for k in numba.prange(n_ch):
        ph   = mix_phase[k]
        st   = mix_step[k]
        # Thread-local buffer: zi history + N freshly mixed samples
        buf  = np.empty(ntaps - 1 + N, dtype=np.complex64)
        for j in range(ntaps - 1):
            buf[j] = zi[k, j]
        for n in range(N):
            buf[ntaps - 1 + n] = raw[n] * ph
            ph *= st
        # FIR convolution
        for n in range(N):
            acc = np.complex64(0.0)
            for j in range(ntaps):
                acc += b_rev[j] * buf[n + j]
            y[k, n] = acc
        # Save updated state
        for j in range(ntaps - 1):
            new_zi[k, j] = buf[N + j]
        new_mix_phase[k] = ph


@numba.njit(parallel=True, cache=True)
def _nb_mix_fir_dp(b_rev, raw_h, raw_v, mix_phase, mix_step,
                   zi_h, zi_v, y_h, y_v, new_zi_h, new_zi_v, new_mix_phase):
    """Dual-pol fused NCO mix + complex64 FIR; H and V share the same NCO grid.

    One prange loop over channels; each iteration processes both H and V so the
    per-sample NCO step is computed once and applied to both polarisations.
    """
    ntaps = b_rev.shape[0]
    N     = raw_h.shape[0]
    n_ch  = mix_phase.shape[0]
    for k in numba.prange(n_ch):
        ph    = mix_phase[k]
        st    = mix_step[k]
        buf_h = np.empty(ntaps - 1 + N, dtype=np.complex64)
        buf_v = np.empty(ntaps - 1 + N, dtype=np.complex64)
        for j in range(ntaps - 1):
            buf_h[j] = zi_h[k, j]
            buf_v[j] = zi_v[k, j]
        for n in range(N):
            buf_h[ntaps - 1 + n] = raw_h[n] * ph
            buf_v[ntaps - 1 + n] = raw_v[n] * ph
            ph *= st
        # Combined H+V FIR — share b_rev[j] load across both accumulators
        for n in range(N):
            acc_h = np.complex64(0.0)
            acc_v = np.complex64(0.0)
            for j in range(ntaps):
                b_j    = b_rev[j]
                acc_h += b_j * buf_h[n + j]
                acc_v += b_j * buf_v[n + j]
            y_h[k, n] = acc_h
            y_v[k, n] = acc_v
        for j in range(ntaps - 1):
            new_zi_h[k, j] = buf_h[N + j]
            new_zi_v[k, j] = buf_v[N + j]
        new_mix_phase[k] = ph


def _mix_fir_lp_sp(b_rev, raw, mix_phase, mix_step, zi):
    """Single-pol fused LP: mix IQ + FIR in one pass per channel.

    Returns (y, new_zi, new_mix_phase), all complex64.
    Eliminates the (n_ch, N) phase-matrix and mixed-signal allocations.
    """
    n_ch  = mix_phase.shape[0]
    N     = raw.shape[0]
    ntaps = b_rev.shape[0]
    y             = np.empty((n_ch, N),       dtype=np.complex64)
    new_zi        = np.empty((n_ch, ntaps-1), dtype=np.complex64)
    new_mix_phase = np.empty(n_ch,            dtype=np.complex64)
    _nb_mix_fir_sp(b_rev, raw, mix_phase, mix_step, zi, y, new_zi, new_mix_phase)
    return y, new_zi, new_mix_phase


def _mix_fir_lp_dp(b_rev, raw_h, raw_v, mix_phase, mix_step, zi_h, zi_v):
    """Dual-pol fused LP: mix IQ + FIR for H and V in one pass per channel.

    Returns (y_h, y_v, new_zi_h, new_zi_v, new_mix_phase), all complex64.
    """
    n_ch  = mix_phase.shape[0]
    N     = raw_h.shape[0]
    ntaps = b_rev.shape[0]
    y_h           = np.empty((n_ch, N),       dtype=np.complex64)
    y_v           = np.empty((n_ch, N),       dtype=np.complex64)
    new_zi_h      = np.empty((n_ch, ntaps-1), dtype=np.complex64)
    new_zi_v      = np.empty((n_ch, ntaps-1), dtype=np.complex64)
    new_mix_phase = np.empty(n_ch,            dtype=np.complex64)
    _nb_mix_fir_dp(b_rev, raw_h, raw_v, mix_phase, mix_step,
                   zi_h, zi_v, y_h, y_v, new_zi_h, new_zi_v, new_mix_phase)
    return y_h, y_v, new_zi_h, new_zi_v, new_mix_phase


# -- Main entry point ----------------------------------------------------------

def apply_channelizer(
    iq_block: np.ndarray,
    state: ChannelizerState,
    n_channels: int            = N_CHANNELS,
    sample_rate: int           = 48000,
    channel_spacing_hz: float  = CHANNEL_SPACING_HZ,
    lp_taps: np.ndarray | None = None,
    decimate_factor: int       = DECIMATE_FACTOR,
    iq_block_v: np.ndarray | None  = None,
    state_v:    ChannelizerState | None = None,
) -> np.ndarray:
    """Mix iq_block into channels, LP-filter, decimate, and HP-filter.

    Parameters
    -----------
    iq_block   : complex IQ at sample_rate Hz, shape (N,).  DC = IF centre.
    state      : ChannelizerState -- mutated in place on return.
    iq_block_v : optional V-channel IQ, same shape as iq_block.  When provided
                 together with state_v, both polarisations are processed in one
                 pass: the fused NCO+FIR kernel processes H and V together so the
                 per-sample NCO step is computed once per channel.
    state_v    : ChannelizerState for V channel.

    Returns
    --------
    When iq_block_v / state_v are None: complex64 array (n_channels, N//decimate_factor).
    When dual-pol args are present:     tuple (ch_h, ch_v), each complex64
                                        (n_channels, N//decimate_factor).
    """
    # Ensure complex64 without copying if dtype already matches.
    raw_h = np.ascontiguousarray(iq_block,   dtype=np.complex64)
    dual  = (iq_block_v is not None) and (state_v is not None)
    raw_v = np.ascontiguousarray(iq_block_v, dtype=np.complex64) if dual else None

    N       = len(raw_h)
    # lp_taps parameter is accepted for API compatibility but not used;
    # the pre-reversed taps stored in state are used instead.
    b_lp_rv = state._b_lp_rev   # pre-reversed float32 taps; no copy needed

    if not dual:
        # ── Single-pol path ────────────────────────────────────────────────────
        # Fused NCO mix + LP FIR: per-channel buffer stays in L1 cache,
        # eliminating the (n_ch, N) phase-matrix and mixed-signal allocations.
        ch_lp, state.zi_c, new_mp = _mix_fir_lp_sp(
            b_lp_rv, raw_h, state._mix_phase, state._mix_step, state.zi_c)
        state._mix_phase = new_mp
        ch_out           = ch_lp[:, ::decimate_factor]
        state.abs_sample += N
        ch_hp, new_hp_zi = _fir_filt_2d_c64(state._hp_taps, ch_out, state._hp_zi)
        state._hp_zi     = new_hp_zi
        return ch_hp

    # ── Dual-pol path — fused NCO+FIR for H and V in one prange pass ──────────
    # Both polarisations share the same NCO grid; the per-channel phase step is
    # computed once and applied to both raw_h and raw_v inside the kernel.
    ch_lp_h, ch_lp_v, new_zi_h, new_zi_v, new_mp = _mix_fir_lp_dp(
        b_lp_rv, raw_h, raw_v, state._mix_phase, state._mix_step,
        state.zi_c, state_v.zi_c)

    state.zi_c   = new_zi_h
    state_v.zi_c = new_zi_v
    state._mix_phase  = new_mp
    state_v._mix_phase = new_mp   # V shares NCO — no copy needed (read-only after this)

    ch_h = ch_lp_h[:, ::decimate_factor]
    ch_v = ch_lp_v[:, ::decimate_factor]

    state.abs_sample  += N
    state_v.abs_sample += N

    # HP filter — batch H+V over 2×n_channels rows (unchanged from before)
    stacked_dec      = np.concatenate([ch_h, ch_v],                   axis=0)
    hp_zi_both       = np.concatenate([state._hp_zi, state_v._hp_zi], axis=0)
    hp_both, new_hp_zi_both = _fir_filt_2d_c64(
        state._hp_taps, stacked_dec, hp_zi_both)
    state._hp_zi   = new_hp_zi_both[:n_channels]
    state_v._hp_zi = new_hp_zi_both[n_channels:]

    return hp_both[:n_channels], hp_both[n_channels:]
