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
"""FFT and channelised MSK144 detection pipeline for incoming IQ sample data.

Pipeline (in order of execution)
---------------------------------
0. **Wideband FFT noise blanker (Linrad-style)** — input is divided into
   NB_FFT_SIZE blocks.  Each block is FFT'd; the per-bin power is normalised
   by a per-bin running average.  If more than NB_BROADBAND_FRAC of bins
   have normalised power above nb_factor² the block is a broadband impulse
   and is zeroed (with Hann-taper fade at the edges).  Narrowband signals
   (meteor bursts, beacons) elevate only a few bins and are not blanked.
   Per-bin averages are updated only from non-blanked blocks, preventing
   impulses from inflating the threshold.

1. **Ring buffer** — cleaned IQ is written to a 30-second circular buffer
   (``Engine._iq_ring``) so ``extract_and_decode`` can read a long window
   around a trigger (aligned with Fast Graph capture regions).

2. Channeliser -- apply_channelizer() splits the 48 kHz IQ block into
   N_CHANNELS=48 channels at CH_SAMPLE_RATE=12 kHz each.  Channel k mixes
   down k x 1000 Hz + channel_offset_hz on the IF, where channel_offset_hz
   = (calling_freq_hz - pan_center_hz) + 1500 Hz.  The +1500 aligns each NCO
   with the USB signal energy (dial + 1500 Hz) for stations on calling_freq + k kHz
   dial.  See channelizer.py.

3. **Per-channel detection** — channelised samples are accumulated in
   ``_ch_buf`` (48 × N) until CH_DETECT_SIZE samples are available.  For each
   complete block:

   a. Square the complex channel samples.
   b. Windowed FFT of the squared signal at 12 kHz.
   c. Pair metric: dB above a rolling 25th-percentile baseline of linear peaks
      in the squared-domain ±1 kHz tone windows (see ``_LO_MASK`` / ``_HI_MASK``).
   d. Coincidence and sustained-signal gates; carrier estimate from peak bins;
      optional ``extract_and_decode`` in a daemon thread.

4. **SNR history** — ``pair_metric`` (48,) is written to ``_ch_snr_history``
   (N_SNR_HIST × 48).  ``displays.py`` draws the 48-channel ridgeline.

5. **Overlap-FFT waterfall** — cleaned IQ in ``_sbuf`` is processed in fft_size
   blocks with 50 % overlap.  Rows are placed using **packet timestamps** so
   the spectrogram matches receive time.

Timing (two clocks)
-------------------
The detection heatmap indexes rows by **wall time** — ``time.time()`` for live
sources and ``_wav_time_cursor`` for WAV — so multiple detection hops inside one
``process_iq_data`` call map to distinct rows.  The waterfall uses **sample
timestamps** (``timestamp_int`` + picosecond ``timestamp_frac``) for
sample-accurate placement; see the ``process_iq_data`` docstring.

Constants
---------
DETECT_THRESH_DB    : 3.5 dB  — threshold: dB above 25th-percentile baseline
DETECT_MERGE_GAP_S  : 1.0 s   — cooldown between triggers on the same channel
CH_DETECT_SIZE      : 512     — per-channel FFT size for detection (at 12 kHz)
_METRIC_HIST_DEPTH  : 300     — rolling history depth for percentile normaliser
N_SNR_HIST          : computed — spans ~one 15 s window at the channeliser hop rate
"""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .channelizer import (
    apply_channelizer,
    N_CHANNELS,
    CHANNEL_SPACING_HZ,
    CH_SAMPLE_RATE,
)
from .channel_plan import NYQUIST_EDGE_CHANNEL_SKIP
from .detection import extract_and_decode

DETECT_THRESH_DB   = 3.0        # dB above 25th-percentile noise baseline
DETECT_MERGE_GAP_S = 0.5        # suppress re-trigger on the same channel within this window (s).
                                 # Per-channel cooldown: parallel channels are fully independent,
                                 # so simultaneous pings at different frequencies are unaffected.
                                 # 500 ms covers the longest underdense ping (5τ at τ=100 ms);
                                 # overdense pings that last longer will re-trigger naturally once
                                 # the cooldown expires, which is desirable for long trails.
CH_DETECT_SIZE     = 512        # samples at 12 kHz per detection FFT block
NB_FACTOR          = 6.0        # blanker amplitude threshold (hot bin power > NB_FACTOR² × avg)
NB_FFT_SIZE        = 256        # detection block size: 5.3 ms at 48 kHz
NB_BROADBAND_FRAC  = 0.30       # fraction of hot bins required to declare broadband impulse
NB_SPEC_AVG_TC     = 2.0        # per-bin spectrum averaging time constant (seconds)
_NB_TAPER_N        = 24         # Hann taper half-width in samples (24 → 0.5 ms at 48 kHz)
# Alias for readability — value lives in ``channel_plan`` (UI + detect share it).
_EDGE_CH_SKIP      = NYQUIST_EDGE_CHANNEL_SKIP
_DC_CH_SKIP        = 3          # skip channels within N of DC (ch 0) — LO leakage / DC offset zone
_SQ_TONE_HZ        = 1000.0     # expected squared-domain tone offset from DC
_SQ_NTOL_HZ        = 200.0      # half-width of each tone search window
_METRIC_HIST_DEPTH = 1500       # frames of rolling linear-peak history for percentile.
                                 # At the 256-sample / 12 kHz hop rate each frame is ~21 ms,
                                 # so 1500 frames ≈ 32 s.  A longer window gives a stable
                                 # per-channel noise floor estimate: narrowband IF spurs (which
                                 # occupy fixed frequencies for seconds at a time) are averaged
                                 # into the baseline rather than transiently biasing it.
                                 # As long as pings occupy < 25 % of that window the 25th
                                 # percentile stays at the noise floor, mirroring the xmed
                                 # normalisation in WSJT-X msk144spd.f90.  Memory cost for
                                 # the history: 1500 × 48 ch × 4 B ≈ 288 KB (negligible).
_METRIC_HIST_STRIDE = 5         # subsample history for partition — evaluate every 5th frame
                                 # (300 effective samples, ~107 ms resolution) at 1/5 the
                                 # compute cost.  Called every 10 hops (~213 ms), so the
                                 # partition sorts 300 floats — well under 1 ms.
_COINCIDENCE_MAX_CLUSTERS = 6   # cluster count gate: if ≥ this many SEPARATE narrow clusters
                                 # fire simultaneously it is likely broadband noise, not multiple
                                 # independent meteor pings.  Each real MSK144 ping spreads into
                                 # 3 adjacent channels (squared-domain tones land at ±1000 Hz,
                                 # straddling the 1 kHz channel boundary), forming one cluster of
                                 # span 2.  Three simultaneous pings at 5 kHz spacing → 3 clusters
                                 # of 3 channels each → 9 total channels; the old raw channel count
                                 # gate (≥8) would incorrectly fire on this legitimate scenario.
                                 # Counting clusters instead: 6 simultaneous pings → 6 clusters
                                 # → gate fires (extremely rare even during a shower peak).
                                 # The per-cluster span gate (_COINCIDENCE_SPAN_CH) still handles
                                 # true wideband impulse noise that forms a single wide cluster.
_SUSTAINED_HOPS      = int(2.0 * CH_SAMPLE_RATE / (CH_DETECT_SIZE // 2))  # 2 s worth of hops
_SUSTAINED_LOCKOUT   = int(30.0 * CH_SAMPLE_RATE / (CH_DETECT_SIZE // 2)) # 30 s lockout
_COINCIDENCE_SPAN_CH = 8        # spread gate: if a single cluster spans > this many channels it
                                 # is wideband noise (real MSK144 ping + sidelobes spans ≤ 3
                                 # channels; impulse noise spans 10+ kHz = 10+ channels).
_CH_DETECT_HOP     = CH_DETECT_SIZE // 2   # 50 % overlap hop
# Enough slots to cover one full 15-second window at the channeliser hop rate
N_SNR_HIST = int(15 * CH_SAMPLE_RATE / _CH_DETECT_HOP) + 2   # ≈ 705

# Pre-computed Hann taper kernel for soft-blanking (avoids rectangular-hole sinc spread)
_NB_TAPER_KERNEL = np.hanning(2 * _NB_TAPER_N + 1).astype(np.float32)
_NB_TAPER_KERNEL /= _NB_TAPER_KERNEL.max()   # normalise peak to 1.0

# Pre-computed constants for the detection hot loop (computed once at import time)
_DETECT_WINDOW = np.hanning(CH_DETECT_SIZE).astype(np.float32)
_SQ_FREQ       = np.fft.fftshift(np.fft.fftfreq(CH_DETECT_SIZE, 1.0 / CH_SAMPLE_RATE))
_LO_MASK       = (_SQ_FREQ >= -_SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <= -_SQ_TONE_HZ + _SQ_NTOL_HZ)
_HI_MASK       = (_SQ_FREQ >=  _SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <=  _SQ_TONE_HZ + _SQ_NTOL_HZ)


def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
    """Process one IQ chunk: wideband FFT blanker, ring buffer, channelised
    detection, and overlap-FFT waterfall.

    Parameters
    ----------
    iq_samples : array_like
        Complex IQ at ``self.sample_rate`` (typically 48 kHz).
    timestamp_int : int
        Integer part of the first-sample time.  For live radio this is normally
        Unix seconds when the source clock is valid; very small values fall back
        to wall-clock in the waterfall path (see implementation).
    timestamp_frac : int
        Sub-second fraction in **picoseconds** so that
        ``t = timestamp_int + timestamp_frac * 1e-12``.  WAV replay passes a
        file-relative timeline (seconds since start) via ``timestamp_int``/``frac``.

    Processing order
    ----------------
    0. **Wideband FFT noise blanker** — NB_FFT_SIZE blocks; Hann-tapered soft
       blanking of broadband impulses; updates ``_nb_env`` / ``_nb_floor`` for display.
    1. **Ring buffer** — append cleaned IQ to ``_iq_ring``; advance ``_iq_abs_sample``.
    2. **Time-domain magnitude** — rolling buffer for the scope trace.
    3. **Channeliser** — 48 × 12 kHz channels; optional TX settle gate discards
       partial ``_ch_buf`` accumulation.
    4. **Per-channel detection** — hop loop: squared FFT, 25th-percentile baseline,
       coincidence / sustained gates, optional ``extract_and_decode`` threads.
       SNR write index anchored to VITA timestamp (live) or WAV cursor (replay).
    5. **Waterfall** — overlap Hanning FFT on ``_sbuf``; ``_sbuf_t0`` from packet
       time.  WAV mode updates ``_wav_time_cursor`` from ``_sbuf_t0`` so the next
       chunk stays aligned with the overlap buffer.

    Both the SNR heatmap and the spectrogram use the packet timestamp so their
    time axes stay aligned.  For live radio the timestamp comes from the VITA-49
    sample-clock field; for WAV replay it uses the WAV cursor accumulated in
    _sbuf_t0.  time.time() is only a fallback for pre-GPS-lock hardware.
    """
    # Detect dual-channel input: (N, 2) → H and V; (N,) or (N, 1) → H only.
    _raw_in = np.asarray(iq_samples, dtype=np.complex64)
    dual_pol = getattr(self, 'dual_pol', False) and _raw_in.ndim == 2 and _raw_in.shape[1] >= 2
    raw   = _raw_in[:, 0] if _raw_in.ndim == 2 else _raw_in.ravel()
    raw_v = _raw_in[:, 1].copy() if dual_pol else None

    # Pre-compute packet time from embedded timestamp so it is available for
    # the detection section (which runs before the waterfall section where
    # _pkt_time is normally computed).
    _pkt_time_early = float(timestamp_int) + float(timestamp_frac) * 1e-12
    _is_wav_early   = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav_early and timestamp_int < 1_000_000_000:
        # Hardware clock not GPS-locked yet; fall back to wall clock.
        _pkt_time_early = time.time()

    # ── 0. Wideband FFT noise blanker (Linrad-style) ─────────────────────────
    # Divide input into NB_FFT_SIZE blocks.  For each block, FFT and compute
    # per-bin power normalised by a per-bin running average.  If the fraction
    # of bins whose normalised power exceeds nb_factor² is greater than
    # NB_BROADBAND_FRAC, the block is a broadband impulse and is blanked.
    #
    # A narrowband signal (meteor burst ≈ 2 kHz, beacon ≈ 500 Hz) elevates
    # only a few of the NB_FFT_SIZE bins — well below NB_BROADBAND_FRAC.
    # A broadband impulse (lightning, power-line click) elevates nearly all
    # bins simultaneously — well above NB_BROADBAND_FRAC.
    # The per-bin average is only updated from non-blanked blocks, so impulses
    # cannot inflate the threshold.
    nb_factor        = float(getattr(self, 'nb_factor', NB_FACTOR))
    metric_threshold = nb_factor * nb_factor   # amplitude ratio → power ratio
    mag              = np.abs(raw).astype(np.float32)

    # Initialise per-bin averages from the first available block.
    if self._nb_spec_avg is None:
        first = raw[:NB_FFT_SIZE] if len(raw) >= NB_FFT_SIZE else raw
        X0 = np.fft.fft(first, n=NB_FFT_SIZE)
        self._nb_spec_avg = np.maximum(np.abs(X0).astype(np.float64) ** 2, 1e-30)
        rms = float(np.sqrt(np.mean(self._nb_spec_avg) / NB_FFT_SIZE))
        self._nb_env   = max(rms, 1e-9)
        self._nb_floor = self._nb_env

    n_blocks = len(raw) // NB_FFT_SIZE
    alpha    = 1.0 - np.exp(-NB_FFT_SIZE / (NB_SPEC_AVG_TC * self.sample_rate))

    # Per-sample blanking mask: 1.0 = blank, 0.0 = keep.
    blank_mask = np.zeros(len(raw), dtype=np.float32)
    self._nb_total_count += len(raw)

    for i in range(n_blocks):
        s = i * NB_FFT_SIZE
        e = s + NB_FFT_SIZE
        X = np.fft.fft(raw[s:e])
        P = np.abs(X).astype(np.float64) ** 2

        # Normalise each bin by its running average and count hot bins.
        n_hot       = int(np.sum(P > metric_threshold * self._nb_spec_avg))
        is_broadband = n_hot > NB_BROADBAND_FRAC * NB_FFT_SIZE

        if is_broadband:
            blank_mask[s:e] = 1.0
            self._nb_blanked_count += NB_FFT_SIZE
            _bl = getattr(self, '_nb_blank_log', None)
            if _bl is not None:
                _bl.append(getattr(self, '_nb_log_t0', 0.0) + s / self.sample_rate)
        else:
            # Update per-bin averages only from clean blocks.
            self._nb_spec_avg = (1.0 - alpha) * self._nb_spec_avg + alpha * P

        # Keep last block for the spectrum display.
        self._nb_last_P   = P
        self._nb_last_hot = n_hot

    # Partial block at end of chunk passes through unchanged (no blanking decision).

    # Soft-blank: Hann-taper the edges of each blanked region so the abrupt
    # transition to zero does not create rectangular-hole sinc sidelobes.
    if np.any(blank_mask):
        soft_blank = np.minimum(
            np.convolve(blank_mask, _NB_TAPER_KERNEL, mode='same'),
            1.0,
        )
        cleaned   = (raw   * (1.0 - soft_blank)).astype(np.complex64)
        cleaned_v = (raw_v * (1.0 - soft_blank)).astype(np.complex64) if dual_pol else None
    else:
        cleaned   = raw.copy()
        cleaned_v = raw_v.copy() if dual_pol else None

    # Update display estimates from per-bin averages.
    # _nb_floor: noise floor amplitude implied by the median per-bin power.
    # _nb_env:   running mean of non-blanked sample amplitudes (signal level).
    rms = float(np.sqrt(np.mean(self._nb_spec_avg) / NB_FFT_SIZE))
    self._nb_floor = max(rms, 1e-9)
    not_blanked = blank_mask == 0.0
    if np.any(not_blanked):
        n_ok      = int(np.sum(not_blanked))
        alpha_env = 1.0 - np.exp(-n_ok / (2.0 * self.sample_rate))
        chunk_env = float(np.mean(mag[not_blanked]))
        self._nb_env = (1.0 - alpha_env) * self._nb_env + alpha_env * chunk_env


    # ── 1. Write cleaned IQ into the circular ring buffer ────────────────────
    # Ring is (ring_n, 2): col 0 = H, col 1 = V.  Single-pol writes zeros to col 1.
    chunk_len = len(cleaned)
    ring_size = self._iq_ring.shape[0]
    pos = self._iq_ring_pos
    _v_col = cleaned_v if dual_pol else np.zeros(chunk_len, dtype=np.complex64)
    _ring_chunk = np.stack([cleaned, _v_col], axis=1)   # (N, 2)
    if pos + chunk_len <= ring_size:
        self._iq_ring[pos:pos + chunk_len] = _ring_chunk
    else:
        first_n = ring_size - pos
        self._iq_ring[pos:]                 = _ring_chunk[:first_n]
        self._iq_ring[:chunk_len - first_n] = _ring_chunk[first_n:]
    self._iq_ring_pos    = (pos + chunk_len) % ring_size
    self._iq_abs_sample += chunk_len

    # ── 1a. Update time-domain magnitude display buffer ───────────────────────
    _td_buf = self._td_mag_buf
    _td_n   = len(_td_buf)
    _td_mag = np.abs(cleaned).astype(np.float32)
    _td_in  = len(_td_mag)
    _td_pos = self._td_mag_pos
    if _td_pos + _td_in <= _td_n:
        _td_buf[_td_pos:_td_pos + _td_in] = _td_mag
    else:
        _first = _td_n - _td_pos
        _td_buf[_td_pos:]      = _td_mag[:_first]
        _td_buf[:_td_in-_first] = _td_mag[_first:]
    self._td_mag_pos = (_td_pos + _td_in) % _td_n

    # ── 1b. V-channel time-domain magnitude + NB spectrum (dual-pol display) ────
    if dual_pol and cleaned_v is not None:
        _td_buf_v = self._td_mag_buf_v
        _td_n_v   = len(_td_buf_v)
        _td_mag_v = np.abs(cleaned_v).astype(np.float32)
        _td_in_v  = len(_td_mag_v)
        _td_pos_v = self._td_mag_pos_v
        if _td_pos_v + _td_in_v <= _td_n_v:
            _td_buf_v[_td_pos_v:_td_pos_v + _td_in_v] = _td_mag_v
        else:
            _first_v = _td_n_v - _td_pos_v
            _td_buf_v[_td_pos_v:]         = _td_mag_v[:_first_v]
            _td_buf_v[:_td_in_v - _first_v] = _td_mag_v[_first_v:]
        self._td_mag_pos_v = (_td_pos_v + _td_in_v) % _td_n_v

        # Per-bin V-channel spectrum tracking (floor + last block, for display).
        # Blanking decisions remain H-only; V updates only on non-blanked H blocks.
        if self._nb_spec_avg_v is None:
            first_v = raw_v[:NB_FFT_SIZE] if len(raw_v) >= NB_FFT_SIZE else raw_v
            X0_v = np.fft.fft(first_v, n=NB_FFT_SIZE)
            self._nb_spec_avg_v = np.maximum(np.abs(X0_v).astype(np.float64) ** 2, 1e-30)
        for _i in range(n_blocks):
            _sv = _i * NB_FFT_SIZE
            _ev = _sv + NB_FFT_SIZE
            Xv = np.fft.fft(raw_v[_sv:_ev])
            Pv = np.abs(Xv).astype(np.float64) ** 2
            self._nb_last_P_v = Pv
            if not blank_mask[_sv]:   # update floor only from clean H blocks
                self._nb_spec_avg_v = (1.0 - alpha) * self._nb_spec_avg_v + alpha * Pv

    # ── 2. Channelise into 48 × 12 kHz channels (H, and V if dual-pol) ─────────
    ch_out = apply_channelizer(
        cleaned, self._ch_state,
        lp_taps=self._ch_taps,
    )   # (N_CHANNELS, chunk_len // 4)
    ch_out_v = (apply_channelizer(cleaned_v, self._ch_state_v, lp_taps=self._ch_taps)
                if dual_pol else None)

    # ── 3. Per-channel detection ──────────────────────────────────────────────
    # Post-TX settling gate — discard channelizer output and skip detection for
    # ~500 ms after TX ends so AGC transients don't cause false broadband triggers.
    _settle = getattr(self, '_tx_settle_remaining', 0)
    # Save leftover sample count before adding new data — used below to anchor
    # the WAV-mode SNR write index to the time of _ch_buf[0], not the end of
    # the previous call.  Without this, the write index jumps by one extra hop
    # at the start and end of replay, leaving dark vertical gaps in the heatmap.
    _ch_buf_carry = self._ch_buf_end
    if _settle > 0:
        self._tx_settle_remaining = max(0, _settle - len(iq_samples))
        self._ch_buf_end   = 0   # discard any partial accumulation
        self._ch_buf_v_end = 0
        _ch_buf_carry = 0
        # Fall through to waterfall FFT — NB still learns the noise floor.
    else:
        # Accumulate into pre-allocated slide-left buffer (no per-call allocation)
        new_n = ch_out.shape[1]
        cap = self._ch_buf.shape[1]
        if self._ch_buf_end + new_n > cap:
            # Buffer overflow: discard oldest samples to make room
            keep = cap - new_n
            if keep > 0:
                self._ch_buf[:, :keep] = self._ch_buf[:, cap - keep:cap]
            self._ch_buf_end = max(keep, 0)
        self._ch_buf[:, self._ch_buf_end:self._ch_buf_end + new_n] = ch_out
        self._ch_buf_end += new_n
        if dual_pol:
            if self._ch_buf_v_end + new_n > cap:
                keep_v = cap - new_n
                if keep_v > 0:
                    self._ch_buf_v[:, :keep_v] = self._ch_buf_v[:, cap - keep_v:cap]
                self._ch_buf_v_end = max(keep_v, 0)
            self._ch_buf_v[:, self._ch_buf_v_end:self._ch_buf_v_end + new_n] = ch_out_v
            self._ch_buf_v_end += new_n

    # Anchor the SNR write index to wall clock once per call, then increment
    # by 1 per hop inside the loop.  Calling time.time() inside the loop gives
    # the same value for consecutive hops (they execute in microseconds, not
    # the 21.3 ms of audio each hop represents), so all hops in one call would
    # overwrite the same row — producing the "every other frame missing" effect.
    if self.source_mode == "wav":
        # _wav_time_cursor is the end of the previous call's buffer.  Subtract
        # the carry-over samples already in _ch_buf so the anchor points to
        # _ch_buf[0], placing each hop at its correct sample-accurate time.
        _loop_wall = float(getattr(self, "_wav_time_cursor", 0.0)) \
                     - _ch_buf_carry / CH_SAMPLE_RATE
    else:
        # Live radio: use the VITA-49 sample-clock timestamp so the heatmap
        # is anchored to the same GPS clock as the spectrogram.  Subtract the
        # carry-over channelised samples so the anchor is at _ch_buf[0].
        _loop_wall = _pkt_time_early - _ch_buf_carry / CH_SAMPLE_RATE
    _snr_write_idx = round(
        (_loop_wall % self.history_secs) * CH_SAMPLE_RATE / _CH_DETECT_HOP
    )

    while self._ch_buf_end >= CH_DETECT_SIZE and not getattr(self, '_tx_settle_remaining', 0):
        ch_block   = self._ch_buf[:, :CH_DETECT_SIZE]            # (48, 512) H channel
        ch_block_v = (self._ch_buf_v[:, :CH_DETECT_SIZE]
                      if dual_pol and self._ch_buf_v_end >= CH_DETECT_SIZE
                      else None)

        sq         = ch_block ** 2                               # complex squaring H
        X_sq       = np.fft.fft(sq * _DETECT_WINDOW[np.newaxis, :], axis=1)
        if ch_block_v is not None:
            sq_v   = ch_block_v ** 2
            X_sq_v = np.fft.fft(sq_v * _DETECT_WINDOW[np.newaxis, :], axis=1)

        current_wall_time = _loop_wall
        t_in_window = current_wall_time % self.history_secs

        # ── Two-window tone detection in the squared domain ───────────────────
        # MSK144 tones at fc_offset ± 500 Hz → after squaring → ±1000 Hz.
        # _SQ_FREQ, _LO_MASK, _HI_MASK are pre-computed module-level constants.
        power_lin = np.abs(X_sq) / CH_DETECT_SIZE               # (48, 512) linear
        plin_all  = np.fft.fftshift(power_lin, axes=1)          # (48, 512)
        lo_peak   = np.max(plin_all[:, _LO_MASK], axis=1)       # (48,) linear
        hi_peak   = np.max(plin_all[:, _HI_MASK], axis=1)       # (48,) linear
        raw_lin   = (lo_peak + hi_peak) / 2.0                   # (48,) linear

        # ── H-channel rolling 25th-percentile baseline ───────────────────────
        self._metric_hist_buf[self._metric_hist_idx] = raw_lin
        self._metric_hist_idx = (self._metric_hist_idx + 1) % _METRIC_HIST_DEPTH
        if self._metric_hist_cnt < _METRIC_HIST_DEPTH:
            self._metric_hist_cnt += 1
        hist  = (self._metric_hist_buf[:self._metric_hist_cnt]
                 if self._metric_hist_cnt < _METRIC_HIST_DEPTH
                 else self._metric_hist_buf)
        self._pct25_ctr = getattr(self, '_pct25_ctr', 0) + 1
        if self._pct25_ctr % 10 == 0 or not hasattr(self, '_pct25'):
            hist_s = hist[::_METRIC_HIST_STRIDE]
            k      = max(0, int(len(hist_s) * 0.25))
            self._pct25 = np.maximum(np.partition(hist_s, k, axis=0)[k], 1e-30)
        pct25 = self._pct25

        pair_metric_h = (
            10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30))
        ).astype(np.float32)   # (48,) dB above baseline

        # ── V-channel rolling 25th-percentile baseline (independent noise floor) ─
        if ch_block_v is not None:
            plin_v_all = np.fft.fftshift(np.abs(X_sq_v) / CH_DETECT_SIZE, axes=1)
            raw_lin_v  = (np.max(plin_v_all[:, _LO_MASK], axis=1) +
                          np.max(plin_v_all[:, _HI_MASK], axis=1)) / 2.0
            self._metric_hist_buf_v[self._metric_hist_idx_v] = raw_lin_v
            self._metric_hist_idx_v = (self._metric_hist_idx_v + 1) % _METRIC_HIST_DEPTH
            if self._metric_hist_cnt_v < _METRIC_HIST_DEPTH:
                self._metric_hist_cnt_v += 1
            hist_v = (self._metric_hist_buf_v[:self._metric_hist_cnt_v]
                      if self._metric_hist_cnt_v < _METRIC_HIST_DEPTH
                      else self._metric_hist_buf_v)
            self._pct25_ctr_v = getattr(self, '_pct25_ctr_v', 0) + 1
            if self._pct25_ctr_v % 10 == 0 or not hasattr(self, '_pct25_v'):
                hist_vs = hist_v[::_METRIC_HIST_STRIDE]
                kv      = max(0, int(len(hist_vs) * 0.25))
                self._pct25_v = np.maximum(np.partition(hist_vs, kv, axis=0)[kv], 1e-30)
            pair_metric_v = (
                10.0 * np.log10(np.maximum(raw_lin_v / self._pct25_v, 1e-30))
            ).astype(np.float32)
        else:
            pair_metric_v = pair_metric_h   # single-pol: V mirrors H (unused)

        # Trigger on either channel; display the max for the combined heatmap
        pair_metric = np.maximum(pair_metric_h, pair_metric_v)

        # Time-indexed SNR write — first hop anchored to wall clock above;
        # subsequent hops in this call increment by 1 (each hop = 21.3 ms).
        ch_boundary = int(current_wall_time / self.history_secs)
        if ch_boundary != self._ch_snr_boundary:
            self._ch_snr_history_h[:] = -999.0
            self._ch_snr_history_v[:] = -999.0
            self._ch_snr_boundary     = ch_boundary
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                marker_list.clear()
            # Do NOT reset _ch_above_thresh here — the coincidence gate must
            # survive the display window boundary so a signal that starts just
            # before the boundary still triggers a jt9 launch after it.
            _snr_write_idx = 0
        self._ch_snr_write_idx = min(max(_snr_write_idx, 0), N_SNR_HIST - 1)
        if 0 <= self._ch_snr_write_idx < N_SNR_HIST:
            self._ch_snr_history_h[self._ch_snr_write_idx, :] = pair_metric_h
            self._ch_snr_history_v[self._ch_snr_write_idx, :] = pair_metric_v
        _snr_write_idx += 1

        # Tick down cooldowns
        self._detect_cooldowns = {
            k: v - 1 for k, v in self._detect_cooldowns.items() if v > 1
        }

        cooldown_hops = max(
            1, int(DETECT_MERGE_GAP_S * CH_SAMPLE_RATE / _CH_DETECT_HOP)
        )

        _triggered = np.where(pair_metric > DETECT_THRESH_DB)[0]
        # Process strongest channel first so the proximity guard blocks weaker
        # adjacent channels rather than the correct center channel.
        if len(_triggered) > 1:
            _triggered = _triggered[np.argsort(pair_metric[_triggered])[::-1]]

        # ── Cluster the triggered channels ────────────────────────────────────
        # Split triggered channels into contiguous clusters (gap > 3 ch = new cluster).
        # Broadband noise is one wide contiguous cluster; two real signals at different
        # frequencies form two separate clusters with a gap between them.
        # For each cluster: suppress if too wide (> _COINCIDENCE_SPAN_CH) or if the
        # global cluster count hits _COINCIDENCE_MAX_CLUSTERS.
        # Only the strongest channel in each surviving cluster launches jt9 — this
        # prevents adjacent sidelobe channels from spawning redundant decode threads
        # while still allowing independent signals in separate clusters to both launch.
        _sorted_ch = _triggered.astype(int)   # already sorted by metric desc above
        _sorted_idx = np.sort(_triggered.astype(int))   # ascending index for clustering
        _clusters = []   # list of arrays of channel indices (ascending)
        if len(_sorted_idx):
            _cur = [_sorted_idx[0]]
            for _ci in _sorted_idx[1:]:
                if _ci - _cur[-1] <= 3:
                    _cur.append(_ci)
                else:
                    _clusters.append(np.array(_cur))
                    _cur = [_ci]
            _clusters.append(np.array(_cur))

        # Global cluster count gate: if a large number of SEPARATE narrow clusters
        # fire simultaneously it is broadband noise, not multiple legitimate pings.
        # Each real MSK144 ping spreads into ~3 adjacent channels (squared-domain
        # ±1000 Hz tones straddle the 1 kHz channel boundary), forming one cluster.
        # Three simultaneous pings at 5 kHz spacing produce 3 clusters of ~3 channels
        # each (9 total channels) — the old raw channel-count gate (≥8) incorrectly
        # fired on this scenario.  Counting clusters (≥6) is far more robust.
        # Only fresh clusters (whose best channel is not already in cooldown) are
        # counted — sidelobe remnants of a prior real detection must not trigger the gate.
        _fresh_clusters = [
            _cl for _cl in _clusters
            if int(_cl[np.argmax(pair_metric[_cl])]) not in self._detect_cooldowns
        ]
        _too_many = len(_fresh_clusters) >= _COINCIDENCE_MAX_CLUSTERS
        if _too_many:
            for _ck in _triggered:
                self._detect_cooldowns[_ck] = cooldown_hops
            self._ch_buf[:, :len(self._ch_buf[0])] = 0
            self._ch_buf_end = 0
            self._ch_buf_v_end = 0   # keep H and V in sync
            break

        # Per-cluster spread gate and best-channel selection.
        # _launch_channels: list of (launch_ch, cluster_channels) tuples.
        _launch_channels = []
        for _cl in _clusters:
            _cl_span = int(_cl[-1]) - int(_cl[0])
            if _cl_span > _COINCIDENCE_SPAN_CH:
                # Wide cluster = broadband noise; suppress all its channels.
                for _ck in _cl:
                    self._detect_cooldowns[_ck] = cooldown_hops
                continue
            # Pick the channel with the highest pair_metric in this cluster.
            _best = int(_cl[np.argmax(pair_metric[_cl])])
            if _best not in self._detect_cooldowns:
                _launch_channels.append((_best, _cl))
                # Launch happened: suppress only the launched channel itself.
                # Suppressing neighbours risks blocking a real signal whose center
                # is one channel away (e.g. ch9 signal blocked by ch8 sidelobe launch).
                # Hop-to-hop re-triggers from the same real channel are suppressed
                # because _best stays in cooldown; the "best-cooled" branch then
                # refreshes its timer without adding fresh neighbours.
                self._detect_cooldowns[_best] = cooldown_hops
            else:
                # Best channel is already in cooldown (decoding in progress or just
                # launched).  Only refresh channels that are already suppressed —
                # do NOT add fresh channels to cooldown.  A fresh channel in this
                # cluster may be the real signal center that will become the cluster
                # best on a subsequent hop once the sidelobe-elevated cooled channels
                # drop below threshold.
                for _ck in _cl:
                    if _ck in self._detect_cooldowns:
                        self._detect_cooldowns[_ck] = cooldown_hops  # refresh timer

        # Sustained-signal gate — update consecutive-hop counters.
        # Channels above threshold this hop increment; all others reset to 0.
        # A channel that stays above for >_SUSTAINED_HOPS (~2 s) is interference
        # (FT8, beacon, spur) not a meteor ping — apply a long lockout.
        _ch_above = getattr(self, '_ch_above_thresh', {})
        for _ck in range(N_CHANNELS):
            if _ck in set(_triggered.tolist()):
                _ch_above[_ck] = _ch_above.get(_ck, 0) + 1
                if _ch_above[_ck] == _SUSTAINED_HOPS:
                    print(f"[detect] ch {_ck} sustained signal — locking out for 30 s", flush=True)
                    self._detect_cooldowns[_ck] = _SUSTAINED_LOCKOUT
            else:
                _ch_above.pop(_ck, None)
        self._ch_above_thresh = _ch_above

        for ch_k, _cl_all in _launch_channels:
            _nyquist_ch = N_CHANNELS // 2  # ch 24 = ±24 kHz Nyquist
            if abs(int(ch_k) - _nyquist_ch) <= _EDGE_CH_SKIP:
                continue

            # Estimate fc from peak bin locations in each window (linear domain)
            lo_freq   = float(_SQ_FREQ[_LO_MASK][np.argmax(plin_all[ch_k][_LO_MASK])])
            hi_freq   = float(_SQ_FREQ[_HI_MASK][np.argmax(plin_all[ch_k][_HI_MASK])])
            fc_offset  = (lo_freq + hi_freq) / 4.0   # squaring doubled freqs
            # ch_k 25-47 are negative-frequency channels (-23 to -1 kHz)
            ch_signed  = int(ch_k) if int(ch_k) <= N_CHANNELS // 2 else int(ch_k) - N_CHANNELS
            # Carrier position in Hz from IQ DC (pan centre).
            # channel_offset_hz = (calling_freq - pan_center) + 1500 from the channelizer state.
            fc_hz      = ch_signed * CHANNEL_SPACING_HZ + self._ch_state.channel_offset_hz + fc_offset

            abs_snap      = self._iq_abs_sample
            ring_state_fn = lambda: (self._iq_ring_pos, self._iq_abs_sample)
            output_dir    = str(Path(__file__).parent.parent / 'MSK144' / 'detections')
            detect_ts     = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H:%M:%S.%f')[:21]

            decode_queue = getattr(self, '_decode_queue', None)
            marker_id    = getattr(self, '_jt9_marker_next_id', 0)
            if hasattr(self, '_jt9_marker_next_id'):
                self._jt9_marker_next_id += 1
            ring_gen     = getattr(self, '_iq_ring_gen', 0)
            ring_gen_fn  = lambda: getattr(self, '_iq_ring_gen', 0)

            # Prune finished threads and enforce a hard cap before spawning.
            # Without this, false detections accumulate unbounded threads that
            # saturate all CPU cores with concurrent numpy/scipy work.
            jt9_list = getattr(self, '_jt9_threads', None)
            if jt9_list is not None:
                # Remove dead threads in-place (O(n) but list stays small)
                jt9_list[:] = [t for t in jt9_list if t.is_alive()]
                if len(jt9_list) >= 12:   # 3× semaphore width; beyond this, drop
                    self._detect_cooldowns[ch_k] = cooldown_hops
                    continue
                # Note: proximity guard removed — the ±1 neighbour cooldown on launch
                # handles hop-to-hop re-triggers of the same channel without blocking
                # independent signals in adjacent channels.

            if not getattr(self, '_analysis_no_decode', False):
                t = threading.Thread(
                    target=extract_and_decode,
                    args=(self._iq_ring, ring_state_fn,
                          abs_snap, self.sample_rate, fc_hz, output_dir,
                          t_in_window, decode_queue, marker_id,
                          ring_gen, ring_gen_fn,
                          getattr(self, 'display_center_freq_mhz', self.center_freq_mhz),
                          detect_ts, None, dual_pol),
                    daemon=True,
                )
                t._fc_hz = fc_hz
                t.start()
                if jt9_list is not None:
                    jt9_list.append(t)
            # Record marker for heatmap overlay in radio-tuning coordinates (fc_hz − 1500 Hz)
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                display_khz = (fc_hz - 1500.0) / 1000.0
                marker_list.append({
                    'id':        marker_id,
                    't':         t_in_window,
                    'freq_khz':  display_khz,
                    'boundary':  self._ch_snr_boundary,
                    'decoded':   False,
                    'outcome':   'launched',
                    'message':   None,
                    'theta_deg': None,   # filled by extract_and_decode on dual-pol decode
                })
            self._detect_cooldowns[ch_k] = cooldown_hops

        # Slide channeliser buffers left by one hop (in-place, no allocation)
        remaining = max(0, self._ch_buf_end - _CH_DETECT_HOP)
        if remaining > 0:
            self._ch_buf[:, :remaining] = self._ch_buf[:, _CH_DETECT_HOP:self._ch_buf_end]
        self._ch_buf_end = remaining
        if dual_pol:
            remaining_v = max(0, self._ch_buf_v_end - _CH_DETECT_HOP)
            if remaining_v > 0:
                self._ch_buf_v[:, :remaining_v] = self._ch_buf_v[:, _CH_DETECT_HOP:self._ch_buf_v_end]
            self._ch_buf_v_end = remaining_v

    # ── 4. Overlap-FFT for waterfall display (blanked signal) ─────────────────
    # Anchor _sbuf_t0 to the VITA packet timestamp so each FFT block is placed
    # at its exact sample-accurate receive time, not at time.time() during
    # processing.  Using time.time() causes dark vertical stripes: when the
    # runtime processes a batch of packets quickly the clock barely advances,
    # mapping many blocks to the same row; the clock then jumps, skipping rows.
    #
    _sbuf_end_before = self._sbuf_end   # samples already buffered before this packet

    new_n = len(cleaned)
    self._sbuf[self._sbuf_end:self._sbuf_end + new_n] = cleaned
    self._sbuf_end += new_n
    if dual_pol and cleaned_v is not None:
        self._sbuf_v[self._sbuf_v_end:self._sbuf_v_end + new_n] = cleaned_v
        self._sbuf_v_end += new_n

    # All sources must deliver timestamp_frac as picoseconds (adapter responsibility).
    _pkt_time = float(timestamp_int) + float(timestamp_frac) * 1e-12
    # WAV files use a 0-based cursor (seconds from file start) — their timestamps are
    # intentionally small and must NOT be replaced by wall-clock time.
    # For live radio sources, a small timestamp_int (< year 2001 Unix epoch) means the
    # hardware clock is not yet locked to UTC (power-on counter, GPS not synced, etc.);
    # fall back to wall-clock so the spectrogram is placed correctly instead of cycling
    # in a 0–1 s window.
    _is_wav = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav and timestamp_int < 1_000_000_000:
        _pkt_time = time.time()
    # Time of _sbuf[0] = packet's first-sample time minus already-buffered samples
    self._sbuf_t0 = _pkt_time - _sbuf_end_before / self.sample_rate

    fft_window  = self._fft_window   # pre-computed float32 Hanning window
    window_gain = self._window_gain  # pre-computed scalar
    full_scale  = 1.0

    while self._sbuf_end >= self.fft_size:
        block = self._sbuf[:self.fft_size]

        X         = np.fft.fftshift(np.fft.fft(block * fft_window))
        magnitude = np.abs(X) / (self.fft_size * window_gain)
        power_db  = 20.0 * np.log10(magnitude / full_scale + 1e-12)

        # Sample-accurate time for this block: _sbuf_t0 is the time of _sbuf[0],
        # which is always the first sample of the current block since the buffer
        # slides by exactly one hop after each block is processed below.
        current_wall_time = self._sbuf_t0
        time_in_window = current_wall_time % self.history_secs

        # ── Accumulated spectrogram ───────────────────────────────────────
        # Each block is written at the row corresponding to its receive time so
        # the column position matches wall-clock time modulo the 15-second window,
        # matching the WSJTX Fast Graph convention.
        spec_boundary = int(current_wall_time / self.history_secs)
        if spec_boundary != self.spec_boundary:
            self.spectrogram_data    = self.spec_staging.copy()
            self.spec_staging_filled = True
            self.accumulated_noise_floor = np.percentile(self.spec_staging, 10, axis=0)
            self.spec_staging        = np.full((self.max_history, self.fft_size), -130.0)
            self.spec_boundary       = spec_boundary
            self.next_boundary = current_wall_time + (
                self.history_secs - (current_wall_time % self.history_secs)
            )
            self._prev_staging_idx = None   # reset gap-fill cursor at window boundary
            if dual_pol:
                self.spectrogram_data_v = self.spec_staging_v.copy()
                self.spec_staging_v     = np.full((self.max_history, self.fft_size), -130.0)

        # Compute the row index for this block once — accumulated and realtime
        # spectrograms share the same time_in_window so the index is identical.
        # Use round() not int(): FP reconstruction of the VITA timestamp is
        # systematically a few tens of picoseconds low, giving values like
        # 46.99999999 that int() would floor to 46 instead of 47.
        row_idx = min(max(round(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        # ── Gap fill: repeat last spectrum into any skipped rows ──────────
        # Skipped rows arise from batched packet delivery — the VITA timestamps
        # correctly place each block, so gaps are real.  Filling prevents dark
        # vertical stripes.  Cursors are reset at each 15 s boundary above /
        # below so gap-fill never bridges across windows.
        _prev_staging = self._prev_staging_idx
        if _prev_staging is not None and row_idx > _prev_staging + 1:
            self.spec_staging[_prev_staging + 1:row_idx] = power_db
        self._prev_staging_idx = row_idx
        self.spec_staging[row_idx] = power_db

        # ── Real-time spectrogram ─────────────────────────────────────────
        realtime_boundary = int(current_wall_time / self.history_secs)
        if realtime_boundary != self._realtime_boundary:
            # Allocate new arrays before replacing the references so the display
            # thread reads either the complete old array or the complete new one,
            # never a partially-reset one.
            self.realtime_data      = np.full((self.max_history, self.fft_size), -130.0)
            self._realtime_boundary = realtime_boundary
            self._prev_rt_idx       = None   # reset gap-fill cursor
            if dual_pol:
                self.realtime_data_v = np.full((self.max_history, self.fft_size), -130.0)

        _prev_rt = self._prev_rt_idx
        if _prev_rt is not None and row_idx > _prev_rt + 1:
            self.realtime_data[_prev_rt + 1:row_idx] = power_db
        self._prev_rt_idx = row_idx
        self.realtime_data[row_idx] = power_db
        self.realtime_filled = True
        self._realtime_dirty = True

        # ── V-channel waterfall (dual-pol only) ───────────────────────────
        if dual_pol and self._sbuf_v_end >= self.fft_size:
            block_v    = self._sbuf_v[:self.fft_size]
            X_v        = np.fft.fftshift(np.fft.fft(block_v * fft_window))
            power_db_v = 20.0 * np.log10(
                np.abs(X_v) / (self.fft_size * window_gain) / full_scale + 1e-12
            )
            if _prev_staging is not None and row_idx > _prev_staging + 1:
                self.spec_staging_v[_prev_staging + 1:row_idx] = power_db_v
            self.spec_staging_v[row_idx] = power_db_v
            if _prev_rt is not None and row_idx > _prev_rt + 1:
                self.realtime_data_v[_prev_rt + 1:row_idx] = power_db_v
            self.realtime_data_v[row_idx] = power_db_v

        # Slide sample buffer left by one hop (in-place copy, no allocation)
        hop = self.fft_size // 2
        remaining = self._sbuf_end - hop
        if remaining > 0:
            self._sbuf[:remaining] = self._sbuf[hop:self._sbuf_end]
        self._sbuf_end = max(0, remaining)
        if dual_pol and self._sbuf_v_end >= hop:
            remaining_v = self._sbuf_v_end - hop
            if remaining_v > 0:
                self._sbuf_v[:remaining_v] = self._sbuf_v[hop:self._sbuf_v_end]
            self._sbuf_v_end = remaining_v
        # Advance the time anchor by exactly one hop so the next block's
        # current_wall_time is sample-accurate without calling time.time().
        self._sbuf_t0 += hop / self.sample_rate

    # Keep _wav_time_cursor in sync for the WAV source step (it uses this to
    # timestamp the next packet it will send into process_iq_data).
    if self.source_mode == "wav":
        self._wav_time_cursor = self._sbuf_t0 + self._sbuf_end / self.sample_rate

