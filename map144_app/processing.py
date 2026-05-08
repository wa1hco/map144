# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; wit hout even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""FFT and channelised MSK144 detection pipeline for incoming IQ sample data.

Pipeline (in order of execution)
---------------------------------
0. **Two-stage noise blanker** — shared Hann-tapered soft blanking mask.

   **0a. Wideband FFT blanker (Linrad-style)** — input is divided into
   NB_FFT_SIZE blocks.  Each block is FFT'd; the per-bin power is normalised
   by a per-bin running average.  If more than NB_BROADBAND_FRAC of bins
   have normalised power above nb_factor² the block is a broadband impulse
   and is zeroed.  Catches impulses ≥ NB_FFT_SIZE samples (~5.3 ms).

   **0b. Time-domain amplitude blanker** — per-sample magnitude compared to
   nb_factor × _nb_floor.  Catches sub-block impulses (e.g. 2-sample ~42 µs
   spikes) that are diluted below the FFT hot-bin threshold.  Uses the noise
   floor from the previous call, so the impulse cannot inflate its own threshold.
   Both stages add to a shared blank_mask; the Hann taper is applied once.

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

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage import convolve1d as _nd_convolve1d

from .channelizer import (
    apply_channelizer,
    N_CHANNELS,
    CHANNEL_SPACING_HZ,
    CH_SAMPLE_RATE,
)
from .channel_plan import NYQUIST_EDGE_CHANNEL_SKIP
from .detection import extract_and_decode
from .msk144_spd import (
    _sync_correlate as _msk144_sync_correlate,
    _sync_correlate_batch as _msk144_sync_correlate_batch,
    NSPM as _SYNC_NSPM,
)

# Step 1 of sensitivity-improvement plan — coherent MSK144 sync-word
# correlator running in parallel with the squared-FFT detector.  Feature flag
# so the change is bisectable (flip to False to recover the legacy path).
ENABLE_SYNC_DETECT = True
SYNC_DETECT_THRESH_DB = 3.0    # dB above 25th-percentile sync-magnitude baseline

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
NB_WIDEBAND_AVGNUM = 200        # EMA length for wideband median floor (~1 s at 5.3 ms/block)
_NB_LOWLEVEL_ALPHA = 0.02       # EMA weight for clean-fraction tracker (matches Linrad)
_NB_TAPER_N        = 2          # Hann taper half-width in samples (2 → 42 µs at 48 kHz)
                                # A 3-sample impulse becomes 7 samples blanked (2.3×).
                                # Raise if blanking edges cause spectral artifacts.
NB_SHORT_IMPULSE_THRESH = 5    # FFT blocks where ≤ this many samples exceed the TD threshold
                                # are treated as short impulses; block-blanking is skipped and
                                # the sample-precise TD mask is used instead.
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
N_SNR_HIST = int(15 * CH_SAMPLE_RATE / _CH_DETECT_HOP)        # = 703 (exact integer hops)
# 703 hops × (256 / 12000) s/hop = 14.997 s (3 ms shy of 15 s, well below display
# resolution).  Choosing this clean integer ratio means write_idx = hop_count %
# N_SNR_HIST advances by exactly 1 per hop with no rounding noise — no gap-fill,
# no wipe-and-refill cycle needed.  The 15-s boundary is a display / TX-RX
# concept; internal DSP state (rolling baselines, AGC) is unaffected.

# Pre-computed Hann taper kernel for soft-blanking (avoids rectangular-hole sinc spread)
_NB_TAPER_KERNEL = np.hanning(2 * _NB_TAPER_N + 1).astype(np.float32)
_NB_TAPER_KERNEL /= _NB_TAPER_KERNEL.max()   # normalise peak to 1.0

# Pre-computed constants for the detection hot loop (computed once at import time)
_DETECT_WINDOW = np.hanning(CH_DETECT_SIZE).astype(np.float32)


def reset_detection_baseline(engine):
    """Clear the 25th-percentile history and sustained-signal counters.

    Call whenever an operator-controlled input shifts channel power by
    more than the baseline can track quickly (gain change, antenna
    swap, large K step).  Without this the old pct25 trails 30+ s
    behind the new level, every channel reports "above threshold",
    and the 2-s sustained-signal gate falsely locks all 48 channels
    out for 30 s.

    Clears per-channel hop counters, rolling history, cached
    percentile, and any in-flight cooldown lockouts.
    """
    for suffix in ('', '_v'):
        if hasattr(engine, f'_metric_hist_buf{suffix}'):
            getattr(engine, f'_metric_hist_buf{suffix}').fill(0.0)
        if hasattr(engine, f'_metric_hist_cnt{suffix}'):
            setattr(engine, f'_metric_hist_cnt{suffix}', 0)
        if hasattr(engine, f'_metric_hist_idx{suffix}'):
            setattr(engine, f'_metric_hist_idx{suffix}', 0)
        # Delete the cached percentile so the detection path re-computes
        # from the fresh history on the very next chunk.  Setting to None
        # instead would leave the attribute present but unusable: the
        # detection path only recomputes when ``not hasattr(self, '_pct25*')``
        # or every 10 chunks, and the intervening calls would divide by None.
        for attr in (f'_pct25{suffix}', f'_pct25_ctr{suffix}'):
            if hasattr(engine, attr):
                delattr(engine, attr)
    if hasattr(engine, '_ch_above_thresh'):
        engine._ch_above_thresh = {}
    if hasattr(engine, '_detect_cooldowns'):
        engine._detect_cooldowns = {}
# Use unshifted fftfreq so _LO_MASK/_HI_MASK index directly into the raw FFT
# output — no np.fft.fftshift (which calls np.roll) needed in the hot loop.
_SQ_FREQ       = np.fft.fftfreq(CH_DETECT_SIZE, 1.0 / CH_SAMPLE_RATE)
_LO_MASK       = (_SQ_FREQ >= -_SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <= -_SQ_TONE_HZ + _SQ_NTOL_HZ)
_HI_MASK       = (_SQ_FREQ >=  _SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <=  _SQ_TONE_HZ + _SQ_NTOL_HZ)


def _compute_sync_metric_db(sync_window,
                            hist_buf, hist_idx_attr, hist_cnt_attr,
                            pct25_attr, pct25_ctr_attr, engine) -> np.ndarray:
    """Per-channel coherent sync-correlation peak in dB above 25th-percentile.

    ``sync_window`` is shape (N_CHANNELS, NSPM) — the most recent NSPM samples
    at the head of _ch_buf.  The channelizer already places signal energy at
    DC in each channel's output (see channelizer.py docstring), which is the
    convention ``_sync_correlate`` expects, so no pre-mix is needed.  The peak
    magnitude per channel is normalised against a rolling 25th-percentile
    baseline, producing a dB metric directly comparable to the squared-FFT
    pair_metric.

    Uses the batched ``_sync_correlate_batch`` (one BLAS matmul over all
    channels) instead of a Python ``for ch in range(48)`` loop — the loop
    version was the largest single hot spot in py-spy profiling (~25 % of
    total CPU on a 6-core i5 during noise bursts).
    """
    sync_window_c64 = np.ascontiguousarray(sync_window, dtype=np.complex64)
    _, peaks, _ = _msk144_sync_correlate_batch(sync_window_c64)

    # Rolling 25th-percentile baseline of sync magnitudes (one row per hop).
    idx = getattr(engine, hist_idx_attr)
    cnt = getattr(engine, hist_cnt_attr)
    hist_buf[idx] = peaks
    setattr(engine, hist_idx_attr, (idx + 1) % _METRIC_HIST_DEPTH)
    if cnt < _METRIC_HIST_DEPTH:
        setattr(engine, hist_cnt_attr, cnt + 1)
        cnt += 1
    hist = hist_buf[:cnt] if cnt < _METRIC_HIST_DEPTH else hist_buf

    pct25_ctr = getattr(engine, pct25_ctr_attr, 0) + 1
    setattr(engine, pct25_ctr_attr, pct25_ctr)
    pct25 = getattr(engine, pct25_attr, None)
    if pct25_ctr % 10 == 0 or pct25 is None:
        hist_s = hist[::_METRIC_HIST_STRIDE]
        k      = max(0, int(len(hist_s) * 0.25))
        tmp    = hist_s.copy()
        tmp.partition(k, axis=0)
        pct25  = np.maximum(tmp[k], 1e-30)
        setattr(engine, pct25_attr, pct25)

    return (10.0 * np.log10(np.maximum(peaks / pct25, 1e-30))).astype(np.float32)


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
    #
    # **Sample-clock anchoring** (see project_sample_clock_anchoring memory):
    # Scrolling displays (waterfall, detection heatmaps) must index by a
    # sample-counter-derived time, never by wall-clock-at-processing time.
    # When the radio provides a valid VITA-49 sample-clock timestamp
    # (timestamp_int >= 1e9), we use it directly — perfectly monotonic.
    # When it doesn't (Flex TSI=0, RTL-SDR, Airspy without GPS lock), the
    # legacy fallback ``time.time()`` injects per-chunk wall-clock jitter
    # that lands on the heatmap row index → randomly-placed dark vertical
    # lines.  Instead, anchor once at the first chunk and derive each
    # subsequent chunk's time from the sample counter.  Result is a
    # monotonic per-sample timeline regardless of OS scheduling jitter.
    _pkt_time_early = float(timestamp_int) + float(timestamp_frac) * 1e-12
    _is_wav_early   = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav_early and timestamp_int < 1_000_000_000:
        _t0 = getattr(self, '_iq_t0_wall', None)
        if _t0 is not None:
            # Subsequent chunk: derive from sample counter.  _iq_abs_sample
            # is the count *before* this chunk has been added, so this is
            # the wall-clock for the first sample of this chunk.
            _pkt_time_early = _t0 + self._iq_abs_sample / self.sample_rate
        else:
            # Very first chunk after start/reset: anchor to wall clock once.
            _pkt_time_early = time.time()


    # ── 0. Noise blanker (Linrad / Bypass / NR0V-Wideband) ────────────────────
    # Backend is selected via ``self.blanker`` (set in engine.__init__, runtime-
    # switchable).  Each backend operates on the engine's ``_nb_*`` state so
    # display and reset paths continue to work without change.  Dual-pol
    # backends return independent H and V blanking masks; single-pipeline
    # backends return only blank_mask (H) and leave blank_mask_v None, in
    # which case the H mask is applied to V for backward compatibility.
    _nb_result   = self.blanker.process(self, raw, raw_v, dual_pol)
    cleaned      = _nb_result.cleaned_h
    cleaned_v    = _nb_result.cleaned_v
    blank_mask   = _nb_result.blank_mask
    mag          = _nb_result.mag_h
    blank_mask_v = _nb_result.blank_mask_v if _nb_result.blank_mask_v is not None else blank_mask
    mag_v_disp   = (_nb_result.mag_v if _nb_result.mag_v is not None else
                    (np.abs(raw_v).astype(np.float32) if dual_pol and raw_v is not None else None))


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
    # Anchor wall-clock for IQ ring sample 0 — see project_sample_clock_anchoring.
    # Set ONCE per session (or after a reset that nulls it).  Re-anchoring per
    # chunk would re-introduce wall-clock jitter into _iq_t0_wall and undo the
    # whole point of sample-clock anchoring.
    #
    # The drift threshold for re-anchoring needs to be high enough that normal
    # source-side timestamp jitter never trips it.  Flex's per-packet timestamp
    # (computed in the source thread as ``_t_start + _abs_samps/_hw_rate``) can
    # oscillate ±1–2 s under CPU contention because ``_t_start`` is anchored
    # once at acquisition start with a `time.time()` reading that itself has
    # OS-scheduling jitter and NTP-correction events.  A 1-s threshold caused
    # constant re-anchoring during noise bursts; each re-anchor jumped
    # ``_iq_t0_wall`` enough to spuriously cross a 15-s boundary in the
    # heatmap-display logic and triggered a wipe.
    #
    # 30 s is well above any normal jitter (Flex packets stay within ±2 s) but
    # well below the "stream is broken" regime — a real source-restart or DAX
    # stream reset would manifest as tens of seconds of discontinuity.  The
    # PERIOD_SLIP fix can absorb a few seconds of drift without misassigning
    # decodes (15 s period, plenty of slack).
    _projected = (self._iq_t0_wall + self._iq_abs_sample / self.sample_rate
                  if self._iq_t0_wall is not None else None)
    if self._iq_t0_wall is None:
        self._iq_t0_wall = _pkt_time_early - self._iq_abs_sample / self.sample_rate
    elif abs(_pkt_time_early - _projected) > 30.0:
        # Major discontinuity — re-anchor and log so the cause is visible.
        self._iq_t0_wall = _pkt_time_early - self._iq_abs_sample / self.sample_rate
        logger.warning("Re-anchored _iq_t0_wall: drift %.2f s from projected",
                       _pkt_time_early - _projected)
    self._iq_abs_sample += chunk_len

    # ── 1a. Update time-domain magnitude display buffers (post- and pre-blanker) ──
    def _write_td_buf(buf, pos_attr, data):
        """Write data into the circular buffer named buf; return new position."""
        n   = len(buf)
        d_n = len(data)
        pos = getattr(self, pos_attr)
        if pos + d_n <= n:
            buf[pos:pos + d_n] = data
        else:
            first = n - pos
            buf[pos:]         = data[:first]
            buf[:d_n - first] = data[first:]
        setattr(self, pos_attr, (pos + d_n) % n)

    _td_mag     = np.abs(cleaned).astype(np.float32)   # post-blanker H
    _write_td_buf(self._td_mag_buf,     '_td_mag_pos',     _td_mag)
    _write_td_buf(self._td_mag_buf_raw, '_td_mag_pos_raw', mag)     # pre-blanker H

    # ── 1b. V-channel time-domain magnitude display buffers ──────────────────
    # Per-channel V detection state (_nb_spec_avg_v, _nb_last_P_v, …) is
    # now owned by the blanker backend (see LinradBlanker._process_channel
    # with suffix='_v').  Only the display-only post/pre-blanker magnitude
    # circular buffers are updated here, using the V mag provided in the
    # BlankerResult so NR0V's delay-aligned pre-blanker trace stays aligned.
    if dual_pol and cleaned_v is not None:
        _td_mag_v = np.abs(cleaned_v).astype(np.float32)            # post-blanker V
        _write_td_buf(self._td_mag_buf_v,     '_td_mag_pos_v',     _td_mag_v)
        if mag_v_disp is not None:
            _write_td_buf(self._td_mag_buf_raw_v, '_td_mag_pos_raw_v', mag_v_disp)

    # ── 2. Channelise into 48 × 12 kHz channels (H, and V if dual-pol) ─────────
    # Dual-pol: pass both H and V in one call so the NCO phase matrix (the most
    # expensive part) is computed once instead of twice, and the FIR filter calls
    # are batched over 2×N_CHANNELS rows instead of two separate passes.
    if dual_pol:
        ch_out, ch_out_v = apply_channelizer(
            cleaned, self._ch_state,
            lp_taps=self._ch_taps,
            iq_block_v=cleaned_v, state_v=self._ch_state_v,
        )
    else:
        ch_out   = apply_channelizer(cleaned, self._ch_state, lp_taps=self._ch_taps)
        ch_out_v = None

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

        # Sync correlator's dedicated rolling ring — FIFO shift-and-append so
        # the most recent NSPM samples per channel are always available
        # regardless of _ch_buf's drain state.  See engine.py for the rationale.
        if ENABLE_SYNC_DETECT:
            _sb_n = self._sync_buf.shape[1]   # = NSPM
            if new_n >= _sb_n:
                self._sync_buf[:] = ch_out[:, -_sb_n:]
            else:
                self._sync_buf[:, :_sb_n - new_n] = self._sync_buf[:, new_n:]
                self._sync_buf[:, _sb_n - new_n:] = ch_out
            if dual_pol:
                if new_n >= _sb_n:
                    self._sync_buf_v[:] = ch_out_v[:, -_sb_n:]
                else:
                    self._sync_buf_v[:, :_sb_n - new_n] = self._sync_buf_v[:, new_n:]
                    self._sync_buf_v[:, _sb_n - new_n:] = ch_out_v

    # Anchor wall-clock time of _ch_buf[0] once per call.  This is no longer
    # used for the per-hop write index (the hop counter handles that), but it
    # is still used inside the loop as ``current_wall_time`` for the 15-s
    # boundary marker-lifecycle check.
    if self.source_mode == "wav":
        # _wav_time_cursor is the end of the previous call's buffer.  Subtract
        # the carry-over samples already in _ch_buf so the anchor points to
        # _ch_buf[0].
        _loop_wall = float(getattr(self, "_wav_time_cursor", 0.0)) \
                     - _ch_buf_carry / CH_SAMPLE_RATE
    else:
        # Live radio: use the VITA-49 sample-clock timestamp so the heatmap
        # is anchored to the same GPS clock as the spectrogram.  Subtract the
        # carry-over channelised samples so the anchor is at _ch_buf[0].
        _loop_wall = _pkt_time_early - _ch_buf_carry / CH_SAMPLE_RATE

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
        lo_peak   = power_lin[:, _LO_MASK].max(axis=1)          # (48,) linear
        hi_peak   = power_lin[:, _HI_MASK].max(axis=1)          # (48,) linear
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
            tmp = hist_s.copy()
            tmp.partition(k, axis=0)          # in-place, bypasses _wrapfunc
            self._pct25 = np.maximum(tmp[k], 1e-30)
        pct25 = self._pct25

        pair_metric_h = (
            10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30))
        ).astype(np.float32)   # (48,) dB above baseline

        # ── V-channel rolling 25th-percentile baseline (independent noise floor) ─
        if ch_block_v is not None:
            plin_v    = np.abs(X_sq_v) / CH_DETECT_SIZE
            raw_lin_v = (plin_v[:, _LO_MASK].max(axis=1) +
                         plin_v[:, _HI_MASK].max(axis=1)) / 2.0
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
                tmp_v = hist_vs.copy()
                tmp_v.partition(kv, axis=0)       # in-place, bypasses _wrapfunc
                self._pct25_v = np.maximum(tmp_v[kv], 1e-30)
            pair_metric_v = (
                10.0 * np.log10(np.maximum(raw_lin_v / self._pct25_v, 1e-30))
            ).astype(np.float32)
        else:
            pair_metric_v = pair_metric_h   # single-pol: V mirrors H (unused)

        # ── Coherent sync-word detector (additive Step 1) ────────────────────
        # Per-channel correlation of the leading NSPM-sample window of _ch_buf
        # against the MSK144 16-chip sync template.  Coherent integration over
        # 8 ms of known waveform buys ~6 dB over the squared-domain power
        # detector at low SNR.  Output magnitude is normalised by a per-channel
        # 25th-percentile baseline (mirrors the squared-FFT path) so the dB
        # metric is comparable, then pair_metric takes the max of the two paths.
        # The window is read fresh per hop, so the metric tracks the current
        # ch_block (not stalely fixed at chunk boundaries).
        # Save the squared-only metrics so we can attribute each surviving
        # trigger to "sq", "sync", or "both" downstream — useful for verifying
        # the sync detector is firing at all and for diagnostic console output.
        sq_metric_h_only = pair_metric_h.copy()
        sq_metric_v_only = pair_metric_v.copy()
        sync_metric_h = None
        sync_metric_v = None

        if ENABLE_SYNC_DETECT:
            # _sync_buf is FIFO-updated per chunk and always holds the most
            # recent NSPM samples, so the correlator can run on every hop
            # regardless of _ch_buf's drain state.  No cache fallback needed
            # (which previously produced stale-row banding under live Flex
            # operation where _ch_buf_end never reaches NSPM).
            sync_metric_h = _compute_sync_metric_db(
                self._sync_buf,
                self._sync_metric_hist_buf, '_sync_metric_hist_idx',
                '_sync_metric_hist_cnt', '_sync_pct25', '_sync_pct25_ctr',
                self,
            )
            pair_metric_h = np.maximum(pair_metric_h, sync_metric_h)

            if ch_block_v is not None:
                sync_metric_v = _compute_sync_metric_db(
                    self._sync_buf_v,
                    self._sync_metric_hist_buf_v, '_sync_metric_hist_idx_v',
                    '_sync_metric_hist_cnt_v', '_sync_pct25_v', '_sync_pct25_ctr_v',
                    self,
                )
                pair_metric_v = np.maximum(pair_metric_v, sync_metric_v)

        # Trigger on either channel; display the max for the combined heatmap
        pair_metric = np.maximum(pair_metric_h, pair_metric_v)

        # ── Display-only 15-s boundary handling ──────────────────────────────
        # Blank the heatmaps and reset the hop counter at each 15-s wall-clock
        # boundary cross (and at session start, when _ch_snr_boundary's initial
        # value differs from the current period number).  Internal DSP state
        # (rolling baselines, AGC, percentile estimators) is NOT reset — those
        # run continuously and the 15-s boundary doesn't apply to them.
        # The hop counter (rather than a sample-clock-to-row map) is what fills
        # rows: each hop fires exactly when the buffer drained 256 samples, so
        # consecutive hops always fill consecutive rows.  This eliminates the
        # rare edge case where a sample-clock map could place the first hop
        # after a boundary at row 1 (or higher) and leave row 0 dark for an
        # entire period — visible as a persistent vertical dark line.
        ch_boundary = int(current_wall_time / self.history_secs)
        if ch_boundary != self._ch_snr_boundary:
            self._ch_snr_history_h[:] = -999.0
            self._ch_snr_history_v[:] = -999.0
            self._sync_snr_history_h[:] = -999.0
            self._sync_snr_history_v[:] = -999.0
            self._ch_snr_hop_count = 0
            self._ch_snr_boundary  = ch_boundary
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                marker_list.clear()
            # Do NOT reset _ch_above_thresh — the coincidence gate must survive
            # the display window boundary so a signal that starts just before
            # the boundary still triggers a jt9 launch after it.

        # Hop-counter row index — guaranteed contiguous across hops.
        self._ch_snr_write_idx = self._ch_snr_hop_count % N_SNR_HIST
        self._ch_snr_history_h[self._ch_snr_write_idx, :] = pair_metric_h
        self._ch_snr_history_v[self._ch_snr_write_idx, :] = pair_metric_v
        if sync_metric_h is not None:
            self._sync_snr_history_h[self._ch_snr_write_idx, :] = sync_metric_h
        if sync_metric_v is not None:
            self._sync_snr_history_v[self._ch_snr_write_idx, :] = sync_metric_v
        # ── DIAGNOSTIC: heatmap state at selected hops (DEBUG-level) ──────────
        # Enable with logging.getLogger("map144_app.processing").setLevel(DEBUG)
        # — useful when investigating detection-plot rendering vs. metric scale
        # vs. baseline staleness.  Six samples per WAV play; negligible cost.
        if (logger.isEnabledFor(logging.DEBUG)
                and self._ch_snr_hop_count in (0, 5, 50, 200, 500, 700)):
            _arr = self._ch_snr_history_h
            _sa  = self._sync_snr_history_h if sync_metric_h is not None else None
            logger.debug(
                "heatmap-diag hop=%d pair_h max=%.2f dB argmax_ch=%d "
                "row_just_written max=%.2f full_arr min=%.2f max=%.2f "
                "nonzero_rows=%d sync max=%.2f boundary=%d",
                self._ch_snr_hop_count,
                float(pair_metric_h.max()), int(np.argmax(pair_metric_h)),
                float(_arr[self._ch_snr_write_idx].max()),
                float(_arr.min()), float(_arr.max()),
                int(np.any(_arr != 0.0, axis=1).sum()),
                float(_sa.max()) if _sa is not None else float('nan'),
                int(self._ch_snr_boundary),
            )
        self._ch_snr_hop_count += 1

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
            _triggered = _triggered[pair_metric[_triggered].argsort()[::-1]]

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
        _sorted_idx = _triggered.astype(int); _sorted_idx.sort()  # .sort() bypasses _wrapfunc
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
                    logger.debug("[detect] ch %d sustained signal — locking out for 30 s", _ck)
                    self._detect_cooldowns[_ck] = _SUSTAINED_LOCKOUT
            else:
                _ch_above.pop(_ck, None)
        self._ch_above_thresh = _ch_above

        for ch_k, _cl_all in _launch_channels:
            _nyquist_ch = N_CHANNELS // 2  # ch 24 = ±24 kHz Nyquist
            if abs(int(ch_k) - _nyquist_ch) <= _EDGE_CH_SKIP:
                continue

            # Detector attribution: which path put this channel above threshold?
            # Examine BOTH the squared-FFT and sync metrics (max of H/V if dual-pol)
            # against DETECT_THRESH_DB.  Result is one of "sq", "sync", "sq+sync".
            _sq_above = (max(sq_metric_h_only[ch_k],
                             sq_metric_v_only[ch_k] if ch_block_v is not None
                             else -np.inf) > DETECT_THRESH_DB)
            _sync_above = False
            if sync_metric_h is not None:
                _sync_h = sync_metric_h[ch_k]
                _sync_v = (sync_metric_v[ch_k]
                           if (sync_metric_v is not None and ch_block_v is not None)
                           else -np.inf)
                _sync_above = max(_sync_h, _sync_v) > DETECT_THRESH_DB
            if _sq_above and _sync_above:
                det_source = 'sq+sync'
            elif _sync_above:
                det_source = 'sync'
            else:
                det_source = 'sq'

            # Estimate fc from peak bin locations in each window (linear domain)
            lo_freq   = float(_SQ_FREQ[_LO_MASK][np.argmax(power_lin[ch_k][_LO_MASK])])
            hi_freq   = float(_SQ_FREQ[_HI_MASK][np.argmax(power_lin[ch_k][_HI_MASK])])
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

            # Per-launch DSP-state snapshot for launches.jsonl.  Lets offline
            # analysis distinguish "real strong signal that didn't decode"
            # from "noise spike at the band edge crossing a low pct25
            # threshold" without needing the saved IQ.  All values in dB.
            _pct25_arr   = getattr(self, '_pct25', None)
            _pct25_v_arr = getattr(self, '_pct25_v', None)
            launch_metrics = {
                'pair_metric_db_h':  round(float(pair_metric_h[ch_k]), 1),
                'pair_metric_db_v':  (round(float(pair_metric_v[ch_k]), 1)
                                      if ch_block_v is not None else None),
                'sq_metric_db_h':    round(float(sq_metric_h_only[ch_k]), 1),
                'sq_metric_db_v':    (round(float(sq_metric_v_only[ch_k]), 1)
                                      if ch_block_v is not None else None),
                'sync_metric_db_h':  (round(float(sync_metric_h[ch_k]), 1)
                                      if sync_metric_h is not None else None),
                'sync_metric_db_v':  (round(float(sync_metric_v[ch_k]), 1)
                                      if sync_metric_v is not None else None),
                'ch_pct25_db_h':     (round(float(10.0 * np.log10(_pct25_arr[ch_k])), 1)
                                      if _pct25_arr is not None else None),
                'ch_pct25_db_v':     (round(float(10.0 * np.log10(_pct25_v_arr[ch_k])), 1)
                                      if _pct25_v_arr is not None else None),
                'ch_signed':         int(ch_signed),
                'n_cluster_chan':    int(len(_cl_all)),
                'det_source':        det_source,
            }

            if not getattr(self, '_analysis_no_decode', False):
                t = threading.Thread(
                    target=extract_and_decode,
                    args=(self._iq_ring, ring_state_fn,
                          abs_snap, self.sample_rate, fc_hz, output_dir,
                          t_in_window, decode_queue, marker_id,
                          ring_gen, ring_gen_fn,
                          getattr(self, 'display_center_freq_mhz', self.center_freq_mhz),
                          detect_ts, None, dual_pol),
                    kwargs={'det_source': det_source,
                            'launch_metrics': launch_metrics},
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
                # det_pol: which channel had the higher detection metric at launch time.
                # Used for immediate circle routing before theta_deg is available.
                _det_pol = ('v'
                            if dual_pol and pair_metric_v[ch_k] > pair_metric_h[ch_k]
                            else 'h')
                marker_list.append({
                    'id':        marker_id,
                    't':         t_in_window,
                    'freq_khz':  display_khz,
                    'boundary':  self._ch_snr_boundary,
                    'decoded':   False,
                    'outcome':   'launched',
                    'message':   None,
                    'theta_deg': None,   # filled by extract_and_decode on dual-pol decode
                    'det_pol':   _det_pol,
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
    # hardware clock is not yet locked to UTC (Flex TSI=0, RTL-SDR, Airspy without GPS).
    # Fall back to the sample-clock-anchored time (_iq_t0_wall + abs_sample/sr), NOT
    # to time.time() — the latter introduces wall-clock jitter that makes _sbuf_t0
    # jump between chunks during CPU bursts, which makes row_idx jump, which fires
    # the spec_staging gap-fill and produces visible 200 ms bands of repeated
    # spectrum content.  See project_sample_clock_anchoring memory.
    _is_wav = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav and timestamp_int < 1_000_000_000:
        _t0 = getattr(self, '_iq_t0_wall', None)
        if _t0 is not None:
            # _iq_abs_sample at this point already includes this chunk's samples
            # (incremented earlier in process_iq_data), so _sbuf_end_before pointing
            # at the same absolute sample count yields a consistent _sbuf_t0.
            _pkt_time = _t0 + (self._iq_abs_sample - new_n) / self.sample_rate
        else:
            _pkt_time = time.time()
    # Time of _sbuf[0] = packet's first-sample time minus already-buffered samples
    self._sbuf_t0 = _pkt_time - _sbuf_end_before / self.sample_rate

    fft_window  = self._fft_window   # pre-computed float32 Hanning window
    window_gain = self._window_gain  # pre-computed scalar
    full_scale  = 1.0
    _fg_win     = getattr(self, '_fast_graph_win', None)
    _fg_visible = _fg_win is None or _fg_win.isVisible()
    _wf_hop     = self.fft_size // 2   # constant; moved out of the loop

    while self._sbuf_end >= self.fft_size:
        try:
            if _fg_visible:
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
                    _gap = row_idx - _prev_staging - 1
                    # Diagnostic: gaps > 2 rows (~43 ms) are worth investigating;
                    # with sample-clock-anchored _sbuf_t0 these should not happen
                    # in steady state.  Log size + recent timing context.
                    if _gap > 2:
                        logger.info(
                            "spec_staging gap-fill: %d rows (~%.0f ms) at row_idx=%d "
                            "_sbuf_t0=%.6f _sbuf_end_before=%d",
                            _gap, _gap * 1000.0 / self.blocks_per_sec,
                            row_idx, self._sbuf_t0, _sbuf_end_before,
                        )
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

        except Exception as _wf_exc:
            import logging as _lg
            _lg.getLogger(__name__).warning("waterfall FFT error (skipping block): %s", _wf_exc)

        finally:
            # Slide always runs — even on exception — so _sbuf_end never overflows.
            remaining = self._sbuf_end - _wf_hop
            if remaining > 0:
                self._sbuf[:remaining] = self._sbuf[_wf_hop:self._sbuf_end]
            self._sbuf_end = max(0, remaining)
            if dual_pol and self._sbuf_v_end >= _wf_hop:
                remaining_v = self._sbuf_v_end - _wf_hop
                if remaining_v > 0:
                    self._sbuf_v[:remaining_v] = self._sbuf_v[_wf_hop:self._sbuf_v_end]
                self._sbuf_v_end = remaining_v
            # Advance the time anchor by exactly one hop so the next block's
            # current_wall_time is sample-accurate without calling time.time().
            self._sbuf_t0 += _wf_hop / self.sample_rate

    # Keep _wav_time_cursor in sync for the WAV source step (it uses this to
    # timestamp the next packet it will send into process_iq_data).
    if self.source_mode == "wav":
        self._wav_time_cursor = self._sbuf_t0 + self._sbuf_end / self.sample_rate

