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

1. **Ring buffer** — cleaned IQ samples are written into a 5-second circular
   buffer.  extract_and_decode reads this buffer when a signal is confirmed.

2. **Channeliser** — apply_channelizer() splits the 48 kHz IQ block into
   N_CHANNELS=48 sub-bands at CH_SAMPLE_RATE=12 kHz each.  Each channel k is
   centred at k × 1 kHz.

3. **Per-channel detection** — channelised samples are accumulated in
   _ch_buffer (48 × N) until CH_DETECT_SIZE samples are available.  For each
   complete block:

   a. Square the complex sub-band samples.
   b. Windowed FFT of the squared signal at 12 kHz.
   c. Peak SNR above median noise floor → per-channel detection metric.
   d. scan_for_pairs() on channels that exceed DETECT_THRESH_DB to get
      a precise carrier estimate; launch extract_and_decode in a daemon thread.

4. **SNR history** — peak_snr (48,) is appended to a rolling circular buffer
   _ch_snr_history (N_SNR_HIST × 48).  displays.py reads this to draw the
   48-channel ridgeline detection plot.

5. **Normal overlap-FFT** — cleaned IQ accumulated in sample_buffer is
   processed in fft_size blocks with 50 % overlap, producing the full-bandwidth
   spectrum for the waterfall displays.

Constants
---------
DETECT_THRESH_DB    : 3.5 dB  — threshold: dB above 25th-percentile baseline
DETECT_MERGE_GAP_S  : 1.0 s   — cooldown between triggers on the same channel
CH_DETECT_SIZE      : 512     — per-channel FFT size for detection (at 12 kHz)
_METRIC_HIST_DEPTH  : 300     — rolling history depth for percentile normaliser
N_SNR_HIST          : 300     — rolling history depth for the ridgeline display
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
    CHANNEL_OFFSET_HZ,
    CH_SAMPLE_RATE,
)
from .detection import extract_and_decode

DETECT_THRESH_DB   = 3.5        # dB above 25th-percentile noise baseline
DETECT_MERGE_GAP_S = 1.0        # suppress re-trigger within this window (s)
CH_DETECT_SIZE     = 512        # samples at 12 kHz per detection FFT block
NB_FACTOR          = 6.0        # blanker amplitude threshold (hot bin power > NB_FACTOR² × avg)
NB_FFT_SIZE        = 256        # detection block size: 5.3 ms at 48 kHz
NB_BROADBAND_FRAC  = 0.30       # fraction of hot bins required to declare broadband impulse
NB_SPEC_AVG_TC     = 2.0        # per-bin spectrum averaging time constant (seconds)
_NB_TAPER_N        = 24         # Hann taper half-width in samples (24 → 0.5 ms at 48 kHz)
_EDGE_CH_SKIP      = 4          # skip channels within N of Nyquist (ch 24) — DAXIQ filter rolloff zone
_DC_CH_SKIP        = 3          # skip channels within N of DC (ch 0) — LO leakage / DC offset zone
_SQ_TONE_HZ        = 1000.0     # expected squared-domain tone offset from DC
_SQ_NTOL_HZ        = 200.0      # half-width of each tone search window
_METRIC_HIST_DEPTH = 300        # frames of rolling linear-peak history for percentile.
                                 # At the 256-sample / 12 kHz hop rate each frame is ~21 ms,
                                 # so 300 frames ≈ 6.4 s.  As long as pings occupy < 25 % of
                                 # that window the 25th percentile stays at the noise floor,
                                 # mirroring the xmed normalisation in WSJT-X msk144spd.f90.
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
    """Process one IQ chunk: time-domain blanker, ring buffer,
    channelised detection, and waterfall display FFT.

    Processing order
    ----------------
    0. Time-domain noise blanker — zero samples whose amplitude exceeds
       nb_factor × running RMS envelope.  The envelope is updated only
       from unblanked samples so large impulses cannot raise the threshold
       and hide themselves.  All downstream stages operate on the cleaned
       signal.

    1. Ring buffer — cleaned IQ written to 5-second circular buffer.
    2. Channeliser — cleaned IQ split into 48 × 12 kHz sub-bands.
    3. Per-channel detection — squared FFT, pair-metric, jt9 launch.
    4. Waterfall FFT — overlapping Hanning FFT on cleaned IQ for display.
    """

    raw = iq_samples.astype(np.complex64)

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
        else:
            # Update per-bin averages only from clean blocks.
            self._nb_spec_avg = (1.0 - alpha) * self._nb_spec_avg + alpha * P

    # Partial block at end of chunk passes through unchanged (no blanking decision).

    # Soft-blank: Hann-taper the edges of each blanked region so the abrupt
    # transition to zero does not create rectangular-hole sinc sidelobes.
    if np.any(blank_mask):
        soft_blank = np.minimum(
            np.convolve(blank_mask, _NB_TAPER_KERNEL, mode='same'),
            1.0,
        )
        cleaned = (raw * (1.0 - soft_blank)).astype(np.complex64)
    else:
        cleaned = raw.copy()

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


    chunk_len = len(cleaned)

    # ── 1. Write cleaned IQ into the circular ring buffer ────────────────────
    ring_size = len(self._iq_ring)
    pos = self._iq_ring_pos
    if pos + chunk_len <= ring_size:
        self._iq_ring[pos:pos + chunk_len] = cleaned
    else:
        first_n = ring_size - pos
        self._iq_ring[pos:]                  = cleaned[:first_n]
        self._iq_ring[:chunk_len - first_n]  = cleaned[first_n:]
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

    # ── 2. Channelise into 48 × 12 kHz sub-bands ─────────────────────────────
    ch_out = apply_channelizer(
        cleaned, self._ch_state,
        lp_taps=self._ch_taps,
    )   # (N_CHANNELS, chunk_len // 4)

    # ── 3. Per-channel detection ──────────────────────────────────────────────
    # Accumulate into pre-allocated slide-left buffer (no per-call allocation)
    new_n = ch_out.shape[1]
    self._ch_buf[:, self._ch_buf_end:self._ch_buf_end + new_n] = ch_out
    self._ch_buf_end += new_n

    while self._ch_buf_end >= CH_DETECT_SIZE:
        ch_block = self._ch_buf[:, :CH_DETECT_SIZE]              # (48, 512) view — no copy

        sq         = ch_block ** 2                               # complex squaring
        X_sq       = np.fft.fft(sq * _DETECT_WINDOW[np.newaxis, :], axis=1)

        # Wall-clock time
        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
        else:
            current_wall_time = time.time()
        t_in_window = current_wall_time % self.history_secs

        # ── Two-window tone detection in the squared domain ───────────────────
        # MSK144 tones at fc_offset ± 500 Hz → after squaring → ±1000 Hz.
        # _SQ_FREQ, _LO_MASK, _HI_MASK are pre-computed module-level constants.
        power_lin = np.abs(X_sq) / CH_DETECT_SIZE               # (48, 512) linear
        plin_all  = np.fft.fftshift(power_lin, axes=1)          # (48, 512)
        lo_peak   = np.max(plin_all[:, _LO_MASK], axis=1)       # (48,) linear
        hi_peak   = np.max(plin_all[:, _HI_MASK], axis=1)       # (48,) linear
        raw_lin   = (lo_peak + hi_peak) / 2.0                   # (48,) linear

        # Rolling 25th-percentile baseline — numpy circular buffer avoids
        # O(N) list.pop(0) and np.stack() allocation on every detection block.
        self._metric_hist_buf[self._metric_hist_idx] = raw_lin
        self._metric_hist_idx = (self._metric_hist_idx + 1) % _METRIC_HIST_DEPTH
        if self._metric_hist_cnt < _METRIC_HIST_DEPTH:
            self._metric_hist_cnt += 1
        hist  = (self._metric_hist_buf[:self._metric_hist_cnt]
                 if self._metric_hist_cnt < _METRIC_HIST_DEPTH
                 else self._metric_hist_buf)
        k     = max(0, int(len(hist) * 0.25))                        # 25th-pct index
        pct25 = np.maximum(np.partition(hist, k, axis=0)[k], 1e-30)  # (48,) O(N) vs O(N log N)

        pair_metric = np.maximum(
            10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30)), 0.0
        ).astype(np.float32)                                     # (48,) dB above baseline

        # Time-indexed SNR write — mirrors spectrogram boundary logic.
        ch_boundary = int(current_wall_time / self.history_secs)
        if ch_boundary != self._ch_snr_boundary:
            self._ch_snr_history[:] = 0.0
            self._ch_snr_boundary   = ch_boundary
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                marker_list.clear()
            self._ch_snr_write_idx  = min(
                max(int(t_in_window * CH_SAMPLE_RATE / _CH_DETECT_HOP), 0),
                N_SNR_HIST - 1,
            )
        if 0 <= self._ch_snr_write_idx < N_SNR_HIST:
            self._ch_snr_history[self._ch_snr_write_idx, :] = pair_metric
        self._ch_snr_write_idx += 1

        # Tick down cooldowns
        self._detect_cooldowns = {
            k: v - 1 for k, v in self._detect_cooldowns.items() if v > 1
        }

        cooldown_hops = max(
            1, int(DETECT_MERGE_GAP_S * CH_SAMPLE_RATE / _CH_DETECT_HOP)
        )

        for ch_k in np.where(pair_metric > DETECT_THRESH_DB)[0]:
            _nyquist_ch = N_CHANNELS // 2  # ch 24 = ±24 kHz Nyquist
            if abs(int(ch_k) - _nyquist_ch) <= _EDGE_CH_SKIP:
                continue
            if ch_k in self._detect_cooldowns:
                continue

            # Estimate fc from peak bin locations in each window (linear domain)
            lo_freq   = float(_SQ_FREQ[_LO_MASK][np.argmax(plin_all[ch_k][_LO_MASK])])
            hi_freq   = float(_SQ_FREQ[_HI_MASK][np.argmax(plin_all[ch_k][_HI_MASK])])
            fc_offset  = (lo_freq + hi_freq) / 4.0   # squaring doubled freqs
            # ch_k 25-47 are negative-frequency channels (-23 to -1 kHz)
            ch_signed  = int(ch_k) if int(ch_k) <= N_CHANNELS // 2 else int(ch_k) - N_CHANNELS
            # CHANNEL_OFFSET_HZ shifts channel centres by +500 Hz so that USB MSK144
            # signals at integer-kHz dial + 1500 Hz audio land at fc_offset = 0.
            fc_hz      = ch_signed * CHANNEL_SPACING_HZ + CHANNEL_OFFSET_HZ + fc_offset

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

            t = threading.Thread(
                target=extract_and_decode,
                args=(self._iq_ring, ring_state_fn,
                      abs_snap, self.sample_rate, fc_hz, output_dir,
                      t_in_window, decode_queue, marker_id,
                      ring_gen, ring_gen_fn,
                      getattr(self, 'display_center_freq_mhz', self.center_freq_mhz),
                      detect_ts),
                daemon=True,
            )
            t.start()
            if jt9_list is not None:
                jt9_list.append(t)
            # Record marker for heatmap overlay; fc_hz is already a signed offset (-24..+24 kHz)
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                display_khz = fc_hz / 1000.0
                marker_list.append({
                    'id':       marker_id,
                    't':        t_in_window,
                    'freq_khz': display_khz,
                    'boundary': self._ch_snr_boundary,
                    'decoded':  False,
                    'message':  None,
                })
            self._detect_cooldowns[ch_k] = cooldown_hops

        # Slide channeliser buffer left by one hop (in-place copy, no allocation)
        remaining = max(0, self._ch_buf_end - _CH_DETECT_HOP)
        if remaining > 0:
            self._ch_buf[:, :remaining] = self._ch_buf[:, _CH_DETECT_HOP:self._ch_buf_end]
        self._ch_buf_end = remaining

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

    # All sources must deliver timestamp_frac as picoseconds (adapter responsibility).
    _pkt_time = float(timestamp_int) + float(timestamp_frac) * 1e-12
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

        staging_idx = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)
        self.spec_staging[staging_idx] = power_db

        # ── Real-time spectrogram ─────────────────────────────────────────
        realtime_boundary = int(current_wall_time / self.history_secs)
        if realtime_boundary != self._realtime_boundary:
            self.realtime_data         = np.full((self.max_history, self.fft_size), -130.0)
            self._realtime_boundary    = realtime_boundary

        rt_idx = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)
        self.realtime_data[rt_idx] = power_db
        self.realtime_filled  = True
        self._realtime_dirty  = True

        # Slide sample buffer left by one hop (in-place copy, no allocation)
        hop = self.fft_size // 2
        remaining = self._sbuf_end - hop
        if remaining > 0:
            self._sbuf[:remaining] = self._sbuf[hop:self._sbuf_end]
        self._sbuf_end = max(0, remaining)
        # Advance the time anchor by exactly one hop so the next block's
        # current_wall_time is sample-accurate without calling time.time().
        self._sbuf_t0 += hop / self.sample_rate

    # Keep _wav_time_cursor in sync for the WAV source step (it uses this to
    # timestamp the next packet it will send into process_iq_data).
    if self.source_mode == "wav":
        self._wav_time_cursor = self._sbuf_t0 + self._sbuf_end / self.sample_rate
