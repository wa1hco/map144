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
1. **Ring buffer** — raw IQ samples are written into a 5-second circular buffer.
   extract_and_decode reads this buffer when a signal is confirmed.

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

5. **Normal overlap-FFT** — raw IQ accumulated in sample_buffer is processed
   in fft_size blocks with 50 % overlap, producing the full-bandwidth spectrum
   for the waterfall displays.  No LP filtering is applied here.

Constants
---------
DETECT_THRESH_DB    : 3.0 dB  — threshold: dB above 25th-percentile baseline
DETECT_MERGE_GAP_S  : 1.0 s   — cooldown between triggers on the same channel
CH_DETECT_SIZE      : 512     — per-channel FFT size for detection (at 12 kHz)
_METRIC_HIST_DEPTH  : 300     — rolling history depth for percentile normaliser
N_SNR_HIST          : 300     — rolling history depth for the ridgeline display
"""

import threading
import time
from pathlib import Path

import numpy as np

from .channelizer import (
    apply_channelizer,
    N_CHANNELS,
    CHANNEL_SPACING_HZ,
    CH_SAMPLE_RATE,
)
from .detection import extract_and_decode

DETECT_THRESH_DB   = 3.0        # dB above 25th-percentile noise baseline
DETECT_MERGE_GAP_S = 1.0        # suppress re-trigger within this window (s)
CH_DETECT_SIZE     = 512        # samples at 12 kHz per detection FFT block
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


def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
    """Process one IQ chunk: update ring buffer, run channelised detection,
    and update the full-bandwidth spectrogram display buffers."""

    # ── 1. Write raw IQ into the circular ring buffer ────────────────────────
    chunk_len = len(iq_samples)
    ring_size = len(self._iq_ring)
    pos = self._iq_ring_pos
    raw = iq_samples.astype(np.complex64)
    if pos + chunk_len <= ring_size:
        self._iq_ring[pos:pos + chunk_len] = raw
    else:
        first_n = ring_size - pos
        self._iq_ring[pos:]          = raw[:first_n]
        self._iq_ring[:chunk_len - first_n] = raw[first_n:]
    self._iq_ring_pos    = (pos + chunk_len) % ring_size
    self._iq_abs_sample += chunk_len

    # ── 2. Channelise into 48 × 12 kHz sub-bands ─────────────────────────────
    ch_out = apply_channelizer(
        raw, self._ch_state,
        lp_taps=self._ch_taps,
    )   # (N_CHANNELS, chunk_len // 4)

    # ── 3. Per-channel detection ──────────────────────────────────────────────
    self._ch_buffer = np.concatenate([self._ch_buffer, ch_out], axis=1)

    while self._ch_buffer.shape[1] >= CH_DETECT_SIZE:
        ch_block = self._ch_buffer[:, :CH_DETECT_SIZE]          # (48, 512)
        self._ch_buffer = self._ch_buffer[:, _CH_DETECT_HOP:]   # slide by hop

        window     = np.hanning(CH_DETECT_SIZE)
        sq         = ch_block ** 2                               # complex squaring
        X_sq       = np.fft.fft(sq * window[np.newaxis, :], axis=1)

        # Wall-clock time
        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
        else:
            current_wall_time = time.time()
        t_in_window = current_wall_time % self.history_secs

        # ── Two-window tone detection in the squared domain ───────────────────
        # MSK144 tones at fc_offset ± 500 Hz → after squaring → ±1000 Hz.
        # Search for the max linear power in a ±NTOL window around each tone.
        # Metric is normalised by the rolling 25th percentile of the raw linear
        # average (mirroring the xmed normalisation in WSJT-X msk144spd.f90).
        # This is immune to the 2·signal·noise cross-product that biases a
        # per-frame median noise estimate during a ping burst.
        sq_freq  = np.fft.fftshift(
            np.fft.fftfreq(CH_DETECT_SIZE, 1.0 / CH_SAMPLE_RATE)
        )
        lo_mask  = (sq_freq >= -_SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                   (sq_freq <= -_SQ_TONE_HZ + _SQ_NTOL_HZ)
        hi_mask  = (sq_freq >=  _SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                   (sq_freq <=  _SQ_TONE_HZ + _SQ_NTOL_HZ)

        power_lin = np.abs(X_sq) / CH_DETECT_SIZE               # (48, 512) linear
        plin_all  = np.fft.fftshift(power_lin, axes=1)          # (48, 512)
        lo_peak   = np.max(plin_all[:, lo_mask], axis=1)        # (48,) linear
        hi_peak   = np.max(plin_all[:, hi_mask], axis=1)        # (48,) linear
        raw_lin   = (lo_peak + hi_peak) / 2.0                   # (48,) linear

        # Rolling 25th-percentile baseline per channel
        self._metric_history.append(raw_lin)
        if len(self._metric_history) > _METRIC_HIST_DEPTH:
            self._metric_history.pop(0)
        hist    = np.stack(self._metric_history)                 # (n_hist, 48)
        pct25   = np.maximum(np.percentile(hist, 25, axis=0), 1e-30)  # (48,)

        pair_metric = np.maximum(
            10.0 * np.log10(raw_lin / pct25), 0.0
        ).astype(np.float32)                                     # (48,) dB above baseline

        # Time-indexed SNR write — mirrors spectrogram boundary logic.
        ch_boundary = int(current_wall_time / self.history_secs)
        if ch_boundary != self._ch_snr_boundary:
            self._ch_snr_history[:] = 0.0
            self._ch_snr_boundary   = ch_boundary
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
            if ch_k in self._detect_cooldowns:
                continue

            # Estimate fc from peak bin locations in each window (linear domain)
            lo_freq   = float(sq_freq[lo_mask][np.argmax(plin_all[ch_k][lo_mask])])
            hi_freq   = float(sq_freq[hi_mask][np.argmax(plin_all[ch_k][hi_mask])])
            fc_offset = (lo_freq + hi_freq) / 4.0   # squaring doubled freqs
            fc_hz     = ch_k * CHANNEL_SPACING_HZ + fc_offset

            abs_snap      = self._iq_abs_sample
            ring_state_fn = lambda: (self._iq_ring_pos, self._iq_abs_sample)
            output_dir    = str(Path(__file__).parent.parent / 'MSK144' / 'detections')

            t = threading.Thread(
                target=extract_and_decode,
                args=(self._iq_ring, ring_state_fn,
                      abs_snap, self.sample_rate, fc_hz, output_dir,
                      t_in_window),
                daemon=True,
            )
            t.start()
            self._detect_cooldowns[ch_k] = cooldown_hops

    # ── 4. Normal overlap-FFT for waterfall display (raw, full bandwidth) ─────
    self.sample_buffer = np.concatenate(
        [self.sample_buffer, raw]
    )

    window      = np.hanning(self.fft_size)
    window_gain = np.sqrt(np.mean(window ** 2))
    full_scale  = 1.0

    while len(self.sample_buffer) >= self.fft_size:
        block             = self.sample_buffer[:self.fft_size]
        self.sample_buffer = self.sample_buffer[self.fft_size // 2:]

        X         = np.fft.fftshift(np.fft.fft(block * window))
        magnitude = np.abs(X) / (self.fft_size * window_gain)
        power_db  = 20.0 * np.log10(magnitude / full_scale + 1e-12)

        # Wall-clock / WAV-cursor time
        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
            wav_block_seconds = (self.fft_size // 2) / self.sample_rate
        else:
            current_wall_time = time.time()
            wav_block_seconds = 0.0
        time_in_window = current_wall_time % self.history_secs

        # ── Accumulated spectrogram ───────────────────────────────────────
        spec_boundary = int(current_wall_time / self.history_secs)
        if spec_boundary != self.spec_boundary:
            self.spectrogram_data    = self.spec_staging.copy()
            self.spec_staging_filled = True
            self.accumulated_noise_floor = np.percentile(self.spec_staging, 10, axis=0)
            self.spec_staging        = np.full((self.max_history, self.fft_size), -130.0)
            self.spec_boundary       = spec_boundary
            self.spec_write_index    = min(
                max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1
            )
            self.next_boundary = current_wall_time + (
                self.history_secs - (current_wall_time % self.history_secs)
            )

        if 0 <= self.spec_write_index < self.max_history:
            self.spec_staging[self.spec_write_index] = power_db
        self.spec_write_index += 1

        # ── Real-time spectrogram ─────────────────────────────────────────
        realtime_boundary = int(current_wall_time / self.history_secs)
        if realtime_boundary != self._realtime_boundary:
            self.realtime_data         = np.full((self.max_history, self.fft_size), -130.0)
            self._realtime_boundary    = realtime_boundary
            self.realtime_write_index  = min(
                max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1
            )

        if 0 <= self.realtime_write_index < self.max_history:
            self.realtime_data[self.realtime_write_index] = power_db
            self.realtime_filled = True
        self.realtime_write_index += 1

        if self.source_mode == "wav":
            self._wav_time_cursor = current_wall_time + wav_block_seconds
