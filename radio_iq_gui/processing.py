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
"""FFT and buffer update pipeline for incoming IQ sample data.

This module implements ``process_iq_data``, the per-chunk DSP callback that
drives all spectral display and MSK144 detection work.  It is mixed into
``RadioIQVisualizer`` as a method and is called once per incoming IQ chunk from
either the radio IQ stream or a WAV file replay.

Pipeline (in order of execution)
---------------------------------
1. **LP filter** — a streaming FIR low-pass filter (10 kHz cutoff, designed in
   ``detection.design_lp_filter``) is applied to each incoming chunk.  Filter
   state (``_lp_zi_re`` / ``_lp_zi_im``) is preserved across calls so there
   are no boundary artefacts between chunks.

2. **Ring buffer** — the LP-filtered complex samples are written into a 5-second
   circular buffer (``_iq_ring``).  When a signal is detected this buffer
   supplies the raw IQ snapshot that ``extract_and_decode`` uses to reconstruct
   a full 15-second MSK144 transmission window.

3. **Overlap-FFT loop** — samples are accumulated in ``sample_buffer`` and
   processed in blocks of ``fft_size`` with 50 % overlap (hop = fft_size // 2).
   Each block produces two spectra:

   * **Normal spectrum** — Hanning-windowed FFT, magnitude normalised to
     ``full_scale = 1.0`` (0 dBFS = ±1.0).  All sources are normalised to the
     ±1.0 convention at ingress in ``runtime.py`` before reaching this
     function.  Used for the main waterfall and real-time spectrogram displays.

   * **Squared spectrum** — the time-domain block is squared (``block²``) before
     the FFT.  Squaring a two-tone MSK signal at fc±Δf produces spectral lines
     at ±2Δf, collapsing the carrier and both sidebands into a single detectable
     peak pair regardless of carrier phase.  Full-scale reference becomes
     ``full_scale²``; an empirical −15 dB display offset is applied.  Used for
     the squared-spectrum waterfall and tone-pair detection.

4. **Tone-pair detection** — every FFT block the squared spectrum is scanned for
   symmetric peak pairs separated by ``_DETECT_SPACING_HZ`` (2000 Hz, i.e. the
   2×1000 Hz MSK144 deviation after squaring).  On a hit the inferred carrier
   frequency is computed and ``extract_and_decode`` is launched in a daemon
   thread with a snapshot of the ring buffer.  A cooldown counter
   (``DETECT_MERGE_GAP_S = 0.4 s``) suppresses re-triggers from the same burst.

5. **Display buffer management** — four rolling buffers are updated each block,
   all keyed to 15-second wall-clock windows:

   ==========================================  =====================================
   Buffer pair                                 Content
   ==========================================  =====================================
   ``spec_staging`` / ``spectrogram_data``     Accumulated normal spectrum (previous window)
   ``realtime_data``                           Normal spectrum (current window, live)
   ``sq_spec_staging`` / ``sq_spectrogram_data``  Accumulated squared spectrum
   ``sq_realtime_data``                        Squared spectrum (current window, live)
   ==========================================  =====================================

   At each 15-second boundary the staging buffer is promoted to the display
   buffer and a fresh staging buffer is initialised to −130 dBFS.  The
   accumulated noise floor is estimated as the 10th-percentile bin value across
   the completed window.

6. **WAV cursor** — in ``source_mode == "wav"`` wall-clock time is simulated by
   advancing ``_wav_time_cursor`` by one hop's worth of seconds per block,
   allowing WAV replay to drive the same time-windowed display logic as live
   DAXIQ without touching ``time.time()``.

Constants
---------
DETECT_MERGE_GAP_S  : 0.4 s  — minimum gap between successive decode triggers
_DETECT_THRESH_DB   : 10 dB  — peak must exceed median noise by this amount
_DETECT_SPACING_HZ  : 2000 Hz — expected tone-pair spacing in squared domain
_DETECT_TOL_HZ      : 200 Hz — tolerance when searching for the paired peak
"""

import threading
import time

import numpy as np

from .detection import (
    apply_lp_filter,
    extract_and_decode,
    fc_from_sq_pair,
    scan_for_pairs,
)

DETECT_MERGE_GAP_S  = 1.0       # suppress re-trigger within this window (seconds)
                                # MSK144 tails decay for ~4× ping width; at max width 300 ms
                                # the tail lasts ~900 ms, so 1.0 s prevents re-triggers on
                                # the same burst while still allowing signals ≥1 s apart
_DETECT_THRESH_DB   = 10.0      # dB above median noise floor
_DETECT_SPACING_HZ  = 2000.0    # squared-domain tone-pair spacing
_DETECT_TOL_HZ      = 200.0     # search tolerance


def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
    """Process IQ samples and update data buffers using wall clock time."""

    # ── LP filter (streaming state maintained between calls) ─────────────────
    filtered, self._lp_zi_re, self._lp_zi_im = apply_lp_filter(
        iq_samples, self._lp_taps, self._lp_zi_re, self._lp_zi_im
    )

    # ── Write LP-filtered samples into the circular ring buffer ──────────────
    chunk_len = len(filtered)
    ring_size = len(self._iq_ring)
    pos = self._iq_ring_pos
    if pos + chunk_len <= ring_size:
        self._iq_ring[pos:pos + chunk_len] = filtered
    else:
        first_n = ring_size - pos
        self._iq_ring[pos:] = filtered[:first_n]
        self._iq_ring[:chunk_len - first_n] = filtered[first_n:]
    self._iq_ring_pos = (pos + chunk_len) % ring_size
    self._iq_abs_sample += chunk_len

    # ── Feed samples into overlap-FFT buffers ────────────────────────────────
    # raw_buffer  → unfiltered IQ for the normal IQ display (full bandwidth)
    # sample_buffer → LP-filtered IQ for the squared display and detection
    self.raw_buffer    = np.concatenate([self.raw_buffer,    iq_samples.astype(np.complex64)])
    self.sample_buffer = np.concatenate([self.sample_buffer, filtered])

    # Precompute window (same each block)
    window = np.hanning(self.fft_size)
    window_gain = np.sqrt(np.mean(window ** 2))
    full_scale = 1.0  # dBFS reference: 0 dBFS = ±1.0 (all sources normalised at ingress)

    while len(self.sample_buffer) >= self.fft_size:
        raw_block = self.raw_buffer[:self.fft_size]
        self.raw_buffer    = self.raw_buffer[self.fft_size // 2:]

        block = self.sample_buffer[:self.fft_size]
        self.sample_buffer = self.sample_buffer[self.fft_size // 2:]

        # ── Normal spectrum (uses unfiltered raw IQ — full receive bandwidth) ─
        X = np.fft.fftshift(np.fft.fft(raw_block * window))
        magnitude = np.abs(X) / (self.fft_size * window_gain)
        power_db = 20 * np.log10(magnitude / full_scale + 1e-12)

        # ── Squared spectrum (uses LP-filtered IQ — 10 kHz detection band) ────
        block_sq = block ** 2
        X_sq = np.fft.fftshift(np.fft.fft(block_sq * window))
        magnitude_sq = np.abs(X_sq) / (self.fft_size * window_gain)
        power_db_sq = 10 * np.log10(magnitude_sq / (full_scale ** 2) + 1e-12)

        # ── Wall-clock / WAV-cursor time ──────────────────────────────────────
        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
            wav_block_seconds = (self.fft_size // 2) / self.sample_rate
        else:
            current_wall_time = time.time()
            wav_block_seconds = 0.0
        time_in_window = current_wall_time % self.history_secs

        # ── Tone-pair detection ───────────────────────────────────────────────
        hop = self.fft_size // 2
        # Tick down all per-frequency cooldowns and discard expired entries
        self._detect_cooldowns = {
            k: v - 1 for k, v in self._detect_cooldowns.items() if v > 1
        }
        sq_freq_hz = self.sq_freq_axis_khz * 1000.0
        pairs = scan_for_pairs(
            power_db_sq, sq_freq_hz,
            spacing_hz=_DETECT_SPACING_HZ,
            tol_hz=_DETECT_TOL_HZ,
            thresh_db=_DETECT_THRESH_DB,
        )
        for f_sq_lo, f_sq_hi in pairs:
            fc_hz = fc_from_sq_pair(f_sq_lo, f_sq_hi)
            # Quantise to nearest 500 Hz to group re-triggers of the same signal
            freq_key = round(fc_hz / 500) * 500
            if freq_key in self._detect_cooldowns:
                continue    # same signal still ringing — skip
            # New signal at this frequency: launch decode thread
            abs_snap      = self._iq_abs_sample
            ring_state_fn = lambda: (self._iq_ring_pos, self._iq_abs_sample)
            output_dir    = str(__import__('pathlib').Path(__file__).parent.parent / 'MSK144' / 'detections')
            t = threading.Thread(
                target=extract_and_decode,
                args=(self._iq_ring, ring_state_fn,
                      abs_snap, self.sample_rate, fc_hz, output_dir,
                      time_in_window),
                daemon=True,
            )
            t.start()
            self._detect_cooldowns[freq_key] = max(1, int(DETECT_MERGE_GAP_S * self.sample_rate / hop))

        # ── Accumulated spectrogram ───────────────────────────────────────────
        spec_boundary = int(current_wall_time / self.history_secs)
        if spec_boundary != self.spec_boundary:
            self.spectrogram_data = self.spec_staging.copy()
            self.spec_staging_filled = True
            self.accumulated_noise_floor = np.percentile(self.spec_staging, 10, axis=0)
            self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)
            self.spec_boundary = spec_boundary
            self.spec_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)
            self.next_boundary = current_wall_time + (self.history_secs - (current_wall_time % self.history_secs))

        if 0 <= self.spec_write_index < self.max_history:
            self.spec_staging[self.spec_write_index] = power_db
        self.spec_write_index += 1

        # ── Real-time spectrogram ─────────────────────────────────────────────
        realtime_boundary = int(current_wall_time / self.history_secs)
        if realtime_boundary != self._realtime_boundary:
            self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)
            self._realtime_boundary = realtime_boundary
            self.realtime_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        if 0 <= self.realtime_write_index < self.max_history:
            self.realtime_data[self.realtime_write_index] = power_db
            self.realtime_filled = True
        self.realtime_write_index += 1

        # ── Accumulated squared spectrogram ───────────────────────────────────
        sq_spec_boundary = int(current_wall_time / self.history_secs)
        if sq_spec_boundary != self.sq_spec_boundary:
            self.sq_spectrogram_data = self.sq_spec_staging.copy()
            self.sq_spec_staging_filled = True
            self.sq_spec_staging = np.full((self.max_history, self.fft_size), -130.0)
            self.sq_spec_boundary = sq_spec_boundary
            self.sq_spec_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        if 0 <= self.sq_spec_write_index < self.max_history:
            self.sq_spec_staging[self.sq_spec_write_index] = power_db_sq
        self.sq_spec_write_index += 1

        # ── Real-time squared spectrogram ─────────────────────────────────────
        sq_realtime_boundary = int(current_wall_time / self.history_secs)
        if sq_realtime_boundary != self._sq_realtime_boundary:
            self.sq_realtime_data = np.full((self.max_history, self.fft_size), -130.0)
            self._sq_realtime_boundary = sq_realtime_boundary
            self.sq_realtime_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        if 0 <= self.sq_realtime_write_index < self.max_history:
            self.sq_realtime_data[self.sq_realtime_write_index] = power_db_sq
            self.sq_realtime_filled = True
        self.sq_realtime_write_index += 1

        if self.source_mode == "wav":
            self._wav_time_cursor = current_wall_time + wav_block_seconds
