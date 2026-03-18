"""FFT and buffer update pipeline for incoming IQ sample data."""

import threading
import time

import numpy as np

from .detection import (
    apply_lp_filter,
    extract_and_decode,
    fc_from_sq_pair,
    scan_for_pairs,
)

DETECT_MERGE_GAP_S  = 0.4       # suppress re-trigger within this window (seconds)
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

    # ── Feed filtered samples into overlap-FFT buffer ────────────────────────
    self.sample_buffer = np.concatenate([self.sample_buffer, filtered])

    # Precompute window (same each block)
    window = np.hanning(self.fft_size)
    window_gain = np.sqrt(np.mean(window ** 2))
    full_scale = 32768.0

    while len(self.sample_buffer) >= self.fft_size:
        block = self.sample_buffer[:self.fft_size]
        self.sample_buffer = self.sample_buffer[self.fft_size // 2:]

        # ── Normal spectrum ───────────────────────────────────────────────────
        X = np.fft.fftshift(np.fft.fft(block * window))
        magnitude = np.abs(X) / (self.fft_size * window_gain)
        power_db = 20 * np.log10(magnitude / full_scale + 1e-12)

        # ── Squared spectrum (time-domain squaring then FFT) ──────────────────
        block_sq = block ** 2
        X_sq = np.fft.fftshift(np.fft.fft(block_sq * window))
        magnitude_sq = np.abs(X_sq) / (self.fft_size * window_gain)
        power_db_sq = 10 * np.log10(magnitude_sq / (full_scale ** 2) + 1e-12) - 15.0

        # ── Wall-clock / WAV-cursor time ──────────────────────────────────────
        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
            wav_block_seconds = (self.fft_size // 2) / self.sample_rate
        else:
            current_wall_time = time.time()
            wav_block_seconds = 0.0
        time_in_window = current_wall_time % self.history_secs

        # ── Tone-pair detection ───────────────────────────────────────────────
        if self._detect_cooldown <= 0:
            sq_freq_hz = self.sq_freq_axis_khz * 1000.0
            pairs = scan_for_pairs(
                power_db_sq, sq_freq_hz,
                spacing_hz=_DETECT_SPACING_HZ,
                tol_hz=_DETECT_TOL_HZ,
                thresh_db=_DETECT_THRESH_DB,
            )
            if pairs:
                f_sq_lo, f_sq_hi = pairs[0]
                fc_hz = fc_from_sq_pair(f_sq_lo, f_sq_hi)
                # Snapshot ring buffer state for the background thread
                ring_snap     = self._iq_ring.copy()
                ring_pos_snap = self._iq_ring_pos
                abs_snap      = self._iq_abs_sample
                output_dir    = str(__import__('pathlib').Path(__file__).parent.parent / 'MSK144' / 'detections')
                t = threading.Thread(
                    target=extract_and_decode,
                    args=(ring_snap, ring_pos_snap, abs_snap,
                          abs_snap, self.sample_rate, fc_hz, output_dir,
                          time_in_window),
                    daemon=True,
                )
                t.start()
                hop = self.fft_size // 2
                self._detect_cooldown = max(1, int(DETECT_MERGE_GAP_S * self.sample_rate / hop))
        else:
            self._detect_cooldown -= 1

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

        # ── Energy overlay ────────────────────────────────────────────────────
        total_energy = float(np.max(power_db))
        energy_boundary = int(current_wall_time / self.history_secs)
        if energy_boundary != self.energy_boundary:
            self.accumulated_energy_buffer = self.realtime_energy_buffer.copy()
            self.accumulated_energy_filled = np.any(~np.isnan(self.accumulated_energy_buffer))
            self.realtime_energy_buffer = np.full(self.max_history, np.nan)
            self.energy_boundary = energy_boundary
            self.energy_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        if 0 <= self.energy_write_index < self.max_history:
            self.realtime_energy_buffer[self.energy_write_index] = total_energy
        self.energy_write_index += 1

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
