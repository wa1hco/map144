"""FFT and buffer update pipeline for incoming IQ sample data."""

import time

import numpy as np


def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
    """Process IQ samples and update data buffers using wall clock time."""
    self.sample_buffer = np.concatenate([self.sample_buffer, iq_samples])

    while len(self.sample_buffer) >= self.fft_size:
        block = self.sample_buffer[:self.fft_size]
        self.sample_buffer = self.sample_buffer[self.fft_size:]

        window = np.hanning(self.fft_size)
        window_gain = np.sqrt(np.mean(window**2))

        spectrum = np.fft.fftshift(np.fft.fft(block * window))

        # Internal level standard: full-scale reference is 32768 for complex float samples.
        # Source-specific conversion to this standard is handled at input ingestion.
        full_scale = 32768.0
        magnitude = np.abs(spectrum) / (self.fft_size * window_gain)
        power_db = 20 * np.log10(magnitude / full_scale + 1e-12)

        # Squared signal spectrum: squaring doubles all component frequencies.
        # For MSK144, tones at fc±500 Hz appear at 2fc±1000 Hz in this spectrum,
        # producing a distinctive ±1000 Hz symmetric pair regardless of center freq.
        block_sq = block ** 2
        spectrum_sq = np.fft.fftshift(np.fft.fft(block_sq * window))
        magnitude_sq = np.abs(spectrum_sq) / (self.fft_size * window_gain)
        power_db_sq = 20 * np.log10(magnitude_sq / full_scale ** 2 + 1e-12)

        if self.source_mode == "wav":
            current_wall_time = float(getattr(self, "_wav_time_cursor", 0.0))
            wav_block_seconds = self.fft_size / self.sample_rate
        else:
            current_wall_time = time.time()
            wav_block_seconds = 0.0
        time_in_window = current_wall_time % self.history_secs

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

        realtime_boundary = int(current_wall_time / self.history_secs)
        if realtime_boundary != self._realtime_boundary:
            self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)
            self._realtime_boundary = realtime_boundary
            self.realtime_write_index = min(max(int(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

        if 0 <= self.realtime_write_index < self.max_history:
            self.realtime_data[self.realtime_write_index] = power_db
            self.realtime_filled = True
        self.realtime_write_index += 1

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

        # Squared signal: accumulated snapshot buffer (mirrors spec_staging logic).
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

        # Squared signal: real-time buffer (mirrors realtime_data logic).
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
