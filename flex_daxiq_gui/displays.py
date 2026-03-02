"""Display update/render logic for spectrogram, noise-floor, and energy overlays."""

import datetime

import numpy as np
from PyQt5 import QtCore


def _format_bandwidth_hz(bandwidth_hz):
    if bandwidth_hz is None:
        return None
    try:
        bw = float(bandwidth_hz)
    except (TypeError, ValueError):
        return None
    if bw <= 0:
        return None
    return f"{int(round(bw / 1e3))} kHz"


def update_displays(self):
    """Update all display panels."""
    if len(self.realtime_data) == 0:
        return

    tuned_freq_mhz, tuned_source, tuned_bandwidth_hz = self._get_tuned_frequency_mhz()
    center_freq_mhz = tuned_freq_mhz if tuned_freq_mhz is not None else self.center_freq_mhz

    if abs(center_freq_mhz - self.display_center_freq_mhz) > 1e-9:
        self.display_center_freq_mhz = center_freq_mhz
        self.freq_axis = self.fft_bin_axis_mhz + center_freq_mhz

    freq_min = self.freq_axis[0]
    freq_max = self.freq_axis[-1]

    if self.spec_staging_filled:
        spec_array = self.spectrogram_data

        valid_acc_rows = np.any(spec_array > -129.5, axis=1)
        if np.any(valid_acc_rows):
            self.accumulated_noise_floor = np.percentile(spec_array[valid_acc_rows], 10, axis=0)

        self.spectrogram_img.setImage(
            spec_array,
            autoLevels=False,
            levels=[self.min_level, self.max_level],
        )

        self.spectrogram_img.setRect(
            QtCore.QRectF(
                0,
                freq_min,
                self.max_time,
                freq_max - freq_min,
            )
        )

        self.spectrogram_plot.setXRange(0, self.max_time, padding=0)
        self.spectrogram_plot.setYRange(freq_min, freq_max, padding=0)

    if self.realtime_filled:
        valid_rt_rows = np.any(self.realtime_data > -129.5, axis=1)
        if np.any(valid_rt_rows):
            self.realtime_noise_floor = np.percentile(self.realtime_data[valid_rt_rows], 10, axis=0)

        self.realtime_img.setImage(
            self.realtime_data,
            autoLevels=False,
            levels=[self.min_level, self.max_level],
        )

        self.realtime_img.setRect(
            QtCore.QRectF(
                0,
                freq_min,
                self.realtime_time,
                freq_max - freq_min,
            )
        )

        self.realtime_plot.setXRange(0, self.realtime_time, padding=0)
        self.realtime_plot.setYRange(freq_min, freq_max, padding=0)

    self.accumulated_noise_curve.setData(self.accumulated_noise_floor, self.freq_axis)
    self.accumulated_noise_plot.setXRange(self.min_level, self.max_level, padding=0)
    self.accumulated_noise_plot.setYRange(freq_min, freq_max, padding=0)

    self.realtime_noise_curve.setData(self.realtime_noise_floor, self.freq_axis)
    self.realtime_noise_plot.setXRange(self.min_level, self.max_level, padding=0)
    self.realtime_noise_plot.setYRange(freq_min, freq_max, padding=0)

    accumulated_valid_mask = ~np.isnan(self.accumulated_energy_buffer)
    if self.accumulated_energy_filled and np.any(accumulated_valid_mask):
        acc_energy_vals = self.accumulated_energy_buffer[accumulated_valid_mask]
        acc_y = self._map_energy_to_freq_band(acc_energy_vals, freq_min, freq_max)
        self.accumulated_energy_curve.setData(self.energy_time_axis[accumulated_valid_mask], acc_y)
    else:
        self.accumulated_energy_curve.setData([], [])

    valid_mask = ~np.isnan(self.realtime_energy_buffer)
    if np.any(valid_mask):
        energy_vals = self.realtime_energy_buffer[valid_mask]
        realtime_y = self._map_energy_to_freq_band(energy_vals, freq_min, freq_max)
        self.realtime_energy_curve.setData(self.energy_time_axis[valid_mask], realtime_y)
    else:
        self.realtime_energy_curve.setData([], [])

    # Squared-signal periodograms.
    # Y-axis uses kHz offset from band center; actual spectral content appears at
    # 2× those offsets because squaring doubled the component frequencies.
    sq_freq_min = self.sq_freq_axis_khz[0]
    sq_freq_max = self.sq_freq_axis_khz[-1]

    if self.sq_spec_staging_filled:
        self.sq_accumulated_img.setImage(
            self.sq_spectrogram_data,
            autoLevels=False,
            levels=[self.sq_min_level, self.sq_max_level],
        )
        self.sq_accumulated_img.setRect(
            QtCore.QRectF(0, sq_freq_min, self.max_time, sq_freq_max - sq_freq_min)
        )
        self.sq_accumulated_plot.setXRange(0, self.max_time, padding=0)
        self.sq_accumulated_plot.setYRange(sq_freq_min, sq_freq_max, padding=0)

    if self.sq_realtime_filled:
        self.sq_realtime_img.setImage(
            self.sq_realtime_data,
            autoLevels=False,
            levels=[self.sq_min_level, self.sq_max_level],
        )
        self.sq_realtime_img.setRect(
            QtCore.QRectF(0, sq_freq_min, self.realtime_time, sq_freq_max - sq_freq_min)
        )
        self.sq_realtime_plot.setXRange(0, self.realtime_time, padding=0)
        self.sq_realtime_plot.setYRange(sq_freq_min, sq_freq_max, padding=0)

    utc_time = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")
    self.utc_clock_label.setText(f"UTC: {utc_time}")

    if tuned_freq_mhz is not None:
        if tuned_source == "Pan":
            bw_text = _format_bandwidth_hz(tuned_bandwidth_hz)
            if bw_text:
                self.tuned_freq_label.setText(f"Pan Center: {tuned_freq_mhz:.3f} MHz BW: {bw_text}")
            else:
                self.tuned_freq_label.setText(f"Pan Center: {tuned_freq_mhz:.3f} MHz")
        else:
            self.tuned_freq_label.setText(f"Tuned ({tuned_source}): {tuned_freq_mhz:.3f} MHz")
        self.setWindowTitle(f'FlexRadio DAXIQ - {tuned_freq_mhz:.3f} MHz')
    else:
        self.tuned_freq_label.setText(f"Tuned: {self.center_freq_mhz:.3f} MHz (requested)")
        self.setWindowTitle(f'FlexRadio DAXIQ - {self.center_freq_mhz:.3f} MHz')

    if self.spec_staging_filled and len(self.spectrogram_data) > 0:
        data_min = np.min(self.spectrogram_data)
        data_max = np.max(self.spectrogram_data)
        data_mean = np.mean(self.spectrogram_data)

        packet_info = ''
        if hasattr(self, 'flex_client') and hasattr(self.flex_client, '_vita') and self.flex_client._vita:
            missed = self.flex_client._vita.missed_count
            total = self.flex_client._vita.packet_count
            if total > 0:
                loss_pct = (missed / total) * 100 if total > 0 else 0
                packet_info = f' | Packets: {total} (loss: {loss_pct:.2f}%)'

        self.statusBar().showMessage(
            f'Rate: {self.sample_rate/1000:.0f} kHz | '
            f'FFT: {self.fft_size} bins | '
            f'Power: {data_min:.1f} to {data_max:.1f} dB (avg {data_mean:.1f})'
            f'{packet_info}'
        )
    else:
        self.statusBar().showMessage(
            f'Rate: {self.sample_rate/1000:.0f} kHz | '
            f'FFT: {self.fft_size} bins | '
            f'Waiting for data...'
        )
