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
"""Display update/render logic for spectrogram, noise-floor, and energy overlays.

This module implements ``update_displays``, mixed into ``RadioIQVisualizer`` as
a method and called by a Qt 100 ms refresh timer.  It reads the shared NumPy
buffers written by ``processing.py`` and pushes updated data to the five
pyqtgraph plot widgets created by ``ui.py``.  No signal processing is done
here; this module is purely concerned with rendering and status text.

Panel layout (matches the grid built in ``ui.py``)
---------------------------------------------------
Row 0  Accumulated IQ spectrogram  |  Accumulated noise-floor curve
Row 1  Real-time IQ spectrogram    |  Real-time noise-floor curve
Row 2  (IQ slider bar — not drawn here)
Row 3  Squared-signal accumulated spectrogram
Row 4  Squared-signal real-time spectrogram

Rendering details
-----------------
Frequency axis
    On each call the current tuned frequency is fetched from the radio client
    via ``_get_tuned_frequency_mhz()`` (or ``center_freq_mhz`` for WAV mode).
    If the centre frequency has shifted since the last frame the ``freq_axis``
    array is recomputed in-place, keeping all display rects in sync without
    rebuilding any Qt objects.

IQ spectrograms (``spectrogram_img``, ``realtime_img``)
    ``setImage`` is called with ``autoLevels=False``; the colour-scale limits
    come from ``min_level`` / ``max_level`` (dBFS), which are updated live by
    the slider handlers in ``ui.py``.  The image rect maps the time axis to
    [0, history_secs] and the frequency axis to [freq_min, freq_max] in MHz.
    The accumulated panel is only refreshed when ``spec_staging_filled`` is
    True (i.e. at least one 15-second window has completed).

Noise-floor curves (``accumulated_noise_curve``, ``realtime_noise_curve``)
    The 10th-percentile bin values across valid rows are re-estimated each
    frame and plotted as power (dBFS) vs. frequency (MHz).  Rows filled with
    the sentinel value −130 dBFS are excluded from the percentile calculation
    to avoid pulling the floor down to the init value.

Squared-signal spectrograms (``sq_accumulated_img``, ``sq_realtime_img``)
    Rendered identically to the IQ spectrograms but using the squared-domain
    buffers.  The Y-axis is labelled in kHz relative to band centre; note that
    the actual RF offset of detected tones is half the displayed kHz value
    because squaring doubles all spectral component frequencies.

Status bar and labels
    - UTC clock updated every frame (``datetime.UTC`` aware).
    - Tuned frequency label distinguishes Slice, Pan (with bandwidth), and WAV
      sources; the window title tracks the tuned frequency.
    - Status bar shows sample rate, FFT size, min/max/mean power, and (when
      available) VITA-49 packet count and loss percentage read from the radio
      client's ``_vita`` tracker.

Helper
------
_format_bandwidth_hz(bandwidth_hz)
    Converts a raw Hz value to a rounded kHz string (e.g. ``"48 kHz"``).
    Returns ``None`` for invalid or non-positive inputs so callers can omit
    the bandwidth field gracefully.
"""

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

    # realtime_noise_plot is now cursor-driven (IQ freq slice); Y range kept in sync
    self.realtime_noise_plot.setXRange(self.min_level, self.max_level, padding=0)
    self.realtime_noise_plot.setYRange(freq_min, freq_max, padding=0)

    # Pin IQ time-slice X range to the realtime window
    self.iq_time_slice_plot.setXRange(0, self.realtime_time, padding=0)

    # ── Channel detection ridgeline ───────────────────────────────────────────
    # Show data from t=0 growing rightward; curves reset each 15-s boundary.
    w_idx  = min(self._ch_snr_write_idx, self._ch_snr_history.shape[0])
    t_axis = self._ch_time_axis[:w_idx]
    offset = self._ch_display_offset

    for ch_k, curve in enumerate(self._ch_curves):
        snr_k = self._ch_snr_history[:w_idx, ch_k].astype(np.float32)
        pos   = self._ch_display_pos[ch_k]
        curve.setData(t_axis, snr_k + pos * offset)

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
        self.setWindowTitle(f'Radio IQ - {tuned_freq_mhz:.3f} MHz')
    else:
        self.tuned_freq_label.setText(f"Tuned: {self.center_freq_mhz:.3f} MHz (requested)")
        self.setWindowTitle(f'Radio IQ - {self.center_freq_mhz:.3f} MHz')

    if self.spec_staging_filled and len(self.spectrogram_data) > 0:
        data_min = np.min(self.spectrogram_data)
        data_max = np.max(self.spectrogram_data)
        data_mean = np.mean(self.spectrogram_data)

        packet_info = ''
        if hasattr(self, 'radio_client') and hasattr(self.radio_client, '_vita') and self.radio_client._vita:
            missed = self.radio_client._vita.missed_count
            total = self.radio_client._vita.packet_count
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
