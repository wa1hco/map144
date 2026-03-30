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

This module implements ``update_displays``, mixed into ``MAP144Visualizer`` as
a method and called by a Qt 100 ms refresh timer.  It reads the shared NumPy
buffers written by ``processing.py`` and pushes updated data to the five
pyqtgraph plot widgets created by ``ui.py``.  No signal processing is done
here; this module is purely concerned with rendering and status text.

Panel layout (matches the grid built in ``ui.py``)
---------------------------------------------------
Row 0  Accumulated IQ spectrogram  |  Decode panel (spans rows 0–1)
Row 1  Real-time IQ spectrogram    |  (covered by decode panel rowspan)
Row 2  IQ colour-scale slider bar
Row 3  Channel detection SNR heatmap
Row 4  Detection colour-scale slider bar

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

    self._noise_floor_ctr = getattr(self, '_noise_floor_ctr', 0) + 1

    if self.spec_staging_filled:
        spec_array = self.spectrogram_data

        if self._noise_floor_ctr % 10 == 1:   # recompute ~once per second
            valid_acc_rows = np.any(spec_array > -129.5, axis=1)
            if np.any(valid_acc_rows):
                _a = spec_array[valid_acc_rows]
                _k = max(0, int(len(_a) * 0.10))
                self.accumulated_noise_floor = np.partition(_a, _k, axis=0)[_k]

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

    if self.realtime_filled and getattr(self, '_realtime_dirty', False):
        self._realtime_dirty = False

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

        # setXRange/setYRange must come AFTER setImage and setRect.
        # sigBoundsChanged fires synchronously on those calls and triggers
        # autoRange; an explicit setXRange after that overrides the result.
        # (The accumulated spectrogram has always used this order and works.)
        self.realtime_plot.setXRange(0, self.realtime_time, padding=0)
        self.realtime_plot.setYRange(freq_min, freq_max, padding=0)

    # ── jt9 decode results — always drain the queue so decode_panel stays current
    all_markers = list(getattr(self, '_jt9_markers', []))
    _marker_by_id = {m['id']: m for m in all_markers if isinstance(m, dict)}

    dq = getattr(self, '_decode_queue', None)
    if dq is not None:
        while not dq.empty():
            try:
                result = dq.get_nowait()
            except Exception:
                break
            mid = result.get('marker_id', -1)
            if mid in _marker_by_id:
                _marker_by_id[mid]['decoded'] = True
                _marker_by_id[mid]['message'] = result.get('message')
            # Prepend to decode panel (most recent at top)
            msg       = result.get('message', '?')
            radio_khz = result.get('radio_khz', 0.0)
            snr       = result.get('jt9_snr')
            utc_time  = result.get('utc_time', '')
            rf_mhz    = radio_khz / 1000.0
            snr_str   = f"{snr:+d} dB" if snr is not None else "  ?"
            self.decode_panel.insertItem(0, f"{utc_time}  {rf_mhz:.3f}  {snr_str:>7}  {msg}")

    # ── Detection heatmap + markers — skip when panel is hidden ──────────────
    _detect_win = getattr(self, '_detect_win', None)
    if _detect_win is None or _detect_win.isVisible():
        # fftshift channels so image row 0 = -24 kHz, row 47 = +23 kHz.
        hm_data = np.fft.fftshift(self._ch_snr_history, axes=1)   # (N_SNR_HIST, N_CH)
        self.ch_detect_img.setImage(
            hm_data,
            autoLevels=False,
            levels=[self.detect_min_level, self.detect_max_level],
        )
        self.ch_detect_img.setRect(QtCore.QRectF(
            0.0,
            self._detect_freq_min_khz,
            float(self.history_secs),
            self._detect_freq_span_khz,
        ))
        self.ch_detect_plot.setXRange(0, float(self.history_secs), padding=0)

        # ── jt9 launch markers ────────────────────────────────────────────────
        # Red = launched / not yet decoded.  Green = successfully decoded.
        markers = [m for m in all_markers
                   if isinstance(m, dict) and m.get('boundary') == self._ch_snr_boundary]

        # Only rebuild circle paths when marker state changes
        n_decoded   = sum(1 for m in markers if m.get('decoded'))
        marker_fp   = (self._ch_snr_boundary, len(markers), n_decoded)
        prev_fp     = getattr(self, '_marker_fp', None)
        if marker_fp != prev_fp:
            self._marker_fp = marker_fp

            r_y   = 2.5   # kHz radius (5 kHz diameter)
            px_x, px_y = self.ch_detect_plot.getViewBox().viewPixelSize()
            r_x   = r_y * (px_x / px_y) if px_y > 0 else r_y * 0.1
            theta = np.linspace(0, 2 * np.pi, 33)

            def _circle_path(mlist):
                xs, ys = [], []
                for m in mlist:
                    xs.append(m.get('t', 0.0)        + r_x * np.cos(theta))
                    ys.append(m.get('freq_khz', 0.0) + r_y * np.sin(theta))
                    xs.append([np.nan])
                    ys.append([np.nan])
                return np.concatenate(xs), np.concatenate(ys)

            undecoded = [m for m in markers if not m['decoded']]
            decoded   = [m for m in markers if     m['decoded']]

            if undecoded:
                self.ch_detect_curve_red.setData(*_circle_path(undecoded), connect='finite')
            else:
                self.ch_detect_curve_red.setData(x=[], y=[])

            if decoded:
                self.ch_detect_curve_green.setData(*_circle_path(decoded), connect='finite')
            else:
                self.ch_detect_curve_green.setData(x=[], y=[])

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
        self.setWindowTitle(f'map144 - {tuned_freq_mhz:.3f} MHz')
    else:
        self.tuned_freq_label.setText(f"Tuned: {self.center_freq_mhz:.3f} MHz (requested)")
        self.setWindowTitle(f'map144 - {self.center_freq_mhz:.3f} MHz')

    if self.spec_staging_filled and len(self.spectrogram_data) > 0:
        data_min = np.min(self.spectrogram_data)
        data_max = np.max(self.spectrogram_data)
        data_mean = np.mean(self.spectrogram_data)

        packet_info = ''
        if hasattr(self, 'radio_client') and hasattr(self.radio_client, '_vita') and self.radio_client._vita:
            missed = self.radio_client._vita.missed_count
            total = self.radio_client._vita.packet_count
            drops = self.radio_client._vita.drop_count
            if total > 0:
                loss_pct = (missed / total) * 100 if total > 0 else 0
                drop_str = f'  drops: {drops}' if drops > 0 else ''
                packet_info = f' | Packets: {total} (loss: {loss_pct:.2f}%{drop_str})'

        _nb_total = getattr(self, '_nb_total_count', 0)
        _nb_blank = getattr(self, '_nb_blanked_count', 0)
        if _nb_total > 0 and _nb_blank > 0:
            nb_str = f' | NB: {100.0 * _nb_blank / _nb_total:.2f}%'
        else:
            nb_str = ''
        self.statusBar().showMessage(
            f'Rate: {self.sample_rate/1000:.0f} kHz | '
            f'FFT: {self.fft_size} bins | '
            f'Power: {data_min:.1f} to {data_max:.1f} dB (avg {data_mean:.1f})'
            f'{packet_info}{nb_str}'
        )
    else:
        self.statusBar().showMessage(
            f'Rate: {self.sample_rate/1000:.0f} kHz | '
            f'FFT: {self.fft_size} bins | '
            f'Waiting for data...'
        )

    # ── Source windows (IQ/NB + per-radio panels) ────────────────────────────
    from .source_windows import update_source_windows
    update_source_windows(self)
