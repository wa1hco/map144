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
"""UI layout and slider handlers for the radio IQ visualizer.

This module is responsible for constructing every Qt widget in the main window
and wiring interactive controls to their handler callbacks.  ``setup_ui`` is
mixed into ``RadioIQVisualizer`` and called once from ``__init__``.  All widget
references created here are stored as instance attributes so that
``displays.py`` can update them from the 100 ms refresh timer.

Window structure
----------------
Menu bar
    File menu with a mutually exclusive action group:
      - "Flex Radio" (default, checked on start) → ``on_select_source_radio``
      - "WAV File" → opens a file dialog via ``on_select_source_wav``

Central widget — QGridLayout (5 rows × 2 columns)
    Row 0, col 0  ``spectrogram_plot``       Accumulated IQ waterfall (15 s snapshot)
    Row 0, col 1  ``accumulated_noise_plot`` Noise-floor vs. frequency (accumulated)
    Row 1, col 0  ``realtime_plot``          Real-time IQ waterfall (current 15 s window)
    Row 1, col 1  ``realtime_noise_plot``    Noise-floor vs. frequency (real-time)
    Row 2, col 0–1 IQ colour-scale slider bar (spans both columns)
    Row 3, col 0  ``sq_accumulated_plot``    Squared-signal accumulated waterfall
    Row 4, col 0  ``sq_realtime_plot``       Squared-signal real-time waterfall

    Column stretch is 4:1 (waterfall wide, noise plot narrow).
    Rows 0, 1, 3, 4 each carry equal stretch weight; row 2 is fixed height.

Plot widgets and image items
    Each waterfall uses a ``pg.PlotWidget`` containing a ``pg.ImageItem`` with
    ``axisOrder='col-major'`` so that time advances left→right and frequency
    runs bottom→top.  Both normal and squared waterfalls share the same 9-stop
    black-blue-cyan-white colour map, providing good dynamic range for signals
    that are 30–60 dB above the noise floor.

    The two noise-floor plots display a single ``PlotCurveItem`` (yellow for
    accumulated, cyan for real-time) drawn as power (dBFS) on the X-axis vs.
    frequency (MHz) on the Y-axis, matching the orientation of the adjacent
    waterfalls.

    Energy overlay curves (green, 2 px wide) are added to both IQ waterfalls.
    They are populated by ``displays.py`` and are not further configured here.

Colour map
    Nine stops from pure black (−∞ / floor) through deep blue, cyan, yellow,
    to white (0 dBFS / full scale).  Positions are uniformly spaced at 0.125
    intervals.  The same ``pg.ColorMap`` instance is applied to all four
    ``ImageItem`` objects.

Slider bar helper (_make_slider_bar)
    A private factory function that builds a compact horizontal QWidget
    containing a bold title label, a "Min" label + slider, and a "Max" label
    + slider.  Label references are stored on ``self`` via ``setattr`` so that
    the value-changed handlers can update the text without searching the widget
    tree.  Fixed height of 54 px keeps the bar from consuming vertical space.

Interactive controls
--------------------
IQ colour scale sliders
    ``on_min_level_changed`` / ``on_max_level_changed`` — update ``self.min_level``
    and ``self.max_level`` (dBFS); ``displays.py`` reads these on every frame.

Squared-spectrum colour scale sliders
    ``on_sq_min_level_changed`` / ``on_sq_max_level_changed`` — update
    ``self.sq_min_level`` / ``self.sq_max_level``.  (Currently the squared
    waterfalls share the IQ colour limits; these handlers are wired for future
    independent scaling.)

Source selection
    ``on_select_source_radio`` — sets ``source_mode = "radio"``, clears
    ``selected_wav_path``; ``run_radio_source`` picks this up on the next loop.

    ``on_select_source_wav`` — opens a ``QFileDialog``; on cancellation
    reverts to Flex Radio mode.  On a valid selection sets ``source_mode =
    "wav"`` and ``selected_wav_path`` for ``run_radio_source`` to load.

Status bar
    Left area: scrolling status text (initialising → live power / packet stats).
    Permanent right-side widgets: bold tuned-frequency label and UTC clock
    label, both updated by ``displays.py`` on every timer tick.

Display timer
    A ``QTimer`` firing every 100 ms is created here and connected to
    ``update_displays``; it is stopped in ``closeEvent`` (``runtime.py``).
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


def setup_ui(self):
    """Create the five-panel display layout."""
    self.setWindowTitle(f'Radio IQ - {self.center_freq_mhz:.3f} MHz')
    self.setGeometry(100, 100, 1400, 1600)

    menu_bar = self.menuBar()
    file_menu = menu_bar.addMenu("&File")

    self.source_action_group = QtWidgets.QActionGroup(self)
    self.source_action_group.setExclusive(True)

    self.source_radio_action = QtWidgets.QAction("Flex Radio", self)
    self.source_radio_action.setCheckable(True)
    self.source_radio_action.setChecked(False)
    self.source_radio_action.triggered.connect(self.on_select_source_radio)
    self.source_action_group.addAction(self.source_radio_action)
    file_menu.addAction(self.source_radio_action)

    self.source_wav_action = QtWidgets.QAction("WAV File", self)
    self.source_wav_action.setCheckable(True)
    self.source_wav_action.triggered.connect(self.on_select_source_wav)
    self.source_action_group.addAction(self.source_wav_action)
    file_menu.addAction(self.source_wav_action)

    central = QtWidgets.QWidget()
    self.setCentralWidget(central)
    layout = QtWidgets.QGridLayout(central)

    pg.setConfigOptions(antialias=True)

    self.spectrogram_plot = pg.PlotWidget(title="Accumulated (15 sec snapshot)")
    self.spectrogram_plot.setLabel('left', 'Frequency', units='MHz')
    self.spectrogram_plot.setLabel('bottom', 'Time', units='s')
    self.spectrogram_img = pg.ImageItem(axisOrder='col-major')
    self.spectrogram_plot.addItem(self.spectrogram_img)
    self.spectrogram_plot.setAspectLocked(False)

    self.realtime_plot = pg.PlotWidget(title="Real-time (15 sec)")
    self.realtime_plot.setLabel('left', 'Frequency', units='MHz')
    self.realtime_plot.setLabel('bottom', 'Time', units='s')
    self.realtime_img = pg.ImageItem(axisOrder='col-major')
    self.realtime_plot.addItem(self.realtime_img)
    self.realtime_plot.setAspectLocked(False)

    colors = [
        (0, 0, 0),
        (0, 0, 64),
        (0, 0, 128),
        (0, 64, 192),
        (0, 128, 255),
        (64, 192, 255),
        (128, 255, 255),
        (255, 255, 128),
        (255, 255, 255),
    ]

    positions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    colormap = pg.ColorMap(positions, colors)
    self.spectrogram_img.setColorMap(colormap)
    self.realtime_img.setColorMap(colormap)

    _SIDE_WIDTH = 260   # fixed pixel width for all column-1 slice/noise plots

    _SIDE_Y_AXIS_W = 52   # fixed pixel width for left-axis tick labels inside pyqtgraph

    _black_pen = pg.mkPen('k')

    def _light_background(p):
        """White background, black axes and tick labels for slice/noise plots."""
        p.setBackground('w')
        for ax in ('left', 'bottom'):
            p.getAxis(ax).setPen(_black_pen)
            p.getAxis(ax).setTextPen(_black_pen)

    self.accumulated_noise_plot = pg.PlotWidget(title="Accumulated Noise Floor")
    self.accumulated_noise_plot.setLabel('bottom', 'Power', units='dB')
    self.accumulated_noise_plot.setLabel('left', 'Frequency', units='MHz')
    self.accumulated_noise_plot.showGrid(x=True, y=True, alpha=0.3)
    self.accumulated_noise_plot.setFixedWidth(_SIDE_WIDTH)
    self.accumulated_noise_plot.getAxis('left').setWidth(_SIDE_Y_AXIS_W)
    _light_background(self.accumulated_noise_plot)
    self.accumulated_noise_curve = self.accumulated_noise_plot.plot(pen='y', width=2)

    # Repurposed as cursor-driven IQ freq slice (power vs frequency at cursor time)
    self.realtime_noise_plot = pg.PlotWidget(title="IQ Freq Slice")
    self.realtime_noise_plot.setLabel('bottom', 'Power', units='dB')
    self.realtime_noise_plot.setLabel('left', 'Frequency', units='MHz')
    self.realtime_noise_plot.showGrid(x=True, y=True, alpha=0.3)
    self.realtime_noise_plot.setFixedWidth(_SIDE_WIDTH)
    self.realtime_noise_plot.getAxis('left').setWidth(_SIDE_Y_AXIS_W)
    _light_background(self.realtime_noise_plot)
    self._iq_freq_slice_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=1))
    self.realtime_noise_plot.addItem(self._iq_freq_slice_curve)

    # IQ time-slice plot: power (dB) vs time (s) at cursor frequency
    self.iq_time_slice_plot = pg.PlotWidget(title="IQ Time Slice")
    self.iq_time_slice_plot.setLabel('left', 'Power', units='dB')
    self.iq_time_slice_plot.setLabel('bottom', 'Time', units='s')
    self.iq_time_slice_plot.setFixedHeight(120)
    self.iq_time_slice_plot.showGrid(x=True, y=True, alpha=0.3)
    _light_background(self.iq_time_slice_plot)
    self._iq_time_slice_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=1))
    self.iq_time_slice_plot.addItem(self._iq_time_slice_curve)

    # Horizontal cursor line on realtime_plot showing IQ frequency equivalent
    self._iq_cursor_hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashLine))
    self.realtime_plot.addItem(self._iq_cursor_hline)

    # Squared-signal periodograms: FFT of the squared real (I) 48 kHz input signal.
    self.sq_realtime_plot = pg.PlotWidget(title="Squared Signal – Real-time")
    self.sq_realtime_plot.setLabel('left', 'Freq (squared)', units='kHz')
    self.sq_realtime_plot.setLabel('bottom', 'Time', units='s')
    self.sq_realtime_img = pg.ImageItem(axisOrder='col-major')
    self.sq_realtime_plot.addItem(self.sq_realtime_img)
    self.sq_realtime_plot.setAspectLocked(False)
    self.sq_realtime_img.setColorMap(colormap)

    # Crosshairs on sq_realtime_plot
    self._sq_cursor_vline = pg.InfiniteLine(angle=90,  pen=pg.mkPen('g', width=1))
    self._sq_cursor_hline = pg.InfiniteLine(angle=0,   pen=pg.mkPen('g', width=1))
    self.sq_realtime_plot.addItem(self._sq_cursor_vline)
    self.sq_realtime_plot.addItem(self._sq_cursor_hline)

    # Frequency-slice plot: power (dB) vs frequency (kHz), X=power, Y=freq
    self.sq_freq_slice_plot = pg.PlotWidget(title="Freq Slice")
    self.sq_freq_slice_plot.setLabel('bottom', 'Power', units='dB')
    self.sq_freq_slice_plot.setLabel('left', 'Freq (squared)', units='kHz')
    self.sq_freq_slice_plot.showGrid(x=True, y=True, alpha=0.3)
    self.sq_freq_slice_plot.setFixedWidth(_SIDE_WIDTH)
    self.sq_freq_slice_plot.getAxis('left').setWidth(_SIDE_Y_AXIS_W)
    _light_background(self.sq_freq_slice_plot)
    self._sq_freq_slice_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=1))
    self.sq_freq_slice_plot.addItem(self._sq_freq_slice_curve)

    # Time-slice plot: power (dB) vs time (s)
    self.sq_time_slice_plot = pg.PlotWidget(title="Time Slice")
    self.sq_time_slice_plot.setLabel('left', 'Power', units='dB')
    self.sq_time_slice_plot.setLabel('bottom', 'Time', units='s')
    self.sq_time_slice_plot.setFixedHeight(120)
    self.sq_time_slice_plot.showGrid(x=True, y=True, alpha=0.3)
    _light_background(self.sq_time_slice_plot)
    self._sq_time_slice_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=1))
    self.sq_time_slice_plot.addItem(self._sq_time_slice_curve)

    # Cursor coordinate label — fixed width so it never widens column 1
    self.sq_cursor_label = QtWidgets.QLabel("")
    self.sq_cursor_label.setStyleSheet("QLabel { color: green; padding: 2px 6px; }")
    self.sq_cursor_label.setFixedWidth(_SIDE_WIDTH)

    # Connect mouse tracking
    self.sq_realtime_plot.scene().sigMouseMoved.connect(self.on_sq_realtime_mouse_moved)

    self.sq_accumulated_plot = pg.PlotWidget(title="Squared Signal – Accumulated (15 sec snapshot)")
    self.sq_accumulated_plot.setLabel('left', 'Freq (squared)', units='kHz')
    self.sq_accumulated_plot.setLabel('bottom', 'Time', units='s')
    self.sq_accumulated_img = pg.ImageItem(axisOrder='col-major')
    self.sq_accumulated_plot.addItem(self.sq_accumulated_img)
    self.sq_accumulated_plot.setAspectLocked(False)
    self.sq_accumulated_img.setColorMap(colormap)

    def _make_slider_bar(title, min_label_ref, min_range, min_default,
                         max_label_ref, max_range, max_default,
                         min_slot, max_slot, tick_interval):
        """Return a fixed-height horizontal bar: title | min label | slider | max label | slider."""
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))

        min_lbl = QtWidgets.QLabel(f"Min: {min_default} dB")
        setattr(self, min_label_ref, min_lbl)
        row.addWidget(min_lbl)
        min_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        min_sl.setMinimum(min_range[0])
        min_sl.setMaximum(min_range[1])
        min_sl.setValue(min_default)
        min_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        min_sl.setTickInterval(tick_interval)
        min_sl.valueChanged.connect(min_slot)
        row.addWidget(min_sl, stretch=1)

        max_lbl = QtWidgets.QLabel(f"Max: {max_default} dB")
        setattr(self, max_label_ref, max_lbl)
        row.addWidget(max_lbl)
        max_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        max_sl.setMinimum(max_range[0])
        max_sl.setMaximum(max_range[1])
        max_sl.setValue(max_default)
        max_sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        max_sl.setTickInterval(tick_interval)
        max_sl.valueChanged.connect(max_slot)
        row.addWidget(max_sl, stretch=1)

        bar.setFixedHeight(54)
        return bar

    iq_sliders = _make_slider_bar(
        "IQ Color Scale",
        "min_level_label",  (-150, -20),  self.min_level,
        "max_level_label",  (-100,   0),  self.max_level,
        self.on_min_level_changed, self.on_max_level_changed, 10,
    )
    sq_sliders = _make_slider_bar(
        "Squared Color Scale",
        "sq_min_level_label",  (-150, -20),  self.sq_min_level,
        "sq_max_level_label",  (-100,   0),  self.sq_max_level,
        self.on_sq_min_level_changed, self.on_sq_max_level_changed, 10,
    )
    # ── Grid layout ─────────────────────────────────────────────────────────
    # Row 0: accumulated IQ spectrogram  | accumulated noise floor
    # Row 1: real-time IQ spectrogram    | IQ freq slice (cursor-driven)
    # Row 2: IQ time slice               | (col 1 empty)
    # Row 3: IQ slider bar               | (spans both cols)
    # Row 4: accumulated squared         | (col 1 empty)
    # Row 5: squared slider bar          | (spans both cols)
    # Row 6: real-time squared           | sq_freq_slice_plot
    # Row 7: sq_time_slice_plot          | cursor label
    layout.addWidget(self.spectrogram_plot,       0, 0)
    layout.addWidget(self.accumulated_noise_plot,  0, 1)
    layout.addWidget(self.realtime_plot,           1, 0)
    layout.addWidget(self.realtime_noise_plot,     1, 1)
    layout.addWidget(self.iq_time_slice_plot,      2, 0)
    layout.addWidget(iq_sliders,                   3, 0, 1, 2)
    layout.addWidget(self.sq_accumulated_plot,     4, 0)
    layout.addWidget(sq_sliders,                   5, 0, 1, 2)
    layout.addWidget(self.sq_realtime_plot,        6, 0)
    layout.addWidget(self.sq_freq_slice_plot,      6, 1)
    layout.addWidget(self.sq_time_slice_plot,      7, 0)
    layout.addWidget(self.sq_cursor_label,         7, 1)

    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 0)   # col-1 width is fixed by setFixedWidth above
    layout.setRowStretch(0, 3)
    layout.setRowStretch(1, 3)
    layout.setRowStretch(2, 1)
    layout.setRowStretch(3, 0)
    layout.setRowStretch(4, 3)
    layout.setRowStretch(5, 0)
    layout.setRowStretch(6, 3)
    layout.setRowStretch(7, 1)

    self.statusBar().showMessage('Initializing...')
    self.tuned_freq_label = QtWidgets.QLabel("Tuned: --")
    self.tuned_freq_label.setStyleSheet("QLabel { font-weight: bold; padding: 0 10px; }")
    self.statusBar().addPermanentWidget(self.tuned_freq_label)
    self.utc_clock_label = QtWidgets.QLabel()
    self.utc_clock_label.setStyleSheet("QLabel { font-weight: bold; padding: 0 10px; }")
    self.statusBar().addPermanentWidget(self.utc_clock_label)

    self.update_timer = QtCore.QTimer()
    self.update_timer.timeout.connect(self.update_displays)
    self.update_timer.start(100)


def on_min_level_changed(self, value):
    self.min_level = value
    self.min_level_label.setText(f"Min: {value} dB")


def on_max_level_changed(self, value):
    self.max_level = value
    self.max_level_label.setText(f"Max: {value} dB")


def on_sq_min_level_changed(self, value):
    self.sq_min_level = value
    self.sq_min_level_label.setText(f"Min: {value} dB")


def on_sq_max_level_changed(self, value):
    self.sq_max_level = value
    self.sq_max_level_label.setText(f"Max: {value} dB")


def on_select_source_radio(self):
    self._connect_radio_client()
    self.source_mode = "radio"
    self.selected_wav_path = None
    self.statusBar().showMessage("Source: Flex Radio")


def on_select_source_wav(self):
    from pathlib import Path as _Path
    _default_dir = str(_Path('MSK144/simulations').resolve()) if _Path('MSK144/simulations').exists() else ""
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select Simulation WAV File",
        _default_dir,
        "WAV Files (*.wav);;All Files (*)",
    )

    if not file_path:
        # Leave whatever was previously selected unchanged
        return

    self.source_mode = "wav"
    self.selected_wav_path = file_path
    self._wav_load_nonce = getattr(self, '_wav_load_nonce', 0) + 1
    self._wav_done = False
    self.statusBar().showMessage(f"Source selected: WAV File ({file_path})")


def on_sq_realtime_mouse_moved(self, pos):
    if not self.sq_realtime_plot.sceneBoundingRect().contains(pos):
        return
    mouse_point = self.sq_realtime_plot.getViewBox().mapSceneToView(pos)
    t_s   = float(mouse_point.x())
    f_khz = float(mouse_point.y())

    self._sq_cursor_vline.setPos(t_s)
    self._sq_cursor_hline.setPos(f_khz)

    sq_data   = self.sq_realtime_data      # shape (max_history, fft_size)
    sq_freq   = self.sq_freq_axis_khz      # kHz relative to centre
    bps       = self.blocks_per_sec

    # Smooth helper: box average
    def _smooth(arr, n=21):
        if n < 2 or arr.size < n:
            return arr
        return np.convolve(arr, np.ones(n) / n, mode='same')

    # Common time/freq indices
    t_idx  = max(0, min(int(round(t_s * bps)), sq_data.shape[0] - 1))
    f_idx  = max(0, min(int(np.argmin(np.abs(sq_freq - f_khz))), sq_data.shape[1] - 1))
    time_axis = np.arange(sq_data.shape[0]) / bps

    # ── Squared slices ────────────────────────────────────────────────────
    self._sq_time_slice_curve.setData(time_axis, _smooth(sq_data[:, f_idx]))
    self._sq_freq_slice_curve.setData(_smooth(sq_data[t_idx, :]), sq_freq)

    # ── IQ slices — squared freq → IQ freq (squaring doubles frequency) ──
    # f_khz is in the squared domain (offset from centre); original signal
    # is at half that offset.  Convert to absolute MHz for the IQ freq axis.
    f_iq_mhz = self.display_center_freq_mhz + f_khz / 2.0 / 1000.0
    self._iq_cursor_hline.setPos(f_iq_mhz)

    iq_data  = self.realtime_data          # shape (max_history, fft_size)
    iq_freq  = self.freq_axis              # MHz absolute

    fi_iq = max(0, min(int(np.argmin(np.abs(iq_freq - f_iq_mhz))), iq_data.shape[1] - 1))
    self._iq_time_slice_curve.setData(time_axis, _smooth(iq_data[:, fi_iq], n=7))
    self._iq_freq_slice_curve.setData(_smooth(iq_data[t_idx, :], n=21), iq_freq)

    self.sq_cursor_label.setText(
        f"sq: t={t_s:.2f}s f={f_khz:.1f}kHz {float(sq_data[t_idx, f_idx]):.1f}dB\n"
        f"iq: {f_iq_mhz*1000:.1f}kHz {float(iq_data[t_idx, fi_iq]):.1f}dB"
    )
