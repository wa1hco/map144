"""UI layout and slider handlers for the DAXIQ visualizer."""

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


def setup_ui(self):
    """Create the five-panel display layout."""
    self.setWindowTitle(f'FlexRadio DAXIQ - {self.center_freq_mhz:.3f} MHz')
    self.setGeometry(100, 100, 1400, 1600)

    menu_bar = self.menuBar()
    file_menu = menu_bar.addMenu("&File")

    self.source_action_group = QtWidgets.QActionGroup(self)
    self.source_action_group.setExclusive(True)

    self.source_flex_action = QtWidgets.QAction("Flex Radio", self)
    self.source_flex_action.setCheckable(True)
    self.source_flex_action.setChecked(True)
    self.source_flex_action.triggered.connect(self.on_select_source_flex)
    self.source_action_group.addAction(self.source_flex_action)
    file_menu.addAction(self.source_flex_action)

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
    self.accumulated_energy_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))
    self.spectrogram_plot.addItem(self.accumulated_energy_curve)

    self.realtime_plot = pg.PlotWidget(title="Real-time (15 sec)")
    self.realtime_plot.setLabel('left', 'Frequency', units='MHz')
    self.realtime_plot.setLabel('bottom', 'Time', units='s')
    self.realtime_img = pg.ImageItem(axisOrder='col-major')
    self.realtime_plot.addItem(self.realtime_img)
    self.realtime_plot.setAspectLocked(False)
    self.realtime_energy_curve = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))
    self.realtime_plot.addItem(self.realtime_energy_curve)

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

    self.accumulated_noise_plot = pg.PlotWidget(title="Accumulated Noise Floor")
    self.accumulated_noise_plot.setLabel('bottom', 'Power', units='dB')
    self.accumulated_noise_plot.setLabel('left', 'Frequency', units='MHz')
    self.accumulated_noise_curve = self.accumulated_noise_plot.plot(pen='y', width=2)

    self.realtime_noise_plot = pg.PlotWidget(title="Real-time Noise Floor")
    self.realtime_noise_plot.setLabel('bottom', 'Power', units='dB')
    self.realtime_noise_plot.setLabel('left', 'Frequency', units='MHz')
    self.realtime_noise_curve = self.realtime_noise_plot.plot(pen='c', width=2)

    # Squared-signal periodograms: FFT of the squared real (I) 48 kHz input signal.
    self.sq_realtime_plot = pg.PlotWidget(title="Squared Signal – Real-time")
    self.sq_realtime_plot.setLabel('left', 'Freq (squared)', units='kHz')
    self.sq_realtime_plot.setLabel('bottom', 'Time', units='s')
    self.sq_realtime_img = pg.ImageItem(axisOrder='col-major')
    self.sq_realtime_plot.addItem(self.sq_realtime_img)
    self.sq_realtime_plot.setAspectLocked(False)
    self.sq_realtime_img.setColorMap(colormap)

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
    # ── Grid layout ─────────────────────────────────────────────────────────
    # Row 0: accumulated IQ spectrogram  | accumulated noise floor
    # Row 1: real-time IQ spectrogram    | real-time noise floor
    # Row 2: IQ slider bar               | (spans both cols)
    # Row 3: accumulated squared         | (col 1 empty)
    # Row 4: real-time squared           | (col 1 empty)
    layout.addWidget(self.spectrogram_plot,      0, 0)
    layout.addWidget(self.accumulated_noise_plot, 0, 1)
    layout.addWidget(self.realtime_plot,          1, 0)
    layout.addWidget(self.realtime_noise_plot,    1, 1)
    layout.addWidget(iq_sliders,                  2, 0, 1, 2)
    layout.addWidget(self.sq_accumulated_plot,    3, 0)
    layout.addWidget(self.sq_realtime_plot,       4, 0)

    layout.setColumnStretch(0, 4)
    layout.setColumnStretch(1, 1)
    layout.setRowStretch(0, 3)
    layout.setRowStretch(1, 3)
    layout.setRowStretch(2, 0)
    layout.setRowStretch(3, 3)
    layout.setRowStretch(4, 3)

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


def on_select_source_flex(self):
    self.source_mode = "flex"
    self.selected_wav_path = None
    self.statusBar().showMessage("Source selected: Flex Radio")


def on_select_source_wav(self):
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select WAV File",
        "",
        "WAV Files (*.wav);;All Files (*)",
    )

    if not file_path:
        self.source_flex_action.setChecked(True)
        self.source_mode = "flex"
        self.statusBar().showMessage("Source selected: Flex Radio")
        return

    self.source_mode = "wav"
    self.selected_wav_path = file_path
    self.statusBar().showMessage(f"Source selected: WAV File ({file_path})")
