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
"""Analysis Window — offline IQ capture viewer.

Layout (top to bottom)
----------------------
  Info bar     filename | source | duration | centre freq | # detections
  Spectrogram  48 kHz bilateral waterfall, ±24 kHz, 15 s
  Heatmap      per-channel detection SNR, same axes as live Tone Detection SNR
  Controls     NB enable checkbox | NB factor slider | Re-run button
  Status bar   running / done / error
"""

from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .channelizer import N_CHANNELS
from .processing import DETECT_THRESH_DB
from .displays import _align_msk144_message

# Light-mode plot styling — must be set BEFORE any PlotWidget is constructed.
# Heatmap colormaps below stay the same (they're explicit), only the axes,
# titles, and curve foreground swap from dark→light.
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# ── colormap (same as main window) ────────────────────────────────────────────
def _make_colormap():
    colors    = [
        (0, 0, 0), (0, 0, 64), (0, 0, 128), (0, 64, 192),
        (0, 128, 255), (64, 192, 255), (128, 255, 255),
        (255, 255, 128), (255, 255, 255),
    ]
    positions = [i / 8.0 for i in range(9)]
    return pg.ColorMap(positions, colors)


_COLORMAP = _make_colormap()


def _audio_spectrogram(audio: np.ndarray, win_size: int = 256, hop: int = 128) -> np.ndarray:
    """Return noise-floor-relative spectrogram (n_frames, n_bins) in dB.

    Each bin's median across all frames is treated as the noise floor and
    subtracted, so noise averages to 0 dB and signal peaks appear as positive
    values.  Typical MSK144 burst peaks land at +10 to +30 dB above floor.
    """
    n = len(audio)
    if n < win_size:
        return np.full((1, win_size // 2 + 1), 0.0, dtype=np.float32)
    window = np.hanning(win_size).astype(np.float32)
    n_frames = (n - win_size) // hop + 1
    s0, s1 = audio.strides[0] * hop, audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, win_size), strides=(s0, s1)
    )
    mag = np.abs(np.fft.rfft(frames * window[np.newaxis, :], axis=1))
    spec = (20.0 * np.log10(np.maximum(mag, 1e-10))).astype(np.float32)
    # Subtract per-bin median (noise floor) so noise ≈ 0 dB everywhere.
    noise_floor = np.median(spec, axis=0)
    return spec - noise_floor[np.newaxis, :]


class AnalysisWindow(QtWidgets.QWidget):
    """Standalone offline analysis window for a single IQ capture file."""

    _SETTINGS = QtCore.QSettings('RadioIQ', 'RadioIQVisualizer')

    def __init__(self, wav_path: Path, reporter=None, parent=None):
        super().__init__(None, QtCore.Qt.Window)
        self._wav_path      = Path(wav_path)
        self._reporter      = reporter   # optional Reporter; None = reporting disabled

        # Read manifest JSON (generated alongside test WAVs).
        # callsigns=False means diagnostic test messages — never report.
        manifest_path = self._wav_path.with_suffix('.json')
        self._callsigns_wav = True  # default: allow reporting if reporter present
        if manifest_path.exists():
            import json
            try:
                manifest = json.loads(manifest_path.read_text())
                self._callsigns_wav = bool(manifest.get('callsigns', True))
            except Exception:
                pass
        self._worker        = None
        self._decode_worker = None
        self._engine        = None   # kept alive after replay for manual_decode
        self._results       = None
        self._drain_timer   = None   # polls _engine._decode_queue after replay
        self._n_pending     = 0      # jt9 threads still outstanding

        self.setWindowTitle(f"Analysis — {self._wav_path.name}")
        self.setMinimumSize(900, 800)

        self._build_ui()
        self._restore_settings()
        self._load_and_run()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Outer: info bar | main area | jt9 args | status ──────────────────
        outer_vbox = QtWidgets.QVBoxLayout(self)
        outer_vbox.setContentsMargins(4, 4, 4, 4)
        outer_vbox.setSpacing(4)

        self._info_label = QtWidgets.QLabel("Loading…")
        self._info_label.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #cccccc; padding: 3px 6px; }"
        )
        outer_vbox.addWidget(self._info_label)

        # ── Main area: rows_vsplit | right_col ───────────────────────────────
        # rows_vsplit: 3 rows, each is an h-splitter (plot | controls/plot)
        # right_col:   decode list | NB + buttons
        main_h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._main_h = main_h
        outer_vbox.addWidget(main_h, stretch=1)

        rows_vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._rows_vsplit = rows_vsplit
        right_col   = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._right_col = right_col
        main_h.addWidget(rows_vsplit)
        main_h.addWidget(right_col)
        main_h.setStretchFactor(0, 5)
        main_h.setStretchFactor(1, 1)

        # ── Helper: vertical min/max slider widget ────────────────────────────
        def _vslider_pane(title, min_attr, min_range, min_default,
                          max_attr, max_range, max_default, callback):
            w = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(w)
            vbox.setContentsMargins(4, 4, 4, 4)
            vbox.setSpacing(2)
            lbl = QtWidgets.QLabel(f"<b>{title}</b>")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(lbl)
            hbox = QtWidgets.QHBoxLayout()
            hbox.setSpacing(6)
            for attr, rng, default, side in (
                (min_attr, min_range, min_default, "min"),
                (max_attr, max_range, max_default, "max"),
            ):
                col_w = QtWidgets.QWidget()
                col_v = QtWidgets.QVBoxLayout(col_w)
                col_v.setContentsMargins(0, 0, 0, 0)
                col_v.setSpacing(1)
                val_lbl = QtWidgets.QLabel(str(default))
                val_lbl.setAlignment(QtCore.Qt.AlignCenter)
                setattr(self, attr + '_label', val_lbl)
                sl = QtWidgets.QSlider(QtCore.Qt.Vertical)
                sl.setMinimum(rng[0]); sl.setMaximum(rng[1])
                sl.setValue(default)
                sl.setInvertedAppearance(True)
                sl.valueChanged.connect(callback)
                sl.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
                setattr(self, attr, sl)
                col_v.addWidget(val_lbl)
                col_v.addWidget(sl, stretch=1)
                col_v.addWidget(QtWidgets.QLabel(side))
                hbox.addWidget(col_w)
            vbox.addLayout(hbox, stretch=1)
            return w

        # ── Row 0: IQ spectrograms (H always; V hidden until dual-pol loaded) ──
        # Both plots share the IQ color sliders via a sub-vertical splitter.
        row0 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        row0_specs = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._row0_specs = row0_specs   # stored for dual-pol resize

        self._spec_plot = pg.PlotWidget(title="H  IQ Spectrogram  (48 kHz, ±24 kHz)")
        self._spec_plot.setLabel('left',   'H  Freq offset (kHz)')
        self._spec_plot.getAxis('bottom').hide()
        self._spec_img = pg.ImageItem(axisOrder='col-major')
        self._spec_img.setColorMap(_COLORMAP)
        self._spec_plot.addItem(self._spec_img)
        self._spec_plot.setAspectLocked(False)
        self._spec_plot.getViewBox().disableAutoRange()
        self._blank_bars = pg.PlotCurveItem(pen=pg.mkPen((255, 80, 80, 100), width=4))
        self._spec_plot.addItem(self._blank_bars)
        self._spec_plot.scene().sigMouseClicked.connect(self._on_spec_click)
        row0_specs.addWidget(self._spec_plot)

        self._spec_plot_v = pg.PlotWidget()
        self._spec_plot_v.setLabel('left',   'V  Freq offset (kHz)')
        self._spec_plot_v.setLabel('bottom', 'Time (s)')
        self._spec_img_v = pg.ImageItem(axisOrder='col-major')
        self._spec_img_v.setColorMap(_COLORMAP)
        self._spec_plot_v.addItem(self._spec_img_v)
        self._spec_plot_v.setAspectLocked(False)
        self._spec_plot_v.getViewBox().disableAutoRange()
        self._spec_plot_v.scene().sigMouseClicked.connect(self._on_spec_click_v)
        self._spec_plot_v.hide()
        row0_specs.addWidget(self._spec_plot_v)

        row0.addWidget(row0_specs)
        row0.addWidget(_vslider_pane(
            "IQ Color",
            '_vmin_slider', (-150, -20), -110,
            '_vmax_slider', (-100,   0), -70,
            self._on_level_changed,
        ))
        row0.setStretchFactor(0, 1); row0.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row0)

        # ── Row TFMF: TFMF wideband matched-filter SNR surface ──────────────
        # Image: (time × freq), ±24 kHz audio range, SNR-dB above per-bin
        # pct25 noise floor.  Yellow circles overlay the peak-picked
        # candidates (post all the live-runtime filters).
        row_tfmf = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        row_tfmf_specs = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._row_tfmf_specs = row_tfmf_specs

        self._tfmf_plot = pg.PlotWidget(
            title="H  TFMF Surface  (dual-sync, 48 kHz, ±24 kHz)   "
                  "yellow = peak-picked candidates"
        )
        self._tfmf_plot.setLabel('left',   'H  Freq offset (kHz)')
        self._tfmf_plot.getAxis('bottom').hide()
        self._tfmf_img = pg.ImageItem(axisOrder='col-major')
        self._tfmf_img.setColorMap(_COLORMAP)
        self._tfmf_plot.addItem(self._tfmf_img)
        self._tfmf_plot.setAspectLocked(False)
        self._tfmf_plot.getViewBox().disableAutoRange()
        self._tfmf_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen('y', width=1.5), brush=pg.mkBrush(None),
            symbol='o',
        )
        self._tfmf_plot.addItem(self._tfmf_scatter)
        row_tfmf_specs.addWidget(self._tfmf_plot)

        self._tfmf_plot_v = pg.PlotWidget()
        self._tfmf_plot_v.setLabel('left',   'V  Freq offset (kHz)')
        self._tfmf_plot_v.setLabel('bottom', 'Time (s)')
        self._tfmf_img_v = pg.ImageItem(axisOrder='col-major')
        self._tfmf_img_v.setColorMap(_COLORMAP)
        self._tfmf_plot_v.addItem(self._tfmf_img_v)
        self._tfmf_plot_v.setAspectLocked(False)
        self._tfmf_plot_v.getViewBox().disableAutoRange()
        self._tfmf_scatter_v = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen('y', width=1.5), brush=pg.mkBrush(None),
            symbol='o',
        )
        self._tfmf_plot_v.addItem(self._tfmf_scatter_v)
        self._tfmf_plot_v.hide()
        row_tfmf_specs.addWidget(self._tfmf_plot_v)

        row_tfmf.addWidget(row_tfmf_specs)
        row_tfmf.addWidget(_vslider_pane(
            "TFMF dB",
            '_tfmf_vmin_slider', (-5, 25),   0,
            '_tfmf_vmax_slider', ( 5, 40),  20,
            self._on_tfmf_level_changed,
        ))
        row_tfmf.setStretchFactor(0, 1); row_tfmf.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row_tfmf)

        # ── Row 1: Detection heatmaps (H always; V hidden until dual-pol) ─────
        row1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        row1_hms = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._row1_hms = row1_hms   # stored for dual-pol resize

        _half_ch = N_CHANNELS // 2
        self._det_freq_min_khz  = -float(_half_ch) - 0.5
        self._det_freq_span_khz =  float(N_CHANNELS)
        self._det_plot = pg.PlotWidget(
            title=f"H  sq_det per-channel SNR  "
                  f"(dB above rolling pct25 of squared-FFT pair metric; "
                  f"detector fires at ≥ {DETECT_THRESH_DB:.0f} dB) — "
                  f"circles: green = decoded by jt9/SPD,  "
                  f"orange = launched but no decode"
        )
        self._det_plot.setLabel('left',   'H  Dial offset (kHz)')
        self._det_plot.getAxis('bottom').hide()
        self._det_img = pg.ImageItem(axisOrder='col-major')
        self._det_img.setColorMap(_COLORMAP)
        self._det_plot.addItem(self._det_img)
        self._det_plot.setAspectLocked(False)
        self._det_plot.getViewBox().disableAutoRange()
        self._det_curve_green         = pg.PlotCurveItem(pen=pg.mkPen('g',           width=1.5))
        self._det_curve_orange        = pg.PlotCurveItem(pen=pg.mkPen((255, 140,  0), width=1.5))
        self._det_curve_manual_green  = pg.PlotCurveItem(pen=pg.mkPen((0, 255, 128), width=2.0))
        self._det_curve_manual_orange = pg.PlotCurveItem(pen=pg.mkPen((255, 140,  0), width=2.0))
        self._det_plot.addItem(self._det_curve_green)
        self._det_plot.addItem(self._det_curve_orange)
        self._det_plot.addItem(self._det_curve_manual_green)
        self._det_plot.addItem(self._det_curve_manual_orange)
        self._det_plot.scene().sigMouseClicked.connect(self._on_heatmap_click)
        row1_hms.addWidget(self._det_plot)

        self._det_plot_v = pg.PlotWidget()
        self._det_plot_v.setLabel('left',   'V  Dial offset (kHz)')
        self._det_plot_v.setLabel('bottom', 'Time (s)')
        self._det_img_v = pg.ImageItem(axisOrder='col-major')
        self._det_img_v.setColorMap(_COLORMAP)
        self._det_plot_v.addItem(self._det_img_v)
        self._det_plot_v.setAspectLocked(False)
        self._det_plot_v.getViewBox().disableAutoRange()
        self._det_curve_green_v         = pg.PlotCurveItem(pen=pg.mkPen('g',           width=1.5))
        self._det_curve_orange_v        = pg.PlotCurveItem(pen=pg.mkPen((255, 140,  0), width=1.5))
        self._det_curve_manual_green_v  = pg.PlotCurveItem(pen=pg.mkPen((0, 255, 128), width=2.0))
        self._det_curve_manual_orange_v = pg.PlotCurveItem(pen=pg.mkPen((255, 140,  0), width=2.0))
        self._det_plot_v.addItem(self._det_curve_green_v)
        self._det_plot_v.addItem(self._det_curve_orange_v)
        self._det_plot_v.addItem(self._det_curve_manual_green_v)
        self._det_plot_v.addItem(self._det_curve_manual_orange_v)
        self._det_plot_v.scene().sigMouseClicked.connect(self._on_heatmap_click_v)
        self._det_plot_v.hide()
        row1_hms.addWidget(self._det_plot_v)

        row1.addWidget(row1_hms)
        row1.addWidget(_vslider_pane(
            "Heatmap",
            '_det_vmin_slider', (0, 30),  0,
            '_det_vmax_slider', (1, 50), 15,
            self._on_det_level_changed,
        ))
        row1.setStretchFactor(0, 1); row1.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row1)

        # ── Row FT: envelope + instantaneous carrier-freq trace ─────────────
        # Mirrors docs/figures/freq_vs_time_corpus/freqtime_*.png — the
        # squared-FFT carrier estimator shows fine-grained freq vs time
        # so dual-Doppler paths (a ping with two reflection components at
        # different velocities) become visually obvious.
        row_ft = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._row_ft = row_ft
        self._env_plot = pg.PlotWidget(
            title="TFMF peak SNR per time row  "
                  "(max over freq, dB above per-bin pct25)")
        self._env_plot.setLabel('left', 'dB')
        self._env_plot.getAxis('bottom').hide()
        self._env_plot.setAspectLocked(False)
        self._env_plot.getViewBox().disableAutoRange()
        self._env_curve = pg.PlotCurveItem(pen=pg.mkPen((100, 180, 255), width=1.5),
                                            connect='finite')
        self._env_plot.addItem(self._env_curve)
        # Burst threshold reference line = TFMF's own detector threshold
        # (17.3 dB above pct25 = same FA rate as 13.5 dB above median on
        # AWGN).  A peak crossing this line is what TFMF itself would have
        # flagged as a candidate at that instant.
        self._env_thresh_line = pg.InfiniteLine(
            pos=17.3, angle=0,
            pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine),
        )
        self._env_plot.addItem(self._env_thresh_line)
        row_ft.addWidget(self._env_plot)

        self._ift_plot = pg.PlotWidget(
            title="TFMF peak freq per time row  (argmax over freq)   "
                  "grey = below 17.3 dB,  green = above"
        )
        self._ift_plot.setLabel('left',   'Freq (Hz)')
        self._ift_plot.setLabel('bottom', 'Time (s)')
        self._ift_plot.setAspectLocked(False)
        self._ift_plot.getViewBox().disableAutoRange()
        # X-link the envelope and inst-freq plots so they pan/zoom together.
        self._env_plot.getViewBox().setXLink(self._ift_plot.getViewBox())
        # Sub-threshold points (grey).
        self._ift_scatter_sub = pg.ScatterPlotItem(
            size=4, pen=pg.mkPen(None),
            brush=pg.mkBrush((180, 180, 180, 200)), symbol='o',
        )
        # In-burst points (green).
        self._ift_scatter_burst = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen('g', width=1.5),
            brush=pg.mkBrush((60, 220, 80, 180)), symbol='o',
        )
        self._ift_plot.addItem(self._ift_scatter_sub)
        self._ift_plot.addItem(self._ift_scatter_burst)
        # Zero-line reference (calling freq).
        self._ift_zero_line = pg.InfiniteLine(
            pos=0.0, angle=0,
            pen=pg.mkPen((200, 200, 200, 90), width=1, style=QtCore.Qt.DashLine),
        )
        self._ift_plot.addItem(self._ift_zero_line)
        row_ft.addWidget(self._ift_plot)
        row_ft.setStretchFactor(0, 1); row_ft.setStretchFactor(1, 2)
        rows_vsplit.addWidget(row_ft)

        # ── Row 2: (audio_plot | tone_plot) | audio_vsliders ─────────────────
        row2 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        _mono9 = self.font()
        _mono9.setFamily("Monospace")
        _mono9.setPointSize(9)

        # Left sub-splitter: audio spectrogram and ping spectrum side by side
        row2_plots = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self._audio_plot = pg.PlotWidget(title="12 kHz Audio Spectrogram  (carrier → 1500 Hz)")
        self._audio_plot.setLabel('left',   'Frequency (Hz)')
        self._audio_plot.setLabel('bottom', 'Time (s)')
        self._audio_img = pg.ImageItem(axisOrder='col-major')
        self._audio_img.setColorMap(_COLORMAP)
        self._audio_plot.addItem(self._audio_img)
        self._audio_plot.setAspectLocked(False)
        self._audio_plot.getViewBox().disableAutoRange()
        row2_plots.addWidget(self._audio_plot)

        # Ping spectrum — Y-linked to audio spectrogram for vertical alignment
        self._tone_plot = pg.PlotWidget(title="Ping Spectrum")
        self._tone_plot.hideAxis('left')
        self._tone_plot.setLabel('bottom', 'dB above noise')
        self._tone_plot.setAspectLocked(False)
        self._tone_plot.getViewBox().disableAutoRange()
        self._tone_plot.getViewBox().setYLink(self._audio_plot.getViewBox())
        self._tone_curve = pg.PlotCurveItem(
            pen=pg.mkPen((100, 200, 255), width=1.5), connect='finite',
        )
        self._tone_plot.addItem(self._tone_curve)
        for tone_hz in (1000.0, 2000.0):
            self._tone_plot.addItem(
                pg.InfiniteLine(pos=tone_hz, angle=0,
                                pen=pg.mkPen((255, 200, 0, 120), width=1,
                                             style=QtCore.Qt.DashLine))
            )
        row2_plots.addWidget(self._tone_plot)
        row2_plots.setStretchFactor(0, 2)
        row2_plots.setStretchFactor(1, 1)

        row2.addWidget(row2_plots)
        row2.addWidget(_vslider_pane(
            "Audio dB",
            '_audio_vmin_slider', (-30, 10), -15,
            '_audio_vmax_slider', (  5, 50),  20,
            self._on_audio_level_changed,
        ))
        row2.setStretchFactor(0, 1); row2.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row2)
        # Operator decision 2026-05-29: drop the bottom 12 kHz audio
        # spectrogram + ping spectrum + audio sliders.  The wideband IQ
        # spectrogram at the top is the single primary (time, freq) view.
        row2.hide()

        # Sync all three row splitters so the plot widths stay equal across rows.
        # setSizes does not emit splitterMoved, so no feedback loop.
        def _sync_rows(src, others):
            def _cb():
                sizes = src.sizes()
                for o in others:
                    o.setSizes(sizes)
            return _cb
        # Keep the H/V-split rows in sync so the H pane / V pane heights
        # match across IQ-spec, TFMF-surface, and sq_det-HM rows.
        _sync_row_list = [row0, row_tfmf, row1]
        for _src in _sync_row_list:
            _others = [r for r in _sync_row_list if r is not _src]
            _src.splitterMoved.connect(_sync_rows(_src, _others))

        # Initial row heights — operator suggested:
        #   IQ specs (row0)   shrunk to ~½ their default
        #   TFMF surface      full
        #   sq_det HM (row1)  shrunk to ~½
        #   freq-vs-time (FT) full
        #   audio + ping (r2) full
        # Use stretch factors 2:3:2:3:3 → row0 / row1 get ~15 %, others ~23 %.
        # Old layout was 1:1:1 (~33 % each), so row0 and row1 are ~½ of before.
        for _idx, _f in enumerate((2, 3, 2, 3, 3)):
            rows_vsplit.setStretchFactor(_idx, _f)

        # ── Right col pane 0: status counts + decode list ────────────────────
        _decode_pane = QtWidgets.QWidget()
        _decode_vbox = QtWidgets.QVBoxLayout(_decode_pane)
        _decode_vbox.setContentsMargins(0, 0, 0, 0)
        _decode_vbox.setSpacing(0)

        # Status counts — one line each, updated by _update_status()
        _counts_style = "QLabel { background: #1e1e1e; color: #aaaaaa; padding: 1px 6px; }"
        self._lbl_decoded   = QtWidgets.QLabel("decoded:     —")
        self._lbl_nodecode  = QtWidgets.QLabel("no-decode:   —")
        self._lbl_suppressed= QtWidgets.QLabel("suppressed:  —")
        self._lbl_blanked   = QtWidgets.QLabel("blanked:     —")
        self._lbl_pending   = QtWidgets.QLabel("")
        for lbl in (self._lbl_decoded, self._lbl_nodecode,
                    self._lbl_suppressed, self._lbl_blanked, self._lbl_pending):
            lbl.setFont(_mono9)
            lbl.setStyleSheet(_counts_style)
            _decode_vbox.addWidget(lbl)

        _decode_hdr = QtWidgets.QLabel("  t(s)   Freq(kHz)   SNR    θ    Message")
        _decode_hdr.setFont(_mono9)
        _decode_hdr.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #aaaaaa; "
            "border: 1px solid #555; border-bottom: none; padding: 2px 4px; }"
        )
        self._decode_list = QtWidgets.QListWidget()
        self._decode_list.setFont(_mono9)
        self._decode_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; color: #e0e0e0; border: 1px solid #555; }"
            "QListWidget::item { padding: 2px 4px; }"
        )
        _decode_vbox.addWidget(_decode_hdr)
        _decode_vbox.addWidget(self._decode_list, stretch=1)
        right_col.addWidget(_decode_pane)

        # ── Right col pane 1: buttons + jt9 args ─────────────────────────────
        from .detection import JT9_BASE_ARGS

        ctrl_pane = QtWidgets.QWidget()
        ctrl_vbox = QtWidgets.QVBoxLayout(ctrl_pane)
        ctrl_vbox.setContentsMargins(6, 6, 6, 6)
        ctrl_vbox.setSpacing(6)

        self._rerun_btn = QtWidgets.QPushButton("Re-run")
        self._rerun_btn.setEnabled(False)
        self._rerun_btn.clicked.connect(self._load_and_run)
        ctrl_vbox.addWidget(self._rerun_btn)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Processing… %p%")
        self._progress_bar.hide()
        ctrl_vbox.addWidget(self._progress_bar)

        self._clear_btn = QtWidgets.QPushButton("Clear Circles")
        self._clear_btn.clicked.connect(self._clear_manual_decodes)
        ctrl_vbox.addWidget(self._clear_btn)

        self._report_btn = QtWidgets.QPushButton("Report All Decodes")
        self._report_btn.setEnabled(self._reporter is not None and self._callsigns_wav)
        _report_tip = (
            "Report all successful decodes in the list to PSKReporter and WSJT-X UDP.\n"
            "WAV decodes are not reported automatically."
            if self._callsigns_wav else
            "Reporting disabled — this WAV was generated without real callsigns."
        )
        self._report_btn.setToolTip(_report_tip)
        self._report_btn.clicked.connect(self._on_report_selected)
        ctrl_vbox.addWidget(self._report_btn)

        # jt9 args as named fields
        from .detection import _JT9_FTOL_ANALYSIS, _JT9_DEPTH
        _jt9_field_defs = [
            ('ftol',  'Ftol ± (Hz)',   str(_JT9_FTOL_ANALYSIS)),
            ('depth', 'decode depth',  str(_JT9_DEPTH)),
        ]
        _mono11 = self.font()
        _mono11.setFamily("Monospace")
        _mono11.setPointSize(11)

        _jt9_hdr = QtWidgets.QLabel("<b>jt9 arguments</b>")
        _jt9_hdr.setFont(_mono11)
        ctrl_vbox.addWidget(_jt9_hdr)

        self._jt9_fields = {}   # name → QLineEdit
        for name, label, default in _jt9_field_defs:
            row_w = QtWidgets.QWidget()
            row_h = QtWidgets.QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)
            lbl = QtWidgets.QLabel(label)
            lbl.setFont(_mono11)
            lbl.setMinimumWidth(120)
            edit = QtWidgets.QLineEdit(default)
            edit.setFont(_mono11)
            edit.setFixedWidth(56)
            self._jt9_fields[name] = edit
            row_h.addWidget(lbl)
            row_h.addWidget(edit)
            ctrl_vbox.addWidget(row_w)

        args_reset_btn = QtWidgets.QPushButton("Reset jt9 args")
        def _reset_jt9():
            self._jt9_fields['ftol'].setText(str(_JT9_FTOL_ANALYSIS))
            self._jt9_fields['depth'].setText(str(_JT9_DEPTH))
        args_reset_btn.clicked.connect(_reset_jt9)
        ctrl_vbox.addWidget(args_reset_btn)
        ctrl_vbox.addStretch(1)

        right_col.addWidget(ctrl_pane)
        right_col.setSizes([400, 300])

    # ── Replay ─────────────────────────────────────────────────────────────────

    def _load_and_run(self):
        from .capture import load_capture
        from .analysis_engine import AnalysisEngine, AnalysisWorker

        self._rerun_btn.setEnabled(False)
        self._lbl_pending.setText("Running replay…")
        self._decode_list.clear()
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        # Stop any previous worker.
        if self._worker is not None:
            self._worker.quit()
            self._worker.wait(2000)
            self._worker = None

        try:
            iq, meta = load_capture(self._wav_path)

            engine = AnalysisEngine(iq, meta)
            self._engine = engine   # keep alive for manual_decode
            self._worker = AnalysisWorker(engine)
            self._worker.finished.connect(self._on_replay_done)
            self._worker.error.connect(self._on_replay_error)
            self._worker.progress.connect(
                lambda f: self._progress_bar.setValue(int(f * 100))
            )
            self._worker.start()

            # Update info bar from metadata (available before replay completes).
            self._update_info_bar(meta, detections=None)

        except Exception as exc:
            self._lbl_pending.setText(f"Error: {exc}")
            self._rerun_btn.setEnabled(True)
            import traceback
            print(f"[analysis] load/start error:\n{traceback.format_exc()}", flush=True)

    def _on_replay_done(self, results: dict):
        self._results  = results
        self._n_pending = results.get('n_launched', 0)
        self._rerun_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._progress_bar.hide()

        markers = results.get('jt9_markers', [])
        n_sup   = sum(1 for m in markers if m.get('outcome') == 'suppressed')
        n_blank = len(results.get('blanked_times_s', []))
        self._update_info_bar(results['meta'],
                              detections=sum(1 for m in markers if m.get('outcome') != 'suppressed'))
        self._update_status()

        self._render(results)

        # Start a 250 ms timer to drain jt9 decode results as they arrive.
        if self._n_pending > 0:
            if self._drain_timer is None:
                self._drain_timer = QtCore.QTimer(self)
                self._drain_timer.timeout.connect(self._drain_decode_queue)
            self._drain_timer.start(250)
        else:
            self._update_status()

    def _on_replay_error(self, msg: str):
        self._rerun_btn.setEnabled(True)
        self._progress_bar.hide()
        self._lbl_pending.setText(f"Replay error: {msg[:120]}")
        print(f"[analysis] replay error:\n{msg}", flush=True)

    def _update_status(self):
        """Refresh the status count labels."""
        if self._results is None:
            return
        markers   = self._results.get('jt9_markers', [])
        n_decoded = sum(1 for m in markers if m.get('outcome') == 'decoded')
        n_no_dec  = sum(1 for m in markers if m.get('outcome') in ('no_decode', 'timeout', 'error'))
        n_sup     = sum(1 for m in markers if m.get('outcome') == 'suppressed')
        n_blank   = len(self._results.get('blanked_times_s', []))
        pending   = self._n_pending
        self._lbl_decoded.setText(   f"decoded:     {n_decoded}")
        self._lbl_nodecode.setText(  f"no-decode:   {n_no_dec}")
        self._lbl_suppressed.setText(f"suppressed:  {n_sup}")
        self._lbl_blanked.setText(   f"blanked:     {n_blank}")
        self._lbl_pending.setText(   f"jt9 running: {pending}" if pending > 0 else "")

    def _drain_decode_queue(self):
        """Poll engine decode queue; update marker outcomes and redraw circles."""
        from PyQt5.QtGui import QColor
        if self._engine is None or self._results is None:
            if self._drain_timer:
                self._drain_timer.stop()
            return

        markers      = self._results.get('jt9_markers', [])
        marker_by_id = {m['id']: m for m in markers}
        dq           = self._engine._decode_queue
        changed      = False
        meta         = self._results['meta']
        fc_mhz       = float(meta.get('center_freq_mhz', 50.260))

        while not dq.empty():
            try:
                result = dq.get_nowait()
            except Exception:
                break
            mid = result.get('marker_id', -1)
            if mid in marker_by_id:
                m       = marker_by_id[mid]
                outcome = result.get('outcome', 'no_decode')
                # Store audio on marker so heatmap click can display it later.
                if result.get('audio') is not None:
                    m['audio'] = result['audio']
                # theta_deg is always available (computed before jt9 runs).
                if result.get('theta_deg') is not None:
                    m['theta_deg'] = result.get('theta_deg')
                if outcome == 'decoded' or result.get('decoded'):
                    m['outcome']   = 'decoded'
                    m['decoded']   = True
                    m['message']   = result.get('message')
                    # Add to decode list
                    t_s      = m.get('t', 0.0)
                    freq_khz = m.get('freq_khz', 0.0)
                    radio_mhz = fc_mhz + (freq_khz / 1000.0)
                    _th = m.get('theta_deg')
                    _th_str = f"{_th:4.0f}°" if _th is not None else "   —"
                    item = QtWidgets.QListWidgetItem(
                        f"  {t_s:5.1f}   {radio_mhz * 1000:9.3f}   {'?':>6}   {_th_str}  {_align_msk144_message(m['message'] or '')}"
                    )
                    item.setForeground(QColor('#80ff80'))
                    item.setData(QtCore.Qt.UserRole, result)
                    self._decode_list.addItem(item)
                else:
                    m['outcome'] = outcome
                self._n_pending = max(0, self._n_pending - 1)
                changed = True

        if changed:
            self._render(self._results)
        self._update_status()

        if self._n_pending <= 0:
            if self._drain_timer:
                self._drain_timer.stop()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self, results: dict):
        meta      = results['meta']
        spec      = results['spectrogram_data']      # (max_history, fft_size)
        spec_v    = results.get('spectrogram_data_v')
        hm        = results['ch_snr_history']         # (N_SNR_HIST, N_CH)
        hm_v      = results.get('ch_snr_history_v')
        dual      = bool(results.get('dual_pol', False))
        markers   = results['jt9_markers']
        blanked   = results['blanked_times_s']
        duration  = float(results['duration_secs'])
        rate      = int(meta.get('sample_rate', 48000))
        fc_mhz    = float(meta.get('center_freq_mhz', 50.260))

        vmin = self._vmin_slider.value()
        vmax = self._vmax_slider.value()

        # Show/hide V panes based on dual-pol; grow window on first dual-pol load.
        _prev_dual = getattr(self, '_dual_shown', None)
        if dual:
            if not self._spec_plot_v.isVisible():
                self._spec_plot_v.show()
            if not self._det_plot_v.isVisible():
                self._det_plot_v.show()
            if not self._tfmf_plot_v.isVisible():
                self._tfmf_plot_v.show()
            # H pane: hide time axis (V pane below has it)
            self._spec_plot.getAxis('bottom').hide()
            self._det_plot.getAxis('bottom').hide()
            self._tfmf_plot.getAxis('bottom').hide()
            # On transition to dual-pol: grow window height and equalize H/V splits.
            if not _prev_dual:
                self._dual_shown = True
                new_h = max(self.height(), int(self.height() * 1.5), 1100)
                self.resize(self.width(), new_h)
                # Give H and V equal space in each inner splitter.
                # Defer until the event loop has processed the resize — calling
                # setSizes immediately reads pre-resize heights and halves H.
                def _equalize():
                    for spl in (self._row0_specs, self._row_tfmf_specs,
                                 self._row1_hms):
                        spl.setSizes([10000, 10000])   # equal large values → 50/50
                QtCore.QTimer.singleShot(0, _equalize)
        else:
            self._dual_shown = False
            self._spec_plot_v.hide()
            self._det_plot_v.hide()
            self._tfmf_plot_v.hide()
            self._spec_plot.getAxis('bottom').show()
            self._det_plot.getAxis('bottom').show()
            self._tfmf_plot.getAxis('bottom').show()

        # ── Spectrogram image ─────────────────────────────────────────────────
        # Y axis: frequency offset in kHz, ±rate/2 kHz.  For audio-burst
        # sources (a 12 kHz mono WAV Hilbert+upsampled here) the only
        # physical content sits in 0..3 kHz; zoom in so the operator can
        # actually see it rather than a sliver in the middle of ±24 kHz.
        _is_audio = bool(results.get('source_was_audio'))
        half_rate_khz = rate / 2000.0    # e.g. 24.0 kHz for 48 kHz

        def _trim_valid(arr, idle: float = 0.0):
            """Return the leading slice of ``arr`` along axis 0 that contains
            rows the engine actually wrote.  ``spec_staging`` is initialised
            to **-130.0** (dB) and ``_ch_snr_history`` to **-999.0** by
            Engine.__init__; pre-fix this function checked != 0 and treated
            every initial-sentinel row as populated, leaving the actual data
            squashed into the leftmost ~20 % of the plot width on a 2.9-s
            burst sitting in a 15-s-period buffer."""
            n_rows = arr.shape[0]
            flat = arr.reshape(n_rows, -1)
            is_idle_row = np.all(np.abs(flat - idle) < 0.01, axis=1)
            nz = ~is_idle_row
            if not nz.any():
                return arr
            last = int(np.where(nz)[0].max()) + 1
            return arr[:last]

        def _set_spec(plot, img, arr):
            arr_v = _trim_valid(arr, idle=-130.0)
            _dw  = max(plot.width(), 1)
            _rep = max(1, -(-_dw // arr_v.shape[0]))
            img.setImage(np.repeat(arr_v[:, ::4], _rep, axis=0),
                         autoLevels=False, levels=[vmin, vmax])
            # Rect spans the full burst duration; image data is trimmed and
            # stretched so the valid samples cover the whole plot width.
            img.setRect(QtCore.QRectF(0.0, -half_rate_khz,
                                       duration, 2 * half_rate_khz))
            plot.setXRange(0, duration, padding=0)
            if _is_audio:
                plot.setYRange(0, 3, padding=0)             # audio band only
            else:
                plot.setYRange(-half_rate_khz, half_rate_khz, padding=0)

        # Top wideband-IQ spec stays visible for all source types — it's the
        # primary, consistent (time, freq) view across captures.  Operator
        # decision 2026-05-29: prefer one consistent IQ-based view at the
        # top over a separate 12 kHz audio spectrogram at the bottom (the
        # latter is dropped — see ``_audio_plot.hide()`` below).
        _set_spec(self._spec_plot, self._spec_img, spec)
        if dual and spec_v is not None:
            _set_spec(self._spec_plot_v, self._spec_img_v, spec_v)

        # Blanked-block overlay: draw a short vertical stroke at each block start
        if len(blanked) > 0:
            block_dur = 256 / rate   # NB_FFT_SIZE / sample_rate
            xs, ys = [], []
            for t0 in blanked:
                t_mid = float(t0) + block_dur / 2.0
                xs.extend([t_mid, t_mid, np.nan])
                ys.extend([-half_rate_khz, half_rate_khz, np.nan])
            self._blank_bars.setData(
                np.array(xs, dtype=np.float32),
                np.array(ys, dtype=np.float32),
                connect='finite',
            )
        else:
            self._blank_bars.setData(x=[], y=[])

        # ── Detection heatmaps ────────────────────────────────────────────────
        # Image data extends across all N_CHANNELS (±24 kHz dial offset); the
        # *viewbox* gets zoomed to the audio band 0..6 kHz when the source is
        # a 12 kHz audio WAV — that's where any real sq_det channel response
        # could possibly sit.
        _dlvl     = [self._det_vmin_slider.value(), self._det_vmax_slider.value()]

        def _set_hm(plot, img, arr):
            arr_v = _trim_valid(arr, idle=-999.0)
            hm_d = np.fft.fftshift(arr_v, axes=1)
            _dw  = max(plot.width(), 1)
            _rep = max(1, -(-_dw // hm_d.shape[0]))
            img.setImage(np.repeat(hm_d, _rep, axis=0),
                         autoLevels=False, levels=_dlvl)
            img.setRect(QtCore.QRectF(0.0, self._det_freq_min_khz,
                                       duration, self._det_freq_span_khz))
            plot.setXRange(0, duration, padding=0)
            if _is_audio:
                plot.setYRange(-0.5, 6.5, padding=0)        # 0–6 kHz audio band
            else:
                plot.setYRange(self._det_freq_min_khz - 0.5,
                                self._det_freq_min_khz + self._det_freq_span_khz + 0.5,
                                padding=0)

        _set_hm(self._det_plot, self._det_img, hm)
        if dual and hm_v is not None:
            _set_hm(self._det_plot_v, self._det_img_v, hm_v)

        # ── TFMF surface (H, optionally V) ────────────────────────────────────
        # _tfmf_surface_* is (n_windows, n_freq_bins) float32 SNR-dB; axis 0
        # is time (col-major image → X axis), axis 1 is freq fftshifted (DC
        # in middle).  ±24 kHz on the 48 kHz IQ.
        from .processing import (_TFMF_FREQ_MIN_KHZ as _TF_F_MIN,
                                  _TFMF_FREQ_SPAN_KHZ as _TF_F_SPAN,
                                  _TFMF_DISP_STRIDE as _TF_STRIDE,
                                  _TFMF_DISP_FS_HZ as _TF_FS)
        _tfmf_lvl = [self._tfmf_vmin_slider.value(), self._tfmf_vmax_slider.value()]
        _tfmf_pan_offset_khz = (50.260 - fc_mhz) * 1000.0  # 0 when fc_mhz IS calling
        # NB: AnalysisEngine sets center_freq_mhz to the calling freq, so the
        # TFMF freq axis (relative to pan centre) lines up with the call-freq
        # offset axis here.  No further translation needed for the rect.

        def _set_tfmf(plot, img, scatter, surf, cands):
            if surf is None:
                img.setImage(np.zeros((1, 1), dtype=np.float32),
                             autoLevels=False, levels=_tfmf_lvl)
                scatter.setData(x=[], y=[])
                return
            n_win = surf.shape[0]
            t_span = n_win * _TF_STRIDE / _TF_FS
            _rect = QtCore.QRectF(0.0, _TF_F_MIN, t_span, _TF_F_SPAN)
            img.setImage(surf, autoLevels=False, levels=_tfmf_lvl)
            img.setRect(_rect)
            plot.setXRange(0, duration, padding=0)
            if _is_audio:
                # Same audio-band zoom as the IQ spectrogram for visual
                # alignment.  The image still extends across ±24 kHz; the
                # viewbox just shows the 0–3 kHz slice.
                plot.setYRange(0, 3, padding=0)
            else:
                plot.setYRange(_TF_F_MIN - 0.5, _TF_F_MIN + _TF_F_SPAN + 0.5,
                                padding=0)
            if cands:
                scatter.setData(x=[c.time_s for c in cands],
                                y=[c.freq_hz / 1000.0 for c in cands])
            else:
                scatter.setData(x=[], y=[])

        _set_tfmf(self._tfmf_plot, self._tfmf_img, self._tfmf_scatter,
                   results.get('tfmf_surface_h'), results.get('tfmf_candidates_h', []))
        if dual:
            _set_tfmf(self._tfmf_plot_v, self._tfmf_img_v, self._tfmf_scatter_v,
                       results.get('tfmf_surface_v'), results.get('tfmf_candidates_v', []))

        # ── Freq-vs-time + envelope traces (derived from TFMF surface) ─────
        _ift_f   = results.get('inst_freq_audio')
        _ift_t   = results.get('inst_freq_t')
        _env_db  = results.get('env_db')
        if _ift_f is not None and _ift_t is not None and _ift_t.size > 0:
            # Envelope (top sub-plot).  Y fixed 0..25 dB — covers the
            # interesting range (17.3 dB threshold visible, room above it
            # for strong-burst peaks) without auto-fit-driven re-zoom that
            # made cross-burst comparisons hard.
            self._env_curve.setData(_ift_t, _env_db)
            self._env_plot.setXRange(0, duration, padding=0)
            self._env_plot.setYRange(0, 25, padding=0)

            # Burst-vs-noise classification at TFMF's own detector threshold
            # (17.3 dB above per-bin pct25).  Points above = "what TFMF would
            # have flagged as a candidate at this instant"; below = "what
            # TFMF saw as noise" so the operator can still see where the
            # argmax was wandering.
            _TFMF_THRESH_DB = 17.3
            _is_burst = (_env_db > _TFMF_THRESH_DB
                          if _env_db is not None else None)
            if _is_burst is None or not np.any(_is_burst):
                self._ift_scatter_sub.setData(x=_ift_t, y=_ift_f)
                self._ift_scatter_burst.setData(x=[], y=[])
            else:
                self._ift_scatter_sub.setData(
                    x=_ift_t[~_is_burst], y=_ift_f[~_is_burst])
                self._ift_scatter_burst.setData(
                    x=_ift_t[_is_burst], y=_ift_f[_is_burst])
            self._ift_plot.setXRange(0, duration, padding=0)
            # Y-range: for audio-burst sources clamp to 0..3 kHz — TFMF is
            # NOT a squared detector (verified by reading
            # ``compute_tf_surface_cpu`` directly), so the matched-filter
            # peak sits at the actual audio carrier (~1500 Hz for MSK144),
            # not at 2·carrier.  Otherwise auto-fit to the in-burst points.
            if _is_audio:
                self._ift_plot.setYRange(0, 3000, padding=0)
            else:
                _y_for_range = (_ift_f[_is_burst]
                                if _is_burst is not None and np.any(_is_burst)
                                else _ift_f)
                if _y_for_range.size > 0:
                    _y_med = float(np.median(_y_for_range))
                    _y_dev = max(50.0,
                                  float(np.std(_y_for_range)) * 3.0,
                                  float(np.max(np.abs(_y_for_range - _y_med)))
                                  + 20.0)
                    self._ift_plot.setYRange(_y_med - _y_dev, _y_med + _y_dev,
                                              padding=0)
                else:
                    self._ift_plot.setYRange(-200, 200, padding=0)
        else:
            self._env_curve.setData([], [])
            self._ift_scatter_sub.setData(x=[], y=[])
            self._ift_scatter_burst.setData(x=[], y=[])

        # ── Audio-burst: auto-show the native 12 kHz spectrogram + tone view ─
        # _show_audio_spectrogram is normally triggered by a manual decode
        # click; for audio-source WAVs we have the burst right here, so
        # render it on load.
        if _is_audio:
            _native_audio = results.get('native_audio')
            _native_sr    = results.get('native_audio_sr')
            if _native_audio is not None and _native_sr:
                try:
                    self._show_audio_spectrogram(
                        _native_audio,
                        label=f"{_native_sr // 1000} kHz audio source",
                    )
                except Exception as _e:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "Auto-show audio spectrogram failed: %s", _e)

        # ── Marker circles ────────────────────────────────────────────────────
        r_y = 2.5   # kHz radius
        px_x, px_y = self._det_plot.getViewBox().viewPixelSize()
        r_x = r_y * (px_x / px_y) if px_y > 0 else r_y * 0.1
        theta = np.linspace(0, 2 * np.pi, 33)

        def _circle_path(mlist):
            xs, ys = [], []
            for m in mlist:
                xs.append(m.get('t', 0.0)        + r_x * np.cos(theta))
                ys.append(m.get('freq_khz', 0.0) + r_y * np.sin(theta))
                xs.append([np.nan])
                ys.append([np.nan])
            return np.concatenate(xs), np.concatenate(ys)

        def _split_pol(mlist):
            """Split markers to H or V pane.

            Primary: theta_deg >= 45° → V (polarization search result).
            Fallback when theta_deg is None: det_pol == 'v' → V (detecting channel).
            """
            h, v = [], []
            for m in mlist:
                th = m.get('theta_deg')
                if dual and th is not None:
                    (v if th >= 45.0 else h).append(m)
                elif dual and m.get('det_pol') == 'v':
                    v.append(m)
                else:
                    h.append(m)
            return h, v

        decoded    = [m for m in markers if m.get('outcome') == 'decoded']
        no_decode  = [m for m in markers if m.get('outcome') in
                      ('no_decode', 'timeout', 'error', 'launched')]
        dec_h,  dec_v  = _split_pol(decoded)
        nod_h,  nod_v  = _split_pol(no_decode)

        for curve, mlist in (
            (self._det_curve_green,  dec_h),
            (self._det_curve_orange, nod_h),
        ):
            if mlist:
                curve.setData(*_circle_path(mlist), connect='finite')
            else:
                curve.setData(x=[], y=[])

        if dual:
            for curve, mlist in (
                (self._det_curve_green_v,  dec_v),
                (self._det_curve_orange_v, nod_v),
            ):
                if mlist:
                    curve.setData(*_circle_path(mlist), connect='finite')
                else:
                    curve.setData(x=[], y=[])

    # ── Manual decode ──────────────────────────────────────────────────────────

    def _on_spec_click(self, event):
        """Click on IQ spectrogram → decode at the IQ frequency offset clicked."""
        if self._engine is None or self._results is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb  = self._spec_plot.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t_s     = float(pos.x())
        iq_khz  = float(pos.y())   # IQ offset kHz (not dial offset)
        duration = self._results.get('duration_secs', 15.0)
        if not (0.0 <= t_s <= duration):
            return
        # IQ offset kHz → DSP fc_hz in Hz
        fc_hz = iq_khz * 1000.0
        self._trigger_decode(fc_hz, t_s)

    def _on_spec_click_v(self, event):
        """Click on V IQ spectrogram → decode at the frequency offset clicked."""
        if self._engine is None or self._results is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb  = self._spec_plot_v.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t_s    = float(pos.x())
        iq_khz = float(pos.y())
        duration = self._results.get('duration_secs', 15.0)
        if not (0.0 <= t_s <= duration):
            return
        self._trigger_decode(iq_khz * 1000.0, t_s)

    def _on_heatmap_click(self, event):
        """Click on the H detection heatmap → trigger a new decode at the click position."""
        if self._engine is None or self._results is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb  = self._det_plot.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t_s    = float(pos.x())
        fc_khz = float(pos.y())    # dial offset kHz
        duration = self._results.get('duration_secs', 15.0)
        if not (0.0 <= t_s <= duration):
            return
        self._trigger_decode(fc_khz * 1000.0 + 1500.0, t_s)

    def _on_heatmap_click_v(self, event):
        """Click on the V detection heatmap → trigger a new decode at the click position."""
        if self._engine is None or self._results is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb  = self._det_plot_v.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t_s    = float(pos.x())
        fc_khz = float(pos.y())
        duration = self._results.get('duration_secs', 15.0)
        if not (0.0 <= t_s <= duration):
            return
        self._trigger_decode(fc_khz * 1000.0 + 1500.0, t_s, force_v=True)

    def _trigger_decode(self, fc_hz: float, t_s: float, force_v: bool = False):
        """Common decode trigger used by both spectrogram and heatmap clicks.

        force_v=True means the user clicked on the V pane; the result circle
        will appear on the V heatmap regardless of measured theta_deg.
        """
        from .analysis_engine import ManualDecodeWorker

        # Build jt9 args from Ftol and depth fields.
        try:
            ftol  = int(self._jt9_fields['ftol'].text().strip())
            depth = int(self._jt9_fields['depth'].text().strip())
            jt9_args = [
                'jt9', '--msk144',
                '-L', str(1500 - ftol), '-H', str(1500 + ftol),
                '-F', str(ftol), '-d', str(depth),
            ]
        except Exception:
            jt9_args = None   # fall back to JT9_BASE_ARGS

        # Stop any running decode worker.
        if self._decode_worker is not None:
            self._decode_worker.quit()
            self._decode_worker.wait(1000)

        self._manual_click_force_v = force_v   # remembered by _on_decode_done
        dial_khz = (fc_hz - 1500.0) / 1000.0
        self._lbl_pending.setText(f"Decoding at {dial_khz:+.1f} kHz dial, t={t_s:.1f} s …")

        worker = ManualDecodeWorker(self._engine, fc_hz=fc_hz, t_click_s=t_s,
                                    jt9_args=jt9_args)
        worker.finished.connect(self._on_decode_done)
        worker.error.connect(self._on_decode_error)
        self._decode_worker = worker
        worker.start()

    def _clear_manual_decodes(self):
        """Clear all circles (auto-detect and manual) and the decode result list."""
        for c in (self._det_curve_green, self._det_curve_orange,
                  self._det_curve_manual_green, self._det_curve_manual_orange,
                  self._det_curve_green_v, self._det_curve_orange_v,
                  self._det_curve_manual_green_v, self._det_curve_manual_orange_v):
            c.setData(x=[], y=[])
        self._decode_list.clear()

    def _show_audio_spectrogram(self, audio: np.ndarray, label: str = ""):
        """Compute and display 12 kHz audio spectrogram in the audio panel."""
        spec = _audio_spectrogram(audio.astype(np.float32))   # (n_frames, n_bins)
        print(f"[audio spec] shape={spec.shape}  min={spec.min():.1f}  max={spec.max():.1f}  "
              f"mean={spec.mean():.1f}  median={float(np.median(spec)):.1f}  dBNF", flush=True)
        n_frames, n_bins = spec.shape
        _DECODE_RATE = 12000
        win_size, hop = 256, 128
        duration = (n_frames * hop + win_size) / _DECODE_RATE
        # Each rfft bin k has center frequency k × (fs/win_size).
        # setRect must place pixel k center at that frequency; shifting y0 by
        # -bin_hz/2 achieves this: center of pixel k = -bin_hz/2 + (k+0.5)*bin_hz = k*bin_hz.
        bin_hz   = _DECODE_RATE / win_size          # 46.875 Hz per bin
        rect_h   = n_bins * bin_hz                  # 129 × 46.875 = 6046.875 Hz
        vmin = self._audio_vmin_slider.value()
        vmax = self._audio_vmax_slider.value()
        self._audio_img.setImage(spec, autoLevels=False, levels=[vmin, vmax])
        self._audio_img.setRect(QtCore.QRectF(0.0, -bin_hz / 2.0, duration, rect_h))
        self._audio_plot.setXRange(0, duration, padding=0)
        self._audio_plot.setYRange(0, 3000, padding=0)   # show 0–3 kHz; signal at 1500 Hz
        if label:
            self._audio_plot.setTitle(f"12 kHz Audio Spectrogram  —  {label}")

        # Tone spectrum: max dB above noise floor across all time frames, per bin.
        # Using max (not mean) highlights the brief ping peaks over the noise baseline.
        power_lin = 10.0 ** (spec / 10.0)           # linear power (n_frames, n_bins)
        tone_db   = 10.0 * np.log10(np.sum(power_lin, axis=0))  # integrated over all frames
        freq_axis = np.arange(n_bins) * bin_hz      # bin k center = k × bin_hz
        self._tone_curve.setData(x=tone_db, y=freq_axis)
        self._tone_plot.setXRange(float(np.min(tone_db)), float(np.max(tone_db)), padding=0.05)
        # Y range is linked to _audio_plot, so setYRange on audio drives both.

    def _on_decode_done(self, result: dict):
        from PyQt5.QtGui import QColor
        outcome   = result.get('outcome', '?')
        message   = result.get('message', '')
        snr       = result.get('jt9_snr')
        t_s       = result.get('t_s', 0.0)
        radio_khz = result.get('radio_khz', 0.0)
        snr_str   = f"{snr:+d} dB" if snr is not None else "  ?"
        color = {'decoded': '#80ff80', 'no_decode': '#aaaaaa',
                 'timeout': '#ffaa44', 'error': '#ff6060'}.get(outcome, '#cccccc')
        theta_deg = result.get('theta_deg')
        th_str    = f"{theta_deg:4.0f}°" if theta_deg is not None else "   —"
        item = QtWidgets.QListWidgetItem(
            f"  {t_s:5.1f}   {radio_khz:9.3f}   {snr_str:>6}   {th_str}  "
            f"{_align_msk144_message(message) if message else f'({outcome})'}"
        )
        item.setForeground(QColor(color))
        item.setData(QtCore.Qt.UserRole, result)
        self._decode_list.insertItem(0, item)
        self._lbl_pending.setText(
            f"Decode: {outcome}  {message}  @ {radio_khz:.3f} kHz  SNR {snr_str}"
        )

        # Display the 12 kHz audio spectrogram for this decode.
        audio = result.get('audio')
        if audio is not None and len(audio) > 0:
            fc_khz = (result.get('fc_hz', 1500.0) - 1500.0) / 1000.0
            self._show_audio_spectrogram(
                audio,
                label=f"click  t={t_s:.1f}s  dial {fc_khz:+.1f} kHz  {outcome}",
            )

        # Add a circle marker on the correct heatmap pane.
        # Manual clicks: honour the pane the user clicked (force_v).
        # Auto-detected circles use theta_deg routing in _render/_split_pol.
        _dual   = bool(self._results.get('dual_pol', False)) if self._results else False
        _use_v  = _dual and getattr(self, '_manual_click_force_v', False)
        if _use_v:
            curve = (self._det_curve_manual_green_v
                     if outcome == 'decoded'
                     else self._det_curve_manual_orange_v)
            ref_plot = self._det_plot_v
        else:
            curve = (self._det_curve_manual_green
                     if outcome == 'decoded'
                     else self._det_curve_manual_orange)
            ref_plot = self._det_plot

        r_y = 2.5
        px_x, px_y = ref_plot.getViewBox().viewPixelSize()
        r_x = r_y * (px_x / px_y) if px_y > 0 else r_y * 0.1
        circ   = np.linspace(0, 2 * np.pi, 33)
        fc_khz = (result.get('fc_hz', 1500.0) - 1500.0) / 1000.0   # → dial offset kHz
        t_val  = result.get('t_s', 0.0)
        xs = np.concatenate([t_val  + r_x * np.cos(circ), [np.nan]])
        ys = np.concatenate([fc_khz + r_y * np.sin(circ), [np.nan]])
        prev_x, prev_y = curve.getData()
        if prev_x is not None and len(prev_x) > 0:
            xs = np.concatenate([prev_x, xs])
            ys = np.concatenate([prev_y, ys])
        curve.setData(xs, ys, connect='finite')

    def _on_decode_error(self, msg: str):
        self._lbl_pending.setText(f"Decode error: {msg[:120]}")
        print(f"[analysis] decode error:\n{msg}", flush=True)

    def _on_report_selected(self):
        """Report all successful decodes in the list to PSKReporter / WSJT-X UDP."""
        if self._reporter is None:
            return
        reported = []
        for i in range(self._decode_list.count()):
            result = self._decode_list.item(i).data(QtCore.Qt.UserRole)
            if result and result.get('outcome') == 'decoded':
                self._reporter.report_decode(result)
                reported.append(f"{result.get('message','?')} @ {result.get('radio_khz',0.0):.3f} kHz")
        if reported:
            self._lbl_pending.setText(f"Reported {len(reported)}: {', '.join(reported)}")
        else:
            self._lbl_pending.setText("No successful decodes to report")

    # ── Slider callbacks ───────────────────────────────────────────────────────

    def _on_level_changed(self):
        if self._results is not None:
            lvl = [self._vmin_slider.value(), self._vmax_slider.value()]
            self._spec_img.setLevels(lvl)
            self._spec_img_v.setLevels(lvl)

    def _on_det_level_changed(self):
        if self._results is not None:
            lvl = [self._det_vmin_slider.value(), self._det_vmax_slider.value()]
            self._det_img.setLevels(lvl)
            self._det_img_v.setLevels(lvl)

    def _on_tfmf_level_changed(self):
        if self._results is not None:
            lvl = [self._tfmf_vmin_slider.value(), self._tfmf_vmax_slider.value()]
            self._tfmf_img.setLevels(lvl)
            self._tfmf_img_v.setLevels(lvl)

    def _on_audio_level_changed(self):
        self._audio_img.setLevels([self._audio_vmin_slider.value(), self._audio_vmax_slider.value()])

    # ── Info bar ───────────────────────────────────────────────────────────────

    def _update_info_bar(self, meta: dict, detections):
        src      = meta.get('source', 'unknown')
        dur      = meta.get('duration_secs', 0.0)
        fc_mhz   = meta.get('center_freq_mhz', 0.0)
        ts       = meta.get('timestamp_utc', '')[:19].replace('T', ' ')
        det_str  = f"  |  detections: {detections}" if detections is not None else ""
        self._info_label.setText(
            f"{self._wav_path.name}   source: {src}   {ts} UTC   "
            f"fc: {fc_mhz:.4f} MHz   duration: {dur:.1f} s{det_str}"
        )

    # ── Settings persistence ───────────────────────────────────────────────────

    def _restore_settings(self):
        """Restore window geometry, slider values, and splitter positions from QSettings."""
        s = self._SETTINGS
        geom = s.value('analysis_window_geometry')
        if geom is not None:
            self.restoreGeometry(geom)
        for attr, key in (
            ('_main_h',      'analysis_main_h_state'),
            ('_rows_vsplit', 'analysis_rows_vsplit_state'),
            ('_right_col',   'analysis_right_col_state'),
        ):
            state = s.value(key)
            if state is not None:
                getattr(self, attr).restoreState(state)
        try:
            self._vmin_slider.setValue(int(s.value('analysis_vmin', -110)))
            self._vmax_slider.setValue(int(s.value('analysis_vmax', -70)))
            self._det_vmin_slider.setValue(int(s.value('analysis_det_vmin', 0)))
            self._det_vmax_slider.setValue(int(s.value('analysis_det_vmax', 15)))
            self._audio_vmin_slider.setValue(int(s.value('analysis_audio_vmin_v3', -15)))
            self._audio_vmax_slider.setValue(int(s.value('analysis_audio_vmax_v3', 20)))
            from .detection import _JT9_FTOL_ANALYSIS, _JT9_DEPTH
            self._jt9_fields['ftol'].setText(str(int(s.value('analysis_jt9_ftol', _JT9_FTOL_ANALYSIS))))
            self._jt9_fields['depth'].setText(str(int(s.value('analysis_jt9_depth', _JT9_DEPTH))))
        except (TypeError, ValueError):
            pass   # leave defaults if settings are malformed

    def _save_settings(self):
        """Persist window geometry, slider values, and splitter positions to QSettings."""
        s = self._SETTINGS
        s.setValue('analysis_window_geometry', self.saveGeometry())
        s.setValue('analysis_main_h_state',      self._main_h.saveState())
        s.setValue('analysis_rows_vsplit_state',  self._rows_vsplit.saveState())
        s.setValue('analysis_right_col_state',    self._right_col.saveState())
        s.setValue('analysis_vmin',         self._vmin_slider.value())
        s.setValue('analysis_vmax',         self._vmax_slider.value())
        s.setValue('analysis_det_vmin',     self._det_vmin_slider.value())
        s.setValue('analysis_det_vmax',     self._det_vmax_slider.value())
        s.setValue('analysis_audio_vmin_v3', self._audio_vmin_slider.value())
        s.setValue('analysis_audio_vmax_v3', self._audio_vmax_slider.value())
        s.setValue('analysis_jt9_ftol',      self._jt9_fields['ftol'].text().strip())
        s.setValue('analysis_jt9_depth',     self._jt9_fields['depth'].text().strip())

    def closeEvent(self, event):
        self._save_settings()
        if self._drain_timer is not None:
            self._drain_timer.stop()
        if self._worker is not None:
            self._worker.quit()
            self._worker.wait(2000)
        if self._decode_worker is not None:
            self._decode_worker.quit()
            self._decode_worker.wait(2000)
        event.accept()
