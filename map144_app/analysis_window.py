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
from .detection import BURST_WAV_MAX_DURATION_S

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


def _find_launch_for_wav(wav_path) -> dict | None:
    """Locate the launches.jsonl entry that produced this saved-burst WAV.

    Match keys (all parsed from the filename):
        timestamp date + HH:MM:SS == launch entry's timestamp prefix
        radio_khz                 == rounded int of launch entry's radio_khz
        message (post-safe-quote) == launch entry's message (safe-quoted)

    Returns the parsed dict from launches.jsonl, or None if not matched.
    Used by the analysis window to overlay the live detector's verdict on
    the offline plots — what the runtime decided, where it placed the
    detection, what metrics it computed.  detection.py format:
        {YYYY}{MM}{DD}_{HH}{MM}{SS}Z_{khz}kHz_{msg_safe}.wav
    """
    import re as _re, json as _json
    name = str(wav_path.name)
    m = _re.match(r'^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})Z_(\d+)kHz_(.+)\.wav$', name)
    if not m:
        return None
    yyyy, MM, dd, hh, mm, ss, khz, msg_safe = m.groups()
    target_ts_prefix = f"{yyyy}-{MM}-{dd}_{hh}:{mm}:{ss}"
    target_radio_khz = int(khz)
    launches = wav_path.parent / 'launches.jsonl'
    if not launches.exists():
        return None
    hits = []
    with launches.open() as f:
        for line in f:
            try:
                r = _json.loads(line)
            except Exception:
                continue
            if not str(r.get('timestamp', '')).startswith(target_ts_prefix):
                continue
            try:
                r_khz = int(round(float(r.get('radio_khz', 0))))
            except (TypeError, ValueError):
                continue
            if r_khz != target_radio_khz:
                continue
            r_msg_safe = _re.sub(r'[^A-Za-z0-9]+', '_',
                                 str(r.get('message', ''))).strip('_')
            if r_msg_safe == msg_safe:
                hits.append(r)
    return hits[0] if hits else None


def _wav_start_within_period(wav_path, duration_s: float) -> float | None:
    """Compute the WAV's first-sample time within its 15-s WSJT period.

    Two filename conventions are handled:

      Map144 burst save  ``YYYYMMDD_HHMMSSZ_<msg>.wav``
        HHMMSS is the BURST-DETECTION time (decoder-fire time), rounded
        to the second.  The WAV's first sample sits a pre-window before:
            SPD save  (~1.9 s WAV): pre = pad + pre_n     = 0.6 s
            jt9 save  (~2.9 s WAV): pre = pad + pre_n_jt9 = 1.6 s
        so wav_start_within_period = (burst_sec_of_day % 15) - pre.

      WSJT-X period save  ``YYMMDD_HHMMSS.wav``
        HHMMSS is the PERIOD START (always on a 15-s boundary), the WAV
        covers the full 15-s period, so wav_start_within_period = 0.

    Returns the offset in seconds (may be negative if the WAV straddles a
    period boundary), or ``None`` if the filename doesn't match either.
    """
    import re as _re
    name = str(wav_path.name)
    # Map144 burst save format
    m = _re.search(r'(\d{8})_(\d{2})(\d{2})(\d{2})Z', name)
    if m:
        hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        burst_sec_of_day = hh * 3600 + mm * 60 + ss
        burst_within_period = burst_sec_of_day % 15
        pre = 0.6 if duration_s < 2.5 else 1.6   # SPD vs jt9 save
        return float(burst_within_period) - pre
    # WSJT-X period save format (15-s full-period WAV; t=0 is period start)
    if _re.match(r'^\d{6}_\d{6}\.wav$', name):
        return 0.0
    return None


def _audio_spectrogram(audio: np.ndarray, win_size: int = 256, hop: int = 128) -> np.ndarray:
    """Return noise-floor-relative spectrogram (n_frames, n_bins) in dB.

    Each bin's 25th percentile across all frames is treated as the noise
    floor and subtracted, so noise averages to 0 dB and burst peaks appear
    as positive values.  This matches the per-bin pct25 convention TFMF
    and sq_det use, so the three rows render with a common visual scale
    and the pre-burst noise region looks the same in all of them.
    pct25 (rather than median) tolerates up to 75 % signal contamination
    per bin — useful when a bin happens to fall inside the carrier band.

    Typical MSK144 burst peaks land at +10 to +30 dB above floor.
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
    # Subtract per-bin 25th percentile (noise floor) so noise ≈ 0 dB everywhere.
    noise_floor = np.percentile(spec, 25, axis=0).astype(np.float32)
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
        # Second info row: live-runtime detection metrics for this WAV
        # (parsed from launches.jsonl by _find_launch_for_wav and filled in
        # during _render).  Empty / hidden until a match is found.
        self._launch_info_label = QtWidgets.QLabel("")
        self._launch_info_label.setStyleSheet(
            "QLabel { background: #233a26; color: #d0eedd; padding: 3px 6px; "
            "font-family: monospace; font-size: 11px; }"
        )
        self._launch_info_label.setVisible(False)
        outer_vbox.addWidget(self._launch_info_label)

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
        self._row_iq_spec = row0           # row-container handle for show/hide
        row0_specs = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._row0_specs = row0_specs   # stored for dual-pol resize

        self._spec_plot = pg.PlotWidget(title="Wideband IQ Spectrogram")
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

        self._tfmf_plot = pg.PlotWidget(title="TFMF Spectrum")
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
        self._det_plot = pg.PlotWidget(title="Squared Detection Spectrum")
        self._det_plot.setLabel('left',   'H  Squared-spec freq (kHz)')
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
        self._det_plot_v.setLabel('left',   'V  Squared-spec freq (kHz)')
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
        self._env_plot = pg.PlotWidget(title="TFMF Peak SNR per frame")
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

        self._ift_plot = pg.PlotWidget(title="TFMF Peak Frequency plot")
        self._ift_plot.setLabel('left',   'Freq (kHz)')
        self._ift_plot.setLabel('bottom', 'Time within 15-s period (s)')
        self._ift_plot.setAspectLocked(False)
        self._ift_plot.getViewBox().disableAutoRange()
        # X-link the envelope and inst-freq plots so they pan/zoom together.
        self._env_plot.getViewBox().setXLink(self._ift_plot.getViewBox())
        # Three-tier classification (operator request 2026-05-29):
        #   below low_thresh         → grey  (noise / sub-threshold)
        #   low_thresh..high_thresh  → green (in-burst, confirmed signal)
        #   above high_thresh        → magenta-pink (very strong signal, distinct
        #                              hue from green but similar luminance per
        #                              operator's "not pure red, same luminance"
        #                              spec — RGB (220,100,200) lum ≈ 148, green
        #                              (60,220,80) lum ≈ 156, near-match).
        self._ift_scatter_sub = pg.ScatterPlotItem(
            size=4, pen=pg.mkPen(None),
            brush=pg.mkBrush((180, 180, 180, 200)), symbol='o',
        )
        self._ift_scatter_burst = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen('g', width=1.5),
            brush=pg.mkBrush((60, 220, 80, 180)), symbol='o',
        )
        self._ift_scatter_strong = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen((220, 100, 200), width=1.5),
            brush=pg.mkBrush((220, 100, 200, 200)), symbol='o',
        )
        self._ift_plot.addItem(self._ift_scatter_sub)
        self._ift_plot.addItem(self._ift_scatter_burst)
        self._ift_plot.addItem(self._ift_scatter_strong)
        # Connecting line through consecutive in-burst dots.  Modelled on
        # docs/figures/freq_vs_time_corpus/freqtime_*.png — matplotlib's
        # 'o-' with NaN-separated arrays.  Same idea with pyqtgraph: feed
        # a Y array that has NaN at sub-threshold positions; connect='finite'
        # then breaks the line over the gaps.  Lets the operator see the
        # frequency walk (Doppler drift / multi-signal alternation) as a
        # chain rather than just isolated dots.
        self._ift_chain_line = pg.PlotCurveItem(
            pen=pg.mkPen((60, 220, 80, 200), width=1.2),
            connect='finite',
        )
        # Drawn UNDER the scatters so dots stay on top.  addItem appends to
        # the end of the item list, which is on top; insert at position 0 via
        # zValue instead.
        self._ift_chain_line.setZValue(-1)
        self._ift_plot.addItem(self._ift_chain_line)
        # Zero-line reference (calling freq).
        self._ift_zero_line = pg.InfiniteLine(
            pos=0.0, angle=0,
            pen=pg.mkPen((200, 200, 200, 90), width=1, style=QtCore.Qt.DashLine),
        )
        self._ift_plot.addItem(self._ift_zero_line)
        row_ft.addWidget(self._ift_plot)
        row_ft.setStretchFactor(0, 1); row_ft.setStretchFactor(1, 2)

        # ── Wrap row_ft horizontally so we can add Y-range sliders on the
        # right (operator request 2026-05-29).  Two slider panes stacked
        # vertically — one for env_plot's dB axis, one for ift_plot's
        # freq axis — let the operator align the env / freq plots against
        # the spectrograms above by trimming the visible Y range.
        row_ft_h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        row_ft_h.addWidget(row_ft)
        _ft_sliders = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        _ft_sliders.addWidget(_vslider_pane(
            "Env dB",
            '_env_ymin_slider', (-10, 25),  0,
            '_env_ymax_slider', (  5, 60), 25,
            self._on_env_y_changed,
        ))
        # ift sliders now select TWO THRESHOLDS that re-classify the freq
        # scatter dots: below min → grey (noise), min..max → green (burst),
        # above max → magenta-pink (very strong).  Defaults: min=17 dB
        # (TFMF's own per-bin pct25 threshold = 17.3, rounded), max=30 dB.
        _ft_sliders.addWidget(_vslider_pane(
            "Thresh dB",
            '_ift_thresh_low',  (-5, 30), 17,
            '_ift_thresh_high', (10, 60), 30,
            self._on_ift_thresh_changed,
        ))
        # Slider sub-splitter stretches must match the env / ift plot ratio
        # (1 : 2) so the slider columns stay vertically aligned with their
        # plot regions even when the user drags the row_ft vertical splitter.
        _ft_sliders.setStretchFactor(0, 1); _ft_sliders.setStretchFactor(1, 2)
        row_ft_h.addWidget(_ft_sliders)
        row_ft_h.setStretchFactor(0, 1); row_ft_h.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row_ft_h)

        # ── Row epoch: per-candidate sync frame-phase + coherence ───────
        # Operator request 2026-05-29.  Two new dimensions on a single row:
        #   X      time within 15-s period (linked to all other rows)
        #   Y      sync onset modulo NSPM (=72 ms at 12 kHz) — same-signal
        #          frames from a single station cluster at one Y; different
        #          stations / paths separate vertically by sync timing.
        #   color  SNR for now; coherence (= phase-lock quality) once that's
        #          computed per candidate.
        #   dot    one per TFMF peak-picked candidate above the in-burst
        #          threshold.
        # Hypothesis from msk144_spd.py:_sync_phase_features: real signals
        # have coherence ≈ 1, noise ≈ 0.1.  TFMF candidates above SNR-only
        # threshold form a diffuse cloud; coherence is what separates real
        # signal frames from threshold-edge noise.
        self._epoch_plot = pg.PlotWidget(title="Sync Frame Phase + Coherence")
        self._epoch_plot.setLabel('left',   'Frame phase (ms)')
        self._epoch_plot.setLabel('bottom', 'Time within 15-s period (s)')
        self._epoch_plot.setAspectLocked(False)
        self._epoch_plot.getViewBox().disableAutoRange()
        self._epoch_scatter = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), symbol='o',
        )
        self._epoch_plot.addItem(self._epoch_scatter)
        # Live-runtime detector marker.  Cross at (t_sec, sync_t_sample/12)
        # showing where the channelized sq_det placed the burst's sync
        # onset, with the live sync_phase_coherence_h as its fill color.
        # Hidden until a matching launches.jsonl entry is found.  Lets the
        # operator compare TFMF's peak-picker against the live detector
        # at a glance — if TFMF dots cluster around the live cross at high
        # coherence, both agree; if not, TFMF is firing on a different cell.
        self._epoch_live_marker = pg.ScatterPlotItem(
            size=18, pen=pg.mkPen((0, 200, 220), width=2.5),
            brush=pg.mkBrush(None), symbol='+',
        )
        self._epoch_live_marker.setZValue(10)   # on top of TFMF dots
        self._epoch_plot.addItem(self._epoch_live_marker)
        # Wrap in horizontal splitter with a slider stub on the right so
        # column alignment matches the rows above (plot left, sliders right).
        row_epoch_h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        row_epoch_h.addWidget(self._epoch_plot)
        row_epoch_h.addWidget(_vslider_pane(
            "Coh thresh",
            '_epoch_coh_thresh', (0, 100), 30,
            '_epoch_snr_thresh', (10, 50), 17,
            self._on_epoch_thresh_changed,
        ))
        row_epoch_h.setStretchFactor(0, 1); row_epoch_h.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row_epoch_h)

        # ── Row 2: (audio_plot | tone_plot) | audio_vsliders ─────────────────
        row2 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._row_audio_spec = row2        # row-container handle for show/hide

        _mono9 = self.font()
        _mono9.setFamily("Monospace")
        _mono9.setPointSize(9)

        # Left sub-splitter: audio spectrogram and ping spectrum side by side
        row2_plots = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self._audio_plot = pg.PlotWidget(title="Audio Spectrum")
        self._audio_plot.setLabel('left',   'Frequency (kHz)')
        self._audio_plot.setLabel('bottom', 'Time within 15-s period (s)')
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
        # Operator request 2026-05-29: hide ping spectrum — gets in the way
        # of the audio spectrogram, which is the actual diagnostic.  The
        # tone_plot widget stays parented so any code that references it
        # (like the Y-link with audio_plot) doesn't break, just invisible.
        self._tone_plot.hide()
        row2_plots.setStretchFactor(0, 1)
        row2_plots.setStretchFactor(1, 0)

        row2.addWidget(row2_plots)
        row2.addWidget(_vslider_pane(
            "Audio dB",
            '_audio_vmin_slider', (-30, 10), -15,
            '_audio_vmax_slider', (  5, 50),  20,
            self._on_audio_level_changed,
        ))
        row2.setStretchFactor(0, 1); row2.setStretchFactor(1, 0)
        rows_vsplit.addWidget(row2)

        # ── Reorder rows for audio-burst-mode display ─────────────────────────
        # Target layout (operator request 2026-05-29):
        #   row 1  audio spectrogram (12 kHz native, 0..6 kHz)
        #   row 2  sq_det squared-spec (WSJT-X-faithful, 12 kHz native)
        #   row 3  TFMF surface
        #   row 4  TFMF peak SNR vs time
        #   row 5  TFMF peak freq scatter (grey / green dots)
        # IQ-spec (row0) is kept in the splitter but hidden — when a true
        # 48 kHz IQ capture file shows up the renderer will reveal it
        # instead of the audio-spec row.  QSplitter.insertWidget moves an
        # already-parented child to the new index, so this just rearranges.
        rows_vsplit.insertWidget(0, row2)         # audio spec   → row 1
        rows_vsplit.insertWidget(1, row1)         # sq_det       → row 2
        rows_vsplit.insertWidget(2, row_tfmf)     # TFMF surface → row 3
        rows_vsplit.insertWidget(3, row_ft_h)     # TFMF env+ift+Y-sliders (rows 4-5)
        rows_vsplit.insertWidget(4, row_epoch_h)  # sync frame phase + coh → row 6
        rows_vsplit.insertWidget(5, row0)         # IQ spec      → row 7 (hidden in audio mode)

        # Lock all six plot X axes together (operator request 2026-05-29).
        # pyqtgraph's setXLink chain propagates pan/zoom from any plot
        # through every other — so when the operator drags time on the
        # audio spec, sq_det / TFMF / env / ift / epoch rows all follow in
        # lockstep.  Routes both directions.
        self._audio_plot.getViewBox().setXLink(self._det_plot.getViewBox())
        self._det_plot.getViewBox().setXLink(self._tfmf_plot.getViewBox())
        self._tfmf_plot.getViewBox().setXLink(self._env_plot.getViewBox())
        self._ift_plot.getViewBox().setXLink(self._epoch_plot.getViewBox())
        # env↔ift link already set in env_plot setup (line ~404).

        # Operator follow-up 2026-05-29: when the analysis window was
        # shrunk horizontally, all rows compressed together EXCEPT
        # sq_det, which kept its tick spacing and ran off under the
        # slider widget on the right.  Cause: pyqtgraph's plot title is
        # a QGraphicsTextItem whose width drives a minimum widget width,
        # and sq_det's title was significantly longer than the others.
        # Titles have been shortened above; also force every plot widget
        # to accept widths down to 0 so the splitter can shrink them in
        # lockstep when the operator drags the window narrower.
        for _p in (self._audio_plot, self._tone_plot,
                   self._spec_plot, self._spec_plot_v,
                   self._det_plot,  self._det_plot_v,
                   self._tfmf_plot, self._tfmf_plot_v,
                   self._env_plot,  self._ift_plot,
                   self._epoch_plot):
            _p.setMinimumWidth(0)

        # ── Detection-time markers (Phase 2 step 2) ──────────────────────
        # A vertical line at the live-runtime's reported detection time
        # (t_sec from launches.jsonl) on every time-locked plot.  Cyan
        # dashed so it's visible on both the dark-image rows and the light
        # line/scatter rows, but doesn't compete with burst content.
        # Hidden until _render finds a matching launches.jsonl entry.
        self._det_time_lines = []
        for _p in (self._audio_plot, self._det_plot, self._tfmf_plot,
                   self._env_plot, self._ift_plot):
            _ln = pg.InfiniteLine(pos=0.0, angle=90,
                                  pen=pg.mkPen((0, 200, 220, 200), width=1.5,
                                               style=QtCore.Qt.DashLine))
            _ln.setVisible(False)
            _p.addItem(_ln)
            self._det_time_lines.append(_ln)

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

        # Initial row heights for the reordered layout (operator request
        # 2026-05-29 — audio-burst mode):
        #   row 1 audio spec        stretch 3
        #   row 2 sq_det            stretch 3
        #   row 3 TFMF surface      stretch 4   (primary detector view)
        #   row 4-5 env + ift       stretch 4   (two sub-plots)
        #   row 6 sync frame phase  stretch 3
        #   row 7 IQ wideband       stretch 1   (hidden by default, ~0 vertical space)
        for _idx, _f in enumerate((3, 3, 4, 4, 3, 1)):
            rows_vsplit.setStretchFactor(_idx, _f)

        # ── "View All" override: reset to operator default ranges ─────────
        # Default pyqtgraph "View All" fits the viewbox to data bounds with
        # padding from InfiniteLine items.  We snap back to the
        # period-time X range stored on self._x_min/_x_max and a row-specific
        # Y so all five rows stay time-aligned.  Closures read those at
        # click-time so View All tracks whatever file is currently loaded.
        self._x_min = 0.0
        self._x_max = float(BURST_WAV_MAX_DURATION_S)
        def _make_reset_view(plot, y_min, y_max):
            def _reset(*args, **kwargs):
                plot.setXRange(self._x_min, self._x_max, padding=0)
                plot.setYRange(y_min, y_max, padding=0)
            return _reset
        # Fixed-Y override list: plots whose Y range is a fixed operator
        # default.  Stable across renders and View All clicks.
        for _plot, _ymin, _ymax in (
                (self._audio_plot,   0,    3),      # 0..3 kHz audio (kHz units)
                (self._det_plot,     0,    6),      # 0..6 kHz squared
                (self._det_plot_v,   0,    6),
                (self._tfmf_plot,    0,    3),      # 0..3 kHz (audio-band TFMF zoom)
                (self._tfmf_plot_v,  0,    3),
                (self._epoch_plot,   0,   72),      # 0..72 ms (NSPM frame phase, ms)
                # IQ wideband spec: 48 kHz IQ default ±24 kHz (matches the
                # ImageItem rect set in _render).  Hidden in audio-burst mode.
                (self._spec_plot,   -24, 24),
                (self._spec_plot_v, -24, 24),
        ):
            _plot.getViewBox().autoRange = _make_reset_view(_plot, _ymin, _ymax)

        # env_plot / ift_plot Y range tracks the operator sliders, so their
        # "View All" reads the current slider values at click-time rather
        # than baking in defaults.  Same X reset (self._x_max), slider Y.
        def _make_reset_view_env(plot):
            def _reset(*args, **kwargs):
                plot.setXRange(self._x_min, self._x_max, padding=0)
                plot.setYRange(self._env_ymin_slider.value(),
                               self._env_ymax_slider.value(), padding=0)
            return _reset
        def _make_reset_view_ift(plot):
            def _reset(*args, **kwargs):
                plot.setXRange(self._x_min, self._x_max, padding=0)
                plot.setYRange(0, 3, padding=0)   # fixed 0..3 kHz
            return _reset
        self._env_plot.getViewBox().autoRange = _make_reset_view_env(self._env_plot)
        self._ift_plot.getViewBox().autoRange = _make_reset_view_ift(self._ift_plot)

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

        # Phase 2 step 1: locate this WAV's entry in launches.jsonl and
        # surface the live-runtime detector metrics so the operator can
        # compare them against the offline TFMF / sq_det / WSJT-X views.
        self._launch_entry = _find_launch_for_wav(self._wav_path)
        self._update_launch_info_bar()

        # Compute per-candidate coherence for the new sync-phase row.
        # MAP144's _sync_phase_features wants 12 kHz channelized baseband
        # of NSPM samples at the sync onset.  For each TFMF candidate we
        # mix the 48 kHz IQ to DC at the candidate's freq, decimate to
        # 12 kHz, take NSPM samples starting at the refined sync onset,
        # then run the existing within-frame coherence ratio.  Cached on
        # self._cand_coherences so slider re-renders don't recompute.
        self._cand_coherences = self._compute_candidate_coherences(results)

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

        # Time-axis offset to put the X axis in "time within 15-s period"
        # coordinates: t=0 is the WSJT period start (the :00/:15/:30/:45
        # boundary that contains this burst), not the WAV start.
        #
        # Preferred source: launches.jsonl t_sec — sub-second precise.  The
        # filename-derived path (HHMMSS rounded to integer) loses up to
        # 1 s of precision, which manifested as a visible ~0.7 s shift of
        # the detection-time line vs the actual burst (the fractional part
        # of t_sec gets dropped).  Use the matched launch entry if found.
        _lentry = getattr(self, '_launch_entry', None)
        _t_offset = None
        if _is_audio:
            if _lentry is not None and _lentry.get('t_sec') is not None:
                _pre = 0.6 if float(duration) < 2.5 else 1.6
                try:
                    _t_offset = float(_lentry['t_sec']) - _pre
                except (TypeError, ValueError):
                    _t_offset = None
            if _t_offset is None:
                _t_offset = _wav_start_within_period(self._wav_path, float(duration))
        if _t_offset is None:
            _t_offset = 0.0
        self._t_offset = _t_offset

        # X axis spans the WAV duration in period-time coords.  No upper
        # cap — both burst saves (~1.9/2.9 s) and full WSJT-X period saves
        # (15 s) should render at their natural extent.
        _x_lo  = float(_t_offset)
        _x_hi  = float(_t_offset) + float(duration)
        _x_default = _x_hi - _x_lo
        self._x_min = _x_lo
        self._x_max = _x_hi

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
            # Keep the full freq resolution (was [:, ::4] — that quartered the
            # display freq bins to 93.75 Hz/bin and made the audio band look
            # blocky).  Time resolution is set by the engine FFT hop (1024 sa
            # at 48 kHz = 21.3 ms / row) — to refine time, repeat each row
            # _rep cols wide so the image stretches smoothly to the plot
            # width rather than rendering as fat blocks.
            arr_v = _trim_valid(arr, idle=-130.0)
            n_t = arr_v.shape[0]
            _dw  = max(plot.width(), 1)
            _rep = max(1, -(-_dw // n_t))
            img.setImage(np.repeat(arr_v, _rep, axis=0),
                         autoLevels=False, levels=[vmin, vmax])
            # Map the image's full time extent to [0, t_filled] where
            # t_filled = n_t * hop / rate is the audio time actually covered
            # by the FFT rows that got written.  Using ``duration`` here
            # would stretch e.g. 134 FFT rows of a 2.86-s audio span across
            # the 2.9-s plot, putting the data at the wrong x positions and
            # making the IQ spec drift relative to TFMF below.
            hop = 1024   # engine FFT hop at 48 kHz (fft_size // 2)
            t_filled = n_t * hop / float(rate)
            img.setRect(QtCore.QRectF(_t_offset, -half_rate_khz,
                                       t_filled, 2 * half_rate_khz))
            plot.setXRange(_x_lo, _x_hi, padding=0)
            if _is_audio:
                plot.setYRange(0, 3, padding=0)             # audio band only
            else:
                plot.setYRange(-half_rate_khz, half_rate_khz, padding=0)
            # Defensive: re-disable auto in case the operator toggled X-Auto
            # / Y-Auto from the right-click menu — otherwise pyqtgraph would
            # immediately auto-fit and undo the setXRange above.
            plot.getViewBox().disableAutoRange()

        # Top wideband-IQ spec stays visible for all source types — it's the
        # primary, consistent (time, freq) view across captures.  Operator
        # decision 2026-05-29: prefer one consistent IQ-based view at the
        # top over a separate 12 kHz audio spectrogram at the bottom (the
        # latter is dropped — see ``_audio_plot.hide()`` below).
        _set_spec(self._spec_plot, self._spec_img, spec)
        if dual and spec_v is not None:
            _set_spec(self._spec_plot_v, self._spec_img_v, spec_v)

        # Blanker overlay disabled in the analysis window: for bursty audio
        # the blanker fires many times per second (often dozens of strokes
        # inside the burst envelope), and with semi-transparent red those
        # overlapping strokes painted a continuous dull-red band across
        # 0–0.6 s that hid the spectrogram beneath.  The blanker is a
        # diagnostic for live runs; in a forensic single-burst view it just
        # obscures the signal we're trying to inspect.
        self._blank_bars.setData(x=[], y=[])

        # ── sq_det squared-signal spectrogram (replaces the channel-SNR HM) ─
        # The image's freq axis is the FULL spectrum from the squared-spec
        # function (0..rate/2 for real input, ±rate/2 for complex IQ); the
        # display viewbox zooms to 0..6 kHz where the squared MSK144 tones
        # actually sit (2·(fc±500) = 2000 and 4000 Hz for fc=1500).
        _sq_db    = results.get('sq_spec_db')
        _sq_freq  = results.get('sq_spec_freq_hz')
        _sq_t     = results.get('sq_spec_time_s')
        _dlvl     = [self._det_vmin_slider.value(), self._det_vmax_slider.value()]

        def _set_sq_spec(plot, img, spec_db, freq_hz_axis, t_arr):
            if spec_db is None or spec_db.size == 0 or freq_hz_axis is None:
                img.setImage(np.zeros((1, 1), dtype=np.float32),
                             autoLevels=False, levels=_dlvl)
                return
            # Display freq axis in kHz; rect spans the actual freq range.
            f_lo_khz = float(freq_hz_axis[0])  / 1000.0
            f_hi_khz = float(freq_hz_axis[-1]) / 1000.0
            n_t, n_f = spec_db.shape
            # Per-bin pct25 normalization (same convention TFMF uses below).
            # Each frequency column gets its own noise reference computed
            # across time, so silent bins and busy-bins both end up at "0 dB
            # above own noise floor" — producing a smooth gradient that the
            # 0..50 dB sliders can render as useful contrast.  The earlier
            # global-pct25 form gave a bimodal display: the squared-FFT
            # epsilon floor (1e-20 → −200 dB) dominated the global pct25,
            # leaving the actual noise/signal continuum collapsed into the
            # all-black or all-white extremes the operator was seeing.
            _clipped = np.maximum(spec_db, -130.0).astype(np.float32)
            _bin_ref = np.percentile(_clipped, 25, axis=0).astype(np.float32)
            spec_db_norm = (_clipped - _bin_ref[None, :])
            _dw  = max(plot.width(), 1)
            _rep = max(1, -(-_dw // n_t))
            img.setImage(np.repeat(spec_db_norm, _rep, axis=0),
                         autoLevels=False, levels=_dlvl)
            # Honest time scale: each FFT column is placed at its center
            # time t_arr[k], and the image spans the half-stride margins
            # on either side.  Previous code placed col-0 at x=0 and col
            # n-1 at x=t_arr[-1] — that compressed the data by one stride
            # and shifted everything left by ~9 ms.  Live ch_detect_img
            # uses the same convention (rect spans the actual buffer
            # extent, see displays.py:392).  For an msk144spd run on a
            # 1.9-s WAV: t_arr[0]=36 ms, t_arr[-1]=1.854 s, stride=18 ms,
            # so the image now spans 27 ms .. 1.863 s — the genuine
            # time extent of the FFT windows that produced this data.
            if t_arr is not None and len(t_arr) > 1:
                stride_s = float(t_arr[1] - t_arr[0])
            else:
                stride_s = 0.018   # WSJT-X msk144spd default stride at 12 kHz
            t_left  = float(t_arr[0]) - stride_s / 2.0
            t_width = stride_s * float(n_t)
            img.setRect(QtCore.QRectF(_t_offset + t_left, f_lo_khz,
                                       t_width, f_hi_khz - f_lo_khz))
            plot.setXRange(_x_lo, _x_hi, padding=0)
            plot.setYRange(0, 6, padding=0)   # 0..6 kHz audio (squared band)
            plot.getViewBox().disableAutoRange()

        _set_sq_spec(self._det_plot, self._det_img, _sq_db, _sq_freq, _sq_t)
        if dual:
            _set_sq_spec(self._det_plot_v, self._det_img_v, _sq_db, _sq_freq, _sq_t)

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
            _rect = QtCore.QRectF(_t_offset, _TF_F_MIN, t_span, _TF_F_SPAN)
            img.setImage(surf, autoLevels=False, levels=_tfmf_lvl)
            img.setRect(_rect)
            plot.setXRange(_x_lo, _x_hi, padding=0)
            if _is_audio:
                # Same audio-band zoom as the IQ spectrogram for visual
                # alignment.  The image still extends across ±24 kHz; the
                # viewbox just shows the 0–3 kHz slice.
                plot.setYRange(0, 3, padding=0)
            else:
                plot.setYRange(_TF_F_MIN - 0.5, _TF_F_MIN + _TF_F_SPAN + 0.5,
                                padding=0)
            plot.getViewBox().disableAutoRange()
            if cands:
                scatter.setData(x=[c.time_s + _t_offset for c in cands],
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
            self._env_curve.setData(_ift_t + _t_offset, _env_db)
            self._env_plot.setXRange(_x_lo, _x_hi, padding=0)
            # Y range comes from the env Y sliders (operator-adjustable on
            # the right of row 4-5).  Default 0..25 dB at slider construction
            # time covers the 17.3 dB TFMF threshold + room above.
            self._env_plot.setYRange(self._env_ymin_slider.value(),
                                     self._env_ymax_slider.value(),
                                     padding=0)
            self._env_plot.getViewBox().disableAutoRange()

            # Three-tier classification driven by the two Thresh-dB sliders
            # on the right of rows 4-5.  See _reclassify_ift_dots for the
            # bucket math; this just calls into it.
            self._reclassify_ift_dots()
            self._ift_plot.setXRange(_x_lo, _x_hi, padding=0)
            # Y range fixed at 0..3 kHz (MSK144 audio carrier band).
            self._ift_plot.setYRange(0, 3, padding=0)
            self._ift_plot.getViewBox().disableAutoRange()
        else:
            self._env_curve.setData([], [])
            self._ift_scatter_sub.setData(x=[], y=[])
            self._ift_scatter_burst.setData(x=[], y=[])
            self._ift_scatter_strong.setData(x=[], y=[])
            self._ift_chain_line.setData(x=[], y=[])

        # ── Audio spectrogram (row 1, audio-burst mode) ───────────────────
        # Reinstated 2026-05-29 with the new row order: audio spec is now
        # the TOP row, computed from the NATIVE 12 kHz mono audio (no
        # Hilbert/resample), per-bin pct25 normalised, displayed 0..6 kHz.
        # For 48 kHz IQ captures (no native_audio in meta), leave the
        # audio panel empty — the wideband IQ spec (row 6) takes its place
        # and that branch is what 48 kHz captures should look at.
        _native_audio    = results.get('native_audio')
        _native_audio_sr = results.get('native_audio_sr')
        if _native_audio is not None and _native_audio_sr:
            self._show_audio_spectrogram(np.asarray(_native_audio, dtype=np.float32))
        else:
            self._audio_img.setImage(np.zeros((1, 1), dtype=np.float32),
                                      autoLevels=False,
                                      levels=[self._audio_vmin_slider.value(),
                                              self._audio_vmax_slider.value()])

        # Show audio-spec (row 1) for audio-burst sources, hide the IQ spec
        # (row 0 → repositioned to bottom by the setup_ui reorder).  When a
        # true 48 kHz IQ capture arrives, swap which row container is shown.
        # V scaffolding stays inside whichever row is visible so future
        # dual-pol captures can light it up.
        if _is_audio:
            self._row_iq_spec.hide()
            self._row_audio_spec.show()
        else:
            self._row_iq_spec.show()
            self._row_audio_spec.hide()

        # ── Row 6: sync frame phase + (eventually) coherence ──────────────
        # Computed from TFMF candidates' refined sync onset.  Each candidate
        # has a sub-sample-refined sample index that, mod NSPM_48k, gives
        # the frame-phase fingerprint — constant across frames from a
        # single transmitter, scrambled across stations.
        self._render_epoch_row()

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

        # Clear the green/orange marker curves on the sq_det row.  Their
        # ``freq_khz`` is "dial offset from calling" (where 0 = the
        # calling-freq channel), inherited from the channel-SNR HM layout.
        # That coordinate doesn't align with the squared-spec Y axis here
        # (which is 2·audio_carrier kHz; signal at audio 1.5 kHz lives at
        # 3 kHz, not 0).  Leave the curves empty until we rewire the marker
        # freq to squared-spec coordinates.
        for curve in (self._det_curve_green, self._det_curve_orange):
            curve.setData(x=[], y=[])
        if dual:
            for curve in (self._det_curve_green_v, self._det_curve_orange_v):
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
        # Image rect is in kHz on the Y axis (matches the operator's request
        # 2026-05-29: all vertical axes in kHz for uniformity).
        bin_khz = bin_hz / 1000.0
        rect_h_khz = rect_h / 1000.0
        # Apply the period-time offset (set in _render) so the audio image
        # lands at the same X coords as the other rows.  Falls back to 0
        # for cases where _t_offset hasn't been set yet (e.g. manual-decode
        # click before any _render has run).
        _t_off = getattr(self, '_t_offset', 0.0)
        self._audio_img.setRect(QtCore.QRectF(_t_off, -bin_khz / 2.0,
                                              duration, rect_h_khz))
        _x_lo = getattr(self, '_x_min', _t_off)
        _x_hi = getattr(self, '_x_max', _t_off + duration)
        self._audio_plot.setXRange(_x_lo, _x_hi, padding=0)
        self._audio_plot.setYRange(0, 3, padding=0)
        self._audio_plot.getViewBox().disableAutoRange()
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

    def _on_env_y_changed(self):
        # Env slider adjusts the viewbox Y range (not image levels).
        self._env_plot.setYRange(self._env_ymin_slider.value(),
                                 self._env_ymax_slider.value(),
                                 padding=0)

    def _on_epoch_thresh_changed(self):
        # Both coherence and SNR thresholds gate which TFMF candidates
        # render on the epoch plot.  Re-render from cached results.
        if self._results is not None:
            self._render_epoch_row()

    def _on_ift_thresh_changed(self):
        # Enforce the invariant low ≤ high.  If the user drags the LOW
        # slider above HIGH, bump HIGH up to match (and vice-versa).  Block
        # signals during the corrective move so we don't recurse.
        lo = self._ift_thresh_low.value()
        hi = self._ift_thresh_high.value()
        if lo > hi:
            sender = self.sender()
            if sender is self._ift_thresh_low:
                self._ift_thresh_high.blockSignals(True)
                self._ift_thresh_high.setValue(lo)
                self._ift_thresh_high.blockSignals(False)
                self._ift_thresh_high_label.setText(str(lo))
            else:
                self._ift_thresh_low.blockSignals(True)
                self._ift_thresh_low.setValue(hi)
                self._ift_thresh_low.blockSignals(False)
                self._ift_thresh_low_label.setText(str(hi))
        # Re-classify scatter dots, but only if we have cached freq/env
        # data from the last render to re-bucket.
        if self._results is not None:
            self._reclassify_ift_dots()

    def _compute_candidate_coherences(self, results: dict) -> list:
        """For each TFMF candidate, return coherence ∈ [0, 1] (or None).

        Channelizes the 48 kHz IQ to 12 kHz at the candidate's freq,
        extracts NSPM=864 samples starting at the refined sync onset, and
        calls MAP144's :func:`_sync_phase_features` (the same function
        that produces ``sync_phase_coherence_h`` in launches.jsonl).
        Real MSK144 → coh ≈ 1; impulse noise → ≈ 0.1.
        """
        cands = results.get('tfmf_candidates_h') or []
        if not cands:
            return []
        iq_h = results.get('iq_h_arr')
        if iq_h is None:
            # iq_h_arr isn't in results today — we'll pull it from the
            # engine instance below (kept after replay for manual_decode).
            iq_h = getattr(self._engine, '_iq_data', None)
            if iq_h is not None and iq_h.ndim == 2:
                iq_h = iq_h[:, 0]
        if iq_h is None:
            return [None] * len(cands)
        from scipy.signal import decimate
        from .msk144_spd import (_sync_phase_features, _sync_correlate_batch,
                                  NSPM)
        FS = 48000
        N_TPL = 1536
        PRE_PAD_48K  = 200       # filter pre-pad (50 samples at 12 kHz)
        POST_PAD_48K = 100
        NEED = NSPM * 4 + PRE_PAD_48K + POST_PAD_48K
        pre_pad_12k = PRE_PAD_48K // 4
        coherences = []
        for c in cands:
            refined_onset = int(round(
                c.time_sample - N_TPL // 2 + c.sample_epoch_offset_samples))
            start = refined_onset - PRE_PAD_48K
            if start < 0 or start + NEED > len(iq_h):
                coherences.append(None)
                continue
            seg = iq_h[start:start + NEED]
            t = np.arange(len(seg), dtype=np.float32) / FS
            seg_dc = (seg * np.exp(-2j * np.pi * float(c.freq_hz) * t)
                      ).astype(np.complex64)
            try:
                seg_12k = decimate(seg_dc, 4, ftype='fir', zero_phase=True
                                   ).astype(np.complex64)
                if len(seg_12k) < pre_pad_12k + NSPM:
                    coherences.append(None)
                    continue
                c_window = seg_12k[pre_pad_12k:pre_pad_12k + NSPM].astype(np.complex64)
                # Locate sync in-window first; ish_best=0 is only valid when
                # refined_onset is sub-stride accurate — not for spurious
                # candidates.  See processing._tfmf_candidate_coherence for
                # the empirical justification.
                _xcc, _peak, _ish = _sync_correlate_batch(c_window[np.newaxis, :])
                ish_best = int(_ish[0])
                coh, _ts, _foff = _sync_phase_features(c_window, ish_best)
                coherences.append(float(coh))
            except Exception:
                coherences.append(None)
        return coherences

    def _render_epoch_row(self):
        """Populate row 6 with one dot per TFMF candidate.

        Each candidate exposes a sub-stride-refined sync onset:
            refined_sample = c.time_sample - N//2 + c.sample_epoch_offset_samples
        Y = ``refined_sample % NSPM_48k`` (converted to ms; range 0..72).
        Same-station candidates cluster at one Y (their per-path delay is
        constant within a 15-s period); different stations land at
        different Ys.  Color (currently SNR; coherence added next pass)
        separates real-signal dots from threshold-edge noise candidates.

        Operator sliders (SNR threshold + coherence threshold) gate which
        candidates render.  Coherence not yet computed per-candidate, so
        the coherence threshold is a no-op for now (placeholder UI).
        """
        r = self._results or {}
        cands = r.get('tfmf_candidates_h') or []
        if not cands:
            self._epoch_scatter.setData(x=[], y=[])
            return
        # TFMF constants — N (template length) = 1536; NSPM at 48 kHz = 864*4.
        from .processing import _TFMF_DISP_FS_HZ as _FS
        NSPM_48K = 864 * 4
        N_TPL    = 1536
        snr_thr_db   = float(self._epoch_snr_thresh.value())
        coh_thr_pct  = float(self._epoch_coh_thresh.value())  # 0..100 → 0..1
        coh_thr      = coh_thr_pct / 100.0
        t_offset     = float(getattr(self, '_t_offset', 0.0))
        coh_list     = getattr(self, '_cand_coherences', [None] * len(cands))
        # Build the scatter spots — one per candidate that passes both
        # SNR and coherence thresholds.  Color by COHERENCE (the actual
        # noise-vs-signal discriminator), size by SNR.
        spots = []
        for i, c in enumerate(cands):
            if float(c.snr_db) < snr_thr_db:
                continue
            coh = coh_list[i] if i < len(coh_list) else None
            if coh is not None and coh < coh_thr:
                continue
            refined = (int(c.time_sample) - N_TPL // 2
                       + float(c.sample_epoch_offset_samples))
            frame_phase_samples = refined % NSPM_48K
            frame_phase_ms = frame_phase_samples / _FS * 1000.0
            x = float(c.time_s) + t_offset
            # Coherence → brush color.  Grey-orange (low, ≈ noise) →
            # bright green (high, ≈ real MSK144).  Equal luminance match
            # to the in-burst green used on row 5.  Unknown coh → grey.
            if coh is None:
                br = pg.mkBrush(160, 160, 160, 180)
            else:
                _n = max(0.0, min(1.0, (coh - 0.2) / 0.7))   # map 0.2..0.9 → 0..1
                r8 = int(220 - _n * (220 - 60))    # 220 → 60
                g8 = int(160 + _n * (220 - 160))   # 160 → 220
                b8 = int(80  + _n * (80  - 80))    # 80
                br = pg.mkBrush(r8, g8, b8, 220)
            # Size → SNR (small near threshold, large for strong).
            _ns = max(0.0, min(1.0, (float(c.snr_db) - 17.0) / 23.0))
            size = 5 + int(_ns * 7)   # 5 (threshold) .. 12 (40+ dB)
            spots.append({
                'pos':   (x, frame_phase_ms),
                'brush': br,
                'size':  size,
            })
        self._epoch_scatter.setData(spots=spots)

        # Live-detector cross marker — from launches.jsonl entry if present.
        # X = t_sec (period-time, no offset adjustment — already absolute).
        # Y = sync_t_sample_h / 12 ms  (channelizer is at 12 kHz, so the
        #     sample-domain peak index in [0, NSPM=864) directly converts
        #     to a 0..72 ms frame phase via /12000 × 1000).
        # Fill color reflects live sync_phase_coherence_h (same colormap
        # as the TFMF dots so visual comparison is direct).
        le = getattr(self, '_launch_entry', None)
        if le is not None:
            try:
                _lt   = float(le.get('t_sec'))
                _lsmp = float(le.get('sync_t_sample_h'))
                _lcoh = le.get('sync_phase_coherence_h')
                _lphase_ms = _lsmp / 12.0
                if _lcoh is None:
                    _br = pg.mkBrush(160, 160, 160, 0)   # outline only
                else:
                    _nc = max(0.0, min(1.0, (float(_lcoh) - 0.2) / 0.7))
                    r8 = int(220 - _nc * (220 - 60))
                    g8 = int(160 + _nc * (220 - 160))
                    b8 = 80
                    _br = pg.mkBrush(r8, g8, b8, 200)
                self._epoch_live_marker.setData(
                    [{'pos': (_lt, _lphase_ms), 'brush': _br, 'size': 18}])
            except (TypeError, ValueError):
                self._epoch_live_marker.setData([])
        else:
            self._epoch_live_marker.setData([])

        self._epoch_plot.setXRange(self._x_min, self._x_max, padding=0)
        self._epoch_plot.setYRange(0, 72, padding=0)
        self._epoch_plot.getViewBox().disableAutoRange()

    def _reclassify_ift_dots(self):
        """Bucket the TFMF peak-freq scatter dots into three colors based on
        the operator's two threshold sliders.  Reads ift_t / ift_f / env_db
        from the most-recent results dict; freq is converted Hz → kHz so the
        Y axis stays consistent with the other rows' kHz convention."""
        r = self._results or {}
        _t   = r.get('inst_freq_t')
        _f   = r.get('inst_freq_audio')
        _env = r.get('env_db')
        if _t is None or _f is None or _env is None or _t.size == 0:
            self._ift_scatter_sub.setData(x=[], y=[])
            self._ift_scatter_burst.setData(x=[], y=[])
            self._ift_scatter_strong.setData(x=[], y=[])
            self._ift_chain_line.setData(x=[], y=[])
            return
        lo = float(self._ift_thresh_low.value())
        hi = float(self._ift_thresh_high.value())
        if hi < lo:
            hi = lo   # defensive: high slider crossed below low
        f_khz = _f / 1000.0   # all vertical axes in kHz
        # Apply the period-time offset to the X coords (same convention
        # the other rows use).  _t_offset is set in _render.
        _t_disp = _t + getattr(self, '_t_offset', 0.0)
        is_sub    = _env <  lo
        is_strong = _env >= hi
        is_mid    = ~is_sub & ~is_strong
        self._ift_scatter_sub.setData   (x=_t_disp[is_sub],    y=f_khz[is_sub])
        self._ift_scatter_burst.setData (x=_t_disp[is_mid],    y=f_khz[is_mid])
        self._ift_scatter_strong.setData(x=_t_disp[is_strong], y=f_khz[is_strong])
        # Chain line through above-threshold dots (mid + strong).  NaN at
        # sub-threshold positions → pyqtgraph breaks the line over gaps
        # (connect='finite').  Same convention as the matplotlib 'o-' plot
        # in freqtime_*.png reference figure.
        _above = is_mid | is_strong
        _y_chain = np.where(_above, f_khz, np.nan).astype(np.float32)
        self._ift_chain_line.setData(x=_t_disp, y=_y_chain)

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

    def _update_launch_info_bar(self):
        """Render the matched launches.jsonl entry's key metrics into the
        second info row.  Hidden when no match (test WAVs, manually-loaded
        captures not from the live runtime, etc.).  Also positions the
        per-plot detection-time vertical lines."""
        r = getattr(self, '_launch_entry', None)
        if not r:
            self._launch_info_label.setVisible(False)
            for _ln in getattr(self, '_det_time_lines', ()):
                _ln.setVisible(False)
            return
        # Place the detection-time marker at the live runtime's t_sec
        # (already in within-15-s-period coordinates).  Same coordinate
        # system as the plots after the period-time offset applied in
        # _render, so a single x value works for all five plots.
        try:
            _t_det = float(r.get('t_sec'))
        except (TypeError, ValueError):
            _t_det = None
        for _ln in getattr(self, '_det_time_lines', ()):
            if _t_det is not None:
                _ln.setPos(_t_det)
                _ln.setVisible(True)
            else:
                _ln.setVisible(False)
        def _f(key, fmt='%.2f', default='—'):
            v = r.get(key)
            if v is None:
                return default
            try:
                return fmt % float(v)
            except (TypeError, ValueError):
                return str(v)
        parts = [
            f"LIVE   t={_f('t_sec', '%5.2f')}s",
            f"radio={_f('radio_khz', '%.1f')}kHz",
            f"ch={r.get('ch_signed', '—')}",
            f"src={r.get('det_source', '—')}",
            f"outcome={r.get('outcome', '—')}",
            f"sq={_f('sq_metric_db_h', '%.1f')}dB",
            f"sync={_f('sync_metric_db_h', '%.1f')}dB",
            f"pair={_f('pair_metric_db_h', '%.1f')}dB",
            f"coh={_f('sync_phase_coherence_h', '%.2f')}",
            f"fc_off={_f('sync_freq_offset_hz_h', '%+.1f')}Hz",
            f"clust={r.get('n_cluster_chan', '—')}ch",
            f"jt9_snr={_f('jt9_snr_db', '%+d')}",
        ]
        self._launch_info_label.setText("  ".join(parts))
        self._launch_info_label.setVisible(True)

    # ── Settings persistence ───────────────────────────────────────────────────

    def _restore_settings(self):
        """Restore window geometry, slider values, and splitter positions from QSettings."""
        s = self._SETTINGS
        geom = s.value('analysis_window_geometry')
        if geom is not None:
            self.restoreGeometry(geom)
        for attr, key in (
            ('_main_h',      'analysis_main_h_state'),
            # Bumped to _v2 (2026-05-29) when rows were reordered to
            # audio/sq_det/TFMF/env/ift/IQ.  Restoring a pre-reorder state
            # squashed the new top row (audio) to near-zero height because
            # the old state had IQ spec at index 0 with a small stretch.
            ('_rows_vsplit', 'analysis_rows_vsplit_state_v3'),
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
        s.setValue('analysis_rows_vsplit_state_v3',  self._rows_vsplit.saveState())
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
