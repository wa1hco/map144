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
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import __version__

# ── Decode-panel age colours (matches MAP65 convention) ───────────────────────
_AGE_COLORS = [
    ( 120, QtGui.QColor('#ffffff')),   # < 2 min  — white
    ( 480, QtGui.QColor('#ffff80')),   # < 8 min  — yellow
    (1200, QtGui.QColor('#ff9040')),   # < 20 min — orange
    (None, QtGui.QColor('#707070')),   # older    — gray
]

def _age_color(age_s: float) -> QtGui.QColor:
    for threshold, color in _AGE_COLORS:
        if threshold is None or age_s < threshold:
            return color
    return _AGE_COLORS[-1][1]

def recolor_decode_panel(panel: QtWidgets.QListWidget) -> None:
    """Update foreground colour of every item in the decode panel by age."""
    now = time.time()
    for i in range(panel.count()):
        item = panel.item(i)
        ts = item.data(QtCore.Qt.UserRole)
        if ts is not None:
            item.setForeground(_age_color(now - ts))


_MSG_COL_WIDTH = 13   # matches WSJT-X displaytext.cpp WORD1_WIDTH / WORD2_WIDTH
                      # worst-case callsign: <...>/AB1CDE = 12 chars → fits with 1-space pad

def _align_msk144_message(msg: str) -> str:
    """Left-justify the three-part MSK144 message into fixed-width columns.

    MSK144 messages have the form ``WORD1 WORD2 WORD3`` where WORD1 and WORD2
    are callsigns (or compounds like ``<...>/AB1CDE``).  Pad each to
    ``_MSG_COL_WIDTH`` chars so the decode panel lines up regardless of
    callsign length, mirroring the WSJT-X ``alignCalls`` mod in
    ``displaytext.cpp``.
    """
    parts = msg.split(None, 2)   # split on whitespace, at most 3 pieces
    if len(parts) < 3:
        return msg               # CQ / short messages — leave as-is
    w1, w2, rest = parts
    return w1.ljust(_MSG_COL_WIDTH) + w2.ljust(_MSG_COL_WIDTH) + rest


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


# Epoch-samples that span the full hue wheel.  The parabolic sub-stride epoch
# clusters tightly (measured 2026-05-31 on 5000 live candidates: std ≈ 2.4
# samples, 91% within ±4, full range ±12 hit by <1%), so mapping the
# theoretical ±stride/2 = ±12 to the wheel wasted ~2/3 of the colours —
# everything landed green→cyan→blue.  8 maps the ±4 bulk across the whole wheel
# for strong per-station contrast; the rare ±8-12 tail aliases (acceptable).
# Tunable: smaller → more contrast + more tail aliasing; larger → calmer.
_CAND_MAP_EPOCH_HUE_SPAN = 8.0


def _cand_map_size(snr_db: float) -> float:
    """Scatter marker size from SNR (dB), clamped to a legible pixel range."""
    return max(4.0, min(22.0, 4.0 + (float(snr_db) - 10.0) * 0.9))


def _cand_map_brush(epoch_samples: float, coherence):
    """Per-candidate brush: hue from sample-epoch (cyclic — same propagation
    delay ⇒ same hue ⇒ same station, even across frequencies), alpha from
    within-frame sync coherence (vivid = MSK144-like, faded = incoherent spur).
    ``coherence`` may be None (not computed) → a neutral mid alpha."""
    import pyqtgraph as pg
    hue = ((float(epoch_samples) / _CAND_MAP_EPOCH_HUE_SPAN) + 0.5) % 1.0
    coh = 0.5 if coherence is None else max(0.0, min(1.0, float(coherence)))
    return pg.mkBrush(pg.hsvColor(hue, 1.0, 1.0, 0.25 + 0.75 * coh))


def _usrp_startup_status(self):
    """B210 startup status line for the receiver label, or None once streaming
    (so the normal flow status takes over).  Driven by USRPSource.startup_phase
    so the operator sees the firmware/FPGA load and slow-USB stream-start with an
    elapsed counter + a USB/power hint, instead of a multi-minute black box."""
    if getattr(self, 'source_mode', None) != 'usrp':
        return None
    uc = getattr(self, 'usrp_client', None)
    if uc is None:
        return None
    phase = getattr(uc, 'startup_phase', 'idle')
    if phase in ('idle', 'streaming'):
        return None
    import time as _t
    el = _t.time() - getattr(uc, 'startup_t0', _t.time())
    _names = {'opening':     'opening device (firmware/FPGA)',
              'configuring': 'configuring',
              'waiting_iq':  'waiting for IQ'}
    msg = f"USRP B210 — {_names.get(phase, phase)} {el:.0f}s"
    detail = getattr(uc, 'startup_detail', '')   # latest UHD log line (Stage 2)
    if detail:
        msg += f"  · {detail}"
    if (phase == 'waiting_iq' and el > 20.0) or (phase == 'opening' and el > 30.0):
        msg += "   ⚠ USB link slow — check hub/power"
    return msg


def update_displays(self):
    """Update all display panels."""
    if len(self.realtime_data) == 0:
        return

    tuned_freq_mhz, tuned_source, tuned_bandwidth_hz = self._get_tuned_frequency_mhz()
    center_freq_mhz = tuned_freq_mhz if tuned_freq_mhz is not None else self.center_freq_mhz

    if abs(center_freq_mhz - self.display_center_freq_mhz) > 1e-9:
        self.display_center_freq_mhz = center_freq_mhz
        self.freq_axis = self.fft_bin_axis_mhz + center_freq_mhz
        # Keep the channelizer NCO aligned to the actual pan centre.
        # This covers both startup (Flex reports a slightly different frequency
        # than the requested value) and any pan moves during operation.
        if abs(center_freq_mhz - self.center_freq_mhz) > 1e-6:
            print(f"[channelizer] pan centre updated: "
                  f"{self.center_freq_mhz:.4f} → {center_freq_mhz:.4f} MHz  "
                  f"(rebuilding NCO)", flush=True)
            self.center_freq_mhz = center_freq_mhz
            self._rebuild_channelizer_state()
        rpt = getattr(self, 'reporter', None)
        if rpt is not None:
            rpt.report_freq(int(round(center_freq_mhz * 1e6)))

        from .channel_plan import (
            channel_summary_lines,
            format_monitor_status,
            monitored_if_span_khz,
        )

        _if_lo, _if_hi, _n_ch = monitored_if_span_khz()
        _mon_lbl = getattr(self, '_msk_monitor_label', None)
        if _mon_lbl is not None:
            _mon_lbl.setText(format_monitor_status(_if_lo, _if_hi, _n_ch))
            _lines = channel_summary_lines(center_freq_mhz)
            _tip = "Coarse IF channel centres (usable rows):\n\n" + "\n".join(_lines[:14])
            if len(_lines) > 14:
                _tip += "\n…"
            _mon_lbl.setToolTip(_tip)

    # Use symmetric ±half_bw around centre rather than the raw fftfreq bin edges.
    # fftfreq for even N is asymmetric (one extra negative bin), so the last
    # positive bin is at centre + (N/2−1)×Δf, not centre + N/2×Δf.  After the
    # [:, ::4] frequency subsampling the mismatch widens further.  Matching the
    # analysis window's ±sample_rate/2 convention keeps both displays identical.
    _half_bw_mhz = self.sample_rate / 2e6          # e.g. 0.024 MHz for 48 kHz
    freq_min = center_freq_mhz - _half_bw_mhz
    freq_max = center_freq_mhz + _half_bw_mhz

    self._noise_floor_ctr = getattr(self, '_noise_floor_ctr', 0) + 1

    # Show/hide panes based on dual_pol and source_mode.
    # V real-time pane follows dual_pol.
    # Previous-15s panes are only meaningful for live sources — hide in WAV mode.
    _dual    = getattr(self, 'dual_pol', False)
    _is_live = getattr(self, 'source_mode', 'idle') not in ('idle', 'wav')
    _rt_v_plot  = getattr(self, 'realtime_plot_v',    None)
    _sp_plot    = getattr(self, 'spectrogram_plot',   None)
    _sp_v_plot  = getattr(self, 'spectrogram_plot_v', None)

    # V real-time: show when dual_pol
    if _rt_v_plot is not None:
        if _dual and not _rt_v_plot.isVisible():
            _rt_v_plot.show()
        elif not _dual and _rt_v_plot.isVisible():
            _rt_v_plot.hide()

    # Previous-15s H pane: show only for live sources
    if _sp_plot is not None:
        if _is_live and not _sp_plot.isVisible():
            _sp_plot.show()
        elif not _is_live and _sp_plot.isVisible():
            _sp_plot.hide()

    # Previous-15s V pane: show only for live dual-pol sources
    if _sp_v_plot is not None:
        if _is_live and _dual and not _sp_v_plot.isVisible():
            _sp_v_plot.show()
        elif (not _is_live or not _dual) and _sp_v_plot.isVisible():
            _sp_v_plot.hide()

    # Pair separator: visible only when the previous-15s pair is visible
    _sep = getattr(self, '_fg_pair_sep', None)
    if _sep is not None:
        if _is_live and not _sep.isVisible():
            _sep.show()
        elif not _is_live and _sep.isVisible():
            _sep.hide()

    _fg_win     = getattr(self, '_fast_graph_win', None)
    _fg_visible = _fg_win is None or _fg_win.isVisible()

    if _fg_visible and self.spec_staging_filled:
        spec_array = self.spectrogram_data

        # Gate setImage to once per 15-second boundary — the accumulated
        # spectrogram only changes when processing.py flips spec_boundary,
        # so calling setImage every 100 ms burns GPU/CPU for no visual gain.
        _acc_boundary = getattr(self, '_acc_boundary_rendered', None)
        if _acc_boundary != self.spec_boundary:
            self._acc_boundary_rendered = self.spec_boundary

            # Repeat time columns so image is wider than the display widget.
            # PyQtGraph leaves 1-px black gaps at column boundaries when the
            # image is upscaled (fewer columns than display pixels).
            _dw  = max(self.spectrogram_plot.width(), 1)
            _rep = max(1, -(-_dw // spec_array.shape[0]))
            self.spectrogram_img.setImage(
                np.repeat(spec_array[:, ::4], _rep, axis=0),
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

            self.spectrogram_plot.setXRange(0, self.max_time + 0.5, padding=0)
            self.spectrogram_plot.setYRange(freq_min, freq_max, padding=0)

            # V-channel accumulated spectrogram
            if _dual and _sp_v_plot is not None:
                spec_v = self.spectrogram_data_v
                _dw_v  = max(_sp_v_plot.width(), 1)
                _rep_v = max(1, -(-_dw_v // spec_v.shape[0]))
                self.spectrogram_img_v.setImage(
                    np.repeat(spec_v[:, ::4], _rep_v, axis=0),
                    autoLevels=False,
                    levels=[self.min_level, self.max_level],
                )
                self.spectrogram_img_v.setRect(QtCore.QRectF(
                    0, freq_min, self.max_time, freq_max - freq_min,
                ))
                _sp_v_plot.setXRange(0, self.max_time + 0.5, padding=0)
                _sp_v_plot.setYRange(freq_min, freq_max, padding=0)

    if _fg_visible and self.realtime_filled and getattr(self, '_realtime_dirty', False):
        self._realtime_dirty = False

        # Capture array references once so the display uses a consistent
        # snapshot even if the processing thread swaps realtime_data mid-render.
        _rt_arr   = self.realtime_data
        _rt_arr_v = self.realtime_data_v if _dual else None

        # Subsample frequency axis by 4; repeat time columns to avoid
        # PyQtGraph upscaling gap artifacts (see accumulated spectrogram above).
        _dw  = max(self.realtime_plot.width(), 1)
        _rep = max(1, -(-_dw // _rt_arr.shape[0]))
        self.realtime_img.setImage(
            np.repeat(_rt_arr[:, ::4], _rep, axis=0),
            autoLevels=False,
            levels=[self.min_level, self.max_level],
        )

        _rt_rect = QtCore.QRectF(0, freq_min, self.realtime_time, freq_max - freq_min)
        self.realtime_img.setRect(_rt_rect)

        # Only update the viewport range when it has actually changed — calling
        # setXRange/setYRange every frame triggers a second repaint per cycle
        # (viewport change → repaint) on top of the setImage repaint, causing
        # the right edge of the spectrogram to flicker.
        _rt_range_key = (self.realtime_time, freq_min, freq_max)
        if getattr(self, '_rt_range_key', None) != _rt_range_key:
            self._rt_range_key = _rt_range_key
            self.realtime_plot.setXRange(0, self.realtime_time + 0.5, padding=0)
            self.realtime_plot.setYRange(freq_min, freq_max, padding=0)

        # V-channel real-time spectrogram
        if _dual and _rt_v_plot is not None and _rt_arr_v is not None:
            _dw_v  = max(_rt_v_plot.width(), 1)
            _rep_v = max(1, -(-_dw_v // _rt_arr_v.shape[0]))
            self.realtime_img_v.setImage(
                np.repeat(_rt_arr_v[:, ::4], _rep_v, axis=0),
                autoLevels=False,
                levels=[self.min_level, self.max_level],
            )
            self.realtime_img_v.setRect(QtCore.QRectF(
                0, freq_min, self.realtime_time, freq_max - freq_min,
            ))
            _rt_v_range_key = (self.realtime_time, freq_min, freq_max)
            if getattr(self, '_rt_v_range_key', None) != _rt_v_range_key:
                self._rt_v_range_key = _rt_v_range_key
                _rt_v_plot.setXRange(0, self.realtime_time + 0.5, padding=0)
                _rt_v_plot.setYRange(freq_min, freq_max, padding=0)

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
            # Control sentinel: a worker thread requests a panel wipe (new WAV
            # load / source start).  decode_panel is a QListWidget and must be
            # mutated only on the GUI thread — which is here.  The worker
            # enqueues {'clear': True} instead of touching the widget directly
            # (see _reset_wav_timeline / _clear_decode_panel_on_source_start).
            if isinstance(result, dict) and result.get('clear'):
                self.decode_panel.clear()
                continue
            mid = result.get('marker_id', -1)
            if mid in _marker_by_id:
                outcome = result.get('outcome', 'decoded')
                if outcome == 'decoded' or result.get('decoded'):
                    _marker_by_id[mid]['decoded']   = True
                    _marker_by_id[mid]['outcome']   = 'decoded'
                    _marker_by_id[mid]['decoder']   = result.get('decoder', 'jt9')
                    _marker_by_id[mid]['message']   = result.get('message')
                    _marker_by_id[mid]['theta_deg'] = result.get('theta_deg')
                else:
                    _marker_by_id[mid]['outcome'] = outcome   # 'no_decode', 'timeout', 'error'
            # Only add successful decodes to the panel — no_decode/timeout/error are noise.
            outcome = result.get('outcome', 'decoded')
            if outcome == 'decoded' or result.get('decoded'):
                msg       = result.get('message', '?')
                radio_khz = result.get('radio_khz', 0.0)
                snr       = result.get('jt9_snr')
                utc_time  = result.get('utc_time', '')
                theta_deg = result.get('theta_deg')
                df_hz     = int(round((radio_khz - self.calling_freq_mhz * 1000) * 1000))
                pol_str   = f"{int(round(theta_deg)):3d}\u00b0" if theta_deg is not None else "  --"
                snr_str   = f"{snr:+d} dB" if snr is not None else "     ?"
                _item = QtWidgets.QListWidgetItem(
                    f"{utc_time}  {radio_khz:9.3f}  {df_hz:+5d}  {pol_str}  {snr_str:>7}  {_align_msk144_message(msg)}"
                )
                _item.setForeground(_age_color(0))        # brand-new — white
                _item.setData(QtCore.Qt.UserRole, time.time())
                # Stash the saved-burst WAV path so the right-click context
                # menu in the main window can open it in the AnalysisWindow.
                _wav_path = result.get('wav_path')
                if _wav_path:
                    _item.setData(QtCore.Qt.UserRole + 1, str(_wav_path))
                self.decode_panel.insertItem(0, _item)
                # Forward to reporter (WSJT-X UDP + PSKReporter).
                # Suppress reporting when replaying a WAV file — test signals
                # and simulations must not appear on PSKReporter or DX cluster.
                rpt = getattr(self, 'reporter', None)
                if rpt is not None and getattr(self, 'source_mode', 'wav') != 'wav':
                    rpt.report_decode(result)

    # ── Detection heatmap + markers — skip when panel is hidden ──────────────
    _detect_win = getattr(self, '_detect_win', None)
    _detect_plot_v = getattr(self, 'ch_detect_plot_v', None)

    # Show/hide V heatmap pane based on dual_pol
    if _detect_plot_v is not None:
        if _dual and not _detect_plot_v.isVisible():
            _detect_plot_v.show()
        elif not _dual and _detect_plot_v.isVisible():
            _detect_plot_v.hide()

    if _detect_win is None or _detect_win.isVisible():
        _hm_rect = QtCore.QRectF(
            0.0, self._detect_freq_min_khz,
            float(self.history_secs), self._detect_freq_span_khz,
        )

        # H heatmap (always shown)
        hm_data = np.fft.fftshift(self._ch_snr_history_h, axes=1)
        _dw  = max(self.ch_detect_plot.width(), 1)
        _rep = max(1, -(-_dw // hm_data.shape[0]))
        self.ch_detect_img.setImage(
            np.repeat(hm_data, _rep, axis=0),
            autoLevels=False,
            levels=[self.detect_min_level, self.detect_max_level],
        )
        self.ch_detect_img.setRect(_hm_rect)
        self.ch_detect_plot.setXRange(0, float(self.history_secs) + 0.5, padding=0)

        # V heatmap (dual-pol only)
        if _dual and _detect_plot_v is not None:
            hm_v = np.fft.fftshift(self._ch_snr_history_v, axes=1)
            _dw_v  = max(_detect_plot_v.width(), 1)
            _rep_v = max(1, -(-_dw_v // hm_v.shape[0]))
            self.ch_detect_img_v.setImage(
                np.repeat(hm_v, _rep_v, axis=0),
                autoLevels=False,
                levels=[self.detect_min_level, self.detect_max_level],
            )
            self.ch_detect_img_v.setRect(_hm_rect)
            _detect_plot_v.setXRange(0, float(self.history_secs) + 0.5, padding=0)

    # ── TFMF time-frequency surface (period-boundary update) ─────────────────
    # _tfmf_surface_h is (n_windows, n_bins) float32 SNR-dB computed once per
    # 15-s period from the post-blanker 48 kHz IQ ring.  Image col=time,
    # row=audio-freq (fftshifted, DC centred).  Candidate scatter uses
    # fftfreq-ordered freq_hz converted to kHz.
    _TFMF_FREQ_MIN_KHZ  = -24.0   # after fftshift, bin 0 → −24 kHz
    _TFMF_FREQ_SPAN_KHZ = 48.0    # ±24 kHz wideband
    _TFMF_DISP_STRIDE   = 24      # samples/window at 48 kHz = 0.5 ms/window
    _TFMF_FS            = 48000   # post-blanker IQ rate

    _sync_win = getattr(self, '_sync_detect_win', None)
    if _sync_win is None or _sync_win.isVisible():
        _sync_plot = getattr(self, 'sync_detect_plot', None)
        if _sync_plot is not None:
            _sync_levels = [getattr(self, 'sync_detect_min_level', self.detect_min_level),
                            getattr(self, 'sync_detect_max_level', self.detect_max_level)]
            _surf_h = getattr(self, '_tfmf_surface_h', None)
            if _surf_h is not None:
                _n_win_h = _surf_h.shape[0]
                _time_span_h = _n_win_h * _TFMF_DISP_STRIDE / _TFMF_FS
                # Image x-anchor is the surface's first-sample time within
                # the current 15-s period.  Normally 0.0 (we have IQ all the
                # way back to the boundary).  As the period fills the image
                # grows from x=0 toward x=15; at the next boundary it resets.
                _offset_h = getattr(self, '_tfmf_surface_offset_s_h', 0.0)
                _tfmf_rect_h = QtCore.QRectF(
                    _offset_h, _TFMF_FREQ_MIN_KHZ, _time_span_h, _TFMF_FREQ_SPAN_KHZ
                )
                self.sync_detect_img.setImage(
                    _surf_h, autoLevels=False, levels=_sync_levels,
                )
                self.sync_detect_img.setRect(_tfmf_rect_h)
                # X range is fixed to the full period so the image's position
                # within the period is visually correct as it fills.
                _sync_plot.setXRange(0, float(self.history_secs) + 0.5, padding=0)

                _scatter = getattr(self, 'sync_detect_scatter', None)
                if _scatter is not None:
                    _cands = getattr(self, '_tfmf_candidates_h', [])
                    if _cands:
                        _scatter.setData(
                            x=[c.time_s for c in _cands],
                            y=[c.freq_hz / 1000.0 for c in _cands],
                        )
                    else:
                        _scatter.setData(x=[], y=[])

            _sync_plot_v = getattr(self, 'sync_detect_plot_v', None)
            if _dual and _sync_plot_v is not None:
                _surf_v = getattr(self, '_tfmf_surface_v', None)
                if _surf_v is not None:
                    _n_win_v = _surf_v.shape[0]
                    _time_span_v = _n_win_v * _TFMF_DISP_STRIDE / _TFMF_FS
                    _offset_v = getattr(self, '_tfmf_surface_offset_s_v', 0.0)
                    _tfmf_rect_v = QtCore.QRectF(
                        _offset_v, _TFMF_FREQ_MIN_KHZ, _time_span_v, _TFMF_FREQ_SPAN_KHZ
                    )
                    self.sync_detect_img_v.setImage(
                        _surf_v, autoLevels=False, levels=_sync_levels,
                    )
                    self.sync_detect_img_v.setRect(_tfmf_rect_v)
                    _sync_plot_v.setXRange(0, float(self.history_secs) + 0.5, padding=0)

                    _scatter_v = getattr(self, 'sync_detect_scatter_v', None)
                    if _scatter_v is not None:
                        _cands_v = getattr(self, '_tfmf_candidates_v', [])
                        if _cands_v:
                            _scatter_v.setData(
                                x=[c.time_s for c in _cands_v],
                                y=[c.freq_hz / 1000.0 for c in _cands_v],
                            )
                        else:
                            _scatter_v.setData(x=[], y=[])
                if _surf_v is not None:
                    if not _sync_plot_v.isVisible():
                        _sync_plot_v.show()
            elif _sync_plot_v is not None and _sync_plot_v.isVisible():
                _sync_plot_v.hide()

    # ── Sync Candidate Map ────────────────────────────────────────────────────
    # Scatter of the same TFMF candidates, encoded by station (sample-epoch →
    # hue), confidence (coherence → alpha) and strength (SNR → size).  Its own
    # window so the colour coding reads on a neutral background instead of
    # fighting the SNR heatmap.  Reads _tfmf_candidates_h (the live accumulator).
    _cm_win = getattr(self, '_cand_map_win', None)
    _cm_scatter = getattr(self, 'cand_map_scatter', None)
    if _cm_scatter is not None and (_cm_win is None or _cm_win.isVisible()):
        _cm_cands = getattr(self, '_tfmf_candidates_h', [])
        if _cm_cands:
            _cm_scatter.setData(
                x=[c.time_s for c in _cm_cands],
                y=[c.freq_hz / 1000.0 for c in _cm_cands],
                size=[_cand_map_size(c.snr_db) for c in _cm_cands],
                brush=[_cand_map_brush(getattr(c, 'sample_epoch_offset_samples', 0.0),
                                       getattr(c, 'sync_phase_coherence', None))
                       for c in _cm_cands],
                pen=None,
            )
        else:
            _cm_scatter.setData(x=[], y=[])

    # Re-anchor the squared-FFT detection block so the sync-marker overlay
    # below still sees a valid `_hm_rect` and `markers` list — restore the
    # outer-block context that is otherwise scoped to the squared-FFT branch.
    if _detect_win is None or _detect_win.isVisible():
        _hm_rect = QtCore.QRectF(
            0.0, self._detect_freq_min_khz,
            float(self.history_secs), self._detect_freq_span_khz,
        )

        # ── jt9 launch markers ────────────────────────────────────────────────
        # Green   = decoded successfully
        # Red     = launched, jt9 still running
        # Orange  = jt9 finished: no_decode / timeout / error
        # In dual-pol mode circles are placed on H pane (θ < 45°) or V pane (θ ≥ 45°).
        markers = [m for m in all_markers
                   if isinstance(m, dict) and m.get('boundary') == self._ch_snr_boundary]

        # Only rebuild circle paths when marker state changes
        n_decoded    = sum(1 for m in markers if m.get('decoded'))
        theta_sig    = tuple(m.get('theta_deg') for m in markers if m.get('decoded'))
        outcomes_sig = tuple((m.get('outcome', 'launched'), m.get('decoder', '')) for m in markers
                             if m.get('outcome') != 'suppressed')
        marker_fp    = (self._ch_snr_boundary, len(markers), n_decoded, outcomes_sig, theta_sig)
        prev_fp      = getattr(self, '_marker_fp', None)
        if marker_fp != prev_fp:
            self._marker_fp = marker_fp

            r_y   = 2.5   # kHz radius
            px_x, px_y = self.ch_detect_plot.getViewBox().viewPixelSize()
            r_x   = r_y * (px_x / px_y) if px_y > 0 else r_y * 0.1
            circ  = np.linspace(0, 2 * np.pi, 33)

            def _circle_path(mlist):
                xs, ys = [], []
                for m in mlist:
                    xs.append(m.get('t', 0.0)        + r_x * np.cos(circ))
                    ys.append(m.get('freq_khz', 0.0) + r_y * np.sin(circ))
                    xs.append([np.nan])
                    ys.append([np.nan])
                return np.concatenate(xs), np.concatenate(ys)

            def _split_by_pol(mlist):
                """Split markers onto H or V pane.

                Primary: theta_deg >= 45° → V (polarization search result).
                Fallback when theta_deg is None: det_pol == 'v' → V (detecting channel).
                """
                h, v = [], []
                for m in mlist:
                    th = m.get('theta_deg')
                    if _dual and th is not None:
                        (v if th >= 45.0 else h).append(m)
                    elif _dual and m.get('det_pol') == 'v':
                        v.append(m)
                    else:
                        h.append(m)
                return h, v

            spd_dec   = [m for m in markers if m.get('outcome') == 'decoded' and m.get('decoder') == 'spd']
            jt9_dec   = [m for m in markers if m.get('outcome') == 'decoded' and m.get('decoder') != 'spd']
            running   = [m for m in markers if m.get('outcome') == 'launched']
            no_decode = [m for m in markers if m.get('outcome') in ('no_decode', 'timeout', 'error')]

            spd_h,  spd_v  = _split_by_pol(spd_dec)
            jt9_h,  jt9_v  = _split_by_pol(jt9_dec)
            run_h,  run_v  = _split_by_pol(running)
            nod_h,  nod_v  = _split_by_pol(no_decode)

            for curve, mlist in (
                (self.ch_detect_curve_cyan,   spd_h),   # SPD decoded — cyan
                (self.ch_detect_curve_green,  jt9_h),   # jt9 decoded — green
                (self.ch_detect_curve_red,    run_h),   # running — red
                (self.ch_detect_curve_orange, nod_h),   # no decode — orange
            ):
                curve.setData(*_circle_path(mlist), connect='finite') if mlist else curve.setData(x=[], y=[])

            if _dual and _detect_plot_v is not None:
                for curve, mlist in (
                    (self.ch_detect_curve_cyan_v,   spd_v),
                    (self.ch_detect_curve_green_v,  jt9_v),
                    (self.ch_detect_curve_red_v,    run_v),
                    (self.ch_detect_curve_orange_v, nod_v),
                ):
                    curve.setData(*_circle_path(mlist), connect='finite') if mlist else curve.setData(x=[], y=[])

            # Mirror the same circle overlay onto the sync-detector heatmap so
            # the user sees the same decode-progress markers (cyan/green/red/
            # orange) over both detector views.  Markers don't depend on the
            # underlying detector — they reflect the launch/decode lifecycle.
            _sync_plot   = getattr(self, 'sync_detect_plot',   None)
            _sync_plot_v = getattr(self, 'sync_detect_plot_v', None)
            if _sync_plot is not None:
                for curve_name, mlist in (
                    ('sync_detect_curve_cyan',   spd_h),
                    ('sync_detect_curve_green',  jt9_h),
                    ('sync_detect_curve_red',    run_h),
                    ('sync_detect_curve_orange', nod_h),
                ):
                    curve = getattr(self, curve_name, None)
                    if curve is None:
                        continue
                    curve.setData(*_circle_path(mlist), connect='finite') if mlist else curve.setData(x=[], y=[])
                if _dual and _sync_plot_v is not None:
                    for curve_name, mlist in (
                        ('sync_detect_curve_cyan_v',   spd_v),
                        ('sync_detect_curve_green_v',  jt9_v),
                        ('sync_detect_curve_red_v',    run_v),
                        ('sync_detect_curve_orange_v', nod_v),
                    ):
                        curve = getattr(self, curve_name, None)
                        if curve is None:
                            continue
                        curve.setData(*_circle_path(mlist), connect='finite') if mlist else curve.setData(x=[], y=[])

    utc_time = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")
    self.utc_clock_label.setText(f"UTC: {utc_time}")

    if tuned_freq_mhz is not None:
        self.tuned_freq_label.setText(f"{tuned_freq_mhz:.3f} MHz")
        self.setWindowTitle(f'map144 v{__version__}  —  {tuned_freq_mhz:.3f} MHz')
    else:
        self.tuned_freq_label.setText(f"{self.center_freq_mhz:.3f} MHz")
        self.setWindowTitle(f'map144 v{__version__}  —  {self.center_freq_mhz:.3f} MHz')

    if self.spec_staging_filled and len(self.spectrogram_data) > 0:
        # Recompute power stats only once per second (every 10 frames) and
        # on a subsampled slice — full (900, 4096) array dominates _wrapreduction.
        if self._noise_floor_ctr % 10 == 1 or not hasattr(self, '_stat_min'):
            _s = self.spectrogram_data[::4, ::8]   # (225, 512) — representative sample
            self._stat_min  = float(np.min(_s))
            self._stat_max  = float(np.max(_s))
            self._stat_mean = float(np.mean(_s))
        data_min  = self._stat_min
        data_max  = self._stat_max
        data_mean = self._stat_mean

        # ── Update receiver label with flow status ────────────────────────────
        _recv_lbl = getattr(self, '_receiver_label', None)
        if _recv_lbl is not None:
            _src = getattr(self, 'source_mode', 'idle')
            if _src == 'wav':
                _recv_lbl.setText("Simulation")
            else:
                # Build a compact one-line status: source name + packet info + NB% + TX flag
                _src_names = {
                    'radio': 'Flex Radio', 'usrp': 'USRP B210',
                    'airspy': 'Airspy HF+', 'rtlsdr': 'RTL-SDR',
                }
                _base = _src_names.get(_src, _src)

                _pkt_str = ''
                if hasattr(self, 'radio_client') and hasattr(self.radio_client, '_vita') and self.radio_client._vita:
                    vita = self.radio_client._vita
                    import time as _t
                    _now = _t.monotonic()
                    if not hasattr(self, '_vita_window'):
                        self._vita_window = []
                    self._vita_window.append((_now, vita.packet_count, vita.missed_count))
                    self._vita_window = [e for e in self._vita_window if _now - e[0] <= 10.0]
                    if len(self._vita_window) >= 2:
                        _dt_secs   = self._vita_window[-1][0] - self._vita_window[0][0]
                        _dt_pkts   = self._vita_window[-1][1] - self._vita_window[0][1]
                        _dt_missed = self._vita_window[-1][2] - self._vita_window[0][2]
                        pkt_rate   = _dt_pkts / _dt_secs if _dt_secs > 0 else 0.0
                        loss_pct   = (_dt_missed / _dt_pkts * 100) if _dt_pkts > 0 else 0.0
                    else:
                        pkt_rate = 0.0
                        loss_pct = 0.0
                    drops = vita.drop_count
                    _drop_str = f'  drops:{drops}' if drops > 0 else ''
                    _pkt_str = f'  {pkt_rate:.0f} pkt/s  loss:{loss_pct:.1f}%{_drop_str}'

                _nb_total = getattr(self, '_nb_total_count', 0)
                _nb_blank = getattr(self, '_nb_blanked_count', 0)
                _nb_str = f'  NB:{100.0 * _nb_blank / _nb_total:.1f}%' if _nb_total > 0 and _nb_blank > 0 else ''

                _rc = getattr(self, 'radio_client', None)
                _tx_str = '  ** TX **' if (_rc is not None and getattr(_rc, 'transmitting', False)) else ''

                _recv_lbl.setText(
                    _usrp_startup_status(self)
                    or f"{_base}{_pkt_str}{_nb_str}{_tx_str}")
    else:
        _recv_lbl = getattr(self, '_receiver_label', None)
        if _recv_lbl is not None:
            _src = getattr(self, 'source_mode', 'idle')
            if _src == 'wav':
                _recv_lbl.setText("Simulation")
            elif _src != 'idle':
                _src_names = {
                    'radio': 'Flex Radio', 'usrp': 'USRP B210',
                    'airspy': 'Airspy HF+', 'rtlsdr': 'RTL-SDR',
                }
                _recv_lbl.setText(
                    _usrp_startup_status(self)
                    or f"{_src_names.get(_src, _src)}  waiting…")

    # ── Source windows (IQ/NB + per-radio panels) ────────────────────────────
    from .source_windows import update_source_windows
    update_source_windows(self)
