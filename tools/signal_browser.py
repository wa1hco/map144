"""
signal_browser.py — PyQt5 browser for the MSK144 signal database.

Layout:
  QMainWindow
  └── QSplitter (horizontal)
      ├── LEFT:  search sidebar (~300 px)
      └── RIGHT: QSplitter (vertical)
          ├── TOP:    browse tabs (Table / Scatter / Histogram)
          └── BOTTOM: signal detail (info + analysis tabs)
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSortFilterProxyModel)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QDateEdit, QTableWidget, QTableWidgetItem, QTabWidget, QComboBox,
    QGroupBox, QScrollArea, QSizePolicy, QAbstractItemView, QHeaderView,
    QFormLayout, QFrame, QProgressBar, QToolButton, QButtonGroup, QMenu, QAction,
)
from PyQt5.QtGui import QFont, QColor, QBrush
from PyQt5.QtWidgets import QApplication as _QApp


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically (NULLs sort last)."""
    def __lt__(self, other):
        a = self.data(Qt.UserRole)
        b = other.data(Qt.UserRole)
        if a is None and b is None:
            return False
        if a is None:
            return False
        if b is None:
            return True
        return a < b

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from tools.signal_db import (
    open_db, get_signal, search, search_summary,
    column_values, set_tag, set_reviewed,
)
from tools.long_ping_gallery import (
    load_wav, stft_power,
    FC_JT9, FS, SAVE_DIR,
    maidenhead_to_latlon, great_circle_km,
)

DB_PATH    = _ROOT / "scratch" / "signals.db"
CACHE_DIR  = _ROOT / "scratch" / "signal_cache"

# ── Matplotlib canvas helper ──────────────────────────────────────────────────

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _make_canvas(w: float = 6.0, h: float = 3.0) -> FigureCanvas:
    fig = Figure(figsize=(w, h), facecolor="#f8f8f8")
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return canvas


# ── STFT colormap ─────────────────────────────────────────────────────────────

from matplotlib.colors import LinearSegmentedColormap

def _cmap():
    stops = [(0,0,0),(0,0,64),(0,0,128),(0,64,192),(0,128,255),
             (64,192,255),(128,255,255),(255,255,128),(255,255,255)]
    return LinearSegmentedColormap.from_list(
        "map144", [(r/255, g/255, b/255) for r, g, b in stops], N=256)

_CMAP = _cmap()
FRAME_DUR = 720 / 12000.0  # MSK144 frame ~60 ms
PLOT_PAD     = 1.5
SYNC_F_RANGE = 300.0   # Hz either side of audio_hz to sweep
SYNC_F_STEP  =   3.0   # Hz per freq bin in sync surface


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1-D cell-center coordinates to N+1 cell-edge coordinates.

    Supplying explicit edges to pcolormesh avoids matplotlib's monotonicity
    warning when it tries to infer edges from centers.
    """
    c = np.asarray(centers, dtype=np.float64)
    e = np.empty(len(c) + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]    = c[0]  - 0.5 * (c[1]  - c[0])  if len(c) > 1 else c[0] - 0.5
    e[-1]   = c[-1] + 0.5 * (c[-1] - c[-2]) if len(c) > 1 else c[-1] + 0.5
    return e


# ── Worker thread: WAV → analysis arrays ─────────────────────────────────────

class AnalysisWorker(QThread):
    done    = pyqtSignal(dict)
    failed  = pyqtSignal(str)

    def __init__(self, sig: dict):
        super().__init__()
        self._sig = sig

    def run(self):
        sig = self._sig
        wav_path = sig.get("wav_path")
        audio_hz = sig.get("audio_hz") or FC_JT9
        dt_s     = sig.get("dt_s") or 7.5
        cache_p  = sig.get("cache_path")

        # Fast path: pre-computed cache
        if cache_p and Path(cache_p).exists():
            try:
                data = dict(np.load(cache_p))
                data["from_cache"] = True
                self.done.emit(data)
                return
            except Exception:
                pass

        # Compute from WAV
        if not wav_path or not Path(wav_path).exists():
            self.failed.emit("")   # empty = no WAV, not an error
            return
        try:
            audio = load_wav(Path(wav_path))
            times, freqs, S = stft_power(audio)

            # Square real audio: carrier doubles to 2·audio_hz, noise fills 0–6 kHz
            y = (audio ** 2).astype(np.float32)
            _, _, Sq = stft_power(y)

            self.done.emit({
                "S":       S.astype(np.float32),
                "Sq":      Sq.astype(np.float32),
                "times":   times.astype(np.float32),
                "freqs":   freqs.astype(np.float32),
                "from_cache": False,
            })
        except Exception as e:
            self.failed.emit(str(e))


# ── Detail: spectrogram tab ───────────────────────────────────────────────────

class SpectrogramTab(QWidget):
    def __init__(self, label: str, use_squared: bool = False):
        super().__init__()
        self._use_sq = use_squared
        # Fixed frequency display range: audio 0-3000 Hz, squared 0-6000 Hz
        self._f_lo = 0.0
        self._f_hi = 6000.0 if use_squared else 3000.0
        self._canvas = _make_canvas(8, 3)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def show_data(self, data: dict, sig: dict) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        S     = data["Sq" if self._use_sq else "S"]
        times = data["times"]
        freqs = data["freqs"]

        audio_hz    = sig.get("audio_hz") or FC_JT9
        dt_s        = sig.get("dt_s") or 7.5
        navg        = sig.get("navg") or 1
        bb_dur_s    = (sig.get("bb_longest_ms") or 0) / 1000.0
        ping_dur_s  = max(bb_dur_s, max(navg, 1) * FRAME_DUR)

        # Show full ping duration plus noise pad on each side
        t_lo = max(0.0,        dt_s - PLOT_PAD)
        t_hi = min(times[-1],  dt_s + ping_dur_s + PLOT_PAD)
        f_lo = max(0.0,        self._f_lo)
        f_hi = min(freqs[-1],  self._f_hi)

        t_mask = (times >= t_lo) & (times <= t_hi)
        f_mask = (freqs >= f_lo) & (freqs <= f_hi)
        t_crop = times[t_mask]
        f_crop = freqs[f_mask]

        # S shape: stft_power returns (n_times, n_freqs)
        # pcolormesh(t, f, Z) expects Z.shape == (len(f), len(t))
        # Convert to dB; use median-anchored range matching long_ping_gallery
        S_crop = 10.0 * np.log10(S[np.ix_(t_mask, f_mask)].T + 1e-12)
        med  = float(np.median(S_crop))
        vmin = med - 8.0    # NOISE_HEADROOM
        vmax = med + 22.0   # NOISE_RANGE
        ax.pcolormesh(t_crop, f_crop, S_crop, cmap=_CMAP,
                      vmin=vmin, vmax=vmax, shading="auto", rasterized=True)

        ax.axvline(dt_s, color="white", lw=0.8, ls=":", alpha=0.8)
        ref_hz = 2.0 * audio_hz if self._use_sq else audio_hz
        ax.axhline(ref_hz, color="lime", lw=0.6, ls="--", alpha=0.6)
        for fi in range(max(navg or 1, 1)):
            ft = dt_s + fi * FRAME_DUR
            ax.axvspan(ft, ft + FRAME_DUR, alpha=0.08, color="cyan")


        msg = sig.get("message", "")
        grid = sig.get("grid", "")
        dist = sig.get("dist_km")
        dist_s = f"  {dist:.0f} km" if dist else ""
        snr  = sig.get("snr_db")
        snr_s = f"  SNR {int(snr):+d} dB" if snr is not None else ""
        navg_s = f"  navg={navg}" if navg else ""
        ax.set_title(f"{msg}  {grid}{dist_s}{snr_s}{navg_s}", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
        fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def clear(self):
        self._canvas.figure.clear()
        self._canvas.draw()


# ── Integration sweep worker & tab ────────────────────────────────────────────

class IntegrationWorker(QThread):
    done   = pyqtSignal(dict)
    failed = pyqtSignal(str)
    MAX_NAVG = 7

    def __init__(self, sig: dict):
        super().__init__()
        self._sig = sig

    def run(self):
        from map144_app.msk144_spd import (
            _freq_search_avg, _sync_phase_features,
            NSPM, FS as SPD_FS, SYNC_THRESHOLD,
        )
        from tools.long_ping_gallery import to_baseband as _to_bb
        sig      = self._sig
        wav_path = sig.get("wav_path")
        audio_hz = sig.get("audio_hz") or FC_JT9
        dt_s     = sig.get("dt_s") or 7.5

        if not wav_path or not Path(wav_path).exists():
            self.failed.emit("")
            return
        try:
            audio = load_wav(Path(wav_path))
            bb    = _to_bb(audio, audio_hz)

            # Window centered on dt_s with enough headroom for 7 frames
            i_start = max(0, int(dt_s * SPD_FS) - NSPM // 2)
            cbase   = np.ascontiguousarray(
                bb[i_start: i_start + (self.MAX_NAVG + 1) * NSPM],
                dtype=np.complex64,
            )

            ns, xmax_ns, cohs, ferrs = [], [], [], []
            for n in range(1, self.MAX_NAVG + 1):
                if n * NSPM > len(cbase):
                    break
                c_avg, xmax_n, ferr, xcc, ish, _ = _freq_search_avg(
                    cbase[: n * NSPM], [1] * n
                )
                if c_avg is None or xcc is None:
                    ns.append(n); xmax_ns.append(0.0)
                    cohs.append(0.0); ferrs.append(0.0)
                    continue
                coh, _, _ = _sync_phase_features(c_avg, ish)
                ns.append(n)
                xmax_ns.append(float(xmax_n))
                cohs.append(float(coh))
                ferrs.append(float(ferr))

            self.done.emit({
                "ns":           ns,
                "xmax_ns":      xmax_ns,
                "cohs":         cohs,
                "ferrs":        ferrs,
                "threshold":    float(SYNC_THRESHOLD),
                "stored_navg":  sig.get("navg"),
            })
        except Exception as e:
            self.failed.emit(str(e))


class IntegrationTab(QWidget):
    def __init__(self):
        super().__init__()
        self._canvas = _make_canvas(7, 4)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def show_data(self, data: dict, sig: dict) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax_xm, ax_coh = fig.subplots(2, 1, sharex=True)

        ns          = data["ns"]
        xmax_ns     = data["xmax_ns"]
        cohs        = data["cohs"]
        threshold   = data["threshold"]
        stored_navg = data.get("stored_navg")
        ns_arr      = np.array(ns, dtype=float)

        # ── Top: xmax_norm vs N ──────────────────────────────────────────────
        ax_xm.plot(ns, xmax_ns, "o-", color="steelblue", label="measured xmax_norm")

        # Theoretical sqrt(N) growth anchored at N=1
        if xmax_ns:
            theory = [xmax_ns[0] * np.sqrt(n) for n in ns]
            ax_xm.plot(ns, theory, "--", color="orange", alpha=0.7,
                       label=f"xmax(1)·√N  ({xmax_ns[0]:.2f}·√N)")

        ax_xm.axhline(threshold, color="red", lw=1.0, ls="--", alpha=0.8,
                      label=f"decode threshold ({threshold})")
        if stored_navg:
            ax_xm.axvline(stored_navg, color="green", lw=1.2, ls=":",
                          alpha=0.9, label=f"stored navg={stored_navg}")

        ax_xm.set_ylabel("xmax_norm")
        ax_xm.set_ylim(bottom=0)
        ax_xm.legend(fontsize=7, loc="upper left")
        ax_xm.grid(True, alpha=0.3)

        snr  = sig.get("snr_db")
        navg = sig.get("navg")
        msg  = sig.get("message", "")
        snr_s  = f"  SNR {int(snr):+d} dB" if snr is not None else ""
        navg_s = f"  navg={navg}" if navg else ""
        ax_xm.set_title(f"{msg}{snr_s}{navg_s}", fontsize=9)

        # ── Bottom: phase coherence vs N ─────────────────────────────────────
        ax_coh.plot(ns, cohs, "s-", color="mediumpurple", label="phase coherence")
        ax_coh.axhline(0.7, color="gray", lw=0.8, ls="--", alpha=0.6,
                       label="coherence 0.7")
        if stored_navg:
            ax_coh.axvline(stored_navg, color="green", lw=1.2, ls=":", alpha=0.9)

        ax_coh.set_ylabel("Coherence")
        ax_coh.set_xlabel("Frames integrated (N)")
        ax_coh.set_ylim(0, 1.05)
        ax_coh.set_xticks(ns)
        ax_coh.legend(fontsize=7, loc="upper left")
        ax_coh.grid(True, alpha=0.3)

        fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def clear(self):
        self._canvas.figure.clear()
        self._canvas.draw()


# ── WSJT-X sq-det worker ─────────────────────────────────────────────────────

class _SqDetWorker(QThread):
    """Compute WSJT-X squared-spectrum + normalised metric using the exact
    sq_det.py algorithm: complex analytic squaring, NFFT=864, rectangular
    window, 18 ms stride, p25 noise normalisation.  One batched FFT pass."""
    done   = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, sig: dict):
        super().__init__()
        self._sig = sig

    def run(self):
        from scipy.signal import hilbert as _hilbert
        from map144_app.sq_det import (
            NSPM, FS, DF, PCT_FOR_XMED, DETECT_THRESHOLD_NORM_LIN,
            DETECT_THRESHOLD_DETMET2_LIN,
        )
        STRIDE = NSPM // 4          # 216 samples = 18 ms  (WSJT-X quarter-frame)
        NTOL   = 100.0              # Hz — default sq_det tolerance

        sig      = self._sig
        wav_path = sig.get("wav_path")
        audio_hz = float(sig.get("audio_hz") or FC_JT9)

        if not wav_path or not Path(wav_path).exists():
            self.failed.emit("")
            return
        try:
            audio = load_wav(Path(wav_path))
            cdat  = _hilbert(audio.astype(np.float64)).astype(np.complex128)

            # ── squared spectrum, all frames in one batched FFT ───────────────
            n       = len(cdat)
            n_steps = max(1, (n - NSPM) // STRIDE)
            idx     = np.arange(NSPM)[None, :] + np.arange(n_steps)[:, None] * STRIDE
            ctmp    = cdat[idx] ** 2                        # (n_steps, NSPM) complex
            spec    = np.fft.fft(ctmp, axis=1)             # full FFT — signal is complex
            tonespec = (np.abs(spec[:, :NSPM // 2 + 1]) ** 2).astype(np.float32)

            freqs_full = np.fft.fftfreq(NSPM, 1.0 / FS)
            fq         = freqs_full[:NSPM // 2 + 1].astype(np.float32)  # 0 … ~6 kHz
            t_centers  = ((np.arange(n_steps) * STRIDE + NSPM / 2.0) / FS).astype(np.float32)

            # ── detection metrics (matches sq_det.py exactly) ─────────────────
            nfhi = 2.0 * (audio_hz + 500.0)
            nflo = 2.0 * (audio_hz - 500.0)
            i_hi = int(np.argmin(np.abs(freqs_full - nfhi)))
            i_lo = int(np.argmin(np.abs(freqs_full - nflo)))
            bw   = int(round(2.0 * NTOL / DF))

            # Use tonespec (positive half) for display but full FFT for metric
            # bins — for fc=1500 both tones are at positive freq so tonespec suffices
            tspec_full = (np.abs(spec) ** 2).astype(np.float64)
            hi_band = tspec_full[:, max(0, i_hi - bw): i_hi + bw + 1]
            lo_band = tspec_full[:, max(0, i_lo - bw): i_lo + bw + 1]
            ah = hi_band.max(axis=1)
            al = lo_band.max(axis=1)
            detmet = np.maximum(ah, al)

            n_hi  = max(1, hi_band.shape[1] - 1)
            n_lo  = max(1, lo_band.shape[1] - 1)
            ahavp = (hi_band.sum(axis=1) - ah) / n_hi
            alavp = (lo_band.sum(axis=1) - al) / n_lo
            detmet2 = np.maximum(
                ah / np.maximum(ahavp, 1e-30),
                al / np.maximum(alavp, 1e-30),
            )

            xmed       = float(np.percentile(detmet, PCT_FOR_XMED))
            detmet_norm = detmet / max(xmed, 1e-30)
            triggers    = (detmet_norm >= DETECT_THRESHOLD_NORM_LIN) | \
                          (detmet2     >= DETECT_THRESHOLD_DETMET2_LIN)

            self.done.emit({
                "tonespec":    tonespec,          # (n_steps, NSPM//2+1) float32
                "tq":          t_centers,
                "fq":          fq,
                "detmet_norm": detmet_norm.astype(np.float32),
                "detmet2":     detmet2.astype(np.float32),
                "t_centers":   t_centers,
                "xmed":        xmed,
                "triggers":    triggers,
            })
        except Exception as e:
            self.failed.emit(str(e))


# ── WSJT-X sq-det display tab ─────────────────────────────────────────────────

class SqDetTab(QWidget):
    """Squared-spectrum tab using WSJT-X exact algorithm.

    Top subplot  — 2-D tonespec (time × freq, power dB)
    Bottom subplot — normalised detection metric vs time with threshold lines
    """

    _DB_RANGE = 35.0    # dB above noise floor shown by default

    def __init__(self):
        super().__init__()
        self._canvas = _make_canvas(8, 5)
        self._data   = None
        self._sig    = None

        # Gain trim spinbox (shifts color window without changing reference)
        self._gain_spin = QSpinBox()
        self._gain_spin.setRange(-30, 30)
        self._gain_spin.setValue(0)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setToolTip("Shift display window relative to xmed noise floor")
        self._gain_spin.setMaximumWidth(90)
        self._gain_spin.valueChanged.connect(self._rerender)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 2, 4, 0)
        ctrl.addWidget(QLabel("Gain:"))
        ctrl.addWidget(self._gain_spin)
        ctrl.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(ctrl)
        lay.addWidget(self._canvas)

    def _rerender(self) -> None:
        if self._data and self._sig:
            self._render(self._data, self._sig)

    def show_data(self, data: dict, sig: dict) -> None:
        self._data = data
        self._sig  = sig
        self._render(data, sig)

    def _render(self, data: dict, sig: dict) -> None:
        from map144_app.sq_det import DETECT_THRESHOLD_NORM_LIN, DETECT_THRESHOLD_DETMET2_LIN

        fig = self._canvas.figure
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        tonespec    = data["tonespec"]     # (n_steps, n_freq) float32
        tq          = data["tq"]
        fq          = data["fq"]
        detmet_norm = data["detmet_norm"]
        detmet2     = data["detmet2"]
        t_centers   = data["t_centers"]
        xmed        = float(data.get("xmed", 1e-30))

        audio_hz = float(sig.get("audio_hz") or FC_JT9)
        dt_s     = sig.get("dt_s") or 7.5
        navg     = sig.get("navg") or 1
        bb_dur_s = (sig.get("bb_longest_ms") or 0) / 1000.0
        ping_dur = max(bb_dur_s, max(navg, 1) * FRAME_DUR)
        t_lo = max(float(tq[0]),  dt_s - PLOT_PAD)
        t_hi = min(float(tq[-1]), dt_s + ping_dur + PLOT_PAD)

        t_mask = (tq >= t_lo) & (tq <= t_hi)

        # ── top: 2-D squared spectrogram ──────────────────────────────────────
        # Limit display to the tone band ± margin (avoids DC bin and out-of-band
        # artefacts that would skew any global statistic).
        f_lo_disp = max(float(fq[1]), 2.0 * audio_hz - 2000.0)
        f_hi_disp = 2.0 * audio_hz + 2000.0
        f_mask = (fq >= f_lo_disp) & (fq <= f_hi_disp)

        Sq_db = 10.0 * np.log10(tonespec[t_mask, :].T + 1e-30)
        Sq_crop = Sq_db[f_mask, :]

        # Median of the cropped region ≈ noise floor (signal is <3% of pixels).
        # Same formula as SpectrogramTab: med-8 … med+22, 30 dB window.
        gain_db = float(self._gain_spin.value())
        med  = float(np.median(Sq_crop))
        vmin = med + gain_db - 8.0
        vmax = med + gain_db + 38.0   # wider bright side: squaring doubles SNR in dB
        ax1.pcolormesh(_bin_edges(tq[t_mask]), _bin_edges(fq[f_mask]), Sq_crop,
                       vmin=vmin, vmax=vmax,
                       cmap=_CMAP, shading="flat", rasterized=True)

        # Tone position markers: draw only in the noise flanks so the ping ridge
        # is not obscured.  Left flank: t_lo … dt_s-pad; right: dt_s+ping+pad … t_hi
        f_lo_tone = 2.0 * audio_hz - 1000.0
        f_hi_tone = 2.0 * audio_hz + 1000.0
        _MARKER_PAD = 0.10   # s gap between marker end and ping edge
        _flank_lo_end  = dt_s - _MARKER_PAD
        _flank_hi_start = dt_s + ping_dur + _MARKER_PAD
        for _f in (f_lo_tone, f_hi_tone):
            if t_lo < _flank_lo_end:
                ax1.hlines(_f, t_lo, min(_flank_lo_end, t_hi),
                           colors="lime", lw=0.9, ls="--", alpha=0.7)
            if _flank_hi_start < t_hi:
                ax1.hlines(_f, max(_flank_hi_start, t_lo), t_hi,
                           colors="lime", lw=0.9, ls="--", alpha=0.7)
        ax1.axvline(dt_s, color="white", lw=0.8, ls=":", alpha=0.8)
        for fi in range(max(navg, 1)):
            ax1.axvspan(dt_s + fi * FRAME_DUR, dt_s + (fi + 1) * FRAME_DUR,
                        alpha=0.07, color="cyan")
        ax1.set_ylim(f_lo_disp, f_hi_disp)
        ax1.set_ylabel("Freq (Hz)")
        ax1.set_title(f"WSJT-X sq det  NFFT=864 rect  18 ms stride", fontsize=8)

        # ── bottom: normalised metric trace ───────────────────────────────────
        tm_mask  = (t_centers >= t_lo) & (t_centers <= t_hi)
        triggers = data["triggers"]

        ax2.plot(t_centers[tm_mask], detmet_norm[tm_mask],
                 color="cyan", lw=0.9, label="detmet/xmed")
        # detmet2 scaled so its threshold (12) maps to the same 3.0 line
        _scale = DETECT_THRESHOLD_NORM_LIN / DETECT_THRESHOLD_DETMET2_LIN
        ax2.plot(t_centers[tm_mask], detmet2[tm_mask] * _scale,
                 color="orange", lw=0.7, alpha=0.7, label=f"detmet2/4")
        ax2.axhline(DETECT_THRESHOLD_NORM_LIN, color="goldenrod", lw=1.0,
                    ls="--", label=f"threshold {DETECT_THRESHOLD_NORM_LIN:.0f}")
        ax2.axvline(dt_s, color="white", lw=0.8, ls=":", alpha=0.8)

        # Onset: first trigger frame in display window — same logic as WSJT-X
        # (triggers = primary | fallback, no debounce)
        _in = (t_centers >= t_lo) & (t_centers <= t_hi)
        _oi = np.where(_in & triggers)[0]
        t_onset = float(t_centers[_oi[0]]) if len(_oi) else None
        if t_onset is not None:
            ax1.axvline(t_onset, color="yellow", lw=1.2, alpha=0.9)
            ax2.axvline(t_onset, color="red", lw=1.2, alpha=0.9,
                        label=f"onset {t_onset:.2f}s")

        peak_norm = float(detmet_norm[tm_mask].max()) if tm_mask.any() else 5.0
        ax2.set_ylim(0, min(max(peak_norm * 1.1, 4.0), 40.0))
        ax2.set_ylabel("detmet/xmed")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=7, loc="upper left")
        ax2.set_xlim(t_lo, t_hi)

        fig.tight_layout(pad=0.4)
        self._canvas.draw_idle()

    def clear(self):
        self._canvas.figure.clear()
        self._canvas.draw_idle()


# ── Sync surface worker ───────────────────────────────────────────────────────

class SyncWorker(QThread):
    done   = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, sig: dict):
        super().__init__()
        self._sig = sig

    def run(self):
        from map144_app.msk144_spd import (
            _sync_correlate, _sync_phase_features, NSPM, FS as SPD_FS,
        )
        sig      = self._sig
        wav_path = sig.get("wav_path")
        audio_hz = sig.get("audio_hz") or FC_JT9

        if not wav_path or not Path(wav_path).exists():
            self.failed.emit("")
            return
        try:
            audio  = load_wav(Path(wav_path))

            # Build baseband at audio_hz once, then shift per freq step
            n      = len(audio)
            t_vec  = np.arange(n, dtype=np.float64) / SPD_FS
            analytic = np.fft.ifft(
                np.where(np.fft.fftfreq(n) >= 0,
                         2 * np.fft.fft(audio.astype(np.float64)), 0)
            ).astype(np.complex64)
            carrier_phasor = np.exp(-2j * np.pi * audio_hz * t_vec).astype(np.complex64)
            bb_ref = (analytic * carrier_phasor).astype(np.complex64)
            rms = float(np.sqrt(np.mean(np.abs(bb_ref.astype(np.complex128)) ** 2)))
            if rms > 1e-12:
                bb_ref = (bb_ref * (np.sqrt(2.0) / rms)).astype(np.complex64)

            freq_offsets = np.arange(-SYNC_F_RANGE, SYNC_F_RANGE + SYNC_F_STEP,
                                     SYNC_F_STEP, dtype=np.float32)
            n_frames = n // NSPM
            times    = (np.arange(n_frames) * NSPM + NSPM / 2.0) / SPD_FS

            xmax_surf = np.zeros((n_frames, len(freq_offsets)), dtype=np.float32)
            coh_surf  = np.zeros((n_frames, len(freq_offsets)), dtype=np.float32)

            NORM = 1.0 / 48.0   # match SPD navg=1 normalisation

            for fi, df in enumerate(freq_offsets):
                shift = np.exp(-2j * np.pi * float(df) * t_vec).astype(np.complex64)
                bb = bb_ref * shift
                for ti in range(n_frames):
                    frame = np.ascontiguousarray(bb[ti * NSPM:(ti + 1) * NSPM])
                    if len(frame) < NSPM:
                        break
                    xcc, xmax, ish = _sync_correlate(frame)
                    xmax_surf[ti, fi] = xmax * NORM
                    coh, _, _ = _sync_phase_features(frame, ish)
                    coh_surf[ti, fi]  = coh

            self.done.emit({
                "times":        times.astype(np.float32),
                "freq_offsets": freq_offsets,
                "audio_hz":     float(audio_hz),
                "xmax_surf":    xmax_surf,
                "coh_surf":     coh_surf,
            })
        except Exception as e:
            self.failed.emit(str(e))


# ── Sync surface display tab ──────────────────────────────────────────────────

class SyncSurfaceTab(QWidget):
    def __init__(self):
        super().__init__()
        self._canvas = _make_canvas(8, 4)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def show_data(self, data: dict, sig: dict) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax_corr, ax_coh = fig.subplots(2, 1, sharex=True)

        times      = data["times"]
        f_offsets  = data["freq_offsets"]
        audio_hz   = data["audio_hz"]
        xmax_surf  = data["xmax_surf"]   # (n_frames, n_freqs)
        coh_surf   = data["coh_surf"]

        freqs = audio_hz + f_offsets      # absolute Hz

        dt_s     = sig.get("dt_s") or 7.5
        bb_dur_s = (sig.get("bb_longest_ms") or 0) / 1000.0
        navg     = sig.get("navg") or 1
        ping_dur = max(bb_dur_s, navg * FRAME_DUR)
        t_lo = max(times[0],  dt_s - PLOT_PAD)
        t_hi = min(times[-1], dt_s + ping_dur + PLOT_PAD)
        t_mask = (times >= t_lo) & (times <= t_hi)
        t_crop = times[t_mask]

        # pcolormesh expects Z.shape == (len(y), len(x))
        X = xmax_surf[t_mask].T    # (n_freqs, n_t_crop)
        C = coh_surf[t_mask].T

        ax_corr.pcolormesh(t_crop, freqs, X,
                           vmin=0.0, vmax=2.0, cmap="inferno",
                           shading="auto", rasterized=True)
        ax_corr.axhline(audio_hz, color="cyan", lw=0.7, ls="--", alpha=0.7)
        ax_corr.axvline(dt_s,     color="white", lw=0.8, ls=":", alpha=0.8)
        ax_corr.axhline(audio_hz, color="cyan", lw=0.7, ls="--", alpha=0.7)
        # threshold line (SPD decode threshold ≈ 1.3 in normalised units)
        ax_corr.set_ylabel("Carrier Hz")
        ax_corr.set_title("Sync correlation (norm)", fontsize=9)

        ax_coh.pcolormesh(t_crop, freqs, C,
                          vmin=0.0, vmax=1.0, cmap="viridis",
                          shading="auto", rasterized=True)
        ax_coh.axhline(audio_hz, color="cyan", lw=0.7, ls="--", alpha=0.7)
        ax_coh.axvline(dt_s,     color="white", lw=0.8, ls=":", alpha=0.8)
        ax_coh.set_ylabel("Carrier Hz")
        ax_coh.set_xlabel("Time (s)")
        ax_coh.set_title("Phase coherence [0–1]", fontsize=9)

        fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def clear(self):
        self._canvas.figure.clear()
        self._canvas.draw()


# ── Signal detail panel ───────────────────────────────────────────────────────

class SignalDetail(QWidget):
    tag_changed = pyqtSignal()

    def __init__(self, conn):
        super().__init__()
        self._conn        = conn
        self._sig         = None
        self._worker      = None
        self._sq_worker   = None
        self._sq_sig_id   = None
        self._sync_worker = None
        self._sync_sig_id = None
        self._intg_worker = None
        self._intg_sig_id = None

        # Info bar
        self._info = QLabel("— no signal selected —")
        self._info.setWordWrap(True)
        font = QFont()
        font.setPointSize(10)
        self._info.setFont(font)

        # Tag / reviewed row
        tag_row = QHBoxLayout()
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("tag to add …")
        self._tag_edit.setMaximumWidth(160)
        btn_add_tag = QPushButton("Add tag")
        btn_add_tag.clicked.connect(self._add_tag)
        btn_rm_tag  = QPushButton("Remove tag")
        btn_rm_tag.clicked.connect(self._rm_tag)
        self._reviewed_btn = QPushButton("Mark reviewed")
        self._reviewed_btn.setCheckable(True)
        self._reviewed_btn.toggled.connect(self._toggle_reviewed)
        tag_row.addWidget(QLabel("Tag:"))
        tag_row.addWidget(self._tag_edit)
        tag_row.addWidget(btn_add_tag)
        tag_row.addWidget(btn_rm_tag)
        tag_row.addStretch()
        tag_row.addWidget(self._reviewed_btn)

        # Analysis tabs
        self._spec_tab  = SpectrogramTab("Spectrogram",   use_squared=False)
        self._sq_tab    = SqDetTab()
        self._sync_tab  = SyncSurfaceTab()
        self._intg_tab  = IntegrationTab()

        self._tabs = QTabWidget()
        self._tabs.addTab(self._spec_tab, "Spectrogram")
        self._tabs.addTab(self._sq_tab,   "Sq Det")
        self._tabs.addTab(self._sync_tab, "Sync")
        self._tabs.addTab(self._intg_tab, "Integration")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Progress indicator
        self._progress = QLabel("")

        # Nav bar
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._pos_label = QLabel("—")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._pos_label)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch()

        # Layout
        lay = QVBoxLayout(self)
        lay.addWidget(self._info)
        lay.addLayout(tag_row)
        lay.addWidget(self._tabs)
        lay.addWidget(self._progress)
        lay.addLayout(nav_row)

        # Navigation state (set by SignalBrowser)
        self._ids: list[int] = []
        self._cur_idx: int   = -1

        # Debounce: don't start WAV load until user stops on a row for 300 ms
        self._pending_id: int | None = None
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(300)
        self._load_timer.timeout.connect(self._do_load)

    def set_result_ids(self, ids: list[int]) -> None:
        self._ids     = ids
        self._cur_idx = -1
        self._update_nav()

    # Called from browse views — updates info bar immediately, defers WAV load
    def load_signal(self, signal_id: int) -> None:
        sig = get_signal(self._conn, signal_id)
        if sig is None:
            return
        self._sig = sig
        self._pending_id = signal_id

        if signal_id in self._ids:
            self._cur_idx = self._ids.index(signal_id)
        self._update_nav()

        # Info bar updates immediately (no I/O cost)
        self._update_info_bar(sig)

        # Restart debounce timer — WAV load fires 300 ms after user stops
        self._load_timer.start()

    def _update_info_bar(self, sig: dict) -> None:
        msg   = sig.get("message", "")
        grid  = sig.get("grid", "")
        dist  = sig.get("dist_km")
        snr   = sig.get("snr_db")
        navg  = sig.get("navg")
        tags  = sig.get("tags", "")
        dist_s = f"  {dist:.0f} km" if dist else ""
        snr_s  = f"  SNR {int(snr):+d} dB" if snr is not None else ""
        navg_s = f"  navg={navg}" if navg else ""
        tags_s = f"  [{tags}]" if tags else ""
        rev_s  = "  ✓reviewed" if sig.get("reviewed") else ""
        self._info.setText(
            f"{msg}  {grid}{dist_s}{snr_s}{navg_s}{tags_s}{rev_s}"
        )
        self._reviewed_btn.blockSignals(True)
        self._reviewed_btn.setChecked(bool(sig.get("reviewed")))
        self._reviewed_btn.blockSignals(False)

    def _do_load(self) -> None:
        """Called by debounce timer — starts WAV analysis worker."""
        if self._sig is None or self._sig["id"] != self._pending_id:
            return
        self._clear_plots()
        self._progress.setText("Loading …")
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            # No wait — let it die asynchronously
        self._worker = AnalysisWorker(dict(self._sig))
        self._worker.done.connect(self._on_data_ready)
        self._worker.failed.connect(self._on_data_failed)
        self._worker.start()

    def _on_data_ready(self, data: dict) -> None:
        if self._sig is None:
            return
        src = "cache" if data.get("from_cache") else "computed"
        self._progress.setText(f"Loaded ({src})")
        self._spec_tab.show_data(data, dict(self._sig))
        # Kick off lazy tabs if currently visible
        cur = self._tabs.currentWidget()
        if cur is self._sq_tab:
            self._start_sq_worker()
        elif cur is self._sync_tab:
            self._start_sync_worker()
        elif cur is self._intg_tab:
            self._start_intg_worker()

    def _on_data_failed(self, msg: str) -> None:
        self._progress.setText(msg if msg else "No WAV file — metadata only")

    def _on_tab_changed(self, idx: int) -> None:
        w = self._tabs.widget(idx)
        if w is self._sq_tab:
            self._start_sq_worker()
        elif w is self._sync_tab:
            self._start_sync_worker()
        elif w is self._intg_tab:
            self._start_intg_worker()

    def _start_sq_worker(self) -> None:
        if self._sig is None:
            return
        sig_id = self._sig["id"]
        if sig_id == self._sq_sig_id:
            return
        if not self._sig.get("wav_path"):
            return
        self._sq_sig_id = sig_id
        self._progress.setText("Computing sq det …")
        if self._sq_worker and self._sq_worker.isRunning():
            self._sq_worker.terminate()
        self._sq_worker = _SqDetWorker(dict(self._sig))
        self._sq_worker.done.connect(self._on_sq_ready)
        self._sq_worker.failed.connect(lambda _: self._progress.setText("Sq det failed"))
        self._sq_worker.start()

    def _on_sq_ready(self, data: dict) -> None:
        if self._sig is None:
            return
        self._progress.setText("Sq det ready")
        self._sq_tab.show_data(data, dict(self._sig))

    def _start_sync_worker(self) -> None:
        if self._sig is None:
            return
        sig_id = self._sig["id"]
        if sig_id == self._sync_sig_id:
            return   # already computed for this signal
        if not self._sig.get("wav_path"):
            return
        self._sync_sig_id = sig_id
        self._progress.setText("Computing sync surface …")
        if self._sync_worker and self._sync_worker.isRunning():
            self._sync_worker.terminate()
        self._sync_worker = SyncWorker(dict(self._sig))
        self._sync_worker.done.connect(self._on_sync_ready)
        self._sync_worker.failed.connect(lambda _: self._progress.setText("Sync failed"))
        self._sync_worker.start()

    def _on_sync_ready(self, data: dict) -> None:
        if self._sig is None:
            return
        self._progress.setText("Sync surface ready")
        self._sync_tab.show_data(data, dict(self._sig))

    def _start_intg_worker(self) -> None:
        if self._sig is None:
            return
        sig_id = self._sig["id"]
        if sig_id == self._intg_sig_id:
            return
        if not self._sig.get("wav_path"):
            return
        self._intg_sig_id = sig_id
        self._progress.setText("Computing integration sweep …")
        if self._intg_worker and self._intg_worker.isRunning():
            self._intg_worker.terminate()
        self._intg_worker = IntegrationWorker(dict(self._sig))
        self._intg_worker.done.connect(self._on_intg_ready)
        self._intg_worker.failed.connect(lambda _: self._progress.setText("Integration failed"))
        self._intg_worker.start()

    def _on_intg_ready(self, data: dict) -> None:
        if self._sig is None:
            return
        self._progress.setText("Integration sweep ready")
        self._intg_tab.show_data(data, dict(self._sig))

    def _clear_plots(self) -> None:
        self._spec_tab.clear()
        self._sq_tab.clear()
        self._sync_tab.clear()
        self._intg_tab.clear()
        self._sq_sig_id   = None
        self._sync_sig_id = None
        self._intg_sig_id = None

    def _add_tag(self) -> None:
        if self._sig is None:
            return
        tag = self._tag_edit.text().strip()
        if tag:
            set_tag(self._conn, self._sig["id"], tag, add=True)
            self._tag_edit.clear()
            self.tag_changed.emit()
            self.load_signal(self._sig["id"])

    def _rm_tag(self) -> None:
        if self._sig is None:
            return
        tag = self._tag_edit.text().strip()
        if tag:
            set_tag(self._conn, self._sig["id"], tag, add=False)
            self._tag_edit.clear()
            self.tag_changed.emit()
            self.load_signal(self._sig["id"])

    def _toggle_reviewed(self, checked: bool) -> None:
        if self._sig is None:
            return
        set_reviewed(self._conn, self._sig["id"], checked)
        self.tag_changed.emit()

    def _go_prev(self) -> None:
        if self._cur_idx > 0:
            self._cur_idx -= 1
            self.load_signal(self._ids[self._cur_idx])

    def _go_next(self) -> None:
        if self._cur_idx < len(self._ids) - 1:
            self._cur_idx += 1
            self.load_signal(self._ids[self._cur_idx])

    def _update_nav(self) -> None:
        n = len(self._ids)
        pos = self._cur_idx + 1 if self._cur_idx >= 0 else 0
        self._pos_label.setText(f"{pos} / {n}")
        self._prev_btn.setEnabled(self._cur_idx > 0)
        self._next_btn.setEnabled(self._cur_idx < n - 1)


# ── Browse: Table view ────────────────────────────────────────────────────────

# (key, header, width_px)
_TABLE_COLS = [
    ("timestamp",      "Timestamp",   150),
    ("snr_db",         "SNR",          42),
    ("navg",           "navg",         42),
    ("dist_km",        "Dist km",      60),
    ("bb_longest_ms",  "Dur ms",       60),
    ("drift_hz_per_s", "Drift Hz/s",   70),
    ("audio_hz",       "Audio Hz",     65),
    ("source",         "Source",       80),
    ("tags",           "Tags",        110),
    ("message",        "Message",     220),
]

class TableView(QWidget):
    signal_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._ids: list[int] = []
        self._tbl = QTableWidget()
        self._tbl.setColumnCount(len(_TABLE_COLS))
        self._tbl.setHorizontalHeaderLabels([c[1] for c in _TABLE_COLS])
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl.setWordWrap(False)
        vh = self._tbl.verticalHeader()
        vh.setDefaultSectionSize(22)
        vh.setSectionResizeMode(QHeaderView.Fixed)  # no drag-to-resize rows
        vh.setVisible(False)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._tbl.horizontalHeader().sectionClicked.connect(self._on_header_click)
        self._tbl.itemSelectionChanged.connect(self._on_select)
        self._tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tbl.customContextMenuRequested.connect(self._on_context_menu)
        for i, (_, _, w) in enumerate(_TABLE_COLS):
            self._tbl.setColumnWidth(i, w)
        self._sort_col   = 0
        self._sort_asc   = True

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._tbl)

    def populate(self, rows: list[dict]) -> None:
        self._ids = [r["id"] for r in rows]
        tbl = self._tbl
        _NUMERIC = {"snr_db", "navg", "dist_km", "bb_longest_ms",
                    "drift_hz_per_s", "audio_hz"}
        tbl.setUpdatesEnabled(False)
        tbl.setSortingEnabled(False)
        tbl.setRowCount(len(rows))
        for row_i, r in enumerate(rows):
            sig_id = r["id"]
            has_wav = bool(r.get("wav_path"))
            for col_i, (key, _, _) in enumerate(_TABLE_COLS):
                val = r.get(key)
                if val is None:
                    text = ""
                elif key in ("snr_db", "navg"):
                    text = str(int(val))
                elif key == "dist_km":
                    text = f"{val:.0f}"
                elif key == "bb_longest_ms":
                    text = f"{val:.0f}" if val else ""
                elif key == "drift_hz_per_s":
                    text = f"{val:.2f}" if val else ""
                elif key == "audio_hz":
                    text = f"{val:.0f}"
                else:
                    text = str(val)
                if key in _NUMERIC:
                    item = _NumericItem(text)
                    item.setData(Qt.UserRole, float(val) if val is not None else None)
                else:
                    item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Store signal_id in first column so sort doesn't break lookup
                if col_i == 0:
                    item.setData(Qt.UserRole, sig_id)
                    item.setData(Qt.UserRole + 1, r.get("wav_path") or "")
                if key == "navg" and val and val >= 2:
                    item.setBackground(QBrush(QColor(210, 240, 210)))
                elif not has_wav:
                    item.setForeground(QBrush(QColor(160, 160, 160)))
                tbl.setItem(row_i, col_i, item)
        tbl.setSortingEnabled(True)
        tbl.setUpdatesEnabled(True)
        self._autosize_columns()

    # Per-column max widths (px); 0 = no cap
    _COL_MAX = [0, 55, 55, 80, 80, 95, 80, 110, 160, 0]

    def _autosize_columns(self) -> None:
        """Size columns from font metrics on first 50 rows; last col stretches."""
        from PyQt5.QtGui import QFontMetrics
        tbl = self._tbl
        fm  = QFontMetrics(tbl.font())
        hdr = tbl.horizontalHeader()
        PAD      = 18
        n_sample = min(50, tbl.rowCount())
        last_col = tbl.columnCount() - 1
        for col_i in range(last_col):
            w = fm.horizontalAdvance(
                tbl.horizontalHeaderItem(col_i).text()) + PAD
            for row_i in range(n_sample):
                item = tbl.item(row_i, col_i)
                if item:
                    w = max(w, fm.horizontalAdvance(item.text()) + PAD)
            cap = self._COL_MAX[col_i]
            if cap:
                w = min(w, cap)
            tbl.setColumnWidth(col_i, w)
        hdr.setSectionResizeMode(last_col, QHeaderView.Stretch)

    def _on_header_click(self, col: int) -> None:
        if col == self._sort_col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        order = Qt.AscendingOrder if self._sort_asc else Qt.DescendingOrder
        self._tbl.sortItems(col, order)

    def _on_select(self) -> None:
        if not self._tbl.selectedItems():
            return
        row_i = self._tbl.currentRow()
        item  = self._tbl.item(row_i, 0)
        if item is None:
            return
        sig_id = item.data(Qt.UserRole)
        if sig_id is not None:
            self.signal_selected.emit(sig_id)

    def _on_context_menu(self, pos) -> None:
        row_i = self._tbl.rowAt(pos.y())
        if row_i < 0:
            return
        col0 = self._tbl.item(row_i, 0)
        if col0 is None:
            return
        sig_id  = col0.data(Qt.UserRole)
        wav     = col0.data(Qt.UserRole + 1) or ""
        ts_item = col0  # timestamp text is col 0 display text
        ts      = col0.text()

        # Build the message columns from visible cells
        msg_col = next((i for i, (k, _, _) in enumerate(_TABLE_COLS) if k == "message"), None)
        msg = self._tbl.item(row_i, msg_col).text() if msg_col is not None else ""

        ref = f"signal_id={sig_id}  {ts}  {msg}"
        if wav:
            ref += f"\n  wav={wav}"

        menu = QMenu(self._tbl)

        act_ref = QAction("Copy signal reference", self._tbl)
        act_ref.triggered.connect(lambda: _QApp.clipboard().setText(ref))
        menu.addAction(act_ref)

        if wav:
            act_wav = QAction("Copy WAV path", self._tbl)
            act_wav.triggered.connect(lambda: _QApp.clipboard().setText(wav))
            menu.addAction(act_wav)

        act_id = QAction(f"Copy signal ID  ({sig_id})", self._tbl)
        act_id.triggered.connect(lambda: _QApp.clipboard().setText(str(sig_id)))
        menu.addAction(act_id)

        menu.exec_(self._tbl.viewport().mapToGlobal(pos))

    def select_id(self, signal_id: int) -> None:
        # After sort the row order is unknown — scan first column
        for row_i in range(self._tbl.rowCount()):
            item = self._tbl.item(row_i, 0)
            if item and item.data(Qt.UserRole) == signal_id:
                self._tbl.selectRow(row_i)
                self._tbl.scrollToItem(item)
                return


# ── Browse: Scatter view ──────────────────────────────────────────────────────

_SCATTER_METRICS = [
    ("snr_db",              "SNR (dB)"),
    ("dist_km",             "Distance (km)"),
    ("audio_hz",            "Audio Hz"),
    ("fest_hz",             "Off-freq Hz"),
    ("navg",                "navg"),
    ("bb_longest_ms",       "Burst duration (ms)"),
    ("drift_hz_per_s",      "Drift (Hz/s)"),
    ("f_std_hz",            "Freq std (Hz)"),
    ("f_range_hz",          "Freq range (Hz)"),
    ("af_corr",             "Amp↔Freq corr"),
    ("anomaly_score",       "Anomaly score"),
    ("cluster_id",          "Cluster ID"),
    ("sync_phase_coherence_h", "Sync coherence H"),
    ("theta_deg",           "Theta (deg)"),
    ("dt_s",                "DT (s)"),
]
_METRIC_NAMES  = [k for k, _ in _SCATTER_METRICS]
_METRIC_LABELS = {k: lbl for k, lbl in _SCATTER_METRICS}


class ScatterView(QWidget):
    signal_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._ids: list[int]     = []
        self._rows: list[dict]   = []

        row = QHBoxLayout()
        self._x_combo  = QComboBox()
        self._y_combo  = QComboBox()
        self._c_combo  = QComboBox()
        for combo in (self._x_combo, self._y_combo, self._c_combo):
            for k, lbl in _SCATTER_METRICS:
                combo.addItem(lbl, k)
        self._x_combo.setCurrentIndex(_METRIC_NAMES.index("dist_km"))
        self._y_combo.setCurrentIndex(_METRIC_NAMES.index("snr_db"))
        self._c_combo.setCurrentIndex(_METRIC_NAMES.index("navg"))
        row.addWidget(QLabel("X:"));   row.addWidget(self._x_combo)
        row.addWidget(QLabel("Y:"));   row.addWidget(self._y_combo)
        row.addWidget(QLabel("Color:")); row.addWidget(self._c_combo)
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._replot)
        row.addWidget(btn_plot)

        self._canvas = _make_canvas(8, 4)
        self._canvas.mpl_connect("pick_event", self._on_pick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self._canvas)

    def populate(self, rows: list[dict]) -> None:
        self._rows = rows
        self._ids  = [r["id"] for r in rows]
        self._replot()

    def _replot(self) -> None:
        x_key = self._x_combo.currentData()
        y_key = self._y_combo.currentData()
        c_key = self._c_combo.currentData()

        xs, ys, cs, ids_used = [], [], [], []
        for r in self._rows:
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is None or yv is None:
                continue
            xs.append(xv)
            ys.append(yv)
            cs.append(r.get(c_key) or 0)
            ids_used.append(r["id"])
        self._scatter_ids = ids_used

        fig = self._canvas.figure
        fig.clear()
        ax  = fig.add_subplot(111)
        if xs:
            sc = ax.scatter(xs, ys, c=cs, cmap="viridis",
                            s=12, alpha=0.7, picker=5)
            fig.colorbar(sc, ax=ax, label=_METRIC_LABELS.get(c_key, c_key))
        ax.set_xlabel(_METRIC_LABELS.get(x_key, x_key))
        ax.set_ylabel(_METRIC_LABELS.get(y_key, y_key))
        ax.set_title(f"{len(xs)} signals")
        fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def _on_pick(self, event) -> None:
        if not hasattr(event, "ind") or not len(event.ind):
            return
        idx = event.ind[0]
        if idx < len(self._scatter_ids):
            self.signal_selected.emit(self._scatter_ids[idx])


# ── Browse: Histogram view ────────────────────────────────────────────────────

class HistoView(QWidget):
    signal_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._rows: list[dict] = []

        row = QHBoxLayout()
        self._metric_combo = QComboBox()
        for k, lbl in _SCATTER_METRICS:
            self._metric_combo.addItem(lbl, k)
        self._n_bins = QSpinBox()
        self._n_bins.setRange(5, 200)
        self._n_bins.setValue(30)
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._replot)
        row.addWidget(QLabel("Metric:"))
        row.addWidget(self._metric_combo)
        row.addWidget(QLabel("Bins:"))
        row.addWidget(self._n_bins)
        row.addWidget(btn_plot)

        self._canvas = _make_canvas(8, 4)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self._canvas)

    def populate(self, rows: list[dict]) -> None:
        self._rows = rows
        self._replot()

    def _replot(self) -> None:
        key = self._metric_combo.currentData()
        vals = [r[key] for r in self._rows if r.get(key) is not None]
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if vals:
            ax.hist(vals, bins=self._n_bins.value(), color="#4488cc", edgecolor="white")
        ax.set_xlabel(_METRIC_LABELS.get(key, key))
        ax.set_ylabel("Count")
        ax.set_title(f"{len(vals)} signals with {key}")
        fig.tight_layout(pad=0.5)
        self._canvas.draw()


# ── Search sidebar ────────────────────────────────────────────────────────────

class SearchSidebar(QWidget):
    search_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(200)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        form  = QFormLayout(inner)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(4)

        def _spin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            return s

        def _dspin(lo, hi, val, step=0.1, dec=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            return s

        def _pair(lo_w, hi_w):
            row = QHBoxLayout()
            row.addWidget(lo_w)
            row.addWidget(QLabel("–"))
            row.addWidget(hi_w)
            w = QWidget(); w.setLayout(row)
            return w

        # Distance
        self._dist_lo = _spin(0, 20000, 0, 100)
        self._dist_hi = _spin(0, 20000, 20000, 100)
        form.addRow("Distance km:", _pair(self._dist_lo, self._dist_hi))

        # Callsign
        self._call_edit = QLineEdit()
        self._call_edit.setPlaceholderText("e.g. N4JP")
        form.addRow("Callsign:", self._call_edit)

        # Grid
        self._grid_edit = QLineEdit()
        self._grid_edit.setPlaceholderText("e.g. EM84")
        form.addRow("Grid:", self._grid_edit)

        # SNR
        self._snr_lo = _spin(-30, 30, -30)
        self._snr_hi = _spin(-30, 30, 5)
        form.addRow("SNR dB:", _pair(self._snr_lo, self._snr_hi))

        # Audio Hz
        self._af_lo = _spin(100, 3000, 300, 50)
        self._af_hi = _spin(100, 3000, 2700, 50)
        form.addRow("Audio Hz:", _pair(self._af_lo, self._af_hi))

        # Off-freq only
        self._off_freq_cb  = QCheckBox("Off-freq only")
        self._off_freq_thr = _spin(0, 1000, 50, 10)
        off_row = QHBoxLayout()
        off_row.addWidget(self._off_freq_cb)
        off_row.addWidget(self._off_freq_thr)
        off_row.addWidget(QLabel("Hz"))
        off_w = QWidget(); off_w.setLayout(off_row)
        form.addRow("", off_w)

        # navg checkboxes
        self._navg_1    = QCheckBox("1")
        self._navg_2    = QCheckBox("2")
        self._navg_3    = QCheckBox("3")
        self._navg_fail = QCheckBox("fail")
        self._navg_unk  = QCheckBox("unknown")
        for cb in (self._navg_1, self._navg_2, self._navg_3,
                   self._navg_fail, self._navg_unk):
            cb.setChecked(True)
        navg_row = QHBoxLayout()
        for cb in (self._navg_1, self._navg_2, self._navg_3,
                   self._navg_fail, self._navg_unk):
            navg_row.addWidget(cb)
        navg_w = QWidget(); navg_w.setLayout(navg_row)
        form.addRow("navg:", navg_w)

        # Duration
        self._dur_lo = _spin(0, 10000, 0, 50)
        self._dur_hi = _spin(0, 10000, 10000, 50)
        form.addRow("Dur ms:", _pair(self._dur_lo, self._dur_hi))

        # Drift
        self._drift_lo = _dspin(0, 100, 0.0, 0.5)
        self._drift_hi = _dspin(0, 100, 100.0, 0.5)
        form.addRow("|Drift| Hz/s:", _pair(self._drift_lo, self._drift_hi))

        # Flutter
        self._flutter_cb = QCheckBox("Flutter only (af_corr > 0.3)")
        form.addRow("", self._flutter_cb)

        # Sync coherence
        self._sync_min = _dspin(0.0, 1.0, 0.0, 0.05, 2)
        form.addRow("Sync coh ≥:", self._sync_min)

        # Multipath only
        self._multipath_cb = QCheckBox("Multipath only")
        form.addRow("", self._multipath_cb)

        # Unreviewed only
        self._unreviewed_cb = QCheckBox("Unreviewed only")
        form.addRow("", self._unreviewed_cb)

        # Source checkboxes
        self._src_map144 = QCheckBox("map144")
        self._src_flex   = QCheckBox("wsjtx_flex")
        self._src_wsjtx  = QCheckBox("wsjtx")
        for cb in (self._src_map144, self._src_flex, self._src_wsjtx):
            cb.setChecked(True)
        src_row = QHBoxLayout()
        for cb in (self._src_map144, self._src_flex, self._src_wsjtx):
            src_row.addWidget(cb)
        src_w = QWidget(); src_w.setLayout(src_row)
        form.addRow("Source:", src_w)

        # Tag filter
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("weak multipath …")
        form.addRow("Tags (AND):", self._tag_edit)

        # Max results
        self._limit_spin = _spin(10, 5000, 500, 100)
        form.addRow("Max results:", self._limit_spin)

        # Buttons
        btn_row = QHBoxLayout()
        self._search_btn = QPushButton("Search")
        self._clear_btn  = QPushButton("Clear")
        self._search_btn.clicked.connect(self._do_search)
        self._clear_btn.clicked.connect(self._do_clear)
        btn_row.addWidget(self._search_btn)
        btn_row.addWidget(self._clear_btn)
        btn_w = QWidget(); btn_w.setLayout(btn_row)
        form.addRow("", btn_w)

        # Summary strip
        self._summary = QLabel("—")
        self._summary.setWordWrap(True)
        form.addRow("Results:", self._summary)

        scroll.setWidget(inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(scroll)

    def update_summary(self, rows: list[dict]) -> None:
        n = len(rows)
        snrs  = [r["snr_db"] for r in rows if r.get("snr_db") is not None]
        dists = [r["dist_km"] for r in rows if r.get("dist_km") is not None]
        mean_snr  = f"  mean SNR {sum(snrs)/len(snrs):.1f} dB"  if snrs  else ""
        mean_dist = f"  mean dist {sum(dists)/len(dists):.0f} km" if dists else ""
        self._summary.setText(f"{n} signals{mean_snr}{mean_dist}")

    def _do_search(self) -> None:
        f: dict = {}

        lo, hi = self._dist_lo.value(), self._dist_hi.value()
        if lo > 0 or hi < 20000:
            f["dist_km_range"] = (lo if lo > 0 else None, hi if hi < 20000 else None)

        if self._call_edit.text().strip():
            f["callsign"] = self._call_edit.text().strip()

        if self._grid_edit.text().strip():
            f["grid"] = self._grid_edit.text().strip()

        lo, hi = self._snr_lo.value(), self._snr_hi.value()
        if lo > -30 or hi < 5:
            f["snr_db_range"] = (lo if lo > -30 else None, hi if hi < 30 else None)

        lo, hi = self._af_lo.value(), self._af_hi.value()
        if lo > 100 or hi < 2700:
            f["audio_hz_range"] = (lo, hi)

        if self._off_freq_cb.isChecked():
            f["off_freq_thresh_hz"] = self._off_freq_thr.value()

        # navg
        navg_list: list = []
        if self._navg_1.isChecked():   navg_list.append(1)
        if self._navg_2.isChecked():   navg_list.append(2)
        if self._navg_3.isChecked():   navg_list.append(3)
        if self._navg_fail.isChecked(): navg_list.append(-1)  # stored as NULL or -1
        if self._navg_unk.isChecked():  navg_list.append(None)
        if len(navg_list) < 5:  # not all selected — apply filter
            f["navg_list"] = navg_list

        lo, hi = self._dur_lo.value(), self._dur_hi.value()
        if lo > 0 or hi < 10000:
            f["duration_ms_range"] = (lo if lo > 0 else None, hi if hi < 10000 else None)

        lo, hi = self._drift_lo.value(), self._drift_hi.value()
        if lo > 0 or hi < 100:
            f["drift_hz_per_s_range"] = (lo if lo > 0 else None, hi if hi < 100 else None)

        if self._flutter_cb.isChecked():
            f["flutter_only"] = True

        if self._sync_min.value() > 0:
            f["sync_coherence_min"] = self._sync_min.value()

        if self._multipath_cb.isChecked():
            f["multipath_only"] = True

        if self._unreviewed_cb.isChecked():
            f["unreviewed_only"] = True

        sources = []
        if self._src_map144.isChecked(): sources.append("map144")
        if self._src_flex.isChecked():   sources.append("wsjtx_flex")
        if self._src_wsjtx.isChecked():  sources.append("wsjtx")
        if len(sources) < 3:
            f["source_list"] = sources

        tag = self._tag_edit.text().strip()
        if tag:
            f["tag_filter"] = tag

        f["_limit"] = self._limit_spin.value()
        self.search_requested.emit(f)

    def _do_clear(self) -> None:
        self._dist_lo.setValue(0)
        self._dist_hi.setValue(20000)
        self._call_edit.clear()
        self._grid_edit.clear()
        self._snr_lo.setValue(-30)
        self._snr_hi.setValue(5)
        self._af_lo.setValue(300)
        self._af_hi.setValue(2700)
        self._off_freq_cb.setChecked(False)
        for cb in (self._navg_1, self._navg_2, self._navg_3,
                   self._navg_fail, self._navg_unk):
            cb.setChecked(True)
        self._dur_lo.setValue(0)
        self._dur_hi.setValue(10000)
        self._drift_lo.setValue(0.0)
        self._drift_hi.setValue(100.0)
        self._flutter_cb.setChecked(False)
        self._sync_min.setValue(0.0)
        self._multipath_cb.setChecked(False)
        self._unreviewed_cb.setChecked(False)
        for cb in (self._src_map144, self._src_flex, self._src_wsjtx):
            cb.setChecked(True)
        self._tag_edit.clear()
        # Don't auto-search after clear — user must click Search


# ── Main window ───────────────────────────────────────────────────────────────

class SignalBrowser(QMainWindow):
    signal_selected = pyqtSignal(int)

    def __init__(self, db_path: Path = DB_PATH):
        super().__init__()
        self._conn = open_db(db_path)
        self.setWindowTitle(f"Signal Browser — {db_path.name}")
        self.resize(1400, 900)

        # Font
        font = QApplication.font()
        font.setPointSize(12)
        QApplication.setFont(font)

        # Sidebar
        self._sidebar = SearchSidebar()
        self._sidebar.search_requested.connect(self._on_search)

        # Browse tabs
        self._table_view   = TableView()
        self._scatter_view = ScatterView()
        self._histo_view   = HistoView()

        self._browse_tabs = QTabWidget()
        self._browse_tabs.addTab(self._table_view,   "Table")
        self._browse_tabs.addTab(self._scatter_view, "Scatter")
        self._browse_tabs.addTab(self._histo_view,   "Histogram")

        # Connect all browse views → signal_selected
        self._table_view.signal_selected.connect(self.signal_selected)
        self._scatter_view.signal_selected.connect(self.signal_selected)

        # Detail panel
        self._detail = SignalDetail(self._conn)
        self._detail.tag_changed.connect(self._on_tag_changed)
        self.signal_selected.connect(self._detail.load_signal)
        self.signal_selected.connect(self._table_view.select_id)

        # Right side: browse (top) + detail (bottom)
        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self._browse_tabs)
        right_split.addWidget(self._detail)
        right_split.setSizes([450, 350])

        # Main splitter
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(self._sidebar)
        main_split.addWidget(right_split)
        main_split.setSizes([300, 1100])

        self.setCentralWidget(main_split)

        # Lazy-render: only replot scatter/histo when tab becomes visible
        self._browse_tabs.currentChanged.connect(self._on_tab_changed)
        self._last_rows: list[dict] = []
        self._scatter_stale = False
        self._histo_stale   = False

    def _on_search(self, filters: dict) -> None:
        limit = filters.pop("_limit", 500)
        rows = search(self._conn, {**filters,
                                   "order_by":   "epoch_s",
                                   "order_desc": True,
                                   "limit":      limit})
        self._last_rows = rows
        self._table_view.populate(rows)
        ids = [r["id"] for r in rows]
        self._detail.set_result_ids(ids)
        self._sidebar.update_summary(rows)

        # Mark plots stale; only render the currently visible one
        self._scatter_stale = True
        self._histo_stale   = True
        self._on_tab_changed(self._browse_tabs.currentIndex())

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1 and self._scatter_stale:
            self._scatter_view.populate(self._last_rows)
            self._scatter_stale = False
        elif idx == 2 and self._histo_stale:
            self._histo_view.populate(self._last_rows)
            self._histo_stale = False

    def _on_tag_changed(self) -> None:
        pass  # could re-run last search; skip for now


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    import signal
    ap = argparse.ArgumentParser(description="MSK144 signal browser")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C exits cleanly
    win = SignalBrowser(db_path=Path(args.db))
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
