#!/usr/bin/env python3
"""Analyze a single MSK144 WAV file.

Two-row diagnostic display:
  Row 1 – Normal spectrogram (original or flattened) + median noise-floor panel.
  Row 2 – Squared-signal spectrogram (msk144spd.f90 algorithm) + detection-metric panel.

The squared-signal row implements the exact sliding-window algorithm from
msk144spd.f90: NSPM=864 samples per frame at 12 kHz (scaled to actual rate),
step=NSPM/4, 12-sample raised-cosine edge window, complex analytic squaring,
then FFT to locate the expected tones at 2*(fc±500) Hz.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import subprocess
import tempfile
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button, Slider

# ── msk144spd.f90 algorithm constants ────────────────────────────────────── #
NSPM_12K      = 864    # samples / MSK144 frame at 12 000 Hz  (72 ms)
STEP_12K      = 216    # step at 12 000 Hz                    (18 ms = NSPM/4)
EDGE_FADE_12K = 12     # raised-cosine edge length at 12 kHz (each end)
DETECT_THRESH = 3.0    # normalized detection metric threshold


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return mono float32 samples in [-1, 1] and sample rate."""
    with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        float_try = np.frombuffer(raw, dtype=np.float32)
        if np.all(np.isfinite(float_try)) and np.max(np.abs(float_try)) <= 10:
            data = float_try.astype(np.float32)
            max_abs = float(np.max(np.abs(data))) if data.size else 0.0
            if max_abs > 1.0:
                data /= max_abs
        else:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data.astype(np.float32), sample_rate


def _real_to_analytic(samples: np.ndarray) -> np.ndarray:
    """Convert a real signal to its analytic (single-sideband) complex form via FFT.

    Zeroing the negative-frequency bins removes the mirror-image component so
    that subsequent squaring places only one copy of the signal in the passband.
    """
    n = samples.size
    X = np.fft.fft(samples.astype(np.float64))
    H = np.zeros(n, dtype=np.float64)
    H[0] = 1.0                  # DC: keep as-is
    if n % 2 == 0:
        H[1:n // 2] = 2.0       # positive freqs: double (to preserve power)
        H[n // 2] = 1.0         # Nyquist: keep as-is
    else:
        H[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * H).astype(np.complex64)


def _compute_spectrogram(
    samples: np.ndarray,
    rate: int,
    nfft: int = 512,
    chunk_samples: int = 504,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_s, freq_hz, spec_db) for a real mono signal.

    spec_db has shape (nstep, nfreq) where nfreq = nfft//2 + 1.
    """
    nfft = max(256, min(int(nfft), samples.size))
    chunk_samples = max(1, int(chunk_samples))

    window = np.hanning(nfft).astype(np.float64)
    win_power = np.sum(window ** 2)
    frames: list[np.ndarray] = []
    times: list[float] = []
    elapsed_s = 0.0

    for start in range(0, samples.size, chunk_samples):
        chunk = samples[start:start + chunk_samples]
        if chunk.size == 0:
            continue
        block = np.zeros(nfft, dtype=np.float64)
        copy_len = min(chunk.size, nfft)
        block[:copy_len] = chunk[:copy_len].astype(np.float64)
        spec = np.fft.rfft(block * window)
        psd = (np.abs(spec) ** 2) / (rate * win_power)
        frames.append(10.0 * np.log10(psd + 1e-20))
        elapsed_s += chunk.size / rate
        times.append(elapsed_s)

    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / rate)
    if not frames:
        return np.array([0.0]), freq_hz, np.full((1, freq_hz.size), -180.0, dtype=np.float64)

    return np.array(times, dtype=np.float64), freq_hz, np.stack(frames)


def _estimate_median(spec_db: np.ndarray) -> np.ndarray:
    """Return per-frequency median noise floor (dB) across all time frames."""
    return np.median(spec_db.astype(np.float64), axis=0)


def _flatten(spec_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten spectrogram by raising bins with low noise floor to the peak floor level.

    Returns:
        spec_flat  – corrected spectrogram (nstep, nfreq)
        floor_flat – corrected noise floor (nfreq,); constant at peak_db
        correction – per-bin additive correction (nfreq,)
    """
    floor = _estimate_median(spec_db)
    peak = float(np.max(floor))
    correction = peak - floor
    spec_flat = spec_db.astype(np.float64) + correction[np.newaxis, :]
    return spec_flat, floor + correction, correction


def _gui_colormap() -> LinearSegmentedColormap:
    """Return the GUI-style spectrogram colormap (black → blue → cyan → white)."""
    colors = [
        (0,   0,   0),
        (0,   0,   64),
        (0,   0,   128),
        (0,   64,  192),
        (0,   128, 255),
        (64,  192, 255),
        (128, 255, 255),
        (255, 255, 128),
        (255, 255, 255),
    ]
    positions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    color_points = [
        (pos, (r / 255.0, g / 255.0, b / 255.0))
        for pos, (r, g, b) in zip(positions, colors)
    ]
    return LinearSegmentedColormap.from_list("flex_gui_spectrogram", color_points)


def _compute_squared_spectrogram(
    samples: np.ndarray,
    rate: int,
    fc_hz: float,
    ntol_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window squared-signal spectrogram (msk144spd.f90 algorithm).

    Returns:
        time_s    (nstep,)        frame-centre times in seconds
        freq_hz   (npos,)         frequency axis 0 … rate/2 Hz
        spec_db   (nstep, npos)   squared-power spectrogram in dB
        det_norm  (nstep,)        normalized detection metric
    """
    # Scale 12 kHz reference parameters to actual sample rate
    nspm   = int(round(NSPM_12K      * rate / 12000))  # frame length (72 ms)
    step   = int(round(STEP_12K      * rate / 12000))  # step   (18 ms)
    edge_n = int(round(EDGE_FADE_12K * rate / 12000))  # edge window
    nfft   = nspm
    df     = rate / nfft
    npos   = nfft // 2 + 1
    freq_hz = np.arange(npos, dtype=np.float64) * df   # 0 … rate/2 Hz

    # Raised-cosine edge window:  rcw[i] = (1 – cos(i·π/edge_n)) / 2
    rcw = (1.0 - np.cos(np.arange(edge_n) * np.pi / edge_n)) / 2.0

    # Expected squared-tone centres and search half-width (in bins)
    i_high = max(0, min(npos - 1, int(round(2.0 * (fc_hz + 500.0) / df))))
    i_low  = max(0, min(npos - 1, int(round(2.0 * (fc_hz - 500.0) / df))))
    hw     = max(1, int(round(2.0 * ntol_hz / df)))

    analytic = _real_to_analytic(samples)
    n        = analytic.size

    frames:   list[np.ndarray] = []
    times:    list[float]      = []
    det_raw:  list[float]      = []

    istp = 0
    while True:
        ns = istp * step
        ne = ns + nspm
        if ne > n:
            break

        blk = analytic[ns:ne].astype(np.complex128)

        # Step 1: square the analytic block
        blk_sq = blk ** 2

        # Step 2: apply raised-cosine edge window (matches rcw in msk144spd.f90)
        blk_sq[:edge_n]        *= rcw
        blk_sq[nspm - edge_n:] *= rcw[::-1]

        # Step 3: FFT → power spectrum (positive-frequency half only)
        tone = np.abs(np.fft.fft(blk_sq, n=nfft)[:npos]) ** 2

        frames.append(10.0 * np.log10(tone + 1e-20))
        times.append((ns + nspm / 2.0) / rate)

        # Step 4: detection metric – peak in ±2*ntol_hz window around each tone
        hi_lo = max(0, i_high - hw);  hi_hi = min(npos - 1, i_high + hw)
        lo_lo = max(0, i_low  - hw);  lo_hi = min(npos - 1, i_low  + hw)
        ah = float(tone[hi_lo:hi_hi + 1].max()) if hi_hi >= hi_lo else 0.0
        al = float(tone[lo_lo:lo_hi + 1].max()) if lo_hi >= lo_lo else 0.0
        det_raw.append(max(ah, al))

        istp += 1

    if not frames:
        return np.array([0.0]), freq_hz, np.zeros((1, npos)), np.zeros(1)

    spec_db = np.stack(frames)
    time_s  = np.array(times,   dtype=np.float64)
    det_arr = np.array(det_raw, dtype=np.float64)

    # Normalize by 25th percentile (mirrors msk144spd "xmed" normalization)
    pct25    = float(np.percentile(det_arr, 25)) if det_arr.size > 1 else float(det_arr[0])
    det_norm = det_arr / max(pct25, 1e-20)

    return time_s, freq_hz, spec_db, det_norm


JT9_PATH = '/home/jeff/ham/wsjtx-2.7.0/build/wsjtx-prefix/src/wsjtx-build/jt9'
JT9_BASE_ARGS = [
    JT9_PATH, "--msk144",
    "-p", "15", "-L", "1400", "-H", "1600",
    "-f", "1500", "-F", "200", "-d", "3",
]


def run_detections(
    wav_path: Path,
    samples: np.ndarray,
    rate: int,
    fc_hz: float,
    ntol_hz: float,
) -> None:
    """Print header line then detect bursts, run jt9 on each, and print results."""
    print(
        f"\n{wav_path.name}: {rate} Hz, {samples.size} samples "
        f"({samples.size / rate:.3f} s), fc={fc_hz:.0f} Hz, ntol=\u00b1{ntol_hz:.0f} Hz"
    )

    t2, _, _, det_norm = _compute_squared_spectrogram(samples, rate, fc_hz, ntol_hz)
    block_dur = t2[1] - t2[0] if len(t2) > 1 else 0.0

    detections: list[tuple[int, int]] = []
    start_idx = None
    for idx, metric in enumerate(det_norm):
        if metric >= DETECT_THRESH:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                detections.append((start_idx, idx - 1))
                start_idx = None
    if start_idx is not None:
        detections.append((start_idx, len(det_norm) - 1))
    detections.sort(key=lambda s_e: det_norm[s_e[0]:s_e[1]+1].max(), reverse=True)

    pad = np.zeros(int(0.5 * rate), dtype=np.float32)

    print(f"\njt9: {' '.join(JT9_BASE_ARGS)} <wav>")
    print(f"{'start':>7}  {'dur':>6}  {'metric':>6}  decode")
    if not detections:
        print("  (no detections above threshold)")
    for d_start, d_end in detections:
        t_start    = t2[d_start]
        t_end      = t2[d_end]
        max_metric = float(det_norm[d_start:d_end+1].max())
        duration   = (d_end - d_start + 1) * block_dur
        segment    = samples[int(t_start * rate):int(t_end * rate)]
        padded     = np.concatenate([pad, segment, pad])

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes((padded * 32767).astype(np.int16).tobytes())
            result = subprocess.run(
                JT9_BASE_ARGS + [tmp_path],
                capture_output=True, text=True,
            )
            decoded = ""
            for line in result.stdout.splitlines():
                s = line.strip()
                if s and not s.startswith('<') and not s.startswith('EOF'):
                    decoded = s
                    break
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        print(f"{t_start:7.3f}  {duration:6.3f}  {max_metric:6.2f}  {decoded or 'no decode'}")


# ═══════════════════════════════════════════════════════════════════════════ #
#   Main plot                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def plot_analysis(
    name:       str,
    samples:    np.ndarray,
    rate:       int,
    output_path: Path,
    show_plots: bool,
    flatten:    bool,
    fc_hz:      float,
    ntol_hz:    float,
    timings:    dict | None = None,
) -> None:
    def _tick(label: str, t0: float) -> float:
        """Record elapsed time for a section and return a new start time."""
        if timings is not None:
            timings[label] = time.perf_counter() - t0
        return time.perf_counter()

    t0 = time.perf_counter()

    # ── Row 1: normal spectrogram ──────────────────────────────────────────
    t1, f1, s1_raw = _compute_spectrogram(samples, rate)
    t0 = _tick('analysis: normal_spectrogram', t0)

    if flatten:
        s1, floor1, _ = _flatten(s1_raw)
        row1_title    = "Spectrogram (flattened)"
    else:
        s1     = s1_raw.astype(np.float64)
        floor1 = _estimate_median(s1)
        row1_title = "Spectrogram (original)"
    t0 = _tick('analysis: median_flatten_row1', t0)

    # ── Row 2: squared-signal spectrogram ──────────────────────────────────
    t2, f2, s2, det_norm = _compute_squared_spectrogram(samples, rate, fc_hz, ntol_hz)
    t0 = _tick('operational: squared_spectrogram', t0)

    floor2 = _estimate_median(s2)
    t0 = _tick('analysis: median_floor_row2', t0)

    # Expected squared-tone frequencies (for reference lines)
    f_sq_high = 2.0 * (fc_hz + 500.0)
    f_sq_low  = 2.0 * (fc_hz - 500.0)

    # ── Figure layout ──────────────────────────────────────────────────────
    gui_cmap = _gui_colormap()
    baseline_default = -110.0
    gain_default     =   40.0
    vmin0 = baseline_default
    vmax0 = baseline_default + gain_default

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 10.0),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.0, 0.7], "height_ratios": [1.0, 1.0, 0.4]},
    )
    fig.suptitle(f"MSK144 Analysis \u2013 {name}", fontsize=13)

    def _extent(t: np.ndarray, f: np.ndarray) -> list[float]:
        return [float(t[0]), float(t[-1]), float(f[0]) / 1000.0, float(f[-1]) / 1000.0]

    # ── [0,0]  Normal spectrogram ──────────────────────────────────────────
    ax00, ax01 = axes[0]
    img1 = ax00.imshow(
        s1.T, aspect='auto', origin='lower',
        extent=_extent(t1, f1), cmap=gui_cmap, vmin=vmin0, vmax=vmax0,
    )
    ax00.set_title(row1_title)
    ax00.set_xlabel("Time (s)")
    ax00.set_ylabel("Frequency (kHz)")
    ax00.set_ylim(0.0, 6.0)
    ax00.xaxis.set_major_locator(MultipleLocator(1.0))

    # ── [0,1]  Median noise floor vs frequency ─────────────────────────────
    floor1_line, = ax01.plot(floor1, f1 / 1000.0, lw=1.2, color='tab:orange')
    ax01.set_title("Median Noise")
    ax01.set_xlabel("Level (dB)")
    ax01.set_ylabel("Frequency (kHz)")
    ax01.set_xlim(vmin0, vmax0)
    ax01.set_ylim(0.0, 6.0)
    ax01.grid(True, alpha=0.25)

    # ── [1,0]  Squared spectrogram ─────────────────────────────────────────
    ax10, ax11 = axes[1]

    # Auto-scale the squared spectrogram (its absolute dB level differs from row 1)
    s2_finite = s2[np.isfinite(s2)]
    sq_vmin = float(np.percentile(s2_finite,  5)) if s2_finite.size else -180.0
    sq_vmax = float(np.percentile(s2_finite, 99)) if s2_finite.size else -100.0
    sq_vmax = max(sq_vmax, sq_vmin + gain_default)

    img2 = ax10.imshow(
        s2.T, aspect='auto', origin='lower',
        extent=_extent(t2, f2), cmap=gui_cmap, vmin=sq_vmin, vmax=sq_vmax,
    )

    # Reference lines at expected squared-tone frequencies
    ax10.axhline(f_sq_high / 1000.0, color='yellow', lw=0.9, ls='--',
                 label=f'2\u00b7(fc+500)={f_sq_high:.0f} Hz')
    ax10.axhline(f_sq_low  / 1000.0, color='cyan',   lw=0.9, ls='--',
                 label=f'2\u00b7(fc\u2212500)={f_sq_low:.0f} Hz')

    ax10.set_title(
        f"Squared-Signal Spectrogram  (fc={fc_hz:.0f} Hz, ntol=\u00b1{ntol_hz:.0f} Hz)"
    )
    ax10.set_xlabel("Time (s)")
    ax10.set_ylabel("Frequency (kHz)")
    ax10.xaxis.set_major_locator(MultipleLocator(1.0))
    ax10.legend(fontsize=8, loc='upper right')

    # ── [1,1]  Squared-spectrum median noise floor vs frequency ────────────
    floor2_line, = ax11.plot(floor2, f2 / 1000.0, lw=1.2, color='tab:orange')
    ax11.set_title("Median Noise")
    ax11.set_xlabel("Level (dB)")
    ax11.set_ylabel("Frequency (kHz)")
    ax11.grid(True, alpha=0.25)

    # ── [2,0]  Detection metric vs time (x-axis aligned with [1,0]) ────────
    ax20, ax21 = axes[2]
    ax20.sharex(ax10)
    ax20.step(t2, det_norm, where='mid', color='tab:green', lw=1.0)
    ax20.axhline(DETECT_THRESH, color='red', lw=0.9, ls='--',
                 label=f'Threshold ({DETECT_THRESH})')
    ax20.fill_between(t2, det_norm, DETECT_THRESH,
                      where=(det_norm >= DETECT_THRESH),
                      color='red', alpha=0.25, label='Detections')
    ax20.set_xlabel("Time (s)")
    ax20.set_ylabel("Norm. metric")
    ax20.set_ylim(bottom=0.0)
    ax20.legend(fontsize=8, loc='upper right')
    ax20.grid(True, alpha=0.25)

    # ── [2,1]  Hidden (no content needed alongside metric strip) ───────────
    ax21.set_visible(False)

    t0 = _tick('analysis: plot_layout_and_axes', t0)

    # ── Baseline / Gain sliders + Open WAV button ──────────────────────────
    if show_plots:
        fig.tight_layout(rect=[0, 0.10, 1, 0.97])
        btn_ax = fig.add_axes([0.01, 0.035, 0.11, 0.030])
        bl_ax  = fig.add_axes([0.15, 0.04,  0.30, 0.025])
        gn_ax  = fig.add_axes([0.58, 0.04,  0.30, 0.025])

        open_btn = Button(btn_ax, 'Open WAV\u2026', color='0.15', hovercolor='0.30')
        bl_sl = Slider(bl_ax, "Baseline (dB)", -240.0, -20.0,
                       valinit=baseline_default, valstep=1.0)
        gn_sl = Slider(gn_ax, "Gain (dB)", 10.0, 220.0,
                       valinit=gain_default, valstep=1.0)

        def _update(_: object) -> None:
            bl = float(bl_sl.val)
            gn = float(gn_sl.val)
            img1.set_clim(bl, bl + gn)
            ax01.set_xlim(bl, bl + gn)
            fig.canvas.draw_idle()

        bl_sl.on_changed(_update)
        gn_sl.on_changed(_update)

        dialog_dir = [output_path.parent]

        def _reload(new_wav: Path) -> None:
            new_samples, new_rate = read_wav_mono(new_wav)

            # Row 1
            t1n, f1n, s1n_raw = _compute_spectrogram(new_samples, new_rate)
            if flatten:
                s1n, floor1n, _ = _flatten(s1n_raw)
                ax00.set_title("Spectrogram (flattened)")
            else:
                s1n = s1n_raw.astype(np.float64)
                floor1n = _estimate_median(s1n)
                ax00.set_title("Spectrogram (original)")
            img1.set_data(s1n.T)
            img1.set_extent(_extent(t1n, f1n))
            floor1_line.set_xdata(floor1n)
            floor1_line.set_ydata(f1n / 1000.0)

            # Row 2
            t2n, f2n, s2n, det_norm_n = _compute_squared_spectrogram(
                new_samples, new_rate, fc_hz, ntol_hz
            )
            floor2n = _estimate_median(s2n)
            s2n_fin = s2n[np.isfinite(s2n)]
            sq_vmin_n = float(np.percentile(s2n_fin,  5)) if s2n_fin.size else -180.0
            sq_vmax_n = float(np.percentile(s2n_fin, 99)) if s2n_fin.size else -100.0
            sq_vmax_n = max(sq_vmax_n, sq_vmin_n + gain_default)
            img2.set_data(s2n.T)
            img2.set_extent(_extent(t2n, f2n))
            img2.set_clim(sq_vmin_n, sq_vmax_n)
            floor2_line.set_xdata(floor2n)
            floor2_line.set_ydata(f2n / 1000.0)
            ax11.relim()
            ax11.autoscale_view(scalex=True, scaley=False)

            # Detection metric strip — clear and replot
            ax20.cla()
            ax20.step(t2n, det_norm_n, where='mid', color='tab:green', lw=1.0)
            ax20.axhline(DETECT_THRESH, color='red', lw=0.9, ls='--',
                         label=f'Threshold ({DETECT_THRESH})')
            ax20.fill_between(t2n, det_norm_n, DETECT_THRESH,
                              where=(det_norm_n >= DETECT_THRESH),
                              color='red', alpha=0.25, label='Detections')
            ax20.set_xlabel("Time (s)")
            ax20.set_ylabel("Norm. metric")
            ax20.set_ylim(bottom=0.0)
            ax20.xaxis.set_major_locator(MultipleLocator(1.0))
            ax20.legend(fontsize=8, loc='upper right')
            ax20.grid(True, alpha=0.25)

            new_out = new_wav.parent / (new_wav.stem + '_analysis.png')
            fig.suptitle(f"MSK144 Analysis \u2013 {new_wav.name}", fontsize=13)
            fig.canvas.draw_idle()
            fig.savefig(new_out, dpi=140)
            print(f"Wrote: {new_out}")
            run_detections(new_wav, new_samples, new_rate, fc_hz, ntol_hz)

        def _on_open_wav(_event) -> None:
            root = tk.Tk()
            root.withdraw()
            ttk.Style(root).theme_use('clam')
            root.lift()
            chosen = filedialog.askopenfilename(
                parent=root,
                title="Select MSK144 WAV file",
                initialdir=str(dialog_dir[0]),
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            )
            root.destroy()
            if not chosen:
                return
            new_wav = Path(chosen)
            dialog_dir[0] = new_wav.parent
            _reload(new_wav)

        open_btn.on_clicked(_on_open_wav)
    else:
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    t0 = time.perf_counter()
    fig.savefig(output_path, dpi=140)
    _tick('analysis: png_save', t0)
    print(f"Wrote: {output_path}")

    if show_plots:
        plt.show()

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════ #
#   Entry point                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Analyze a single MSK144 WAV: normal + squared-signal periodograms.",
        )
        parser.add_argument('wav', nargs='?', default=None,
                            help='Input WAV file path (opens dialog if omitted)')
        parser.add_argument(
            '--fc-hz', type=float, default=1500.0,
            help='Expected signal centre frequency in Hz (default: 1500)',
        )
        parser.add_argument(
            '--ntol-hz', type=float, default=50.0,
            help='Search tolerance around each squared tone in Hz (default: 50)',
        )
        parser.add_argument(
            '--flatten-spectrum', dest='flatten', action='store_true', default=True,
            help='Flatten row-1 spectrogram by median noise floor (default)',
        )
        parser.add_argument(
            '--no-flatten-spectrum', dest='flatten', action='store_false',
            help='Show raw (unflattened) spectrogram in row 1',
        )
        parser.add_argument(
            '--show-plots', dest='show', action='store_true', default=True,
            help='Display interactive plot window (default)',
        )
        parser.add_argument(
            '--no-show-plots', dest='show', action='store_false',
            help='Save PNG only; do not open a window',
        )
        parser.add_argument(
            '--plot-output', default='',
            help='Output PNG path (default: <wav_stem>_analysis.png)',
        )
        parser.add_argument(
            '--profile', action='store_true', default=False,
            help='Run cProfile and print top functions by cumulative time (forces --no-show-plots)',
        )
        args = parser.parse_args()

        if args.wav is None:
            root = tk.Tk()
            root.withdraw()
            ttk.Style(root).theme_use('clam')
            root.lift()
            chosen = filedialog.askopenfilename(
                parent=root,
                title="Select MSK144 WAV file",
                initialdir=str(Path.cwd()),
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            )
            root.destroy()
            if not chosen:
                raise SystemExit("No file selected.")
            wav_path = Path(chosen)
        else:
            wav_path = Path(args.wav)
            if not wav_path.exists():
                raise SystemExit(f"File not found: {wav_path}")

        if args.plot_output:
            out_path = Path(args.plot_output)
        else:
            out_path = wav_path.parent / (wav_path.stem + '_analysis.png')

        samples, rate = read_wav_mono(wav_path)
        show = args.show and not args.profile
        timings: dict | None = {} if args.profile else None

        run_detections(wav_path, samples, rate, args.fc_hz, args.ntol_hz)

        call_kwargs = dict(
            name=wav_path.name,
            samples=samples,
            rate=rate,
            output_path=out_path,
            show_plots=show,
            flatten=args.flatten,
            fc_hz=args.fc_hz,
            ntol_hz=args.ntol_hz,
            timings=timings,
        )

        if args.profile:
            prof_path = wav_path.parent / (wav_path.stem + '_profile.prof')
            pr = cProfile.Profile()
            pr.enable()
            plot_analysis(**call_kwargs)
            pr.disable()
            pr.dump_stats(str(prof_path))

            # ── Section timing table ──────────────────────────────────────────
            total = sum(timings.values())
            print("\nSection timing  (tag: operational = would run in detector, analysis = display only)")
            print(f"  {'Section':<40}  {'Time (s)':>8}  {'%':>5}")
            print(f"  {'-'*40}  {'-'*8}  {'-'*5}")
            for label, secs in timings.items():
                tag = label.split(':')[0].strip()
                name_part = label.split(':', 1)[1].strip()
                pct = 100.0 * secs / total if total > 0 else 0.0
                marker = '***' if tag == 'operational' else ''
                print(f"  [{tag}]  {name_part:<30}  {secs:>8.3f}  {pct:>4.1f}%  {marker}")
            print(f"  {'TOTAL':<40}  {total:>8.3f}")

            # ── cProfile detail ───────────────────────────────────────────────
            buf = io.StringIO()
            ps  = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(25)
            print(buf.getvalue())
            print(f"Profile saved: {prof_path}")
            print(f"  snakeviz {prof_path}          # interactive sunburst in browser")
            print(f"  gprof2dot -f pstats {prof_path} | dot -Tsvg -o callgraph.svg  # call graph")
        else:
            plot_analysis(**call_kwargs)
    except KeyboardInterrupt:
        print("\nExiting on user interrupt (Ctrl+C). Goodbye!")


if __name__ == '__main__':
    main()
