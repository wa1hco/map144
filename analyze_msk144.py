#!/usr/bin/env python3
"""Analyze a single MSK144 WAV file.

Standalone diagnostic and decode tool for MSK144 audio files.  Produces a
four-panel matplotlib figure, runs the WSJT-X ``jt9`` decoder on every
detected burst, and can save results interactively.  Accepts stereo IQ (L=I,
R=Q) or mono real audio; handles 8-bit, 16-bit, 32-bit int, and float32 WAV.

Display layout (3 rows × 2 columns)
------------------------------------
[0, 0]  Normal spectrogram — Hanning-windowed, 50 % overlap FFT, expressed as
        power-spectral density in dB/Hz.  For complex IQ input the bilateral
        FFT is used (−Fs/2 … +Fs/2); for mono real input rfft is used
        (0 … Fs/2).  Optional flattening (``--flatten-spectrum``, default on)
        equalises the noise floor across frequency by adding the median-floor
        correction per bin, making weak signals visible against a tilted
        passband.

[0, 1]  Median noise-floor curve for row 0 — per-frequency 50th-percentile
        power across all time frames.

[1, 0]  Squared-signal spectrogram — implements the exact sliding-window
        algorithm from WSJT-X ``msk144spd.f90``:
          * Frame length NSPM = 864 samples at 12 kHz (72 ms), scaled to
            the actual input sample rate.
          * Step = NSPM / 4 (18 ms, 75 % overlap).
          * 12-sample raised-cosine edge window applied to each frame after
            squaring: ``rcw[i] = (1 − cos(i·π/edge_n)) / 2``.
          * For complex IQ input, the IQ is LP-filtered to ±10 kHz before
            squaring so out-of-band signals do not alias into the squared
            domain.  For mono input, the signal is converted to its analytic
            form first (via FFT, zeroing negative-frequency bins) to eliminate
            the mirror-image tone pair that would appear at negative fc when
            squaring a real signal.
          * FFT of each squared, windowed frame gives the squared spectrum.
          * Reference lines mark the expected squared-tone centres at
            2·(fc±500) Hz.
          * A display offset (``sq_offset``) aligns the squared-spectrum noise
            median with the normal-spectrum noise median so both rows share the
            same colour scale.

[1, 1]  Median noise-floor curve for the squared spectrum.

[2, 0]  Detection metric strip — the per-frame tone-pair strength normalised by
        its 25th percentile (matching the ``xmed`` normalisation in
        ``msk144spd.f90``).  Red dashed threshold line at ``DETECT_THRESH = 3.0``.
        Red fill marks frames above threshold.  X-axis is shared with [1, 0].

[2, 1]  Hidden (no content needed alongside the metric strip).

Detection and decode pipeline (``run_detections``)
---------------------------------------------------
1. **Threshold crossing** — contiguous runs of frames where
   ``det_norm >= DETECT_THRESH`` are collected as candidate bursts.
2. **Gap merging** — bursts separated by ≤ ``DETECT_MERGE_GAP_S = 0.4 s``
   are merged into a single detection.
3. **Ranking** — bursts sorted strongest-first by peak metric; all are
   attempted regardless of rank.
4. **Segment extraction** — ±0.5 s around each burst is extracted from the
   source array, zero-padded by 0.1 s each end.
5. **Decode** — the segment is written to a temporary 16-bit mono WAV at the
   source sample rate and passed to ``jt9 --msk144 -p 15 -L 1400 -H 1600
   -f 1500 -F 200 -d 3``.
6. **Save** — on a successful decode the padded segment WAV is saved to
   ``<source_dir>/detections/<stem>_t<time>.wav``; detection files
   (filenames containing ``_t<digits>``) are not re-saved to avoid recursion.

Interactive controls (``--show-plots`` mode)
--------------------------------------------
Scroll wheel (over any spectrogram or noise panel)
    Plain scroll — shifts both colour limits together (brightness).
    Ctrl + scroll — expands or contracts the span (contrast).
    Step size: 2 dB per click.  Current limits shown in a status line below
    the figure.

"Open WAV…" button
    Opens a file dialog; reloads the figure with the new file in-place
    (updates image data, axes, noise curves, detection metric, and saves a
    new PNG) without opening a second window.

Persistent file-picker Toplevel (TkAgg backend)
    A separate Tk window listing all ``*.wav`` files in the current directory
    appears alongside the figure.  Double-click or Enter loads a file; the
    Browse Dir… button switches directory.

Command-line arguments
----------------------
wav                     Input WAV file path; opens a file dialog if omitted.
--fc-hz FLOAT           Expected carrier frequency in Hz (default: 1500).
--ntol-hz FLOAT         ±Search tolerance around each squared tone (default: 50).
--flatten-spectrum      Flatten row-1 spectrogram by median noise floor (default on).
--no-flatten-spectrum   Show raw (unflattened) spectrogram in row 1.
--show-plots            Open interactive window (default on).
--no-show-plots         Save PNG only; no window.
--plot-output PATH      Output PNG path (default: ``<stem>_analysis.png``).
--profile               Run ``cProfile`` + section timing table; implies
                        ``--no-show-plots``.

Algorithm constants
-------------------
NSPM_12K      : 864    samples/frame at 12 kHz (72 ms, matches msk144spd.f90)
STEP_12K      : 216    step at 12 kHz (18 ms = NSPM/4)
EDGE_FADE_12K : 12     raised-cosine edge length at 12 kHz
DETECT_THRESH     : 3.0   normalised detection threshold
DETECT_MERGE_GAP_S: 0.4 s gap within which successive detections are merged
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import re
import subprocess
import tempfile
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button, Slider

# ── msk144spd.f90 algorithm constants ────────────────────────────────────── #
NSPM_12K      = 864    # samples / MSK144 frame at 12 000 Hz  (72 ms)
STEP_12K      = 216    # step at 12 000 Hz                    (18 ms = NSPM/4)
EDGE_FADE_12K = 12     # raised-cosine edge length at 12 kHz (each end)
DETECT_THRESH     = 3.0    # normalized detection metric threshold
DETECT_MERGE_GAP_S = 0.4  # merge successive detections with gaps <= this (seconds)

# ── Per-channel detection constants (mirror processing.py) ──────────────── #
_PDET_FFT_SIZE   = 512
_PDET_HOP        = _PDET_FFT_SIZE // 2
_PDET_SQ_TONE    = 1000.0   # Hz — squared-domain tone offset from DC
_PDET_SQ_NTOL    = 200.0    # Hz — half-width of each tone search window
_PDET_THRESH_DB  = 3.0      # dB above 25th-percentile noise baseline (WSJT-X style)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file.

    Returns complex64 IQ (L=I, R=Q) for stereo files, or float32 mono
    for single-channel files.  Samples are normalised to ±1.
    """
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

    if channels == 2:
        frames = data.reshape(-1, 2)
        return (frames[:, 0] + 1j * frames[:, 1]).astype(np.complex64), sample_rate

    if channels > 2:
        data = data.reshape(-1, channels).mean(axis=1)

    return data.astype(np.float32), sample_rate


# Keep the old name as an alias for callers that expect mono
def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    return read_wav(path)


def _lp_filter_iq(samples: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    """FFT-domain LP filter for complex IQ: keeps |f| <= cutoff_hz with a 10 % taper."""
    n = samples.size
    freq = np.fft.fftfreq(n, 1.0 / rate)
    X = np.fft.fft(samples.astype(np.complex128))
    taper_bw = 0.10 * cutoff_hz
    f_abs = np.abs(freq)
    gain = np.zeros(n, dtype=np.float64)
    gain[f_abs <= cutoff_hz] = 1.0
    trans = (f_abs > cutoff_hz) & (f_abs <= cutoff_hz + taper_bw)
    if np.any(trans):
        t = (f_abs[trans] - cutoff_hz) / taper_bw
        gain[trans] = 0.5 + 0.5 * np.cos(np.pi * t)
    return np.fft.ifft(X * gain).astype(np.complex64)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_s, freq_hz, spec_db) with 50% overlap.

    For complex IQ input the bilateral FFT is used (freq_hz spans -rate/2 … +rate/2).
    For real mono input rfft is used (0 … rate/2).
    spec_db shape: (nstep, nfreq).
    """
    is_complex = np.iscomplexobj(samples)
    nfft = max(256, min(int(nfft), samples.size))
    hop = nfft // 2

    window = np.hanning(nfft).astype(np.float64)
    win_power = np.sum(window ** 2)
    frames: list[np.ndarray] = []
    times: list[float] = []

    for start in range(0, samples.size, hop):
        chunk = samples[start:start + nfft]
        if is_complex:
            block = np.zeros(nfft, dtype=np.complex128)
            block[:chunk.size] = chunk.astype(np.complex128)
            spec = np.fft.fftshift(np.fft.fft(block * window))
        else:
            block = np.zeros(nfft, dtype=np.float64)
            block[:chunk.size] = chunk.astype(np.float64)
            spec = np.fft.rfft(block * window)
        psd = (np.abs(spec) ** 2) / (rate * win_power)
        frames.append(10.0 * np.log10(psd + 1e-20))
        times.append((start + nfft / 2) / rate)

    if is_complex:
        freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate))
    else:
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
    return LinearSegmentedColormap.from_list("radio_iq_spectrogram", color_points)


def _build_file_picker(
    parent_win,
    dialog_dir: list,
    reload_fn,
    initial_filename: str = "",
) -> None:
    """Create a persistent WAV-file selector Toplevel alongside the figure.

    parent_win     – the TkAgg figure window (fig.canvas.manager.window)
    dialog_dir     – mutable list[Path] pointing to the current directory
    reload_fn      – callable(Path) that reloads the figure
    initial_filename – wav filename to highlight on first open
    """
    win = tk.Toplevel(parent_win)
    win.title("WAV Files")
    win.geometry("300x440")
    win.resizable(True, True)

    dir_var = tk.StringVar(value=str(dialog_dir[0]))
    tk.Label(
        win, textvariable=dir_var, wraplength=280,
        justify=tk.LEFT, relief=tk.SUNKEN, anchor='w', padx=4,
    ).pack(fill=tk.X, padx=6, pady=(6, 2))

    list_frame = tk.Frame(win)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

    sb = tk.Scrollbar(list_frame)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    lb = tk.Listbox(
        list_frame, yscrollcommand=sb.set, selectmode=tk.SINGLE,
        exportselection=False, activestyle='dotbox',
    )
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.config(command=lb.yview)

    def _populate(highlight: str = "") -> None:
        lb.delete(0, tk.END)
        for w in sorted(dialog_dir[0].glob('*.wav')):
            lb.insert(tk.END, w.name)
        dir_var.set(str(dialog_dir[0]))
        if highlight:
            items = lb.get(0, tk.END)
            if highlight in items:
                idx = list(items).index(highlight)
                lb.selection_set(idx)
                lb.see(idx)

    def _load(_event=None) -> None:
        sel = lb.curselection()
        if not sel:
            return
        chosen = dialog_dir[0] / lb.get(sel[0])
        if chosen.exists():
            dialog_dir[0] = chosen.parent
            reload_fn(chosen)

    def _browse() -> None:
        chosen_dir = filedialog.askdirectory(
            parent=win, title="Select directory",
            initialdir=str(dialog_dir[0]),
        )
        if chosen_dir:
            dialog_dir[0] = Path(chosen_dir)
            _populate()

    # Double-click or Enter to load; single click just highlights
    lb.bind('<Double-Button-1>', _load)
    lb.bind('<Return>', _load)

    btn_bar = tk.Frame(win)
    btn_bar.pack(fill=tk.X, padx=6, pady=(2, 6))
    ttk.Button(btn_bar, text='Load',        command=_load).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_bar, text='Refresh',     command=_populate).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_bar, text='Browse Dir…', command=_browse).pack(side=tk.LEFT, padx=2)

    _populate(highlight=initial_filename)


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

    is_complex = np.iscomplexobj(samples)
    if is_complex:
        # LP-filter IQ to ±10 kHz before squaring so out-of-band signals don't
        # alias into the squared spectrum.  After squaring, a ±10 kHz band maps
        # to ±20 kHz in the squared domain — well within the ±24 kHz display range.
        analytic = _lp_filter_iq(samples, rate, cutoff_hz=10000.0).astype(np.complex128)
        freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
        n_spec = nfft          # bilateral: all bins
    else:
        analytic = _real_to_analytic(samples)
        freq_hz = np.arange(npos, dtype=np.float64) * df   # 0 … rate/2
        n_spec = npos

    n = analytic.size

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

        # Step 3: FFT → power spectrum
        fft_sq = np.fft.fft(blk_sq, n=nfft)
        tone_pos = np.abs(fft_sq[:npos]) ** 2          # always use positive half for detection

        if is_complex:
            spec_frame = 10.0 * np.log10(
                np.fft.fftshift(np.abs(fft_sq) ** 2) + 1e-20
            )
        else:
            spec_frame = 10.0 * np.log10(tone_pos + 1e-20)

        frames.append(spec_frame)
        times.append((ns + nspm / 2.0) / rate)

        # Step 4: detection metric – peak in ±2*ntol_hz window around each tone
        hi_lo = max(0, i_high - hw);  hi_hi = min(npos - 1, i_high + hw)
        lo_lo = max(0, i_low  - hw);  lo_hi = min(npos - 1, i_low  + hw)
        ah = float(tone_pos[hi_lo:hi_hi + 1].max()) if hi_hi >= hi_lo else 0.0
        al = float(tone_pos[lo_lo:lo_hi + 1].max()) if lo_hi >= lo_lo else 0.0
        det_raw.append(max(ah, al))

        istp += 1

    if not frames:
        return np.array([0.0]), freq_hz, np.zeros((1, n_spec)), np.zeros(1)

    spec_db = np.stack(frames)
    time_s  = np.array(times,   dtype=np.float64)
    det_arr = np.array(det_raw, dtype=np.float64)

    # Normalize by 25th percentile (mirrors msk144spd "xmed" normalization).
    # Floor at 0.1 % of the mean prevents blowup when the detection bins contain
    # near-zero power (e.g. complex IQ input where signals are not at fc_hz).
    pct25    = float(np.percentile(det_arr, 25)) if det_arr.size > 1 else float(det_arr[0])
    mean_val = float(np.mean(det_arr)) if det_arr.size > 0 else 1.0
    ref      = max(pct25, mean_val * 0.001, 1e-20)
    det_norm = det_arr / ref

    return time_s, freq_hz, spec_db, det_norm


JT9_BASE_ARGS = [
    'jt9', "--msk144",
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
    if np.iscomplexobj(samples):
        print("  (complex IQ input – jt9 decode requires mono; skipping detection)")
        return

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

    # Merge detections whose gap is within DETECT_MERGE_GAP_S
    if block_dur > 0:
        gap_frames = int(DETECT_MERGE_GAP_S / block_dur)
        merged: list[list[int]] = []
        for d_start, d_end in detections:   # still in time order here
            if merged and (d_start - merged[-1][1]) <= gap_frames:
                merged[-1][1] = d_end
            else:
                merged.append([d_start, d_end])
        detections = [tuple(d) for d in merged]  # type: ignore[assignment]

    detections.sort(key=lambda s_e: det_norm[s_e[0]:s_e[1]+1].max(), reverse=True)


    pad = np.zeros(int(0.1 * rate), dtype=np.float32)  # Set to 0.0 for manual testing

    print(f"\njt9: {' '.join(JT9_BASE_ARGS)} <wav>")
    pad = np.zeros(int(0.1 * rate), dtype=np.float32)  # 0.1 seconds of padding, per manual test
    if not detections:
        print("  (no detections above threshold)")
        return
    for d_start, d_end in detections:
        t_start    = t2[d_start]
        t_end      = t2[d_end]
        max_metric = float(det_norm[d_start:d_end+1].max())
        duration   = (d_end - d_start + 1) * block_dur
        pre_post   = int(0.5 * rate)
        seg_start  = max(0, int(t_start * rate) - pre_post)
        seg_end    = min(samples.size, int(t_end * rate) + pre_post)
        segment    = samples[seg_start:seg_end]
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
            # Save to permanent file if valid decode, but skip if this file
            # is itself a saved detection (stem already contains _t<digits>)
            is_detection_file = bool(re.search(r'_t\d+\.\d+', wav_path.stem))
            if decoded and not is_detection_file:
                det_dir = wav_path.parent / 'detections'
                det_dir.mkdir(exist_ok=True)
                out_name = f"{wav_path.stem}_t{t_start:.2f}.wav"
                out_path = det_dir / out_name
                with wave.open(str(out_path), 'wb') as wf_out:
                    wf_out.setnchannels(1)
                    wf_out.setsampwidth(2)
                    wf_out.setframerate(rate)
                    wf_out.writeframes((padded * 32767).astype(np.int16).tobytes())
                print(f"Saved detection to: {out_path}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        print(f"{t_start:7.3f}  {duration:6.3f}  {max_metric:6.2f}  {decoded or 'no decode'}")


# ═══════════════════════════════════════════════════════════════════════════ #
#   Per-channel detection diagnostic                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def _compute_per_channel_metrics(
    samples: np.ndarray,
    rate: int,
) -> dict | None:
    """Run the channelizer + per-channel detection pipeline on complex IQ input.

    Mirrors the algorithm in radio_iq_gui/processing.py.  Returns None for
    mono (real) input or if the channelizer module is unavailable.

    Returns a dict with keys:
        time_s          (n_frames,)
        channel_hz      (N_CHANNELS,)
        pair_metric     (n_frames, N_CHANNELS) — dB above 25th-percentile baseline
        lo_peak         (n_frames, N_CHANNELS) — lo-window peak power (dB)
        hi_peak         (n_frames, N_CHANNELS) — hi-window peak power (dB)
        pct25_lin       (N_CHANNELS,)           — per-channel 25th-percentile linear baseline
        sq_freq         (_PDET_FFT_SIZE,)      — centered frequency axis (Hz)
        lo_mask / hi_mask (_PDET_FFT_SIZE,) bool
        peak_frame_pdb  (N_CHANNELS, _PDET_FFT_SIZE) — pdb_all at peak frame
        peak_frame_idx  int
    """
    if not np.iscomplexobj(samples):
        return None
    try:
        from radio_iq_gui.channelizer import (
            apply_channelizer,
            design_channelizer_filter,
            make_channelizer_state,
            N_CHANNELS,
            CHANNEL_SPACING_HZ,
            DECIMATE_FACTOR,
        )
    except ImportError:
        print("  per-channel analysis: radio_iq_gui.channelizer not available")
        return None

    lp_taps = design_channelizer_filter(rate)
    state   = make_channelizer_state(N_CHANNELS, lp_taps)
    ch_out  = apply_channelizer(
        samples.astype(np.complex64), state, lp_taps=lp_taps, sample_rate=rate,
    )  # (N_CHANNELS, n_ch_samples)

    ch_rate  = rate // DECIMATE_FACTOR
    fft_sz   = _PDET_FFT_SIZE
    hop      = _PDET_HOP
    window   = np.hanning(fft_sz)

    sq_freq = np.fft.fftshift(np.fft.fftfreq(fft_sz, 1.0 / ch_rate))
    lo_mask = (sq_freq >= -_PDET_SQ_TONE - _PDET_SQ_NTOL) & \
              (sq_freq <= -_PDET_SQ_TONE + _PDET_SQ_NTOL)
    hi_mask = (sq_freq >=  _PDET_SQ_TONE - _PDET_SQ_NTOL) & \
              (sq_freq <=  _PDET_SQ_TONE + _PDET_SQ_NTOL)

    n_ch_samples = ch_out.shape[1]
    n_frames     = max(0, (n_ch_samples - fft_sz) // hop + 1)
    if n_frames == 0:
        return None

    lo_list   = []
    hi_list   = []
    raw_list  = []   # linear (lo_lin + hi_lin) / 2 per frame — for percentile
    pdb_list  = []
    t_list    = []

    for i in range(n_frames):
        start    = i * hop
        ch_block = ch_out[:, start : start + fft_sz]                # (N_CH, fft_sz)
        sq       = ch_block ** 2
        X_sq     = np.fft.fft(sq * window[np.newaxis, :], axis=1)

        # dB spectrum for display in the right panels
        power_db = 10.0 * np.log10(np.abs(X_sq) / fft_sz + 1e-12)  # (N_CH, fft_sz)
        pdb_all  = np.fft.fftshift(power_db, axes=1)                # (N_CH, fft_sz)

        # Linear-domain peaks for WSJT-X-style 25th-percentile normalization
        power_lin = np.abs(X_sq) / fft_sz                           # (N_CH, fft_sz)
        plin_all  = np.fft.fftshift(power_lin, axes=1)              # (N_CH, fft_sz)
        lo_lin    = np.max(plin_all[:, lo_mask], axis=1)            # (N_CH,) linear
        hi_lin    = np.max(plin_all[:, hi_mask], axis=1)            # (N_CH,) linear
        raw_lin   = (lo_lin + hi_lin) / 2.0                         # (N_CH,) linear

        # dB peaks for display in ax_ch
        lo_peak   = np.max(pdb_all[:, lo_mask], axis=1)             # (N_CH,) dB
        hi_peak   = np.max(pdb_all[:, hi_mask], axis=1)             # (N_CH,) dB

        lo_list .append(lo_peak)
        hi_list .append(hi_peak)
        raw_list.append(raw_lin)
        pdb_list.append(pdb_all)
        t_list  .append((start + fft_sz / 2.0) / ch_rate)

    # WSJT-X xmed normalization: 25th percentile over all frames per channel.
    # Using the full file ensures quiet frames dominate even if pings are long.
    raw_arr    = np.stack(raw_list)                                  # (n_frames, N_CH)
    pct25_lin  = np.percentile(raw_arr, 25, axis=0)                 # (N_CH,) linear
    pair_metric = 10.0 * np.log10(
        raw_arr / np.maximum(pct25_lin, 1e-30)
    )                                                                # (n_frames, N_CH) dB

    peak_fi    = int(np.argmax(pair_metric.max(axis=1)))
    pdb_frames = np.stack(pdb_list)                                  # (n_frames, N_CH, fft_sz)

    return dict(
        time_s          = np.array(t_list),
        channel_hz      = np.arange(N_CHANNELS) * float(CHANNEL_SPACING_HZ),
        pair_metric     = pair_metric,
        lo_peak         = np.stack(lo_list),    # (n_frames, N_CH) dB — display only
        hi_peak         = np.stack(hi_list),    # (n_frames, N_CH) dB — display only
        pct25_lin       = pct25_lin,            # (N_CH,) linear 25th-percentile baseline
        sq_freq         = sq_freq,
        lo_mask         = lo_mask,
        hi_mask         = hi_mask,
        peak_frame_pdb  = pdb_frames[peak_fi],  # (N_CH, fft_sz) kept for compat
        peak_frame_idx  = peak_fi,
        pdb_all_frames  = pdb_frames,           # (n_frames, N_CH, fft_sz)
    )


def _load_manifest_pings(wav_path: Path) -> list:
    """Return the 'placements' list from <wav_stem>.json, or [] if absent."""
    import json as _json
    json_path = wav_path.with_suffix('.json')
    if not json_path.exists():
        return []
    try:
        data = _json.load(open(json_path))
        return sorted(data.get('placements', []), key=lambda p: p['delay_s'])
    except Exception as exc:
        print(f"  manifest load failed: {exc}")
        return []


def plot_channel_detection(
    name:        str,
    result:      dict,
    show_plots:  bool,
    output_path: Path,
    wav_path:    Path | None = None,
) -> None:
    """Interactive per-channel detection figure.

    Layout (75 % / 25 % width split):
      Left  — top ~3/7: pair_metric heatmap with vertical + horizontal cursors
              bot ~4/7: selected-channel lo/hi/noise time series (x shared)
      Right — 7 stacked squared-spectrum panels for selected ch ± 3

    If a companion JSON manifest exists alongside *wav_path*, left/right arrow
    keys cycle through signal placements and update all panels.
    """
    time_s         = result['time_s']
    channel_hz     = result['channel_hz']
    pair_metric    = result['pair_metric']       # (n_frames, N_CH) dB above 25th-pct
    lo_peak        = result['lo_peak']           # (n_frames, N_CH) dB — display
    hi_peak        = result['hi_peak']           # (n_frames, N_CH) dB — display
    pct25_lin      = result['pct25_lin']         # (N_CH,) linear 25th-percentile baseline
    pct25_dB       = 10.0 * np.log10(np.maximum(pct25_lin, 1e-30))  # (N_CH,) dB
    sq_freq        = result['sq_freq']
    lo_mask        = result['lo_mask']
    hi_mask        = result['hi_mask']
    pdb_all_frames = result['pdb_all_frames']    # (n_frames, N_CH, fft_sz)
    peak_fi        = result['peak_frame_idx']
    N_CH           = pair_metric.shape[1]
    ch_spacing     = float(channel_hz[1] - channel_hz[0]) if len(channel_hz) > 1 else 1000.0
    half_ch        = N_CH // 2

    # ── fftshift display arrays ───────────────────────────────────────────
    # Channel k is centred at k*1 kHz (0..47 kHz).  In complex IQ, channels
    # 24-47 are the negative-frequency half (-24..-1 kHz).  fftshift reorders
    # so display slot 0 = -24 kHz, slot 24 = 0 kHz, slot 47 = +23 kHz —
    # matching the -24/+24 kHz scale used by the spectrogram displays.
    #
    # Conversion: original channel k  →  display slot (k - half_ch) % N_CH
    #             display slot d       →  original channel (d + half_ch) % N_CH
    display_hz = (np.arange(N_CH) - half_ch) * ch_spacing   # -24000..23000 Hz
    pm_disp    = np.fft.fftshift(pair_metric, axes=1)        # (n_frames, N_CH)
    lo_disp    = np.fft.fftshift(lo_peak,    axes=1)
    hi_disp    = np.fft.fftshift(hi_peak,    axes=1)
    pct25_dB_d = np.fft.fftshift(pct25_dB)                  # (N_CH,)

    # ── Manifest pings ────────────────────────────────────────────────────
    pings   = _load_manifest_pings(wav_path) if wav_path is not None else []
    n_pings = len(pings)

    def _ch_for_ping(p):
        """Map ping center_hz to display slot index."""
        k = int(round(p['center_hz'] / ch_spacing)) % N_CH
        return int((k - half_ch) % N_CH)

    def _frame_for_ping(p):
        """Find the frame where pm_disp peaks for this ping's display slot and time window."""
        ch_d    = _ch_for_ping(p)
        t_start = p['delay_s']
        t_end   = p['delay_s'] + p.get('width_ms', 0) / 1000.0
        mask    = (time_s >= t_start) & (time_s <= t_end)
        if np.any(mask):
            idxs = np.where(mask)[0]
            return int(idxs[np.argmax(pm_disp[idxs, ch_d])])
        # Fallback: nearest frame to ping centre
        return int(np.argmin(np.abs(time_s - (t_start + (t_end - t_start) / 2.0))))

    # Initial state: use ping nearest to peak frame, else peak frame itself.
    # state['ch'] is always a DISPLAY slot index.
    if n_pings > 0:
        init_pi = int(np.argmin([abs(_frame_for_ping(p) - peak_fi) for p in pings]))
        init_fi = _frame_for_ping(pings[init_pi])
        init_ch = _ch_for_ping(pings[init_pi])
    else:
        init_pi = 0
        init_fi = peak_fi
        init_ch = int(np.argmax(pm_disp[peak_fi]))

    state = {'ping_idx': init_pi, 'frame_idx': init_fi, 'ch': init_ch}

    # ── Figure layout ─────────────────────────────────────────────────────
    # Outer 1×2 grid splits 75 % / 25 % horizontally.
    # Left sub-grid: heatmap (shorter) over channel-power (taller), with
    # enough hspace so the power x-axis labels don't crowd the heatmap.
    # Right sub-grid: 7 tightly-spaced spectrum panels.
    # NO colorbar on the heatmap — it would narrow ax_hm and break x-alignment.
    fig      = plt.figure(figsize=(16, 10))
    outer_gs = fig.add_gridspec(
        1, 2, width_ratios=[3, 1], wspace=0.28,
        left=0.07, right=0.97, top=0.88, bottom=0.07,
    )
    left_gs  = outer_gs[0, 0].subgridspec(2, 1, height_ratios=[2, 2.5], hspace=0.22)
    right_gs = outer_gs[0, 1].subgridspec(7, 1, hspace=0.05)

    ax_hm = fig.add_subplot(left_gs[0])
    ax_ch = fig.add_subplot(left_gs[1], sharex=ax_hm)
    sq_axes: list = []
    for i in range(7):
        kw = {} if i == 0 else {'sharex': sq_axes[0], 'sharey': sq_axes[0]}
        sq_axes.append(fig.add_subplot(right_gs[i], **kw))
    for a in sq_axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    # ── Heatmap (static image, dynamic cursors) ───────────────────────────
    vmax_pm  = max(_PDET_THRESH_DB * 3, float(pm_disp.max()) * 1.05)
    # Extent must place pixel *centres* at (time_s[i], display_hz[k]/1000),
    # so push edges out by half a pixel in each direction.
    dt   = float(time_s[1] - time_s[0]) / 2.0 if len(time_s) > 1 else 0.0
    half = ch_spacing / 2000.0   # half channel spacing in kHz
    hm_ext   = [
        float(time_s[0])   - dt,  float(time_s[-1])   + dt,
        float(display_hz[0])  / 1000.0 - half,
        float(display_hz[-1]) / 1000.0 + half,
    ]
    im = ax_hm.imshow(
        pm_disp.T, aspect='auto', origin='lower',
        extent=hm_ext, cmap='inferno', vmin=0.0, vmax=vmax_pm,
    )
    ax_hm.set_ylabel("Frequency (kHz)", fontsize=8)
    ax_hm.set_xlabel("Time (s)", fontsize=8)
    ax_hm.set_title("Pair Metric Heatmap", fontsize=9)

    vline_hm = ax_hm.axvline(float(time_s[init_fi]),               color='cyan', lw=1.2, ls='-')
    hline_hm = ax_hm.axhline(float(display_hz[init_ch]) / 1000.0,  color='cyan', lw=0.7, ls=':')

    # ── Channel power plot (dynamic lines) ───────────────────────────────
    lo_ln,  = ax_ch.plot(time_s, lo_disp[:, init_ch], color='tab:blue',  lw=1.0, label='lo peak')
    hi_ln,  = ax_ch.plot(time_s, hi_disp[:, init_ch], color='tab:red',   lw=1.0, label='hi peak')
    nf_ln,  = ax_ch.plot(time_s, np.full(len(time_s), pct25_dB_d[init_ch]),
                         color='tab:gray', lw=1.0, ls='--', label='25th-pct baseline')
    thr_ln, = ax_ch.plot(time_s, np.full(len(time_s), pct25_dB_d[init_ch] + _PDET_THRESH_DB),
                         color='orange', lw=0.8, ls='--', label=f'baseline+{_PDET_THRESH_DB} dB')
    vline_ch = ax_ch.axvline(float(time_s[init_fi]), color='cyan', lw=1.2, ls='-')
    ax_ch.set_xlabel("Time (s)", fontsize=8)
    ax_ch.set_ylabel("Power (dB)", fontsize=8)
    ax_ch.legend(fontsize=7, loc='upper right')
    ax_ch.grid(True, alpha=0.25)

    # ── Right panels: 7 stacked squared-spectrum subplots ─────────────────
    lo_f = sq_freq[lo_mask]
    hi_f = sq_freq[hi_mask]
    sq_lines: list = []
    sq_nf_lines: list = []
    sq_thr_lines: list = []
    sq_labels: list = []

    def _spectrum_at(fi, ch_d):
        """Return squared-spectrum for display slot ch_d at frame fi."""
        k = int((ch_d + half_ch) % N_CH)   # display slot → original channel
        return pdb_all_frames[fi, k].copy()

    for idx, offset in enumerate(range(-3, 4)):
        ax   = sq_axes[idx]
        ch   = (init_ch + offset) % N_CH   # wrap display slot
        y    = _spectrum_at(init_fi, ch)
        nf_v = float(pct25_dB_d[ch])

        ax.axvspan(lo_f.min() / 1000.0, lo_f.max() / 1000.0, alpha=0.12, color='blue')
        ax.axvspan(hi_f.min() / 1000.0, hi_f.max() / 1000.0, alpha=0.12, color='red')

        ln,  = ax.plot(sq_freq / 1000.0, y, color='tab:green', lw=0.9)
        nfl  = ax.axhline(nf_v,                      color='gray',   lw=0.8, ls='--')
        thrl = ax.axhline(nf_v + _PDET_THRESH_DB,    color='orange', lw=0.8, ls=':')

        sq_lines    .append(ln)
        sq_nf_lines .append(nfl)
        sq_thr_lines.append(thrl)

        ax.set_xlim(float(sq_freq[0]) / 1000.0, float(sq_freq[-1]) / 1000.0)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.20)

        lbl_str = f'{display_hz[ch]/1000:.0f} kHz'
        if offset == 0:
            lbl_str += ' ←'
        txt = ax.text(0.03, 0.93, lbl_str, transform=ax.transAxes,
                      fontsize=6, va='top', ha='left',
                      color='white' if offset == 0 else 'lightgray')
        sq_labels.append(txt)

    sq_axes[-1].set_xlabel("Frequency (kHz)", fontsize=7)
    sq_axes[3].set_ylabel("Power (dB)", fontsize=7)     # middle panel gets the y label
    sq_axes[0].set_title("Sq. spectrum\n(± 3 ch)", fontsize=7)

    # ── Suptitle / ping info ──────────────────────────────────────────────
    def _ping_str(idx):
        if n_pings == 0:
            return f"(no manifest)  frame={peak_fi}  ch={init_ch}"
        p = pings[idx]
        return (
            f"Ping {idx+1}/{n_pings}:  '{p['msg']}'   "
            f"fc={p['center_hz']/1000:.1f} kHz   "
            f"t={p['delay_s']:.2f} s   "
            f"SNR={p['snr_db']} dB   "
            f"width={p.get('width_ms','?')} ms"
            + (f"   [← →  to navigate]" if n_pings > 1 else "")
        )

    sup = fig.suptitle(
        f"Per-Channel Detection  –  {name}\n{_ping_str(state['ping_idx'])}",
        fontsize=9, y=0.975,
    )

    # ── Update all dynamic elements for current state ────────────────────
    def _update():
        fi = state['frame_idx']
        ch = state['ch']

        # Heatmap cursors
        vline_hm.set_xdata([float(time_s[fi])] * 2)
        hline_hm.set_ydata([float(display_hz[ch]) / 1000.0] * 2)
        vline_ch.set_xdata([float(time_s[fi])] * 2)

        # Channel power lines
        lo_ln .set_ydata(lo_disp[:, ch])
        hi_ln .set_ydata(hi_disp[:, ch])
        nf_ln .set_ydata(np.full(len(time_s), pct25_dB_d[ch]))
        thr_ln.set_ydata(np.full(len(time_s), pct25_dB_d[ch] + _PDET_THRESH_DB))
        ax_ch.set_title(
            f"{display_hz[ch]/1000:.0f} kHz  lo/hi peaks vs 25th-pct baseline",
            fontsize=8,
        )
        ax_ch.relim()
        ax_ch.autoscale_view(scalex=False, scaley=True)

        # Right panels — spectrum lines + noise/threshold markers
        y_vals = []
        for idx, offset in enumerate(range(-3, 4)):
            ach  = (ch + offset) % N_CH   # wrap display slot
            y    = _spectrum_at(fi, ach)
            nf_v = float(pct25_dB_d[ach])
            sq_lines    [idx].set_ydata(y)
            sq_nf_lines [idx].set_ydata([nf_v,                   nf_v])
            sq_thr_lines[idx].set_ydata([nf_v + _PDET_THRESH_DB, nf_v + _PDET_THRESH_DB])
            lbl = f'{display_hz[ach]/1000:.0f} kHz'
            if offset == 0:
                lbl += ' ←'
            sq_labels[idx].set_text(lbl)
            valid = y[np.isfinite(y)]
            if valid.size:
                y_vals.extend([float(valid.min()), float(valid.max())])

        # Common y-range for all 7 right panels
        if y_vals:
            sq_axes[0].set_ylim(min(y_vals) - 2, max(y_vals) + 5)

        # Title
        sup.set_text(
            f"Per-Channel Detection  –  {name}\n{_ping_str(state['ping_idx'])}"
        )
        fig.canvas.draw_idle()

    # ── Arrow-key ping navigation ─────────────────────────────────────────
    def _on_key(event):
        if event.key not in ('right', 'left') or n_pings == 0:
            return
        new_pi = (state['ping_idx'] + (1 if event.key == 'right' else -1)) % n_pings
        state['ping_idx']  = new_pi
        state['frame_idx'] = _frame_for_ping(pings[new_pi])
        state['ch']        = _ch_for_ping(pings[new_pi])
        _update()

    if show_plots:
        fig.canvas.mpl_connect('key_press_event', _on_key)

    _update()   # apply initial state

    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"Wrote: {output_path}")
    if not show_plots:
        plt.close(fig)


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

    # ── Row 2: detection heatmap (complex IQ) or squared spectrogram (mono) ─
    per_ch = _compute_per_channel_metrics(samples, rate)
    t0 = _tick('operational: per_channel_detection', t0)

    if per_ch is None:
        t2, f2, s2, det_norm = _compute_squared_spectrogram(samples, rate, fc_hz, ntol_hz)
        floor2    = _estimate_median(s2)
        sq_offset = float(np.median(floor1)) - float(np.median(floor2))
        s2        = s2     + sq_offset
        floor2    = floor2 + sq_offset
        t0 = _tick('analysis: median_floor_row2', t0)

    # ── Figure layout ──────────────────────────────────────────────────────
    gui_cmap = _gui_colormap()
    vmin0 = -110.0
    vmax0 =  -70.0

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
    f1_min_khz = 0.0
    f1_max_khz = 3.0
    ax00, ax01 = axes[0]
    img1 = ax00.imshow(
        s1.T, aspect='auto', origin='lower',
        extent=_extent(t1, f1), cmap=gui_cmap, vmin=vmin0, vmax=vmax0,
    )
    ax00.set_title(row1_title)
    ax00.set_xlabel("Time (s)")
    ax00.set_ylabel("Frequency (kHz)")
    ax00.set_ylim(f1_min_khz, f1_max_khz)
    ax00.xaxis.set_major_locator(MultipleLocator(1.0))

    # ── [0,1]  Median noise floor vs frequency ─────────────────────────────
    floor1_line, = ax01.plot(floor1, f1 / 1000.0, lw=1.2, color='tab:orange')
    ax01.set_title("Median Noise")
    ax01.set_xlabel("Level (dB)")
    ax01.set_ylabel("Frequency (kHz)")
    ax01.set_xlim(vmin0, vmax0)
    ax01.set_ylim(f1_min_khz, f1_max_khz)
    ax01.grid(True, alpha=0.25)

    # ── [1,0]  Detection heatmap or squared spectrogram ────────────────────
    ax10, ax11 = axes[1]
    floor2_line = None   # only set in the mono / squared-spectrogram path

    if per_ch is not None:
        _N_CH    = per_ch['pair_metric'].shape[1]
        _ch_sp   = float(per_ch['channel_hz'][1] - per_ch['channel_hz'][0]) if len(per_ch['channel_hz']) > 1 else 1000.0
        _half_ch = _N_CH // 2
        _disp_hz = (np.arange(_N_CH) - _half_ch) * _ch_sp
        pm_disp  = np.fft.fftshift(per_ch['pair_metric'], axes=1)
        time_pc  = per_ch['time_s']
        vmax_pm  = max(_PDET_THRESH_DB * 3, float(pm_disp.max()) * 1.05)
        _dt_pc   = float(time_pc[1] - time_pc[0]) / 2.0 if len(time_pc) > 1 else 0.0
        _half_f  = _ch_sp / 2000.0
        ext_pc   = [float(time_pc[0]) - _dt_pc, float(time_pc[-1]) + _dt_pc,
                    float(_disp_hz[0]) / 1000.0 - _half_f,
                    float(_disp_hz[-1]) / 1000.0 + _half_f]
        img2 = ax10.imshow(pm_disp.T, aspect='auto', origin='lower',
                           extent=ext_pc, cmap='inferno', vmin=0.0, vmax=vmax_pm)
        ax10.set_title(
            f"Detection Heatmap — pair metric (dB above 25th-pct baseline, "
            f"threshold = {_PDET_THRESH_DB} dB)"
        )
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("Frequency (kHz)")
        ax10.set_ylim(ext_pc[2], ext_pc[3])
        ax10.xaxis.set_major_locator(MultipleLocator(1.0))
        ax11.set_visible(False)
        t2_det       = time_pc
        det_norm_det = pm_disp.max(axis=1)
        det_thresh   = _PDET_THRESH_DB
        det_label    = "Pair metric (dB)"
    else:
        f2_min_khz = float(f2[0])  / 1000.0
        f2_max_khz = float(f2[-1]) / 1000.0
        img2 = ax10.imshow(
            s2.T, aspect='auto', origin='lower',
            extent=_extent(t2, f2), cmap=gui_cmap, vmin=vmin0, vmax=vmax0,
        )
        ax10.set_title(
            f"Squared-Signal Spectrogram  (fc={fc_hz:.0f} Hz, ntol=\u00b1{ntol_hz:.0f} Hz,"
            f" sq_offset={sq_offset:+.0f} dB)"
        )
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("Frequency (kHz)")
        ax10.set_ylim(f2_min_khz, f2_max_khz)
        ax10.xaxis.set_major_locator(MultipleLocator(1.0))
        # ── [1,1]  Squared-spectrum median noise floor vs frequency ──────────
        floor2_line, = ax11.plot(floor2, f2 / 1000.0, lw=1.2, color='tab:orange')
        ax11.set_title("Median Noise")
        ax11.set_xlabel("Level (dB)")
        ax11.set_ylabel("Frequency (kHz)")
        ax11.set_xlim(vmin0, vmax0)
        ax11.set_ylim(f2_min_khz, f2_max_khz)
        ax11.grid(True, alpha=0.25)
        t2_det       = t2
        det_norm_det = det_norm
        det_thresh   = DETECT_THRESH
        det_label    = "Norm. metric"

    # ── [2,0]  Detection metric vs time (x-axis aligned with [1,0]) ────────
    ax20, ax21 = axes[2]
    ax20.sharex(ax10)
    ax20.step(t2_det, det_norm_det, where='mid', color='tab:green', lw=1.0)
    ax20.axhline(det_thresh, color='red', lw=0.9, ls='--',
                 label=f'Threshold ({det_thresh})')
    ax20.fill_between(t2_det, det_norm_det, det_thresh,
                      where=(det_norm_det >= det_thresh),
                      color='red', alpha=0.25, label='Detections')
    ax20.set_xlabel("Time (s)")
    ax20.set_ylabel(det_label)
    ax20.set_ylim(bottom=0.0)
    ax20.legend(fontsize=8, loc='upper right')
    ax20.grid(True, alpha=0.25)

    # ── [2,1]  Hidden (no content needed alongside metric strip) ───────────
    ax21.set_visible(False)

    t0 = _tick('analysis: plot_layout_and_axes', t0)

    # ── Color-range controls ───────────────────────────────────────────────
    # Sliders generate many events during a drag and flood the renderer.
    # Instead use discrete scroll-wheel events (one redraw per click):
    #   scroll up/down          → shift min+max together  (brightness)
    #   Ctrl + scroll up/down   → expand/contract span    (contrast)
    # Current values are shown as text in the button bar.
    if show_plots:
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        btn_ax = fig.add_axes([0.01, 0.015, 0.10, 0.030])
        open_btn = Button(btn_ax, 'Open WAV\u2026', color='0.15', hovercolor='0.30')

        clim = [vmin0, vmax0]          # mutable so closures can update it
        SCROLL_STEP = 2.0              # dB per scroll click

        baseline_ax = fig.add_axes([0.18, 0.015, 0.28, 0.025])
        gain_ax     = fig.add_axes([0.60, 0.015, 0.28, 0.025])
        baseline_slider = Slider(baseline_ax, 'Min (dB)', -200.0, 0.0,
                                 valinit=vmin0, valstep=1.0)
        gain_slider     = Slider(gain_ax,     'Span (dB)',  5.0, 120.0,
                                 valinit=vmax0 - vmin0, valstep=1.0)

        def _apply_clim() -> None:
            img1.set_clim(clim[0], clim[1])
            ax01.set_xlim(clim[0], clim[1])
            if per_ch is None:
                img2.set_clim(clim[0], clim[1])
                ax11.set_xlim(clim[0], clim[1])
            fig.canvas.draw_idle()

        def _on_slider(_val) -> None:
            clim[0] = float(baseline_slider.val)
            clim[1] = clim[0] + float(gain_slider.val)
            _apply_clim()

        baseline_slider.on_changed(_on_slider)
        gain_slider.on_changed(_on_slider)

        def _on_scroll(event) -> None:
            _scroll_axes = (ax00, ax01) if per_ch is not None else (ax00, ax01, ax10, ax11)
            if event.inaxes not in _scroll_axes:
                return
            delta = SCROLL_STEP if event.button == 'up' else -SCROLL_STEP
            if event.key == 'control':
                # Ctrl+scroll: expand/contract span via gain slider
                gain_slider.set_val(max(5.0, gain_slider.val + 2 * delta))
            else:
                # Plain scroll: shift baseline slider (brightness)
                baseline_slider.set_val(
                    float(np.clip(baseline_slider.val + delta, -200.0, 0.0))
                )

        fig.canvas.mpl_connect('scroll_event', _on_scroll)
        _apply_clim()   # set initial clim

        def _full_redraw() -> None:
            _apply_clim()

        dialog_dir = [output_path.parent]

        def _reload(new_wav: Path) -> None:
            new_samples, new_rate = read_wav(new_wav)

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
            ax00.set_ylim(0.0, 3.0)
            ax01.set_ylim(0.0, 3.0)
            floor1_line.set_xdata(floor1n)
            floor1_line.set_ydata(f1n / 1000.0)

            # Row 2 — detection heatmap or squared spectrogram
            per_ch_n = _compute_per_channel_metrics(new_samples, new_rate)
            if per_ch_n is not None:
                _N_CH_n    = per_ch_n['pair_metric'].shape[1]
                _ch_sp_n   = float(per_ch_n['channel_hz'][1] - per_ch_n['channel_hz'][0]) if len(per_ch_n['channel_hz']) > 1 else 1000.0
                _half_ch_n = _N_CH_n // 2
                _disp_hz_n = (np.arange(_N_CH_n) - _half_ch_n) * _ch_sp_n
                pm_disp_n  = np.fft.fftshift(per_ch_n['pair_metric'], axes=1)
                time_pc_n  = per_ch_n['time_s']
                vmax_pm_n  = max(_PDET_THRESH_DB * 3, float(pm_disp_n.max()) * 1.05)
                _dt_pc_n   = float(time_pc_n[1] - time_pc_n[0]) / 2.0 if len(time_pc_n) > 1 else 0.0
                _half_f_n  = _ch_sp_n / 2000.0
                ext_pc_n   = [float(time_pc_n[0]) - _dt_pc_n, float(time_pc_n[-1]) + _dt_pc_n,
                               float(_disp_hz_n[0]) / 1000.0 - _half_f_n,
                               float(_disp_hz_n[-1]) / 1000.0 + _half_f_n]
                img2.set_data(pm_disp_n.T)
                img2.set_extent(ext_pc_n)
                img2.set_clim(0.0, vmax_pm_n)
                ax10.set_ylim(ext_pc_n[2], ext_pc_n[3])
                ax10.set_title(
                    f"Detection Heatmap — pair metric (dB above 25th-pct baseline, "
                    f"threshold = {_PDET_THRESH_DB} dB)"
                )
                t2_det_n       = time_pc_n
                det_norm_det_n = pm_disp_n.max(axis=1)
                det_thresh_n   = _PDET_THRESH_DB
                det_label_n    = "Pair metric (dB)"
            else:
                t2n, f2n, s2n, det_norm_n = _compute_squared_spectrogram(
                    new_samples, new_rate, fc_hz, ntol_hz
                )
                floor2n = _estimate_median(s2n)
                sq_offset_n = float(np.median(floor1n)) - float(np.median(floor2n))
                s2n     = s2n     + sq_offset_n
                floor2n = floor2n + sq_offset_n
                ax10.set_title(
                    f"Squared-Signal Spectrogram  (fc={fc_hz:.0f} Hz, ntol=\u00b1{ntol_hz:.0f} Hz,"
                    f" sq_offset={sq_offset_n:+.0f} dB)"
                )
                img2.set_data(s2n.T)
                img2.set_extent(_extent(t2n, f2n))
                f2n_min_khz = float(f2n[0])  / 1000.0
                f2n_max_khz = float(f2n[-1]) / 1000.0
                ax10.set_ylim(f2n_min_khz, f2n_max_khz)
                if floor2_line is not None:
                    floor2_line.set_xdata(floor2n)
                    floor2_line.set_ydata(f2n / 1000.0)
                    ax11.set_ylim(f2n_min_khz, f2n_max_khz)
                    ax11.relim()
                    ax11.autoscale_view(scalex=True, scaley=False)
                t2_det_n       = t2n
                det_norm_det_n = det_norm_n
                det_thresh_n   = DETECT_THRESH
                det_label_n    = "Norm. metric"

            # Detection metric strip — clear and replot
            ax20.cla()
            ax20.step(t2_det_n, det_norm_det_n, where='mid', color='tab:green', lw=1.0)
            ax20.axhline(det_thresh_n, color='red', lw=0.9, ls='--',
                         label=f'Threshold ({det_thresh_n})')
            ax20.fill_between(t2_det_n, det_norm_det_n, det_thresh_n,
                              where=(det_norm_det_n >= det_thresh_n),
                              color='red', alpha=0.25, label='Detections')
            ax20.set_xlabel("Time (s)")
            ax20.set_ylabel(det_label_n)
            ax20.set_ylim(bottom=0.0)
            ax20.xaxis.set_major_locator(MultipleLocator(1.0))
            ax20.legend(fontsize=8, loc='upper right')
            ax20.grid(True, alpha=0.25)

            new_out = new_wav.parent / (new_wav.stem + '_analysis.png')
            fig.suptitle(f"MSK144 Analysis \u2013 {new_wav.name}", fontsize=13)
            _full_redraw()   # rebuilds blit background for the new file
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

        try:
            _build_file_picker(
                fig.canvas.manager.window,
                dialog_dir,
                _reload,
                initial_filename=name,
            )
        except Exception:
            pass  # non-TkAgg backend: file picker unavailable, Open WAV button still works
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

        samples, rate = read_wav(wav_path)
        show = args.show and not args.profile
        timings: dict | None = {} if args.profile else None

        run_detections(wav_path, samples, rate, args.fc_hz, args.ntol_hz)

        # Per-channel detection diagnostic (complex IQ only).
        # Create the figure before plot_analysis so both windows appear together
        # when plt.show() is called.
        ch_result = _compute_per_channel_metrics(samples, rate)
        if ch_result is not None:
            ch_out_path = wav_path.parent / (wav_path.stem + '_ch_detection.png')
            plot_channel_detection(wav_path.name, ch_result, show, ch_out_path,
                                   wav_path=wav_path)

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
