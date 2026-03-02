#!/usr/bin/env python3
"""Generate a combined 48 kHz complex MSK144 test stream from WAV burst files.

This tool:
- Reads mono WAV bursts (default from ./MSK144)
- Resamples each file from source sample rate (expected 12 kHz) to 48 kHz
- Frequency-shifts each burst from a fixed source center (1500 Hz)
  to user-selected target centers
- Sums all shifted bursts into one complex stream
- Writes complex output as .npy and optional stereo I/Q WAV
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider


MEDIAN_UPDATE_STEP_DB = 0.0625
MEDIAN_EDGE_EXCLUDE_FRAMES = 0
MEDIAN_REPLAY_PASSES = 8
MEDIAN_FREQ_WINDOW_BINS = 7
MEDIAN_INVALID_EDGE_GUARD_DB = -170.0
MEDIAN_INVALID_EDGE_MARGIN_DB = 6.0
DIAGNOSTIC_NFFT = 512
DIAGNOSTIC_CHUNK_SAMPLES = 504
EMBEDDED_POWER_MIN_SPAN_DB = 1.5
SOURCE_CENTER_HZ = 1500.0


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
        # Try float32 first; if not finite, fall back to int32 scaling.
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


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample using linear interpolation (dependency-free fallback)."""
    if src_rate == dst_rate:
        return samples.astype(np.float32)

    if samples.size == 0:
        return np.array([], dtype=np.float32)

    out_len = int(round(samples.size * (dst_rate / src_rate)))
    old_x = np.arange(samples.size, dtype=np.float64)
    new_x = np.linspace(0, samples.size - 1, out_len, dtype=np.float64)
    out = np.interp(new_x, old_x, samples.astype(np.float64))
    return out.astype(np.float32)


def _real_to_analytic(samples: np.ndarray) -> np.ndarray:
    """Convert a real signal to its analytic (single-sideband) complex form via FFT.

    Zeroing the negative-frequency bins removes the mirror-image component so
    that subsequent frequency shifting places only one copy of the signal in
    the complex passband instead of creating an in-band image at the reflected
    frequency.
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
    # Negative-frequency bins (n//2+1 onward) stay zero → image suppressed
    return np.fft.ifft(X * H).astype(np.complex64)


def freq_shift_real_to_complex(samples: np.ndarray, shift_hz: float, sample_rate: int) -> np.ndarray:
    """Frequency-shift a real signal into a complex passband without mirror images.

    Converts the real input to its analytic form first (suppressing the
    negative-frequency image), then applies the complex LO shift.  Without
    this step, a real signal at ±f_tone would produce two copies in the
    complex band at (shift ± f_tone), polluting the passband with a ghost
    signal at (shift - f_tone) as well as the intended (shift + f_tone).
    """
    analytic = _real_to_analytic(samples)
    n = np.arange(analytic.size, dtype=np.float64)
    lo = np.exp(1j * 2.0 * np.pi * shift_hz * n / sample_rate).astype(np.complex64)
    return analytic * lo


def write_iq_wav(path: Path, iq: np.ndarray, sample_rate: int) -> None:
    """Write complex IQ as stereo int16 WAV (left=I, right=Q)."""
    i = np.real(iq)
    q = np.imag(iq)
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = i
    interleaved[1::2] = q

    peak = float(np.max(np.abs(interleaved))) if interleaved.size else 1.0
    scale = 0.95 / peak if peak > 0 else 1.0
    pcm = np.clip(interleaved * scale, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def _gui_like_colormap() -> LinearSegmentedColormap:
    """Return the same color table used by the Flex GUI spectrogram."""
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
    color_points = [
        (pos, (r / 255.0, g / 255.0, b / 255.0))
        for pos, (r, g, b) in zip(positions, colors)
    ]
    return LinearSegmentedColormap.from_list("flex_gui_spectrogram", color_points)


def _compute_spectrogram_db(
    samples: np.ndarray,
    sample_rate: int,
    nfft: int = DIAGNOSTIC_NFFT,
    chunk_samples: int = DIAGNOSTIC_CHUNK_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrogram-like PSD matrix in dB/Hz from real-time style chunk loop.

    Samples are consumed in `chunk_samples` blocks (default 504) to emulate
    receive cadence. Each chunk is zero-padded/truncated to `nfft` for FFT.
    Complex (IQ) input uses the full bilateral FFT; real input uses rfft.
    """
    is_complex = np.iscomplexobj(samples)

    if samples.size == 0:
        nfft = max(256, int(nfft))
        if is_complex:
            freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
        else:
            freq_hz = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        return np.array([0.0]), freq_hz, np.full((1, freq_hz.size), -180.0, dtype=np.float64)

    nfft = max(256, min(int(nfft), samples.size))
    chunk_samples = max(1, int(chunk_samples))

    window = np.hanning(nfft).astype(np.float64)
    win_power = np.sum(window ** 2)
    frames = []
    times = []
    elapsed_s = 0.0

    for start in range(0, samples.size, chunk_samples):
        chunk = samples[start:start + chunk_samples]
        if chunk.size == 0:
            continue

        if is_complex:
            block = np.zeros(nfft, dtype=np.complex128)
            copy_len = min(chunk.size, nfft)
            block[:copy_len] = chunk[:copy_len].astype(np.complex128)
            spec = np.fft.fftshift(np.fft.fft(block * window))
        else:
            block = np.zeros(nfft, dtype=np.float64)
            copy_len = min(chunk.size, nfft)
            block[:copy_len] = chunk[:copy_len].astype(np.float64)
            spec = np.fft.rfft(block * window)

        psd = (np.abs(spec) ** 2) / (sample_rate * win_power)
        frames.append(psd)
        elapsed_s += chunk.size / sample_rate
        times.append(elapsed_s)

    if not frames:
        if is_complex:
            block = np.zeros(nfft, dtype=np.complex128)
            copy_len = min(samples.size, nfft)
            block[:copy_len] = samples[:copy_len].astype(np.complex128)
            spec = np.fft.fftshift(np.fft.fft(block * window))
        else:
            block = np.zeros(nfft, dtype=np.float64)
            copy_len = min(samples.size, nfft)
            block[:copy_len] = samples[:copy_len].astype(np.float64)
            spec = np.fft.rfft(block * window)
        psd = (np.abs(spec) ** 2) / (sample_rate * win_power)
        frames = [psd]
        times = [copy_len / sample_rate]

    spec_lin = np.stack(frames, axis=0)
    spec_db = 10.0 * np.log10(spec_lin + 1e-20)
    if is_complex:
        freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    else:
        freq_hz = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    return np.array(times, dtype=np.float64), freq_hz, spec_db


def _median_update_mask(
    num_frames: int,
    edge_exclude_frames: int = MEDIAN_EDGE_EXCLUDE_FRAMES,
    spec_db: np.ndarray | None = None,
    invalid_edge_guard_db: float = MEDIAN_INVALID_EDGE_GUARD_DB,
    invalid_edge_margin_db: float = MEDIAN_INVALID_EDGE_MARGIN_DB,
) -> np.ndarray:
    """Return frame mask for median updates, excluding unreliable start/end frames.

    In addition to fixed edge exclusion, this detects contiguous leading/trailing
    frames that sit near the PSD numerical floor (e.g., zero-filled WAV padding)
    and excludes them from median updates.
    """
    if num_frames <= 0:
        return np.zeros(0, dtype=bool)

    edge = max(0, int(edge_exclude_frames))
    if (2 * edge) >= num_frames:
        return np.ones(num_frames, dtype=bool)

    mask = np.ones(num_frames, dtype=bool)
    if edge > 0:
        mask[:edge] = False
        mask[-edge:] = False

    if spec_db is not None and spec_db.size and spec_db.ndim == 2 and spec_db.shape[0] == num_frames:
        frame_peak_db = np.max(spec_db.astype(np.float64, copy=False), axis=1)
        floor_db = float(np.min(frame_peak_db))
        if floor_db <= float(invalid_edge_guard_db):
            invalid_thresh_db = floor_db + float(invalid_edge_margin_db)
            edge_invalid = frame_peak_db <= invalid_thresh_db

            lead = 0
            while lead < num_frames and edge_invalid[lead]:
                lead += 1

            trail = 0
            while trail < (num_frames - lead) and edge_invalid[num_frames - 1 - trail]:
                trail += 1

            if lead > 0:
                mask[:lead] = False
            if trail > 0:
                mask[-trail:] = False

    return mask


def _initial_median_estimate_db(spec_db: np.ndarray, update_mask: np.ndarray) -> np.ndarray:
    """Pick initial median estimate from the first frame allowed by the update mask."""
    if spec_db.size == 0:
        return np.array([], dtype=np.float64)

    valid_idx = np.flatnonzero(update_mask)
    first_idx = int(valid_idx[0]) if valid_idx.size else 0
    return spec_db[first_idx].astype(np.float64).copy()


def _median_smooth_bins_db(values_db: np.ndarray, window_bins: int = MEDIAN_FREQ_WINDOW_BINS) -> np.ndarray:
    """Median-smooth one spectrum across neighboring frequency bins."""
    if values_db.size == 0:
        return values_db

    window = max(1, int(window_bins))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values_db.copy()

    half = window // 2
    padded = np.pad(values_db, (half, half), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    return np.median(windows, axis=1).astype(np.float64, copy=False)


def _update_median_estimate_db(
    estimate_db: np.ndarray,
    frame_db: np.ndarray,
    step_db: float = MEDIAN_UPDATE_STEP_DB,
    freq_window_bins: int = MEDIAN_FREQ_WINDOW_BINS,
) -> np.ndarray:
    """Update a per-bin streaming median estimate in dB with constant cost per frame."""
    if estimate_db.size == 0:
        return estimate_db

    step = float(step_db)
    if step <= 0.0:
        return estimate_db

    smoothed_frame_db = _median_smooth_bins_db(frame_db, window_bins=freq_window_bins)
    above = smoothed_frame_db > estimate_db
    estimate_db[above] += step
    estimate_db[~above] -= step
    estimate_db[:] = _median_smooth_bins_db(estimate_db, window_bins=freq_window_bins)
    return estimate_db


def _estimate_median_energy_bins_db(
    spec_db: np.ndarray,
    step_db: float = MEDIAN_UPDATE_STEP_DB,
    edge_exclude_frames: int = MEDIAN_EDGE_EXCLUDE_FRAMES,
    replay_passes: int = MEDIAN_REPLAY_PASSES,
    freq_window_bins: int = MEDIAN_FREQ_WINDOW_BINS,
) -> np.ndarray:
    """Return per-bin streaming median estimate (dB), replaying data to emulate continuity."""
    if spec_db.size == 0:
        return np.array([], dtype=np.float64)

    spec_db = spec_db.astype(np.float64, copy=False)
    update_mask = _median_update_mask(
        spec_db.shape[0],
        edge_exclude_frames=edge_exclude_frames,
        spec_db=spec_db,
    )
    estimate = _initial_median_estimate_db(spec_db, update_mask)

    passes = max(1, int(replay_passes))
    for _ in range(passes):
        for frame_idx, frame_db in enumerate(spec_db):
            if not update_mask[frame_idx]:
                continue
            _update_median_estimate_db(
                estimate,
                frame_db,
                step_db=step_db,
                freq_window_bins=freq_window_bins,
            )

    return estimate


def _median_energy_series_db(
    spec_db: np.ndarray,
    step_db: float = MEDIAN_UPDATE_STEP_DB,
    edge_exclude_frames: int = MEDIAN_EDGE_EXCLUDE_FRAMES,
    freq_window_bins: int = MEDIAN_FREQ_WINDOW_BINS,
) -> np.ndarray:
    """Return running per-bin streaming median estimate (dB) for each time frame."""
    if spec_db.size == 0:
        return np.zeros_like(spec_db, dtype=np.float64)

    spec_db = spec_db.astype(np.float64, copy=False)
    update_mask = _median_update_mask(
        spec_db.shape[0],
        edge_exclude_frames=edge_exclude_frames,
        spec_db=spec_db,
    )
    estimate = _initial_median_estimate_db(spec_db, update_mask)

    out = np.empty_like(spec_db, dtype=np.float64)
    for idx, frame_db in enumerate(spec_db):
        if update_mask[idx]:
            _update_median_estimate_db(
                estimate,
                frame_db,
                step_db=step_db,
                freq_window_bins=freq_window_bins,
            )
        out[idx] = estimate

    return out


def _flatten_spectrum_by_median_noise_floor(spec_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten spectrum across frequency using median-noise-floor-derived correction.

    Returns:
        spec_db_flat: Per-frame flattened spectrum in dB.
        median_floor_flat_db: Flattened median floor curve in dB (peaks at reference).
        correction_db: Per-bin additive correction applied to each frame.
    """
    if spec_db.size == 0:
        empty = np.array([], dtype=np.float64)
        return spec_db.astype(np.float64, copy=False), empty, empty

    spec_db_f64 = spec_db.astype(np.float64, copy=False)
    median_floor_db = _estimate_median_energy_bins_db(spec_db_f64)
    if median_floor_db.size == 0:
        empty = np.array([], dtype=np.float64)
        return spec_db_f64.copy(), empty, empty

    peak_db = float(np.max(median_floor_db))
    correction_db = peak_db - median_floor_db
    spec_db_flat = spec_db_f64 + correction_db[np.newaxis, :]
    median_floor_flat_db = median_floor_db + correction_db
    return spec_db_flat, median_floor_flat_db, correction_db


def _embedded_power_trace(time_s: np.ndarray, spec_db: np.ndarray, freq_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map frame power to a low-frequency band so it can be overlaid on the periodogram image."""
    if spec_db.size == 0 or time_s.size == 0 or freq_hz.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    frame_power = np.median(spec_db, axis=1)

    # Noise-referenced linear mapping with no clipping/compression.
    # This intentionally allows excursions so algorithm tuning can inspect behavior.
    p_noise = float(np.percentile(frame_power, 10))
    p_ref = float(np.percentile(frame_power, 99.5))
    span = max(p_ref - p_noise, EMBEDDED_POWER_MIN_SPAN_DB)
    norm = (frame_power - p_noise) / span

    f_min = float(freq_hz[0])
    f_max = float(freq_hz[-1])
    band = 0.10 * (f_max - f_min)
    y = f_min + norm * band
    return time_s, y


def _apply_lp_taper(
    spec_db: np.ndarray,
    freq_hz: np.ndarray,
    cutoff_hz: float,
) -> np.ndarray:
    """Apply a smooth cosine roll-off taper above cutoff_hz.

    Passband   |f| <= cutoff_hz         : unchanged.
    Transition cutoff_hz < |f| <= 2*cutoff_hz : cosine taper 0 → -60 dB.
    Stopband   |f| > 2*cutoff_hz        : set to -200 dB.
    """
    lo = float(cutoff_hz)
    hi = 2.0 * lo
    out = spec_db.astype(np.float64, copy=True)
    f_abs = np.abs(freq_hz)
    trans = (f_abs > lo) & (f_abs <= hi)
    if np.any(trans):
        t = (f_abs[trans] - lo) / (hi - lo)
        gain_db = -60.0 * (0.5 - 0.5 * np.cos(np.pi * t))
        out[:, trans] += gain_db[np.newaxis, :]
    stop = f_abs > hi
    if np.any(stop):
        out[:, stop] = -200.0
    return out


def plot_wav_diagnostics(
    all_data: list[tuple[str, np.ndarray, int]],
    output_path: Path,
    show_plots: bool,
    flatten_spectrum: bool = True,
    combined_output_rate: int | None = None,
    combined_target_centers_hz: list[float] | None = None,
    combined_lp_cutoff_hz: float = 3000.0,
):
    """Create GUI-style diagnostics: periodogram+embedded-power (left), noise-floor (right)."""
    if not all_data:
        return

    rows = len(all_data)
    has_combined = (
        combined_output_rate is not None
        and combined_target_centers_hz is not None
        and len(combined_target_centers_hz) == len(all_data)
    )
    total_rows = rows + (1 if has_combined else 0)
    gui_cmap = _gui_like_colormap()
    baseline_default = -110.0
    gain_default = 40.0
    min_level_default = baseline_default
    max_level_default = baseline_default + gain_default

    fig, axes = plt.subplots(
        total_rows,
        2,
        figsize=(14, 3.3 * total_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.0, 0.6]},
    )
    fig.suptitle("MSK144 WAV Diagnostics", fontsize=14)
    periodogram_images = []
    noise_axes = []
    noise_lines = []
    power_lines = []
    periodogram_buffers = []
    plot_datasets = []

    for row, (name, samples, sample_rate) in enumerate(all_data):
        t_s, freq_hz, spec_db = _compute_spectrogram_db(
            samples,
            sample_rate,
            nfft=DIAGNOSTIC_NFFT,
            chunk_samples=DIAGNOSTIC_CHUNK_SAMPLES,
        )
        if flatten_spectrum:
            spec_db_plot, noise_floor_db, freq_flatten_correction_db = _flatten_spectrum_by_median_noise_floor(spec_db)
        else:
            spec_db_plot = spec_db.astype(np.float64, copy=False)
            noise_floor_db = _estimate_median_energy_bins_db(spec_db_plot)
            freq_flatten_correction_db = np.zeros(spec_db_plot.shape[1], dtype=np.float64)
        median_update_mask = _median_update_mask(spec_db.shape[0], spec_db=spec_db)
        median_estimate_db = _initial_median_estimate_db(spec_db_plot, median_update_mask)
        if show_plots and median_estimate_db.size and noise_floor_db.size:
            median_estimate_db[:] = noise_floor_db
        t_overlay, y_overlay_hz = _embedded_power_trace(t_s, spec_db_plot, freq_hz)

        ax_periodogram = axes[row][0]
        ax_noise = axes[row][1]

        t_min = float(t_s[0]) if t_s.size else 0.0
        t_max = float(t_s[-1]) if t_s.size else 1.0
        if t_max <= t_min:
            t_max = t_min + 1e-3
        f_min = float(freq_hz[0]) if freq_hz.size else 0.0
        f_max = float(freq_hz[-1]) if freq_hz.size else sample_rate / 2.0

        image = ax_periodogram.imshow(
            spec_db_plot.T,
            aspect='auto',
            origin='lower',
            extent=[t_min, t_max, f_min / 1000.0, f_max / 1000.0],
            cmap=gui_cmap,
            vmin=min_level_default,
            vmax=max_level_default,
        )
        periodogram_images.append(image)
        if show_plots:
            buffer = np.full_like(spec_db.T, min_level_default, dtype=np.float64)
            image.set_data(buffer)
            periodogram_buffers.append(buffer)
            power_line, = ax_periodogram.plot([], [], color='red', linewidth=1.5)
        else:
            periodogram_buffers.append(None)
            if t_overlay.size and y_overlay_hz.size:
                power_line, = ax_periodogram.plot(t_overlay, y_overlay_hz / 1000.0, color='red', linewidth=1.5)
            else:
                power_line, = ax_periodogram.plot([], [], color='red', linewidth=1.5)
        power_lines.append(power_line)
        ax_periodogram.set_title(f"{name} - Periodogram")
        ax_periodogram.set_xlabel("Time (s)")
        ax_periodogram.set_ylabel("Frequency (kHz)")
        ax_periodogram.set_ylim(0.0, sample_rate / 2000.0)

        if show_plots:
            noise_x = median_estimate_db
        else:
            noise_x = noise_floor_db
        noise_line, = ax_noise.plot(noise_x, freq_hz / 1000.0, lw=1.2, color='tab:orange')
        ax_noise.set_title("Median Power")
        ax_noise.set_xlabel("Level (dB)")
        ax_noise.set_ylabel("Frequency (kHz)")
        ax_noise.set_xlim(min_level_default, max_level_default)
        ax_noise.set_ylim(0.0, sample_rate / 2000.0)
        ax_noise.grid(True, alpha=0.25)
        noise_axes.append(ax_noise)
        noise_lines.append(noise_line)

        plot_datasets.append(
            {
                "name": name,
                "time_s": t_s,
                "freq_hz": freq_hz,
                "spec_db": spec_db_plot,
                "spec_db_raw": spec_db,
                "median_update_mask": median_update_mask,
                "median_estimate_db": median_estimate_db,
                "freq_flatten_correction_db": freq_flatten_correction_db,
                "overlay_t": t_overlay,
                "overlay_y_khz": y_overlay_hz / 1000.0,
            }
        )

    if has_combined:
        # Build combined bilateral spectrogram from individual flattened, LP-filtered spectra.
        # Each source's passband (0..cutoff_hz) is frequency-shifted to its target center
        # in the combined 48 kHz bilateral grid, then summed as linear power.
        n_comb_bins = DIAGNOSTIC_NFFT
        freq_hz_comb = np.fft.fftshift(np.fft.fftfreq(n_comb_bins, 1.0 / combined_output_rate))
        max_frames = max(d["spec_db"].shape[0] for d in plot_datasets)
        power_lin_combined = np.zeros((max_frames, n_comb_bins), dtype=np.float64)

        for data, target_hz in zip(plot_datasets, combined_target_centers_hz):
            spec_db_lp = _apply_lp_taper(data["spec_db"], data["freq_hz"], combined_lp_cutoff_hz)
            power_lin_src = 10.0 ** (spec_db_lp / 10.0)
            shift_hz = float(target_hz) - SOURCE_CENTER_HZ
            freq_hz_shifted = data["freq_hz"] + shift_hz
            n_src = data["spec_db"].shape[0]
            for t in range(n_src):
                power_lin_combined[t] += np.interp(
                    freq_hz_comb, freq_hz_shifted, power_lin_src[t], left=0.0, right=0.0,
                )

        spec_db_combined = 10.0 * np.log10(np.maximum(power_lin_combined, 1e-30))

        # Align noise floor in signal-band bins to match individual rows.
        # Signal band for each source: [shift_hz, shift_hz + cutoff_hz] in combined space.
        signal_band_mask = np.zeros(n_comb_bins, dtype=bool)
        for target_hz in combined_target_centers_hz:
            f_lo = float(target_hz) - SOURCE_CENTER_HZ
            f_hi = f_lo + combined_lp_cutoff_hz
            signal_band_mask |= (freq_hz_comb >= min(f_lo, f_hi)) & (freq_hz_comb <= max(f_lo, f_hi))

        if np.any(signal_band_mask):
            ref_noise_db = float(np.mean([np.mean(d["median_estimate_db"]) for d in plot_datasets]))
            combined_noise_db = float(np.percentile(spec_db_combined[:, signal_band_mask], 10))
            spec_db_combined += ref_noise_db - combined_noise_db

        # Use time axis from the source dataset with the most frames.
        t_s = max(plot_datasets, key=lambda d: d["spec_db"].shape[0])["time_s"]
        freq_hz = freq_hz_comb

        noise_floor_db = _estimate_median_energy_bins_db(spec_db_combined)
        median_update_mask = _median_update_mask(spec_db_combined.shape[0], spec_db=spec_db_combined)
        median_estimate_db = _initial_median_estimate_db(spec_db_combined, median_update_mask)
        if show_plots and median_estimate_db.size and noise_floor_db.size:
            median_estimate_db[:] = noise_floor_db
        t_overlay, y_overlay_hz = _embedded_power_trace(t_s, spec_db_combined, freq_hz)

        ax_periodogram = axes[rows][0]
        ax_noise = axes[rows][1]

        t_min = float(t_s[0]) if t_s.size else 0.0
        t_max = float(t_s[-1]) if t_s.size else 1.0
        if t_max <= t_min:
            t_max = t_min + 1e-3
        f_min = float(freq_hz[0]) if freq_hz.size else -combined_output_rate / 2.0
        f_max = float(freq_hz[-1]) if freq_hz.size else combined_output_rate / 2.0

        image = ax_periodogram.imshow(
            spec_db_combined.T,
            aspect='auto',
            origin='lower',
            extent=[t_min, t_max, f_min / 1000.0, f_max / 1000.0],
            cmap=gui_cmap,
            vmin=min_level_default,
            vmax=max_level_default,
        )
        periodogram_images.append(image)
        if show_plots:
            buffer = np.full_like(spec_db_combined.T, min_level_default, dtype=np.float64)
            image.set_data(buffer)
            periodogram_buffers.append(buffer)
            power_line, = ax_periodogram.plot([], [], color='red', linewidth=1.5)
        else:
            periodogram_buffers.append(None)
            if t_overlay.size and y_overlay_hz.size:
                power_line, = ax_periodogram.plot(t_overlay, y_overlay_hz / 1000.0, color='red', linewidth=1.5)
            else:
                power_line, = ax_periodogram.plot([], [], color='red', linewidth=1.5)
        power_lines.append(power_line)
        ax_periodogram.set_title("Combined (flattened + LP filtered) – Periodogram")
        ax_periodogram.set_xlabel("Time (s)")
        ax_periodogram.set_ylabel("Frequency (kHz)")
        ax_periodogram.set_ylim(f_min / 1000.0, f_max / 1000.0)

        if show_plots:
            noise_x = median_estimate_db
        else:
            noise_x = noise_floor_db
        noise_line, = ax_noise.plot(noise_x, freq_hz / 1000.0, lw=1.2, color='tab:orange')
        ax_noise.set_title("Median Power")
        ax_noise.set_xlabel("Level (dB)")
        ax_noise.set_ylabel("Frequency (kHz)")
        ax_noise.set_xlim(min_level_default, max_level_default)
        ax_noise.set_ylim(f_min / 1000.0, f_max / 1000.0)
        ax_noise.grid(True, alpha=0.25)
        noise_axes.append(ax_noise)
        noise_lines.append(noise_line)

        plot_datasets.append(
            {
                "name": "Combined",
                "time_s": t_s,
                "freq_hz": freq_hz,
                "spec_db": spec_db_combined,
                "spec_db_raw": spec_db_combined,
                "median_update_mask": median_update_mask,
                "median_estimate_db": median_estimate_db,
                "freq_flatten_correction_db": np.zeros(n_comb_bins, dtype=np.float64),
                "overlay_t": t_overlay,
                "overlay_y_khz": y_overlay_hz / 1000.0,
            }
        )

    if show_plots:
        # Reserve space at bottom for slider controls.
        fig.tight_layout(rect=[0, 0.10, 1, 0.98])

        baseline_ax = fig.add_axes([0.14, 0.04, 0.32, 0.025])
        gain_ax = fig.add_axes([0.58, 0.04, 0.32, 0.025])

        baseline_slider = Slider(
            baseline_ax,
            "Baseline (dB)",
            -240.0,
            -20.0,
            valinit=baseline_default,
            valstep=1.0,
        )
        gain_slider = Slider(
            gain_ax,
            "Gain (dB)",
            10.0,
            220.0,
            valinit=gain_default,
            valstep=1.0,
        )

        def _update_levels(_):
            baseline = float(baseline_slider.val)
            gain = float(gain_slider.val)
            vmin = baseline
            vmax = baseline + gain
            for image in periodogram_images:
                image.set_clim(vmin=vmin, vmax=vmax)
            for axis in noise_axes:
                axis.set_xlim(vmin, vmax)
            fig.canvas.draw_idle()

        baseline_slider.on_changed(_update_levels)
        gain_slider.on_changed(_update_levels)
        _update_levels(None)

        stop_requested = False

        def _request_stop(*_args):
            nonlocal stop_requested
            stop_requested = True

        def _on_key(event):
            if event.key in {"q", "escape"}:
                _request_stop()
                plt.close(fig)

        fig.canvas.mpl_connect('close_event', _request_stop)
        fig.canvas.mpl_connect('key_press_event', _on_key)

        manager = getattr(fig.canvas, "manager", None)
        key_handler_id = getattr(manager, "key_press_handler_id", None) if manager else None
        if key_handler_id is not None:
            fig.canvas.mpl_disconnect(key_handler_id)

        # Animate frame-by-frame to simulate real-time receive flow across the periodogram.
        plt.show(block=False)
        plt.pause(0.001)

        max_frames = max(data["spec_db"].shape[0] for data in plot_datasets)
        step_candidates = []
        for data in plot_datasets:
            t_vals = data["time_s"]
            if t_vals.size > 1:
                diffs = np.diff(t_vals)
                positive = diffs[diffs > 0]
                if positive.size:
                    step_candidates.append(float(np.median(positive)))
        pause_dt = min(step_candidates) if step_candidates else 0.04
        pause_dt = max(0.005, min(pause_dt, 0.20))

        try:
            start_wall = time.perf_counter()
            last_drawn_idx = -1
            while True:
                if stop_requested or not plt.fignum_exists(fig.number):
                    break

                elapsed_s = max(0.0, time.perf_counter() - start_wall)
                current_idx = int(elapsed_s / pause_dt) % max_frames
                if current_idx == last_drawn_idx:
                    fig.canvas.flush_events()
                    fig.canvas.start_event_loop(0.001)
                    continue

                if last_drawn_idx < 0:
                    indices = [current_idx]
                elif current_idx > last_drawn_idx:
                    indices = list(range(last_drawn_idx + 1, current_idx + 1))
                else:
                    indices = list(range(last_drawn_idx + 1, max_frames)) + list(range(0, current_idx + 1))

                for row_idx, data in enumerate(plot_datasets):
                    spec_db = data["spec_db"]
                    n_frames = spec_db.shape[0]
                    buffer = periodogram_buffers[row_idx]
                    freq_khz = data["freq_hz"] / 1000.0

                    for frame_idx in indices:
                        if frame_idx >= n_frames:
                            continue
                        buffer[:, frame_idx] = spec_db[frame_idx]
                        if data["median_update_mask"][frame_idx]:
                            _update_median_estimate_db(
                                data["median_estimate_db"],
                                spec_db[frame_idx],
                                step_db=MEDIAN_UPDATE_STEP_DB,
                            )

                    periodogram_images[row_idx].set_data(buffer)

                    overlay_t = data["overlay_t"]
                    overlay_y = data["overlay_y_khz"]
                    if overlay_t.size:
                        end = min(current_idx + 1, overlay_t.size)
                        power_lines[row_idx].set_data(overlay_t[:end], overlay_y[:end])

                    noise_lines[row_idx].set_data(data["median_estimate_db"], freq_khz)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                fig.canvas.start_event_loop(0.001)
                last_drawn_idx = current_idx
        except KeyboardInterrupt:
            print("Plot window interrupted (Ctrl+C), exiting cleanly.")
    else:
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    fig.savefig(output_path, dpi=140)
    print(f"Wrote diagnostics plot: {output_path}")

    if not show_plots:
        plt.close(fig)


def _equalize_and_lp_filter(
    samples: np.ndarray,
    correction_db: np.ndarray,
    sample_rate: int,
    nfft_correction: int,
    lp_cutoff_hz: float,
) -> np.ndarray:
    """Apply equalization and LP taper to a real signal in one FFT pass.

    correction_db: per-bin power-dB correction from rfft at sample_rate / nfft_correction.
    lp_cutoff_hz: passband edge; cosine taper to 0 over [cutoff, 2*cutoff] Hz.
    Returns float32 at the same sample rate.
    """
    n = samples.size
    if n == 0:
        return samples.astype(np.float32)

    freq = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Equalization: interpolate per-bin correction onto this signal's rfft grid.
    # correction_db is zero-filled outside the source Nyquist range.
    if correction_db.size > 0:
        freq_corr = np.fft.rfftfreq(nfft_correction, 1.0 / sample_rate)
        corr_interp = np.interp(freq, freq_corr, correction_db, left=0.0, right=0.0)
        eq_gain = 10.0 ** (corr_interp / 20.0)  # power dB → amplitude gain
    else:
        eq_gain = np.ones(freq.size, dtype=np.float64)

    # LP taper: cosine roll-off above lp_cutoff_hz (mirrors _apply_lp_taper).
    lo, hi = float(lp_cutoff_hz), 2.0 * float(lp_cutoff_hz)
    lp_gain = np.ones(freq.size, dtype=np.float64)
    trans = (freq > lo) & (freq <= hi)
    if np.any(trans):
        t = (freq[trans] - lo) / (hi - lo)
        lp_gain[trans] = 10.0 ** ((-60.0 * (0.5 - 0.5 * np.cos(np.pi * t))) / 20.0)
    lp_gain[freq > hi] = 0.0

    X = np.fft.rfft(samples.astype(np.float64))
    x_out = np.fft.irfft(X * eq_gain * lp_gain, n=n)
    return x_out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 48 kHz combined MSK144 complex test signal")
    parser.add_argument('--input-dir', default='MSK144', help='Folder containing source WAV files')
    parser.add_argument('--pattern', default='*.wav', help='Glob pattern for source files')
    parser.add_argument('--output-rate', type=int, default=48000,
                        help='Output sample rate in Hz')
    parser.add_argument('--target-centers-hz', default='-9000,-3000,3000,9000',
                        help='Comma-separated target center frequencies in Hz for each file')
    parser.add_argument('--output-npy', default='msk144_combined_iq_48k.npy',
                        help='Output complex .npy file path')
    parser.add_argument('--output-iq-wav', default='msk144_combined_iq_48k.wav',
                        help='Optional output stereo IQ WAV path; use empty string to skip')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating diagnostics plots for input WAV files')
    parser.add_argument('--show-plots', action='store_true', dest='show_plots',
                        help='Display diagnostics plots interactively (default)')
    parser.add_argument('--no-show-plots', action='store_false', dest='show_plots',
                        help='Do not display diagnostics plots interactively')
    parser.add_argument('--flatten-spectrum', action='store_true', dest='flatten_spectrum',
                        help='Flatten diagnostics spectrum using median noise floor vs frequency (default)')
    parser.add_argument('--no-flatten-spectrum', action='store_false', dest='flatten_spectrum',
                        help='Do not apply frequency flattening to diagnostics spectrum')
    parser.set_defaults(show_plots=True, flatten_spectrum=True)
    parser.add_argument('--plot-output', default='msk144_wav_diagnostics.png',
                        help='Output PNG path for diagnostics plots')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {input_dir} matching {args.pattern}")

    target_centers = list(reversed(parse_float_list(args.target_centers_hz)))

    if len(target_centers) != len(files):
        raise SystemExit(
            f"Need one target center per file: found {len(files)} files but {len(target_centers)} centers"
        )

    prepared: list[np.ndarray] = []
    diagnostics_data: list[tuple[str, np.ndarray, int]] = []
    max_len = 0

    print("Input files:")
    for index, path in enumerate(files):
        samples, src_rate = read_wav_mono(path)
        diagnostics_data.append((path.name, samples, src_rate))

        # Compute the same flattening correction used by the combined plot,
        # then apply it together with the 3 kHz LP filter before resampling.
        _, _, spec_db_src = _compute_spectrogram_db(samples, src_rate)
        if spec_db_src.size > 0:
            _, _, correction_db = _flatten_spectrum_by_median_noise_floor(spec_db_src)
        else:
            correction_db = np.array([], dtype=np.float64)
        samples_proc = _equalize_and_lp_filter(
            samples, correction_db, src_rate, DIAGNOSTIC_NFFT, lp_cutoff_hz=3000.0,
        )

        up = resample_linear(samples_proc, src_rate, args.output_rate)

        shift_hz = target_centers[index] - SOURCE_CENTER_HZ
        shifted = freq_shift_real_to_complex(up, shift_hz, args.output_rate)

        prepared.append(shifted)
        max_len = max(max_len, shifted.size)

        print(
            f"  {path.name}: src_rate={src_rate} Hz, samples={samples.size}, "
            f"upsampled={up.size}, target_center={target_centers[index]:.1f} Hz, "
            f"shift={shift_hz:+.1f} Hz, start=0.000 s"
        )

    combined = np.zeros(max_len, dtype=np.complex64)
    for shifted in prepared:
        combined[:shifted.size] += shifted

    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0:
        combined *= (0.95 / peak)

    out_npy = Path(args.output_npy)
    np.save(out_npy, combined)
    print(f"Wrote complex stream: {out_npy} ({combined.size} samples @ {args.output_rate} Hz)")

    if args.output_iq_wav.strip():
        out_wav = Path(args.output_iq_wav)
        write_iq_wav(out_wav, combined, args.output_rate)
        print(f"Wrote IQ WAV: {out_wav}")

    if not args.skip_plots:
        plot_wav_diagnostics(
            diagnostics_data,
            output_path=Path(args.plot_output),
            show_plots=args.show_plots,
            flatten_spectrum=args.flatten_spectrum,
            combined_output_rate=args.output_rate,
            combined_target_centers_hz=target_centers,
            combined_lp_cutoff_hz=3000.0,
        )


if __name__ == '__main__':
    main()
