"""MSK144 detection helpers: LP filter, pair scanning, and decode pipeline."""

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
from scipy.signal import decimate, firwin, lfilter

JT9_PATH = '/home/jeff/ham/wsjtx-2.7.0/build/wsjtx-prefix/src/wsjtx-build/jt9'
JT9_BASE_ARGS = [
    JT9_PATH, "--msk144",
    "-p", "15", "-L", "1400", "-H", "1600",
    "-f", "1500", "-F", "200", "-d", "3",
]

_DECODE_RATE = 12000        # target WAV sample rate for jt9
_DECIMATE_FACTOR = 4        # 48 kHz → 12 kHz
_TARGET_FC_HZ = 1500.0      # jt9 expects the signal at this frequency


def design_lp_filter(sample_rate: int, cutoff_hz: float = 10000.0, numtaps: int = 101) -> np.ndarray:
    """Design a symmetric FIR low-pass filter.  Returns tap array (float64)."""
    nyq = sample_rate / 2.0
    taps = firwin(numtaps, cutoff_hz / nyq)
    return taps.astype(np.float64)


def apply_lp_filter(
    block: np.ndarray,
    taps: np.ndarray,
    zi_re: np.ndarray,
    zi_im: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply FIR LP filter to I and Q independently, preserving streaming state.

    Returns (filtered_iq, new_zi_re, new_zi_im).
    """
    i_filt, new_zi_re = lfilter(taps, 1.0, np.real(block).astype(np.float64), zi=zi_re)
    q_filt, new_zi_im = lfilter(taps, 1.0, np.imag(block).astype(np.float64), zi=zi_im)
    filtered = (i_filt + 1j * q_filt).astype(np.complex64)
    return filtered, new_zi_re, new_zi_im


def scan_for_pairs(
    power_db: np.ndarray,
    freq_hz: np.ndarray,
    spacing_hz: float = 2000.0,
    tol_hz: float = 200.0,
    thresh_db: float = 10.0,
) -> list[tuple[float, float]]:
    """Scan a power spectrum for pairs of peaks spaced spacing_hz apart.

    power_db  – full fftshift'd squared spectrum in dB  (shape: fft_size,)
    freq_hz   – corresponding frequency axis in Hz  (same shape)

    Returns list of (f_lo_hz, f_hi_hz) for each detected pair, ordered by
    peak power (strongest pair first).
    """
    noise_floor = float(np.median(power_db))
    threshold = noise_floor + thresh_db

    # Local-maximum peak detector with ±5-bin exclusion zone
    n = len(power_db)
    win = 5
    peaks = []
    for i in range(win, n - win):
        if power_db[i] >= threshold:
            if power_db[i] == power_db[max(0, i - win):i + win + 1].max():
                peaks.append(i)

    if not peaks:
        return []

    df = float(freq_hz[1] - freq_hz[0]) if len(freq_hz) > 1 else 1.0

    pairs = []
    seen = set()
    for i_lo in sorted(peaks, key=lambda i: -power_db[i]):
        if i_lo in seen:
            continue
        f_lo = float(freq_hz[i_lo])
        f_target_hi = f_lo + spacing_hz
        best_i_hi = None
        best_dist = tol_hz + 1.0
        for i_hi in peaks:
            if i_hi == i_lo or i_hi in seen:
                continue
            dist = abs(float(freq_hz[i_hi]) - f_target_hi)
            if dist <= tol_hz and dist < best_dist:
                best_dist = dist
                best_i_hi = i_hi
        if best_i_hi is not None:
            pairs.append((f_lo, float(freq_hz[best_i_hi])))
            seen.add(i_lo)
            seen.add(best_i_hi)

    return pairs


def fc_from_sq_pair(f_sq_lo: float, f_sq_hi: float) -> float:
    """Recover original signal fc from a squared-domain tone pair.

    Squaring maps a complex tone at f → 2f, so the pair midpoint divided by 2
    gives the original carrier centre frequency.
    """
    return (f_sq_lo + f_sq_hi) / 4.0


def _read_ring(
    iq_ring: np.ndarray,
    ring_pos: int,
    abs_sample: int,
    read_start_abs: int,
    n_samples: int,
) -> np.ndarray:
    """Copy n_samples from the circular ring buffer starting at read_start_abs.

    ring_pos   – index where the NEXT write will go (write head)
    abs_sample – absolute sample count at ring_pos
    """
    ring_size = len(iq_ring)
    oldest_abs = abs_sample - ring_size

    if read_start_abs < oldest_abs:
        n_skip = oldest_abs - read_start_abs
        n_samples -= n_skip
        read_start_abs = oldest_abs

    if n_samples <= 0 or read_start_abs >= abs_sample:
        return np.zeros(0, dtype=np.complex64)

    n_samples = min(n_samples, abs_sample - read_start_abs)

    samples_from_end = abs_sample - read_start_abs
    start_ring = (ring_pos - samples_from_end) % ring_size
    end_ring = (start_ring + n_samples) % ring_size

    if start_ring + n_samples <= ring_size:
        return iq_ring[start_ring:start_ring + n_samples].copy()
    part1 = iq_ring[start_ring:]
    part2 = iq_ring[:end_ring]
    return np.concatenate([part1, part2])


def extract_and_decode(
    iq_ring: np.ndarray,
    ring_pos: int,
    abs_sample: int,
    detect_sample: int,
    sample_rate: int,
    fc_hz: float,
    output_dir: str,
    t_in_window: float = 0.0,
) -> None:
    """Extract IQ around detect_sample, mix fc to 1500 Hz, decimate to 12 kHz, run jt9.

    t_in_window – seconds within the 15-s display window (0–15), used for logging.
    Intended to run in a background daemon thread.
    """
    pre_n  = int(0.300 * sample_rate)   # 300 ms before detection
    post_n = int(0.300 * sample_rate)   # 300 ms after detection
    pad_n  = int(0.100 * sample_rate)   # 100 ms zero padding each end

    iq_seg = _read_ring(iq_ring, ring_pos, abs_sample,
                        detect_sample - pre_n, pre_n + post_n)

    if iq_seg.size == 0:
        print("[MSK144] ring buffer empty, skipping decode", flush=True)
        return

    # Zero-pad
    pad = np.zeros(pad_n, dtype=np.complex64)
    iq_padded = np.concatenate([pad, iq_seg, pad])

    # Mix: shift fc_hz → _TARGET_FC_HZ
    shift_hz = fc_hz - _TARGET_FC_HZ
    t = np.arange(len(iq_padded), dtype=np.float64)
    iq_mixed = (iq_padded * np.exp(-2j * np.pi * shift_hz * t / sample_rate)).astype(np.complex64)

    # Decimate I channel: 48 kHz → 12 kHz
    i_dec = decimate(np.real(iq_mixed).astype(np.float64), _DECIMATE_FACTOR, zero_phase=True)
    audio = i_dec.astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak * 0.9

    t_sec = t_in_window
    fc_khz = fc_hz / 1000.0

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_DECODE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            cmd = JT9_BASE_ARGS + [tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)

            decoded = ""
            for line in result.stdout.splitlines():
                s = line.strip()
                if s and not s.startswith('<') and not s.startswith('EOF'):
                    decoded = s
                    break

            if decoded:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                save_name = f"msk144_t{t_sec:.2f}_fc{fc_khz:.2f}kHz.wav"
                shutil.move(tmp_path, str(out_dir / save_name))
                print(f"[MSK144 DECODE]  t={t_sec:.2f}s  fc={fc_khz:.2f} kHz  {decoded}", flush=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        print(f"[MSK144]  jt9 timeout at t={t_sec:.2f}s", flush=True)
    except Exception as exc:
        print(f"[MSK144]  extract_and_decode error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
