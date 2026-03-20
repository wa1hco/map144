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
"""MSK144 detection helpers: LP filter, pair scanning, and decode pipeline.

This module provides the signal-processing primitives used to detect and
decode MSK144 bursts from a complex IQ stream.  It is deliberately stateless
(no class, no globals beyond configuration constants) so that every function
can be unit-tested in isolation and called freely from ``processing.py`` and
the background thread launched by ``runtime.py``.

Functions
---------
design_lp_filter(sample_rate, cutoff_hz, numtaps)
    Design a linear-phase FIR low-pass filter using a Kaiser-windowed sinc
    (via ``scipy.signal.firwin``).  The default cutoff of 10 kHz passes the
    full ±24 kHz MSK144 burst bandwidth while attenuating alias energy from
    the 48 kHz DAXIQ stream.  Returns float64 tap array for use with
    ``apply_lp_filter``.

apply_lp_filter(block, taps, zi_re, zi_im)
    Apply the FIR filter to I and Q independently using ``scipy.signal.lfilter``
    with explicit initial-condition vectors ``zi_re`` / ``zi_im``.  Returning
    updated state vectors allows the caller to chain calls across consecutive
    chunks without discontinuities at chunk boundaries (streaming-FIR pattern).
    Input may be any complex dtype; output is complex64.

scan_for_pairs(power_db, freq_hz, spacing_hz, tol_hz, thresh_db)
    Scan an fftshift'd squared-domain power spectrum for symmetric peak pairs
    spaced ``spacing_hz`` apart.  MSK144 at fc±500 Hz produces, after squaring,
    spectral lines at ±1000 Hz relative to 2·fc; in the squared FFT these appear
    as a pair separated by 2000 Hz.  The detector:

      1. Computes median noise floor and applies a ``thresh_db`` guard.
      2. Finds local maxima with a ±5-bin exclusion zone.
      3. Pairs each peak with the nearest partner at the expected spacing
         (within ``tol_hz`` tolerance).
      4. Returns pairs ordered strongest-first so callers can act on the
         highest-confidence candidate immediately.

fc_from_sq_pair(f_sq_lo, f_sq_hi)
    Recover the original signal carrier frequency from a squared-domain tone
    pair.  Squaring shifts every spectral component from f to 2f, so the pair
    midpoint (f_lo + f_hi) / 2 equals 2·fc; dividing by 2 yields fc.

_read_ring(iq_ring, ring_pos, abs_sample, read_start_abs, n_samples)
    Copy a contiguous segment from the circular ring buffer maintained in
    ``RadioIQVisualizer``.  Handles wrap-around and the case where
    ``read_start_abs`` predates the oldest retained sample (clips the start,
    returns whatever is still in buffer).  Returns complex64 array.

extract_and_decode(iq_ring, ring_state_fn, detect_sample,
                   sample_rate, fc_hz, output_dir, t_in_window)
    End-to-end decode pipeline, intended to run in a daemon thread:

      1. **Extract** — waits for up to ``max_post`` of new IQ data to arrive
         in the ring buffer after ``detect_sample``, then reads 300 ms before
         detection plus everything up to where the signal envelope drops below
         5 % of its peak (plus a 50 ms margin), zero-padded 100 ms each end.
      2. **Mix** — complex-multiplies by ``exp(-j2π·Δf·t)`` to shift the
         carrier from ``fc_hz`` to 1500 Hz (the frequency ``jt9`` expects).
      3. **Decimate** — extracts the real (I) channel and applies
         ``scipy.signal.decimate`` with factor 4 to go from 48 kHz → 12 kHz.
      4. **Normalise** — scales audio peak to 90 % of full scale.
      5. **Write WAV** — writes a temporary 16-bit mono 12 kHz WAV file.
      6. **Run jt9** — calls the WSJT-X ``jt9 --msk144`` decoder with the
         WAV file; timeout 10 s.
      7. **Log / save** — on a successful decode, moves the WAV to
         ``MSK144/detections/`` with a timestamped filename and prints the
         decoded callsigns to stdout.

Constants
---------
JT9_PATH           : absolute path to the jt9 binary (WSJT-X build tree)
JT9_BASE_ARGS      : fixed jt9 arguments (mode, period, freq window, depth)
_DECODE_RATE       : 12000 Hz — target WAV sample rate expected by jt9
_DECIMATE_FACTOR   : 4       — ratio 48 kHz / 12 kHz
_TARGET_FC_HZ      : 1500 Hz — jt9 audio centre frequency after mixing
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
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
_JT9_SEMAPHORE = threading.Semaphore(4)   # max concurrent jt9 processes


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
    # Estimate noise floor from bins that are above the 1e-12 clamp floor
    # (~-135 dBFS for the squared spectrum).  Using only the populated bins
    # prevents the median from collapsing to the floor when signals occupy
    # a small fraction of the band, which would make the threshold meaningless.
    floor_db    = 10 * np.log10(1e-12) - 15.0   # matches power_db_sq floor
    active      = power_db[power_db > floor_db + 5.0]
    if active.size < 10:
        return []   # not enough data above floor to make a reliable estimate
    noise_floor = float(np.percentile(active, 50))
    threshold   = noise_floor + thresh_db

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
    ring_state_fn,
    detect_sample: int,
    sample_rate: int,
    fc_hz: float,
    output_dir: str,
    t_in_window: float = 0.0,
) -> None:
    """Extract IQ around detect_sample, mix fc to 1500 Hz, decimate to 12 kHz, run jt9.

    iq_ring      – live ring buffer array (shared reference, not a snapshot)
    ring_state_fn – callable returning (ring_pos, abs_sample) from the live writer
    t_in_window  – seconds within the 15-s display window (0–15), used for logging.
    Intended to run in a background daemon thread.
    """
    pre_n  = int(0.300 * sample_rate)   # 300 ms before detection
    post_n = int(1.200 * sample_rate)   # 1200 ms after detection — MSK144 envelope e·t·exp(-t)
                                        # decays to ~10% at 4× width; for max 300 ms ping that
                                        # tail extends ~900 ms past peak, so 1200 ms captures it
    pad_n  = int(0.100 * sample_rate)   # 100 ms zero padding each end

    # ── Wait for post_n samples to arrive after detect_sample ────────────────
    deadline = time.monotonic() + 2.0
    while True:
        ring_pos, abs_sample = ring_state_fn()
        if abs_sample >= detect_sample + post_n:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.020)

    ring_pos, abs_sample = ring_state_fn()

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

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    launch_ts = datetime.now(timezone.utc).isoformat()

    def _log_launch(outcome: str, message: str = "", jt9_snr=None, jt9_line: str = ""):
        entry = {
            "timestamp": launch_ts,
            "t_sec":     round(t_sec, 3),
            "fc_khz":    round(fc_khz, 3),
            "outcome":   outcome,   # "decoded" | "no_decode" | "timeout" | "error"
            "message":   message,
            "jt9_snr_db": jt9_snr,
            "jt9_line":  jt9_line,
        }
        with open(out_dir / "launches.jsonl", "a") as lf:
            lf.write(json.dumps(entry) + "\n")

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
            with _JT9_SEMAPHORE:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20.0)

            decoded = ""
            for line in result.stdout.splitlines():
                s = line.strip()
                if s and not s.startswith('<') and not s.startswith('EOF'):
                    decoded = s
                    break

            if decoded:
                # jt9 MSK144 output: "HHMMSS  SNR  dt  freq  mode  MESSAGE"
                tokens   = decoded.split()
                bare_msg = tokens[-1] if tokens else decoded
                jt9_snr  = int(tokens[1]) if len(tokens) >= 2 else None

                # Name the WAV after the decoded message so it is easy to
                # correlate with the comparison report.  Add a sequence number
                # if a file for this message already exists (re-trigger case).
                seq = 1
                while True:
                    seq_sfx = f"_{seq:02d}" if seq > 1 else ""
                    save_name = f"{bare_msg}{seq_sfx}.wav"
                    if not (out_dir / save_name).exists():
                        break
                    seq += 1
                shutil.move(tmp_path, str(out_dir / save_name))
                print(f"[MSK144 DECODE]  t={t_sec:.2f}s  fc={fc_khz:.2f} kHz  {decoded}", flush=True)

                # Append to decode log for post-run comparison with manifest
                decode_entry = {
                    "timestamp":  launch_ts,
                    "t_sec":      round(t_sec, 3),
                    "fc_khz":     round(fc_khz, 3),
                    "message":    bare_msg,
                    "jt9_snr_db": jt9_snr,
                    "jt9_line":   decoded,
                }
                with open(out_dir / "decodes.jsonl", "a") as lf:
                    lf.write(json.dumps(decode_entry) + "\n")

                _log_launch("decoded", message=bare_msg, jt9_snr=jt9_snr, jt9_line=decoded)
            else:
                _log_launch("no_decode")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        print(f"[MSK144]  jt9 timeout (>20s) at t={t_sec:.2f}s  fc={fc_khz:.2f} kHz", flush=True)
        _log_launch("timeout")
    except Exception as exc:
        print(f"[MSK144]  extract_and_decode error: {exc}", flush=True)
        _log_launch("error")
        import traceback
        traceback.print_exc()
