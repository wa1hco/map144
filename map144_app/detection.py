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
"""MSK144 detection helpers: pair scanning, SNR estimate, ring readout, decode.

This module sits after the 48-channel polyphase channelizer in channelizer.py
and the per-channel squared-FFT detector in processing.py.  It is stateless
(no class, no mutable globals beyond threading primitives) so functions can be
tested in isolation and called from processing.py daemon threads.

Frequency plan
--------------
Channel k captures signals from stations with USB dial at calling_freq + k × 1000 Hz.
In USB radio, signal energy is 1500 Hz above the dial.  The channelizer NCO for
channel k is placed at (k × 1000 + channel_offset_hz) Hz from IQ DC, where
channel_offset_hz = (calling_freq - pan_center) + 1500.  The +1500 aligns the NCO
with the signal energy so the carrier lands at DC inside the channel row.

fc_hz — carrier position in Hz from IQ DC (pan centre):

    fc_hz = ch_signed x 1000 + channel_offset_hz + fc_offset

For a station on calling_freq (k=0, fc_offset≈0, pan=calling_freq):
    fc_hz ≈ 0 + 1500 + 0 = 1500 Hz

extract_and_decode (this module) mixes the carrier from fc_hz to 1500 Hz:

    shift_hz = fc_hz - 1500

For that same station: shift_hz = 0, so no mixing needed — the carrier is
already at 1500 Hz.  For any other station: the mix places its carrier at 1500 Hz
in the audio regardless of where it was in the IF.

radio_khz logged = pan_center_khz + (fc_hz - 1500) / 1000 = USB dial frequency.

Any error in center_freq_mhz or fc_hz propagates into both radio_khz and the
carrier position in the WAV; a shifted carrier causes jt9 to miss.

Functions
----------
scan_for_pairs(power_db, freq_hz, spacing_hz, tol_hz, thresh_db)
    Scan an fftshift'd squared-domain power spectrum for symmetric peak pairs
    spaced spacing_hz apart.  MSK144 at fc+-500 Hz produces, after squaring,
    spectral lines at +-1000 Hz relative to 2*fc; in the squared FFT these appear
    as a pair separated by 2000 Hz.  Steps:
      1. Estimate median noise floor; apply thresh_db guard.
      2. Find local maxima with a +-5-bin exclusion zone.
      3. Pair each peak with the nearest partner at the expected spacing
         (within tol_hz tolerance).
      4. Return pairs ordered strongest-first.

fc_from_sq_pair(f_sq_lo, f_sq_hi)
    Recover original signal fc from a squared-domain tone pair.
    Squaring maps f -> 2f, so pair midpoint / 2 = fc.

_read_ring(iq_ring, ring_pos, abs_sample, read_start_abs, n_samples)
    Copy a contiguous segment from the circular IQ ring buffer (Engine._iq_ring).
    Handles wrap-around; clips start if read_start_abs predates oldest sample.
    Returns complex64 array.

extract_and_decode(iq_ring, ring_state_fn, detect_sample, sample_rate, fc_hz, ...)
    End-to-end decode pipeline, run in a daemon thread:
      1. Extract -- wait for post-detection IQ; read 500 ms pre-burst + tail.
      2. Mix -- shift carrier from fc_hz to 1500 Hz.
      3. Decimate -- real I channel, 48 kHz -> 12 kHz (factor 4).
      4. Normalise -- scale audio peak to 90% full scale.
      5. Write WAV -- 16-bit mono 12 kHz temp file.
      6. Run jt9 -- WSJT-X jt9 --msk144 decoder; timeout 10 s.
      7. Log / save -- on decode, move WAV to MSK144/detections/ with timestamp.

Constants
----------
JT9_BASE_ARGS  : fixed jt9 arguments (mode, period, freq window, depth)
_DECODE_RATE   : 12000 Hz -- target WAV sample rate expected by jt9
_TARGET_FC_HZ  : 1500 Hz  -- jt9 audio centre frequency after mixing
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import decimate, firwin, filtfilt

from .msk144_spd import msk144_spd_decode

_JT9_FTOL          = 200   # Hz — search half-width around 1500 Hz audio centre.
                            # Observed 2026-05-11: 85% of WSJT-X-decoded MSK144
                            # signals on calling channel land at audio 1600-1640 Hz
                            # (peak at 1603-1606 Hz), well above the prior 100 Hz
                            # half-width.  Widening to 200 Hz brings MAP144 in line
                            # with SPD's ntol=200 default and with the audio-freq
                            # distribution we actually see in operation.  Costs
                            # ~30-50% more jt9 CPU per decode — covered by the
                            # 4-process semaphore.
_JT9_FTOL_ANALYSIS = 400   # Hz — wider default for analysis window (human click accuracy)
# jt9 decode depth.  Depth 3 (deep) widens the hypothesis search and adds
# hash-based LDPC parity shortcuts using the master callsign/grid table.
# That makes it prone to phantom decodes on noise — popular calls
# (e.g. K1JT, WA2FZW) appear with arbitrary grids when LDPC parity
# happens to pass on noise that hashes to a known call.  WSJT-X itself
# defaults to depth 1 (normal) for live operation; matching here.
# Observed 2026-05-11: -d 3 produced a 24-decode "CQ K1JT FN20" phantom
# cluster over 3 hours from a persistent narrowband signal at 50265 kHz
# (K1JT operates from FN42, FN20 grid is wrong — hash artifact).  Those
# were going out over UDP/PSKReporter as spurious reports.
_JT9_DEPTH   = 1

JT9_BASE_ARGS = [
    'jt9', "--msk144",
    "-L", str(1500 - _JT9_FTOL), "-H", str(1500 + _JT9_FTOL),
    "-F", str(_JT9_FTOL), "-d", str(_JT9_DEPTH),
]

_DECODE_RATE = 12000        # target WAV sample rate for jt9
_DECIMATE_FACTOR = 4        # 48 kHz → 12 kHz
_TARGET_FC_HZ = 1500.0      # jt9 expects the signal at this frequency
_JT9_SEMAPHORE = threading.Semaphore(4)   # max concurrent jt9 processes

# Pre-computed SNR estimator constants (at _DECODE_RATE = 12000 Hz)
_SNR_N_FFT    = 1024                         # ≈ 11.7 Hz/bin  → 85 ms window
_SNR_WINDOW   = np.hanning(_SNR_N_FFT).astype(np.float64)
_SNR_FREQS    = np.fft.rfftfreq(_SNR_N_FFT, 1.0 / _DECODE_RATE)
_SNR_BIN_HZ   = float(_DECODE_RATE) / _SNR_N_FFT
# Signal band: MSK144 tones at ±500 Hz around 1500 Hz carrier → 1000–2000 Hz.
# Widen slightly to cover jt9's ±600 Hz search window.
_SNR_SIG_MASK  = (_SNR_FREQS >= 900.0) & (_SNR_FREQS <= 2100.0)
# Noise band: above DC artefacts, below Nyquist rolloff, excluding signal band.
_SNR_NOISE_MASK = (_SNR_FREQS > 200.0) & (_SNR_FREQS < 5500.0) & ~_SNR_SIG_MASK
_SNR_REF_BINS  = 2500.0 / _SNR_BIN_HZ  # 2500 Hz reference bandwidth in bins
_SNR_N_SIG_BINS = int(_SNR_SIG_MASK.sum())  # bins in signal band (noise floor correction)

# ── SPD audio BPF (300–2700 Hz modelled as complex lowpass ±1200 Hz) ─────────
# Applied to complex baseband IQ at 12 kHz before the SPD.  The MSK144 signal
# occupies roughly 300–2700 Hz in the USB audio passband; modelling the receiver's
# SSB filter reduces the noise bandwidth from ~4800 Hz (IIR decimation) to ~2400 Hz,
# eliminating detection failures entirely (det_miss → 0) and shifting the SPD 50%
# threshold by ~1.5 dB.
#
# With the carrier mixed to 0 Hz baseband the audio BPF is symmetric about DC,
# so it becomes a simple lowpass at half_bw = (2700−300)/2 = 1200 Hz.  Applying
# the same real FIR independently to I and Q preserves I/Q statistical independence,
# which the BP decoder requires for correct LLR computation.  filtfilt gives zero
# phase so there is no group-delay shift and no onset transient on the 72 ms frame.
_SPD_BPF_NTAPS = 129
_SPD_BPF_TAPS  = firwin(_SPD_BPF_NTAPS, 1200.0 / (_DECODE_RATE / 2.0)).astype(np.float64)


def _apply_audio_bpf(iq_bb: np.ndarray) -> np.ndarray:
    """Apply 300–2700 Hz audio BPF to complex baseband IQ at 12 kHz.

    Carrier must already be at 0 Hz (baseband).  Filters I and Q with the same
    real lowpass FIR independently, then recombines.
    """
    i_filt = filtfilt(_SPD_BPF_TAPS, 1.0, np.real(iq_bb).astype(np.float64))
    q_filt = filtfilt(_SPD_BPF_TAPS, 1.0, np.imag(iq_bb).astype(np.float64))
    return (i_filt + 1j * q_filt).astype(np.complex64)


def _estimate_snr_db(audio: np.ndarray) -> int | None:
    """Estimate MSK144 SNR in dB referenced to 2500 Hz bandwidth (WSJTX convention).

    jt9 cannot compute a meaningful SNR from a short burst WAV because it has
    no noise-only baseline to compare against.  This function uses the audio
    we already have to produce an equivalent estimate:

    Noise
        Overlap-averaged PSD (50 % hop, Hanning window) over the first
        500 ms of pre-burst audio (audio[1200:7200] at 12 kHz).  When
        called from the jt9 fallback path the audio has 1500 ms of
        pre-burst, so the full noise_end window is used; the SPD path
        supplies 500 ms.  The median power across out-of-band bins
        (200–5500 Hz, excluding the signal band) gives the noise density.

    Signal
        A 1024-sample (≈ 85 ms) Hanning-windowed FFT is slid through the
        burst region in 256-sample steps.  The peak sum of in-band power
        (900–2100 Hz, covering both MSK144 tones at 1000 Hz and 2000 Hz
        plus jt9's ±600 Hz search window) represents the burst power at
        its strongest moment.

    Reference bandwidth
        SNR = 10·log10(signal_bins / (noise_per_bin × ref_bins))
        where ref_bins = 2500 Hz / bin_width (≈ 213 bins at 12 kHz /
        1024-point FFT).  Both signal and noise use the same FFT
        normalisation so units cancel.  The 2500 Hz reference matches the
        WSJTX convention for all weak-signal modes.

    audio   – float32 mono at _DECODE_RATE (12 kHz), structured as:
                [pad_n_dec] [pre_n_dec (noise)] [burst + post] [pad_n_dec]
              where pad_n_dec = 100 ms, pre_n_dec matches the caller's pre_n.

    Returns SNR rounded to the nearest integer dB, or None on failure.
    """
    pad_n_dec  = int(0.100 * _DECODE_RATE)   # 1200 samples
    # Use whatever pre-burst was captured; if less than 500 ms, skip to avoid
    # indexing into the burst region.
    pre_n_dec  = int(0.500 * _DECODE_RATE)   # 6000 samples minimum noise window
    noise_end  = pad_n_dec + pre_n_dec        # 7200
    burst_start = noise_end

    if len(audio) <= noise_end:
        return None

    # ── Noise floor: overlap-averaged PSD from the pre-burst segment ─────────
    noise_seg = audio[pad_n_dec:noise_end].astype(np.float64)
    hop       = _SNR_N_FFT // 2
    psds      = []
    for off in range(0, max(1, len(noise_seg) - _SNR_N_FFT + 1), hop):
        blk = noise_seg[off:off + _SNR_N_FFT]
        if len(blk) < _SNR_N_FFT:
            blk = np.pad(blk, (0, _SNR_N_FFT - len(blk)))
        psds.append(np.abs(np.fft.rfft(blk * _SNR_WINDOW)) ** 2)
    if not psds:
        return None
    _arr = np.mean(psds, axis=0)[_SNR_NOISE_MASK]  # fancy-index returns new writable array
    _k   = len(_arr) // 2
    _arr.partition(_k)                              # in-place O(n); bypasses _wrapfunc/_quantile
    noise_per_bin = float(_arr[_k])
    if noise_per_bin <= 0.0:
        return None

    # ── Signal: peak in-band power over N_FFT-sample windows in burst region ─
    burst_seg  = audio[burst_start:].astype(np.float64)
    best_sig   = 0.0
    for off in range(0, max(1, len(burst_seg) - _SNR_N_FFT + 1), _SNR_N_FFT // 4):
        blk = burst_seg[off:off + _SNR_N_FFT]
        if len(blk) < _SNR_N_FFT:
            break
        psd     = np.abs(np.fft.rfft(blk * _SNR_WINDOW)) ** 2
        in_band = float(np.sum(psd[_SNR_SIG_MASK]))
        best_sig = max(best_sig, in_band)

    if best_sig <= 0.0:
        return None

    # Subtract the noise contribution from the signal-band sum so the estimator
    # does not bottom out at 10*log10(n_sig_bins/ref_bins) ≈ −3 dB when SNR→0.
    # Without this, the noise power in the ~103 signal bins dominates at low SNR
    # and clamps the reported value to 0–2 dB even when the true SNR is −8 dB.
    signal_only = best_sig - noise_per_bin * _SNR_N_SIG_BINS
    if signal_only <= 0.0:
        return None   # signal indistinguishable from noise — caller uses jt9_snr_db

    snr_db = 10.0 * np.log10(signal_only / (noise_per_bin * _SNR_REF_BINS))
    return int(round(snr_db))



def scan_for_pairs(
    power_db: np.ndarray,
    freq_hz: np.ndarray,
    spacing_hz: float = 2000.0,
    tol_hz: float = 200.0,
    thresh_db: float = 10.0,
    center_hz: float | None = None,
    center_tol_hz: float = 500.0,
) -> list[tuple[float, float]]:
    """Scan a power spectrum for pairs of peaks spaced spacing_hz apart.

    power_db      – full fftshift'd spectrum in dB  (shape: fft_size,)
    freq_hz       – corresponding frequency axis in Hz  (same shape)
    center_hz     – if set, keep only pairs whose midpoint is within
                    center_tol_hz of this frequency.  Pass 0.0 with
                    center_tol_hz=500 to require the pair to straddle DC,
                    which naturally rejects adjacent-channel ghost detections.

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

    if center_hz is not None:
        pairs = [
            (fl, fh) for fl, fh in pairs
            if abs((fl + fh) / 2.0 - center_hz) <= center_tol_hz
        ]

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
    return np.concatenate([part1, part2], axis=0)   # axis=0 works for (N,) and (N,2)


def _eval_pair_metric(iq: np.ndarray, sample_rate: int) -> float:
    """Scalar squared-spectrum pair metric for a short IQ segment at 12 kHz."""
    from .processing import CH_DETECT_SIZE, _DETECT_WINDOW, _LO_MASK, _HI_MASK
    block_len = min(CH_DETECT_SIZE, len(iq))
    block = np.zeros(CH_DETECT_SIZE, dtype=np.complex64)
    block[-block_len:] = iq[-block_len:]
    X = np.fft.fft(block ** 2 * _DETECT_WINDOW)
    plin = np.abs(np.fft.fftshift(X)) / CH_DETECT_SIZE
    return float((np.max(plin[_LO_MASK]) + np.max(plin[_HI_MASK])) / 2.0)


def _polarization_search(
    iq_h: np.ndarray,
    iq_v: np.ndarray,
    sample_rate: int,
    n_theta: int = 9,
    n_dphi: int = 8,
) -> tuple:
    """Grid search over (θ, Δφ) to maximise the squared-spectrum pair metric.

    Combines the pre-mixed H and V channel segments:
        combined = iq_h * cos θ  +  iq_v * exp(j·Δφ) * sin θ

    Returns (theta_deg, delta_phi_deg, combined_iq).
    Both iq_h and iq_v should already be carrier-mixed to baseband.
    """
    best_metric   = -1.0
    best_theta    =  0.0
    best_dphi     =  0.0
    best_combined = iq_h

    thetas = np.linspace(0.0, np.pi / 2.0, n_theta)           # 0° … 90°
    dphis  = np.linspace(0.0, 2.0 * np.pi, n_dphi, endpoint=False)  # 0° … 315°

    for theta in thetas:
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        for dphi in dphis:
            combined = (iq_h * cos_t +
                        iq_v * np.exp(1j * dphi).astype(np.complex64) * sin_t)
            metric = _eval_pair_metric(combined, sample_rate)
            if metric > best_metric:
                best_metric   = metric
                best_theta    = theta
                best_dphi     = dphi
                best_combined = combined

    return np.degrees(best_theta), np.degrees(best_dphi), best_combined


def extract_and_decode(
    iq_ring: np.ndarray,
    ring_state_fn,
    detect_sample: int,
    sample_rate: int,
    fc_hz: float,
    output_dir: str,
    t_in_window: float = 0.0,
    decode_queue=None,
    marker_id: int = -1,
    ring_gen: int = 0,
    ring_gen_fn=None,
    center_freq_mhz: float = 0.0,
    detect_ts: str = "",
    jt9_args: list | None = None,
    dual_pol: bool = False,
    iq_t0_wall: float | None = None,
    det_source: str = "sq",
    launch_metrics: dict | None = None,
) -> None:
    """Extract IQ around detect_sample, mix fc to 1500 Hz, decimate to 12 kHz, run jt9.

    iq_ring         – live ring buffer array (shared reference, not a snapshot).
    ring_state_fn   – callable returning ``(ring_pos, abs_sample)`` from the writer.
    detect_sample   – absolute sample index at detection (matches ``_iq_abs_sample`` snap).
    fc_hz           – MSK carrier offset **in Hz relative to the IQ stream's tuned
                      centre** (DC in baseband), after channeliser placement and
                      fine estimate — **not** the USB dial frequency by itself; see
                      module docstring.
    output_dir      – directory for saved WAV on success.
    t_in_window     – seconds within the 15 s display window; markers and logging.
    decode_queue    – optional queue for decode results (GUI / analysis).
    marker_id       – id for heatmap / outcome correlation.
    ring_gen        – generation counter snapshot; invalidates extract if the ring wrapped.
    ring_gen_fn     – callable returning current ``_iq_ring_gen``.
    center_freq_mhz – receiver LO / digital **tuned centre** in MHz for the IQ
                      stream (same reference as spectrogram centre); combined with
                      ``fc_hz`` for dial-style logging — must stay consistent with
                      how the radio applies USB and the 1500 Hz audio convention.
    detect_ts       – UTC timestamp string for filenames and JSONL.
    jt9_args        – optional extra CLI args appended to the jt9 invocation.
    iq_t0_wall      – wall-clock for IQ ring sample 0 (Unix seconds). When provided,
                      the period epoch is derived from ``detect_sample`` and the IQ
                      sample rate, which is immune to pipeline / queue latency.
                      ``None`` falls back to the launch-time-minus-pre-window heuristic
                      (used by the offline analysis-engine click path).

    Intended to run in a background daemon thread.
    """
    theta_deg = 0.0   # set by pol search below; initialised here so except blocks can reference it
    dphi_deg  = 0.0
    pre_n  = int(0.500 * sample_rate)   # 500 ms before detection — sufficient for SPD path
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

    # If the ring buffer was reset (source switch) this launch is stale — exit silently.
    if ring_gen_fn is not None and ring_gen_fn() != ring_gen:
        return

    ring_pos, abs_sample = ring_state_fn()

    iq_seg_raw = _read_ring(iq_ring, ring_pos, abs_sample,
                            detect_sample - pre_n, pre_n + post_n)

    if iq_seg_raw.size == 0:
        return

    # Separate H and V, mix both to baseband (fc_hz → 1500 Hz audio centre).
    shift_hz = fc_hz - _TARGET_FC_HZ
    _seg_len  = iq_seg_raw.shape[0]
    _t_mix    = np.arange(_seg_len, dtype=np.float64)
    _mix      = np.exp(-2j * np.pi * shift_hz * _t_mix / sample_rate).astype(np.complex64)

    if dual_pol and iq_seg_raw.ndim == 2 and iq_seg_raw.shape[1] >= 2:
        iq_h_mixed = (iq_seg_raw[:, 0] * _mix).astype(np.complex64)
        iq_v_mixed = (iq_seg_raw[:, 1] * _mix).astype(np.complex64)
        # Decimate to CH_SAMPLE_RATE (12 kHz) before pol search.
        # At 48 kHz the noise bandwidth is 4× wider than at 12 kHz, making the
        # squared-spectrum metric noise-dominated.  Decimating first matches the
        # noise bandwidth that _eval_pair_metric / _polarization_search expect.
        from .processing import CH_DETECT_SIZE, CH_SAMPLE_RATE
        _factor = max(1, sample_rate // CH_SAMPLE_RATE)
        if _factor > 1:
            _h_i = decimate(np.real(iq_h_mixed).astype(np.float64), _factor, zero_phase=False)
            _h_q = decimate(np.imag(iq_h_mixed).astype(np.float64), _factor, zero_phase=False)
            _v_i = decimate(np.real(iq_v_mixed).astype(np.float64), _factor, zero_phase=False)
            _v_q = decimate(np.imag(iq_v_mixed).astype(np.float64), _factor, zero_phase=False)
            _iq_h_lp = (_h_i + 1j * _h_q).astype(np.complex64)
            _iq_v_lp = (_v_i + 1j * _v_q).astype(np.complex64)
            _pre_lp  = pre_n // _factor
        else:
            _iq_h_lp, _iq_v_lp, _pre_lp = iq_h_mixed, iq_v_mixed, pre_n
        # Evaluate only the CH_DETECT_SIZE samples starting at the detection moment.
        _ps = _pre_lp
        _pe = min(len(_iq_h_lp), _pre_lp + CH_DETECT_SIZE)
        theta_deg, dphi_deg, _ = _polarization_search(
            _iq_h_lp[_ps:_pe], _iq_v_lp[_ps:_pe], CH_SAMPLE_RATE
        )
        # Reconstruct full-length combined IQ at original sample rate using found theta/dphi.
        _cos_t = float(np.cos(np.radians(theta_deg)))
        _sin_t = float(np.sin(np.radians(theta_deg)))
        _phase = np.exp(1j * np.radians(dphi_deg)).astype(np.complex64)
        iq_seg = (_cos_t * iq_h_mixed + _sin_t * _phase * iq_v_mixed).astype(np.complex64)
    else:
        iq_h_mixed = ((iq_seg_raw[:, 0] if iq_seg_raw.ndim == 2 else iq_seg_raw) * _mix
                      ).astype(np.complex64)
        iq_seg     = iq_h_mixed
        theta_deg  = 0.0
        dphi_deg   = 0.0

    # Zero-pad
    pad = np.zeros(pad_n, dtype=np.complex64)
    iq_padded = np.concatenate([pad, iq_seg, pad])
    iq_mixed  = iq_padded   # already mixed above

    # Decimate to 12 kHz — both I and Q for the SPD complex path.
    # Real audio (I only) is kept for the jt9 fallback and SNR estimation.
    i_dec = decimate(np.real(iq_mixed).astype(np.float64), _DECIMATE_FACTOR, zero_phase=False)
    q_dec = decimate(np.imag(iq_mixed).astype(np.float64), _DECIMATE_FACTOR, zero_phase=False)
    audio = i_dec.astype(np.float32)   # real 12 kHz audio, carrier at _TARGET_FC_HZ

    # SNR estimate uses real audio before any normalisation (2500 Hz BW reference)
    est_snr = _estimate_snr_db(audio)

    # ── SPD: complex IQ + audio BPF ─────────────────────────────────────────
    # Mix from _TARGET_FC_HZ (1500 Hz) to 0 Hz baseband, apply BPF, run SPD.
    _n12  = len(i_dec)
    _t12  = np.arange(_n12, dtype=np.float64) / _DECODE_RATE
    _iq12 = ((i_dec + 1j * q_dec) *
             np.exp(-2j * np.pi * _TARGET_FC_HZ * _t12)).astype(np.complex64)
    _iq12_bpf = _apply_audio_bpf(_iq12)
    spd_msg, spd_snr, spd_navg, spd_fest, spd_xmax = msk144_spd_decode(
        complex_baseband=_iq12_bpf)

    # Normalise real audio for jt9 fallback and WAV archive
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak * 0.9

    t_sec  = t_in_window
    # Logging frequency in kHz: tuned centre (kHz) plus offset of the detected
    # carrier from DC (kHz), minus the 1500 Hz nominal MSK audio centre so the
    # number matches **USB dial / contest** convention (N1MM, GridTracker, …).
    # That only matches operator expectations if ``center_freq_mhz`` and ``fc_hz``
    # share the same IQ reference and the USB + 1500 Hz geometry matches reality.
    radio_khz = center_freq_mhz * 1000.0 + (fc_hz - _TARGET_FC_HZ) / 1000.0

    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    launch_ts = detect_ts or datetime.now(timezone.utc).strftime('%Y-%m-%d_%H:%M:%S.%f')[:21]
    _ts_file  = launch_ts[:10].replace('-', '') + '_' + launch_ts[11:19].replace(':', '') + 'Z'

    # MSK144 period timestamp: WSJT-X convention is the START of the Rx
    # period (the 15-second boundary that contains the ping).  A ping
    # received at 09:40:12 is in the period 09:40:00–09:40:15; both
    # WSJT-X and MAP144 log it as 09:40:00.  This matches WSJT-X's
    # ALL.TXT trigger-time-plus-dt convention floored to the 15-second
    # boundary, and the way compare_decoders.parse_wsjtx() canonicalises
    # WSJT-X timestamps — so periods line up at ±0 tolerance.
    #
    # (Prior convention was period END / start-of-next-window;
    #  changed 2026-05-11 to floor instead of ceiling.  Old data in
    #  decodes.jsonl from before this change is offset +15 s.)
    #
    # PERIOD_SLIP correction: anchor the period to the IQ sample index, not to
    # ``launch_ts``.  ``detect_sample`` is the absolute ring sample of the burst
    # and ``iq_t0_wall`` is the wall-clock for ring sample 0 (refreshed every
    # input chunk in process_iq_data).  The resulting epoch is sample-accurate
    # and immune to pipeline / queue latency — the dominant cause of slips with
    # a wall-clock-anchored estimator.  The ``None`` branch keeps the legacy
    # heuristic for the analysis-engine click path, which has no ring epoch.
    try:
        _launch_dt = datetime.strptime(launch_ts, "%Y-%m-%d_%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        _launch_dt = datetime.strptime(launch_ts, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    if iq_t0_wall is not None:
        _ping_epoch_est = iq_t0_wall + detect_sample / sample_rate
    else:
        _ping_epoch_est = _launch_dt.timestamp() - pre_n / sample_rate
    _period_epoch   = int(_ping_epoch_est // 15) * 15                # period START
    _period_dt      = datetime.fromtimestamp(_period_epoch, tz=timezone.utc)
    period_ts       = _period_dt.strftime("%Y-%m-%d_%H:%M:%S")

    def _log_launch(outcome: str, message: str = "", jt9_snr=None, jt9_line: str = ""):
        entry = {
            "timestamp": launch_ts,
            "t_sec":     round(t_sec, 3),
            "radio_khz":    int(round(radio_khz)),
            "outcome":   outcome,   # "decoded" | "no_decode" | "timeout" | "error"
            "message":   message,
            "jt9_snr_db": jt9_snr,
            "jt9_line":  jt9_line,
            "theta_deg": round(theta_deg, 1) if dual_pol else None,
        }
        # DSP-state snapshot at the moment of launch — used by offline analysis
        # (analyze_launches.py / compare_decoders.py) to separate real-signal
        # launches from threshold-edge noise spikes without re-running the
        # detector.  Absent when older runs predate the metric capture.
        if launch_metrics:
            entry.update(launch_metrics)
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

            if spd_msg is not None:
                # ── SPD decoded ───────────────────────────────────────────────
                full_msg = spd_msg
                snr      = spd_snr if spd_snr is not None else est_snr
                jt9_line = (f"[SPD navg={spd_navg} xmax={spd_xmax:.2f}"
                            f" fest={spd_fest:+.0f}Hz]")

                msg_safe  = re.sub(r'[^A-Za-z0-9]+', '_', full_msg).strip('_')
                rf_int    = int(round(radio_khz))
                save_name = f"{_ts_file}_{rf_int}kHz_{msg_safe}.wav"
                shutil.move(tmp_path, str(out_dir / save_name))
                _pol_str = f"  θ={theta_deg:.0f}°" if dual_pol else ""
                print(f"[MSK144 SPD]  t={t_sec:.2f}s  radio={radio_khz:.3f} kHz{_pol_str}"
                      f"  det={det_source}  navg={spd_navg}  {full_msg}", flush=True)

                if decode_queue is not None:
                    decode_queue.put({
                        'marker_id': marker_id,
                        'outcome':   'decoded',
                        'decoder':   'spd',
                        'decoded':   True,
                        'message':   full_msg,
                        't_sec':     t_sec,
                        'radio_khz': radio_khz,
                        'jt9_snr':   snr,
                        'utc_time':  _period_dt.strftime('%H:%M:%S'),
                        'audio':     audio.copy(),
                        'theta_deg': round(theta_deg, 1),
                        'dphi_deg':  round(dphi_deg, 1),
                    })

                decode_entry = {
                    "timestamp":  launch_ts,
                    "period_ts":  period_ts,
                    "t_sec":      round(t_sec, 3),
                    "radio_khz":  int(round(radio_khz)),
                    "message":    full_msg,
                    "jt9_snr_db": snr,
                    "est_snr_db": est_snr,
                    "jt9_line":   jt9_line,
                    "theta_deg":  round(theta_deg, 1) if dual_pol else None,
                }
                with open(out_dir / "decodes.jsonl", "a") as lf:
                    lf.write(json.dumps(decode_entry) + "\n")

                _log_launch("decoded", message=full_msg, jt9_snr=snr, jt9_line=jt9_line)

            else:
                # ── jt9 fallback (SPD missed — handles longer bursts) ─────────
                #
                # Extend the pre-burst noise window to 1500 ms before running jt9.
                # jt9's SNR estimate needs a clean noise reference; 500 ms gives only
                # ~9 PSD frames while 1500 ms gives ~34.  The ring buffer still holds
                # this history — re-read it now that SPD has confirmed we need jt9.
                # theta_deg/dphi_deg from the pol search are already determined and
                # reused as-is; we skip the pol search on the extended buffer.
                _jt9_pre_used = pre_n   # updated below if extended buffer is read
                _pre_n_jt9 = int(1.500 * sample_rate)
                _rp2, _as2 = ring_state_fn()
                _iq_ext = _read_ring(iq_ring, _rp2, _as2,
                                     detect_sample - _pre_n_jt9, _pre_n_jt9 + post_n)
                if _iq_ext.size > 0:
                    _slen2  = _iq_ext.shape[0]
                    _t2     = np.arange(_slen2, dtype=np.float64)
                    _mix2   = np.exp(-2j * np.pi * shift_hz * _t2 / sample_rate
                                     ).astype(np.complex64)
                    if dual_pol and _iq_ext.ndim == 2 and _iq_ext.shape[1] >= 2:
                        _cos_t2 = float(np.cos(np.radians(theta_deg)))
                        _sin_t2 = float(np.sin(np.radians(theta_deg)))
                        _ph2    = np.exp(1j * np.radians(dphi_deg)).astype(np.complex64)
                        _seg2   = (_cos_t2 * (_iq_ext[:, 0] * _mix2) +
                                   _sin_t2 * _ph2 * (_iq_ext[:, 1] * _mix2)
                                   ).astype(np.complex64)
                    else:
                        _ch2  = _iq_ext[:, 0] if _iq_ext.ndim == 2 else _iq_ext
                        _seg2 = (_ch2 * _mix2).astype(np.complex64)
                    _pad2     = np.zeros(pad_n, dtype=np.complex64)
                    _iq_pad2  = np.concatenate([_pad2, _seg2, _pad2])
                    _i2       = decimate(np.real(_iq_pad2).astype(np.float64),
                                         _DECIMATE_FACTOR, zero_phase=False)
                    audio_jt9 = _i2.astype(np.float32)
                    est_snr   = _estimate_snr_db(audio_jt9)
                    _jt9_pre_used = _pre_n_jt9  # extended buffer in use
                    # Overwrite the WAV with the extended-noise version for jt9
                    with wave.open(tmp_path, 'wb') as _wf2:
                        _wf2.setnchannels(1)
                        _wf2.setsampwidth(2)
                        _wf2.setframerate(_DECODE_RATE)
                        _wf2.writeframes((audio_jt9 * 32767).astype(np.int16).tobytes())

                cmd = (jt9_args if jt9_args is not None else JT9_BASE_ARGS) + [tmp_path]
                with _JT9_SEMAPHORE:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20.0)

                decoded = ""
                for line in result.stdout.splitlines():
                    s = line.strip()
                    if s and not s.startswith('<') and not s.startswith('EOF'):
                        decoded = s
                        break

                if not decoded and result.returncode != 0:
                    print(f"[jt9 error]  rc={result.returncode}  fc={fc_hz:.0f}Hz"
                          f"  stdout: {result.stdout.strip()!r}", flush=True)

                if decoded:
                    # jt9 MSK144 output: "HHMMSS  SNR  dt  freq  mode  MESSAGE..."
                    tokens   = decoded.split()
                    full_msg = " ".join(tokens[5:]) if len(tokens) >= 6 else decoded
                    bare_msg = full_msg
                    try:
                        jt9_snr = int(tokens[1]) if len(tokens) >= 2 else None
                    except (ValueError, IndexError):
                        jt9_snr = None
                    # jt9 cannot compute SNR from a short burst WAV (no noise baseline).
                    # Use our pre-burst estimate when jt9 gives nothing.
                    snr = jt9_snr if jt9_snr is not None else est_snr

                    # Refine period using jt9's dt field (exact ping time in WAV).
                    # dt is seconds from WAV start; the burst sits (_jt9_pre_used + pad_n)
                    # samples after WAV start.  When the IQ ring epoch is available we use
                    # it (exact); otherwise fall back to the launch-anchored form.
                    try:
                        _dt_jt9 = float(tokens[2])
                        _wav_offset = (_jt9_pre_used + pad_n) / sample_rate
                        if iq_t0_wall is not None:
                            _burst_epoch = iq_t0_wall + detect_sample / sample_rate
                        else:
                            _burst_epoch = _launch_dt.timestamp()
                        _ping_epoch_jt9 = _burst_epoch - _wav_offset + _dt_jt9
                        # period START (floor); see SPD-path comment above
                        # for the convention rationale.
                        _period_epoch   = int(_ping_epoch_jt9 // 15) * 15
                        _period_dt      = datetime.fromtimestamp(_period_epoch, tz=timezone.utc)
                        period_ts       = _period_dt.strftime("%Y-%m-%d_%H:%M:%S")
                    except (ValueError, IndexError, TypeError):
                        pass  # keep pre-burst-window heuristic

                    msg_safe  = re.sub(r'[^A-Za-z0-9]+', '_', full_msg).strip('_')
                    rf_int    = int(round(radio_khz))
                    save_name = f"{_ts_file}_{rf_int}kHz_{msg_safe}.wav"
                    shutil.move(tmp_path, str(out_dir / save_name))
                    _pol_str = f"  θ={theta_deg:.0f}°" if dual_pol else ""
                    print(f"[MSK144 DECODE]  t={t_sec:.2f}s  radio={radio_khz:.3f} kHz{_pol_str}"
                          f"  det={det_source}  {decoded}", flush=True)

                    if decode_queue is not None:
                        decode_queue.put({
                            'marker_id': marker_id,
                            'outcome':   'decoded',
                            'decoder':   'jt9',
                            'decoded':   True,
                            'message':   bare_msg,
                            't_sec':     t_sec,
                            'radio_khz': radio_khz,
                            'jt9_snr':   snr,
                            'utc_time':  _period_dt.strftime('%H:%M:%S'),
                            'audio':     audio.copy(),
                            'theta_deg': round(theta_deg, 1),
                            'dphi_deg':  round(dphi_deg, 1),
                        })

                    decode_entry = {
                        "timestamp":  launch_ts,
                        "period_ts":  period_ts,
                        "t_sec":      round(t_sec, 3),
                        "radio_khz":  int(round(radio_khz)),
                        "message":    bare_msg,
                        "jt9_snr_db": snr,
                        "est_snr_db": est_snr,
                        "jt9_line":   decoded,
                        "theta_deg":  round(theta_deg, 1) if dual_pol else None,
                    }
                    with open(out_dir / "decodes.jsonl", "a") as lf:
                        lf.write(json.dumps(decode_entry) + "\n")

                    _log_launch("decoded", message=bare_msg, jt9_snr=snr, jt9_line=decoded)
                else:
                    _log_launch("no_decode")
                    if decode_queue is not None:
                        decode_queue.put({'marker_id': marker_id, 'outcome': 'no_decode',
                                          'theta_deg': round(theta_deg, 1) if dual_pol else None,
                                          'audio': audio.copy()})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        print(f"[MSK144]  jt9 timeout (>20s) at t={t_sec:.2f}s  rf={radio_khz:.3f} kHz", flush=True)
        _log_launch("timeout")
        if decode_queue is not None:
            decode_queue.put({'marker_id': marker_id, 'outcome': 'timeout',
                              'theta_deg': round(theta_deg, 1) if dual_pol else None})
    except Exception as exc:
        print(f"[MSK144]  extract_and_decode error: {exc}", flush=True)
        _log_launch("error")
        if decode_queue is not None:
            decode_queue.put({'marker_id': marker_id, 'outcome': 'error',
                              'theta_deg': round(theta_deg, 1) if dual_pol else None})
        import traceback
        traceback.print_exc()
