#!/usr/bin/env python3
"""Diagnose the 6 dB sensitivity gap between direct and pipeline decode paths.

Tests three paths at identical labeled SNR:
  A) Direct: complex AWGN → msk144decodeframe  (baseline)
  B) Via Hilbert: real(signal) + real_AWGN → Hilbert+mix → msk144decodeframe
  C) Via jt9: real(signal) + real_AWGN at 12kHz → jt9 (no decimation)
  D) Via jt9: 48kHz IQ → decimate → jt9 (full pipeline)

If A ≈ B, the Hilbert path is fine and the gap is elsewhere.
If A >> B, the Hilbert path itself is losing SNR.
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, decimate

sys.path.insert(0, str(Path(__file__).parent))
from map144_app.msk144_decode import msk144decodeframe, NSPM, FS
from map144_app.detection import JT9_BASE_ARGS
from generate_msk144 import generate_ping_msk144code

BW  = 2500.0
FC  = 1500.0   # carrier Hz for passband representation

# ── Build a reference flat MSK144 frame at complex baseband ──────────────────

def _get_ref_frame(msg: str = 'CQ K2GV FN30') -> np.ndarray:
    """Generate one clean timing-aligned complex baseband MSK frame."""
    result = subprocess.run(['msk144code', msg], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    sym_lines = [l for l in lines if set(l.strip()) <= {'0','1'} and len(l.strip()) >= 60]
    bits = [int(b) for line in sym_lines[:2] for b in line.strip()]
    assert len(bits) == 144, f"Expected 144 bits, got {len(bits)}"

    chips = np.array([1 if b else -1 for b in bits], dtype=np.float64)
    sps   = FS / 2000.0            # 6 samples per chip at 12kHz
    n_burst = int(np.ceil(144 * sps))   # 864 = NSPM

    phase = np.zeros(n_burst, dtype=np.float64)
    running = 0.0
    for i, chip in enumerate(chips):
        s = int(round(i * sps))
        e = min(int(round((i + 1) * sps)), n_burst)
        n = e - s
        if n <= 0: continue
        k = np.arange(n, dtype=np.float64)
        pulse = np.sin(np.pi * k / sps)
        norm  = pulse.sum()
        steps = chip * (np.pi / 2.0) * pulse / norm if norm > 0 else np.zeros(n)
        phase[s:e] = running + np.cumsum(steps)
        running = float(phase[e - 1])

    return np.exp(1j * phase).astype(np.complex64)


_REF = _get_ref_frame()    # unit-amplitude complex baseband MSK frame


# ── Path A: direct complex inject ────────────────────────────────────────────

def test_direct(snr_db: float, seed: int) -> bool:
    """Complex AWGN + msk144decodeframe — the cleanest possible baseline."""
    rng = np.random.default_rng(seed)
    snr_lin = 10**(snr_db / 10.0)
    # noise_sigma = 1.0 per component; amp calibrated for WSJT-X convention
    amp = np.sqrt(2.0 * snr_lin * BW / FS)
    noise = (rng.standard_normal(NSPM) + 1j * rng.standard_normal(NSPM)).astype(np.complex64)
    frame = _REF * amp + noise
    msg, _ = msk144decodeframe(frame)
    return msg is not None


# ── Path B: real audio at 12kHz → Hilbert+mix → msk144decodeframe ────────────

def test_hilbert(snr_db: float, seed: int) -> bool:
    """Convert complex frame to real audio, Hilbert back, decode."""
    rng = np.random.default_rng(seed)
    snr_lin = 10**(snr_db / 10.0)
    amp  = np.sqrt(2.0 * snr_lin * BW / FS)

    # Real bandpass signal at FC Hz
    t = np.arange(NSPM, dtype=np.float64) / FS
    audio_sig = np.real(_REF * amp * np.exp(1j * 2 * np.pi * FC * t)).astype(np.float64)

    # Real AWGN with same sigma as complex path (sigma_real = 1)
    audio_noise = rng.standard_normal(NSPM)
    audio = audio_sig + audio_noise

    # Hilbert + mix back to baseband
    analytic = hilbert(audio)
    cbase = (analytic * np.exp(-1j * 2 * np.pi * FC * t)).astype(np.complex64)

    msg, _ = msk144decodeframe(cbase)
    return msg is not None


# ── Path B2: larger buffer (512ms noise before/after) → same Hilbert test ────

def test_hilbert_buffer(snr_db: float, seed: int) -> bool:
    """Same as B but in a 1500ms buffer with signal centred — avoids edge effects."""
    rng = np.random.default_rng(seed)
    snr_lin = 10**(snr_db / 10.0)
    amp = np.sqrt(2.0 * snr_lin * BW / FS)

    pre_n  = int(0.5 * FS)   # 500 ms noise before signal
    post_n = int(0.5 * FS)   # 500 ms noise after signal
    total  = pre_n + NSPM + post_n

    audio = rng.standard_normal(total)

    t_frame = np.arange(NSPM, dtype=np.float64) / FS
    audio_sig = np.real(_REF * amp * np.exp(1j * 2 * np.pi * FC * t_frame))
    audio[pre_n : pre_n + NSPM] += audio_sig

    # Hilbert on full buffer, extract frame region
    t_big = np.arange(total, dtype=np.float64) / FS
    analytic = hilbert(audio)
    cbase = (analytic * np.exp(-1j * 2 * np.pi * FC * t_big)).astype(np.complex64)

    frame = cbase[pre_n : pre_n + NSPM]
    msg, _ = msk144decodeframe(frame)
    return msg is not None


# ── Path C: jt9 on 12kHz real audio (no decimation) ─────────────────────────

def test_jt9_12k(snr_db: float, seed: int) -> bool:
    """jt9 on 12kHz real audio with precisely calibrated SNR — no 48kHz chain."""
    rng = np.random.default_rng(seed)
    snr_lin = 10**(snr_db / 10.0)
    sigma = 0.01    # real audio noise sigma

    # Amplitude calibrated for WSJT-X convention: SNR = A²/(2*sigma²*BW/FS)
    amp = sigma * np.sqrt(2.0 * snr_lin * BW / FS)

    pad_n  = int(0.100 * FS)
    pre_n  = int(0.500 * FS)
    post_n = int(1.200 * FS)
    total  = pad_n + pre_n + post_n + pad_n

    # Signal at pre_n position
    t_frame = np.arange(NSPM, dtype=np.float64) / FS
    audio_sig = np.real(_REF * amp * np.exp(1j * 2 * np.pi * FC * t_frame)).astype(np.float32)

    audio = (rng.standard_normal(total) * sigma).astype(np.float32)
    audio[pad_n + pre_n : pad_n + pre_n + NSPM] += audio_sig

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(FS))
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        result = subprocess.run(JT9_BASE_ARGS + [tmp_path],
                                capture_output=True, text=True, timeout=15)
        for line in result.stdout.splitlines():
            if len(line.split()) >= 6:
                return True
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Path D: 48kHz IQ → decimate → jt9 (full pipeline) ───────────────────────

def test_jt9_48k(snr_db: float, seed: int) -> bool:
    """Full pipeline: 48kHz complex → decimate to 12kHz → jt9."""
    rng = np.random.default_rng(seed)
    noise_sigma = 0.01
    ping_h, _, _ = generate_ping_msk144code(
        'CQ K2GV FN30',
        center_hz=FC,
        delay_s=0.500,
        snr_db=snr_db,
        noise_sigma=noise_sigma,
        flat_frame=True,
        rng=rng,
        sample_rate=48000,
    )
    total_iq = int((0.500 + 1.200) * 48000)
    if len(ping_h) < total_iq:
        ping_h = np.pad(ping_h, (0, total_iq - len(ping_h)))
    else:
        ping_h = ping_h[:total_iq]

    noise_iq = ((rng.standard_normal(total_iq) +
                 1j * rng.standard_normal(total_iq)) * noise_sigma).astype(np.complex64)
    iq_noisy = ping_h + noise_iq

    audio = decimate(np.real(iq_noisy).astype(np.float64), 4,
                     zero_phase=False).astype(np.float32)

    pad_n = int(0.100 * FS)
    audio = np.concatenate([np.zeros(pad_n, np.float32), audio, np.zeros(pad_n, np.float32)])

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(FS))
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        result = subprocess.run(JT9_BASE_ARGS + [tmp_path],
                                capture_output=True, text=True, timeout=15)
        for line in result.stdout.splitlines():
            if len(line.split()) >= 6:
                return True
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Sweep runner ─────────────────────────────────────────────────────────────

def sweep(name: str, fn, snr_range, n_trials: int = 50):
    print(f'\n=== {name} ===')
    print(f'  {"SNR":>6}  {"p(decode)":>10}')
    print(f'  {"─"*6}  {"─"*10}')
    for snr_db in snr_range:
        hits = sum(fn(snr_db, seed) for seed in range(n_trials))
        print(f'  {snr_db:+6.1f}  {hits/n_trials:10.2f}')


if __name__ == '__main__':
    SNR = list(np.arange(-6, 7, 1.0))
    N   = 50

    sweep('A: direct complex → msk144decodeframe',  test_direct,         SNR, N)
    sweep('B: Hilbert@NSPM  → msk144decodeframe',   test_hilbert,        SNR, N)
    sweep('B2: Hilbert@buffer → msk144decodeframe',  test_hilbert_buffer, SNR, N)
    sweep('C: jt9 at 12kHz (no decimation)',         test_jt9_12k,        SNR, N)
    sweep('D: jt9 via 48kHz→12kHz decimation',       test_jt9_48k,        SNR, N)
