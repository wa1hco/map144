#!/home/jeff/ham/map144/.venv/bin/python3
"""Generate a combined 48 kHz complex MSK144 test stream for pipeline testing.

This tool produces a synthetic complex IQ test vector that exercises the full
detection and decode pipeline of map144 (processing.py → detection.py → jt9).
It supports two source modes and a rich set of signal-conditioning steps.

Operating modes (selected by ``--count``)
-----------------------------------------
Mode 1 — msk144code ping generation (``--count N``, default)
    ``N`` pings are generated using the ``msk144code`` binary (from the WSJT-X
    build tree) to obtain exact 144-chip symbol sequences.  Each ping's
    frequency, time offset, SNR, and duration are chosen randomly within
    configurable ranges and encoded into a self-describing 10-character MSK144
    free-text message of the form ``A[P/N][ff][ttt][P/N][dd]``:

    ======= ========= =============================================
    Field   Width     Meaning
    ======= ========= =============================================
    A       1 char    literal prefix
    [P/N]   1 char    sign of freq (P=positive/zero, N=negative)
    [ff]    2 chars   |freq| in kHz, 00–17
    [ttt]   3 chars   time in deciseconds, 005–145
    [P/N]   1 char    sign of SNR (P=positive/zero, N=negative)
    [dd]    2 chars   |SNR| in dB, 03–10
    ======= ========= =============================================

    The chips are modulated with phase-continuous MSK at 2000 baud (±500 Hz
    tones), wrapped in a Gamma meteor-trail envelope, and placed at the exact
    channelizer channel frequency so the squared-spectrum detector (±1 kHz
    tone windows) sees the signal at DC in the correct channel row.

Mode 2 — existing WAV files (``--count 0``)
    WAV files are loaded from ``--input-dir`` (default ``./MSK144``), placed at
    random centre frequencies within ±``--freq-range-hz`` and random time
    offsets up to ``--max-delay-s``.


Frequency placement and the channelizer
----------------------------------------
Channel k captures signals from stations whose USB dial is at
``calling_freq + k × 1000 Hz``.

In USB radio the dial is the suppressed carrier; signal energy is 1500 Hz above
the dial.  A station with dial at ``calling_freq + k × 1000 Hz`` transmits energy
at ``calling_freq + k × 1000 + 1500 Hz``.  In the IQ stream centred at
``calling_freq``, that energy appears at ``k × 1000 + 1500 Hz`` from DC.
Channel k's NCO is placed exactly there (see channelizer.py).

For the squaring detector to find the MSK144 tone pair at ±1000 Hz, the signal
must land at DC in the channel output.  This tool therefore places each test
signal at::

    target_hz = freq_khz × 1000 + 1500   [Hz from IQ DC, pan = calling_freq]

which matches channel ``freq_khz``'s NCO exactly.

After detection, ``extract_and_decode`` (detection.py) mixes the carrier from
``fc_hz`` to 1500 Hz so jt9 receives audio centred at 1500 Hz::

    shift_hz = fc_hz − 1500

For an on-channel signal: ``fc_hz = freq_khz × 1000 + 1500``, so
``shift_hz = freq_khz × 1000`` and the carrier lands at 1500 Hz in the audio.

The frequency logged for the operator::

    radio_khz = pan_center_khz + (fc_hz − 1500) / 1000 = pan_center_khz + freq_khz

This is the station's USB dial offset from ``calling_freq`` in kHz — the number
on the operator's radio.

Nyquist edge guard
~~~~~~~~~~~~~~~~~~
Signals at ±18–24 kHz alias through the polyphase filter bank edges into ch0/ch47,
producing spurious triggers.  ``FREQ_MIN/MAX_KHZ`` is clamped to ±17 kHz.

Signal conditioning pipeline (Mode 1)
--------------------------------------
1. **msk144code** — get 144 BPSK chips for the message text.
2. **Tile + modulate** — tile chips to cover the echo duration and modulate once
   with phase-continuous MSK (2000 baud, ±500 Hz) so burst boundaries are
   seamless.  Tiling the *waveform* instead would create phase jumps at every
   72 ms burst boundary that corrupt jt9 correlation.
3. **Gamma envelope** — multiply by ``e·(t/τ)·exp(−t/τ)`` to simulate a
   meteor trail with decay constant τ = width_ms / 1000 s.
4. **Frequency shift** — mix to ``freq_khz × 1000 + 1500 Hz`` via complex LO,
   matching the channelizer NCO for channel ``freq_khz``.
5. **SNR scaling** — amplitude calibrated so peak SNR matches the encoded value
   (WSJT-X 2500 Hz reference bandwidth convention).
6. **AWGN** — complex Gaussian noise added at ``--noise-floor-dbfs``.

Outputs
-------
Stereo (or 4-channel) IEEE float32 WAV, left=I, right=Q, auto-named by
timestamp and signal count (e.g. ``msk144_20260411_103800_10sig.wav``).
A JSON manifest records each signal's parameters for use by ``compare_msk144.py``.

Diagnostic display (``plot_wav_diagnostics``)
---------------------------------------------
An interactive matplotlib figure shows one row per source plus an optional
combined-output row.  Each row contains a periodogram (time × frequency) and
a median noise-floor curve.  In interactive mode the periodogram is animated
frame-by-frame to simulate real-time receive flow using the streaming median
estimator described below.  Baseline and gain sliders control the colour scale
on all images simultaneously.

Streaming median estimator
--------------------------
The noise floor tracker (``_estimate_median_energy_bins_db``) uses a
constant-cost per-bin sign-SGD update: each frame, the per-bin estimate is
nudged up or down by ``MEDIAN_UPDATE_STEP_DB = 0.0625 dB`` depending on
whether the current frame exceeds the estimate.  The data is replayed
``MEDIAN_REPLAY_PASSES = 8`` times to bootstrap the estimate from a cold start.
A ±``MEDIAN_FREQ_WINDOW_BINS = 7`` bin median smoother is applied after each
update to suppress salt-and-pepper noise in the estimate.  Edge frames that sit
near the PSD numerical floor (zero-filled WAV padding) are excluded from updates
via ``_median_update_mask``.

Key constants
-------------
FREQ_MIN/MAX_KHZ            : ±17 kHz — signal placement range (within monitored band)
MSK144SIM_GENERATE_SNR_DB   : 55 dB   — msk144sim SNR for reference ping generation
PING_EXTRACT_START_S        : 1.0 s   — where first ping lands in msk144sim output
PING_EXTRACT_DURATION_S     : 1.0 s   — window extracted per ping
MSK144_SNR_BW_HZ            : 2500 Hz — WSJT-X SNR reference bandwidth
DIAGNOSTIC_NFFT             : 512     — FFT size used for all diagnostic spectrograms
DIAGNOSTIC_CHUNK_SAMPLES    : 504     — chunk size emulating real-time receive cadence
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
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

# ── msk144sim ping generation ──────────────────────────────────────────────
MSK144SIM_TR_PERIOD = 15         # seconds; msk144sim TR period argument
MSK144SIM_AUDIO_CENTER_HZ = 1500.0
MSK144SIM_SAMPLE_RATE = 12000
PING_EXTRACT_START_S = 1.0       # first ping lands at t=1 s in msk144sim output
PING_EXTRACT_DURATION_S = 1.0    # one second of ping audio to extract

# Parameter encoding ranges (match message format "A[P/N][ff][ttt][P/N][dd]")
# ±17 kHz keeps signals within the reliably monitored channelizer band.
# Channels within 4 of the Nyquist bin (ch 24 = ±24 kHz) are edge-skipped by
# the detector.  Signals at ±18–24 kHz can also alias through the polyphase
# filter bank edges into ch0/ch47, causing spurious triggers at DC.
FREQ_MIN_KHZ = -17
FREQ_MAX_KHZ = +17
TIME_MIN_DS  =   5               # deciseconds ( 0.5 s)
TIME_MAX_DS  = 145               # deciseconds (14.5 s)
SNR_MIN_DB   =   0
SNR_MAX_DB   =  +6
WIDTH_MIN_MS = 200
WIDTH_MAX_MS = 500
WIDTH_CLAMP_MIN_MS = 10          # avoid division-by-zero inside msk144sim
MSK144_SNR_BW_HZ = 2500.0        # WSJT-X reference bandwidth for MSK144 SNR
MSK144SIM_GENERATE_SNR_DB = 55   # SNR passed to msk144sim.  55 dB is near the
                                 # int16 clipping limit (≈ 58 dB); going higher
                                 # would saturate iwave = 30*wave in msk144sim.


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


def write_iq_wav(path: Path, iq: np.ndarray, sample_rate: int,
                 scale: float | None = None,
                 iq_ch1: np.ndarray | None = None) -> None:
    """Write complex IQ as float32 WAV.

    Single-channel (iq_ch1=None): stereo WAV, left=I, right=Q.
    Dual-channel (iq_ch1 provided): 4-channel WAV, ch0_I, ch0_Q, ch1_I, ch1_Q.

    The 4-channel format is backward-compatible in the sense that single-channel
    readers will still open the file; only readers that check n_channels==4 will
    decode ch1 (runtime._load_wav_complex does this automatically).

    If *scale* is None (default) the array is peak-normalised to 0.95 full
    scale across all channels.  Pass an explicit scale (e.g. 1.0) to write
    with a fixed gain so that calibrated noise floors are preserved.

    Float32 format eliminates the int16 quantization noise floor (~−134 dBFS)
    that otherwise appears as wideband noise across the full ±24 kHz spectrum
    when signal amplitudes are small relative to ±1.0 full scale.  The reader
    in runtime._load_wav_complex detects float32 samples automatically.
    """
    if iq_ch1 is not None:
        # 4-channel interleave: ch0_I, ch0_Q, ch1_I, ch1_Q per frame
        n = min(len(iq), len(iq_ch1))
        interleaved = np.empty(n * 4, dtype=np.float32)
        interleaved[0::4] = np.real(iq[:n])
        interleaved[1::4] = np.imag(iq[:n])
        interleaved[2::4] = np.real(iq_ch1[:n])
        interleaved[3::4] = np.imag(iq_ch1[:n])
        n_channels = 4
    else:
        interleaved = np.empty(iq.size * 2, dtype=np.float32)
        interleaved[0::2] = np.real(iq)
        interleaved[1::2] = np.imag(iq)
        n_channels = 2

    if scale is None:
        peak = float(np.max(np.abs(interleaved))) if interleaved.size else 1.0
        scale = 0.95 / peak if peak > 0 else 1.0
    pcm = np.clip(interleaved * scale, -1.0, 1.0).astype(np.float32)

    # Write RIFF/WAVE with format code 3 (IEEE float32) using manual packing.
    # Python's wave module always writes format code 1 (PCM) — using it here
    # would cause _load_wav_complex to reinterpret float32 bits as int32.
    import struct as _struct
    data_bytes  = pcm.tobytes()
    sampwidth   = 4                  # float32 = 4 bytes
    fmt_code    = 3                  # WAVE_FORMAT_IEEE_FLOAT
    byte_rate   = sample_rate * n_channels * sampwidth
    block_align = n_channels * sampwidth
    with open(str(path), 'wb') as f:
        data_size  = len(data_bytes)
        chunk_size = 36 + data_size
        f.write(b'RIFF')
        f.write(_struct.pack('<I', chunk_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(_struct.pack('<I', 16))
        f.write(_struct.pack('<H', fmt_code))
        f.write(_struct.pack('<H', n_channels))
        f.write(_struct.pack('<I', sample_rate))
        f.write(_struct.pack('<I', byte_rate))
        f.write(_struct.pack('<H', block_align))
        f.write(_struct.pack('<H', sampwidth * 8))
        f.write(b'data')
        f.write(_struct.pack('<I', data_size))
        f.write(data_bytes)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(',') if item.strip()]


# ── Ping message encoding / decoding ──────────────────────────────────────

def encode_ping_message(freq_khz: int, time_ds: int, snr_db: int, width_ms: int = 0) -> str:
    """Encode ping parameters into a 10-character MSK144 free-text message.

    Format: A[P/N][ff][ttt][P/N][dd]
      A        – literal prefix character
      [P/N]    – sign of freq_khz  ('P' = positive/zero, 'N' = negative)
      [ff]     – abs(freq_khz) as 2 digits, 00–24
      [ttt]    – time in deciseconds as 3 digits, 005–150
      [P/N]    – sign of snr_db   ('P' = positive/zero, 'N' = negative)
      [dd]     – abs(snr_db) as 2 digits, 00–10
    Total: 1+1+2+3+1+2 = 10 characters.
    P/N are used instead of +/- because '+' is not reliably preserved by the
    MSK144 message encoder.  width_ms is accepted but ignored (MSK144 burst
    is always 60 ms when generated via msk144code).
    """
    freq_sign = 'P' if freq_khz >= 0 else 'N'
    snr_sign  = 'P' if snr_db  >= 0 else 'N'
    return f"A{freq_sign}{abs(freq_khz):02d}{time_ds:03d}{snr_sign}{abs(snr_db):02d}"


def decode_ping_message(msg: str) -> dict:
    """Decode a ping message back to its parameters."""
    freq_khz = (1 if msg[1] == 'P' else -1) * int(msg[2:4])
    time_ds  = int(msg[4:7])
    snr_db   = (1 if msg[7] == 'P' else -1) * int(msg[8:10])
    return {'freq_khz': freq_khz, 'time_ds': time_ds, 'snr_db': snr_db, 'width_ms': 60}


def _modulate_msk144(chips: np.ndarray, sample_rate: int) -> np.ndarray:
    """Modulate BPSK chips as MSK with half-sine pulse shaping.

    MSK144: 2000 baud, ±500 Hz tones, 72 ms per 144-chip burst.
    Returns unit-amplitude complex64 of length ceil(len(chips) * sample_rate / 2000).
    Phase is continuous across all chips — pass tiled chips for phase-continuous repeats.
    """
    sps     = sample_rate / 2000.0
    n_burst = int(np.ceil(len(chips) * sps))
    phase   = np.zeros(n_burst, dtype=np.float64)

    running_phase = 0.0
    for i, chip in enumerate(chips):
        sym_start = int(round(i * sps))
        sym_end   = min(int(round((i + 1) * sps)), n_burst)
        n_sym = sym_end - sym_start
        if n_sym <= 0:
            continue
        k     = np.arange(n_sym, dtype=np.float64)
        pulse = np.sin(np.pi * k / sps)
        norm  = np.sum(pulse)
        steps = chip * (np.pi / 2.0) * pulse / norm if norm > 0 else np.zeros(n_sym)
        phase[sym_start:sym_end] = running_phase + np.cumsum(steps)
        running_phase = float(phase[sym_end - 1])

    return np.exp(1j * phase).astype(np.complex64)


def generate_ping_msk144code(msg: str, center_hz: float, delay_s: float,
                             snr_db: float, noise_sigma: float,
                             width_ms: int = 300,
                             pol_scenario: str = 'pure-H',
                             rng=None,
                             sample_rate: int = 48000,
                             flat_frame: bool = False) -> tuple:
    """Generate a realistic MSK144 meteor-scatter echo from msk144code symbols.

    Physical model
    --------------
    The transmitter is H-polarized.  On reflection from the meteor trail, the
    polarization plane is rotated by angle θ (Faraday rotation through the
    ionosphere plus the trail geometry contribution).  θ is effectively frozen
    across a single 72 ms ping but is random from ping to ping.  The receiver
    sees two orthogonal components:

        rH(t) = cos θ(t) · s(t) · g_H
        rV(t) = sin θ(t) · s(t) · g_V · exp(j·Δφ(t))

    where s(t) is the Gamma-enveloped, frequency-shifted, amplitude-scaled
    baseband burst.  Independent AWGN per antenna is added by the caller.

    Phase continuity across burst boundaries is preserved by tiling the 144
    chips and modulating the full sequence in one pass — tiling the
    already-modulated waveform would introduce a phase jump at every 72 ms
    boundary that jt9 cannot correlate through.

    SNR definition
    --------------
    snr_db is defined as the peak SNR a perfectly-aligned H receiver would see
    at θ=0° (equivalently, the SNR of an MRC-combined H+V receiver for any θ):

        peak² / (2·σ²·BW/fs) = 10^(snr_db/10)     BW = 2500 Hz

    After polarization splitting:
        SNR_H   = snr_db + 20·log₁₀(|cos θ|)      (−∞ at θ=90°)
        SNR_V   = snr_db + 20·log₁₀(|sin θ|)      (−∞ at θ=0°)
        SNR_MRC = snr_db                           (MRC H+V, independent of θ)

    Parameters
    ----------
    msg          : MSK144 message string (passed to msk144code)
    center_hz    : signal carrier in the IQ stream (Hz from DC).
                   Must equal freq_khz × 1000 + 1500 to land at DC in the channel.
    delay_s      : time of echo onset (t=0 of Gamma envelope) in seconds
    snr_db       : peak SNR at θ=0° (WSJT-X definition, 2500 Hz bandwidth)
    noise_sigma  : per-sample AWGN sigma used for SNR calibration
    width_ms     : trail decay time constant τ in ms; echo duration ≈ 10τ s
                   (ignored when flat_frame=True)
    flat_frame   : if True, emit exactly one 72 ms frame at constant amplitude
                   (no Gamma envelope) — use this to match QEX-style sensitivity tests
    pol_scenario : polarization scenario — see _apply_pol_split for names
    rng          : numpy Generator; created internally if None
    sample_rate  : output sample rate in Hz

    Returns
    -------
    (ping_h, ping_v, pol_meta)
        ping_h   : complex64 array — H-antenna signal (no noise)
        ping_v   : complex64 array — V-antenna signal (no noise), same shape
        pol_meta : dict — pol_scenario, theta0_deg, delta_phi0_deg, w_h, w_v
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── 1. Get 144 channel symbols from msk144code ────────────────────────
    result = subprocess.run(['msk144code', msg], capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    sym_lines = [l for l in lines if set(l.strip()) <= {'0', '1'} and len(l.strip()) >= 60]
    if len(sym_lines) < 2:
        raise RuntimeError(f"msk144code output unexpected for {msg!r}:\n{result.stdout}")
    bits = [int(b) for line in sym_lines[:2] for b in line.strip()]
    if len(bits) != 144:
        raise RuntimeError(f"Expected 144 symbols, got {len(bits)}")

    chips = np.array([1 if b else -1 for b in bits], dtype=np.float64)

    # ── 2. Build burst stream ─────────────────────────────────────────────
    sps     = sample_rate / 2000.0
    n_burst = int(np.ceil(144 * sps))

    if flat_frame:
        # Single frame, constant amplitude — no meteor propagation model.
        # Matches the QEX / WSJT-X sensitivity test: one 72 ms frame at
        # uniform power.
        echo_n = n_burst
        stream = _modulate_msk144(chips, sample_rate)[:echo_n]
        stream = stream.astype(np.complex64)
    else:
        width_s  = max(width_ms, 10) / 1000.0
        echo_dur = 10.0 * width_s
        echo_n   = int(np.ceil(echo_dur * sample_rate))
        n_reps   = int(np.ceil(echo_n / n_burst))
        chips_tiled = np.tile(chips, n_reps)
        stream = _modulate_msk144(chips_tiled, sample_rate)[:echo_n]

        # ── 3. Gamma envelope A(t) = e·(t/τ)·exp(−t/τ) ──────────────────
        t_norm = np.arange(echo_n, dtype=np.float64) / sample_rate / width_s
        env    = np.where(t_norm > 0, np.e * t_norm * np.exp(-t_norm), 0.0).astype(np.float32)
        stream = (stream * env).astype(np.complex64)

    # t is needed for the frequency shift in step 4
    t = np.arange(echo_n, dtype=np.float64) / sample_rate

    # ── 4. Frequency shift to IF centre ───────────────────────────────────
    stream = (stream * np.exp(1j * 2.0 * np.pi * center_hz * t)).astype(np.complex64)

    # ── 5. SNR-calibrated amplitude ────────────────────────────────────────
    # Scale so that at θ=0° the H channel reaches the target peak SNR.
    # The cos/sin polarization split preserves total power: SNR_H + SNR_V = snr_db.
    BW  = 2500.0
    amp = noise_sigma * np.sqrt(2.0 * (10.0 ** (snr_db / 10.0)) * BW / sample_rate)
    stream = (stream * amp).astype(np.complex64)

    # ── 6. Polarization split ─────────────────────────────────────────────
    rH, rV, pol_meta = _apply_pol_split(stream, pol_scenario, rng, sample_rate)

    # ── 7. Place echo at delay in output arrays ────────────────────────────
    delay_n = int(round(delay_s * sample_rate))
    out_h   = np.zeros(delay_n + echo_n, dtype=np.complex64)
    out_v   = np.zeros(delay_n + echo_n, dtype=np.complex64)
    out_h[delay_n:] = rH
    out_v[delay_n:] = rV

    return out_h, out_v, pol_meta


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
    return LinearSegmentedColormap.from_list("radio_iq_spectrogram", color_points)


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


def _bandpass_real_fft(
    samples: np.ndarray,
    sample_rate: int,
    lo_hz: float,
    hi_hz: float,
) -> np.ndarray:
    """FFT-domain bandpass filter for a real signal with cosine tapers at both edges.

    Removes out-of-band noise (e.g. the wideband Gaussian noise embedded in
    msk144sim output) before converting to analytic/IQ.  The taper width is
    10 % of the passband bandwidth at each edge.
    """
    n = samples.size
    if n == 0:
        return samples.astype(np.float32)

    freq = np.fft.rfftfreq(n, 1.0 / sample_rate)
    X = np.fft.rfft(samples.astype(np.float64))
    gain = np.zeros(freq.size, dtype=np.float64)

    bw = hi_hz - lo_hz
    taper = max(10.0, 0.10 * bw)

    # Flat passband
    flat = (freq >= lo_hz) & (freq <= hi_hz)
    gain[flat] = 1.0

    # Low-edge cosine rise: (lo_hz - taper) … lo_hz
    lo_taper_lo = max(0.0, lo_hz - taper)
    trans_lo = (freq >= lo_taper_lo) & (freq < lo_hz)
    if np.any(trans_lo):
        t = (freq[trans_lo] - lo_taper_lo) / (lo_hz - lo_taper_lo)
        gain[trans_lo] = 0.5 - 0.5 * np.cos(np.pi * t)

    # High-edge cosine fall: hi_hz … (hi_hz + taper)
    hi_taper_hi = hi_hz + taper
    trans_hi = (freq > hi_hz) & (freq <= hi_taper_hi)
    if np.any(trans_hi):
        t = (freq[trans_hi] - hi_hz) / (hi_taper_hi - hi_hz)
        gain[trans_hi] = 0.5 + 0.5 * np.cos(np.pi * t)

    x_out = np.fft.irfft(X * gain, n=n)
    return x_out.astype(np.float32)


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


def _plot_output_spectrogram(
    combined: np.ndarray,
    sample_rate: int,
    output_path: Path,
    placements: list[dict],
) -> None:
    """Save a spectrogram of the combined IQ output with signal placement markers."""
    t_s, freq_hz, spec_db = _compute_spectrogram_db(combined, sample_rate)
    gui_cmap = _gui_like_colormap()
    vmin, vmax = -110.0, -70.0

    fig, ax = plt.subplots(figsize=(14, 5))
    f_min = float(freq_hz[0]) / 1000.0
    f_max = float(freq_hz[-1]) / 1000.0
    t_min = float(t_s[0])
    t_max = float(t_s[-1])
    ax.imshow(
        spec_db.T, aspect='auto', origin='lower',
        extent=[t_min, t_max, f_min, f_max],
        cmap=gui_cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Combined IQ Test Signal – Spectrogram')

    # Mark each signal's center frequency and start time
    for p in placements:
        ax.axhline(p['center_hz'] / 1000.0, color='red', lw=0.7, ls='--', alpha=0.6)
        ax.axvline(p['delay_s'], color='yellow', lw=0.7, ls=':', alpha=0.6)
        ax.text(p['delay_s'] + 0.05, p['center_hz'] / 1000.0 + 0.2,
                f"{p['center_hz']/1000:+.1f}kHz", color='red', fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote spectrogram: {output_path}")


def _apply_pol_split(signal: np.ndarray, pol_scenario: str, rng,
                     sample_rate: int = 48000) -> tuple:
    """Split a complex baseband signal into H and V receive channels.

    Models an H-polarized transmitter reflected by a meteor trail.  The
    received polarization angle θ is set by the scenario.  Faraday rotation
    and trail geometry together determine θ; it is frozen across a single ping
    (~72 ms) but random from ping to ping.  Within-ping drift models trail
    aspect-angle evolution as the trail expands — not Faraday rotation, which
    changes on timescales of minutes.

    Channel model:
        rH(t) = g_H · cos θ(t) · s(t)
        rV(t) = g_V · sin θ(t) · s(t) · exp(j·Δφ)

    Parameters
    ----------
    signal       : complex64 array — pre-scaled baseband signal s(t)
    pol_scenario : scenario name or numeric angle in degrees (see below)
    rng          : numpy Generator
    sample_rate  : samples per second (used for within-ping drift rate)

    Returns
    -------
    (rH, rV, pol_meta)
        rH       : complex64 array — H-antenna signal
        rV       : complex64 array — V-antenna signal
        pol_meta : dict — pol_scenario, theta0_deg, delta_phi0_deg, w_h, w_v

    Scenarios
    ---------
    pure-H          θ=0°            all power in H; V sees nothing
    pure-V          θ=90°           all power in V; H sees nothing
    linear-30/45/60 θ=30/45/60°    fixed linear angle
    near-cross      θ=85°           deep fade on H-only receiver
    random          θ~U[0°,90°]    realistic default; new draw each call
    trail-drift     θ0 random, +30°/s drift  — slow trail aspect evolution
    trail-fast      θ0 random, +180°/s drift — rapid trail aspect evolution
    elliptical      θ=45°, Δφ=90°  circular polarization
    cal-error       θ=45°, ±3 dB gain imbalance, Δφ=15° systematic offset
    faraday-slow    alias for trail-drift (legacy)
    faraday-fast    alias for trail-fast  (legacy)
    <number>        θ=<number>° fixed, Δφ=0°
    """
    n = len(signal)
    t = np.arange(n, dtype=np.float64) / sample_rate
    pol = pol_scenario.strip().lower()

    theta0_deg     = 0.0
    drift_deg_s    = 0.0
    delta_phi0_deg = 0.0
    g_h_db         = 0.0
    g_v_db         = 0.0

    if pol == 'pure-h':
        theta0_deg = 0.0
    elif pol == 'pure-v':
        theta0_deg = 90.0
    elif pol in ('linear-30', 'linear30'):
        theta0_deg = 30.0
    elif pol in ('linear-45', 'linear45'):
        theta0_deg = 45.0
    elif pol in ('linear-60', 'linear60'):
        theta0_deg = 60.0
    elif pol == 'near-cross':
        theta0_deg = 85.0
    elif pol == 'random':
        theta0_deg = float(rng.uniform(0.0, 90.0))
    elif pol in ('trail-drift', 'faraday-slow'):
        theta0_deg  = float(rng.uniform(0.0, 90.0))
        drift_deg_s = 30.0
    elif pol in ('trail-fast', 'faraday-fast'):
        theta0_deg  = float(rng.uniform(0.0, 90.0))
        drift_deg_s = 180.0
    elif pol == 'elliptical':
        theta0_deg     = 45.0
        delta_phi0_deg = 90.0
    elif pol == 'cal-error':
        theta0_deg     = 45.0
        g_h_db         = float(rng.uniform(-3.0, 3.0))
        g_v_db         = -g_h_db
        delta_phi0_deg = 15.0
    else:
        try:
            theta0_deg = float(pol_scenario)
        except ValueError:
            raise SystemExit(
                f"Unknown pol_scenario {pol_scenario!r}. "
                "Use: pure-H, pure-V, linear-30/45/60, near-cross, random, "
                "trail-drift, trail-fast, elliptical, cal-error, "
                "or a numeric angle in degrees."
            )

    theta_rad     = np.deg2rad(theta0_deg + drift_deg_s * t)
    delta_phi_rad = np.deg2rad(delta_phi0_deg)
    g_h           = 10.0 ** (g_h_db / 20.0)
    g_v           = 10.0 ** (g_v_db / 20.0)

    rH = (g_h * np.cos(theta_rad) * signal).astype(np.complex64)
    rV = (g_v * np.sin(theta_rad) * np.exp(1j * delta_phi_rad) * signal).astype(np.complex64)

    # MRC combiner weights at mid-ping
    mid = n // 2
    w_h = float(np.cos(theta_rad[mid]))
    w_v = complex(np.sin(theta_rad[mid]) * np.exp(-1j * delta_phi_rad))

    pol_meta = {
        'pol_scenario':   pol_scenario,
        'theta0_deg':     round(theta0_deg, 1),
        'delta_phi0_deg': round(delta_phi0_deg, 1),
        'w_h':            round(w_h, 4),
        'w_v':            str(round(w_v.real, 4) + round(w_v.imag, 4) * 1j),
    }
    return rH, rV, pol_meta


# ── Ramp-mode helpers (calibrated SNR sweep) ─────────────────────────────────

RAMP_AUDIO_RATE = 12000        # WSJT-X expects 12 kHz mono
RAMP_IQ_RATE    = 48000        # MAP144 channelizer input rate
RAMP_AUDIO_CENTER_HZ = 1500.0  # WSJT-X USB audio centre


def _write_audio_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono real-audio WAV (PCM int16) — the format WSJT-X File→Open
    and headless ``jt9 -m`` accept directly.  Peak-normalised to 0.95 full scale
    only if necessary (signal already exceeds ±1.0); otherwise written at unity
    gain so that calibrated noise floors are preserved."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    scale = 0.95 / peak if peak > 1.0 else 1.0
    pcm = np.clip(samples * scale, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _make_msk144_burst_real(msg: str, sample_rate: int, n_frames: int = 1) -> np.ndarray:
    """Return ``n_frames`` MSK144 frames at unit-amplitude COMPLEX baseband, 0 Hz.

    Phase continuity across frame boundaries is preserved by tiling the chip
    sequence first and modulating the full stream in one pass — tiling the
    already-modulated waveform would introduce a phase jump at every 72 ms
    boundary that the matched-filter decoder cannot correlate through.
    Caller multiplies by the desired carrier and amplitude.
    """
    result = subprocess.run(['msk144code', msg], capture_output=True, text=True, check=True)
    sym_lines = [l for l in result.stdout.strip().splitlines()
                 if set(l.strip()) <= {'0', '1'} and len(l.strip()) >= 60]
    if len(sym_lines) < 2:
        raise RuntimeError(f"msk144code output unexpected for {msg!r}")
    bits = [int(b) for line in sym_lines[:2] for b in line.strip()]
    if len(bits) != 144:
        raise RuntimeError(f"Expected 144 symbols, got {len(bits)}")
    chips = np.array([1 if b else -1 for b in bits], dtype=np.float64)
    if n_frames > 1:
        chips = np.tile(chips, n_frames)
    return _modulate_msk144(chips, sample_rate)   # unit-amplitude complex


def _run_ramp_mode(args) -> None:
    """SNR-ramp emit path.

    Builds a 12 kHz mono real-audio buffer (the canonical, SNR-calibrated test
    signal that WSJT-X / headless ``jt9 --msk144`` ingests directly) containing
    N MSK144 pings at monotonically increasing SNRs in a 2500 Hz reference
    bandwidth.

    The 48 kHz complex-IQ output is the **signal** taken from a noise-free copy
    of the audio buffer (Hilbert → resample → optional frequency shift) PLUS a
    **fresh full-band complex AWGN** floor that fills the entire ±24 kHz IQ
    Nyquist band.  This matches what an on-air receiver actually delivers to
    MAP144's channelizer; deriving IQ noise via Hilbert from the 12 kHz audio
    would give a 6 kHz-wide noise blob whose ``|·|²`` (the squared-FFT
    detection step) splatters into the rest of the IF band and triggers
    spurious detections in channels with no real signal.  See
    https://… (this comment block) for the calibration derivation.

    Calibration: SNR-in-2500-Hz matches between files
    -------------------------------------------------
    Audio (real 12 kHz mono):
        signal mean power      = A_audio² / 2          (real cos)
        noise power in 2500 Hz = σ_a²  · 2500 / 6000   (one-sided over fs/2)
        SNR_audio = (A²/2) / (σ_a² · 2500/6000) = (3·A²) / (2500·σ_a²)

    IQ (complex 48 kHz, fresh full-band AWGN, signal via Hilbert+resample):
        signal mean power      = A_audio²              (Hilbert doubles |·|²)
        noise power in 2500 Hz = σ_iq² · 2500 / 48000  (two-sided over full IQ)
        SNR_iq   = A² / (σ_iq² · 2500/48000) = (48000·A²) / (2500·σ_iq²)

    Setting SNR_iq = SNR_audio yields ``σ_iq² = 16·σ_a²``, i.e. **σ_iq = 4·σ_a**.

    Each ping carries a self-describing ``A...`` message encoding its intended
    SNR; the truth JSON sidecar lists the full plan for the scoring step.
    """
    if args.ramp_snr_max < args.ramp_snr_min:
        raise SystemExit("--ramp-snr-max must be >= --ramp-snr-min")
    n_steps = int(round((args.ramp_snr_max - args.ramp_snr_min) / args.ramp_snr_step)) + 1
    if n_steps <= 0:
        raise SystemExit("ramp produces zero pings — check --ramp-snr-min/-max/-step")

    snr_seq = [args.ramp_snr_min + k * args.ramp_snr_step for k in range(n_steps)]
    delays  = [args.ramp_t0 + k * args.ramp_spacing for k in range(n_steps)]
    if delays[-1] >= args.duration - 0.1:
        raise SystemExit(
            f"Ramp does not fit in {args.duration:.1f} s: last ping at {delays[-1]:.2f} s. "
            f"Reduce --ramp-snr-step or --ramp-spacing, or extend --duration.")

    rng = np.random.default_rng(args.seed)
    n_audio = int(args.duration * RAMP_AUDIO_RATE)
    n_iq    = int(args.duration * RAMP_IQ_RATE)

    # Audio noise floor: full-scale ±1.0 in the WAV.  noise_floor_dbfs sets the
    # per-bin floor at the receiver's display FFT (4096 bins over 0..fs/2).
    # Per-sample sigma_audio is back-derived from the per-bin level.
    display_fft_size = 4096
    noise_floor_amp = 10.0 ** (args.noise_floor_dbfs / 20.0)
    sigma_audio = float(np.sqrt(display_fft_size) * noise_floor_amp)
    sigma_iq    = 4.0 * sigma_audio   # see docstring derivation

    # ── Build noise-only audio (real, 12 kHz) and signal-only audio side ────
    audio_noise = rng.normal(0.0, sigma_audio, size=n_audio).astype(np.float32)
    audio_signal = np.zeros(n_audio, dtype=np.float32)
    BW = MSK144_SNR_BW_HZ
    fs = float(RAMP_AUDIO_RATE)

    truth = []
    for k, (snr_db, delay_s) in enumerate(zip(snr_seq, delays)):
        # Self-describing message: freq-field always 0 in ramp mode (IQ offset
        # is in the sidecar, not the message).  SNR + decisecond-time encoded.
        time_ds = int(round(min(max(delay_s, 0.5), args.duration - 0.5) * 10))
        snr_int = int(round(snr_db))
        msg = encode_ping_message(freq_khz=0, time_ds=time_ds, snr_db=snr_int)

        # Unit-amplitude complex MSK144 baseband, then up-shift to 1500 Hz
        # and take Re{} to get a real audio burst at the WSJT-X centre.
        burst_c = _make_msk144_burst_real(msg, RAMP_AUDIO_RATE, n_frames=args.ramp_frames)
        t = np.arange(len(burst_c), dtype=np.float64) / fs
        burst_real = np.real(burst_c * np.exp(1j * 2.0 * np.pi * RAMP_AUDIO_CENTER_HZ * t))

        # Real-signal amplitude for target SNR over the 2500 Hz reference,
        # given the real AWGN at sigma_audio per sample (see docstring).
        snr_lin = 10.0 ** (snr_db / 10.0)
        amp = 2.0 * sigma_audio * np.sqrt(snr_lin * BW / fs)
        burst_real = (burst_real * amp).astype(np.float32)

        delay_n = int(round(delay_s * fs))
        end_n   = min(delay_n + len(burst_real), n_audio)
        audio_signal[delay_n:end_n] += burst_real[:end_n - delay_n]

        truth.append({
            'message':       msg,
            't_offset_s':    round(delay_s, 4),
            'snr_db_input':  round(float(snr_db), 2),
            'audio_freq_hz': RAMP_AUDIO_CENTER_HZ,
            'iq_freq_hz':    RAMP_AUDIO_CENTER_HZ + args.iq_offset_hz,
        })

    audio_buf = audio_noise + audio_signal

    # ── Build IQ: flat full-band complex AWGN at 48 kHz + signal-only path ──
    from scipy import signal as _scipy_signal
    iq_noise_re = rng.normal(0.0, sigma_iq / np.sqrt(2.0), size=n_iq)
    iq_noise_im = rng.normal(0.0, sigma_iq / np.sqrt(2.0), size=n_iq)
    iq_buf = (iq_noise_re + 1j * iq_noise_im).astype(np.complex64)

    # Signal-only path: Hilbert(real audio signal) → resample to 48 kHz → shift.
    # Hilbert doubles |·|² of the signal which the calibration above accounts
    # for; resample_poly is band-limited interpolation that preserves SNR.
    analytic_sig = _scipy_signal.hilbert(audio_signal).astype(np.complex64)
    iq_signal = _scipy_signal.resample_poly(
        analytic_sig, up=RAMP_IQ_RATE, down=RAMP_AUDIO_RATE).astype(np.complex64)
    if args.iq_offset_hz != 0.0:
        t_iq = np.arange(len(iq_signal), dtype=np.float64) / RAMP_IQ_RATE
        iq_signal = (iq_signal * np.exp(1j * 2.0 * np.pi * args.iq_offset_hz * t_iq)).astype(np.complex64)
    # Length-match (resample can be off by 1 sample) and add.
    n_add = min(len(iq_buf), len(iq_signal))
    iq_buf[:n_add] += iq_signal[:n_add]

    # ── Output paths ─────────────────────────────────────────────────────────
    _ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    _stem = f"ramp_{_ts}_{n_steps}sig_{int(round(args.ramp_snr_min)):+d}to{int(round(args.ramp_snr_max)):+d}dB"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{_stem}_audio.wav"
    iq_path    = out_dir / f"{_stem}_iq.wav"
    truth_path = out_dir / f"{_stem}_truth.json"

    _write_audio_wav(audio_path, audio_buf, RAMP_AUDIO_RATE)
    write_iq_wav(iq_path, iq_buf, RAMP_IQ_RATE, scale=1.0)

    truth_doc = {
        'generated_at':     _ts,
        'duration_s':       float(args.duration),
        'audio_rate_hz':    RAMP_AUDIO_RATE,
        'iq_rate_hz':       RAMP_IQ_RATE,
        'iq_offset_hz':     float(args.iq_offset_hz),
        'noise_floor_dbfs': float(args.noise_floor_dbfs),
        'snr_reference_bw_hz': MSK144_SNR_BW_HZ,
        'pings':            truth,
    }
    truth_path.write_text(json.dumps(truth_doc, indent=2))

    # MAP144's built-in WAV-comparison thread (see runtime._run_wav_comparison)
    # looks for ``<wav>.json`` next to the WAV and expects a ``placements`` list
    # in the schema used by the ``--count N`` path.  Write that sidecar too so
    # MAP144 emits its comparison report automatically at end of WAV playback.
    iq_manifest_path = iq_path.with_suffix('.json')
    iq_manifest = {
        'generated_at':      datetime.now(timezone.utc).isoformat(),
        'output_wav':        str(iq_path),
        'channels':          1,
        'channel_format':    'single (I=L, Q=R)',
        'output_rate':       RAMP_IQ_RATE,
        'duration_s':        float(args.duration),
        'noise_floor_dbfs':  float(args.noise_floor_dbfs),
        'no_noise':          False,
        'atten_db':          0.0,
        'mode':              'ramp',
        'iq_offset_hz':      float(args.iq_offset_hz),
        'snr_reference_bw_hz': MSK144_SNR_BW_HZ,
        'placements': [
            {
                'msg':       p['message'],
                'center_hz': p['iq_freq_hz'],
                'delay_s':   p['t_offset_s'],
                # MAP144's _run_wav_comparison report uses ``f"{v:>+4d}"`` on
                # ``snr_db`` (see runtime._snr_str); the existing --count path
                # always writes int.  Cast to match that schema.
                'snr_db':    int(round(p['snr_db_input'])),
                'width_ms':  0,
            }
            for p in truth
        ],
    }
    iq_manifest_path.write_text(json.dumps(iq_manifest, indent=2))

    print(f"Ramp: {n_steps} pings, SNR {args.ramp_snr_min:+.1f} → {args.ramp_snr_max:+.1f} dB"
          f"  step {args.ramp_snr_step:.1f} dB  spacing {args.ramp_spacing:.2f} s")
    print(f"  audio (WSJT-X / jt9): {audio_path}")
    print(f"  IQ    (MAP144):       {iq_path}  (offset {args.iq_offset_hz:+.0f} Hz)")
    print(f"  manifest (MAP144):    {iq_manifest_path}")
    print(f"  truth (score_ramp):   {truth_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 48 kHz complex IQ test stream: AWGN noise floor with MSK144 "
            "bursts placed at random frequencies and time offsets.\n\n"
            "With --count N, pings are generated via msk144sim with randomly chosen\n"
            "parameters encoded into self-describing 13-char messages."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--count', type=int, default=10,
                        help='Number of random pings to generate via msk144sim; 0 = use --input-dir')
    parser.add_argument('--input-dir', default='MSK144',
                        help='Folder containing 15-sec 12 kHz mono WAV files')
    parser.add_argument('--pattern', default='*.wav',
                        help='Glob pattern for source WAV files')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Output duration in seconds')
    parser.add_argument('--output-rate', type=int, default=48000,
                        help='Output sample rate in Hz')
    parser.add_argument('--freq-range-hz', type=float, default=12000.0,
                        help='Place each signal at a random centre in ±freq-range-hz; '
                             'ignored when --count is used (frequency is encoded in message)')
    parser.add_argument('--max-delay-s', type=float, default=1.0,
                        help='Maximum random start-time delay in seconds; '
                             'ignored when --count is used (time is encoded in message)')
    parser.add_argument('--atten-db', type=float, default=10.0,
                        help='Attenuation applied to each signal before mixing, dB')
    parser.add_argument('--snr-min', type=int, default=SNR_MIN_DB,
                        help=f'Minimum SNR in dB for generated pings (default: {SNR_MIN_DB})')
    parser.add_argument('--snr-max', type=int, default=SNR_MAX_DB,
                        help=f'Maximum SNR in dB for generated pings (default: {SNR_MAX_DB})')
    parser.add_argument('--width-min-ms', type=int, default=WIDTH_MIN_MS,
                        help=f'Minimum Gamma envelope τ in ms (default: {WIDTH_MIN_MS})')
    parser.add_argument('--width-max-ms', type=int, default=WIDTH_MAX_MS,
                        help=f'Maximum Gamma envelope τ in ms (default: {WIDTH_MAX_MS})')
    parser.add_argument('--noise-floor-dbfs', type=float, default=-96.0,
                        help='Target noise floor in dBFS (0 dBFS = ±1.0); '
                             'per-sample σ = √4096 × 10^(noise_floor_dbfs/20)')
    parser.add_argument('--no-noise', action='store_true',
                        help='Omit AWGN entirely; write pure signal at the calibrated amplitude')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible placement; None = random')
    parser.add_argument('-o', '--output-iq-wav', default=None,
                        help='Output IQ WAV path (I=L, Q=R for single-channel; 4-ch for dual); '
                             'None = auto-name in --output-dir')
    parser.add_argument('--dual-channel', action='store_true',
                        help='Generate a second polarization channel (ch1=V antenna); '
                             'writes a 4-channel WAV (ch0_I, ch0_Q, ch1_I, ch1_Q)')
    parser.add_argument('--pol-scenario', default='pure-H',
                        help='Polarization scenario applied to every ping: '
                             'pure-H (default, no polarization loss), pure-V, '
                             'linear-30/45/60, near-cross, random (uniform θ per ping), '
                             'trail-drift, trail-fast, elliptical, cal-error, '
                             'or a numeric angle in degrees. '
                             'Use random with --dual-channel for realistic dual-pol testing.')
    parser.add_argument('--plot-output', default=None,
                        help='Output PNG spectrogram path; None = auto-name in --output-dir')
    parser.add_argument('--source-bp-lo', type=float, default=300.0,
                        help='Source bandpass lower edge Hz – strip out-of-band noise')
    parser.add_argument('--source-bp-hi', type=float, default=2700.0,
                        help='Source bandpass upper edge Hz')
    parser.add_argument('--callsigns', action='store_true',
                        help='Use real callsign messages (CQ/QSO format) instead of diagnostic A... messages')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip output spectrogram plot')
    parser.add_argument('--output-dir', default='MSK144/simulations',
                        help='Directory for output files')

    # ── Ramp mode (calibrated SNR sweep for WSJT-X vs MAP144 comparison) ──
    parser.add_argument('--ramp', action='store_true',
                        help='Calibrated SNR-ramp mode: emit a 12 kHz mono real-audio WAV '
                             '(for WSJT-X File→Open / headless jt9), a 48 kHz complex-IQ WAV '
                             '(for MAP144 replay), and a JSON truth sidecar.  Overrides --count.')
    parser.add_argument('--ramp-snr-min', type=float, default=-15.0,
                        help='Lowest SNR (dB, WSJT-X 2500 Hz reference) in the ramp')
    parser.add_argument('--ramp-snr-max', type=float, default=0.0,
                        help='Highest SNR (dB, WSJT-X 2500 Hz reference) in the ramp')
    parser.add_argument('--ramp-snr-step', type=float, default=1.0,
                        help='SNR step in dB; pings = floor((max-min)/step) + 1')
    parser.add_argument('--ramp-spacing', type=float, default=0.9,
                        help='Seconds between consecutive ping starts.  Default 0.9 s lets a '
                             '16-step ramp fit in 15 s with margin on both ends.')
    parser.add_argument('--ramp-t0', type=float, default=0.5,
                        help='Time of first ping in seconds (≥ 0.5 s leaves pre-burst noise '
                             'window for SPD-style estimators).')
    parser.add_argument('--ramp-frames', type=int, default=5,
                        help='Number of 72 ms MSK144 frames per ping at constant amplitude. '
                             '5 frames (~360 ms) gives MDS ≈ −6 dB on jt9, close to the '
                             'published WSJT-X MSK144 sensitivity.  Single-frame (1) is too '
                             'short for the matched-filter decoder and shifts MDS by ~10 dB.')
    parser.add_argument('--iq-offset-hz', type=float, default=0.0,
                        help='Frequency offset (Hz) added to the IQ render only.  The audio '
                             'render always places the signal at 1500 Hz (WSJT-X convention). '
                             'Use this to stress MAP144 corner cases (channelizer edges, BPF '
                             'skirts) without changing what WSJT-X sees.')

    args = parser.parse_args()

    if args.ramp:
        return _run_ramp_mode(args)

    from datetime import datetime as _dt
    _ts = _dt.now().strftime('%Y%m%d_%H%M%S')
    _n = args.count if args.count > 0 else 0
    _stem = f"msk144_{_ts}_{_n}sig"
    _out_dir = Path(args.output_dir)
    _out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_iq_wav is None:
        args.output_iq_wav = str(_out_dir / f"{_stem}.wav")
    if args.plot_output is None:
        args.plot_output = str(_out_dir / f"{_stem}.png")

    rng = np.random.default_rng(args.seed)
    n_total = int(args.duration * args.output_rate)

    # Display FFT size used by the receiver pipeline (must match processing.py)
    display_fft_size = 4096
    # Per-bin noise floor amplitude: tone at this amplitude displays at noise_floor_dbfs
    noise_floor_amp = 10.0 ** (args.noise_floor_dbfs / 20.0)
    # Per-sample noise sigma: 20*log10(noise_sigma / sqrt(N)) == noise_floor_dbfs
    noise_sigma = np.sqrt(display_fft_size) * noise_floor_amp

    print(f"Output: {args.duration:.1f} s  @  {args.output_rate} Hz  ({n_total} samples)")
    print(f"Noise floor: {args.noise_floor_dbfs:.1f} dBFS  "
          f"(σ={noise_sigma:.6f}  bin floor={20*np.log10(noise_floor_amp):.1f} dBFS)\n")

    sig_buf  = np.zeros(n_total, dtype=np.complex64)
    sig_buf1 = np.zeros(n_total, dtype=np.complex64) if args.dual_channel else None
    placements: list[dict] = []   # keys: msg, center_hz, delay_s, snr_db, width_ms

    # Pool of real callsign messages for --callsigns mode
    _CALLSIGN_MESSAGES = [
        "CQ W1AW FN31",
        "CQ K1TTT FN32",
        "CQ W2FU FN20",
        "CQ KA1G FN42",
        "CQ VE3EJ FN03",
        "CQ W4RX EM84",
        "CQ K8TS EN82",
        "CQ K2TER FN30",
        "CQ W1ZM FN42",
        "CQ N1JEZ FN42",
        "W1AW K1TTT FN32",
        "K1TTT W1AW FN31",
    ]

    if args.count > 0:
        # ── Generate pings via msk144code (pure MSK, no embedded noise) ────────
        if shutil.which('msk144code') is None:
            print("error: msk144code not found on PATH", file=sys.stderr)
            sys.exit(1)

        if args.callsigns:
            print(f"Generating {args.count} pings via msk144code (callsign messages):")
        else:
            print(f"Generating {args.count} pings via msk144code:")
            print(f"  Message format: A[P/N][ff][ttt][P/N][dd]")
        print(f"  freq {FREQ_MIN_KHZ}..{FREQ_MAX_KHZ} kHz  "
              f"time {TIME_MIN_DS}..{TIME_MAX_DS} ds  "
              f"snr {args.snr_min}..{args.snr_max} dB  "
              f"width {args.width_min_ms}..{args.width_max_ms} ms\n")

        def _ping_conflicts(placements, freq_khz, delay_s, width_ms):
            """Return True if candidate ping conflicts with already-placed pings.

            Guards against two detector failure modes:
            1. Same-channel cooldown: two pings on the same 1-kHz channel within
               the detector's DETECT_MERGE_GAP_S=1.0 s cooldown window — the second
               will be suppressed.  Guard conservatively at 1.5 s.
            2. Coincidence-gate over-trigger: too many channels firing simultaneously
               trips _COINCIDENCE_MAX_CH=8.  Each ping spans ~2 detector channels, so
               cap simultaneous overlapping pings at 3 (3×2=6 channels, safely < 8).
            """
            new_start = delay_s
            new_end   = delay_s + width_ms / 1000.0 * 1.5   # rough gamma tail bound

            n_overlap = 0
            for p in placements:
                p_freq_khz = round((p['center_hz'] - 1500.0) / 1000.0)
                p_start    = p['delay_s']
                p_end      = p_start + p['width_ms'] / 1000.0 * 1.5

                # Temporal overlap — needed for simultaneous count
                if new_start < p_end and new_end > p_start:
                    n_overlap += 1

                # Same-channel conflict: within 1.5 kHz in frequency AND 1.5 s in time
                if abs(freq_khz - p_freq_khz) < 1.5 and abs(delay_s - p['delay_s']) < 1.5:
                    return True

            # Too many simultaneous pings would trip the coincidence gate
            if n_overlap >= 3:
                return True

            return False

        placed   = 0
        retries  = 0
        _MAX_RETRIES = args.count * 50

        while placed < args.count:
            if retries >= _MAX_RETRIES:
                print(f"  [warning] only placed {placed}/{args.count} pings after "
                      f"{_MAX_RETRIES} retry attempts — time window too dense or "
                      f"count too high for the available spread")
                break

            freq_khz = int(rng.integers(FREQ_MIN_KHZ, FREQ_MAX_KHZ + 1))
            time_ds  = int(rng.integers(TIME_MIN_DS,  TIME_MAX_DS  + 1))
            snr_db   = int(rng.integers(args.snr_min,  args.snr_max  + 1))
            width_ms = int(rng.integers(args.width_min_ms, args.width_max_ms + 1))
            delay_s  = time_ds * 0.1

            if _ping_conflicts(placements, freq_khz, delay_s, width_ms):
                retries += 1
                continue

            if args.callsigns:
                msg = _CALLSIGN_MESSAGES[placed % len(_CALLSIGN_MESSAGES)]
            else:
                msg = encode_ping_message(freq_khz, time_ds, snr_db)

            # Place signal at the channelizer NCO for channel freq_khz.
            # NCO(k) = k*1000 + channel_offset_hz, and channel_offset_hz
            # includes +1500 Hz (USB audio offset).  Signal at k*1000+1500 Hz
            # in the IF lands at DC in channel k, which the squaring detector
            # requires.  extract_and_decode later mixes fc_hz → 1500 Hz for jt9.
            target_hz = float(freq_khz) * 1000.0 + 1500.0

            try:
                ping_h, ping_v, pol_meta = generate_ping_msk144code(
                    msg, center_hz=target_hz, delay_s=delay_s,
                    snr_db=snr_db, noise_sigma=noise_sigma,
                    width_ms=width_ms, pol_scenario=args.pol_scenario,
                    rng=rng, sample_rate=args.output_rate)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"  [{placed+1:3d}/{args.count}] {msg}  FAILED ({exc}), skipping")
                retries += 1
                continue

            n_copy = min(len(ping_h), n_total)
            sig_buf[:n_copy] += ping_h[:n_copy]
            if sig_buf1 is not None:
                sig_buf1[:n_copy] += ping_v[:n_copy]
            _dphi = pol_meta['delta_phi0_deg']
            _dphi_str = f"  Δφ={_dphi:.1f}°" if _dphi != 0.0 else ""
            pol_str = (f"  θ₀={pol_meta['theta0_deg']:.1f}°{_dphi_str}"
                       f"  scenario={pol_meta['pol_scenario']}")

            print(f"  [{placed+1:3d}/{args.count}] {msg}  "
                  f"freq={freq_khz:+d} kHz  t={delay_s:.1f} s  "
                  f"snr={snr_db:+d} dB  width={width_ms} ms{pol_str}")
            placements.append({'msg': msg, 'center_hz': target_hz, 'delay_s': delay_s,
                                'snr_db': snr_db, 'width_ms': width_ms,
                                **pol_meta})
            placed  += 1
            retries  = 0

    else:
        # ── Original mode: read WAV files from --input-dir ────────────────────
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob(args.pattern))
        if not files:
            raise SystemExit(f"No WAV files found in {input_dir} matching {args.pattern}")

        print(f"Source files ({len(files)}):")
        for path in files:
            samples, src_rate = read_wav_mono(path)

            up = resample_linear(samples, src_rate, args.output_rate)
            up = _bandpass_real_fft(up, args.output_rate, args.source_bp_lo, args.source_bp_hi)

            target_center_hz = float(rng.uniform(-args.freq_range_hz, args.freq_range_hz))
            shift_hz = target_center_hz - SOURCE_CENTER_HZ
            shifted = freq_shift_real_to_complex(up, shift_hz, args.output_rate)

            delay_n = int(rng.uniform(0.0, args.max_delay_s) * args.output_rate)

            n_copy = min(shifted.size, n_total - delay_n)
            if n_copy > 0:
                scaled = (shifted[:n_copy] * atten).astype(np.complex64)
                rH, rV, _ = _apply_pol_split(scaled, args.pol_scenario, rng, args.output_rate)
                sig_buf[delay_n:delay_n + n_copy] += rH
                if sig_buf1 is not None:
                    sig_buf1[delay_n:delay_n + n_copy] += rV

            delay_s = delay_n / args.output_rate
            print(
                f"  {path.name}: src={src_rate} Hz, "
                f"centre={target_center_hz:+.0f} Hz, "
                f"delay={delay_s:.3f} s"
            )
            placements.append({'msg': path.name, 'center_hz': target_center_hz,
                                'delay_s': delay_s, 'snr_db': None, 'width_ms': None})

    # ── Pass 2: optionally add calibrated AWGN, write with fixed scale ────────
    # Pings are scaled to WSJT-X SNR in MSK144_SNR_BW_HZ bandwidth relative to
    # the noise floor.  The combined peak is noise-dominated (~5·σ ≈ 0.005) and
    # well within ±1.0, so writing with scale=1.0 preserves the noise floor at
    # exactly noise_floor_dbfs dBFS after the int16 round-trip.
    if args.no_noise:
        combined  = sig_buf.copy()
        combined1 = sig_buf1.copy() if sig_buf1 is not None else None
        comb_peak = float(np.max(np.abs(combined)))
        sig_peak  = float(np.max(np.abs(sig_buf))) if sig_buf.size else 0.0
        print(f"\nNo-noise mode.  Signal peak: {sig_peak:.2e} "
              f"({20*np.log10(sig_peak+1e-30):.1f} dBFS)  combined peak: {comb_peak:.4f}")
    else:
        noise_i = rng.standard_normal(n_total).astype(np.float32) * noise_sigma
        noise_q = rng.standard_normal(n_total).astype(np.float32) * noise_sigma
        combined = sig_buf + (noise_i + 1j * noise_q).astype(np.complex64)
        if sig_buf1 is not None:
            # Independent noise on ch1
            noise_i1 = rng.standard_normal(n_total).astype(np.float32) * noise_sigma
            noise_q1 = rng.standard_normal(n_total).astype(np.float32) * noise_sigma
            combined1 = sig_buf1 + (noise_i1 + 1j * noise_q1).astype(np.complex64)
        else:
            combined1 = None
        comb_peak = float(np.max(np.abs(combined)))
        sig_peak  = float(np.max(np.abs(sig_buf))) if sig_buf.size else 0.0
        print(f"\nSignal peak: {sig_peak:.2e} ({20*np.log10(sig_peak+1e-30):.1f} dBFS)  "
              f"noise σ: {noise_sigma:.2e}  combined peak: {comb_peak:.4f}")

    if args.output_iq_wav.strip():
        out_wav = Path(args.output_iq_wav)
        # With AWGN: use scale=1.0 so the calibrated noise floor is preserved.
        # Without AWGN: the signal amplitude is tiny (few int16 counts at scale=1.0),
        # causing ~25–30 dB quantization distortion.  Auto peak-normalize instead so
        # the full 16-bit range is used and quantization artifacts are negligible.
        write_iq_wav(out_wav, combined, args.output_rate,
                     scale=None if args.no_noise else 1.0,
                     iq_ch1=combined1)
        ch_str = "dual-channel 4-ch" if combined1 is not None else "single-channel stereo"
        print(f"Wrote ({ch_str}): {out_wav}")

    # ── Write manifest JSON ────────────────────────────────────────────────────
    # Each placement entry includes decoded-message parameters so that
    # compare_msk144.py can match jt9 decode output back to known signals.
    manifest_path = Path(args.output_iq_wav).with_suffix('.json')
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_wav": str(Path(args.output_iq_wav)),
        "callsigns": args.callsigns,
        "channels": 2 if args.dual_channel else 1,
        "channel_format": "dual (ch0_I, ch0_Q, ch1_I, ch1_Q)" if args.dual_channel else "single (I=L, Q=R)",
        "ch0_antenna": "H",
        "ch1_antenna": "V" if args.dual_channel else None,
        "pol_scenario": args.pol_scenario if args.dual_channel else None,
        "output_rate": args.output_rate,
        "duration_s": args.duration,
        "noise_floor_dbfs": args.noise_floor_dbfs,
        "no_noise": args.no_noise,
        "atten_db": args.atten_db,
        "placements": [
            {k: v for k, v in p.items() if v is not None}
            for p in placements
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}  ({len(placements)} signals)")

    if not args.skip_plots:
        _plot_output_spectrogram(combined, args.output_rate, Path(args.plot_output), placements)


if __name__ == '__main__':
    main()
