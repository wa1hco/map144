"""Python port of WSJT-X msk144spd / msk144sync coherent frame-averaging decoder.

Architecture
------------
This module implements the short-ping-decoder (SPD) path used by WSJT-X's
mskrtd / msk144spd routines.  The key sensitivity improvement over the plain
jt9-WAV-file path comes from *coherent frame averaging*: because the MSK144
mark tone is at exactly 2000 Hz and each frame is exactly 72 ms, successive
frames are naturally phase-coherent.  Averaging N frames before running the
sync correlator improves SNR by √N before any decoding takes place.

Algorithm (mirrors msk144spd.f90 → msk144sync.f90 → msk144_freq_search.f90)
-----------------------------------------------------------------------------
1. Detection
   Slide an NSPM=864-sample (72 ms) window in steps of 216 samples (18 ms)
   over the complex baseband signal.  Square each window and FFT; look for
   tone energy at ±1000 Hz (the squared mark/space tones at baseband).
   Find up to MAXCAND=5 candidate frame positions.

2. Coherent averaging
   For each candidate, extract a 3-frame (3 × 864) window.  Try six averaging
   patterns (single-frame and two- and three-frame combinations, from
   navpatterns in msk144spd.f90).

3. Frequency search + averaging  (msk144_freq_search.f90)
   For each pattern, sweep ±ntol Hz in delf=2 Hz steps.  For each trial
   frequency, apply a continuous-phase correction to the multi-frame window,
   sum the selected frames, run the sync correlator, track the best peak.

4. Sync correlation  (msk144sync.f90)
   Correlate the 864-sample averaged frame simultaneously with both sync-word
   positions (offset 0 and 336, 0-indexed) using the 42-sample complex
   half-sine-pulse template.  Threshold: normalised peak ≥ 1.3 → nsuccess.

5. Timing recovery
   Circular-shift the averaged frame by ish_best to align the frame start.
   Try ish±1 to handle rounding (mirrors msk144spd is=1,2,3 loop).

6. Decode
   Reconstruct real 12 kHz audio from the phase-aligned complex frame and
   call jt9 for soft-bit extraction + LDPC decoding.

Constants
---------
NSPM    = 864     samples per MSK144 frame at 12 kHz (72 ms)
FS      = 12000   sample rate (Hz)
FC_JT9  = 1500    carrier frequency jt9 expects (Hz)
"""

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from .msk144_decode import msk144decodeframe as _msk144decodeframe

# ── Constants ─────────────────────────────────────────────────────────────────

NSPM   = 864        # samples per MSK144 frame at 12 kHz
FS     = 12000.0    # sample rate (Hz)
FC_JT9 = 1500.0     # jt9 expects the carrier at this frequency

# Success threshold for normalised sync peak (mirrors msk144sync.f90 line 98)
SYNC_THRESHOLD = 1.3

# Maximum decode candidates per input buffer
MAXCAND = 5

# Six averaging patterns for a 3-frame window (from msk144spd navpatterns)
# Each row is [frame0, frame1, frame2]; 1 = include in average
_NAV_PATTERNS = [
    [0, 1, 0],   # middle frame only        (tpat=1.5)
    [1, 0, 0],   # first frame only         (tpat=0.5)
    [0, 0, 1],   # last frame only          (tpat=2.5)
    [1, 1, 0],   # first two frames         (tpat=1.0)
    [0, 1, 1],   # last two frames          (tpat=2.0)
    [1, 1, 1],   # all three frames         (tpat=1.5)
]


# ── Sync template (built once at module import) ───────────────────────────────

def _build_sync_template() -> np.ndarray:
    """Build the 42-sample complex MSK144 sync-word template.

    Translated from msk144sync.f90 / msk144decodeframe.f90 (identical in both).
    Fortran convention (1-indexed) mapped to Python (0-indexed).

    s8 = [0,1,1,1,0,0,1,0]  →  bipolar s8 = [-1,+1,+1,+1,-1,-1,+1,-1]
    pp = sin((i-1)*π/12)  for i=1..12  (half-sine pulse, 12 samples)

    Q channel  (cbq, 42 samples):
        cbq[ 0: 6] = pp[6:12] * s8[0]
        cbq[ 6:18] = pp       * s8[2]
        cbq[18:30] = pp       * s8[4]
        cbq[30:42] = pp       * s8[6]

    I channel  (cbi, 42 samples):
        cbi[ 0:12] = pp       * s8[1]
        cbi[12:24] = pp       * s8[3]
        cbi[24:36] = pp       * s8[5]
        cbi[36:42] = pp[0:6]  * s8[7]
    """
    s8 = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.float32)
    s8 = 2 * s8 - 1                                    # bipolar
    pp = np.sin(np.arange(12) * np.pi / 12).astype(np.float32)

    cbq = np.zeros(42, dtype=np.float32)
    cbq[0:6]   = pp[6:12] * s8[0]
    cbq[6:18]  = pp       * s8[2]
    cbq[18:30] = pp       * s8[4]
    cbq[30:42] = pp       * s8[6]

    cbi = np.zeros(42, dtype=np.float32)
    cbi[0:12]  = pp       * s8[1]
    cbi[12:24] = pp       * s8[3]
    cbi[24:36] = pp       * s8[5]
    cbi[36:42] = pp[0:6]  * s8[7]

    return (cbi + 1j * cbq).astype(np.complex64)


_SYNC_CB = _build_sync_template()   # module-level singleton


# ── Sync correlator ───────────────────────────────────────────────────────────

def _sync_correlate(c: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Correlate a complex NSPM-sample frame with the MSK144 sync template.

    Mirrors msk144_freq_search.f90 line 36::

        cc(ish) = dot_product(ct2(1+ish:42+ish) + ct2(337+ish:378+ish), cb(1:42))

    Fortran ``dot_product(a, b)`` for complex arrays = Σ conj(a_i) * b_i.

    Both sync positions (offset 0 and 336, 0-indexed) are combined before the
    dot product, not after — this is the key to sensitivity.

    Parameters
    ----------
    c : complex64 array, length NSPM

    Returns
    -------
    xcc      : float32 array, length NSPM  — correlation magnitude per shift
    xmax     : float  — peak magnitude
    ish_best : int    — 0-based shift index of the peak
    """
    cb   = _SYNC_CB                          # (42,) complex64
    ct2  = np.concatenate([c, c])            # (2*NSPM,) doubled for circular shifts

    # Build index arrays for vectorised dot product over all 864 shifts
    k    = np.arange(42, dtype=np.int32)
    ish  = np.arange(NSPM, dtype=np.int32)
    idx1 = ish[:, None] + k[None, :]        # (NSPM, 42) — sync block at position 0
    idx2 = idx1 + 336                        # (NSPM, 42) — sync block at position 336

    combined = ct2[idx1] + ct2[idx2]         # (NSPM, 42) complex

    # Σ conj(combined) * cb  (Fortran dot_product convention)
    cc   = np.sum(np.conj(combined) * cb[None, :], axis=1)
    xcc  = np.abs(cc).astype(np.float32)
    ish_best = int(np.argmax(xcc))
    return xcc, float(xcc[ish_best]), ish_best


# ── Frequency search + coherent averaging ────────────────────────────────────

def _freq_search_avg(
    cbase_window: np.ndarray,
    navmask: list,
    ntol: float = 8.0,
    delf: float = 2.0,
) -> tuple:
    """Frequency search with coherent frame averaging.

    Mirrors msk144_freq_search.f90.  The input window is already at baseband
    (carrier removed).  We sweep residual frequency offsets ±ntol Hz and pick
    the one that produces the largest normalised sync peak.

    Phase coherence across frames:
        The mix ``exp(-j·2π·ferr·t)`` is applied to the *entire* window with
        continuous time (not reset per frame), so inter-frame phase drift due
        to the residual Doppler error is corrected before averaging.

    Parameters
    ----------
    cbase_window : complex64 array, shape (≥ nframes * NSPM,)
        Baseband IQ signal (carrier at 0 Hz).  May be longer than nframes*NSPM
        to allow non-circular timing extraction.
    navmask : list of int (0 or 1), length nframes
        Which frames to include in the coherent sum.
    ntol  : float — freq search half-width in Hz (default 8 Hz, inner search)
    delf  : float — freq step in Hz (default 2 Hz)

    Returns
    -------
    c_avg      : complex64 (NSPM,) — best coherently averaged frame (unshifted)
    xmax_norm  : float             — normalised sync peak (threshold 1.3)
    ferr_best  : float             — best residual frequency correction (Hz)
    xcc        : float32 (NSPM,)  — correlation magnitude array
    ish_best   : int               — best timing shift (0-based)
    best_mixed : complex64 (len(cbase_window),) — freq-corrected signal at best ferr
    """
    nframes = len(navmask)
    n_search = nframes * NSPM           # use exactly nframes*NSPM for freq search
    assert len(cbase_window) >= n_search, (
        f"cbase_window length {len(cbase_window)} < nframes*NSPM={n_search}"
    )

    navg = int(np.sum(navmask))
    if navg == 0:
        return None, 0.0, 0.0, None, 0, None

    fac = 1.0 / (48.0 * np.sqrt(float(navg)))      # normalisation (msk144_freq_search ln 18)

    n_ifr   = int(round(ntol / delf))
    t_full  = np.arange(len(cbase_window), dtype=np.float64) / FS   # continuous time over full window

    best_xmax  = -1.0
    best_c     = None
    best_ferr  = 0.0
    best_xcc   = None
    best_ish   = 0
    best_mixed = None

    for ifr in range(-n_ifr, n_ifr + 1):
        ferr  = ifr * delf

        # Continuous-phase mix across the whole window (preserves inter-frame coherence)
        mixed = cbase_window * np.exp(-1j * 2 * np.pi * ferr * t_full).astype(np.complex64)

        # Sum selected frames (use first n_search samples for sync)
        c = np.zeros(NSPM, dtype=np.complex64)
        for i, m in enumerate(navmask):
            if m:
                c += mixed[i * NSPM:(i + 1) * NSPM]

        # Sync correlation
        xcc, xmax_raw, ish = _sync_correlate(c)
        xb = xmax_raw * fac

        if xb > best_xmax:
            best_xmax  = xb
            best_c     = c.copy()
            best_ferr  = ferr
            best_xcc   = xcc.copy()
            best_ish   = ish
            best_mixed = mixed

    return best_c, best_xmax, best_ferr, best_xcc, best_ish, best_mixed


# ── jt9 decode of an aligned complex frame ───────────────────────────────────

def _frame_to_wav_and_decode(
    c_aligned: np.ndarray,
    fc_out: float,
    jt9_args: list | None = None,
) -> tuple[str | None, int | None]:
    """Convert an aligned complex baseband frame to real 12 kHz audio and call jt9.

    Parameters
    ----------
    c_aligned : complex64 (NSPM,) — frame aligned to symbol boundary (ish=0)
    fc_out    : float — carrier frequency in the output WAV (Hz); jt9 searches
                        a window around 1500 Hz so keep this close to FC_JT9.
    jt9_args  : list  — override jt9 argument list; None → JT9_BASE_ARGS

    Returns
    -------
    decoded_msg : str or None
    jt9_snr     : int or None

    WAV layout (matches detection.py / jt9 expectation):
        [100 ms zeros][500 ms AWGN][72 ms frame][500 ms AWGN][100 ms zeros]
    The 100 ms zero pads satisfy jt9's minimum-duration requirement.
    The 500 ms AWGN regions give jt9 a realistic noise baseline for its
    internal RMS normalisation and SNR estimation.
    """
    from map144_app.detection import JT9_BASE_ARGS
    if jt9_args is None:
        jt9_args = JT9_BASE_ARGS

    # Upconvert complex baseband → real audio at fc_out Hz
    t_frame  = np.arange(NSPM, dtype=np.float64) / FS
    audio_f  = np.real(c_aligned * np.exp(1j * 2 * np.pi * fc_out * t_frame)).astype(np.float32)

    # Estimate noise RMS from the frame itself (at low SNR, noise dominates std).
    # This is used to scale the AWGN padding so jt9 sees a realistic noise floor.
    noise_rms = float(np.std(audio_f))

    # Build WAV: [100 ms zeros][500 ms AWGN][frame][500 ms AWGN][100 ms zeros]
    pad_n  = int(0.100 * FS)   # 100 ms zeros  (jt9 minimum-duration guard)
    pre_n  = int(0.500 * FS)   # 500 ms AWGN   (jt9 noise baseline)
    post_n = int(0.500 * FS)   # 500 ms AWGN

    rng = np.random.default_rng()
    pre_noise  = (rng.standard_normal(pre_n)  * noise_rms).astype(np.float32)
    post_noise = (rng.standard_normal(post_n) * noise_rms).astype(np.float32)

    audio = np.concatenate([
        np.zeros(pad_n,  dtype=np.float32),
        pre_noise,
        audio_f,
        post_noise,
        np.zeros(pad_n,  dtype=np.float32),
    ])

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    else:
        return None, None

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(FS))
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        result = subprocess.run(
            jt9_args + [tmp_path],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.splitlines():
            tokens = line.split()
            if len(tokens) >= 6:
                msg = ' '.join(tokens[5:]).strip()
                try:
                    snr = int(tokens[1])
                except (ValueError, IndexError):
                    snr = None
                return msg, snr
    except Exception:
        pass
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return None, None


# ── Top-level decoder ─────────────────────────────────────────────────────────

def msk144_spd_decode(
    audio_real: np.ndarray | None = None,
    fc: float = FC_JT9,
    fs: float = FS,
    ntol: float = 200.0,
    jt9_args: list | None = None,
    success_threshold: float = SYNC_THRESHOLD,
    complex_baseband: np.ndarray | None = None,
) -> tuple[str | None, int | None, int, float, float]:
    """MSK144 short-ping decoder with coherent frame averaging.

    Python port of WSJT-X msk144spd (msk144spd.f90 + msk144sync.f90 +
    msk144_freq_search.f90).

    Parameters
    ----------
    audio_real       : float32 array — real 12 kHz audio containing the MSK144
                       signal at ``fc`` Hz.  Mutually exclusive with
                       ``complex_baseband``.
    fc               : float — nominal carrier frequency in Hz (default 1500).
                       Used only when ``audio_real`` is provided.
    fs               : float — sample rate; must be 12000 Hz (real audio path only)
    ntol             : float — outer freq search half-width for detection (Hz)
    jt9_args         : list  — override jt9 command arguments
    success_threshold: float — normalised sync peak threshold (default 1.3)
    complex_baseband : complex64 array — complex IQ at 12 kHz with the MSK144
                       signal already mixed to 0 Hz (baseband).  When provided,
                       the Hilbert transform and mix-down are skipped entirely,
                       recovering the ~3 dB noise-doubling penalty of the real-
                       audio path.  Pass the complex channelizer output (or
                       decimated complex IQ from an SDR) directly.

    Returns
    -------
    decoded_msg : str or None
    jt9_snr     : int or None  — SNR reported by jt9 (dB, 2500 Hz ref)
    navg        : int          — number of frames averaged for the decode
    fest        : float        — estimated carrier frequency (Hz); 0 Hz offset
                                 from baseband when complex_baseband is used
    xmax_norm   : float        — normalised sync peak of the best decode attempt
    """
    # ── Step 1: produce complex baseband ─────────────────────────────────────
    if complex_baseband is not None:
        # Direct complex IQ path — no Hilbert, no mix-down.
        # Input must already be at 0 Hz (baseband) at FS = 12 kHz.
        #
        # Scale to match the Hilbert analytic-signal convention used by the
        # Fortran msk144spd / msk144_freq_search code from which the sync
        # threshold (1.3) and normalisation constant (48) were derived.
        #
        # In the real-audio path, RMS-normalising the real audio to 1.0 and
        # then applying Hilbert produces an analytic signal with RMS ≈ √2
        # (both I and Q channels carry equal power).  The sync correlator and
        # its normalisation constant fac=1/48 were calibrated for that √2
        # scale.  A direct complex-IQ input RMS-normalised to 1.0 would be
        # √2 under-scaled, causing the normalised sync peak to fall √2 below
        # the 1.3 threshold for the same true SNR — a 3 dB detection penalty
        # that partially cancels the 2.5 dB Hilbert noise-doubling benefit.
        #
        # Fix: RMS-normalise to 1.0 then scale by √2 so the sync correlator
        # sees the same amplitude convention as the Hilbert path.  The decoder
        # (msk144decodeframe) normalises its soft bits internally and is
        # therefore unaffected by this constant factor.
        n = len(complex_baseband)
        rms = float(np.sqrt(np.mean(np.abs(complex_baseband.astype(np.complex128)) ** 2)))
        if rms < 1e-12:
            return None, None, 0, 0.0, 0.0
        _SQRT2 = np.float32(np.sqrt(2.0))
        cbase   = (complex_baseband * (_SQRT2 / rms)).astype(np.complex64)
        fc_base = 0.0
    else:
        # Real audio path: RMS normalise + Hilbert + baseband mix.
        # mskrtd.f90 normalises the 16-bit audio by its RMS before calling analytic():
        #   rms = sqrt(sum(d**2)/NZ)
        #   d   = d / rms
        # This ensures the noise floor is at 1.0 (in complex-amplitude units), which
        # is what the sync threshold of 1.3 was calibrated for.  Peak-normalisation
        # (to 0.9) suppresses the signal at low SNR (noise peak >> signal amplitude),
        # making the sync peak far too small.
        assert audio_real is not None, "Provide either audio_real or complex_baseband"
        assert abs(fs - FS) < 1, f"Expected fs={FS} Hz, got {fs}"
        n = len(audio_real)
        rms = float(np.sqrt(np.mean(audio_real.astype(np.float64) ** 2)))
        if rms < 1e-12:
            return None, None, 0, fc, 0.0
        audio_norm = (audio_real.astype(np.float64) / rms).astype(np.float32)
        analytic = hilbert(audio_norm.astype(np.float64)).astype(np.complex64)
        t_big    = np.arange(n, dtype=np.float64) / FS
        cbase    = analytic * np.exp(-1j * 2 * np.pi * fc * t_big).astype(np.complex64)
        fc_base  = fc

    # ── Step 2: Squared-spectrum detection (mirrors msk144spd.f90 ll 77-120) ─
    nstep  = (n - NSPM) // 216
    detmet = np.zeros(nstep, dtype=np.float32)
    detfer = np.zeros(nstep, dtype=np.float32)

    nfft  = NSPM
    df    = FS / nfft
    # After squaring baseband IQ: mark at +500 Hz → +1000 Hz; space at -500 Hz → -1000 Hz.
    # In an NFFT=864-bin FFT at 12 kHz, -1000 Hz wraps to bin NSPM - round(1000/df).
    idx_pos = int(round(1000.0 / df))            # +1000 Hz bin  ≈ 72
    idx_neg = NSPM - idx_pos                      # -1000 Hz bin  ≈ 792
    half_win = max(2, int(round(ntol * 2 / df)))  # search window half-width in bins

    # Raised-cosine edge window (rcw from msk144spd.f90 ll 57-59)
    rcw = ((1.0 - np.cos(np.arange(12) * np.pi / 12)) / 2).astype(np.float32)

    for istp in range(nstep):
        ns  = istp * 216
        ne  = ns + NSPM
        if ne > n:
            break
        seg_sq = (cbase[ns:ne] ** 2).copy()
        seg_sq[:12]      *= rcw
        seg_sq[NSPM-12:] *= rcw[::-1]

        spec     = np.fft.fft(seg_sq, n=nfft)
        tonespec = np.abs(spec) ** 2

        hi_lo = max(0, idx_pos - half_win)
        hi_hi = min(nfft // 2, idx_pos + half_win)
        lo_lo = max(nfft // 2, idx_neg - half_win)
        lo_hi = min(nfft, idx_neg + half_win)

        ah = float(np.max(tonespec[hi_lo:hi_hi])) if hi_hi > hi_lo else 0.0
        al = float(np.max(tonespec[lo_lo:lo_hi])) if lo_hi > lo_lo else 0.0
        detmet[istp] = max(ah, al)
        detfer[istp] = 0.0   # coarse freq error; inner freq search refines below

    # Normalise detection metric (median of lower quartile → noise floor = 1.0)
    # mirrors msk144spd.f90 ll 122-124
    q_idx = max(1, nstep // 4)
    sorted_met = np.sort(detmet[:nstep])
    xmed = float(sorted_met[q_idx]) if q_idx > 0 else 1.0
    if xmed > 0:
        detmet[:nstep] /= xmed

    # Find up to MAXCAND candidates where metric > 3.0
    dmet_copy  = detmet.copy()
    candidates: list[tuple[int, float]] = []
    for _ in range(MAXCAND):
        il = int(np.argmax(dmet_copy[:nstep]))
        if dmet_copy[il] < 3.0:
            break
        nstart_cand = il * 216          # 0-based sample index of candidate window
        candidates.append((nstart_cand, float(detfer[il])))
        dmet_copy[il] = 0.0

    if not candidates:
        return None, None, 0, fc, 0.0

    # ── Step 3: Sync / decode for each candidate ─────────────────────────────
    ntol_inner = 8.0    # ±8 Hz inner frequency search (msk144spd ntol0)
    delf_inner = 2.0    # 2 Hz steps

    for nstart_cand, _ferr_coarse in candidates:
        # Extract 3-frame baseband window (msk144spd: cdat = cbig(ib:ie))
        # Add one extra NSPM of margin so the timing extraction can index
        # beyond the 3-frame boundary without circular wrap.
        #
        # Window placement: the squared-spectrum detector peaks at the first
        # 216-sample step whose NSPM-sample window is fully covered by the signal
        # (nstart_cand ≥ signal_start, offset ∈ [0, 216)).  With the naive
        # placement ib = nstart_cand - NSPM, the signal falls at the END of the
        # first frame [NSPM-216:NSPM], so the coherent-averaged frame for any
        # single-frame nav-pattern sees only ONE of the two sync blocks — the
        # sync peak is halved and cannot pass the 1.3 threshold.
        #
        # Fix: shift ib back by one detection step (216 samples).  The signal
        # then falls at the BEGINNING of the middle frame [0:216], so both sync
        # blocks are visible and the sync correlation gives the correct ish_best.
        ib = max(0, nstart_cand - NSPM - 216)
        ie_search = ib + 3 * NSPM
        if ie_search > n:
            ie_search = n
            ib = max(0, ie_search - 3 * NSPM)

        # Extended window: up to 4*NSPM from ib (zero-pad if near end of cbase)
        ie_ext = min(ib + 4 * NSPM, n)
        cbase_ext = cbase[ib:ie_ext]
        if ie_ext < ib + 3 * NSPM:
            continue     # not enough data even for the search window

        # Pad to exactly 3*NSPM for the freq-search portion
        if len(cbase_ext) < 3 * NSPM:
            continue
        cbase_3    = cbase_ext[:3 * NSPM]

        for navmask in _NAV_PATTERNS:
            # Freq search uses exactly 3*NSPM; pass extended window so best_mixed
            # can be used for non-circular timing extraction below.
            c_avg, xmax_norm, ferr_best, xcc, ish_best, best_mixed = _freq_search_avg(
                cbase_ext, navmask,
                ntol=ntol_inner, delf=delf_inner,
            )

            if c_avg is None or xmax_norm < success_threshold:
                continue

            navg_val = int(np.sum(navmask))

            # Find top-2 timing peaks (msk144spd npeaks=2)
            xcc_copy  = xcc.copy()
            peak_locs = []
            for _ in range(2):
                pk = int(np.argmax(xcc_copy))
                peak_locs.append(pk)
                lo = max(0, pk - 7)
                hi = min(NSPM, pk + 8)
                xcc_copy[lo:hi] = 0.0

            # Try each peak location and ±1 timing nudge (msk144spd is=1,2,3)
            nframes = len(navmask)
            mixed_len = len(best_mixed)
            for ic0 in peak_locs:
                for delta in (0, -1, 1):
                    ic = (ic0 + delta) % NSPM

                    # Non-circular frame extraction: sum selected frames starting
                    # at offset ic within each frame in best_mixed.  Avoids the
                    # circular-wrap artefact that would corrupt the last ~ic
                    # matched-filter windows if we used np.roll instead.
                    c_shifted = np.zeros(NSPM, dtype=np.complex64)
                    for fi, m in enumerate(navmask):
                        if m:
                            start = fi * NSPM + ic
                            end   = start + NSPM
                            if end <= mixed_len:
                                c_shifted += best_mixed[start:end]
                            else:
                                # Near end of buffer — take what's available and
                                # zero-pad the remainder (affects at most a few
                                # matched-filter windows near the frame boundary)
                                avail = max(0, mixed_len - start)
                                if avail > 0:
                                    c_shifted[:avail] += best_mixed[start:start + avail]

                    # Soft-decision decode directly on aligned frame (no jt9 re-detection)
                    decoded_msg, _nhe = _msk144decodeframe(c_shifted)
                    jt9_snr = None
                    if decoded_msg is not None:
                        fest = fc_base + ferr_best
                        return decoded_msg, jt9_snr, navg_val, fest, xmax_norm

    return None, None, 0, fc, 0.0


# ── Convenience wrapper for decode_sensitivity.py ────────────────────────────

def spd_run_trial(
    audio_real: np.ndarray,
    fc: float = FC_JT9,
    jt9_args: list | None = None,
) -> tuple[str | None, int | None]:
    """Thin wrapper: decode a pre-built 12 kHz real audio array.

    Parameters
    ----------
    audio_real : float32 ndarray, 12 kHz real audio
    fc         : carrier frequency (Hz, default 1500)
    jt9_args   : override jt9 args

    Returns
    -------
    decoded_msg : str or None
    jt9_snr     : int or None
    """
    msg, snr, _navg, _fest, _xmax = msk144_spd_decode(audio_real, fc=fc, jt9_args=jt9_args)
    return msg, snr
