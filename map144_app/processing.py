# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; wit hout even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""FFT and channelised MSK144 detection pipeline for incoming IQ sample data.

Pipeline (in order of execution)
---------------------------------
0. **Two-stage noise blanker** — shared Hann-tapered soft blanking mask.

   **0a. Wideband FFT blanker (Linrad-style)** — input is divided into
   NB_FFT_SIZE blocks.  Each block is FFT'd; the per-bin power is normalised
   by a per-bin running average.  If more than NB_BROADBAND_FRAC of bins
   have normalised power above nb_factor² the block is a broadband impulse
   and is zeroed.  Catches impulses ≥ NB_FFT_SIZE samples (~5.3 ms).

   **0b. Time-domain amplitude blanker** — per-sample magnitude compared to
   nb_factor × _nb_floor.  Catches sub-block impulses (e.g. 2-sample ~42 µs
   spikes) that are diluted below the FFT hot-bin threshold.  Uses the noise
   floor from the previous call, so the impulse cannot inflate its own threshold.
   Both stages add to a shared blank_mask; the Hann taper is applied once.

1. **Ring buffer** — cleaned IQ is written to a 30-second circular buffer
   (``Engine._iq_ring``) so ``extract_and_decode`` can read a long window
   around a trigger (aligned with Fast Graph capture regions).

2. Channeliser -- apply_channelizer() splits the 48 kHz IQ block into
   N_CHANNELS=48 channels at CH_SAMPLE_RATE=12 kHz each.  Channel k mixes
   down k x 1000 Hz + channel_offset_hz on the IF, where channel_offset_hz
   = (calling_freq_hz - pan_center_hz) + 1500 Hz.  The +1500 aligns each NCO
   with the USB signal energy (dial + 1500 Hz) for stations on calling_freq + k kHz
   dial.  See channelizer.py.

3. **Per-channel detection** — channelised samples are accumulated in
   ``_ch_buf`` (48 × N) until CH_DETECT_SIZE samples are available.  For each
   complete block:

   a. Square the complex channel samples.
   b. Windowed FFT of the squared signal at 12 kHz.
   c. Pair metric: dB above a rolling 25th-percentile baseline of linear peaks
      in the squared-domain ±1 kHz tone windows (see ``_LO_MASK`` / ``_HI_MASK``).
   d. Coincidence and sustained-signal gates; carrier estimate from peak bins;
      optional ``extract_and_decode`` in a daemon thread.

4. **SNR history** — ``pair_metric`` (48,) is written to ``_ch_snr_history``
   (N_SNR_HIST × 48).  ``displays.py`` draws the 48-channel ridgeline.

5. **Overlap-FFT waterfall** — cleaned IQ in ``_sbuf`` is processed in fft_size
   blocks with 50 % overlap.  Rows are placed using **packet timestamps** so
   the spectrogram matches receive time.

Timing (two clocks)
-------------------
The detection heatmap indexes rows by **wall time** — ``time.time()`` for live
sources and ``_wav_time_cursor`` for WAV — so multiple detection hops inside one
``process_iq_data`` call map to distinct rows.  The waterfall uses **sample
timestamps** (``timestamp_int`` + picosecond ``timestamp_frac``) for
sample-accurate placement; see the ``process_iq_data`` docstring.

Constants
---------
DETECT_THRESH_DB    : 3.5 dB  — threshold: dB above 25th-percentile baseline
DETECT_MERGE_GAP_S  : 1.0 s   — cooldown between triggers on the same channel
CH_DETECT_SIZE      : 512     — per-channel FFT size for detection (at 12 kHz)
_METRIC_HIST_DEPTH  : 300     — rolling history depth for percentile normaliser
N_SNR_HIST          : computed — spans ~one 15 s window at the channeliser hop rate
"""

import json as _json
import logging
import sys as _sys
import threading
import time

import scipy.fft as _sp_fft   # multi-threaded FFT for the wideband TFMF surface
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── CuPy / GPU initialisation ────────────────────────────────────────────────
# Two failure modes to handle gracefully:
#   1. cupy not installed (e.g. CPU-only laptop): ImportError on import.
#   2. cupy installed but GPU not accessible (driver mismatch, headless ssh,
#      no NVIDIA hardware): import succeeds but the first cuda call raises.
# Both must drop cleanly back to the numpy CPU path; only the GPU compute
# path checks _HAS_CUPY, so a False here disables the GPU path everywhere.
#
# The memory pool cap is scaled to half of available VRAM (so we don't
# starve Xorg/other GPU clients) and additionally capped at 4 GiB — peak
# per-dispatch alloc is ~1.4 GiB, so 4 GiB is comfortable on 16 GiB cards
# and 50 % of 4 GiB (T2000) gives 2 GiB which is the minimum that fits a
# full 15-s surface.
try:
    import cupy as _cp                                  # type: ignore[import-not-found]
    # Probe the runtime — raises if no driver / no GPU.
    _cp.cuda.runtime.runtimeGetVersion()
    _cp_dev_props = _cp.cuda.runtime.getDeviceProperties(0)
    _cp_dev_name  = _cp_dev_props['name'].decode() if isinstance(
        _cp_dev_props['name'], (bytes, bytearray)) else str(_cp_dev_props['name'])
    _cp_dev_total = int(_cp_dev_props['totalGlobalMem'])
    _cp_pool_cap  = min(_cp_dev_total // 2, 4 * 1024 ** 3)
    _cp.get_default_memory_pool().set_limit(size=_cp_pool_cap)
    _HAS_CUPY = True
    logger.info(
        "[tfmf] GPU: %s, %.0f MiB total, pool cap %.0f MiB",
        _cp_dev_name, _cp_dev_total / (1024 ** 2), _cp_pool_cap / (1024 ** 2),
    )
except Exception as _cupy_err:   # ImportError, RuntimeError, AttributeError, …
    _cp = None                                          # type: ignore[assignment]
    _HAS_CUPY = False
    logger.info("[tfmf] CPU fallback (cupy unavailable: %s)",
                type(_cupy_err).__name__ + ": " + str(_cupy_err))

import numpy as np
from scipy.ndimage import convolve1d as _nd_convolve1d

from .channelizer import (
    apply_channelizer,
    N_CHANNELS,
    CHANNEL_SPACING_HZ,
    CH_SAMPLE_RATE,
)
from .channel_plan import NYQUIST_EDGE_CHANNEL_SKIP
from .detection import extract_and_decode, production_output_dir, _read_ring
from .msk144_spd import (
    _sync_correlate as _msk144_sync_correlate,
    _sync_correlate_batch as _msk144_sync_correlate_batch,
    _sync_phase_features as _msk144_sync_phase_features,
    NSPM as _SYNC_NSPM,
)

# WSJT-X-faithful sq_det port (2026-05-15).  Used for SHADOW logging at launch
# time only — does NOT affect triggering.  Per-launch enrichment so we can
# compare the new metric against the legacy `sq_metric_db_h` field after a run.
# Disable via env: MAP144_SHADOW_SQDET=0
import os as _os
from .sq_det import sq_det as _sq_det_new, to_db as _sq_to_db
_SHADOW_SQDET_ENABLED = _os.environ.get('MAP144_SHADOW_SQDET', '1') != '0'

# Step 1 of sensitivity-improvement plan — coherent MSK144 sync-word
# correlator running in parallel with the squared-FFT detector.  Feature flag
# so the change is bisectable (flip to False to recover the legacy path).
ENABLE_SYNC_DETECT = True
SYNC_DETECT_THRESH_DB = 3.0    # dB above 25th-percentile sync-magnitude baseline

DETECT_THRESH_DB   = 3.0        # dB above 25th-percentile noise baseline

# Diagnostic raw-power display offsets.  When the operator enables
# Diagnostics → "Show raw power", the heatmap shows raw power instead
# of dB-above-pct25.  The squared-FFT and sync-correlator paths sit at
# very different absolute scales, so each needs its own offset to land
# the typical noise floor in the same display window the existing
# colour-scale sliders are calibrated for (~0..30 dB).
#
# Squared-FFT path: 10·log10(raw_lin) sits around −90..−50 dBFS on this
# Flex setup.  +85 dB offset → typical −80 dBFS floor maps to +5 dB
# displayed, well inside the slider window.
#
# Sync path: peaks magnitude is ~√(2·42) × |c|_rms = 9.2 × |c|_rms in
# *linear* IQ units.  For a typical Flex setup |c|_rms is ~1e-3..1e-2,
# so 10·log10(peaks) lands around −20..−10 dB at the noise floor (NOT
# the +9.5 dB I estimated initially — that assumed unit-amplitude IQ,
# which is full-scale, not realistic).  Offset +30 dB puts the noise
# floor in the +10..+20 dB display window for typical Flex gain.  Real
# levels depend on user gain settings — if sync display sits dark or
# saturates, this offset is the calibration to adjust.
#
# Channel-to-channel deltas are preserved exactly in both cases — only
# the absolute axis is shifted.
_RAW_POWER_DISPLAY_OFFSET_DB_SQ   = 85.0
_RAW_POWER_DISPLAY_OFFSET_DB_SYNC = 30.0
DETECT_MERGE_GAP_S = 0.5        # suppress re-trigger on the same channel within this window (s).
                                 # Per-channel cooldown: parallel channels are fully independent,
                                 # so simultaneous pings at different frequencies are unaffected.
                                 # 500 ms covers the longest underdense ping (5τ at τ=100 ms);
                                 # overdense pings that last longer will re-trigger naturally once
                                 # the cooldown expires, which is desirable for long trails.
CH_DETECT_SIZE     = 512        # samples at 12 kHz per detection FFT block
NB_FACTOR          = 6.0        # blanker amplitude threshold (hot bin power > NB_FACTOR² × avg)
NB_FFT_SIZE        = 256        # detection block size: 5.3 ms at 48 kHz
NB_BROADBAND_FRAC  = 0.30       # fraction of hot bins required to declare broadband impulse
NB_SPEC_AVG_TC     = 2.0        # per-bin spectrum averaging time constant (seconds)
NB_WIDEBAND_AVGNUM = 200        # EMA length for wideband median floor (~1 s at 5.3 ms/block)
_NB_LOWLEVEL_ALPHA = 0.02       # EMA weight for clean-fraction tracker (matches Linrad)
_NB_TAPER_N        = 2          # Hann taper half-width in samples (2 → 42 µs at 48 kHz)
                                # A 3-sample impulse becomes 7 samples blanked (2.3×).
                                # Raise if blanking edges cause spectral artifacts.
NB_SHORT_IMPULSE_THRESH = 5    # FFT blocks where ≤ this many samples exceed the TD threshold
                                # are treated as short impulses; block-blanking is skipped and
                                # the sample-precise TD mask is used instead.
# Alias for readability — value lives in ``channel_plan`` (UI + detect share it).
_EDGE_CH_SKIP      = NYQUIST_EDGE_CHANNEL_SKIP
# NOTE: not currently used anywhere in processing.py.  MAP144's
# detection invariant is that the 48 kHz IQ at the engine input
# always has CLEAN DC — channels at ch_signed = 0, ±1, ±2 are
# LEGITIMATE operating frequencies (real QSOs at calling±1/±2 kHz
# happen routinely) and must NOT be suppressed.  How DC stays clean
# depends on the radio:
#   - Flex (digital IQ, oversampled): pan center IS the calling freq,
#     DC genuinely zero by design — no offset, no decimation in MAP144.
#   - B210 (analog IQ, real DC artifact): handled at the RADIO CONFIG
#     layer — tune offset + run at higher sample rate + decimate to
#     48 kHz, which moves the real DC out of band before MAP144 sees
#     the IQ.  Not MAP144's responsibility.
# The DetectorBlock (Block-pipeline path) carries a configurable
# ``dc_ch_skip`` as a last-resort hardware escape hatch, but its
# default is 0 to match this invariant.  See memory entry
# ``project_dc_lo_offset`` for the full rationale.
_DC_CH_SKIP        = 0          # default: no DC suppression (see note above)
_SQ_TONE_HZ        = 1000.0     # expected squared-domain tone offset from DC
_SQ_NTOL_HZ        = 200.0      # half-width of each tone search window
_METRIC_HIST_DEPTH = 1500       # frames of rolling linear-peak history for percentile.
                                 # At the 256-sample / 12 kHz hop rate each frame is ~21 ms,
                                 # so 1500 frames ≈ 32 s.  A longer window gives a stable
                                 # per-channel noise floor estimate: narrowband IF spurs (which
                                 # occupy fixed frequencies for seconds at a time) are averaged
                                 # into the baseline rather than transiently biasing it.
                                 # As long as pings occupy < 25 % of that window the 25th
                                 # percentile stays at the noise floor, mirroring the xmed
                                 # normalisation in WSJT-X msk144spd.f90.  Memory cost for
                                 # the history: 1500 × 48 ch × 4 B ≈ 288 KB (negligible).
_METRIC_HIST_STRIDE = 5         # subsample history for partition — evaluate every 5th frame
                                 # (300 effective samples, ~107 ms resolution) at 1/5 the
                                 # compute cost.  Called every 10 hops (~213 ms), so the
                                 # partition sorts 300 floats — well under 1 ms.
_COINCIDENCE_MAX_CLUSTERS = 6   # cluster count gate: if ≥ this many SEPARATE narrow clusters
                                 # fire simultaneously it is likely broadband noise, not multiple
                                 # independent meteor pings.  Each real MSK144 ping spreads into
                                 # 3 adjacent channels (squared-domain tones land at ±1000 Hz,
                                 # straddling the 1 kHz channel boundary), forming one cluster of
                                 # span 2.  Three simultaneous pings at 5 kHz spacing → 3 clusters
                                 # of 3 channels each → 9 total channels; the old raw channel count
                                 # gate (≥8) would incorrectly fire on this legitimate scenario.
                                 # Counting clusters instead: 6 simultaneous pings → 6 clusters
                                 # → gate fires (extremely rare even during a shower peak).
                                 # The per-cluster span gate (_COINCIDENCE_SPAN_CH) still handles
                                 # true wideband impulse noise that forms a single wide cluster.
_SUSTAINED_HOPS      = int(2.0 * CH_SAMPLE_RATE / (CH_DETECT_SIZE // 2))  # 2 s worth of hops
_SUSTAINED_LOCKOUT   = int(30.0 * CH_SAMPLE_RATE / (CH_DETECT_SIZE // 2)) # 30 s lockout
_COINCIDENCE_SPAN_CH = 8        # spread gate: if a single cluster spans > this many channels it
                                 # is wideband noise (real MSK144 ping + sidelobes spans ≤ 3
                                 # channels; impulse noise spans 10+ kHz = 10+ channels).
_CH_DETECT_HOP     = CH_DETECT_SIZE // 2   # 50 % overlap hop
# Enough slots to cover one full 15-second window at the channeliser hop rate
N_SNR_HIST = int(15 * CH_SAMPLE_RATE / _CH_DETECT_HOP)        # = 703 (exact integer hops)
# 703 hops × (256 / 12000) s/hop = 14.997 s (3 ms shy of 15 s, well below display
# resolution).  Choosing this clean integer ratio means write_idx = hop_count %
# N_SNR_HIST advances by exactly 1 per hop with no rounding noise — no gap-fill,
# no wipe-and-refill cycle needed.  The 15-s boundary is a display / TX-RX
# concept; internal DSP state (rolling baselines, AGC) is unaffected.

# Pre-computed Hann taper kernel for soft-blanking (avoids rectangular-hole sinc spread)
_NB_TAPER_KERNEL = np.hanning(2 * _NB_TAPER_N + 1).astype(np.float32)
_NB_TAPER_KERNEL /= _NB_TAPER_KERNEL.max()   # normalise peak to 1.0

# Pre-computed constants for the detection hot loop (computed once at import time)
_DETECT_WINDOW = np.hanning(CH_DETECT_SIZE).astype(np.float32)

# ── TFMF display surface constants ────────────────────────────────────────────
# One surface is computed per 15-s period (at the boundary) from the last 15 s
# of post-blanker 48 kHz IQ pulled from the IQ ring.  TFMF is a wideband matched
# filter: running it pre-channelizer (full ±24 kHz audio) avoids the channelizer
# filter response colouring the noise floor and gives the full DAX-IQ search
# bandwidth.
_TFMF_DISP_FS_HZ    = 48_000        # post-blanker IQ rate from DAX-IQ
_TFMF_DISP_STRIDE   = 24            # samples between TFMF windows (0.5 ms at 48 kHz)
_TFMF_DISP_TMPL_N   = 1536          # dual-sync template length at 48 kHz (64 sym × 24 sps)
_TFMF_FREQ_MIN_KHZ  = -_TFMF_DISP_FS_HZ / 2 / 1000.0   # −24.0 kHz (after fftshift)
_TFMF_FREQ_SPAN_KHZ = _TFMF_DISP_FS_HZ / 1000.0        # 48.0 kHz
_TFMF_PERIOD_SAMPLES = 15 * _TFMF_DISP_FS_HZ           # 720 000 samples
# How often to re-dispatch the surface compute (seconds).  The busy-gate
# drops overlapping dispatches, so the effective rate is bounded by
# GPU compute time (~50-200 ms early in a period, ~600 ms at full 15 s).
# 0.1 = 10 Hz nominal, settling to ~1.4 Hz once the surface fills the period.
_TFMF_REDISPLAY_INTERVAL_S = 0.1
# Impulse-row rejection threshold.  Each surface time-row's median-across-freq
# magnitude is compared to the global median of those row-medians.  A row
# whose ratio exceeds this threshold is treated as a broadband impulse
# (energy spread across all freq bins of that time window) — its candidates
# are dropped.  Without this, a single short impulse produces ~46 spurious
# circles spread across freq.  3.0 catches roughly +10 dB impulses and above;
# raise to be less aggressive, lower for more impulses caught.
_TFMF_IMPULSE_ROW_THRESHOLD = 3.0

# Per-time-bucket burst suppression.  After peak-pick + impulse-row rejection,
# bucket the remaining candidates by their period-relative time into
# ``_TFMF_BURST_BUCKET_S``-wide windows.  If a bucket has more than
# ``_TFMF_BURST_THRESHOLD`` candidates, drop the bucket entirely — it's a
# narrowband-or-mixed noise burst that survived impulse-row rejection but
# clearly isn't a single localised signal.  Real MS pings produce 1–5
# candidates in a 200 ms bucket; 10 leaves comfortable headroom.
_TFMF_BURST_BUCKET_S = 0.2
_TFMF_BURST_THRESHOLD = 10

# Cross-dispatch dedup tolerance (seconds, Hz).  Each dispatch's peak-pick
# deduplicates within its own surface, but two candidates within one peak-
# pick footprint can land on either side of a dispatch boundary and both
# survive into the accumulator.  When extending the accumulator, drop a
# new candidate if any existing one is within these distances.  Match the
# peak-pick footprint (15 windows × 1 kHz half-width × 2).
_TFMF_DEDUP_TIME_S = 0.0075
_TFMF_DEDUP_FREQ_HZ = 1000.0
# Lazy-initialised dual-sync template (built once on first period boundary)
_TFMF_TEMPLATE_48K: 'np.ndarray | None' = None
_TFMF_TEMPLATE_48K_CP = None    # cupy.ndarray when _HAS_CUPY, else unused

# Frequency bands to suppress in TFMF candidate output.  The detector's
# per-bin median normalisation cleanly handles flat passband noise, but two
# real-world structures violate the AWGN model and produce dense FA clusters:
#   ±22-24 kHz : Flex DAX-IQ analog anti-alias filter rolloff zone — depressed
#                noise floor inflates relative SNR.
#   −10 kHz    : observed PSU harmonic at 50.249 MHz with the operator's
#                pan center 50.259 MHz; non-Rayleigh narrowband content.
# Each entry is (lo_hz, hi_hz), inclusive lo / exclusive hi.
_TFMF_EXCLUDED_BANDS_HZ = (
    # Extended to ±20-24 kHz after the first overnight bake: the band-edge
    # cluster persisted past ±22 kHz into ±20-22 kHz (6.4 k cands in the
    # +20..+22 bin alone, second-largest FA peak after the calling-freq
    # channel).  ±20 kHz exclusion still leaves +0..+19 kHz audio search,
    # well past the useful 6-m MSK144 band.
    (-24500.0, -20000.0),
    ( 20000.0,  24500.0),
    (-10500.0,  -9500.0),
)


def _tfmf_in_excluded_band(freq_hz: float) -> bool:
    for _lo, _hi in _TFMF_EXCLUDED_BANDS_HZ:
        if _lo <= freq_hz < _hi:
            return True
    return False


def _build_tfmf_display_surface(iq_samples: np.ndarray):
    """Compute the TFMF SNR surface and peak-picked candidates for the sync-det display.

    GPU path is used when CuPy is importable (~10 ms surface + ~500 ms peak-pick
    per 15-s period on RTX-class hardware); CPU fallback otherwise (~4-7 s).
    Templates are kept resident on the GPU between calls to avoid retransfer.

    Returns
    -------
    snr_db : float32 ndarray shape (n_windows, _TFMF_DISP_TMPL_N), freq axis
             fftshifted (DC in centre column).  None when the input is too short.
    candidates : list[Candidate]  peak-picked detections.  Candidates falling
                 inside ``_TFMF_EXCLUDED_BANDS_HZ`` (band-edge rolloff, known
                 spurs) are dropped before returning.
    """
    global _TFMF_TEMPLATE_48K, _TFMF_TEMPLATE_48K_CP
    if _TFMF_TEMPLATE_48K is None:
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from experiments.gpu_tfmf.sync_template import build_dual_sync_template
        _TFMF_TEMPLATE_48K = build_dual_sync_template(_TFMF_DISP_FS_HZ)

    h = _TFMF_TEMPLATE_48K
    N = len(h)                           # 1536
    stride = _TFMF_DISP_STRIDE           # 24
    iq = np.ascontiguousarray(iq_samples, dtype=np.complex64)
    n_samp = len(iq)
    n_windows = max(0, (n_samp - N) // stride + 1)
    if n_windows == 0:
        return None, []

    if _HAS_CUPY:
        # ── GPU path ──────────────────────────────────────────────────────
        from experiments.gpu_tfmf.tfmf import (
            compute_tf_surface_gpu, extract_candidates, TFMFConfig,
        )
        if _TFMF_TEMPLATE_48K_CP is None:
            _TFMF_TEMPLATE_48K_CP = _cp.asarray(h, dtype=_cp.complex64)
        iq_cp = _cp.asarray(iq, dtype=_cp.complex64)
        surface_cp = compute_tf_surface_gpu(
            iq_cp, _TFMF_TEMPLATE_48K_CP, stride_samples=stride,
        )
        # Noise floor is the 25th percentile of |MF| per freq bin (sq_det's
        # convention; tolerates ≤75% signal contamination per bin vs ≤50%
        # for median).  Threshold above pct25 needs to be +3.82 dB above
        # the median-based threshold to give the same AWGN FA rate:
        # 13.5 dB above median → 17.3 dB above pct25.
        cfg = TFMFConfig(
            sample_rate_hz=_TFMF_DISP_FS_HZ,
            stride_samples=stride,
            threshold_db=17.3,
            noise_floor_percentile=25.0,
        )
        candidates = extract_candidates(surface_cp, cfg, template_length=N)
        # Display surface: per-bin pct25 normalisation, fftshifted, in dB.
        # Uses a 10× subsample to keep the partition cheap (1.6% relative
        # error vs full pct25 on AWGN).
        magnitude_cp = _cp.abs(surface_cp)
        _sub_cp = magnitude_cp[::10]
        _k_pct25 = max(0, _sub_cp.shape[0] // 4)
        noise_floor_cp = _cp.partition(_sub_cp, _k_pct25, axis=0)[_k_pct25]
        noise_floor_cp = _cp.maximum(noise_floor_cp, 1e-30)
        ratio_cp = _cp.maximum(magnitude_cp / noise_floor_cp[None, :], 1e-30)
        snr_db_unshifted_cp = (20.0 * _cp.log10(ratio_cp)).astype(_cp.float32)
        snr_db = _cp.asnumpy(_cp.fft.fftshift(snr_db_unshifted_cp, axes=1))
        # Impulse-row mask: row-median(|MF|) across freq, then compare to
        # the global median.  Broadband impulses elevate the whole row.
        row_median_np = _cp.asnumpy(_cp.median(magnitude_cp, axis=1))
        # Explicitly release per-dispatch GPU intermediates.  Peak alloc per
        # 15-s dispatch is ~1.1 GB (surface + magnitude + ratio + snr stages);
        # without the explicit del + free_all_blocks the pool would hold onto
        # them until Python GC, which can lag the next dispatch and cause an
        # OOM at higher update rates.  The template (shared across dispatches)
        # stays resident.
        del iq_cp, surface_cp, magnitude_cp, ratio_cp
        del noise_floor_cp, snr_db_unshifted_cp
        _cp.get_default_memory_pool().free_all_blocks()
    else:
        # ── CPU path ──────────────────────────────────────────────────────
        # Build (n_windows, N) matrix via advanced indexing — one copy, contiguous.
        row_idx = np.arange(n_windows, dtype=np.int32)[:, np.newaxis] * stride
        col_idx = np.arange(N, dtype=np.int32)[np.newaxis, :]
        windows = iq[row_idx + col_idx]
        h_conj  = np.conj(h).astype(np.complex64)
        products = windows * h_conj[np.newaxis, :]
        # Multi-threaded scipy FFT — ~2-3× faster than numpy on this shape.
        surface  = _sp_fft.fft(products, axis=1, workers=-1).astype(np.complex64)

        magnitude_unshifted = np.abs(surface)
        # Per-bin noise floor: 25th percentile, 10× subsample.  pct25 tolerates
        # up to ~75% signal contamination per bin vs ~50% for median; under
        # pure AWGN, pct25 ≈ 0.7585·σ vs median ≈ 1.177·σ, so equivalent FA-rate
        # threshold is +3.82 dB above the median-based value (13.5 → 17.3 dB).
        _sub = magnitude_unshifted[::10]
        _k_pct25 = max(0, _sub.shape[0] // 4)
        noise_floor = np.partition(_sub, _k_pct25, axis=0)[_k_pct25]
        noise_floor = np.maximum(noise_floor, 1e-30)
        ratio_unshifted = np.maximum(
            magnitude_unshifted / noise_floor[np.newaxis, :], 1e-30,
        )
        snr_db_unshifted = 20.0 * np.log10(ratio_unshifted).astype(np.float32)
        snr_db = np.fft.fftshift(snr_db_unshifted, axes=1)
        candidates = _peak_pick_from_snr_db(
            snr_db_unshifted, magnitude_unshifted,
            stride_samples=stride, template_length=N,
            sample_rate_hz=_TFMF_DISP_FS_HZ,
            peak_picking_time_windows=15,
            peak_picking_freq_hz=500.0,
            threshold_db=17.3,
        )
        # Per-row median across freq — used by impulse-row rejection below.
        row_median_np = np.median(magnitude_unshifted, axis=1)

    # Impulse-row rejection: time windows whose median magnitude across freq
    # is anomalously high are broadband impulses, not localised signals.  A
    # single 500 ms burst can produce ~100 spurious candidates without this
    # filter; with it, those rows' candidates are dropped.  See
    # ``_TFMF_IMPULSE_ROW_THRESHOLD`` for the cutoff.
    if candidates and row_median_np.size:
        _global_med = float(np.median(row_median_np))
        if _global_med > 0:
            _impulse_thresh = _TFMF_IMPULSE_ROW_THRESHOLD * _global_med
            _impulse_mask = row_median_np > _impulse_thresh   # (n_windows,)
            if _impulse_mask.any():
                _n_w = row_median_np.shape[0]
                _half_n = N // 2
                _kept = []
                for _c in candidates:
                    _row = (_c.time_sample - _half_n) // stride
                    if 0 <= _row < _n_w and _impulse_mask[_row]:
                        continue
                    _kept.append(_c)
                candidates = _kept

    # Drop candidates in known-noisy bands (band-edge rolloff, PSU spur).
    # Configured in ``_TFMF_EXCLUDED_BANDS_HZ`` at module top.
    if candidates and _TFMF_EXCLUDED_BANDS_HZ:
        candidates = [c for c in candidates
                      if not _tfmf_in_excluded_band(c.freq_hz)]

    # Per-time-bucket burst suppression: candidates that cluster densely in
    # time are a noise burst, not a localised signal.  Bucket by
    # _TFMF_BURST_BUCKET_S; drop buckets with > _TFMF_BURST_THRESHOLD entries.
    if candidates and _TFMF_BURST_THRESHOLD > 0:
        from collections import defaultdict
        _buckets = defaultdict(list)
        for c in candidates:
            _buckets[int(c.time_s / _TFMF_BURST_BUCKET_S)].append(c)
        candidates = [c for _, cs in _buckets.items()
                      if len(cs) <= _TFMF_BURST_THRESHOLD
                      for c in cs]

    return snr_db, candidates


def _peak_pick_from_snr_db(snr_db_unshifted: np.ndarray,
                            magnitude_unshifted: np.ndarray,
                            *,
                            stride_samples: int,
                            template_length: int,
                            sample_rate_hz: int,
                            peak_picking_time_windows: int,
                            peak_picking_freq_hz: float,
                            threshold_db: float) -> list:
    """2-D local-max peak-pick on a precomputed SNR-dB surface.

    Equivalent to ``experiments.gpu_tfmf.tfmf.extract_candidates_cpu`` but
    skips the duplicate magnitude / median computation by accepting them as
    inputs.  Saves ~1.9 s on the wideband 48 kHz surface.

    Inputs are un-shifted (fftfreq ordering) — same as extract_candidates_cpu.
    """
    from scipy.ndimage import maximum_filter as _mf
    from experiments.gpu_tfmf.tfmf import (
        Candidate, _parabolic_subpixel, _parabolic_freq_offset,
    )

    n_freq = snr_db_unshifted.shape[1]
    freq_res = sample_rate_hz / n_freq
    footprint_t = peak_picking_time_windows
    footprint_f = max(1, round(2.0 * peak_picking_freq_hz / freq_res))
    if footprint_f % 2 == 0:
        footprint_f += 1

    local_max = _mf(snr_db_unshifted, size=(footprint_t, footprint_f),
                     mode='constant', cval=float('-inf'))
    is_peak = (snr_db_unshifted >= local_max - 1e-6) & (snr_db_unshifted > threshold_db)
    if not is_peak.any():
        return []

    freq_axis = np.fft.fftfreq(n_freq, 1.0 / sample_rate_hz)
    bin_width_hz = sample_rate_hz / n_freq

    out: list = []
    win_idxs, freq_idxs = np.where(is_peak)
    for t, f in zip(win_idxs.tolist(), freq_idxs.tolist()):
        sample_idx = t * stride_samples + template_length // 2
        delta_t = _parabolic_subpixel(magnitude_unshifted[:, f], t)
        freq_hz = float(freq_axis[f])
        # 3-point parabolic refinement with circular wrap on freq.
        left  = float(magnitude_unshifted[t, (f - 1) % n_freq])
        peak  = float(magnitude_unshifted[t, f])
        right = float(magnitude_unshifted[t, (f + 1) % n_freq])
        delta = _parabolic_freq_offset(left, peak, right)
        freq_hz = freq_hz + delta * bin_width_hz
        out.append(Candidate(
            time_sample=int(sample_idx),
            time_s=float(sample_idx / sample_rate_hz),
            freq_hz=freq_hz,
            snr_db=float(snr_db_unshifted[t, f]),
            raw_magnitude=float(magnitude_unshifted[t, f]),
            sample_epoch_offset_samples=delta_t * stride_samples,
        ))
    return out


# ── TFMF candidate JSONL logger ──────────────────────────────────────────────
# One line per accepted candidate, appended to
# ``production_output_dir() / 'tfmf_candidates.jsonl'``.  Thread-safe (the
# worker runs in a daemon thread).  Line-buffered so a crash mid-period
# doesn't lose the most-recent lines.  At ~8 FA / 15-s period on AWGN plus
# real activity, an overnight run produces ~50–100 k lines (~10 MB).
_TFMF_LOG_LOCK = threading.Lock()
_TFMF_LOG_FH = None
_TFMF_LOG_PATH = None


def _tfmf_log_candidates(cands, period_idx: int, side: str,
                         period_secs: float = 15.0) -> None:
    """Append candidates to the JSONL log.  Caller passes only the newly-added
    accumulator entries (not the full accumulator) to avoid duplicate writes.
    """
    if not cands:
        return
    global _TFMF_LOG_FH, _TFMF_LOG_PATH
    period_start_wall = period_idx * period_secs
    with _TFMF_LOG_LOCK:
        try:
            if _TFMF_LOG_FH is None:
                _TFMF_LOG_PATH = production_output_dir() / 'tfmf_candidates.jsonl'
                _TFMF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                _TFMF_LOG_FH = open(_TFMF_LOG_PATH, 'a', buffering=1)
                logger.info("[tfmf] candidate log: %s", _TFMF_LOG_PATH)
            for c in cands:
                _abs_wall = period_start_wall + float(c.time_s)
                _ts = datetime.fromtimestamp(_abs_wall, tz=timezone.utc).strftime(
                    "%Y-%m-%d_%H:%M:%S.%f")[:-5]
                _TFMF_LOG_FH.write(_json.dumps({
                    'timestamp':      _ts,
                    'period_idx':     int(period_idx),
                    'pol':            side,
                    't_s_in_period':  float(c.time_s),
                    'freq_hz':        float(c.freq_hz),
                    'snr_db':         float(c.snr_db),
                    'raw_magnitude':  float(c.raw_magnitude),
                }) + '\n')
        except Exception as _e:
            logger.warning("TFMF log write failed: %s", _e)


def _tfmf_period_worker(engine, wb_iq_h: np.ndarray,
                         wb_iq_v: 'np.ndarray | None',
                         offset_in_period_s: float,
                         period_idx: int) -> None:
    """Background-thread worker: compute TFMF surface for H (and V if present)
    and publish results onto the engine.  Wideband CPU compute is ~7 s (CPU)
    or ~0.7 s (GPU); running either on the main process_iq_data thread starves
    the VITA receiver queue.

    Reads only the IQ snapshot passed in (already copied out of the ring by
    the caller), so it is safe to run concurrently with ring writes.  Result
    publication is per-attribute assignment, atomic under the GIL.

    ``offset_in_period_s`` is the wall-clock time (in seconds) inside the
    current 15-s period at which the FIRST sample of ``wb_iq_h`` was
    captured.  Non-zero only on cold-start mid-period entry.

    ``period_idx`` identifies the wall-clock period this surface belongs to.
    Used to detect period rollover so the candidate accumulator is reset at
    the boundary, and to "freeze" already-published circles: only candidates
    in the newly-covered time region (past ``_tfmf_period_max_t_*``) are
    added on each dispatch.  This keeps the displayed circles stable across
    the 1 Hz redisplays — the older circles do NOT shift when the surface
    is re-rendered with a different (longer-window) per-bin noise floor.
    """

    def _publish(surf, cands, side: str) -> None:
        # Period rollover — drop accumulator from the previous period.
        idx_attr = f'_tfmf_period_idx_{side}'
        max_t_attr = f'_tfmf_period_max_t_{side}'
        acc_attr = f'_tfmf_period_candidates_{side}'
        if getattr(engine, idx_attr) != period_idx:
            setattr(engine, acc_attr, [])
            setattr(engine, max_t_attr, 0.0)
            setattr(engine, idx_attr, period_idx)

        # Only NEW circles (time after previously-covered region) are added.
        # Within that, also dedup against existing accumulator entries —
        # peak-pick deduplicates within one surface, but candidates near
        # the dispatch boundary can land on either side of prev_max_t and
        # produce visually close-together circles.  Drop a new candidate
        # if any existing one is within the peak-pick footprint.
        prev_max_t = getattr(engine, max_t_attr)
        accumulator = getattr(engine, acc_attr)
        _newly_added = []
        for c in cands:
            if c.time_s <= prev_max_t:
                continue
            _too_close = False
            for _e in accumulator:
                if (abs(c.time_s - _e.time_s) < _TFMF_DEDUP_TIME_S
                        and abs(c.freq_hz - _e.freq_hz) < _TFMF_DEDUP_FREQ_HZ):
                    _too_close = True
                    break
            if not _too_close:
                accumulator.append(c)
                _newly_added.append(c)

        # Log newly-added candidates to the JSONL bake-in log (one line per
        # candidate; thread-safe via _tfmf_log_candidates' internal lock).
        if _newly_added:
            _tfmf_log_candidates(_newly_added, period_idx, side)

        # Advance covered-time pointer to the end of the current surface.
        n_win = surf.shape[0]
        surface_end_t = (offset_in_period_s
                         + n_win * _TFMF_DISP_STRIDE / _TFMF_DISP_FS_HZ)
        if surface_end_t > prev_max_t:
            setattr(engine, max_t_attr, surface_end_t)

        # Publish.  _tfmf_candidates_* is the live accumulator for display.
        setattr(engine, f'_tfmf_surface_{side}', surf)
        setattr(engine, f'_tfmf_candidates_{side}', accumulator)
        setattr(engine, f'_tfmf_surface_offset_s_{side}', offset_in_period_s)
        setattr(engine, f'_tfmf_surface_dirty_{side}', True)

    try:
        _surf_h, _cands_h = _build_tfmf_display_surface(wb_iq_h)
        if _surf_h is not None:
            if offset_in_period_s:
                for _c in _cands_h:
                    _c.time_s += offset_in_period_s
            _publish(_surf_h, _cands_h, 'h')
    except Exception as _e:
        logger.warning("TFMF worker (H): %s", _e)
    if wb_iq_v is not None:
        try:
            _surf_v, _cands_v = _build_tfmf_display_surface(wb_iq_v)
            if _surf_v is not None:
                if offset_in_period_s:
                    for _c in _cands_v:
                        _c.time_s += offset_in_period_s
                _publish(_surf_v, _cands_v, 'v')
        except Exception as _e:
            logger.warning("TFMF worker (V): %s", _e)
    engine._tfmf_worker_busy = False


def reset_detection_baseline(engine):
    """Clear the 25th-percentile history and sustained-signal counters.

    Call whenever an operator-controlled input shifts channel power by
    more than the baseline can track quickly (gain change, antenna
    swap, large K step).  Without this the old pct25 trails 30+ s
    behind the new level, every channel reports "above threshold",
    and the 2-s sustained-signal gate falsely locks all 48 channels
    out for 30 s.

    Clears per-channel hop counters, rolling history, cached
    percentile, and any in-flight cooldown lockouts.
    """
    for suffix in ('', '_v'):
        if hasattr(engine, f'_metric_hist_buf{suffix}'):
            getattr(engine, f'_metric_hist_buf{suffix}').fill(0.0)
        if hasattr(engine, f'_metric_hist_cnt{suffix}'):
            setattr(engine, f'_metric_hist_cnt{suffix}', 0)
        if hasattr(engine, f'_metric_hist_idx{suffix}'):
            setattr(engine, f'_metric_hist_idx{suffix}', 0)
        # Delete the cached percentile so the detection path re-computes
        # from the fresh history on the very next chunk.  Setting to None
        # instead would leave the attribute present but unusable: the
        # detection path only recomputes when ``not hasattr(self, '_pct25*')``
        # or every 10 chunks, and the intervening calls would divide by None.
        for attr in (f'_pct25{suffix}', f'_pct25_ctr{suffix}'):
            if hasattr(engine, attr):
                delattr(engine, attr)
    if hasattr(engine, '_ch_above_thresh'):
        engine._ch_above_thresh = {}
    if hasattr(engine, '_detect_cooldowns'):
        engine._detect_cooldowns = {}
# Use unshifted fftfreq so _LO_MASK/_HI_MASK index directly into the raw FFT
# output — no np.fft.fftshift (which calls np.roll) needed in the hot loop.
_SQ_FREQ       = np.fft.fftfreq(CH_DETECT_SIZE, 1.0 / CH_SAMPLE_RATE)
_LO_MASK       = (_SQ_FREQ >= -_SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <= -_SQ_TONE_HZ + _SQ_NTOL_HZ)
_HI_MASK       = (_SQ_FREQ >=  _SQ_TONE_HZ - _SQ_NTOL_HZ) & \
                  (_SQ_FREQ <=  _SQ_TONE_HZ + _SQ_NTOL_HZ)


def _compute_sync_metric_db(sync_window,
                            hist_buf, hist_idx_attr, hist_cnt_attr,
                            pct25_attr, pct25_ctr_attr, engine,
                            skip_history_update: bool = False,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel coherent sync-correlation peak in dB above 25th-percentile.

    ``sync_window`` is shape (N_CHANNELS, NSPM) — the most recent NSPM samples
    at the head of _ch_buf.  The channelizer already places signal energy at
    DC in each channel's output (see channelizer.py docstring), which is the
    convention ``_sync_correlate`` expects, so no pre-mix is needed.  The peak
    magnitude per channel is normalised against a rolling 25th-percentile
    baseline, producing a dB metric directly comparable to the squared-FFT
    pair_metric.

    Uses the batched ``_sync_correlate_batch`` (one BLAS matmul over all
    channels) instead of a Python ``for ch in range(48)`` loop — the loop
    version was the largest single hot spot in py-spy profiling (~25 % of
    total CPU on a 6-core i5 during noise bursts).

    Returns
    -------
    metric_db : float32 (N_CHANNELS,)
        dB above the rolling pct25 baseline.
    ish_best_arr : int32 (N_CHANNELS,)
        Per-channel peak shift index from the sync correlator.  Forwarded
        unchanged from ``_sync_correlate_batch``; consumed at launch time
        by ``_sync_phase_features`` (#29 phase-derived features).
    """
    sync_window_c64 = np.ascontiguousarray(sync_window, dtype=np.complex64)
    _, peaks, ish_best_arr = _msk144_sync_correlate_batch(sync_window_c64)

    # Rolling 25th-percentile baseline of sync magnitudes (one row per hop).
    # When the upstream chunk had blanker activity (TODO #41c), skip the
    # history append so blanker zeros don't pollute the baseline — same
    # rationale as the squared-FFT path.  pct25 recompute still runs on
    # the existing history so thresholding stays current.
    idx = getattr(engine, hist_idx_attr)
    cnt = getattr(engine, hist_cnt_attr)
    if not skip_history_update:
        hist_buf[idx] = peaks
        setattr(engine, hist_idx_attr, (idx + 1) % _METRIC_HIST_DEPTH)
        if cnt < _METRIC_HIST_DEPTH:
            setattr(engine, hist_cnt_attr, cnt + 1)
            cnt += 1
    hist = hist_buf[:cnt] if cnt < _METRIC_HIST_DEPTH else hist_buf

    pct25_ctr = getattr(engine, pct25_ctr_attr, 0) + 1
    setattr(engine, pct25_ctr_attr, pct25_ctr)
    pct25 = getattr(engine, pct25_attr, None)
    # Guard against an empty history when #41c's blanker-gate
    # suppressed the very first hop's append.  ``cnt == 0`` means
    # hist_s would be a zero-length array and ``tmp[k]`` would raise
    # IndexError.  Defer the recompute and fall back to ones until
    # a clean hop populates real history.
    if (pct25_ctr % 10 == 0 or pct25 is None) and cnt > 0:
        hist_s = hist[::_METRIC_HIST_STRIDE]
        k      = max(0, int(len(hist_s) * 0.25))
        tmp    = hist_s.copy()
        tmp.partition(k, axis=0)
        pct25  = np.maximum(tmp[k], 1e-30)
        setattr(engine, pct25_attr, pct25)
    if pct25 is None:
        pct25 = np.ones(peaks.shape, dtype=np.float32)
        setattr(engine, pct25_attr, pct25)

    return (
        (10.0 * np.log10(np.maximum(peaks / pct25, 1e-30))).astype(np.float32),
        ish_best_arr,
    )


def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
    """Process one IQ chunk: wideband FFT blanker, ring buffer, channelised
    detection, and overlap-FFT waterfall.

    Parameters
    ----------
    iq_samples : array_like
        Complex IQ at ``self.sample_rate`` (typically 48 kHz).
    timestamp_int : int
        Integer part of the first-sample time.  For live radio this is normally
        Unix seconds when the source clock is valid; very small values fall back
        to wall-clock in the waterfall path (see implementation).
    timestamp_frac : int
        Sub-second fraction in **picoseconds** so that
        ``t = timestamp_int + timestamp_frac * 1e-12``.  WAV replay passes a
        file-relative timeline (seconds since start) via ``timestamp_int``/``frac``.

    Processing order
    ----------------
    0. **Wideband FFT noise blanker** — NB_FFT_SIZE blocks; Hann-tapered soft
       blanking of broadband impulses; updates ``_nb_env`` / ``_nb_floor`` for display.
    1. **Ring buffer** — append cleaned IQ to ``_iq_ring``; advance ``_iq_abs_sample``.
    2. **Time-domain magnitude** — rolling buffer for the scope trace.
    3. **Channeliser** — 48 × 12 kHz channels; optional TX settle gate discards
       partial ``_ch_buf`` accumulation.
    4. **Per-channel detection** — hop loop: squared FFT, 25th-percentile baseline,
       coincidence / sustained gates, optional ``extract_and_decode`` threads.
       SNR write index anchored to VITA timestamp (live) or WAV cursor (replay).
    5. **Waterfall** — overlap Hanning FFT on ``_sbuf``; ``_sbuf_t0`` from packet
       time.  WAV mode updates ``_wav_time_cursor`` from ``_sbuf_t0`` so the next
       chunk stays aligned with the overlap buffer.

    Both the SNR heatmap and the spectrogram use the packet timestamp so their
    time axes stay aligned.  For live radio the timestamp comes from the VITA-49
    sample-clock field; for WAV replay it uses the WAV cursor accumulated in
    _sbuf_t0.  time.time() is only a fallback for pre-GPS-lock hardware.
    """
    # Detect dual-channel input: (N, 2) → H and V; (N,) or (N, 1) → H only.
    _raw_in = np.asarray(iq_samples, dtype=np.complex64)
    dual_pol = getattr(self, 'dual_pol', False) and _raw_in.ndim == 2 and _raw_in.shape[1] >= 2
    raw   = _raw_in[:, 0] if _raw_in.ndim == 2 else _raw_in.ravel()
    raw_v = _raw_in[:, 1].copy() if dual_pol else None

    # Pre-compute packet time from embedded timestamp so it is available for
    # the detection section (which runs before the waterfall section where
    # _pkt_time is normally computed).
    #
    # **Sample-clock anchoring** (see project_sample_clock_anchoring memory):
    # Scrolling displays (waterfall, detection heatmaps) must index by a
    # sample-counter-derived time, never by wall-clock-at-processing time.
    # When the radio provides a valid VITA-49 sample-clock timestamp
    # (timestamp_int >= 1e9), we use it directly — perfectly monotonic.
    # When it doesn't (Flex TSI=0, RTL-SDR, Airspy without GPS lock), the
    # legacy fallback ``time.time()`` injects per-chunk wall-clock jitter
    # that lands on the heatmap row index → randomly-placed dark vertical
    # lines.  Instead, anchor once at the first chunk and derive each
    # subsequent chunk's time from the sample counter.  Result is a
    # monotonic per-sample timeline regardless of OS scheduling jitter.
    _pkt_time_early = float(timestamp_int) + float(timestamp_frac) * 1e-12
    _is_wav_early   = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav_early and timestamp_int < 1_000_000_000:
        _t0 = getattr(self, '_iq_t0_wall', None)
        if _t0 is not None:
            # Subsequent chunk: derive from sample counter.  _iq_abs_sample
            # is the count *before* this chunk has been added, so this is
            # the wall-clock for the first sample of this chunk.
            _pkt_time_early = _t0 + self._iq_abs_sample / self.sample_rate
        else:
            # Very first chunk after start/reset: anchor to wall clock once.
            _pkt_time_early = time.time()


    # ── 0. Noise blanker (Linrad / Bypass / NR0V-Wideband) ────────────────────
    # Backend is selected via ``self.blanker`` (set in engine.__init__, runtime-
    # switchable).  Each backend operates on the engine's ``_nb_*`` state so
    # display and reset paths continue to work without change.  Dual-pol
    # backends return independent H and V blanking masks; single-pipeline
    # backends return only blank_mask (H) and leave blank_mask_v None, in
    # which case the H mask is applied to V for backward compatibility.
    _nb_result   = self.blanker.process(self, raw, raw_v, dual_pol)
    cleaned      = _nb_result.cleaned_h
    cleaned_v    = _nb_result.cleaned_v
    blank_mask   = _nb_result.blank_mask
    mag          = _nb_result.mag_h
    blank_mask_v = _nb_result.blank_mask_v if _nb_result.blank_mask_v is not None else blank_mask
    mag_v_disp   = (_nb_result.mag_v if _nb_result.mag_v is not None else
                    (np.abs(raw_v).astype(np.float32) if dual_pol and raw_v is not None else None))

    # TODO #41c: skip pct25 baseline update for hops whose underlying
    # input chunk had any blanked samples.  Without this gate, the
    # blanker zeroes (or Hann-tapers) noise-burst samples, which the
    # squared-FFT measures as LOW power; that pollutes the rolling
    # 25-percentile baseline downward, and once the burst ends the
    # baseline is too low — normal noise looks elevated, producing
    # the 100× post-burst false-trigger spike documented in
    # ``project_post_burst_rebound`` memory.  The fix freezes the
    # baseline during blanking; pct25 stays at its pre-burst value,
    # so post-burst normal noise reads as normal.  The detection
    # threshold still fires on real signals (raw_lin / pct25 is
    # still computed every hop), it just doesn't pollute the
    # rolling reference any more.
    #
    # Per-chunk granularity (one chunk → 1-3 detection hops, all
    # derived from the same input samples) is the natural unit; if
    # any sample in the chunk was blanked, every hop derived from
    # that chunk is tainted.
    _chunk_had_blank_h = bool(blank_mask.any())
    _chunk_had_blank_v = (
        bool(_nb_result.blank_mask_v.any())
        if (dual_pol and _nb_result.blank_mask_v is not None)
        else _chunk_had_blank_h   # single-pol: H mask applies to V
    )
    self._skip_pct25_update_h = _chunk_had_blank_h
    self._skip_pct25_update_v = _chunk_had_blank_v


    # ── 1. Write cleaned IQ into the circular ring buffer ────────────────────
    # Ring is (ring_n, 2): col 0 = H, col 1 = V.  Single-pol writes zeros to col 1.
    chunk_len = len(cleaned)
    ring_size = self._iq_ring.shape[0]
    pos = self._iq_ring_pos
    _v_col = cleaned_v if dual_pol else np.zeros(chunk_len, dtype=np.complex64)
    _ring_chunk = np.stack([cleaned, _v_col], axis=1)   # (N, 2)
    if pos + chunk_len <= ring_size:
        self._iq_ring[pos:pos + chunk_len] = _ring_chunk
    else:
        first_n = ring_size - pos
        self._iq_ring[pos:]                 = _ring_chunk[:first_n]
        self._iq_ring[:chunk_len - first_n] = _ring_chunk[first_n:]
    self._iq_ring_pos    = (pos + chunk_len) % ring_size
    # Anchor wall-clock for IQ ring sample 0 — see project_sample_clock_anchoring.
    # Set ONCE per session (or after a reset that nulls it).  Re-anchoring per
    # chunk would re-introduce wall-clock jitter into _iq_t0_wall and undo the
    # whole point of sample-clock anchoring.
    #
    # The drift threshold for re-anchoring needs to be high enough that normal
    # source-side timestamp jitter never trips it.  Flex's per-packet timestamp
    # (computed in the source thread as ``_t_start + _abs_samps/_hw_rate``) can
    # oscillate ±1–2 s under CPU contention because ``_t_start`` is anchored
    # once at acquisition start with a `time.time()` reading that itself has
    # OS-scheduling jitter and NTP-correction events.  A 1-s threshold caused
    # constant re-anchoring during noise bursts; each re-anchor jumped
    # ``_iq_t0_wall`` enough to spuriously cross a 15-s boundary in the
    # heatmap-display logic and triggered a wipe.
    #
    # 30 s is well above any normal jitter (Flex packets stay within ±2 s) but
    # well below the "stream is broken" regime — a real source-restart or DAX
    # stream reset would manifest as tens of seconds of discontinuity.  The
    # PERIOD_SLIP fix can absorb a few seconds of drift without misassigning
    # decodes (15 s period, plenty of slack).
    _projected = (self._iq_t0_wall + self._iq_abs_sample / self.sample_rate
                  if self._iq_t0_wall is not None else None)
    if self._iq_t0_wall is None:
        self._iq_t0_wall = _pkt_time_early - self._iq_abs_sample / self.sample_rate
    elif abs(_pkt_time_early - _projected) > 30.0:
        # Major discontinuity — re-anchor and log so the cause is visible.
        self._iq_t0_wall = _pkt_time_early - self._iq_abs_sample / self.sample_rate
        logger.warning("Re-anchored _iq_t0_wall: drift %.2f s from projected",
                       _pkt_time_early - _projected)
    self._iq_abs_sample += chunk_len

    # ── 1a. Update time-domain magnitude display buffers (post- and pre-blanker) ──
    def _write_td_buf(buf, pos_attr, data):
        """Write data into the circular buffer named buf; return new position."""
        n   = len(buf)
        d_n = len(data)
        pos = getattr(self, pos_attr)
        if pos + d_n <= n:
            buf[pos:pos + d_n] = data
        else:
            first = n - pos
            buf[pos:]         = data[:first]
            buf[:d_n - first] = data[first:]
        setattr(self, pos_attr, (pos + d_n) % n)

    _td_mag     = np.abs(cleaned).astype(np.float32)   # post-blanker H
    _write_td_buf(self._td_mag_buf,     '_td_mag_pos',     _td_mag)
    _write_td_buf(self._td_mag_buf_raw, '_td_mag_pos_raw', mag)     # pre-blanker H

    # ── 1b. V-channel time-domain magnitude display buffers ──────────────────
    # Per-channel V detection state (_nb_spec_avg_v, _nb_last_P_v, …) is
    # now owned by the blanker backend (see LinradBlanker._process_channel
    # with suffix='_v').  Only the display-only post/pre-blanker magnitude
    # circular buffers are updated here, using the V mag provided in the
    # BlankerResult so NR0V's delay-aligned pre-blanker trace stays aligned.
    if dual_pol and cleaned_v is not None:
        _td_mag_v = np.abs(cleaned_v).astype(np.float32)            # post-blanker V
        _write_td_buf(self._td_mag_buf_v,     '_td_mag_pos_v',     _td_mag_v)
        if mag_v_disp is not None:
            _write_td_buf(self._td_mag_buf_raw_v, '_td_mag_pos_raw_v', mag_v_disp)

    # ── 2. Channelise into 48 × 12 kHz channels (H, and V if dual-pol) ─────────
    # Dual-pol: pass both H and V in one call so the NCO phase matrix (the most
    # expensive part) is computed once instead of twice, and the FIR filter calls
    # are batched over 2×N_CHANNELS rows instead of two separate passes.
    if dual_pol:
        ch_out, ch_out_v = apply_channelizer(
            cleaned, self._ch_state,
            lp_taps=self._ch_taps,
            iq_block_v=cleaned_v, state_v=self._ch_state_v,
        )
    else:
        ch_out   = apply_channelizer(cleaned, self._ch_state, lp_taps=self._ch_taps)
        ch_out_v = None

    # ── 3. Per-channel detection ──────────────────────────────────────────────
    # Post-TX settling gate — discard channelizer output and skip detection for
    # ~500 ms after TX ends so AGC transients don't cause false broadband triggers.
    _settle = getattr(self, '_tx_settle_remaining', 0)
    # Save leftover sample count before adding new data — used below to anchor
    # the WAV-mode SNR write index to the time of _ch_buf[0], not the end of
    # the previous call.  Without this, the write index jumps by one extra hop
    # at the start and end of replay, leaving dark vertical gaps in the heatmap.
    _ch_buf_carry = self._ch_buf_end
    if _settle > 0:
        self._tx_settle_remaining = max(0, _settle - len(iq_samples))
        self._ch_buf_end   = 0   # discard any partial accumulation
        self._ch_buf_v_end = 0
        _ch_buf_carry = 0
        # Fall through to waterfall FFT — NB still learns the noise floor.
    else:
        # Accumulate into pre-allocated slide-left buffer (no per-call allocation)
        new_n = ch_out.shape[1]
        cap = self._ch_buf.shape[1]
        if self._ch_buf_end + new_n > cap:
            # Buffer overflow: discard oldest samples to make room
            keep = cap - new_n
            if keep > 0:
                self._ch_buf[:, :keep] = self._ch_buf[:, cap - keep:cap]
            self._ch_buf_end = max(keep, 0)
        self._ch_buf[:, self._ch_buf_end:self._ch_buf_end + new_n] = ch_out
        self._ch_buf_end += new_n
        if dual_pol:
            if self._ch_buf_v_end + new_n > cap:
                keep_v = cap - new_n
                if keep_v > 0:
                    self._ch_buf_v[:, :keep_v] = self._ch_buf_v[:, cap - keep_v:cap]
                self._ch_buf_v_end = max(keep_v, 0)
            self._ch_buf_v[:, self._ch_buf_v_end:self._ch_buf_v_end + new_n] = ch_out_v
            self._ch_buf_v_end += new_n

        # Sync correlator's dedicated rolling ring — FIFO shift-and-append so
        # the most recent NSPM samples per channel are always available
        # regardless of _ch_buf's drain state.  See engine.py for the rationale.
        if ENABLE_SYNC_DETECT:
            _sb_n = self._sync_buf.shape[1]   # = NSPM
            if new_n >= _sb_n:
                self._sync_buf[:] = ch_out[:, -_sb_n:]
            else:
                self._sync_buf[:, :_sb_n - new_n] = self._sync_buf[:, new_n:]
                self._sync_buf[:, _sb_n - new_n:] = ch_out
            if dual_pol:
                if new_n >= _sb_n:
                    self._sync_buf_v[:] = ch_out_v[:, -_sb_n:]
                else:
                    self._sync_buf_v[:, :_sb_n - new_n] = self._sync_buf_v[:, new_n:]
                    self._sync_buf_v[:, _sb_n - new_n:] = ch_out_v

    # Anchor wall-clock time of _ch_buf[0] once per call.  This is no longer
    # used for the per-hop write index (the hop counter handles that), but it
    # is still used inside the loop as ``current_wall_time`` for the 15-s
    # boundary marker-lifecycle check.
    if self.source_mode == "wav":
        # _wav_time_cursor is the end of the previous call's buffer.  Subtract
        # the carry-over samples already in _ch_buf so the anchor points to
        # _ch_buf[0].
        _loop_wall = float(getattr(self, "_wav_time_cursor", 0.0)) \
                     - _ch_buf_carry / CH_SAMPLE_RATE
    else:
        # Live radio: use the VITA-49 sample-clock timestamp so the heatmap
        # is anchored to the same GPS clock as the spectrogram.  Subtract the
        # carry-over channelised samples so the anchor is at _ch_buf[0].
        _loop_wall = _pkt_time_early - _ch_buf_carry / CH_SAMPLE_RATE

    # ── Period-aligned TFMF wideband surface dispatch ──────────────────────
    # Re-displays at ~_TFMF_REDISPLAY_INTERVAL_S, with the surface anchored to
    # the START of the current 15-s wall-clock period.  This matches the
    # other panels (fast-graph, sq_det heatmap) which all reset at the period
    # boundary; the TFMF X axis is "seconds since the period started", so a
    # feature at t = 5 s in the surface corresponds to 5 s into the period —
    # the same X-coordinate it would have on those other panels.
    #
    # As the period progresses, each dispatch reads a growing window of IQ
    # from period_start to now.  At period boundary the window resets to a
    # short slice, which redraws as the new period fills.
    #
    # ``offset_in_period_s`` handles the cold-start case (engine entered
    # mid-period): the surface's first sample is the OLDEST available IQ,
    # which may be later than period_start — the offset shifts the image
    # rect right so it lands at the correct period-relative time.
    if (self._iq_t0_wall is not None
            and not self._tfmf_worker_busy
            and (_loop_wall - self._tfmf_last_dispatch_time
                 >= _TFMF_REDISPLAY_INTERVAL_S)):
        _period_idx = int(_loop_wall / self.history_secs)
        _period_start_wall = _period_idx * float(self.history_secs)
        _period_start_abs = int(
            (_period_start_wall - self._iq_t0_wall) * self.sample_rate
        )
        _ring_size = self._iq_ring.shape[0]
        _oldest_abs = self._iq_abs_sample - _ring_size
        _actual_start_abs = max(_period_start_abs, _oldest_abs)
        _n_samples = self._iq_abs_sample - _actual_start_abs
        # Skip until we have at least one matched-filter template length of
        # IQ inside the current period — surface compute returns None for
        # shorter inputs.
        if _n_samples >= _TFMF_DISP_TMPL_N:
            _offset_s = (_actual_start_abs - _period_start_abs) / self.sample_rate
            _wb_iq = _read_ring(self._iq_ring, self._iq_ring_pos,
                                self._iq_abs_sample,
                                _actual_start_abs, _n_samples)
            if _wb_iq.shape[0] >= _TFMF_DISP_TMPL_N:
                _wb_h = np.ascontiguousarray(_wb_iq[:, 0])
                _wb_v_raw = _wb_iq[:, 1]
                _wb_v = (np.ascontiguousarray(_wb_v_raw)
                         if np.any(_wb_v_raw) else None)
                self._tfmf_worker_busy = True
                self._tfmf_last_dispatch_time = _loop_wall
                threading.Thread(
                    target=_tfmf_period_worker,
                    args=(self, _wb_h, _wb_v, _offset_s, _period_idx),
                    daemon=True, name='tfmf-period',
                ).start()

    while self._ch_buf_end >= CH_DETECT_SIZE and not getattr(self, '_tx_settle_remaining', 0):
        ch_block   = self._ch_buf[:, :CH_DETECT_SIZE]            # (48, 512) H channel
        ch_block_v = (self._ch_buf_v[:, :CH_DETECT_SIZE]
                      if dual_pol and self._ch_buf_v_end >= CH_DETECT_SIZE
                      else None)

        sq         = ch_block ** 2                               # complex squaring H
        X_sq       = np.fft.fft(sq * _DETECT_WINDOW[np.newaxis, :], axis=1)
        if ch_block_v is not None:
            sq_v   = ch_block_v ** 2
            X_sq_v = np.fft.fft(sq_v * _DETECT_WINDOW[np.newaxis, :], axis=1)

        current_wall_time = _loop_wall
        t_in_window = current_wall_time % self.history_secs

        # ── Two-window tone detection in the squared domain ───────────────────
        # MSK144 tones at fc_offset ± 500 Hz → after squaring → ±1000 Hz.
        # _SQ_FREQ, _LO_MASK, _HI_MASK are pre-computed module-level constants.
        power_lin = np.abs(X_sq) / CH_DETECT_SIZE               # (48, 512) linear
        lo_peak   = power_lin[:, _LO_MASK].max(axis=1)          # (48,) linear
        hi_peak   = power_lin[:, _HI_MASK].max(axis=1)          # (48,) linear
        raw_lin   = (lo_peak + hi_peak) / 2.0                   # (48,) linear

        # ── Shadow: WSJT-X-style MAX combine + parallel rolling history ──────
        # Computed every hop (free — lo_peak/hi_peak already available).
        # Maintained so launch-time shadow can compute detmet_norm (the WSJT-X
        # primary metric).  Per-channel buffer, same shape and update rule as
        # the legacy _metric_hist_buf.  Disable: env MAP144_SHADOW_SQDET=0.
        new_raw_lin = np.maximum(lo_peak, hi_peak)              # (48,) WSJT-X max-combine
        if _SHADOW_SQDET_ENABLED:
            if not hasattr(self, '_metric_hist_buf_new'):
                self._metric_hist_buf_new = np.zeros(
                    (_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32
                )
                self._metric_hist_idx_new = 0
                self._metric_hist_cnt_new = 0
            if not getattr(self, '_skip_pct25_update_h', False):
                self._metric_hist_buf_new[self._metric_hist_idx_new] = new_raw_lin
                self._metric_hist_idx_new = (self._metric_hist_idx_new + 1) % _METRIC_HIST_DEPTH
                if self._metric_hist_cnt_new < _METRIC_HIST_DEPTH:
                    self._metric_hist_cnt_new += 1

        # ── H-channel rolling 25th-percentile baseline ───────────────────────
        # Skip the history append when this chunk had blanker activity
        # (TODO #41c — see _skip_pct25_update_h set after the blanker
        # call above).  The pct25 recompute still runs on the existing
        # history so detection thresholding stays current; only the
        # *update* is gated so blanker zeros don't pollute the baseline.
        if not getattr(self, '_skip_pct25_update_h', False):
            self._metric_hist_buf[self._metric_hist_idx] = raw_lin
            self._metric_hist_idx = (self._metric_hist_idx + 1) % _METRIC_HIST_DEPTH
            if self._metric_hist_cnt < _METRIC_HIST_DEPTH:
                self._metric_hist_cnt += 1
        hist  = (self._metric_hist_buf[:self._metric_hist_cnt]
                 if self._metric_hist_cnt < _METRIC_HIST_DEPTH
                 else self._metric_hist_buf)
        self._pct25_ctr = getattr(self, '_pct25_ctr', 0) + 1
        # Guard against an empty history — possible when #41c's
        # blanker-gate suppresses the FIRST chunk's append (a noise
        # impulse on chunk 0 happens to trip the blanker before any
        # clean hop has populated the history).  Without this guard
        # ``tmp[k]`` indexes into a zero-length array and raises
        # IndexError.  When hist is empty, defer the recompute — the
        # ``pct25 = self._pct25`` below uses whatever default was
        # set in engine.__init__ (np.full(n_channels, 1.0)) until a
        # clean hop populates real history.
        if (self._pct25_ctr % 10 == 0 or not hasattr(self, '_pct25')) and self._metric_hist_cnt > 0:
            hist_s = hist[::_METRIC_HIST_STRIDE]
            k      = max(0, int(len(hist_s) * 0.25))
            tmp = hist_s.copy()
            tmp.partition(k, axis=0)          # in-place, bypasses _wrapfunc
            self._pct25 = np.maximum(tmp[k], 1e-30)
        # Defensive: if neither the recompute fired nor was _pct25
        # ever set (rare — only on engine restart where the first
        # chunk had blanking), fall back to ones so detection
        # thresholds compute without raising AttributeError.
        if not hasattr(self, '_pct25'):
            self._pct25 = np.ones(raw_lin.shape, dtype=np.float32)
        pct25 = self._pct25

        pair_metric_h = (
            10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30))
        ).astype(np.float32)   # (48,) dB above baseline

        # ── V-channel rolling 25th-percentile baseline (independent noise floor) ─
        if ch_block_v is not None:
            plin_v    = np.abs(X_sq_v) / CH_DETECT_SIZE
            raw_lin_v = (plin_v[:, _LO_MASK].max(axis=1) +
                         plin_v[:, _HI_MASK].max(axis=1)) / 2.0
            # Same blanker-gate as the H path above (TODO #41c).
            if not getattr(self, '_skip_pct25_update_v', False):
                self._metric_hist_buf_v[self._metric_hist_idx_v] = raw_lin_v
                self._metric_hist_idx_v = (self._metric_hist_idx_v + 1) % _METRIC_HIST_DEPTH
                if self._metric_hist_cnt_v < _METRIC_HIST_DEPTH:
                    self._metric_hist_cnt_v += 1
            hist_v = (self._metric_hist_buf_v[:self._metric_hist_cnt_v]
                      if self._metric_hist_cnt_v < _METRIC_HIST_DEPTH
                      else self._metric_hist_buf_v)
            self._pct25_ctr_v = getattr(self, '_pct25_ctr_v', 0) + 1
            # Same empty-history guard as the H path (TODO #41c).
            if (self._pct25_ctr_v % 10 == 0 or not hasattr(self, '_pct25_v')) and self._metric_hist_cnt_v > 0:
                hist_vs = hist_v[::_METRIC_HIST_STRIDE]
                kv      = max(0, int(len(hist_vs) * 0.25))
                tmp_v = hist_vs.copy()
                tmp_v.partition(kv, axis=0)       # in-place, bypasses _wrapfunc
                self._pct25_v = np.maximum(tmp_v[kv], 1e-30)
            if not hasattr(self, '_pct25_v'):
                self._pct25_v = np.ones(raw_lin_v.shape, dtype=np.float32)
            pair_metric_v = (
                10.0 * np.log10(np.maximum(raw_lin_v / self._pct25_v, 1e-30))
            ).astype(np.float32)
        else:
            pair_metric_v = pair_metric_h   # single-pol: V mirrors H (unused)

        # ── Coherent sync-word detector (additive Step 1) ────────────────────
        # Per-channel correlation of the leading NSPM-sample window of _ch_buf
        # against the MSK144 16-chip sync template.  Coherent integration over
        # 8 ms of known waveform buys ~6 dB over the squared-domain power
        # detector at low SNR.  Output magnitude is normalised by a per-channel
        # 25th-percentile baseline (mirrors the squared-FFT path) so the dB
        # metric is comparable, then pair_metric takes the max of the two paths.
        # The window is read fresh per hop, so the metric tracks the current
        # ch_block (not stalely fixed at chunk boundaries).
        # Save the squared-only metrics so we can attribute each surviving
        # trigger to "sq", "sync", or "both" downstream — useful for verifying
        # the sync detector is firing at all and for diagnostic console output.
        sq_metric_h_only = pair_metric_h.copy()
        sq_metric_v_only = pair_metric_v.copy()
        sync_metric_h = None
        sync_metric_v = None
        sync_ish_best_h = None
        sync_ish_best_v = None

        if ENABLE_SYNC_DETECT:
            # _sync_buf is FIFO-updated per chunk and always holds the most
            # recent NSPM samples, so the correlator can run on every hop
            # regardless of _ch_buf's drain state.  No cache fallback needed
            # (which previously produced stale-row banding under live Flex
            # operation where _ch_buf_end never reaches NSPM).
            sync_metric_h, sync_ish_best_h = _compute_sync_metric_db(
                self._sync_buf,
                self._sync_metric_hist_buf, '_sync_metric_hist_idx',
                '_sync_metric_hist_cnt', '_sync_pct25', '_sync_pct25_ctr',
                self,
                skip_history_update=getattr(self, '_skip_pct25_update_h', False),
            )
            pair_metric_h = np.maximum(pair_metric_h, sync_metric_h)

            if ch_block_v is not None:
                sync_metric_v, sync_ish_best_v = _compute_sync_metric_db(
                    self._sync_buf_v,
                    self._sync_metric_hist_buf_v, '_sync_metric_hist_idx_v',
                    '_sync_metric_hist_cnt_v', '_sync_pct25_v', '_sync_pct25_ctr_v',
                    self,
                    skip_history_update=getattr(self, '_skip_pct25_update_v', False),
                )
                pair_metric_v = np.maximum(pair_metric_v, sync_metric_v)
            else:
                sync_ish_best_v = None

        # Trigger on either channel; display the max for the combined heatmap
        pair_metric = np.maximum(pair_metric_h, pair_metric_v)

        # ── Display-only 15-s boundary handling ──────────────────────────────
        # Blank the heatmaps and reset the hop counter at each 15-s wall-clock
        # boundary cross (and at session start, when _ch_snr_boundary's initial
        # value differs from the current period number).  Internal DSP state
        # (rolling baselines, AGC, percentile estimators) is NOT reset — those
        # run continuously and the 15-s boundary doesn't apply to them.
        # The hop counter (rather than a sample-clock-to-row map) is what fills
        # rows: each hop fires exactly when the buffer drained 256 samples, so
        # consecutive hops always fill consecutive rows.  This eliminates the
        # rare edge case where a sample-clock map could place the first hop
        # after a boundary at row 1 (or higher) and leave row 0 dark for an
        # entire period — visible as a persistent vertical dark line.
        ch_boundary = int(current_wall_time / self.history_secs)
        if ch_boundary != self._ch_snr_boundary:
            self._ch_snr_history_h[:] = -999.0
            self._ch_snr_history_v[:] = -999.0
            self._sync_snr_history_h[:] = -999.0
            self._sync_snr_history_v[:] = -999.0
            self._ch_snr_hop_count = 0
            self._ch_snr_boundary  = ch_boundary
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                marker_list.clear()
            # Do NOT reset _ch_above_thresh — the coincidence gate must survive
            # the display window boundary so a signal that starts just before
            # the boundary still triggers a jt9 launch after it.

            # (TFMF dispatch moved out of the period-boundary block — see the
            # periodic-trigger block above the per-hop loop, which fires every
            # ``_TFMF_REDISPLAY_INTERVAL_S`` on the most-recent 15 s of IQ.)
            pass

        # Hop-counter row index — guaranteed contiguous across hops.
        self._ch_snr_write_idx = self._ch_snr_hop_count % N_SNR_HIST
        # ── Diagnostic toggle: raw power vs pct25-normalised metric ────────
        # Default heatmap shows pair_metric / sync_metric (dB above the
        # rolling 25th-pct baseline), which absorbs steady spurs and
        # asymmetric noise-floor rolloffs.  When ``_show_raw_power`` is set
        # (UI checkbox in Diagnostics menu), substitute the raw power in dB
        # so persistent / asymmetric features become visible.  Detection
        # logic continues to use pair_metric — only the display changes.
        #
        # Display scaling: see _RAW_POWER_DISPLAY_OFFSET_DB_SQ /
        # _RAW_POWER_DISPLAY_OFFSET_DB_SYNC at module top.  Squared-FFT
        # raw power is ~−90..−50 dBFS so the SQ offset shifts it +85
        # dB into the slider window.  Sync correlator output is already
        # at +9..+30 dB so SYNC offset is 0.  Channel-to-channel
        # differences are preserved exactly in both cases.
        if getattr(self, '_show_raw_power', False):
            raw_db_h = (
                10.0 * np.log10(np.maximum(raw_lin, 1e-30))
                + _RAW_POWER_DISPLAY_OFFSET_DB_SQ
            ).astype(np.float32)
            if ch_block_v is not None:
                raw_db_v = (
                    10.0 * np.log10(np.maximum(raw_lin_v, 1e-30))
                    + _RAW_POWER_DISPLAY_OFFSET_DB_SQ
                ).astype(np.float32)
            else:
                raw_db_v = raw_db_h
            self._ch_snr_history_h[self._ch_snr_write_idx, :] = raw_db_h
            self._ch_snr_history_v[self._ch_snr_write_idx, :] = raw_db_v
        else:
            self._ch_snr_history_h[self._ch_snr_write_idx, :] = pair_metric_h
            self._ch_snr_history_v[self._ch_snr_write_idx, :] = pair_metric_v
        # Sync history — independent panel.  Same toggle logic, separate
        # pct25 (``_sync_pct25`` / ``_sync_pct25_v``).  We recover raw sync
        # magnitude in dB by adding 10·log10(sync_pct25) back to
        # sync_metric (which is already 10·log10(peaks/sync_pct25)).
        if sync_metric_h is not None:
            if getattr(self, '_show_raw_power', False):
                sp25_h = getattr(self, '_sync_pct25', None)
                if sp25_h is not None:
                    raw_sync_db_h = (
                        sync_metric_h
                        + 10.0 * np.log10(np.maximum(sp25_h, 1e-30))
                        + _RAW_POWER_DISPLAY_OFFSET_DB_SYNC
                    ).astype(np.float32)
                    self._sync_snr_history_h[self._ch_snr_write_idx, :] = raw_sync_db_h
                else:
                    self._sync_snr_history_h[self._ch_snr_write_idx, :] = sync_metric_h
            else:
                self._sync_snr_history_h[self._ch_snr_write_idx, :] = sync_metric_h
        if sync_metric_v is not None:
            if getattr(self, '_show_raw_power', False):
                sp25_v = getattr(self, '_sync_pct25_v', None)
                if sp25_v is not None:
                    raw_sync_db_v = (
                        sync_metric_v
                        + 10.0 * np.log10(np.maximum(sp25_v, 1e-30))
                        + _RAW_POWER_DISPLAY_OFFSET_DB_SYNC
                    ).astype(np.float32)
                    self._sync_snr_history_v[self._ch_snr_write_idx, :] = raw_sync_db_v
                else:
                    self._sync_snr_history_v[self._ch_snr_write_idx, :] = sync_metric_v
            else:
                self._sync_snr_history_v[self._ch_snr_write_idx, :] = sync_metric_v
        # ── DIAGNOSTIC: heatmap state at selected hops (DEBUG-level) ──────────
        # Enable with logging.getLogger("map144_app.processing").setLevel(DEBUG)
        # — useful when investigating detection-plot rendering vs. metric scale
        # vs. baseline staleness.  Six samples per WAV play; negligible cost.
        if (logger.isEnabledFor(logging.DEBUG)
                and self._ch_snr_hop_count in (0, 5, 50, 200, 500, 700)):
            _arr = self._ch_snr_history_h
            _sa  = self._sync_snr_history_h if sync_metric_h is not None else None
            logger.debug(
                "heatmap-diag hop=%d pair_h max=%.2f dB argmax_ch=%d "
                "row_just_written max=%.2f full_arr min=%.2f max=%.2f "
                "nonzero_rows=%d sync max=%.2f boundary=%d",
                self._ch_snr_hop_count,
                float(pair_metric_h.max()), int(np.argmax(pair_metric_h)),
                float(_arr[self._ch_snr_write_idx].max()),
                float(_arr.min()), float(_arr.max()),
                int(np.any(_arr != 0.0, axis=1).sum()),
                float(_sa.max()) if _sa is not None else float('nan'),
                int(self._ch_snr_boundary),
            )
        self._ch_snr_hop_count += 1

        # Block-primary cutover (Option 2): metrics + heatmap history
        # writes above have already happened — that's what feeds the
        # GUI sq_det / sync_det / detection-heatmap displays.  Skip
        # the rest of the hop body (cooldown ticks, threshold gate,
        # cluster gate, decode dispatch, launches.jsonl write) — the
        # Block runtime's DetectorBlock + DecoderBlock now own those.
        # Inline the per-hop _ch_buf drain that normally runs at the
        # bottom of this loop body (lines ~1318-1325) so the loop
        # advances; then ``continue`` to the next hop.
        if getattr(self, 'use_blocks', False):
            remaining = max(0, self._ch_buf_end - _CH_DETECT_HOP)
            if remaining > 0:
                self._ch_buf[:, :remaining] = self._ch_buf[:, _CH_DETECT_HOP:self._ch_buf_end]
            self._ch_buf_end = remaining
            if dual_pol:
                remaining_v = max(0, self._ch_buf_v_end - _CH_DETECT_HOP)
                if remaining_v > 0:
                    self._ch_buf_v[:, :remaining_v] = self._ch_buf_v[:, _CH_DETECT_HOP:self._ch_buf_v_end]
                self._ch_buf_v_end = remaining_v
            continue

        # Tick down cooldowns
        self._detect_cooldowns = {
            k: v - 1 for k, v in self._detect_cooldowns.items() if v > 1
        }

        cooldown_hops = max(
            1, int(DETECT_MERGE_GAP_S * CH_SAMPLE_RATE / _CH_DETECT_HOP)
        )

        _triggered = np.where(pair_metric > DETECT_THRESH_DB)[0]
        # Process strongest channel first so the proximity guard blocks weaker
        # adjacent channels rather than the correct center channel.
        if len(_triggered) > 1:
            _triggered = _triggered[pair_metric[_triggered].argsort()[::-1]]

        # ── Cluster the triggered channels ────────────────────────────────────
        # Split triggered channels into contiguous clusters (gap > 3 ch = new cluster).
        # Broadband noise is one wide contiguous cluster; two real signals at different
        # frequencies form two separate clusters with a gap between them.
        # For each cluster: suppress if too wide (> _COINCIDENCE_SPAN_CH) or if the
        # global cluster count hits _COINCIDENCE_MAX_CLUSTERS.
        # Only the strongest channel in each surviving cluster launches jt9 — this
        # prevents adjacent sidelobe channels from spawning redundant decode threads
        # while still allowing independent signals in separate clusters to both launch.
        _sorted_ch = _triggered.astype(int)   # already sorted by metric desc above
        _sorted_idx = _triggered.astype(int); _sorted_idx.sort()  # .sort() bypasses _wrapfunc
        _clusters = []   # list of arrays of channel indices (ascending)
        if len(_sorted_idx):
            _cur = [_sorted_idx[0]]
            for _ci in _sorted_idx[1:]:
                if _ci - _cur[-1] <= 3:
                    _cur.append(_ci)
                else:
                    _clusters.append(np.array(_cur))
                    _cur = [_ci]
            _clusters.append(np.array(_cur))

        # Global cluster count gate: if a large number of SEPARATE narrow clusters
        # fire simultaneously it is broadband noise, not multiple legitimate pings.
        # Each real MSK144 ping spreads into ~3 adjacent channels (squared-domain
        # ±1000 Hz tones straddle the 1 kHz channel boundary), forming one cluster.
        # Three simultaneous pings at 5 kHz spacing produce 3 clusters of ~3 channels
        # each (9 total channels) — the old raw channel-count gate (≥8) incorrectly
        # fired on this scenario.  Counting clusters (≥6) is far more robust.
        # Only fresh clusters (whose best channel is not already in cooldown) are
        # counted — sidelobe remnants of a prior real detection must not trigger the gate.
        _fresh_clusters = [
            _cl for _cl in _clusters
            if int(_cl[np.argmax(pair_metric[_cl])]) not in self._detect_cooldowns
        ]
        _too_many = len(_fresh_clusters) >= _COINCIDENCE_MAX_CLUSTERS
        if _too_many:
            for _ck in _triggered:
                self._detect_cooldowns[_ck] = cooldown_hops
            self._ch_buf[:, :len(self._ch_buf[0])] = 0
            self._ch_buf_end = 0
            self._ch_buf_v_end = 0   # keep H and V in sync
            break

        # Per-cluster spread gate and best-channel selection.
        # _launch_channels: list of (launch_ch, cluster_channels) tuples.
        _launch_channels = []
        for _cl in _clusters:
            _cl_span = int(_cl[-1]) - int(_cl[0])
            if _cl_span > _COINCIDENCE_SPAN_CH:
                # Wide cluster = broadband noise; suppress all its channels.
                for _ck in _cl:
                    self._detect_cooldowns[_ck] = cooldown_hops
                continue
            # Pick the channel with the highest pair_metric in this cluster.
            _best = int(_cl[np.argmax(pair_metric[_cl])])
            if _best not in self._detect_cooldowns:
                _launch_channels.append((_best, _cl))
                # Launch happened: suppress only the launched channel itself.
                # Suppressing neighbours risks blocking a real signal whose center
                # is one channel away (e.g. ch9 signal blocked by ch8 sidelobe launch).
                # Hop-to-hop re-triggers from the same real channel are suppressed
                # because _best stays in cooldown; the "best-cooled" branch then
                # refreshes its timer without adding fresh neighbours.
                self._detect_cooldowns[_best] = cooldown_hops
            else:
                # Best channel is already in cooldown (decoding in progress or just
                # launched).  Only refresh channels that are already suppressed —
                # do NOT add fresh channels to cooldown.  A fresh channel in this
                # cluster may be the real signal center that will become the cluster
                # best on a subsequent hop once the sidelobe-elevated cooled channels
                # drop below threshold.
                for _ck in _cl:
                    if _ck in self._detect_cooldowns:
                        self._detect_cooldowns[_ck] = cooldown_hops  # refresh timer

        # Sustained-signal gate — update consecutive-hop counters.
        # Channels above threshold this hop increment; all others reset to 0.
        # A channel that stays above for >_SUSTAINED_HOPS (~2 s) is interference
        # (FT8, beacon, spur) not a meteor ping — apply a long lockout.
        _ch_above = getattr(self, '_ch_above_thresh', {})
        for _ck in range(N_CHANNELS):
            if _ck in set(_triggered.tolist()):
                _ch_above[_ck] = _ch_above.get(_ck, 0) + 1
                if _ch_above[_ck] == _SUSTAINED_HOPS:
                    logger.debug("[detect] ch %d sustained signal — locking out for 30 s", _ck)
                    self._detect_cooldowns[_ck] = _SUSTAINED_LOCKOUT
            else:
                _ch_above.pop(_ck, None)
        self._ch_above_thresh = _ch_above

        # Compute the actual IF Nyquist channel for the Nyquist-edge mask.
        # The naive "_nyquist_ch = N_CHANNELS // 2 = 24" is correct ONLY when
        # the channel grid is centered on IQ DC (channel_offset_hz == 0).
        # The USB-convention offset (channel_offset_hz = 1500 Hz when
        # pan == calling) shifts the channel grid by 1.5 channels relative
        # to IQ DC.  ch_k 24's NCO is 24*1000 + 1500 = 24500 Hz, which wraps
        # to IF −23.5 kHz — NOT at the actual IF Nyquist (±24 kHz).
        # The actual IF Nyquist sits at ch_k = (sample_rate/2 −
        # channel_offset_hz) / CHANNEL_SPACING_HZ = (24000 − 1500) / 1000 = 22.5.
        #
        # Centering the mask on ch_k 24 instead of 22.5 leaves channel
        # ch_k = 19 (IF +20.5 kHz) — which sits deep in the IF passband
        # rolloff — UNmasked, while the equivalent negative-IF channel
        # ch_k = 28 (IF −18.5 kHz, OUTSIDE the rolloff zone) IS masked.
        # That asymmetric coverage is the root cause of the ~6× excess
        # trigger rate at ch_signed +19 (the "STATIONARY_SPUR" / "rolloff
        # bins" puzzle from project_launch_pattern_findings).
        _ch_offset_in_channels = (
            self._ch_state.channel_offset_hz / CHANNEL_SPACING_HZ
        )
        _nyquist_ch_real = (N_CHANNELS / 2) - _ch_offset_in_channels  # 22.5 typical

        for ch_k, _cl_all in _launch_channels:
            # Mask channels within _EDGE_CH_SKIP of the actual IF Nyquist.
            # Uses the wrap-aware distance: ch_k is unsigned 0..47, but
            # frequency-space distance to Nyquist needs to consider both
            # +N_CHANNELS and -N_CHANNELS aliases.
            _ch_int = int(ch_k)
            _dist_to_nyquist = min(
                abs(_ch_int - _nyquist_ch_real),
                abs(_ch_int - _nyquist_ch_real - N_CHANNELS),
                abs(_ch_int - _nyquist_ch_real + N_CHANNELS),
            )
            if _dist_to_nyquist <= _EDGE_CH_SKIP:
                continue

            # Detector attribution: which path put this channel above threshold?
            # Examine BOTH the squared-FFT and sync metrics (max of H/V if dual-pol)
            # against DETECT_THRESH_DB.  Result is one of "sq", "sync", "sq+sync".
            _sq_above = (max(sq_metric_h_only[ch_k],
                             sq_metric_v_only[ch_k] if ch_block_v is not None
                             else -np.inf) > DETECT_THRESH_DB)
            _sync_above = False
            if sync_metric_h is not None:
                _sync_h = sync_metric_h[ch_k]
                _sync_v = (sync_metric_v[ch_k]
                           if (sync_metric_v is not None and ch_block_v is not None)
                           else -np.inf)
                _sync_above = max(_sync_h, _sync_v) > DETECT_THRESH_DB
            if _sq_above and _sync_above:
                det_source = 'sq+sync'
            elif _sync_above:
                det_source = 'sync'
            else:
                det_source = 'sq'

            # Estimate fc from peak bin locations in each window (linear domain)
            lo_freq   = float(_SQ_FREQ[_LO_MASK][np.argmax(power_lin[ch_k][_LO_MASK])])
            hi_freq   = float(_SQ_FREQ[_HI_MASK][np.argmax(power_lin[ch_k][_HI_MASK])])
            fc_offset  = (lo_freq + hi_freq) / 4.0   # squaring doubled freqs
            # ch_k 25-47 are negative-frequency channels (-23 to -1 kHz)
            ch_signed  = int(ch_k) if int(ch_k) <= N_CHANNELS // 2 else int(ch_k) - N_CHANNELS
            # Carrier position in Hz from IQ DC (pan centre).
            # channel_offset_hz = (calling_freq - pan_center) + 1500 from the channelizer state.
            fc_hz      = ch_signed * CHANNEL_SPACING_HZ + self._ch_state.channel_offset_hz + fc_offset

            abs_snap      = self._iq_abs_sample
            ring_state_fn = lambda: (self._iq_ring_pos, self._iq_abs_sample)
            # Resolved via the env-var-aware helper so tests can redirect
            # writes to a TempDir and not contaminate the operator's real
            # decode log.  See detection.py:production_output_dir().
            output_dir    = str(production_output_dir())
            detect_ts     = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H:%M:%S.%f')[:21]

            decode_queue = getattr(self, '_decode_queue', None)
            marker_id    = getattr(self, '_jt9_marker_next_id', 0)
            if hasattr(self, '_jt9_marker_next_id'):
                self._jt9_marker_next_id += 1
            ring_gen     = getattr(self, '_iq_ring_gen', 0)
            ring_gen_fn  = lambda: getattr(self, '_iq_ring_gen', 0)

            # Prune finished threads and enforce a hard cap before spawning.
            # Without this, false detections accumulate unbounded threads that
            # saturate all CPU cores with concurrent numpy/scipy work.
            jt9_list = getattr(self, '_jt9_threads', None)
            if jt9_list is not None:
                # Remove dead threads in-place (O(n) but list stays small)
                jt9_list[:] = [t for t in jt9_list if t.is_alive()]
                if len(jt9_list) >= 12:   # 3× semaphore width; beyond this, drop
                    self._detect_cooldowns[ch_k] = cooldown_hops
                    continue
                # Note: proximity guard removed — the ±1 neighbour cooldown on launch
                # handles hop-to-hop re-triggers of the same channel without blocking
                # independent signals in adjacent channels.

            # Burst-context features for offline analysis + ML classifier.
            # Track recent launches in a rolling 5-second window so each launch
            # carries "how busy was the band recently" as numerical context.
            #
            # n_chans_300ms — distinct channels with launches in the last 300 ms
            #   (excluding this one).  ≥ 8 ⇒ this launch is *inside* a wideband
            #   burst (analyze_launches WIDEBAND_BURST rule).
            # n_chans_2s — distinct channels in the last 2 s.  Captures the
            #   post-burst tail and noisy-episode envelope without explicit
            #   transition tracking.  High n_chans_2s with low n_chans_300ms
            #   = post-burst rebound zone (where ~100 % of launches are noise
            #   per the 2026-05-08 analysis: 0/657 decoded in the 5 s post-
            #   burst window).
            _now_epoch = datetime.now(timezone.utc).timestamp()
            _recent = getattr(self, '_recent_launch_log', None)
            if _recent is None:
                _recent = []
                self._recent_launch_log = _recent
            # Drop entries older than 5 s.  In-place since this is single-thread.
            _cutoff_5s = _now_epoch - 5.0
            while _recent and _recent[0][0] < _cutoff_5s:
                _recent.pop(0)
            _cutoff_300ms = _now_epoch - 0.300
            _cutoff_2s    = _now_epoch - 2.000
            _n_chans_300ms = len({c for t, c in _recent if t > _cutoff_300ms})
            _n_chans_2s    = len({c for t, c in _recent if t > _cutoff_2s})

            # Per-launch DSP-state snapshot for launches.jsonl.  Lets offline
            # analysis distinguish "real strong signal that didn't decode"
            # from "noise spike at the band edge crossing a low pct25
            # threshold" without needing the saved IQ.  All values in dB.
            _pct25_arr   = getattr(self, '_pct25', None)
            _pct25_v_arr = getattr(self, '_pct25_v', None)
            launch_metrics = {
                'pair_metric_db_h':  round(float(pair_metric_h[ch_k]), 1),
                'pair_metric_db_v':  (round(float(pair_metric_v[ch_k]), 1)
                                      if ch_block_v is not None else None),
                'sq_metric_db_h':    round(float(sq_metric_h_only[ch_k]), 1),
                'sq_metric_db_v':    (round(float(sq_metric_v_only[ch_k]), 1)
                                      if ch_block_v is not None else None),
                'sync_metric_db_h':  (round(float(sync_metric_h[ch_k]), 1)
                                      if sync_metric_h is not None else None),
                'sync_metric_db_v':  (round(float(sync_metric_v[ch_k]), 1)
                                      if sync_metric_v is not None else None),
                'ch_pct25_db_h':     (round(float(10.0 * np.log10(_pct25_arr[ch_k])), 1)
                                      if _pct25_arr is not None else None),
                'ch_pct25_db_v':     (round(float(10.0 * np.log10(_pct25_v_arr[ch_k])), 1)
                                      if _pct25_v_arr is not None else None),
                'ch_signed':         int(ch_signed),
                'n_cluster_chan':    int(len(_cl_all)),
                'det_source':        det_source,
                # Burst-context features (see _recent_launch_log block above):
                'n_chans_300ms':     int(_n_chans_300ms),
                'n_chans_2s':        int(_n_chans_2s),
            }
            # Append this launch to the recent log so subsequent launches in
            # this hop's loop see it in their burst-context counts.
            _recent.append((_now_epoch, int(ch_signed)))

            # ── #29: sync-correlator phase-derived features ─────────────────
            # Three physical-structure features the magnitude-only sync
            # metric cannot expose.  Computed only at launch time (rare
            # event), so they don't affect the detector hot path.  None when
            # the sync detector path is disabled or didn't run on this hop.
            if sync_ish_best_h is not None:
                _coh_h, _ts_h, _foff_h = _msk144_sync_phase_features(
                    np.ascontiguousarray(self._sync_buf[ch_k], dtype=np.complex64),
                    int(sync_ish_best_h[ch_k]),
                )
                launch_metrics['sync_phase_coherence_h'] = round(_coh_h, 3)
                launch_metrics['sync_t_sample_h']         = round(_ts_h, 2)
                launch_metrics['sync_freq_offset_hz_h']   = round(_foff_h, 2)
            else:
                launch_metrics['sync_phase_coherence_h'] = None
                launch_metrics['sync_t_sample_h']         = None
                launch_metrics['sync_freq_offset_hz_h']   = None
            if sync_ish_best_v is not None:
                _coh_v, _ts_v, _foff_v = _msk144_sync_phase_features(
                    np.ascontiguousarray(self._sync_buf_v[ch_k], dtype=np.complex64),
                    int(sync_ish_best_v[ch_k]),
                )
                launch_metrics['sync_phase_coherence_v'] = round(_coh_v, 3)
                launch_metrics['sync_t_sample_v']         = round(_ts_v, 2)
                launch_metrics['sync_freq_offset_hz_v']   = round(_foff_v, 2)
            else:
                launch_metrics['sync_phase_coherence_v'] = None
                launch_metrics['sync_t_sample_v']         = None
                launch_metrics['sync_freq_offset_hz_v']   = None

            # ── SHADOW: WSJT-X-faithful sq_det metric on the same window ─────
            # Compute the new sq_det algorithm at launch time only (per-launch
            # cost, not per-hop) so we can compare against the legacy fields
            # (`sq_metric_db_h`) without affecting triggering.  Rolls out the
            # 2026-05-15 dB-scale-mismatch and fc-from-strong-band fixes for
            # operator-side analysis.  Disable: env MAP144_SHADOW_SQDET=0.
            if _SHADOW_SQDET_ENABLED:
                try:
                    _new_h = _sq_det_new(
                        np.ascontiguousarray(self._sync_buf[ch_k], dtype=np.complex128),
                        fc=0.0,        # _sync_buf is complex baseband (channelized at 0 Hz)
                    )
                    # Strong-band fc-offset (the regression-fix from 2026-05-15
                    # — read the freq-error from whichever band fired stronger,
                    # not the noise-driven argmax of the un-signaled band).
                    _new_h_fc_off = (_new_h.ferr_hi if _new_h.ah >= _new_h.al
                                     else _new_h.ferr_lo)
                    launch_metrics['sq_new_detmet_h']      = round(float(_new_h.detmet),      4)
                    launch_metrics['sq_new_detmet2_lin_h'] = round(float(_new_h.detmet2),     3)
                    launch_metrics['sq_new_detmet2_db_h']  = round(float(_sq_to_db(_new_h.detmet2)), 2)
                    launch_metrics['sq_new_ah_h']          = round(float(_new_h.ah),          4)
                    launch_metrics['sq_new_al_h']          = round(float(_new_h.al),          4)
                    launch_metrics['sq_new_fc_offset_hz_h']= round(float(_new_h_fc_off),      2)

                    # ── Three noise references on the same hop ──────────────
                    # detmet_norm (WSJT-X primary) = current MAX-combine raw_lin
                    # divided by pct25 of recent values on this channel.  Log
                    # three alternative noise references so offline analysis
                    # can pick the best one for Step 6 threshold calibration:
                    #   (1) rolling per-channel pct25 of NEW (max-combine) values
                    #   (2) cross-channel pct25 at THIS hop (48 channels)
                    #   (3) wideband median-floor estimator (decoupled from
                    #       per-channel state entirely; see noise_blanker.py)
                    # Plus the LEGACY per-channel rolling pct25 (avg-combine)
                    # for direct comparison against the formula port.
                    _new_raw_now = float(new_raw_lin[ch_k])
                    launch_metrics['sq_new_raw_lin_h'] = round(_new_raw_now, 4)

                    # (1) Rolling per-channel pct25 of new max-combine values
                    if self._metric_hist_cnt_new > 0:
                        _hist_new = self._metric_hist_buf_new[:self._metric_hist_cnt_new, ch_k]
                        _hist_new_s = _hist_new[::_METRIC_HIST_STRIDE]
                        _pct25_chan_rolling = float(np.percentile(_hist_new_s, 25)) if len(_hist_new_s) else 0.0
                    else:
                        _pct25_chan_rolling = 0.0
                    launch_metrics['sq_new_pct25_chan_rolling_h'] = round(_pct25_chan_rolling, 6)

                    # (2) Cross-channel pct25 of NEW values at this hop
                    _pct25_cross_chan = float(np.percentile(new_raw_lin, 25))
                    launch_metrics['sq_new_pct25_cross_chan_h'] = round(_pct25_cross_chan, 6)

                    # (3) Wideband floor from the SELLIM-style estimator (in
                    # noise_blanker.py).  None if estimator hasn't initialised
                    # yet (early in run before lowlevel_frac threshold met).
                    _wbf = getattr(self, '_nb_wideband_floor', None)
                    launch_metrics['sq_new_wideband_floor_h'] = (
                        round(float(_wbf), 6) if _wbf is not None else None
                    )

                    # Legacy per-channel rolling pct25 (AVG-combine) for direct
                    # apples-to-apples comparison against (1).
                    if hasattr(self, '_pct25'):
                        launch_metrics['sq_new_pct25_chan_legacy_h'] = round(
                            float(self._pct25[ch_k]), 6
                        )

                    # detmet_norm (WSJT-X primary metric) using reference (1)
                    _denom = max(_pct25_chan_rolling, 1e-30)
                    _detmet_norm = _new_raw_now / _denom
                    launch_metrics['sq_new_detmet_norm_h']    = round(_detmet_norm, 3)
                    launch_metrics['sq_new_detmet_norm_db_h'] = round(
                        float(_sq_to_db(_detmet_norm)), 2
                    )
                except Exception as _exc:    # noqa: BLE001
                    launch_metrics['sq_new_error_h'] = str(_exc)[:80]

                if dual_pol and getattr(self, '_sync_buf_v', None) is not None:
                    try:
                        _new_v = _sq_det_new(
                            np.ascontiguousarray(self._sync_buf_v[ch_k], dtype=np.complex128),
                            fc=0.0,
                        )
                        _new_v_fc_off = (_new_v.ferr_hi if _new_v.ah >= _new_v.al
                                         else _new_v.ferr_lo)
                        launch_metrics['sq_new_detmet_v']      = round(float(_new_v.detmet),      4)
                        launch_metrics['sq_new_detmet2_lin_v'] = round(float(_new_v.detmet2),     3)
                        launch_metrics['sq_new_detmet2_db_v']  = round(float(_sq_to_db(_new_v.detmet2)), 2)
                        launch_metrics['sq_new_ah_v']          = round(float(_new_v.ah),          4)
                        launch_metrics['sq_new_al_v']          = round(float(_new_v.al),          4)
                        launch_metrics['sq_new_fc_offset_hz_v']= round(float(_new_v_fc_off),      2)
                    except Exception as _exc:    # noqa: BLE001
                        launch_metrics['sq_new_error_v'] = str(_exc)[:80]

            if not getattr(self, '_analysis_no_decode', False):
                t = threading.Thread(
                    target=extract_and_decode,
                    args=(self._iq_ring, ring_state_fn,
                          abs_snap, self.sample_rate, fc_hz, output_dir,
                          t_in_window, decode_queue, marker_id,
                          ring_gen, ring_gen_fn,
                          getattr(self, 'display_center_freq_mhz', self.center_freq_mhz),
                          detect_ts, None, dual_pol),
                    kwargs={'det_source': det_source,
                            'launch_metrics': launch_metrics},
                    daemon=True,
                )
                t._fc_hz = fc_hz
                t.start()
                if jt9_list is not None:
                    jt9_list.append(t)
            # Record marker for heatmap overlay in radio-tuning coordinates (fc_hz − 1500 Hz)
            marker_list = getattr(self, '_jt9_markers', None)
            if marker_list is not None:
                display_khz = (fc_hz - 1500.0) / 1000.0
                # det_pol: which channel had the higher detection metric at launch time.
                # Used for immediate circle routing before theta_deg is available.
                _det_pol = ('v'
                            if dual_pol and pair_metric_v[ch_k] > pair_metric_h[ch_k]
                            else 'h')
                marker_list.append({
                    'id':        marker_id,
                    't':         t_in_window,
                    'freq_khz':  display_khz,
                    'boundary':  self._ch_snr_boundary,
                    'decoded':   False,
                    'outcome':   'launched',
                    'message':   None,
                    'theta_deg': None,   # filled by extract_and_decode on dual-pol decode
                    'det_pol':   _det_pol,
                })
            self._detect_cooldowns[ch_k] = cooldown_hops

        # Slide channeliser buffers left by one hop (in-place, no allocation)
        remaining = max(0, self._ch_buf_end - _CH_DETECT_HOP)
        if remaining > 0:
            self._ch_buf[:, :remaining] = self._ch_buf[:, _CH_DETECT_HOP:self._ch_buf_end]
        self._ch_buf_end = remaining
        if dual_pol:
            remaining_v = max(0, self._ch_buf_v_end - _CH_DETECT_HOP)
            if remaining_v > 0:
                self._ch_buf_v[:, :remaining_v] = self._ch_buf_v[:, _CH_DETECT_HOP:self._ch_buf_v_end]
            self._ch_buf_v_end = remaining_v

    # ── 4. Overlap-FFT for waterfall display (blanked signal) ─────────────────
    # Anchor _sbuf_t0 to the VITA packet timestamp so each FFT block is placed
    # at its exact sample-accurate receive time, not at time.time() during
    # processing.  Using time.time() causes dark vertical stripes: when the
    # runtime processes a batch of packets quickly the clock barely advances,
    # mapping many blocks to the same row; the clock then jumps, skipping rows.
    #
    _sbuf_end_before = self._sbuf_end   # samples already buffered before this packet

    new_n = len(cleaned)
    self._sbuf[self._sbuf_end:self._sbuf_end + new_n] = cleaned
    self._sbuf_end += new_n
    if dual_pol and cleaned_v is not None:
        self._sbuf_v[self._sbuf_v_end:self._sbuf_v_end + new_n] = cleaned_v
        self._sbuf_v_end += new_n

    # All sources must deliver timestamp_frac as picoseconds (adapter responsibility).
    _pkt_time = float(timestamp_int) + float(timestamp_frac) * 1e-12
    # WAV files use a 0-based cursor (seconds from file start) — their timestamps are
    # intentionally small and must NOT be replaced by wall-clock time.
    # For live radio sources, a small timestamp_int (< year 2001 Unix epoch) means the
    # hardware clock is not yet locked to UTC (Flex TSI=0, RTL-SDR, Airspy without GPS).
    # Fall back to the sample-clock-anchored time (_iq_t0_wall + abs_sample/sr), NOT
    # to time.time() — the latter introduces wall-clock jitter that makes _sbuf_t0
    # jump between chunks during CPU bursts, which makes row_idx jump, which fires
    # the spec_staging gap-fill and produces visible 200 ms bands of repeated
    # spectrum content.  See project_sample_clock_anchoring memory.
    _is_wav = getattr(self, 'source_mode', '') == 'wav'
    if not _is_wav and timestamp_int < 1_000_000_000:
        _t0 = getattr(self, '_iq_t0_wall', None)
        if _t0 is not None:
            # _iq_abs_sample at this point already includes this chunk's samples
            # (incremented earlier in process_iq_data), so _sbuf_end_before pointing
            # at the same absolute sample count yields a consistent _sbuf_t0.
            _pkt_time = _t0 + (self._iq_abs_sample - new_n) / self.sample_rate
        else:
            _pkt_time = time.time()
    # Time of _sbuf[0] = packet's first-sample time minus already-buffered samples
    self._sbuf_t0 = _pkt_time - _sbuf_end_before / self.sample_rate

    fft_window  = self._fft_window   # pre-computed float32 Hanning window
    window_gain = self._window_gain  # pre-computed scalar
    full_scale  = 1.0
    _fg_win     = getattr(self, '_fast_graph_win', None)
    _fg_visible = _fg_win is None or _fg_win.isVisible()
    _wf_hop     = self.fft_size // 2   # constant; moved out of the loop

    while self._sbuf_end >= self.fft_size:
        try:
            if _fg_visible:
                block = self._sbuf[:self.fft_size]

                X         = np.fft.fftshift(np.fft.fft(block * fft_window))
                magnitude = np.abs(X) / (self.fft_size * window_gain)
                power_db  = 20.0 * np.log10(magnitude / full_scale + 1e-12)

                # Sample-accurate time for this block: _sbuf_t0 is the time of _sbuf[0],
                # which is always the first sample of the current block since the buffer
                # slides by exactly one hop after each block is processed below.
                current_wall_time = self._sbuf_t0
                time_in_window = current_wall_time % self.history_secs

                # ── Accumulated spectrogram ───────────────────────────────────────
                # Each block is written at the row corresponding to its receive time so
                # the column position matches wall-clock time modulo the 15-second window,
                # matching the WSJTX Fast Graph convention.
                spec_boundary = int(current_wall_time / self.history_secs)
                if spec_boundary != self.spec_boundary:
                    self.spectrogram_data    = self.spec_staging.copy()
                    self.spec_staging_filled = True
                    self.accumulated_noise_floor = np.percentile(self.spec_staging, 10, axis=0)
                    self.spec_staging        = np.full((self.max_history, self.fft_size), -130.0)
                    self.spec_boundary       = spec_boundary
                    self.next_boundary = current_wall_time + (
                        self.history_secs - (current_wall_time % self.history_secs)
                    )
                    self._prev_staging_idx = None   # reset gap-fill cursor at window boundary
                    if dual_pol:
                        self.spectrogram_data_v = self.spec_staging_v.copy()
                        self.spec_staging_v     = np.full((self.max_history, self.fft_size), -130.0)

                # Compute the row index for this block once — accumulated and realtime
                # spectrograms share the same time_in_window so the index is identical.
                # Use round() not int(): FP reconstruction of the VITA timestamp is
                # systematically a few tens of picoseconds low, giving values like
                # 46.99999999 that int() would floor to 46 instead of 47.
                row_idx = min(max(round(time_in_window * self.blocks_per_sec), 0), self.max_history - 1)

                # ── Gap fill: repeat last spectrum into any skipped rows ──────────
                # Skipped rows arise from batched packet delivery — the VITA timestamps
                # correctly place each block, so gaps are real.  Filling prevents dark
                # vertical stripes.  Cursors are reset at each 15 s boundary above /
                # below so gap-fill never bridges across windows.
                _prev_staging = self._prev_staging_idx
                if _prev_staging is not None and row_idx > _prev_staging + 1:
                    _gap = row_idx - _prev_staging - 1
                    # Diagnostic: gaps > 2 rows (~43 ms) are worth investigating;
                    # with sample-clock-anchored _sbuf_t0 these should not happen
                    # in steady state.  Log size + recent timing context.
                    if _gap > 2:
                        logger.info(
                            "spec_staging gap-fill: %d rows (~%.0f ms) at row_idx=%d "
                            "_sbuf_t0=%.6f _sbuf_end_before=%d",
                            _gap, _gap * 1000.0 / self.blocks_per_sec,
                            row_idx, self._sbuf_t0, _sbuf_end_before,
                        )
                    self.spec_staging[_prev_staging + 1:row_idx] = power_db
                self._prev_staging_idx = row_idx
                self.spec_staging[row_idx] = power_db

                # ── Real-time spectrogram ─────────────────────────────────────────
                realtime_boundary = int(current_wall_time / self.history_secs)
                if realtime_boundary != self._realtime_boundary:
                    # Allocate new arrays before replacing the references so the display
                    # thread reads either the complete old array or the complete new one,
                    # never a partially-reset one.
                    self.realtime_data      = np.full((self.max_history, self.fft_size), -130.0)
                    self._realtime_boundary = realtime_boundary
                    self._prev_rt_idx       = None   # reset gap-fill cursor
                    if dual_pol:
                        self.realtime_data_v = np.full((self.max_history, self.fft_size), -130.0)

                _prev_rt = self._prev_rt_idx
                if _prev_rt is not None and row_idx > _prev_rt + 1:
                    self.realtime_data[_prev_rt + 1:row_idx] = power_db
                self._prev_rt_idx = row_idx
                self.realtime_data[row_idx] = power_db
                self.realtime_filled = True
                self._realtime_dirty = True

                # ── V-channel waterfall (dual-pol only) ───────────────────────────
                if dual_pol and self._sbuf_v_end >= self.fft_size:
                    block_v    = self._sbuf_v[:self.fft_size]
                    X_v        = np.fft.fftshift(np.fft.fft(block_v * fft_window))
                    power_db_v = 20.0 * np.log10(
                        np.abs(X_v) / (self.fft_size * window_gain) / full_scale + 1e-12
                    )
                    if _prev_staging is not None and row_idx > _prev_staging + 1:
                        self.spec_staging_v[_prev_staging + 1:row_idx] = power_db_v
                    self.spec_staging_v[row_idx] = power_db_v
                    if _prev_rt is not None and row_idx > _prev_rt + 1:
                        self.realtime_data_v[_prev_rt + 1:row_idx] = power_db_v
                    self.realtime_data_v[row_idx] = power_db_v

        except Exception as _wf_exc:
            import logging as _lg
            _lg.getLogger(__name__).warning("waterfall FFT error (skipping block): %s", _wf_exc)

        finally:
            # Slide always runs — even on exception — so _sbuf_end never overflows.
            remaining = self._sbuf_end - _wf_hop
            if remaining > 0:
                self._sbuf[:remaining] = self._sbuf[_wf_hop:self._sbuf_end]
            self._sbuf_end = max(0, remaining)
            if dual_pol and self._sbuf_v_end >= _wf_hop:
                remaining_v = self._sbuf_v_end - _wf_hop
                if remaining_v > 0:
                    self._sbuf_v[:remaining_v] = self._sbuf_v[_wf_hop:self._sbuf_v_end]
                self._sbuf_v_end = remaining_v
            # Advance the time anchor by exactly one hop so the next block's
            # current_wall_time is sample-accurate without calling time.time().
            self._sbuf_t0 += _wf_hop / self.sample_rate

    # Keep _wav_time_cursor in sync for the WAV source step (it uses this to
    # timestamp the next packet it will send into process_iq_data).
    if self.source_mode == "wav":
        self._wav_time_cursor = self._sbuf_t0 + self._sbuf_end / self.sample_rate

