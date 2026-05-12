"""Measure instantaneous frequency vs time within MS pings.

Purpose
-------
Test the operator's question (2026-05-12):

  "How much Doppler does the path apply?  How much does DF change
   during integration?  How coherent is the signal over time?
   If the signal flutters in amplitude, does it flutter in
   frequency as well?"

These distributions don't appear to be published in measured form
for 6 m MS.  This tool builds them from WSJT-X-saved WAV files.

For each WAV (15 s, 12 kHz mono, signal centered at 1500 Hz):

  1. Bandpass to the MSK144 audio band (1300-1700 Hz).
  2. Hilbert transform → complex analytic signal.
  3. Instantaneous frequency = (1/2π) × d(phase)/dt, smoothed
     over a configurable window (default 20 ms).
  4. Identify burst regions where |envelope| > median + threshold.
  5. Within each burst, compute:
       - duration (ms)
       - peak amplitude (dB above median)
       - mean / max-min / σ of freq (Hz)
       - amplitude-frequency correlation (Pearson r)

Outputs a plot (envelope + freq overlay) and a text report.

Usage
-----
    python tools/measure_ping_freq_vs_time.py --wav <PATH>           # single WAV
    python tools/measure_ping_freq_vs_time.py --date 20260512        # batch
    python tools/measure_ping_freq_vs_time.py --date 20260512 --csv  # batch + CSV
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# ── Configuration ─────────────────────────────────────────────────────────────
SR_AUDIO        = 12000
# MSK144 audio band — the two modulation tones live at 1000 Hz and 2000 Hz
# (±500 Hz around the suppressed 1500 Hz carrier).  The bandpass must
# preserve both tones for the squared-spectrum carrier estimator to work:
# squaring tones at 1000 Hz and 2000 Hz produces the carrier-identifying
# lines at 2·f₀ ± 1000 = 2000 Hz and 4000 Hz; if either input tone is
# attenuated by the bandpass, the squared lines disappear.
# 600-2400 Hz captures both tones with ~400 Hz margin (covers the main
# spectral lobe of MSK144) and is similar in spirit to MAP144's own
# 300-2700 Hz audio BPF in detection.py:_apply_audio_bpf.
F_LO, F_HI      = 600, 2400         # MSK144 audio band — must straddle both tones
INST_FREQ_BIN_S = 0.020             # 20 ms smoothing
ENV_BIN_S       = 0.010             # 10 ms envelope resolution
BURST_THRESH_DB = 4.0               # burst above (median + 4 dB)
MIN_BURST_S     = 0.050             # ignore < 50 ms transients


# ── Sync-correlation-based carrier tracker (matched filter, low-SNR robust) ──
# The squared-FFT carrier estimator (used for σ_f stats elsewhere in this
# file) is noise-limited: it needs the signal to dominate the squared
# spectrum, which requires roughly +4 dB envelope above noise.  For weak
# signals where coherent integration would actually help, that method gives
# noise.
#
# A sync-correlation-vs-freq scan is a matched filter against the known
# MSK144 sync sequence.  It gains the matched-filter processing gain
# (~6 dB for the 42-chip sync), extending reliable carrier estimation
# into the sub-+4-dB regime where coherent-integration design changes
# matter.  This is exactly the inner search WSJT-X's msk144_freq_search
# already does internally — we expose the full curve instead of just the
# winning frequency.

def sync_corr_freq_track(
    audio_real: np.ndarray,
    sr: int = SR_AUDIO,
    f_min_hz: float = -150.0,
    f_max_hz: float = +150.0,
    f_step_hz: float = 5.0,
    frame_n: int | None = None,        # 864 = 72 ms at 12 kHz (one MSK frame)
    step_s: float = 0.020,             # 20 ms hop between estimates
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, f_peak_hz, mag_peak) arrays — sync-correlator carrier
    estimate over time.

    For each ``step_s`` hop, takes a ``frame_n``-sample window, scans the
    candidate frequency range, applies the freq shift, correlates against
    the MSK144 sync template, and records the freq at peak magnitude.
    """
    from scipy.signal import hilbert as _hilbert
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from map144_app.msk144_spd import _sync_correlate, NSPM

    if frame_n is None:
        frame_n = NSPM    # 864 = 72 ms = one MSK144 frame
    step_n = max(1, int(round(step_s * sr)))

    # Real audio → complex analytic → mix down by 1500 Hz so signal at DC.
    # Then RMS-normalise to 1.0 and scale by √2 so the sync correlator sees
    # the amplitude convention its sync threshold was calibrated for
    # (matches the complex_baseband normalisation in msk144_spd_decode).
    analytic = _hilbert(audio_real.astype(np.float64)).astype(np.complex64)
    t_full = np.arange(len(analytic), dtype=np.float64) / sr
    baseband = (analytic * np.exp(-2j * np.pi * 1500.0 * t_full)).astype(np.complex64)
    rms = float((np.abs(baseband.astype(np.complex128)) ** 2).mean() ** 0.5)
    if rms > 1e-12:
        baseband = (baseband * (np.sqrt(2.0) / rms)).astype(np.complex64)

    freqs = np.arange(f_min_hz, f_max_hz + f_step_hz/2, f_step_hz, dtype=np.float64)
    n_steps = max(0, (len(baseband) - frame_n) // step_n + 1)

    t_arr = np.empty(n_steps, dtype=np.float32)
    f_arr = np.empty(n_steps, dtype=np.float32)
    m_arr = np.empty(n_steps, dtype=np.float32)

    t_frame = np.arange(frame_n, dtype=np.float64) / sr
    # Pre-compute all freq-shift phasors (frame_n × n_freqs is small)
    phasors = np.exp(-2j * np.pi * np.outer(t_frame, freqs)).astype(np.complex64)
    # phasors[:, k] is the shift for freqs[k]

    for i in range(n_steps):
        start = i * step_n
        frame = baseband[start:start + frame_n]
        # Apply each candidate shift and pick the one with the highest
        # sync-correlation peak magnitude.
        best_f, best_m = 0.0, -1.0
        for k in range(len(freqs)):
            shifted = (frame * phasors[:, k]).astype(np.complex64)
            _, xmax, _ = _sync_correlate(shifted)
            if xmax > best_m:
                best_m = xmax
                best_f = float(freqs[k])
        t_arr[i] = (start + frame_n / 2) / sr
        f_arr[i] = best_f
        m_arr[i] = best_m

    return t_arr, f_arr, m_arr


@dataclass
class BurstFreqStat:
    t_start: float
    t_end:   float
    duration_ms: float
    peak_db: float
    f_mean_hz:  float
    f_std_hz:   float
    f_range_hz: float    # max - min within burst
    af_corr:    float    # Pearson r(|A|, f) within burst


# ── WSJT-X ALL.TXT lookup for decoded messages per WAV ───────────────────────

def _load_wsjtx_messages(all_txt_path: Path) -> dict[tuple[int, int, int], list[str]]:
    """Return {(yyyymmdd, hhmm, ss-floored-to-15s): [message, ...]} index.

    Lets us look up which MSK144 messages WSJT-X decoded for the 15-s
    period a given WAV covers.  WAV filename ``YYMMDD_HHMMSS.wav`` →
    period_start at HHMMSS (already 15-s-aligned by WSJT-X save logic).

    Uses the same period-attribution math as the wav_corpus_test.py fix:
    period = floor(timestamp/15s) + dt, then floor to 15s again.
    """
    import re
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    if not all_txt_path.exists():
        return {}
    pat = re.compile(
        r'^(\d{6})_(\d{6})\s+[\d.]+\s+Rx\s+MSK144\s+'
        r'[-+]?\d+\s+([-+]?[\d.]+)\s+\d+\s+(.+)$'
    )
    idx: dict[tuple[int, int, int], list[str]] = {}
    with all_txt_path.open(errors='replace') as fh:
        for ln in fh:
            m = pat.match(ln.strip())
            if not m:
                continue
            yymmdd, hhmmss, dt_s, msg = m.groups()
            try:
                ts = _dt.strptime(f"20{yymmdd}_{hhmmss}", "%Y%m%d_%H%M%S").replace(tzinfo=_tz.utc)
                # floor ts to 15s, add dt, floor again — matches WAV period
                window = _dt(ts.year, ts.month, ts.day, ts.hour, ts.minute,
                             (ts.second // 15) * 15, tzinfo=_tz.utc)
                signal_t = window + _td(seconds=float(dt_s))
                period_start = _dt(
                    signal_t.year, signal_t.month, signal_t.day,
                    signal_t.hour, signal_t.minute,
                    (signal_t.second // 15) * 15, tzinfo=_tz.utc)
                key = (int(period_start.strftime('%Y%m%d')),
                       period_start.hour * 100 + period_start.minute,
                       period_start.second)
                msg_clean = ' '.join(msg.strip().split())
                if msg_clean not in idx.setdefault(key, []):
                    idx[key].append(msg_clean)
            except (ValueError, IndexError):
                continue
    return idx


def _msgs_for_wav(wav_path: Path,
                  msg_index: dict[tuple[int, int, int], list[str]]) -> list[str]:
    """Look up decoded messages for the 15-s period covered by ``wav_path``."""
    import re
    m = re.match(r'(\d{6})_(\d{6})\.wav$', wav_path.name)
    if not m:
        return []
    yymmdd, hhmmss = m.groups()
    yyyymmdd = int(f"20{yymmdd}")
    hh = int(hhmmss[:2]); mm = int(hhmmss[2:4]); ss = int(hhmmss[4:6])
    key = (yyyymmdd, hh*100 + mm, (ss // 15) * 15)
    return msg_index.get(key, [])


# ── Signal processing ────────────────────────────────────────────────────────

def _bandpass(audio: np.ndarray, sr: int = SR_AUDIO,
              f_lo: float = F_LO, f_hi: float = F_HI) -> np.ndarray:
    """Real-valued 4th-order Butterworth bandpass."""
    b, a = butter(4, [f_lo, f_hi], btype='bandpass', fs=sr)
    return filtfilt(b, a, audio).astype(np.float32)


def _instantaneous_freq(audio: np.ndarray, sr: int = SR_AUDIO,
                        smooth_s: float = INST_FREQ_BIN_S) -> np.ndarray:
    """Return f_inst(t) in Hz, length matches input audio length.

    Uses *squared-spectrum* peak — the same trick MAP144's tone-pair
    detector uses to find the carrier without modulation contamination.

    For MSK144 with carrier f₀ and ±500 Hz tone spacing, squaring the
    real audio produces two CW lines at 2·f₀ ± 1000 Hz.  The carrier
    is the midpoint divided by two, independent of the bit pattern.

    A sliding FFT (window ~50 ms) over the squared signal picks the
    two tones; we report (tone_lo + tone_hi)/4 as the instantaneous
    carrier estimate at the window centre.  Smoothing then fills
    samples between window centres.

    This is cleaner than Hilbert-phase-derivative which leaks MSK
    modulation into the estimate, inflating σ(f) within bursts.
    """
    win_s = 0.064               # ~768 samples — covers ~1 MSK frame
    win_n = int(win_s * sr)
    if win_n < 256:
        win_n = 256
    if win_n % 2:
        win_n += 1
    step_n = win_n // 4         # 25% overlap; smooth_s is ignored here
    nfft   = win_n
    df     = sr / nfft

    # Square first → modulation tones become CW
    sq = audio.astype(np.float64) ** 2
    # Search ranges for the squared tones (around 2*1500 ± 1000 = 2000 and 4000 Hz)
    # Working in bin coords:
    bin_lo_centre = int(round((2*1500 - 1000) / df))   # 2000 Hz → bin
    bin_hi_centre = int(round((2*1500 + 1000) / df))   # 4000 Hz → bin
    bin_half_win  = int(round(300.0 / df))             # ±300 Hz search

    f_estimates = np.full(len(audio), 1500.0, dtype=np.float32)   # default: nominal
    # Pre-compute a Hann window
    hann = np.hanning(win_n).astype(np.float64)
    for start in range(0, len(audio) - win_n + 1, step_n):
        seg = sq[start:start + win_n] * hann
        spec = np.abs(np.fft.rfft(seg))
        lo_lo = max(0, bin_lo_centre - bin_half_win)
        lo_hi = min(len(spec), bin_lo_centre + bin_half_win)
        hi_lo = max(0, bin_hi_centre - bin_half_win)
        hi_hi = min(len(spec), bin_hi_centre + bin_half_win)
        if lo_hi <= lo_lo or hi_hi <= hi_lo:
            continue
        i_lo = lo_lo + int(np.argmax(spec[lo_lo:lo_hi]))
        i_hi = hi_lo + int(np.argmax(spec[hi_lo:hi_hi]))
        # Sub-bin parabolic peak interpolation for each tone
        def _parabolic(i, s):
            if 1 <= i < len(s) - 1:
                a, b, c = s[i-1], s[i], s[i+1]
                den = (a - 2*b + c)
                if den != 0:
                    return i + 0.5 * (a - c) / den
            return float(i)
        f_lo = _parabolic(i_lo, spec) * df
        f_hi = _parabolic(i_hi, spec) * df
        f_carrier = (f_lo + f_hi) / 4.0    # /2 for squaring, /2 for midpoint
        # Fill the window range with this estimate (centered)
        win_centre = start + win_n // 2
        n_fill = step_n
        fill_lo = max(0, win_centre - n_fill // 2)
        fill_hi = min(len(f_estimates), win_centre + n_fill // 2)
        f_estimates[fill_lo:fill_hi] = f_carrier

    return f_estimates


def _envelope_db(audio: np.ndarray, sr: int = SR_AUDIO,
                 bin_s: float = ENV_BIN_S) -> tuple[np.ndarray, float]:
    """Return (envelope_db, bin_seconds) for bin_s-second RMS bins."""
    win = max(1, int(bin_s * sr))
    n_bins = len(audio) // win
    env = np.array([np.sqrt(np.mean(audio[i*win:(i+1)*win]**2))
                    for i in range(n_bins)], dtype=np.float64)
    return 20.0 * np.log10(np.maximum(env, 1e-10)), bin_s


def _find_bursts(env_db: np.ndarray, bin_s: float,
                 thresh_db: float = BURST_THRESH_DB,
                 min_duration_s: float = MIN_BURST_S) -> list[tuple[int, int]]:
    """Return [(idx_start, idx_end)] in env_db where signal exceeds median+thresh."""
    median = np.median(env_db)
    above = env_db > (median + thresh_db)
    bursts = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1; continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        if (j - i) * bin_s >= min_duration_s:
            bursts.append((i, j))
        i = j + 1
    return bursts


def analyse_wav(path: Path) -> tuple[np.ndarray, np.ndarray, float, list[BurstFreqStat]]:
    """Return (env_db, freq_t, env_bin_s, burst_stats) for a WAV file."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    if sr != SR_AUDIO:
        raise ValueError(f"expected {SR_AUDIO} Hz, got {sr}")

    audio_bp  = _bandpass(audio, sr)
    f_inst    = _instantaneous_freq(audio_bp, sr)
    env_db, env_bin_s = _envelope_db(audio_bp, sr)

    # For each envelope bin, the median instantaneous freq within that bin
    # (10 ms bin × 120 samples; freq array is per-sample).
    win = int(env_bin_s * sr)
    n_bins = len(env_db)
    freq_per_bin = np.array([np.median(f_inst[i*win:(i+1)*win]) for i in range(n_bins)],
                            dtype=np.float32)

    # Per-burst statistics
    bursts = _find_bursts(env_db, env_bin_s)
    median_db = float(np.median(env_db))
    stats: list[BurstFreqStat] = []
    for i_start, i_end in bursts:
        env_segment = env_db[i_start:i_end]
        freq_segment = freq_per_bin[i_start:i_end]
        # Drop NaN/inf
        good = np.isfinite(freq_segment) & np.isfinite(env_segment)
        if not good.any():
            continue
        env_seg = env_segment[good].astype(np.float64)
        f_seg   = freq_segment[good].astype(np.float64)
        if len(f_seg) < 2:
            continue
        peak_db = float(env_seg.max() - median_db)
        f_mean  = float(np.mean(f_seg))
        f_std   = float(np.std(f_seg))
        f_range = float(f_seg.max() - f_seg.min())
        # Pearson r between |A| (linear amplitude proportional) and f
        amp_lin = 10.0 ** ((env_seg - median_db) / 20.0)
        if f_std > 0 and np.std(amp_lin) > 0:
            af_corr = float(np.corrcoef(amp_lin, f_seg)[0, 1])
        else:
            af_corr = 0.0
        stats.append(BurstFreqStat(
            t_start=i_start * env_bin_s,
            t_end=i_end * env_bin_s,
            duration_ms=(i_end - i_start) * env_bin_s * 1000.0,
            peak_db=peak_db,
            f_mean_hz=f_mean,
            f_std_hz=f_std,
            f_range_hz=f_range,
            af_corr=af_corr,
        ))
    return env_db, freq_per_bin, env_bin_s, stats


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_wav(path: Path, env_db: np.ndarray, freq_t: np.ndarray, env_bin_s: float,
             bursts: list[BurstFreqStat], out_path: Path | None = None,
             pre_post_buffer_s: float = 0.5,
             decoded_msgs: list[str] | None = None,
             sync_track: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> Path:
    """Plot envelope + instantaneous freq, zoomed to the ping(s) with a
    pre/post-burst buffer for noise-floor reference.

    Zoom behaviour:
        - If any bursts detected: x-axis spans from
            (earliest burst start − pre_post_buffer_s)
          to
            (latest   burst end   + pre_post_buffer_s)
          clamped to the WAV duration.  Covers all bursts in the WAV
          including long extended pings.
        - If no bursts: falls back to the full WAV (15 s).

    Frequency-display tiers:
        Envelope > median + BURST_THRESH_DB (=4 dB) → dark solid points
          (in-burst — these are the signal-dominated estimates).
        median + 1 dB < envelope < BURST_THRESH_DB → light grey points
          (transition / sub-threshold signal — visible but de-emphasised).
        envelope < median + 1 dB → not plotted (noise-dominated; estimate
          is meaningless).

    Title includes ``decoded_msgs`` (decoded WSJT-X messages for this WAV's
    15-s period) when provided.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(len(env_db)) * env_bin_s
    median_db = float(np.median(env_db))
    wav_duration = float(len(env_db) * env_bin_s)

    # Compute zoom window
    if bursts:
        t_start = max(0.0, min(b.t_start for b in bursts) - pre_post_buffer_s)
        t_end   = min(wav_duration, max(b.t_end for b in bursts) + pre_post_buffer_s)
        # Guarantee minimum 1-second window even if everything is squeezed
        if t_end - t_start < 1.0:
            mid = (t_end + t_start) / 2.0
            t_start = max(0.0, mid - 0.5)
            t_end   = min(wav_duration, mid + 0.5)
        zoom_note = f"  [zoom {t_start:.2f}–{t_end:.2f} s, ±{pre_post_buffer_s:.1f} s noise reference]"
    else:
        t_start, t_end = 0.0, wav_duration
        zoom_note = "  [no bursts — full WAV shown]"

    fig, (ax_a, ax_f) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax_a.plot(t, env_db - median_db, color='C0', lw=1.0)
    ax_a.axhline(BURST_THRESH_DB, color='r', ls='--', lw=0.5,
                 label=f'{BURST_THRESH_DB} dB threshold')
    ax_a.set_ylabel('Envelope (dB above median)')
    # Title — include decoded messages if any
    if decoded_msgs:
        msg_str = "  /  ".join(decoded_msgs[:3])
        if len(decoded_msgs) > 3:
            msg_str += f"  /  +{len(decoded_msgs)-3} more"
        title = f'{path.name} — {msg_str}{zoom_note}'
    else:
        title = f'{path.name} — envelope (top) and instantaneous freq (bottom){zoom_note}'
    ax_a.set_title(title)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.set_xlim(t_start, t_end)
    # Auto-scale envelope y-axis to the visible zoom window so WAV-boundary
    # zeros (huge negative dB at file edges) don't squash the data range.
    env_zoom_mask = (t >= t_start) & (t <= t_end)
    if env_zoom_mask.any():
        env_visible = env_db[env_zoom_mask] - median_db
        env_lo = float(np.nanmin(env_visible)) - 2.0
        env_hi = float(np.nanmax(env_visible)) + 2.0
        # Ensure the burst-threshold line stays visible
        env_hi = max(env_hi, BURST_THRESH_DB + 1.0)
        # Sensible floor — don't zoom below -10 dB so noise spread shows
        env_lo = max(env_lo, -10.0)
        ax_a.set_ylim(env_lo, env_hi)

    # Two-tier frequency plot:
    #   tier A (in-burst):   env > median + BURST_THRESH_DB → dark solid
    #   tier B (sub-threshold): median+1 dB < env < BURST_THRESH_DB → light grey
    #   below median+1 dB: skipped (pure noise, estimate meaningless)
    tier_a_mask = env_db > (median_db + BURST_THRESH_DB)
    tier_b_mask = (env_db > (median_db + 1.0)) & ~tier_a_mask
    # Plot tier B first so tier A appears on top
    fb = np.where(tier_b_mask, freq_t - 1500.0, np.nan)
    fa = np.where(tier_a_mask, freq_t - 1500.0, np.nan)
    ax_f.plot(t, fb, 'o', color='lightgrey', markersize=3, label='sq-FFT  sub-threshold')
    ax_f.plot(t, fa, 'o-', color='C2', markersize=3.5, lw=0.8,
              label=f'sq-FFT  in-burst (>{BURST_THRESH_DB:.0f} dB)')
    # Optional: overlay sync-correlation carrier track (matched filter).
    # Threshold-filter to show only the high-confidence picks — the
    # correlator returns *some* freq for every window even on noise, and
    # plotting all of them clutters the plot beyond use.
    #
    # Threshold = median magnitude + 1.5 × MAD (robust against the long tail
    # of high-signal samples).  Empirically this keeps the in-burst markers
    # and the strongest sub-threshold matches, while suppressing the
    # noise-floor scatter.  Marker size within the kept set scales with
    # confidence so the strongest locks visually dominate.
    if sync_track is not None:
        ts, fs, ms = sync_track
        if len(ms) > 0:
            med = float(np.median(ms))
            mad = float(np.median(np.abs(ms - med)))
            thr = med + 1.5 * max(mad, 1.0)
            keep = ms > thr
            ts_k, fs_k, ms_k = ts[keep], fs[keep], ms[keep]
            if len(ms_k) > 0:
                m_norm = (ms_k - thr) / max(1e-6, ms_k.max() - thr)
                sizes = 10.0 + 40.0 * m_norm    # 10 (just above thr) .. 50 (strongest)
                # fs is already the offset from 1500 Hz audio — do NOT subtract 1500
                ax_f.scatter(ts_k, fs_k, s=sizes, marker='x', color='C3',
                             alpha=0.85, linewidths=1.4,
                             label=f'sync-corr peak  (mag > {thr:.0f}, '
                                   f'{keep.sum()}/{len(ms)} pts kept)')
    ax_f.axhline(0.0, color='k', ls=':', lw=0.5)
    ax_f.set_ylabel('Inst. freq − 1500 Hz')
    ax_f.set_xlabel('Time within WAV (s)')
    ax_f.grid(alpha=0.3)
    ax_f.set_xlim(t_start, t_end)
    ax_f.legend(loc='upper right', fontsize=7)
    # Combined mask for y-axis auto-scaling (use both tiers)
    mask = tier_a_mask | tier_b_mask

    # Auto-scale freq y-axis based on the data visible in the zoom window.
    # Use 2nd–98th percentile to be robust against outliers from measurement
    # glitches (e.g., the squared-FFT peak finder jumping between two signals
    # in a multi-station period can produce one-bin spikes >1 kHz that would
    # otherwise dominate the scale).
    zoom_mask = (t >= t_start) & (t <= t_end) & mask
    if zoom_mask.any():
        f_visible = freq_t[zoom_mask] - 1500.0
        f_visible = f_visible[np.isfinite(f_visible)]
        if len(f_visible) >= 3:
            p_lo = float(np.percentile(f_visible, 2))
            p_hi = float(np.percentile(f_visible, 98))
        else:
            p_lo = float(np.nanmin(f_visible))
            p_hi = float(np.nanmax(f_visible))
        # Pad and ensure zero line is visible
        f_lo = min(p_lo - 5.0, -10.0)
        f_hi = max(p_hi + 5.0, +10.0)
        # Minimum span 30 Hz; hard cap of ±200 Hz so outliers can't blow it up
        if (f_hi - f_lo) < 30.0:
            mid = (f_lo + f_hi) / 2.0
            f_lo, f_hi = mid - 15.0, mid + 15.0
        f_lo = max(f_lo, -200.0)
        f_hi = min(f_hi, +200.0)
        ax_f.set_ylim(f_lo, f_hi)
    else:
        ax_f.set_ylim(-150, 150)

    # Annotate each burst region with its stats
    for b in bursts:
        for ax in (ax_a, ax_f):
            ax.axvspan(b.t_start, b.t_end, color='C1', alpha=0.15)
        ax_a.annotate(
            f'{b.duration_ms:.0f} ms\n+{b.peak_db:.1f} dB',
            xy=((b.t_start + b.t_end)/2, ax_a.get_ylim()[1]*0.85),
            ha='center', fontsize=7,
        )
        ax_f.annotate(
            f'σ={b.f_std_hz:.1f}\nΔ={b.f_range_hz:.1f}\nr={b.af_corr:+.2f}',
            xy=((b.t_start + b.t_end)/2, ax_f.get_ylim()[1]*0.78),
            ha='center', fontsize=7,
        )

    if out_path is None:
        out_path = Path('/tmp') / f'freqtime_{path.stem}.png'
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None, help='Single WAV path or filename in --wsjtx-dir/save/')
    ap.add_argument('--wsjtx-dir', type=Path,
                    default=Path("/home/jeff/.local/share/WSJT-X - flex"),
                    help='WSJT-X user directory containing save/')
    ap.add_argument('--date', default=None,
                    help='Process all WAVs for this UTC date (YYYYMMDD)')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Write per-burst stats to this CSV (batch mode)')
    ap.add_argument('--max-wavs', type=int, default=None,
                    help='Limit batch to first N WAVs')
    ap.add_argument('--plot-each-burst', type=Path, default=None,
                    metavar='DIR',
                    help='In batch mode, generate a freq-vs-time plot for every '
                         'WAV that has at least one burst, saving into DIR.  '
                         'Filenames: freqtime_<wav-stem>.png')
    ap.add_argument('--sync-corr', action='store_true',
                    help='Overlay the sync-correlation-peak carrier track on the '
                         'freq panel.  Matched filter against the MSK144 sync '
                         'template — works at lower SNR than the squared-FFT '
                         'estimator that drives the dots/lines.  Slower (~5-10x).')
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'

    # ── Single-WAV mode ────────────────────────────────────────────────────
    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        if not wav_path.exists():
            print(f"error: WAV not found: {args.wav}", file=sys.stderr)
            return 1
        env_db, freq_t, bin_s, bursts = analyse_wav(wav_path)
        # Look up decoded messages from WSJT-X ALL.TXT for the title
        msg_index = _load_wsjtx_messages(args.wsjtx_dir / 'ALL.TXT')
        decoded_msgs = _msgs_for_wav(wav_path, msg_index) if msg_index else []
        # Optional sync-correlation freq track overlay
        sync_track = None
        if args.sync_corr:
            import wave
            with wave.open(str(wav_path), 'rb') as wf:
                _sr = wf.getframerate()
                _audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                          .astype(np.float32) / 32768.0)
            print("  computing sync-correlation freq track...", file=sys.stderr)
            sync_track = sync_corr_freq_track(_audio, _sr)
            print(f"  sync-corr: {len(sync_track[0])} samples", file=sys.stderr)
        png = plot_wav(wav_path, env_db, freq_t, bin_s, bursts,
                       decoded_msgs=decoded_msgs, sync_track=sync_track)
        msgs_str = '  /  '.join(decoded_msgs) if decoded_msgs else '(no decoded message)'
        print(f"{wav_path.name}  ({len(bursts)} bursts)  WSJT-X: {msgs_str}")
        for b in bursts:
            print(f"  t={b.t_start:5.2f}-{b.t_end:5.2f}s  dur={b.duration_ms:5.0f} ms  "
                  f"peak +{b.peak_db:.1f} dB  f̄={b.f_mean_hz-1500:+6.1f}  "
                  f"σ_f={b.f_std_hz:5.2f}  Δf={b.f_range_hz:5.1f}  r(|A|,f)={b.af_corr:+.2f}")
        print(f"\nPlot: {png}")
        return 0

    # ── Batch mode ──────────────────────────────────────────────────────────
    if not args.date:
        print("error: provide --wav or --date", file=sys.stderr)
        return 1
    date_prefix = args.date[2:]
    wavs = sorted(save_dir.glob(f'{date_prefix}_*.wav'))
    if args.max_wavs:
        wavs = wavs[:args.max_wavs]
    if not wavs:
        print(f"error: no WAVs for date {args.date}", file=sys.stderr)
        return 1

    print(f"Processing {len(wavs)} WAVs from {save_dir}", file=sys.stderr)
    if args.plot_each_burst:
        args.plot_each_burst.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be written to {args.plot_each_burst}/", file=sys.stderr)
    # Pre-load WSJT-X ALL.TXT messages so each plot can show the decoded calls
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt)
    if msg_index:
        print(f"Loaded {len(msg_index)} periods of WSJT-X decoded messages "
              f"from {all_txt.name}", file=sys.stderr)
    all_bursts: list[tuple[str, BurstFreqStat]] = []
    n_plotted = 0
    for i, p in enumerate(wavs):
        try:
            env_db, freq_t, bin_s, bursts = analyse_wav(p)
        except Exception as e:
            print(f"  {p.name}: ERROR {e}", file=sys.stderr)
            continue
        for b in bursts:
            all_bursts.append((p.name, b))
        if args.plot_each_burst and bursts:
            out_png = args.plot_each_burst / f"freqtime_{p.stem}.png"
            decoded_msgs = _msgs_for_wav(p, msg_index)
            try:
                plot_wav(p, env_db, freq_t, bin_s, bursts, out_path=out_png,
                         decoded_msgs=decoded_msgs)
                n_plotted += 1
            except Exception as e:
                print(f"  {p.name}: PLOT ERROR {e}", file=sys.stderr)
        if (i+1) % 25 == 0 or i+1 == len(wavs):
            extra = f", {n_plotted} plotted" if args.plot_each_burst else ""
            print(f"  [{i+1}/{len(wavs)}]  {len(all_bursts)} bursts total{extra}",
                  file=sys.stderr)

    # CSV output
    if args.csv:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 't_start_s', 't_end_s', 'dur_ms',
                        'peak_db', 'f_mean_hz_minus_1500', 'f_std_hz',
                        'f_range_hz', 'af_corr'])
            for name, b in all_bursts:
                w.writerow([name, f'{b.t_start:.3f}', f'{b.t_end:.3f}',
                            f'{b.duration_ms:.1f}', f'{b.peak_db:.2f}',
                            f'{b.f_mean_hz-1500:+.2f}', f'{b.f_std_hz:.3f}',
                            f'{b.f_range_hz:.2f}', f'{b.af_corr:+.3f}'])
        print(f"CSV written: {args.csv}")

    # ── Summary distributions ────────────────────────────────────────────────
    if not all_bursts:
        print("No bursts found.")
        return 0
    arr_dur   = np.array([b.duration_ms      for _, b in all_bursts])
    arr_peak  = np.array([b.peak_db          for _, b in all_bursts])
    arr_fstd  = np.array([b.f_std_hz         for _, b in all_bursts])
    arr_fran  = np.array([b.f_range_hz       for _, b in all_bursts])
    arr_corr  = np.array([b.af_corr          for _, b in all_bursts])

    print()
    print(f"{'metric':<28}  {'n':>5}  {'min':>8}  {'p25':>8}  {'med':>8}  {'p75':>8}  {'max':>8}")
    for label, arr, unit in [
        ('burst duration (ms)',   arr_dur,  'ms'),
        ('peak above noise (dB)', arr_peak, 'dB'),
        ('σ(f) within burst',     arr_fstd, 'Hz'),
        ('Δf within burst',       arr_fran, 'Hz'),
        ('r(|A|, f) within burst',arr_corr, ''),
    ]:
        print(f"  {label:<26}  {len(arr):>5}  "
              f"{arr.min():>+8.2f}  {np.percentile(arr,25):>+8.2f}  "
              f"{np.median(arr):>+8.2f}  {np.percentile(arr,75):>+8.2f}  "
              f"{arr.max():>+8.2f}")

    # Bucket by burst duration: does longer-burst = more drift?
    print()
    print("σ(f) by burst-duration bucket:")
    duration_buckets = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 2000)]
    for lo, hi in duration_buckets:
        mask = (arr_dur >= lo) & (arr_dur < hi)
        if not mask.any():
            continue
        sub = arr_fstd[mask]
        print(f"  duration {lo:>4}-{hi:>4} ms (n={mask.sum():>3}): "
              f"σ_f median={np.median(sub):.2f}  p75={np.percentile(sub,75):.2f}  "
              f"max={sub.max():.2f}")

    # Pearson |A|↔f correlation
    print()
    print(f"r(|A|, f) distribution: how often does amplitude flutter track freq flutter?")
    print(f"  |r| > 0.5 (strong)  : {np.sum(np.abs(arr_corr) > 0.5):>4} / {len(arr_corr)} bursts")
    print(f"  |r| > 0.3 (moderate): {np.sum(np.abs(arr_corr) > 0.3):>4} / {len(arr_corr)}")
    print(f"  |r| < 0.1 (none)    : {np.sum(np.abs(arr_corr) < 0.1):>4} / {len(arr_corr)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
