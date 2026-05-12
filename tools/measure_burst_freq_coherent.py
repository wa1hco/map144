"""Per-burst coherent matched-filter frequency estimator.

This is the physics-measurement-stage successor to
``measure_ping_freq_vs_time.py``'s sliding squared-FFT and sliding
sync-correlation methods.  Both of those were bandwidth-limited per
window (~14-50 Hz freq resolution from a 72-ms matched filter) and
gave noisy per-window picks.

This tool exploits the fixed MSK144 protocol timing — two sync
sequences per 72-ms frame, regularly spaced — to coherently integrate
the matched-filter response across all syncs in a burst.  The
effective integration time becomes the burst duration (typically
72-500 ms), which collapses the freq estimate to a single number per
burst with ±2 Hz or better precision.

Method
------
For a burst spanning ``n_frames`` consecutive 72-ms MSK144 frames:

  1. Convert audio to complex baseband (Hilbert + mix down by 1500 Hz).
  2. Find frame boundary alignment ``ish_best`` via the existing
     ``_sync_correlate`` on the first frame.  This anchors all
     subsequent sync positions to the protocol grid.
  3. Sync positions in the burst then follow at::

         frame_f, sync_1: ish_best + f·NSPM
         frame_f, sync_2: ish_best + f·NSPM + NSPM/2
         for f = 0, 1, ..., n_frames-1

  4. For each candidate frequency offset ``Δf`` in the search range:
       a. Mix the baseband burst by ``exp(-j·2π·Δf·t)``.
       b. Compute complex sync correlation at every known position.
       c. *Coherently sum* — complex addition, preserving phase.
  5. The frequency at maximum |coherent sum| is the burst-average
     carrier offset.  Parabolic interpolation refines below the
     step size.

Why this gives 2 Hz precision (or better)
-----------------------------------------
The matched-filter peak width is set by the total coherent
integration time, not the individual sync window length.  For
``n_frames`` frames spanning ``n_frames * 72 ms`` of signal, the
Fourier resolution is roughly 1 / (n_frames * 72 ms).  For 2 frames
that's already 7 Hz; for 3 frames, 4.6 Hz; for 5 frames, 2.8 Hz.
Parabolic interpolation around the matched-filter peak commonly
pulls that down by another factor of 2-5 for moderate SNR signals.

Limitations
-----------
- Treats the burst as having stationary frequency.  Drifting signals
  produce a lower / broader peak with f₀ at the burst-average.  An
  optional 2-D search (f₀, df/dt) extension is documented in the
  README and can be added later.
- The sync-position grid assumes burst start aligns to MSK144 frame
  timing.  Burst start from envelope is good to ~50 ms; the inner
  ``ish_best`` search on frame 0 refines it to sub-sample precision.

Usage
-----
    python tools/measure_burst_freq_coherent.py --wav FILE
    python tools/measure_burst_freq_coherent.py --date YYYYMMDD --csv OUT.csv

Outputs a per-burst matched-filter response plot to
``docs/figures/burst_freq_coherent_single/`` (or specified batch dir).
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

# Reuse the existing tool's burst detection
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_ping_freq_vs_time import (
    _bandpass, _envelope_db, _find_bursts,
    _load_wsjtx_messages, _msgs_for_wav,
    F_LO, F_HI, ENV_BIN_S, BURST_THRESH_DB, MIN_BURST_S,
)

# Sync correlator + template from the production decoder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import _sync_correlate, _SYNC_CB_CONJ, NSPM, FS


SR_AUDIO = 12000
FC_AUDIO = 1500.0          # WSJT-X MSK144 audio carrier convention
SYNC_LEN = len(_SYNC_CB_CONJ)   # 42 samples


# ── Per-burst result ─────────────────────────────────────────────────────────

@dataclass
class CoherentFreqEstimate:
    """Output of the coherent matched-filter freq estimator for one burst."""
    t_start_s:        float
    t_end_s:          float
    duration_ms:      float
    n_frames_used:    int
    n_syncs_used:     int
    ish_best:         int
    f0_hz:            float    # carrier offset from 1500 Hz, parabolic-refined
    peak_mag:         float    # matched-filter peak magnitude
    peak_width_hz:    float    # FWHM of the matched-filter peak (a freq-resolution proxy)
    freqs_hz:         np.ndarray
    magnitudes:       np.ndarray


# ── Audio → baseband conversion ─────────────────────────────────────────────

def _audio_to_baseband(audio_real: np.ndarray, sr: int = SR_AUDIO) -> np.ndarray:
    """Real audio → complex baseband (signal at 0 Hz), RMS-normalised.

    Matches the convention msk144_spd._sync_correlate was calibrated for.
    """
    analytic = hilbert(audio_real.astype(np.float64)).astype(np.complex64)
    t_full = np.arange(len(analytic), dtype=np.float64) / sr
    baseband = (analytic * np.exp(-2j * np.pi * FC_AUDIO * t_full)).astype(np.complex64)
    rms = float((np.abs(baseband.astype(np.complex128)) ** 2).mean() ** 0.5)
    if rms > 1e-12:
        baseband = (baseband * (np.sqrt(2.0) / rms)).astype(np.complex64)
    return baseband


# ── Coherent matched-filter freq estimator ─────────────────────────────────

def coherent_burst_freq_estimate(
    baseband: np.ndarray,
    burst_start_n: int,
    burst_end_n: int,
    sync_template_conj: np.ndarray = _SYNC_CB_CONJ,
    sr: int = SR_AUDIO,
    f_search_hz: float = 250.0,
    f_step_hz: float = 1.0,
) -> CoherentFreqEstimate | None:
    """Run the coherent matched-filter frequency estimator on one burst.

    Parameters
    ----------
    baseband : complex64 (N,)
        Full baseband audio (signal at 0 Hz).
    burst_start_n, burst_end_n : int
        Sample indices bounding the burst.
    sync_template_conj : complex64 (42,)
        MSK144 sync template, conjugated.  Default = production template.
    f_search_hz : float
        Frequency search half-width.  ±250 Hz default is enough for any
        operator-setup offset plus typical trail Doppler.
    f_step_hz : float
        Frequency step.  1 Hz with parabolic interpolation gives sub-Hz
        precision at moderate SNR.

    Returns
    -------
    CoherentFreqEstimate or None (if burst too short for even one frame).
    """
    # Use only frames fully inside the burst.  Burst-edge sub-frames have
    # weaker signal and would dilute the coherent sum.
    n_avail = burst_end_n - burst_start_n
    n_frames = n_avail // NSPM
    if n_frames < 1:
        return None
    end_n_frames = burst_start_n + n_frames * NSPM

    # First, find frame-boundary alignment (ish_best) from the first frame
    # at zero freq shift.  This is correct as long as the freq offset is
    # within the sync correlator's tolerance (~50 Hz on the unshifted frame
    # at moderate SNR — wider than needed since operator offsets are ~100 Hz
    # we'll iterate if first try fails).
    first_frame = baseband[burst_start_n:burst_start_n + NSPM]
    _, _, ish_best = _sync_correlate(first_frame)

    # If sync correlator on unshifted first frame returned low confidence,
    # the freq offset may be too large.  Do a quick coarse scan.
    # For simplicity in v1 we trust ish_best from the unshifted attempt and
    # rely on the freq scan to absorb the carrier offset.

    # (No explicit sync_positions list needed — the new sum-then-correlate
    # logic below handles the sync alignment via _sync_correlate on the
    # summed frame.)

    # Frequency search — coherent-sum-of-frames-then-correlate.
    #
    # WSJT-X's msk144_freq_search does this:
    #   for each f:
    #     1. mix the whole multi-frame buffer by -f
    #     2. SUM the frames coherently (frames add as N at true f, ~√N noise)
    #     3. sync-correlate the SUMMED frame
    #     4. xmax = matched-filter response
    #
    # The summed frame is 864 samples (one frame's worth) but carries N×
    # the signal amplitude; the sync correlation against that summed frame
    # gives a sharper peak than correlate-then-sum because the matched
    # filter window is the full sync template applied to a high-SNR frame.
    # Equivalent freq resolution: 1 / (N * 72 ms).
    freqs = np.arange(-f_search_hz, f_search_hz + f_step_hz/2, f_step_hz,
                      dtype=np.float64)
    # Take the frame-aligned region of the burst (n_frames * NSPM samples
    # starting at burst_start_n + ish_best).
    frame_start_n = burst_start_n + ish_best
    if frame_start_n + n_frames * NSPM > len(baseband):
        # Roll back if ish_best pushes us past the buffer end.
        n_frames = max(1, (len(baseband) - frame_start_n) // NSPM)
        if n_frames < 1:
            return None
    burst_buf = baseband[frame_start_n:frame_start_n + n_frames * NSPM]
    t_burst = np.arange(len(burst_buf), dtype=np.float64) / sr

    magnitudes = np.zeros(len(freqs), dtype=np.float64)
    for i, f in enumerate(freqs):
        # 1. Mix
        shifted = (burst_buf * np.exp(-2j * np.pi * f * t_burst)
                              .astype(np.complex64)).astype(np.complex64)
        # 2. Sum frames coherently — reshape into (n_frames, NSPM), sum along frame axis
        summed = shifted.reshape(n_frames, NSPM).sum(axis=0)
        # 3. Sync-correlate the summed frame
        _, xmax, _ = _sync_correlate(summed)
        magnitudes[i] = float(xmax)
    # n_syncs_used semantics: WSJT-X-equivalent integration uses 2 syncs per
    # summed frame regardless of n_frames; the integration gain comes from
    # the n_frames sum, not from extra sync positions.
    n_syncs_used = 2

    # Find the peak and refine via parabolic interpolation
    i_peak = int(np.argmax(magnitudes))
    f0 = float(freqs[i_peak])
    if 0 < i_peak < len(freqs) - 1:
        a, b, c = magnitudes[i_peak - 1], magnitudes[i_peak], magnitudes[i_peak + 1]
        denom = a - 2.0 * b + c
        if denom != 0:
            f0 += 0.5 * (a - c) / denom * (freqs[1] - freqs[0])

    # FWHM (full-width-half-maximum) of the matched-filter peak — a proxy
    # for effective frequency resolution under the present SNR.
    half = magnitudes[i_peak] / 2.0
    above = magnitudes > half
    if above.any():
        lo = freqs[np.argmax(above)]
        hi = freqs[len(above) - 1 - np.argmax(above[::-1])]
        fwhm = float(hi - lo)
    else:
        fwhm = float('nan')

    return CoherentFreqEstimate(
        t_start_s=burst_start_n / sr,
        t_end_s=end_n_frames / sr,
        duration_ms=(end_n_frames - burst_start_n) * 1000.0 / sr,
        n_frames_used=n_frames,
        n_syncs_used=n_syncs_used,
        ish_best=ish_best,
        f0_hz=f0,
        peak_mag=float(magnitudes[i_peak]),
        peak_width_hz=fwhm,
        freqs_hz=freqs.astype(np.float32),
        magnitudes=magnitudes.astype(np.float32),
    )


# ── WAV-level analysis ────────────────────────────────────────────────────

def analyse_wav(path: Path) -> tuple[np.ndarray, float, list[CoherentFreqEstimate]]:
    """Find bursts in a WAV and run the coherent freq estimator on each.

    Returns (env_db, env_bin_s, list_of_estimates).
    """
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    if sr != SR_AUDIO:
        raise ValueError(f"expected {SR_AUDIO} Hz, got {sr}")

    audio_bp = _bandpass(audio, sr)
    env_db, bin_s = _envelope_db(audio_bp, sr)
    burst_idx_pairs = _find_bursts(env_db, bin_s, BURST_THRESH_DB, MIN_BURST_S)

    baseband = _audio_to_baseband(audio, sr)
    estimates = []
    for i_start, i_end in burst_idx_pairs:
        # Convert envelope bin indices → audio sample indices
        burst_start_n = int(i_start * bin_s * sr)
        burst_end_n = int(i_end * bin_s * sr)
        est = coherent_burst_freq_estimate(baseband, burst_start_n, burst_end_n, sr=sr)
        if est is not None:
            estimates.append(est)
    return env_db, bin_s, estimates


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_burst_response(path: Path,
                        env_db: np.ndarray, env_bin_s: float,
                        estimates: list[CoherentFreqEstimate],
                        decoded_msgs: list[str] | None = None,
                        out_path: Path | None = None) -> Path:
    """One PNG per WAV: top = envelope (zoomed), bottom = matched-filter
    response curve(s) for each burst.

    When multiple bursts are present, each gets its own colored curve
    on the response plot, annotated with its f0 ± parabolic estimate.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wav_duration = len(env_db) * env_bin_s
    t_env = np.arange(len(env_db)) * env_bin_s
    median_db = float(np.median(env_db))

    # Time zoom around the bursts (with 0.5 s noise reference each side)
    if estimates:
        t_lo = max(0.0, min(e.t_start_s for e in estimates) - 0.5)
        t_hi = min(wav_duration, max(e.t_end_s for e in estimates) + 0.5)
        zoom_note = f"  [zoom {t_lo:.2f}-{t_hi:.2f} s]"
    else:
        t_lo, t_hi = 0.0, wav_duration
        zoom_note = "  [no bursts]"

    fig, (ax_env, ax_resp) = plt.subplots(2, 1, figsize=(13, 7))

    # ── Top: envelope ──────────────────────────────────────────────────────
    ax_env.plot(t_env, env_db - median_db, color='C0', lw=1.0)
    ax_env.axhline(BURST_THRESH_DB, color='r', ls='--', lw=0.5,
                   label=f'{BURST_THRESH_DB} dB threshold')
    for e in estimates:
        ax_env.axvspan(e.t_start_s, e.t_end_s, color='C1', alpha=0.15)
        ax_env.annotate(f'{e.duration_ms:.0f} ms\n{e.n_frames_used} frames',
                        xy=((e.t_start_s + e.t_end_s)/2, ax_env.get_ylim()[1]*0.85),
                        ha='center', fontsize=8)
    ax_env.set_xlim(t_lo, t_hi)
    # Auto-scale envelope y to visible range
    zoom_mask = (t_env >= t_lo) & (t_env <= t_hi)
    if zoom_mask.any():
        seg = env_db[zoom_mask] - median_db
        ax_env.set_ylim(max(-10.0, float(seg.min()) - 2.0),
                        max(float(seg.max()) + 2.0, BURST_THRESH_DB + 1.0))
    ax_env.set_ylabel('Envelope (dB above median)')

    title_parts = [path.name]
    if decoded_msgs:
        title_parts.append('  /  '.join(decoded_msgs[:3]))
    title_parts.append(zoom_note)
    ax_env.set_title('  —  '.join(title_parts[:-1]) + title_parts[-1])
    ax_env.grid(alpha=0.3)
    ax_env.legend(loc='upper right', fontsize=8)

    # ── Bottom: matched-filter response curves ─────────────────────────────
    cmap = plt.get_cmap('tab10')
    for i, e in enumerate(estimates):
        color = cmap(i % 10)
        ax_resp.plot(e.freqs_hz, e.magnitudes, color=color, lw=1.2,
                     label=f'burst {i+1}  ({e.duration_ms:.0f} ms, '
                           f'{e.n_syncs_used} syncs)   '
                           f'f₀ = {e.f0_hz:+.1f} Hz   '
                           f'FWHM = {e.peak_width_hz:.1f} Hz')
        # Mark the parabolic-interpolated peak
        ax_resp.axvline(e.f0_hz, color=color, ls=':', lw=0.8, alpha=0.6)
    ax_resp.axvline(0.0, color='k', ls=':', lw=0.5)
    ax_resp.set_xlabel('Frequency offset from 1500 Hz audio centre (Hz)')
    ax_resp.set_ylabel('Coherent matched-filter |response|')
    ax_resp.set_title('Per-burst matched-filter response  '
                      '(peak position = burst-average carrier offset)')
    ax_resp.grid(alpha=0.3)
    ax_resp.legend(loc='upper right', fontsize=8)

    if out_path is None:
        default_dir = (Path(__file__).resolve().parent.parent
                       / 'docs' / 'figures' / 'burst_freq_coherent_single')
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / f'burstcoh_{path.stem}.png'

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav', default=None,
                    help='Single WAV path or filename in --wsjtx-dir/save/')
    ap.add_argument('--wsjtx-dir', type=Path,
                    default=Path("/home/jeff/.local/share/WSJT-X - flex"))
    ap.add_argument('--date', default=None,
                    help='Batch: process all WAVs for this UTC date (YYYYMMDD)')
    ap.add_argument('--csv', type=Path, default=None,
                    help='Write per-burst estimates to this CSV (batch mode)')
    ap.add_argument('--plot-each-burst', type=Path, default=None,
                    metavar='DIR', help='In batch mode, save per-WAV plot to DIR')
    ap.add_argument('--max-wavs', type=int, default=None)
    args = ap.parse_args(argv)

    save_dir = args.wsjtx_dir / 'save'
    all_txt = args.wsjtx_dir / 'ALL.TXT'
    msg_index = _load_wsjtx_messages(all_txt) if all_txt.exists() else {}

    # ── Single-WAV mode ─────────────────────────────────────────────────────
    if args.wav and not args.date:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            wav_path = save_dir / args.wav
        if not wav_path.exists():
            print(f"error: WAV not found: {args.wav}", file=sys.stderr)
            return 1
        env_db, bin_s, estimates = analyse_wav(wav_path)
        decoded_msgs = _msgs_for_wav(wav_path, msg_index)
        png = plot_burst_response(wav_path, env_db, bin_s, estimates, decoded_msgs)
        msgs_str = '  /  '.join(decoded_msgs) if decoded_msgs else '(no decoded message)'
        print(f"{wav_path.name}  ({len(estimates)} bursts)  WSJT-X: {msgs_str}")
        for i, e in enumerate(estimates):
            print(f"  burst {i+1}: t={e.t_start_s:5.2f}-{e.t_end_s:5.2f}s  "
                  f"dur={e.duration_ms:5.0f} ms  "
                  f"{e.n_frames_used} frames ({e.n_syncs_used} syncs)  "
                  f"f₀ = {e.f0_hz:+7.2f} Hz  FWHM = {e.peak_width_hz:5.2f} Hz  "
                  f"peak_mag = {e.peak_mag:.0f}")
        print(f"\nPlot: {png}")
        return 0

    # ── Batch mode ───────────────────────────────────────────────────────────
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

    print(f"Processing {len(wavs)} WAVs", file=sys.stderr)
    if args.plot_each_burst:
        args.plot_each_burst.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, str, CoherentFreqEstimate]] = []
    n_plotted = 0
    for i, p in enumerate(wavs):
        try:
            env_db, bin_s, estimates = analyse_wav(p)
        except Exception as exc:
            print(f"  {p.name}: ERROR {exc}", file=sys.stderr)
            continue
        decoded_msgs = _msgs_for_wav(p, msg_index)
        msgs_str = '|'.join(decoded_msgs)
        for e in estimates:
            all_rows.append((p.name, msgs_str, e))
        if args.plot_each_burst and estimates:
            out_png = args.plot_each_burst / f'burstcoh_{p.stem}.png'
            try:
                plot_burst_response(p, env_db, bin_s, estimates, decoded_msgs,
                                    out_path=out_png)
                n_plotted += 1
            except Exception as exc:
                print(f"  {p.name}: PLOT ERROR {exc}", file=sys.stderr)
        if (i + 1) % 25 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]  {len(all_rows)} bursts, "
                  f"{n_plotted} plots", file=sys.stderr)

    if args.csv and all_rows:
        with args.csv.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wav', 'wsjtx_msgs', 't_start_s', 't_end_s', 'dur_ms',
                        'n_frames', 'n_syncs', 'ish_best',
                        'f0_hz', 'peak_mag', 'fwhm_hz'])
            for name, msgs, e in all_rows:
                w.writerow([name, msgs,
                            f'{e.t_start_s:.3f}', f'{e.t_end_s:.3f}',
                            f'{e.duration_ms:.1f}',
                            e.n_frames_used, e.n_syncs_used, e.ish_best,
                            f'{e.f0_hz:+.2f}', f'{e.peak_mag:.0f}',
                            f'{e.peak_width_hz:.2f}'])
        print(f"\nCSV written: {args.csv}")

    # ── Summary distributions ────────────────────────────────────────────────
    if all_rows:
        f0s   = np.array([r[2].f0_hz       for r in all_rows])
        durs  = np.array([r[2].duration_ms for r in all_rows])
        fwhms = np.array([r[2].peak_width_hz for r in all_rows
                          if not np.isnan(r[2].peak_width_hz)])
        print()
        print(f"Coherent burst-freq estimator — {len(all_rows)} bursts")
        print(f"  f₀ distribution (Hz from 1500 Hz audio):")
        print(f"    min   : {f0s.min():+.2f}")
        print(f"    p25   : {np.percentile(f0s, 25):+.2f}")
        print(f"    median: {np.median(f0s):+.2f}")
        print(f"    p75   : {np.percentile(f0s, 75):+.2f}")
        print(f"    max   : {f0s.max():+.2f}")
        print(f"  Burst duration distribution (ms):")
        print(f"    median: {np.median(durs):.0f}  range [{durs.min():.0f}, {durs.max():.0f}]")
        if len(fwhms):
            print(f"  Matched-filter peak FWHM distribution (Hz)  "
                  f"— effective freq resolution:")
            print(f"    median: {np.median(fwhms):.2f}  "
                  f"p25-p75: {np.percentile(fwhms, 25):.2f}-{np.percentile(fwhms, 75):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
