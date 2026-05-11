#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Quantitative test for per-channel banding in tone and sync detection.

Replays a WAV file (or generates flat AWGN) through the MAP144 Engine,
captures per-channel detection metrics at every hop, and reports
whether persistent channel-to-channel bias exists.

The metric of interest is **time-averaged peak-to-peak (P-P) across
channels** — if channel A has consistently higher detection value than
channel B over many hops (rather than randomly varying), that's
banding.  Per-hop noise produces large per-channel variance but small
time-averaged-across-channels variance.  Banding produces small
per-hop variance but large time-averaged-across-channels variance.

Usage::

    # AWGN through fresh Engine
    python3 tools/measure_banding.py --synthetic 30

    # Replay a known WAV through fresh Engine
    python3 tools/measure_banding.py tests/fixtures/parity_strong10.wav

    # Same but capture diagnostic raw-power instead of pct25-normalized
    python3 tools/measure_banding.py --raw-power tests/fixtures/parity_strong10.wav

    # Plot the time evolution
    python3 tools/measure_banding.py --plot banding.png MSK144/sim/short50.wav

Output (always printed):

  - Per-channel mean over the stable region (last N hops)
  - Time-averaged P-P across channels  (this is the banding amplitude)
  - Per-hop std across channels (statistical noise floor)
  - Banding/noise ratio (>3 = real banding; ~1 = no banding)
  - Time evolution: banding amplitude at t=10s, 30s, 60s, 120s, 300s

The script makes NO assumption about cause.  It measures the
phenomenon.  Variables to manipulate between runs (each gives a
separate measurement):

  - Engine blanker choice          → Engine().set_blanker('Bypass'/'NR0V-Wideband'/'Linrad')
  - Input source                   → AWGN vs WAV
  - Diagnostic raw-power           → --raw-power
  - Run duration                   → AWGN seconds argument
  - With/without #41c              → git revert / restore between runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.channelizer import N_CHANNELS                      # noqa: E402
from map144_app.engine import Engine                                # noqa: E402


@dataclass
class BandingMeasurement:
    """Per-channel detection-metric statistics over the run."""

    label: str                         # e.g. "tone" or "sync"
    n_hops: int                        # how many hops we measured
    per_channel_mean: np.ndarray       # (N_CHANNELS,)
    per_channel_std: np.ndarray        # (N_CHANNELS,)
    per_hop_std_across_ch: float       # median per-hop std across channels
    pp_across_channels: float          # max(mean) - min(mean) — banding amplitude
    band_to_noise_ratio: float         # pp_across_channels / per_hop_std_across_ch
    time_series: np.ndarray            # (n_hops, N_CHANNELS) — full evolution


def _capture(engine: Engine, duration_s: float, source: str,
             wav_path: Path | None, sample_rate: int,
             use_raw_power: bool, window_hops: int,
             ) -> tuple[BandingMeasurement, BandingMeasurement]:
    """Feed IQ through engine.process_iq_data and capture the
    per-channel ``_ch_snr_history_h`` (tone) and
    ``_sync_snr_history_h`` (sync) rolling buffers as time series.
    """
    engine._show_raw_power = use_raw_power

    # Source IQ.
    if source == "synthetic":
        n_total = int(duration_s * sample_rate)
        rng = np.random.default_rng(1234)
        iq = (rng.standard_normal(n_total).astype(np.float32)
              + 1j * rng.standard_normal(n_total).astype(np.float32)
              ) * np.complex64(0.001)
    else:
        from map144_app.runtime import _load_wav_complex  # noqa: PLC0415
        if wav_path is None:
            raise ValueError("wav_path required when source='wav'")
        samples, _ = _load_wav_complex(str(wav_path), sample_rate)
        iq = samples.reshape(-1) if samples.ndim == 2 else samples

    # Capture buffer: collect history snapshots at each chunk.  Since
    # the engine's _ch_snr_history_h is a CIRCULAR buffer of N_SNR_HIST
    # rows, we instead capture per-chunk by reading the row that was
    # just written.  Track via _ch_snr_write_idx.
    tone_hops: list[np.ndarray] = []
    sync_hops: list[np.ndarray] = []
    last_idx = -1

    chunk = 4096
    pos = 0
    while pos < len(iq):
        engine.process_iq_data(iq[pos:pos + chunk], 0, 0)
        pos += chunk
        # The detector may have run multiple hops inside this chunk.
        # Capture every populated row since the last call.
        cur_idx = engine._ch_snr_write_idx
        if cur_idx != last_idx:
            # Capture every NEW row.  Handle wraparound.
            n_hist = engine._ch_snr_history_h.shape[0]
            if cur_idx > last_idx:
                fresh = range(last_idx + 1, cur_idx + 1)
            else:
                fresh = list(range(last_idx + 1, n_hist)) + list(range(0, cur_idx + 1))
            for i in fresh:
                row_h = engine._ch_snr_history_h[i].copy()
                row_s = engine._sync_snr_history_h[i].copy()
                if row_h[0] > -100:   # populated
                    tone_hops.append(row_h)
                    sync_hops.append(row_s)
            last_idx = cur_idx

    tone_ts = np.array(tone_hops)
    sync_ts = np.array(sync_hops)
    return (
        _summarise(tone_ts, "tone", window_hops),
        _summarise(sync_ts, "sync", window_hops),
    )


def _summarise(ts: np.ndarray, label: str, window_hops: int) -> BandingMeasurement:
    """Compute banding stats over the LAST ``window_hops`` of the run
    — this is the CURRENT band state, not an average from start."""
    if len(ts) < 20:
        return BandingMeasurement(
            label=label, n_hops=len(ts),
            per_channel_mean=np.full(N_CHANNELS, np.nan),
            per_channel_std=np.full(N_CHANNELS, np.nan),
            per_hop_std_across_ch=np.nan,
            pp_across_channels=np.nan,
            band_to_noise_ratio=np.nan,
            time_series=ts,
        )
    stable = ts[-window_hops:] if window_hops < len(ts) else ts
    per_ch_mean = stable.mean(axis=0)
    per_ch_std  = stable.std(axis=0)
    # Per-hop std across channels (how much channels vary at a single time)
    per_hop_std = stable.std(axis=1)
    pp_across = float(per_ch_mean.max() - per_ch_mean.min())
    median_per_hop_std = float(np.median(per_hop_std))
    ratio = pp_across / max(median_per_hop_std, 1e-6)
    return BandingMeasurement(
        label=label,
        n_hops=len(ts),
        per_channel_mean=per_ch_mean,
        per_channel_std=per_ch_std,
        per_hop_std_across_ch=median_per_hop_std,
        pp_across_channels=pp_across,
        band_to_noise_ratio=ratio,
        time_series=ts,
    )


def _evolution_over_time(ts: np.ndarray, hops_per_sec: float,
                         checkpoint_times_s: list[float],
                         window_s: float) -> list[tuple[float, float, float, int]]:
    """Banding amplitude in a SLIDING window of ``window_s`` ending
    at each checkpoint time.  Tracks the CURRENT state at each time,
    not the cumulative average from t=0 — so a slowly-growing pattern
    shows up as a rising P-P curve.

    Returns list of (checkpoint_time_s, pp_db, ratio, n_hops_in_window).
    """
    out = []
    window_hops = max(20, int(window_s * hops_per_sec))
    for t in checkpoint_times_s:
        end = int(t * hops_per_sec)
        if end > len(ts):
            end = len(ts)
        start = max(0, end - window_hops)
        if end - start < 20:
            out.append((t, float("nan"), float("nan"), end - start))
            continue
        window = ts[start:end]
        per_ch_mean   = window.mean(axis=0)
        per_hop_std   = window.std(axis=1)
        pp            = float(per_ch_mean.max() - per_ch_mean.min())
        median_h_std  = float(np.median(per_hop_std))
        ratio         = pp / max(median_h_std, 1e-6)
        out.append((t, pp, ratio, end - start))
    return out


def _print_report(m: BandingMeasurement, hops_per_sec: float,
                  window_s: float) -> None:
    total_run_s = m.n_hops / hops_per_sec
    print(f"\n── {m.label.upper()} detection ─────────────────────────")
    print(f"  total run:                                 {total_run_s:.1f} s "
          f"({m.n_hops} hops)")
    print(f"  measurement window (final state):          last {window_s:.0f} s")
    print(f"  per-hop std across channels (noise floor): {m.per_hop_std_across_ch:6.2f} dB")
    print(f"  P-P across channels (banding amplitude):   {m.pp_across_channels:6.2f} dB")
    print(f"  banding / noise ratio:                     {m.band_to_noise_ratio:6.2f}")
    verdict = ("** REAL BANDING **" if m.band_to_noise_ratio > 3.0
               else "no significant banding")
    print(f"  verdict: {verdict}")

    # Time evolution — SLIDING window so a slowly-growing pattern
    # shows up.  Each checkpoint reports banding in the last
    # ``window_s`` of run-time ending at that checkpoint.
    if total_run_s >= 10.0:
        # Generate checkpoints up to the actual run duration.
        candidate = [10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0,
                     420.0, 600.0]
        checkpoints = [t for t in candidate if t <= total_run_s + 1.0]
        if checkpoints and checkpoints[-1] < total_run_s - 5.0:
            checkpoints.append(round(total_run_s, 1))
        evo = _evolution_over_time(m.time_series, hops_per_sec,
                                    checkpoints, window_s=window_s)
        print(f"  evolution (sliding {window_s:.0f}-s window — current band state at each t):")
        print(f"     {'t':>8} {'P-P':>7} {'noise':>6} {'ratio':>6} {'n_hops':>7}")
        for t, pp, ratio, n in evo:
            if np.isnan(pp):
                continue
            # Compute per-hop noise from the same window for ratio context
            print(f"     {t:7.0f}s {pp:6.2f}dB        {ratio:5.2f} {n:7d}")

    # Per-channel mean profile (the band shape)
    print(f"  per-channel mean (signed-channel order):")
    db_min = m.per_channel_mean.min()
    db_max = m.per_channel_mean.max()
    span = max(db_max - db_min, 0.1)
    for ks in range(-N_CHANNELS // 2, N_CHANNELS // 2):
        k = ks if ks >= 0 else ks + N_CHANNELS
        v = m.per_channel_mean[k]
        bar = '█' * max(1, int((v - db_min) / span * 40))
        print(f"     ch {ks:+3d} {v:6.2f} dB  {bar}")


def _plot(m_tone: BandingMeasurement, m_sync: BandingMeasurement,
          out_path: Path, hops_per_sec: float, window_s: float) -> None:
    """Three-panel plot:
      1. Tone heatmap (time × channel) — for visual check vs the GUI
      2. Sync heatmap (time × channel)
      3. Banding amplitude over time (sliding window) — directly shows
         whether bands are growing, stable, or shrinking with time.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available — skipping plot {out_path}",
              file=sys.stderr)
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Heatmaps (top two panels)
    for ax, m in zip(axes[:2], (m_tone, m_sync)):
        ts = m.time_series
        ts_signed = np.fft.fftshift(ts, axes=1)
        vmin = float(np.percentile(ts_signed, 2))
        vmax = float(np.percentile(ts_signed, 98))
        ax.imshow(
            ts_signed.T, aspect="auto", origin="lower",
            vmin=vmin, vmax=vmax,
            extent=[0, ts.shape[0] / hops_per_sec,
                    -N_CHANNELS // 2, N_CHANNELS // 2 - 1],
        )
        ax.set_ylabel(f"{m.label}  ch_signed")
        ax.set_title(
            f"{m.label}: last-{window_s:.0f}s P-P = "
            f"{m.pp_across_channels:.2f} dB, ratio = "
            f"{m.band_to_noise_ratio:.2f}"
        )

    # Banding-amplitude-over-time curve (third panel) — sliding window
    # rolled across the whole run.  This is the test for "do bands grow
    # over time?".  Compute one point per ~5 % of run length.
    ax3 = axes[2]
    for m, colour in [(m_tone, "C0"), (m_sync, "C1")]:
        n_hops = m.n_hops
        if n_hops < 20:
            continue
        window_hops = max(20, int(window_s * hops_per_sec))
        step_hops   = max(20, n_hops // 60)
        times, pps  = [], []
        for end in range(window_hops, n_hops + 1, step_hops):
            start = end - window_hops
            window = m.time_series[start:end]
            per_ch_mean = window.mean(axis=0)
            pps.append(float(per_ch_mean.max() - per_ch_mean.min()))
            times.append(end / hops_per_sec)
        ax3.plot(times, pps, label=m.label, color=colour, linewidth=2)
    ax3.set_xlabel("time within run (s)")
    ax3.set_ylabel("banding amplitude (dB)")
    ax3.set_title(f"Banding amplitude in a sliding {window_s:.0f}-s window — "
                  f"shows growth/stable/decay")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    print(f"  plot saved → {out_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?",
                    help="WAV file to replay (omit with --synthetic)")
    ap.add_argument("--synthetic", type=float, default=None, metavar="SECONDS",
                    help="instead of WAV, generate N seconds of flat AWGN")
    ap.add_argument("--raw-power", action="store_true",
                    help="capture diagnostic raw-power instead of pct25-normalised")
    ap.add_argument("--blanker", default=None,
                    choices=["Linrad", "Bypass", "NR0V-Wideband", "NR0V-Spectral"],
                    help="override blanker backend (default: Engine's LinradBlanker)")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--plot", default=None, metavar="PATH",
                    help="save a heatmap plot of the time series + banding-vs-time curve")
    ap.add_argument("--window", type=float, default=30.0, metavar="SEC",
                    help="sliding-window length for current-state and evolution measurements (default 30 s)")
    args = ap.parse_args()

    if args.synthetic is None and args.wav is None:
        ap.error("either WAV path or --synthetic SECONDS required")

    e = Engine()
    if args.blanker is not None:
        e.set_blanker(args.blanker)

    # Channelizer hop rate (at 12 kHz channel rate / 256-sample stride =
    # ~46.875 hops/sec).  Compute exactly from settings.
    from map144_app.processing import CH_DETECT_SIZE
    from map144_app.channelizer import CH_SAMPLE_RATE
    hops_per_sec = CH_SAMPLE_RATE / (CH_DETECT_SIZE // 2)
    window_hops = max(20, int(args.window * hops_per_sec))

    t0 = time.monotonic()
    if args.synthetic is not None:
        print(f"Source: flat AWGN, {args.synthetic} s @ {args.sample_rate} Hz")
        m_tone, m_sync = _capture(
            e, args.synthetic, "synthetic", None, args.sample_rate,
            args.raw_power, window_hops,
        )
    else:
        print(f"Source: {args.wav}")
        m_tone, m_sync = _capture(
            e, 0.0, "wav", Path(args.wav), args.sample_rate,
            args.raw_power, window_hops,
        )
    elapsed = time.monotonic() - t0
    print(f"Engine: {type(e.blanker).__name__}  "
          f"diagnostic raw_power={args.raw_power}  "
          f"runtime {elapsed:.1f} s")

    _print_report(m_tone, hops_per_sec, args.window)
    _print_report(m_sync, hops_per_sec, args.window)

    if args.plot:
        _plot(m_tone, m_sync, Path(args.plot), hops_per_sec, args.window)

    return 0


if __name__ == "__main__":
    sys.exit(main())
