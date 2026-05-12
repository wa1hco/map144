#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Compare temporal features (rise time, duty cycle, fading rate) across
capture WAVs grouped by classification.

Goal — validate or refute the assumption that range-band classification
correctly separates MS pings from tropo-ducting / forward-scatter /
sporadic-E.  Range alone is ambiguous: ducting and forward-scatter can
deliver continuous-but-fading signals at MS-zone distances (300–2200
km), masquerading as frequent pings to a range-only classifier.

Three temporal features are extracted per WAV from the squared-tone-pair
detection metric (the same algorithm WSJT-X's ``msk144spd.f90`` and
``analyze_msk144.py`` use).  Each feature has a different sensitivity:

  rise_time_ms       — 10 % → 90 % SNR transition leading the peak.
                        MS pings: tens to hundreds of ms.
                        Tropo / forward / Es: typically seconds.
  duty_cycle         — fraction of frames above 3 dB above noise.
                        Pings: low (single brief event in the window).
                        Sustained modes: high (signal present most of
                        the window).
  fading_peak_hz     — frequency of the dominant peak in the SNR(t)
                        envelope FFT, in the 0.05–2 Hz band.
                        Tropo / forward scatter: ~0.1–1 Hz.
                        Sporadic-E: << 0.1 Hz (slow drift).
                        MS pings: no characteristic rate; the spectrum
                        is dominated by the impulse, not a rhythm.
                        Only computed on windows ≥ 4 s.

WAVs come from ``MSK144/detections/`` (MAP144's per-launch captures,
1.7 s each) by default; passing ``--wsjtx-save-dir`` adds the WSJT-X
SaveAll dir (15 s WAVs — long enough for fading-rate analysis).

Usage::

    python analyze_rise_time.py \\
        --true-ping  K2IL,K9NW,N4JP,KA6U,W4UCK \\
        --weak-fading WA2FZW \\
        --strong-local N1KWF,KB1MGI \\
        --max-wavs-per-call 6 \\
        --output docs/figures/rise_time_compare.png
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / batch
import matplotlib.pyplot as plt
import numpy as np

# Reuse compare_decoders' parsers + sender attribution + capture lookup so
# the WAV-to-callsign mapping stays consistent with the rest of the tools.
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Stash argv to avoid compare_decoders' argparse seeing our flags during import.
_OUR_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from compare_decoders import (  # noqa: E402
    MAP144_LOG, WSJTX_ALL,
    parse_map144, parse_wsjtx, is_test_message,
    wavs_for_sender,
)
sys.argv = _OUR_ARGV


# ── DSP constants — match analyze_msk144 / msk144_spd.py ─────────────
NSPM_REF        = 864      # frame length at the 12 kHz reference rate (72 ms)
STEP_DIV        = 4        # 75 % overlap
FC_HZ_DEFAULT   = 1500.0   # MSK144 audio carrier (the 1500 Hz centre after dial mix)
SQUARED_TONE_HZ = 1000.0   # MSK144 ±500 Hz tones squared land at ±1000 Hz
TONE_TOL_HZ     = 200.0    # ±tolerance window around each squared tone
DETECT_THRESH_DB = 3.0     # frame counted as "signal present" above this


# ── WAV I/O ──────────────────────────────────────────────────────────


def load_wav_real(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV → (samples_real, rate).  Mono real audio expected
    (MAP144 captures and WSJT-X 12 kHz save files are both this shape).
    """
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if ch >= 2:
        data = data.reshape(-1, ch)[:, 0]
    return data, rate


# ── SNR envelope ─────────────────────────────────────────────────────


def real_to_analytic(samples: np.ndarray) -> np.ndarray:
    """FFT-based Hilbert transform — real → analytic complex.

    Squaring a real signal produces mirror-image tone pairs at ±fc,
    swamping the negative-frequency squared spectrum.  Converting to
    analytic form first eliminates this and matches the convention in
    msk144spd.f90 / analyze_msk144.
    """
    n = len(samples)
    spec = np.fft.fft(samples.astype(np.float64))
    spec[1 : n // 2] *= 2.0
    spec[n // 2 + 1 :] = 0.0
    return np.fft.ifft(spec).astype(np.complex64)


def compute_snr_envelope(
    samples_real: np.ndarray,
    rate: int,
    fc_hz: float = FC_HZ_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame MSK144 detection metric, dB above 25th-percentile baseline.

    Algorithm (mirrors ``analyze_msk144._compute_squared_spectrogram`` /
    ``msk144_spd.py``):

    1. Convert real audio to analytic.
    2. Mix carrier ``fc_hz`` to DC.
    3. Square — places MSK144 ±500 Hz tones at ±1000 Hz.
    4. Sliding-window FFT (NSPM frames at NSPM/4 step ≈ 18 ms at 12 kHz).
    5. Per frame: peak power in the ±SQUARED_TONE_HZ ± TONE_TOL_HZ
       windows (lo and hi separately), average them.
    6. Normalise by the 25th percentile of the per-frame metric.
    7. Convert to dB.

    Returns ``(frame_times_s, snr_db)``.
    """
    analytic = real_to_analytic(samples_real)
    n = len(analytic)
    if n < 16:
        return np.array([0.0]), np.array([0.0])

    # Mix to DC.
    t_idx = np.arange(n)
    mix = np.exp(-2j * np.pi * fc_hz * t_idx / rate).astype(np.complex64)
    mixed = analytic * mix

    # Squared signal (MSK144 tones now at ±1000 Hz around DC).
    sq = (mixed ** 2).astype(np.complex64)

    nspm = max(64, int(round(NSPM_REF * rate / 12000)))
    step = max(8, nspm // STEP_DIV)
    n_frames = max(1, (n - nspm) // step + 1)
    if n_frames < 2:
        return np.array([0.0]), np.array([0.0])

    win = np.hanning(nspm).astype(np.float32)
    freqs = np.fft.fftfreq(nspm, 1.0 / rate)
    lo_mask = (freqs >= -SQUARED_TONE_HZ - TONE_TOL_HZ) & (freqs <= -SQUARED_TONE_HZ + TONE_TOL_HZ)
    hi_mask = (freqs >=  SQUARED_TONE_HZ - TONE_TOL_HZ) & (freqs <=  SQUARED_TONE_HZ + TONE_TOL_HZ)

    metric = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * step
        chunk = sq[start : start + nspm] * win
        spec = np.abs(np.fft.fft(chunk)) / nspm
        lo_pk = float(spec[lo_mask].max()) if lo_mask.any() else 1e-30
        hi_pk = float(spec[hi_mask].max()) if hi_mask.any() else 1e-30
        metric[i] = 0.5 * (lo_pk + hi_pk)

    pct25 = max(1e-30, float(np.percentile(metric, 25)))
    snr_db = 10.0 * np.log10(np.maximum(metric / pct25, 1e-30))
    snr_db = snr_db.astype(np.float32)

    frame_times = (np.arange(n_frames) * step / rate).astype(np.float32)
    return frame_times, snr_db


# ── Feature extraction ───────────────────────────────────────────────


def compute_features(
    times_s: np.ndarray,
    snr_db: np.ndarray,
    fading_min_window_s: float = 4.0,
    edge_window_s: float = 0.2,
) -> dict:
    """Extract temporal features.  NaN where a feature isn't well-defined
    (e.g. fading-rate from a < 4 s window, or rise-time from a clip
    with no above-threshold peak).

    ``peak_to_start_db`` and ``peak_to_end_db`` measure how much the
    SNR has dropped at the window edges relative to the peak, averaged
    over ``edge_window_s`` seconds.  These distinguish a sustained
    keyed signal (small peak-to-end gap — still keyed at window end)
    from a real ping (large peak-to-end gap — signal returned to
    baseline).  Critical when rise_time alone is misleading: a station
    that keys up mid-period has a fast rise too, but its peak_to_end
    is small.
    """
    out: dict = {
        "duration_s":            float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
        "n_frames":              int(len(snr_db)),
        "peak_db":               float(snr_db.max()) if len(snr_db) else 0.0,
        "rise_time_ms":          float("nan"),
        "fall_time_ms":          float("nan"),
        "duty_cycle":            float("nan"),
        "longest_run_ms":        float("nan"),
        "peak_to_start_db":      float("nan"),
        "peak_to_end_db":        float("nan"),
        "fading_peak_hz":        float("nan"),
        "fading_band_power_frac": float("nan"),
    }

    if len(snr_db) < 4 or float(snr_db.max()) < DETECT_THRESH_DB:
        # No signal or too short to characterise.
        return out

    peak = float(snr_db.max())
    peak_idx = int(np.argmax(snr_db))
    p10_thr = peak * 0.10
    p90_thr = peak * 0.90

    # ── Rise time (10% → 90%, scanning back from the peak) ───────────
    # Walk backward from the peak; first index ≤ p10 is the rise start.
    i_p10_rise = 0
    for j in range(peak_idx, -1, -1):
        if snr_db[j] <= p10_thr:
            i_p10_rise = j
            break
    # Walk forward from the rise start; first index ≥ p90 is the rise end.
    i_p90_rise = peak_idx
    for j in range(i_p10_rise, peak_idx + 1):
        if snr_db[j] >= p90_thr:
            i_p90_rise = j
            break
    out["rise_time_ms"] = float((times_s[i_p90_rise] - times_s[i_p10_rise]) * 1000)

    # ── Fall time (90% → 10%, scanning forward from peak) ───────────
    i_p90_fall = peak_idx
    for j in range(peak_idx, len(snr_db)):
        if snr_db[j] >= p90_thr:
            i_p90_fall = j
        else:
            break
    i_p10_fall = len(snr_db) - 1
    for j in range(i_p90_fall, len(snr_db)):
        if snr_db[j] <= p10_thr:
            i_p10_fall = j
            break
    out["fall_time_ms"] = float((times_s[i_p10_fall] - times_s[i_p90_fall]) * 1000)

    # ── Duty cycle (fraction of frames above threshold) ──────────────
    above = snr_db > DETECT_THRESH_DB
    out["duty_cycle"] = float(above.sum() / len(snr_db))

    # ── Longest contiguous run above threshold (ms) ─────────────────
    # A ping shows ONE short run; a sustained signal shows ONE long run
    # spanning most of the window.  duty_cycle alone treats fragmented
    # bursts and a single sustained burst the same; longest_run pulls
    # them apart.
    if above.any():
        runs = []
        cur = 0
        for v in above:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        max_run_frames = max(runs) if runs else 0
        # Frame step in seconds
        dt_s = float(times_s[1] - times_s[0]) if len(times_s) > 1 else 0.0
        out["longest_run_ms"] = float(max_run_frames * dt_s * 1000)
    else:
        out["longest_run_ms"] = 0.0

    # ── Peak-to-edge differences ─────────────────────────────────────
    # Sample the leading / trailing ``edge_window_s`` seconds and report
    # the mean SNR there relative to the peak.  Local keying mid-period
    # → small peak_to_end (signal still strong at end).  Real ping with
    # full decay → large peak_to_end (signal gone by end).
    dt_s = float(times_s[1] - times_s[0]) if len(times_s) > 1 else 0.0
    if dt_s > 0:
        edge_n = max(1, int(round(edge_window_s / dt_s)))
        edge_n = min(edge_n, len(snr_db) // 4 or 1)
        start_mean = float(snr_db[:edge_n].mean())
        end_mean = float(snr_db[-edge_n:].mean())
        out["peak_to_start_db"] = peak - start_mean
        out["peak_to_end_db"] = peak - end_mean

    # ── Fading rate (FFT of the SNR envelope; only if window long enough) ─
    if out["duration_s"] >= fading_min_window_s and len(snr_db) >= 32:
        env = snr_db - float(snr_db.mean())
        spec = np.abs(np.fft.rfft(env))
        # frame rate (Hz) for the envelope sequence
        dt = float(times_s[1] - times_s[0]) if len(times_s) > 1 else 1.0
        f_axis = np.fft.rfftfreq(len(snr_db), d=dt)
        # Peak in the 0.05–2 Hz band — the relevant tropo / forward / Es
        # range.  Below 0.05 Hz is the slow envelope drift; above 2 Hz
        # starts to overlap MSK144 frame structure.
        band = (f_axis >= 0.05) & (f_axis <= 2.0)
        if band.any():
            band_spec = spec[band]
            band_freqs = f_axis[band]
            out["fading_peak_hz"] = float(band_freqs[int(np.argmax(band_spec))])
            tropo_band = (f_axis >= 0.1) & (f_axis <= 1.0)
            if tropo_band.any():
                tropo_pwr = float((spec[tropo_band] ** 2).sum())
                total_pwr = float((spec ** 2).sum()) + 1e-30
                out["fading_band_power_frac"] = tropo_pwr / total_pwr
    return out


# ── WAV gathering ────────────────────────────────────────────────────


def _load_decodes() -> list[dict]:
    ws = parse_wsjtx(WSJTX_ALL)
    mp = parse_map144(MAP144_LOG)
    mp = [d for d in mp if not is_test_message(d["message"])]
    return mp + ws


def find_wsjtx_wavs_for_sender(
    wsjtx_save_dir: Path,
    sender: str,
    decodes_ws: list[dict],
    max_count: int,
) -> list[tuple[dict, Path]]:
    """Find WSJT-X "save decoded" 15-s WAVs for ``sender``.

    WSJT-X filenames are ``YYMMDD_HHMMSS.wav`` (no callsign info), so we
    cross-reference against the parsed WSJT-X decode list: a decode at
    timestamp T whose sender matches the target gets paired with the
    WAV file at T (if it exists).
    """
    if not wsjtx_save_dir.is_dir():
        return []
    out: list[tuple[dict, Path]] = []
    # Re-import sender_in_message lazily to avoid circular dep at module load.
    from compare_decoders import sender_in_message  # noqa: PLC0415
    for d in decodes_ws:
        if d.get("source") != "wsjtx":
            continue
        if sender_in_message(d["message"]) != sender:
            continue
        # WSJT-X SaveAll filename is the period timestamp at second
        # precision (no fractional).  d['ts'] is already period-floored
        # in parse_wsjtx, so format directly.
        ts = d["ts"]
        fname = ts.strftime("%y%m%d_%H%M%S") + ".wav"
        wav = wsjtx_save_dir / fname
        if wav.is_file():
            out.append((d, wav))
    out.sort(key=lambda x: x[0]["ts"], reverse=True)
    return out[:max_count]


# ── Plotting ─────────────────────────────────────────────────────────


_CLASS_COLORS = {
    "TRUE_PING":    "#1f77b4",
    "WEAK_FADING":  "#2ca02c",
    "STRONG_LOCAL": "#d62728",
    "CONT_PROP":    "#ff7f0e",
}


def plot_results(results_by_class: dict, output_path: Path) -> None:
    classes = [c for c in ("TRUE_PING", "WEAK_FADING", "CONT_PROP", "STRONG_LOCAL")
               if results_by_class.get(c)]
    if not classes:
        print("No data — nothing to plot.")
        return

    n_classes = len(classes)
    max_examples = max(len(results_by_class[c]) for c in classes)
    max_examples = min(max_examples, 6)   # cap visual grid width

    # Layout: top n_classes rows of SNR(t) traces, bottom row of histograms (3 cols).
    rows = n_classes + 1
    cols = max(max_examples, 3)
    fig = plt.figure(figsize=(2.8 * cols, 2.0 * n_classes + 3.5))

    # SNR(t) traces — one row per class.
    for ci, cls in enumerate(classes):
        items = results_by_class[cls][:max_examples]
        for xi in range(max_examples):
            ax = fig.add_subplot(rows, max_examples, ci * max_examples + xi + 1)
            if xi < len(items):
                item = items[xi]
                ax.plot(item["times_s"], item["snr_db"],
                        color=_CLASS_COLORS.get(cls, "k"), lw=0.8)
                ax.axhline(DETECT_THRESH_DB, color="red", ls=":", lw=0.5)
                ax.set_ylim(-3, max(15, float(item["snr_db"].max()) + 3))
                ax.set_xlim(0, max(0.1, float(item["times_s"].max())))
                short = item["filename"]
                if len(short) > 28:
                    short = short[:28]
                rt = item["features"]["rise_time_ms"]
                rt_str = f"{rt:.0f} ms" if rt == rt else "—"  # NaN check
                dc = item["features"]["duty_cycle"]
                dc_str = f"{dc:.2f}" if dc == dc else "—"
                ax.set_title(
                    f"{item['callsign']}  rise={rt_str}  duty={dc_str}",
                    fontsize=7,
                )
                ax.tick_params(labelsize=6)
                if xi == 0:
                    ax.set_ylabel(f"{cls}\nSNR (dB)", fontsize=8)
            else:
                ax.set_visible(False)

    # Feature histograms (bottom row, 4 panels).  The four shown together
    # disambiguate the failure mode where rise_time alone misclassifies
    # a station that keys mid-period as a ping: fall_time and
    # peak_to_end_dB stay LOW for the keying station (signal sustained)
    # but stay HIGH for a real ping that decayed before window end.
    feature_specs = [
        ("rise_time_ms",     "rise time (ms)",      [0, 1000]),
        ("fall_time_ms",     "fall time (ms)",      [0, 1500]),
        ("longest_run_ms",   "longest run (ms)",    None),       # auto x
        ("peak_to_end_db",   "peak − end (dB)",     [0, 25]),
    ]
    for fi, (fname, label, xlim) in enumerate(feature_specs):
        ax = fig.add_subplot(rows, len(feature_specs), n_classes * len(feature_specs) + fi + 1)
        for cls in classes:
            vals = [r["features"][fname] for r in results_by_class[cls]
                    if not np.isnan(r["features"][fname])]
            if vals:
                ax.hist(
                    vals, bins=12,
                    range=xlim if isinstance(xlim, list) else None,
                    alpha=0.55,
                    color=_CLASS_COLORS.get(cls, "k"),
                    label=f"{cls} (n={len(vals)})",
                )
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        if isinstance(xlim, list):
            ax.set_xlim(xlim)

    fig.suptitle(
        "Temporal-feature comparison across classifications "
        "(SNR(t) traces top; per-feature histograms bottom)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    print(f"Wrote {output_path}")
    plt.close(fig)


# ── Summary ──────────────────────────────────────────────────────────


def print_summary(results_by_class: dict) -> None:
    """Print medians per class for every feature, with column groups
    that distinguish similar-rise but different-after-peak signals.
    """
    print()
    print(
        f"{'class':<14} {'n':>3} | "
        f"{'rise_ms':>8} {'fall_ms':>8} | "
        f"{'duty':>5} {'longrun_ms':>10} | "
        f"{'p2start_dB':>11} {'p2end_dB':>9} | "
        f"{'fade_Hz':>8}"
    )
    print("-" * 96)
    for cls, results in results_by_class.items():
        if not results:
            continue

        def med(key):
            vals = [r["features"][key] for r in results
                    if r["features"].get(key) is not None
                    and r["features"].get(key) == r["features"].get(key)]   # NaN-safe
            return np.median(vals) if vals else float("nan")

        n = len(results)
        rise = med("rise_time_ms")
        fall = med("fall_time_ms")
        duty = med("duty_cycle")
        runl = med("longest_run_ms")
        p2s = med("peak_to_start_db")
        p2e = med("peak_to_end_db")
        fade = med("fading_peak_hz")

        def fmt(x, w):
            return f"{x:>{w}.0f}" if (x == x and not np.isinf(x)) else " " * (w - 1) + "—"

        def fmt_f(x, w, dec):
            return f"{x:>{w}.{dec}f}" if (x == x and not np.isinf(x)) else " " * (w - 1) + "—"

        print(
            f"{cls:<14} {n:>3} | "
            f"{fmt(rise, 8)} {fmt(fall, 8)} | "
            f"{fmt_f(duty, 5, 2)} {fmt(runl, 10)} | "
            f"{fmt_f(p2s, 11, 1)} {fmt_f(p2e, 9, 1)} | "
            f"{fmt_f(fade, 8, 2)}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--true-ping", default="",
                    help="Comma-separated TRUE_PING candidate callsigns")
    ap.add_argument("--weak-fading", default="",
                    help="Comma-separated WEAK_FADING candidate callsigns")
    ap.add_argument("--strong-local", default="",
                    help="Comma-separated STRONG_LOCAL candidate callsigns")
    ap.add_argument("--cont-prop", default="",
                    help="Comma-separated CONT_PROP candidate callsigns")
    ap.add_argument("--max-wavs-per-call", type=int, default=4,
                    help="Per-callsign cap on number of WAVs analysed (default 4)")
    ap.add_argument("--captures-dir",
                    default=str(_REPO / "MSK144" / "detections"),
                    help="MAP144 captures directory (1.7-s WAVs)")
    ap.add_argument("--wsjtx-save-dir",
                    default=str(Path.home() / ".local" / "share"
                                / "WSJT-X - Local" / "save"),
                    help="WSJT-X SaveAll/SaveDecoded directory (15-s WAVs).  "
                         "Pass empty string to disable.")
    ap.add_argument("--output", default=str(_REPO / "docs" / "figures" / "rise_time_compare.png"))
    ap.add_argument("--summary-only", action="store_true",
                    help="Print summary table only; skip plot")
    args = ap.parse_args()

    classes_args = {
        "TRUE_PING":    args.true_ping,
        "WEAK_FADING":  args.weak_fading,
        "CONT_PROP":    args.cont_prop,
        "STRONG_LOCAL": args.strong_local,
    }
    classes_calls = {
        k: [s.strip() for s in v.split(",") if s.strip()]
        for k, v in classes_args.items()
    }
    if not any(classes_calls.values()):
        ap.error(
            "Provide at least one of --true-ping / --weak-fading / "
            "--cont-prop / --strong-local"
        )

    print("Loading decodes (MAP144 + WSJT-X)…", flush=True)
    decodes = _load_decodes()
    print(f"  {len(decodes)} decodes parsed")

    captures_dir = Path(args.captures_dir)
    wsjtx_save_dir = Path(args.wsjtx_save_dir) if args.wsjtx_save_dir else None
    if wsjtx_save_dir and wsjtx_save_dir.is_dir():
        n_wsjtx = sum(1 for _ in wsjtx_save_dir.glob("*.wav"))
        print(f"  WSJT-X save dir: {wsjtx_save_dir}  ({n_wsjtx} WAVs)")
    else:
        print(f"  WSJT-X save dir: not found / disabled")
        wsjtx_save_dir = None

    # Pre-filter the WSJT-X-only decodes for SaveAll lookup.
    ws_decodes = [d for d in decodes if d.get("source") == "wsjtx"]

    results_by_class: dict[str, list[dict]] = {}
    for cls, callsigns in classes_calls.items():
        if not callsigns:
            continue
        bucket: list[dict] = []
        for cs in callsigns:
            # MAP144 captures (1.7 s) — always available
            map_wavs = wavs_for_sender(cs, decodes, captures_dir)[: args.max_wavs_per_call]
            # WSJT-X SaveAll (15 s) — if dir present
            ws_wavs: list = []
            if wsjtx_save_dir is not None:
                ws_wavs = find_wsjtx_wavs_for_sender(
                    wsjtx_save_dir, cs, ws_decodes, args.max_wavs_per_call,
                )
            wavs = map_wavs + ws_wavs
            if not wavs:
                print(f"  {cls:<14} {cs:<10} 0 WAVs — skipped")
                continue
            print(f"  {cls:<14} {cs:<10} "
                  f"{len(map_wavs)} MAP + {len(ws_wavs)} WSJT-X = {len(wavs)} WAVs")
            for d, wav in wavs:
                try:
                    samples, rate = load_wav_real(wav)
                    times, snr = compute_snr_envelope(samples, rate)
                    features = compute_features(times, snr)
                    bucket.append({
                        "callsign": cs,
                        "filename": wav.name,
                        "decode":   d,
                        "times_s":  times,
                        "snr_db":   snr,
                        "features": features,
                    })
                except Exception as exc:
                    print(f"    ! failed on {wav.name}: {exc}")
        results_by_class[cls] = bucket

    print_summary(results_by_class)
    if not args.summary_only:
        plot_results(results_by_class, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
