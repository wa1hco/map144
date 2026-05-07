#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Unsupervised characterization of MAP144 launches by their (channel × time) pattern.

99 % of launches in launches.jsonl don't decode.  We don't have ground-
truth labels for the failure types — but the launches themselves cluster
into a small number of physically-distinct *patterns* in (channel × time)
space.  This script applies rule-based pattern detection and reports the
distribution + decode-rate per pattern.

Patterns considered (rule-based, hierarchy is exclusive, most-specific first):

  WIDEBAND_BURST    ≥ N_WIDE distinct channels co-fire within ±300 ms.
                    Atmospheric / impulse / wideband noise.
  SSB_VOICE         Same channel, sustained rapid-retrigger
                    (≥ N_VOICE_RAPID launches/sec averaged over span).
                    Voice-modulated SSB on a fixed frequency.
                    NOTE: per-channel cooldown limits live SSB triggers
                    to ~2/sec, so this rule rarely fires in practice.
  LOCAL_EDGE        3+ adjacent channels (±3) co-fire within ±100 ms.
                    Strong-local TX rising edge sidelobes.
  STATIONARY_SPUR   Channel fires in many periods (across-period
                    persistence ≥ SPUR_PERIOD_FRAC) at <SPUR_DECODE_FRAC
                    decode rate, AND not on the calling frequency.
                    CW carriers, narrow modulated spurs that fire
                    1–2 times per period, persistently.
  SUSTAINED         Same channel, ≥ N_SUSTAINED launches within a
                    single period, but not voice-rapid.  Continuous TX
                    with cooldown re-triggers.
  ISOLATED          Single launch, no neighbours in time or frequency.
                    Either real-signal-shaped or unexplained noise.

Implementation: vectorized sliding-window counts over a per-channel
multiplicity histogram — O(N) total time, runs in seconds even on
700k launches.

Usage::

    python analyze_launches.py [--since YYYYMMDD] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent
LAUNCHES = REPO / "MSK144" / "detections" / "launches.jsonl"
CALLING_KHZ_DEFAULT = 50260.0


# ── Rule thresholds ──────────────────────────────────────────────────────
WIDE_WINDOW_S          = 0.300
WIDE_N_DISTINCT_CHAN   = 8
LOCAL_EDGE_WINDOW_S    = 0.100
LOCAL_EDGE_N_NEAR_CHAN = 3
SSB_RAPID_PER_SEC      = 5.0
SUSTAINED_N_PER_PERIOD = 3

# Stationary-spur thresholds.  A channel that's present in ≥ this fraction
# of all observed periods *and* decodes in < SPUR_DECODE_FRAC of its launches
# is treated as a stationary spur (NOT applied to the calling channel,
# which may legitimately be busy + low-decode-rate during noisy bands).
SPUR_PERIOD_FRAC       = 0.10
SPUR_DECODE_FRAC       = 0.02
SPUR_MIN_LAUNCHES      = 30      # don't flag spurs from too few samples
CALLING_PROTECT_KHZ    = 2       # don't flag channels within ±2 kHz of calling


# ── Loaders ──────────────────────────────────────────────────────────────


def parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d_%H:%M:%S.%f", "%Y-%m-%d_%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def load_launches(path: Path, since: datetime | None = None) -> dict:
    """Load launches.jsonl into a column-oriented numpy structure.

    Sorted by time.  Returns a dict of equal-length arrays:
        ts (datetime[]), ts_epoch (float64), t_sec, khz (int),
        outcome_decoded (bool), period_key (int = floor(ts/15)*15)
    plus the raw record list for reporting.
    """
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(r.get("timestamp", ""))
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            khz = r.get("radio_khz")
            t_sec = r.get("t_sec")
            if khz is None or t_sec is None:
                continue
            outcome = r.get("outcome", "?")
            rows.append({
                "ts":           ts,
                "ts_epoch":     ts.timestamp(),
                "t_sec":        float(t_sec),
                "khz":          int(round(float(khz))),
                "outcome":      outcome,
                "decoded":      (outcome == "decoded"),
            })
    rows.sort(key=lambda r: r["ts_epoch"])

    n = len(rows)
    return {
        "rows":      rows,
        "ts_epoch":  np.array([r["ts_epoch"] for r in rows], dtype=np.float64),
        "khz":       np.array([r["khz"]      for r in rows], dtype=np.int32),
        "t_sec":     np.array([r["t_sec"]    for r in rows], dtype=np.float32),
        "decoded":   np.array([r["decoded"]  for r in rows], dtype=bool),
        "period_key": np.array(
            [int(r["ts_epoch"] - r["t_sec"]) // 15 * 15 for r in rows],
            dtype=np.int64,
        ),
    }


# ── Vectorized pattern classifier ────────────────────────────────────────


def classify(data: dict, calling_khz: float = CALLING_KHZ_DEFAULT) -> np.ndarray:
    """Return ``labels`` array of dtype object (strings)."""
    n = len(data["ts_epoch"])
    if n == 0:
        return np.array([], dtype=object)

    times = data["ts_epoch"]
    chans = data["khz"] - int(round(calling_khz))     # signed offsets
    chan_min = int(chans.min())
    chan_idx = (chans - chan_min).astype(np.int32)    # 0-based for histogram
    n_chan = int(chan_idx.max()) + 1

    # ── Sliding-window histograms over (chan_idx) per time window ────
    # window_count[c] = number of launches with chan_idx == c currently
    # within [t_i - W, t_i + W].  Maintained via two-pointer slide.

    def sliding_distinct_count(window_s: float) -> np.ndarray:
        """For each launch i, return the number of *distinct* channels
        firing in [t_i - window_s, t_i + window_s]."""
        n_dist = np.zeros(n, dtype=np.int32)
        wc = np.zeros(n_chan, dtype=np.int32)   # multiplicities currently in window
        n_active = 0    # = (wc > 0).sum() maintained incrementally

        lo = 0
        hi = 0
        for i in range(n):
            t = times[i]
            # Expand right edge: add events with times[hi] <= t + window_s
            t_hi_max = t + window_s
            while hi < n and times[hi] <= t_hi_max:
                c = chan_idx[hi]
                if wc[c] == 0:
                    n_active += 1
                wc[c] += 1
                hi += 1
            # Contract left edge: remove events with times[lo] < t - window_s
            t_lo_min = t - window_s
            while lo < n and times[lo] < t_lo_min:
                c = chan_idx[lo]
                wc[c] -= 1
                if wc[c] == 0:
                    n_active -= 1
                lo += 1
            n_dist[i] = n_active
        return n_dist

    # n_distinct_wide: how many channels fired anywhere in the band within ±300 ms
    print("  computing wideband distinct-channel counts…", flush=True)
    n_distinct_wide = sliding_distinct_count(WIDE_WINDOW_S)

    # For LOCAL_EDGE we need: how many distinct channels within ±3 of THIS launch's
    # channel fired within ±LOCAL_EDGE_WINDOW_S.  Compute via sliding-window
    # multiplicity histogram + slice ±3 around each launch's channel.

    print("  computing local-edge counts…", flush=True)
    n_local_distinct = np.zeros(n, dtype=np.int32)
    wc = np.zeros(n_chan, dtype=np.int32)
    lo = 0
    hi = 0
    for i in range(n):
        t = times[i]
        t_hi_max = t + LOCAL_EDGE_WINDOW_S
        while hi < n and times[hi] <= t_hi_max:
            c = chan_idx[hi]
            wc[c] += 1
            hi += 1
        t_lo_min = t - LOCAL_EDGE_WINDOW_S
        while lo < n and times[lo] < t_lo_min:
            c = chan_idx[lo]
            wc[c] -= 1
            lo += 1
        my_c = int(chan_idx[i])
        c0 = max(0, my_c - 3)
        c1 = min(n_chan, my_c + 4)
        n_local_distinct[i] = int(np.count_nonzero(wc[c0:c1]))

    # ── Per-(period, channel) bucket stats for SSB / SUSTAINED ──────
    print("  computing per-period per-channel buckets…", flush=True)
    period_key = data["period_key"]
    bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
    bucket_times: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i in range(n):
        k = (int(period_key[i]), int(chan_idx[i]))
        bucket_counts[k] += 1
        bucket_times[k].append(float(times[i]))

    # Compute average launches/sec on each bucket if span > 0
    bucket_rate: dict[tuple[int, int], float] = {}
    for k, ts in bucket_times.items():
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            bucket_rate[k] = (len(ts) - 1) / span if span > 0 else float("inf")
        else:
            bucket_rate[k] = 0.0

    # ── Stationary-spur detection (cross-period channel persistence) ─
    print("  detecting stationary spurs…", flush=True)
    n_periods_total = len(set(period_key.tolist()))
    # Per-channel: number of distinct periods present + total launches +
    # decoded count.  Use khz (absolute) to keep readable in output.
    chan_periods: dict[int, set] = defaultdict(set)
    chan_launches = Counter()
    chan_decoded = Counter()
    for i in range(n):
        khz = int(data["khz"][i])
        chan_periods[khz].add(int(period_key[i]))
        chan_launches[khz] += 1
        if data["decoded"][i]:
            chan_decoded[khz] += 1
    spur_channels: set[int] = set()
    calling_int = int(round(calling_khz))
    for khz, periods_set in chan_periods.items():
        if abs(khz - calling_int) <= CALLING_PROTECT_KHZ:
            continue
        n_launches_ch = chan_launches[khz]
        if n_launches_ch < SPUR_MIN_LAUNCHES:
            continue
        period_frac = len(periods_set) / max(1, n_periods_total)
        decode_frac = chan_decoded[khz] / max(1, n_launches_ch)
        if period_frac >= SPUR_PERIOD_FRAC and decode_frac < SPUR_DECODE_FRAC:
            spur_channels.add(khz)

    print(f"  identified {len(spur_channels)} stationary-spur channel(s): "
          f"{sorted(spur_channels)}")

    # ── Apply hierarchy ──────────────────────────────────────────────
    print("  applying classification hierarchy…", flush=True)
    labels = np.full(n, "ISOLATED", dtype=object)

    # WIDEBAND_BURST first (most-exclusive)
    is_wideband = n_distinct_wide >= WIDE_N_DISTINCT_CHAN
    labels[is_wideband] = "WIDEBAND_BURST"

    # STATIONARY_SPUR — second tier.  All launches on a spur-flagged
    # channel get this label, regardless of within-period count.
    if spur_channels:
        khz_arr = data["khz"]
        is_spur = np.array([int(k) in spur_channels for k in khz_arr])
        # only relabel ones that aren't already WIDEBAND_BURST
        relabel = is_spur & (labels == "ISOLATED")
        labels[relabel] = "STATIONARY_SPUR"

    # SSB_VOICE / SUSTAINED — bucket-driven, only for still-ISOLATED
    for i in range(n):
        if labels[i] != "ISOLATED":
            continue
        k = (int(period_key[i]), int(chan_idx[i]))
        bn = bucket_counts[k]
        if bn < SUSTAINED_N_PER_PERIOD:
            continue
        rate = bucket_rate[k]
        if rate >= SSB_RAPID_PER_SEC:
            labels[i] = "SSB_VOICE"
        else:
            labels[i] = "SUSTAINED"

    # LOCAL_EDGE — among still-ISOLATED launches, check tight cluster
    # of adjacent channels.  3+ distinct channels within ±3 ch and ±100 ms.
    is_local_edge_candidate = (n_local_distinct >= LOCAL_EDGE_N_NEAR_CHAN)
    still_iso = (labels == "ISOLATED")
    labels[is_local_edge_candidate & still_iso] = "LOCAL_EDGE"

    return labels


# ── Reporting ────────────────────────────────────────────────────────────


def print_summary(data: dict, labels: np.ndarray) -> None:
    n = len(labels)
    label_counts = Counter(labels.tolist())
    decoded_per_label: dict[str, int] = defaultdict(int)
    for lbl, dec in zip(labels, data["decoded"]):
        if dec:
            decoded_per_label[str(lbl)] += 1

    print(f"\nTotal launches: {n:,}")
    print(f"{'pattern':<18} {'launches':>10}  {'%':>6}  {'decoded':>9}  {'decode%':>8}")
    print("-" * 62)
    for pat in ("WIDEBAND_BURST", "SSB_VOICE", "LOCAL_EDGE",
                "STATIONARY_SPUR", "SUSTAINED", "ISOLATED"):
        nn = label_counts.get(pat, 0)
        if nn == 0:
            continue
        d = decoded_per_label.get(pat, 0)
        pct = 100 * nn / n
        dpct = 100 * d / nn if nn else 0
        print(f"{pat:<16} {nn:>10,}  {pct:>5.1f}%  {d:>9,}  {dpct:>7.2f}%")
    n_decoded = int(data["decoded"].sum())
    print(f"\nOverall decode rate: {n_decoded:,} / {n:,} = {100*n_decoded/n:.2f}%")

    # Channel-distribution by pattern (top channels per pattern)
    chans = data["khz"]
    print("\nTop channels by launches, per pattern:")
    for pat in ("WIDEBAND_BURST", "SSB_VOICE", "LOCAL_EDGE",
                "STATIONARY_SPUR", "SUSTAINED", "ISOLATED"):
        mask = labels == pat
        if not mask.any():
            continue
        ch_counts = Counter(chans[mask].tolist())
        top = sorted(ch_counts.items(), key=lambda x: -x[1])[:5]
        top_str = "  ".join(f"{c}kHz:{n}" for c, n in top)
        print(f"  {pat:<16} {top_str}")


def plot_summary(
    data: dict, labels: np.ndarray, out_path: Path,
    calling_khz: float = CALLING_KHZ_DEFAULT,
) -> None:
    n = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    pat_order = ["WIDEBAND_BURST", "SSB_VOICE", "LOCAL_EDGE",
                 "STATIONARY_SPUR", "SUSTAINED", "ISOLATED"]
    pat_color = {
        "WIDEBAND_BURST":  "#d62728",
        "SSB_VOICE":       "#ff7f0e",
        "LOCAL_EDGE":      "#9467bd",
        "STATIONARY_SPUR": "#8c564b",
        "SUSTAINED":       "#2ca02c",
        "ISOLATED":        "#1f77b4",
    }

    label_counts = Counter(labels.tolist())
    decoded_per_label: dict[str, int] = defaultdict(int)
    for lbl, dec in zip(labels, data["decoded"]):
        if dec:
            decoded_per_label[str(lbl)] += 1

    # 0,0: launches per pattern (log-scale bar)
    ax = axes[0, 0]
    counts = [label_counts.get(p, 0) for p in pat_order]
    ax.bar(pat_order, counts, color=[pat_color[p] for p in pat_order])
    ax.set_yscale("log")
    ax.set_ylabel("launches (log)")
    ax.set_title(f"Launches by pattern (n={n:,})")
    ax.tick_params(axis="x", rotation=20)
    for p, c in zip(pat_order, counts):
        if c:
            ax.text(p, c, f"{c:,}", ha="center", va="bottom", fontsize=8)

    # 0,1: decode rate per pattern
    ax = axes[0, 1]
    rates = [100 * decoded_per_label.get(p, 0) / max(1, label_counts.get(p, 0))
             for p in pat_order]
    ax.bar(pat_order, rates, color=[pat_color[p] for p in pat_order])
    ax.set_ylabel("decode rate (%)")
    ax.set_title("P(decoded) per pattern")
    ax.tick_params(axis="x", rotation=20)
    for p, r, c in zip(pat_order, rates, counts):
        if c:
            ax.text(p, r, f"{r:.2f}%", ha="center", va="bottom", fontsize=8)

    # 1,0: channel distribution
    ax = axes[1, 0]
    chans_off = data["khz"] - int(round(calling_khz))
    bins = np.arange(chans_off.min() - 0.5, chans_off.max() + 1.5, 1)
    bottom = np.zeros(len(bins) - 1)
    for p in pat_order:
        mask = labels == p
        if not mask.any():
            continue
        h, _ = np.histogram(chans_off[mask], bins=bins)
        ax.bar(bins[:-1], h, width=1.0, bottom=bottom,
               color=pat_color[p], label=p, align="edge")
        bottom += h
    ax.set_xlabel(f"channel offset (kHz from {int(calling_khz)})")
    ax.set_ylabel("launches")
    ax.set_yscale("log")
    ax.set_title("Frequency distribution by pattern")
    ax.legend(fontsize=8, loc="upper right")

    # 1,1: t_sec distribution
    ax = axes[1, 1]
    bins = np.linspace(0, 15, 31)
    bottom = np.zeros(len(bins) - 1)
    for p in pat_order:
        mask = labels == p
        if not mask.any():
            continue
        h, _ = np.histogram(data["t_sec"][mask], bins=bins)
        ax.bar(bins[:-1], h, width=0.5, bottom=bottom,
               color=pat_color[p], label=p, align="edge")
        bottom += h
    ax.set_xlabel("t_sec within 15-s period")
    ax.set_ylabel("launches")
    ax.set_yscale("log")
    ax.set_title("Time-within-period by pattern")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None,
                    help="Restrict analysis to launches after YYYYMMDD (UTC)")
    ap.add_argument("--launches", default=str(LAUNCHES))
    ap.add_argument("--calling-khz", type=float, default=CALLING_KHZ_DEFAULT)
    ap.add_argument("--output", default=str(REPO / "docs" / "launch_patterns.png"))
    args = ap.parse_args()

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y%m%d").replace(tzinfo=timezone.utc)

    print(f"Loading {args.launches}…", flush=True)
    data = load_launches(Path(args.launches), since=since)
    rows = data["rows"]
    if not rows:
        print("No launches.")
        return 1
    print(f"  {len(rows):,} launches loaded "
          f"({rows[0]['ts'].date()} → {rows[-1]['ts'].date()})")

    print("Classifying…", flush=True)
    labels = classify(data, calling_khz=args.calling_khz)
    print_summary(data, labels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_summary(data, labels, Path(args.output),
                 calling_khz=args.calling_khz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
