"""
build_signal_db.py — ingest all MSK144 signal sources into scratch/signals.db.

Phase 1 (fast, default): parses JSON + ALL.TXT → metadata only, no DSP.
Phase 2 (slow, --precompute): runs SPD + WAV analysis tools → computed metrics
                              + STFT cache files in scratch/signal_cache/.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from tools.signal_db import open_db, set_tag

# Reuse helpers from long_ping_gallery
from tools.long_ping_gallery import (
    grid_from_message,
    calls_from_message,
    maidenhead_to_latlon,
    great_circle_km,
    period_wav,
    load_wav,
    to_baseband,
    find_min_navg,
    FC_JT9,
    SAVE_DIR as WSJTX_FLEX_SAVE,
    ALL_TXT as WSJTX_FLEX_ALL_TXT,
    MY_GRID,
    MY_CALL,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

DETECTIONS_DIR  = _ROOT / "MSK144" / "detections"
DECODES_JSONL   = DETECTIONS_DIR / "decodes.jsonl"
LAUNCHES_JSONL  = DETECTIONS_DIR / "launches.jsonl"
QRZ_CACHE       = DETECTIONS_DIR / "qrz_grid_cache.json"

WSJTX_IC7300_ALL_TXT = Path("/home/jeff/.local/share/WSJT-X/ALL.TXT")
WSJTX_IC7300_SAVE    = Path("/home/jeff/.local/share/WSJT-X/save")

SCRATCH_DIR   = _ROOT / "scratch"
DB_PATH       = SCRATCH_DIR / "signals.db"
CACHE_DIR     = SCRATCH_DIR / "signal_cache"

FC_JT9_HZ = 1500.0  # MSK144 calling frequency audio offset

# ── JT9-line parser ───────────────────────────────────────────────────────────

_JT9_RE = re.compile(
    r"^\s*\S+\s+"          # first token (navmask / time offset)
    r"([+-]?\d+)\s+"       # SNR dB
    r"([+-]?[\d.]+)\s+"    # DT s
    r"([\d.]+)\s+"         # audio Hz
    r"[&@]\s+"             # separator
    r"(.+)$"               # message
)

def _parse_jt9_line(line: str) -> dict | None:
    m = _JT9_RE.match(line.strip())
    if not m:
        return None
    return {
        "snr_db":   int(m.group(1)),
        "dt_s":     float(m.group(2)),
        "audio_hz": float(m.group(3)),
        "message":  m.group(4).strip(),
    }


# ── QRZ grid cache (read-only at ingest) ─────────────────────────────────────

def _load_qrz_cache() -> dict[str, str]:
    if not QRZ_CACHE.exists():
        return {}
    try:
        raw = json.loads(QRZ_CACHE.read_text())
        return {k.upper(): v.get("grid", "")[:4].upper()
                for k, v in raw.items() if isinstance(v, dict) and v.get("grid")}
    except Exception:
        return {}


def _resolve_grid(message: str, qrz: dict[str, str],
                  my_call: str = MY_CALL) -> tuple[str | None, str]:
    grid = grid_from_message(message)
    if grid:
        return grid, "msg"
    for call in calls_from_message(message):
        if call.upper() == my_call.upper():
            continue
        g = qrz.get(call.upper())
        if g:
            return g, f"qrz:{call}"
    return None, ""


# ── MAP144 WAV file matching ──────────────────────────────────────────────────

def _map144_wav(timestamp: str, radio_khz: int, message: str) -> Path | None:
    """Reconstruct MAP144 WAV filename from decode record.

    Example: "2026-04-02_22:50:53.0", 50246, "CQ K8TS EN82"
          → MSK144/detections/20260402_225053Z_50246kHz_CQ_K8TS_EN82.wav
    """
    # timestamp: "2026-04-02_22:50:53.0" or "2026-04-02_22:50:53"
    ts = timestamp.replace("-", "").replace(":", "").replace(".", "")
    # ts is now like "20260402_225053" (the fractional second may still be there)
    ts = ts.split("_")
    if len(ts) < 2:
        return None
    date_part = ts[0][:8]
    time_part = ts[1][:6]
    msg_part = message.replace(" ", "_")
    fname = f"{date_part}_{time_part}Z_{radio_khz}kHz_{msg_part}.wav"
    p = DETECTIONS_DIR / fname
    return p if p.exists() else None


# ── Timestamp → epoch_s ───────────────────────────────────────────────────────

def _ts_to_epoch(timestamp: str) -> float:
    """Convert various timestamp formats to UTC epoch seconds."""
    # MAP144 format: "2026-04-02_22:50:53.0"
    for fmt in ("%Y-%m-%d_%H:%M:%S.%f", "%Y-%m-%d_%H:%M:%S",
                "%y%m%d_%H%M%S", "%y%m%d_%H%M%S.%f"):
        try:
            dt = datetime.strptime(timestamp.split(".")[0], fmt.split(".")[0])
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def _ts_to_iso(timestamp: str) -> str:
    """Normalise any timestamp string to 'YYYY-MM-DD HH:MM:SS' for storage."""
    # MAP144: "2026-04-02_22:50:53.0"  → normalize to ISO
    # ALL.TXT: "260507_124700" (YYMMDD_HHMMSS)
    ts = timestamp.strip()
    for fmt, out_fmt in [
        ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S"),
        ("%y%m%d_%H%M%S",     "%Y-%m-%d %H:%M:%S"),
    ]:
        try:
            base = ts.split(".")[0]
            dt = datetime.strptime(base, fmt)
            return dt.strftime(out_fmt)
        except ValueError:
            continue
    return ts  # fallback: store as-is


def _wav_fingerprint(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(
        f"{path}:{st.st_mtime}:{st.st_size}".encode()
    ).hexdigest()[:16]


# ── Launches.jsonl index (timestamp + radio_khz → theta_deg) ─────────────────

def _build_launch_index(path: Path) -> dict[tuple[str, int], float | None]:
    idx: dict[tuple[str, int], float | None] = {}
    if not path.exists():
        return idx
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("timestamp", "")
            khz = rec.get("radio_khz")
            theta = rec.get("theta_deg")
            if ts and khz is not None:
                idx[(ts, int(khz))] = theta
    return idx


# ── Distance helper ───────────────────────────────────────────────────────────

def _dist_km(grid: str | None, my_lat: float, my_lon: float) -> float | None:
    if not grid:
        return None
    try:
        r_lat, r_lon = maidenhead_to_latlon(grid)
        return great_circle_km(my_lat, my_lon, r_lat, r_lon)
    except Exception:
        return None


# ── Ingest MAP144 decodes.jsonl ───────────────────────────────────────────────

def _ingest_map144(qrz: dict[str, str],
                   my_lat: float, my_lon: float,
                   launch_idx: dict) -> list[dict]:
    records: list[dict] = []
    if not DECODES_JSONL.exists():
        print(f"  [warn] {DECODES_JSONL} not found, skipping map144 source")
        return records

    with open(DECODES_JSONL) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            jt9 = _parse_jt9_line(rec.get("jt9_line", ""))
            snr_db   = rec.get("jt9_snr_db") or (jt9["snr_db"] if jt9 else None)
            dt_s     = jt9["dt_s"]     if jt9 else rec.get("t_sec")
            audio_hz = jt9["audio_hz"] if jt9 else None
            if audio_hz is None:
                continue

            ts        = rec["timestamp"]
            radio_khz = int(rec["radio_khz"])
            message   = rec.get("message", "").strip()

            grid, grid_src = _resolve_grid(message, qrz)
            dist = _dist_km(grid, my_lat, my_lon)

            theta = launch_idx.get((ts, radio_khz))

            wav = _map144_wav(ts, radio_khz, message)
            callsigns = calls_from_message(message)
            tx = callsigns[0] if callsigns else None
            rx = callsigns[1] if len(callsigns) > 1 else None

            records.append(dict(
                source          = "map144",
                timestamp       = _ts_to_iso(ts),
                epoch_s         = _ts_to_epoch(ts),
                wav_path        = str(wav) if wav else None,
                wav_fingerprint = _wav_fingerprint(wav) if wav else None,
                snr_db          = snr_db,
                dt_s            = dt_s,
                audio_hz        = audio_hz,
                radio_khz       = radio_khz,
                message         = message,
                callsign_tx     = tx,
                callsign_rx     = rx,
                grid            = grid,
                dist_km         = dist,
                fest_hz         = audio_hz - FC_JT9_HZ,
                theta_deg       = theta,
            ))

    print(f"  map144: {len(records)} decoded signals from {DECODES_JSONL.name}")
    return records


# ── Ingest WSJT-X ALL.TXT ────────────────────────────────────────────────────

def _ingest_wsjtx_all_txt(all_txt: Path, save_dir: Path, source_label: str,
                           qrz: dict[str, str],
                           my_lat: float, my_lon: float) -> list[dict]:
    records: list[dict] = []
    if not all_txt.exists():
        print(f"  [warn] {all_txt} not found, skipping {source_label}")
        return records

    n_skip = 0
    with open(all_txt, errors="replace") as fh:
        for line in fh:
            if "MSK144" not in line or " Rx " not in line:
                continue
            parts = line.split()
            # Format: YYMMDD_HHMMSS  dial_MHz  Rx  MSK144  SNR  DT  audio_hz  msg...
            if len(parts) < 8:
                continue
            try:
                snr_db   = int(parts[4])
                dt_s     = float(parts[5])
                audio_hz = float(parts[6])
                dial_mhz = float(parts[1])
            except (ValueError, IndexError):
                n_skip += 1
                continue

            ts      = parts[0]        # "260507_124700"
            message = " ".join(parts[7:]).strip()

            try:
                wav = period_wav(ts)
                if not wav.exists():
                    wav = None
            except (ValueError, IndexError):
                wav = None

            grid, _ = _resolve_grid(message, qrz)
            dist = _dist_km(grid, my_lat, my_lon)

            callsigns = calls_from_message(message)
            tx = callsigns[0] if callsigns else None
            rx = callsigns[1] if len(callsigns) > 1 else None

            records.append(dict(
                source          = source_label,
                timestamp       = _ts_to_iso(ts),
                epoch_s         = _ts_to_epoch(ts),
                wav_path        = str(wav) if wav else None,
                wav_fingerprint = _wav_fingerprint(wav) if wav else None,
                snr_db          = snr_db,
                dt_s            = dt_s,
                audio_hz        = audio_hz,
                radio_khz       = int(dial_mhz * 1000),
                message         = message,
                callsign_tx     = tx,
                callsign_rx     = rx,
                grid            = grid,
                dist_km         = dist,
                fest_hz         = audio_hz - FC_JT9_HZ,
                theta_deg       = None,
            ))

    print(f"  {source_label}: {len(records)} MSK144 Rx lines "
          f"({n_skip} parse errors) from {all_txt.name}")
    return records


# ── Multipath detection ───────────────────────────────────────────────────────

_MULTIPATH_TIME_TOL_S  = 2.0   # same meteor if within ±2 s
_MULTIPATH_FREQ_MIN_HZ = 20.0  # different path if audio_hz differs ≥ 20 Hz

def _detect_multipath(records: list[dict]) -> list[tuple[int, int]]:
    """Return list of (idx_a, idx_b) pairs that form multipath events."""
    pairs: list[tuple[int, int]] = []
    n = len(records)
    for i in range(n):
        a = records[i]
        if not a.get("message"):
            continue
        for j in range(i + 1, n):
            b = records[j]
            if abs(a["epoch_s"] - b["epoch_s"]) > _MULTIPATH_TIME_TOL_S:
                if b["epoch_s"] - a["epoch_s"] > _MULTIPATH_TIME_TOL_S:
                    break  # records are roughly time-sorted after this point
                continue
            if a["message"] != b["message"]:
                continue
            if a.get("audio_hz") is None or b.get("audio_hz") is None:
                continue
            if abs(a["audio_hz"] - b["audio_hz"]) < _MULTIPATH_FREQ_MIN_HZ:
                continue
            pairs.append((i, j))
    return pairs


def _assign_multipath_groups(
    records: list[dict],
    pairs: list[tuple[int, int]],
    conn,
) -> None:
    """Union-find → assign group IDs; write multipath_groups table."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    # Group by root
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        if any(i == idx or j == idx for i, j in pairs):
            groups[find(idx)].append(idx)

    for root, members in groups.items():
        if len(members) < 2:
            continue
        first = records[members[0]]
        freq_spread = max(
            abs(records[a]["audio_hz"] - records[b]["audio_hz"])
            for a in members for b in members
            if records[a].get("audio_hz") is not None
            and records[b].get("audio_hz") is not None
        )
        cur = conn.execute(
            "INSERT INTO multipath_groups (epoch_s, message, n_paths, freq_spread_hz) "
            "VALUES (?, ?, ?, ?)",
            (first["epoch_s"], first["message"], len(members), freq_spread),
        )
        gid = cur.lastrowid
        for idx in members:
            records[idx]["multipath_group_id"] = gid


# ── Auto-tagging (metadata-only tags) ────────────────────────────────────────

def _auto_tags(rec: dict) -> list[str]:
    tags: list[str] = []
    if rec.get("snr_db") is not None and rec["snr_db"] <= -5:
        tags.append("weak")
    if rec.get("audio_hz") is not None:
        if rec["audio_hz"] > 2400 or rec["audio_hz"] < 600:
            tags.append("near_nyquist")
    if rec.get("dist_km") is not None and rec["dist_km"] < 100:
        tags.append("groundwave")
    if rec.get("multipath_group_id") is not None:
        tags.append("multipath")
    return tags


# ── Deduplication (same decode logged by map144 AND wsjtx) ───────────────────

def _dedup_key(r: dict) -> tuple:
    """Coarse dedup: same period (±7.5 s), same message, same audio_hz (±10 Hz)."""
    period = round(r["epoch_s"] / 15.0)
    freq   = round((r.get("audio_hz") or 0.0) / 10.0) * 10
    return (period, r.get("message", ""), freq)


# ── Phase 1: metadata ingest ─────────────────────────────────────────────────

def run_ingest(conn, sources: list[str], no_dedup: bool = False) -> int:
    qrz      = _load_qrz_cache()
    my_lat, my_lon = maidenhead_to_latlon(MY_GRID)
    launch_idx = _build_launch_index(LAUNCHES_JSONL)

    all_records: list[dict] = []

    if "map144" in sources:
        all_records.extend(_ingest_map144(qrz, my_lat, my_lon, launch_idx))

    if "wsjtx_flex" in sources:
        all_records.extend(_ingest_wsjtx_all_txt(
            WSJTX_FLEX_ALL_TXT, WSJTX_FLEX_SAVE, "wsjtx_flex",
            qrz, my_lat, my_lon))

    if "wsjtx" in sources:
        all_records.extend(_ingest_wsjtx_all_txt(
            WSJTX_IC7300_ALL_TXT, WSJTX_IC7300_SAVE, "wsjtx",
            qrz, my_lat, my_lon))

    # Sort by epoch for multipath detection
    all_records.sort(key=lambda r: r["epoch_s"])

    # Dedup (keep first occurrence of each period×message×freq)
    if not no_dedup:
        seen: set = set()
        deduped: list[dict] = []
        for r in all_records:
            k = _dedup_key(r)
            if k not in seen:
                seen.add(k)
                deduped.append(r)
        n_dup = len(all_records) - len(deduped)
        all_records = deduped
        print(f"  Dedup: removed {n_dup} duplicates, {len(all_records)} unique signals")

    # Multipath detection
    print("  Detecting multipath groups …")
    pairs = _detect_multipath(all_records)
    _assign_multipath_groups(all_records, pairs, conn)
    n_multi = sum(1 for r in all_records if r.get("multipath_group_id"))
    print(f"  Multipath: {len(pairs)} pairs → {n_multi} signals in groups")

    # Auto-tag
    for r in all_records:
        r["tags"] = " ".join(_auto_tags(r))

    # Insert
    cols = [
        "source", "timestamp", "epoch_s", "wav_path", "wav_fingerprint",
        "snr_db", "dt_s", "audio_hz", "radio_khz",
        "message", "callsign_tx", "callsign_rx", "grid", "dist_km", "fest_hz",
        "theta_deg", "multipath_group_id", "tags",
    ]
    placeholders = ", ".join("?" * len(cols))
    col_names    = ", ".join(cols)
    sql = f"INSERT INTO signals ({col_names}) VALUES ({placeholders})"

    rows = []
    for r in all_records:
        rows.append(tuple(r.get(c) for c in cols))

    conn.executemany(sql, rows)
    conn.commit()
    print(f"  Inserted {len(rows)} signals → {DB_PATH}")
    return len(rows)


# ── Phase 2: WAV precompute ───────────────────────────────────────────────────

def _stft_cache_path(signal_id: int) -> Path:
    return CACHE_DIR / f"{signal_id}.npz"


def _stft_vectorised(audio: np.ndarray, nfft: int, hop: int, fs: float):
    """Vectorised STFT — builds frame matrix then calls rfft once, no Python loop."""
    win  = np.hanning(nfft).astype(np.float32)
    n_fr = 1 + (len(audio) - nfft) // hop
    idx  = np.arange(nfft)[None, :] + np.arange(n_fr)[:, None] * hop
    S    = np.abs(np.fft.rfft(audio[idx] * win, nfft, axis=1)) ** 2
    times = (np.arange(n_fr) * hop + nfft / 2.0) / fs
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return times.astype(np.float32), freqs.astype(np.float32), S.astype(np.float32)


def _compute_and_cache(signal_id: int, wav_path: str, audio_hz: float,
                        dt_s: float | None) -> dict:
    """Compute STFT + squared arrays for one signal; save to .npz; return metrics."""
    from tools.long_ping_gallery import NFFT, HOP, FS as _FS

    try:
        audio = load_wav(Path(wav_path))
    except Exception as e:
        return {"error": str(e)}

    times, freqs, S  = _stft_vectorised(audio, NFFT, HOP, _FS)
    # Square real audio: carrier doubles to 2·audio_hz, noise fills 0–6 kHz
    y    = (audio ** 2).astype(np.float32)
    _, _, Sq = _stft_vectorised(y, NFFT, HOP, _FS)

    out = _stft_cache_path(signal_id)
    np.savez_compressed(str(out), S=S, Sq=Sq, times=times, freqs=freqs)
    return {"cache_path": str(out)}


def _run_wav_metrics(wav_path: str, audio_hz: float,
                      dt_s: float | None) -> dict:
    """Run all WAV analysis tools; return dict of DB column names → values."""
    from tools.measure_ping_length import measure_wav
    from tools.measure_ping_freq_vs_time import analyse_wav as freq_analyse
    from tools.measure_burst_freq_track import analyse_wav as track_analyse
    from tools.long_ping_gallery import to_baseband, FC_JT9, find_min_navg, FS

    p = Path(wav_path)
    result: dict = {}

    # Ping length
    try:
        meas = measure_wav(p)
        result["bb_longest_ms"] = meas.bb_longest_ms
        result["ab_longest_ms"] = meas.ab_longest_ms
        result["bb_total_ms"]   = meas.bb_total_ms
        result["ab_total_ms"]   = meas.ab_total_ms
    except Exception:
        pass

    # Freq stability
    try:
        _, _, _, burst_stats = freq_analyse(p)
        if burst_stats:
            s = burst_stats[0]  # primary burst
            result["f_std_hz"]   = s.f_std_hz
            result["f_range_hz"] = s.f_range_hz
            result["af_corr"]    = s.af_corr
    except Exception:
        pass

    # Freq track (drift)
    try:
        _, _, tracks = track_analyse(p)
        if tracks:
            t = tracks[0]
            result["drift_hz_per_s"]  = t.drift_hz_per_s
            result["drift_resid_hz"]  = t.resid_hz
            result["n_frames_used"]   = t.n_frames
    except Exception:
        pass

    # navg (requires SPD — this is slow)
    if dt_s is not None:
        try:
            audio = load_wav(p)
            fc_off = audio_hz - FC_JT9
            bb = to_baseband(audio, audio_hz)
            navg, xmax = find_min_navg(bb, dt_s, fc_off)
            result["navg"]      = navg
            result["xmax_norm"] = xmax
        except Exception:
            pass

    # Auto-tags from computed metrics
    extra_tags: list[str] = []
    if result.get("f_std_hz", 0.0) >= 1.7:
        extra_tags.append("fluttery")
    if abs(result.get("drift_hz_per_s", 0.0) or 0.0) >= 0.5:
        extra_tags.append("drifting")
    if result.get("bb_longest_ms", 9999.0) <= 500.0:
        extra_tags.append("short_burst")
    result["_extra_tags"] = extra_tags

    return result


# ── Worker (top-level so multiprocessing can pickle it) ──────────────────────

def _worker(args: tuple) -> tuple[int, dict]:
    """Process one WAV in a subprocess; return (signal_id, metrics_dict)."""
    sid, wav_path, wav_fp, audio_hz, dt_s, cache_dir, stft_only = args
    p = Path(wav_path)
    if not p.exists():
        return sid, {}

    cache_p = Path(cache_dir) / f"{sid}.npz"

    # Skip if cache already current
    if cache_p.exists() and wav_fp and _wav_fingerprint(p) == wav_fp:
        return sid, {"cache_path": str(cache_p), "_already_cached": True}

    result = _compute_and_cache(sid, wav_path, audio_hz, dt_s)
    if not stft_only and "error" not in result:
        result.update(_run_wav_metrics(wav_path, audio_hz, dt_s))
    return sid, result


def run_precompute(conn, n_workers: int | None = None,
                   limit: int | None = None, stft_only: bool = False,
                   force: bool = False) -> None:
    """Phase 2: for each signal with a WAV, run analysis + STFT cache.

    Runs on all CPU cores in parallel (ProcessPoolExecutor).
    stft_only=True skips the slow analysis tools; only builds the display cache.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or os.cpu_count()

    if force:
        conn.execute("UPDATE signals SET cache_path = NULL")
        conn.commit()
        for f in CACHE_DIR.glob("*.npz"):
            f.unlink()
    rows = conn.execute(
        "SELECT id, wav_path, wav_fingerprint, audio_hz, dt_s, cache_path "
        "FROM signals WHERE wav_path IS NOT NULL "
        "ORDER BY epoch_s"
    ).fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    mode = "stft-only" if stft_only else "full metrics"
    print(f"  Precomputing {total} WAV signals ({mode}, {n_workers} workers) …")
    t0 = time.monotonic()

    args_iter = (
        (row["id"], row["wav_path"], row["wav_fingerprint"],
         row["audio_hz"], row["dt_s"], str(CACHE_DIR), stft_only)
        for row in rows
    )

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args_iter}
        for fut in as_completed(futures):
            sid, metrics = fut.result()
            done += 1

            if not metrics or "_already_cached" in metrics:
                if metrics.get("cache_path"):
                    conn.execute("UPDATE signals SET cache_path=? WHERE id=?",
                                 (metrics["cache_path"], sid))
            else:
                extra_tags = metrics.pop("_extra_tags", [])
                update_cols = [k for k in metrics
                               if k not in ("error",) and not k.startswith("_")]
                if update_cols:
                    set_clause = ", ".join(f"{c}=?" for c in update_cols)
                    vals = [metrics[c] for c in update_cols] + [sid]
                    conn.execute(
                        f"UPDATE signals SET {set_clause} WHERE id=?", vals)
                if extra_tags:
                    row = conn.execute(
                        "SELECT tags FROM signals WHERE id=?", (sid,)
                    ).fetchone()
                    existing = set((row["tags"] or "").split()) if row else set()
                    existing.update(extra_tags)
                    conn.execute("UPDATE signals SET tags=? WHERE id=?",
                                 (" ".join(sorted(existing)), sid))

            if done % 100 == 0 or done == total:
                conn.commit()
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total}  ({rate:.1f}/s  ETA {eta:.0f}s)")

    conn.commit()
    print(f"  Precompute done in {time.monotonic()-t0:.0f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build scratch/signals.db from all MSK144 signal sources")
    ap.add_argument("--db",         default=str(DB_PATH),
                    help="SQLite DB path (default: scratch/signals.db)")
    ap.add_argument("--rebuild",    action="store_true",
                    help="Drop and recreate the DB before ingesting")
    ap.add_argument("--sources",    default="map144,wsjtx_flex,wsjtx",
                    help="Comma-separated sources: map144,wsjtx_flex,wsjtx")
    ap.add_argument("--no-dedup",   action="store_true",
                    help="Skip deduplication of identical decodes across sources")
    ap.add_argument("--precompute", action="store_true",
                    help="Run WAV analysis + STFT cache after ingest")
    ap.add_argument("--precompute-only", action="store_true",
                    help="Skip ingest; only run precompute on existing DB")
    ap.add_argument("--workers",    type=int, default=None,
                    help="Worker processes (default: all CPU cores)")
    ap.add_argument("--stft-only",  action="store_true",
                    help="Only build STFT display cache; skip slow analysis tools")
    ap.add_argument("--limit",      type=int, default=None,
                    help="Limit precompute to N WAVs (for testing)")
    ap.add_argument("--force-cache", action="store_true",
                    help="Clear cached paths and recompute all WAV signals")
    args = ap.parse_args()

    db_path = Path(args.db)
    sources = [s.strip() for s in args.sources.split(",")]

    if args.rebuild and db_path.exists() and not args.precompute_only:
        print(f"Removing existing DB: {db_path}")
        db_path.unlink()

    conn = open_db(db_path)

    if not args.precompute_only:
        print(f"Phase 1: metadata ingest → {db_path}")
        n = run_ingest(conn, sources, no_dedup=args.no_dedup)
        print(f"Phase 1 complete: {n} signals ingested")

    if args.precompute or args.precompute_only:
        print(f"\nPhase 2: WAV precompute → {CACHE_DIR}")
        run_precompute(conn,
                       n_workers=args.workers,
                       limit=args.limit,
                       stft_only=args.stft_only,
                       force=args.force_cache)

    # Summary
    stats = conn.execute(
        "SELECT source, count(*) as n FROM signals GROUP BY source"
    ).fetchall()
    print("\nDB summary:")
    for row in stats:
        print(f"  {row['source']:15s}  {row['n']:6d} signals")
    total = conn.execute("SELECT count(*) FROM signals").fetchone()[0]
    n_wav = conn.execute(
        "SELECT count(*) FROM signals WHERE wav_path IS NOT NULL"
    ).fetchone()[0]
    n_multi = conn.execute(
        "SELECT count(*) FROM signals WHERE multipath_group_id IS NOT NULL"
    ).fetchone()[0]
    n_groups = conn.execute(
        "SELECT count(*) FROM multipath_groups"
    ).fetchone()[0]
    print(f"  {'TOTAL':15s}  {total:6d} signals  "
          f"{n_wav} with WAV  {n_multi} multipath ({n_groups} groups)")

    conn.close()


if __name__ == "__main__":
    main()
