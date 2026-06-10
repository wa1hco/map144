"""
signal_db.py — SQLite schema and query helpers for the signal browser.

No Qt, no DSP — pure database layer.
"""

import sqlite3
from pathlib import Path
from typing import Any

# ── Schema ──────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS multipath_groups (
    id              INTEGER PRIMARY KEY,
    epoch_s         REAL,
    message         TEXT,
    n_paths         INTEGER,
    freq_spread_hz  REAL
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY,

    -- Identity
    source          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    epoch_s         REAL NOT NULL,
    wav_path        TEXT,
    wav_fingerprint TEXT,
    cache_path      TEXT,

    -- Decode result
    snr_db          REAL,
    dt_s            REAL,
    audio_hz        REAL,
    radio_khz       REAL,
    message         TEXT,
    callsign_tx     TEXT,
    callsign_rx     TEXT,
    grid            TEXT,
    dist_km         REAL,
    fest_hz         REAL,

    -- SPD coherent-integration metrics
    navg            INTEGER,
    xmax_norm       REAL,
    sync_phase_coherence_h  REAL,
    sync_phase_coherence_v  REAL,
    sync_t_sample_h         REAL,
    sync_t_sample_v         REAL,
    sync_freq_offset_hz_h   REAL,
    sync_freq_offset_hz_v   REAL,

    -- Detection statistics (from launches.jsonl)
    sq_metric_db_h          REAL,
    sq_metric_db_v          REAL,
    sync_metric_db_h        REAL,
    sync_metric_db_v        REAL,
    pair_metric_db_h        REAL,
    pair_metric_db_v        REAL,
    sq_new_detmet_norm_db_h REAL,
    sq_new_detmet_norm_db_v REAL,
    sq_new_fc_offset_hz_h   REAL,
    sq_new_fc_offset_hz_v   REAL,
    ch_pct25_db_h           REAL,
    ch_pct25_db_v           REAL,
    n_chans_300ms           INTEGER,
    n_chans_2s              INTEGER,
    det_source              TEXT,
    theta_deg               REAL,

    -- Ping duration (from measure_ping_length.py)
    bb_longest_ms           REAL,
    ab_longest_ms           REAL,
    bb_total_ms             REAL,
    ab_total_ms             REAL,

    -- Carrier frequency & drift (from measure_burst_freq_track.py)
    f0_hz                   REAL,
    peak_to_noise           REAL,
    peak_width_hz           REAL,
    drift_hz_per_s          REAL,
    drift_resid_hz          REAL,
    n_frames_used           INTEGER,

    -- Frequency stability (from measure_ping_freq_vs_time.py)
    f_std_hz                REAL,
    f_range_hz              REAL,
    af_corr                 REAL,

    -- Multipath
    multipath_group_id INTEGER REFERENCES multipath_groups(id),

    -- Annotation & clustering
    tags            TEXT DEFAULT '',
    reviewed        INTEGER DEFAULT 0,
    cluster_id      INTEGER,
    anomaly_score   REAL
);

CREATE INDEX IF NOT EXISTS sig_epoch   ON signals(epoch_s);
CREATE INDEX IF NOT EXISTS sig_dist    ON signals(dist_km);
CREATE INDEX IF NOT EXISTS sig_call    ON signals(callsign_tx, callsign_rx);
CREATE INDEX IF NOT EXISTS sig_grid    ON signals(grid);
CREATE INDEX IF NOT EXISTS sig_navg    ON signals(navg);
CREATE INDEX IF NOT EXISTS sig_audio   ON signals(audio_hz);
CREATE INDEX IF NOT EXISTS sig_drift   ON signals(drift_hz_per_s);
CREATE INDEX IF NOT EXISTS sig_dur     ON signals(bb_longest_ms);
CREATE INDEX IF NOT EXISTS sig_multi   ON signals(multipath_group_id);
CREATE INDEX IF NOT EXISTS sig_source  ON signals(source);
CREATE INDEX IF NOT EXISTS sig_ts      ON signals(timestamp);
"""


def open_db(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}{'?mode=ro' if read_only else ''}"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not read_only:
        for stmt in _DDL.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
    return conn


# ── Query helpers ────────────────────────────────────────────────────────────

def get_signal(conn: sqlite3.Connection, signal_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM signals WHERE id = ?", (signal_id,)
    ).fetchone()
    return dict(row) if row else None


def get_multipath_group(conn: sqlite3.Connection, group_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM signals WHERE multipath_group_id = ? ORDER BY audio_hz",
        (group_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def search(conn: sqlite3.Connection, filters: dict[str, Any]) -> list[dict]:
    """
    Return signals matching *filters*.

    Supported filter keys
    ---------------------
    dist_km_range       : (min, max) km — None means unbounded
    callsign            : str — matches TX or RX (case-insensitive prefix)
    grid                : str — 2- or 4-char prefix match
    snr_db_range        : (min, max)
    audio_hz_range      : (min, max)
    off_freq_thresh_hz  : float — |audio_hz - 1500| > threshold
    navg_list           : list[int | None] — None in list matches NULL navg
    duration_ms_range   : (min, max) — compares bb_longest_ms
    drift_hz_per_s_range: (min, max) — |drift_hz_per_s| range
    flutter_only        : bool — af_corr > 0 (correlation amplitude↔freq)
    sync_coherence_min  : float — min(sync_phase_coherence_h, _v) ≥ threshold
    multipath_only      : bool
    date_range          : (str, str) — ISO dates 'YYYY-MM-DD' inclusive
    source_list         : list[str] — 'map144', 'wsjtx_flex', 'wsjtx'
    tag_filter          : str — space-separated tags; ALL must be present
    unreviewed_only     : bool
    order_by            : str — column name (default: epoch_s)
    order_desc          : bool
    limit               : int
    """
    clauses: list[str] = []
    params: list[Any] = []

    dist_range = filters.get("dist_km_range")
    if dist_range:
        lo, hi = dist_range
        if lo is not None:
            clauses.append("dist_km >= ?"); params.append(lo)
        if hi is not None:
            clauses.append("dist_km <= ?"); params.append(hi)

    callsign = filters.get("callsign")
    if callsign:
        pat = callsign.upper() + "%"
        clauses.append("(UPPER(callsign_tx) LIKE ? OR UPPER(callsign_rx) LIKE ?)")
        params.extend([pat, pat])

    grid = filters.get("grid")
    if grid:
        clauses.append("grid LIKE ?"); params.append(grid.upper() + "%")

    snr_range = filters.get("snr_db_range")
    if snr_range:
        lo, hi = snr_range
        if lo is not None:
            clauses.append("snr_db >= ?"); params.append(lo)
        if hi is not None:
            clauses.append("snr_db <= ?"); params.append(hi)

    audio_range = filters.get("audio_hz_range")
    if audio_range:
        lo, hi = audio_range
        if lo is not None:
            clauses.append("audio_hz >= ?"); params.append(lo)
        if hi is not None:
            clauses.append("audio_hz <= ?"); params.append(hi)

    off_freq = filters.get("off_freq_thresh_hz")
    if off_freq:
        clauses.append("ABS(audio_hz - 1500.0) > ?"); params.append(off_freq)

    navg_list = filters.get("navg_list")
    if navg_list is not None:
        int_vals = [v for v in navg_list if v is not None]
        has_null = any(v is None for v in navg_list)
        sub: list[str] = []
        if int_vals:
            placeholders = ",".join("?" * len(int_vals))
            sub.append(f"navg IN ({placeholders})")
            params.extend(int_vals)
        if has_null:
            sub.append("navg IS NULL")
        if sub:
            clauses.append(f"({' OR '.join(sub)})")

    dur_range = filters.get("duration_ms_range")
    if dur_range:
        lo, hi = dur_range
        if lo is not None:
            clauses.append("bb_longest_ms >= ?"); params.append(lo)
        if hi is not None:
            clauses.append("bb_longest_ms <= ?"); params.append(hi)

    drift_range = filters.get("drift_hz_per_s_range")
    if drift_range:
        lo, hi = drift_range
        if lo is not None:
            clauses.append("ABS(drift_hz_per_s) >= ?"); params.append(lo)
        if hi is not None:
            clauses.append("ABS(drift_hz_per_s) <= ?"); params.append(hi)

    if filters.get("flutter_only"):
        clauses.append("af_corr IS NOT NULL AND af_corr > 0.3")

    sync_min = filters.get("sync_coherence_min")
    if sync_min is not None:
        clauses.append(
            "MIN(COALESCE(sync_phase_coherence_h, 0), "
            "    COALESCE(sync_phase_coherence_v, 0)) >= ?"
        )
        params.append(sync_min)

    if filters.get("multipath_only"):
        clauses.append("multipath_group_id IS NOT NULL")

    date_range = filters.get("date_range")
    if date_range:
        lo, hi = date_range
        if lo:
            clauses.append("timestamp >= ?"); params.append(lo)
        if hi:
            clauses.append("timestamp <= ?"); params.append(hi + " 23:59:59")

    source_list = filters.get("source_list")
    if source_list:
        placeholders = ",".join("?" * len(source_list))
        clauses.append(f"source IN ({placeholders})")
        params.extend(source_list)

    tag_filter = filters.get("tag_filter", "").strip()
    if tag_filter:
        for tag in tag_filter.split():
            clauses.append("(' ' || tags || ' ') LIKE ?")
            params.append(f"% {tag} %")

    if filters.get("unreviewed_only"):
        clauses.append("reviewed = 0")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_col = filters.get("order_by", "epoch_s")
    order_dir = "DESC" if filters.get("order_desc") else "ASC"
    limit_clause = f"LIMIT {int(filters['limit'])}" if filters.get("limit") else ""

    sql = f"SELECT * FROM signals {where} ORDER BY {order_col} {order_dir} {limit_clause}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_summary(conn: sqlite3.Connection, filters: dict[str, Any]) -> dict:
    """Return aggregate stats for the current filter result."""
    rows = search(conn, {**filters, "limit": None})
    if not rows:
        return {"count": 0}
    import statistics
    snrs = [r["snr_db"] for r in rows if r["snr_db"] is not None]
    dists = [r["dist_km"] for r in rows if r["dist_km"] is not None]
    return {
        "count": len(rows),
        "mean_snr_db": statistics.mean(snrs) if snrs else None,
        "mean_dist_km": statistics.mean(dists) if dists else None,
        "n_with_wav": sum(1 for r in rows if r["wav_path"]),
        "n_multipath": sum(1 for r in rows if r["multipath_group_id"]),
    }


def column_values(
    conn: sqlite3.Connection,
    column: str,
    filters: dict[str, Any] | None = None,
) -> list:
    """Return non-NULL values of *column* matching optional *filters*."""
    ALLOWED = {
        "snr_db", "dist_km", "audio_hz", "fest_hz", "navg", "dt_s",
        "bb_longest_ms", "ab_longest_ms", "bb_total_ms", "ab_total_ms",
        "drift_hz_per_s", "drift_resid_hz", "f_std_hz", "f_range_hz",
        "af_corr", "f0_hz", "peak_to_noise", "peak_width_hz",
        "sync_phase_coherence_h", "sync_phase_coherence_v",
        "sync_freq_offset_hz_h", "sync_freq_offset_hz_v",
        "sq_metric_db_h", "sq_metric_db_v",
        "sq_new_detmet_norm_db_h", "sq_new_detmet_norm_db_v",
        "anomaly_score", "cluster_id", "theta_deg",
        "n_chans_300ms", "n_chans_2s",
    }
    if column not in ALLOWED:
        raise ValueError(f"column_values: disallowed column '{column}'")
    if filters:
        rows = search(conn, filters)
        return [r[column] for r in rows if r[column] is not None]
    rows = conn.execute(
        f"SELECT {column} FROM signals WHERE {column} IS NOT NULL"
    ).fetchall()
    return [r[0] for r in rows]


def write_cluster_results(
    conn: sqlite3.Connection,
    ids: list[int],
    labels: list[int],
    scores: list[float],
) -> None:
    conn.executemany(
        "UPDATE signals SET cluster_id = ?, anomaly_score = ? WHERE id = ?",
        [(int(lbl), float(sc), int(sid)) for lbl, sc, sid in zip(labels, scores, ids)],
    )
    conn.commit()


def set_tag(conn: sqlite3.Connection, signal_id: int, tag: str, add: bool = True) -> None:
    row = conn.execute("SELECT tags FROM signals WHERE id = ?", (signal_id,)).fetchone()
    if row is None:
        return
    tags = set(row["tags"].split()) if row["tags"] else set()
    if add:
        tags.add(tag)
    else:
        tags.discard(tag)
    conn.execute(
        "UPDATE signals SET tags = ? WHERE id = ?",
        (" ".join(sorted(tags)), signal_id),
    )
    conn.commit()


def set_reviewed(conn: sqlite3.Connection, signal_id: int, reviewed: bool = True) -> None:
    conn.execute(
        "UPDATE signals SET reviewed = ? WHERE id = ?",
        (1 if reviewed else 0, signal_id),
    )
    conn.commit()
