#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thumbnail HTML gallery for the DSP-regression corpus.

Filters ``tests/corpus/index.csv`` by ``--tag`` (AND across multiple) or
``--any-tag`` (OR), renders a spectrogram thumbnail per matching case,
and emits one HTML page with a CSS grid of tiles.

Thumbnails are cached under ``tests/corpus/thumbs/`` keyed by SHA-1 of
``(path + mtime + size)``; re-runs reuse cached PNGs.  Cache invalidates
automatically when a WAV changes on disk.

Per-tile content
----------------
* Spectrogram of the WSJT-X save WAV (12 kHz, 0-4 kHz), or — if absent —
  the first MAP144 detection WAV for that period.  A small badge shows
  which source was used.
* Period timestamp (click opens the WAV in the OS default app)
* WSJT-X decodes (count, SNR range, message text)
* MAP144 decodes / event count (red if event count > 0 and decodes == 0)
* Key metrics chosen for triage value:
    sq=<max_sq_metric_db>  coh=<median_coherence_h>  n2s=<max_n_chans_2s>
    freq=<freq_khz>
* Tag chips, color-coded by pass criterion (green=catch test,
  red=false-positive test, gray=orthogonal)

Workflow
--------
::

    # First the mission-critical bucket — weak signals MAP144 missed
    python3 tools/gallery_corpus.py --tag weak --tag wsjtx_miss \\
        --out /tmp/weak_miss.html
    xdg-open /tmp/weak_miss.html

    # Or: all noise-driven activity, capped
    python3 tools/gallery_corpus.py --tag persistent_nodecode --max 100 \\
        --out /tmp/noise.html

The cache is shared across all runs — switching filters re-renders only
the new WAVs.  ``--clear-cache`` forces a fresh render.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Non-interactive matplotlib backend — we render in batch, no display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


_REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_INDEX  = _REPO_ROOT / "tests" / "corpus" / "index.csv"
DEFAULT_THUMBS = _REPO_ROOT / "tests" / "corpus" / "thumbs"


# ── Spectrogram rendering ────────────────────────────────────────────────────

THUMB_W_PX = 240
THUMB_H_PX = 140
NFFT       = 512
HOP        = 256
FMAX_HZ    = 4000.0   # MSK144 audio band ceiling; full band would dilute features

# Noise-floor-relative dynamic range.  Peak-normalisation washes out brief
# weak pings (a 500 ms ping in a 15 s window is ~3% of bins, so its peak
# barely shifts the cmap).  P25 was the first attempt but sits BELOW the
# noise speckle (typical p50 - p25 ≈ 7 dB on WSJT-X audio WAVs), making the
# whole image look warm and washing out signals.  Anchoring at the MEDIAN
# (typical bin value) puts the speckle in deep blue and lets signal bins
# stand out brightly.  22 dB above median saturates the top end on strong
# pings (peak ≈ p99 + 5 dB) — acceptable for thumbnails.
# Noise-floor anchor + display range.
#
# Empirically on WSJT-X audio WAVs:
#   * median (p50) is a robust noise-floor estimate — sparse bright pings
#     (31 bins of 120 000 for a weak case) don't move it
#   * p99 of bin values sits ~10 dB above median (noise speckle peaks)
#   * strong-ping peaks sit ~22 dB above median; weak-ping peaks ~15 dB
#
# Display window: [median - HEADROOM_DB, median + RANGE_DB].  The HEADROOM
# lifts typical noise off the cmap bottom so it reads as navy/blue (matching
# the live MAP144 GUI look) instead of pure black.  RANGE_DB above puts
# strong-ping peaks near the top of the cmap (cyan / pale-yellow / white).
NOISE_PCT             = 50
NOISE_HEADROOM_DB     = 8.0    # cmap bottom this far below median anchor
NOISE_REL_RANGE_DB    = 22.0   # cmap top this far above median anchor


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV as float32 in [-1, 1]; mix down if stereo."""
    with wave.open(str(path), 'rb') as w:
        nch = w.getnchannels()
        fs  = w.getframerate()
        sw  = w.getsampwidth()
        n   = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        raise RuntimeError(f"unsupported sample width {sw} in {path}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, fs


def _spectrogram_db(x: np.ndarray, fs: int,
                    ) -> tuple[np.ndarray, float, float, float, float]:
    """STFT magnitude in dB, anchored at the per-image median.

    Returns (S_db, t_max_s, f_max_hz, vmin_db, vmax_db).  Orientation: rows
    = freq ascending, cols = time ascending — feed to imshow with
    ``origin='lower'`` and ``vmin``/``vmax`` from the return.

    See module-level NOISE_PCT / NOISE_REL_RANGE_DB constants for the
    contrast-tuning rationale.
    """
    win = np.hanning(NFFT).astype(np.float32)
    n_frames = max(1, 1 + (len(x) - NFFT) // HOP)
    S = np.empty((NFFT // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i*HOP : i*HOP + NFFT]
        if len(seg) < NFFT:
            seg = np.pad(seg, (0, NFFT - len(seg)))
        F = np.fft.rfft(seg * win)
        S[:, i] = np.abs(F)
    eps = 1e-12
    S_db = 20.0 * np.log10(S + eps)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    keep  = freqs <= FMAX_HZ
    S_db  = S_db[keep, :]
    anchor = float(np.percentile(S_db, NOISE_PCT))
    vmin   = anchor - NOISE_HEADROOM_DB
    vmax   = anchor + NOISE_REL_RANGE_DB
    t_max = n_frames * HOP / fs
    return S_db, t_max, min(FMAX_HZ, freqs[keep][-1]), vmin, vmax


_CMAP_CACHE = None
def _cmap():
    """MAP144 GUI colormap.

    Mirrors map144_app/ui.py:295-303 (the pyqtgraph ColorMap used by the
    realtime / spectrogram / channel-detect / sync-detect image items) so
    the gallery thumbnails read the same as the live displays.

    Stops: black → dark navy → blue → cyan → pale-yellow → white.  No
    red/orange — that's what gives the live MAP144 displays their light,
    blue-dominant look.
    """
    global _CMAP_CACHE
    if _CMAP_CACHE is None:
        # Stops copied verbatim from ui.py; positions are evenly spaced i/8.
        stops_rgb = [
            (0, 0, 0), (0, 0, 64), (0, 0, 128), (0, 64, 192),
            (0, 128, 255), (64, 192, 255), (128, 255, 255),
            (255, 255, 128), (255, 255, 255),
        ]
        stops = [(i/8.0, (r/255.0, g/255.0, b/255.0))
                 for i, (r, g, b) in enumerate(stops_rgb)]
        _CMAP_CACHE = LinearSegmentedColormap.from_list(
            'map144_gui',
            [c for _, c in stops], N=256)
    return _CMAP_CACHE


def _render_thumbnail(wav_path: Path, out_png: Path,
                      width_px: int = THUMB_W_PX,
                      height_px: int = THUMB_H_PX) -> bool:
    """Render one PNG thumbnail.  Returns False if the WAV can't be read.

    Width/height default to gallery thumb size; review_corpus.py uses a
    larger size for the single-case detail view.
    """
    try:
        x, fs = _read_wav_mono(wav_path)
    except Exception as exc:
        print(f"  warning: read failed {wav_path}: {exc}", file=sys.stderr)
        return False
    if len(x) < NFFT:
        return False
    S_db, t_max, f_max, vmin, vmax = _spectrogram_db(x, fs)
    fig = plt.figure(figsize=(width_px/80.0, height_px/80.0), dpi=80)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(S_db, aspect='auto', origin='lower',
              extent=[0, t_max, 0, f_max],
              vmin=vmin, vmax=vmax, cmap=_cmap(),
              interpolation='nearest')
    ax.set_axis_off()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80, pad_inches=0)
    plt.close(fig)
    return True


# ── Cache key ────────────────────────────────────────────────────────────────

def _cache_key(wav_path: Path) -> str | None:
    """Hash of (abs path + mtime_ns + size).  None if file missing."""
    try:
        st = wav_path.stat()
    except FileNotFoundError:
        return None
    h = hashlib.sha1()
    h.update(str(wav_path).encode('utf-8'))
    h.update(str(st.st_mtime_ns).encode('utf-8'))
    h.update(str(st.st_size).encode('utf-8'))
    return h.hexdigest()


def _ensure_thumb(thumbs_dir: Path, wav_path: Path) -> Path | None:
    """Return the path to the thumbnail PNG, rendering if not cached."""
    key = _cache_key(wav_path)
    if key is None:
        return None
    out = thumbs_dir / f"{key}.png"
    if out.is_file():
        return out
    if _render_thumbnail(wav_path, out):
        return out
    return None


# ── Index filtering ──────────────────────────────────────────────────────────

def _row_tags(row: dict) -> set[str]:
    return set(row['tags'].split(';')) if row['tags'] else set()


def _filter_rows(rows: list[dict], require: list[str], any_of: list[str],
                 max_n: int | None, require_wav: bool = True,
                 sort_key: str = 'snr_asc',
                 hours_back: float | None = None,
                 ) -> tuple[list[dict], int, int]:
    """Apply tag + WAV-availability + time-window filters, then sort.
    Returns (selected, n_tag_match, n_dropped_no_wav).  ``hours_back`` (UTC
    hours) scopes review to recent activity; None means no time filter.
    The cap is applied AFTER all filtering AND sorting, so --max keeps the
    top of the sorted list."""
    from datetime import datetime, timezone, timedelta
    req     = set(require)
    any_set = set(any_of)
    cutoff = None
    if hours_back is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    tag_matched = []
    for r in rows:
        if cutoff is not None:
            try:    ts = datetime.fromisoformat(r['period_ts'])
            except: continue
            if ts < cutoff:
                continue
        rt = _row_tags(r)
        if req and not req.issubset(rt):
            continue
        if any_set and not (any_set & rt):
            continue
        tag_matched.append(r)
    if require_wav:
        with_wav = [r for r in tag_matched
                    if (r.get('wsjtx_wav') or r.get('map144_wavs'))]
    else:
        with_wav = tag_matched
    n_dropped = len(tag_matched) - len(with_wav)
    with_wav = _sort_rows(with_wav, sort_key)
    if max_n and len(with_wav) > max_n:
        with_wav = with_wav[:max_n]
    return with_wav, len(tag_matched), n_dropped


def _sort_rows(rows: list[dict], key: str) -> list[dict]:
    """Sort rows by named key.  Missing values go to the end."""
    BIG = 1e18
    if key == 'time':
        return sorted(rows, key=lambda r: r.get('period_ts', ''))
    if key == 'snr_asc':   # weakest WSJT-X signal first — most actionable
        def k(r):
            v = r.get('wsjtx_snr_min', '')
            try:    return int(v)
            except: return BIG
        return sorted(rows, key=k)
    if key == 'snr_desc':
        def k(r):
            v = r.get('wsjtx_snr_max', '')
            try:    return -int(v)
            except: return BIG
        return sorted(rows, key=k)
    if key == 'sq_desc':   # strongest MAP144 squared-FFT metric first
        def k(r):
            v = r.get('max_sq_metric_db', '')
            try:    return -float(v)
            except: return BIG
        return sorted(rows, key=k)
    if key == 'sq_asc':
        def k(r):
            v = r.get('max_sq_metric_db', '')
            try:    return float(v)
            except: return BIG
        return sorted(rows, key=k)
    if key == 'events_desc':   # noisiest periods first
        def k(r):
            try:    return -int(r.get('n_map144_events') or 0)
            except: return 0
        return sorted(rows, key=k)
    # Fallback: chronological
    return sorted(rows, key=lambda r: r.get('period_ts', ''))


def _select_wav(row: dict) -> tuple[Path | None, str]:
    """Pick which WAV to render.  Returns (path, source_label)."""
    if row.get('wsjtx_wav'):
        return Path(row['wsjtx_wav']), 'wsjtx'
    ms = row.get('map144_wavs') or ''
    if ms:
        first = ms.split(';', 1)[0]
        if first:
            return Path(first), 'map144'
    return None, ''


# ── Tag classification (for chip colour) ─────────────────────────────────────

def _load_tag_classes() -> dict[str, str]:
    """{tag_name: pass_when} from tests.corpus.tags — drives chip color."""
    sys.path.insert(0, str(_REPO_ROOT))
    from tests.corpus.tags import TAGS
    return {t.name: t.pass_when for t in TAGS}


def _chip_class(pass_when: str) -> str:
    return {'decoded': 'catch', 'no_decode': 'fp', 'any': 'any'}.get(pass_when, 'any')


# ── HTML emission ────────────────────────────────────────────────────────────

_CSS = """
body { font-family: sans-serif; font-size: 12pt; background: #fafafa; color: #333;
       margin: 0; padding: 16px; }
h1 { font-size: 15pt; margin: 0 0 6px 0; font-weight: 500; color: #555; }
.meta { color: #888; margin-bottom: 16px; font-size: 10.5pt; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
        gap: 10px; }
.tile { background: #fff; border: 1px solid #ececec; border-radius: 4px;
        padding: 8px; box-shadow: 0 1px 1px rgba(0,0,0,0.03); }
.tile .imgwrap { position: relative; }
.tile img { width: 100%; height: 140px; image-rendering: pixelated;
            border: 1px solid #eee; display: block; }
.tile .src { position: absolute; bottom: 3px; right: 3px;
             background: rgba(255,255,255,0.78); color: #222; font-size: 8.5pt;
             padding: 0 4px; border-radius: 2px; font-family: monospace; }
.tile .hdrline { margin-top: 6px; display: flex; align-items: center;
                 justify-content: space-between; gap: 4px; }
.tile .caseid { font-family: monospace; font-size: 9.5pt; color: #444;
                cursor: text; user-select: all; }
.tile .tagbtn { font-family: monospace; font-size: 8.5pt; color: #666;
                background: #f0f0f0; border: 1px solid #e0e0e0; border-radius: 3px;
                padding: 1px 6px; cursor: pointer; }
.tile .tagbtn:hover { background: #e6e6e6; color: #333; }
.tile .freq { font-family: monospace; font-size: 9pt; color: #888;
              float: right; }
.tile .msg { font-family: monospace; font-size: 9.5pt; margin-top: 3px;
             color: #444; word-wrap: break-word; line-height: 1.3; }
.tile .msg.miss { color: #a52; }
.tile .msg.got  { color: #275; }
.tile .submsg { font-family: monospace; font-size: 8.5pt; color: #888;
                margin-left: 10px; line-height: 1.25; }
.tile .note { font-family: serif; font-style: italic; font-size: 10pt;
              color: #555; margin-top: 4px; padding: 3px 6px;
              background: #fafbe8; border-left: 2px solid #d4d090; }
.chips { margin-top: 5px; }
.chip { display: inline-block; background: #eee; color: #555; font-size: 8.5pt;
        padding: 1px 6px; border-radius: 9px; margin: 2px 2px 0 0;
        font-family: monospace; }
.chip.catch  { background: #e6f4dd; color: #275; }
.chip.fp     { background: #faeaea; color: #944; }
.chip.any    { background: #eee;    color: #555; }
.chip.manual { background: #e8e0f4; color: #524080;
               box-shadow: inset 0 0 0 1px #cfc2e6; }
.tile .metrics { font-family: monospace; font-size: 8.5pt; color: #888;
                 margin-top: 4px; }
.tile a { color: inherit; text-decoration: none; }
.tile a:hover .hdrline { text-decoration: underline; }
"""

# Tiny JS: clipboard helper for the per-tile "tag" button.  No frameworks,
# no fetch — just navigator.clipboard.writeText().  The button copies a
# ready-to-paste tag_case.py invocation; the user adds tags in the terminal.
_JS = """
function copyTagCmd(caseId, btn) {
  const cmd = 'python3 tools/tag_case.py ' + caseId + ' ';
  navigator.clipboard.writeText(cmd).then(function() {
    const old = btn.textContent;
    btn.textContent = '✓ copied';
    setTimeout(function() { btn.textContent = old; }, 1100);
  });
}
"""


def _tile_html(row: dict, thumb_url: str, src_label: str,
               tag_classes: dict[str, str], wav_path: Path,
               wsjtx_per_decode: list[dict],
               peak_event: dict | None) -> str:
    """Render one tile.  ``wsjtx_per_decode`` and ``peak_event`` come from
    JSON blobs in the index CSV (parsed by caller).  We surface per-decode
    detail and, when MAP144 produced launches but no decode, the peak
    event's channel / detector / metric so the operator sees *what*
    triggered.
    """
    case_id = row['case_id']
    period  = row['period_ts'].replace('T', ' ').replace('+00:00', '')
    freq    = row.get('freq_khz') or ''
    n_wd    = int(row.get('n_wsjtx_decodes') or 0)
    n_md    = int(row.get('n_map144_decodes') or 0)
    n_ev    = int(row.get('n_map144_events')  or 0)
    sq      = row.get('max_sq_metric_db') or ''
    coh     = row.get('median_coherence_h') or ''
    nc2s    = int(row.get('max_n_chans_2s') or 0)
    tags    = sorted(_row_tags(row))
    manual_tags = set((row.get('manual_tags') or '').split(';')) - {''}
    note    = row.get('manual_note') or ''

    # ── WSJT-X line (header) + one submsg per decode ─────────────────────────
    if n_wd == 0:
        wsjtx_block = '<div class="msg miss">WSJT-X×0  (none)</div>'
    else:
        wsjtx_block = f'<div class="msg got">WSJT-X×{n_wd}</div>'
        for d in wsjtx_per_decode:
            wsjtx_block += (
                f'<div class="submsg">'
                f'snr {d["snr"]:+d}  dt {d["dt"]:+.1f}  af {d["af"]}  '
                f'{html.escape(d["msg"])}'
                f'</div>'
            )

    # ── MAP144 line (header) + peak-event submsg ─────────────────────────────
    if n_md:
        # Show decoded messages; peak event detail probably less interesting here
        map144_block = f'<div class="msg got">MAP144×{n_md}d/{n_ev}ev  {html.escape(row.get("map144_msgs") or "")}</div>'
    elif n_ev:
        map144_block = f'<div class="msg miss">MAP144×0d/{n_ev}ev</div>'
        if peak_event:
            t_sec = peak_event.get('t_sec')
            ch    = peak_event.get('ch_signed')
            det   = peak_event.get('det')
            sqv   = peak_event.get('sq')
            pcoh  = peak_event.get('coh')
            pieces = []
            if t_sec is not None: pieces.append(f'peak t={t_sec:.2f}')
            if ch is not None:    pieces.append(f'ch{ch:+d}')
            if det:               pieces.append(f'det={det}')
            if sqv is not None:   pieces.append(f'sq={sqv:.1f}')
            if pcoh is not None:  pieces.append(f'coh={pcoh:.2f}')
            if pieces:
                map144_block += '<div class="submsg">' + '  '.join(pieces) + '</div>'
    else:
        map144_block = '<div class="msg">MAP144×0d/0ev  (no launches)</div>'

    # ── Tag chips (manual ones styled distinctly) ────────────────────────────
    chip_html = []
    for t in tags:
        if t in manual_tags:
            cls = 'manual'
        else:
            cls = _chip_class(tag_classes.get(t, 'any'))
        chip_html.append(
            f'<span class="chip {cls}">{html.escape(t)}</span>'
        )
    chips = ' '.join(chip_html)

    # ── Aggregate metrics line ───────────────────────────────────────────────
    metrics = []
    if sq:        metrics.append(f"sq_max={sq}")
    if coh:       metrics.append(f"coh_med={coh}")
    if nc2s >= 4: metrics.append(f"n2s={nc2s}")
    metrics_line = ('<div class="metrics">' + '  '.join(metrics) + '</div>') if metrics else ''

    note_block = ''
    if note:
        note_block = f'<div class="note">{html.escape(note)}</div>'

    # ── Header line: case_id + tag-helper button + freq ──────────────────────
    hdr = (
        '<div class="hdrline">'
        f'<code class="caseid" title="{html.escape(case_id)}">{html.escape(case_id)}</code>'
        f'<button class="tagbtn" onclick="copyTagCmd(\'{html.escape(case_id)}\', this); '
        f'event.preventDefault(); event.stopPropagation();">tag</button>'
        '</div>'
    )

    # Period + freq subhead
    subhdr = (
        f'<div class="metrics">{html.escape(period)}'
        + (f'  <span class="freq">{freq}kHz</span>' if freq else '')
        + '</div>'
    )

    return (
        '<div class="tile">'
        f'<a href="file://{html.escape(str(wav_path))}" title="{html.escape(str(wav_path))}">'
        f'<div class="imgwrap"><img src="{html.escape(thumb_url)}" loading="lazy">'
        f'<span class="src">{html.escape(src_label)}</span></div>'
        '</a>'
        f'{hdr}{subhdr}'
        f'{wsjtx_block}{map144_block}'
        f'{metrics_line}{note_block}'
        f'<div class="chips">{chips}</div>'
        '</div>'
    )


def _render_html(rows: list[dict], thumbs_dir: Path, out_html: Path,
                 tag_classes: dict[str, str], filter_desc: str,
                 ) -> tuple[int, int, int]:
    """Returns (n_tiles, n_skip_no_wav, n_render_failed)."""
    tiles = []
    n_no_wav = n_fail = 0
    for r in rows:
        wav, label = _select_wav(r)
        if wav is None:
            n_no_wav += 1
            continue
        tp = _ensure_thumb(thumbs_dir, wav)
        if tp is None:
            n_fail += 1
            continue
        # Parse per-decode + peak-event JSON blobs (empty-string-safe)
        try:
            wsjtx_per = json.loads(r.get('wsjtx_decodes_json') or '[]')
        except Exception:
            wsjtx_per = []
        try:
            peak = json.loads(r.get('map144_peak_json') or 'null')
        except Exception:
            peak = None
        # Absolute file:// URL — robust to moving the HTML
        thumb_url = f"file://{tp}"
        tiles.append(_tile_html(r, thumb_url, label, tag_classes, wav,
                                wsjtx_per, peak))

    head = (
        '<h1>Corpus gallery</h1>'
        f'<div class="meta">{html.escape(filter_desc)}<br>'
        f'{len(tiles)} tiles'
        + (f' &nbsp; ({n_no_wav} skipped — no WAV)' if n_no_wav else '')
        + (f' &nbsp; ({n_fail} render failed)' if n_fail else '')
        + f' &nbsp; generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}'
        + '</div>'
    )
    body = '<div class="grid">' + '\n'.join(tiles) + '</div>'
    doc = (
        '<!doctype html><html><head>'
        '<meta charset="utf-8">'
        f'<title>{html.escape(filter_desc[:60])} — gallery</title>'
        f'<style>{_CSS}</style>'
        f'<script>{_JS}</script>'
        '</head><body>'
        f'{head}{body}'
        '</body></html>'
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding='utf-8')
    return len(tiles), n_no_wav, n_fail


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index',   type=Path, default=DEFAULT_INDEX,
                    help='Corpus index CSV (default: %(default)s)')
    ap.add_argument('--thumbs',  type=Path, default=DEFAULT_THUMBS,
                    help='Thumbnail cache directory')
    ap.add_argument('--out',     type=Path,
                    default=_REPO_ROOT / 'scratch' / 'galleries' / 'corpus_gallery.html',
                    help='Output HTML path.  Default writes to scratch/galleries/ '
                         '(in-tree, gitignored).  scratch/ is the canonical '
                         'project-local scratch area; see scratch/README.md.')
    ap.add_argument('--tag',     action='append', default=[],
                    help='Require this tag (repeat for AND).  '
                         'See tests/corpus/tags.py for vocabulary.')
    ap.add_argument('--any-tag', action='append', default=[],
                    help='Match if any of these tags present (OR).')
    ap.add_argument('--max',     type=int, default=500,
                    help='Cap tile count (default 500)')
    ap.add_argument('--sort',    default='snr_asc',
                    choices=('time', 'snr_asc', 'snr_desc',
                             'sq_asc', 'sq_desc', 'events_desc'),
                    help='Tile order.  Default snr_asc puts weakest signals '
                         'first (most actionable).  time = chronological.')
    ap.add_argument('--hours-back', type=float, default=None,
                    help='Only show periods from the last N UTC hours.  Use to '
                         'scope review to a recent run (e.g. --hours-back 16 '
                         'for last night/this morning).')
    ap.add_argument('--clear-cache', action='store_true',
                    help='Delete cached thumbnails before rendering')
    ap.add_argument('--include-no-wav', action='store_true',
                    help='Do not pre-filter rows that have no WAV (default: '
                         'drop them so --max counts renderable rows only)')
    args = ap.parse_args(argv)

    if args.clear_cache and args.thumbs.is_dir():
        n = 0
        for f in args.thumbs.glob('*.png'):
            f.unlink(); n += 1
        print(f'cleared {n} cached thumbnails', file=sys.stderr)

    if not args.index.is_file():
        print(f'error: index not found at {args.index}; '
              f'run tools/select_stress_corpus.py first', file=sys.stderr)
        return 1

    with args.index.open() as fh:
        rows = list(csv.DictReader(fh))
    sel, n_tag, n_drop = _filter_rows(rows, args.tag, args.any_tag, args.max,
                                      require_wav=not args.include_no_wav,
                                      sort_key=args.sort,
                                      hours_back=args.hours_back)

    # Cleaner title parts ("require" and "any" labels make the raw dict
    # form readable in the tab title and meta line)
    parts = []
    if args.tag:     parts.append('all of [' + ', '.join(args.tag)     + ']')
    if args.any_tag: parts.append('any of [' + ', '.join(args.any_tag) + ']')
    sel_desc = ' & '.join(parts) if parts else 'all rows'
    filter_desc = (
        f"{sel_desc}  ·  sort={args.sort}  ·  "
        f"matched {n_tag}, {n_drop} no-wav, showing {len(sel)}/{args.max}"
    )

    print(f'corpus gallery', file=sys.stderr)
    print(f'  index         : {args.index}', file=sys.stderr)
    print(f'  tag matched   : {n_tag}', file=sys.stderr)
    print(f'  dropped no-wav: {n_drop}', file=sys.stderr)
    print(f'  rendering     : {len(sel)} (cap {args.max})', file=sys.stderr)
    print(f'  thumbs        : {args.thumbs}', file=sys.stderr)
    print(f'  out           : {args.out}', file=sys.stderr)

    tag_classes = _load_tag_classes()
    n_tiles, n_no_wav, n_fail = _render_html(sel, args.thumbs, args.out,
                                             tag_classes, filter_desc)
    print(f'wrote {n_tiles} tiles to {args.out} '
          f'({n_no_wav} no-wav, {n_fail} render-failed)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
