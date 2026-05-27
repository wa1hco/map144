#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-case keyboard-driven corpus review tool.

A tiny local HTTP server + browser UI for fast per-case annotation.
Reviews one period at a time at large thumbnail size, supports keyboard
navigation (← → for prev/next), single-letter tag shortcuts (see
:mod:`tests.corpus.shortcuts`), free-form notes, and ad-hoc new tag
creation.  Annotations persist to ``tests/corpus/manual_tags.jsonl``
exactly as ``tools/tag_case.py`` writes them.

Usage
-----
::

    # Review the mission-gap bucket (same filter syntax as gallery_corpus.py)
    python3 tools/review_corpus.py --tag weak --tag wsjtx_only

    # Bound the set
    python3 tools/review_corpus.py --tag canonical_strong --max 50

Server listens on 127.0.0.1:8765 by default; opens the browser
automatically.  Ctrl-C to stop.

Keyboard while focus is OUTSIDE an input field
-----------------------------------------------
* ``→`` / ``Space``  next case
* ``←``              previous case
* ``a``-``z`` / ``A``-``Z``  toggle the tag bound to that letter
                             (see shortcuts panel on the page)
* ``n``              focus the note input
* ``t``              focus the new-tag input

Single keypress = single annotation = one POST = one line appended to
``manual_tags.jsonl``.  Re-pressing the same letter toggles
(``+tag`` → ``-tag`` → ``+tag``).  Multi-session resume works: re-open
the same filter and previously-annotated tags are pre-loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the existing thumbnail renderer + cache-key helpers
from tools.gallery_corpus import (                          # noqa: E402
    _render_thumbnail, _cache_key, _row_tags, _filter_rows,
    _load_tag_classes,
)
from tests.corpus.shortcuts import SHORTCUTS                # noqa: E402
from tests.corpus.tags     import TAGS                      # noqa: E402

DEFAULT_INDEX  = _REPO_ROOT / "tests" / "corpus" / "index.csv"
DEFAULT_THUMBS = _REPO_ROOT / "tests" / "corpus" / "thumbs"
DEFAULT_MANUAL = _REPO_ROOT / "tests" / "corpus" / "manual_tags.jsonl"

# Larger detail-view thumb size — gallery thumbs (240x140) are too small
# to read while annotating.  Stored in a size-suffixed cache filename so
# both caches coexist.
REVIEW_W_PX = 720
REVIEW_H_PX = 360

# Append-only log validation (mirrors tag_case.py)
_CASE_RE = re.compile(r'^period_\d{6}_\d{6}$')
_TAG_RE  = re.compile(r'^[a-z][a-z0-9_]*$')

# Server state — populated by main(), read by request handler
STATE: dict = {
    'rows':         [],            # filtered, sorted list of case rows (dicts)
    'idx_by_id':    {},            # case_id -> index in rows
    'thumbs_dir':   DEFAULT_THUMBS,
    'manual_log':   DEFAULT_MANUAL,
    'manual_state': {},            # case_id -> {'add': set, 'rem': set, 'note': str}
    'manual_lock':  threading.Lock(),
    'tag_classes':  {},            # tag_name -> pass_when
    'filter_desc':  '',
}


# ── manual_tags.jsonl replay (mirrors select_stress_corpus.load_manual_tags) ─

def _load_manual_state(path: Path) -> dict[str, dict]:
    """Replay manual_tags.jsonl into {case_id: {add:set, rem:set, note:str}}."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            case = rec.get('case')
            if not case:
                continue
            ent = out.setdefault(case, {'add': set(), 'rem': set(), 'note': ''})
            for op in rec.get('ops', []):
                if not op or op[0] not in '+-':
                    continue
                tag = op[1:]
                if op[0] == '+':
                    ent['add'].add(tag); ent['rem'].discard(tag)
                else:
                    ent['rem'].add(tag); ent['add'].discard(tag)
            note = rec.get('note') or ''
            if note:
                ent['note'] = note
    return out


def _append_manual(case_id: str, ops: list[str], note: str) -> None:
    """Append one record to manual_tags.jsonl (same format as tag_case.py)."""
    rec = {
        'ts':   datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'case': case_id,
        'ops':  ops,
        'note': note,
    }
    with STATE['manual_lock']:
        STATE['manual_log'].parent.mkdir(parents=True, exist_ok=True)
        with STATE['manual_log'].open('a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # Update in-memory state
        ent = STATE['manual_state'].setdefault(
            case_id, {'add': set(), 'rem': set(), 'note': ''})
        for op in ops:
            if op and op[0] in '+-':
                tag = op[1:]
                if op[0] == '+':
                    ent['add'].add(tag); ent['rem'].discard(tag)
                else:
                    ent['rem'].add(tag); ent['add'].discard(tag)
        if note:
            ent['note'] = note


# ── Thumbnail (review-size, cache key includes dimensions) ───────────────────

def _review_thumb_path(wav_path: Path) -> Path | None:
    key = _cache_key(wav_path)
    if key is None:
        return None
    return STATE['thumbs_dir'] / f"{key}_{REVIEW_W_PX}x{REVIEW_H_PX}.png"


def _ensure_review_thumb(wav_path: Path) -> Path | None:
    p = _review_thumb_path(wav_path)
    if p is None:
        return None
    if p.is_file():
        return p
    if _render_thumbnail(wav_path, p, REVIEW_W_PX, REVIEW_H_PX):
        return p
    return None


def _select_wav(row: dict) -> tuple[Path | None, str]:
    """Prefer WSJT-X save WAV, fall back to first MAP144 detection WAV."""
    if row.get('wsjtx_wav'):
        return Path(row['wsjtx_wav']), 'wsjtx'
    ms = row.get('map144_wavs') or ''
    if ms:
        first = ms.split(';', 1)[0]
        if first:
            return Path(first), 'map144'
    return None, ''


# ── HTML / CSS / JS (single page) ────────────────────────────────────────────

_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Corpus review</title>
<style>
body { font-family: sans-serif; font-size: 12pt; background: #fafafa; color: #222;
       margin: 0; padding: 12px; }
h1 { font-size: 14pt; margin: 0 0 4px 0; font-weight: 500; color: #555; }
.meta { color: #888; font-size: 10pt; margin-bottom: 8px; }
.bar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.bar button { font-family: monospace; font-size: 11pt; padding: 4px 10px;
              border: 1px solid #ccc; background: #fff; border-radius: 3px;
              cursor: pointer; }
.bar button:hover { background: #f0f0f0; }
.progress { font-family: monospace; font-size: 11pt; color: #666; }
.case { background: #fff; border: 1px solid #ddd; border-radius: 4px;
        padding: 14px; }
.case .hdrline { display: flex; justify-content: space-between;
                 align-items: baseline; margin-bottom: 6px; }
.case .caseid { font-family: monospace; font-size: 12pt; color: #333;
                user-select: all; }
.case .freq { font-family: monospace; font-size: 11pt; color: #888; }
.case img { display: block; width: 720px; max-width: 100%; height: auto;
            image-rendering: pixelated; border: 1px solid #eee;
            background: #000;
            filter: brightness(var(--bright, 1)) contrast(var(--contrast, 1)); }
.sliders { display: flex; gap: 16px; align-items: center; margin-bottom: 8px;
           padding: 6px 10px; background: #f4f4f4; border: 1px solid #e5e5e5;
           border-radius: 4px; font-family: monospace; font-size: 10pt;
           color: #555; }
.sliders label { display: flex; gap: 6px; align-items: center; }
.sliders input[type=range] { width: 140px; }
.sliders .val { display: inline-block; min-width: 36px; text-align: right;
                color: #335; }
.sliders button { font-family: monospace; font-size: 10pt; padding: 2px 8px;
                  border: 1px solid #ccc; background: #fff; border-radius: 3px;
                  cursor: pointer; }
.sliders button:hover { background: #f0f0f0; }
.case .meta-rows { margin-top: 8px; font-family: monospace; font-size: 11pt;
                   line-height: 1.45; }
.case .got  { color: #275; }
.case .miss { color: #a52; }
.case .submsg { color: #888; margin-left: 14px; font-size: 10.5pt; }
.chips { margin-top: 8px; }
.chip { display: inline-block; background: #eee; color: #444; font-size: 9.5pt;
        padding: 1px 7px; border-radius: 10px; margin: 2px 3px 0 0;
        font-family: monospace; }
.chip.catch  { background: #e6f4dd; color: #275; }
.chip.fp     { background: #faeaea; color: #944; }
.chip.any    { background: #eee;    color: #555; }
.chip.manual { background: #e8e0f4; color: #524080;
               box-shadow: inset 0 0 0 1px #cfc2e6; }
.shortcuts { margin-top: 12px; padding: 8px; background: #f7f7f0;
             border: 1px solid #e3e0c8; border-radius: 4px;
             font-family: monospace; font-size: 10pt; color: #555;
             columns: 4; column-gap: 14px; }
.shortcuts .row { break-inside: avoid; line-height: 1.5; }
.shortcuts .key { display: inline-block; min-width: 16px; color: #804;
                  font-weight: bold; }
.shortcuts .row.applied { background: #e8e0f4; padding: 0 4px; border-radius: 3px; }
.input-row { margin-top: 8px; display: flex; gap: 6px; align-items: center; }
.input-row label { font-family: monospace; font-size: 10.5pt; color: #555;
                   min-width: 72px; }
.input-row input { font-family: monospace; font-size: 11pt; padding: 3px 6px;
                   border: 1px solid #ccc; border-radius: 3px; flex: 1; }
.note-display { margin-top: 6px; font-family: serif; font-style: italic;
                font-size: 11pt; color: #444; padding: 4px 8px;
                background: #fafbe8; border-left: 2px solid #d4d090; }
.help { color: #888; font-size: 9.5pt; margin-top: 6px; }
.kbd { display: inline-block; padding: 0 4px; border: 1px solid #bbb;
       border-radius: 2px; background: #f4f4f4; font-family: monospace;
       font-size: 9.5pt; color: #444; }
</style>
</head><body>

<h1>Corpus review</h1>
<div class="meta" id="filter-desc"></div>

<div class="bar">
  <button onclick="prev()">← prev</button>
  <button onclick="next()">next →</button>
  <span class="progress" id="progress"></span>
  <span style="flex:1"></span>
  <span class="help">
    <span class="kbd">←</span> <span class="kbd">→</span> nav &nbsp;
    keys toggle tags &nbsp;
    <span class="kbd">n</span> note &nbsp;
    <span class="kbd">t</span> new tag &nbsp;
    <span class="kbd">Esc</span> exit input
  </span>
</div>

<div class="sliders">
  <label>brightness
    <input id="brightness" type="range" min="0.4" max="2.5" step="0.05" value="1">
    <span class="val" id="brightness-val">1.00</span>
  </label>
  <label>contrast
    <input id="contrast" type="range" min="0.4" max="2.5" step="0.05" value="1">
    <span class="val" id="contrast-val">1.00</span>
  </label>
  <button onclick="resetSliders()">reset</button>
  <span style="color:#888;font-size:9.5pt">(applies to all cases this session)</span>
</div>

<div class="case" id="case"><em>Loading…</em></div>

<script>
let cases = [];           // [{case_id, ...}, ...]
let cur = 0;
let vocab = {};           // {tag_name: pass_when}
let shortcuts = {};       // {letter: tag_name}
let cached = {};          // case_id -> full case detail (lazy fetched)

async function loadInit() {
  const r = await fetch('/api/init');
  const j = await r.json();
  cases = j.cases;
  vocab = j.vocab;
  shortcuts = j.shortcuts;
  document.getElementById('filter-desc').textContent = j.filter_desc;
  if (cases.length === 0) {
    document.getElementById('case').innerHTML = '<em>No cases match the filter.</em>';
    return;
  }
  renderCurrent();
}

async function renderCurrent() {
  if (cases.length === 0) return;
  if (cur < 0) cur = 0;
  if (cur >= cases.length) cur = cases.length - 1;
  document.getElementById('progress').textContent =
    `${cur + 1} / ${cases.length}`;
  const caseId = cases[cur].case_id;
  let detail = cached[caseId];
  if (!detail) {
    const r = await fetch('/api/case/' + caseId);
    detail = await r.json();
    cached[caseId] = detail;
  }
  renderCase(detail);
}

function chip(tag, classExtra) {
  const cls = (classExtra ? classExtra + ' ' : '') +
              (vocab[tag] === 'decoded' ? 'catch' :
               vocab[tag] === 'no_decode' ? 'fp' : 'any');
  return `<span class="chip ${cls}">${escapeHtml(tag)}</span>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderCase(d) {
  const tagsApplied = new Set([...(d.auto_tags||[]), ...(d.manual_add||[])].filter(t => !(d.manual_rem||[]).includes(t)));

  // WSJT-X block
  let wsjtxBlock = '';
  if (d.wsjtx_decodes && d.wsjtx_decodes.length) {
    wsjtxBlock = `<div class="got">WSJT-X×${d.wsjtx_decodes.length}</div>`;
    for (const dec of d.wsjtx_decodes) {
      wsjtxBlock += `<div class="submsg">snr ${dec.snr>=0?'+':''}${dec.snr}  dt ${dec.dt>=0?'+':''}${dec.dt.toFixed(2)}  af ${dec.af}  ${escapeHtml(dec.msg)}</div>`;
    }
  } else {
    wsjtxBlock = '<div class="miss">WSJT-X×0  (none)</div>';
  }

  // MAP144 block
  let mapBlock = '';
  if (d.n_map144_decodes > 0) {
    mapBlock = `<div class="got">MAP144×${d.n_map144_decodes}d/${d.n_map144_events}ev  ${escapeHtml(d.map144_msgs||'')}</div>`;
  } else if (d.n_map144_events > 0) {
    mapBlock = `<div class="miss">MAP144×0d/${d.n_map144_events}ev</div>`;
    if (d.peak_event) {
      const p = d.peak_event;
      const pieces = [];
      if (p.t_sec != null) pieces.push(`peak t=${p.t_sec.toFixed(2)}`);
      if (p.ch_signed != null) pieces.push(`ch${p.ch_signed>=0?'+':''}${p.ch_signed}`);
      if (p.det) pieces.push(`det=${p.det}`);
      if (p.sq != null) pieces.push(`sq=${p.sq.toFixed(1)}`);
      if (p.coh != null) pieces.push(`coh=${p.coh.toFixed(2)}`);
      mapBlock += `<div class="submsg">${pieces.join('  ')}</div>`;
    }
  } else {
    mapBlock = `<div>MAP144×0d/0ev  (no launches)</div>`;
  }

  // Chips: auto-applied first, then manual
  const chipsHtml = Array.from(tagsApplied).sort().map(t => {
    const isManual = (d.manual_add||[]).includes(t);
    return chip(t, isManual ? 'manual' : '');
  }).join(' ');

  // Shortcuts panel
  let shortHtml = '';
  for (const [key, tag] of Object.entries(shortcuts)) {
    const applied = tagsApplied.has(tag);
    shortHtml += `<div class="row${applied?' applied':''}"><span class="key">${escapeHtml(key)}</span>${escapeHtml(tag)}</div>`;
  }

  const noteBlock = d.note ?
    `<div class="note-display">${escapeHtml(d.note)}</div>` : '';

  document.getElementById('case').innerHTML = `
    <div class="hdrline">
      <code class="caseid">${escapeHtml(d.case_id)}</code>
      <span class="freq">${escapeHtml(d.period_ts || '')}  ·  ${d.freq_khz || ''}kHz</span>
    </div>
    <a href="file://${encodeURI(d.wav_path)}"><img src="/thumb/${encodeURIComponent(d.case_id)}"></a>
    <div class="meta-rows">${wsjtxBlock}${mapBlock}</div>
    <div class="chips">${chipsHtml}</div>
    ${noteBlock}
    <div class="shortcuts">${shortHtml}</div>
    <div class="input-row">
      <label>note:</label>
      <input id="note-input" type="text" placeholder="(free-form, Enter to save)" value="${escapeHtml(d.note||'')}">
    </div>
    <div class="input-row">
      <label>new tag:</label>
      <input id="newtag-input" type="text" placeholder="(snake_case, Enter to add as +tag)">
    </div>
  `;

  // Wire inputs after re-render.  Enter = apply AND advance to next case
  // (Shift-Enter to apply without advancing — for multi-tag cases).
  // Escape blurs without committing.
  const noteEl = document.getElementById('note-input');
  noteEl.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      saveNote(noteEl.value);
      noteEl.blur();
      if (!ev.shiftKey) next();
    }
    if (ev.key === 'Escape') { ev.preventDefault(); noteEl.blur(); }
  });
  const newtagEl = document.getElementById('newtag-input');
  newtagEl.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      const v = newtagEl.value;
      newtagEl.value = '';
      newtagEl.blur();
      addNewTag(v);              // calls toggleTag, which re-renders
      if (!ev.shiftKey) next();
    }
    if (ev.key === 'Escape') { ev.preventDefault(); newtagEl.blur(); }
  });
}

function next() { if (cur < cases.length - 1) { cur++; renderCurrent(); } }
function prev() { if (cur > 0) { cur--; renderCurrent(); } }

async function toggleTag(tag) {
  const caseId = cases[cur].case_id;
  const d = cached[caseId];
  if (!d) return;
  const applied = new Set([...(d.auto_tags||[]), ...(d.manual_add||[])].filter(t => !(d.manual_rem||[]).includes(t)));
  // If currently applied, queue a -tag (overrides any +tag).  If not, +tag.
  const op = applied.has(tag) ? '-' + tag : '+' + tag;
  await fetch('/api/tag', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({case_id: caseId, ops: [op], note: ''}),
  });
  // Invalidate cache for this case so re-render reflects new state
  delete cached[caseId];
  renderCurrent();
}

async function saveNote(text) {
  const caseId = cases[cur].case_id;
  await fetch('/api/tag', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({case_id: caseId, ops: [], note: text}),
  });
  delete cached[caseId];
  renderCurrent();
}

async function addNewTag(tag) {
  tag = tag.trim();
  if (!tag) return;
  if (!/^[a-z][a-z0-9_]*$/.test(tag)) {
    alert('tag name must match [a-z][a-z0-9_]*');
    return;
  }
  toggleTag(tag);
}

document.addEventListener('keydown', ev => {
  // Skip if focus is in an input
  const tag = (document.activeElement || {}).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  // Skip if a modifier is held — don't eat Ctrl-C / Cmd-C / Ctrl-A etc.
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

  if (ev.key === 'ArrowRight' || ev.key === ' ') { ev.preventDefault(); next(); return; }
  if (ev.key === 'ArrowLeft')                    { ev.preventDefault(); prev(); return; }
  if (ev.key === 'n') {
    ev.preventDefault();
    const el = document.getElementById('note-input');
    if (el) el.focus();
    return;
  }
  if (ev.key === 't') {
    ev.preventDefault();
    const el = document.getElementById('newtag-input');
    if (el) el.focus();
    return;
  }
  // Letter shortcuts
  if (shortcuts[ev.key]) {
    ev.preventDefault();
    toggleTag(shortcuts[ev.key]);
  }
});

// ── Brightness / contrast sliders ────────────────────────────────────────
function bindSlider(id, varName) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + '-val');
  const apply = () => {
    document.documentElement.style.setProperty('--' + varName, el.value);
    out.textContent = parseFloat(el.value).toFixed(2);
  };
  el.addEventListener('input', apply);
  apply();
}
function resetSliders() {
  for (const [id, v] of [['brightness', 1], ['contrast', 1]]) {
    const el = document.getElementById(id);
    el.value = v;
    el.dispatchEvent(new Event('input'));
  }
}
bindSlider('brightness', 'bright');
bindSlider('contrast',   'contrast');

loadInit();
</script>
</body></html>
"""


# ── HTTP handler ─────────────────────────────────────────────────────────────

def _row_to_summary(r: dict) -> dict:
    return {
        'case_id':   r['case_id'],
        'period_ts': r['period_ts'],
    }


def _row_to_detail(r: dict) -> dict:
    """Full case detail for the UI."""
    wav, src = _select_wav(r)
    try:
        wsjtx_decodes = json.loads(r.get('wsjtx_decodes_json') or '[]')
    except Exception:
        wsjtx_decodes = []
    try:
        peak = json.loads(r.get('map144_peak_json') or 'null')
    except Exception:
        peak = None
    auto_tags = sorted(_row_tags(r) - set((r.get('manual_tags') or '').split(';')) - {''})
    ent = STATE['manual_state'].get(r['case_id'], {'add': set(), 'rem': set(), 'note': ''})
    return {
        'case_id':           r['case_id'],
        'period_ts':         r['period_ts'].replace('T', ' ').replace('+00:00', ''),
        'freq_khz':          r.get('freq_khz', ''),
        'wav_path':          str(wav) if wav else '',
        'wav_source':        src,
        'n_wsjtx_decodes':   int(r.get('n_wsjtx_decodes') or 0),
        'n_map144_decodes':  int(r.get('n_map144_decodes') or 0),
        'n_map144_events':   int(r.get('n_map144_events') or 0),
        'wsjtx_decodes':     wsjtx_decodes,
        'map144_msgs':       r.get('map144_msgs', ''),
        'peak_event':        peak,
        'auto_tags':         auto_tags,
        'manual_add':        sorted(ent['add']),
        'manual_rem':        sorted(ent['rem']),
        'note':              ent['note'] or (r.get('manual_note') or ''),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass    # silence default logging

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json; charset=utf-8')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text):
        body = text.encode('utf-8')
        self.send_response(200)
        self.send_header('content-type', 'text/html; charset=utf-8')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header('content-type', 'image/png')
        self.send_header('content-length', str(len(data)))
        self.send_header('cache-control', 'private, max-age=3600')
        self.end_headers()
        self.wfile.write(data)

    def _send_err(self, status, msg):
        body = msg.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'text/plain; charset=utf-8')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        if path == '/':
            self._send_html(_HTML); return
        if path == '/api/init':
            self._send_json({
                'cases':       [_row_to_summary(r) for r in STATE['rows']],
                'vocab':       STATE['tag_classes'],
                'shortcuts':   SHORTCUTS,
                'filter_desc': STATE['filter_desc'],
            }); return
        if path.startswith('/api/case/'):
            cid = path[len('/api/case/'):]
            idx = STATE['idx_by_id'].get(cid)
            if idx is None:
                self._send_err(404, f'unknown case: {cid}'); return
            self._send_json(_row_to_detail(STATE['rows'][idx])); return
        if path.startswith('/thumb/'):
            cid = path[len('/thumb/'):]
            idx = STATE['idx_by_id'].get(cid)
            if idx is None:
                self._send_err(404, 'no such case'); return
            wav, _ = _select_wav(STATE['rows'][idx])
            if wav is None:
                self._send_err(404, 'no wav'); return
            tp = _ensure_review_thumb(wav)
            if tp is None or not tp.is_file():
                self._send_err(500, 'thumb render failed'); return
            self._send_png(tp); return
        self._send_err(404, 'not found')

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        n = int(self.headers.get('content-length', '0') or '0')
        raw = self.rfile.read(n) if n > 0 else b''
        if path == '/api/tag':
            try:
                body = json.loads(raw.decode('utf-8'))
            except Exception:
                self._send_err(400, 'bad JSON'); return
            case_id = body.get('case_id', '')
            ops     = body.get('ops', [])
            note    = body.get('note', '') or ''
            if not _CASE_RE.match(case_id):
                self._send_err(400, f'bad case_id {case_id!r}'); return
            # Validate ops
            for op in ops:
                if not op or op[0] not in '+-':
                    self._send_err(400, f'bad op {op!r}'); return
                if not _TAG_RE.match(op[1:]):
                    self._send_err(400, f'bad tag {op[1:]!r}'); return
            if not ops and not note:
                self._send_err(400, 'no-op'); return
            _append_manual(case_id, ops, note)
            self._send_json({'ok': True}); return
        self._send_err(404, 'not found')


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index',   type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--thumbs',  type=Path, default=DEFAULT_THUMBS)
    ap.add_argument('--manual-log', type=Path, default=DEFAULT_MANUAL)
    ap.add_argument('--tag',     action='append', default=[],
                    help='Require this tag (repeat for AND)')
    ap.add_argument('--any-tag', action='append', default=[],
                    help='Match if any of these tags present (OR)')
    ap.add_argument('--max',     type=int, default=500)
    ap.add_argument('--sort',    default='snr_asc',
                    choices=('time','snr_asc','snr_desc','sq_asc','sq_desc','events_desc'))
    ap.add_argument('--hours-back', type=float, default=None,
                    help='Only show periods from the last N UTC hours (e.g. '
                         '--hours-back 16 for last night/this morning).')
    ap.add_argument('--port',    type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true',
                    help="Don't open the browser automatically")
    args = ap.parse_args(argv)

    if not args.index.is_file():
        print(f'error: index not found at {args.index}', file=sys.stderr)
        return 1

    with args.index.open() as fh:
        rows = list(csv.DictReader(fh))
    sel, n_tag, n_drop = _filter_rows(rows, args.tag, args.any_tag, args.max,
                                      require_wav=True, sort_key=args.sort,
                                      hours_back=args.hours_back)

    parts = []
    if args.tag:     parts.append('all of [' + ', '.join(args.tag)     + ']')
    if args.any_tag: parts.append('any of [' + ', '.join(args.any_tag) + ']')
    sel_desc = ' & '.join(parts) if parts else 'all rows'
    filter_desc = (
        f"{sel_desc}  ·  sort={args.sort}  ·  "
        f"matched {n_tag}, {n_drop} no-wav, showing {len(sel)}/{args.max}"
    )

    STATE['rows']         = sel
    STATE['idx_by_id']    = {r['case_id']: i for i, r in enumerate(sel)}
    STATE['thumbs_dir']   = args.thumbs
    STATE['manual_log']   = args.manual_log
    STATE['manual_state'] = _load_manual_state(args.manual_log)
    STATE['tag_classes']  = {t.name: t.pass_when for t in TAGS}
    STATE['filter_desc']  = filter_desc

    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'review server on {url}  ·  {filter_desc}', file=sys.stderr)
    print(f'  index       : {args.index}', file=sys.stderr)
    print(f'  manual log  : {args.manual_log}', file=sys.stderr)
    print(f'  thumbs      : {args.thumbs}', file=sys.stderr)
    print(f'  press Ctrl-C to stop', file=sys.stderr)

    if not args.no_browser:
        try:    webbrowser.open(url)
        except: pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nbye.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
