# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Controlled-vocabulary tags for the DSP-regression corpus.

A *case* in the corpus is a 15-s period.  Each case carries zero or more
tags from this module.  ``tools/select_stress_corpus.py`` applies these
predicates while building ``tests/corpus/index.csv``; the audit tool
``tools/audit_taxonomy.py`` reads the index and reports per-tag counts
plus tag co-occurrence so the taxonomy can be refined without moving
files.

Each tag is declared via :class:`Tag` with:

* ``name``         — short snake_case identifier; used as CSV column header
* ``description``  — one line of intent (what stress regime this targets)
* ``pass_when``    — ``"decoded"`` (catch-rate test) or ``"no_decode"``
                     (false-positive test) or ``"any"`` (orthogonal/source)
* ``predicate``    — ``Callable[[Period], bool]`` or ``None`` if the
                     predicate isn't yet implementable from available data
                     (audit tool will flag tags with ``None`` predicates).

The :class:`Period` aggregate passed to predicates is built per-period
by ``select_stress_corpus.py`` and described in its docstring.

Restructuring policy
--------------------
Tags are expected to change over time.  Renaming or merging tags should
not require moving WAVs — re-running ``select_stress_corpus.py`` rebuilds
``index.csv`` from current predicates.  Old baselines need a tag-mapping
note when renames happen (see ``M5`` baseline plan).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable


# ── Period aggregate (what predicates receive) ────────────────────────────────

@dataclass
class Period:
    """A single 15-s window's worth of corpus inputs.

    Built by ``tools/select_stress_corpus.py``.  Fields:

    * ``period_ts``       — UTC datetime, floored to 15-s boundary
    * ``wsjtx_wav``       — path to ``yymmdd_HHMMSS.wav`` if present, else None
    * ``map144_wavs``     — list of MAP144 detection WAV paths in this period
    * ``wsjtx_decodes``   — list of parsed ALL.TXT MSK144 entries
                            (keys: snr, dt, af, msg, raw_ts)
    * ``map144_events``   — list of launches.jsonl rows for this period
                            (full schema; see launches.jsonl docstring)

    Predicates should treat empty lists / None fields as "no data on that
    side" rather than asserting.
    """
    period_ts: object  # datetime — quoted to avoid circular import
    wsjtx_wav: object | None
    map144_wavs: list
    wsjtx_decodes: list[dict]
    map144_events: list[dict]


# ── Tag declaration ───────────────────────────────────────────────────────────

@dataclass
class Tag:
    name: str
    description: str
    pass_when: str                          # "decoded" | "no_decode" | "any"
    predicate: Callable[[Period], bool] | None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decoded_events(p: Period) -> list[dict]:
    return [e for e in p.map144_events if e.get('outcome') == 'decoded']

def _nodecode_events(p: Period) -> list[dict]:
    return [e for e in p.map144_events if e.get('outcome') == 'no_decode']

def _wsjtx_unique_msgs(p: Period) -> set[str]:
    """Distinct normalised messages WSJT-X decoded in this period."""
    return {' '.join(d['msg'].upper().split()) for d in p.wsjtx_decodes}

def _map144_unique_msgs(p: Period) -> set[str]:
    """Distinct normalised messages MAP144 decoded in this period
    (from launches.jsonl events with outcome=decoded)."""
    out: set[str] = set()
    for e in p.map144_events:
        if e.get('outcome') != 'decoded': continue
        m = e.get('message') or ''
        if m: out.add(' '.join(m.upper().split()))
    return out

def _wsjtx_callsigns(p: Period) -> set[str]:
    """Distinct receiver callsigns from WSJT-X decodes.  Approximate —
    pulls the second whitespace token, which for typical CQ/QSO messages
    is the called or responding callsign.  Good enough for multi-station
    period tagging."""
    out: set[str] = set()
    for d in p.wsjtx_decodes:
        toks = d['msg'].split()
        if len(toks) >= 2:
            out.add(toks[1])
    return out

def _max_n_chans_2s(p: Period) -> int:
    return max((e.get('n_chans_2s') or 0 for e in p.map144_events), default=0)

def _median_coherence_h(p: Period) -> float | None:
    vals = [e.get('sync_phase_coherence_h') for e in p.map144_events]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None

def _max_abs_ch_signed(p: Period) -> int:
    return max((abs(e.get('ch_signed') or 0) for e in p.map144_events), default=0)

def _max_sq_metric(p: Period) -> float:
    return max((e.get('sq_metric_db_h') or -99.0 for e in p.map144_events), default=-99.0)

def _wsjtx_snrs(p: Period) -> list[int]:
    return [d['snr'] for d in p.wsjtx_decodes]


# ── Tag table ─────────────────────────────────────────────────────────────────

TAGS: list[Tag] = [

    # ── catch-rate tests: pass criterion = decode ────────────────────────────

    Tag('canonical_strong',
        'Strong on-frequency single ping; sanity floor.  Both sides decode, '
        'detection cluster narrow.',
        pass_when='decoded',
        predicate=lambda p: (
            len(p.wsjtx_decodes) >= 1 and max(_wsjtx_snrs(p), default=-99) >= 8
            and len(_decoded_events(p)) >= 1
            and max((e.get('n_cluster_chan') or 0 for e in p.map144_events), default=0) <= 2
        )),

    Tag('weak',
        'Near-floor signal.  WSJT-X SNR <= -3 dB.  Per-event SNR may differ; '
        'WSJT-X side is the reference.',
        pass_when='decoded',
        predicate=lambda p: (
            len(p.wsjtx_decodes) >= 1 and min(_wsjtx_snrs(p), default=99) <= -3
        )),

    Tag('wsjtx_only',
        'MAP144 missed, WSJT-X decoded.  Period has >= 1 ALL.TXT entry '
        'and zero MAP144 decoded events.  The mission-critical gap '
        '(~70/day in current operation).  Symmetric with map144_only.',
        pass_when='decoded',
        predicate=lambda p: (
            len(p.wsjtx_decodes) >= 1 and len(_decoded_events(p)) == 0
        )),

    Tag('map144_only',
        'WSJT-X missed, MAP144 decoded.  Period has >= 1 MAP144 decoded '
        'event and zero ALL.TXT entries.  Possible MAP144-advantage case '
        'OR a phantom — manual triage required.  Symmetric with wsjtx_only.',
        pass_when='decoded',
        predicate=lambda p: (
            len(_decoded_events(p)) >= 1 and len(p.wsjtx_decodes) == 0
        )),

    Tag('both_decoded',
        'Both sides decoded.  Baseline correctness sample.  Note: does NOT '
        'mean MAP144 caught EVERY WSJT-X decode in the period — only that '
        'each side got at least one.  See wsjtx_more / map144_more for the '
        'asymmetric partial-coverage cases.',
        pass_when='decoded',
        predicate=lambda p: (
            len(p.wsjtx_decodes) >= 1 and len(_decoded_events(p)) >= 1
        )),

    Tag('wsjtx_more',
        'WSJT-X had MORE raw decode events than MAP144 in this period — '
        'both sides decoded SOMETHING but MAP144 has a smaller decode count.  '
        'NOTE: counts duplicates (e.g. WSJT-X getting 3 averaging-pattern '
        'hits on the same CQ counts as 3); to distinguish actual missed '
        'callsigns from "just fewer wins on the same message," see '
        'wsjtx_missed_callsign.',
        pass_when='decoded',
        predicate=lambda p: (
            len(p.wsjtx_decodes) > len(_decoded_events(p)) >= 1
        )),

    Tag('map144_more',
        'MAP144 had MORE raw decode events than WSJT-X did in this period.  '
        'Either a real MAP144-advantage case OR a phantom; same '
        'duplicate-counting caveat as wsjtx_more.  See map144_extra_callsign '
        'for unique-message-level cases.',
        pass_when='decoded',
        predicate=lambda p: (
            len(_decoded_events(p)) > len(p.wsjtx_decodes) >= 1
        )),

    Tag('wsjtx_missed_callsign',
        'MAP144 decoded a STRICT SUBSET of the unique messages WSJT-X got '
        '(both >= 1, but WSJT-X has at least one unique message MAP144 did '
        'not).  This is the direct "MAP144 missed a callsign" case — '
        'message-set comparison, ignores duplicate decodes of the same '
        'message.  The most actionable partial-miss bucket.',
        pass_when='decoded',
        predicate=lambda p: (
            _wsjtx_unique_msgs(p) > _map144_unique_msgs(p)
            and len(_map144_unique_msgs(p)) >= 1
        )),

    Tag('map144_extra_callsign',
        'MAP144 decoded at least one unique message WSJT-X did NOT.  Both '
        'sides decoded something; MAP144 caught a callsign the reference '
        'missed.  Possible MAP144 win (e.g. caught a brief ping WSJT-X '
        "missed) OR a phantom matching no real station's transmission.",
        pass_when='decoded',
        predicate=lambda p: (
            _map144_unique_msgs(p) - _wsjtx_unique_msgs(p)
            and len(_wsjtx_unique_msgs(p)) >= 1
        )),

    Tag('multi_station_period',
        'Period contains decodes from 3+ distinct callsigns — stresses '
        'classifier signature-aware repeat suppression.',
        pass_when='decoded',
        predicate=lambda p: len(_wsjtx_callsigns(p)) >= 3),

    Tag('near_nyquist',
        'Signal placed near the channel-grid Nyquist boundary; stresses '
        'the dynamic-Nyquist mask fix (commit cecce87 era).',
        pass_when='decoded',
        predicate=lambda p: _max_abs_ch_signed(p) >= 20),

    # ── propagation-regime tags (predicate placeholders — need M2 measure) ───

    Tag('short_burst',
        'Burst width <= 500 ms.  Predicate not yet implemented: requires '
        'a width estimate that is not currently in launches.jsonl.',
        pass_when='decoded',
        predicate=None),

    Tag('fluttery',
        'Per-burst sigma_f >= 1.7 Hz (58% of corpus per 2026-05-11 finding).  '
        'Predicate requires running measure_burst_freq_coherent over the '
        'corpus and persisting sigma_f — to be added in M2.',
        pass_when='decoded',
        predicate=None),

    Tag('drifting',
        'Linear |df/dt| >= 0.5 Hz/s within burst.  Same data dependency as '
        'fluttery — added in M2.',
        pass_when='decoded',
        predicate=None),

    Tag('multipath',
        'Moderate correlation r(|A|, f).  Same data dependency as fluttery.',
        pass_when='decoded',
        predicate=None),

    # ── Mode tags applied by human review (no automatic predicate) ───────────
    # These are inferences from the spectrogram + decode context; the predicate
    # is None because they can't be reliably derived from data we have.  They
    # exist here so the gallery / audit tools recognise them as declared
    # vocabulary rather than ad-hoc.

    Tag('weak_fading',
        'Slow time-varying weak signal of unknown specific propagation mode.  '
        'Catch-all for tropo / aircraft-enhanced / terrestrial multipath / '
        'unidentified weak slow-varying signals.  Use this when you can see '
        'a signal but cannot confidently say which mode produced it — that '
        'is most of the time.  Human-applied only.',
        pass_when='decoded',
        predicate=None),

    Tag('groundwave',
        'Short-distance signal that is weak because of antenna geometry or '
        'low power, not propagation.  Antennas pointed away, side-lobe '
        'reception, or local QRP.  Distinct from weak_fading (which implies '
        'propagation-mediated weakness).  Human-applied only.',
        pass_when='decoded',
        predicate=None),

    Tag('meteor_ping',
        'Discrete impulsive burst with classic MS signature (brief rise, '
        'short tail, no significant Doppler).  Human-applied — the classifier '
        'will eventually try to learn this label from launches.jsonl features.',
        pass_when='decoded',
        predicate=None),

    # ── false-positive tests: pass criterion = no decode ─────────────────────

    Tag('post_burst_rebound',
        'Wideband burst end followed by ramp of false triggers (project '
        'memory: post-burst rebound).  Real-side filter: n_chans_2s >= 8 '
        'AND no decode anywhere in period AND no decode in adjacent period.',
        pass_when='no_decode',
        predicate=lambda p: (
            _max_n_chans_2s(p) >= 8
            and len(p.wsjtx_decodes) == 0
            and len(_decoded_events(p)) == 0
        )),

    Tag('persistent_nodecode',
        'Period with many MAP144 launches and zero decodes either side.  '
        'Catch-all for noise-driven false-positive activity.',
        pass_when='no_decode',
        predicate=lambda p: (
            len(_nodecode_events(p)) >= 10
            and len(p.wsjtx_decodes) == 0
            and len(_decoded_events(p)) == 0
        )),

    Tag('low_coherence_launch',
        'Median sync_phase_coherence_h < 0.35 across launches in period.  '
        'Suggests noise-driven trigger activity.',
        pass_when='no_decode',
        predicate=lambda p: (
            (_median_coherence_h(p) or 1.0) < 0.35 and len(p.map144_events) >= 3
        )),

    # ── orthogonal tags (source / availability) ──────────────────────────────

    Tag('has_wsjtx_wav',
        'WSJT-X save/ contains the 12-kHz audio WAV for this period.',
        pass_when='any',
        predicate=lambda p: p.wsjtx_wav is not None),

    Tag('has_map144_wav',
        'MSK144/detections/ contains at least one MAP144 detection WAV in '
        'this period.',
        pass_when='any',
        predicate=lambda p: len(p.map144_wavs) >= 1),

    Tag('both_wavs_available',
        'Both WSJT-X audio and a MAP144 detection WAV exist — enables direct '
        'jt9 A/B (same period, both decoder inputs).',
        pass_when='any',
        predicate=lambda p: p.wsjtx_wav is not None and len(p.map144_wavs) >= 1),
]


# Lookup map: name → Tag
TAG_BY_NAME: dict[str, Tag] = {t.name: t for t in TAGS}
