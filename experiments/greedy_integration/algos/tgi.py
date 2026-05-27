# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""TGI — Tone-Guided Integration.

Greedy speculative coherent integration of MSK144 frames, decisions
driven by WSJT-X sq_det (squared-signal tone amplitude at 2000 / 4000 Hz)
metric improvement.

Algorithm
---------
Given an analytic-signal stream and a trigger position t_anchor:

    integrated ← audio[t_anchor : t_anchor + NSPM]
    sq        ← sq_det(integrated).detmet
    for k = 1, 2, 3, ... up to max_span_frames:
        for offset in (+k, -k):                 # alternate forward/backward
            candidate ← audio[anchor + offset*NSPM : anchor + (offset+1)*NSPM]
            trial      = integrated + candidate  # coherent sum
            trial_sq   = sq_det(trial).detmet
            if trial_sq > sq:
                integrated ← trial               # KEEP
                sq         ← trial_sq
                consecutive_rejects ← 0
            else:
                consecutive_rejects += 1         # REJECT
                if consecutive_rejects >= max_consecutive_rejects:
                    return ...

The hypothesis: phase-coherent addition of frames on the SAME signal
adds amplitudes (+6 dB peak per doubling).  Frames on a different
Doppler track or with destructive multipath produce the opposite —
sq_det stays the same or DECREASES — and are rejected automatically.

The decision metric is ``detmet2`` (per-frame peak/local-in-band-noise
ratio).  This is scale-invariant: doubling the number of summed noise
frames doubles BOTH the peak (ah) AND the in-band noise floor (ahavp),
so detmet2 stays put.  But coherent in-phase signal addition doubles
ah (4× power for amplitude×2) while ahavp grows ~2× (incoherent noise),
so detmet2 GROWS.  This gives greedy the right discriminator: signal
coherence ↑ detmet2 ↑ KEEP; noise addition leaves detmet2 ≈ same →
REJECT (small fluctuations average out).

The raw ``detmet`` (peak power) is the WRONG criterion because it grows
with sample count regardless of signal — pure-noise summing inflates it
spuriously.  Block normalisation by xmed at the FINAL stage compares
the integrated detmet against the surrounding-block reference (where
xmed was computed at single-frame scale, so the integrated detmet/xmed
correctly grows N× for N coherent frames).

Coherent SUM vs AVERAGE
-----------------------
We use **sum** (not average) so the coherent gain is visible in
detmet's natural scale.  Two in-phase frames give 2× detmet (+6 dB);
averaging would give 1× and hide the gain in the metric used for the
greedy decision.  Block normalisation (xmed from the per-frame trace)
remains in single-frame units, so detmet_sum / xmed correctly grows
N-fold for N coherent frames.

Order of candidates
-------------------
Outward from anchor, alternating ±k.  Operator's call: most
appropriate for MS pings where signal is concentrated near the
trigger and the strong frames cluster.  An option is left for
``forward_only`` where the trigger arrives at the START of the burst.

Output
------
TGIResult contains:
    integrated_frame   : complex array, NSPM samples (the final coherent sum)
    detmet             : float (raw, of integrated_frame)
    detmet2            : float (per-frame ratio, of integrated_frame)
    n_kept             : int total frames in the sum (>= 1; trigger always kept)
    offsets_kept       : list of integer NSPM-offsets actually added
    offsets_rejected   : list of integer NSPM-offsets tried and rejected
    span_used_s        : seconds from min-offset frame start to max-offset frame end
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from map144_app.sq_det import (
    sq_det, NSPM, FS, DEFAULT_FC, DEFAULT_NTOL,
    DETECT_THRESHOLD_NORM,
)


@dataclass
class TGIConfig:
    fc: float = DEFAULT_FC                   # audio carrier (Hz)
    ntol: float = DEFAULT_NTOL               # freq tolerance for sq_det search masks
    max_span_frames: int = 8                 # max forward (or backward) frames to TRY
    max_consecutive_rejects: int = 3         # stop after this many in a row failed
    forward_only: bool = False               # True → only try +k offsets

    # Per-candidate phase search.  MSK144 IS continuous-phase, but the
    # PER-FRAME phase rotation in the squared FFT bin is non-zero in
    # general:
    #     Δφ_per_frame_in_FFT_bin = 2π·fc·NSPM/FS  +  Δφ_data_one_frame
    #                              = 2π·108         +  data-dependent
    # The carrier part is ZERO (integer 2π) for nominal fc=1500, but the
    # data part is the cumulative MSK phase change over 144 symbols —
    # message-content-dependent, anywhere in [0, 2π) for typical messages.
    # In the squared spectrum, this becomes 2·Δφ_data per frame.
    #
    # Result: frame K = frame 0 · exp(j · K · 2·Δφ_data).  Without
    # compensation, the coherent sum cancels for adverse Δφ_data values.
    # Phase search at each candidate finds the constructive rotation.
    #
    # 8 rotations gives 22.5° resolution — enough for sq_det (which is
    # ~smooth in phase since it's a power spectrum, not a phase-coherent
    # demodulator).  Cost: 8× per candidate, still cheap.
    n_phase_search: int = 8                  # rotations to try per candidate
    # Set n_phase_search=1 to disable (the v0 "naive sum" variant — useful
    # to measure how much the phase search itself contributes).
    #
    # Future optimisation: once the +1 candidate's best phase φ*_1 is found,
    # the +k candidates need k·φ*_1 by the constant-rotation-per-frame
    # property above.  Could skip the search for k>1.  Not done in v1.

    # Greedy decision criterion.  Each option is a different way of
    # asking "did this frame addition improve the SIGNAL detectability,
    # net of noise growth?":
    #
    # 'self'   : detmet2 = ah_trial / ahavp_trial  (self-normalised per
    #            trial; the v1 default).  Scale-invariant under coherent
    #            scaling, but in-band signal bleed contaminates ahavp
    #            slightly so the metric understates real signal growth.
    #
    # 'anchor' : ah_trial / ahavp_anchor  (operator's intuition: use the
    #            first-frame noise estimate as a fixed reference).  Cleaner
    #            if anchor were pure noise, but anchor has signal (we
    #            triggered there) so still imperfect.
    #
    # 'xmed'   : ah_trial - K · xmed  (block-level noise reference,
    #            multiplied by K to account for incoherent noise growth
    #            across K integrated frames).  Cleanest in principle —
    #            xmed is noise-only by construction (25th percentile of
    #            slide trace).  Requires the caller to supply xmed.
    criterion: str = 'self'                  # 'self' | 'anchor' | 'xmed'
    xmed: float = 0.0                        # required if criterion='xmed'

    # Option A: noise-growth-aware margin.  Reject candidates whose metric
    # improvement looks like noise inflation rather than coherent signal.
    #
    # margin_mode = 'none'  : keep on ANY improvement (v1 behaviour, broken
    #                          per Experiment B').
    # margin_mode = 'fixed' : keep iff new > M · old, M = margin_fixed.
    # margin_mode = 'kadapt': keep iff new > [(K+1)/K · safety] · old.
    #                          Tracks the linear-in-K signal-growth expectation.
    margin_mode: str = 'none'                # 'none' | 'fixed' | 'kadapt'
    margin_fixed: float = 1.5                # used when margin_mode='fixed'
    margin_safety: float = 1.0               # used when margin_mode='kadapt'


@dataclass
class TGIResult:
    integrated_frame: np.ndarray             # complex64, NSPM samples
    detmet:           float                  # raw peak amplitude max(ah, al)
    detmet2:          float                  # per-frame peak/local in-band noise ratio
    n_kept:           int                    # 1 = trigger only; no additions
    offsets_kept:     list[int]              # positive forward, negative backward
    offsets_rejected: list[int]
    span_used_s:      float                  # extent of the included frames in seconds

    def detection_score_norm(self, xmed: float) -> float:
        """Block-normalised detection metric (detmet / xmed).  Compare to
        DETECT_THRESHOLD_NORM = 3 to decide if the integrated frame fires."""
        return self.detmet / max(xmed, 1e-30)


def tgi_integrate(audio: np.ndarray, t_anchor_s: float,
                  config: Optional[TGIConfig] = None,
                  ) -> TGIResult:
    """Run TGI on an analytic-signal block, anchored at t_anchor_s.

    Parameters
    ----------
    audio : complex array
        Analytic signal at FS=12 kHz.  Audio carrier is at config.fc.
    t_anchor_s : float
        Time (seconds) where the trigger fired — the centre of the FIRST
        frame to integrate.  Adjacent frames are at ±NSPM samples.
    config : TGIConfig, optional
        Algorithm parameters.

    Returns
    -------
    TGIResult with the final integrated frame and sq_det metrics.

    Notes
    -----
    * If the anchor frame's window doesn't fit in the audio (negative or
      past-end), we clamp to the nearest valid position.
    * Each candidate frame must also fit in the audio; out-of-range
      offsets are silently skipped and counted as rejected.
    """
    if config is None:
        config = TGIConfig()
    n = audio.shape[0]
    n_anchor = int(round(t_anchor_s * FS))
    # Trigger frame: clamp anchor so [n_anchor : n_anchor + NSPM] is in range
    n_anchor = max(0, min(n - NSPM, n_anchor))

    integrated = audio[n_anchor : n_anchor + NSPM].astype(np.complex128).copy()
    r0 = sq_det(integrated, fc=config.fc, ntol=config.ntol)
    # Greedy decision uses detmet2 (scale-invariant, peak/local-in-band-noise
    # ratio).  See module docstring for why detmet (raw peak power) is wrong.
    cur_metric = r0.detmet2
    n_kept = 1
    kept: list[int] = []
    rejected: list[int] = []

    # Build outward-alternating offset sequence: 1, -1, 2, -2, 3, -3, ...
    # (or forward-only if configured)
    offset_seq: list[int] = []
    for k in range(1, config.max_span_frames + 1):
        offset_seq.append(+k)
        if not config.forward_only:
            offset_seq.append(-k)

    consecutive_rejects = 0
    for off in offset_seq:
        n_start = n_anchor + off * NSPM
        n_end   = n_start + NSPM
        if n_start < 0 or n_end > n:
            # Out-of-range candidate frame — counted as a reject for stopping
            rejected.append(off)
            consecutive_rejects += 1
            if consecutive_rejects >= config.max_consecutive_rejects:
                break
            continue

        candidate = audio[n_start : n_end].astype(np.complex128)

        # Per-candidate phase search: try n_phase_search rotations, keep the
        # one with highest detmet2 (scale-invariant — see module docstring).
        best_trial_metric = -np.inf
        best_trial_phase  = 0.0
        for k in range(config.n_phase_search):
            phi = 2.0 * np.pi * k / max(1, config.n_phase_search)
            trial = integrated + candidate * np.exp(1j * phi)
            m = sq_det(trial, fc=config.fc, ntol=config.ntol).detmet2
            if m > best_trial_metric:
                best_trial_metric = m
                best_trial_phase  = phi

        # Option A — noise-growth-aware margin
        if config.margin_mode == 'none':
            required_metric = cur_metric
        elif config.margin_mode == 'fixed':
            required_metric = cur_metric * config.margin_fixed
        elif config.margin_mode == 'kadapt':
            # K-adaptive: require improvement to exceed (K+1)/K signal-growth
            # expectation times safety factor.  K = current n_kept BEFORE add.
            required_metric = cur_metric * ((n_kept + 1) / n_kept) * config.margin_safety
        else:
            raise ValueError(f"unknown margin_mode {config.margin_mode!r}")

        if best_trial_metric > required_metric:
            integrated = integrated + candidate * np.exp(1j * best_trial_phase)
            cur_metric = best_trial_metric
            n_kept += 1
            kept.append(off)
            consecutive_rejects = 0
        else:
            rejected.append(off)
            consecutive_rejects += 1
            if consecutive_rejects >= config.max_consecutive_rejects:
                break

    final = sq_det(integrated, fc=config.fc, ntol=config.ntol)
    if kept or [0]:
        all_offs = sorted([0] + kept)
        # Span = end of latest kept frame minus start of earliest kept frame
        span_samples = (all_offs[-1] - all_offs[0]) * NSPM + NSPM
        span_used_s = span_samples / FS
    else:
        span_used_s = NSPM / FS

    return TGIResult(
        integrated_frame=integrated.astype(np.complex64),
        detmet=final.detmet, detmet2=final.detmet2,
        n_kept=n_kept,
        offsets_kept=kept, offsets_rejected=rejected,
        span_used_s=span_used_s,
    )
