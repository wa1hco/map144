# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Dual-clock time base — the single owner of sample↔UTC time handling.

See ``docs/block-stream-design.md`` §3.5 ("Two clocks, by construction" /
"Period anchoring").  This is the refactored home for all the time logic that
used to be scattered through ``processing.py`` as ``_iq_t0_wall`` /
``_loop_wall`` / ``_pkt_time*`` mutable engine state — the sprawl that produced
the 2026-05-31 "1970 timestamp" bug when a WAV-relative anchor leaked into a
live run.

Design contract (do not weaken without a doc revision):

* **Two clocks, kept independent, never slewed.**
  - *Sample clock*: monotonic sample counter; ``sample / sample_rate_hz`` is
    deterministic time, drives all DSP.  TimeBase never reads or corrects it.
  - *Wall clock*: UTC seconds (``time.time()``); fresh per source/period read.
    TimeBase never derives it from the sample count.
* **Period anchoring.**  The only place the two clocks meet is a *synchronised
  re-pair* captured once per 15-s period: ``mark_period_start(sample, wall)``.
  An event at ``event_sample`` within that period is stamped
  ``period_start_wall + (event_sample - period_start_sample) / sample_rate_hz``
  — sample-precise within the period, fresh UTC per period, no cross-period
  drift, no event-processing-latency jitter.
* **One owner, one reset point.**  ``reset(sample, wall)`` is called exactly
  once per source open with that source's anchor policy (radio =
  ``time.time()``; WAV = an explicit chosen UTC, *not* ≈0 by accident).  Because
  every source open re-anchors, a stale anchor cannot leak across a source
  switch — that is the structural fix for the 1970 bug.

This object is **pure**: it does not call ``time.time()`` or touch any I/O.  The
Source reads the two clocks and hands them in.  That keeps it trivially
testable (including the drift and 1970-regression paths) — see
``tests/test_timebase.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


class TimeBaseError(RuntimeError):
    """Raised when a UTC is requested before the time base is anchored."""


@dataclass
class _Anchor:
    sample: int
    wall: float


class TimeBase:
    """Owns the sample↔UTC mapping for one signal source.

    Parameters
    ----------
    sample_rate_hz : float
        Nominal sample rate.  Used only to convert *short* intra-period sample
        offsets to seconds; never used to extrapolate UTC across periods.
    period_secs : float
        WSJT period length (15.0 s on 6 m MSK144).  Defines the UTC grid that
        ``period_index`` snaps to.
    """

    def __init__(self, sample_rate_hz: float, period_secs: float = 15.0):
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
        if period_secs <= 0:
            raise ValueError(f"period_secs must be > 0, got {period_secs}")
        self.sample_rate_hz = float(sample_rate_hz)
        self.period_secs = float(period_secs)
        self._session: _Anchor | None = None   # at source open (policy anchor)
        self._period: _Anchor | None = None     # current period re-pair

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(self, *, sample: int, wall: float) -> None:
        """Anchor at source open with the source's policy.

        ``wall`` is the UTC that the source assigns to ``sample``: for live
        radio, ``time.time()`` read at the first packet; for WAV replay, the
        explicit policy UTC (recording time, or "now") — chosen, never an
        accidental ≈0.  Called on *every* source open, which is what makes a
        cross-source anchor leak impossible.
        """
        self._session = _Anchor(int(sample), float(wall))
        # The session anchor doubles as the first period anchor so utc_at works
        # immediately; the source re-pairs it at each 15-s boundary thereafter.
        self._period = _Anchor(int(sample), float(wall))

    def is_anchored(self) -> bool:
        return self._period is not None

    # ── period anchoring ─────────────────────────────────────────────────────
    def mark_period_start(self, *, sample: int, wall: float) -> None:
        """Capture one synchronised ``(sample, wall)`` pair at a period boundary.

        Both clocks are read together by the caller (the Source / period
        tracker) and handed in.  This is the per-period re-pair: it does NOT
        slew either clock, it re-pegs the pair so intra-period stamping stays
        fresh.  Call once per 15-s period (see ``should_mark_period``).
        """
        self._period = _Anchor(int(sample), float(wall))

    def should_mark_period(self, wall: float) -> bool:
        """True when ``wall`` has crossed into a new UTC period since the last
        anchor — i.e. the caller should ``mark_period_start`` now."""
        if self._period is None:
            return True
        return self.period_index(wall) != self.period_index(self._period.wall)

    # ── queries ──────────────────────────────────────────────────────────────
    def utc_at(self, sample: int) -> float:
        """Absolute UTC (seconds) for ``sample``, anchored to the current period.

        ``period_start_wall + (sample - period_start_sample) / sample_rate_hz``.
        Valid for samples within (or near) the current period; the caller marks
        a new period at each boundary so the extrapolation stays short.
        """
        if self._period is None:
            raise TimeBaseError("utc_at() before reset()/mark_period_start()")
        return self._period.wall + (int(sample) - self._period.sample) / self.sample_rate_hz

    def period_index(self, wall: float) -> int:
        """Which UTC 15-s period a wall-clock time falls in (floor to the grid).

        WSJT periods are UTC-aligned, so period assignment is a pure wall-clock
        operation — never sample-derived (that is the drift trap)."""
        import math
        return int(math.floor(wall / self.period_secs))

    def current_period_index(self) -> int:
        """Period index of the currently-anchored period."""
        if self._period is None:
            raise TimeBaseError("current_period_index() before anchoring")
        return self.period_index(self._period.wall)

    def period_start_utc(self) -> float:
        """UTC the current period anchor was pegged to."""
        if self._period is None:
            raise TimeBaseError("period_start_utc() before anchoring")
        return self._period.wall
