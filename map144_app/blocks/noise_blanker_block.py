# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""NoiseBlankerBlock — Block-form wideband noise blanker.

Wraps the existing :mod:`map144_app.noise_blanker` backends (Linrad / Bypass /
NR0V-Wideband) as a Phase 3 :class:`Block`.  No DSP changes — same backend, same
per-bin spectrum averages, same Hann-tapered soft blanking.  This is a packaging
change toward Option 3 of the cut-over (TODO #13.7 / Phase 5): moving the noise
blanker off the legacy ``process_iq_data`` inline path so the monolith can be
retired.  Not yet wired into a graph.

Data flow
---------
Input : ``IQStream``  complex64, shape ``(N,)`` at ``sample_rate_hz`` (48 kHz)
Output: ``IQStream``  complex64, shape ``(N,)`` — cleaned IQ, same rate, same
        ``start_sample`` (blanking is sample-for-sample, no decimation, no carry).

Decoupling from the Engine
--------------------------
The legacy blanker backends take the *Engine* and read/write their warm-up and
display state (``_nb_spec_avg``, ``_nb_floor``, ``_nb_blanked_count`` …) directly
on it via get/setattr.  This block hands them a minimal :class:`_BlankerState`
holder instead of the whole Engine — a narrower interface (CLAUDE.md "small
interfaces").  ``blanker.reset(state)`` seeds the warm-up attributes the backend
reads *without* getattr defaults; the ``+=`` counters ``_nb_total_count`` /
``_nb_blanked_count`` are seeded to 0.

Sample-clock and wall-clock anchoring
-------------------------------------
Per §3.5 the blanker **inherits** ``start_sample`` and
``wall_clock_at_production`` from the upstream record unchanged — it is a
deterministic, sample-aligned function of the input and the Source remains the
authoritative anchor.  No samples are added or dropped.

Out of scope for v1
-------------------
- Dual-pol.  The backends support independent H/V masks; a follow-up will
  consume a :class:`CoherentPair` and emit cleaned H + V.  v1 is mono (H only).
- Exposing the blanker's display state (env / floor / blanked-sample log) to a
  DisplayBlock.  v1 surfaces only ``blanked_fraction`` in the output record's
  metadata — enough for the #41c pct25-freeze gate the DetectorBlock will read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import noise_blanker
from ..noise_blanker import NB_FACTOR
from .block import Block
from .types import BlockConfig, Record, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class NoiseBlankerBlockConfig(BlockConfig):
    """Per-NoiseBlanker config."""

    # Input / output ports.  Output is cleaned IQ at the same rate, so the
    # default port name mirrors the input ("iq").
    input_port_name: str = "iq"
    output_port_name: str = "iq"

    # Backend name passed to ``noise_blanker.make()`` ("Linrad" / "Bypass" /
    # "NR0V-Wideband").  Unknown / unavailable names fall back to Linrad.
    blanker_name: str = "Linrad"

    sample_rate_hz: int = 48_000
    nb_factor: float = NB_FACTOR
    nb_factor_v: float | None = None

    # ``Stream.get`` timeout — short enough that ``stop()`` stays responsive.
    get_timeout_s: float = 0.250


class _BlankerState:
    """Minimal stand-in for the Engine that a Blanker backend get/setattrs its
    ``_nb_*`` warm-up + display state onto.

    Decouples the backend from the full Engine: the block owns exactly the state
    the blanker needs and nothing else.  ``Blanker.reset()`` seeds the warm-up
    attributes the backend reads without getattr defaults (``_nb_spec_avg`` …);
    the cumulative ``+=`` counters are seeded here.
    """

    def __init__(self, *, sample_rate_hz: int, nb_factor: float,
                 nb_factor_v: float | None = None) -> None:
        self.sample_rate = int(sample_rate_hz)
        self.nb_factor = float(nb_factor)
        if nb_factor_v is not None:
            self.nb_factor_v = float(nb_factor_v)
        # Cumulative counters the backends assume already exist (reset() does
        # not seed these — they are running totals, not warm-up state).
        self._nb_total_count = 0
        self._nb_blanked_count = 0


class NoiseBlankerBlock(Block):
    """Wideband noise blanker as a Block.  See module docstring."""

    config_type = NoiseBlankerBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._blanker = None
        self._state: _BlankerState | None = None
        # Next expected input start_sample (gap-free invariant, §3.1).  None
        # until the first record, then adopted from it.
        self._n_in_samples: int | None = None

    # --- Lifecycle ----------------------------------------------------

    def on_start(self) -> None:
        super().on_start()
        cfg: NoiseBlankerBlockConfig = self.config  # type: ignore[assignment]
        self._blanker = noise_blanker.make(cfg.blanker_name)
        self._state = _BlankerState(
            sample_rate_hz=cfg.sample_rate_hz,
            nb_factor=cfg.nb_factor,
            nb_factor_v=cfg.nb_factor_v,
        )
        # Seed the backend's warm-up state (Linrad reads _nb_spec_avg etc. with
        # no getattr default; Bypass.reset is a no-op).
        self._blanker.reset(self._state)
        self._n_in_samples = None
        log.info("%s: blanker=%s @ %d Hz, nb_factor=%.2f",
                 self.name, self._blanker.name, cfg.sample_rate_hz, cfg.nb_factor)

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: NoiseBlankerBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.input_port_name
        out_port = cfg.output_port_name
        if in_port not in self.inputs:
            raise RuntimeError(f"{self.name}: input port {in_port!r} not connected")
        if out_port not in self.outputs:
            raise RuntimeError(f"{self.name}: output port {out_port!r} not connected")

        try:
            rec: Record = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates to the run loop and exits cleanly.

        if not isinstance(rec, Record):
            log.warning("%s: dropped non-Record %r on input port",
                        self.name, type(rec).__name__)
            return
        if rec.sample_rate_hz != cfg.sample_rate_hz:
            raise RuntimeError(
                f"{self.name}: upstream record sample_rate_hz={rec.sample_rate_hz} "
                f"!= configured {cfg.sample_rate_hz}")

        # Gap-free invariant: a discontinuity invalidates the blanker's warm-up
        # reference (per-bin spectrum averages, block-RMS median), so reseed.
        if self._n_in_samples is None:
            self._n_in_samples = rec.start_sample
        elif rec.start_sample != self._n_in_samples:
            log.warning("%s: input gap — got start_sample=%d, expected %d; reseeding",
                        self.name, rec.start_sample, self._n_in_samples)
            self._blanker.reset(self._state)
            self._n_in_samples = rec.start_sample

        raw = np.ascontiguousarray(rec.data, dtype=np.complex64).ravel()
        result = self._blanker.process(self._state, raw, None, False)
        cleaned = np.ascontiguousarray(result.cleaned_h, dtype=np.complex64)
        blanked_fraction = (float(result.blank_mask.mean())
                            if result.blank_mask.size else 0.0)

        out_meta = dict(rec.metadata)
        out_meta["blanker"] = self._blanker.name
        out_meta["blanked_fraction"] = blanked_fraction

        out_record = Record(
            data=cleaned,
            sample_rate_hz=rec.sample_rate_hz,
            start_sample=rec.start_sample,
            wall_clock_at_production=rec.wall_clock_at_production,  # inherit (§3.5)
            metadata=out_meta,
        )
        try:
            self.outputs[out_port].put(out_record)
        except StreamClosed:
            raise StopIteration

        self._n_in_samples += rec.n_samples
