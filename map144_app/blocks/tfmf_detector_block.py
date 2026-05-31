# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""TFMF detector block (TODO R4, first slice).

A :class:`Block` that lifts the wideband Time-Frequency Matched-Filter detector
out of the legacy ``process_iq_data`` monolith into the block/stream graph.  It
consumes wideband IQ :class:`Record`\\s, accumulates one WSJT period, runs the
matched-filter surface + candidate extraction (GPU when CuPy is available, CPU
fallback — see ``_build_tfmf_display_surface``), and emits one
``tfmf_candidate`` :class:`Event` per detection.

Why a block, and block-only (never bolted onto the legacy path): the matched
filter is *linear*, so unlike sq_det it needs no channelizer (squaring is what
forced the channelizer; see ``experiments/gpu_tfmf/README.md``).  In GPU mode
this block *replaces* ``Channelizer → Detector(sq+sync)`` rather than running
beside it — the graph-builder topology switch is the next R4 slice.

Dual-clock stamping follows block-stream-design §3.5 "Period anchoring": the
period's first record supplies the synchronized ``(start_sample, wall_clock)``
anchor, and each candidate's ``signal_wall_clock`` is
``period_start_wall + candidate.time_s`` — the same formula
:class:`~map144_app.timebase.TimeBase` uses, carried here in event metadata so
no downstream stage re-derives time.  (This is the structural cure for the
1970-timestamp bug — there is no relative anchor anywhere.)

R4 follow-on slices (not here): the ``(mode, gpu)`` graph-builder topology
switch, device-residency across GPU blocks, and the GPU-vs-CPU parity gate.
"""
from __future__ import annotations

import time

import numpy as np

from .block import Block
from .types import BlockConfig, Event, Record, StreamClosed


class TFMFDetectorBlockConfig(BlockConfig):
    """Config for :class:`TFMFDetectorBlock`."""

    def __init__(self, name: str = "",
                 input_port_name: str = "iq",
                 candidate_port_name: str = "candidates",
                 period_secs: float = 15.0,
                 get_timeout_s: float = 1.0):
        super().__init__()
        self.name = name
        self.input_port_name = input_port_name
        self.candidate_port_name = candidate_port_name
        self.period_secs = float(period_secs)
        self.get_timeout_s = float(get_timeout_s)


class TFMFDetectorBlock(Block):
    """Wideband TFMF detector: IQ Records in, ``tfmf_candidate`` Events out."""

    config_type = TFMFDetectorBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # Per-period accumulator.  ``_buf`` is a list of complex64 H-channel
        # chunks; concatenated and trimmed to one period when full.
        self._buf: list[np.ndarray] | None = None
        self._buf_n: int = 0
        self._buf_start_sample: int | None = None
        self._buf_start_wall: float | None = None
        self._sample_rate: int | None = None

    # --- one consume → maybe-emit cycle -----------------------------------
    def tick(self) -> None:
        cfg: TFMFDetectorBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.input_port_name
        if in_port not in self.inputs:
            raise RuntimeError(f"{self.name}: input port {in_port!r} not connected")
        try:
            rec = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except StreamClosed:
            self._stopping.set()
            return
        if rec is None or not isinstance(rec, Record):
            return

        data = np.asarray(rec.data)
        # Wideband IQ; H channel is column 0 of a dual-pol record, else 1-D.
        iq = (data[:, 0] if data.ndim == 2 else data.ravel()).astype(np.complex64)

        if self._buf is None:
            self._buf = []
            self._buf_n = 0
            self._buf_start_sample = int(rec.start_sample)
            self._buf_start_wall = float(rec.wall_clock_at_production)
            self._sample_rate = int(rec.sample_rate_hz)
        self._buf.append(iq)
        self._buf_n += iq.shape[0]

        period_n = int(round(cfg.period_secs * self._sample_rate))
        if self._buf_n >= period_n:
            wb = np.concatenate(self._buf)[:period_n]
            self._emit_candidates(wb)
            self._buf = None   # start a fresh period

    # --- compute + emit ----------------------------------------------------
    def _emit_candidates(self, wb_iq: np.ndarray) -> None:
        cfg: TFMFDetectorBlockConfig = self.config  # type: ignore[assignment]
        out_port = cfg.candidate_port_name
        # Compute is GPU-when-available / CPU-fallback inside this call.  It is
        # a pure function of the IQ window (no engine state).
        from ..processing import _build_tfmf_display_surface
        try:
            _surf, candidates = _build_tfmf_display_surface(wb_iq)
        except Exception:
            # Detection must never take the block down; a bad window is dropped.
            return
        if not candidates or out_port not in self.outputs:
            return

        start_sample = int(self._buf_start_sample or 0)
        start_wall = float(self._buf_start_wall or 0.0)
        rate = int(self._sample_rate or cfg.period_secs)  # rate already known
        now = time.time()
        out = self.outputs[out_port]
        for c in candidates:
            ev = Event(
                kind="tfmf_candidate",
                occurred_at_sample=start_sample + int(c.time_sample),
                sample_rate_hz=int(self._sample_rate or 48000),
                wall_clock_at_production=now,
                payload={
                    "freq_hz": float(c.freq_hz),
                    "snr_db": float(c.snr_db),
                    "raw_magnitude": float(c.raw_magnitude),
                    "time_s": float(c.time_s),
                    "sample_epoch_offset_samples": float(c.sample_epoch_offset_samples),
                    # §3.5 period anchoring: signal time = period_start_wall + offset.
                    "signal_wall_clock": start_wall + float(c.time_s),
                },
            )
            out.put(ev)
