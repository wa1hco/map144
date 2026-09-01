# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Per-dial NCO + decimate bank → WsjtxAudioExporter sinks.

Input: raw complex IQ at ``hw_rate`` with DC = ``pan_center``.
Each branch mixes ``usb_center = dial + 1500`` to DC, anti-alias decimates to
12 kHz, and feeds ``WsjtxAudioExporter`` (which upconverts DC → 1500 Hz audio).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.signal import firwin, upfirdn

from ..wsjtx_audio_export import WsjtxAudioExporter
from .band_plan import DialChannel

log = logging.getLogger(__name__)

AUDIO_FS = 12_000


def _make_decimator(factor: int, ntaps: int = 65):
    """Stateful integer decimator (complex64), same contract as map65_export."""
    taps = firwin(ntaps, 0.9 / factor).astype(np.float32)
    ntaps_m1 = len(taps) - 1
    skip = ntaps_m1 // factor
    state = np.zeros(ntaps_m1, dtype=np.complex64)

    def apply(iq: np.ndarray) -> np.ndarray:
        nonlocal state
        x = np.ascontiguousarray(iq, dtype=np.complex64)
        x_ext = np.concatenate((state, x))
        yr = upfirdn(taps, x_ext.real, up=1, down=factor)
        yi = upfirdn(taps, x_ext.imag, up=1, down=factor)
        state = (x[-ntaps_m1:] if len(x) >= ntaps_m1
                 else np.concatenate((state, x))[-ntaps_m1:])
        n_out = len(iq) // factor
        out = np.empty(n_out, dtype=np.complex64)
        out.real[:] = yr[skip:skip + n_out]
        out.imag[:] = yi[skip:skip + n_out]
        return out

    return apply


def _nco_shift(chunk: np.ndarray, phase: float, step: float):
    """Multiply by exp(j*(phase + step*n)); return (shifted, new_phase)."""
    n = len(chunk)
    ph = phase + step * np.arange(n, dtype=np.float64)
    rot = np.exp(1j * ph).astype(np.complex64)
    out = np.ascontiguousarray(chunk, dtype=np.complex64) * rot
    new_phase = float((phase + step * n) % (2.0 * np.pi))
    return out, new_phase


@dataclass
class DialBranch:
    channel: DialChannel
    rf: int | None
    exporter: WsjtxAudioExporter
    mix_hz: float
    _phase: float = 0.0
    _step: float = 0.0
    _decimate: Callable | None = None

    def process(self, raw_iq: np.ndarray) -> None:
        shifted, self._phase = _nco_shift(raw_iq, self._phase, self._step)
        bb = self._decimate(shifted)
        if len(bb):
            self.exporter.feed(bb)


class DialAudioBank:
    """N dial → audio sink branches sharing one raw-IQ stream (one RF port)."""

    def __init__(
        self,
        channels: Sequence[DialChannel],
        *,
        pan_center_mhz: float,
        hw_rate_hz: int,
        rf: int | None = None,
        audio_fs: int = AUDIO_FS,
        exporter_factory=None,
    ):
        if hw_rate_hz % audio_fs != 0:
            raise ValueError(f"hw_rate {hw_rate_hz} not divisible by audio_fs {audio_fs}")
        self.pan_center_mhz = float(pan_center_mhz)
        self.hw_rate_hz = int(hw_rate_hz)
        self.audio_fs = int(audio_fs)
        self.rf = rf
        self.decim = self.hw_rate_hz // self.audio_fs
        factory = exporter_factory or WsjtxAudioExporter

        pan_hz = self.pan_center_mhz * 1e6
        self.branches: list[DialBranch] = []
        for i, ch in enumerate(channels):
            # Bring usb_center down to DC: mix by -(usb_center - pan).
            mix_hz = -(ch.usb_center_hz - pan_hz)
            step = 2.0 * np.pi * mix_hz / self.hw_rate_hz
            sink = ch.sink_name(rf)
            desc = ch.description(rf)
            stream = f"{ch.band} {ch.mode} {ch.dial_mhz:.3f} 12k feed -> {sink}"
            # index selects MAP144_WSJTX_DEVICE list entry / auto-matched cable
            # on Windows/macOS (Linux ignores it and creates a PipeWire sink).
            exp = factory(
                sink_name=sink,
                label=f"{ch.band}-{ch.mode}-{ch.dial_khz_token}",
                description=desc,
                stream_name=stream,
                fs=self.audio_fs,
                index=i,
                rf=rf,
            )
            br = DialBranch(
                channel=ch, rf=rf, exporter=exp, mix_hz=mix_hz,
                _phase=0.0, _step=step, _decimate=_make_decimator(self.decim),
            )
            self.branches.append(br)
            log.info("DialAudioBank: %s  mix=%+.1f Hz -> %s",
                     ch.description(rf), mix_hz, sink)

    def process(self, raw_iq: np.ndarray) -> None:
        """Feed one raw IQ block (complex) at hw_rate to every branch."""
        for br in self.branches:
            br.process(raw_iq)

    def close(self) -> None:
        for br in self.branches:
            try:
                br.exporter.close()
            except Exception:
                pass
        self.branches.clear()

    @property
    def sink_names(self) -> list[str]:
        return [br.exporter.sink for br in self.branches]
