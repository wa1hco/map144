# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Router engine: plan LO, open B210, run dial-audio + wideband banks.

Standalone process owns the B210 (do not run beside MAP144 on the same
radio).  DSP stays in dial_audio / wideband_iq; this module owns lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from .band_plan import DialChannel
from .dial_audio import DialAudioBank
from .lo_planner import DEFAULT_HW_RATE_HZ, LoPlan, LoPlanError, plan_lo
from .wideband_iq import WidebandDest, WidebandIqBank

log = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    dials: Sequence[DialChannel] = field(default_factory=list)
    map65: WidebandDest | None = None
    qmap: WidebandDest | None = None
    dual_rf_audio: bool = False   # second DialAudioBank on RF1 when dual hardware
    dual_channel: bool = False    # open B210 with both RF ports
    gain_db: float = 30.0
    device_args: str = ""
    max_hw_rate_hz: int = DEFAULT_HW_RATE_HZ
    pan_center_mhz: float | None = None  # override; else planner midpoint


class RouterEngine:
    """Start/stop the B210 multi-mode router."""

    def __init__(self):
        self.config: RouterConfig | None = None
        self.plan: LoPlan | None = None
        self._src = None
        self._audio_rf0: DialAudioBank | None = None
        self._audio_rf1: DialAudioBank | None = None
        self._wideband: WidebandIqBank | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def sink_names(self) -> list[str]:
        names: list[str] = []
        if self._audio_rf0 is not None:
            names.extend(self._audio_rf0.sink_names)
        if self._audio_rf1 is not None:
            names.extend(self._audio_rf1.sink_names)
        return names

    def plan_only(self, config: RouterConfig) -> LoPlan:
        """Compute LO plan without opening hardware (for GUI preview)."""
        from .lo_planner import WIDEBAND_HALF_HZ

        centres = []
        if config.map65 is not None:
            centres.append(float(config.map65.center_mhz))
        if config.qmap is not None:
            centres.append(float(config.qmap.center_mhz))

        if not centres:
            return plan_lo(
                config.dials,
                wideband_center_mhz=None,
                max_hw_rate_hz=config.max_hw_rate_hz,
                pan_center_mhz=config.pan_center_mhz,
            )

        # One or two IQ destinations: cover the union of their ±48 kHz windows.
        lo = min(centres) * 1e6 - WIDEBAND_HALF_HZ
        hi = max(centres) * 1e6 + WIDEBAND_HALF_HZ
        mid_mhz = 0.5 * (lo + hi) / 1e6
        half_hz = 0.5 * (hi - lo)
        return plan_lo(
            config.dials,
            wideband_center_mhz=mid_mhz,
            wideband_half_hz=half_hz,
            max_hw_rate_hz=config.max_hw_rate_hz,
            pan_center_mhz=config.pan_center_mhz,
        )

    def start(self, config: RouterConfig,
              *,
              usrp_factory=None,
              audio_exporter_factory=None,
              map65_exporter_factory=None) -> LoPlan:
        if self._running:
            raise RuntimeError("router already running; stop() first")
        if not config.dials and config.map65 is None and config.qmap is None:
            raise LoPlanError("select at least one dial channel or MAP65/QMAP")

        plan = self.plan_only(config)
        if plan.hw_rate_hz != DEFAULT_HW_RATE_HZ:
            # USRPSource is hard-wired to 192 kHz today.
            raise LoPlanError(
                f"planned hw_rate {plan.hw_rate_hz} Hz not supported by "
                f"USRPSource yet (only {DEFAULT_HW_RATE_HZ} Hz); shrink selection"
            )

        from ..usrp_source import USRPSource
        factory = usrp_factory or USRPSource

        dual = bool(config.dual_channel or config.dual_rf_audio
                    or config.map65 is not None or config.qmap is not None)
        # MAP65 dual-pol path historically required dual_channel; for mono IQ
        # Map65Exporter(n_channels=1) works with raw_v=raw_h in WidebandIqBank.
        if config.map65 is None and config.qmap is None:
            dual = bool(config.dual_channel or config.dual_rf_audio)

        src = factory(
            center_freq_mhz=plan.pan_center_mhz,
            pan_center_mhz=plan.pan_center_mhz,
            gain_db=config.gain_db,
            device_args=config.device_args,
            dual_channel=dual,
        )
        src.raw_only = True

        n_ch = 2 if dual else 1
        audio_rf = 0 if (config.dual_rf_audio and dual) else None
        self._audio_rf0 = DialAudioBank(
            config.dials,
            pan_center_mhz=plan.pan_center_mhz,
            hw_rate_hz=plan.hw_rate_hz,
            rf=audio_rf,
            exporter_factory=audio_exporter_factory,
        ) if config.dials else None

        self._audio_rf1 = None
        if config.dials and config.dual_rf_audio and dual:
            self._audio_rf1 = DialAudioBank(
                config.dials,
                pan_center_mhz=plan.pan_center_mhz,
                hw_rate_hz=plan.hw_rate_hz,
                rf=1,
                exporter_factory=audio_exporter_factory,
            )

        self._wideband = WidebandIqBank(
            pan_center_mhz=plan.pan_center_mhz,
            n_channels=n_ch if (config.map65 or config.qmap) else 1,
            map65=config.map65,
            qmap=config.qmap,
            exporter_factory=map65_exporter_factory,
        ) if (config.map65 or config.qmap) else None

        def _on_raw(raw_h, raw_v, ts_epoch):
            if self._audio_rf0 is not None:
                self._audio_rf0.process(raw_h)
            if self._audio_rf1 is not None and raw_v is not None:
                self._audio_rf1.process(raw_v)
            if self._wideband is not None:
                self._wideband.process(raw_h, raw_v, ts_epoch)

        src.raw_iq_callback = _on_raw
        src.start()

        self._src = src
        self.config = config
        self.plan = plan
        self._running = True
        log.info("RouterEngine started: pan=%.6f MHz hw=%d Hz sinks=%s",
                 plan.pan_center_mhz, plan.hw_rate_hz, self.sink_names)
        return plan

    def stop(self) -> None:
        self._running = False
        src, self._src = self._src, None
        if src is not None:
            try:
                src.raw_iq_callback = None
                src.stop()
            except Exception as exc:
                log.warning("RouterEngine: USRP stop: %s", exc)
        for bank in (self._audio_rf0, self._audio_rf1):
            if bank is not None:
                try:
                    bank.close()
                except Exception:
                    pass
        self._audio_rf0 = self._audio_rf1 = None
        if self._wideband is not None:
            try:
                self._wideband.close()
            except Exception:
                pass
            self._wideband = None
        log.info("RouterEngine stopped")
