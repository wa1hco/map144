# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Router engine: plan LO, open B210, blank, run dial-audio + wideband banks.

Standalone process owns the B210 (do not run beside MAP144 on the same
radio).  DSP stays in dial_audio / wideband_iq / noise_blanker; this module
owns lifecycle and live gain/blanker control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..blocks.noise_blanker_block import _BlankerState
from ..noise_blanker import NB_FACTOR, available as blanker_names, make as make_blanker
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
    gain_db: float = 50.0
    gain_db_ch1: float | None = None
    antenna: str = "RX2"
    antenna_ch1: str = "RX2"
    blanker_name: str = "Linrad"
    nb_factor: float = NB_FACTOR
    nb_factor_v: float | None = None
    device_args: str = ""
    max_hw_rate_hz: int = DEFAULT_HW_RATE_HZ
    pan_center_mhz: float | None = None  # override; else planner midpoint


@dataclass
class RouterStatus:
    """Snapshot for the GUI status timer."""

    running: bool = False
    pan_mhz: float | None = None
    hw_rate_hz: int | None = None
    dual: bool = False
    serial: str = "—"
    firmware: str = "—"
    fpga: str = "—"
    recv_count: int = 0
    buf_per_s: float = 0.0
    rms_h_dbfs: float | None = None
    rms_v_dbfs: float | None = None
    blanker_name: str = "—"
    blanked_pct: float = 0.0
    gain_db: float | None = None
    gain_db_ch1: float | None = None
    startup_phase: str = "idle"


class RouterEngine:
    """Start/stop the B210 multi-mode router."""

    def __init__(self):
        self.config: RouterConfig | None = None
        self.plan: LoPlan | None = None
        self._src = None
        self._audio_rf0: DialAudioBank | None = None
        self._audio_rf1: DialAudioBank | None = None
        self._wideband: WidebandIqBank | None = None
        self._blanker = None
        self._nb_state: _BlankerState | None = None
        self._running = False
        self._hw_info_cached = False
        self._serial = "—"
        self._firmware = "—"
        self._fpga = "—"
        self._blanked_ema = 0.0
        self._rms_h = None
        self._rms_v = None
        self._prev_recv_count = 0
        self._prev_recv_t = 0.0
        self._buf_per_s = 0.0
        # Pending control values (applied on start; live while running).
        self.gain_db = 50.0
        self.gain_db_ch1 = 50.0
        self.antenna = "RX2"
        self.antenna_ch1 = "RX2"
        self.blanker_name = "Linrad"
        self.nb_factor = float(NB_FACTOR)
        self.nb_factor_v = float(NB_FACTOR)

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
            raise LoPlanError(
                f"planned hw_rate {plan.hw_rate_hz} Hz not supported by "
                f"USRPSource yet (only {DEFAULT_HW_RATE_HZ} Hz); shrink selection"
            )

        from ..usrp_source import USRPSource
        factory = usrp_factory or USRPSource

        dual = bool(config.dual_channel or config.dual_rf_audio
                    or config.map65 is not None or config.qmap is not None)
        if config.map65 is None and config.qmap is None:
            dual = bool(config.dual_channel or config.dual_rf_audio)

        self.gain_db = float(config.gain_db)
        self.gain_db_ch1 = float(
            config.gain_db_ch1 if config.gain_db_ch1 is not None else config.gain_db)
        self.antenna = str(config.antenna)
        self.antenna_ch1 = str(config.antenna_ch1)
        self.blanker_name = str(config.blanker_name)
        self.nb_factor = float(config.nb_factor)
        self.nb_factor_v = float(
            config.nb_factor_v if config.nb_factor_v is not None else config.nb_factor)

        src = factory(
            center_freq_mhz=plan.pan_center_mhz,
            pan_center_mhz=plan.pan_center_mhz,
            gain_db=self.gain_db,
            antenna=self.antenna,
            device_args=config.device_args,
            dual_channel=dual,
        )
        src._gain_ch1 = self.gain_db_ch1
        src._antenna_ch1 = self.antenna_ch1
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

        self._init_blanker(plan.hw_rate_hz)

        def _on_raw(raw_h, raw_v, ts_epoch):
            cleaned_h, cleaned_v = self._blank_raw(raw_h, raw_v, dual)
            if self._audio_rf0 is not None:
                self._audio_rf0.process(cleaned_h)
            if self._audio_rf1 is not None and cleaned_v is not None:
                self._audio_rf1.process(cleaned_v)
            if self._wideband is not None:
                self._wideband.process(cleaned_h, cleaned_v, ts_epoch)

        src.raw_iq_callback = _on_raw
        src.start()

        self._src = src
        self.config = config
        self.plan = plan
        self._running = True
        self._hw_info_cached = False
        self._prev_recv_count = 0
        self._prev_recv_t = time.monotonic()
        self._buf_per_s = 0.0
        log.info("RouterEngine started: pan=%.6f MHz hw=%d Hz blanker=%s "
                 "gain=%.0f/%.0f dB sinks=%s",
                 plan.pan_center_mhz, plan.hw_rate_hz, self.blanker_name,
                 self.gain_db, self.gain_db_ch1, self.sink_names)
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
        self._blanker = None
        self._nb_state = None
        log.info("RouterEngine stopped")

    # ── Live controls ─────────────────────────────────────────────────────

    def set_gain_db(self, gain_db: float, channel: int = 0) -> None:
        gain_db = float(gain_db)
        if int(channel) == 0:
            self.gain_db = gain_db
        else:
            self.gain_db_ch1 = gain_db
        src = self._src
        if src is not None and hasattr(src, "set_rx_gain"):
            try:
                src.set_rx_gain(gain_db, int(channel))
            except Exception as exc:
                log.warning("set_gain_db ch%d: %s", channel, exc)

    def set_antenna(self, antenna: str, channel: int = 0) -> None:
        antenna = str(antenna)
        if int(channel) == 0:
            self.antenna = antenna
        else:
            self.antenna_ch1 = antenna
        src = self._src
        if src is not None and hasattr(src, "set_rx_antenna"):
            try:
                src.set_rx_antenna(antenna, int(channel))
            except Exception as exc:
                log.warning("set_antenna ch%d: %s", channel, exc)

    def set_blanker(self, name: str) -> None:
        name = str(name)
        self.blanker_name = name
        if not self._running:
            return
        rate = self.plan.hw_rate_hz if self.plan is not None else DEFAULT_HW_RATE_HZ
        self._init_blanker(rate)

    def set_nb_factor(self, k: float, channel: str = "h") -> None:
        k = float(k)
        if channel == "v":
            self.nb_factor_v = k
            if self._nb_state is not None:
                self._nb_state.nb_factor_v = k
        else:
            self.nb_factor = k
            if self._nb_state is not None:
                self._nb_state.nb_factor = k

    def status(self) -> RouterStatus:
        src = self._src
        dual = bool(getattr(src, "dual_channel", False)) if src else False
        if src is not None and not self._hw_info_cached:
            self._cache_hw_info(src)

        recv = int(getattr(src, "recv_count", 0) or 0) if src else 0
        now = time.monotonic()
        dt = now - self._prev_recv_t
        if dt >= 0.5:
            self._buf_per_s = (recv - self._prev_recv_count) / dt if dt > 0 else 0.0
            self._prev_recv_count = recv
            self._prev_recv_t = now

        def _dbfs(rms):
            if rms is None or rms < 1e-12:
                return None
            return float(20.0 * np.log10(rms))

        gain0 = self.gain_db
        gain1 = self.gain_db_ch1
        usrp = getattr(src, "_usrp", None) if src is not None else None
        if usrp is not None:
            try:
                gain0 = float(usrp.get_rx_gain(0))
            except Exception:
                pass
            if dual:
                try:
                    gain1 = float(usrp.get_rx_gain(1))
                except Exception:
                    pass

        return RouterStatus(
            running=self._running,
            pan_mhz=(self.plan.pan_center_mhz if self.plan else None),
            hw_rate_hz=(self.plan.hw_rate_hz if self.plan else None),
            dual=dual,
            serial=self._serial,
            firmware=self._firmware,
            fpga=self._fpga,
            recv_count=recv,
            buf_per_s=self._buf_per_s,
            rms_h_dbfs=_dbfs(self._rms_h),
            rms_v_dbfs=_dbfs(self._rms_v) if dual else None,
            blanker_name=self.blanker_name,
            blanked_pct=100.0 * self._blanked_ema,
            gain_db=gain0,
            gain_db_ch1=gain1 if dual else None,
            startup_phase=str(getattr(src, "startup_phase", "idle") if src else "idle"),
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _init_blanker(self, hw_rate_hz: int) -> None:
        self._blanker = make_blanker(self.blanker_name)
        self._nb_state = _BlankerState(
            sample_rate_hz=int(hw_rate_hz),
            nb_factor=self.nb_factor,
            nb_factor_v=self.nb_factor_v,
        )
        self._blanker.reset(self._nb_state)
        log.info("RouterEngine blanker=%s @ %d Hz K=%.1f/%.1f",
                 self._blanker.name, hw_rate_hz, self.nb_factor, self.nb_factor_v)

    def _blank_raw(self, raw_h, raw_v, dual: bool):
        if self._blanker is None or self._nb_state is None:
            return raw_h, raw_v
        try:
            result = self._blanker.process(
                self._nb_state, raw_h, raw_v if dual else None, bool(dual and raw_v is not None))
        except Exception as exc:
            log.warning("blanker error: %s — passing raw", exc)
            return raw_h, raw_v

        frac = float(result.blank_mask.mean()) if result.blank_mask.size else 0.0
        self._blanked_ema = 0.9 * self._blanked_ema + 0.1 * frac

        mag = result.mag_h
        if mag is not None and mag.size:
            rms = float(np.sqrt(np.mean(np.square(mag))))
            self._rms_h = rms if self._rms_h is None else 0.95 * self._rms_h + 0.05 * rms
        if result.mag_v is not None and result.mag_v.size:
            rmsv = float(np.sqrt(np.mean(np.square(result.mag_v))))
            self._rms_v = rmsv if self._rms_v is None else 0.95 * self._rms_v + 0.05 * rmsv

        return result.cleaned_h, result.cleaned_v

    def _cache_hw_info(self, src) -> None:
        usrp = getattr(src, "_usrp", None)
        if usrp is None:
            return
        try:
            info = usrp.get_usrp_rx_info(0)
            self._serial = info.get("mboard_serial", "—") or "—"
        except Exception:
            self._serial = "—"
        try:
            tree = usrp.get_tree()
            self._firmware = tree.access_str("/mboards/0/fw_version").get() or "—"
        except Exception:
            self._firmware = "—"
        try:
            tree = usrp.get_tree()
            self._fpga = tree.access_str("/mboards/0/fpga_version").get() or "—"
        except Exception:
            self._fpga = "—"
        self._hw_info_cached = True


__all__ = ["RouterConfig", "RouterEngine", "RouterStatus", "blanker_names"]
