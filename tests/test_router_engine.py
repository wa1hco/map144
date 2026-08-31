"""Lifecycle tests for RouterEngine with fakes (no UHD / PipeWire)."""
import types

import numpy as np
import pytest

from map144_app.router.band_plan import DialChannel
from map144_app.router.engine import RouterConfig, RouterEngine
from map144_app.router.lo_planner import LoPlanError
from map144_app.router.wideband_iq import WidebandDest


class _FakeExporter:
    instances = []

    def __init__(self, sink_name=None, label=None, description=None,
                 stream_name=None, fs=12000, **kw):
        self.sink = sink_name
        self.label = label
        self.description = description
        self.stream_name = stream_name
        self.fs = fs
        self.feeds = 0
        self.closed = False
        _FakeExporter.instances.append(self)

    def feed(self, ch_iq):
        self.feeds += 1

    def close(self):
        self.closed = True


class _FakeMap65:
    def __init__(self, host="127.0.0.1", port=50002, pan_center_mhz=144.1,
                 map65_center_mhz=144.125, n_channels=1, **kw):
        self.host = host
        self.port = port
        self.pan_center_mhz = pan_center_mhz
        self.map65_center_mhz = map65_center_mhz
        self.n_channels = n_channels
        self.enabled = False
        self.packets = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.enabled = True
        self.started = True

    def stop(self):
        self.enabled = False
        self.stopped = True

    def process(self, raw_h, raw_v, ts_epoch):
        self.packets += 1


class _FakeUSRP:
    def __init__(self, **kw):
        self.kw = kw
        self.raw_only = False
        self.raw_iq_callback = None
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_start_stop_creates_sinks_and_tears_down():
    _FakeExporter.instances = []
    eng = RouterEngine()
    dials = [
        DialChannel("2m", "MSK144", 144.150, "MSK144"),
        DialChannel("2m", "FT8", 144.174, "FT8"),
    ]
    cfg = RouterConfig(dials=dials, dual_channel=False)
    plan = eng.start(
        cfg,
        usrp_factory=_FakeUSRP,
        audio_exporter_factory=_FakeExporter,
    )
    assert eng.running
    assert plan.hw_rate_hz == 192_000
    assert len(eng.sink_names) == 2
    assert eng._src.raw_only is True
    assert eng._src.raw_iq_callback is not None

    # Simulate one raw block through the callback
    n = 3840
    raw = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    eng._src.raw_iq_callback(raw, None, 0.0)
    assert all(e.feeds >= 1 for e in _FakeExporter.instances)

    eng.stop()
    assert not eng.running
    assert eng._src is None
    assert all(e.closed for e in _FakeExporter.instances)


def test_map65_destination_starts():
    eng = RouterEngine()
    cfg = RouterConfig(
        dials=[],
        map65=WidebandDest("MAP65", "127.0.0.1", 50002, 144.125),
        dual_channel=True,
    )
    fake_m65 = []

    def factory(**kw):
        e = _FakeMap65(**kw)
        fake_m65.append(e)
        return e

    eng.start(cfg, usrp_factory=_FakeUSRP, map65_exporter_factory=factory)
    assert len(fake_m65) == 1
    assert fake_m65[0].started
    raw = np.zeros(3840, np.complex64)
    eng._src.raw_iq_callback(raw, raw, 1.0)
    assert fake_m65[0].packets == 1
    eng.stop()
    assert fake_m65[0].stopped


def test_empty_config_raises():
    eng = RouterEngine()
    with pytest.raises(LoPlanError):
        eng.start(RouterConfig(), usrp_factory=_FakeUSRP)


def test_plan_only_preview():
    eng = RouterEngine()
    dials = [DialChannel("6m", "MSK144", 50.260, "MSK144"),
             DialChannel("6m", "FT8", 50.313, "FT8")]
    plan = eng.plan_only(RouterConfig(dials=dials))
    assert plan.hw_rate_hz == 192_000
    assert not eng.running
