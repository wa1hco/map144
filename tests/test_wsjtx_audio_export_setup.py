"""Tests for the B210 WSJT-X per-RF-port audio-export setup/teardown.

These exercise the gating, naming, and routing-table logic in
runtime._setup_wsjtx_audio_export / _teardown_wsjtx_audio_export without
touching PipeWire — the real WsjtxAudioExporter is replaced by a fake that
records its sink_name/label and a close() call.
"""
import sys
import types
import pytest

import map144_app.runtime as runtime
import map144_app.wsjtx_audio_export as wae


class _FakeExporter:
    instances = []

    def __init__(self, sink_name=None, label=None, description=None,
                 stream_name=None, **kw):
        self.sink = sink_name
        self.label = label
        self.description = description
        self.stream_name = stream_name
        self.closed = False
        _FakeExporter.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_exporter(monkeypatch):
    _FakeExporter.instances = []
    # _setup imports the symbol from the module at call time, so patch there.
    monkeypatch.setattr(wae, "WsjtxAudioExporter", _FakeExporter)
    monkeypatch.setattr(sys, "platform", "linux")
    return _FakeExporter


def _obj():
    """Minimal stand-in for the visualizer (only the attrs the helpers touch)."""
    return types.SimpleNamespace(_wsjtx_exports={})


def test_dual_creates_rf0_and_rf1(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    assert set(self._wsjtx_exports) == {0, 1}
    # programmatic name = dotted machine token, with .rx direction suffix
    assert self._wsjtx_exports[0].sink == "map144.RF0.rx"
    assert self._wsjtx_exports[1].sink == "map144.RF1.rx"
    assert self._wsjtx_exports[0].label == "RF0"
    assert self._wsjtx_exports[1].label == "RF1"


def test_description_and_stream_name_are_human_readable(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    # description is what WSJT-X/pavucontrol show -> spaced prose with the arrow
    assert self._wsjtx_exports[0].description == "MAP144 RF0 RX -> WSJT-X"
    assert self._wsjtx_exports[1].description == "MAP144 RF1 RX -> WSJT-X"
    # stream_name labels the paplay producer and names the target sink
    assert "RF0 RX" in self._wsjtx_exports[0].stream_name
    assert "map144.RF0.rx" in self._wsjtx_exports[0].stream_name


def test_single_channel_creates_only_rf0(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=False)
    assert set(self._wsjtx_exports) == {0}
    assert self._wsjtx_exports[0].sink == "map144.RF0.rx"


def test_default_on_when_env_unset(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    assert self._wsjtx_exports, "bridge should be on by default for B210/Linux"


def test_env_zero_disables(fake_exporter, monkeypatch):
    monkeypatch.setenv("MAP144_WSJTX_AUDIO", "0")
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    assert self._wsjtx_exports == {}


def test_windows_enabled_by_default(fake_exporter, monkeypatch):
    """#45: bridge is no longer Linux-only — PortAudio path on win/mac."""
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    assert set(self._wsjtx_exports) == {0, 1}
    assert self._wsjtx_exports[0].sink == "map144.RF0.rx"


def test_idempotent_when_already_built(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    n_after_first = len(_FakeExporter.instances)
    runtime._setup_wsjtx_audio_export(self, dual=True)   # second call no-ops
    assert len(_FakeExporter.instances) == n_after_first


def test_one_failed_port_does_not_block_other(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)

    def _maybe_fail(sink_name=None, label=None, **kw):
        if sink_name == "map144.RF0.rx":
            raise RuntimeError("pactl exploded")
        return _FakeExporter(sink_name=sink_name, label=label, **kw)

    monkeypatch.setattr(wae, "WsjtxAudioExporter", _maybe_fail)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    assert set(self._wsjtx_exports) == {1}              # RF1 still came up


def test_teardown_closes_and_clears(fake_exporter, monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_AUDIO", raising=False)
    self = _obj()
    runtime._setup_wsjtx_audio_export(self, dual=True)
    exps = list(self._wsjtx_exports.values())
    runtime._teardown_wsjtx_audio_export(self)
    assert all(e.closed for e in exps)
    assert self._wsjtx_exports == {}
