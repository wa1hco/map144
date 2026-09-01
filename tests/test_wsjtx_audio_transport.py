"""Tests for #45 Transport abstraction and PortAudio device resolution."""
import sys
import types

import numpy as np
import pytest

import map144_app.wsjtx_audio_export as wae


class _FakeTransport(wae.AudioTransport):
    def __init__(self, hint="fake-hint"):
        self.wsjtx_input_hint = hint
        self.writes = []
        self.closed = False

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def close(self) -> None:
        self.closed = True


def test_exporter_feed_writes_pcm_via_injected_transport():
    tr = _FakeTransport()
    exp = wae.WsjtxAudioExporter(
        sink_name="map144.test.rx",
        description="MAP144 test -> WSJT-X",
        transport=tr,
    )
    # One chunk of unit DC → audio at 1500 Hz after upconvert
    n = 1200
    exp.feed(np.ones(n, dtype=np.complex64))
    # Writer is async; give it a moment
    import time
    for _ in range(50):
        if tr.writes:
            break
        time.sleep(0.01)
    exp.close()
    assert tr.closed
    assert tr.writes, "expected at least one PCM write"
    pcm = tr.writes[0]
    assert len(pcm) == n * 2  # int16
    assert exp._transport is None


def test_resolve_explicit_device_name():
    name = wae.resolve_output_device(device_name="Cable A Input", index=0)
    assert name == "Cable A Input"


def test_resolve_env_comma_list(monkeypatch):
    monkeypatch.setenv("MAP144_WSJTX_DEVICE", "Cable A Input, Cable B Input")
    assert wae.resolve_output_device(index=0) == "Cable A Input"
    assert wae.resolve_output_device(index=1) == "Cable B Input"
    assert wae.resolve_output_device(index=9) == "Cable B Input"  # clamp


def test_resolve_env_rf_override(monkeypatch):
    monkeypatch.setenv("MAP144_WSJTX_DEVICE", "Cable A Input")
    monkeypatch.setenv("MAP144_WSJTX_DEVICE_RF1", "VoiceMeeter Input")
    assert wae.resolve_output_device(index=1, rf=1) == "VoiceMeeter Input"
    assert wae.resolve_output_device(index=0, rf=0) == "Cable A Input"


def test_resolve_auto_match_fake_sd(monkeypatch):
    monkeypatch.delenv("MAP144_WSJTX_DEVICE", raising=False)
    monkeypatch.delenv("MAP144_WSJTX_DEVICE_RF0", raising=False)

    devices = [
        {"name": "Speakers", "max_output_channels": 2},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
        {"name": "Cable B Input (VB-Audio Cable B)", "max_output_channels": 2},
    ]
    fake_sd = types.SimpleNamespace(query_devices=lambda: devices)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    # Force re-import path inside resolve by calling list_matched directly
    matched = wae.list_matched_output_devices(fake_sd)
    assert len(matched) >= 2
    assert "CABLE Input" in matched[0][1] or "Cable" in matched[0][1]

    monkeypatch.setattr(sys, "platform", "win32")
    # resolve imports sounddevice — already in sys.modules
    name0 = wae.resolve_output_device(index=0)
    name1 = wae.resolve_output_device(index=1)
    assert "CABLE Input" in name0 or "Cable A" in name0 or "Cable" in name0
    assert name1 != "" 


def test_make_transport_linux_uses_pipewire(monkeypatch):
    created = {}

    class _FakePW(wae.AudioTransport):
        def __init__(self, sink_name, description, stream_name, fs):
            created["sink"] = sink_name
            self.wsjtx_input_hint = f"Monitor of {description}"

        def write(self, pcm): pass
        def close(self): pass

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(wae, "PipeWireTransport", _FakePW)
    t = wae.make_transport(
        sink_name="map144.RF0.rx",
        description="MAP144 RF0 RX -> WSJT-X",
        stream_name="feed",
        fs=12000,
    )
    assert created["sink"] == "map144.RF0.rx"
    assert "Monitor of" in t.wsjtx_input_hint


def test_make_transport_win_uses_portaudio(monkeypatch):
    created = {}

    class _FakePA(wae.AudioTransport):
        def __init__(self, device_name, fs, device_index=None):
            created["device"] = device_name
            self.wsjtx_input_hint = "CABLE Output"

        def write(self, pcm): pass
        def close(self): pass

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(wae, "PortAudioTransport", _FakePA)
    monkeypatch.setattr(wae, "resolve_output_device",
                        lambda **kw: "CABLE Input (VB-Audio Virtual Cable)")
    t = wae.make_transport(
        sink_name="map144.RF0.rx",
        description="MAP144 RF0 RX -> WSJT-X",
        stream_name="feed",
        fs=12000,
        index=0,
        rf=0,
    )
    assert "CABLE Input" in created["device"]
    assert t.wsjtx_input_hint == "CABLE Output"
