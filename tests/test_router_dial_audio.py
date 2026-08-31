"""Offline tone test for DialAudioBank NCO + decimate path."""
import numpy as np
import pytest

from map144_app.router.band_plan import DialChannel
from map144_app.router.dial_audio import AUDIO_FS, DialAudioBank


class _CaptureExporter:
    """Stand-in for WsjtxAudioExporter: records baseband IQ fed to it."""

    def __init__(self, sink_name=None, label=None, description=None,
                 stream_name=None, fs=AUDIO_FS, **kw):
        self.sink = sink_name
        self.label = label
        self.description = description
        self.stream_name = stream_name
        self.fs = fs
        self.chunks = []
        self.closed = False

    def feed(self, ch_iq):
        self.chunks.append(np.asarray(ch_iq, dtype=np.complex64).copy())

    def close(self):
        self.closed = True

    @property
    def iq(self):
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.complex64)


def _peak_hz(iq, fs):
    """Dominant frequency of complex IQ via FFT (Hz, signed)."""
    n = len(iq)
    win = np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(iq * win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    return float(freqs[int(np.argmax(np.abs(spec)))])


def test_tone_at_dial_lands_at_dc_after_bank():
    """A CW tone at dial+1500 in the raw IF must appear near DC at 12 kHz."""
    hw = 192_000
    pan_mhz = 50.2865  # midpoint-ish of 50.260 / 50.313
    dials = [
        DialChannel("6m", "MSK144", 50.260, "MSK144"),
        DialChannel("6m", "FT8", 50.313, "FT8"),
    ]
    bank = DialAudioBank(
        dials,
        pan_center_mhz=pan_mhz,
        hw_rate_hz=hw,
        exporter_factory=_CaptureExporter,
    )
    # 100 ms of raw IQ: two unit tones at each USB centre
    n = hw // 10
    t = np.arange(n, dtype=np.float64) / hw
    pan_hz = pan_mhz * 1e6
    raw = np.zeros(n, dtype=np.complex64)
    for ch in dials:
        f_if = ch.usb_center_hz - pan_hz
        raw += np.exp(1j * 2 * np.pi * f_if * t).astype(np.complex64)

    # Process in 20 ms blocks (matches USRP recv size at 192 kHz)
    block = 3840
    for i in range(0, n - block + 1, block):
        bank.process(raw[i:i + block])

    assert len(bank.branches) == 2
    for br in bank.branches:
        iq = br.exporter.iq
        # Drop startup transient from decimator state
        iq = iq[len(iq) // 5:]
        assert len(iq) > 500
        peak = _peak_hz(iq, AUDIO_FS)
        assert abs(peak) < 50.0, f"{br.channel.mode}: peak {peak:.1f} Hz"

    names = bank.sink_names
    assert "map144.6m.msk144.50260.rx" in names
    assert "map144.6m.ft8.50313.rx" in names
    exps = [br.exporter for br in bank.branches]
    bank.close()
    assert all(e.closed for e in exps)


def test_bank_close_closes_exporters():
    dials = [DialChannel("2m", "FT8", 144.174, "FT8")]
    bank = DialAudioBank(
        dials, pan_center_mhz=144.174, hw_rate_hz=192_000,
        exporter_factory=_CaptureExporter,
    )
    exps = [br.exporter for br in bank.branches]
    bank.close()
    assert all(e.closed for e in exps)
    assert bank.branches == []
