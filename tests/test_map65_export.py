# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for map144_app/map65_export.py (MAP65 timf2 I/Q export).

Covers the DSP (decimate, NCO tone placement, int16 scaling) and the transport
(timf2 header layout, int16 interleave I_h,Q_h,I_v,Q_v, packetization/ptr).
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map144_app.map65_export import (
    Map65Exporter, build_timf2_header, _make_decimator, _nco_shift,
    _clip_int16, TIMF2_HEADER_SIZE, TIMF2_HEADER_FMT, HW_RATE_IN, MAP65_RATE,
    DECIMATE, BYTES_PER_CHANNEL, NET_MULTICAST_PAYLOAD,
)


# ── timf2 header ─────────────────────────────────────────────────────────────

def test_header_size_is_24():
    assert TIMF2_HEADER_SIZE == 24
    assert struct.calcsize(TIMF2_HEADER_FMT) == 24


def test_header_roundtrips_fields():
    pkt = build_timf2_header(144.125, 12_345_678, ptr=2048, block_no=7,
                             userx_freq=0.0, userx_no=0, passband_direction=1)
    pc, t, uf, ptr, blk, uno, pdir = struct.unpack(TIMF2_HEADER_FMT, pkt)
    assert pc == 144.125            # passband_center is the *delivered* centre
    assert t == 12_345_678
    assert ptr == 2048
    assert blk == 7
    assert uno == 0
    assert pdir == 1


def test_header_is_little_endian():
    # block_no=1 little-endian uint16 -> 0x01 0x00 at its offset (20).
    pkt = build_timf2_header(0.0, 0, 0, 1)
    assert pkt[20] == 1 and pkt[21] == 0


# ── int16 scaling ────────────────────────────────────────────────────────────

def test_clip_int16_rounds_and_saturates():
    # Saturates just INSIDE the int16 range — never emits the ±full-scale
    # extremes that crash MAP65's INTEGER*2 sync (-32768 / +32767).
    x = np.array([-40000.0, -0.4, 0.6, 1.5, 40000.0])
    out = _clip_int16(x)
    assert out.dtype == np.int16
    assert list(out) == [-32766, 0, 1, 2, 32766]


# ── decimator ────────────────────────────────────────────────────────────────

def test_decimator_output_length_and_dc_gain():
    dec = _make_decimator(DECIMATE)
    x = np.ones(3840, dtype=np.complex64)         # DC, amplitude 1
    y = dec(x)
    assert len(y) == 3840 // DECIMATE             # exact integer decimation
    # firwin is unity-gain at DC; steady-state output ~= 1.
    assert abs(y[-1] - 1.0) < 1e-3


def test_decimator_rejects_out_of_band():
    # A tone near the input Nyquist (96 kHz) is in the /2 stopband -> attenuated.
    dec = _make_decimator(DECIMATE)
    n = 8192
    f = 80_000.0                                   # > 48 kHz output Nyquist
    t = np.arange(n)
    x = np.exp(1j * 2 * np.pi * f * t / HW_RATE_IN).astype(np.complex64)
    y = dec(x)
    # Compare residual power to a DC (passband) reference of equal input power.
    ref = dec(np.ones(n, dtype=np.complex64))
    p_stop = np.mean(np.abs(y[-100:]) ** 2)
    p_pass = np.mean(np.abs(ref[-100:]) ** 2)
    assert p_stop < 0.01 * p_pass                  # > 20 dB rejection


# ── NCO + decimate: tone placement ───────────────────────────────────────────

def _peak_bin_freq(y, fs):
    """Return the frequency (Hz, signed) of the strongest FFT bin of y."""
    Y = np.fft.fftshift(np.fft.fft(y))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(y), 1.0 / fs))
    return freqs[np.argmax(np.abs(Y))]


def test_map65_center_tone_lands_at_dc():
    # shift_hz default = 25 kHz.  A tone at +25 kHz (the MAP65 centre relative to
    # the 144.100 DC) must end up at DC in the 96 kHz stream.
    exp = Map65Exporter(pan_center_mhz=144.100, map65_center_mhz=144.125)
    n = 19_200
    t = np.arange(n)
    tone = np.exp(1j * 2 * np.pi * 25_000.0 * t / HW_RATE_IN).astype(np.complex64)
    shifted, _ = _nco_shift(tone, 0.0, exp._step)
    y = _make_decimator(DECIMATE)(shifted)
    assert abs(_peak_bin_freq(y, MAP65_RATE)) < 200.0   # at DC


def test_dc_artifact_lands_below_center():
    # The hardware DC artifact (a tone at 0 Hz = 144.100) must appear at
    # -shift_hz (= -25 kHz) in MAP65's window -> 144.100 MHz, below activity.
    exp = Map65Exporter(pan_center_mhz=144.100, map65_center_mhz=144.125)
    n = 19_200
    dc = np.ones(n, dtype=np.complex64)
    shifted, _ = _nco_shift(dc, 0.0, exp._step)
    y = _make_decimator(DECIMATE)(shifted)
    assert abs(_peak_bin_freq(y, MAP65_RATE) - (-25_000.0)) < 200.0


# ── End-to-end: interleave order + packetization ─────────────────────────────

class _FakeSock:
    def __init__(self):
        self.sent = []
    def sendto(self, data, addr):
        self.sent.append((bytes(data), addr))
    def close(self):
        pass


def _make_exporter_no_shift(n_channels=2, scale=1.0):
    # map65_center == pan_center -> shift_hz = 0 -> NCO is identity, so a DC
    # (constant) input passes through the decimator unchanged at steady state,
    # letting us read back the exact interleaved values.
    exp = Map65Exporter(host="127.0.0.1", port=50002,
                        pan_center_mhz=144.100, map65_center_mhz=144.100,
                        level_scale=scale, n_channels=n_channels)
    exp._sock = _FakeSock()
    exp.enabled = True
    assert exp.shift_hz == 0.0
    return exp


def test_payload_is_always_net_multicast_payload():
    # The packet payload must equal NET_MULTICAST_PAYLOAD for both channel
    # counts, or MAP65 mis-aligns its ring buffer (the dense-lines bug).
    exp1 = _make_exporter_no_shift(n_channels=1)
    exp2 = _make_exporter_no_shift(n_channels=2)
    assert exp1.frames_per_packet * exp1._bytes_per_frame == NET_MULTICAST_PAYLOAD
    assert exp2.frames_per_packet * exp2._bytes_per_frame == NET_MULTICAST_PAYLOAD
    assert exp1.userx_no == 1 and exp2.userx_no == 2


def test_interleave_order_i_h_q_h_i_v_q_v():
    exp = _make_exporter_no_shift(n_channels=2, scale=1.0)
    bpf = BYTES_PER_CHANNEL * 2                       # 8 bytes/frame (dual)
    n = 4096
    h = np.full(n, 10 + 20j, dtype=np.complex64)
    v = np.full(n, 30 + 40j, dtype=np.complex64)
    exp.process(h, v, ts_epoch=0.0)
    assert exp.packets_sent >= 1
    data, addr = exp._sock.sent[-1]
    assert addr == ("127.0.0.1", 50002)
    assert len(data) == TIMF2_HEADER_SIZE + NET_MULTICAST_PAYLOAD
    payload = data[TIMF2_HEADER_SIZE:]
    frame = struct.unpack("<4h", payload[-bpf:])     # last (steady) frame
    assert frame == (10, 20, 30, 40)                 # I_h, Q_h, I_v, Q_v


def test_single_channel_format_and_interleave():
    exp = _make_exporter_no_shift(n_channels=1, scale=1.0)
    assert exp._bytes_per_frame == BYTES_PER_CHANNEL  # 4 bytes/frame (I,Q)
    h = np.full(2048, 7 + 9j, dtype=np.complex64)
    exp.process(h, None, ts_epoch=0.0)               # v ignored in single-pol
    assert exp.packets_sent >= 1
    data = exp._sock.sent[-1][0]
    _, _, _, _, _, uno, _ = struct.unpack(TIMF2_HEADER_FMT, data[:TIMF2_HEADER_SIZE])
    assert uno == 1                                  # +1 => 1 channel, int16
    frame = struct.unpack("<2h", data[-BYTES_PER_CHANNEL:])
    assert frame == (7, 9)                           # I_h, Q_h


def test_packetization_count_and_ptr_advances():
    exp = _make_exporter_no_shift(n_channels=1, scale=1.0)
    fpp = exp.frames_per_packet                      # 348 (= 1392 / 4)
    pkt_bytes = fpp * exp._bytes_per_frame           # == NET_MULTICAST_PAYLOAD
    assert pkt_bytes == NET_MULTICAST_PAYLOAD
    # Feed exactly 3 packets' worth: 3*fpp output frames = 3*fpp*DECIMATE input.
    n_in = 3 * fpp * DECIMATE
    exp.process(np.ones(n_in, dtype=np.complex64), None, ts_epoch=0.0)
    assert exp.packets_sent == 3
    assert len(exp._pending) == 0
    pc, t, uf, ptr, blk, uno, pdir = struct.unpack(
        TIMF2_HEADER_FMT, exp._sock.sent[-1][0][:TIMF2_HEADER_SIZE])
    assert ptr == (2 * pkt_bytes) % exp.buffer_size  # ptr increments AFTER send
    assert blk == 2


def test_disabled_exporter_sends_nothing():
    exp = Map65Exporter()
    # not started -> enabled False, no socket
    exp.process(np.ones(3840, dtype=np.complex64),
                np.ones(3840, dtype=np.complex64), 0.0)
    assert exp.packets_sent == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
