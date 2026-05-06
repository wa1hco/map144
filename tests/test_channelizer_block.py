#!/usr/bin/env python3
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
"""Tests for ``map144_app.blocks.ChannelizerBlock``.

Validates:

- output shape ``(N_CHANNELS, N // decimate_factor)``
- output ``sample_rate_hz`` is decimated correctly
- output ``start_sample`` advances at the decimated rate, gap-free
- ``wall_clock_at_production`` is inherited from the upstream record (§3.5)
- a complex tone at ``k * channel_spacing_hz`` lands in channel ``k``
- carry buffering across ticks for misaligned record lengths
- end-to-end SyntheticSource → ChannelizerBlock pipeline

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_channelizer_block -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    ChannelizerBlock,
    ChannelizerBlockConfig,
    Record,
    Runtime,
    Stream,
    SyntheticSource,
    SyntheticSourceConfig,
    make_sample_stream,
)


N_CHANNELS = 48
DECIMATE = 4
SR_IN = 48_000
SR_OUT = SR_IN // DECIMATE


def _push_record(stream: Stream, data: np.ndarray, *, start_sample: int,
                 sample_rate_hz: int = SR_IN, wall_clock: float = 0.0) -> None:
    """Helper: place a Record on the stream with explicit anchors."""
    stream.put(Record(
        data=np.ascontiguousarray(data, dtype=np.complex64),
        sample_rate_hz=sample_rate_hz,
        start_sample=start_sample,
        wall_clock_at_production=wall_clock,
    ))


def _drain_records(stream: Stream, max_items: int = 200, timeout: float = 0.05):
    items: list[Record] = []
    for _ in range(max_items):
        try:
            items.append(stream.get(timeout=timeout))
        except Exception:
            break
    return items


# ---------------------------------------------------------------------
# Basic shape / contract
# ---------------------------------------------------------------------


class TestChannelizerShape(unittest.TestCase):

    def _run_and_get(self, in_records: list[np.ndarray],
                     start_samples: list[int],
                     wall_clocks: list[float] | None = None) -> list[Record]:
        if wall_clocks is None:
            wall_clocks = [1_700_000_000.0 + 0.001 * i for i in range(len(in_records))]

        rt = Runtime()
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=SR_IN,
                n_channels=N_CHANNELS,
                decimate_factor=DECIMATE,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        in_stream = make_sample_stream(
            "iq", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN, n_samples_per_record=1024,
        )
        out_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_OUT, n_samples_per_record=256,
        )
        ch.inputs["iq"] = in_stream
        ch.outputs["channels"] = out_stream

        rt.start()
        try:
            for arr, ss, wc in zip(in_records, start_samples, wall_clocks):
                _push_record(in_stream, arr, start_sample=ss, wall_clock=wc)
            time.sleep(0.2)
        finally:
            rt.stop(timeout_s=1.0)

        return _drain_records(out_stream)

    def test_basic_shape_and_rate(self):
        # 1024 samples in → 256 samples out, shape (48, 256).
        data = np.zeros(1024, dtype=np.complex64)
        recs = self._run_and_get([data], [0])
        self.assertEqual(len(recs), 1)
        out = recs[0]
        self.assertEqual(out.data.shape, (N_CHANNELS, 256))
        self.assertEqual(out.sample_rate_hz, SR_OUT)
        self.assertEqual(out.start_sample, 0)

    def test_wall_clock_inherited(self):
        """§3.5: Channelizer inherits wall_clock_at_production from upstream."""
        data = np.zeros(1024, dtype=np.complex64)
        wc = 1_700_000_042.5
        recs = self._run_and_get([data], [0], [wc])
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].wall_clock_at_production, wc, places=6)

    def test_start_sample_advances_at_decimated_rate(self):
        """Sequential aligned records → output start_samples are 0, 256, 512, ..."""
        data = np.zeros(1024, dtype=np.complex64)
        recs = self._run_and_get(
            [data, data, data],
            [0, 1024, 2048],
        )
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0].start_sample, 0)
        self.assertEqual(recs[1].start_sample, 256)
        self.assertEqual(recs[2].start_sample, 512)


# ---------------------------------------------------------------------
# Spectral correctness — tone lands in the right channel
# ---------------------------------------------------------------------


class TestChannelizerSpectral(unittest.TestCase):

    def test_tone_appears_in_expected_channel_at_expected_offset(self):
        """A complex tone at fc(k) + delta lands in channel k's output at
        +delta Hz (in the channel's local 12 kHz spectrum).

        Note: each channel has a 300 Hz high-pass at its output, so a tone
        at exactly fc(k) (DC after mixing) would be *rejected*.  The test
        deliberately puts the tone +500 Hz off the channel center.

        Validates two things at once:
        1. channel k captures the tone (peak in channel k's spectrum)
        2. the tone appears at the expected offset (channelizer NCO math
           is correct)
        """
        rt = Runtime()
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=SR_IN,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        in_stream = make_sample_stream(
            "iq", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN, n_samples_per_record=4096,
        )
        out_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_OUT, n_samples_per_record=1024,
        )
        ch.inputs["iq"] = in_stream
        ch.outputs["channels"] = out_stream

        # 0.5 s of a +5500 Hz tone — fc(5) + 500.  Channel 5 sees it at
        # +500 Hz (HP- and LP-pass); channel 6 sees it at -500 Hz (also
        # in passband, so equal power); channel 4 at +1500 Hz, channel 7
        # at -1500 Hz (also in passband).  We do not test channel-level
        # peak power because the LP/HP shapes do not give a unique winner
        # with 1 kHz channel spacing.
        n = 24_000
        t = np.arange(n) / SR_IN
        tone_hz = 5500.0
        delta_hz = 500.0  # offset from fc(5) = 5000
        tone = (0.5 * np.exp(2j * np.pi * tone_hz * t)).astype(np.complex64)

        rt.start()
        try:
            _push_record(in_stream, tone, start_sample=0)
            time.sleep(0.3)
        finally:
            rt.stop(timeout_s=1.0)

        recs = _drain_records(out_stream)
        self.assertEqual(len(recs), 1, f"expected 1 output record, got {len(recs)}")
        ch_out = recs[0].data            # shape (48, n//4)

        # Drop the FIR transient at the start.
        ch5 = ch_out[5, 200:]
        # FFT of channel 5's output; spike should be at +delta_hz.
        spec = np.fft.fftshift(np.fft.fft(ch5))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(ch5), 1.0 / SR_OUT))
        peak_idx = int(np.argmax(np.abs(spec)))
        peak_hz = float(freqs[peak_idx])
        self.assertAlmostEqual(
            peak_hz, delta_hz, delta=10.0,
            msg=f"channel 5 spectrum peak at {peak_hz:.1f} Hz, expected {delta_hz:.1f} Hz",
        )
        # And channel 20 (well outside ±2700 Hz of the tone) is in the
        # LP stopband for the tone — its mean power should be tiny vs
        # channel 5's.
        p5 = float(np.mean(np.abs(ch5) ** 2))
        p20 = float(np.mean(np.abs(ch_out[20, 200:]) ** 2))
        ratio_db = 10.0 * np.log10(p5 / max(p20, 1e-30))
        self.assertGreater(
            ratio_db, 40.0,
            f"channel 5 vs 20 power ratio = {ratio_db:.1f} dB, expected > 40 dB",
        )


# ---------------------------------------------------------------------
# Carry buffering across misaligned record lengths
# ---------------------------------------------------------------------


class TestChannelizerCarry(unittest.TestCase):
    """When input record lengths aren't divisible by ``decimate_factor``,
    the block must carry the trailing samples to the next tick so no
    sample is lost or misanchored.
    """

    def test_misaligned_records_dont_lose_samples(self):
        """Push 4 records of 101 samples each (4 * 101 = 404 input samples;
        404 / 4 = 101 output samples).  Verify cumulative output count is
        exactly 101 and start_sample advances correctly.
        """
        rt = Runtime()
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=SR_IN,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        in_stream = make_sample_stream(
            "iq", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN, n_samples_per_record=128,
        )
        out_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_OUT, n_samples_per_record=32,
        )
        ch.inputs["iq"] = in_stream
        ch.outputs["channels"] = out_stream

        n_per = 101
        rt.start()
        try:
            for i in range(4):
                _push_record(
                    in_stream,
                    np.zeros(n_per, dtype=np.complex64),
                    start_sample=i * n_per,
                )
            time.sleep(0.2)
        finally:
            rt.stop(timeout_s=1.0)

        recs = _drain_records(out_stream)

        # Sum of output sample counts must equal total_input // decim.
        total_out = sum(r.data.shape[-1] for r in recs)
        self.assertEqual(total_out, (4 * n_per) // DECIMATE)

        # start_sample is gap-free at the output.
        cursor = 0
        for r in recs:
            self.assertEqual(r.start_sample, cursor,
                             f"output gap: got start_sample={r.start_sample}, expected {cursor}")
            cursor += r.data.shape[-1]

    def test_subaligned_first_record_buffers_silently(self):
        """A first record shorter than ``decimate_factor`` cannot produce
        any output yet — the block holds it as carry and emits nothing
        until enough samples arrive.
        """
        rt = Runtime()
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=SR_IN,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        in_stream = make_sample_stream(
            "iq", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN, n_samples_per_record=128,
        )
        out_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_OUT, n_samples_per_record=32,
        )
        ch.inputs["iq"] = in_stream
        ch.outputs["channels"] = out_stream

        rt.start()
        try:
            # 3 samples — less than DECIMATE (4); should hold as carry.
            _push_record(in_stream, np.zeros(3, dtype=np.complex64), start_sample=0)
            time.sleep(0.10)
            recs_partial = _drain_records(out_stream, timeout=0.02)
            self.assertEqual(len(recs_partial), 0,
                             "expected no output yet; carry should be holding 3 samples")
            # Push more samples; total now 3+5=8, aligned=8, output=2 samples.
            _push_record(in_stream, np.zeros(5, dtype=np.complex64), start_sample=3)
            time.sleep(0.10)
        finally:
            rt.stop(timeout_s=1.0)

        recs = _drain_records(out_stream)
        self.assertGreater(len(recs), 0)
        total_out = sum(r.data.shape[-1] for r in recs)
        self.assertEqual(total_out, 2,
                         f"expected 8/4=2 output samples total, got {total_out}")


# ---------------------------------------------------------------------
# End-to-end with SyntheticSource
# ---------------------------------------------------------------------


class TestSourceToChannelizerEndToEnd(unittest.TestCase):

    def test_synthetic_tone_into_channelizer(self):
        """SyntheticSource (1500 Hz tone) → ChannelizerBlock.

        With ``channel_offset_hz=0`` channel 1 is centred at 1000 Hz, so
        a 1500 Hz tone appears at +500 Hz in channel 1's output spectrum.
        We verify that spectral peak rather than checking which channel
        has the highest mean power — the 300 Hz HP filter at each channel
        output and the wide LP rolloff make multiple neighbours bright.
        """
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=SR_IN,
                samples_per_record=4096,
                signal_mode="tone",
                tone_hz=1500.0,
                tone_amplitude=0.5,
                realtime_pacing=False,
                stop_after_samples=20_480,    # 5 records of 4096
            ),
        )
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=SR_IN,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        rt.connect(src, "iq", ch, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=SR_IN, n_samples_per_record=4096)
        out_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_OUT, n_samples_per_record=1024,
        )
        ch.outputs["channels"] = out_stream

        rt.start()
        try:
            time.sleep(0.5)
        finally:
            rt.stop(timeout_s=1.0)

        recs = _drain_records(out_stream)
        self.assertGreater(len(recs), 0)

        # Concatenate per-channel; drop FIR transient.
        ch_out = np.concatenate([r.data for r in recs], axis=1)
        self.assertEqual(ch_out.shape[0], N_CHANNELS)
        ch1 = ch_out[1, 400:]  # drop FIR transient

        # Channel 1's spectrum should peak at +500 Hz.
        spec = np.fft.fftshift(np.fft.fft(ch1))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(ch1), 1.0 / SR_OUT))
        peak_hz = float(freqs[int(np.argmax(np.abs(spec)))])
        self.assertAlmostEqual(
            peak_hz, 500.0, delta=10.0,
            msg=f"channel 1 spectrum peak at {peak_hz:.1f} Hz, expected +500 Hz",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
