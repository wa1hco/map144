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
"""Source-layer tests: Source / SyntheticSource / WavSource / JsonlSink.

Validates the §3.5 dual-clock contract end-to-end:

- A produced :class:`Record` carries an independent sample-clock anchor
  (``start_sample``) and wall-clock anchor (``wall_clock_at_production``)
  with neither derived from the other.
- ``test_clock_drift_ppm`` injects synthetic drift into wall-clock-at-
  production at the configured ppm rate.
- ``ClockHealth`` events fire at ``clock_health_period_s``.
- CHARACTERIZE crossings log at INFO; WARN crossings log at WARNING.
- WavSource replays files faithfully; JsonlSink records events as JSONL.

Run as::

    cd ~/ham/map144
    python3 -m unittest tests.test_sources -v
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    Event,
    JsonlSink,
    JsonlSinkConfig,
    Record,
    Runtime,
    Source,
    Stream,
    SyntheticSource,
    SyntheticSourceConfig,
    WavSource,
    WavSourceConfig,
    make_event_stream,
    make_sample_stream,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _drain(stream: Stream, max_items: int = 1000, timeout: float = 0.01):
    """Drain whatever is currently in a stream into a list."""
    items = []
    for _ in range(max_items):
        try:
            items.append(stream.get(timeout=timeout))
        except Exception:
            break
    return items


def _write_iq_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate_hz: int = 48_000,
) -> None:
    """Write a tiny PCM int16 IQ WAV (L=I, R=Q) compatible with
    ``map144_app.runtime._load_wav_complex``.
    """
    if samples.dtype not in (np.complex64, np.complex128):
        raise ValueError("samples must be complex")
    re = np.real(samples).astype(np.float32)
    im = np.imag(samples).astype(np.float32)
    # Scale to int16; assume already in [-1, 1].
    peak = max(1e-9, max(np.abs(re).max(), np.abs(im).max()))
    if peak > 1.0:
        re = re / peak
        im = im / peak
    re_i16 = (re * 32767.0).astype(np.int16)
    im_i16 = (im * 32767.0).astype(np.int16)
    interleaved = np.empty(2 * len(samples), dtype=np.int16)
    interleaved[0::2] = re_i16
    interleaved[1::2] = im_i16
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(interleaved.tobytes())


# ---------------------------------------------------------------------
# §3.5 Dual-clock contract
# ---------------------------------------------------------------------


class TestDualClockContract(unittest.TestCase):
    """A produced Record carries both clock anchors as independent fields."""

    def test_record_has_both_anchors(self):
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=48_000,
                samples_per_record=1024,
                signal_mode="zero",
                realtime_pacing=False,
                stop_after_samples=4096,
            ),
        )
        consumer_stream = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=1024,
        )
        # Wire by hand so we can inspect the stream.
        src.outputs["iq"] = consumer_stream

        rt.start()
        try:
            time.sleep(0.05)
            recs = _drain(consumer_stream, max_items=100, timeout=0.1)
        finally:
            rt.stop(timeout_s=1.0)

        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIsInstance(r, Record)
            self.assertEqual(r.sample_rate_hz, 48_000)
            # start_sample is the canonical sample-clock time.
            self.assertGreaterEqual(r.start_sample, 0)
            # wall_clock_at_production is captured at production time.
            # Bound it loosely: should be near "now" at test time.
            self.assertGreater(r.wall_clock_at_production, time.time() - 60)
            self.assertLess(r.wall_clock_at_production, time.time() + 5)

        # Gap-free: each record's start_sample == previous end_sample.
        for prev, nxt in zip(recs[:-1], recs[1:]):
            self.assertEqual(nxt.start_sample, prev.end_sample)

    def test_drift_injection_produces_offset(self):
        """test_clock_drift_ppm=10000 (10 ms/s) makes wall-clock drift
        measurably ahead of real time over a short test window.
        """
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=48_000,
                samples_per_record=1024,
                signal_mode="zero",
                realtime_pacing=True,        # so sample clock tracks real time
                test_clock_drift_ppm=10_000.0,  # 10 ms drift per real second
            ),
        )
        out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=1024,
        )
        src.outputs["iq"] = out

        t0 = time.time()
        rt.start()
        try:
            time.sleep(0.5)
            recs = _drain(out, max_items=100, timeout=0.05)
        finally:
            rt.stop(timeout_s=1.0)
        elapsed = time.time() - t0

        self.assertGreater(len(recs), 1)
        last = recs[-1]
        # Implied drift = wall_clock_at_production - real_time_at_emission.
        # With 10 000 ppm = 1% rate skew, after ~0.5 s we expect ~5 ms drift.
        # Compare against the test's real elapsed for a generous bound.
        injected_offset = last.wall_clock_at_production - (t0 + last.start_sample / 48_000)
        # Expect drift to be POSITIVE (wall clock running ahead of sample clock).
        self.assertGreater(injected_offset, 0.0005,
                           f"injected_offset={injected_offset*1000:.2f} ms, expected > 0.5 ms")
        # And bounded by roughly elapsed * ppm.
        self.assertLess(injected_offset, elapsed * 10_000 * 1e-6 * 2,
                        f"injected_offset={injected_offset*1000:.2f} ms way over expected")


# ---------------------------------------------------------------------
# ClockHealth + tier logging (§3.5)
# ---------------------------------------------------------------------


class TestClockHealth(unittest.TestCase):

    def test_clock_health_event_emitted(self):
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=48_000,
                samples_per_record=1024,
                signal_mode="zero",
                realtime_pacing=True,
                clock_health_period_s=0.05,  # fast for the test
            ),
        )
        iq_out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=1024,
        )
        ch_out = make_event_stream("clock_health", capacity=64)
        src.outputs["iq"] = iq_out
        src.outputs["clock_health"] = ch_out

        rt.start()
        try:
            time.sleep(0.25)  # ~5 clock-health periods
            events = _drain(ch_out, max_items=50, timeout=0.05)
        finally:
            rt.stop(timeout_s=1.0)

        # Should have multiple ClockHealth events.
        ch_events = [e for e in events if isinstance(e, Event) and e.kind == "clock_health"]
        self.assertGreaterEqual(
            len(ch_events), 2,
            f"expected ≥2 clock_health events in 0.25 s at 0.05s period, got {len(ch_events)}",
        )
        # Each carries the §3.5 payload fields.
        e = ch_events[0]
        self.assertIn("implied_drift_seconds", e.payload)
        self.assertIn("drift_ppm", e.payload)
        self.assertIn("real_wall_clock_now", e.payload)
        self.assertIn("elapsed_seconds", e.payload)
        self.assertIn("test_drift_injection_ppm", e.payload)

    def test_characterize_log_at_info_level(self):
        """High-ppm injection in a short test window crosses the
        CHARACTERIZE threshold (2 ms) within ~0.5 s.  Verify the
        log line lands at INFO, not WARNING.
        """
        with self.assertLogs("map144_app.blocks.source", level="INFO") as caplog:
            rt = Runtime()
            src = rt.add(
                SyntheticSource("char-src"),
                SyntheticSourceConfig(
                    sample_rate_hz=48_000,
                    samples_per_record=1024,
                    signal_mode="zero",
                    realtime_pacing=True,
                    test_clock_drift_ppm=10_000.0,    # 10 ms/s; 2 ms in 0.2 s
                    clock_health_period_s=0.05,
                    drift_log_rate_limit_s=0.0,        # no rate limit for the test
                ),
            )
            iq_out = make_sample_stream(
                "iq", queue_depth_seconds=2.0,
                sample_rate_hz=48_000, n_samples_per_record=1024,
            )
            src.outputs["iq"] = iq_out
            rt.start()
            try:
                time.sleep(0.6)  # well past 2 ms drift
            finally:
                rt.stop(timeout_s=1.0)

        # At least one CHARACTERIZE line at INFO level.
        info_lines = [r for r in caplog.records
                      if r.levelname == "INFO" and "characterization" in r.getMessage()]
        warn_lines = [r for r in caplog.records
                      if r.levelname == "WARNING" and "WARN threshold" in r.getMessage()]
        self.assertGreater(
            len(info_lines), 0,
            f"expected at least one CHARACTERIZE INFO line, got {[r.getMessage() for r in caplog.records]}",
        )
        # Should NOT have crossed WARN (100 ms) in 0.6 s at 10 ms/s.
        # 0.6 * 0.01 = 6 ms drift << 100 ms.
        self.assertEqual(
            len(warn_lines), 0,
            f"unexpected WARN line: {[r.getMessage() for r in warn_lines]}",
        )

    def test_warn_log_at_warning_level(self):
        """Aggressive injection crosses the WARN threshold (100 ms) inside
        the test window.  Verify the log line lands at WARNING.
        """
        with self.assertLogs("map144_app.blocks.source", level="INFO") as caplog:
            rt = Runtime()
            src = rt.add(
                SyntheticSource("warn-src"),
                SyntheticSourceConfig(
                    sample_rate_hz=48_000,
                    samples_per_record=1024,
                    signal_mode="zero",
                    realtime_pacing=True,
                    test_clock_drift_ppm=500_000.0,   # 0.5 s/s — extreme
                    clock_health_period_s=0.05,
                    drift_log_rate_limit_s=0.0,
                ),
            )
            iq_out = make_sample_stream(
                "iq", queue_depth_seconds=2.0,
                sample_rate_hz=48_000, n_samples_per_record=1024,
            )
            src.outputs["iq"] = iq_out
            rt.start()
            try:
                time.sleep(0.6)  # 500 000 ppm × 0.6 s = 300 ms drift
            finally:
                rt.stop(timeout_s=1.0)

        warn_lines = [r for r in caplog.records
                      if r.levelname == "WARNING" and "WARN threshold" in r.getMessage()]
        self.assertGreater(
            len(warn_lines), 0,
            f"expected at least one WARN line, got: {[r.getMessage() for r in caplog.records]}",
        )


# ---------------------------------------------------------------------
# WavSource
# ---------------------------------------------------------------------


class TestWavSource(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_replay_round_trip(self):
        # 1 second of a 1 kHz tone at 48 kHz IQ.
        sr = 48_000
        n = 48_000
        t = np.arange(n) / sr
        samples = (0.5 * np.exp(1j * 2 * np.pi * 1000 * t)).astype(np.complex64)
        wav_path = self.tmpdir / "tone.wav"
        _write_iq_wav(wav_path, samples, sample_rate_hz=sr)

        rt = Runtime()
        src = rt.add(
            WavSource("wav"),
            WavSourceConfig(
                wav_path=str(wav_path),
                sample_rate_hz=sr,
                samples_per_record=1024,
                realtime_pacing=False,  # fast replay
            ),
        )
        iq_out = make_sample_stream(
            "iq", queue_depth_seconds=4.0,
            sample_rate_hz=sr, n_samples_per_record=1024,
        )
        src.outputs["iq"] = iq_out

        rt.start()
        try:
            # WavSource raises StopIteration at EOF — let it run to completion.
            recs = []
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    recs.append(iq_out.get(timeout=0.1))
                except Exception:
                    if not src._thread or not src._thread.is_alive():
                        break
        finally:
            rt.stop(timeout_s=1.0)
            recs.extend(_drain(iq_out, max_items=100, timeout=0.05))

        # Total samples == file content (within partial-trailing-record tolerance).
        total = sum(r.n_samples for r in recs)
        self.assertEqual(total, n)

        # Reassemble + spot-check that the WAV-roundtrip preserved the tone
        # frequency.  PCM int16 quantisation costs a few dB SNR, so we do
        # not check sample-exact equality — just that the dominant tone is
        # at 1 kHz.
        joined = np.concatenate([r.data for r in recs])
        self.assertEqual(joined.size, n)
        spec = np.abs(np.fft.fft(joined))
        peak_bin = int(np.argmax(spec))
        peak_hz = peak_bin * sr / n
        self.assertAlmostEqual(peak_hz, 1000.0, delta=5.0,
                               msg=f"expected dominant tone near 1 kHz, got {peak_hz:.1f} Hz")

    def test_missing_file_raises(self):
        src = WavSource("missing")
        src.configure(WavSourceConfig(
            wav_path=str(self.tmpdir / "nope.wav"),
            sample_rate_hz=48_000,
        ))
        with self.assertRaises(FileNotFoundError):
            src.on_start()

    def test_empty_path_raises(self):
        src = WavSource("blank")
        src.configure(WavSourceConfig(wav_path="", sample_rate_hz=48_000))
        with self.assertRaises(RuntimeError):
            src.on_start()


# ---------------------------------------------------------------------
# JsonlSink
# ---------------------------------------------------------------------


class TestJsonlSink(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_writes_events_as_jsonl(self):
        out_path = self.tmpdir / "events.jsonl"
        rt = Runtime()
        sink = rt.add(
            JsonlSink("sink"),
            JsonlSinkConfig(
                output_path=str(out_path),
                input_port_name="events",
                get_timeout_s=0.05,
            ),
        )
        # Wire an event stream by hand so the test owns the producer side.
        events_stream = make_event_stream("events", capacity=16)
        sink.inputs["events"] = events_stream

        rt.start()
        try:
            for i in range(5):
                events_stream.put(Event(
                    kind="detection",
                    occurred_at_sample=i * 256,
                    sample_rate_hz=12_000,
                    wall_clock_at_production=1_700_000_000.0 + i * 0.1,
                    payload={"channel": i, "fc_hz": 50_400_000 + i * 1000},
                ))
            time.sleep(0.2)
        finally:
            rt.stop(timeout_s=1.0)

        # Read back and parse.
        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 5)
        for i, line in enumerate(lines):
            obj = json.loads(line)
            self.assertEqual(obj["kind"], "detection")
            self.assertEqual(obj["occurred_at_sample"], i * 256)
            self.assertEqual(obj["sample_rate_hz"], 12_000)
            self.assertEqual(obj["payload"]["channel"], i)

    def test_kind_filter(self):
        out_path = self.tmpdir / "filtered.jsonl"
        rt = Runtime()
        sink = rt.add(
            JsonlSink("sink"),
            JsonlSinkConfig(
                output_path=str(out_path),
                input_port_name="events",
                kind_filter=["detection"],   # drop everything else
                get_timeout_s=0.05,
            ),
        )
        events_stream = make_event_stream("events", capacity=16)
        sink.inputs["events"] = events_stream

        rt.start()
        try:
            events_stream.put(Event("detection", 0, 12_000, 0.0, {}))
            events_stream.put(Event("clock_health", 0, 12_000, 0.0, {}))
            events_stream.put(Event("detection", 256, 12_000, 0.1, {}))
            events_stream.put(Event("decode", 256, 12_000, 0.2, {}))
            time.sleep(0.2)
        finally:
            rt.stop(timeout_s=1.0)

        lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
        self.assertEqual(len(lines), 2)
        kinds = [json.loads(ln)["kind"] for ln in lines]
        self.assertEqual(kinds, ["detection", "detection"])


# ---------------------------------------------------------------------
# §10 mini-validation: dual-clock-aware end-to-end
# ---------------------------------------------------------------------


class TestEndToEndClockHealth(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_synthetic_source_to_jsonl_sink(self):
        """SyntheticSource → ClockHealth events → JsonlSink → JSONL file.

        Smaller cousin of the §10 drift-injection test: verifies the
        Source / Stream / Sink pipeline records ClockHealth events with
        the dual-clock fields intact.
        """
        out_path = self.tmpdir / "ch.jsonl"
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=48_000,
                samples_per_record=1024,
                signal_mode="zero",
                realtime_pacing=True,
                clock_health_period_s=0.05,
            ),
        )
        sink = rt.add(
            JsonlSink("sink"),
            JsonlSinkConfig(
                output_path=str(out_path),
                input_port_name="events",
                get_timeout_s=0.05,
            ),
        )
        # Source needs an iq output (sample stream) plus the clock_health
        # event stream that feeds the sink.
        iq_out = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=1024,
        )
        src.outputs["iq"] = iq_out
        rt.connect(src, "clock_health", sink, "events",
                   kind=Stream.KIND_EVENTS, capacity=64)

        rt.start()
        try:
            time.sleep(0.3)
        finally:
            rt.stop(timeout_s=1.0)

        # JSONL file should contain multiple clock_health events.
        text = out_path.read_text(encoding="utf-8").strip()
        self.assertGreater(len(text), 0)
        lines = text.split("\n")
        for line in lines:
            obj = json.loads(line)
            self.assertEqual(obj["kind"], "clock_health")
            self.assertIn("implied_drift_seconds", obj["payload"])
            self.assertIn("drift_ppm", obj["payload"])
            # wall_clock_at_production is a real timestamp.
            self.assertGreater(obj["wall_clock_at_production"], time.time() - 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
