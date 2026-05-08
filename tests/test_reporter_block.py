#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Tests for ``map144_app.blocks.ReporterBlock`` (Phase 4 #12).

The wrapped :class:`Reporter` does real network I/O (PSKReporter
IPFIX, DX cluster TCP, WSJT-X UDP).  These tests substitute a fake
Reporter so we can verify the block's filtering + dict-conversion
behaviour without opening sockets.

Validates:

- Decoded events trigger ``reporter.report_decode`` with the legacy
  decode-dict shape (message, jt9_snr, t_sec, radio_khz, utc_time)
- no_decode / timeout / error events are silently dropped
- ``apply_settings`` and ``set_dial_freq`` forward to the wrapped
  Reporter
- Stats: decoded_in, decoded_sent, other_skipped accounted correctly

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_reporter_block -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    Event,
    ReporterBlock,
    ReporterBlockConfig,
    Runtime,
    make_event_stream,
)


# --- Fake Reporter so tests don't open sockets ----------------------


class _FakeReporter:
    """Stub for ``map144_app.reporting.Reporter`` used in tests."""

    def __init__(self):
        self.my_call = ""
        self.my_grid = ""
        self.wsjtx_enabled = False
        self.pskreporter_enabled = False
        self.dx_enabled = False

        # What we record:
        self.decode_calls: list[dict] = []
        self.freq_calls:   list[int]  = []
        self.applied_settings: list[dict] = []
        self.started = False
        self.stopped = False

        # Dummy stat fields the block reads in stats().
        self.stat_udp_sent      = 0
        self.stat_psk_uploaded  = 0
        self.stat_psk_queued    = 0
        self.stat_dx_sent       = 0
        self.stat_dx_status     = "fake"

    def apply_settings(self, **kwargs):
        self.applied_settings.append(dict(kwargs))
        for k in ("my_call", "my_grid", "wsjtx_enabled",
                  "pskreporter_enabled", "dx_enabled"):
            if k in kwargs:
                setattr(self, k, kwargs[k])

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def report_decode(self, decode: dict):
        self.decode_calls.append(dict(decode))

    def report_freq(self, freq_hz: int):
        self.freq_calls.append(int(freq_hz))


# --- helpers ---------------------------------------------------------


def _decode_event(outcome: str = "decoded",
                  message: str = "WA1HCO FN42",
                  jt9_snr: int = 12,
                  t_sec: float = 7.5,
                  radio_khz: float = 50260.5) -> Event:
    return Event(
        kind="decode",
        occurred_at_sample=12345,
        sample_rate_hz=48_000,
        wall_clock_at_production=1700000000.0,
        payload={
            "outcome":   outcome,
            "message":   message,
            "channel_index": 5,
            "fc_hz":     1500.0,
            "jt9_snr":   jt9_snr,
            "t_sec":     t_sec,
            "radio_khz": radio_khz,
            "utc_time":  "12:34:56",
            "decoder":   "spd",
            "marker_id": 99,
        },
    )


class _Harness:
    """Build a Runtime with one ReporterBlock + fake Reporter.

    ReporterBlock.configure() lazy-imports map144_app.reporting.Reporter
    and instantiates it.  We monkey-patch that class with a factory that
    returns our fake so configure() picks up the fake, not a real socket
    Reporter.  Restored by stop_and_unpatch().
    """

    def __init__(self):
        self.fake = _FakeReporter()
        # Patch reporting.Reporter -> a factory that returns OUR fake.
        # This persists for the harness lifetime; tearDown via the
        # caller's tearDown method or via stop_and_unpatch().
        import map144_app.reporting as _rep
        self._real_reporter_cls = _rep.Reporter
        # Note: ReporterBlock.configure() does ``self._reporter = Reporter()``
        # so the factory must be callable with no args and return our fake.
        _rep.Reporter = lambda: self.fake

        self.rt = Runtime()
        self.blk = self.rt.add(
            ReporterBlock("reporter"),
            ReporterBlockConfig(
                decode_port_name="decodes",
                get_timeout_s=0.05,
            ),
        )

        self.dec = make_event_stream("decodes", capacity=64, durable=True)
        self.blk.inputs["decodes"] = self.dec

    def stop_and_unpatch(self, timeout_s: float = 1.0) -> None:
        """Stop the runtime and restore the real Reporter class."""
        try:
            self.rt.stop(timeout_s=timeout_s)
        finally:
            import map144_app.reporting as _rep
            _rep.Reporter = self._real_reporter_cls

    def start_and_wait_ready(self, timeout: float = 1.0) -> None:
        """Start the runtime and wait until the worker thread has
        called on_start() (which calls reporter.start())."""
        self.rt.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.fake.started:
                return
            time.sleep(0.005)
        raise TimeoutError(f"ReporterBlock not ready within {timeout}s")


# --- tests -----------------------------------------------------------


class TestDecodedEventForwarding(unittest.TestCase):

    def test_decoded_event_triggers_report_decode(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            h.dec.put(_decode_event(outcome="decoded",
                                    message="VE3RA FN03",
                                    jt9_snr=8, t_sec=4.2,
                                    radio_khz=50260.5))
            time.sleep(0.15)
        finally:
            h.stop_and_unpatch(timeout_s=1.0)

        self.assertEqual(len(h.fake.decode_calls), 1)
        d = h.fake.decode_calls[0]
        self.assertEqual(d["outcome"], "decoded")
        self.assertEqual(d["message"], "VE3RA FN03")
        self.assertEqual(d["jt9_snr"], 8)
        self.assertAlmostEqual(d["t_sec"], 4.2)
        self.assertAlmostEqual(d["radio_khz"], 50260.5)
        self.assertEqual(d["utc_time"], "12:34:56")
        self.assertEqual(d["marker_id"], 99)

    def test_multiple_decodes_all_forwarded(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            for k in range(5):
                h.dec.put(_decode_event(message=f"TEST{k} CALL FN42",
                                        jt9_snr=10 + k))
            time.sleep(0.20)
        finally:
            h.stop_and_unpatch(timeout_s=1.0)

        self.assertEqual(len(h.fake.decode_calls), 5)
        snrs = [d["jt9_snr"] for d in h.fake.decode_calls]
        self.assertEqual(snrs, [10, 11, 12, 13, 14])


class TestNonDecodedFiltered(unittest.TestCase):

    def test_no_decode_outcome_skipped(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            h.dec.put(_decode_event(outcome="no_decode"))
            h.dec.put(_decode_event(outcome="timeout"))
            h.dec.put(_decode_event(outcome="error"))
            h.dec.put(_decode_event(outcome="decoded", message="REAL DECODE"))
            time.sleep(0.20)
        finally:
            h.stop_and_unpatch(timeout_s=1.0)

        self.assertEqual(len(h.fake.decode_calls), 1)
        self.assertEqual(h.fake.decode_calls[0]["message"], "REAL DECODE")
        # Stats account the skipped ones separately.
        self.assertEqual(h.blk._n_other_skipped, 3)


class TestSettingsForwarding(unittest.TestCase):

    def test_apply_settings_forwards_to_reporter(self):
        h = _Harness()
        # Apply BEFORE start so the reporter has settings when start fires.
        h.blk.apply_settings(my_call="WA1HCO", my_grid="FN42",
                             wsjtx_enabled=True,
                             wsjtx_host="127.0.0.1", wsjtx_port=2237,
                             pskreporter_enabled=True)
        h.start_and_wait_ready()
        try:
            self.assertEqual(len(h.fake.applied_settings), 1)
            settings = h.fake.applied_settings[0]
            self.assertEqual(settings["my_call"], "WA1HCO")
            self.assertEqual(settings["my_grid"], "FN42")
            self.assertTrue(settings["wsjtx_enabled"])
            self.assertTrue(settings["pskreporter_enabled"])
            self.assertEqual(h.fake.my_call, "WA1HCO")
        finally:
            h.stop_and_unpatch(timeout_s=1.0)

    def test_set_dial_freq_forwards(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            h.blk.set_dial_freq(50_260_000)
            h.blk.set_dial_freq(50_265_000)
        finally:
            h.stop_and_unpatch(timeout_s=1.0)
        self.assertEqual(h.fake.freq_calls, [50_260_000, 50_265_000])

    def test_apply_settings_before_configure_buffers(self):
        # Direct ReporterBlock with no configure() — apply_settings and
        # set_dial_freq get BUFFERED, not raised.  This matches the GUI's
        # typical ordering: settings load from QSettings before the
        # runtime starts the block.
        blk = ReporterBlock("reporter")
        blk.apply_settings(my_call="X", my_grid="FN42")
        blk.set_dial_freq(50_000_000)
        # No raise; values are buffered for configure() to replay.
        self.assertIsNotNone(blk._pending_settings)
        self.assertEqual(blk._pending_settings.get("my_call"), "X")
        self.assertEqual(blk._pending_dial_freq, 50_000_000)


class TestStats(unittest.TestCase):

    def test_stats_reflect_event_counts(self):
        h = _Harness()
        h.start_and_wait_ready()
        try:
            h.dec.put(_decode_event(outcome="decoded"))
            h.dec.put(_decode_event(outcome="decoded"))
            h.dec.put(_decode_event(outcome="no_decode"))
            time.sleep(0.20)
            stats = h.blk.stats()
        finally:
            h.stop_and_unpatch(timeout_s=1.0)
        self.assertEqual(stats["decoded_in"], 2)
        self.assertEqual(stats["decoded_sent"], 2)
        self.assertEqual(stats["other_skipped"], 1)

    def test_stats_include_reporter_passthroughs(self):
        h = _Harness()
        h.fake.stat_udp_sent = 7
        h.fake.stat_psk_uploaded = 3
        h.fake.stat_dx_status = "connected"
        h.start_and_wait_ready()
        try:
            stats = h.blk.stats()
        finally:
            h.stop_and_unpatch(timeout_s=1.0)
        self.assertEqual(stats["reporter_udp_sent"], 7)
        self.assertEqual(stats["reporter_psk_uploaded"], 3)
        self.assertEqual(stats["reporter_dx_status"], "connected")


class TestLifecycle(unittest.TestCase):

    def test_reporter_started_and_stopped(self):
        h = _Harness()
        h.start_and_wait_ready()
        self.assertTrue(h.fake.started)
        self.assertFalse(h.fake.stopped)
        h.stop_and_unpatch(timeout_s=1.0)
        self.assertTrue(h.fake.stopped)


if __name__ == "__main__":
    unittest.main()
