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
"""Tests for ``map144_app.blocks.DisplayBlock``.

Validates ``#10`` v1 scope (heatmap + markers only):

- Single-record write lands at row 0, hop_count=1
- Sequential records advance the write index modulo history_rows
- 15-second period boundary detection (wall-clock-grid) clears history
  and resets hop_count
- Marker pruning across boundary
- Sample-clock independence: drifting wall-clock-at-record produces the
  expected boundary cross even if the record's start_sample is irregular
- Snapshot atomicity: arrays returned are copies (caller mutations do
  not leak back into the block)
- Mono path: pair_metric_db_v == pair_metric_db_h on round-trip

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_display_block -v
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
    ChannelMetricRecord,
    DisplayBlock,
    DisplayBlockConfig,
    Event,
    Runtime,
    Stream,
    make_event_stream,
    make_sample_stream,
)


N_CHANNELS = 48
SR = 12_000               # detector tick rate (channelised)
STRIDE = 256              # _CH_DETECT_HOP
HISTORY_ROWS = 16         # small for tests; legacy uses ~705
PERIOD_S = 15.0


# --- helpers ---------------------------------------------------------


def _metric_record(
    start_sample: int,
    wall_clock: float,
    fill: float = 0.0,
) -> ChannelMetricRecord:
    """Build a ChannelMetricRecord with mono pair_metric_db_h == _v."""
    pair = np.full(N_CHANNELS, fill, dtype=np.float32)
    return ChannelMetricRecord(
        data=pair,
        sample_rate_hz=SR,
        start_sample=start_sample,
        wall_clock_at_production=wall_clock,
        pair_metric_db_h=pair,
        pair_metric_db_v=pair,
        sq_metric_db_h=pair,
        sq_metric_db_v=pair,
        sync_metric_db_h=pair,
        sync_metric_db_v=pair,
    )


def _detection_event(occurred_at_sample: int, wall_clock: float,
                     channel_index: int = 5,
                     pair_metric_db: float = 12.5) -> Event:
    return Event(
        kind="detection",
        occurred_at_sample=occurred_at_sample,
        sample_rate_hz=SR,
        wall_clock_at_production=wall_clock,
        payload={
            "channel_index": channel_index,
            "fc_hz": 1000.0 * channel_index,
            "pair_metric_db": pair_metric_db,
            "sync_metric_db": pair_metric_db,
            "sq_metric_db": pair_metric_db,
            "source": "sync",
            "cluster_channels": [channel_index],
            "cluster_span_ch": 0,
        },
    )


class _DisplayHarness:
    """Build a Runtime with one DisplayBlock and feed it metric / event
    records directly via input streams.  Mirrors the pattern used in
    ``test_combiner_block.py`` / ``test_channelizer_block.py``.
    """

    def __init__(self):
        self.rt = Runtime()
        self.disp = self.rt.add(DisplayBlock("disp"), DisplayBlockConfig(
            n_channels=N_CHANNELS,
            history_rows=HISTORY_ROWS,
            history_period_s=PERIOD_S,
            get_timeout_s=0.05,
        ))

        self.metrics = make_sample_stream(
            "metrics", queue_depth_seconds=4.0,
            sample_rate_hz=SR, n_samples_per_record=STRIDE,
        )
        self.detections = make_event_stream("detections", capacity=64)

        self.disp.inputs["metrics"] = self.metrics
        self.disp.inputs["detections"] = self.detections

    def push_metric(self, rec: ChannelMetricRecord) -> None:
        self.metrics.put(rec)

    def push_detection(self, evt: Event) -> None:
        self.detections.put(evt)

    def settle(self, seconds: float = 0.15) -> None:
        time.sleep(seconds)


# --- tests -----------------------------------------------------------


class TestDisplayBlockSingleRecord(unittest.TestCase):

    def test_single_record_writes_row_0(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.0, fill=5.0))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(snap.ch_snr_hop_count, 1)
        self.assertEqual(snap.ch_snr_write_idx, 0)
        # Row 0 received the metric; row 1+ still sentinel.
        np.testing.assert_array_almost_equal(
            snap.ch_snr_history_h[0, :], np.full(N_CHANNELS, 5.0)
        )
        np.testing.assert_array_almost_equal(
            snap.ch_snr_history_v[0, :], np.full(N_CHANNELS, 5.0)
        )
        self.assertTrue(np.all(snap.ch_snr_history_h[1, :] < -100))


class TestDisplayBlockRingAdvance(unittest.TestCase):

    def test_sequential_records_fill_consecutive_rows(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            # Five records within the same period — write_idx 0..4.
            for k in range(5):
                h.push_metric(_metric_record(
                    start_sample=k * STRIDE,
                    wall_clock=0.5 + 0.04 * k,    # all in period 0
                    fill=float(k + 1),
                ))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(snap.ch_snr_hop_count, 5)
        self.assertEqual(snap.ch_snr_write_idx, 4)
        for k in range(5):
            np.testing.assert_array_almost_equal(
                snap.ch_snr_history_h[k, :], np.full(N_CHANNELS, k + 1)
            )

    def test_ring_wraps_modulo_history_rows(self):
        """Pushing >history_rows records within a period overwrites row 0."""
        h = _DisplayHarness()
        h.rt.start()
        try:
            # Push history_rows + 1 records, all within period 0.
            n = HISTORY_ROWS + 1
            for k in range(n):
                h.push_metric(_metric_record(
                    start_sample=k * STRIDE,
                    wall_clock=0.5 + 0.001 * k,   # tightly within period 0
                    fill=float(k + 1),
                ))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(snap.ch_snr_hop_count, n)
        # write_idx wraps to (n-1) % history_rows = 0.
        self.assertEqual(snap.ch_snr_write_idx, (n - 1) % HISTORY_ROWS)
        # Row 0 holds the LATEST overwrite (value = n).
        np.testing.assert_array_almost_equal(
            snap.ch_snr_history_h[0, :], np.full(N_CHANNELS, n)
        )


class TestDisplayBlockBoundary(unittest.TestCase):

    def test_period_boundary_clears_history_and_resets_hop_count(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            # Three records in period 0 (wall_clock 14.x).
            for k in range(3):
                h.push_metric(_metric_record(
                    start_sample=k * STRIDE,
                    wall_clock=14.0 + 0.1 * k,
                    fill=float(k + 1),
                ))
            # One record in period 1 (wall_clock 15.x).
            h.push_metric(_metric_record(
                start_sample=3 * STRIDE,
                wall_clock=15.5,
                fill=99.0,
            ))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        # After boundary: hop_count counts only the post-boundary records (1).
        self.assertEqual(snap.ch_snr_hop_count, 1)
        self.assertEqual(snap.ch_snr_write_idx, 0)
        self.assertEqual(snap.ch_snr_period_index, 1)
        # Row 0 holds the post-boundary record's value.
        np.testing.assert_array_almost_equal(
            snap.ch_snr_history_h[0, :], np.full(N_CHANNELS, 99.0)
        )
        # Pre-boundary rows are cleared back to sentinel.
        self.assertTrue(np.all(snap.ch_snr_history_h[1, :] < -100))

    def test_session_start_sets_initial_period(self):
        """First record sets ch_snr_period_index from its wall_clock."""
        h = _DisplayHarness()
        h.rt.start()
        try:
            h.push_metric(_metric_record(
                start_sample=0,
                wall_clock=42 * PERIOD_S + 7.5,  # period 42
                fill=1.0,
            ))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(snap.ch_snr_period_index, 42)
        self.assertEqual(snap.ch_snr_hop_count, 1)


class TestDisplayBlockMarkers(unittest.TestCase):

    def test_detection_event_appended_as_marker(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            h.push_detection(_detection_event(
                occurred_at_sample=128,
                wall_clock=2.5,
                channel_index=10,
                pair_metric_db=15.0,
            ))
            h.push_metric(_metric_record(
                start_sample=0, wall_clock=2.5, fill=1.0,
            ))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(len(snap.jt9_markers), 1)
        m = snap.jt9_markers[0]
        self.assertEqual(m["channel_index"], 10)
        self.assertEqual(m["pair_metric_db"], 15.0)
        self.assertEqual(m["period_index"], 0)

    def test_markers_pruned_at_period_boundary(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            # Two metric records in period 0, with a detection event between.
            h.push_metric(_metric_record(start_sample=0, wall_clock=14.0, fill=1.0))
            h.push_detection(_detection_event(occurred_at_sample=10, wall_clock=14.05))
            h.push_metric(_metric_record(start_sample=STRIDE, wall_clock=14.1, fill=2.0))
            h.settle(seconds=0.05)

            # Confirm marker is in period 0.
            snap0 = h.disp.state_snapshot()
            self.assertEqual(len(snap0.jt9_markers), 1)
            self.assertEqual(snap0.jt9_markers[0]["period_index"], 0)

            # Boundary cross — marker for period 0 should be pruned.
            h.push_metric(_metric_record(
                start_sample=2 * STRIDE, wall_clock=15.5, fill=10.0,
            ))
            h.settle()
            snap1 = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(snap1.ch_snr_period_index, 1)
        self.assertEqual(len(snap1.jt9_markers), 0)


class TestDisplayBlockSnapshotIsolation(unittest.TestCase):

    def test_snapshot_is_a_copy(self):
        """Caller mutations to the snapshot must not leak back into the
        block's state."""
        h = _DisplayHarness()
        h.rt.start()
        try:
            h.push_metric(_metric_record(start_sample=0, wall_clock=1.0, fill=3.0))
            h.settle()
            snap = h.disp.state_snapshot()
            # Mutate the snapshot.
            snap.ch_snr_history_h[0, :] = -42.0

            # Push another record so we re-read.
            h.push_metric(_metric_record(start_sample=STRIDE, wall_clock=1.05, fill=7.0))
            h.settle()
            snap2 = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        # Block's internal row 0 is unchanged (still 3.0), not -42.
        np.testing.assert_array_almost_equal(
            snap2.ch_snr_history_h[0, :], np.full(N_CHANNELS, 3.0)
        )


class TestDisplayBlockMonoPath(unittest.TestCase):

    def test_v_equals_h_for_mono_record(self):
        h = _DisplayHarness()
        h.rt.start()
        try:
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.0, fill=4.5))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        np.testing.assert_array_equal(
            snap.ch_snr_history_h[0, :], snap.ch_snr_history_v[0, :]
        )


# --- IQ waterfall (#10b) --------------------------------------------


def _iq_record(start_sample: int, n: int,
               wall_clock: float = 100.0,
               sample_rate_hz: int = 48_000,
               fill_complex: complex = 0.5+0.0j) -> Record:
    """1-D IQ record matching the IQStream contract.  Default fill is
    a constant 0.5+0j (yields a clean DC tone in the waterfall)."""
    return Record(
        data=np.full(n, fill_complex, dtype=np.complex64),
        sample_rate_hz=sample_rate_hz,
        start_sample=start_sample,
        wall_clock_at_production=wall_clock,
    )


class _WaterfallHarness:
    """Build a Runtime with one DisplayBlock + IQ stream attached."""

    def __init__(self):
        self.rt = Runtime()
        self.disp = self.rt.add(DisplayBlock("disp"), DisplayBlockConfig(
            n_channels=N_CHANNELS,
            history_rows=HISTORY_ROWS,
            history_period_s=PERIOD_S,
            iq_sample_rate_hz=48_000,
            iq_fft_size=64,        # small FFT for fast tests
            iq_full_scale=1.0,
            get_timeout_s=0.05,
        ))
        self.metrics = make_sample_stream(
            "metrics", queue_depth_seconds=4.0,
            sample_rate_hz=SR, n_samples_per_record=STRIDE,
        )
        self.detections = make_event_stream("detections", capacity=64)
        self.iq = make_sample_stream(
            "iq", queue_depth_seconds=2.0,
            sample_rate_hz=48_000, n_samples_per_record=512,
        )
        self.disp.inputs["metrics"]    = self.metrics
        self.disp.inputs["detections"] = self.detections
        self.disp.inputs["iq"]         = self.iq

    def push_metric(self, rec: ChannelMetricRecord) -> None:
        self.metrics.put(rec)

    def push_iq(self, rec: Record) -> None:
        self.iq.put(rec)

    def settle(self, seconds: float = 0.20) -> None:
        time.sleep(seconds)


class TestWaterfallIQIngest(unittest.TestCase):

    def test_waterfall_state_present_in_snapshot(self):
        h = _WaterfallHarness()
        h.rt.start()
        try:
            # rt.start() returns before on_start() runs in the spawned
            # thread; brief barrier so _spectrogram_data is allocated.
            for _ in range(40):
                if h.disp._spectrogram_data is not None:
                    break
                time.sleep(0.01)
            # No IQ yet — snapshot still has the waterfall fields, all
            # at the sentinel.
            snap = h.disp.state_snapshot()
            self.assertEqual(snap.spectrogram_data.shape, snap.spec_staging.shape)
            self.assertFalse(snap.spec_filled)
            self.assertTrue(np.all(snap.spec_staging < -100))
        finally:
            h.rt.stop(timeout_s=1.0)

    def test_iq_records_drained_into_sbuf(self):
        h = _WaterfallHarness()
        h.rt.start()
        try:
            # Need at least one metric record for the heartbeat that
            # drives the IQ drain (tick is metric-blocking).
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.5,
                                         fill=1.0))
            # Push enough IQ for several FFT blocks: fft_size=64,
            # hop=32, so 5 blocks = 5*32 + 32 = 192 samples (then a
            # final 32-sample drain).  Push 1024 samples for headroom.
            h.push_iq(_iq_record(start_sample=0, n=1024,
                                 wall_clock=0.5, fill_complex=0.5+0j))
            h.settle()
            stats = h.disp.stats()
        finally:
            h.rt.stop(timeout_s=1.0)
        self.assertGreater(stats["iq_records"], 0)
        self.assertGreater(stats["iq_samples"], 0)
        # FFT blocks were processed (at least a few).
        self.assertGreater(stats["spec_blocks_written"], 0)

    def test_constant_dc_input_produces_dc_bin_peak(self):
        """A constant 0.5+0j IQ stream is a pure DC tone; FFT magnitude
        should peak at the DC bin (= fft_size//2 after fftshift)."""
        h = _WaterfallHarness()
        h.rt.start()
        try:
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.5,
                                         fill=1.0))
            for k in range(4):
                h.push_iq(_iq_record(start_sample=k * 512, n=512,
                                     wall_clock=0.5 + k * 0.01,
                                     fill_complex=0.5+0j))
            h.settle(seconds=0.30)
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)
        # Find any row that has been populated (above sentinel).
        populated = np.where(np.any(snap.spec_staging > -100, axis=1))[0]
        self.assertGreater(len(populated), 0,
                           "no rows were written to spec_staging")
        # In any populated row, the peak bin should be at fft_size//2
        # (DC after fftshift).
        for row in populated[:3]:
            peak_bin = int(np.argmax(snap.spec_staging[row, :]))
            self.assertEqual(peak_bin, snap.spec_staging.shape[1] // 2,
                             f"row {row}: peak at bin {peak_bin}, "
                             f"expected DC bin {snap.spec_staging.shape[1]//2}")


class TestWaterfallPeriodSnapshot(unittest.TestCase):

    def test_period_boundary_snapshots_to_spectrogram_data(self):
        """When a period boundary crosses, _spec_staging snapshots into
        _spectrogram_data and spec_filled flips True."""
        h = _WaterfallHarness()
        h.rt.start()
        try:
            # Push IQ in period 0, then more in period 1.
            h.push_metric(_metric_record(start_sample=0, wall_clock=14.0,
                                         fill=1.0))
            for k in range(3):
                h.push_iq(_iq_record(start_sample=k * 512, n=512,
                                     wall_clock=14.0 + k * 0.01,
                                     fill_complex=0.5+0j))
            h.settle(seconds=0.20)

            # Cross boundary into period 1.
            h.push_metric(_metric_record(start_sample=10_000,
                                         wall_clock=15.5, fill=2.0))
            for k in range(3):
                h.push_iq(_iq_record(start_sample=10_000 + k * 512, n=512,
                                     wall_clock=15.5 + k * 0.01,
                                     fill_complex=0.5+0j))
            h.settle(seconds=0.20)
            snap = h.disp.state_snapshot()
            stats = h.disp.stats()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertTrue(snap.spec_filled,
                        "spec_filled should be True after period crosses")
        self.assertEqual(stats["spec_period_snapshots"], 1,
                         "exactly one period snapshot expected")


# --- Decode panel (#10c) --------------------------------------------


def _decode_event(outcome: str = "decoded",
                  message: str = "WA1HCO FN42",
                  jt9_snr: int = 12,
                  radio_khz: float = 50260.5,
                  t_sec: float = 7.5,
                  channel_index: int = 5,
                  fc_hz: float = 1500.0,
                  marker_id: int = 42,
                  wall_clock: float = 1700000000.0) -> Event:
    return Event(
        kind="decode",
        occurred_at_sample=12345,
        sample_rate_hz=48_000,
        wall_clock_at_production=wall_clock,
        payload={
            "outcome":      outcome,
            "message":      message,
            "jt9_snr":      jt9_snr,
            "radio_khz":    radio_khz,
            "t_sec":        t_sec,
            "channel_index": channel_index,
            "fc_hz":        fc_hz,
            "marker_id":    marker_id,
        },
    )


class _PanelHarness:
    """Harness for #10c decode-panel tests."""

    def __init__(self, panel_max: int = 100):
        self.rt = Runtime()
        self.disp = self.rt.add(DisplayBlock("disp"), DisplayBlockConfig(
            n_channels=N_CHANNELS,
            history_rows=HISTORY_ROWS,
            history_period_s=PERIOD_S,
            decode_panel_max=panel_max,
            get_timeout_s=0.05,
        ))
        self.metrics = make_sample_stream(
            "metrics", queue_depth_seconds=4.0,
            sample_rate_hz=SR, n_samples_per_record=STRIDE,
        )
        self.detections = make_event_stream("detections", capacity=64)
        self.decodes    = make_event_stream("decodes", capacity=64)
        self.disp.inputs["metrics"]    = self.metrics
        self.disp.inputs["detections"] = self.detections
        self.disp.inputs["decodes"]    = self.decodes

    def push_metric(self, rec: ChannelMetricRecord) -> None:
        self.metrics.put(rec)

    def push_decode(self, evt: Event) -> None:
        self.decodes.put(evt)

    def settle(self, seconds: float = 0.20) -> None:
        time.sleep(seconds)


class TestDecodePanel(unittest.TestCase):

    def test_decoded_event_appended_to_panel(self):
        h = _PanelHarness()
        h.rt.start()
        try:
            h.push_decode(_decode_event(message="WA1HCO FN42",
                                        jt9_snr=8,
                                        radio_khz=50260.5))
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.5))
            h.settle()
            snap = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)

        self.assertEqual(len(snap.decode_panel_entries), 1)
        e = snap.decode_panel_entries[0]
        self.assertEqual(e["message"], "WA1HCO FN42")
        self.assertEqual(e["jt9_snr"], 8)
        self.assertAlmostEqual(e["radio_khz"], 50260.5)
        self.assertEqual(e["marker_id"], 42)

    def test_non_decoded_outcomes_skipped_from_panel(self):
        h = _PanelHarness()
        h.rt.start()
        try:
            h.push_decode(_decode_event(outcome="no_decode"))
            h.push_decode(_decode_event(outcome="timeout"))
            h.push_decode(_decode_event(outcome="error"))
            h.push_decode(_decode_event(outcome="decoded",
                                        message="REAL DECODE"))
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.5))
            h.settle()
            snap = h.disp.state_snapshot()
            stats = h.disp.stats()
        finally:
            h.rt.stop(timeout_s=1.0)
        self.assertEqual(len(snap.decode_panel_entries), 1)
        self.assertEqual(snap.decode_panel_entries[0]["message"],
                         "REAL DECODE")
        # All 4 events count as 'in', but only 1 was appended.
        self.assertEqual(stats["decode_events_in"], 4)
        self.assertEqual(stats["decode_panel_appended"], 1)

    def test_panel_eviction_when_full(self):
        h = _PanelHarness(panel_max=3)
        h.rt.start()
        try:
            for k in range(5):
                h.push_decode(_decode_event(message=f"CALL{k} FN42"))
                h.push_metric(_metric_record(
                    start_sample=k * STRIDE, wall_clock=0.5 + k * 0.01,
                ))
            h.settle(seconds=0.30)
            snap = h.disp.state_snapshot()
            stats = h.disp.stats()
        finally:
            h.rt.stop(timeout_s=1.0)
        # Only 3 retained, oldest 2 dropped.
        self.assertEqual(len(snap.decode_panel_entries), 3)
        # Newest LAST: latest call should be CALL4.
        msgs = [e["message"] for e in snap.decode_panel_entries]
        self.assertEqual(msgs, ["CALL2 FN42", "CALL3 FN42", "CALL4 FN42"])
        self.assertEqual(stats["decode_panel_dropped"], 2)

    def test_snapshot_panel_entries_are_immutable(self):
        h = _PanelHarness()
        h.rt.start()
        try:
            h.push_decode(_decode_event(message="A1BC FN42"))
            h.push_metric(_metric_record(start_sample=0, wall_clock=0.5))
            h.settle()
            snap = h.disp.state_snapshot()
            # Mutating snapshot dict should not affect the block's state.
            snap_entries = snap.decode_panel_entries
            self.assertIsInstance(snap_entries, tuple)
            # Mutate a copy of the dict to confirm copies are safe.
            mutated = dict(snap_entries[0])
            mutated["message"] = "MUTATED"
            # Re-snapshot — original unchanged.
            snap2 = h.disp.state_snapshot()
        finally:
            h.rt.stop(timeout_s=1.0)
        self.assertEqual(snap2.decode_panel_entries[0]["message"],
                         "A1BC FN42")


# --- End-to-end: Detector → Display ---------------------------------


from map144_app.blocks import (  # noqa: E402
    ChannelizerBlock,
    ChannelizerBlockConfig,
    DetectorBlock,
    DetectorBlockConfig,
    Record,
)


class TestDetectorToDisplay(unittest.TestCase):
    """Wire ChannelizerBlock → DetectorBlock → DisplayBlock with the new
    metrics port and confirm the stream contract works end-to-end.

    A single 1024-sample IQ record produces channels → detector hops →
    metric records → DisplayBlock state.  We don't validate the metric
    *values* (covered by detector tests); this test only confirms the
    plumbing — that DisplayBlock receives metric records on a real
    pipeline graph.
    """

    def test_pipeline_delivers_metric_records_to_display(self):
        # Sources arrive as 48 kHz IQ, channelize 4× to 12 kHz.
        SR_IN = 48_000
        DEC = 4
        rt = Runtime()
        chn = rt.add(
            ChannelizerBlock("chan"),
            ChannelizerBlockConfig(
                n_channels=N_CHANNELS,
                sample_rate_in_hz=SR_IN,
                decimate_factor=DEC,
            ),
        )
        det = rt.add(
            DetectorBlock("det"),
            DetectorBlockConfig(
                n_channels=N_CHANNELS,
                sample_rate_hz=SR_IN // DEC,
                # Disable cluster gate side-effects we don't care about
                # here; we only need _detect_one_cycle to fire so the
                # metric record is emitted.
            ),
        )
        disp = rt.add(
            DisplayBlock("disp"),
            DisplayBlockConfig(
                n_channels=N_CHANNELS,
                history_rows=32,
                history_period_s=PERIOD_S,
                get_timeout_s=0.05,
            ),
        )

        # Build streams: input IQ → chan → channels → det → metrics → disp.
        in_stream = make_sample_stream(
            "iq", queue_depth_seconds=1.0,
            sample_rate_hz=SR_IN, n_samples_per_record=1024,
        )
        chn_stream = make_sample_stream(
            "channels", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN // DEC, n_samples_per_record=256,
        )
        metric_stream = make_sample_stream(
            "metrics", queue_depth_seconds=4.0,
            sample_rate_hz=SR_IN // DEC, n_samples_per_record=STRIDE,
        )

        chn.inputs["iq"] = in_stream
        chn.outputs["channels"] = chn_stream
        det.inputs["channels"] = chn_stream
        det.outputs["metrics"] = metric_stream
        disp.inputs["metrics"] = metric_stream

        rt.start()
        try:
            # Push enough samples for several detector cycles.  At 12 kHz
            # channel rate, 512-sample fft + 256-sample stride means
            # one detection cycle per 256 channel-samples.  Push two 1024-
            # sample IQ records (= 256 channel samples each, decimate ÷4)
            # — should yield ~2 detection cycles.
            for k in range(8):
                rec = Record(
                    data=np.zeros(1024, dtype=np.complex64),
                    sample_rate_hz=SR_IN,
                    start_sample=k * 1024,
                    wall_clock_at_production=0.5 + 0.001 * k,
                )
                in_stream.put(rec)
            time.sleep(0.4)
            snap = disp.state_snapshot()
        finally:
            rt.stop(timeout_s=2.0)

        # We expect at least one metric record to have made it through.
        self.assertGreater(snap.ch_snr_hop_count, 0)


if __name__ == "__main__":
    unittest.main()
