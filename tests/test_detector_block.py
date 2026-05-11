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
"""Tests for ``map144_app.blocks.DetectorBlock``.

Two layers:

1. **Cluster-gate unit tests** — call :meth:`DetectorBlock._cluster_gate`
   directly with synthetic ``triggered`` and ``pair_metric`` arrays.
   Verifies each of the four CLAUDE.md detection invariants in isolation,
   plus the per-cluster span gate and the sustained-signal lockout.
2. **End-to-end pipeline test** — SyntheticSource (tone) →
   ChannelizerBlock → DetectorBlock; verify the rolling pct25 baseline
   warms up and that a strong tone eventually produces detection events.

The unit-test layer is the substantive correctness check; the end-to-end
test is a smoke check that the moving parts integrate.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_detector_block -v
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
    DetectorBlock,
    DetectorBlockConfig,
    Event,
    Record,
    Runtime,
    Stream,
    SyntheticSource,
    SyntheticSourceConfig,
    make_event_stream,
    make_sample_stream,
)


def _make_detector(**overrides) -> DetectorBlock:
    """Construct + on_start a DetectorBlock so private state is allocated.

    Tests poke ``_cluster_gate`` directly so the buffers must exist.
    """
    cfg_kwargs = dict(
        n_channels=48,
        coincidence_max_clusters=6,
        coincidence_span_ch=8,
        sustained_hops=100,        # ~2 s of hops
        sustained_lockout=1500,
    )
    cfg_kwargs.update(overrides)
    cfg = DetectorBlockConfig(**cfg_kwargs)
    det = DetectorBlock("det")
    det.configure(cfg)
    # on_start allocates buffers without spawning a thread.
    det.on_start()
    return det


# ---------------------------------------------------------------------
# Cluster-gate unit tests — each CLAUDE.md invariant numbered in test
# names so failures point at the invariant.
# ---------------------------------------------------------------------


class TestClusterGateInvariant1(unittest.TestCase):
    """Invariant 1: cluster definition — gap > 3 channels = new cluster."""

    def test_adjacent_channels_form_one_cluster(self):
        det = _make_detector()
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[[5, 6, 7]] = [5.0, 4.5, 4.0]
        triggered = np.array([5, 6, 7])
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(len(launches), 1)
        launch_ch, cluster = launches[0]
        self.assertEqual(launch_ch, 5)  # strongest in cluster
        self.assertEqual(sorted(cluster.tolist()), [5, 6, 7])

    def test_gap_of_3_keeps_one_cluster(self):
        """Gap of exactly 3 (e.g. ch 5 and ch 8) joins them into one cluster."""
        det = _make_detector()
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[[5, 8]] = [5.0, 4.5]
        triggered = np.array([5, 8])
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(len(launches), 1)
        launch_ch, cluster = launches[0]
        self.assertEqual(launch_ch, 5)
        self.assertEqual(sorted(cluster.tolist()), [5, 8])

    def test_gap_of_4_splits_into_two_clusters(self):
        det = _make_detector()
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[[5, 9]] = [5.0, 4.5]
        triggered = np.array([5, 9])
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(len(launches), 2)
        chs = sorted(lc[0] for lc in launches)
        self.assertEqual(chs, [5, 9])


class TestClusterGateInvariant2(unittest.TestCase):
    """Invariant 2: only-best cooldown on launch.

    The launched channel goes into cooldown; cluster neighbours do NOT.
    """

    def test_only_best_channel_is_cooled(self):
        det = _make_detector()
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[[5, 6, 7]] = [5.0, 4.0, 4.5]
        triggered = np.array([5, 7, 6])  # not necessarily sorted
        det._cluster_gate(triggered, pair_metric)
        # Only channel 5 is in cooldown.  6 and 7 must NOT be cooled.
        self.assertIn(5, det._detect_cooldowns)
        self.assertNotIn(6, det._detect_cooldowns)
        self.assertNotIn(7, det._detect_cooldowns)


class TestClusterGateInvariant3(unittest.TestCase):
    """Invariant 3: fresh-channel too-many gate.

    The broadband-noise gate (``coincidence_max_clusters``) counts only
    *fresh* (non-cooldown) clusters.  Sidelobe remnants of a prior real
    detection must not inflate the gate count.
    """

    def test_cooled_clusters_dont_inflate_gate(self):
        det = _make_detector(coincidence_max_clusters=3)
        # 5 clusters total, but 3 of their best channels are already in
        # cooldown (sidelobes of a recent real detection).  Only 2 are
        # fresh; gate threshold is 3, so the gate should NOT fire.
        pair_metric = np.zeros(48, dtype=np.float32)
        cluster_centers = [4, 10, 16, 22, 28]   # gap of 6 between each
        for ch in cluster_centers:
            pair_metric[ch] = 5.0
        # Mark first 3 as cooled (their cluster-best is in cooldown).
        for ch in cluster_centers[:3]:
            det._detect_cooldowns[ch] = 5
        triggered = np.array(cluster_centers)
        # Reset broadband-gate counter so we can detect it firing.
        det._n_too_many_clusters_gates = 0
        launches = det._cluster_gate(triggered, pair_metric)
        # Gate did NOT fire (too_many_clusters_gates unchanged).
        self.assertEqual(det._n_too_many_clusters_gates, 0)
        # Two fresh clusters launch.
        launch_chs = sorted(lc[0] for lc in launches)
        self.assertEqual(launch_chs, [22, 28])

    def test_all_fresh_clusters_trip_gate(self):
        det = _make_detector(coincidence_max_clusters=3)
        pair_metric = np.zeros(48, dtype=np.float32)
        cluster_centers = [4, 10, 16, 22, 28]
        for ch in cluster_centers:
            pair_metric[ch] = 5.0
        triggered = np.array(cluster_centers)
        det._n_too_many_clusters_gates = 0
        launches = det._cluster_gate(triggered, pair_metric)
        # 5 fresh clusters >= 3 → gate fires; no launches.
        self.assertEqual(launches, [])
        self.assertEqual(det._n_too_many_clusters_gates, 1)


class TestClusterGateInvariant4(unittest.TestCase):
    """Invariant 4: no proximity guard.

    Adjacent-channel signals in *separate* clusters (gap > 3) must both
    launch.  A re-triggering channel must not pre-cool its neighbours.
    """

    def test_adjacent_clusters_both_launch(self):
        det = _make_detector()
        # Two clusters, gap of 5 (just over 3).
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[[10, 16]] = [5.0, 4.5]
        triggered = np.array([10, 16])
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(len(launches), 2)
        chs = sorted(lc[0] for lc in launches)
        self.assertEqual(chs, [10, 16])

    def test_no_neighbour_cooldown_on_launch(self):
        det = _make_detector()
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[10] = 5.0
        triggered = np.array([10])
        det._cluster_gate(triggered, pair_metric)
        # Only ch 10 is cooled; ±1, ±2, ±3 must remain available.
        for delta in (-3, -2, -1, 1, 2, 3):
            self.assertNotIn(
                10 + delta, det._detect_cooldowns,
                f"channel {10 + delta} should not be in cooldown (no proximity guard)",
            )


class TestClusterGateSpanGate(unittest.TestCase):
    """Per-cluster span gate: cluster spanning > coincidence_span_ch
    channels is wideband noise — suppress all its channels.
    """

    def test_wide_cluster_suppressed(self):
        det = _make_detector(coincidence_span_ch=8)
        # 10-channel-wide single cluster (span = 9 > 8).
        pair_metric = np.zeros(48, dtype=np.float32)
        chs = list(range(10, 20))   # span = 9
        for ch in chs:
            pair_metric[ch] = 5.0
        triggered = np.array(chs)
        det._n_clusters_too_wide = 0
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(launches, [])
        self.assertEqual(det._n_clusters_too_wide, 1)
        # All 10 channels are now in cooldown.
        for ch in chs:
            self.assertIn(ch, det._detect_cooldowns)

    def test_narrow_cluster_passes(self):
        det = _make_detector(coincidence_span_ch=8)
        chs = [10, 11, 12]   # span = 2
        pair_metric = np.zeros(48, dtype=np.float32)
        for ch in chs:
            pair_metric[ch] = 5.0
        triggered = np.array(chs)
        det._n_clusters_too_wide = 0
        launches = det._cluster_gate(triggered, pair_metric)
        self.assertEqual(len(launches), 1)
        self.assertEqual(det._n_clusters_too_wide, 0)


class TestClusterGateBestCooledRefresh(unittest.TestCase):
    """When the cluster best is already in cooldown, only refresh
    *already-cooled* members; do NOT add fresh members.
    A fresh member may be the real signal centre that becomes the
    cluster's best on a subsequent cycle.
    """

    def test_fresh_member_stays_fresh_when_best_is_cooled(self):
        det = _make_detector()
        # Cluster: ch 5 (best, cooled), ch 6 (fresh, not as strong).
        pair_metric = np.zeros(48, dtype=np.float32)
        pair_metric[5] = 5.0
        pair_metric[6] = 4.0
        det._detect_cooldowns[5] = 10  # ch 5 is cooled
        triggered = np.array([5, 6])
        launches = det._cluster_gate(triggered, pair_metric)
        # No launch (best is cooled).
        self.assertEqual(launches, [])
        # Ch 5 cooldown refreshed (still in cooldowns).
        self.assertIn(5, det._detect_cooldowns)
        # Ch 6 must NOT have been added to cooldowns.
        self.assertNotIn(6, det._detect_cooldowns)


# ---------------------------------------------------------------------
# Sustained-signal lockout
# ---------------------------------------------------------------------


class TestSustainedLockout(unittest.TestCase):
    """Channel above threshold for ``sustained_hops`` cycles → long
    lockout.  Models a beacon / FT8 / spur, not a meteor ping.
    """

    def test_lockout_after_sustained(self):
        det = _make_detector(sustained_hops=5, sustained_lockout=100)
        # Simulate ch 7 above threshold for 5 consecutive cycles.
        for _ in range(5):
            det._update_sustained_state(np.array([7]))
        self.assertIn(7, det._detect_cooldowns)
        self.assertEqual(det._detect_cooldowns[7], 100)

    def test_below_threshold_resets_counter(self):
        det = _make_detector(sustained_hops=5, sustained_lockout=100)
        for _ in range(3):
            det._update_sustained_state(np.array([7]))
        # Ch 7 below threshold for one cycle → counter resets.
        det._update_sustained_state(np.array([]))
        # Now 4 more cycles above — should NOT yet trip lockout.
        for _ in range(4):
            det._update_sustained_state(np.array([7]))
        self.assertNotIn(7, det._detect_cooldowns)


# ---------------------------------------------------------------------
# Edge / DC channel skip
# ---------------------------------------------------------------------


class TestEdgeChannelSuppress(unittest.TestCase):
    """Edge and DC channels are masked out before the cluster gate runs."""

    def test_dc_channel_skipped(self):
        # We can only test this through _detect_one_cycle since the
        # masking happens there, not in _cluster_gate.  Bypass DSP by
        # writing pair_metric directly through a stubbed compute path.
        # Default ``dc_ch_skip=0`` matches legacy parity — explicitly
        # set 3 here to exercise the DC-mask code path itself.
        # ch 10 is well clear of both DC (0..dc_skip) and the
        # wrap-aware Nyquist mask (≈ ch 24 ± edge_skip with default
        # channel_offset_hz=0).
        det = _make_detector(dc_ch_skip=3)
        nch = det.config.n_channels  # type: ignore[union-attr]
        # Stub the compute methods to return a controlled metric.
        big = np.full(nch, 0.0, dtype=np.float32)
        big[0] = 10.0   # DC channel — should be suppressed
        big[10] = 10.0  # ordinary — should trigger
        det._compute_sq_metric_db = lambda _b: big.copy()  # type: ignore[method-assign]
        det._compute_sync_metric_db = lambda: big.copy()    # type: ignore[method-assign]

        # Need a queued event sink to capture launches.
        events = make_event_stream("evt", capacity=8, durable=True)
        det.outputs[det.config.detection_port_name] = events  # type: ignore[union-attr]

        det._detect_one_cycle(wall_clock_for_events=1.0)

        out_events = []
        try:
            while True:
                out_events.append(events.get(timeout=0.01))
        except Exception:
            pass
        # Should have exactly one event for ch 10, none for ch 0.
        self.assertEqual(len(out_events), 1)
        self.assertEqual(out_events[0].payload["channel_index"], 10)


# ---------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------


class TestDetectionEventPayload(unittest.TestCase):
    """Detection events carry the §6 payload fields."""

    def test_event_fields(self):
        # Use ch 10 — clear of the wrap-aware Nyquist mask (centred
        # near ch 23.5 for offset=500) AND of the DC mask.
        det = _make_detector(channel_offset_hz=500.0, channel_spacing_hz=1000.0)
        nch = det.config.n_channels  # type: ignore[union-attr]
        sync_db = np.zeros(nch, dtype=np.float32)
        sq_db = np.zeros(nch, dtype=np.float32)
        sync_db[10] = 6.0
        sq_db[10] = 4.5
        det._compute_sync_metric_db = lambda: sync_db.copy()  # type: ignore[method-assign]
        det._compute_sq_metric_db = lambda _b: sq_db.copy()    # type: ignore[method-assign]

        events = make_event_stream("evt", capacity=8, durable=True)
        det.outputs[det.config.detection_port_name] = events  # type: ignore[union-attr]

        det._detect_one_cycle(wall_clock_for_events=42.5)

        out: Event = events.get(timeout=0.1)
        self.assertEqual(out.kind, "detection")
        self.assertEqual(out.sample_rate_hz, det.config.sample_rate_hz)  # type: ignore[union-attr]
        self.assertAlmostEqual(out.wall_clock_at_production, 42.5, places=6)
        p = out.payload
        self.assertEqual(p["channel_index"], 10)
        # fc_hz = 10 (signed, since 10 <= 24) * 1000 + 500 = 10500
        self.assertAlmostEqual(p["fc_hz"], 10_500.0)
        self.assertAlmostEqual(p["sync_metric_db"], 6.0, places=4)
        self.assertAlmostEqual(p["sq_metric_db"], 4.5, places=4)
        # max → pair_metric = 6.0
        self.assertAlmostEqual(p["pair_metric_db"], 6.0, places=4)
        # both above threshold (3.0) → "both"
        self.assertEqual(p["source"], "both")
        self.assertEqual(p["cluster_channels"], [10])
        self.assertEqual(p["cluster_span_ch"], 0)


class TestSignedChannelFcHz(unittest.TestCase):
    """Negative-frequency-half channels (ch_idx > N/2) must wrap to a
    signed index when computing fc_hz — same convention as legacy
    processing.py:952.  Without the wrap, fc_hz comes out 48 kHz too
    high and extract_and_decode mixes the signal to the wrong audio
    frequency (no decode possible).
    """

    def test_high_channel_index_wraps_to_negative_fc_hz(self):
        det = _make_detector(channel_offset_hz=1500.0, channel_spacing_hz=1000.0)
        # ch_idx 42 = ch_signed -6: fc_hz = -6 * 1000 + 1500 = -4500
        self.assertAlmostEqual(det._fc_hz_for_channel(42), -4500.0, places=4)
        # ch_idx 47 = ch_signed -1: fc_hz = -1 * 1000 + 1500 = +500
        self.assertAlmostEqual(det._fc_hz_for_channel(47), 500.0, places=4)
        # ch_idx 5 (positive half) unaffected: fc_hz = 5 * 1000 + 1500 = 6500
        self.assertAlmostEqual(det._fc_hz_for_channel(5), 6500.0, places=4)
        # ch_idx 24 (boundary): treated as positive: 24 * 1000 + 1500 = 25500
        self.assertAlmostEqual(det._fc_hz_for_channel(24), 25_500.0, places=4)


class TestNyquistMaskUSBConvention(unittest.TestCase):
    """The Nyquist edge mask must follow the actual IF Nyquist when
    channel_offset_hz != 0 — not the naive 0..edge_skip / N-edge_skip..N
    band edges.  See ``project_launch_pattern_findings`` memory: the
    pre-fix behaviour left the rolloff zone (ch_k ≈ 19 with USB
    convention) unmasked, producing a ~6× false-trigger rate at
    50279 kHz.  Same fix as processing.py commit b318072.
    """

    def test_usb_convention_masks_real_nyquist(self):
        # USB convention: channel_offset_hz = 1500 → IF Nyquist sits at
        # ch_k = 24 - 1.5 = 22.5.  With edge_skip=4, channels in
        # [18.5, 26.5] are suppressed.
        det = _make_detector(
            channel_offset_hz=1500.0, channel_spacing_hz=1000.0,
            edge_ch_skip=4, dc_ch_skip=3,
        )
        nch = det.config.n_channels  # type: ignore[union-attr]

        # Strong signal at ch 19 — used to slip through the broken mask.
        # Should now be SUPPRESSED.
        big = np.zeros(nch, dtype=np.float32)
        big[19] = 10.0
        det._compute_sq_metric_db = lambda _b: big.copy()  # type: ignore[method-assign]
        det._compute_sync_metric_db = lambda: big.copy()    # type: ignore[method-assign]

        events = make_event_stream("evt", capacity=8, durable=True)
        det.outputs[det.config.detection_port_name] = events  # type: ignore[union-attr]
        det._detect_one_cycle(wall_clock_for_events=1.0)

        out_events = []
        try:
            while True:
                out_events.append(events.get(timeout=0.01))
        except Exception:
            pass
        self.assertEqual(
            len(out_events), 0,
            "ch_k=19 sits inside the corrected Nyquist mask "
            "[18.5 ± 4] for USB convention; should be suppressed.",
        )


class TestTooManyResetAndAdvanceInteraction(unittest.TestCase):
    """REGRESSION: when ``_detect_one_cycle``'s cluster-gate too-many
    branch fires (broadband suppression), it sets ``_ch_buf_end = 0``
    inline.  The caller then runs ``_advance_one_stride`` anyway
    because the outer ``while _ch_buf_end >= fft_size`` loop entered
    before the reset.  Without clamping ``_ch_buf_end`` to non-negative
    in ``_advance_one_stride``, it goes to ``-stride_samples``.

    The NEXT outer-loop iteration's slice arithmetic then mixes a
    negative start with a positive end:

        _ch_buf[:, -256 : -256 + 512]   →   _ch_buf[:, 256 : 256]

    which evaluates to an EMPTY slice — numpy resolves -256 to
    arr.shape[1]-256 = 256 but leaves the positive end as 256
    literally.  Writing a (48, 512) record into a (48, 0) slot raises
    ``ValueError: could not broadcast input array from shape (48,512)
    into shape (48,0)``.

    Caught live during the 2026-05-11 shadow-runtime run, ~17 s after
    start, when the cluster gate fired its first too_many on a burst.
    """

    def test_too_many_then_drain_then_advance_does_not_underflow(self):
        det = _make_detector()
        cfg = det.config

        # Stub the per-channel metric to produce 6+ fresh clusters,
        # tripping the too_many gate.  pair_metric ≥ threshold (3.0)
        # on widely-spaced channels guarantees N independent
        # clusters (gap > 3 channels each).
        big = np.zeros(cfg.n_channels, dtype=np.float32)
        # Pick 7 channels with gap > 3 between each pair so they
        # become 7 distinct clusters.  Avoid the Nyquist mask zone
        # (default channel_offset_hz=0 → mask centred near ch 24).
        cluster_channels = [4, 8, 12, 16, 28, 32, 36]
        for ch in cluster_channels:
            big[ch] = 10.0
        det._compute_sq_metric_db   = lambda _b: big.copy()  # type: ignore[method-assign]
        det._compute_sync_metric_db = lambda: big.copy()      # type: ignore[method-assign]

        # Fill the buffer to fft_size so _detect_one_cycle is allowed
        # to run.
        det._ch_buf[:] = 0.5 + 0.5j   # nonzero so cluster gate processes
        det._ch_buf_end = cfg.fft_size

        # Trigger the gate.  After this, too_many should have fired
        # and _ch_buf_end = 0 (the inline reset).
        det._detect_one_cycle(wall_clock_for_events=1.0)
        self.assertEqual(
            det._ch_buf_end, 0,
            "too_many gate should reset _ch_buf_end to 0 inline",
        )
        self.assertGreater(
            det._n_too_many_clusters_gates, 0,
            "test fixture must actually trip the too_many gate; "
            "if zero, the cluster construction needs adjustment",
        )

        # Now _advance_one_stride runs (the caller does this
        # unconditionally inside the inner drain loop).  Pre-fix this
        # would leave _ch_buf_end at -stride_samples = -256.  Post-fix
        # the clamp keeps it at 0.
        det._advance_one_stride()
        self.assertGreaterEqual(
            det._ch_buf_end, 0,
            f"_ch_buf_end must not go negative after a too_many reset "
            f"+ advance — got {det._ch_buf_end}",
        )

        # And the subsequent slice arithmetic must produce a slot
        # matching the record's data shape.  Simulate the tick()
        # writing a 512-sample record into the buffer.
        space = cfg.fft_size - det._ch_buf_end
        take = min(space, 512)
        slot = det._ch_buf[:, det._ch_buf_end:det._ch_buf_end + take]
        self.assertEqual(
            slot.shape, (cfg.n_channels, take),
            f"slice [_ch_buf_end : _ch_buf_end + take] must produce a "
            f"slot with width=take; got shape {slot.shape} "
            f"(buf_end={det._ch_buf_end}, take={take})",
        )


# ---------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """SyntheticSource → ChannelizerBlock → DetectorBlock with a strong
    tone.  Verifies that the moving parts wire together and that the
    block emits events without crashing.

    Detection event timing depends on the rolling-pct25 warm-up — we
    just check that the pipeline runs cleanly and that the block's
    stats reflect the input it received, not that any specific event
    was emitted (the warm-up is too long to wait for in unit tests).
    """

    def test_pipeline_runs_clean(self):
        rt = Runtime()
        src = rt.add(
            SyntheticSource("src"),
            SyntheticSourceConfig(
                sample_rate_hz=48_000,
                samples_per_record=4096,
                signal_mode="tone",
                tone_hz=5500.0,    # fc(5)+500 with offset 0
                tone_amplitude=0.5,
                realtime_pacing=False,
                stop_after_samples=48_000 * 2,  # 2 s of input
            ),
        )
        ch = rt.add(
            ChannelizerBlock("ch"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=48_000,
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        det = rt.add(
            DetectorBlock("det"),
            DetectorBlockConfig(
                input_port_name="channels",
                detection_port_name="detections",
                channel_offset_hz=0.0,
                get_timeout_s=0.05,
            ),
        )
        rt.connect(src, "iq", ch, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=48_000, n_samples_per_record=4096)
        rt.connect(ch, "channels", det, "channels",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=12_000, n_samples_per_record=1024)
        det_events = make_event_stream("detections", capacity=64, durable=True)
        det.outputs["detections"] = det_events

        rt.start()
        try:
            time.sleep(1.5)
        finally:
            rt.stop(timeout_s=2.0)

        # The block ran without raising; stats should reflect the
        # detection cycles it executed.
        s = det.stats()
        # tick_count > 0 means at least one ChannelStream record was
        # consumed.  Channelizer-rate is 12 kHz; in ~1.5 s of run time
        # at faster-than-real-time we expect at least a handful of
        # ticks.
        self.assertGreater(s["ticks_executed"], 0)
        # samples_in["channels"] should be > 0 (Records consumed).
        ch_in = s["samples_in"].get("channels", 0)
        self.assertGreater(ch_in, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
