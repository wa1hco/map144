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
"""Tests for ``map144_app.blocks.CombinerBlock``.

Validates:

- Combination math at boundary mixing angles (θ=0, π/2, π/4)
- Per-channel array θ / Δφ broadcast
- Sample-alignment invariant (§3.2): mismatched ``start_sample``
  raises a warning and skips combine
- Wall-clock inheritance (§3.5): combined record carries H's anchor
- Shape / sample-rate mismatch is rejected
- Metadata: combined records carry ``theta_rad``,
  ``delta_phi_rad``, and ``polarization_combined`` markers
- End-to-end: H + V records flowing through a Runtime produce one
  combined record per pair, gap-free in start_sample

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_combiner_block -v
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
    CombinerBlock,
    CombinerBlockConfig,
    Record,
    Runtime,
    Stream,
    make_sample_stream,
)


N_CHANNELS = 48
SR = 12_000


def _make_record(start_sample: int, fill: complex, n: int = 256,
                 sample_rate_hz: int = SR, wall_clock: float = 1.0) -> Record:
    return Record(
        data=np.full((N_CHANNELS, n), fill, dtype=np.complex64),
        sample_rate_hz=sample_rate_hz,
        start_sample=start_sample,
        wall_clock_at_production=wall_clock,
    )


def _drain(stream: Stream, max_items: int = 200, timeout: float = 0.05):
    items = []
    for _ in range(max_items):
        try:
            items.append(stream.get(timeout=timeout))
        except Exception:
            break
    return items


# ---------------------------------------------------------------------
# Combination math
# ---------------------------------------------------------------------


class TestCombinerMath(unittest.TestCase):
    """combined = H * cos(θ) + V * exp(jΔφ) * sin(θ).

    Each test runs the block once with two records and verifies the
    output sample values match the closed-form result.
    """

    def _run_one(self, theta_rad: float, delta_phi_rad: float,
                 h_fill: complex, v_fill: complex) -> np.ndarray:
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h",
                input_port_v="v",
                output_port="out",
                theta_rad=theta_rad,
                delta_phi_rad=delta_phi_rad,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out

        rt.start()
        try:
            sh.put(_make_record(0, h_fill))
            sv.put(_make_record(0, v_fill))
            time.sleep(0.15)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 1)
        return recs[0].data

    def test_theta_zero_passes_h_through(self):
        """θ=0: V contribution vanishes (sin 0 = 0), output = H."""
        out = self._run_one(theta_rad=0.0, delta_phi_rad=1.234,
                            h_fill=0.5 + 0.0j, v_fill=99 + 99j)
        np.testing.assert_allclose(out, 0.5 + 0.0j, atol=1e-6)

    def test_theta_half_pi_passes_v_with_phase(self):
        """θ=π/2, Δφ=0: H contribution vanishes (cos π/2 = 0), output = V."""
        out = self._run_one(theta_rad=np.pi / 2, delta_phi_rad=0.0,
                            h_fill=99 + 99j, v_fill=0.3 + 0.0j)
        np.testing.assert_allclose(out, 0.3 + 0.0j, atol=1e-5)

    def test_theta_half_pi_with_delta_phi_rotates_v(self):
        """θ=π/2, Δφ=π/2: output = V * exp(jπ/2) = j*V."""
        out = self._run_one(theta_rad=np.pi / 2, delta_phi_rad=np.pi / 2,
                            h_fill=99 + 99j, v_fill=1.0 + 0.0j)
        np.testing.assert_allclose(out, 1.0j, atol=1e-5)

    def test_theta_quarter_pi(self):
        """θ=π/4, Δφ=0: output = (H + V) / sqrt(2)."""
        out = self._run_one(theta_rad=np.pi / 4, delta_phi_rad=0.0,
                            h_fill=1.0 + 0.0j, v_fill=1.0 + 0.0j)
        expected = (1.0 + 1.0) / np.sqrt(2.0)
        np.testing.assert_allclose(out.real, expected, atol=1e-5)
        np.testing.assert_allclose(out.imag, 0.0, atol=1e-5)


class TestCombinerPerChannelArrays(unittest.TestCase):
    """θ and Δφ may be per-channel arrays — broadcast over the channel axis."""

    def test_per_channel_theta(self):
        # Half the channels at θ=0 (passes H), half at θ=π/2, Δφ=0 (passes V).
        theta = np.zeros(N_CHANNELS, dtype=np.float64)
        theta[24:] = np.pi / 2
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h", input_port_v="v", output_port="out",
                theta_rad=theta, delta_phi_rad=0.0,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out

        h_data = np.full((N_CHANNELS, 256), 0.5 + 0.0j, dtype=np.complex64)
        v_data = np.full((N_CHANNELS, 256), 0.0 + 0.7j, dtype=np.complex64)
        rt.start()
        try:
            sh.put(Record(data=h_data, sample_rate_hz=SR, start_sample=0,
                          wall_clock_at_production=1.0))
            sv.put(Record(data=v_data, sample_rate_hz=SR, start_sample=0,
                          wall_clock_at_production=1.0))
            time.sleep(0.15)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 1)
        result = recs[0].data
        # First 24 channels: θ=0 → H = 0.5+0j
        np.testing.assert_allclose(result[:24].real, 0.5, atol=1e-5)
        np.testing.assert_allclose(result[:24].imag, 0.0, atol=1e-5)
        # Last 24 channels: θ=π/2, Δφ=0 → V = 0+0.7j
        np.testing.assert_allclose(result[24:].real, 0.0, atol=1e-5)
        np.testing.assert_allclose(result[24:].imag, 0.7, atol=1e-4)


# ---------------------------------------------------------------------
# Sample-alignment invariant (§3.2)
# ---------------------------------------------------------------------


class TestCombinerAlignment(unittest.TestCase):

    def _wire(self) -> tuple[Runtime, CombinerBlock, Stream, Stream, Stream]:
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h", input_port_v="v", output_port="out",
                theta_rad=np.pi / 4, delta_phi_rad=0.0,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out
        return rt, comb, sh, sv, out

    def test_misaligned_start_sample_logs_warning_and_skips(self):
        rt, comb, sh, sv, out = self._wire()
        with self.assertLogs("map144_app.blocks.combiner_block", level="WARNING") as cap:
            rt.start()
            try:
                sh.put(_make_record(0, 0.5 + 0j))
                sv.put(_make_record(256, 0.5 + 0j))   # wrong start_sample
                time.sleep(0.15)
            finally:
                rt.stop(timeout_s=1.0)
        # No combined record emitted.
        recs = _drain(out)
        self.assertEqual(len(recs), 0)
        # Alignment-error counter incremented.
        self.assertGreater(comb._n_alignment_errors, 0)
        # Warning logged with both start_samples mentioned.
        self.assertTrue(any("misalignment" in r.getMessage() for r in cap.records))

    def test_aligned_pair_passes(self):
        rt, comb, sh, sv, out = self._wire()
        rt.start()
        try:
            for i in range(3):
                sh.put(_make_record(i * 256, 0.5 + 0j))
                sv.put(_make_record(i * 256, 0.5 + 0j))
            time.sleep(0.20)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 3)
        for i, r in enumerate(recs):
            self.assertEqual(r.start_sample, i * 256)
        self.assertEqual(comb._n_alignment_errors, 0)


# ---------------------------------------------------------------------
# Record-level invariants
# ---------------------------------------------------------------------


class TestCombinerRecordInvariants(unittest.TestCase):

    def test_wall_clock_inherited_from_h(self):
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h", input_port_v="v", output_port="out",
                theta_rad=np.pi / 4, delta_phi_rad=0.0,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out

        wc_h = 1_700_000_001.5
        wc_v = 1_700_000_002.5  # different — combiner should pick H's
        rt.start()
        try:
            sh.put(Record(data=np.full((N_CHANNELS, 256), 0.5 + 0j, dtype=np.complex64),
                          sample_rate_hz=SR, start_sample=0,
                          wall_clock_at_production=wc_h))
            sv.put(Record(data=np.full((N_CHANNELS, 256), 0.5 + 0j, dtype=np.complex64),
                          sample_rate_hz=SR, start_sample=0,
                          wall_clock_at_production=wc_v))
            time.sleep(0.15)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].wall_clock_at_production, wc_h, places=6)

    def test_metadata_carries_combination_params(self):
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h", input_port_v="v", output_port="out",
                theta_rad=0.7, delta_phi_rad=-0.3,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out

        rt.start()
        try:
            sh.put(_make_record(0, 0.5 + 0j))
            sv.put(_make_record(0, 0.5 + 0j))
            time.sleep(0.15)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 1)
        meta = recs[0].metadata
        self.assertAlmostEqual(meta["theta_rad"], 0.7, places=4)
        self.assertAlmostEqual(meta["delta_phi_rad"], -0.3, places=4)
        self.assertTrue(meta["polarization_combined"])

    def test_shape_mismatch_skipped(self):
        rt = Runtime()
        comb = rt.add(
            CombinerBlock("c"),
            CombinerBlockConfig(
                input_port_h="h", input_port_v="v", output_port="out",
                theta_rad=np.pi / 4, delta_phi_rad=0.0,
                get_timeout_s=0.05,
            ),
        )
        sh = make_sample_stream("h", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        sv = make_sample_stream("v", queue_depth_seconds=2.0,
                                sample_rate_hz=SR, n_samples_per_record=256)
        out = make_sample_stream("out", queue_depth_seconds=2.0,
                                 sample_rate_hz=SR, n_samples_per_record=256)
        comb.inputs["h"] = sh
        comb.inputs["v"] = sv
        comb.outputs["out"] = out

        rt.start()
        try:
            sh.put(_make_record(0, 0.5 + 0j, n=256))
            sv.put(_make_record(0, 0.5 + 0j, n=128))   # different size
            time.sleep(0.15)
        finally:
            rt.stop(timeout_s=1.0)
        recs = _drain(out)
        self.assertEqual(len(recs), 0)
        self.assertGreater(comb._n_alignment_errors, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
