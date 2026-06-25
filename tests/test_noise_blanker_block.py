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
"""Tests for ``map144_app.blocks.NoiseBlankerBlock``.

Validates the packaging-only block wrapper around the noise_blanker backends:

- Bypass backend is an exact passthrough (cleaned == raw, blanked_fraction 0)
- output preserves shape, sample_rate, start_sample and inherits wall_clock (§3.5)
- no decimation: N samples in → N samples out
- Linrad blanks an impulse burst more than clean noise
- a sample-index gap reseeds the backend (warm-up state cleared)

These drive ``tick()`` directly (no Runtime thread) for determinism.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_noise_blanker_block -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    NoiseBlankerBlock,
    NoiseBlankerBlockConfig,
    Record,
    make_sample_stream,
)

SR = 48_000
N = 4096
WALL = 1_700_000_000.5


def _make_block(blanker_name: str = "Linrad", nb_factor: float = 3.0) -> NoiseBlankerBlock:
    blk = NoiseBlankerBlock("nb")
    blk.configure(NoiseBlankerBlockConfig(
        input_port_name="iq", output_port_name="iq",
        blanker_name=blanker_name, sample_rate_hz=SR,
        nb_factor=nb_factor, get_timeout_s=0.05,
    ))
    blk.inputs["iq"] = make_sample_stream(
        "iq", queue_depth_seconds=4.0, sample_rate_hz=SR, n_samples_per_record=N)
    blk.outputs["iq"] = make_sample_stream(
        "iq", queue_depth_seconds=4.0, sample_rate_hz=SR, n_samples_per_record=N)
    blk.on_start()
    return blk


def _push(blk: NoiseBlankerBlock, data: np.ndarray, start_sample: int,
          wall: float = WALL) -> None:
    blk.inputs["iq"].put(Record(
        data=np.ascontiguousarray(data, dtype=np.complex64),
        sample_rate_hz=SR, start_sample=start_sample,
        wall_clock_at_production=wall,
    ))


def _tick_get(blk: NoiseBlankerBlock) -> Record:
    blk.tick()
    return blk.outputs["iq"].get(timeout=0.05)


def _noise(n: int = N, std: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            .astype(np.complex64) * std)


class TestNoiseBlankerBlock(unittest.TestCase):

    def test_bypass_is_exact_passthrough(self):
        blk = _make_block(blanker_name="Bypass")
        raw = _noise(seed=1)
        _push(blk, raw, start_sample=0)
        out = _tick_get(blk)
        np.testing.assert_array_equal(out.data, raw)
        self.assertEqual(out.metadata["blanked_fraction"], 0.0)
        self.assertEqual(out.metadata["blanker"], "Bypass")

    def test_anchors_and_shape_preserved(self):
        # No decimation: N in -> N out; rate, start_sample, wall inherited.
        blk = _make_block()
        raw = _noise(seed=2)
        _push(blk, raw, start_sample=12_345, wall=1_700_000_123.25)
        out = _tick_get(blk)
        self.assertEqual(out.n_samples, N)
        self.assertEqual(out.sample_rate_hz, SR)
        self.assertEqual(out.start_sample, 12_345)
        self.assertEqual(out.wall_clock_at_production, 1_700_000_123.25)
        self.assertEqual(out.data.dtype, np.complex64)

    def test_start_sample_gap_free_across_records(self):
        blk = _make_block()
        for i in range(3):
            _push(blk, _noise(seed=10 + i), start_sample=i * N)
            out = _tick_get(blk)
            self.assertEqual(out.start_sample, i * N)
            self.assertEqual(out.end_sample, (i + 1) * N)

    def test_linrad_blanks_impulse_more_than_clean(self):
        blk = _make_block(nb_factor=3.0)
        # Warm up on clean noise so the backend's reference is established.
        _push(blk, _noise(seed=20), start_sample=0)
        _tick_get(blk)
        # Clean record: measure baseline blanked fraction.
        _push(blk, _noise(seed=21), start_sample=N)
        clean_frac = _tick_get(blk).metadata["blanked_fraction"]
        # Impulse record: a strong burst should blank noticeably more.
        impulse = _noise(seed=22)
        impulse[2000:2030] += 30.0  # ~3000x the 0.01 noise std
        _push(blk, impulse, start_sample=2 * N)
        impulse_frac = _tick_get(blk).metadata["blanked_fraction"]
        self.assertGreater(impulse_frac, 0.0)
        self.assertGreater(impulse_frac, clean_frac)

    def test_blanked_fraction_in_unit_range(self):
        blk = _make_block()
        _push(blk, _noise(seed=30), start_sample=0)
        frac = _tick_get(blk).metadata["blanked_fraction"]
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_gap_triggers_reseed(self):
        # A non-contiguous start_sample must reseed without raising; output
        # continues with the new anchor.
        blk = _make_block()
        _push(blk, _noise(seed=40), start_sample=0)
        _tick_get(blk)
        _push(blk, _noise(seed=41), start_sample=10 * N)  # gap
        out = _tick_get(blk)
        self.assertEqual(out.start_sample, 10 * N)
        self.assertEqual(blk._n_in_samples, 11 * N)


if __name__ == "__main__":
    unittest.main()
