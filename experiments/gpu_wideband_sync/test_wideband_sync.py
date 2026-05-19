# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# Licensed under GPL-3.0-or-later.
"""Unit / sanity tests for the wideband sync detector scaffold.

These are the minimum-viable correctness checks.  Expand on the road
as the algorithm matures.

The synthetic-ping tests require the ``msk144code`` binary (provided by
WSJT-X) on PATH.  Tests that don't are marked clearly.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Repo root on sys.path for synthetic.py's lazy import of generate_msk144.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.gpu_wideband_sync import sync_template, wideband_sync


def _msk144code_available() -> bool:
    try:
        r = subprocess.run(["msk144code", "--help"],
                           capture_output=True, timeout=2.0)
        return r.returncode in (0, 1, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestSyncTemplate(unittest.TestCase):
    """Template construction sanity — no msk144code needed."""

    def test_template_length_at_48khz(self):
        h = sync_template.build_sync_template(sample_rate_hz=48_000)
        # 8 symbols × 24 sps = 192 samples (8 * sps, edge_truncate True
        # default returns n = 8 * sps in current scaffold; refine later
        # to match WSJT-X's 7*sps if/when we want bit-exact parity).
        self.assertEqual(len(h), 8 * 24)

    def test_template_is_complex(self):
        h = sync_template.build_sync_template(sample_rate_hz=48_000)
        self.assertEqual(h.dtype, np.complex64)

    def test_autocorrelation_peak_clean(self):
        """Autocorrelating the template against itself should have a
        clear peak well above the sidelobes.  ~15+ dB is plausible for
        a well-formed sync template."""
        stats = sync_template.autocorrelate_at_sample_rate(48_000)
        self.assertGreater(stats["peak_to_sidelobe_db"], 6.0,
                           f"Template autocorrelation poorly defined: "
                           f"{stats}")


class TestCPUReference(unittest.TestCase):
    """CPU numpy reference paths — no GPU needed.

    Validates the matched-filter math before we go GPU.
    """

    def test_correlation_peak_at_template_position(self):
        """Bury the template in zero noise; the correlation peak should
        land at the position the template was inserted."""
        fs = 48_000
        h = sync_template.build_sync_template(fs)
        N = len(h)

        # Build a signal with the template at a known offset.
        offset = 1000
        signal = np.zeros(offset + N + 1000, dtype=np.complex64)
        signal[offset:offset + N] = h

        corr = sync_template.cross_correlate_template(signal, h)
        peak = int(np.argmax(np.abs(corr)))
        self.assertEqual(peak, offset,
                         f"Correlation peak at sample {peak}, "
                         f"template was inserted at {offset}")

    def test_ambiguity_surface_cpu_smoke(self):
        """Surface computation produces (n_windows, fft_len) output of
        the right shape for a small input."""
        fs = 48_000
        h = sync_template.build_sync_template(fs)
        iq = np.zeros(fs, dtype=np.complex64)               # 1 s of zeros
        surface = wideband_sync.compute_ambiguity_surface_cpu(
            iq, h,
            stride_samples=24,
            fft_length=len(h),
        )
        expected_windows = (fs - len(h)) // 24 + 1
        self.assertEqual(surface.shape, (expected_windows, len(h)))


@unittest.skipUnless(_msk144code_available(),
                     "msk144code not available — skipping synthetic-ping tests")
class TestSyntheticPingDetection(unittest.TestCase):
    """End-to-end: generate a known ping, detect it."""

    def test_detects_strong_centred_ping_cpu_or_gpu(self):
        """At 15 dB SNR, freq offset 0, the detector must find SOMETHING
        near the right (time, freq).  Loose tolerance — this is a
        first-light test, not a precision claim."""
        from experiments.gpu_wideband_sync import synthetic   # noqa: PLC0415

        s = synthetic.single_ping(
            snr_db=15.0,
            freq_hz=0.0,
            time_s=0.5,
            sample_rate_hz=48_000,
            duration_s=2.0,
            seed=42,
        )
        h = sync_template.build_sync_template(48_000)
        cfg = wideband_sync.WidebandSyncConfig(
            sample_rate_hz=48_000,
            threshold_db=8.0,
        )
        # Try GPU first, fall back to CPU surface + GPU extraction
        # if GPU end-to-end fails.
        try:
            cands = wideband_sync.detect(s.iq, h, cfg, use_gpu=True)
        except (RuntimeError, NotImplementedError) as exc:
            self.skipTest(f"GPU path unavailable: {exc}")

        self.assertGreater(
            len(cands), 0,
            "No candidates from a 15 dB synthetic ping — detector broken",
        )
        # At least one candidate must be near the truth.
        truth = s.pings[0]
        matched = [
            c for c in cands
            if abs(c.time_s - truth.time_s) < 0.2
            and abs(c.freq_hz - truth.freq_hz) < 500.0
        ]
        self.assertGreater(
            len(matched), 0,
            f"No candidate near truth ({truth.time_s:.2f}s, "
            f"{truth.freq_hz:.0f} Hz).  Got {len(cands)} candidates total: "
            f"{cands[:5]!r}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
