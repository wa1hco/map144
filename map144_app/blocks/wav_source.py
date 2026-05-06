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
"""WavSource — replay IQ WAV files through the Block / Stream pipeline.

See ``docs/block-stream-design.md`` §9 #10 (replay tools land in Phase 3
because §10 bit-exact-replay validation depends on them).

Reads a complex-IQ WAV file once, hands chunks of ``samples_per_record``
samples down the ``iq`` output stream, and raises ``StopIteration`` at
EOF for a clean shutdown.

WAV parsing reuses :func:`map144_app.runtime._load_wav_complex` — that
function already handles every format variant the project encounters
(PCM 8 / 16 / 32-bit, IEEE float32, mono real → analytic, stereo IQ,
dual-channel dual-pol) and resamples to the target rate.  Importing it
keeps a single source of truth; future cleanup can move that helper
into a shared ``io/`` module without touching this block.

Mono only for v1.  A separate ``DualPolWavSource`` will use
:class:`CoherentPair` once the dual-pol replay path needs it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .source import Source, SourceConfig

log = logging.getLogger(__name__)


@dataclass
class WavSourceConfig(SourceConfig):
    """Per-WavSource config.  Inherits clock / drift settings from SourceConfig."""

    wav_path: str = ""
    samples_per_record: int = 1024

    # If True, ``produce()`` paces emission so the sample clock advances at
    # real-time rate.  Default False — replay validation does not need
    # real-time, and faster replay is desirable.  The drift-injection §10
    # test is the case that wants ``True``.
    realtime_pacing: bool = False


class WavSource(Source):
    """Replay an IQ WAV file as an :class:`IQStream`.

    On ``on_start`` the file is loaded entirely into memory (typical
    capture sizes are tens of MB; streaming would complicate format
    handling without payoff at v1).  Each ``produce()`` returns the next
    ``samples_per_record`` samples; EOF raises ``StopIteration`` to
    signal clean run-loop exit.
    """

    config_type = WavSourceConfig

    def __init__(self, name: str):
        super().__init__(name)
        self._samples: np.ndarray | None = None
        self._cursor: int = 0
        self._produce_anchor_wall: float = 0.0

    def on_start(self) -> None:
        super().on_start()
        cfg: WavSourceConfig = self.config  # type: ignore[assignment]
        if not cfg.wav_path:
            raise RuntimeError(f"{self.name}: WavSourceConfig.wav_path is required")
        path = Path(cfg.wav_path)
        if not path.is_file():
            raise FileNotFoundError(f"{self.name}: WAV file not found: {path}")

        # Lazy import to avoid pulling runtime.py at module-import time
        # (runtime.py drags in Qt and a lot of DSP code; we only need the
        # WAV parser).
        from ..runtime import _load_wav_complex  # noqa: PLC0415

        samples, _ = _load_wav_complex(str(path), cfg.sample_rate_hz)
        # _load_wav_complex returns shape (N, 1) for mono / single-IQ-pair
        # files, (N, 2) for dual-channel.  v1 supports mono only.
        if samples.ndim == 2 and samples.shape[1] == 2:
            log.warning(
                "%s: dual-channel WAV detected; v1 emits H channel only "
                "(use DualPolWavSource for coherent-pair replay)", self.name,
            )
            samples = samples[:, 0]
        else:
            samples = samples.reshape(-1)
        self._samples = samples.astype(np.complex64, copy=False)
        self._cursor = 0
        self._produce_anchor_wall = self._real_anchor_wall
        log.info(
            "%s: loaded %d samples from %s (%.2f s @ %d Hz)",
            self.name, self._samples.size, path,
            self._samples.size / cfg.sample_rate_hz,
            cfg.sample_rate_hz,
        )

    def produce(self) -> np.ndarray | None:
        cfg: WavSourceConfig = self.config  # type: ignore[assignment]
        if self._samples is None:
            raise RuntimeError(f"{self.name}: produce() before on_start()")

        if self._cursor >= self._samples.size:
            raise StopIteration

        # Optional real-time pacing — used by the drift-injection §10 test
        # so that sample-clock and wall-clock advance at the same rate, and
        # only the synthetic test_clock_drift_ppm produces measurable drift.
        if cfg.realtime_pacing:
            target_wall = (
                self._produce_anchor_wall
                + (self._sample_counter + cfg.samples_per_record) / cfg.sample_rate_hz
            )
            slack = target_wall - time.time()
            if slack > 0:
                end_at = time.time() + slack
                while not self._stopping.is_set():
                    remaining = end_at - time.time()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.050))
                if self._stopping.is_set():
                    return None

        end = min(self._cursor + cfg.samples_per_record, self._samples.size)
        chunk = self._samples[self._cursor:end]
        self._cursor = end
        # Trailing partial chunk is fine — the contract says record length
        # may vary tick-to-tick (§3.1).  Consumers that need fixed windows
        # accumulate themselves.
        return np.ascontiguousarray(chunk)
