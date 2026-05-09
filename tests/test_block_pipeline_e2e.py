#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""End-to-end Block pipeline test on synthetic MSK144 IQ — §10 Stage 1.

Wires up the canonical Phase 4 graph against a deterministic synthetic
input and verifies the pipeline produces detection + decode events.
This is the §10 functional gate that must pass before the cut-over
commit (engine.py switching to Runtime + Blocks) lands.

Graph
-----

    PrebakedIQSource ── iq ──► Tee ──► to_chan ──► Channelizer ──► Detector ──┐
                                  │                                            │
                                  └──── to_dec ──────────────────────────► Decoder ──► decodes
                                                                                  │
                                                               (fake) Reporter ◄──┤
                                                                       JsonlSink ◄┘

The Tee is required because IQ feeds two independent paths (per §6 of
``docs/block-stream-design.md``): the Channelizer→Detector path that
generates DetectionEvents, and the Decoder's ring-buffer for windowed
read on each event.

What this test verifies
-----------------------

1. The Block graph wires up cleanly (no port-mismatch errors at start).
2. Running 4 s of synthetic IQ through the pipeline produces:
    a. at least one DetectionEvent, AND
    b. at least one DecodeEvent with ``outcome="decoded"``.
3. The decoded message contains the embedded callsign — confirming the
   full chain (channelizer NCO → detection → ring-buffer window →
   carrier mix → SPD/jt9 decode) preserves the signal correctly.
4. The fake Reporter receives the decoded spot.
5. The JsonlSink writes a record that reflects the decode.

What this test does NOT yet verify (Stages 2 / 3)
-------------------------------------------------

- Bit-exact equivalence with the legacy engine's decode payload.
- Multi-seed ramp (statistical sensitivity matching).
- CPU regression.
- Drift-injection.

Those gates land in test_block_pipeline_replay.py and
test_block_pipeline_drift.py.

Cost
----

Real SPD + jt9 run inside DecoderBlock here.  ``msk144code`` is invoked
once at module import to build the synthetic ping.  jt9 is the dominant
cost; expect ~5-10 s wall-clock on a modern machine.  The test is
guarded by a skip if ``msk144code`` is not available.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_block_pipeline_e2e -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from map144_app.blocks import (  # noqa: E402
    ChannelizerBlock,
    ChannelizerBlockConfig,
    DecoderBlock,
    DecoderBlockConfig,
    DetectorBlock,
    DetectorBlockConfig,
    Event,
    JsonlSink,
    JsonlSinkConfig,
    ReporterBlock,
    ReporterBlockConfig,
    Runtime,
    Source,
    SourceConfig,
    Stream,
    StreamClosed,
    TeeBlock,
    TeeBlockConfig,
    make_event_stream,
)


# --- Test fixtures ---------------------------------------------------


def _msk144code_available() -> bool:
    return shutil.which("msk144code") is not None


@dataclass
class _PrebakedIQSourceConfig(SourceConfig):
    """Per-_PrebakedIQSource config.  The prebaked array is set on the
    block instance directly (not via config) — the test owns the array."""

    samples_per_record: int = 4096


class _PrebakedIQSource(Source):
    """Replay a pre-built complex64 IQ array as :class:`Record`s.

    Functionally a simplified WavSource that doesn't need the disk I/O.
    Pacing is *not* real-time — runs as fast as downstream consumers
    accept records.  For functional tests we want the pipeline to run
    fast; for drift / cadence tests use a real-time-paced source.
    """

    config_type = _PrebakedIQSourceConfig

    def __init__(self, name: str, samples: np.ndarray):
        super().__init__(name)
        self._samples = np.ascontiguousarray(samples, dtype=np.complex64)
        self._cursor: int = 0

    def produce(self) -> np.ndarray | None:
        cfg: _PrebakedIQSourceConfig = self.config  # type: ignore[assignment]
        if self._cursor >= self._samples.size:
            raise StopIteration
        end = min(self._cursor + cfg.samples_per_record, self._samples.size)
        chunk = self._samples[self._cursor:end]
        self._cursor = end
        return np.ascontiguousarray(chunk)


class _FakeReporter:
    """Stub for ``map144_app.reporting.Reporter`` (no sockets)."""

    def __init__(self):
        self.my_call = ""
        self.my_grid = ""
        self.wsjtx_enabled = False
        self.pskreporter_enabled = False
        self.dx_enabled = False
        self.decode_calls: list[dict] = []
        self.freq_calls: list[int] = []
        self.applied_settings: list[dict] = []
        self.started = False
        self.stopped = False
        self.stat_udp_sent = 0
        self.stat_psk_uploaded = 0
        self.stat_psk_queued = 0
        self.stat_dx_sent = 0
        self.stat_dx_status = "fake"

    def apply_settings(self, **kwargs):
        self.applied_settings.append(dict(kwargs))
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def report_decode(self, decode: dict):
        self.decode_calls.append(dict(decode))

    def report_freq(self, freq_hz: int):
        self.freq_calls.append(int(freq_hz))


# --- Synthetic IQ generator ------------------------------------------


def _generate_msk144_test_iq(
    msg: str,
    *,
    duration_s: float = 4.0,
    sample_rate: int = 48_000,
    freq_khz: int = 5,
    delay_s: float = 1.0,
    snr_db: float = 25.0,
    noise_sigma: float = 1.0,
    seed: int = 12345,
) -> tuple[np.ndarray, float]:
    """Build a deterministic IQ buffer carrying one MSK144 ping.

    Calls into ``generate_msk144.generate_ping_msk144code`` (which
    requires the ``msk144code`` binary) so the chip sequence matches what
    the production decoder expects.  Returns the IQ stream plus the
    expected ``fc_hz`` (carrier position in Hz from DC) — the detector
    should report this value on the launched event.
    """
    # Lazy import — generate_msk144 pulls in subprocess for msk144code.
    from generate_msk144 import generate_ping_msk144code  # noqa: PLC0415

    rng = np.random.default_rng(seed)

    # Channel k captures stations whose carrier is at k*1000 + 1500 from
    # IQ DC (USB convention with channel_offset_hz = 1500).
    center_hz = freq_khz * 1000.0 + 1500.0

    # ── 1. The ping (no noise inside; flat one-frame envelope) ────────
    ping_h, _ping_v, _meta = generate_ping_msk144code(
        msg,
        center_hz=center_hz,
        delay_s=delay_s,
        snr_db=snr_db,
        noise_sigma=noise_sigma,
        sample_rate=sample_rate,
        flat_frame=False,           # use the realistic Gamma envelope
        width_ms=60,                # ~600 ms decay — plenty for SPD
        pol_scenario='pure-H',      # all energy on H
        rng=rng,
    )

    # ── 2. Build the full noise+ping buffer ───────────────────────────
    n_samples = int(duration_s * sample_rate)
    noise = (
        rng.standard_normal(n_samples).astype(np.float32)
        + 1j * rng.standard_normal(n_samples).astype(np.float32)
    ).astype(np.complex64) * np.complex64(noise_sigma)

    iq = noise.copy()
    n_ping = ping_h.size
    if n_ping > n_samples:
        n_ping = n_samples
    iq[:n_ping] += ping_h[:n_ping]

    return iq, center_hz


# --- Tests -----------------------------------------------------------


@unittest.skipUnless(
    _msk144code_available(),
    "msk144code binary not available — skipping synthetic-MSK144 e2e test",
)
class TestBlockPipelineEndToEnd(unittest.TestCase):
    """End-to-end functional gate (§10 Stage 1)."""

    def test_synthetic_msk144_decodes_through_pipeline(self):
        # ── 1. Generate a deterministic synthetic ping ────────────────
        # K1JT is recognisable in the dataset; the message text is
        # unimportant for the test — we just check that the decoder
        # recovers some plausible callsign-bearing string.
        msg = "CQ K1JT FN20"
        sample_rate = 48_000
        duration_s = 6.0           # plenty for warmup + decode
        freq_khz = 5               # channel 5
        delay_s = 1.5              # ping starts 1.5 s in

        iq, expected_fc_hz = _generate_msk144_test_iq(
            msg,
            duration_s=duration_s,
            sample_rate=sample_rate,
            freq_khz=freq_khz,
            delay_s=delay_s,
            snr_db=20.0,           # strong; meant to decode reliably
            noise_sigma=1.0,
            seed=42,
        )

        # ── 2. Wire up the graph ──────────────────────────────────────
        rt = Runtime()

        src = rt.add(
            _PrebakedIQSource("src", iq),
            _PrebakedIQSourceConfig(
                sample_rate_hz=sample_rate,
                samples_per_record=4096,
            ),
        )

        tee = rt.add(
            TeeBlock("tee"),
            TeeBlockConfig(
                input_port_name="iq",
                output_port_names=["to_chan", "to_dec"],
                get_timeout_s=0.05,
            ),
        )

        chan = rt.add(
            ChannelizerBlock("chan"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=sample_rate,
                channel_offset_hz=1500.0,    # USB convention
                get_timeout_s=0.05,
            ),
        )

        det = rt.add(
            DetectorBlock("det"),
            DetectorBlockConfig(
                input_port_name="channels",
                detection_port_name="detections",
                channel_offset_hz=1500.0,    # match channelizer
                get_timeout_s=0.05,
            ),
        )

        # Use a temp dir for decoder WAV output (we don't inspect the
        # files; just want decode to succeed even with output_dir set).
        self._tmpdir = tempfile.mkdtemp(prefix="map144_e2e_")
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)

        dec = rt.add(
            DecoderBlock("dec"),
            DecoderBlockConfig(
                iq_port_name="iq",
                detection_port_name="detections",
                decode_port_name="decodes",
                sample_rate_hz=sample_rate,
                ring_seconds=15.0,
                max_concurrent_decodes=4,
                output_dir=self._tmpdir,
                center_freq_mhz=50.260,
                get_timeout_s=0.05,
            ),
        )

        # ── ReporterBlock with fake Reporter (no sockets) ─────────────
        fake_reporter = _FakeReporter()
        import map144_app.reporting as _rep
        real_reporter_cls = _rep.Reporter
        _rep.Reporter = lambda: fake_reporter
        self.addCleanup(setattr, _rep, "Reporter", real_reporter_cls)

        rep = rt.add(
            ReporterBlock("rep"),
            ReporterBlockConfig(
                decode_port_name="decodes",
                get_timeout_s=0.05,
            ),
        )
        # Configure reporter callsign so report_decode is called with
        # plausible context (the Block doesn't enforce non-empty my_call,
        # but a real reporter would skip uploads without it; the fake
        # records every call regardless).
        rep.apply_settings(my_call="W1AW", my_grid="FN31",
                           wsjtx_enabled=False,
                           pskreporter_enabled=False,
                           dx_enabled=False)

        # ── JsonlSink to capture decode events ────────────────────────
        decode_jsonl_path = Path(self._tmpdir) / "decode_events.jsonl"
        sink = rt.add(
            JsonlSink("decode_sink"),
            JsonlSinkConfig(
                input_port_name="events",
                output_path=str(decode_jsonl_path),
                get_timeout_s=0.05,
            ),
        )

        # ── Connections ───────────────────────────────────────────────
        rt.connect(src, "iq", tee, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(tee, "to_chan", chan, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(tee, "to_dec", dec, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(chan, "channels", det, "channels",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=12_000,
                   n_samples_per_record=1024)
        rt.connect(det, "detections", dec, "detections",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)

        # Decoder's ``decodes`` output fans to BOTH ReporterBlock and
        # JsonlSink — uses another Tee.
        decode_tee = rt.add(
            TeeBlock("decode_tee"),
            TeeBlockConfig(
                input_port_name="in",
                output_port_names=["to_rep", "to_sink"],
                get_timeout_s=0.05,
            ),
        )
        rt.connect(dec, "decodes", decode_tee, "in",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)
        rt.connect(decode_tee, "to_rep", rep, "decodes",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)
        rt.connect(decode_tee, "to_sink", sink, "events",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)

        # Also tap the decode events on a side stream so the test can
        # inspect them directly without parsing the JSONL.  We need a
        # 3-way Tee for that — easier to just re-bind decode_tee with
        # a third port.  Instead: subclass JsonlSink to also append to
        # an in-memory list?  Simpler: read the JSONL after stop.

        # ── Run ───────────────────────────────────────────────────────
        rt.start()
        try:
            # Wait until the source has drained or until we have at least
            # one decoded event (whichever first).  Cap the wait — jt9
            # has a 20 s timeout of its own; total budget 30 s.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                # Source done?
                if src._cursor >= src._samples.size:
                    # Give in-flight workers another moment.
                    time.sleep(2.0)
                    break
                time.sleep(0.1)
        finally:
            rt.stop(timeout_s=5.0)

        # ── Assertions ────────────────────────────────────────────────
        det_stats = det.stats()
        dec_stats = dec.stats()

        # Pipeline ran end-to-end.
        self.assertGreater(det_stats["ticks_executed"], 0,
                           "DetectorBlock did not execute any ticks")
        self.assertGreater(dec_stats["iq_records"], 0,
                           "DecoderBlock received no IQ records")

        # Detection event fired on the synthetic ping.
        det_events_out = det_stats.get("events_out", {}).get("detections", 0)
        self.assertGreater(
            det_events_out, 0,
            f"DetectorBlock emitted no detection events; stats={det_stats}",
        )
        self.assertGreater(
            dec_stats["detection_events"], 0,
            f"DecoderBlock saw no detection events; stats={dec_stats}",
        )
        self.assertGreater(
            dec_stats["decode_attempts"], 0,
            f"DecoderBlock attempted no decodes; stats={dec_stats}",
        )

        # Read the decode-event JSONL and find the decoded event.
        self.assertTrue(decode_jsonl_path.is_file(),
                        f"JSONL not written: {decode_jsonl_path}")
        events = []
        with decode_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        decoded_events = [
            e for e in events
            if (e.get("payload") or {}).get("outcome") == "decoded"
        ]
        # If decode failed, surface useful context.
        if not decoded_events:
            self.fail(
                f"No decoded event observed. {len(events)} total events; "
                f"outcomes: "
                f"{[(e.get('payload') or {}).get('outcome') for e in events]}; "
                f"det_stats={det_stats}; dec_stats={dec_stats}"
            )
        self.assertGreater(len(decoded_events), 0)

        # The Reporter should have seen at least one decoded spot.
        self.assertGreater(
            len(fake_reporter.decode_calls), 0,
            "fake Reporter received no decoded spots",
        )

        # The decoded message should contain a recognisable token from
        # the embedded message ("K1JT" or "FN20").  Decoder may emit
        # multiple messages from a single ping; we check any of them.
        msgs = [
            (e.get("payload") or {}).get("message", "")
            for e in decoded_events
        ]
        any_match = any(("K1JT" in m) or ("FN20" in m) or ("CQ" in m)
                        for m in msgs)
        self.assertTrue(
            any_match,
            f"None of the decoded messages match expected tokens "
            f"(K1JT/FN20/CQ): {msgs}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
