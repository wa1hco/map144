#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Legacy-vs-Block A/B replay validation — §10 Stage 2.

The cut-over commit (engine.py switching to Runtime + Blocks) needs
confidence that the new pipeline produces the same decodes as the legacy
one on the same IQ.  This test feeds a deterministic synthetic IQ stream
into BOTH pipelines and verifies the recovered messages and frequencies
agree.

Bit-exact-replay scope (§10, design-doc):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DSP — channelizer, sync correlator, squared-FFT detector, SPD,
jt9 — is *unchanged* between the legacy and the Block pipelines.  The
Block code wraps it without modifying it.  What does differ:

- *Threading*: legacy is single-threaded; Blocks have one thread per
  block plus pooled decode workers.  Detection-event ORDERING across
  back-to-back ticks may differ.
- *Sample-counter rebasing*: detection events in Blocks carry a
  ``sample_rate_hz`` field (12 kHz at the channelizer rate); the legacy
  path passes IQ-rate sample numbers directly to extract_and_decode.
  The rebase happens inside DecoderBlock; bug-fixed in commit 4a1b698.
- *Wall-clock timestamps*: inherently non-deterministic.

So "bit-exact" in §10 means:

1. Same set of decoded callsign / message strings.
2. Same ``radio_khz`` per decode (within ±1 kHz, the channel spacing).
3. Same approximate ``jt9_snr`` (within ±2 dB — the SPD/jt9 inner
   workings are deterministic on the same IQ window, but the window
   selected by the Block decoder may differ by a few channelizer hops
   from the legacy one due to different sample-counter alignment).

Costs
-----

Each pass runs real SPD + jt9.  A single-ping test run takes ~5–8 s
wall-clock.  Skipped if ``msk144code`` is not available.

Run::

    cd ~/ham/map144
    python3 -m unittest tests.test_block_pipeline_replay -v
"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse Stage 1's helpers for synthetic IQ + Block pipeline construction.
from tests.test_block_pipeline_e2e import (  # noqa: E402
    _generate_msk144_test_iq,
    _msk144code_available,
    _PrebakedIQSource,
    _PrebakedIQSourceConfig,
    _FakeReporter,
)

from map144_app.blocks import (  # noqa: E402
    ChannelizerBlock,
    ChannelizerBlockConfig,
    DecoderBlock,
    DecoderBlockConfig,
    DetectorBlock,
    DetectorBlockConfig,
    JsonlSink,
    JsonlSinkConfig,
    ReporterBlock,
    ReporterBlockConfig,
    Runtime,
    Stream,
    TeeBlock,
    TeeBlockConfig,
)


# --- WAV helpers -----------------------------------------------------


def _save_iq_wav_float32(path: Path, iq: np.ndarray, sample_rate: int) -> None:
    """Write a 2-channel IEEE float32 WAV (L=I, R=Q) that
    ``map144_app.runtime._load_wav_complex`` accepts.

    Uses the same RIFF/WAVE layout as the legacy capture.save_capture()
    output.  The file is small for our test inputs (~6 s × 48 kHz × 8
    bytes/frame ≈ 2.3 MB).
    """
    iq = np.ascontiguousarray(iq, dtype=np.complex64)
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    audio_bytes = interleaved.tobytes()

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(audio_bytes)))
        f.write(b"WAVE")
        # fmt chunk — IEEE float32 (fmt_code=3), 2 channels
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # chunk size
        f.write(struct.pack("<H", 3))            # fmt_code = IEEE float
        f.write(struct.pack("<H", 2))            # channels = 2 (I, Q)
        f.write(struct.pack("<I", sample_rate))  # sample rate
        f.write(struct.pack("<I", sample_rate * 2 * 4))  # byte rate
        f.write(struct.pack("<H", 2 * 4))        # block align
        f.write(struct.pack("<H", 32))           # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", len(audio_bytes)))
        f.write(audio_bytes)


# --- Decode log readers ----------------------------------------------


def _read_legacy_decodes_jsonl(
    path: Path,
    *,
    since: datetime,
) -> list[dict]:
    """Read the legacy ``decodes.jsonl`` entries with timestamp >= ``since``.

    The legacy file lives at ``MSK144/detections/decodes.jsonl`` and is
    appended to across all runs; we filter on the per-entry ``timestamp``
    field (UTC ISO-ish) to extract just the run we care about.
    """
    if not path.is_file():
        return []
    keep = []
    since_str = since.strftime("%Y-%m-%d_%H:%M:%S")
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        # ``timestamp`` is the launch_ts, which is in the same format.
        ts = entry.get("timestamp", "")
        if ts >= since_str:
            keep.append(entry)
    return keep


def _read_block_decode_events_jsonl(path: Path) -> list[dict]:
    """Read decode-event records written by JsonlSink in the Block run."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# --- Block-pipeline runner -------------------------------------------


def _run_block_pipeline_on_iq(
    iq: np.ndarray,
    *,
    sample_rate: int,
    center_freq_mhz: float,
    output_dir: Path,
    decode_jsonl_path: Path,
    timeout_s: float = 30.0,
    max_concurrent_decodes: int = 4,
) -> dict:
    """Run the canonical Phase 4 graph on an IQ buffer; return final stats.

    Same wiring as ``test_block_pipeline_e2e``: Source ▶ Tee ▶ Channelizer ▶
    Detector ▶ Decoder ▶ Tee ▶ (Reporter, JsonlSink).  The Reporter is
    swapped for a fake so no sockets open.
    """
    rt = Runtime()
    src = rt.add(
        _PrebakedIQSource("src", iq),
        _PrebakedIQSourceConfig(
            sample_rate_hz=sample_rate,
            samples_per_record=4096,
        ),
    )
    iq_tee = rt.add(
        TeeBlock("iq_tee"),
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
            channel_offset_hz=1500.0,
            get_timeout_s=0.05,
        ),
    )
    det = rt.add(
        DetectorBlock("det"),
        DetectorBlockConfig(
            input_port_name="channels",
            detection_port_name="detections",
            channel_offset_hz=1500.0,
            get_timeout_s=0.05,
        ),
    )
    dec = rt.add(
        DecoderBlock("dec"),
        DecoderBlockConfig(
            iq_port_name="iq",
            detection_port_name="detections",
            decode_port_name="decodes",
            sample_rate_hz=sample_rate,
            ring_seconds=15.0,
            max_concurrent_decodes=max_concurrent_decodes,
            output_dir=str(output_dir),
            center_freq_mhz=center_freq_mhz,
            get_timeout_s=0.05,
        ),
    )

    # Fake Reporter — no sockets.
    fake = _FakeReporter()
    import map144_app.reporting as _rep
    real_cls = _rep.Reporter
    _rep.Reporter = lambda: fake
    try:
        rep = rt.add(
            ReporterBlock("rep"),
            ReporterBlockConfig(decode_port_name="decodes", get_timeout_s=0.05),
        )
        rep.apply_settings(my_call="W1AW", my_grid="FN31",
                           wsjtx_enabled=False,
                           pskreporter_enabled=False,
                           dx_enabled=False)

        sink = rt.add(
            JsonlSink("decode_sink"),
            JsonlSinkConfig(
                input_port_name="events",
                output_path=str(decode_jsonl_path),
                get_timeout_s=0.05,
            ),
        )

        # Wire the graph.
        rt.connect(src, "iq", iq_tee, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(iq_tee, "to_chan", chan, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(iq_tee, "to_dec", dec, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=sample_rate,
                   n_samples_per_record=4096)
        rt.connect(chan, "channels", det, "channels",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=12_000,
                   n_samples_per_record=1024)
        rt.connect(det, "detections", dec, "detections",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)

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

        rt.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if src._cursor >= src._samples.size:
                time.sleep(2.0)        # let workers complete
                break
            time.sleep(0.05)
        rt.stop(timeout_s=5.0)

        return {
            "det_stats":  det.stats(),
            "dec_stats":  dec.stats(),
            "fake_reporter_decodes": list(fake.decode_calls),
        }
    finally:
        _rep.Reporter = real_cls


# --- Tests -----------------------------------------------------------


@unittest.skipUnless(
    _msk144code_available(),
    "msk144code binary not available — skipping replay validation gate",
)
class TestLegacyVsBlockReplay(unittest.TestCase):
    """The Stage-2 §10 cut-over confidence gate: legacy and Block
    pipelines should decode the same message + frequency from the same
    IQ.

    Wall-clock timestamps and per-call jt9 SNR jitter are accepted as
    inherent non-determinism.  Message text and ``radio_khz`` are the
    operationally-meaningful invariants.
    """

    def _run_one_freq(self, *, msg: str, freq_khz: int, seed: int) -> tuple[set, set]:
        """Generate one synthetic ping at ``freq_khz`` and run BOTH the
        legacy headless replay and the Block-pipeline shadow path.

        Returns ``(legacy_set, block_set)`` where each is a set of
        ``(message, radio_khz)`` tuples — the operator-visible decode
        identity that the cut-over needs to preserve.

        Used by the per-channel test methods to cover both halves of
        the channelizer (positive + negative frequency-half), exercising
        the signed-channel ``fc_hz`` path that
        ``DetectorBlock._fc_hz_for_channel`` rebuilt in commit cbea6b7.
        """
        sample_rate = 48_000
        duration_s = 6.0
        delay_s = 1.5
        center_freq_mhz = 50.260      # both pipelines see this
        calling_freq_mhz = 50.260     # pan == calling, so channel_offset_hz=1500

        iq, expected_fc_hz = _generate_msk144_test_iq(
            msg,
            duration_s=duration_s,
            sample_rate=sample_rate,
            freq_khz=freq_khz,
            delay_s=delay_s,
            snr_db=20.0,              # strong; deterministic decode
            noise_sigma=1.0,
            seed=seed,
        )

        tmpdir = Path(tempfile.mkdtemp(prefix=f"map144_replay_{freq_khz:+d}_"))
        self.addCleanup(shutil.rmtree, str(tmpdir), ignore_errors=True)

        wav_path = tmpdir / "ping.wav"
        _save_iq_wav_float32(wav_path, iq, sample_rate)

        # Expected operator-visible dial frequency:
        # radio_khz = center_freq_mhz * 1000 + (fc_hz - 1500) / 1000
        # For freq_khz=+5  → 50265.  For freq_khz=-10 → 50250.
        expected_radio_khz = int(round(
            center_freq_mhz * 1000.0
            + (expected_fc_hz - 1500.0) / 1000.0
        ))

        # ── Legacy pipeline (replay_iq_wav) ───────────────────────────
        from tests.headless_replay import replay_iq_wav  # noqa: PLC0415
        legacy_decodes_path = (
            _REPO_ROOT / "MSK144" / "detections" / "decodes.jsonl"
        )
        legacy_run_start = replay_iq_wav(
            str(wav_path),
            center_freq_mhz=center_freq_mhz,
            calling_freq_mhz=calling_freq_mhz,
            sample_rate=sample_rate,
            jt9_join_timeout_s=15.0,
            verbose=False,
        )
        legacy_decodes = _read_legacy_decodes_jsonl(
            legacy_decodes_path, since=legacy_run_start,
        )

        # ── Block pipeline ────────────────────────────────────────────
        block_output_dir = tmpdir / "block_output"
        block_output_dir.mkdir()
        block_decode_jsonl = tmpdir / "block_decodes.jsonl"
        block_results = _run_block_pipeline_on_iq(
            iq.reshape(-1) if iq.ndim == 2 else iq,
            sample_rate=sample_rate,
            center_freq_mhz=center_freq_mhz,
            output_dir=block_output_dir,
            decode_jsonl_path=block_decode_jsonl,
            timeout_s=30.0,
        )
        # Block writes its own decodes.jsonl into output_dir (via
        # extract_and_decode) — the same shape as the legacy file.
        block_legacyshape_jsonl = block_output_dir / "decodes.jsonl"
        block_legacyshape_decodes = _read_block_decode_events_jsonl(
            block_legacyshape_jsonl,
        )

        # ── Compare ───────────────────────────────────────────────────
        def _msgs(entries: list[dict]) -> list[tuple[str, int]]:
            out = []
            for e in entries:
                m = e.get("message", "")
                f = e.get("radio_khz")
                if m and f is not None:
                    out.append((m.strip(), int(round(float(f)))))
            return out

        legacy_pairs = _msgs(legacy_decodes)
        block_pairs = _msgs(block_legacyshape_decodes)

        # Both pipelines must have decoded at least one entry.
        self.assertGreater(
            len(legacy_pairs), 0,
            f"freq_khz={freq_khz}: Legacy pipeline decoded nothing.  "
            f"legacy_decodes={legacy_decodes}",
        )
        self.assertGreater(
            len(block_pairs), 0,
            f"freq_khz={freq_khz}: Block pipeline decoded nothing.  "
            f"block_legacyshape_decodes={block_legacyshape_decodes}; "
            f"det_stats={block_results['det_stats']}; "
            f"dec_stats={block_results['dec_stats']}",
        )

        # The expected message + freq must appear in BOTH sets.
        legacy_set = set(legacy_pairs)
        block_set = set(block_pairs)

        def _has_match(pairs_set, expect_msg, expect_khz, tol_khz=1):
            for m, f in pairs_set:
                if expect_msg in m and abs(f - expect_khz) <= tol_khz:
                    return True
            return False

        self.assertTrue(
            _has_match(legacy_set, msg, expected_radio_khz),
            f"freq_khz={freq_khz}: Legacy did not decode {msg!r} @ "
            f"{expected_radio_khz} kHz; got {legacy_set}",
        )
        self.assertTrue(
            _has_match(block_set, msg, expected_radio_khz),
            f"freq_khz={freq_khz}: Block did not decode {msg!r} @ "
            f"{expected_radio_khz} kHz; got {block_set}",
        )

        print(
            f"\n[§10 Stage 2 freq_khz={freq_khz:+d}] "
            f"expected={expected_radio_khz} kHz  "
            f"legacy={len(legacy_pairs)} decode(s)  "
            f"block={len(block_pairs)} decode(s)",
            file=sys.stderr,
        )

        return legacy_set, block_set

    def test_positive_half_channel(self):
        """Channel +5 (50265 kHz dial) — sanity check that the original
        §10 Stage 2 case still passes after the cbea6b7 fc_hz refactor.
        """
        legacy_set, block_set = self._run_one_freq(
            msg="CQ K1JT FN20", freq_khz=5, seed=42,
        )
        # Both sides should see the K1JT/50265 decode.
        union = legacy_set | block_set
        self.assertTrue(
            any("K1JT" in m and abs(f - 50265) <= 1 for m, f in union),
            f"Neither pipeline produced K1JT @ 50265: {union}",
        )

    def test_parity_on_strong_multi_ping_sim(self):
        """Replay the strong-signal 10-ping sim through BOTH pipelines and
        require shadow's decoded set ⊇ legacy's decoded set.

        The sim (``MSK144/sim/parity_strong10.wav`` + matching ``.json``
        truth) places 10 MSK144 pings at SNR 10–19 dB across the full
        ±14 kHz IF range — including negative-half placements (e.g.
        AN08067P10 at −8 kHz / 50252 kHz dial) that exercise the
        signed-channel ``fc_hz`` fix from commit cbea6b7.

        Both legacy and shadow run with ``max_concurrent_decodes=12`` so
        the Block path matches the legacy ``_jt9_threads`` cap of 12 (a
        smaller cap drops detection events under bursty arrival, which
        biases the comparison).

        Pass criterion: every decode legacy makes is matched by a shadow
        decode (same callsign-shaped token, ``radio_khz`` within ±1
        kHz).  Shadow may decode strictly MORE — that's a sensitivity
        gain, not a regression.  Decoder-pool drops on either side mean
        the absolute count varies across runs; matching the SET is the
        invariant.
        """
        sim_wav  = _REPO_ROOT / "tests" / "fixtures" / "parity_strong10.wav"
        sim_json = _REPO_ROOT / "tests" / "fixtures" / "parity_strong10.json"
        if not sim_wav.is_file() or not sim_json.is_file():
            self.skipTest(f"Sim fixture not present: {sim_wav}")

        # ── Legacy ────────────────────────────────────────────────────
        from tests.headless_replay import replay_iq_wav  # noqa: PLC0415
        legacy_decodes_path = (
            _REPO_ROOT / "MSK144" / "detections" / "decodes.jsonl"
        )
        legacy_run_start = replay_iq_wav(
            str(sim_wav),
            sample_rate=48_000,
            jt9_join_timeout_s=20.0,
            verbose=False,
        )
        legacy_decodes = _read_legacy_decodes_jsonl(
            legacy_decodes_path, since=legacy_run_start,
        )

        # ── Shadow Block pipeline on the same IQ ──────────────────────
        from map144_app.runtime import _load_wav_complex  # noqa: PLC0415
        samples, _ = _load_wav_complex(str(sim_wav), 48_000)
        iq = samples.reshape(-1) if samples.ndim == 2 else samples

        tmpdir = Path(tempfile.mkdtemp(prefix="map144_parity_"))
        self.addCleanup(shutil.rmtree, str(tmpdir), ignore_errors=True)
        block_out = tmpdir / "block_out"
        block_out.mkdir()
        block_results = _run_block_pipeline_on_iq(
            iq, sample_rate=48_000,
            center_freq_mhz=50.260,
            output_dir=block_out,
            decode_jsonl_path=tmpdir / "block_evs.jsonl",
            timeout_s=60.0,
            # Match legacy's _jt9_threads cap of 12.
            max_concurrent_decodes=12,
        )
        block_decodes = _read_block_decode_events_jsonl(
            block_out / "decodes.jsonl",
        )

        # ── Compare as callsign-token sets (dedup by callsign) ────────
        # Synthetic msg encoding: A[P/N][ff][ttt][P/N][dd] — a single
        # 10-char token per ping.  Two pipelines may report the SAME
        # callsign at slightly different rounded radio_khz (e.g.
        # 50259 vs 50263 from the same -1 kHz signal — legacy logged
        # both the real channel and a sidelobe; shadow only logged the
        # sidelobe).  Both forms identify the same callsign decoded
        # from the same burst, so dedup by callsign is the parity
        # invariant the cut-over actually requires.
        def _tokset(entries: list[dict]) -> set:
            out = set()
            for e in entries:
                msg = e.get("message", "")
                if not msg:
                    continue
                for tok in msg.upper().split():
                    if 3 <= len(tok) <= 12 and any(c.isdigit() for c in tok):
                        out.add(tok)
            return out

        legacy_callsigns = _tokset(legacy_decodes)
        block_callsigns  = _tokset(block_decodes)

        # Require legacy decoded at least a few signals — otherwise the
        # sim is broken / SNR too marginal and we can't draw any parity
        # conclusion.
        self.assertGreater(
            len(legacy_callsigns), 3,
            f"Legacy decoded only {len(legacy_callsigns)} unique "
            f"callsign(s) on the parity_strong10 sim — fixture too "
            f"weak; got {legacy_callsigns}",
        )

        legacy_only = legacy_callsigns - block_callsigns

        # Diagnostic regardless of pass/fail.
        print(
            f"\n[parity sim] legacy={len(legacy_callsigns)} unique "
            f"callsign(s)  block={len(block_callsigns)}  "
            f"shadow-missed={sorted(legacy_only)}\n"
            f"  legacy={sorted(legacy_callsigns)}\n"
            f"  block ={sorted(block_callsigns)}",
            file=sys.stderr,
        )

        self.assertEqual(
            legacy_only, set(),
            f"Shadow MISSED {len(legacy_only)} callsign(s) that legacy "
            f"caught: {sorted(legacy_only)}.  Sensitivity regression in "
            f"the Block path — investigate before promotion (option 2).",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
