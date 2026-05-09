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
            max_concurrent_decodes=4,
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

    def test_single_ping_decoded_by_both_pipelines(self):
        # ── 1. Generate one strong synthetic ping ─────────────────────
        msg = "CQ K1JT FN20"
        sample_rate = 48_000
        duration_s = 6.0
        freq_khz = 5
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
            seed=42,
        )
        # 1.5 (mono) → reshape to (N, 1) so legacy WAV+_polarization_combine
        # treats it as single-pol IQ.

        tmpdir = Path(tempfile.mkdtemp(prefix="map144_replay_"))
        self.addCleanup(shutil.rmtree, str(tmpdir), ignore_errors=True)

        wav_path = tmpdir / "ping.wav"
        _save_iq_wav_float32(wav_path, iq, sample_rate)

        # Expected operator-visible dial frequency:
        # radio_khz = center_freq_mhz * 1000 + (fc_hz - 1500) / 1000
        #           = 50260 + (6500 - 1500) / 1000
        #           = 50265
        expected_radio_khz = (
            center_freq_mhz * 1000.0
            + (expected_fc_hz - 1500.0) / 1000.0
        )

        # ── 2. Legacy pipeline (replay_iq_wav) ────────────────────────
        # Legacy writes to MSK144/detections/decodes.jsonl unconditionally.
        # We capture by filtering on the run_start timestamp.
        from tests.headless_replay import replay_iq_wav  # noqa: PLC0415

        legacy_decodes_path = (
            _REPO_ROOT / "MSK144" / "detections" / "decodes.jsonl"
        )

        # Note: replay_iq_wav uses the engine's own settings; we pass
        # center_freq_mhz so radio_khz logging matches what the Block
        # path will compute too.
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

        # ── 3. Block pipeline ─────────────────────────────────────────
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
        block_decode_events = _read_block_decode_events_jsonl(
            block_decode_jsonl,
        )
        block_decoded = [
            e for e in block_decode_events
            if (e.get("payload") or {}).get("outcome") == "decoded"
        ]

        # Block also writes its own decodes.jsonl into output_dir (via
        # extract_and_decode) — the same shape as the legacy file.
        block_legacyshape_jsonl = block_output_dir / "decodes.jsonl"
        block_legacyshape_decodes = _read_block_decode_events_jsonl(
            block_legacyshape_jsonl,
        )

        # ── 4. Compare ────────────────────────────────────────────────
        # Pull message + radio_khz from each side.
        def _msgs(entries: list[dict], path_to_msg, path_to_freq):
            out = []
            for e in entries:
                m = path_to_msg(e)
                f = path_to_freq(e)
                if m and f is not None:
                    out.append((m.strip(), int(round(float(f)))))
            return out

        legacy_pairs = _msgs(
            legacy_decodes,
            lambda e: e.get("message", ""),
            lambda e: e.get("radio_khz"),
        )
        # Block legacy-shape JSONL has the same field layout as the legacy
        # one (extract_and_decode writes both in the same format).
        block_pairs = _msgs(
            block_legacyshape_decodes,
            lambda e: e.get("message", ""),
            lambda e: e.get("radio_khz"),
        )

        # Both pipelines must have decoded at least one entry.
        self.assertGreater(
            len(legacy_pairs), 0,
            f"Legacy pipeline decoded nothing.  "
            f"legacy_decodes={legacy_decodes}",
        )
        self.assertGreater(
            len(block_pairs), 0,
            f"Block pipeline decoded nothing.  "
            f"block_legacyshape_decodes={block_legacyshape_decodes}; "
            f"det_stats={block_results['det_stats']}; "
            f"dec_stats={block_results['dec_stats']}",
        )

        # The expected message + freq must appear in BOTH sets.
        legacy_set = set(legacy_pairs)
        block_set = set(block_pairs)

        # Tolerance: ±1 kHz (channel spacing).  On strong synthetic IQ
        # the freq should match exactly, but allow ±1 to defend against
        # an off-by-one in the legacy fc_offset estimation rounding.
        def _has_match(pairs_set, expect_msg, expect_khz, tol_khz=1):
            for m, f in pairs_set:
                # Either pipeline may prepend "<…>" or otherwise decorate;
                # match on the substring being present.
                if expect_msg in m and abs(f - expect_khz) <= tol_khz:
                    return True
            return False

        self.assertTrue(
            _has_match(legacy_set, msg, int(round(expected_radio_khz))),
            f"Legacy did not decode {msg!r} @ {int(expected_radio_khz)} kHz; "
            f"got {legacy_set}",
        )
        self.assertTrue(
            _has_match(block_set, msg, int(round(expected_radio_khz))),
            f"Block did not decode {msg!r} @ {int(expected_radio_khz)} kHz; "
            f"got {block_set}",
        )

        # The intersection of the two pipelines, by (msg-substring, khz±1)
        # must contain at least the expected entry — this is the §10
        # core invariant: same IQ → same operationally-visible decode.
        # Phrased as "is at least the expected decode shared"; not "are
        # the sets identical" (sidelobe / spurious-noise decodes can
        # differ between pipelines without compromising the cut-over).
        shared = []
        for m_legacy, f_legacy in legacy_set:
            for m_block, f_block in block_set:
                # Same callsign tokens, freq within 1 kHz.
                tokens_legacy = set(m_legacy.split())
                tokens_block = set(m_block.split())
                shared_toks = tokens_legacy & tokens_block
                if shared_toks and abs(f_legacy - f_block) <= 1:
                    shared.append(
                        (sorted(shared_toks), f_legacy, f_block)
                    )
                    break
        self.assertGreater(
            len(shared), 0,
            f"No decode pair (msg-token + freq) shared between pipelines.\n"
            f"  legacy={legacy_set}\n"
            f"  block={block_set}",
        )

        # Diagnostic: print summary so a maintainer can see at a glance
        # that the gate passed without parsing the verbose output.
        print(
            f"\n[§10 Stage 2] legacy={len(legacy_pairs)} decode(s) "
            f"block={len(block_pairs)} decode(s) "
            f"shared={len(shared)} "
            f"e.g. {shared[0] if shared else None}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
