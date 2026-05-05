#!/usr/bin/env python3
# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Headless MAP144 IQ-WAV replay — same DSP path as the GUI, no Qt.

Constructs a minimal :class:`map144_app.engine.Engine`, feeds chunks of an IQ
WAV through ``process_iq_data`` exactly the way the live runtime does, and
joins all spawned jt9 decode threads before returning.  Used by
``run_ramp_tests.py`` so the multi-seed ramp sweep can score MAP144 results
without manual GUI replay.

Usage from CLI:
    python3 headless_replay.py path/to/file_iq.wav

From Python::
    from headless_replay import replay_iq_wav
    run_start = replay_iq_wav('foo_iq.wav')
    # decodes.jsonl entries with timestamp >= run_start belong to this run.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow ``python tests/headless_replay.py`` from anywhere — Python normally
# adds only the script's directory to sys.path, but ``map144_app`` lives at
# the repo root one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from map144_app.engine import Engine
from map144_app.runtime import (
    _load_wav_complex,
    _set_dual_pol,
    _reset_wav_timeline,
    _polarization_combine,
)


def replay_iq_wav(wav_path: str | Path, *,
                  center_freq_mhz: float = 50.260,
                  calling_freq_mhz: float = 50.260,
                  sample_rate: int = 48000,
                  fft_size: int = 2048,
                  jt9_join_timeout_s: float = 30.0,
                  verbose: bool = True) -> datetime:
    """Replay an IQ WAV through the MAP144 DSP pipeline once.

    Returns the wall-clock datetime captured at the start of the run; callers
    can filter ``decodes.jsonl`` entries by ``timestamp >= run_start``.
    """
    eng = Engine(center_freq_mhz=center_freq_mhz,
                 calling_freq_mhz=calling_freq_mhz,
                 sample_rate=sample_rate,
                 fft_size=fft_size)
    # Engine defaults display_center_freq_mhz to -1.0 (a sentinel meaning "not
    # yet reported"); processing.py reads it for radio_khz computation in
    # decodes.jsonl.  Without this, replay decodes log radio_khz=-1000.xxx
    # which breaks MAP144's manifest-based comparison report.
    eng.display_center_freq_mhz = float(center_freq_mhz)

    samples, _ = _load_wav_complex(str(wav_path), eng.sample_rate)
    eng._wav_samples = samples
    eng._wav_path_loaded = str(wav_path)
    eng._wav_index = 0
    eng._wav_done  = False
    is_dual = samples.ndim == 2 and samples.shape[1] == 2
    _set_dual_pol(eng, is_dual)        # safe: _fast_graph_win is None → no-op
    _reset_wav_timeline(eng)

    eng.source_mode = 'wav'
    eng.selected_wav_path = str(wav_path)
    run_start = datetime.now(timezone.utc)
    eng._wav_run_start_time = run_start
    if verbose:
        print(f"[headless] {wav_path}  ({len(samples)} samples @ {sample_rate} Hz, "
              f"dual_pol={'ON' if is_dual else 'OFF'})", flush=True)

    # Feed chunks the same way runtime._process_wav_source_step does.  The chunk
    # size matches the GUI loop; using the same value keeps the channeliser /
    # waterfall buffer geometry identical to live replay.
    chunk_size = eng.fft_size * 4
    n = len(samples)
    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk = _polarization_combine(samples[pos:end].astype(np.complex64))
        wav_seconds = float(eng._wav_time_cursor)
        ts_int  = int(wav_seconds)
        ts_frac = int((wav_seconds - ts_int) * 1e12)
        eng.process_iq_data(chunk, ts_int, ts_frac)
        pos = end

    # Wait for any extract_and_decode daemon threads spawned during the loop
    # to finish — they write directly to MSK144/detections/decodes.jsonl, so
    # if we exit before they complete the decode log will be partial.
    threads = list(getattr(eng, '_jt9_threads', []))
    n_alive = sum(1 for t in threads if t.is_alive())
    if verbose and n_alive:
        print(f"[headless] joining {n_alive} pending decode thread(s) "
              f"(timeout {jt9_join_timeout_s:.0f}s each)...", flush=True)
    for t in threads:
        t.join(timeout=jt9_join_timeout_s)
    # Final settle: a few decode threads can still be in flight via SPD's
    # internal subprocess wait; give them a brief grace period.
    time.sleep(0.5)

    if verbose:
        print(f"[headless] replay complete: {wav_path}", flush=True)
    return run_start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wav', help='Path to IQ WAV (signed-PCM or float32; 2-ch I/Q or 4-ch dual-pol)')
    ap.add_argument('--center-freq-mhz',  type=float, default=50.260)
    ap.add_argument('--calling-freq-mhz', type=float, default=50.260)
    ap.add_argument('--sample-rate',      type=int,   default=48000)
    ap.add_argument('--fft-size',         type=int,   default=2048)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    replay_iq_wav(args.wav,
                  center_freq_mhz=args.center_freq_mhz,
                  calling_freq_mhz=args.calling_freq_mhz,
                  sample_rate=args.sample_rate,
                  fft_size=args.fft_size,
                  verbose=not args.quiet)


if __name__ == '__main__':
    main()
