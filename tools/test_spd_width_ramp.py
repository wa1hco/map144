"""Direct SPD test against a synthetic width ramp.

For each ping in a width-ramped WAV (produced by
``generate_msk144.py --ramp --ramp-mode width``), this harness:

  1. Reads the ping's t_offset and width_ms from the truth JSON sidecar
  2. Slices a ~1.7 s window of audio centered on t_offset
  3. Converts the real audio to complex baseband (Hilbert → shift −1500 Hz)
  4. Applies the audio BPF that MAP144 applies on the live path
  5. Calls ``msk144_spd_decode`` directly
  6. Records the returned navg / xmax / fest / decoded-msg

Then prints a side-by-side table: truth width vs SPD-reported navg.
If navg=1 dominates regardless of width, that confirms the suspected
bug in SPD's coherent-averaging cost function.

Usage
-----
First generate the ramp WAV::

    python3 generate_msk144.py --ramp --ramp-mode width \\
        --ramp-width-min-ms 30 --ramp-width-max-ms 200 --ramp-width-step-ms 20 \\
        --ramp-fixed-snr-db 6 --output-dir /tmp

then run::

    python3 tools/test_spd_width_ramp.py \\
        --audio-wav /tmp/rampw_<timestamp>_<n>sig_<min>to<max>ms_<snr>dB_audio.wav

Output is a table; the program returns 0 always (no pass/fail
assertion — interpretation is up to the operator).
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

# Inject the map144_app package on sys.path so we can call SPD directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import msk144_spd_decode
from map144_app.detection import _apply_audio_bpf, _TARGET_FC_HZ


# ── Defaults ───────────────────────────────────────────────────────────────────
SR_AUDIO     = 12000          # WSJT-X / SPD canonical audio rate
SLICE_LEN_S  = 1.792          # SPD analysis buffer length (~1.7 s, matches
                              # the ring-buffer slice MAP144 hands to SPD)
PRE_BURST_S  = 0.300          # leading noise window before the ping anchor


# ── Slice + baseband conversion ────────────────────────────────────────────────

def _slice_and_baseband(audio_real: np.ndarray, t_anchor_s: float,
                       fs: int = SR_AUDIO) -> np.ndarray:
    """Return complex baseband (carrier at 0 Hz) for a 1.7-s window around the
    ping anchor.  Replicates the SPD pre-processing in detection.py:

        Hilbert → shift down 1500 Hz → audio BPF.
    """
    pre_n   = int(round(PRE_BURST_S * fs))
    slice_n = int(round(SLICE_LEN_S * fs))
    anchor_idx = int(round(t_anchor_s * fs))
    start = max(0, anchor_idx - pre_n)
    end   = min(len(audio_real), start + slice_n)
    if end - start < slice_n:
        # Zero-pad at the tail if the ping is too close to end-of-file
        pad = np.zeros(slice_n - (end - start), dtype=audio_real.dtype)
        seg = np.concatenate([audio_real[start:end], pad])
    else:
        seg = audio_real[start:end]

    # Hilbert → complex analytic signal at the audio rate
    analytic = hilbert(seg).astype(np.complex128)
    # Shift the 1500 Hz carrier down to DC
    t = np.arange(len(analytic), dtype=np.float64) / fs
    iq_bb = (analytic * np.exp(-2j * np.pi * _TARGET_FC_HZ * t)).astype(np.complex64)
    # MAP144's audio BPF (300–2700 Hz at audio rate, applied at baseband)
    iq_bbf = _apply_audio_bpf(iq_bb)
    return iq_bbf


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--audio-wav', type=Path, required=True,
                    help='Width-ramp audio WAV (12 kHz mono real) from '
                         'generate_msk144.py --ramp --ramp-mode width.')
    ap.add_argument('--truth-json', type=Path, default=None,
                    help='Truth JSON sidecar.  Default: <audio>.replace("_audio.wav","_truth.json").')
    args = ap.parse_args(argv)

    if args.truth_json is None:
        args.truth_json = Path(str(args.audio_wav).replace('_audio.wav', '_truth.json'))

    # ── Load WAV ───────────────────────────────────────────────────────────────
    with wave.open(str(args.audio_wav), 'rb') as wf:
        if wf.getframerate() != SR_AUDIO:
            print(f"warning: expected {SR_AUDIO} Hz, got {wf.getframerate()}",
                  file=sys.stderr)
        nfr = wf.getnframes()
        audio = (np.frombuffer(wf.readframes(nfr), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
    print(f"WAV : {args.audio_wav.name}  ({nfr/SR_AUDIO:.2f} s)")

    # ── Load truth ─────────────────────────────────────────────────────────────
    truth = json.loads(args.truth_json.read_text())
    pings = truth.get('pings', [])
    print(f"Truth pings: {len(pings)}")

    # ── Predicted navg from FWHM (rule of thumb: pings shorter than one
    #    frame → 1, between 1 and 2 frames → 2, longer → 3) ────────────────
    def predict_navg(fwhm_ms: float) -> int:
        if fwhm_ms < 100:   return 1
        if fwhm_ms < 200:   return 2
        return 3

    # ── Run SPD on each ping ───────────────────────────────────────────────────
    print()
    print("=" * 96)
    print(f"{'#':>2}  {'τ_ms':>5}  {'FWHM_ms':>8}  {'predicted':>10}  "
          f"{'SPD navg':>9}  {'xmax':>6}  {'fest_Hz':>8}  {'decoded msg':<30}")
    print("=" * 96)

    counts = {'1': 0, '2': 0, '3': 0, 'no_decode': 0}
    for i, p in enumerate(pings):
        t_anchor = float(p['t_offset_s'])
        width_ms = int(p.get('width_ms', 0))
        fwhm_ms  = float(p.get('fwhm_ms', 0.0))
        predicted = predict_navg(fwhm_ms)

        iq_bb = _slice_and_baseband(audio, t_anchor)
        msg, snr, navg, fest, xmax = msk144_spd_decode(complex_baseband=iq_bb)
        if msg:
            counts[str(navg)] = counts.get(str(navg), 0) + 1
            msg_disp = msg
        else:
            counts['no_decode'] += 1
            msg_disp = '(no decode)'

        print(f"{i:>2}  {width_ms:>5}  {fwhm_ms:>6.1f}    {predicted:>10}  "
              f"{navg if msg else '—':>9}  {xmax:>6.2f}  {fest:>+8.1f}  {msg_disp:<30}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary across all pings:")
    print(f"  navg=1     : {counts.get('1', 0):>3}")
    print(f"  navg=2     : {counts.get('2', 0):>3}")
    print(f"  navg=3     : {counts.get('3', 0):>3}")
    print(f"  no decode  : {counts.get('no_decode', 0):>3}")

    if counts.get('1', 0) >= max(2, len(pings) - 1) and counts.get('2', 0) <= 1:
        print()
        print("VERDICT: SPD reports navg=1 for ~all pings regardless of width.")
        print("         This is consistent with a bug in the frame-averaging cost")
        print("         function or navmask enumeration.  Real MS pings on the")
        print("         operator's fast graph are >150 ms — navg=2/3 should win")
        print("         on most longer-width inputs but doesn't.")
    elif counts.get('2', 0) + counts.get('3', 0) >= len(pings) // 2:
        print()
        print("VERDICT: SPD scales navg with width as expected.  No bug visible.")
    else:
        print()
        print("VERDICT: Mixed results — needs deeper look at xmax cost function.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
