#!/home/jeff/ham/map144/.venv/bin/python3
"""Standalone test of the improved decode pipeline against sparse50.wav.

Reads MSK144/sim/sparse50.wav and MSK144/sim/sparse50.json, then for each
placement runs the exact extract-and-decode path that detection.py uses:
  mix to baseband → decimate 48→12 kHz → BPF → SPD (complex IQ) → jt9 fallback

Reports per-SNR decode rates, comparing SPD vs jt9 contributions.

Usage:
    python test_sparse50.py [--no-jt9]
"""

import argparse
import json
import sys
import wave
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).parent))

from map144_app.detection import (
    _DECIMATE_FACTOR, _DECODE_RATE, _TARGET_FC_HZ,
    _apply_audio_bpf, JT9_BASE_ARGS,
)
from map144_app.msk144_spd import msk144_spd_decode


def load_wav_iq(path: str) -> tuple[np.ndarray, int]:
    """Load stereo WAV as complex IQ (L=I, R=Q). Returns (complex64 array, sample_rate)."""
    import scipy.io.wavfile as _sio
    fs, data = _sio.read(path)

    if data.ndim == 2 and data.shape[1] >= 2:
        i_ch = data[:, 0].astype(np.float32)
        q_ch = data[:, 1].astype(np.float32)
        iq = (i_ch + 1j * q_ch).astype(np.complex64)
    elif data.ndim == 1:
        iq = data.astype(np.complex64)
    else:
        raise ValueError(f"Unexpected shape {data.shape}")

    return iq, int(fs)


def decode_segment(iq_full: np.ndarray, fs: int, fc_hz: float,
                   t_start_s: float, pre_s: float = 0.5, post_s: float = 1.2,
                   pad_s: float = 0.1, use_jt9: bool = True) -> tuple[str | None, str]:
    """Extract a segment around a known ping and run the improved decode pipeline.

    Returns (decoded_message or None, decoder_name).
    decoder_name is one of: 'spd', 'jt9', 'none'.
    """
    import subprocess, tempfile, wave as _wave

    pre_n  = int(pre_s  * fs)
    post_n = int(post_s * fs)
    pad_n  = int(pad_s  * fs)

    # Detection is expected at t_start_s + some fraction of the ping duration.
    # Use the start of the ping as the "detect_sample" reference.
    detect_n = int(t_start_s * fs)
    start_n  = max(0, detect_n - pre_n)
    end_n    = min(len(iq_full), detect_n + post_n)

    seg = iq_full[start_n:end_n]

    # Zero-pad to expected length
    pad = np.zeros(pad_n, dtype=np.complex64)
    seg = np.concatenate([pad, seg, pad])

    # Mix carrier from fc_hz to _TARGET_FC_HZ (1500 Hz)
    shift_hz = fc_hz - _TARGET_FC_HZ
    t_mix = np.arange(len(seg), dtype=np.float64) / fs
    mix   = np.exp(-2j * np.pi * shift_hz * t_mix).astype(np.complex64)
    seg_mixed = seg * mix

    # Decimate I and Q to 12 kHz
    i_dec = decimate(np.real(seg_mixed).astype(np.float64), _DECIMATE_FACTOR, zero_phase=False)
    q_dec = decimate(np.imag(seg_mixed).astype(np.float64), _DECIMATE_FACTOR, zero_phase=False)
    audio = i_dec.astype(np.float32)

    # SPD: complex IQ + audio BPF
    n12   = len(i_dec)
    t12   = np.arange(n12, dtype=np.float64) / _DECODE_RATE
    iq12  = ((i_dec + 1j * q_dec) *
             np.exp(-2j * np.pi * _TARGET_FC_HZ * t12)).astype(np.complex64)
    iq12_bpf = _apply_audio_bpf(iq12)

    spd_msg, spd_snr, spd_navg, spd_fest, spd_xmax = msk144_spd_decode(
        complex_baseband=iq12_bpf)

    if spd_msg is not None:
        return spd_msg.strip(), 'spd'

    if not use_jt9:
        return None, 'none'

    # jt9 fallback — normalize, write WAV, call jt9
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    else:
        return None, 'none'

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with _wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_DECODE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        result = subprocess.run(
            JT9_BASE_ARGS + [tmp_path],
            capture_output=True, text=True, timeout=20.0,
        )
        for line in result.stdout.splitlines():
            s = line.strip()
            if s and not s.startswith('<') and not s.startswith('EOF'):
                tokens = s.split()
                msg = ' '.join(tokens[5:]) if len(tokens) >= 6 else s
                return msg.strip(), 'jt9'
    except Exception:
        pass
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return None, 'none'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav',      default='MSK144/sim/sparse50.wav')
    parser.add_argument('--manifest', default='MSK144/sim/sparse50.json')
    parser.add_argument('--no-jt9',   action='store_true',
                        help='Skip jt9 fallback (SPD only)')
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()

    print(f"Loading {args.wav} ...")
    iq, fs = load_wav_iq(args.wav)
    print(f"  {len(iq)/fs:.1f} s  at {fs} Hz  complex IQ")

    with open(args.manifest) as f:
        manifest = json.load(f)
    placements = manifest['placements']
    print(f"  {len(placements)} placements in manifest\n")

    # fc_hz in the IQ passband → carrier is at center_hz + 1500 Hz
    # (center_hz is the offset from DC; the MSK tone pair straddles
    # center_hz ± 500 Hz, so the carrier is at center_hz + 0 Hz...
    # wait, need to check the generate_msk144 convention).
    #
    # generate_msk144 places center_hz as the signal's centre frequency
    # in the 48 kHz IQ passband (relative to DC at 0 Hz).  In USB radio
    # the carrier is 1500 Hz above the dial, and the two MSK144 tones are
    # at carrier ± 500 Hz.  The detection.py convention: fc_hz is the
    # carrier position in the IQ passband; mix shift_hz = fc_hz - 1500.
    # After mixing the carrier lands at 1500 Hz in the 12 kHz audio.
    #
    # In sparse50.json, center_hz is the frequency of the MSK144 signal
    # centre in the IQ passband.  The MSK carrier is at that same centre.
    # So fc_hz = center_hz for calling extract_and_decode logic.

    by_snr   = defaultdict(list)  # snr_db → list of (decoded_bool, decoder)
    n_total  = 0
    n_spd    = 0
    n_jt9    = 0
    n_miss   = 0

    for i, p in enumerate(placements):
        msg_expected = p['msg'].strip()
        fc_hz        = float(p['center_hz'])   # carrier in IQ passband
        t_start      = float(p['delay_s'])
        snr_db       = int(p.get('snr_db', 0))

        decoded_msg, decoder = decode_segment(
            iq, fs, fc_hz, t_start, use_jt9=not args.no_jt9)

        # Consider it a decode if the message content matches
        # (strip whitespace; both should be 10-13 char codes)
        hit = (decoded_msg is not None and
               decoded_msg.strip().upper() == msg_expected.upper())

        by_snr[snr_db].append((hit, decoder))
        n_total += 1
        if hit:
            if decoder == 'spd':  n_spd += 1
            else:                 n_jt9 += 1
        else:
            n_miss += 1

        status = 'OK' if hit else 'MISS'
        if args.verbose or not hit:
            decoded_str = repr(decoded_msg) if decoded_msg else 'None'
            print(f"  [{i+1:3d}] snr={snr_db:+3d} dB  fc={fc_hz:+7.0f} Hz  "
                  f"{status:4s}  [{decoder:3s}]  got={decoded_str}  "
                  f"expected={msg_expected!r}")
        else:
            print(f"  [{i+1:3d}] snr={snr_db:+3d} dB  fc={fc_hz:+7.0f} Hz  "
                  f"{status}  [{decoder}]", flush=True)

    print()
    print("=" * 60)
    print(f"TOTAL: {n_spd+n_jt9}/{n_total} decoded  "
          f"(SPD={n_spd} jt9={n_jt9} miss={n_miss})")
    print()
    print(f"{'SNR':>6}  {'Trials':>6}  {'Decoded':>7}  {'Rate':>6}  {'SPD':>4}  {'jt9':>4}")
    for snr in sorted(by_snr.keys()):
        results = by_snr[snr]
        n       = len(results)
        hits    = sum(1 for h, _ in results if h)
        spd_    = sum(1 for h, d in results if h and d == 'spd')
        jt9_    = sum(1 for h, d in results if h and d == 'jt9')
        print(f"  {snr:+3d} dB  {n:6d}  {hits:7d}  {hits/n*100:5.0f}%  {spd_:4d}  {jt9_:4d}")


if __name__ == '__main__':
    main()
