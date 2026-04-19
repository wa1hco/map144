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
"""Capture I/O — save and load 15-second IQ captures with JSON metadata.

A capture is a pair of files in MSK144/captures/:
  YYYYMMDD_HHMMSSZ_capture_{source}_{center_kHz}kHz.wav   stereo float32, L=I R=Q
  YYYYMMDD_HHMMSSZ_capture_{source}_{center_kHz}kHz.json  metadata

The WAV amplitude is not peak-normalised so that SNR is preserved for analysis.

Two windows are capturable:
  'current'  — most recent 15 s (upper Fast Graph)
  'previous' — the 15 s before that (lower Fast Graph)

Both are extracted from the 30-second IQ ring buffer in Engine.
"""

import json
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CAPTURES_DIR = Path('MSK144') / 'captures'
_SAMPLE_RATE = 48000


# ── Save ──────────────────────────────────────────────────────────────────────

def collect_capture_iq(engine, window: str = 'current') -> np.ndarray:
    """Extract 15 s of IQ from the engine ring buffer.

    window  'current'  — [abs_sample - 15s .. abs_sample]
            'previous' — [abs_sample - 30s .. abs_sample - 15s]

    Returns complex64 array of length sample_rate*15 (may be shorter if the
    ring has not yet been filled to 30 s).
    """
    from .detection import _read_ring

    n_win = int(15 * engine.sample_rate)
    abs_now = engine._iq_abs_sample
    ring_pos = engine._iq_ring_pos

    if window == 'current':
        start_abs = abs_now - n_win
    else:
        start_abs = abs_now - 2 * n_win

    return _read_ring(engine._iq_ring, ring_pos, abs_now, start_abs, n_win)


def save_capture(iq: np.ndarray,
                 center_freq_mhz: float,
                 source: str,
                 nb_factor: float,
                 nb_enabled: bool,
                 sample_rate: int = _SAMPLE_RATE,
                 captures_dir: Path = CAPTURES_DIR,
                 is_test_wav: bool = False,
                 source_wav_path: str = None) -> tuple[Path, Path]:
    """Write IQ to a stereo float32 WAV and a companion JSON metadata file.

    Returns (wav_path, json_path).
    """
    captures_dir = Path(captures_dir)
    captures_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime('%Y%m%d_%H%M%SZ')
    center_khz = int(round(center_freq_mhz * 1000))
    # Sanitize source label for use in filename (spaces/colons → underscore)
    import re as _re
    source_safe = _re.sub(r'[^A-Za-z0-9]+', '_', source).strip('_')
    stem = f"{ts_str}_capture_{source_safe}_{center_khz}kHz"
    wav_path  = captures_dir / f"{stem}.wav"
    json_path = captures_dir / f"{stem}.json"

    # Write float32 WAV:
    #   single-pol  — stereo  (L=I, R=Q)
    #   dual-pol    — 4-ch    (H_I, H_Q, V_I, V_Q)
    # Use raw struct packing — Python's wave module only supports int PCM.
    dual = iq.ndim == 2 and iq.shape[1] == 2
    n = iq.shape[0]
    if dual:
        n_channels = 4
        interleaved = np.empty(n * 4, dtype=np.float32)
        interleaved[0::4] = np.real(iq[:, 0])
        interleaved[1::4] = np.imag(iq[:, 0])
        interleaved[2::4] = np.real(iq[:, 1])
        interleaved[3::4] = np.imag(iq[:, 1])
    else:
        n_channels = 2
        interleaved = np.empty(n * 2, dtype=np.float32)
        interleaved[0::2] = np.real(iq).astype(np.float32)
        interleaved[1::2] = np.imag(iq).astype(np.float32)

    data_bytes = interleaved.tobytes()
    sampwidth   = 4          # float32
    fmt_code    = 3          # WAVE_FORMAT_IEEE_FLOAT

    with open(wav_path, 'wb') as f:
        # RIFF header
        data_size   = len(data_bytes)
        chunk_size  = 36 + data_size
        f.write(b'RIFF')
        f.write(struct.pack('<I', chunk_size))
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))               # chunk size
        f.write(struct.pack('<H', fmt_code))          # IEEE float
        f.write(struct.pack('<H', n_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * n_channels * sampwidth))  # byte rate
        f.write(struct.pack('<H', n_channels * sampwidth))                # block align
        f.write(struct.pack('<H', sampwidth * 8))                         # bits/sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(data_bytes)

    # Write JSON metadata
    meta = {
        'format_version':  1,
        'timestamp_utc':   ts.isoformat(),
        'center_freq_mhz': center_freq_mhz,
        'sample_rate':     sample_rate,
        'duration_secs':   n / sample_rate,
        'n_samples':       n,
        'source':          source,
        'nb_factor':       nb_factor,
        'nb_enabled':      nb_enabled,
        'is_test_wav':     is_test_wav,
        'source_wav_path': source_wav_path,
        'dual_pol':        dual,
    }
    json_path.write_text(json.dumps(meta, indent=2))

    return wav_path, json_path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_capture(wav_path: Path) -> tuple[np.ndarray, dict]:
    """Load a capture WAV and its companion JSON metadata.

    Returns (iq_complex64, meta_dict).
    Raises FileNotFoundError or ValueError on bad format.
    """
    wav_path  = Path(wav_path)
    json_path = wav_path.with_suffix('.json')

    # Read stereo float32 WAV
    with open(wav_path, 'rb') as f:
        data = f.read()

    # Parse RIFF/WAVE manually (same writer, so format is known)
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError(f"Not a RIFF/WAVE file: {wav_path}")

    pos = 12
    fmt_code = n_channels = sample_rate = sampwidth = None
    audio_bytes = None
    while pos < len(data) - 8:
        chunk_id   = data[pos:pos+4]
        chunk_size = struct.unpack_from('<I', data, pos+4)[0]
        pos += 8
        if chunk_id == b'fmt ':
            fmt_code   = struct.unpack_from('<H', data, pos)[0]
            n_channels = struct.unpack_from('<H', data, pos+2)[0]
            sample_rate = struct.unpack_from('<I', data, pos+4)[0]
            sampwidth  = struct.unpack_from('<H', data, pos+14)[0] // 8
        elif chunk_id == b'data':
            audio_bytes = data[pos:pos+chunk_size]
        pos += chunk_size

    if audio_bytes is None or fmt_code != 3 or n_channels not in (2, 4):
        raise ValueError(f"Expected stereo or 4-ch float32 capture WAV: {wav_path}")

    interleaved = np.frombuffer(audio_bytes, dtype=np.float32)
    if n_channels == 4:
        h = (interleaved[0::4] + 1j * interleaved[1::4]).astype(np.complex64)
        v = (interleaved[2::4] + 1j * interleaved[3::4]).astype(np.complex64)
        iq = np.stack([h, v], axis=1)   # shape (N, 2)
    else:
        iq = (interleaved[0::2] + 1j * interleaved[1::2]).astype(np.complex64)

    meta = {}
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            pass

    # Fill defaults for any missing metadata keys
    meta.setdefault('center_freq_mhz', 0.0)
    meta.setdefault('sample_rate',     sample_rate or _SAMPLE_RATE)
    meta.setdefault('source',          'unknown')
    meta.setdefault('nb_factor',       6.0)
    meta.setdefault('nb_enabled',      True)
    meta.setdefault('is_test_wav',     False)
    meta.setdefault('source_wav_path', None)
    meta.setdefault('dual_pol',        n_channels == 4)

    return iq, meta


# ── Browse ────────────────────────────────────────────────────────────────────

def browse_captures(parent_widget=None,
                    captures_dir: Path = CAPTURES_DIR) -> Path | None:
    """Open a file dialog pointing at captures_dir.  Returns selected path or None."""
    from PyQt5 import QtWidgets
    captures_dir = Path(captures_dir).resolve()
    captures_dir.mkdir(parents=True, exist_ok=True)
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent_widget,
        "Open IQ Capture",
        str(captures_dir),
        "IQ Capture WAV (*.wav);;All Files (*)",
    )
    return Path(path) if path else None
