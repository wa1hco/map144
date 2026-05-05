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
"""NR0V wideband noise blanker — Stage 2 backend.

Wraps Warren Pratt's WDSP ``nob.c`` (the ANB / "automatic noise blanker")
via ctypes.  The algorithm is a time-domain state machine with a
ring-buffer delay line, so the blanker can zero samples *before* the
impulse reaches the output — impossible to do with a non-causal
numpy-vectorised approach.

For dual-polarisation inputs, H and V are each processed by an
independent ANB instance.  Joint detection is a Stage-2 tuning
experiment (see docs/noise_blanker_plan.md) and not enabled here; the
two instances see the same threshold and timing parameters so their
outputs stay statistically comparable.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .noise_blanker import (
    Blanker, BlankerResult,
    NB_FFT_SIZE, NB_SPEC_AVG_TC,
)


# ── Parameter defaults ────────────────────────────────────────────────────────
# Chosen for 192 kHz operation on MSK144.  All time parameters are in seconds,
# so WDSP rescales to sample counts automatically.  Starting point only; the
# Stage-2 sweep in docs/noise_blanker_plan.md will tune these against a corpus.
NR0V_TAU_DEFAULT        = 1.0e-4    # 100 µs transition (signal ↔ zero)
NR0V_HANGTIME_DEFAULT   = 1.0e-4    # 100 µs hold after trailing edge
NR0V_ADVTIME_DEFAULT    = 1.0e-4    # 100 µs pre-blank (possible due to delay line)
NR0V_BACKTAU_DEFAULT    = 1.0e-2    # 10 ms averaging time constant
NR0V_THRESHOLD_DEFAULT  = 30.0      # trip when mag > threshold × running avg


# ── Shared-library loader ─────────────────────────────────────────────────────

_LIB_PATH = Path(__file__).resolve().parent.parent / "vendor" / "wdsp" / "libnob.so"
_lib: Optional[ctypes.CDLL] = None


def _load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"libnob.so not found at {_LIB_PATH}. "
            f"Build it with: make -C {_LIB_PATH.parent}"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    # ANB is an opaque pointer on the Python side — we only pass it to C.
    ANB_P = ctypes.c_void_p

    # ANB create_anb(int run, int buffsize, double *in, double *out,
    #                double samplerate, double tau, double hangtime,
    #                double advtime, double backtau, double threshold);
    lib.create_anb.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]
    lib.create_anb.restype = ANB_P

    lib.destroy_anb.argtypes = [ANB_P]
    lib.destroy_anb.restype  = None

    lib.flush_anb.argtypes = [ANB_P]
    lib.flush_anb.restype  = None

    # MAP144 shim: update buffers and size in place, then run xanb.
    lib.xanb_buf.argtypes = [
        ANB_P,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.xanb_buf.restype = None

    # Threshold setter — mutates threshold in place, no state reset.
    lib.pSetRCVRANBThreshold.argtypes = [ANB_P, ctypes.c_double]
    lib.pSetRCVRANBThreshold.restype  = None

    _lib = lib
    return lib


# ── Single-channel wrapper ────────────────────────────────────────────────────

class _ANBInstance:
    """Owns one WDSP ANB instance and its I/O scratch buffers."""

    def __init__(self, samplerate: float, chunk_hint: int = 4096):
        lib = _load_library()
        self._lib = lib
        # Buffers hold interleaved real/imag doubles: length = 2 * chunk_hint.
        self._bufsize_samples = chunk_hint
        self._in_buf  = (ctypes.c_double * (2 * chunk_hint))()
        self._out_buf = (ctypes.c_double * (2 * chunk_hint))()
        self._anb = lib.create_anb(
            1,                              # run
            chunk_hint,                     # buffsize (updated per-call via shim)
            ctypes.cast(self._in_buf,  ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(self._out_buf, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(samplerate),
            ctypes.c_double(NR0V_TAU_DEFAULT),
            ctypes.c_double(NR0V_HANGTIME_DEFAULT),
            ctypes.c_double(NR0V_ADVTIME_DEFAULT),
            ctypes.c_double(NR0V_BACKTAU_DEFAULT),
            ctypes.c_double(NR0V_THRESHOLD_DEFAULT),
        )
        if not self._anb:
            raise RuntimeError("create_anb returned NULL")

    def _grow(self, n: int) -> None:
        """Enlarge scratch buffers if a larger chunk arrives."""
        if n <= self._bufsize_samples:
            return
        self._bufsize_samples = n
        self._in_buf  = (ctypes.c_double * (2 * n))()
        self._out_buf = (ctypes.c_double * (2 * n))()

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Blank one chunk of complex IQ.  Returns a new complex64 array."""
        n = len(iq)
        self._grow(n)
        # Interleave complex → doubles into the C-side input buffer.
        interleaved_in = np.empty(2 * n, dtype=np.float64)
        interleaved_in[0::2] = iq.real
        interleaved_in[1::2] = iq.imag
        ctypes.memmove(self._in_buf, interleaved_in.ctypes.data, interleaved_in.nbytes)

        self._lib.xanb_buf(
            self._anb,
            ctypes.cast(self._in_buf,  ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(self._out_buf, ctypes.POINTER(ctypes.c_double)),
            n,
        )

        interleaved_out = np.frombuffer(self._out_buf, dtype=np.float64, count=2 * n)
        out_c = np.empty(n, dtype=np.complex64)
        out_c.real = interleaved_out[0::2]
        out_c.imag = interleaved_out[1::2]
        return out_c

    def flush(self) -> None:
        """Reset WDSP state (drain ring buffer, re-seed averages)."""
        self._lib.flush_anb(self._anb)

    def set_threshold(self, thresh: float) -> None:
        """Change the trip threshold without resetting state."""
        self._lib.pSetRCVRANBThreshold(self._anb, ctypes.c_double(thresh))

    def __del__(self):
        try:
            if getattr(self, "_anb", None):
                self._lib.destroy_anb(self._anb)
                self._anb = None
        except Exception:
            pass


# ── Blanker backend ───────────────────────────────────────────────────────────

class NR0VWidebandBlanker(Blanker):
    """WDSP-style time-domain noise blanker (NR0V / Warren Pratt).

    Two independent ANB instances in dual-pol mode (H and V).
    """

    name = "NR0V-Wideband"

    def __init__(self):
        self._rate: Optional[float] = None
        self._anb_h: Optional[_ANBInstance] = None
        self._anb_v: Optional[_ANBInstance] = None
        # Per-channel thresholds so H and V can be tuned independently.
        self._last_threshold_h: Optional[float] = None
        self._last_threshold_v: Optional[float] = None
        # Tail of previous chunk's input magnitude, length = ring-buffer delay.
        # Used to align the pre-blanker display trace with the post-blanker
        # trace, which is delayed by (tau + advtime) samples internally.
        self._prev_mag_h_tail: Optional[np.ndarray] = None
        self._prev_mag_v_tail: Optional[np.ndarray] = None

    def _ensure(self, rate: float, dual_pol: bool) -> None:
        # Re-create if sample rate changed (buffsize changes are handled
        # without reset inside _ANBInstance).
        if self._rate != rate:
            self._anb_h = _ANBInstance(rate)
            self._anb_v = _ANBInstance(rate) if dual_pol else None
            self._rate  = rate
            self._last_threshold_h = None
            self._last_threshold_v = None
            self._prev_mag_h_tail  = None
            self._prev_mag_v_tail  = None
        elif dual_pol and self._anb_v is None:
            self._anb_v = _ANBInstance(rate)
            self._last_threshold_v = None
        elif not dual_pol and self._anb_v is not None:
            self._anb_v = None
            self._last_threshold_v = None
            self._prev_mag_v_tail  = None

    @staticmethod
    def _delay_samples(rate: float) -> int:
        """Ring-buffer delay in samples: tau + advtime at the current rate.

        Must match WDSP nob.c's ``a->trans_count + a->adv_count`` exactly,
        which floors each term separately before summing.  Computing
        ``int((tau + advtime) * rate)`` can disagree by one sample at
        lower sample rates (e.g. 48 kHz), which causes the mask-derivation
        comparison to compare two *independent* noise samples and produce
        a ~3.85% false-alarm rate (P(A<0.2B) for iid Rayleigh).
        """
        trans = max(int(NR0V_TAU_DEFAULT     * rate), 2)  # WDSP clamps to ≥ 2
        adv   =     int(NR0V_ADVTIME_DEFAULT * rate)
        return trans + adv

    def _align_mag(self, in_mag: np.ndarray, prev_tail: Optional[np.ndarray],
                   delay: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (aligned_mag_for_display, new_tail).

        aligned_mag[t] = |input| at the same time index the blanker's output
        currently represents — i.e. in_mag shifted backward by `delay`, with
        the `delay` leading samples drawn from the previous chunk's tail.
        The new_tail is the last `delay` samples of this chunk for use on
        the next call.
        """
        n = len(in_mag)
        if delay <= 0 or delay >= n:
            return in_mag, in_mag[-delay:].copy() if delay > 0 and delay <= n else None
        prev = prev_tail if prev_tail is not None and len(prev_tail) == delay \
               else np.zeros(delay, dtype=np.float32)
        aligned = np.empty(n, dtype=np.float32)
        aligned[:delay]  = prev
        aligned[delay:]  = in_mag[:-delay]
        new_tail         = in_mag[-delay:].copy()
        return aligned, new_tail

    def process(self, engine, raw, raw_v, dual_pol):
        rate  = float(engine.sample_rate)
        run_v = dual_pol and raw_v is not None
        self._ensure(rate, run_v)

        # Per-channel K → NR0V threshold.  NR0V compares magnitude to
        # running-average magnitude, so K maps to threshold directly.
        k_h = float(getattr(engine, 'nb_factor',   NR0V_THRESHOLD_DEFAULT))
        k_v = float(getattr(engine, 'nb_factor_v', k_h))
        if self._last_threshold_h != k_h and self._anb_h is not None:
            self._anb_h.set_threshold(k_h)
            self._last_threshold_h = k_h
        if run_v and self._last_threshold_v != k_v and self._anb_v is not None:
            self._anb_v.set_threshold(k_v)
            self._last_threshold_v = k_v

        cleaned_h = self._anb_h.process(raw)
        cleaned_v = self._anb_v.process(raw_v) if run_v else None

        # Align pre-blanker magnitudes to the WDSP output timebase (ring
        # buffer delays output by tau + advtime samples) so display traces
        # line up and the mask-derivation comparison is at matching times.
        delay = self._delay_samples(rate)
        in_mag_h = np.abs(raw).astype(np.float32)
        aligned_mag_h, self._prev_mag_h_tail = self._align_mag(
            in_mag_h, self._prev_mag_h_tail, delay)
        aligned_mag_v = None
        if run_v:
            in_mag_v = np.abs(raw_v).astype(np.float32)
            aligned_mag_v, self._prev_mag_v_tail = self._align_mag(
                in_mag_v, self._prev_mag_v_tail, delay)

        # Per-sample mask per channel.
        out_mag_h = np.abs(cleaned_h).astype(np.float32)
        mask_h    = np.where(out_mag_h < 0.2 * aligned_mag_h, 1.0, 0.0).astype(np.float32)
        mask_v    = None
        if run_v:
            out_mag_v = np.abs(cleaned_v).astype(np.float32)
            mask_v    = np.where(out_mag_v < 0.2 * aligned_mag_v, 1.0, 0.0).astype(np.float32)

        # Noise-floor state: one per channel.  Linrad's _nb_wideband_floor
        # is FFT-based state NR0V cannot produce; clear it so the display
        # falls through to the time-domain _nb_floor estimates here.
        self._update_floor(engine, '',   aligned_mag_h, mask_h, rate)
        engine._nb_wideband_floor = None
        if run_v:
            self._update_floor(engine, '_v', aligned_mag_v, mask_v, rate)
            engine._nb_wideband_floor_v = None

        # Status-bar counters: include both channels so the displayed
        # "% blanked" responds to K V changes in dual-pol mode.
        engine._nb_total_count   += len(raw)
        engine._nb_blanked_count += int(mask_h.sum())
        if run_v:
            engine._nb_total_count   += len(raw_v)
            engine._nb_blanked_count += int(mask_v.sum())

        # Display-only spectrum state.  NR0V is a time-domain blanker and
        # doesn't need FFT power for detection, but the blanker spectrum
        # plot consumes _nb_last_P / _nb_spec_avg / _nb_last_blanked_P
        # and would freeze otherwise.  One FFT per chunk per channel —
        # trivial at NB_FFT_SIZE=256.
        self._update_display_spectrum(engine, raw,   cleaned_h, mask_h, '')
        if run_v:
            self._update_display_spectrum(engine, raw_v, cleaned_v, mask_v, '_v')

        return BlankerResult(
            cleaned_h    = cleaned_h,
            cleaned_v    = cleaned_v,
            blank_mask   = mask_h,
            mag_h        = aligned_mag_h,
            blank_mask_v = mask_v,
            mag_v        = aligned_mag_v,
        )

    @staticmethod
    def _update_display_spectrum(engine, raw, cleaned, mask, suffix):
        """Populate _nb_last_P / _nb_spec_avg / _nb_last_blanked_P so the
        Blanker Spectrum panel has something to draw while NR0V is active."""
        n = len(raw)
        if n < NB_FFT_SIZE:
            return
        # Use the last NB_FFT_SIZE samples of the chunk — most recent block.
        X = np.fft.fft(raw[-NB_FFT_SIZE:])
        P = np.abs(X).astype(np.float64) ** 2
        setattr(engine, f'_nb_last_P{suffix}', P)

        # Running per-bin average.  Seed from this block if empty; else
        # EMA with the same time constant Linrad uses (NB_SPEC_AVG_TC s).
        spec_avg = getattr(engine, f'_nb_spec_avg{suffix}', None)
        alpha    = 1.0 - float(np.exp(-NB_FFT_SIZE / (NB_SPEC_AVG_TC * engine.sample_rate)))
        if spec_avg is None:
            med = max(float(np.median(P)), 1e-30)
            spec_avg = np.full(NB_FFT_SIZE, med, dtype=np.float64)
        else:
            spec_avg = (1.0 - alpha) * spec_avg + alpha * P
        setattr(engine, f'_nb_spec_avg{suffix}', spec_avg)

        # Freeze a "blanked spectrum" snapshot whenever this chunk blanked
        # anything, mirroring Linrad's display contract.
        if mask is not None and mask.any():
            setattr(engine, f'_nb_last_blanked_P{suffix}', P)

    @staticmethod
    def _update_floor(engine, suffix, aligned_mag, mask, rate):
        """EMA-update engine._nb_floor{suffix} from non-blanked magnitudes."""
        not_blanked = mask == 0.0
        if not not_blanked.any():
            return
        n_ok = int(not_blanked.sum())
        rms  = float(np.sqrt(np.mean(aligned_mag[not_blanked] ** 2)))
        prev = getattr(engine, f'_nb_floor{suffix}', None)
        if prev is None or prev <= 0.0:
            setattr(engine, f'_nb_floor{suffix}', max(rms, 1e-9))
        else:
            alpha = 1.0 - float(np.exp(-n_ok / (2.0 * rate)))
            setattr(engine, f'_nb_floor{suffix}',
                    max((1.0 - alpha) * float(prev) + alpha * rms, 1e-9))

    def reset(self, engine) -> None:
        if self._anb_h is not None:
            self._anb_h.flush()
        if self._anb_v is not None:
            self._anb_v.flush()
        self._prev_mag_h_tail = None
        self._prev_mag_v_tail = None
        self._last_threshold  = None
