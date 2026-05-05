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
"""Noise-blanker backends — selectable impulse-blanking strategies.

Architecture
------------
The blanker runs on complex IQ at the SDR sample rate (192 kHz for B210 /
SDRangel) before the 4:1 polyphase decimator takes us to 48 kHz.  Each
backend implements a narrow interface (``Blanker.process``) that returns
cleaned H and V streams, a per-sample blanking mask, and the pre-blanker
magnitude buffer the display path already consumes.

Stage 1 scope (no behavior change)
----------------------------------
* ``BypassBlanker`` — passthrough; preserved as an A/B reference.
* ``LinradBlanker`` — current production blanker, moved here verbatim.

Future stages (planned)
-----------------------
* ``NR0VWidebandBlanker`` — WDSP-style wideband impulse blanker.
* ``NR0VSpectralBlanker`` — WDSP-style LPC-based spectral blanker.

Interface contract
------------------
Each backend accepts the engine (``Processing`` host) so it can update
the existing ``_nb_*`` state attributes that ``displays.py`` and the
runtime reset paths depend on.  This keeps the Stage 1 change structural
only — display and reset paths do not move.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import convolve1d as _nd_convolve1d


# ── Constants (mirrored from processing.py) ──────────────────────────────────

NB_FACTOR          = 6.0        # amplitude threshold (hot bin power > factor² × avg)
NB_FFT_SIZE        = 256        # detection block size: 5.3 ms at 48 kHz
NB_BROADBAND_FRAC  = 0.30       # fraction of hot bins for broadband impulse
NB_SPEC_AVG_TC     = 2.0        # per-bin spectrum averaging time constant (s)
NB_WIDEBAND_AVGNUM = 200        # EMA length for wideband median floor
_NB_LOWLEVEL_ALPHA = 0.02       # EMA weight for clean-fraction tracker
_NB_TAPER_N        = 2          # Hann taper half-width in samples
NB_SHORT_IMPULSE_THRESH = 5     # block-blanking bypass threshold

_NB_TAPER_KERNEL = np.hanning(2 * _NB_TAPER_N + 1).astype(np.float32)
_NB_TAPER_KERNEL /= _NB_TAPER_KERNEL.max()


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class BlankerResult:
    """Output of one blanker invocation.

    Attributes are the minimal set consumed by ``process_iq_data`` after
    the blanker runs.  Display-only state (per-bin spectrum averages,
    blanked-sample counters, noise-floor estimates) is written directly
    onto the engine object so existing UI readers keep working.

    ``blank_mask_v`` is the V-channel per-sample mask when the backend
    runs independent H/V detection.  ``None`` means "apply the H mask to
    V" (legacy single-pipeline behavior); dual-pipeline backends must
    set it to a length-matched float32 array.
    """
    cleaned_h: np.ndarray                      # complex64 H after blanking
    cleaned_v: Optional[np.ndarray]            # complex64 V after blanking, or None
    blank_mask: np.ndarray                     # float32 per-sample H mask: 0.0 keep, 1.0 blank
    mag_h: np.ndarray                          # float32 |raw H| (pre-blanker, for display buffer)
    blank_mask_v: Optional[np.ndarray] = None  # float32 per-sample V mask; None = use H mask
    mag_v: Optional[np.ndarray] = None         # float32 |raw V| (pre-blanker, for display buffer)


# ── Base class ───────────────────────────────────────────────────────────────

class Blanker(ABC):
    """Narrow blanker interface.

    Backends may cache per-instance state; state that the UI reads (and
    the runtime reset path clears) lives on the engine for backward
    compatibility.  ``reset`` is called when the caller wants the next
    chunk to re-seed any warm-up state.
    """

    name: str = ""

    @abstractmethod
    def process(self,
                engine,
                raw: np.ndarray,
                raw_v: Optional[np.ndarray],
                dual_pol: bool) -> BlankerResult:
        ...

    def reset(self, engine) -> None:
        """Clear backend state.  Default is a no-op; subclasses override."""
        return


# ── Bypass ───────────────────────────────────────────────────────────────────

class BypassBlanker(Blanker):
    """Passthrough — no blanking, used as an A/B reference."""

    name = "Bypass"

    def process(self, engine, raw, raw_v, dual_pol):
        mag = np.abs(raw).astype(np.float32)
        engine._nb_total_count += len(raw)
        run_v   = dual_pol and raw_v is not None
        mask_v  = np.zeros(len(raw_v), dtype=np.float32) if run_v else None
        mag_v   = np.abs(raw_v).astype(np.float32)       if run_v else None
        return BlankerResult(
            cleaned_h    = raw.copy(),
            cleaned_v    = raw_v.copy() if run_v else None,
            blank_mask   = np.zeros(len(raw), dtype=np.float32),
            mag_h        = mag,
            blank_mask_v = mask_v,
            mag_v        = mag_v,
        )


# ── Linrad (current production blanker) ──────────────────────────────────────

class LinradBlanker(Blanker):
    """Wideband FFT blanker + time-domain amplitude blanker.

    Runs the full detection + blanking pipeline per channel when the
    source delivers dual-polarisation IQ: each of H and V has its own
    per-bin spectrum averages, block-RMS median, noise floor, and
    threshold (``nb_factor`` / ``nb_factor_v``).  The H and V blank masks
    are returned independently in ``BlankerResult``.

    In single-pol mode the V path is skipped and ``BlankerResult.
    blank_mask_v`` is None.
    """

    name = "Linrad"

    def process(self, engine, raw, raw_v, dual_pol):
        run_v = dual_pol and raw_v is not None

        blank_mask_h, mag_h = self._process_channel(
            engine, raw, nb_factor=float(getattr(engine, 'nb_factor', NB_FACTOR)),
            suffix='',
        )
        blank_mask_v = None
        mag_v        = None
        if run_v:
            blank_mask_v, mag_v = self._process_channel(
                engine, raw_v,
                nb_factor=float(getattr(engine, 'nb_factor_v',
                                        getattr(engine, 'nb_factor', NB_FACTOR))),
                suffix='_v',
            )

        # Soft-blank with independent Hann-tapered masks per channel.
        cleaned   = self._apply_soft_blank(raw,   blank_mask_h)
        cleaned_v = self._apply_soft_blank(raw_v, blank_mask_v) if run_v else None

        return BlankerResult(
            cleaned_h    = cleaned,
            cleaned_v    = cleaned_v,
            blank_mask   = blank_mask_h,
            mag_h        = mag_h,
            blank_mask_v = blank_mask_v,
            mag_v        = mag_v,
        )

    @staticmethod
    def _apply_soft_blank(raw, mask):
        """Hann-taper the blanked regions of ``raw`` via ``mask`` and return complex64."""
        if mask is None or not mask.any():
            return raw.copy().astype(np.complex64, copy=False)
        soft = np.minimum(
            _nd_convolve1d(mask, _NB_TAPER_KERNEL, mode='constant', cval=0.0),
            1.0,
        )
        return (raw * (1.0 - soft)).astype(np.complex64)

    def _process_channel(self, engine, raw, nb_factor, suffix):
        """Run FFT + time-domain Linrad detection on one channel.

        ``suffix`` selects the engine state attributes:
          ''   → H (``_nb_spec_avg``, ``_nb_floor`` …)
          '_v' → V (``_nb_spec_avg_v``, ``_nb_floor_v`` …)

        Reads state off the engine at entry, mutates locals through the
        loop, writes state back at exit.  Returns (blank_mask, mag).
        """
        # Pull state into locals.  Suffix allows the same code to operate
        # on H and V without keyword-sharing between channels.
        def _g(name):  return getattr(engine, f'_nb_{name}{suffix}')
        def _s(name, val): setattr(engine, f'_nb_{name}{suffix}', val)

        spec_avg         = _g('spec_avg')
        env              = _g('env')
        floor            = _g('floor')
        wideband_floor   = _g('wideband_floor')
        lowlevel_frac    = _g('lowlevel_frac')
        blkrms_median    = _g('blkrms_median')
        stuck_counter    = getattr(engine, f'_nb_stuck_count{suffix}', 0)

        # Stale-state detection: if the stored broadband reference is far
        # below the current block statistics for 100+ consecutive blocks,
        # the state has been invalidated (e.g. user hot-swapped antennas or
        # radically changed gain).  Force a reseed so the blanker recovers
        # immediately instead of waiting ~10 s for blkrms_median step-up.
        _STUCK_LIMIT = 100   # blocks — ~0.5 s at 192 kHz, ~2 s at 48 kHz

        metric_threshold = nb_factor * nb_factor
        mag              = np.abs(raw).astype(np.float32)

        # Seed per-bin spectrum average from the first block's median power.
        if spec_avg is None:
            first = raw[:NB_FFT_SIZE] if len(raw) >= NB_FFT_SIZE else raw
            X0 = np.fft.fft(first, n=NB_FFT_SIZE)
            P0 = np.abs(X0).astype(np.float64) ** 2
            median_P0 = max(float(np.median(P0)), 1e-30)
            spec_avg = np.full(NB_FFT_SIZE, median_P0, dtype=np.float64)
            rms   = float((median_P0 / NB_FFT_SIZE) ** 0.5)
            env   = max(rms, 1e-9)
            floor = env

        n_blocks = len(raw) // NB_FFT_SIZE
        alpha    = 1.0 - np.exp(-NB_FFT_SIZE / (NB_SPEC_AVG_TC * engine.sample_rate))

        # Pre-compute TD hot mask using the previous call's floor so an
        # impulse cannot inflate its own threshold.
        _td_ref = blkrms_median if blkrms_median is not None else floor
        _td_hot = mag > (nb_factor * _td_ref) if _td_ref is not None \
                  else np.zeros(len(mag), dtype=bool)

        blank_mask = np.zeros(len(raw), dtype=np.float32)
        engine._nb_total_count += len(raw)

        last_P         = None
        last_hot       = 0
        last_blanked_P = None

        for i in range(n_blocks):
            s = i * NB_FFT_SIZE
            e = s + NB_FFT_SIZE

            block_rms = float(np.sqrt(np.mean(mag[s:e] ** 2)))
            if blkrms_median is None:
                blkrms_median = max(float(np.median(mag[s:e])), 1e-12)
            else:
                step = max(blkrms_median, 1e-12) / NB_WIDEBAND_AVGNUM
                blkrms_median = (blkrms_median + step if block_rms > blkrms_median
                                 else max(blkrms_median - step, 1e-12))
            is_broadband_rms = block_rms > nb_factor * blkrms_median

            X = np.fft.fft(raw[s:e])
            P = np.abs(X).astype(np.float64) ** 2

            n_hot            = int((P > metric_threshold * spec_avg).sum())
            is_broadband_fft = n_hot > NB_BROADBAND_FRAC * NB_FFT_SIZE
            is_broadband     = is_broadband_rms or is_broadband_fft

            if is_broadband:
                stuck_counter += 1
                # Auto-reseed when stuck: every block broadband for too
                # long means our reference is orders of magnitude stale.
                # Reset per-bin avg and blkrms_median from this block's
                # statistics so the next block is compared against current
                # conditions, not the pre-event baseline.
                if stuck_counter >= _STUCK_LIMIT:
                    spec_avg      = np.maximum(P.copy(), 1e-30)
                    blkrms_median = max(float(np.median(mag[s:e])), 1e-12)
                    stuck_counter = 0
                    # Treat this block as clean so the mask doesn't carry
                    # the bogus flag forward.
                    continue
                n_td_in_block = int(_td_hot[s:e].sum())
                if n_td_in_block <= NB_SHORT_IMPULSE_THRESH:
                    pass  # short impulse — let the TD path handle it sample-precise
                else:
                    blank_mask[s:e] = 1.0
                    engine._nb_blanked_count += NB_FFT_SIZE
                    last_blanked_P = P   # freeze for display
                    if suffix == '':
                        _bl = getattr(engine, '_nb_blank_log', None)
                        if _bl is not None:
                            _bl.append(getattr(engine, '_nb_log_t0', 0.0) + s / engine.sample_rate)
            else:
                spec_avg = (1.0 - alpha) * spec_avg + alpha * P
                stuck_counter = 0   # broadband streak broken

            last_P   = P
            last_hot = n_hot

            # Wideband median noise-floor estimator (SELLIM-style).
            block_median = float(np.median(P))
            clean_frac   = float((P <= metric_threshold * spec_avg).sum()) / NB_FFT_SIZE
            lowlevel_frac = ((1.0 - _NB_LOWLEVEL_ALPHA) * lowlevel_frac
                             + _NB_LOWLEVEL_ALPHA * clean_frac)
            if lowlevel_frac >= 0.1:
                avgnum = float(getattr(engine, 'nb_avgnum', NB_WIDEBAND_AVGNUM))
                alpha_wb = 1.0 / max(avgnum, 1.0)
                wideband_floor = (block_median if wideband_floor is None
                                  else (1.0 - alpha_wb) * wideband_floor + alpha_wb * block_median)

        # Time-domain mask merged in after the FFT loop.
        _td_new = _td_hot & (blank_mask == 0.0)
        if _td_hot.any():
            blank_mask[_td_hot] = 1.0
            engine._nb_blanked_count += int(_td_new.sum())

        # Update floor/env for display.
        rms   = float((spec_avg.mean() / NB_FFT_SIZE) ** 0.5)
        floor = max(rms, 1e-9)
        not_blanked = blank_mask == 0.0
        if not_blanked.any():
            n_ok      = int(not_blanked.sum())
            alpha_env = 1.0 - np.exp(-n_ok / (2.0 * engine.sample_rate))
            chunk_env = float(mag[not_blanked].mean())
            env = (1.0 - alpha_env) * (env if env is not None else chunk_env) + alpha_env * chunk_env

        # Write state back onto engine.
        _s('spec_avg',       spec_avg)
        _s('env',            env)
        _s('floor',          floor)
        _s('wideband_floor', wideband_floor)
        _s('lowlevel_frac',  lowlevel_frac)
        _s('blkrms_median',  blkrms_median)
        _s('last_P',         last_P)
        _s('last_hot',       last_hot)
        setattr(engine, f'_nb_stuck_count{suffix}', stuck_counter)
        if last_blanked_P is not None:
            _s('last_blanked_P', last_blanked_P)

        return blank_mask, mag

    def reset(self, engine) -> None:
        """Clear Linrad warm-up state for both H and V."""
        for suffix in ('', '_v'):
            setattr(engine, f'_nb_env{suffix}',            None)
            setattr(engine, f'_nb_floor{suffix}',          None)
            setattr(engine, f'_nb_wideband_floor{suffix}', None)
            setattr(engine, f'_nb_lowlevel_frac{suffix}',  0.75)
            setattr(engine, f'_nb_spec_avg{suffix}',       None)
            setattr(engine, f'_nb_last_P{suffix}',         None)
            setattr(engine, f'_nb_last_hot{suffix}',       0)
            setattr(engine, f'_nb_blkrms_median{suffix}',  None)
            setattr(engine, f'_nb_last_blanked_P{suffix}', None)
            setattr(engine, f'_nb_stuck_count{suffix}',    0)


# ── Registry and factory ─────────────────────────────────────────────────────

_REGISTRY: dict[str, type[Blanker]] = {
    "Linrad": LinradBlanker,
    "Bypass": BypassBlanker,
}

# Optional backends that depend on a compiled shared library.  Registered
# lazily so a missing libnob.so does not break import of this module.
try:
    from .nb_nr0v_wideband import NR0VWidebandBlanker
    _REGISTRY["NR0V-Wideband"] = NR0VWidebandBlanker
except (ImportError, RuntimeError):
    pass


def available() -> list[str]:
    """Names of registered blanker backends, in preferred UI order."""
    return list(_REGISTRY.keys())


def make(name: str) -> Blanker:
    """Instantiate a blanker by name.  Unknown names fall back to Linrad."""
    cls = _REGISTRY.get(name, LinradBlanker)
    return cls()
