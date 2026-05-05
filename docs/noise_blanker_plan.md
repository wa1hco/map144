# MAP144 Noise Blanker Plan

Selectable wideband noise blanker backends running on complex IQ at the SDR
sample rate (192 kHz for B210 / SDRangel) before the 4:1 polyphase decimator
takes the stream to 48 kHz for MSK144 processing.

Blanking at 192 kHz rather than 48 kHz gives ~5 µs vs ~21 µs impulse-time
resolution, and ensures the blanker sees the full impulse shape before the
decimation filter smears it across adjacent samples.

Flex Radio does not run DAXIQ reliably at 192 kHz, so the Flex source path is
unaffected — the blanker is B210/SDRangel only.

---

## Pipeline placement

```
B210 / SDRangel  (192 kHz IQ, H and V)
    │
    ▼
[ Noise Blanker ]       ← 192 kHz, ~5 µs impulse resolution
    │
    ▼
[ 4:1 Polyphase Decimator ]   ← 192 → 48 kHz
    │
    ▼
[ Channelizer → audio BPF → SPD / jt9 ]   ← unchanged
```

Key invariants:

- Blanker runs **before** decimation. Reversing the order is wrong — the
  decimation filter spreads impulse energy across ~4 output samples and
  destroys the impulse shape the blanker needs to identify.
- DC offset at 192 kHz does not matter for threshold-based detection; do not
  DC-correct before blanking.
- H and V are blanked **jointly** (detection on the sum or the stronger of the
  two) rather than independently, because impulse noise is strongly
  correlated across polarizations. Per-channel detection doubles false-alarm
  rate without recovering more signal. (To be reconfirmed during Stage 2
  tuning against real captures.)

---

## Current state (Stage 1 — DONE)

The pluggable backend framework already exists in
[map144_app/noise_blanker.py](../map144_app/noise_blanker.py). The interface:

```python
class Blanker(ABC):
    name: str
    def process(self, engine, raw, raw_v, dual_pol) -> BlankerResult: ...
    def reset(self, engine) -> None: ...

@dataclass
class BlankerResult:
    cleaned_h: np.ndarray           # complex64 H after blanking
    cleaned_v: Optional[np.ndarray] # complex64 V after blanking, or None
    blank_mask: np.ndarray          # float32 per-sample mask, 0 keep / 1 blank
    mag_h: np.ndarray               # float32 |raw H|, pre-blanker, for display
```

Display-only state (per-bin spectrum averages, blanked-sample counters,
noise-floor estimates) is written onto the engine host object so existing UI
readers keep working across backend swaps.

Backends in place:

- **`BypassBlanker`** — passthrough. Preserved as the A/B reference against
  which all other backends are measured.
- **`LinradBlanker`** — current production blanker. Two-stage wideband design
  derived from Linrad (SM5BSZ): FFT-based broadband-impulse block blanking
  driven by fraction-of-hot-bins (`NB_BROADBAND_FRAC = 0.30`), with per-bin
  threshold updated only from non-blanked blocks. Followed by time-domain
  soft blanking with a Hann-tapered mask (half-width `_NB_TAPER_N`) to avoid
  sinc sidelobes from rectangular windowing. Tuning parameters
  (`NB_FACTOR = 6.0`, `NB_FFT_SIZE = 256`, `NB_SPEC_AVG_TC = 2.0`) are
  documented in the module.

Runtime mode selection already exists via the existing noise-blanker UI
controls (see briefing.md §"Noise Blanker"). The status bar already surfaces
blanking activity %.

---

## Stage 2 — NR0V Wideband blanker (NEXT)

WDSP-style wideband impulse blanker from Warren Pratt (NR0V), as used in
OpenHPSDR / cuSDR / linHPSDR / Thetis.

Behavior: threshold detection with configurable slew (leading-edge
sensitivity), pre-blanking count (samples zeroed before the threshold
crossing), and post-blanking count (samples zeroed after the trailing-edge
crossing). Zero-fill on detected impulse regions. Fast response, minimal
signal distortion on non-impulsive content, very effective for MSK144.

Reference: WDSP sources `wdsp/nbp.c` and `wdsp/anb.c` (the "auto" wideband
blanker). Upstream:
- <https://github.com/TAPR/OpenHPSDR-Firmware> (original)
- <https://github.com/df8oe/HermesLite2-v2> (cleanly separable C subset)

### Integration approach

Two viable paths, to be chosen before starting:

1. **ctypes wrapper around a compiled `.so`.** Extract `nbp.c`/`anb.c` plus
   the minimum supporting WDSP ring-buffer glue, build a small shared
   library, call it from Python via `ctypes` or `cffi`. Pros: preserves
   WDSP numerics exactly; cheapest to validate against reference
   implementations. Cons: adds a C build step and a compile-time dep.
2. **Pure-numpy reimplementation.** Port ~200 lines of C to numpy. Pros:
   no build step, matches the rest of the MAP144 codebase. Cons: need to
   verify numerical parity against the C reference on known test
   vectors.

Default choice: **start with the ctypes wrapper** (faster to a working
result), keep a numpy port as a longer-term goal if the ctypes path proves
brittle across distros.

### Class name and file

`NR0VWidebandBlanker` in [map144_app/noise_blanker.py](../map144_app/noise_blanker.py),
joining the existing `BypassBlanker` and `LinradBlanker`. If the backend
grows large enough to warrant its own file, split to
`map144_app/nb_nr0v_wideband.py` and re-export from `noise_blanker.py`.

### Parameters to tune at 192 kHz

WDSP's reference parameters target a different sample rate; all sample-count
parameters need rescaling:

- **Threshold** (dB above running noise floor)
- **Slew** (leading-edge sensitivity, derivative threshold)
- **Pre-blank count** (samples before threshold crossing)
- **Post-blank count** (samples after trailing-edge crossing)
- **Hold-off** (minimum time between consecutive blank events)

Initial sweep: grid over threshold ∈ {4, 6, 8, 10} dB × pre/post ∈ {2, 4, 8}
samples × slew ∈ {default, 0.5×, 2×} against the test corpus. Commit a
results table with decode count + blank% per combination; pick the
Pareto-optimal setting.

### Latency budget

MSK144 frames are 72 ms; blanker group delay must be < 1 frame. WDSP NB is
a ring-buffer design with group delay ≈ (pre + post + slew) samples; at
192 kHz, even 64-sample windows are only 0.33 ms. This is not tight.

---

## Stage 3 — NR0V Spectral / LPC blanker (LATER)

WDSP-style spectral noise blanker using Linear Predictive Coding. Detects
impulses by comparing signal to an LPC-predicted value; replaces detected
impulse with the LPC reconstruction rather than zero-filling. Preserves more
underlying signal energy at low SNR, where zero-filling can destroy
recoverable pings.

Reference: WDSP `wdsp/snb.c`, `wdsp/snba.c` (spectral noise blanker
"automatic"). Same upstream as Stage 2.

Design notes:

- LPC order: WDSP default is p = 16. At 192 kHz, 16-tap prediction window is
  ~83 µs — well within the 72 ms frame budget.
- Computational cost is O(p²) per prediction. For p = 16 this is still
  negligible at 192 kHz in numpy.
- Parameter tuning: same grid approach as Stage 2, plus LPC order.

Class name: `NR0VSpectralBlanker`.

Acceptance gate: only proceed to Stage 3 if Stage 2 demonstrates measurable
decode-count improvement over `LinradBlanker` on the corpus. If Stage 2 is a
wash, Stage 3 is unlikely to help and the time is better spent on the
hardware path (QRM Eliminator / two-antenna canceller).

---

## Test corpus and acceptance criteria

### Corpus

Minimum set, matching the categories required by `CLAUDE.md`:

1. **Noise-only IQ** (no pings). Used to confirm no false blanking under
   clean conditions. `blank_pct` should be near zero.
2. **Strong-signal IQ** (high-SNR meteor pings, no impulse noise). Used to
   confirm the blanker does not distort strong signals.
3. **Marginal-SNR IQ** (weak pings at ~ −4 dB to ~ −8 dB, no impulse). Used
   to confirm the blanker preserves weak decodes.
4. **Impulse noise + strong signal** (SMPS/power-line + clearly decodable
   pings). Used to confirm blanking recovers decodes that noise would
   otherwise corrupt.
5. **Impulse noise + marginal signal** (the hard case — the one the blanker
   is actually for).
6. **Timing and frequency offset cases** — pings near the edge of the
   15-second period, or offset from the nominal 1500 Hz audio frequency.
7. **Pathological** — very dense pulse trains (every few ms), wideband
   ignition-style noise, simultaneous SMPS + pulse train.

Captures should be saved as `.wav` (complex IQ) at 192 kHz under a
`captures/nb_corpus/` directory with a short metadata sidecar noting source,
date, observed noise type, and ground-truth ping count where known.

### Primary metric

**MSK144 decode count per 15-second period**, on the `LinradBlanker` as
baseline. Acceptance criterion per backend:

- On corpora #1–#3 (no impulse): decode count ≥ baseline − 0 (no
  regression). `blank_pct` ≤ 1% on #1, ≤ 5% on #2–#3.
- On corpora #4–#5 (impulse present): decode count ≥ baseline, with
  measurable improvement on at least one corpus.
- On corpus #7 (pathological): does not crash, does not drop to zero
  decodes; no worse than baseline.

### Secondary metrics

- `blank_pct` — fraction of samples zeroed. Lower is better at equal decode
  count.
- SNR of unblanked region as reported by SPD — the blanker should raise SNR
  on impulse-corrupted pings by at least the ratio of impulse power to signal
  power.
- Per-period false-alarm and miss counts against ground truth, where the
  corpus has it.

### Composite score (for quick sweeps)

```
score = decode_count / (1 + blank_pct/100)
```

Useful for parameter grid sweeps but **not sufficient alone** for
acceptance — a weak blanker that passes more noise through can score high
by coincidence. Always inspect `blank_pct` and the per-corpus breakdown
directly before declaring a winner.

---

## File layout (current and proposed)

Current (Stage 1):

```
map144_app/
  noise_blanker.py     ← Blanker / BlankerResult / BypassBlanker / LinradBlanker
```

After Stage 2, if backends stay small:

```
map144_app/
  noise_blanker.py     ← +NR0VWidebandBlanker
```

After Stage 2, if NR0V integration grows (ctypes wrapper, WDSP vendored
subset):

```
map144_app/
  noise_blanker.py            ← base classes, factory, registry
  nb_linrad.py                ← LinradBlanker moved out
  nb_nr0v_wideband.py         ← NR0VWidebandBlanker
  nb_nr0v_spectral.py         ← NR0VSpectralBlanker (Stage 3)
vendor/wdsp/                  ← vendored WDSP C sources, if the ctypes path
  nbp.c  anb.c  snb.c ...
  build.sh
```

Do not split modules speculatively — wait until the larger backend actually
lands and pushes `noise_blanker.py` past ~500 lines.

---

## Licensing

WDSP is GPL-compatible. MAP144 is GPL-3. Vendoring a subset of WDSP sources
under `vendor/wdsp/` with the original copyright headers is compliant.
Record the upstream commit hash in a `vendor/wdsp/SOURCE.md` when the
initial import is done, so future re-syncs can diff against a known base.

Linrad code (the origin of the current `LinradBlanker`) is GPL-2-or-later
per SM5BSZ's published licensing. Compatible with GPL-3.

---

## References

- WDSP library — OpenHPSDR-Firmware:
  <https://github.com/TAPR/OpenHPSDR-Firmware>
- Linrad wideband blanker (SM5BSZ):
  <http://www.sm5bsz.com/linuxdsp/blanker/blanker.htm>
- Current briefing: [briefing.md](briefing.md) §"Noise Blanker"
- Current code: [map144_app/noise_blanker.py](../map144_app/noise_blanker.py)

---

## Notes

- **SMPS / solar / wideband non-impulse noise.** A two-antenna hardware
  canceller (QRM Eliminator-style) is the best practical answer for noise
  that isn't short-duration impulsive. The spectral LPC blanker (Stage 3)
  is the best DSP approach for these sources but is still a DSP-only
  mitigation and will not match a hardware canceller on a strong interferer.
- **MSK144 timing.** Tx delay is 0.2 s by default, so RF is active roughly
  t=0.2 s to t=15 s within a period; the blanker's group delay budget
  against one 72 ms frame is very loose at 192 kHz.
- **Dual-polarization.** The `Blanker.process` contract already returns
  separate cleaned H and V streams plus a single shared `blank_mask`. That
  shape is correct for joint detection with per-channel output and does not
  need to change for Stages 2 or 3.
- **222 MHz MSK144.** Calling frequency is 222.090 MHz (de facto standard).
  Noted here for completeness since the blanker is band-agnostic.
