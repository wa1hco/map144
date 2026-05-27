# Detection baseline — vocabulary and reference

**Status:** outline only — review for framing/structure before filling in.

This document is the canonical reference for how detection performance
is described and compared in MAP144.  It exists to stop the
re-litigation of the same dB/SNR/scaling questions every session.  When
a discussion starts to drift on these points, the answer is "look it up
in this doc and update the doc if it's wrong" — not "re-derive from
scratch".

This is a **vocabulary** doc, not a design doc.  It says what words
mean.  It does not propose new architecture.

---

## 1. Scope

What this doc covers, what it doesn't.  Pointer to companion docs
(`detection_test_protocol.md`, `detection_results.md`).

## 2. Conventions (the rules that keep changing)

- **dB is always power.**  Two valid forms only: `10·log10(power)` or
  `20·log10(amplitude)`.  There is no "amplitude-dB".
- **SNR is in 2.5 kHz reference BW.**  The 12 kHz audio from WSJT-X or
  from a typical SSB receiver is already bandlimited to ≈ 2.5 kHz by
  the analog audio filter; no BW correction needed.
- **SNR is a property of the input signal, not the detector's output.**
  A detector produces a *metric* (detmet, MF magnitude, etc.) that is
  proportional to the input SNR in some algorithm-specific way.  When
  we say "the detector estimated SNR = X dB", we mean the metric was
  mapped back to an input-SNR estimate via that algorithm-specific
  relationship.  The mapping accuracy varies — see §5.
- **Coherent integration**, unqualified, refers to averaging 72 ms
  MSK144 frames coherently across a repeated transmission.  Other
  uses (multi-channel summation, polarization combination, etc.) get
  their own qualifier.
- **Sample rates and where each applies** (48 kHz IQ, 12 kHz audio,
  per-channel rates, etc. — to be filled in §3).
- **Phase sign conventions** (FFT, mixing, NCO direction — to be
  filled in).

(Pointers back to `CLAUDE.md` for the canonical versions of dB and
SNR — this section repeats the rules briefly so the doc reads
self-contained, but `CLAUDE.md` is the source of truth.)

## 3. Pipeline reference with units

Each processing block declares its own input and output units.  Use
the per-block table below to track what a given quantity actually is
at any point in the chain.  When two blocks' outputs need to be
compared (e.g., when computing an SNR or a dB), translate both back
to a common reference — usually "input-audio voltage" or "input-audio
power" — using the block's declared transformation.

Convention used in this table: **A = input audio voltage amplitude**.
Powers are written as A², energies as A²·T (time-integrated).  When
a block multiplies samples together (squaring, correlation against
a template), the output's scaling in A is whatever the algebra
produces — stated explicitly per block.

### Signal flow

```
SSB Rx audio path:
    SSB Rx → analog SSB filter → 12 kHz real audio
           → analytic transform → 12 kHz complex IQ
           → sq_det (per-frame metric)
           → per-channel sync correlator (post-channelizer)

IQ Rx path:
    IQ Rx (48 / 96 / 192 kHz wideband) → noise blanker → decimate
           → 48 kHz complex IQ
           → channelizer (48 channels @ ~1 kHz BW) → per-channel sync
           OR
           → TFMF wideband matched filter
```

### Per-block units (to be filled in by detector subsections)

| Block | Input units | Output units | Time scaling |
|---|---|---|---|
| Analog SSB filter | A (voltage) | A (voltage), BW ≈ 2.5 kHz | no change |
| Hilbert / analytic | real A | complex A (I+jQ) | no change |
| Channelizer | 48 kHz complex A | per-channel complex A | decimation N→N/M |
| Squarer (sq_det stage 1) | complex A | complex A² | no change |
| FFT(squared) (sq_det stage 2) | complex A² | complex A² per bin | NSPM → 1 frame |
| \|·\|² (sq_det stage 3) | complex A² | real A⁴ per bin | no change |
| Sync MF correlator | complex A, length N | scalar complex A (∑signal·conj(h)) | N → 1 |
| TFMF surface | complex A, length N | complex A per cell (n_windows × n_freq) | stride→1, FFT |
| Per-bin median normalisation | A per cell | dimensionless ratio | trailing window |
| 20·log10(ratio) | dimensionless | power-dB | none |

(Open: the FFT row in sq_det — the bins ARE in A² units immediately
after FFT(squared); the |·|² operation is what produces A⁴.  WSJT-X's
`tonespec = |FFT(s²)|²` is the A⁴ quantity.  Verified in
sq_det.py:206.)

## 3.5 Receiver chain phase / amplitude — calibration baseline

**Design point:** MAP144's own audio filter chain introduces negligible
phase or amplitude distortion within the signal band (300–2700 Hz).
By design, **all internal filters are linear-phase FIR or zero-phase
filtfilt** with one exception (the scipy IIR decimate), and that
exception operates with its non-flat region above the signal band.
**Therefore MAP144 does not require receiver-chain calibration** —
upstream radio behaviour aside.

### Filter inventory (verified in code)

| Stage | File | Filter | Phase behaviour |
|---|---|---|---|
| Source decimation (SDR → 48 kHz) | `airspy_source.py:121` | 15-tap `firwin` LP via `lfilter` | **Linear phase** |
| Channelizer LP (anti-alias before 48→12 kHz) | `channelizer.py:142` | 23-tap `firwin` LP at 2700 Hz | **Linear phase** |
| Channelizer HP (per-channel 300 Hz DC reject) | `channelizer.py:152` | 15-tap `firwin` HP at 300 Hz | **Linear phase** |
| `scipy.signal.decimate` (48→12 kHz for jt9 input) | `detection.py:661` | IIR (Chebyshev I, 8th-order), `zero_phase=False` | **Nonlinear phase** — but cutoff is ~4800 Hz, *above* the 300–2700 Hz signal band |
| SPD audio BPF (300–2700 Hz on complex baseband) | `detection.py:269` | 129-tap `firwin` LP via `filtfilt` | **Zero phase** |

Definitions:
- **Linear phase** — constant group delay across the band; introduces a
  fixed time offset but no in-band distortion.
- **Zero phase** — no delay AND no distortion (filtfilt forward+reverse
  pass cancels phase).
- **Nonlinear phase** — group delay varies with frequency; introduces
  real distortion within the affected band.

### Why this matters operationally

The "audio filter chain is flat" property has three consequences:

1. **No phase-equalization step is needed inside MAP144.** WSJT-X has
   a `Tools → Phase Equalization` feature designed to compensate for
   nonlinear-phase passband filters on analog SSB receivers; MAP144's
   chain doesn't introduce equivalent distortion to compensate.
2. **Calibration would target upstream**, not internal. The only
   plausible source of in-band amplitude/phase distortion is the
   receiver itself (SDR firmware, audio filter on an analog SSB
   receiver, etc.).  For Flex DAX audio at the recommended 4–5 kHz
   DIGU filter, AK4AO's setup doc (Vienna Wireless, 2016) confirms
   the audio is "already flat"; MAP144 inherits that.
3. **The SPD audio BPF is for noise-bandwidth matching, not distortion
   correction.** The 300–2700 Hz BPF on the SPD path exists because
   the IIR decimation upstream passes noise across ~4800 Hz while the
   signal occupies only ~2400 Hz; without the BPF, noise power is 2×
   the signal bandwidth and SNR estimates are pessimistic.  Bandlimiting
   noise to match the signal recovers ~+1.5 dB sensitivity.  This is
   noise-bandwidth control, not phase/amplitude calibration.

### Caveats

- The IIR `scipy.signal.decimate` step is the only nonlinear-phase
  filter in the chain.  If a future change tightens its cutoff toward
  the signal band, or if a narrower receiver filter is used upstream,
  the group-delay nonlinearity could move into the signal band and
  start to matter.  At the current configuration it doesn't.
- The 300 Hz per-channel HP filter rejects energy at exactly DC in
  each channel — see `project_channelizer_hp_dc_reject` memory.  This
  shapes the channel response near 300 Hz audio (= channel-centre)
  but is symmetric and well-understood; not a distortion needing
  correction.
- This baseline assumes the operator's radio chain feeds clean IQ
  (or clean DAX audio in 4–5 kHz filter).  Analog SSB receivers with
  narrow 2.4 kHz contest filters have nonlinear phase at the band
  edges (see QEX 2017 Figure 7); MAP144 does not currently compensate
  for those.

### Verification approach (if ever needed)

To empirically verify the "flat" assumption on a specific receiver
chain:

1. Capture a strong continuous reference signal (e.g., W1NB MSK144
   recording, or an injected CW tone sweep)
2. Compute the cross-spectrum vs a clean reference (locally generated
   MSK144 waveform with known phase)
3. Plot magnitude and phase difference across 300–2700 Hz
4. If magnitude ripple > ~0.5 dB or phase departure > 0.2 rad within
   the band, calibration would buy real performance.

The W1NB recording at `/home/jeff/.local/share/WSJT-X - flex/save/
260507_114000.wav` is a suitable reference signal for this measurement
on the operator's current Flex setup.

## 4. Per-detector reference pages

One short subsection per detector.  Each says: what's the input, what
gets computed, what's the output's scaling, what's the noise
distribution at the output, what's the threshold statistic, what's the
theoretical false-alarm rate vs threshold, what's the accuracy of the
SNR estimate.

### Threshold sharpness: many cells vs few cells

There IS a real "soft threshold vs hard threshold" effect, but the
right axis isn't bandwidth — it's **number of independent cells the
detector evaluates per measurement period**.

- A detector that evaluates **few cells** (e.g., one per channel per
  frame) has a noise distribution at its peak with relatively high
  variance.  The threshold that gives a target FA rate sits in a
  region where noise fluctuations are common.  Detection probability
  vs SNR rolls off gradually as SNR drops — **soft threshold**.
- A detector that evaluates **many cells** (e.g., 5.76 M cells per
  15 s for TFMF) has a noise distribution at its peak with very
  tight extreme-value statistics — the worst noise excursion across
  millions of cells is a sharply-defined number.  The threshold is
  set high enough to suppress those, and signals either clear it or
  don't.  Detection probability vs SNR transitions sharply — **hard
  threshold**.

(This is the same intuition you described as "narrowband soft /
wideband brick wall"; the underlying mechanism is extreme-value
statistics of the cell count, not the bandwidth per se.  A
narrowband search with 5 M cells would also have a sharp threshold;
a wideband search with 100 cells would not.)

Cell counts on the 16-ping, 15-s ramp test:

| Detector | Cells per 15 s | Expected curve shape |
|---|---|---|
| sq_det per channel, sliding window | ~1 666 | soft |
| sq_det × 40 channels | ~67 000 | medium |
| TFMF wideband (48 kHz, default config) | ~5 760 000 | hard |

The two-sub-block sum in TFMF *doesn't* affect cell count — it
affects matched-filter gain (signal lift) and noise lift equally, so
the FA distribution is unchanged.  TFMF's hard-threshold character
comes from cell count, not from sub-block summation.

### Compare against decoder floor, not just detector floor

The mission goal is **decodable signals**, not the weakest detectable.
Every detector subsection notes detection floor; the comparison plot
in §6 should overlay the **decoder decode-floor curve** so the
"sensitivity surplus" (detector sees it but decoder can't use it)
is visible.  For MSK144, the decode floor curves are:
- Confident single-frame decode (Pc ≥ 0.9): SNR_2.5k ≈ **0 dB**
- Confident K=7-averaged decode (jt9 -d 3): SNR_2.5k ≈ **−8.5 dB**
See §7 for sourcing.

### 4.1 sq_det (WSJT-X-faithful)

- **Input:** 12 kHz analytic IQ frame (NSPM = 864 samples).
- **Output:** `detmet` (A⁴ scaling — peak of `|FFT(s²)|²` at the
  squared-tone bins), normalised by 25th-percentile of detmet across
  a sliding block.
- **Threshold:**
  - Native form: linear `detmet_norm ≥ 3.0` (WSJT-X canonical).
  - In **input-audio power dB** (the form that's directly comparable
    across detectors): **≈ 2.4 dB**.  Derivation: detmet ratio = 3
    is an A⁴ ratio; equivalent A² (power) ratio = √3 = 1.73; in
    dB = 10·log10(1.73) = 2.39.
  - **NOT** 9.5 dB.  That's the test-driver's mislabel from applying
    `20·log10` to detmet_norm as if it were a voltage.  (See CLAUDE.md
    "dB convention".)
- **FA model:** distribution of `|FFT(s²)|²` on noise.  Not Rayleigh
  (because of the squaring step); empirically chi-squared-ish.
  **Open:** derive or measure the FA-rate-vs-threshold curve for
  sq_det specifically; current value (`detmet_norm ≥ 3`) is set by
  WSJT-X heritage, not derived from a stated FA target.
- **SNR estimate accuracy:** **Open.**  detmet_norm at a known-SNR
  signal vs detmet_norm on noise has high variance; the mapping back
  to input-SNR is not well-characterised in MAP144.
- **Detection probability vs SNR curve:** to be measured per the
  test protocol §10.

### 4.2 Per-channel sync correlator (MAP144 legacy)

- **Input:** 12 kHz channelised complex IQ per channel (1 kHz BW),
  N = 42-sample sync template (two sub-blocks coherently pre-summed).
- **Output:** `|dot(signal, conj(template))|` — A scaling (amplitude
  of complex inner product), normalised by 25th-percentile across a
  sliding block.
- **Threshold:**
  - In code today (`processing.py`): `SYNC_DETECT_THRESH_DB = 3.0`,
    but this is the unit-bug form: `10·log10(amplitude_ratio)`.
  - In **correct power dB** (= `20·log10(amplitude_ratio)`): **6 dB**.
    Same physical trigger condition; just expressed correctly.
  - The 2-line fix to use `20·log10` plus changing the constant to
    `6.0` is queued as a TODO (see [project_detection_metric_db_scale_mismatch]
    memory).
- **FA model:** matched-filter magnitude on complex Gaussian noise is
  **Rayleigh** distributed.  `P_FA(per_cell) = exp(−ln(2)·10^(T_dB/10))`
  for threshold expressed in correct power dB.
  At 6 dB: P_FA = exp(−ln(2)·4) = 0.063 per cell.  With ~1 666
  windows × 48 channels = 80 000 cells per 15 s, that's ~5 000 FA/15 s
  per cell-trial — but operationally far fewer because clusters get
  merged and most channels are quiet.  Operational FA rate empirically
  ~10 fires per 15 s across all channels.  Gap between per-cell math
  and operational rate is a **TODO to investigate** (clustering,
  cooldown, channel correlation).
- **SNR estimate accuracy:** **Open.**  At signals well above
  threshold, the metric scales approximately linearly with input
  audio voltage.  Calibration uncertain.
- **Detection probability vs SNR curve:** to be measured per protocol §10.

### 4.3 TFMF (wideband matched filter)

- **Input:** 48 kHz wideband complex IQ (full Nyquist, not
  channelised).  Template length N = 192 samples per sub-block; with
  `two_subblock_sum=True` (default), effective coherent length is
  384 samples covering both MSK144 sync sub-blocks.
- **Output:** matched-filter time-frequency surface, magnitude
  scaled as A, normalised by per-bin median across windows.
- **Threshold:**
  - Native form (already correct power dB):
    `snr_db = 20·log10(|MF| / median(|MF|)) > cfg.threshold_db`.
  - Current default `cfg.threshold_db = 13.5 dB`, chosen so the
    theoretical FA rate is ≈ 1 per 15 s across 5.76 M cells.
  - Translating the 13.5 dB matched-filter-output threshold to an
    **input-audio power dB** requires the matched-filter gain.  For
    N_eff = 384: theoretical √N voltage gain = 25.8 dB.
    threshold-at-input = 13.5 − 25.8 = **−12.3 dB** audio power.
  - **Empirical detection floor on the 16-ping ramp at this threshold:
    SNR ≈ +1 dB**, not −12 dB.  The ~13 dB gap between theory and
    measurement is **open** — possible causes: matched-filter
    mismatch loss, median-normalisation behaviour differing from
    pure σ, signal not aligned to bin/sample.  Needs investigation
    before quoting a TFMF detection floor as fact.
- **FA model:** Rayleigh on the per-cell magnitude (verified
  empirically on AWGN to ~10% accuracy — see [TFMF README "False
  alarm rate vs threshold"](../experiments/gpu_tfmf/README.md)).
  Per-cell `P_FA = exp(−ln(2)·10^(T_dB/10))`; total expected FAs =
  P_FA × N_cells.
- **SNR estimate accuracy:** at signals well above threshold, the
  reported `sync_db` correlates with input SNR but with large
  per-realisation variance (~±2 dB on the 16-ping ramp).  At
  signals near threshold, `sync_db` is **pinned near the threshold
  itself** — useless as an estimator there.  Documented previously
  as a reason TFMF can't usefully gate weak-signal rejection.
- **Detection probability vs SNR curve:** to be measured per
  protocol §10.

## 5. SNR vs detection metric — what each is, what it isn't

**SNR is a property of the input signal**, not a quantity the
detector computes.  At any point in the pipeline, the input signal
has a true SNR (in 2.5 kHz BW per the convention) that is fixed by
the propagation conditions and the receiver.  We don't measure it
directly — we infer it.

**Detection metrics are what detectors compute**: detmet, |MF|,
ratios thereof, normalised forms.  Each metric has *some*
relationship to the input SNR, but the relationship is:

- algorithm-specific (each detector's metric scales differently with
  the underlying signal amplitude — see §3 per-block units)
- noisy (single-realisation metric value vs single-realisation noise
  produces high variance in the implied SNR)
- only useful in the regime where the detector is operating cleanly
  (well above its own noise threshold)

The phrase **"detector-reported SNR"** is a shorthand for: "the
detector's metric, mapped back through its algorithm-specific
amplitude relationship to an input-audio-power dB number".  When that
mapping is well-calibrated and the signal is above the regime where
the metric pins near the threshold, the reported SNR is informative.
When either condition fails, it isn't.

In this doc:

- **"input SNR" / "SNR_2.5k"** — the actual SNR of the input signal
  in 2.5 kHz reference BW.  Ground truth, not measured by the detector.
- **"detector metric"** — the raw detmet, |MF|, etc.  Quoted in the
  detector's native units.
- **"detector-reported SNR"** — only used where the mapping is
  documented; otherwise stick to the raw metric.

## 6. Comparing detectors fairly

Standard methodology: **match FA rate**, then compare detection rate
vs SNR_2.5k.  Steps:

1. Run each detector on noise-only IQ of the target duration.
2. Pick threshold per detector to achieve target FA rate (e.g., 10
   FA/15s).
3. Run both at those thresholds on signal-bearing IQ.
4. Plot detection-rate vs SNR_2.5k for each.

This avoids the "TFMF's 13.5 dB vs sq_det's 9.5 dB" cross-convention
trap — each detector's threshold is in its own units, but both have
the same operationally-observed FA rate, which is the only thing that
matters for fair comparison.

## 6.5 Streaming-mode constraint

TFMF (and any new detector that replaces it) must operate in
**streaming mode** with **1–2 seconds total processing delay** from
IQ-sample arrival to detection-event emission.  This is an
operational requirement, not a "nice to have" — MAP144 needs
detections delivered to the **GUI** and **jt9 decoder** in near-real-
time so the operator sees activity and the decoder can run within
the WSJT period.  (PSKReporter is a separate consumer downstream of
the decoder and is rate-limited at the PSKReporter end — it doesn't
drive this latency budget.)  The existing legacy pipeline meets this
constraint.

Implications:

- **No 15-second batch processing.**  The current TFMF design (build
  surface for full 15-s window, then median-normalise) is a research
  scaffold.  Production design must process incrementally.
- **Rolling noise-floor estimator.**  Median across recent ~1–2 s
  of windows (~3000–6000 windows per freq bin), not across a full
  15-s period.  Update incrementally as new windows arrive.
- **K-frame coherent integration** at K=7 needs 7 × 72 ms = 504 ms
  of buffered IQ.  Fits within the budget; K=4 (288 ms) leaves more
  headroom.
- **Two-sub-block sum** needs 1344 samples = 28 ms of lookahead.
  Trivial within the budget.
- **2D local-max suppression** (when added) must run over a sliding
  window, not the full surface.
- **Latency vs sensitivity trade-off** — anything that requires
  more lookahead than 1–2 s (e.g., K=20 coherent integration) is out
  of scope; document the constraint and choose algorithms that fit
  inside it.

A streaming version of the detector is a separate concern from the
batch sensitivity comparison protocol in §6, but it's a design
gate: **any algorithm that wins on batch sensitivity but can't be
made to fit the streaming budget is not a production candidate.**

## 7. Known floors and open questions

Cite what's verified, what's measured, what's open.

- **MSK144 decode floor** (per QEX 2017, Franke & Taylor, *The MSK144
  Protocol for Meteor-Scatter Communication*):
  - **Single-frame decode** (Figure 6, short messages; Figure 4,
    standard messages): Pc ≈ 0 at SNR_2.5k = −6 dB; Pc ≈ 0.5 at
    −3 dB; **Pc ≈ 0.9+ (confident decode) at ≈ 0 dB**.  Operational
    reference floor.
  - **With K-frame coherent averaging**, the decode curve slides
    left by `10·log10(K)`.  Article-quoted gains: N=2: 3 dB; N=3:
    4.8 dB; N=4: 6 dB; N=5: 7 dB; **N=7: 8.5 dB**.
  - The "decoded with SNR as low as −8 dB" claim in the article
    abstract is the **best-case K=7 floor**, not the operational
    threshold.  Confident K=7 decode: ≈ 0 − 8.5 = **−8.5 dB**.
  - K-frame averaging requires pings ≥ 100 ms and frequency known
    to better than 1 Hz over the averaging interval (article p. 11).
- **K-frame coherent integration gain formula**: `10·log10(K)` dB,
  authoritative per the article.  Memory entry
  `project_msk144_sync_fraction_floor.md` had this wrong as
  `5·log10(K)` (= 10·log10(√K)) — that gave ~half the correct value
  and led to the prior "K-frame integration is a dead end"
  conclusion.  Both were wrong.
- sq_det's per-frame detection floor: TBD by 2.5-kHz-referenced
  measurement.
- TFMF's per-frame detection floor at matched FA rate: TBD.
- Sync template covers 16/144 = 11.1% of frame energy — structural
  property of the protocol.

## 8. Glossary

Short defined-term list: A, sq_det, detmet, detmet_norm, sync MF,
TFMF, SNR_2.5k, K-frame coherent integration, two-sub-block,
parabolic interpolation, channelizer, FA, ROC.

## 9. Maintenance

This doc is updated when a convention is clarified or a finding
changes.  Memory entries that touched these topics
(`project_detection_metric_db_scale_mismatch`,
`project_msk144_sync_fraction_floor`, etc.) become **pointers** to
sections of this doc rather than carrying their own claims.

---

## Companion docs

- `docs/detection_test_protocol.md` — how to run the empirical TFMF
  vs sq_det comparison.  (To be written.)
- `docs/detection_results.md` — append-only log of test runs.  (To be
  written.)
