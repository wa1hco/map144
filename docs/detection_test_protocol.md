# Detection test protocol — TFMF vs sq_det

**Status:** outline only — review for structure before filling in.

This document specifies the empirical procedure for comparing
detection performance across MAP144's detectors.  Companion to
`detection_baseline.md` (vocabulary) and `detection_results.md`
(run log).

The protocol's goal is to produce **apples-to-apples numbers** that
survive future re-readings.  Anyone running the protocol on the same
input should get the same answer; anyone reading the results should
know exactly what was measured.

---

## 1. Scope

What this protocol measures (detection rate at matched FA rate vs
SNR_2.5k) and what it doesn't (decode rate, end-to-end QSO success,
real-world propagation effects).

## 2. Detectors under test

- **TFMF**: wideband 48 kHz matched filter (`experiments/gpu_tfmf/`).
- **sq_det × 40 channels**: per-channel WSJT-X-faithful sq_det run on
  each of the 40-48 channelizer outputs.
- (Optional) **per-channel sync correlator**: MAP144's legacy sync
  matched filter, post-channelization.

Each detector's configuration is captured as a row in the results
CSV so a single test run can sweep multiple variants.

## 3. Test input

Two input modes, both produced by the existing simulator:

- **Synthetic ramp** — 15 s of wideband IQ with N MSK144 pings placed
  at known (time, freq, SNR_2.5k) values.  Default: 16 pings, SNR
  −15 dB to +5 dB in 1 dB steps, ±20 kHz freq spread.
- **Noise-only** — same duration, same noise model, no signals.
  Used for FA calibration.

Both files come from the same generator with deterministic seeds so
runs are reproducible.

## 4. FA-rate calibration

For each detector under test:

1. Run the detector on **noise-only IQ**.
2. Count candidates above each threshold across a fine grid.
3. Pick the threshold that produces the target FA rate (e.g., 10
   FA / 15 s).
4. Record the calibrated threshold and observed FA rate for the
   results CSV.

This step is what makes the comparison fair: each detector is set
to the **same operationally-observed FA rate**, so threshold-
convention differences don't bias the comparison.

## 5. Detection measurement

For each detector at its calibrated threshold:

1. Run on **signal-bearing IQ** (the synthetic ramp).
2. Match each candidate to a truth ping within time + freq tolerance.
3. Count detected pings per SNR bucket.
4. Record per-ping freq error, time error.

Detection counted = ≥1 candidate within (Δt, Δf) tolerance of a truth
ping.  No double-counting of one truth ping by multiple candidates.

## 6. Tolerances

What counts as a "match":

- Time tolerance: ±0.5 s (one MSK144 frame ≈ 0.072 s; 0.5 s is loose
  enough to absorb timing jitter).
- Frequency tolerance: ±500 Hz (one sq_det search window; for narrow
  comparisons, tighten to ±100 Hz).

Tolerances are part of the test config so they're explicit and
sweepable.

## 7. Metrics

For each detector × configuration:

- **Detection rate per SNR bucket** (primary).
- **Threshold at calibrated FA rate** (from §4).
- **Observed FA rate on noise-only** (sanity check vs target).
- **Candidate count on signal+noise** (operational cost indicator).
- **Frequency-error statistics** (mean, std, max-abs per SNR bucket).
- **Time-error statistics** (same).

## 8. Output format

Single CSV, one row per (detector, configuration, SNR bucket).
Columns:

```
detector, config_label, fa_target, fa_observed, threshold,
snr_bucket_lo, snr_bucket_hi, n_truth, n_detected,
freq_err_mean, freq_err_std, time_err_mean, time_err_std,
candidates_total
```

Plus a JSON sidecar capturing the full test config (input file
seeds, simulator parameters, tolerances).

## 9. Reproducibility

- Simulator seeds recorded in JSON sidecar.
- Detector code revisions recorded as git SHA.
- The protocol script accepts `--seed` and `--config` and produces
  deterministic output for the same inputs.

## 10. Reading the results

Standard plot: detection rate vs SNR_2.5k per detector, on the same
axes, with the calibrated FA rate noted in the legend.  This is the
single chart that answers "which detector is more sensitive at this
operating FA rate".

## 11. Where the code lives

`experiments/detection_compare/` — protocol implementation.

- `measure_detection_curves.py` — top-level driver: calibrates FA-matched
  thresholds on noise-only IQ, then sweeps SNR with N trials per bucket.
  Writes `detection_curves.csv`.
- `plot_detection_curves.py` — parametric plotter with `--algos`,
  `--metric`, `--decode-floor` flags.

Companion infrastructure (existing, reused):

- `experiments/gpu_tfmf/synthetic.py` — synthetic IQ generator
  (`single_ping`, `noise_only`) using WSJT-X 2500 Hz SNR convention.
- `experiments/gpu_tfmf/tfmf.py` — TFMF detector + surface helpers.
- `map144_app/sq_det.py` — WSJT-X-faithful sq_det implementation.

## 12. TODOs / open items

- **Streaming-mode performance.**  This protocol measures **batch
  sensitivity** — a full 15-s file processed offline.  Production
  TFMF must run in streaming mode with 1–2 s total latency.  A
  detector that wins on batch sensitivity but can't be made to fit
  the streaming budget is not a production candidate.  Streaming
  variants of each detector should be added to this protocol once
  the batch comparison is settled; the streaming variant is allowed
  to perform slightly worse than its batch counterpart, but must
  stay close (e.g., within 1 dB on detection floor).  See
  `detection_baseline.md` §6.5.
- **2D local-max suppression for TFMF.**  Currently the matched-filter
  surface produces multiple candidates per real signal (mainlobe
  spilling across multiple cells in time and freq).  Zero-padding
  for finer freq resolution is blocked on this — without local-max
  suppression, padding inflates the candidate count and eats any
  sensitivity benefit.
- **Polarization (dual H/V).**  Currently single-polarization only.
  Dual-pol combination (see CLAUDE.md "Polarization combination") adds
  another axis to the comparison.  Important to keep working — flag as
  a required extension before any architectural decision based on this
  protocol's results lands in production.  Note: no dual-pol antenna
  available for live testing at present, so initially synthetic-only.
- **Real-data corpus.**  Blocked on 48 kHz capture-with-truth pipeline
  (current capture is post-channelizer at 12 kHz).  Stub
  `detection_field_test.md` to be drafted when infrastructure exists.

## 13. Maintenance

This doc is updated when the test methodology changes (new metric,
changed tolerance, new detector under test).  Historical results in
`detection_results.md` cite the protocol version that produced them.
