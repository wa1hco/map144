# MAP144 — TODO

Canonical numbered backlog. Mirrors the in-session TodoWrite list. The
numbers are stable IDs referenced from
[`block-stream-design.md`](block-stream-design.md) and from project memory
under `~/.claude/projects/-home-jeff-ham-map144/memory/` — do **not**
renumber to close gaps (e.g. the missing `#19`).

Last updated: 2026-05-07 (Phase 3 #5–#9a landed; metric enrichment
landed-uncommitted; ML / clustering / classifier items added as #29–#39).

---

## Status legend

- ✅ `completed`
- 🔄 `in_progress`
- ⬜ `pending`

---

## ✅ Phase 2 — profiling & vectorisation (done)

1. ✅ Profile sync correlator and channelizer with py-spy during a noise
   burst (flame graph + speedscope timeline)
2. ✅ Vectorise sync correlator across 48 channels (single batched matmul
   instead of Python for-loop calling `_sync_correlate` 48×)
3. ✅ Confirm fast-graph gap-fill is gone after Flex-runtime
   `timestamp_int=0` fix; remove `spec_staging` gap-fill diagnostic
4. ✅ Numba `@njit` on `_sync_correlate` / `_sync_correlate_batch`
   (4.63 ms BLAS → 1.22 ms numba; doubled-buffer branchless inner loop
   SIMD-vectorises)

## ✅ Phase 3 — Block / Stream architecture (mostly done)

5. ✅ Design doc — typed-stream protocol, lifecycle, backpressure, glossary
   (commit `744ea2f` → [docs/block-stream-design.md](block-stream-design.md))
6. ✅ `Block` / `Stream` / `Runtime` / `Record` / `Event` / `CoherentPair`
   base classes; 25 tests (commit `744ea2f`)
7. ✅ Source block — `Source` base + `SyntheticSource` + `WavSource` +
   `JsonlSink` (commit `59478d5`); `FlexSource` wrapping
   `flexclient.FlexDAXIQ` (commit `4198a41`)
8. ✅ `ChannelizerBlock` wrapping `apply_channelizer` (commit `21ebca1`)
9. **DetectorBlock + Combiner**
   - 9a. ✅ DetectorBlock — sync correlator + squared-FFT detector + cluster
     gate; all four CLAUDE.md detection invariants tested (commit `c4db7cd`)
   - 9b. ✅ CombinerBlock — dual-pol merge at `ChannelStream` granularity
     (`H·cosθ + V·exp(jΔφ)·sinθ` per §6.1; static θ/Δφ from config; per-
     channel arrays accepted; commit `e4d2914`)

## ✅/⬜ Phase 4 — block-ify the rest

10. **DisplayBlock**
    - 10a. ✅ v1 scope: heatmap + marker overlay. `ChannelMetricRecord`
      stream landed in [types.py](../map144_app/blocks/types.py); new
      `metrics` output port on
      [detector_block.py](../map144_app/blocks/detector_block.py);
      [display_block.py](../map144_app/blocks/display_block.py) owns
      `_ch_snr_history` + markers + scrolling state. Boundary detection
      mirrors legacy bit-for-bit (UTC-grid period via wall clock,
      hop-count row index within period). 10 tests; commit `aea7086`.
    - 10b. ⬜ IQ waterfall + squared-domain spectrogram. Deferred —
      lands alongside `#11` Decoder which also taps `IQStream`.
    - 10c. ⬜ Decode panel + noise-floor curves + status bar. Needs
      `DecodeEventStream` from `#11` and uses `Block.stats()` for the
      status bar.
11. ⬜ DecoderBlock — consumes `DetectionEventStream` + `IQStream` window
    via Tap; runs SPD + jt9 fallback; emits `DecodeEventStream`
12. ⬜ ReporterBlock — durable consumer of `DecodeEventStream` →
    PSKReporter + DXcluster

## ⬜ Phase 5 — cleanup after the cut-over

13. ⬜ Delete squared-FFT detector path (after sync detector validated
    against multi-seed ramp run on new arch)
14. ⬜ Delete boundary-wipe state plumbing (`_ch_snr_boundary`,
    `_ch_snr_hop_count`) once Display block owns its own scrolling
15. ⬜ Delete orphaned `_drain_t0` / `_drain_samples` wall-clock accounting
    in Flex drain loop (bypassed by `timestamp_int=0`)
16. ⬜ Rename `hop` → `stride` / `tick` across the new architecture
    (legacy `_CH_DETECT_HOP` migrates as consumers move into blocks)

## ⬜ Detection tuning

17. ⬜ Tighten cluster gate to reject "many sync-only triggers in adjacent
    channels" as broadband noise (don't launch decode pool)
18. ⬜ Noise-floor-rise hysteresis — when global per-channel metric jumps
    abruptly across most channels, enter "noise-rise" state and require
    extra dB margin

> `#19` was misframed (sync-trigger-timing hypothesis disproven, see
> `project_todo19_misframed` memory) and dropped 2026-05-07. The number
> is intentionally left as a gap so existing `#20`+ cross-references
> stay valid. The real lever lives in `#20` below.

## ⬜ Sensitivity

20. ⬜ **Sensitivity Step 2** — frame-coherent integration in detection:
    expose `msk144_spd.py`'s coherent-averaging core as a detection-mode
    entry point (+3–7 dB)

## ⬜ Diagnostics & comparison

21. ⬜ Display Plot **B** — within-frame correlation trace (864-bin `xcc`
    array), click-to-inspect on any trigger
22. ⬜ Display Plot **C** — coherent-gain ladder (multi-frame integration
    heatmap rows); after `#20` lands
23. ⬜ Extend `run_ramp_tests.py` to pool decode rates across many noise
    seeds in one invocation (statistically meaningful S-curves)

## ⬜ Infrastructure

24. ⬜ Reimplement the flexclient subset MAP144 uses (~280 lines, drops
    6 transitive deps including selenium); revisit **after** Phase 3
    cut-over so the swap is contained behind one block boundary

## 🔄 Offline analysis & metadata enrichment

25. 🔄 Signal-category classifier in `compare_decoders.py` (TRUE_PING /
    WEAK_FADING / STRONG_LOCAL / MIXED). Groundwork landed via
    `analyze_launches.py` (commit `44ccc25`):
    `WIDEBAND_BURST / STATIONARY_SPUR / SUSTAINED / ISOLATED` rules
    characterise the 99 % no-decode population. `compare_decoders`
    integration still pending. See `project_signal_categories` and
    `project_launch_pattern_findings` memories.
26. ⬜ Enrich `decodes.jsonl` with parsed `tx_grid` + great-circle distance
    (<200 km LOCAL, 200–800 WEAK, 800–2200 PING, >2200 long-haul)
27. ✅ Per-launch metric enrichment in `launches.jsonl` — 11 fields
    landed in commit `1203338` (2026-05-08):
    `pair_metric_db_h/v`, `sq_metric_db_h/v`, `sync_metric_db_h/v`,
    `ch_pct25_db_h/v`, `ch_signed`, `n_cluster_chan`, `det_source`.
    `ch_pct25_db_*` is the legacy per-channel form of `#32`'s
    `background_db`. Heartbeat snapshots, `source_metrics` bag, and
    `source_type` field still pending — split out below as `#27b`.

  - 27b. ⬜ 15-s heartbeat snapshots for MAP144-quiet periods +
    `source_metrics` dict bag (Flex S-meter / USRP under/overrun /
    Airspy AGC) + `source_type` field. Requires source-side hooks;
    lands cleanly post-Phase-3 cut-over.
28. ⬜ Fall back to GridTracker SQLite (or qrz.com) for callsigns lacking
    a grid in the message; respect existing GridTracker session

## ⬜ Detection metrics, clustering, classification, ML (added 2026-05-07)

This block of items is the post-Phase-3 sensitivity / ML programme.
See `project_ml_qso_classifier_plan` memory for the full plan; see
`project_band_state_tracker` for the architectural home (`#32`); see
`project_launch_pattern_findings` for the dataset that motivated it.

29. ✅ **Sync-correlator phase-derived features** — 6 fields landed in
    commit `b23e9f2` (2026-05-08). Per polarisation:

    - `sync_phase_coherence_h/v` ∈ [0, 1] — coherent / noncoherent
      magnitude ratio at peak. Real MSK144 → ≈1; impulse noise →
      ≈1/√(2·42) ≈ 0.109. Validated on synthetic inputs.
    - `sync_t_sample_h/v` — sub-sample peak position via parabolic
      interpolation; bounded `ish_best ± 0.5`.
    - `sync_freq_offset_hz_h/v` — single-frame frequency offset from
      phase rotation between the two sync sub-blocks (336 samples / 28 ms
      apart). Aliases at multiples of ~35.7 Hz.

    Renamed from the memory's `sync_drift_hz_per_s` because this is a
    single-frame estimate, not cross-frame drift. True cross-frame drift
    requires SPD output and lands with `#11` Decoder — split out as
    `#29b`.

  - 29b. ⬜ Cross-frame phase drift `sync_drift_hz_per_s` over the SPD
    window. Requires SPD intermediate state to be exposed at decode time;
    lands with `#11` DecoderBlock or just after.
30. ⬜ **Sync-detector primacy for `fc_hz`** — when
    `det_source ∈ ('sync', 'sq+sync')`, take `fc_hz` from the
    sync-correlator peak (σ_f ≈ 5–15 Hz at typical detection SNR vs
    10–30 Hz for the squared-FFT path). Fall back to squared-FFT only
    when sync did not fire on this channel. Requires Phase 3 cut-over.
31. ⬜ **Fix asymmetric upper-IF noise-floor rolloff** — six channels at
    +14..+19 from pan center produce 28 % of all launches with 0 %
    decode rate, because pct25 there is artificially low (4.7 dB drop
    at +19 vs flat −80 dB baseline elsewhere). The asymmetry rules out
    symmetric Nyquist rolloff; cause is in the upper-IF chain (Flex
    DSP or MAP144 channelizer). The right fix is **NOT** channel-list
    hard-suppression (that would block real signals there) — it's
    cross-channel `pct25` reference, which is `#32` BandStateTracker
    Stage 2. Originally framed as "STATIONARY_SPUR hard-suppression"
    (analyze_launches.py 2026-05-07); reframed 2026-05-08 after
    user-driven diagnosis. See `project_launch_pattern_findings`
    memory for the pct25-by-channel data.
32. ⬜ **BandStateTracker** — single per-channel state object
    (background level + signal label + claim-based repeat-suppression)
    replacing the current pct25 / cooldown / sustained-lockout /
    cluster-gate machinery. **This is also the home for the shared
    noise / channel-background estimator** (renamed from the earlier
    "noise floor estimator" framing per
    [project_band_state_tracker:7](../../.claude/projects/-home-jeff-ham-map144/memory/project_band_state_tracker.md)
    — most of what it tracks is *other signals* on the band, not
    AWGN). API: `snr_db(ch)`, `background_db(ch)`, `signal_db(ch)`,
    `label(ch)`, `is_wideband_burst()`, `claim()`, `update()`.

    **Consumers (the "support calls by detection, decode, and ??"
    question):**
    - Detection — `snr_db(ch)` is the cluster-gate threshold reference
      (replaces today's per-channel `pct25`).
    - Decode — `background_db(ch)` for the WSJT-style SNR reported in
      decode events. Replaces
      [`_estimate_snr_db`](../map144_app/detection.py)'s private
      median-of-out-of-band-bins computation.
    - DisplayBlock (`#10`) — heatmap normalisation
      (SNR-above-background instead of raw squared-FFT power).
    - Logging / launch metrics (`#27`) — per-launch
      `ch_pct25_db_h/v` snapshot is already the legacy form of this.
    - ML classifier (`#33`–`#35`) — `background_db` and history are
      direct features.
    - Noise-floor-rise hysteresis (`#18`) — becomes a method on the
      tracker.
    - Reporter (`#12`) — downstream consumer of the SNR computed via
      Decode.

    4-stage migration: (1) drop-in wrapper around existing pct25 /
    cooldown, same algorithms, new API; (2) cross-channel median +
    label-based suppression replacing per-channel cooldown; (3) ML
    cluster classifier trained on `launches.jsonl` outcomes; (4)
    eventual 2-D blob detection. Direct path to closing the 45 %
    TRUE_PING / 17 % SPD gaps versus WSJT-X. Lands post-cut-over.
33. ⬜ **ML Stage A — xgboost / lightgbm baseline classifier** on
    `(launch_metrics → outcome=decoded)` from `launches.jsonl`. Output:
    calibrated `P(decode)`, permutation feature importance, ROC-AUC.
    Acceptance: AUC > 0.85 on held-out week. Sub-ms inference.
34. ⬜ **ML Stage B — HDBSCAN clustering of no-decode population** in
    standardised feature space. Goal: find natural clusters that don't
    match the hand-coded `WIDEBAND_BURST` / `SSB_VOICE` / etc. patterns.
    Where they disagree is where the rules are wrong.
35. ⬜ **ML Stage C — `cluster_id` feature retrain.** Add cluster ID as
    a feature, retrain. If AUC moves materially, cluster ID captured
    non-linear structure. If not, drop the feature (validates that raw
    features were sufficient).
36. ⬜ **Signature-aware `claim()` in BandStateTracker** — replace the
    simple `(channel, period)` claim with a per-period history of
    `(t_sec, fc_offset_hz, callsign, metric_db)` tuples. A re-launch is
    redundant only if it looks like the *same signal* (same burst
    within 0.5 s AND same `fc_offset_hz` within ~100 Hz AND not
    appreciably stronger metric). Preserves the second-sender-on-same-
    channel case at MS shower peaks.
37. ⬜ **ML Stage D — soft-then-hard gate rollout.** Wire the trained
    classifier into the runtime in **soft mode** first (predict per
    launch + log to `launches.jsonl` but don't gate). After two weeks
    of soft-mode data, choose τ that suppresses the bottom 50 % of
    launches at < 1 % real-QSO loss. Flip to **hard mode**.
38. ⬜ **ML Stage E — drift detection.** Weekly: re-cluster recent
    no-decodes; new clusters → environment shift → trigger retrain.
    Operational hygiene.
39. ⬜ **QSO-aware training labels** — recompute labels per the
    QSO-level loss function:

    ```
    positive  = first decode of a unique (callsign, period)
    positive  = decode of (callsign, period) when a DIFFERENT callsign
                already decoded in the same period (the multi-sender
                case we must keep)
    negative  = decode of the SAME (callsign, period) we already have
                (true redundancy)
    negative  = no_decode in a period where the QSO got caught by
                another launch
    ```

    Requires `compare_decoders.py` linkage to mark each launch with its
    QSO outcome.

---

## Sequencing — what blocks what

1. **Phase 3 cut-over** (`#9b` → `#10` → `#11` → `#12` plus §10
   validation gates in `block-stream-design.md`) — gives us the data
   flow architecture. Required before sync-detector primacy.
2. **Sync-detector primacy** (`#30`) — gives us the better `fc_hz` and
   the phase-derived features. Lands as part of the Phase 3 detector
   block work or just after.
3. **BandStateTracker Stage 1–2** (`#32`) — centralises per-channel
   state, replaces pct25 / cooldown machinery.
4. **ML Stages A–C** (`#33`, `#34`, `#35`) — train baseline classifier,
   discover residual clusters, validate.
5. **BandStateTracker Stage 3 + ML Stage D** (`#37`) — wire classifier
   into runtime as the cluster gate. Soft mode → hard mode.
6. **Stage E** (`#38`) — operational hygiene; retrain on drift.

The phase-derived features (`#29`) and the STATIONARY_SPUR rule (`#31`)
are independent and can land at any time.

---

## How this list is maintained

- The TodoWrite list in the active session is the working copy. It is
  refreshed from this file at session start when memory recall surfaces
  the pointer.
- This file is the persistent canonical copy. Update it whenever an
  item changes status in a way that should survive session loss
  (especially `pending → in_progress → completed` transitions tied to
  commits).
- Cross-references in `docs/block-stream-design.md` and in project
  memory entries use the `#N` IDs from this file. Do **not** renumber
  to close gaps — `#19` is intentionally absent.
