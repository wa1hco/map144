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
    - 10b. ✅ IQ waterfall — `_sbuf` accumulator + Hanning FFT writes
      `spec_staging` / `spectrogram_data` matching legacy semantics;
      commit `5eb4775`.  Squared-domain spectrogram remains for follow-up
      (needs DetectorBlock to expose squared-FFT power as a separate
      stream).
    - 10c. ✅ Decode panel — DecodeEventStream → rolling
      `decode_panel_entries` list with same fields the legacy
      decode_panel widget consumes; commit `d0b8472`.
11. ✅ DecoderBlock — `_DecodeQueueShim` adapts `decode_queue.put`
    interface; daemon worker pool with semaphore; ring buffer; sample-rate
    rebase from detection-event clock to IQ clock; commits `23a55d2`
    (Stage 1 skeleton) + `743895d` (Stage 2 wiring) + `4a1b698`
    (rate-conversion bug fix found by Stage 1 e2e test).
12. ✅ ReporterBlock — durable consumer of `DecodeEventStream` →
    PSKReporter + DXcluster + WSJT-X UDP; pre-configure settings buffering;
    commit `b5e6d13`.

## ✅ Phase 4 §10 — pre-cut-over validation gates (all green)

13.5 Functional / parity / smoke gates that must pass before the
    cut-over commit lands.  All green as of 2026-05-09.

    - **Stage 1** (`commit 4a1b698`,
      [tests/test_block_pipeline_e2e.py](../tests/test_block_pipeline_e2e.py)):
      synthetic MSK144 IQ → full Block graph (Source ▶ Tee ▶ Channelizer
      ▶ Detector ▶ Decoder ▶ Reporter+JsonlSink) → decoded "CQ K1JT FN20"
      via SPD.  Introduces TeeBlock (1-to-N fan-out primitive); fixes
      sample-rate-conversion bug in DecoderBlock (events at 12 kHz →
      IQ ring at 48 kHz; production-blocking).
    - **Stage 2** (`commit a5c23eb`,
      [tests/test_block_pipeline_replay.py](../tests/test_block_pipeline_replay.py)):
      same IQ → both pipelines decode the same (msg-token, freq ±1 kHz)
      pair.  Result: shared = ['CQ', 'FN20', 'K1JT'] @ 50265 kHz.
    - **Stage 3a** sensitivity smoke (2 seeds × 1 SNR): Block ≥ legacy.
      Result: legacy=1, block=2.
    - **Stage 3b** CPU regression bound: Block / legacy < 3×.  Result:
      0.86× (Block slightly faster than legacy on the strong-ping case).
    - **Stage 3c** drift injection: pipeline decodes under 1000 ppm
      injected drift.  Confirms sample-counter ring is drift-immune.
    - All Stage 3 in `commit a7d5a21`,
      [tests/test_block_pipeline_stage3.py](../tests/test_block_pipeline_stage3.py).

## ✅/⬜ Cut-over commits

13.7 Switch `engine.process_iq_data` from the legacy `_hop_loop` in
    [processing.py](../map144_app/processing.py) to a Runtime-driven
    Block graph.  Three-stage rollout per
    [project_phase3_progress](../README.md) memory:

    1. ✅ **Behind a feature flag** (`MAP144_USE_BLOCKS=1` env var or
       `Engine.use_blocks=True`).  Lowest risk; legacy still authoritative
       for GUI/decode reporting; Block path runs in parallel and writes
       shadow JSONL to `MSK144/detections/blocks/decode_events.jsonl` for
       offline A/B.  Default off = zero behaviour change.  Lands in
       commit `294169f`; 6 new tests in
       [tests/test_engine_block_shadow.py](../tests/test_engine_block_shadow.py).
       **Live bake-in pending** before option 2 lands.
    2. ⬜ **Replace detect+decode only** — when flag on, BYPASS the
       legacy detect/decode portion of `process_iq_data` (channelizer
       + hop loop + extract_and_decode dispatch).  Block path becomes
       primary; updates the same `_jt9_markers` / `decode_panel` /
       `_decode_queue` the GUI reads via a small adapter.  Legacy noise
       blanker + spectrogram still run inline.  Lands AFTER bake-in.
    3. ⬜ **Full replacement** — move noise blanker into a
       `NoiseBlankerBlock` and the spectrogram into its own block;
       replace `process_iq_data` wholesale.  Cleanest architecturally;
       lands as part of Phase 5 cleanup (#13–#16).

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

19. deleted

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
31. ✅ **Nyquist mask off-center vs IF Nyquist (USB-convention offset
    bug)** — the legacy detection gate at
    [processing.py:891-894](../map144_app/processing.py#L891-L894) used
    `_nyquist_ch = N_CHANNELS // 2 = 24` as the Nyquist channel center.
    That assumes the channel grid is centered on IQ DC, which it
    isn't: the USB-convention offset
    `channel_offset_hz = (calling − pan) + 1500` shifts the grid by
    1.5 channels (typical pan=calling case). The actual IF Nyquist is
    at `ch_k = (sample_rate/2 − channel_offset_hz) / CHANNEL_SPACING_HZ
    = 22.5`, **not 24**.

    Consequence: with `_EDGE_CH_SKIP = 4` the legacy mask covered
    ch_k 20..28 — IF +20.5..+23.5 kHz on positive (4 channels) and
    IF −23.5..−18.5 kHz on negative (5 channels). The negative-side
    rolloff zone was fully masked, but the **positive-side rolloff
    bin `ch_k = 19` (IF +20.5) was unmasked**, producing the ~6×
    false-trigger rate at `ch_signed +19` (1666 launches in the
    overnight 2026-05-08 dataset, 0 % decode).

    Fix landed: dynamic Nyquist channel based on actual
    `channel_offset_hz`, with wrap-aware distance computation. New
    mask covers ch_k 19..26 (IF ±19.5..±23.5, symmetric around the
    real IF Nyquist).

    Was originally framed (2026-05-07) as "STATIONARY_SPUR
    hard-suppression" of channels 50274–50279, then (2026-05-08
    intermediate) as "asymmetric IF rolloff, cause unidentified".
    Final root cause: the IF rolloff IS symmetric in absolute
    frequency; the launch-count asymmetry came entirely from the
    mask being centered on the wrong channel.

    See `project_launch_pattern_findings` memory for the diagnostic
    data and the toggle (`Diagnostics → Show raw power`) used to
    visualise the underlying symmetric rolloff.
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

40. ⬜ **Polyphase channelizer prototype FIR stopband ripple** —
    surfaced by the `Diagnostics → Show raw power` toggle (commits
    `cecce87` / `5358177` / `3272457`). With the rolling pct25
    normalisation bypassed, the per-channel raw spectrum shows ~7
    horizontal "bands" between channels separated by 1-channel-wide
    darker lines, with **non-uniform widths and spacing** in both
    the squared-FFT and sync detection panels. Source is upstream
    of both detectors — the channelizer's per-channel output itself.

    User-confirmed evidence (2026-05-08): the amplitude of the
    energy in the darker bands **varies with nearby-frequency
    activity**. That's the fingerprint of stopband leakage: a
    strong signal at one channel leaks into others through the
    prototype FIR's imperfect stopband, and the leakage amount
    depends on (a) where the signal sits on the FIR's stopband
    ripple curve and (b) its amplitude. Channels at FIR ripple
    notches see less alias (darker), channels at ripple peaks see
    more.

    Why it matters: cross-channel noise-floor variation that today
    is partially absorbed by per-channel pct25. With `#32`
    BandStateTracker Stage 2's cross-channel pct25 reference, the
    variation becomes a *bias* against ripple-peak channels (their
    noise floor looks higher than the cross-channel median, so the
    threshold sits above their actual noise → reduced sensitivity
    in those channels). Worth flattening at the source.

    Fix path: redesign the channelizer's prototype FIR. Trade-offs
    are filter length (more taps = sharper transition + more CPU)
    vs stopband attenuation depth + ripple amplitude (Parks-McClellan
    or Kaiser-window design with target stopband floor). Today's
    prototype is in [channelizer.py](../map144_app/channelizer.py);
    measure its actual stopband response first to set targets.

    Operationally this is lower priority than `#20` (sensitivity
    Step 2) and the `#33`–`#39` ML programme — it's a second-order
    cleanup that doesn't generate false triggers, just causes
    cross-channel sensitivity bias. Worth doing before `#32`'s
    cross-channel pct25 ships, since otherwise the cross-channel
    median has to compensate for FIR-induced bias instead of just
    real-world noise variation.

41. **Burst-context features for ML / runtime gating**
    - 41a. ✅ `n_chans_300ms` and `n_chans_2s` per-launch features
      added to launches.jsonl (commit `e4caf43`). Discriminate
      in-burst, post-burst-rebound, and quiet-baseline launches
      from per-launch context alone (no transition tracking
      needed at consumer side). See `project_post_burst_rebound`
      memory for the 100× post-burst spike that motivated them.
    - 41b. ⬜ Optional runtime hard gate on these features:
      "suppress launches with `n_chans_300ms ≥ 8` OR
      (`n_chans_2s ≥ 8` AND within 1 s of a wideband burst-end
      transition)." Catches ~225 launches/day on 2026-05-08 data
      at zero decode cost. Hold for ML soft-mode rollout (`#37`)
      so the classifier learns the rule before we hard-code it.
    - 41c. ⬜ Principled DSP fix: exclude blanked samples from
      the pct25 update path in `_compute_sq_metric_db` so the
      noise blanker stops biasing the noise-floor estimator —
      addresses the *cause* of the burst-end adaptive transient
      rather than just recognising it. ~20 LOC.
    - 41d. ⬜ Verification of the blanker→pct25 mechanism:
      requires `nb_blanked_pct` per launch (TODO `#27b`
      heartbeat). Run correlation analysis once available.

42. ⬜ **Period-mode decode for weak-signal propagation modes** —
    user-stated 2026-05-08, see `project_propagation_modes` memory.
    MSK144 reception involves multiple modes (true MS pings,
    airplane scatter, forward scatter, tropo, Es) with very
    different time scales. Only true MS pings have signal
    localised to a 1.7-s burst window. The other modes have
    signal **continuously** across the full 15-s WSJT period, often
    individually below noise — a ping is just a brief 10–20 dB
    enhancement that crosses the detection threshold. WSJT-X
    catches these because `jt9 -p 15` integrates over the full
    period; MAP144's 1.7-s decode window captures only ~10 % of
    the available signal and misses the decode.

    Hypothesis: most of the ~70 WSJT-X-decoded NO_DECODE events
    MAP144 misses per day (per the 2026-05-08 compare run) are
    weak-signal-mode reception. Their per-launch coherence median
    is 0.379 (within the noise distribution), exactly matching
    the weak-signal signature.

    Strategy: **detect with ping, decode over period**. The ping
    triggers a launch and identifies *which channel* and *which
    period*. Decode-mode fans out to:
    - **Burst mode** (1.7 s, current): always run, cheap, catches
      strong MS pings.
    - **Period mode** (15 s, new): run after burst-mode fails, at
      end-of-period boundary; integrates the full 15 s to recover
      below-noise weak-signal-mode messages.

    Period-mode dedup: at most once per `(channel, period)` —
    multiple pings from one event (tropo opening, fading airplane)
    don't trigger redundant period-mode attempts. Mirrors `#36`
    signature-aware claim mechanism.

    CPU budget: period mode is ~10 × burst-mode CPU. Affordable
    only after `#31`, `#32`, `#41`, and the ML programme have
    suppressed 80–90 % of hopeless launches. The free CPU is
    exactly what funds period-mode integration.

    Architectural impact absorbed into `#11` Stage 2:
    DecoderBlock IQ ring sized for 15 s (was 5 s) so period-mode
    can read the full window when the time comes.

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
  to close gaps — `#19` is deleted.
