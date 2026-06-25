# MAP144 — TODO

Canonical numbered backlog. Mirrors the in-session TodoWrite list. The
numbers are stable IDs referenced from
[`block-stream-design.md`](block-stream-design.md) and from project memory
under `~/.claude/projects/-home-jeff-ham-map144/memory/` — do **not**
renumber to close gaps (e.g. the missing `#19`).

Last updated: 2026-06-02 (#25a objective envelope-morphology metrics added —
ping/flutter/fade, the principled replacement for the subjective buckets;
#43 Phase 1: documented use-after-free mechanism REFUTED, paused for
catch-in-the-wild; #44 WSJT-X per-RF-port audio bridge landed; #43 also
gained a live-path repro from the power-outage recovery).
Earlier: 2026-05-30 (#43 WAV-open paint-segfault diagnosed + filed);
2026-05-07 (Phase 3 #5–#9a landed; metric enrichment landed-uncommitted;
ML / clustering / classifier items added as #29–#39).

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
    2. ✅ **Replace detect+decode only** — LANDED (`28b6ed0`). With the flag
       on, the legacy hop loop computes the GUI sq/sync displays but SKIPS
       detect+decode ([processing.py:1600-1610](../map144_app/processing.py#L1600-L1610));
       the Block runtime's DetectorBlock + DecoderBlock own those, and
       `_EngineBridgeSink` ([engine.py:86](../map144_app/engine.py#L86)) forwards
       block decodes into `engine._decode_queue` in the legacy shape so the GUI /
       reporter are unchanged. e2e gate: `tests/test_engine_block_primary.py`.
       **Still default-off + live-bake-in pending** before flipping the default
       (operator-gated; needs an A/B block-vs-legacy decode-count run). Legacy
       noise blanker + spectrogram still run inline under the flag.
    3. 🔄 **Full replacement** — move noise blanker + spectrogram into blocks,
       then replace `process_iq_data` wholesale (Phase 5 cleanup #13–#16).
       - ✅ **`NoiseBlankerBlock`** (2026-06-25) — wraps the `noise_blanker`
         backends (Linrad/Bypass/NR0V) as a Block; mono v1; passes a minimal
         `_BlankerState` holder instead of the Engine (narrower interface), seeded
         identically to the Engine's `_nb_*` init so it's bit-equivalent at
         cut-over; `blanked_fraction` in output metadata for the #41c gate. Not
         wired into a graph. 6 tests in `tests/test_noise_blanker_block.py`.
       - ⬜ **SpectrogramBlock** — the block `DisplayBlock` already owns the IQ
         waterfall (#10b), so this is mostly consolidating the legacy inline
         spectrogram, not net-new DSP. Check overlap with DisplayBlock first.
       - ⬜ Delete `process_iq_data` wholesale — **only after bake-in + default-on**
         (don't delete the still-default path).

## 🔄 Stream-oriented refactor — de-monolith `process_iq_data` (active, 2026-05-31)

Supersedes the stalled big-bang cut-over (§13.7 above): that step had no
incremental payoff, so it kept losing to detection work. Re-sliced so each step
ships a standalone, testable win. **Policy:** a legacy-path bug triggers the
matching slice instead of an in-place patch (CLAUDE.md "Refactor / migration
policy"). Built on the ratified dual-clock contract (block-stream-design §3.5).

- R1. 🔄 **`TimeBase` — dual-clock value object owned by the Source.** Captures
  the independent `(sample_counter, wall_clock)` pair per §3.5 (never derives one
  from the other, never slews either). Per-period anchoring: `mark_period_start(
  sample)` reads one synchronized `(period_start_sample, period_start_wall)` pair
  at each 15-s boundary; `utc_at(event_sample) = period_start_wall + (event_sample
  - period_start_sample)/rate` (sample-precise within the period, fresh UTC per
  period, no cross-period drift). Also `period_index`. One reset point (source
  open) with an explicit per-source anchor policy (radio = `time.time()`; WAV = a
  chosen policy, not ≈0 by accident). Removes the scattered
  `_iq_t0_wall` / `_loop_wall` / `_pkt_time*` arithmetic and the
  `source_mode=='wav'` time branches from the DSP core. **Acceptance test: the
  1970 TFMF-timestamp bug cannot recur** (WAV→radio transition; long-run drift).
  Closes that bug structurally rather than by patch.

  **Status (2026-06-25) — acceptance MET; residual legacy sweep deferred.**
  - ✅ `TimeBase` class landed pure + tested (`009ca0f`,
    [timebase.py](../map144_app/timebase.py)); 11 tests incl. the 1970-regression
    + 11-h no-drift acceptance tests (`tests/test_timebase.py`) — all green.
  - ✅ Wired into the engine and **re-anchors on every source open** (`caaa7e8`),
    so a WAV-relative anchor can no longer leak into a live run. The 1970 bug is
    structurally fixed and verified (headless WAV replay stamps 2026, not 1970).
  - 🔄 **Residual sweep NOT done:** TimeBase currently feeds only the TFMF
    candidate stamp. The scattered `_iq_t0_wall` / `_loop_wall` / `_pkt_time*`
    arithmetic still drives the main detect + display + decode-period path
    ([processing.py:1017-1028](../map144_app/processing.py#L1017-L1028),
    [:1113-1122](../map144_app/processing.py#L1113-L1122),
    [:1243-1249](../map144_app/processing.py#L1243-L1249),
    [:2082-2103](../map144_app/processing.py#L2082-L2103)). Completing it is a
    real design slice, not a mechanical variable-swap — two findings from reading
    the consumers (2026-06-25), both verified in code:
    1. **Live time is not always `_iq_t0_wall`-derived.** For a GPS-locked source
       (`timestamp_int >= 1e9`) `_pkt_time_early` is the **hardware VITA timestamp
       used directly**; the `_iq_t0_wall` derivation is only the fallback for
       unlocked clocks (Flex TSI=0, RTL/Airspy w/o GPS). The current TimeBase
       wiring passes unconditional `time.time()`, so routing live display/decode
       through `utc_at()` blind would **regress GPS-locked timing to OS-clock
       accuracy**. Correct design: Source hands TimeBase the *hardware* timestamp
       as the per-period wall when locked, `time.time()` only as fallback (§3.5
       "Source reads two clocks").
    2. **WAV replay already runs 3 inconsistent time conventions:** display +
       decode-period assignment is 0-based file-relative (`_wav_time_cursor` from
       0, [runtime.py:823](../map144_app/runtime.py#L823),
       [:971-974](../map144_app/runtime.py#L971-L974)); the TFMF stamp is
       *wall-clock-at-replay-start* (the `caaa7e8` `reset(wall=time.time())` fires
       for WAV too, [processing.py:1271-1273](../map144_app/processing.py#L1271-L1273));
       and **neither recovers the capture's true recording UTC**. The sweep must
       pick one. WAV source files are arbitrary-length IQ captures (not 15-s
       period files, not the 2–5 s jt9 snippets), replayed as a continuous stream.
  - **WAV anchor decision (operator chose 2026-06-25, then refined on reading
    the spec test):** WAV has **two intentional time roles that must NOT be
    collapsed** — discovered via
    [test_timebase_wiring.py:61-63](../tests/test_timebase_wiring.py#L61-L63),
    which deliberately asserts the WAV TimeBase anchors to replay-now wall-clock
    (so outward stamps read 2026, not 1970):
    1. **Internal period grid** (display + decode-period assignment) = 0-based
       file-relative — correct/needed (WSJT WAVs start on a 15-s boundary, so a
       0-based grid keeps replay periods aligned to file content; a wall-clock
       grid would split a ping's 15-s period across two display windows).
    2. **Outward timestamps** (TFMF candidates, decode reports) = replay-now
       wall-clock — deliberate, avoids 1970-looking logs.
    "Make the TFMF WAV stamp 0-based" (an earlier mis-statement of option A)
    would regress role 2 to ~1970 and break that test — rejected.
    **Decision: migrate LIVE path only; leave WAV untouched** (0-based grid +
    wall-clock outward stamp, both as-is). True recording-UTC for WAV (role 3,
    currently absent) deferred to a future slice if #26/#48 geometry forces it.
  - ✅ **Live-only sweep landed (2026-06-25).** Live `_pkt_time_early` / `_pkt_time`
    (→ `_loop_wall`, `_sbuf_t0`, heatmap + spectrogram period grid) now derive from
    `timebase.utc_at()`; TimeBase is anchored + period-re-paired at the top of
    `process_iq_data`; the Engine `_iq_t0_wall` single-anchor **and** the 30-s
    drift-guard heuristic are deleted (`_iq_t0_wall` ≡ `utc_at(0)`). Per-period
    re-peg (one `time.time()` read/period) replaces the drift guard — PC UTC is
    only ms-accurate so no finer coupling is pursued; GPS-locked VITA-as-wall
    deliberately skipped (Flex is TSI=0). **WAV untouched** (0-based grid +
    wall-clock outward stamp — its two roles). Decode epoch unaffected
    (`extract_and_decode` is called with `iq_t0_wall=None`; epoch uses `detect_ts`
    = `datetime.now()`). Tests: 6 in `test_timebase_wiring.py` (live-via-utc_at,
    WAV-stays-0-based, attribute-retired) + full `tests/` green (283 pass; the
    lone `test_sources` drift failure is the documented load-flake, passes alone).
  - **Still open (future R1/R2 slices):** (a) optionally feed `extract_and_decode`
    `timebase.utc_at(detect_sample)` to activate the sample-accurate PERIOD_SLIP
    epoch (currently dormant — behavior change, do under its own test); (b) WAV
    true-recording-UTC if #26/#48 geometry needs it; (c) R2 Source-emits-records.
- R2. ⬜ **Source emits timestamped sample-records.** Every record carries both
  clocks as metadata (§3.5 "records carry both anchors"); downstream stages read
  time from the record, never re-derive. Deletes per-section time recomputation
  in `process_iq_data`.
- R3. ⬜ **Split `process_iq_data` into stream stages** (blank → ring/buffer →
  channelize → detect → display), each consuming/producing typed records. This is
  the real cut-over (folds §13.7 option 2/3); lands stage-by-stage behind the
  existing shadow/parity gates (Phase 4 §10), not all at once.
- R4. 🔄 **TFMF as a block — block-only, never on the legacy path.**
  - R4.1 ✅ **`TFMFDetectorBlock`** (commit `3459194`): consumes wideband IQ
    Records, accumulates one period, runs `_build_tfmf_display_surface`
    (GPU/CPU), emits `tfmf_candidate` Events with §3.5 period-anchored
    dual-clock stamps. 4 unit tests (compute stubbed). Not yet wired into a
    graph.
  - R4.2 ⬜ **Graph-builder `(mode, gpu)` topology switch.** Capability probe →
    topology: CPU-parity graph = `Channelizer → Detector(sq+sync)`; GPU-beat
    graph = `TFMFDetectorBlock` (drops channelizer + sq_det — matched filter is
    linear, no intermod, per `experiments/gpu_tfmf/README.md`). Wire R4.1's
    block into the graph here.
  - R4.3 ⬜ **Device-residency.** Keep IQ on the GPU across GPU blocks;
    host↔device transfer only at ingress (upload IQ) / egress (download the
    small candidate set).
  - R4.4 ⬜ **GPU-vs-CPU parity gate** — first-class shadow test.
  - **Forward intent (beat-WSJTX phase):** TFMF/GPU eventually owns *frame
    selection* — finding the frames worth coherent integration, and finding
    multipath frames that can be combined (multi-Doppler). Draw the R4 block
    boundaries so they don't foreclose that.

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

    **25a. ⬜ Objective envelope-morphology metrics (operator 2026-06-02 — the
    principled replacement for the subjective geometry buckets above).** The
    TRUE_PING/WEAK_FADING/STRONG_LOCAL labels are subjective and not tied to
    P(decode).  Replace them with measurements of the per-period amplitude
    *envelope* `E(t)` at the detected `fc` (per channel), which directly encode
    the propagation mode.  Three morphologies and their measurable features:

      1. **Ping** — single rise/peak/fall hump.  `rise_time_s`, `fall_time_s`,
         `peak_width_s` (FWHM), `peak_count` (=1), rise/fall asymmetry (classic
         MS ping = fast rise, slower exponential decay as the trail diffuses).
         Short duration (~0.05–2 s).
      2. **Flutter** — sustained above noise with roughly sinusoidal / Rayleigh
         amplitude variation.  Defining feature: envelope-modulation rate in the
         **0.5–3 Hz** band (FFT of `E(t)`) -> `flutter_rate_hz`, `flutter_depth_db`
         (peak-to-trough).  Long duration.  (Aircraft scatter / moving-reflector
         multipath Doppler.)  **Flutter is NOT mode-unique:** a strong *local*
         beamed at you flutters from short-path multipath too (e.g. W1NB FN43,
         56 km, +24 dB — grid-adjacent to FN42, definitely not a ping), so
         flutter must be read jointly with range to separate local-multipath
         (tens of km) from aircraft/tropo (~hundreds of km).  Argues for joint
         range x morphology clustering over morphology-only labels.
      3. **Fading** — long-duration fade-in/peak/fade-out over seconds, possibly
         several humps per 15 s.  `env_duration_s`, `peak_count` (1–few), dominant
         modulation **< 0.5 Hz** (slow), broad `peak_width_s`.  Long duration =>
         unlikely a real ping.  (Tropo / Es slow fading.)

    Shared envelope features (per period, per candidate): `env_duration_s` (time
    above noise / fraction of 15 s), `peak_count` (prominence-thresholded local
    maxima), `peak_snr_db`, per-peak rise/fall/width, `env_mod_rate_hz` +
    `env_mod_depth_db` (envelope-modulation spectrum), Rayleigh-fit of the
    amplitude distribution (fading vs deterministic).  Discriminators collapse to
    **duration x peak_count x peak-shape x modulation-rate-band** (ping =
    transient, flutter = 0.5–3 Hz, fade = < 0.5 Hz).  Compute from the per-period
    detection-metric time series (sq/sync vs t) or the period IQ at `fc`; the
    TFMF period surface already carries most of it.  These are objective,
    P(decode)-relevant features that let the **cluster analysis (#34 HDBSCAN)
    define the categories** instead of hand-labels — feeds the ML track (#33–#37).

    **Operator research question (dedicated analysis, needs #26 range):** tag each
    decode with morphology (ping / flutter / fade) AND range (great-circle from
    `tx_grid`), then compare the **range distribution and count of ping-like vs
    faded signals** — *do meteor pings reach farther than tropo / aircraft-scatter
    (sustained/fading) signals?*  Physical prior: MS trails at ~100 km favour
    ~500–2000 km single-hop, while tropo/Es and aircraft scatter have different
    range envelopes, so the range x morphology joint distribution should separate
    them.
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
    - 41c. ✅ Principled DSP fix landed: blanker activity on a chunk
      sets `_skip_pct25_update_h` / `_v` flags on the Engine.  The
      per-hop loop checks the flag and skips the `_metric_hist_buf`
      append when set (pct25 recompute still runs on existing
      history so detection thresholding stays current; only the
      *update* is gated).  Same gate plumbed through
      `_compute_sync_metric_db` for the sync-correlator path.  4
      regression tests in `tests/test_pct25_blanker_gate.py` cover:
      clean chunks grow history, blanker-firing chunks freeze it,
      LinradBlanker on quiet noise doesn't false-trigger the gate,
      and the same flag propagates to the sync-correlator history.
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

43. ⬜ **Segfault: pyqtgraph zero-copy use-after-free on paint when
    opening a WAV** — intermittent `Fatal Python error: Segmentation
    fault` shortly after `Loaded WAV source`. Diagnosed 2026-05-30.

    **Mechanism (consistent with evidence; code-grounded, not yet
    caught in a live native frame):** `process_iq_data` *reassigns*
    the spectrogram buffers at each 15-s boundary — `self.spectrogram_data
    = self.spec_staging.copy()` and `self.spec_staging = np.full(...)`
    (`processing.py:2052`, `:2055`; V variants `:2062-2063`). The GUI
    thread paints from that same attribute via
    `self.spectrogram_img.setImage(self.spectrogram_data)`
    (`displays.py:232-246`). pyqtgraph's ImageItem can hold a **zero-copy
    raw C pointer** into the numpy buffer for deferred QImage rendering;
    if the worker thread replaces the Python array object while Qt is
    mid-paint, the old buffer is freed under Qt → dangling pointer →
    segfault on the paint event. This exact hazard is already documented
    in `_reset_wav_timeline` (`runtime.py:741-744`), which deliberately
    fills in-place (`[:]`) to avoid it — `process_iq_data` does **not**.

    **Verified:** offscreen Qt (`QT_QPA_PLATFORM=offscreen`) never
    crashes (no paint); 5 live `:1` repros across Flex-connected,
    all-windows-open, and exact no-Flex conditions ran clean → it is an
    intermittent paint-timing race, not the DSP path and not Flex
    teardown (`FlexDAXIQ.stop()` after a failed `start()` is a no-op —
    all native handles still `None`, `client.py:275-281`).

    **2026-06-01 — live-path repro (corrects the "5 live repros ran
    clean" note above):** the post-power-outage B210 recovery process
    segfaulted in the *same* paint path — faulthandler traceback
    `pyqtgraph try_make_qimage → ImageItem.render → paint → paintEvent →
    map144.py:241 main` — while running the **live B210 source**, not
    WAV-open. So the period-boundary spectrogram reassignment in
    `process_iq_data` exposes the use-after-free on the **live path too**;
    WAV-open is a frequent trigger, not a precondition. This raises the
    priority of the in-place / snapshot fix over the WAV-only framing.
    (Captured in the session recovery log; 6th repro overall.)

    **Precondition that exposed it 2026-05-30:** power outage left the
    Flex powered off; startup auto-restores the last source mode
    (`ui.py:798`), `'radio'` → auto-select Flex → fails "No FlexRadio
    found" → operator then opened a WAV, routing a fresh process straight
    into the WAV-paint path. Normally (Flex on) the restored source
    streams Flex and the WAV path isn't entered at startup.

    **Fix direction (SUPERSEDED — see Phase 1 below):** make the
    period-boundary spectrogram updates in-place like `_reset_wav_timeline`
    already does, or have the GUI `setImage` copy / hold a snapshot ref.

    **Phase 1 (2026-06-02) — mechanism REFUTED; paused, catch-in-the-wild.**
    Instrumented `pyqtgraph.ImageItem.setImage`/`paint`
    (`scratch/paint_probe.py`). Finding (verified): **every** data-bearing
    `setImage` hands pyqtgraph an array that OWNS its data (`base is None`,
    a fresh `np.repeat`/`fftshift` copy) on the **MainThread** — never a
    VIEW into a worker buffer. So the worker reassigning `spectrogram_data`
    frees nothing pyqtgraph paints from: **the use-after-free-on-reassign
    story cannot occur, and the in-place fix above would NOT cure it.** Do
    not apply it blind. NaN/inf-into-`makeARGB` hypothesis also tested —
    not present in the images that fired. Real cause still unknown (needs
    the native C frame; the bug is rare). **Strategy:** catch it in
    production — run the real app under gdb via
    `scratch/catch_segfault_probe.sh` (self-tagging probe). On the next
    natural crash: native frame → `scratch/gdb_segfault_probe.log`, culprit
    image → `scratch/last_paint.txt`, setImage history →
    `scratch/paint_probe.log`. Offscreen Qt does paint but doesn't repro;
    `Xvfb` not installed → can't auto-hunt headless.

44. ✅ **WSJT-X audio bridge — per-RF-port PipeWire sinks, B210 default-on,
    lifecycle tracks source** (2026-06-01, commits `5d34eca` + earlier
    `a883214`). One null-sink per B210 RF port (`map144.RF0.rx` /
    `map144.RF1.rx`), created by default on Linux when the B210 starts
    (opt out `MAP144_WSJTX_AUDIO=0`); RF0/RF1 = hardware ports, not H/V.
    Read-only tap of each port's calling channel; PipeWire owns the
    12→48 k resample + clock-drift absorption. Naming: dotted machine
    name (GUI-invisible) + human `device.description` "MAP144 RFn RX ->
    WSJT-X" (WSJT-X Input = its monitor) + `paplay --stream-name` labels
    the producer in pavucontrol. Lifecycle: torn down whenever the B210
    is no longer the active source (incl. the push-based Flex path via a
    central guard in `run_radio_source`); each sink unloaded on close (no
    `object.linger`) so a dead bridge disappears instead of going silently
    mute. See `project_wsjtx_audio_bridge` memory; 9 tests in
    `tests/test_wsjtx_audio_export_setup.py`.

45. ⬜ **Cross-platform WSJT-X audio bridge (Windows/macOS via PortAudio).**
    The #44 bridge is Linux-only: only two steps are OS-specific — creating
    the virtual sink (`pactl`) and playing into it (`paplay`). Everything
    else (PCM gen, DC→1500 Hz upconvert, auto-level, drop-queue, per-port
    routing in `WsjtxAudioExporter`) is portable numpy. Plan: abstract a
    `Transport` in `wsjtx_audio_export.py` —
    - Linux: keep the auto-created null-sink + paplay (zero user setup).
    - Windows/macOS: `PortAudioTransport(device_name=...)` via `sounddevice`,
      writing PCM to a user-installed virtual cable. `feed()` and the DSP stay
      byte-for-byte identical.
    User-side prerequisites (can't create audio devices without a kernel
    driver): **Windows = VB-CABLE** (VB-Audio; free single cable, or VB-CABLE
    A+B / VoiceMeeter for two ports → RF0=Cable A, RF1=Cable B); **macOS =
    BlackHole**. Add a device-name config setting (no auto-discovery off
    Linux). Bounded effort: transport plug-in + config; the hard part (the
    audio) is done. Deferred — surfaced 2026-06-02 while shifting MAP144
    testing to Windows (the program runs there already, just without this
    feature). See `project_wsjtx_audio_bridge` memory.

## 🔄 External spot correlation & propagation geometry (added 2026-06-03)

External reception-report feeds give MAP144 something it cannot get from its
own antenna: **independent witnesses** of what was on the air. Used two ways
(see `project_signal_categories` and the geometry thesis below):
the *presence oracle* (Role A — was there anything to hear on 2 m overnight,
to separate "deaf hardware / high NF" from "empty band"), and the *independent
mode label* (Role B — a spot's simultaneous multi-receiver footprint
corroborates the #25a envelope-morphology guess: MS = brief/narrow/idiosyncratic
footprint, Es/tropo = broad simultaneous regional footprint, aircraft = narrow
moving footprint). Capture raw now, join offline later — consume the shared
feeds exactly once.

46. ✅ **PSKReporter MQTT spot logger** (landed 2026-06-03, `tools/pskr_logger.py`).
    Passive subscriber to the live feed (`mqtt.pskreporter.info:1883`, hosted by
    Tom M0LTE, data from N1DQ). Topic `pskr/filter/v2/{band}/{mode}/#`; defaults
    to 2 m + 6 m MSK144. Appends raw spot JSON (+ `_rx` local epoch, `_topic`) to
    `MSK144/detections/pskr_spots.jsonl`; heartbeat to `MSK144/logs/pskr_logger.out`.
    paho-mqtt 1.x/2.x compatible, reconnect-with-backoff, clean session. There is
    **no authentication** on PSKReporter (the HTTP retrieve API is capped at last
    100 records / ≤6 h / 1 query per 5 min — MQTT is the uncapped research path).
    Each spot carries `sc/sl` (tx call/grid), `rc/rl` (rx call/grid), `f`, `rp`
    (SNR), `t_tx` (15-s-normalised tx start), `b`, `sa/ra` (DXCC), `sq` (dedup).
    **Pending — the offline join** to `decodes.jsonl` on
    `(2nd-callsign-from-message, 15-s period)` ↔ `(sc, t_tx)` (tx = 2nd call, per
    `project_msk144_callsign_order_tx_is_second`). PSKReporter supplies the grids
    MAP144 messages usually omit (`RR73`/report messages carry no locator), so it
    also feeds #26 range and #28 grid fallback.

47. ⬜ **DX-cluster spot logger** — second external spot source (Telnet to a
    cluster node, or an aggregator feed). Different population from PSKReporter:
    human-logged DX/skeds and 2 m MS announcements rather than auto-decode
    reports, so it can reveal *intended* activity (skeds) that produced no decode
    anywhere. Same capture-first design: append raw to a JSONL, normalise to a
    common spot schema with the PSKReporter records for the geometry engine (#48).
    Bounded: a Telnet client + line parser + the shared spot schema.

48. ⬜ **Propagation-geometry correlation engine** (operator research thesis,
    2026-06-03). Depends on a grid for every callsign (#28 QRZ/GridTracker — the
    operator has QRZ auth) and great-circle math (#26). Stages:

    a. **Reflection-point estimate per spot/path.** For Es, the reflection
       (ground) point is well-approximated by the **great-circle midpoint** of the
       TX↔RX path (single hop, ~100 km E-layer). **Meteor scatter is similar but
       not identical:** the useful reflection obeys specular geometry (the trail
       must be tangent to the prolate ellipsoid with TX/RX as foci), so the
       hot-spots sit at ~85–110 km altitude and are typically **offset to either
       side of the path midpoint**, not exactly on it. Start with the midpoint as
       a first-order estimate; refine to the offset specular hot-spots later.
       *(Physics note, not yet verified in code — confirm the hot-spot offset
       model before relying on it quantitatively.)*

    b. **Path-pair correlation — did one meteor enable two paths?** Search the
       spot stream for pairs of near-simultaneous spots (same 15-s period, or
       tighter) whose estimated reflection points coincide in space (and the trail
       geometry is mutually consistent). A coincidence is evidence a single
       ionised trail served both paths. Sparse for MS (few simultaneous spots),
       so this is a rare-event search — but a confirmed common-trail pair is a
       strong, citable result.

    c. **Predicted-decodable-station map.** Given an estimated trail/cloud
       position and the on-air station set (from the spot feeds), compute which
       stations *should* have been workable from FN42 via that reflector
       (specular geometry + range). Turns "MAP144 decoded nothing" into a
       falsifiable prediction: "trail at X should have made stations A, B
       decodable here" → did MAP144 see them?

    d. **Nearby-station NF correlation (the gold-standard deafness test).** A
       PSKReporter *receiver* `rl` within tens of km of FN42 shares MAP144's
       illumination geometry, so its overnight 2 m decode list ≈ what MAP144's
       antenna was exposed to. Neighbour logs N stations, MAP144 logs 0 →
       isolates NF / antenna, not propagation. Whether such a NE-US 2 m MSK144
       witness exists is itself empirical — the #46 capture will show it.

    The single-receiver caveat that motivates all of this: a PSKReporter spot is
    `sc heard by rc`; for **MS the path is bistatic and trail-specific**, so
    "someone in the NE heard W1XYZ" does *not* mean W1XYZ was workable from FN42.
    Presence is necessary, not sufficient — the geometry engine is what turns raw
    presence into a per-path workability estimate.

    **Meteor-geometry refinement (2026-06-04, from the live run + WSJT-X azdist).**
    The "reflection point = great-circle midpoint" model is wrong. MS reflects off
    an extended ionized **trail** (a line ~10–25 km at ~85–115 km alt), specular
    in the Fresnel sense (trail tangent to the TX/RX ellipsoid) for underdense,
    diffuse for overdense. Antenna pointing *selects* off-path hot spots, so the
    reflection is **off the direct path and only roughly locatable** (diffuse,
    not a point). WSJT-X `azdist.f90` already encodes this: it returns **Hot A/B**
    azimuths = direct `Az ± daz` (`daztab` offset ~8–21°, distance-dependent), and
    `HotABetter` flips by **time of day** (`tmid`, breakpoint at dawn) and
    hemisphere — the terminator-drift effect, in code. **Upgrade path for the
    blob:** shoot a Hot-A ray from each endpoint, intersect → off-path common
    volume, *no antenna logs needed* (heading derived from grids + time). Keep the
    blob diffuse regardless — Hot-A is a statistical bias, not a locator. See
    `project_band_map` and `project_meteor_geometry` memories.

49. ✅ **Band Map panel** (landed 2026-06-04). View-menu toggle window; geographic
    NA map (pyqtgraph, light mode, real lon/lat + cos(lat0) aspect lock; basemap
    loaded at runtime from GridTracker `shapes.json`). Layers: **PSKReporter**
    (blue, age-faded, midpoint ticks) + **MAP144 live decodes** (green, from
    `decodes.jsonl`, near-instant) + **same-second co-path "meteors"** (one diffuse
    colour-coded blob per group; arcs colour-coded so meteors are separable).
    Auto-frame (damped, outlier-robust, home-in) with in-range filter (default
    2500 km). Grid resolution: message > live (PSKReporter) > static
    (ADIF/QRZ home), with recency + `/M`·`/R` rules and canonical position per
    call. Pure-layered: `geo.py` (math) ← `spot_bus.py` (ingest) ← panel (Qt). 49
    geo/spot_bus tests. Files: `map144_app/{geo,spot_bus,band_map_window}.py`.
    See `project_band_map` memory.

    Band-Map backlog:
    - **49a ⬜ ALL.TXT grid source + WSJT-X decode layer.** `ALL.TXT` (730 MB)
      carries `CQ <call> <grid>` for everyone *heard* (not just worked) and is
      self-updating — richer than the ADIF log. Tail it incrementally (cached
      index, not full re-read) for grid fill AND as a decode layer (multi-instance
      WSJT-X). Operator: "new info makes it into all.txt, available next time."
    - **49b ⬜ Hot-A off-path blob biasing** (see #48 refinement above) — port
      `azdist` daz/HotABetter to nudge the blob off-path; keep it diffuse.
    - **49c ⬜ Dateline split** — `AK` Aleutians wrap ±180° → one stray line in a
      whole-world view; split rings at the dateline.
    - **49d ⬜ Terminator-drift time-lapse** — activity longitude vs UTC should
      show the dawn band as a diagonal streak (operator prediction, in the data).
    - **49e ⬜ QRZ cache pre-warmer** (optional) — walk `decodes.jsonl`, look up
      calls via `tools/qrz_lookup.py`, fill `qrz_grid_cache.json` for never-worked
      stations the ADIF index misses.

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
