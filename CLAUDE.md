# MAP144 Project Guidelines

This file contains MAP144- and WSJT-X-specific guidance. General collaboration style, architectural rules, coding preferences, and safety rails are in `~/.claude/CLAUDE.md` and apply here unless overridden below.

## How to use this file

Use this document as standing project guidance for any task in this repository unless the prompt explicitly overrides it.

This repo uses a persistent memory system at `~/.claude/projects/.../memory/` (indexed by `MEMORY.md`). Cross-session findings — algorithm decisions, sensitivity investigation results, architectural constraints — are stored there. Read relevant memory entries before starting work on detection, decode, or sensitivity topics; update them when a finding changes.

Before proposing code changes:

- Read the relevant files first.
- Explain the current signal flow in plain language.
- Preserve architecture unless a restructuring is explicitly requested.
- Prefer small, testable changes over broad rewrites.
- If a mathematical assumption is unclear, ask instead of guessing.

When making code changes:

- Separate DSP math from UI, logging, transport, and file I/O.
- Keep new logic observable and testable.
- Avoid changing multiple abstractions at once.
- Do not silently change numerical conventions, normalization, indexing, or phase sign conventions.
- If a change may alter decode behavior, detection sensitivity, or numerical stability, say so explicitly.

## Working modes: development vs debugging

This project has two distinct working modes. The right collaboration style is different for each, and confusing them causes problems.

### Development mode

Goal: build a working prototype quickly; head in the generally right direction and adjust.

- Fast architectural sketches are useful; speed beats precision.
- Plausible guesses that get us moving are fine.
- Hedges like "probably" and "should" are acceptable.
- Iterate by trying it and looking at what happens.

### Debugging mode

Goal: explain an anomaly accurately so we either patch it or revise the architecture.

This mode has a different contract. **Plausible-sounding guesses substituted for verified facts are the failure mode** — they force the operator to interrogate progressively deeper while answers get progressively more superficial. Every anomaly needs an accurate explanation, not a series of confident guesses.

Rules:

1. **Separate observation from inference on the page.** Every response has two parts: what the data/code *shows* (verified, with `file:line` when from code) and what I *infer* (explicitly marked speculation). No blurring.

2. **Read code before claiming a mechanism.** Before saying "the X causes Y", open the file and find the logic. If I haven't read it, the claim is speculation and labeled as such — not stated as explanation.

3. **State confidence level explicitly** on every claim:
   - **(verified)** — read in code or measured in data
   - **(consistent with data)** — observation supports it but mechanism unconfirmed
   - **(speculation)** — plausible but needs verification

4. **On pushback, do not immediately produce a replacement narrative.** The reflex to give a new answer is the failure mode. Better response: "I was wrong about X. Before guessing again let me read Y."

5. **Make "I don't know" load-bearing.** If I don't know how something works, that is the answer until I look — not "probably", not "looks like". Read the code; report what it says.

6. **Cite evidence by location, not by claim.** Saying "I verified this" or "the data shows X" does not ground a claim — the `file:line` reference, the quoted code snippet, or the exact log entry does. Verification means *checking the output, not asserting that I checked*. Research finding: when I push back at the operator with an evidence claim ("the data shows X") and that claim is itself unverified, I'm at the highest risk of regressive sycophancy. The fix is to require evidence-by-location for any pushback I make, not just any claim I make.

7. **Pre-commit before pivoting.** When the operator pushes back on a wrong claim, restate my prior reasoning chain and identify the specific step the correction invalidates. The new answer must attach to a specific link in the old chain, not start from scratch with a fresh plausible narrative. (Starting fresh is the failure mode — the new narrative looks unconnected to the old one, which is exactly how cascade substitution happens.)

8. **Re-read this section every ~5–10 turns of a long debug thread.** Published research finds multi-turn instruction adherence drops ~39% on average; periodic reinjection is the documented mitigation. If I notice myself slipping — or if the operator says "label that" / "go read the code" — the right response is to re-read this section before continuing, not to defend.

The operator can call out slips explicitly ("that's a guess, label it" or "go read the code first") and I will fix the response, not defend it.

### Mode signals

Debugging mode: corner cases, anomalies, failure modes, "why doesn't X work", "explain again why...", investigation of specific log entries / no-decode periods / unexpected metrics.

Development mode: new feature, prototype, refactor proposal, architecture discussion, "should we...".

When in doubt about the mode, ask.

## GUI preferences

- Framework: PyQt5 (Qt)
- Theme: light mode
- Font size: 12 pt

## Project goals

### MAP144

MAP144 is a weak-signal DSP and decoding project in the meteor-scatter / amateur-radio domain. The main purpose is to detect and decode signals reliably under low SNR conditions while keeping the signal-processing pipeline understandable and modifiable.

The preferred workflow is:

1. Keep the detection path clear.
2. Keep the decode path clear.
3. Make intermediate values inspectable.
4. Optimize only after correctness is established.

### WSJT-X related work

WSJT-X work in this repo is usually exploratory or patch-oriented. Changes should be conservative, easy to review, and aligned with existing structure and naming as much as practical.

For WSJT-X-related edits:

- Respect existing project style and module boundaries.
- Avoid large-scale rewrites unless explicitly requested.
- Prefer minimal patches that can be explained in a short commit message.
- Preserve behavior unless the task is explicitly about changing behavior.

## Signal-flow expectations

Always reason about the code as a pipeline.

The current MAP144 production pipeline is:

RF input (SDR) -> polyphase channelizer (complex IQ per channel) -> audio BPF (129-tap FIR, ±1200 Hz on complex baseband) -> SPD coherent-averaging decoder (primary) -> jt9 fallback (if SPD misses) -> reporting

Key invariants of this pipeline:

- The channelizer produces complex IQ; the Hilbert-to-analytic conversion step is intentionally absent (removed to recover ~3 dB sensitivity).
- The audio BPF on complex baseband is the primary sensitivity mechanism; removing it reverts to detection-limited behavior.
- SPD runs first on every candidate; jt9 is fallback only, not co-primary.
- RMS normalization is used in SPD (not peak normalization); peak normalization suppresses signal at low SNR.

For polarization-related experiments, the flow may include parallel H and V streams and a combination stage before or during detection/decode depending on the experiment.

When discussing modifications, always state:

- What the inputs are.
- What intermediate representation is produced.
- What downstream stage consumes it.
- Whether the change affects timing, phase, scaling, or statistical assumptions.

## Refactor / migration policy

MAP144 is mid-migration from a legacy monolith (`process_iq_data`, ~1100 LOC in
`processing.py`) to the block/stream architecture (`docs/block-stream-design.md`).
The migration stalled once because the cut-over was framed as one big-bang step
with no incremental payoff. To keep it moving:

- **A bug in pre-refactor (legacy-monolith) code is a trigger to migrate that
  piece onto the block/stream path — not to patch it in place.** Do not add
  another patch to the legacy path. Fix the bug by moving the responsibility to
  its refactored home, behind tests; the bug becomes that slice's acceptance
  test. (Example: the 2026-05-31 "1970 TFMF timestamp" bug is fixed by the
  `TimeBase` migration, not an anchor-reset patch.)
- **Slice the refactor so each step ships a concrete, testable win on its own.**
  No step should depend on the whole cut-over.
- Exception (in-place legacy patch allowed): a production-down emergency, or a
  fix far smaller than the migration with the migration already scheduled — and
  say so explicitly when taking the exception.

Corollary: when asked to "fix" a legacy-path bug, first state which refactor
slice owns it and migrate that, rather than reaching for the patch.

## Mathematical conventions

Do not change mathematical conventions casually. If a convention is inferred rather than explicit, ask for confirmation.

### Complex signals

- Preserve clear distinction between I/Q samples, complex spectra, complex correlations, and real-valued metrics.
- Be explicit about whether a quantity is magnitude, magnitude-squared, power, phase, or a complex phasor.
- Do not replace complex-domain logic with magnitude-only shortcuts unless explicitly approved.

### dB convention

**dB is always power.**  There is no such thing as "amplitude-dB" in
this project.  The two valid forms are:

- `10 · log10(power_ratio)` — for quantities that already scale as power
  (e.g. `|X|²`, energy, variance).
- `20 · log10(amplitude_ratio)` — equivalent, for quantities that scale
  as voltage/amplitude (e.g. `|X|`, signal magnitude, matched-filter
  output).  This is just `10 · log10(amplitude_ratio²)`; same dB number.

Applying `10 · log10` to an amplitude-scaled quantity is a **unit bug**,
not an alternative convention.  Same for `20 · log10` on a power-scaled
quantity.  When in doubt about a quantity's scaling, work out what power
of the underlying audio voltage it scales as (e.g. `|FFT(s²)|²` scales
as `A⁴`) and pick the conversion factor that makes the final dB number
mean "audio power dB".

### SNR convention

All SNR values in this project are in the WSJT-X standard reference
bandwidth of **2.5 kHz**.  This applies to:

- ground-truth labels (simulator outputs, `truth.snr_db`, etc.)
- decoder-reported SNR (jt9 `snr_db`)
- any "SNR" used in analysis, plots, or discussion

The 12 kHz audio stream from the SSB receiver is already bandlimited
to approximately 300–2700 Hz by the radio's audio filter (and may be
narrowed further by MAP144 DSP).  The audio noise bandwidth is therefore
already ≈ 2.5 kHz — **no sample-rate-to-2.5-kHz BW correction is
needed**.  Treating the 12 kHz sample rate as the noise bandwidth is a
common mistake; it would over-state SNR by ~6.8 dB.

When proposing or reviewing an SNR-related calculation:

- State what reference bandwidth is in use; if unstated, assume 2.5 kHz.
- Do not convert SNR between bandwidths without explicit justification
  tied to the actual noise spectrum at that point in the pipeline.
- If a calculation appears to need a "12 kHz to 2.5 kHz" correction,
  stop and check whether you're looking at audio (already at 2.5 kHz)
  or at some earlier-stage IQ where the noise is genuinely wideband.

### FFT and indexing

Be explicit about:

- FFT length
- windowing
- overlap
- normalization
- bin ordering
- frequency sign convention
- whether a shift has been applied

Do not assume that FFT scaling is irrelevant. In weak-signal work it often affects thresholds, comparisons, and interpretation.

### Correlation and accumulation

When implementing correlation or coherent accumulation:

- State whether accumulation is coherent or noncoherent.
- State what is being phase-aligned, and how.
- Preserve the distinction between detection statistics and decoded-symbol logic.

### Normalization conventions in SPD

RMS normalization must be preserved in `msk144_spd.py`. Peak normalization (e.g. scaling to 0.9 peak) suppresses low-SNR signals and costs roughly 1.5 dB of sensitivity. Do not switch normalization strategy without calling it out.

The LLR scaling parameter `sigma=0.60` is confirmed optimal for the Python `bpdecode128_90` implementation. Do not change it without a controlled sensitivity sweep.

### Polarization combination

For dual-polarization experiments, the working model may include a combined signal of the form:

$$\text{combined} = H \cdot \cos\theta + V \cdot e^{j\Delta\phi} \cdot \sin\theta$$

where:

- $H$ is the horizontal-polarization complex signal
- $V$ is the vertical-polarization complex signal
- $\theta$ controls the mixing angle
- $\Delta\phi$ is the relative phase offset

When working in this area:

- Keep phase and amplitude effects separate in the explanation.
- Do not silently convert this into an amplitude-only combination.
- State where the combination occurs in the pipeline.
- Make clear whether optimization is for detection, decode quality, or both.

### Numerical stability

Prefer numerically stable forms when equivalent.

Examples:

- Avoid subtracting nearly equal large values when a better formulation exists.
- Be careful with normalization when magnitudes may approach zero.
- Guard divisions when a denominator can be zero or nearly zero.
- Preserve precision in accumulation loops when practical.

## Testing expectations

For algorithmic or DSP changes, propose tests whenever practical.

Minimum useful test categories:

- noise-only input
- strong signal input
- marginal-SNR input
- timing offset case
- frequency offset case
- edge/pathological case

When relevant, suggest both:

- deterministic synthetic tests
- real capture / representative sample validation

If exact expected values are not available, propose invariant-based tests, such as:

- output shape and units remain correct
- monotonic improvement under coherent accumulation
- no regression in previously detected cases
- thresholds move in expected direction after normalization changes

## Safety rails (MAP144-specific)

In addition to the general safety rails in `~/.claude/CLAUDE.md`, do not do the following without explicit instruction:

- Remove the audio BPF from the SPD path — it is the primary mechanism that makes the system detection-capable at low SNR.
- Re-add a proximity guard (±N channel suppression on active threads) to the detection gating — it was deliberately removed because it blocked legitimate adjacent-channel signals and same-frequency re-triggers.
- Switch from RMS to peak normalization in SPD.
- Change `sigma` in `bpdecode128_90` LLR scaling without a controlled sweep.

## MAP144-specific guidance

- Preserve the conceptual separation between detection and decode.
- Do not blur candidate generation with final decode scoring unless explicitly requested.
- Keep intermediate observables available for plotting, inspection, or offline analysis.
- When simulating channels or polarization effects, document assumptions clearly.
- Favor code that makes it easy to compare algorithm variants side by side.

### Detection gating invariants

The cluster-based detection gating in `processing.py` has several non-obvious invariants that must be preserved:

- A cluster is defined as a group of triggered channels with no gap wider than 3 channels; two clusters can both launch independently.
- The `_too_many` broadband gate counts only *fresh* (non-cooldown) triggered channels — previously-cooled sidelobe remnants must not count against the gate threshold.
- On launch, only the single best channel (`_best`, ±0 radius) is suppressed; suppressing ±1 blocks legitimate adjacent-channel signals.
- The proximity guard (±2ch `fc_hz` check on active threads) was intentionally removed — it incorrectly blocked same-frequency re-triggers and legitimate adjacent signals.

When touching detection gating logic, state which invariant is affected and why the change is safe.

### Polarization optimization targets

If proposing changes related to polarization optimization, state whether the optimization target is:

- best instantaneous SNR
- best integrated detection statistic
- best decode probability
- best recovered symbol quality

These are not always the same objective.

## WSJT-X-specific guidance

- Match existing code style and naming when editing upstream-like code.
- Prefer localized patches.
- Be very cautious about changing behavior in mature decoder paths.
- Explain any performance impact, even if expected to be small.
- Avoid speculative cleanup in files touched for a narrow behavioral fix.

## Questions to ask for MAP144 work

If key context is missing, ask focused questions such as:

- Is this change intended to affect detection, decode, or only diagnostics?
- Is coherent phase preservation required at this stage?
- What normalization convention is currently assumed here?
- Should this remain compatible with existing outputs or file formats?
- Is this experimental-only code or production-path code?
- Should the optimization favor weak-signal sensitivity, speed, or maintainability?

## Maintenance notes

This file should be updated whenever one of these happens:

- a repeated Claude failure mode is discovered
- a new architectural rule becomes important for MAP144
- a math convention is clarified
- a project-specific testing pattern proves useful
- a sensitivity finding changes the pipeline architecture

Treat this as a living engineering contract between the human and the assistant.
