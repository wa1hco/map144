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

## Mathematical conventions

Do not change mathematical conventions casually. If a convention is inferred rather than explicit, ask for confirmation.

### Complex signals

- Preserve clear distinction between I/Q samples, complex spectra, complex correlations, and real-valued metrics.
- Be explicit about whether a quantity is magnitude, magnitude-squared, power, phase, or a complex phasor.
- Do not replace complex-domain logic with magnitude-only shortcuts unless explicitly approved.

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
