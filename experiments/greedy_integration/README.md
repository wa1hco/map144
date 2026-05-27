# Greedy sq_det-driven coherent integration

Experimental decoder-front-end algorithm: when a frame triggers WSJT-X's
sq_det (peak/local-noise ≥ 3 on the squared-signal spectrum at 2000 or
4000 Hz), greedily add adjacent whole frames one at a time, **keeping
each only if it improves sq_det**.

The hypothesis: phase-coherent summing of frames on the same Doppler
track gives +3 dB per doubling; frames on a different Doppler track
partially cancel and **fail the sq_det-improvement test**.  Thus the
algorithm self-selects frames that contribute constructively, without
any explicit Doppler-track classifier.

Sealed-off from production:

* All code lives under this directory.
* The shared sq_det implementation (`algos/sq_det.py`) matches the
  WSJT-X `msk144spd.f90` formula exactly so results are comparable
  to that reference decoder.

## Layout

| dir              | contents                                          |
|------------------|---------------------------------------------------|
| `algos/`         | sq_det implementation + algorithm variants        |
| `scenarios/`     | synthetic signal generator extensions             |
| `tests/`         | pytest-level unit tests                           |
| `results/`       | CSVs + PNG plots from comparison runs (gitignored — actually under scratch/ per project rule) |

## Algorithms compared

1. `single_frame` — decode the trigger frame alone, no integration
2. `blind_n_frame` — coherent sum of all N frames within a fixed span
3. `greedy_sqdet` — proposed: greedy frame inclusion with sq_det-improvement criterion
4. `wsjtx_style` — coherent integrate first, then sync-correlate (reference)

## Scenarios

1. Stable carrier (no Doppler / no flutter) — pure SNR sweep
2. Two-Doppler (overdense cloud model) — two reflections, controllable Δf and amplitude ratio
3. Linear drift — single carrier with controllable Hz/s drift rate
4. Multi-burst — N sub-bursts with gaps, all on same carrier

## Output

For each (scenario, SNR, algorithm) triple, we record:
- final sq_det score after integration
- carrier-frequency estimate error vs truth
- decode success / failure (via jt9)
- frames included / total span used

Plots compare the four algorithms across the SNR sweep for each scenario.
