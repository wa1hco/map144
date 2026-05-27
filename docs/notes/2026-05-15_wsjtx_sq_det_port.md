# 2026-05-15 — WSJT-X sq_det formula port (overnight-run README)

**Status:** landed in working tree, NOT yet committed.
**Test gate:** 177/177 production tests pass (`pytest --ignore=tools`).
**Smoke test:** three known-decode WAVs produced expected decodes
(CQ N1KWF FN32 from 260514_112645 / _112715, plus a third).

## What changed

Three of WSJT-X's `msk144spd.f90` sq_det design choices ported into
MAP144's per-channel detector.  One change deferred (NFFT) because
it broke a sanity test under MAP144's polyphase channelizer geometry.

| Knob | Before | After | File / line |
|---|---|---|---|
| Window | `np.hanning(CH_DETECT_SIZE)` | `np.ones(CH_DETECT_SIZE)` | [processing.py:243](../../map144_app/processing.py#L243) |
| LO/HI combine | `(lo_peak + hi_peak) / 2.0` | `np.maximum(lo_peak, hi_peak)` | [processing.py:657](../../map144_app/processing.py#L657) (H), [705](../../map144_app/processing.py#L705) (V) |
| Threshold | `3.0` (dB) | `4.77` (dB ≡ linear 3.0) | [processing.py:111](../../map144_app/processing.py#L111) |
| NFFT | 512 | **kept at 512** (WSJT-X uses 864) | [processing.py:151](../../map144_app/processing.py#L151) |

Block-pipeline detector ([detector_block.py](../../map144_app/blocks/detector_block.py))
got the matching combine-rule and threshold update; NFFT held at 512
there too via the shared `CH_DETECT_SIZE` import.

## Why each piece

* **Rectangular window** — for the squared signal at NFFT=NSPM=864, the
  ±1000 Hz tones land on integer-cycle bin centres ⇒ no spectral
  leakage ⇒ a window only attenuates in-band signal.  WSJT-X exploits
  this; we should too.  At NFFT=512 (current) the bin centres are at
  ±1000 Hz × (512/12000) Hz/bin = ±42.7 bins — *not* integer; some
  leakage remains and a window would help slightly.  Net direction is
  still toward less attenuation, so rectangular is at worst a wash.

* **max(lo, hi) combine** — WSJT-X uses `max`.  Asymmetric multipath
  fades that null one tone but leave the other now fire cleanly
  instead of being averaged away.  At pct25-baseline, both LO and HI
  bins also see the max ⇒ the percentile baseline scales with the
  signal ⇒ raw_lin/pct25 ratio is invariant on noise.

* **Threshold 4.77 dB = 10·log10(3.0)** — WSJT-X compares the raw
  ratio against 3.0 (linear).  MAP144 compares in dB, so 4.77 dB is
  the dB equivalent.  Old MAP144 value 3.0 dB (= linear 2.0) was 1.77
  dB more sensitive, which produced more launches per period — at the
  cost of more no_decode launches (false-positive rate measured 4×
  higher in Experiment B at universal threshold).

## Why NFFT was deferred

Bumping `CH_DETECT_SIZE` 512 → 864 broke
`tests/test_block_pipeline_replay.py::test_positive_half_channel`
(20 dB SNR sanity-check WAV).  Diagnosis from the launches.jsonl on
the failing run:

```
ch_signed=5 (signal channel 6500 Hz)  sq_metric=1.2 dB  sync=8.7 dB
ch_signed=6 (neighbour    7500 Hz)    sq_metric=17.6 dB sync=6.7 dB
```

The signal channel collapsed; an adjacent channel went hot.  The
likely culprit is squared-tone aliasing under the wider bin spacing,
combined with the polyphase channelizer's cross-channel leakage at the
±1000 Hz offset that the squared signal lives on — but the analysis
isn't complete enough to commit a confident fix tonight.

NFFT change is a separate item with its own risk surface; it should
be revisited with a proper end-to-end retest later.

## Expected behaviour overnight

* **Trigger rate** — should drop modestly versus prior runs (1.77 dB
  threshold raise), most of the loss in noise-only launches that
  weren't decoding anyway.  Real-signal launches should be unaffected.
* **SPD vs jt9 split** — combine-rule change makes the SPD path slightly
  more aggressive at low SNR (max ≥ mean), so SPD%/jt9% may shift a
  point or two toward SPD.
* **Decode count** — net direction is small.  Experiment A measured
  +2.33 dB sensitivity from the formula port at the per-frame trigger
  stage; the fraction of that which converts to additional decodes
  depends on how much of MAP144's decode gap was trigger-limited
  vs. averaging-limited.

## Revert if anything goes wrong

```sh
cd ~/ham/map144
git checkout -- map144_app/processing.py map144_app/blocks/detector_block.py
# tests/test_detector_block.py has a fix for the new threshold;
# revert that too if the production code is reverted:
git checkout -- tests/test_detector_block.py
```

(The detector_block test fix raised a hardcoded `sq_db=4.5` to
`sq_db=5.5` so it still exceeds the new 4.77 dB threshold; reverting
processing/detector_block without reverting the test would leave the
test passing but no longer exercising the same threshold-edge case.)

## Files touched (uncommitted)

* `map144_app/processing.py` — three formula edits, one deferred-NFFT note
* `map144_app/blocks/detector_block.py` — combine + threshold comment alignment
* `tests/test_detector_block.py` — `test_event_fields` sq_db threshold input

## Related memory

* `project_wsjtx_sq_det_formula` — exact formula port + +2.33 dB measurement
* `project_tgi_experiment_findings` — TGI dead-end; this is the
  deployable finding from that branch
* `project_msk144_sync_fraction_floor` — why the formula port helps but
  doesn't lift the decode floor
