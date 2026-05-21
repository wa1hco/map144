# GPU TFMF matched-filter detector

**Status:** scaffold — first concrete CuPy implementation is the TODO.

## Goal

Replace the legacy `ChannelizerBlock + DetectorBlock(sq_det + per-channel sync)`
pipeline with a single **Time-Frequency Matched Filter (TFMF) on GPU** that produces
detection candidates directly from full-bandwidth IQ.

**Why this works:** the sync correlator is a *linear* operator, unlike the
squared-FFT detector.  Squaring is what forced the channelizer in the first
place (squaring intermods strong off-channel signals into the squared
spectrum, so each channel had to be band-limited to ~1 kHz first).  A
matched filter has no such restriction — strong off-channel signals
contribute only to *their own* freq bin in the matched-filter output, not to
neighbours.  So we can:

- drop the channelizer entirely
- drop sq_det entirely
- search the full (time × frequency) time-frequency correlation surface in one pass

## Acceptance criteria

The detector is "working" when, on synthetic IQ:

1. **Detection rate vs SNR** is ≥ the legacy sync detector across SNR
   −5 dB to +20 dB.  Same signals, same noise seeds.
2. **Frequency-offset coverage**: detects equally well anywhere in
   ±20 kHz from band centre (current `Channelizer + sq_det` has known
   weak spots at Nyquist edges and channel boundaries).
3. **Same/better false-positive rate** at matched detection threshold
   on noise-only input.
4. **Real-time throughput**: process ≥ 60 s of 48-kHz IQ per second of
   wall-clock on the laptop's GPU.

## Design

### Algorithm

For each ~3.5-ms sliding window of input IQ at 48 kHz:

1. Multiply elementwise by `conj(sync_template)` — gives the input "ungated"
   by the sync, with any signal at zero freq offset producing a constant.
2. FFT of the product, length N samples.
3. The k'th bin of the FFT magnitude = correlation magnitude at frequency
   offset `k * (fs / N)` from the centre.
4. Detection candidates: bins above threshold (relative to per-bin noise
   floor estimated across recent windows).

### Time and frequency stepping (default config, 48 kHz IQ)

**Time axis:**

| Parameter | Value | Where set |
|---|---|---|
| Window length (= template length) | 192 samples = **4 ms** | `sync_template.build_sync_template(48000)` |
| Stride between windows | 24 samples = **0.5 ms** = 1 MSK144 symbol | `TFMFConfig.stride_samples` |
| Window count for a 15-s input | (n_samples − 192) / 24 + 1 = **29,993** | derived |
| **Time increment** | **0.5 ms** | `stride_samples / sample_rate` |
| Time span | window-centre 2 ms → 14.998 s for 15-s input | derived |

Sub-symbol stride buys nothing — correlation barely changes within a
symbol — so 24 samples is the natural lower bound at 48 kHz.

**Frequency axis:**

| Parameter | Value | Where set |
|---|---|---|
| FFT length | 192 (= template length; no zero-pad by default) | `TFMFConfig.fft_length=None` |
| Frequency span | **−24 kHz to +24 kHz** (full Nyquist, signed) | `np.fft.fftfreq(192, 1/48000)` |
| Number of bins | 192 | = FFT length |
| **Frequency increment** | **250 Hz/bin** | `fs / fft_length` |

Bin layout follows `np.fft.fftfreq`: bins 0…96 cover 0…+12 kHz, bin 97
wraps to −11.75 kHz, bins 97…191 cover −24 kHz…−250 Hz.  No `fftshift`
in the hot path.

**Resulting surface:**

For a 15-s input at 48 kHz with default config: **29,993 × 192 ≈ 5.76 M
cells**, each `complex64` (8 B) → **46 MB per 15-s surface**.  Trivial
for the laptop's T2000 (4 GB GDDR6, ~128 GB/s bandwidth).

**Both axes are tunable** via `TFMFConfig`:

| Knob | Default | What it does |
|---|---|---|
| `stride_samples` | 24 | Smaller → finer time, more compute.  Larger → coarser, faster. |
| `fft_length` | `None` (= 192) | Set to ``N_padded > 192`` to zero-pad and view the same correlation surface at finer bin resolution (does **not** improve underlying resolution — window length sets that).  At `fft_length=768` (4× pad): 62.5 Hz/bin. |

**Practical caveats for MSK144 on 6 m:**

- **Time resolution 0.5 ms** vs ping durations of 100 ms to several
  seconds — much finer than needed; cheap headroom.
- **Frequency resolution 250 Hz** vs typical 6 m signal stability of
  ~10 Hz within a burst — **coarse**.  This is why the sensitivity-test
  output shows bin-quantised freq errors at low SNR (multiples of
  250 Hz).  Parabolic interpolation across the 3 bins around the peak
  (TODO #1) is the recommended sub-bin refinement; zero-padding via
  `fft_length` is the alternative.

### Why FFT-based search over freq?

Mathematically, the matched filter at frequency offset `f` is:

    y(t; f) = ∫ x(τ) · h*(τ - t) · exp(-j 2π f (τ - t)) dτ

For a single window (fixed `t`):

    y(f) = exp(j 2π f t) · ∫ x(τ) · h*(τ - t) · exp(-j 2π f τ) dτ
         = exp(j 2π f t) · FFT { x · h*_shifted } (f)

So one FFT per window gives the correlation at *every* frequency offset
simultaneously.  At fs=48 kHz with N=168-sample window (8 sync symbols at
48 kHz), this gives:

- Frequency resolution: 48000 / 168 ≈ 286 Hz/bin
- Frequency range: −24 kHz to +24 kHz (entire Nyquist band)
- Bins per window: 168

Time resolution is set by the stride between windows.  At stride=24 samples
(one symbol at 48 kHz, 500 µs), we get 2000 windows per second.  Total
(time, freq) cells per second: 2000 × 168 = 336k — light for a GPU.

### Frequency resolution refinement (TODO #1)

286 Hz/bin is coarse compared to MSK144's expected freq stability (~10 Hz
within a burst).  Two refinement strategies:

1. **Zero-pad the window** to N_padded samples before FFT.  Increases freq
   resolution by N_padded/N (but doesn't improve the underlying matched-
   filter resolution — just gets us a finer view of the same correlation
   surface).
2. **Frequency-interpolation around the peak** using parabolic
   interpolation across the 3 bins centred on the strongest detection.
   Same trick the legacy sync correlator uses for sub-sample time
   refinement.

Start with #2.  Cheap, no extra FFT cost.

### Two sync sub-blocks per MSK144 message (TODO #2)

WSJT-X's MSK144 message has **two** 8-symbol sync sub-blocks per 144-symbol
message frame — at positions 0–7 and 56–63 (336 samples apart at 12 kHz =
1344 samples apart at 48 kHz).  The legacy sync correlator at
`map144_app/msk144_spd.py::_sync_correlate` correlates both sub-blocks
and sums them coherently.

Scaffold uses **single-sub-block** correlation initially.  Two-sub-block
coherent sum adds ~3 dB on real MSK144 signals; tackle once single-block
detection is proven.

### Sync template construction

Ported from `map144_app/msk144_spd.py::_build_sync_template`, retimed from
12 kHz (42 samples) to 48 kHz (168 samples).  See `sync_template.py`.

## Files

| File | Purpose |
|---|---|
| `sync_template.py` | Build the MSK144 sync template at any sample rate |
| `tfmf.py` | Core GPU matched filter + candidate extraction |
| `synthetic.py` | Synthetic-IQ test signal generator (uses repo's `generate_msk144.py`) |
| `benchmark.py` | Sweep harness: SNR × freq-offset × position → detection-rate table |
| `compare_to_legacy.py` | Same signals through legacy `sq_det + sync`, ROC compare |
| `test_tfmf.py` | Unit tests — template correctness, single-ping detection at high SNR, noise-only false-positive rate |

## Dev setup (laptop side)

```bash
# After git clone + ./install.sh:
nvidia-smi   # confirm driver + CUDA version
.venv/bin/pip install cupy-cuda12x   # adjust for your CUDA major version
                                      #   12.x: cupy-cuda12x
                                      #   11.x: cupy-cuda11x

# Quick GPU sanity:
.venv/bin/python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU(s) visible')"

# Run the (currently stub) sanity test:
.venv/bin/pytest experiments/gpu_tfmf/test_tfmf.py -v
```

## Suggested work sequence on the road

1. **Template parity** — port `_build_sync_template` to support `sample_rate_hz`
   parameter; verify the 48-kHz template autocorrelates to a clean peak.
2. **Single-window matched filter on CPU** (numpy) — get the math right.
   Verify it detects a known synthetic ping at known (t, f, SNR).
3. **Port to CuPy** — same math, cupy arrays.  Profile.
4. **Sliding window** — vectorise the windowing.  Use `cupy.lib.stride_tricks`
   or just iterate (start simple).
5. **Candidate extraction** — peak-finding above threshold.
6. **Sweep harness** — SNR × freq × position → CSV.  ROC curve.
7. **Compare to legacy** — same signals through `sq_det + sync`.  This is
   the result that decides whether GPU mode is worth promoting.
8. **Real WAVs** — feed in `MSK144/detections/*.wav` files and compare
   detected candidates against `launches.jsonl` ground truth.

## Open design questions

- **Two sync sub-blocks**: when to add — start of trip or once single-block
  is solid?  Single-block first; coherent-sum once everything else works.
- **Frequency-interpolation method**: parabolic vs gaussian peak fit?
  Probably parabolic (cheaper, similar accuracy in noise-dominated regime).
- **Output schema**: emit `Event` records compatible with existing
  `DetectorBlock`, or define a new `TFMFCandidate` record and bridge?
  Defer until the math is proven — start with a simple list of `(time_s,
  freq_hz, snr_db)` tuples.
- **Compute budget per candidate** in GPU mode: at GPU speeds we can
  afford multi-Doppler hypothesis testing per candidate — but not until
  the basic detector is working.

## Resources on the laptop

(See parent `README.md` and `docs/ALPHA_NOTES.md` for repo basics.)

For validation data, you need on the laptop:

- A subset of `MSK144/detections/*.wav` for real-signal testing
- `MSK144/detections/launches.jsonl` — ground-truth labels of what the
  current detector did (`sq_metric_db`, `sync_metric_db`, `fc_hz`,
  decode outcome).  Compare candidates against this.
- `tests/corpus/smoke_set.json` + `tests/corpus/manual_tags.jsonl` —
  curated set of manually-classified pings, smaller than full WAV dir.
- `~/.local/share/WSJT-X - flex/ALL.TXT` — WSJT-X-decoded ground truth.
