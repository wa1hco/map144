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

### False alarm rate vs threshold

The detector threshold is **per-bin SNR in dB above the rolling-median
noise floor** for that frequency bin (see `extract_candidates` in
`tfmf.py`).  Per cell:

    snr_lin = |surface[t, f]| / median_t(|surface[·, f]|)
    snr_db  = 20 · log10(snr_lin)
    fire    = snr_db > cfg.threshold_db

On AWGN, each matched-filter magnitude `|surface[t, f]|` is Rayleigh
distributed.  Normalised by its median (= σ·√(2·ln 2) ≈ 1.177σ), the
per-cell tail probability above threshold `T_dB` is:

    P_FA(cell) = exp( −ln(2) · 10^(T_dB / 10) )

(Derivation: P(X > T) = exp(−T²/2σ²); set R = X/median_R, then
P(R > r) = exp(−r²·ln 2); with r = 10^(T_dB/20), r² = 10^(T_dB/10).)

For the default surface (29,993 windows × 192 bins = 5.76 M cells per
15 s), expected false alarms per 15-s period:

| threshold_db | P_FA per cell | FA / 15 s | Use case |
|---|---|---|---|
| 5.0  | 1.1 × 10⁻¹ | 6.4 × 10⁵ | (research only — surface is noise-saturated) |
| 10.0 | 9.8 × 10⁻⁴ | 5 600   | Old default — too loose for production |
| 12.0 | 1.7 × 10⁻⁵ | 97      | Tuning / characterisation |
| **13.5** | **1.8 × 10⁻⁷** | **≈ 1** | **Default — one FA per 15 s** |
| 14.0 | 2.7 × 10⁻⁸ | 0.16    | Mildly conservative |
| 15.0 | 3.1 × 10⁻¹⁰ | 1.8 × 10⁻³ | Clean detection logs |
| 17.0 | 8.3 × 10⁻¹⁶ | 4.8 × 10⁻⁹ | Very confident detections only |
| 20.0 | ≪ 10⁻³⁰ | ≈ 0 | Hard ceiling (effectively zero FAs) |

Empirical verification on 15 s of synthetic complex-Gaussian noise at
48 kHz (template autocorrelation magnitudes normalised by per-bin
median) closely matched the table above — Rayleigh statistics hold.

**Why 13.5 dB as the default**: at one expected FA per 15-s period
the detector reports ≈ 1 FP per WSJT-period, which is the natural cap
before downstream stages (decoder gate, sync sub-block sum, etc.)
discount false candidates.  For sensitivity sweeps or ROC plots, lower
`threshold_db` and rely on downstream stages to reject; for clean
detection logs, raise it.

**Caveats:**
- These numbers assume *independent* cells.  Adjacent (t, f) cells are
  weakly correlated (overlapping windows, sinc-shaped per-bin response),
  so the realised FA count is slightly less than the i.i.d. prediction.
  In practice the table is accurate to ~10–20 %.
- A 2D local-max filter (TODO in `extract_candidates`) would deduplicate
  the multi-cell ring around each real or false detection, dropping the
  effective FA rate by another ~10×.

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

### Frequency resolution refinement (implemented)

250 Hz/bin is coarse compared to MSK144's expected freq stability (~10 Hz
within a burst).  Implemented refinement (`cfg.parabolic_freq_interp`,
default True): **parabolic interpolation across the 3 bins centred on
the peak**.  Standard 3-point fit:

    δ = 0.5 · (left − right) / (left − 2·peak + right)

Returns a sub-bin offset in (−0.5, +0.5) × bin_width that is added to
the bin-centre frequency.  Cheap (no extra FFT), no math change to the
matched filter.

Empirical accuracy on synthetic pings at bin-centre (1500 Hz = bin 6 at
250 Hz/bin):

- At SNR ≥ 0 dB: typically < 50 Hz error (signal dominates the peak).
- At SNR < −5 dB: 50–250 Hz error (noise pushes peak to neighbouring bin).

Alternative: **zero-pad the window** (`cfg.fft_length`) to N_padded.
Increases bin resolution by N_padded/N but doesn't change the underlying
matched-filter accuracy — strictly a finer view of the same surface.
At fft_length=768 (4× pad): 62.5 Hz/bin.  More expensive; use only if
parabolic interp proves insufficient.

### Two sync sub-blocks per MSK144 message (implemented)

WSJT-X's MSK144 message has **two** 8-symbol sync sub-blocks per
144-symbol message frame — at positions 0–7 and 56–63 (336 samples
apart at 12 kHz = **1344 samples = 7 × FFT length at 48 kHz**).  The
legacy sync correlator at `map144_app/msk144_spd.py::_sync_correlate`
correlates both sub-blocks and sums them coherently.

Implemented (`cfg.two_subblock_sum`, default True): after building the
time-frequency surface, sum each row with the row 56 samples later
(`SYNC_SPACING_SAMPLES / stride_samples = 1344 / 24`).  At FFT-bin
frequencies the phase between sub-blocks is `2π · k · 7` mod 2π = 0
for all integer `k`, so the coherent sum is a plain elementwise complex
add — no phase correction needed.  This was the design intent of the
1344-sample spacing.

**Theoretical gain:** +3 dB SNR at the same threshold and FA rate.
Both signal amplitude doubles (+6 dB power) and independent-noise
variance doubles (+3 dB power), so the detector statistic
`|MF|/median(|MF|)` is scale-invariant on pure noise (threshold and
FA distribution unchanged) but signals get +3 dB lift.

**Empirical on a 16-ping ramp (−10 to +5 dB SNR, 1 dB steps):**

| Config | Detections | Candidates | Per-ping snr_db shift |
|---|---|---|---|
| Single sub-block (off) | 5/16 (+1..+5 dB) | 73 | baseline |
| Two-sub-block (on)     | 5/16 (+1..+5 dB) | 134 | +1 to +3 dB (noisy) |

Detection count on this small ramp didn't improve — the +3 dB lift on
already-detected pings is real but doesn't pull the next-dB-down ping
above threshold reliably because per-realization snr_db variance is
~1–2 dB at low SNR (5 pings is too few to average that out).  Candidate
count nearly doubled because each strong ping produces phantom peaks at
row offsets ±56 in the combined surface (sync-1 alone or sync-2 alone
adding to noise still exceeds threshold near a strong source).

**Side-lobe caveat:** for any detected ping at row `i₀`, the combined
surface also lights up at `i₀ ± 56` (a single sub-block matching plus
noise from the other window).  These side-lobes are ~3 dB weaker than
the main peak but can still trip the threshold for strong signals.  A
2D local-max suppressor in `extract_candidates` (TODO) would collapse
them.

### Sensitivity floor caveat

The sync template covers only 16 of 144 symbols = **11.1% of frame
energy** = −9.5 dB vs the full-frame energy that `sq_det` sees.  Per
`memory/project_msk144_sync_fraction_floor.md`, even with K=7 coherent
frame averaging (+4.2 dB) and two-sub-block (+3 dB), the sync matched
filter sits ~2.3 dB below sq_det in per-frame sensitivity on AWGN.

TFMF's value over sq_det is therefore *not* per-frame sensitivity — it
is:

1. **Wide-bandwidth coverage** with no channelizer/intermod limits.
2. **Precise (t, f) handoff to the decoder** — narrowing jt9's FTOL
   search reduces redundant work (jt9 re-does its own sync search;
   high-precision input lets it skip much of that).
3. **GPU throughput** — variable compute per candidate (multi-Doppler
   testing, etc.) becomes affordable.

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
