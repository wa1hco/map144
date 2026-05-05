# DSP levels and decode reliability — working plan

Copy to `dsp-reliability-plan.md` (gitignored) for personal checklists and
capture notes; edit freely. This `.example.md` file stays canonical in git.

---

## 1. Signal-level contract

- **Internal IQ (DSP + map144 spectrograms):** I and Q are **peak-normalized**
  to roughly **`[-1, 1]`** (full-scale **±1.0 per component**). Not “RMS = 1.”
- **Separate domains** (do not merge without an explicit conversion):
  - **Ingress:** radio-specific scaling only at the driver (`runtime`); downstream
    assumes the internal contract.
  - **jt9 path:** real audio at 12 kHz, **peak-normalized with headroom** before
    int16 — not comparable to IQ spectrogram dB.
  - **Reporting SNR:** WSJT-style reference bandwidth — downstream of decode.
- **analyze_msk144-style PSD (dB/Hz)** vs **map144 FFT magnitude (dB vs
  full_scale=1)** are different units.
- **Safety — no clipping for valid signals:** use **scaling / headroom** (e.g.
  peak normalize to 0.9 FS before int16), not **hard clip** as policy for
  on-air or test vectors. `np.clip(±1)` is a last resort for malformed files only.
- **Deliverable:** one short spec (module doc or this doc’s “Contract” section
  expanded) + **named constants** (`FLEX_DAXIQ_FULL_SCALE`, jt9 headroom,
  generator peak factor) + **capture metadata** (rate, nominal noise floor,
  float vs int16).

---

## 2. analyze_msk144.py

- **Direction:** retire when Analysis window fully replaces it.
- **Before removal:** update `.vscode/launch.json`, `generate_msk144.py`
  header, `test_square.py` comments.
- **Optional migration:** matplotlib diagnostic (heatmap + lo/hi traces + seven
  stacked squared spectra, selected channel ±3) only if still needed after
  instrumentation below.

---

## 3. Auto-decode vs click-decode

**Observation:** visible signal, auto fails, Analysis click succeeds →
`extract_and_decode` / jt9 are likely fine; gap is **trigger / time / frequency /
gating**.

**Hypotheses**

- **Wrong automatic frequency:** `fc_hz` from squared-domain peaks is sometimes
  off; clicking **blob center** fixes decode (mixer alignment vs jt9 search window).
- **Strong ping never reached jt9:** **coincidence / spread gate** may classify
  wide sidelobe energy as impulse noise (`_COINCIDENCE_MAX_CH`,
  `_COINCIDENCE_SPAN_CH`, buffer flush).
- **Earlier list:** no launch (cooldown, dup thread, thread cap, edge/DC skip);
  bad `detect_sample` vs click time; live vs replay (blanker, etc.).

**Work order**

1. **Instrumentation** — log or debug UI: skip reason (coincidence vs sustained vs
   cooldown vs dup-thread vs cap), and on fire: channels, span, `fc_hz`,
   `detect_sample` / time.
2. **A/B on Analysis** — same buffer: decode with **auto** `(fc_hz, t)` vs
   **click** `(fc_hz, t)`.
3. **Overlay** — markers for auto trigger time/frequency vs click.
4. **Algorithm tweaks** — only after (1)–(2), one change at a time, with saved
   captures.
5. **Rich squared-spectrum panels** — only if `fc_hz` still unclear.

**Habit:** when auto misses and click hits, save capture + note time/dial; append
a one-line case to your local `dsp-reliability-plan.md`.

---

## 4. Local checklist (copy to gitignored file and tick)

- [ ] Instrumentation: skip reasons + fire parameters logged
- [ ] A/B: auto vs click on same IQ
- [ ] Overlay: auto vs click on spectrogram/heatmap
- [ ] Level spec + constants centralized
- [ ] Coincidence / fc / timing change (evidence-based, one at a time)
- [ ] Retire `analyze_msk144` + repo cleanups
- [ ] Optional: migrate 7-stack squared-spectrum diagnostic
