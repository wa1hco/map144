# B210 Audio / IQ Router

Standalone derivative of MAP144: **band → WSJT-X dial list → named virtual
audio ports** (+ optional MAP65 / QMAP TIMF2 IQ).

## What it does

1. You pick a band (6 m, 2 m, 70 cm, … — B210-tunable / VHF+).
2. The program lists WSJT-X working dial frequencies for that band (FT8, FT4,
   MSK144, Q65, …).
3. You select the dials you want (and optionally MAP65 / QMAP).
4. **Apply** retunes the B210 so one IF covers the selection, creates one
   virtual audio port per dial, and streams USB-like audio into each.
5. Use the inline **USRP B210** panel (gain / antenna / status) and **Noise
   Blanker** panel (backend + K) — same controls as MAP144; live while running.
6. Point each WSJT-X instance at the matching input (see platform notes below).

This process **owns the B210** while running. Do not run MAP144 against the
same radio at the same time.

## Run

### Linux

```bash
./run-router.sh
# or:
python3 router_app.py          # auto-re-execs into .venv when present
```

### Windows

**Full install & start guide:** [`router-windows.md`](router-windows.md)

```bat
run-router.bat
```

Short form once MAP144+B210 and VB-CABLE are already installed:

1. `env\python.exe -m pip install sounddevice` (once)
2. `run-router.bat`
3. Band → select dials → **Apply**
4. WSJT-X **Audio Input** = **CABLE Output** (or Cable A/B Output)

**macOS:** install [BlackHole](https://existential.audio/blackhole/) and
`sounddevice`.

## WSJT-X Audio Input

| Platform | What to select in WSJT-X |
|---|---|
| Linux | `Monitor of MAP144 … -> WSJT-X` (PipeWire null-sink monitor) |
| Windows | `CABLE Output` / `Cable A Output` / VoiceMeeter output (the *Output* side of the cable we write into as *Input*) |
| macOS | `BlackHole 2ch` (or 16ch) |

### Device selection (Windows / macOS)

Priority:

1. Explicit per call / GUI (future)
2. `MAP144_WSJTX_DEVICE_RF0` / `_RF1`
3. `MAP144_WSJTX_DEVICE` — single name, or comma-list for dial/RF index 0,1,…
4. Auto-match: VB-CABLE / Cable A/B / VoiceMeeter / BlackHole

Examples:

```bat
set MAP144_WSJTX_DEVICE=CABLE Input
set MAP144_WSJTX_DEVICE=Cable A Input, Cable B Input
```

## Signal path

```
B210 raw IQ @ 192 kHz (DC = pan_center)
  ├─ per selected dial:
  │    NCO (usb_center = dial+1500 → DC) → /16 → 12 kHz IQ
  │    → WsjtxAudioExporter (DC→1500 Hz audio) → Transport
  │         Linux: PipeWire null sink + paplay
  │         Win/mac: PortAudio → virtual cable
  └─ optional MAP65 / QMAP:
       NCO → 96 kHz → Linrad TIMF2 UDP
```

On Linux, PipeWire resamples 12→48 kHz and absorbs B210-vs-system clock drift.
On Windows/macOS the host API may resample; residual clock drift can cause rare
glitches (same class of issue as any SDR→virtual-cable bridge).

## Naming (Linux sinks)

| Role | Example |
|---|---|
| Sink name (machine) | `map144.2m.msk144.144150.rx` |
| Description (GUI) | `MAP144 2m MSK144 144.150 -> WSJT-X` |
| WSJT-X Input | `Monitor of MAP144 2m MSK144 144.150 -> WSJT-X` |

## Library layout

```
map144_app/router/
  band_plan.py    # frequency table
  lo_planner.py   # pan + hw_rate from selection
  dial_audio.py   # per-dial NCO/decim + exporter bank
  wideband_iq.py  # MAP65/QMAP TIMF2
  engine.py       # lifecycle
  gui.py          # thin Qt UI
map144_app/wsjtx_audio_export.py   # DSP + PipeWire / PortAudio transports
map144_app/data/wsjtx_frequencies.json
router_app.py
run-router.sh / run-router.bat
```

## Limits (v1)

- Hardware rate fixed at **192 kHz** (selection that needs a wider IF errors out).
- No TX / PTT.
- Frequency table is a curated WSJT-X default subset; edit the JSON to extend.
- Multiple dials on Windows need multiple virtual cables (A+B or VoiceMeeter).
