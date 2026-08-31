# B210 Audio / IQ Router

Standalone derivative of MAP144: **band → WSJT-X dial list → named PipeWire
ports** (+ optional MAP65 / QMAP TIMF2 IQ).

## What it does

1. You pick a band (6 m, 2 m, 70 cm, … — B210-tunable / VHF+).
2. The program lists WSJT-X working dial frequencies for that band (FT8, FT4,
   MSK144, Q65, …).
3. You select the dials you want (and optionally MAP65 / QMAP).
4. **Apply** retunes the B210 so one IF covers the selection, creates one
   PipeWire null sink per dial, and streams USB-like audio into each.
5. Point each WSJT-X instance at `Monitor of MAP144 … -> WSJT-X`.

This process **owns the B210** while running. Do not run MAP144 against the
same radio at the same time.

## Run

```bash
./run-router.sh
# or:
python3 router_app.py          # auto-re-execs into .venv when present
```

Use the project venv (same as `./run.sh` for MAP144).  A bare system
`python` will miss `numba` / UHD / PyQt deps.

Requires: UHD + B210, PipeWire/Pulse (`pactl` / `paplay`), PyQt5 (in `.venv`).

Opt out of nothing by default — sinks are created only for selected dials.

## Signal path

```
B210 raw IQ @ 192 kHz (DC = pan_center)
  ├─ per selected dial:
  │    NCO (usb_center = dial+1500 → DC) → /16 → 12 kHz IQ
  │    → WsjtxAudioExporter (DC→1500 Hz audio) → PipeWire sink
  └─ optional MAP65 / QMAP:
       NCO → 96 kHz → Linrad TIMF2 UDP
```

PipeWire resamples 12→48 kHz and absorbs B210-vs-system clock drift (same as
MAP144 TODO #44).

## Naming

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
map144_app/data/wsjtx_frequencies.json
router_app.py
```

## Limits (v1)

- Hardware rate fixed at **192 kHz** (selection that needs a wider IF errors out).
- Linux PipeWire/Pulse only (Windows/macOS = TODO #45 pattern).
- No TX / PTT.
- Frequency table is a curated WSJT-X default subset; edit the JSON to extend.
