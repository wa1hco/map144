# MAP144 — Alpha tester notes

Welcome, and thank you for testing. This page captures what MAP144 is, how to
run it, what to expect, and where to send feedback. Read it once before your
first run.

## What MAP144 is

A weak-signal DSP and decode pipeline for 144/50 MHz MSK144 meteor-scatter
operation. It runs alongside (or in place of) WSJT-X, taking IQ from a Flex
SmartSDR DAX-IQ stream and producing decodes to a GUI decode panel,
PSKReporter, WSJT-X UDP, and DXcluster.

**Mission priority (please keep this in mind when reporting):**

1. **PRIMARY** — decode MS pings (short bursts, random timing within the 15-s period)
2. **SECONDARY** — decode weak tropo / weak fading distant signals
3. **NOT THE GOAL** — strong locals, strong Es, regular tropo at workable SNR; those go to FT8 / SSB / CW

A missed MS ping is a sensitivity bug worth reporting. A missed strong-local
re-decode is usually not a bug — MAP144 deliberately caps launches per channel
to conserve CPU.

## What's in this alpha

- Real-time MSK144 detect + decode pipeline (legacy DSP path is the default)
- Side-by-side comparison vs WSJT-X via `compare_decoders.py`
- Captures browser with substring filter + scrollable list (right-click a
  callsign in the comparison GUI)
- Block-pipeline architecture available behind a flag (see below — *not*
  enabled by default)

## How to run

Standard mode (legacy DSP path, the stable one):

```bash
python3 main.py
```

That's the default. Don't set any env var.

## Known limitations

- **Block-pipeline mode is opt-in only and not recommended yet.** Set
  `MAP144_USE_BLOCKS=1` to enable it. It runs the new dataflow architecture
  as primary detection + decode. Currently has a higher launch rate than
  legacy with a different launch-to-decode profile; under investigation.
  Default is off; flip back by unsetting the env var. No code change.
- **Heatmap launch-marker overlay** is dark in block-primary mode. The decode
  list and reporting work; only the heatmap circles are missing. Legacy mode
  is unaffected.
- **The flag's old "shadow mode" semantics are gone.** If you previously set
  `MAP144_USE_BLOCKS=1` and got both pipelines running in parallel, that
  configuration no longer exists.

## What to report

In order of usefulness:

1. **Missed MS pings** — a station you saw decoded in WSJT-X on the same
   audio but not in MAP144's decode panel. Include the timestamp +
   callsign + roughly the WSJT-X SNR / dt if you have it. The 15-s period
   is the unit we compare on. `compare_decoders.py --date YYYYMMDD` will
   produce a side-by-side report you can attach.
2. **Phantom decodes** — anything in the decode panel that looks like garbage
   (random-character callsigns, impossible grids). MAP144 has zero phantoms
   so far in overnight runs; one would be notable. The format check is
   `[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z]` for callsigns and
   `[A-R]{2}\d{2}([A-X]{2})?` for grids.
3. **GUI weirdness** — anything stalled, blank, mislabeled, or drawing oddly.
4. **Crashes / tracebacks** — please include the log file at
   `MSK144/logs/map144_<timestamp>.log` if possible.

## Useful files

- `MSK144/detections/decodes.jsonl` — the authoritative decode log
- `MSK144/detections/launches.jsonl` — per-launch metrics (every detection
  candidate, decoded or not)
- `MSK144/detections/*.wav` — captured 1.7-s IQ around each launch
- `MSK144/logs/map144_*.log` — Python log; includes errors and tracebacks

## Tools you may find useful

- `compare_decoders.py --date YYYYMMDD` — interactive comparison vs WSJT-X
- `compare_decoders.py --date YYYYMMDD --utc HH:MM-HH:MM` — narrow the window
- Right-click a callsign in the comparison GUI to launch `analyze_msk144.py`
  on a specific WAV
- `analyze_msk144.py <wav>` — single-WAV spectrogram + per-frame analysis

## Sending feedback

If you can attach `MSK144/detections/decodes.jsonl` and the relevant
`map144_*.log` from the timeframe in question, that's enough to reproduce
most issues. WSJT-X `ALL.TXT` for the same window is also gold for the
"WSJT-X decoded it, MAP144 didn't" cases.
