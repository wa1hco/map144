# MAP144 — Alpha tester notes

Welcome, and thank you for testing. This page captures what MAP144 is, how to
run it, what to expect, and where to send feedback. Read it once before your
first run.

## What MAP144 is

Looks at 40 kHz of bandwidth around the meteor scatter calling frequency looking for signals, similar to map65 and QMAP from WSJTX family. It runs alongside WSJT-X, taking IQ from a FlexRadio or other DSP and producing decodes displayed on a GUI decode panel or sent to PSKReporter, N1MM, GridTracker, and DXcluster.

I've done most of my testing with FlexRadio and USRP B210, the program has been tested with Airspy HF+, and nesdr smart

The most significant feature, beside decoding over a wide bandwidth, is the ability to work with dual channel digital receivers and use dual polarization antennas.  Like map65, map144 finds the best combination of two antennas before decoding.  On 50 MHz, Faraday rotation makes the polarization of a received ping random.  map144 should have about 3 dB advantage over single polarity receivers.

**Mission priority (please keep this in mind when reporting):**

1. **PRIMARY** — decode MS pings (short bursts, random timing within the 15-s period)
2. **SECONDARY** — decode weak tropo / weak fading distant signals if using msk144
3. **NOT THE GOAL** — strong locals, strong Es, regular tropo at workable SNR; those go to FT8 / SSB / CW

A missed MS ping is a sensitivity bug worth reporting. A missed strong-local
re-decode is usually not a bug — MAP144 deliberately caps launches per channel
to conserve CPU.

## What's in this alpha
- Real-time MSK144 detect + decode pipeline
- Display windows for 
  - Radio interface status and settings
  - Fast Graph like wsjtx
  - Detection algorithms (two ways)
  - Noise blanker status and settings
  - Reporting options
- Side-by-side comparison vs WSJT-X via `compare_decoders.py` tool

## Prerequisites

- **Python 3.10 or newer** (3.14 verified working as of 2026-05-18).  Windows: install from [python.org](https://www.python.org/downloads/) — *not* the Microsoft Store version (that's a stub that opens the Store on first use).  Linux: your distro's `python3` package is fine on Ubuntu 22.04 / 24.04 and equivalent.
- **WSJT-X** — provides the `jt9` decoder.  Install from [wsjt.sourceforge.io](https://wsjt.sourceforge.io/) on either OS.  MAP144 auto-discovers `jt9` in the standard install locations; you do **not** need to add it to PATH.
- **Git** — to clone the repository.

## Install

### Recommended location

- **Linux / macOS:** `~/map144` (in your home directory)
- **Windows:** `C:\map144` (top-level on the system drive)

You can install elsewhere if you prefer — the install script doesn't care about location.  On Windows, **avoid OneDrive-synced paths** (e.g. `C:\Users\<you>\OneDrive\Documents\…`): MAP144 writes WAV captures and JSONL logs continuously, and OneDrive sync churn against that volume causes lag.

### Fresh install

#### Linux / macOS

```bash
git clone https://github.com/wa1hco/map144.git ~/map144
cd ~/map144
./install.sh
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/wa1hco/map144.git C:\map144
cd C:\map144
.\install.ps1
```

### Upgrade (existing install)

Pull the latest source and re-run the install script.  It's idempotent — reuses any existing `.venv`, re-resolves dependencies, re-verifies `jt9`.

#### Linux / macOS

```bash
cd ~/map144
git pull
./install.sh
```

#### Windows (PowerShell)

```powershell
cd C:\map144
git pull
.\install.ps1
```

The install script creates a `.venv/`, installs dependencies, verifies that `jt9` was found, and prints how to launch.

## How to run

### Linux / macOS (from the map144 directory)

```bash
./run.sh
```

### Windows (from the map144 directory)

```powershell
.\run.bat
```

That's the default — uses the stable legacy DSP path.  Don't set any env var.

The first launch may take ~10 s while numba JIT-compiles the channelizer hot paths.  Subsequent launches are fast (numba caches the compilation).

## Known limitations

map144 is currently about 1 dB less sensitive than WSJT-X.  Most of the time both programs decode the ping, but sometimes only one program catches the weak ping and mostly wsjtx wins.  I'm working on parity and eventually using more CPU power to beat WSJTX.

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
