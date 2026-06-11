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

### All platforms

- **Python 3.10 or newer** (3.14 verified working as of 2026-05-18).
- **WSJT-X** — provides the `jt9` decoder.  Install from [wsjt.sourceforge.io](https://wsjt.sourceforge.io/).  MAP144 auto-discovers `jt9` in the standard install locations; you do **not** need to add it to PATH.  If you have a non-standard install, set the `MAP144_JT9` environment variable to the full path of `jt9` / `jt9.exe`.
- **Git** — to clone the repository.

### Windows-specific setup

1. **Install Python from [python.org](https://www.python.org/downloads/)** — *not* the Microsoft Store version (that's a stub that opens the Store on first use).  During install, tick "Add python.exe to PATH".  Verify with `python --version` in a new PowerShell window.
2. **Install Git** from [git-scm.com](https://git-scm.com/download/win).  Default options are fine.
3. **Install WSJT-X** from [wsjt.sourceforge.io](https://wsjt.sourceforge.io/).  MAP144 will auto-find `jt9.exe` at `C:\WSJT\wsjtx\bin\jt9.exe`, `C:\Program Files\WSJT-X\bin\jt9.exe`, or the standard Program Files (x86) path.
4. **Avoid OneDrive-synced install locations** (e.g. `C:\Users\<you>\OneDrive\Documents\…`).  MAP144 writes WAV captures and JSONL logs continuously; OneDrive sync churn against that volume causes lag.  Recommended location: `C:\map144`.
5. **PowerShell execution policy** — `install.ps1` and `run.bat` are designed to avoid this concern (they invoke `.venv\Scripts\python.exe` directly, so no `Activate.ps1` step is needed).  No execution-policy change required.

### Linux-specific setup

Your distro's `python3` package is fine on Ubuntu 22.04 / 24.04 and equivalent.  WSJT-X comes from the distro package (`sudo apt install wsjtx`) or upstream.

### Radio-specific setup

MAP144 supports several SDRs.  The pip dependencies in `requirements.txt` cover **FlexRadio** (works out of the box once installed below) and **NESDR / RTL-SDR** (Linux only; uses `librtlsdr`).  Other radios need extra setup:

- **FlexRadio 6000 series** — no extra setup; runs on Windows or Linux.  This is the best-tested path.
- **AirSpy HF+** — needs `libairspyhf`.  Linux: `sudo apt install libairspyhf1`.  Windows: download the AirSpy Windows driver from [airspy.com](https://airspy.com/download/) and ensure `airspyhf.dll` is on PATH.
- **RTL-SDR / NESDR Smart** — Linux only.  `sudo apt install librtlsdr2`.
- **USRP B210** — needs the UHD library and Python bindings.
    - *Linux:* `sudo apt install python3-uhd uhd-host && sudo uhd_images_downloader`
    - *macOS:* `brew install uhd && uhd_images_downloader`
    - *Windows:* **use the B210 install kit** — see "USRP B210 install kit (Windows)" below.  Manual recipe lives at the end of this file as a fallback.

      For Flex-only runs the pip `.venv` + `run.bat` path is fine — the B210 kit is only needed when you want to use the B210.

## USRP B210 install kit (Windows)

The kit is a single directory containing everything needed to install B210 support on a clean Windows 11 PC, with no internet access required at install time.  Use this in the field (mountaintop contests, etc.); use the manual recipe at the end of this file only if the kit isn't an option.

### What the kit contains

- Ettus UHD Windows installer (~233 MB) — WinUSB driver + FX3/FPGA firmware
- `map144-env.zip` (~500 MB) — portable Python env with `uhd` + map144 deps pre-installed
- `map144-src.zip` — MAP144 source at the build's git SHA
- `install-b210.bat` — double-click installer

Total kit size: ~1.5 GB.  Fits on any USB stick.

### Building the kit (home machine, one-time)

On a home machine with Miniconda + git installed, from the map144 repo root:

```powershell
.\tools\build-b210-kit.ps1
```

Output: `dist\map144-b210-kit-<gitSha>\` — copy this whole directory to a USB stick (or zip + upload as a GitHub release).

The build script downloads the Ettus installer (caches it in `tools\cache\`), creates a fresh conda env, packs it with `conda-pack`, snapshots the repo via `git archive`, and assembles the kit.  Takes ~15 minutes the first time, faster on re-runs.

### Installing on the contest PC

Prerequisites on the target PC: Windows 11 (or Win10 build 17763+), 3 GB free on `C:`, administrator access, WSJT-X installed separately for the `jt9` decoder.

1. Plug the B210 into a **USB 3.0** port (blue port) on the target PC.
2. Copy the kit directory to the PC (or leave it on the USB stick).
3. Double-click `install-b210.bat`.
4. Approve the UAC prompt.
5. Wait ~5 minutes.  The last line will read either `READY` or `FAILED: <reason>`.
6. Launch with `C:\map144\run-b210.bat`.

The installer is idempotent — re-running the same kit version is a no-op; re-running a different kit version prompts to overwrite.  Install log is at `C:\map144\install.log` for diagnosing failures after the fact.

### What the installer does

1. Self-elevates to admin (UAC).
2. Pre-flight: Windows version, disk space, not under OneDrive.
3. Silently installs Ettus UHD (skipped if already present).
4. Extracts the portable Python env to `C:\map144\env\`.
5. Runs `conda-unpack.exe` to rewrite the env's internal paths (this is the slow step).
6. Extracts the map144 source to `C:\map144\`.
7. Smoke tests: `uhd_find_devices` sees the B210; the env's `python.exe` can `import uhd`.

### Manual B210 setup (fallback when the kit isn't an option)

If you can't use the kit (no home machine to build it on, conda-pack not working on a particular Windows version, etc.), here's the by-hand recipe.  Verified end-to-end on Win11 with a B210 attached on 2026-06-02.

1. **Install the Ettus UHD Windows binary** — provides the WinUSB driver and bundled firmware images.  Grab `uhd_4.10.0.0-release_Win64_VS2022.exe` from <https://files.ettus.com/binaries/uhd/latest_release/Windows11/VS2022/>.  Default install location: `C:\Program Files\UHD\`.

2. **Plug in the B210** (USB 3.0 port).  In Device Manager, look under "USRPs" for an "Ettus" entry with no yellow ⚠.  Verify with `& "C:\Program Files\UHD\bin\uhd_find_devices.exe"`.

3. **Install Miniconda** from <https://docs.conda.io/projects/miniconda/>.  During install: "Install for Just Me", do not add to PATH, do not register as system Python.

4. **Create the conda env and install UHD + map144 deps** (in Anaconda Prompt):
   ```
   conda create -n map144 python=3.11 -y
   conda activate map144
   conda install -c conda-forge uhd -y
   cd C:\Users\<you>\Documents\map144
   pip install -r requirements.txt
   ```

5. **Smoke test** the Python bindings:
   ```
   python -c "import uhd; print(uhd.__version__)"
   ```
   Should print `4.10.0.0-release`.

6. **Launch with the conda env** (`run-b210.bat` auto-detects it):
   ```
   run-b210.bat
   ```

## Install

### Recommended location

- **Linux / macOS:** `~/map144` (in your home directory)
- **Windows:** `C:\map144` (top-level on the system drive)

You can install elsewhere if you prefer — the install script doesn't care about location.  (Windows OneDrive-path warning is covered in the prerequisites above.)

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
