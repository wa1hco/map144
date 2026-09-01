# map144 on Windows

Windows is supported but has had **less soak testing** than Linux. Prefer the
python.org installer (not the Microsoft Store stub).

Product overview and radio sources: see the top-level [README.md](README.md).

## Requirements

- Python **3.10–3.14** from [python.org](https://www.python.org/downloads/)  
  - `install.ps1` **prefers 3.14**, then 3.13 → 3.12 → 3.11 → 3.10  
  - Requirements pin **`numpy==1.26.4`** (UHD ABI). Official PyPI builds of
    1.26.4 have **no cp314 wheels**, so a 3.14 attempt usually fails and
    `install.ps1` automatically retries with 3.13/3.12  
  - Tick **Add python.exe to PATH**  
  - Avoid the Microsoft Store `WindowsApps\python.exe` stub  
- [WSJT-X](https://wsjt.sourceforge.io/) (provides `jt9.exe`)
- Git for Windows (for `git clone` / `git pull`)
- PowerShell 5.1+ (built into Windows 10/11)

FlexRadio + WAV playback are the best-exercised Windows paths. USRP / Airspy /
RTL-SDR are primarily Linux-oriented.

## Install (first time)

In **PowerShell** or **cmd** (from the folder you want the tree in):

```powershell
git clone https://github.com/wa1hco/map144.git C:\WSJT\map144
cd C:\WSJT\map144
.\install.ps1
```

If execution policy blocks scripts, either:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

or right-click → **Run with PowerShell**. `install.ps1` / `run.bat` call
`.venv\Scripts\python.exe` directly — you do **not** need to activate the venv
or change execution policy permanently.

What `install.ps1` does (re-runnable = update):

- Creates/reuses `.venv`
- Upgrades pip and installs `requirements.txt`
- Verifies `jt9` discovery (warning only if missing)

## Update

```powershell
cd C:\WSJT\map144
git pull
.\install.ps1
```

## Run

```powershell
.\run.bat
```

Or:

```powershell
.venv\Scripts\python.exe map144.py
```

Optional arguments:

```powershell
.venv\Scripts\python.exe map144.py --bind-client-id <uuid> --log-level DEBUG
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bind-client-id UUID` | — | FlexRadio `client bind` when required |
| `--bind-client UUID` | — | Deprecated alias |
| `--log-level LEVEL` | INFO | `DEBUG` … `CRITICAL` |

Sample rate and source type are chosen in the app (**Source** menu).

First launch may take ~10 s while numba JIT-compiles hot paths.

Also available: `run-b210.bat`, `run-router.bat` for specialized station layouts
(see [docs/router-windows.md](docs/router-windows.md) if you use the router kit).

## Reporting setup

Open **View → Reporting**:

- **My Station** — callsign and grid (required)
- **WSJT-X UDP** — GridTracker / N1MM / JTAlert (port 2237)
- **PSKReporter** — IPFIX to `report.pskreporter.info:4739`
- **DX Cluster** — telnet (default `dxc.ve7cc.net:7373`)

## Test signals

```powershell
.venv\Scripts\python.exe generate_msk144.py --count 10
.venv\Scripts\python.exe generate_msk144.py --count 10 --callsigns
```

Output under `MSK144\simulations\`. Load with **Source → WAV File**.

## jt9 not found

Install WSJT-X so `jt9.exe` is discoverable, or set before launch:

```powershell
$env:MAP144_JT9 = "C:\Path\To\jt9.exe"
.\run.bat
```

## Common Windows pitfalls

| Symptom | Fix |
|---------|-----|
| `python` opens the Microsoft Store | Install python.org build; fix PATH so it wins over `WindowsApps` |
| pip “Building wheel for numpy” on 3.14 | Expected — `numpy==1.26.4` has no cp314 wheel. Re-run `.\install.ps1` (it falls back to 3.13/3.12), or install Python 3.12 and `rmdir /s /q .venv` first |
| `install.ps1` “cannot be loaded” | `powershell -ExecutionPolicy Bypass -File .\install.ps1` |
| Missing `jt9` | Install WSJT-X or set `MAP144_JT9` |
| First start feels hung | Wait for numba JIT (~10 s) |

## Linux?

Use **[README-Linux.md](README-Linux.md)** (`./install.sh` / `./run.sh`).
