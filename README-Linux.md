# map144 on Linux

Primary development and contest station platform (tested on **Ubuntu 22.04 / 24.04**).

Product overview and radio sources: see the top-level [README.md](README.md).

## Requirements

- Python **3.10+** (`python3` from your distro)
- [WSJT-X](https://wsjt.sourceforge.io/) (provides `jt9` / `msk144sim`)
- Git
- Optional: `python3-uhd` (apt) for USRP/B210 — `install.sh` uses a venv with
  `--system-site-packages` so UHD is visible
- Optional NVIDIA GPU: CuPy extras via `./install.sh --gpu` or auto-detect

## Install (first time)

```bash
git clone https://github.com/wa1hco/map144.git
cd map144
./install.sh
```

What `install.sh` does (re-runnable = update):

- Creates/reuses `.venv` (with system site-packages for UHD)
- `pip install -r requirements.txt`
- Optional GPU extras (`requirements-gpu.txt`) if `--gpu` / `MAP144_GPU=1` / NVIDIA detected
- Verifies `jt9` discovery (warning only if missing)

Override Python: `MAP144_PYTHON=/path/to/python3 ./install.sh`

## Update

From inside the repo:

```bash
git pull
./install.sh
```

## Run

```bash
./run.sh
```

Or:

```bash
.venv/bin/python map144.py
```

Optional arguments:

```bash
.venv/bin/python map144.py --bind-client-id <uuid> --log-level DEBUG
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bind-client-id UUID` | — | FlexRadio `client bind` when required |
| `--bind-client UUID` | — | Deprecated alias |
| `--log-level LEVEL` | INFO | `DEBUG` … `CRITICAL` |

Sample rate and source type are chosen in the app (**Source** menu), not on the CLI.

First launch may take ~10 s while numba JIT-compiles hot paths.

## Reporting setup

Open **View → Reporting**:

- **My Station** — callsign and grid (required)
- **WSJT-X UDP** — GridTracker / N1MM / JTAlert (port 2237)
- **PSKReporter** — IPFIX to `report.pskreporter.info:4739`
- **DX Cluster** — telnet (default `dxc.ve7cc.net:7373`)

## Test signals

```bash
.venv/bin/python generate_msk144.py --count 10
.venv/bin/python generate_msk144.py --count 10 --callsigns
```

Output under `MSK144/simulations/`. Load with **Source → WAV File**.

## jt9 not found

Install WSJT-X so `jt9` is on `PATH` (often `/usr/bin/jt9` or `~/.local/bin/jt9`),
or set:

```bash
export MAP144_JT9=/full/path/to/jt9
```

## USRP / B210 note

Install distro UHD bindings, e.g. Debian/Ubuntu:

```bash
sudo apt install python3-uhd
./install.sh   # venv must see system site-packages
```

## Windows?

Use **[README-Windows.md](README-Windows.md)** (`install.ps1` / `run.bat`).
