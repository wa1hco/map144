# map144 — MSK144 Meteor Scatter Decoder

map144 is a real-time MSK144 meteor scatter decoder for amateur radio. It monitors
the MSK144 calling frequency and ±20 kHz either side, detects signals and decodes them using the jt9 engine from
[WSJT-X](https://wsjt.sourceforge.io/).

Decoded contacts are reported to [PSKReporter](https://pskreporter.info),
[GridTracker](https://gridtracker.org), N1MM Logger+, and DX cluster nodes via
the standard WSJT-X UDP protocol.

## Features

- **48-channel polyphase channelizer** — resolves ±24 kHz into 1 kHz-spaced channels
- **Paired-tone detection** — identifies MSK144 bursts by their squared-domain signature
- **Coincidence gate** — suppresses broadband noise events (lightning, static crashes)
- **Noise blanker** — removes impulsive interference before channelization
- **Live spectrogram** — accumulated and real-time IQ spectrograms with colour scale
- **Detection heatmap** — SNR history across all 48 channels with decode markers
- **Reporting** — PSKReporter (IPFIX UDP), WSJT-X UDP (GridTracker / N1MM / JTAlert), DX cluster (telnet)
- **WAV playback** — replay saved IQ files from the **Source** menu for testing and development

## Supported Radio Sources

| Source                  | Interface                          |
| ----------------------- | ---------------------------------- |
| FlexRadio 6000 series   | DAXIQ via TCP/UDP (SmartSDR)       |
| USRP (Ettus Research)   | UHD Python API (`python3-uhd`)     |
| Airspy HF+              | libairspyhf (ctypes)               |
| RTL-SDR (NooElec NESDR) | librtlsdr (ctypes)                 |
| IQ WAV file             | stereo float32 WAV, left=I right=Q |

## Requirements

- Python 3.10+
- [WSJT-X](https://wsjt.sourceforge.io/) installed (provides the `jt9` and `msk144sim` binaries)
- Linux (tested on Ubuntu 22.04 / 24.04)
- Windows (but has less testing)

Python dependencies:

```
numpy
scipy
PyQt5
pyqtgraph
matplotlib
```

## Installation

```bash
git clone https://github.com/wa1hco/map144.git
cd map144
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Ensure `jt9` is on your PATH (installed with WSJT-X, typically at
`/usr/bin/jt9` or `~/.local/bin/jt9`).

## Usage

Run the GUI (the only entry point provided by `map144.py`):

```bash
.venv/bin/python map144.py
```

Optional arguments:

```bash
.venv/bin/python map144.py --bind-client-id <uuid> --log-level DEBUG
```

Select the radio or WAV source from the **Source** menu. The spectrogram and
detection panels update in real time as IQ data arrives. For IQ WAV files, use
**Source → WAV File** and choose a stereo float32 file (left=I, right=Q).

### Command-line options

| Option                  | Default | Description                                                                 |
| ----------------------- | ------- | ----------------------------------------------------------------------------- |
| `--bind-client-id UUID` | —     | FlexRadio `client bind client_id=<uuid>` when the radio requires binding    |
| `--bind-client UUID`    | —     | Deprecated alias for `--bind-client-id`                                       |
| `--log-level LEVEL`     | INFO    | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` for root and `flexclient` |

Sample rate and source type are chosen in the application (not via `map144.py`).

## Reporting Setup

Open **View → Reporting** to configure:

- **My Station** — callsign and grid square (required for all reporting)
- **WSJT-X UDP** — sends Heartbeat, Status, and Decode messages to GridTracker / N1MM / JTAlert on port 2237
- **PSKReporter** — uploads spots via IPFIX UDP to `report.pskreporter.info:4739` every 5 minutes
- **DX Cluster** — sends spots via telnet to any DX cluster node (default: `dxc.ve7cc.net:7373`)

## Test Signal Generator

Generate a synthetic IQ WAV file containing MSK144 bursts for pipeline testing:

```bash
# Diagnostic A-format messages (encode frequency, time, SNR, width)
.venv/bin/python generate_msk144.py --count 10

# Real callsign messages (for testing reporting and GridTracker)
.venv/bin/python generate_msk144.py --count 10 --callsigns
```

Output is written to `MSK144/simulations/`. Load with **Source → WAV File** in the GUI.

## Directory Structure

```
map144.py                   — main entry point
map144_app/
  visualizer.py             — MAP144Visualizer class and shared state
  engine.py                 — signal processing parameters and buffer setup
  ui.py                     — Qt UI layout, menus, sliders
  processing.py             — channelizer dispatch, detection, FFT pipeline
  channelizer.py            — 48-channel polyphase channelizer
  detection.py              — jt9 decode subprocess management
  displays.py               — spectrogram and heatmap rendering (100 ms timer)
  runtime.py                — radio source lifecycle, WAV playback, shutdown
  reporting.py              — WSJT-X UDP, PSKReporter IPFIX, DX cluster
  reporting_window.py       — reporting settings UI panel
  source_windows.py         — per-source status panels
  airspy_source.py          — Airspy HF+ source (libairspyhf ctypes)
  rtlsdr_source.py          — RTL-SDR source (librtlsdr ctypes)
  usrp_source.py            — USRP source (UHD)
flexclient/                 — FlexRadio TCP/VITA-49 client library
generate_msk144.py  — synthetic IQ test vector generator
```

## How It Works

1. IQ samples arrive from the radio at 48 kHz
2. An optional noise blanker removes impulsive interference
3. A 48-channel polyphase channelizer splits the band into 1 kHz-spaced channels at 12 kHz each
4. Each channel is squared to produce a double-frequency tone; an FFT detects the
   paired tones that characterise an MSK144 burst
5. When a channel exceeds the SNR threshold the surrounding 15-second IQ window is
   saved and passed to `jt9` for decoding
6. Decoded messages are displayed in the GUI decode panel and forwarded to reporting services

## License

GNU General Public License v3 — see [LICENSE](LICENSE) or
<https://www.gnu.org/licenses/gpl-3.0.html>.

Copyright © 2026 Jeff Millar, WA1HCO \<wa1hco@gmail.com\>
