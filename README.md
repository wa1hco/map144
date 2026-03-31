# map144 — MSK144 Meteor Scatter Decoder

map144 is a real-time MSK144 meteor scatter decoder for amateur radio.  It monitors
the MSK144 calling frequency and ±20 kHz either side, detects bursts using a
48-channel polyphase channelizer, and decodes them using the jt9 engine from
[WSJT-X](https://wsjt.sourceforge.io/).

Decoded contacts are reported to [PSKReporter](https://pskreporter.info),
[GridTracker](https://gridtracker.org), N1MM Logger+, and DX cluster nodes via
the standard WSJT-X UDP protocol.

## Features

- **48-channel polyphase channelizer** — resolves ±24 kHz into 1 kHz sub-bands
- **Paired-tone detection** — identifies MSK144 bursts by their squared-domain signature
- **Coincidence gate** — suppresses broadband noise events (lightning, static crashes)
- **Noise blanker** — removes impulsive interference before channelization
- **Live spectrogram** — accumulated and real-time IQ spectrograms with colour scale
- **Detection heatmap** — SNR history across all 48 channels with decode markers
- **Reporting** — PSKReporter (IPFIX UDP), WSJT-X UDP (GridTracker / N1MM / JTAlert), DX cluster (telnet)
- **Headless mode** — runs without GUI for unattended overnight operation
- **WAV playback** — replay saved IQ files for testing and development

## Supported Radio Sources

| Source | Interface |
|--------|-----------|
| FlexRadio 6000 series | DAXIQ via TCP/UDP (SmartSDR) |
| USRP (Ettus Research) | UHD via GNU Radio |
| Airspy HF+ | libairspyhf (ctypes) |
| RTL-SDR (NooElec NESDR) | librtlsdr (ctypes) |
| IQ WAV file | stereo float32 WAV, left=I right=Q |

## Requirements

- Python 3.10+
- [WSJT-X](https://wsjt.sourceforge.io/) installed (provides the `jt9` and `msk144sim` binaries)
- Linux (tested on Ubuntu 22.04 / 24.04)

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

### GUI mode (default)

```bash
.venv/bin/python map144.py
```

Select the radio source from the **Source** menu.  The spectrogram and detection
panels update in real time as IQ data arrives.

### WAV file playback

```bash
.venv/bin/python map144.py --source wav --wav path/to/file.wav
```

### Headless (no GUI)

```bash
.venv/bin/python map144.py --headless --source radio
```

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--rate HZ` | 48000 | IQ sample rate in Hz |
| `--source {radio,wav}` | radio | Source mode for headless operation |
| `--wav PATH` | — | WAV file path for `--source wav` |
| `--bind-client-id UUID` | — | FlexRadio client UUID for `client bind` |
| `--headless` | — | Run without GUI |
| `--log-level LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR / CRITICAL |

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
.venv/bin/python generate_msk144_test_signal.py --count 10

# Real callsign messages (for testing reporting and GridTracker)
.venv/bin/python generate_msk144_test_signal.py --count 10 --callsigns
```

Output is written to `MSK144/simulations/`.  Load with **Source → WAV File** in the GUI.

## Directory Structure

```
map144.py                   — main entry point
map144gui/
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
generate_msk144_test_signal.py  — synthetic IQ test vector generator
```

## How It Works

1. IQ samples arrive from the radio at 48 kHz
2. An optional noise blanker removes impulsive interference
3. A 48-channel polyphase channelizer splits the band into 1 kHz sub-bands at 12 kHz each
4. Each channel is squared to produce a double-frequency tone; an FFT detects the
   paired tones that characterise an MSK144 burst
5. When a channel exceeds the SNR threshold the surrounding 15-second IQ window is
   saved and passed to `jt9` for decoding
6. Decoded messages are displayed in the GUI decode panel and forwarded to reporting services

## License

GNU General Public License v3 — see [LICENSE](LICENSE) or
<https://www.gnu.org/licenses/gpl-3.0.html>.

Copyright © 2026 Jeff Millar, WA1HCO \<wa1hco@gmail.com\>
