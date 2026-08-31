# map144 — MSK144 Meteor Scatter Decoder

map144 is a real-time MSK144 meteor scatter decoder for amateur radio. It monitors
the MSK144 calling frequency and ±20 kHz either side, detects signals and decodes
them using the `jt9` engine from [WSJT-X](https://wsjt.sourceforge.io/).

Decoded contacts are reported to [PSKReporter](https://pskreporter.info),
[GridTracker](https://gridtracker.org), N1MM Logger+, and DX cluster nodes via
the standard WSJT-X UDP protocol.

## Install / run by platform

| Platform | Guide | Install | Run |
|----------|--------|---------|-----|
| **Linux** (primary) | **[README-Linux.md](README-Linux.md)** | `./install.sh` | `./run.sh` |
| **Windows** | **[README-Windows.md](README-Windows.md)** | `.\install.ps1` | `.\run.bat` |

Linux is the best-tested path (Ubuntu 22.04 / 24.04). Windows works but has had
less soak testing.

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

| Source                  | Interface                          | Notes |
| ----------------------- | ---------------------------------- | ----- |
| FlexRadio 6000 series   | DAXIQ via TCP/UDP (SmartSDR)       | Linux + Windows |
| USRP (Ettus Research)   | UHD Python API (`python3-uhd`)     | Linux (system UHD) |
| Airspy HF+              | libairspyhf (ctypes)               | Linux primarily |
| RTL-SDR (NooElec NESDR) | librtlsdr (ctypes)                 | Linux primarily |
| IQ WAV file             | stereo float32 WAV, left=I right=Q | Linux + Windows |

## How It Works

1. IQ samples arrive from the radio at 48 kHz
2. An optional noise blanker removes impulsive interference
3. A 48-channel polyphase channelizer splits the band into 1 kHz-spaced channels at 12 kHz each
4. Each channel is squared to produce a double-frequency tone; an FFT detects the
   paired tones that characterise an MSK144 burst
5. When a channel exceeds the SNR threshold the surrounding 15-second IQ window is
   saved and passed to `jt9` for decoding
6. Decoded messages are displayed in the GUI decode panel and forwarded to reporting services

## Related docs

- [README-Linux.md](README-Linux.md) — Linux install, update, run
- [README-Windows.md](README-Windows.md) — Windows install, update, run
- [docs/ALPHA_NOTES.md](docs/ALPHA_NOTES.md) — alpha tester notes
- [CHANGELOG.md](CHANGELOG.md)

## License

GNU General Public License v3 — see [LICENSE](LICENSE) or
<https://www.gnu.org/licenses/gpl-3.0.html>.

Copyright © 2026 Jeff Millar, WA1HCO \<wa1hco@gmail.com\>
