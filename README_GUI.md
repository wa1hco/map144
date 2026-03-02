# FlexRadio DAXIQ Visualizer GUI

Real-time I/Q data visualization for FlexRadio with three synchronized displays.

## Features

### Three-Panel Display:
1. **Center**: Spectrogram (15 seconds of history x frequency)
   - Shows time-frequency representation of received signals
   - Color-coded power levels (dB)
   
2. **Right**: Noise Floor vs Frequency
   - Vertical plot aligned with spectrogram frequency axis
   - Shows averaged noise floor across the band
   
3. **Bottom**: Energy vs Time  
   - Shows total received power over time
   - Useful for monitoring signal activity

## Requirements

- Python 3

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

### Basic usage (48 kHz sample rate):
```bash
.venv/bin/python flex_gui.py
```

### Custom sample rate:
```bash
.venv/bin/python flex_gui.py --rate 48000
```

### Bind to a specific SmartSDR GUI client UUID:
```bash
.venv/bin/python flex_gui.py --bind-client-id <uuid>
```

### Enable debug logging:
```bash
.venv/bin/python flex_gui.py --log-level DEBUG
```

Accepted log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

## Test Signal Generator

Generate the combined MSK144 test IQ stream and diagnostics plot:

```bash
.venv/bin/python generate_msk144_test_signal.py
```

All source bursts are aligned at `t=0` in the combined output stream.
Source center is fixed at `1500 Hz`; each burst is shifted from `1500 Hz` to its configured target center.

This opens the **interactive diagnostics GUI by default** and writes:
- `msk144_combined_iq_48k.npy`
- `msk144_combined_iq_48k.wav`
- `msk144_wav_diagnostics.png`

Diagnostics spectrum flattening (based on median noise floor vs frequency) is **enabled by default**.

To run without opening the live interactive diagnostics window (Qt/Matplotlib), pass:

```bash
.venv/bin/python generate_msk144_test_signal.py --no-show-plots
```

To disable frequency flattening for diagnostics (use raw spectrum), pass:

```bash
.venv/bin/python generate_msk144_test_signal.py --no-flatten-spectrum
```

Example (headless + raw spectrum):

```bash
.venv/bin/python generate_msk144_test_signal.py --no-show-plots --no-flatten-spectrum
```

## Before Running

1. Make sure FlexRadio is powered on and connected to your network
2. In SmartSDR, set the panadapter center/bandwidth as desired (GUI owns pan tuning)
3. In SmartSDR, enable DAXIQ channel 1 on slice A:
   - Open a panadapter
   - Right-click on slice A
   - Enable DAXIQ channel 1

## How It Works

The GUI connects to your FlexRadio via:
- **Discovery**: Finds radio on network via UDP broadcast
- **Control**: TCP connection on port 4992 for commands
- **Data**: Uses a dynamically selected local UDP port for I/Q streaming

The spectrogram is updated in real-time as IQ samples are received and processed with FFT.

## Files

- `flex_gui.py` - Compatibility launcher (still run this file)
- `flex_daxiq_gui/visualizer.py` - `DAXIQVisualizer` class and shared state
- `flex_daxiq_gui/ui.py` - UI layout and slider handlers
- `flex_daxiq_gui/processing.py` - FFT and buffer processing pipeline
- `flex_daxiq_gui/displays.py` - Plot/update rendering logic
- `flex_daxiq_gui/runtime.py` - Flex client thread lifecycle and shutdown
- `flexclient/` - Modular FlexRadio client package (`core`, `client`, `setup`, `tcp_client`, `vita`, `discovery`, `models`, `common`)
- `flex_client.py` - Legacy CLI launcher and compatibility shim for `flexclient.core`
- `install_gui_deps.sh` - Dependency installation script

## Troubleshooting

If the GUI doesn't display data:
1. Check that DAXIQ is enabled in SmartSDR on slice A
2. Verify the radio is on the same network as your computer
3. Check that no firewall is blocking the dynamically assigned local UDP listen port

### Bind and Diagnostics Behavior

At startup, the client may auto-bind to a discovered SmartSDR GUI client UUID (or use `--bind-client-id` when provided). Typical logs include:
- `Sending bind command: client bind client_id=<uuid>`
- `Bind command response payload: '...'`

Pan/stream troubleshooting probes (`display pan list`, `display pan all`, `stream list`) were removed because they are often rejected by radio policy and are not required for normal operation.

When bind succeeds but a control command is rejected, this typically indicates radio ownership/policy limits for that operation in the current firmware/context.

### Tuning Display

The GUI now displays pan tuning as:
- `Pan Center: xx.xxx MHz BW: zz kHz`

Pan center and bandwidth values come from SmartSDR status updates.

## Controls

- The display updates automatically every 100ms
- Close the window to stop streaming and disconnect
- Status bar shows packet count and elapsed time
