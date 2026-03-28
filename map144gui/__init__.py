# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""map144gui — MSK144 meteor scatter decoder package.

map144 detects and decodes MSK144 meteor-scatter bursts from a FlexRadio
6000 series transceiver.  IQ samples are streamed via DAX IQ, channelized
into 1 kHz sub-bands, and each sub-band is monitored for the paired-tone
signature of an MSK144 ping.  Detections are passed to the jt9 decoder from
WSJT-X; results are logged to launches.jsonl and saved as WAV files.

Package structure
-----------------
engine.py       ``Engine`` — Qt-free base class containing all DSP state and
                the headless run loop.  Can be instantiated directly for
                unattended operation without a display.

visualizer.py   ``MAP144Visualizer`` — PyQt5 QMainWindow that inherits Engine
                and adds the diagnostic GUI panels.

processing.py   ``process_iq_data`` — per-chunk DSP pipeline: channelizer,
                SNR normalization, MSK144 tone-pair detection trigger, and
                spectrogram buffer management.

detection.py    ``extract_and_decode`` — ring-buffer readout, carrier
                recovery, decimation, jt9 invocation, result parsing, WAV
                and JSONL logging.

runtime.py      Source lifecycle: FlexRadio DAX IQ client startup/shutdown,
                WAV file replay, sample ingress, and tuned-frequency query.

displays.py     Qt rendering: live spectrogram, detection heatmap, SNR
                history, decode log, and status labels updated on a 100 ms
                timer tick.

ui.py           Widget construction: panel layout, pyqtgraph plots, colour
                map, sliders, menu bar, and control wiring.

channelizer.py  Polyphase channelizer filter design and state management.

Public API
----------
Only ``MAP144Visualizer`` is exported; everything else is an implementation
detail internal to the package.
"""

from .visualizer import MAP144Visualizer

__all__ = ["MAP144Visualizer"]
