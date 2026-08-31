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
"""map144_app — MSK144 meteor scatter decoder application package.

map144 detects and decodes MSK144 meteor-scatter bursts from a FlexRadio
6000 series transceiver.  IQ samples are streamed via DAX IQ, channelized
into 1 kHz-spaced channels, and each channel is monitored for the paired-tone
signature of an MSK144 ping.  Detections are passed to the jt9 decoder from
WSJT-X; results are logged to launches.jsonl and saved as WAV files.

Package structure
-----------------
engine.py       ``Engine`` — Qt-free base class holding DSP state and
                ``process_iq_data``; subclassed by ``MAP144Visualizer``.

visualizer.py   ``MAP144Visualizer`` — PyQt5 QMainWindow that inherits Engine
                and adds the diagnostic GUI panels.

processing.py   ``process_iq_data`` — per-chunk DSP: wideband FFT blanker,
                channeliser, MSK144 detection, spectrogram / heatmap buffers.

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

router/         B210 multi-mode audio / IQ router (standalone derivative).

Public API
----------
``MAP144Visualizer`` is the main GUI entry; ``__version__`` is always
available.  Submodules (including ``router``) must be importable without
pulling the full GUI/DSP stack — hence the lazy ``MAP144Visualizer`` load.
"""

#: Package version.  Single source of truth — referenced by:
#:   - the main-window title bar (map144_app.displays / map144_app.ui)
#:   - the startup log line in map144.py
#:   - the "ready" banner printed by install.sh / install.ps1
#:   - bug reports (lands in MSK144/logs/map144_*.log automatically)
#: Bumped together with a git tag at each release.  See CHANGELOG.md.
__version__ = "0.1.3-alpha"

__all__ = ["MAP144Visualizer", "__version__"]


def __getattr__(name):
    # Lazy: importing map144_app.router (or __version__) must not require
    # numba / the full Engine+Visualizer stack.
    if name == "MAP144Visualizer":
        from .visualizer import MAP144Visualizer
        return MAP144Visualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
