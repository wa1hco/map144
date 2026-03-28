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
"""map144gui — modular radio IQ visualiser and MSK144 decoder.

Package structure
-----------------
visualizer.py   ``RadioIQVisualizer`` — PyQt5 QMainWindow subclass that owns
                all shared state (buffers, axes, configuration).  The
                processing, display, UI, and runtime functions are mixed in as
                methods so that each concern lives in its own file without
                requiring multiple classes or cross-module callbacks.

processing.py   ``process_iq_data`` — per-chunk DSP pipeline: LP filtering,
                ring-buffer write, overlap-FFT (normal and squared spectra),
                MSK144 tone-pair detection trigger, and spectrogram/energy
                buffer management.

detection.py    Stateless signal-processing helpers: FIR filter design,
                streaming filter application, squared-spectrum peak-pair
                scanner, carrier-frequency recovery, ring-buffer readout, and
                the full extract → mix → decimate → jt9 decode pipeline.

runtime.py      Source lifecycle and background thread: radio client
                startup/shutdown, WAV file loading and replay, sample ingress
                normalisation, tuned-frequency query, and window close handler.

displays.py     Qt rendering: pushes NumPy buffer data into pyqtgraph
                ImageItems and PlotCurveItems, updates noise-floor curves,
                energy overlays, status bar, UTC clock, and tuned-frequency
                label on every 100 ms timer tick.

ui.py           Widget construction: builds the five-panel grid layout,
                pyqtgraph plot widgets, colour map, slider bars, menu bar, and
                status bar; wires all interactive controls to their handlers.

Public API
----------
Only ``RadioIQVisualizer`` is exported; everything else is an implementation
detail internal to the package.
"""

from .visualizer import RadioIQVisualizer

__all__ = ["RadioIQVisualizer"]
