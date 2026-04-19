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
"""Shared custom PyQtGraph/Qt widgets."""

from PyQt5 import QtCore
import pyqtgraph as pg


class ClickableImageItem(pg.ImageItem):
    """ImageItem that emits sigClicked(x_data, y_data, modifiers) on mouse press.

    x_data and y_data are in the ViewBox data coordinate system (not pixels),
    so callers receive (time_sec, freq_mhz) or (time_sec, freq_khz) directly.
    modifiers is a Qt.KeyboardModifiers bitmask — test with Qt.ShiftModifier etc.
    """

    sigClicked = QtCore.pyqtSignal(float, float, object)   # x, y, modifiers

    def mousePressEvent(self, event):
        vb = self.getViewBox()
        if vb is not None:
            pos = vb.mapSceneToView(event.scenePos())
            self.sigClicked.emit(pos.x(), pos.y(), event.modifiers())
        event.accept()   # prevent propagation into PyQtGraph drag machinery
