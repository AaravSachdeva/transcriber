"""Bottom-centre pill with live mic bars while dictating.

It never takes focus: the paste must land in the window that had it when recording began.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from . import theme
from ..controller import State

BARS = 15
W, H, MARGIN = 180, 52, 24
NEON_BLUE = "#00b4ff"
# ponytail: fixed dB window fitted to one quiet mic (silence ~-60 dB, speech ~-40..-30 dB).
# Bars too twitchy in silence: raise FLOOR_DB. Speech never fills them: lower CEIL_DB.
FLOOR_DB, CEIL_DB = -58.0, -30.0
ENVELOPE = [0.4 + 0.6 * math.sin(math.pi * (i + 0.5) / BARS) for i in range(BARS)]


class Overlay(QWidget):
    def __init__(self, level: Callable[[], float]) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(W, H)
        self._level = level
        self._levels = [0.0] * BARS
        self._smooth = 0.0
        self._state = State.IDLE
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_state(self, state: State) -> None:
        self._state = state
        if state == State.RECORDING:
            self._levels, self._smooth = [0.0] * BARS, 0.0
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            area = screen.availableGeometry()
            self.move(area.center().x() - W // 2, area.bottom() - H - MARGIN)
            self.show()
            self._timer.start()
        elif state != State.PROCESSING:
            self._timer.stop()
            self.hide()

    def _tick(self) -> None:
        t = time.monotonic()
        if self._state == State.RECORDING:
            db = 20 * math.log10(max(self._level(), 1e-6))
            target = min(1.0, max(0.0, (db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)))
            # Instant attack, quick release: bars jump on a word and drop between words.
            self._smooth = target if target > self._smooth else self._smooth * 0.8
            self._levels = [self._smooth * e * (0.8 + 0.2 * math.sin(t * (6 + i % 5) + i))
                            for i, e in enumerate(ENVELOPE)]
        else:
            self._levels = [0.3 + 0.2 * math.sin(t * 6 - i * 0.8) for i in range(BARS)]
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(24, 24, 28, 225))
        p.drawRoundedRect(QRectF(self.rect()), H / 2, H / 2)
        colour = NEON_BLUE if self._state == State.RECORDING else theme.PROCESSING
        p.setBrush(QColor(colour))
        pad, bar_w, max_h = 20, 5, H - 12
        step = (W - 2 * pad) / BARS
        for i, v in enumerate(self._levels):
            h = max(bar_w, v * max_h)
            x = pad + i * step + (step - bar_w) / 2
            p.drawRoundedRect(QRectF(x, (H - h) / 2, bar_w, h), bar_w / 2, bar_w / 2)
        p.end()
