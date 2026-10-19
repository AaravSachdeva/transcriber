"""Bottom-centre pill with live mic bars while dictating, and a × that cancels.

It never takes focus, even when clicked: the paste must land in the window that had it
when recording began.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme
from ..controller import State

# Small enough to sit over a document without hiding a line of it.
W, H, MARGIN = 120, 32, 24  # the × sits in the rightmost H x H
BARS = 11  # newest on the right; one per SAMPLE_TICKS frames, so about 0.7 s of speech
SAMPLE_TICKS = 4
NEON_BLUE = "#00b4ff"
# ponytail: fixed dB window fitted to one quiet mic (silence ~-60 dB, speech ~-40..-30 dB).
# Bars too twitchy in silence: raise FLOOR_DB. Speech never fills them: lower CEIL_DB.
FLOOR_DB, CEIL_DB = -58.0, -30.0


class Overlay(QWidget):
    def __init__(self, level: Callable[[], float], on_cancel: Callable[[], None]) -> None:
        # WindowDoesNotAcceptFocus is WS_EX_NOACTIVATE on Windows, so clicking the × leaves
        # focus where it was.
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(W, H)
        self._level = level
        self._on_cancel = on_cancel
        self._levels = [0.0] * BARS
        self._smooth = 0.0
        self._ticks = 0
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
            self._ticks += 1
            if self._ticks % SAMPLE_TICKS == 0:
                # A scrolling record of the real level, not a decorative waveform.
                self._levels = self._levels[1:] + [self._smooth]
        else:
            # Transcribing: a slow bump travels across the bars.
            self._levels = [0.4 * max(0.0, math.sin(t * 4 - i * 0.5)) for i in range(BARS)]
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.position().x() >= W - H:
            self._on_cancel()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Windows 11 flyout: dark fill, hairline light stroke so it holds on dark windows too.
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.setBrush(QColor(32, 32, 32, 242))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), H / 2, H / 2)
        p.setPen(Qt.NoPen)

        dot = NEON_BLUE if self._state == State.RECORDING else theme.PROCESSING
        p.setBrush(QColor(dot))
        p.drawEllipse(QPointF(H / 2, H / 2), 4, 4)

        p.setBrush(QColor(255, 255, 255, 215))
        bar_w, step, max_h = 2.5, 5.5, H - 14
        for i, v in enumerate(self._levels):
            h = max(bar_w, v * max_h)
            x = H - 2 + i * step
            p.drawRoundedRect(QRectF(x, (H - h) / 2, bar_w, h), bar_w / 2, bar_w / 2)

        cx, cy, r = W - H / 2, H / 2, 3.5
        hover = self.mapFromGlobal(QCursor.pos()).x() >= W - H and self.underMouse()
        if hover:
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawEllipse(QPointF(cx, cy), 10, 10)
        p.setPen(QPen(QColor(255, 255, 255, 255 if hover else 150), 1.5,
                      Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QLineF(cx - r, cy - r, cx + r, cy + r))
        p.drawLine(QLineF(cx - r, cy + r, cx + r, cy - r))
        p.end()
