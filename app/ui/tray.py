"""Tray icon: a coloured dot that says what the app is doing.

QSystemTrayIcon rather than pystray, so there is one event loop and one dependency
instead of two. The icon is painted with QPainter, so Pillow is not needed either.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from . import theme
from ..controller import State

STATE_COLOURS = {
    State.IDLE: theme.IDLE,
    State.RECORDING: theme.RECORDING,
    State.PROCESSING: theme.PROCESSING,
    State.ERROR: theme.ERROR,
}

STATE_LABELS = {
    State.IDLE: "Ready",
    State.RECORDING: "Recording",
    State.PROCESSING: "Transcribing",
    State.ERROR: "Problem",
}


def dot_icon(colour: str, size: int = 64) -> QIcon:
    """A filled circle, drawn at 64px so Windows can scale it down cleanly."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(colour))
    margin = size // 8
    painter.drawEllipse(QRect(margin, margin, size - 2 * margin, size - 2 * margin))
    painter.end()
    return QIcon(pixmap)


class Tray(QSystemTrayIcon):
    def __init__(self, on_open: Callable[[], None], on_fix: Callable[[], None],
                 on_quit: Callable[[], None]) -> None:
        super().__init__()
        self._icons = {state: dot_icon(colour) for state, colour in STATE_COLOURS.items()}
        self._state = State.IDLE
        self._message = "Ready"

        menu = QMenu()
        open_action = QAction("Open Transcriber", menu)
        open_action.triggered.connect(lambda: on_open())
        menu.addAction(open_action)
        fix_action = QAction("Fix last dictation...", menu)
        fix_action.triggered.connect(lambda: on_fix())
        menu.addAction(fix_action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(lambda: on_quit())
        menu.addAction(quit_action)
        self.setContextMenu(menu)

        # Left click opens the window; the context menu handles the rest.
        self.activated.connect(
            lambda reason: on_open() if reason == QSystemTrayIcon.Trigger else None
        )
        self.set_state(State.IDLE)

    def set_state(self, state: State) -> None:
        self._state = state
        self.setIcon(self._icons[state])
        self._refresh_tooltip()

    def set_message(self, message: str) -> None:
        self._message = message
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        self.setToolTip(f"Transcriber - {STATE_LABELS[self._state]}\n{self._message}")
