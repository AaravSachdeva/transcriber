"""The application window: a sidebar with the live status and navigation, and a stack
of four screens.

Closing the window hides it to the tray rather than quitting, which is what a dictation
tool should do. Quitting is explicit, from the tray menu.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QCloseEvent, QPalette
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QStackedWidget, QVBoxLayout, QWidget,
)

from . import theme
from .history import HistoryScreen
from .settings import SettingsScreen
from .stats import StatsScreen
from .tray import STATE_COLOURS, STATE_LABELS, dot_icon
from .vocabulary import VocabularyScreen
from ..config import AppConfig
from ..controller import Controller, State
from ..store import Store

SCREENS = ["History", "Stats", "Vocabulary", "Settings"]


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, store: Store, controller: Controller) -> None:
        super().__init__()
        self._controller = controller
        self.setWindowTitle("Transcriber")
        self.setWindowIcon(dot_icon(self.palette().color(QPalette.Accent).name()))
        self.resize(1040, 720)
        self.setMinimumSize(QSize(820, 560))

        # --- sidebar: what the app is doing, then where to go ---
        self._dot = QLabel()
        self._state = theme.label("", "SectionTitle")
        state_row = QHBoxLayout()
        state_row.setSpacing(10)
        state_row.addWidget(self._dot)
        state_row.addWidget(self._state, 1)

        self._status = theme.label("Starting...", "Caption", wrap=True)

        self._nav = QListWidget()
        self._nav.setFrameShape(QFrame.NoFrame)
        self._nav.viewport().setAutoFillBackground(False)
        for name in SCREENS:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(0, 38))
            self._nav.addItem(item)
        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(self._on_nav)

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 24, 8, 16)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addLayout(state_row)
        sidebar_layout.addWidget(self._status)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self._nav, 1)

        # --- screens ---
        self.history = HistoryScreen(store)
        self.stats = StatsScreen(store)
        self.vocabulary = VocabularyScreen(cfg)
        self.settings = SettingsScreen(cfg)

        self._stack = QStackedWidget()
        for screen in (self.history, self.stats, self.vocabulary, self.settings):
            self._stack.addWidget(screen)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sidebar)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        controller.stateChanged.connect(self._on_state)
        controller.statusMessage.connect(self._on_message)
        controller.dictationFinished.connect(self._on_dictation)
        self._on_state(controller.state)

    def _on_nav(self, row: int) -> None:
        self._stack.setCurrentIndex(row)
        # Recompute on arrival rather than on every dictation, so a long session does
        # not keep re-querying screens nobody is looking at.
        if self._stack.currentWidget() is self.stats:
            self.stats.reload()
        elif self._stack.currentWidget() is self.history:
            self.history.reload()

    def _on_state(self, state: State) -> None:
        self._dot.setPixmap(dot_icon(STATE_COLOURS[state]).pixmap(14, 14))
        self._state.setText(STATE_LABELS[state])

    def _on_message(self, message: str) -> None:
        self._status.setText(message)

    def _on_dictation(self, _row_id: int) -> None:
        if self._stack.currentWidget() is self.history:
            self.history.reload()
        elif self._stack.currentWidget() is self.stats:
            self.stats.reload()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hide to the tray. Quit is explicit, from the tray menu."""
        event.ignore()
        self.hide()

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def fix_last(self) -> None:
        """The "that was wrong" entry point: History, newest dictation, correction open."""
        self.show_and_raise()
        self._nav.setCurrentRow(0)
        self.history.correct_latest()
