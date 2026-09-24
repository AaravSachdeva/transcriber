"""Entry point.

    uv run python -m app.main

Qt owns the main thread. The Whisper model is loaded on a worker thread, so the UI stays
responsive through a model load, or the first-use download of a new model.
"""

from __future__ import annotations

import signal
import sys
import threading

from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from . import config
from .controller import Controller
from .store import Store
from .ui import theme
from .ui.tray import Tray
from .ui.window import MainWindow


def _setup_logging() -> None:
    log_path = config.app_dir() / "transcriber.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="2 MB", retention=3, level="INFO", enqueue=True)


def main() -> int:
    _setup_logging()
    cfg = config.load()
    store = Store()

    app = QApplication(sys.argv)
    app.setApplicationName("Transcriber")
    theme.apply(app)
    # Closing the window hides it to the tray; without this the app would exit.
    app.setQuitOnLastWindowClosed(False)

    controller = Controller(cfg, store)
    window = MainWindow(cfg, store, controller)

    def quit_app() -> None:
        controller.stop()
        store.close()
        app.quit()

    tray = Tray(on_open=window.show_and_raise, on_quit=quit_app)
    controller.stateChanged.connect(tray.set_state)
    controller.statusMessage.connect(tray.set_message)
    tray.show()

    window.show()

    # Ctrl+C in a terminal: Python signal handlers only run between bytecode
    # instructions, and Qt's event loop blocks in C++. A periodic no-op timer gives the
    # interpreter a chance to notice the signal.
    signal.signal(signal.SIGINT, lambda *_: quit_app())
    heartbeat = QTimer()
    heartbeat.start(400)
    heartbeat.timeout.connect(lambda: None)

    # Off the GUI thread: a model not yet in the HuggingFace cache downloads here (1.6 GB
    # for large-v3-turbo), and blocking the event loop that long reads as "Not Responding".
    threading.Thread(target=controller.start, daemon=True).start()

    logger.info("Transcriber started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
