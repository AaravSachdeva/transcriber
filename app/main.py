"""Entry point.

    uv run python -m app.main            the app
    uv run python -m app.main --hidden   the app, straight to the tray (autostart uses this)
    uv run python -m app.main --setup    the installer's hardware test; --force reruns it

Qt owns the main thread. The Whisper model is loaded on a worker thread, so the UI stays
responsive through a model load, or the first-use download of a new model.
"""

from __future__ import annotations

import ctypes
import signal
import sys
import threading

from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from . import autosetup, autostart, config
from .controller import Controller
from .store import Store
from .ui import theme
from .ui.overlay import Overlay
from .ui.tray import Tray
from .ui.window import MainWindow


def _setup_logging() -> None:
    log_path = config.app_dir() / "transcriber.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="2 MB", retention=3, level="INFO", enqueue=True)


# Also the installer's AppMutex, so setup asks for the app to be closed before replacing it.
INSTANCE = "TranscriberRunning"
ERROR_ALREADY_EXISTS = 183


def _already_running() -> bool:
    """Autostart and the Start menu can both launch the app, and two instances would each
    paste every dictation. The mutex lives until the process exits."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, INSTANCE)
    return ctypes.get_last_error() == ERROR_ALREADY_EXISTS


def main() -> int:
    _setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Transcriber")
    theme.apply(app)

    if "--setup" in sys.argv:
        return autosetup.run(app, force="--force" in sys.argv)

    if _already_running():
        # Ask the running instance to show its window, so the Start menu entry is not a no-op.
        poke = QLocalSocket()
        poke.connectToServer(INSTANCE)
        poke.waitForConnected(1000)
        logger.info("Already running; raised the existing window.")
        return 0

    cfg = config.load()
    # Uninstall removes the login entry but keeps settings, and security software was seen
    # denying the installer's own write. Either way, this launch puts it back. On a thread:
    # 360 Total Security held the write for 31 seconds (its prompt) before denying it.
    if cfg.autostart and not autostart.is_enabled():
        threading.Thread(target=autostart.set_enabled, args=(True,), daemon=True).start()
    store = Store()
    # Closing the window hides it to the tray; without this the app would exit.
    app.setQuitOnLastWindowClosed(False)

    controller = Controller(cfg, store)
    window = MainWindow(cfg, store, controller)

    def quit_app() -> None:
        controller.stop()
        store.close()
        app.quit()

    tray = Tray(on_open=window.show_and_raise, on_fix=window.fix_last, on_quit=quit_app)
    controller.stateChanged.connect(tray.set_state)
    controller.statusMessage.connect(tray.set_message)
    tray.show()

    overlay = Overlay(lambda: controller.level, controller.cancel)
    controller.stateChanged.connect(overlay.set_state)

    server = QLocalServer()
    server.newConnection.connect(lambda: (server.nextPendingConnection(), window.show_and_raise()))
    server.listen(INSTANCE)
    if "--hidden" not in sys.argv:
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
