"""Put text where the cursor is, without destroying what was on the clipboard.

Clipboard via Qt's QClipboard, the Ctrl+V keystroke via pynput's Controller. Neither
pyautogui nor pyperclip is needed: pynput is already a dependency for the hotkey, and Qt
is already a dependency for the window.

Must be called from the Qt GUI thread. QClipboard is not thread-safe, so the controller
marshals the call rather than letting a worker touch it.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Optional

from loguru import logger
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QGuiApplication
from pynput import keyboard

from app.winctx import user32

_controller = keyboard.Controller()

_MODIFIERS = (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
              keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)


def _clear_modifiers() -> None:
    """Synthesize key-up for modifiers the user may still be physically holding.

    The edit hotkey is right Ctrl + Shift, and the selection is copied while both are
    still down. Without this, the synthetic Ctrl+C reaches the application as
    Ctrl+Shift+C, which in Chrome and Edge opens developer tools instead of copying.

    A synthetic key-up clears the OS modifier state even though the physical key is
    still down. Modifier keys do not auto-repeat on Windows, so the state stays clear
    until the user physically releases and presses the key again - which is what we
    want for the rest of this dictation.
    """
    for key in _MODIFIERS:
        try:
            _controller.release(key)
        except Exception:
            pass  # not all of these exist on every layout


def _snapshot() -> Optional[QMimeData]:
    """Copy what is on the clipboard now so it can be put back afterwards.

    QClipboard hands out a borrowed QMimeData that is invalidated as soon as the
    clipboard changes, so every format is copied into a new object up front. Returns
    None when the clipboard is empty or unreadable.
    """
    source = QGuiApplication.clipboard().mimeData()
    if source is None:
        return None
    try:
        saved = QMimeData()
        for fmt in source.formats():
            saved.setData(fmt, source.data(fmt))
        return saved if saved.formats() else None
    except Exception as exc:
        logger.warning(f"Could not snapshot the clipboard ({exc}); it will not be restored.")
        return None


def _wait(ms: int) -> None:
    """Sleep on the GUI thread while still answering other apps' clipboard requests.

    Qt puts data on the clipboard with delayed rendering. When another app reads it,
    Windows sends WM_RENDERFORMAT to this thread, and the reader keeps the clipboard open
    until we answer. time.sleep() answers nothing: the reader stalls, the restore fails
    with CLIPBRD_E_CANT_OPEN (0x800401D0), and the failed restore drops the dictation, so
    the paste comes out blank. Ctrl+C stalls the same way on WM_DESTROYCLIPBOARD.

    Only messages sent from other threads are dispatched here. Qt events and timers do
    not run, so nothing can re-enter the controller during the wait.
    """
    msg = wintypes.MSG()
    end = time.monotonic() + ms / 1000.0
    while (left := end - time.monotonic()) > 0:
        # QS_SENDMESSAGE, MWMO_INPUTAVAILABLE
        user32.MsgWaitForMultipleObjectsEx(0, None, int(left * 1000) + 1, 0x40, 0x4)
        # PM_QS_SENDMESSAGE: dispatch sent messages only; posted ones stay queued for Qt.
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0x40 << 16)


def paste_text(text: str, settle_ms: int = 150, restore: bool = True) -> None:
    """Paste `text` at the cursor, then put the previous clipboard contents back.

    `settle_ms` is load bearing. Ctrl+V is asynchronous: the keystroke returns
    immediately and the receiving application reads the clipboard a moment later.
    Restoring too early truncates or blanks the paste in slower applications.
    """
    if not text:
        logger.warning("paste_text called with empty text; nothing to do.")
        return

    clipboard = QGuiApplication.clipboard()
    saved = _snapshot() if restore else None

    clipboard.setText(text)
    if not clipboard.ownsClipboard():
        # Qt has already retried for 300 ms. Ctrl+V now would paste the old clipboard.
        logger.error("Another app is holding the clipboard; nothing pasted.")
        return
    _clear_modifiers()
    with _controller.pressed(keyboard.Key.ctrl):
        _controller.tap("v")
    logger.info(f"Pasted {len(text.split())} words at the cursor.")

    if saved is None:
        return
    # ponytail: fixed settle delay, not a handshake. If a slow app ever pastes
    # truncated text, poll GetClipboardSequenceNumber instead of sleeping.
    _wait(settle_ms)
    clipboard.setMimeData(saved)
    if not clipboard.ownsClipboard():
        logger.warning("Could not restore the previous clipboard; another app is holding it.")


def copy_selection(settle_ms: int = 120) -> str:
    """Read whatever text is selected in the foreground app, via Ctrl+C.

    Returns "" when nothing was selected. The previous clipboard is restored either way,
    so voice-editing does not clobber it.
    """
    clipboard = QGuiApplication.clipboard()
    saved = _snapshot()

    # Clear first, so a failed copy is distinguishable from a copy of identical text.
    clipboard.clear()
    _clear_modifiers()
    with _controller.pressed(keyboard.Key.ctrl):
        _controller.tap("c")
    _wait(settle_ms)

    selected = clipboard.text() or ""
    if saved is not None:
        clipboard.setMimeData(saved)
        if not clipboard.ownsClipboard():
            logger.warning("Could not restore the previous clipboard; another app is holding it.")

    if not selected:
        logger.info("Ctrl+C produced nothing; treating it as an empty selection.")
    return selected
