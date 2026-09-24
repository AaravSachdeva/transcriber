from __future__ import annotations

import time
from threading import Thread
from typing import Callable, Optional

from loguru import logger
from pynput import keyboard

# Right Ctrl is the trigger. It is deliberately NOT suppressed: pynput's
# win32_event_filter can suppress a key, but suppressing it also stops on_press and
# on_release from firing (pynput issue #679), so suppression and detection are mutually
# exclusive. Right Ctrl pressed alone is a no-op in essentially every application, so
# letting it through costs nothing.
TRIGGER = keyboard.Key.ctrl_r


class Dictation:
    """What the user asked for by the way they pressed the key."""

    START = "start"      # recording should begin
    STOP = "stop"        # recording should end and be transcribed
    NOTHING = "nothing"  # ignore


class HoldToggleHotkey:
    """Right Ctrl as both push-to-talk and a toggle.

    Held longer than `hold_threshold_ms` behaves as push-to-talk: recording starts on
    press and stops on release. A quicker tap latches recording on, and the next tap
    stops it. Holding Shift as well marks the dictation as an edit instruction for
    whatever text is currently selected.
    """

    def __init__(
        self,
        on_start: Callable[[bool], None],
        on_stop: Callable[[], None],
        hold_threshold_ms: int = 400,
    ) -> None:
        """:param on_start: called with is_edit=True when Shift was also held."""
        self._on_start = on_start
        self._on_stop = on_stop
        self._hold_threshold_s = hold_threshold_ms / 1000.0

        self._listener: Optional[keyboard.Listener] = None
        self._pressed_at: Optional[float] = None  # None means the key is not down
        self._shift_down = False
        self._latched = False  # recording continues after release

    # --- decision logic, kept free of pynput so it can be tested directly ---

    def press(self, now: float) -> str:
        """Right Ctrl went down. Windows auto-repeat fires this repeatedly while the
        key is held, so a press while already down is ignored."""
        if self._pressed_at is not None:
            return Dictation.NOTHING
        self._pressed_at = now
        if self._latched:
            # Key is down again while latched; the decision belongs to release().
            return Dictation.NOTHING
        return Dictation.START

    def release(self, now: float) -> str:
        """Right Ctrl came up. Long press ends push-to-talk; short press toggles."""
        pressed_at, self._pressed_at = self._pressed_at, None
        if pressed_at is None:
            return Dictation.NOTHING  # release without a press we saw

        held_s = now - pressed_at
        if self._latched:
            # Any release while latched stops it, long or short.
            self._latched = False
            return Dictation.STOP
        if held_s >= self._hold_threshold_s:
            return Dictation.STOP  # push-to-talk finished
        self._latched = True
        return Dictation.NOTHING  # tap: keep recording until the next tap

    @property
    def latched(self) -> bool:
        return self._latched

    def reset(self) -> None:
        """Forget all key state. Used when the pipeline aborts on an error."""
        self._pressed_at = None
        self._latched = False

    # --- pynput plumbing ---

    def _on_press(self, key) -> None:
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_down = True
            return
        if key != TRIGGER:
            return
        if self.press(time.monotonic()) == Dictation.START:
            is_edit = self._shift_down
            logger.info(f"Right Ctrl down: starting recording (edit={is_edit}).")
            self._on_start(is_edit)

    def _on_release(self, key) -> None:
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_down = False
            return
        if key != TRIGGER:
            return
        if self.release(time.monotonic()) == Dictation.STOP:
            logger.info("Right Ctrl up: stopping recording.")
            self._on_stop()
        elif self._latched:
            logger.info("Right Ctrl tapped: latched, recording until the next tap.")

    def start(self) -> None:
        def run() -> None:
            with keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            ) as listener:
                self._listener = listener
                listener.join()

        Thread(target=run, daemon=True).start()
        logger.info(f"Listening for right Ctrl (hold threshold {self._hold_threshold_s:.2f}s).")

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            logger.info("Hotkey listener stopped.")
