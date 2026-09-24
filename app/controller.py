"""The dictation pipeline, and the state everything else reads.

Threading, which is the whole difficulty here:

- Qt owns the main thread and the event loop.
- pynput's listener runs on its own thread and calls straight into `_hotkey_start` /
  `_hotkey_stop`. Those only emit signals. With Qt's default AutoConnection an emit from
  another thread is queued onto the GUI thread, so every slot below runs there.
- PortAudio calls the recorder's callback on its own thread; the recorder only appends
  to a deque under a lock.
- Whisper and Ollama run on a plain worker thread and report back by signal.

Clipboard work is therefore always on the GUI thread, which is what QClipboard requires.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Optional

from loguru import logger
from PySide6.QtCore import QObject, Signal, Slot

from . import output, winctx
from .audio import AudioRecorder, RecordedAudio
from .config import AppConfig
from .hotkey import HoldToggleHotkey
from .llm import OllamaClient
from .store import Store
from .transcription import TranscriptionService, build_initial_prompt


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


def normalize_trigger(text: str) -> str:
    """Normalise a transcript for snippet lookup: lowercase, no surrounding
    punctuation or whitespace. Whisper punctuates, so 'my address.' must match the
    trigger 'my address'."""
    return text.strip().strip(".,!?;:\"'").strip().lower()


class Controller(QObject):
    stateChanged = Signal(State)
    dictationFinished = Signal(int)   # history row id, for the UI to refresh
    statusMessage = Signal(str)       # human-readable, for the tray tooltip

    # Emitted from the pynput listener thread; queued onto the GUI thread by Qt.
    _startRequested = Signal(bool)    # is_edit
    _stopRequested = Signal()
    _workerFinished = Signal(object)  # dict payload, or None when nothing to paste

    def __init__(self, cfg: AppConfig, store: Store) -> None:
        super().__init__()
        self.cfg = cfg
        self.store = store

        self._recorder = AudioRecorder(
            sample_rate=cfg.audio.sample_rate,
            channels=cfg.audio.channels,
            block_size=cfg.audio.block_size,
            device_index=cfg.audio.device_index,
        )
        self._transcriber: Optional[TranscriptionService] = None  # built by start()
        self._ollama = OllamaClient(cfg.ollama)
        self._hotkey = HoldToggleHotkey(
            on_start=lambda is_edit: self._startRequested.emit(is_edit),
            on_stop=lambda: self._stopRequested.emit(),
            hold_threshold_ms=cfg.hotkey.hold_threshold_ms,
        )

        self._state = State.IDLE
        # Captured when recording starts, because that is when the user was looking at
        # the target window. By the time text is pasted the focus is the same, but the
        # selection for an edit must be read before the user does anything else.
        self._foreground: Optional[winctx.Foreground] = None
        self._selection = ""
        self._is_edit = False

        self._startRequested.connect(self._on_start)
        self._stopRequested.connect(self._on_stop)
        self._workerFinished.connect(self._on_worker_finished)

    # --- lifecycle ---

    def start(self) -> None:
        """Load the model, warm it up, then begin listening for the hotkey.

        Runs on a worker thread; it only emits signals, and the hotkey that reads
        _transcriber is not started until the model is ready.
        """
        self.statusMessage.emit(
            f"Loading Whisper {self.cfg.whisper.model_size} (first use downloads it)..."
        )
        try:
            self._transcriber = TranscriptionService(self.cfg.whisper)
            self._transcriber.warmup()
        except Exception as exc:
            logger.exception(f"Whisper failed to load: {exc}")
            self._set_state(State.ERROR)
            self.statusMessage.emit(f"Whisper failed to load: {exc}")
            return
        self._ollama.is_available(force=True)
        self._hotkey.start()
        self._set_state(State.IDLE)
        self.statusMessage.emit("Ready. Hold right Ctrl to dictate.")

    def stop(self) -> None:
        """Release the hotkey hook and the pooled HTTP connection."""
        self._hotkey.stop()
        self._ollama.close()

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, state: State) -> None:
        if state == self._state:
            return
        self._state = state
        logger.info(f"State: {state.value}")
        self.stateChanged.emit(state)

    # --- hotkey slots, on the GUI thread ---

    @Slot(bool)
    def _on_start(self, is_edit: bool) -> None:
        if self._state in (State.RECORDING, State.PROCESSING):
            logger.debug("Start ignored; already busy.")
            return
        if self._transcriber is None:
            self.statusMessage.emit("Whisper is not loaded.")
            return

        self._foreground = winctx.foreground()
        self._is_edit = is_edit
        self._selection = ""

        if not self._foreground.usable:
            # UIPI will silently swallow the paste, so say so rather than fail quietly.
            self._set_state(State.ERROR)
            self.statusMessage.emit(
                "The focused window is elevated; Windows blocks pasting into it."
            )
            self._hotkey.reset()
            return

        if is_edit:
            self._selection = output.copy_selection()
            if not self._selection:
                self._set_state(State.ERROR)
                self.statusMessage.emit("Nothing selected to edit.")
                self._hotkey.reset()
                return

        try:
            self._recorder.start()
        except Exception as exc:
            logger.exception(f"Could not start recording: {exc}")
            self._set_state(State.ERROR)
            self.statusMessage.emit(f"Microphone unavailable: {exc}")
            self._hotkey.reset()
            return

        self._set_state(State.RECORDING)
        target = self._foreground.exe or "the active window"
        self.statusMessage.emit(f"{'Editing' if is_edit else 'Recording'} for {target}...")

    @Slot()
    def _on_stop(self) -> None:
        if self._state != State.RECORDING:
            logger.debug("Stop ignored; not recording.")
            return

        try:
            recorded = self._recorder.stop()
        except Exception as exc:
            logger.exception(f"Could not stop recording: {exc}")
            self._set_state(State.ERROR)
            self.statusMessage.emit(f"Recording failed: {exc}")
            return

        if recorded is None or recorded.duration_s < self.cfg.audio.min_duration_s:
            # A stray tap on right Ctrl should not spend two seconds in Whisper.
            logger.info("Recording too short; discarding.")
            self._set_state(State.IDLE)
            self.statusMessage.emit("Too short. Hold right Ctrl while you speak.")
            return

        self._set_state(State.PROCESSING)
        self.statusMessage.emit("Transcribing...")
        threading.Thread(
            target=self._work, args=(recorded, self._is_edit, self._selection,
                                     self._foreground), daemon=True,
        ).start()

    # --- worker thread ---

    def _work(self, recorded: RecordedAudio, is_edit: bool, selection: str,
              fg: Optional[winctx.Foreground]) -> None:
        """Transcribe, then refine or edit. No Qt widget and no clipboard here."""
        payload: Optional[dict] = None
        try:
            t0 = time.perf_counter()
            raw = self._transcriber.transcribe(
                recorded.data, build_initial_prompt(self.cfg.vocabulary)
            )
            asr_ms = int((time.perf_counter() - t0) * 1000)

            if not raw:
                payload = {"error": "Nothing was transcribed. Check the microphone level."}
            elif is_edit:
                edited, llm_ms = self._ollama.edit_selection(selection, raw)
                if edited is None:
                    payload = {"error": "The edit failed; the selection was left alone."}
                else:
                    payload = {"text": edited, "raw": raw, "refined": edited,
                               "asr_ms": asr_ms, "llm_ms": llm_ms,
                               "audio_s": recorded.duration_s, "fg": fg}
            else:
                payload = self._refine(raw, asr_ms, recorded.duration_s, fg)
        except Exception as exc:
            logger.exception(f"Dictation pipeline failed: {exc}")
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._workerFinished.emit(payload)

    def _refine(self, raw: str, asr_ms: int, audio_s: float,
                fg: Optional[winctx.Foreground]) -> dict:
        """Snippet, then the LLM gate, then whatever survives."""
        base = {"raw": raw, "asr_ms": asr_ms, "audio_s": audio_s, "fg": fg}

        snippet = self.cfg.snippets.get(normalize_trigger(raw))
        if snippet:
            logger.info("Snippet matched; bypassing the LLM.")
            return {**base, "text": snippet, "refined": snippet, "llm_ms": None}

        if len(raw.split()) < self.cfg.llm_word_threshold:
            logger.info(
                f"Only {len(raw.split())} words; below the threshold of "
                f"{self.cfg.llm_word_threshold}, so skipping the LLM."
            )
            return {**base, "text": raw, "refined": None, "llm_ms": None}

        context = fg.context if (fg and self.cfg.context_formatting) else None
        refined, llm_ms = self._ollama.refine(raw, context)
        # refined is the raw transcript when the model was unavailable or its output was
        # rejected; llm_ms is None in exactly those cases, so it doubles as the flag.
        return {**base, "text": refined,
                "refined": refined if llm_ms is not None else None, "llm_ms": llm_ms}

    # --- back on the GUI thread ---

    @Slot(object)
    def _on_worker_finished(self, payload: Optional[dict]) -> None:
        if payload is None or "error" in payload:
            message = (payload or {}).get("error", "Dictation failed.")
            self._set_state(State.ERROR)
            self.statusMessage.emit(message)
            return

        text = payload["text"]
        fg: Optional[winctx.Foreground] = payload.get("fg")
        try:
            output.paste_text(text, settle_ms=self.cfg.paste_settle_ms)
        except Exception as exc:
            logger.exception(f"Pasting failed: {exc}")
            self._set_state(State.ERROR)
            self.statusMessage.emit(f"Could not paste: {exc}")
            return

        try:
            row_id = self.store.add(
                raw=payload["raw"], refined=payload.get("refined"),
                audio_s=payload["audio_s"],
                app_exe=fg.exe if fg else None, app_title=fg.title if fg else None,
                context=fg.context if fg else None,
                asr_ms=payload.get("asr_ms"), llm_ms=payload.get("llm_ms"),
            )
            self.dictationFinished.emit(row_id)
        except Exception as exc:
            # The text is already in the document; a failed log must not look like a
            # failed dictation.
            logger.exception(f"Could not write history: {exc}")

        self._set_state(State.IDLE)
        words = len(text.split())
        timing = f"{payload.get('asr_ms', 0)}ms ASR"
        if payload.get("llm_ms"):
            timing += f" + {payload['llm_ms']}ms LLM"
        self.statusMessage.emit(f"{words} words ({timing})")
