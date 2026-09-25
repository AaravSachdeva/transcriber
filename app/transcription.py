from __future__ import annotations

import queue
import re
import threading
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
from loguru import logger

from .config import WhisperConfig

SAMPLE_RATE = 16_000
VAD_WINDOW = 512   # samples Silero scores at once at 16 kHz (32 ms)
VAD_CONTEXT = 64   # tail of the previous window Silero sees alongside each one

# Whisper's prompt window is 224 tokens. Overflowing it silently truncates the prompt,
# and an over-long prompt also makes the model more likely to echo it into the output.
# Roughly 4 characters per token, kept well clear of the limit.
MAX_PROMPT_CHARS = 600

# Whole segments Whisper invents on silence. Seen alone ("you") and after a real
# sentence ("...will that save time  you").
HALLUCINATIONS = {"you"}

# A phrase of one to four words said twice or more in a row: "such as such as".
# ponytail: also collapses the rare legitimate repeat ("had had", "very very"); add an
# allowlist if that bites.
_REPEAT = re.compile(r"\b(\w+(?:\s+\w+){0,3})(?:[\s,]+\1\b)+", re.I)
_FILLER = re.compile(r"\b(?:um+|uh+|erm)\b[,.]?\s*", re.I)


def tidy(text: str) -> str:
    """Remove fillers and stutters without the LLM, so they go even when it is skipped,
    unavailable, or rejected."""
    return _REPEAT.sub(r"\1", _FILLER.sub("", text)).strip()


def build_initial_prompt(vocabulary: list[str]) -> Optional[str]:
    """Turn the user's vocabulary into an initial_prompt.

    initial_prompt biases decoding towards these spellings, which is how names, jargon
    and product terms come back correct. Truncated rather than dropped on overflow, so a
    long list degrades instead of failing.
    """
    terms = [t.strip() for t in vocabulary if t.strip()]
    if not terms:
        return None
    prompt = ", ".join(terms)
    if len(prompt) > MAX_PROMPT_CHARS:
        kept: list[str] = []
        used = 0
        for term in terms:
            if used + len(term) + 2 > MAX_PROMPT_CHARS:
                break
            kept.append(term)
            used += len(term) + 2
        logger.warning(
            f"Vocabulary is too long for Whisper's prompt window; using "
            f"{len(kept)} of {len(terms)} terms."
        )
        prompt = ", ".join(kept)
    return prompt


class TranscriptionService:
    """Thin wrapper around Faster-Whisper for local transcription."""

    def __init__(self, cfg: WhisperConfig) -> None:
        self._cfg = cfg
        self._model_size = cfg.model_size

        device = self._select_device(cfg.device)
        compute_type = self._select_compute_type(cfg.compute_type, device)
        logger.info(
            f"Initializing Faster-Whisper model {cfg.model_size!r} "
            f"on device={device!r}, compute_type={compute_type!r}"
        )
        self._device = device
        self._compute_type = compute_type
        self._model = WhisperModel(self._model_size, device=device, compute_type=compute_type)

    @property
    def device(self) -> str:
        return self._device

    @staticmethod
    def _select_device(requested: str) -> str:
        requested = (requested or "auto").strip().lower()
        if requested in {"cpu", "cuda"}:
            return requested
        try:
            import ctranslate2  # noqa: PLC0415 - probed lazily, only when auto-detecting

            count = int(ctranslate2.get_cuda_device_count())
            if count > 0:
                logger.info(f"CUDA detected ({count} device(s)); using GPU for Whisper.")
                return "cuda"
        except Exception as exc:
            logger.info(f"CUDA not available for Whisper (falling back to CPU). Details: {exc}")
        return "cpu"

    @staticmethod
    def _select_compute_type(requested: str, device: str) -> str:
        requested = (requested or "auto").strip().lower()
        if requested != "auto":
            return requested
        return "int8_float16" if device == "cuda" else "int8"

    def _reinit_model(self, device: str, compute_type: str) -> None:
        logger.warning(
            f"Reinitializing Whisper model {self._model_size!r} "
            f"on device={device!r}, compute_type={compute_type!r}"
        )
        self._model = WhisperModel(self._model_size, device=device, compute_type=compute_type)
        self._device = device
        self._compute_type = compute_type

    @staticmethod
    def _looks_like_cuda_runtime_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        needles = ["cublas", "cudart", "cudnn", "cuda", "dll is not found", "cannot be loaded"]
        return any(n in msg for n in needles)

    def warmup(self) -> None:
        """Run half a second of silence through the model at startup.

        Without this, the first real dictation pays model load and CUDA kernel
        compilation on top of its own transcription, which is the difference between
        feeling instant and feeling broken.
        """
        try:
            list(self._model.transcribe(
                np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                language="en", beam_size=1,
            )[0])
            logger.info(f"Whisper warmed up on {self._device}.")
        except Exception as exc:
            logger.warning(f"Whisper warmup failed (not fatal): {exc}")

    def transcribe(self, audio: np.ndarray, initial_prompt: Optional[str] = None) -> Optional[str]:
        """Transcribe a mono 16 kHz float32 buffer in [-1, 1]."""
        if audio is None or audio.size == 0:
            logger.warning("Empty audio buffer passed to transcribe().")
            return None

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:  # defensive: callers should hand over mono already
            audio = audio[:, 0] if audio.shape[1] == 1 else np.mean(audio, axis=1)
        audio = np.clip(audio, -1.0, 1.0)

        logger.info("Running Whisper transcription.")
        try:
            text = self._run(audio, initial_prompt)
        except RuntimeError as exc:
            # Common on Windows when a CUDA device exists but its runtime DLLs do not.
            if self._device == "cuda" and self._looks_like_cuda_runtime_error(exc):
                logger.exception(
                    "Whisper CUDA execution failed (likely missing CUDA runtime DLLs). "
                    "Falling back to CPU and retrying once."
                )
                self._reinit_model(
                    device="cpu",
                    compute_type=self._select_compute_type(self._cfg.compute_type, "cpu"),
                )
                text = self._run(audio, initial_prompt)
            else:
                raise

        if not text:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            logger.warning(f"Whisper produced empty text. audio_rms={rms:.4f}, audio_peak={peak:.4f}")
        return text

    def _run(self, audio: np.ndarray, initial_prompt: Optional[str]) -> str:
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self._cfg.beam_size,
            # distil-large-v3's own documented setting. Left at its True default, one
            # dictation's text conditions the next one and bleeds across utterances.
            condition_on_previous_text=False,
            vad_filter=True,  # Silero VAD via onnxruntime; trims silence before decoding
            without_timestamps=True,  # nothing here needs word timings
            initial_prompt=initial_prompt,
        )
        # Materialise the lazy generator here so CUDA errors raised during iteration are
        # caught by the caller's fallback rather than escaping later.
        text = " ".join(t for seg in segments
                        if (t := seg.text.strip()) and t.lower().strip(".!") not in HALLUCINATIONS)
        logger.debug(f"Transcription info: language={info.language}, duration={info.duration:.2f}s")
        return text


class LiveTranscript:
    """Transcribes a dictation segment by segment while the user is still talking.

    The recorder pushes blocks from its PortAudio thread. A worker thread scores them with
    the Silero VAD that faster-whisper already bundles and, at each natural pause, sends
    the finished segment to Whisper. By the time the key comes up only the tail is left.
    """

    def __init__(self, transcriber: TranscriptionService, prompt: Optional[str],
                 pause_ms: int, threshold: float = 0.5) -> None:
        self._transcriber = transcriber
        self._prompt = prompt
        self._pause = pause_ms * SAMPLE_RATE // 1000
        self._threshold = threshold
        self._session = get_vad_model().session
        # Silero's recurrent state, carried across calls so it hears one continuous stream.
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT, dtype=np.float32)
        self._audio = np.zeros(0, dtype=np.float32)  # current segment, not yet sent
        self._scored = 0      # samples of _audio the VAD has seen
        self._quiet = 0       # samples of silence since the last speech
        self._heard = False   # speech in the current segment
        self._texts: list[str] = []
        self._error: Optional[Exception] = None
        self._blocks: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, block: np.ndarray) -> None:
        """Called on the PortAudio thread, so it only enqueues."""
        self._blocks.put(block)

    def close(self) -> None:
        """No more audio is coming."""
        self._blocks.put(None)

    def finish(self) -> str:
        """Wait for the segments already cut, transcribe the tail, return the whole text."""
        self.close()
        self._thread.join()
        if self._error is not None:
            raise self._error
        self._send(self._audio)
        return " ".join(t for t in self._texts if t)

    def _run(self) -> None:
        try:
            while (block := self._blocks.get()) is not None:
                for segment in self._cut(block.reshape(len(block), -1).mean(axis=1)):
                    self._send(segment)
        except Exception as exc:  # re-raised by finish(), so no audio is lost silently
            logger.exception(f"Live transcription failed: {exc}")
            self._error = exc

    def _send(self, segment: np.ndarray) -> None:
        if segment.size:
            self._texts.append(self._transcriber.transcribe(segment, self._prompt) or "")

    def _cut(self, audio: np.ndarray) -> list[np.ndarray]:
        """Append audio; return every segment that has ended in a pause of pause_ms.

        A segment keeps its trailing pause and any leading silence; vad_filter trims both.
        """
        self._audio = np.concatenate([self._audio, audio])
        n = (self._audio.size - self._scored) // VAD_WINDOW
        if n == 0:
            return []
        windows = self._audio[self._scored:self._scored + n * VAD_WINDOW].reshape(n, VAD_WINDOW)
        segments = []
        for prob in self._score(windows):
            self._scored += VAD_WINDOW
            if prob >= self._threshold:
                self._heard, self._quiet = True, 0
            elif self._heard:
                self._quiet += VAD_WINDOW
                if self._quiet >= self._pause:
                    logger.info(f"Pause detected; sending {self._scored / SAMPLE_RATE:.1f}s to Whisper.")
                    segments.append(self._audio[:self._scored])
                    self._audio = self._audio[self._scored:]
                    self._scored = self._quiet = 0
                    self._heard = False
        return segments

    def _score(self, windows: np.ndarray) -> np.ndarray:
        """Speech probability per window. Same row layout as faster-whisper's own
        SileroVADModel.__call__, which resets the state on every call and so cannot stream."""
        context = np.vstack([self._context, windows[:-1, -VAD_CONTEXT:]])
        self._context = windows[-1, -VAD_CONTEXT:].copy()
        probs, self._h, self._c = self._session.run(
            None, {"input": np.hstack([context, windows]), "h": self._h, "c": self._c},
        )
        return probs
