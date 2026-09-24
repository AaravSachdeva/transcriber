from __future__ import annotations

from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from .config import WhisperConfig

SAMPLE_RATE = 16_000

# Whisper's prompt window is 224 tokens. Overflowing it silently truncates the prompt,
# and an over-long prompt also makes the model more likely to echo it into the output.
# Roughly 4 characters per token, kept well clear of the limit.
MAX_PROMPT_CHARS = 600


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
        text = " ".join(seg.text for seg in segments).strip()
        logger.debug(f"Transcription info: language={info.language}, duration={info.duration:.2f}s")
        return text
