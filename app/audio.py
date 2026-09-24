from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
from loguru import logger


@dataclass
class RecordedAudio:
    """Container for recorded audio in memory."""

    data: np.ndarray
    sample_rate: int
    duration_s: float
    rms: float
    peak: float


class AudioRecorder:
    """
    Simple, cross-platform microphone recorder using sounddevice.

    Designed to be controlled from a hotkey callback: `start()` when recording
    begins, `stop()` when it ends, then retrieve audio with `get_recording()`.
    """

    def __init__(self, sample_rate: int = 16_000, channels: int = 1, block_size: int = 1024,
                 device_index: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device_index = device_index

        self._frames: "deque[np.ndarray]" = deque()
        self._stream: Optional[sd.InputStream] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self.level = 0.0  # RMS of the latest block, for the overlay

    def _callback(self, indata, frames, _stream_time, status) -> None:  # type: ignore[override]
        if status:
            logger.warning(f"Audio stream status: {status}")
        # Keep the callback lightweight and non-blocking.
        block = indata.copy()
        self.level = float(np.sqrt(np.mean(np.square(block))))
        with self._lock:
            self._frames.append(block)

    def start(self) -> None:
        if self._stream is not None:
            logger.debug("AudioRecorder.start() called, but stream already running.")
            return

        logger.info("Starting audio recording.")
        with self._lock:
            self._frames.clear()
        self._started_at = time.monotonic()

        # Validate settings early to fail fast with a useful error. A saved device that
        # cannot record at our rate, or an index PortAudio has since renumbered (it does
        # whenever a Bluetooth headset connects), falls back to the system default.
        # ponytail: devices saved by index; save by name if the wrong mic gets picked.
        try:
            sd.check_input_settings(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
            )
        except (sd.PortAudioError, ValueError) as exc:
            if self.device_index is None:
                raise
            logger.warning(
                f"Input device {self.device_index} is unusable ({exc}); "
                f"using the system default."
            )
            self.device_index = None
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            device=self.device_index,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> Optional[RecordedAudio]:
        if self._stream is None:
            logger.debug("AudioRecorder.stop() called, but stream not running.")
            return None

        logger.info("Stopping audio recording.")
        started_at = self._started_at
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._started_at = None
        self.level = 0.0

        with self._lock:
            if len(self._frames) == 0:
                logger.warning("No audio frames captured.")
                return None
            frames: List[np.ndarray] = list(self._frames)
            self._frames.clear()

        data = np.concatenate(frames, axis=0)
        # Ensure mono 1D float32 in [-1, 1] for Whisper.
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            if data.shape[1] == 1:
                data = data[:, 0]
            else:
                data = np.mean(data, axis=1)
        data = np.clip(data, -1.0, 1.0)

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        duration_s = float(data.size / self.sample_rate) if self.sample_rate else 0.0
        wall_s = float(time.monotonic() - started_at) if started_at is not None else duration_s

        logger.info(
            f"Recorded audio: samples={data.size}, sr={self.sample_rate}, "
            f"duration~{duration_s:.2f}s (wall {wall_s:.2f}s), rms={rms:.4f}, peak={peak:.4f}"
        )
        if peak < 0.01:
            logger.warning(
                "Audio level is extremely low (near silence). "
                "If you spoke clearly, check mic mute/privacy permissions or input device selection."
            )

        return RecordedAudio(
            data=data,
            sample_rate=self.sample_rate,
            duration_s=duration_s,
            rms=rms,
            peak=peak,
        )

