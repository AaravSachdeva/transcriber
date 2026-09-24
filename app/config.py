from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def app_dir() -> Path:
    """Per-user data directory. Settings, history database and logs live here."""
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "transcriber"


SETTINGS_PATH = app_dir() / "settings.json"
DB_PATH = app_dir() / "history.db"


class OutputMode(str, Enum):
    CLIPBOARD_AND_PASTE = "clipboard_and_paste"
    MARKDOWN_FILE = "markdown_file"


@dataclass
class WhisperConfig:
    # "distil-large-v3" is a short name faster-whisper resolves to
    # Systran/faster-distil-whisper-large-v3. English only.
    model_size: str = "distil-large-v3"
    device: str = "auto"  # "auto", "cpu", "cuda"
    # int8_float16 on a 4 GB card leaves room for the refinement model beside it.
    compute_type: str = "int8_float16"
    beam_size: int = 5


@dataclass
class OllamaConfig:
    # 127.0.0.1 rather than localhost, deliberately. On Windows "localhost" resolves to
    # ::1 first; Ollama binds IPv4 only, so every request paid an IPv6 connect timeout
    # of roughly 2.2 seconds before falling back. Measured: 2762ms versus 557ms.
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:1.7b"
    enabled: bool = True
    timeout_seconds: int = 60
    # Small context keeps the KV cache small, which is what actually costs VRAM here.
    num_ctx: int = 1024
    num_predict: int = 512
    # Keep the model resident between dictations so no reload lands in the latency path.
    keep_alive: str = "30m"


@dataclass
class HotkeyConfig:
    # Right Ctrl alone. Held longer than hold_threshold_ms is push-to-talk; a quicker
    # tap latches into toggle mode. Deliberately not suppressed - see pynput #679.
    hold_threshold_ms: int = 400
    # Right Ctrl + Shift dictates an edit instruction over the current selection.
    edit_modifier: str = "shift"


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_size: int = 1024
    device_index: Optional[int] = None  # None = system default
    # Below this, treat the recording as a stray key tap and skip transcription.
    min_duration_s: float = 0.35


@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output_mode: OutputMode = OutputMode.CLIPBOARD_AND_PASTE
    markdown_output_path: str = "notes.md"

    # Words fed to Whisper as initial_prompt so names and jargon come back spelled right.
    vocabulary: list[str] = field(default_factory=list)
    # Spoken trigger phrase -> canned text. Matched before the LLM, bypassing it.
    snippets: dict[str, str] = field(default_factory=dict)
    # Transcripts shorter than this skip the LLM pass entirely and paste immediately.
    llm_word_threshold: int = 6
    # Adapt tone and formatting to the foreground application.
    context_formatting: bool = True
    autostart: bool = False
    # Milliseconds to wait after synthesizing Ctrl+V before restoring the old clipboard.
    paste_settle_ms: int = 150

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic, so a crash mid-write cannot leave a truncated file
        logger.debug(f"Settings saved to {path}")


def _overlay(obj: Any, data: dict) -> None:
    """Apply a dict onto a dataclass instance in place, recursing into nested
    dataclasses. Unknown keys are ignored and absent keys keep their default, so an
    older settings.json still loads against newer code. Nesting is resolved from the
    live default instance rather than from field annotations, which are strings here
    because of `from __future__ import annotations`.
    """
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _overlay(current, value)
        elif isinstance(current, OutputMode):
            setattr(obj, key, OutputMode(value))
        else:
            setattr(obj, key, value)


def load(path: Path = SETTINGS_PATH) -> AppConfig:
    """Read settings, falling back to defaults on anything unreadable."""
    cfg = AppConfig()
    if not path.exists():
        logger.info(f"No settings file at {path}; using defaults.")
        return cfg
    try:
        _overlay(cfg, json.loads(path.read_text(encoding="utf-8")))
        logger.info(f"Settings loaded from {path}")
    except Exception as exc:
        logger.exception(f"Settings at {path} are unreadable ({exc}); using defaults.")
        return AppConfig()
    return cfg


DEFAULT_CONFIG = AppConfig()
