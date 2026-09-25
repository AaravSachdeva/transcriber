"""Settings screen.

Writes through to settings.json on Save. Model and device changes need a restart,
because both are fixed when the Whisper model is constructed; everything else the
controller reads live from the same config object.
"""

from __future__ import annotations

from typing import Callable, Optional

import requests
import sounddevice as sd
from loguru import logger
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from . import theme
from .. import autostart
from ..config import AppConfig, SETTINGS_PATH
from ..hotkey import MIN_HOLD_THRESHOLD_MS
from ..transcription import SAMPLE_RATE

WHISPER_MODELS = [
    "distil-large-v3", "large-v3-turbo", "large-v3", "medium", "small", "base",
]
COMPUTE_TYPES = ["int8_float16", "int8", "float16"]
DEVICES = ["auto", "cuda", "cpu"]


def input_devices() -> list[tuple[int, str]]:
    """(index, name) for every device that can record at our sample rate.

    PortAudio lists each microphone once per Windows host API. WASAPI and WDM-KS open
    only at the device's native rate and fail at 16 kHz with "Invalid sample rate"; MME
    and DirectSound resample, so those are the entries that survive.
    """
    found = []
    try:
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) < 1:
                continue
            try:
                sd.check_input_settings(device=index, channels=1, samplerate=SAMPLE_RATE)
            except Exception:
                continue
            found.append((index, device["name"]))
    except Exception as exc:
        logger.warning(f"Could not enumerate input devices: {exc}")
    return found


def ollama_models(base_url: str) -> list[str]:
    try:
        tags = requests.get(f"{base_url}/api/tags", timeout=3).json()
        return sorted(m["name"] for m in tags.get("models", []))
    except Exception:
        return []


class SettingsScreen(QWidget):
    def __init__(self, cfg: AppConfig, on_saved: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._on_saved = on_saved

        # One form for every section, so the label column lines up down the page.
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        def section(title: str) -> None:
            heading = theme.label(title, "SectionTitle")
            if form.rowCount():
                heading.setContentsMargins(0, 22, 0, 2)
            form.addRow(heading)

        def note(text: str) -> None:
            form.addRow(theme.label(text, "Caption", wrap=True))

        # --- audio ---
        section("Microphone")
        self._device = QComboBox()
        self._device.addItem("System default", None)
        for index, name in input_devices():
            self._device.addItem(f"{index}: {name}", index)
        # Device names run long; elide them rather than widen the page.
        self._device.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._device.setMinimumContentsLength(32)
        self._select_data(self._device, cfg.audio.device_index)
        form.addRow("Input device", self._device)
        form.addRow("Sample rate", QLabel(f"{SAMPLE_RATE:,} Hz (fixed, what Whisper expects)"))

        # --- whisper ---
        section("Transcription")
        self._model = QComboBox()
        self._model.addItems(WHISPER_MODELS)
        self._model.setCurrentText(cfg.whisper.model_size)
        self._compute = QComboBox()
        self._compute.addItems(COMPUTE_TYPES)
        self._compute.setCurrentText(cfg.whisper.compute_type)
        self._whisper_device = QComboBox()
        self._whisper_device.addItems(DEVICES)
        self._whisper_device.setCurrentText(cfg.whisper.device)
        self._beam = QSpinBox()
        self._beam.setRange(1, 10)
        self._beam.setValue(cfg.whisper.beam_size)
        form.addRow("Model", self._model)
        form.addRow("Compute type", self._compute)
        form.addRow("Device", self._whisper_device)
        form.addRow("Beam size", self._beam)
        note(
            "Changes here apply after a restart. On a 4 GB card, int8_float16 leaves "
            "room for the refinement model beside Whisper. Raise the beam size for "
            "accuracy, lower it for speed.")

        # --- refinement ---
        section("Refinement")
        self._llm_enabled = QCheckBox("Clean up transcripts with a local LLM")
        self._llm_enabled.setChecked(cfg.ollama.enabled)
        self._llm_model = QComboBox()
        self._llm_model.setEditable(True)
        available = ollama_models(cfg.ollama.base_url)
        self._llm_model.addItems(available or [cfg.ollama.model])
        self._llm_model.setCurrentText(cfg.ollama.model)
        self._threshold = QSpinBox()
        self._threshold.setRange(0, 100)
        self._threshold.setValue(cfg.llm_word_threshold)
        self._num_ctx = QSpinBox()
        self._num_ctx.setRange(256, 8192)
        self._num_ctx.setSingleStep(256)
        self._num_ctx.setValue(cfg.ollama.num_ctx)
        self._keep_alive = QLineEdit(cfg.ollama.keep_alive)
        self._context_formatting = QCheckBox("Adapt tone to the app being dictated into")
        self._context_formatting.setChecked(cfg.context_formatting)

        form.addRow("", self._llm_enabled)
        form.addRow("Model", self._llm_model)
        form.addRow("Skip below", self._threshold)
        form.addRow("Context size", self._num_ctx)
        form.addRow("Keep loaded for", self._keep_alive)
        form.addRow("", self._context_formatting)
        note(
            "Transcripts shorter than the skip threshold are pasted straight away, so "
            "short replies stay instant. A smaller context size uses less VRAM."
            + ("" if available else " Ollama is not reachable, so the model list is empty."))

        # --- behaviour ---
        section("Behaviour")
        self._hold = QSpinBox()
        self._hold.setRange(MIN_HOLD_THRESHOLD_MS, 2000)
        self._hold.setSingleStep(50)
        self._hold.setSuffix(" ms")
        self._hold.setValue(cfg.hotkey.hold_threshold_ms)
        self._settle = QSpinBox()
        self._settle.setRange(30, 1000)
        self._settle.setSingleStep(10)
        self._settle.setSuffix(" ms")
        self._settle.setValue(cfg.paste_settle_ms)
        self._autostart = QCheckBox("Start Transcriber when I log in")
        self._autostart.setChecked(autostart.is_enabled())

        form.addRow("Hotkey", QLabel("Right Ctrl. Hold to dictate, tap to latch."))
        form.addRow("Hold threshold", self._hold)
        form.addRow("Paste settle delay", self._settle)
        form.addRow("", self._autostart)
        note(
            "Held longer than the threshold is push-to-talk. A quicker tap latches "
            "recording on until you tap again. Raise the settle delay if a slow app "
            "ever pastes truncated text.")

        # Sections scroll; the title and the Save row stay put.
        sections = QWidget()
        sections_layout = QVBoxLayout(sections)
        sections_layout.setContentsMargins(32, 0, 32, 8)
        sections_layout.addLayout(form)
        sections_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(sections)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setAutoFillBackground(False)
        sections.setAutoFillBackground(False)

        save = QPushButton("Save")
        save.setDefault(True)
        save.clicked.connect(self._save)
        self._saved_note = theme.label(f"Stored in {SETTINGS_PATH}", "Caption")

        footer = QHBoxLayout()
        footer.setContentsMargins(32, 12, 32, 20)
        footer.setSpacing(12)
        footer.addWidget(save)
        footer.addWidget(self._saved_note, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 24, 0, 0)
        layout.setSpacing(16)
        title = theme.label("Settings", "PageTitle")
        title.setContentsMargins(32, 0, 32, 0)
        layout.addWidget(title)
        layout.addWidget(scroll, 1)
        layout.addLayout(footer)

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _save(self) -> None:
        cfg = self._cfg
        cfg.audio.device_index = self._device.currentData()
        cfg.whisper.model_size = self._model.currentText()
        cfg.whisper.compute_type = self._compute.currentText()
        cfg.whisper.device = self._whisper_device.currentText()
        cfg.whisper.beam_size = self._beam.value()

        cfg.ollama.enabled = self._llm_enabled.isChecked()
        cfg.ollama.model = self._llm_model.currentText().strip()
        cfg.ollama.num_ctx = self._num_ctx.value()
        cfg.ollama.keep_alive = self._keep_alive.text().strip() or "30m"
        cfg.llm_word_threshold = self._threshold.value()
        cfg.context_formatting = self._context_formatting.isChecked()

        cfg.hotkey.hold_threshold_ms = self._hold.value()
        cfg.paste_settle_ms = self._settle.value()

        wanted = self._autostart.isChecked()
        if wanted != autostart.is_enabled():
            autostart.set_enabled(wanted)
        cfg.autostart = autostart.is_enabled()
        self._autostart.setChecked(cfg.autostart)

        cfg.save()
        self._saved_note.setText("Saved. Model and device changes need a restart.")
        if self._on_saved:
            self._on_saved()
