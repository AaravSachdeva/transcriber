"""Vocabulary and snippets.

Vocabulary terms are fed to Whisper as initial_prompt, which biases decoding towards
those spellings. Snippets map a spoken trigger phrase to canned text; a match bypasses
the LLM entirely and pastes the canned text as-is.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QMessageBox, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import theme
from ..config import AppConfig
from ..controller import normalize_trigger
from ..transcription import MAX_PROMPT_CHARS, build_initial_prompt


class VocabularyScreen(QWidget):
    def __init__(self, cfg: AppConfig, on_saved: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._on_saved = on_saved

        # --- vocabulary ---
        self._vocab = QPlainTextEdit("\n".join(cfg.vocabulary))
        self._vocab.setPlaceholderText(
            "One term per line. Names, product names, jargon, acronyms."
        )
        self._vocab.setMinimumHeight(120)
        self._vocab.textChanged.connect(self._update_budget)
        self._budget = theme.label("", "Caption", wrap=True)

        # --- snippets ---
        self._snippets = QTableWidget(0, 2)
        self._snippets.setHorizontalHeaderLabels(["Say this", "Paste this"])
        self._snippets.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._snippets.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._snippets.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._snippets.verticalHeader().setVisible(False)
        self._snippets.verticalHeader().setDefaultSectionSize(36)
        self._snippets.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._snippets.setShowGrid(False)
        self._snippets.setMinimumHeight(140)
        for trigger, text in sorted(cfg.snippets.items()):
            self._add_row(trigger, text)

        add = QPushButton("Add snippet")
        add.clicked.connect(lambda: self._add_row("", ""))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)

        snippet_buttons = QHBoxLayout()
        snippet_buttons.addWidget(add)
        snippet_buttons.addWidget(remove)
        snippet_buttons.addStretch(1)

        save = QPushButton("Save")
        save.setDefault(True)
        save.clicked.connect(self._save)
        self._saved_note = theme.label("", "Caption")

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addWidget(save)
        buttons.addWidget(self._saved_note, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 20)
        layout.setSpacing(8)
        layout.addWidget(theme.label("Vocabulary", "PageTitle"))
        layout.addSpacing(8)
        layout.addWidget(theme.label(
            "Names, jargon and product terms Whisper keeps getting wrong. They bias "
            "its spelling towards yours.", wrap=True))
        layout.addWidget(self._vocab)
        layout.addWidget(self._budget)
        layout.addSpacing(20)
        layout.addWidget(theme.label("Snippets", "SectionTitle"))
        layout.addWidget(theme.label(
            "Say a trigger phrase, get the text pasted exactly. Triggers match the whole "
            "transcript, ignoring case and trailing punctuation, and skip the LLM.",
            wrap=True))
        layout.addWidget(self._snippets, 1)
        layout.addLayout(snippet_buttons)
        layout.addSpacing(12)
        layout.addLayout(buttons)

        self._update_budget()

    def _add_row(self, trigger: str, text: str) -> None:
        row = self._snippets.rowCount()
        self._snippets.insertRow(row)
        self._snippets.setItem(row, 0, QTableWidgetItem(trigger))
        self._snippets.setItem(row, 1, QTableWidgetItem(text))

    def _remove_selected(self) -> None:
        for index in sorted(self._snippets.selectionModel().selectedRows(),
                            key=lambda i: i.row(), reverse=True):
            self._snippets.removeRow(index.row())

    def _terms(self) -> list[str]:
        return [line.strip() for line in self._vocab.toPlainText().splitlines() if line.strip()]

    def _update_budget(self) -> None:
        """Whisper's prompt window is 224 tokens, so a long list gets truncated. Say so
        before it silently drops terms."""
        terms = self._terms()
        prompt = build_initial_prompt(terms) or ""
        used = len(", ".join(terms))
        if used > MAX_PROMPT_CHARS:
            kept = len(prompt.split(", ")) if prompt else 0
            self._budget.setText(
                f"{len(terms)} terms, {used} characters. Whisper's prompt window fits "
                f"about {MAX_PROMPT_CHARS}, so only the first {kept} will be used."
            )
        else:
            self._budget.setText(
                f"{len(terms)} terms, {used} of about {MAX_PROMPT_CHARS} characters used."
            )

    def _save(self) -> None:
        snippets: dict[str, str] = {}
        for row in range(self._snippets.rowCount()):
            trigger_item = self._snippets.item(row, 0)
            text_item = self._snippets.item(row, 1)
            trigger = normalize_trigger(trigger_item.text() if trigger_item else "")
            text = (text_item.text() if text_item else "").strip()
            if not trigger or not text:
                continue
            if trigger in snippets:
                QMessageBox.warning(
                    self, "Duplicate trigger",
                    f"More than one snippet is triggered by {trigger!r}. "
                    "Only the last one will be kept.",
                )
            snippets[trigger] = text

        self._cfg.vocabulary = self._terms()
        self._cfg.snippets = snippets
        self._cfg.save()
        self._saved_note.setText(
            f"Saved {len(self._cfg.vocabulary)} terms and {len(snippets)} snippets."
        )
        if self._on_saved:
            self._on_saved()
