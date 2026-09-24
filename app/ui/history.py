"""History screen: every past dictation, searchable, with the raw transcript beside
what was actually pasted.

A plain QAbstractTableModel over rows already fetched from SQLite, with a
QSortFilterProxyModel for the search box. The history is bounded at a few hundred rows,
so filtering in memory is simpler and faster than re-querying on every keystroke.
"""

from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from html import escape
from typing import Any, Optional

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QSize, QSortFilterProxyModel, Qt, Signal,
)
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLineEdit, QPushButton, QSplitter,
    QStyledItemDelegate, QTextBrowser, QTreeView, QVBoxLayout, QWidget,
)

from . import theme
from ..store import Store

COLUMNS = ["When", "App", "Words", "Text"]

EMPTY_HINT = "Nothing dictated yet. Hold right Ctrl in any app and speak."
SELECT_HINT = "Select a dictation to see what was pasted, and what the LLM changed."


def _word_key(word: str) -> str:
    """Case and punctuation are the LLM's routine fixes; matching on them would mark
    nearly every word as changed."""
    return word.strip(".,!?;:\"'()").lower()


def _diff_html(raw: str, refined: str, dim: str, accent: str) -> str:
    """What Whisper heard against what was pasted: removed words struck through in
    `dim`, added words in `accent`, unchanged words as pasted."""
    a, b = raw.split(), refined.split()
    matcher = SequenceMatcher(a=[_word_key(w) for w in a], b=[_word_key(w) for w in b],
                              autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out.append(escape(" ".join(b[j1:j2])))
            continue
        if i2 > i1:
            out.append(f'<s style="color:{dim}">{escape(" ".join(a[i1:i2]))}</s>')
        if j2 > j1:
            out.append(f'<span style="color:{accent}">{escape(" ".join(b[j1:j2]))}</span>')
    return " ".join(out)


class HistoryModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict] = []

    def set_rows(self, rows: list) -> None:
        self.beginResetModel()
        self._rows = [dict(r) for r in rows]
        self.endResetModel()

    def row_at(self, source_row: int) -> Optional[dict]:
        if 0 <= source_row < len(self._rows):
            return self._rows[source_row]
        return None

    def rowCount(self, _parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, _parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section: int, orientation, role=Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role == Qt.DisplayRole:
            column = index.column()
            if column == 0:
                return _when(row["ts"])
            if column == 1:
                return row["app_exe"] or "-"
            if column == 2:
                return row["words"]
            if column == 3:
                text = row["refined"] or row["raw"]
                return text.replace("\n", " ")
        if role == Qt.UserRole:  # sort key: the ISO timestamp sorts; "24 Sep" does not
            return row["ts"] if index.column() == 0 else self.data(index)
        if role == Qt.TextAlignmentRole and index.column() == 2:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class _TallRows(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), 34))
        return size


def _when(iso: str) -> str:
    """ISO 8601 UTC from the database, rendered in local time."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %H:%M")
    except Exception:
        return iso


class HistoryScreen(QWidget):
    countChanged = Signal(int)

    def __init__(self, store: Store) -> None:
        super().__init__()
        self._store = store
        self._model = HistoryModel()

        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setFilterKeyColumn(-1)  # search every column
        self._proxy.setSortRole(Qt.UserRole)

        title = theme.label("History", "PageTitle")

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search transcripts, or the app they went to")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._proxy.setFilterFixedString)

        # A tree view without branches, not a QTableView: the windows11 style highlights
        # a selected tree row as one row, but a table row as separate cells.
        self._table = QTreeView()
        self._table.setModel(self._proxy)
        self._table.setItemDelegate(_TallRows(self._table))
        self._table.setRootIsDecorated(False)
        self._table.setUniformRowHeights(True)
        self._table.setAllColumnsShowFocus(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(0, Qt.DescendingOrder)
        self._table.setWordWrap(False)
        header = self._table.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.selectionModel().selectionChanged.connect(self._show_detail)

        self._detail = QTextBrowser()

        self._copy = QPushButton("Copy pasted text")
        self._copy.setDefault(True)
        self._copy.setEnabled(False)
        self._copy.clicked.connect(self._copy_selected)

        self._delete = QPushButton("Delete")
        self._delete.setEnabled(False)
        self._delete.clicked.connect(self._delete_selected)

        detail_buttons = QHBoxLayout()
        detail_buttons.addWidget(self._copy)
        detail_buttons.addWidget(self._delete)
        detail_buttons.addStretch(1)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 8, 0, 0)
        detail_layout.addWidget(self._detail, 1)
        detail_layout.addLayout(detail_buttons)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._table)
        splitter.addWidget(detail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(self._search)
        layout.addWidget(splitter, 1)

        self.reload()

    def reload(self) -> None:
        rows = self._store.recent()
        self._model.set_rows(rows)
        self._detail.setPlaceholderText(SELECT_HINT if rows else EMPTY_HINT)
        self.countChanged.emit(len(rows))

    def _selected_row(self) -> Optional[dict]:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        return self._model.row_at(self._proxy.mapToSource(indexes[0]).row())

    def _show_detail(self) -> None:
        row = self._selected_row()
        self._copy.setEnabled(row is not None)
        self._delete.setEnabled(row is not None)
        if row is None:
            self._detail.clear()
            return

        palette = self.palette()
        # HexArgb: dark mode's placeholder colour is translucent white, and .name()
        # would drop the alpha and return plain white.
        dim = palette.color(QPalette.PlaceholderText).name(QColor.HexArgb)
        accent = palette.color(QPalette.Accent).name()

        meta = [_when(row["ts"]), row["app_exe"] or "unknown app"]
        if row["context"]:
            meta.append(row["context"])
        meta.append(f"{row['audio_s']:.1f}s of audio")
        if row["asr_ms"]:
            meta.append(f"{row['asr_ms']} ms Whisper")
        meta.append(f"{row['llm_ms']} ms LLM" if row["llm_ms"] else "LLM skipped")

        pasted = row["refined"] or row["raw"]
        html = [
            f'<p style="color:{dim}">{escape(", ".join(meta))}</p>',
            f'<p style="white-space:pre-wrap">{escape(pasted)}</p>',
        ]
        if row["refined"] and row["refined"] != row["raw"]:
            html += [
                '<p style="font-weight:600; margin-top:14px">What the LLM changed</p>',
                f"<p>{_diff_html(row['raw'], row['refined'], dim, accent)}</p>",
            ]
        if row["app_title"]:
            html.append(f'<p style="color:{dim}; margin-top:14px">'
                        f'Window: {escape(row["app_title"])}</p>')
        self._detail.setHtml("".join(html))

    def _copy_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        QGuiApplication.clipboard().setText(row["refined"] or row["raw"])

    def _delete_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self._store.delete(int(row["id"]))
        self.reload()
        self._detail.clear()
