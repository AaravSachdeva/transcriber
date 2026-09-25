"""Dictation history and the stats derived from it.

sqlite3 from the stdlib. Connections are per-thread: the Qt GUI thread reads for the
History and Stats screens while a worker thread writes a finished dictation, and a
sqlite3 connection must not be shared across threads.
"""

from __future__ import annotations

import sqlite3
import threading
import wave
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS dictations (
  id          INTEGER PRIMARY KEY,
  ts          TEXT    NOT NULL,
  app_exe     TEXT,
  app_title   TEXT,
  context     TEXT,
  raw         TEXT    NOT NULL,
  refined     TEXT,
  audio_s     REAL    NOT NULL,
  words       INTEGER NOT NULL,
  llm_ms      INTEGER,
  asr_ms      INTEGER
);
CREATE INDEX IF NOT EXISTS dictations_ts ON dictations(ts);
"""

# Columns added after the first release. Added on open when missing, so an existing
# history.db gains them without losing rows.
# final:        what the text ended up as after the user saw it
# final_source: 'explicit' (corrected in History), 'implicit' (read back from the text
#               box), 'unedited' (read back, not changed), or NULL (not known)
# is_edit:      a selection edit, whose raw is a spoken instruction, not a transcript
ADDED_COLUMNS = ("final TEXT", "final_source TEXT", "is_edit INTEGER NOT NULL DEFAULT 0")

# How much a final text counts as a correction. A deliberate fix is the clean signal.
# Unedited is weak approval only: people leave small errors alone.
SIGNAL_WEIGHT = {"explicit": 1.0, "implicit": 0.5, "unedited": 0.1}

# Typing speed assumed when estimating time saved. Deliberately conservative: a higher
# baseline would inflate the number.
TYPING_WPM = 40.0


@dataclass
class Stats:
    count: int
    words: int
    speaking_wpm: float   # how fast you talk
    seconds_saved: float  # versus typing the same words at TYPING_WPM
    llm_share: float      # fraction of dictations that went through the LLM


def wpm(words: int, audio_s: float) -> float:
    """Speaking rate. Zero-length audio yields 0 rather than dividing by zero."""
    if audio_s <= 0:
        return 0.0
    return words / (audio_s / 60.0)


def seconds_saved(words: int, audio_s: float) -> float:
    """Time saved against typing the same text. Clamped at zero, because a slow
    dictation genuinely saves nothing and should not report a negative."""
    return max(0.0, (words / TYPING_WPM) * 60.0 - audio_s)


class Store:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            have = {r["name"] for r in conn.execute("PRAGMA table_info(dictations)")}
            for column in ADDED_COLUMNS:
                if column.split()[0] not in have:
                    conn.execute(f"ALTER TABLE dictations ADD COLUMN {column}")
        logger.info(f"History database at {self._path}")

    def _conn(self) -> sqlite3.Connection:
        """One connection per thread, created on first use by that thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def add(
        self,
        raw: str,
        refined: Optional[str],
        audio_s: float,
        app_exe: Optional[str],
        app_title: Optional[str],
        context: Optional[str],
        asr_ms: Optional[int] = None,
        llm_ms: Optional[int] = None,
        is_edit: bool = False,
    ) -> int:
        text = refined or raw
        conn = self._conn()
        with conn:
            cur = conn.execute(
                "INSERT INTO dictations "
                "(ts, app_exe, app_title, context, raw, refined, audio_s, words, "
                " asr_ms, llm_ms, is_edit) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    app_exe, app_title, context, raw, refined, audio_s,
                    len(text.split()), asr_ms, llm_ms, int(is_edit),
                ),
            )
        return int(cur.lastrowid)

    def set_final(self, row_id: int, text: str, source: str) -> None:
        """Record what a dictation ended up as. Called from the capture thread as well
        as the GUI. Read-back never overwrites a correction the user made by hand."""
        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE dictations SET final = ?, final_source = ? WHERE id = ? "
                "AND (? = 'explicit' OR final_source IS NOT 'explicit')",
                (text, source, row_id, source),
            )

    def corrections(self) -> list[dict]:
        """Every dictation whose final text is known, oldest first, with its weight."""
        rows = self._conn().execute(
            "SELECT * FROM dictations WHERE final_source IS NOT NULL ORDER BY ts"
        ).fetchall()
        return [{**dict(r), "weight": SIGNAL_WEIGHT[r["final_source"]]} for r in rows]

    def audio_path(self, row_id: int) -> Path:
        return self._path.parent / "audio" / f"{row_id}.wav"

    def save_audio(self, row_id: int, data: np.ndarray, sample_rate: int) -> None:
        """Keep a dictation's recording as 16-bit mono WAV, named by its history id."""
        path = self.audio_path(row_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes((np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes())

    def recent(self, limit: int = 500) -> list[sqlite3.Row]:
        return self._conn().execute(
            "SELECT * FROM dictations ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()

    def stats(self) -> Stats:
        row = self._conn().execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(words), 0) AS words, "
            "COALESCE(SUM(audio_s), 0) AS audio_s, "
            "COALESCE(SUM(refined IS NOT NULL), 0) AS refined_n FROM dictations"
        ).fetchone()
        n, words, audio_s = row["n"], row["words"], row["audio_s"]
        return Stats(
            count=n,
            words=words,
            speaking_wpm=wpm(words, audio_s),
            seconds_saved=seconds_saved(words, audio_s),
            llm_share=(row["refined_n"] / n) if n else 0.0,
        )

    def daily_words(self, days: int = 30) -> list[tuple[str, int]]:
        """(local date, words) for each of the last `days` calendar days, oldest first.
        Quiet days are included as zero, so a chart's x-axis is real time."""
        rows = self._conn().execute(
            "SELECT date(ts, 'localtime') AS day, SUM(words) AS words FROM dictations "
            "GROUP BY day"
        ).fetchall()
        by_day = {r["day"]: r["words"] for r in rows}
        today = date.today()
        return [(d, by_day.get(d, 0)) for d in
                (str(today - timedelta(days=n)) for n in range(days - 1, -1, -1))]

    def delete(self, row_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM dictations WHERE id = ?", (row_id,))
        self.audio_path(row_id).unlink(missing_ok=True)

    def close(self) -> None:
        """Close this thread's connection. Other threads keep theirs; they are closed
        when their thread ends. Windows will not delete an open database file, so
        anything that removes the file must call this first."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
