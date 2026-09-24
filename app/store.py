"""Dictation history and the stats derived from it.

sqlite3 from the stdlib. Connections are per-thread: the Qt GUI thread reads for the
History and Stats screens while a worker thread writes a finished dictation, and a
sqlite3 connection must not be shared across threads.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

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
    ) -> int:
        text = refined or raw
        conn = self._conn()
        with conn:
            cur = conn.execute(
                "INSERT INTO dictations "
                "(ts, app_exe, app_title, context, raw, refined, audio_s, words, "
                " asr_ms, llm_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    app_exe, app_title, context, raw, refined, audio_s,
                    len(text.split()), asr_ms, llm_ms,
                ),
            )
        return int(cur.lastrowid)

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

    def close(self) -> None:
        """Close this thread's connection. Other threads keep theirs; they are closed
        when their thread ends. Windows will not delete an open database file, so
        anything that removes the file must call this first."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
