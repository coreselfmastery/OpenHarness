"""Shared SQLite helpers for the video learning stores.

The learning stores (semantic memory, style memory, failure patterns,
prompt tracker) are all small single-file SQLite databases under
``<data_dir>/video/``. They share the same operational contract:

- WAL journal + NORMAL sync (fast, multi-process safe enough),
- ``check_same_thread=False`` with an RLock around every access,
- graceful degradation: a broken/unwritable DB disables the store
  (methods return empty/zero results instead of raising).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def video_store_dir() -> Path:
    """Directory holding the video learning-store databases."""
    from openharness.config.paths import get_data_dir

    directory = get_data_dir() / "video"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class SQLiteStore:
    """Minimal robust base for the learning stores."""

    def __init__(self, path: Path, schema: str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._broken = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._path), check_same_thread=False, timeout=5.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(schema)
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.warning("%s unavailable (%s): store disabled", type(self).__name__, exc)
            self._conn = None
            self._broken = True

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._conn is not None

    @property
    def path(self) -> Path:
        return self._path

    def execute(
        self, sql: str, params: tuple | dict[str, Any] = ()
    ) -> sqlite3.Cursor | None:
        """Run one statement and commit. Returns None when disabled."""
        if self._conn is None:
            return None
        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
                self._conn.commit()
                return cursor
            except sqlite3.Error:
                log.debug("sqlite execute failed: %s", sql, exc_info=True)
                return None

    def query(self, sql: str, params: tuple | dict[str, Any] = ()) -> list[sqlite3.Row]:
        """Run a SELECT and return all rows ([] when disabled/broken)."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                return list(self._conn.execute(sql, params).fetchall())
            except sqlite3.Error:
                log.debug("sqlite query failed: %s", sql, exc_info=True)
                return []

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    @staticmethod
    def now() -> float:
        return time.time()
