"""Cross-session persistent store backed by SQLite (#19).

A single small SQLite database (``~/.openharness/cache/store.sqlite3``)
shared by:

- the render cache (#14 / #17) — scene fingerprints -> rendered outputs,
- the semantic LLM cache (#18) — MinHash signatures -> responses,
- the QA failure predictor (#10) — failure-category counters.

Everything is stdlib (``sqlite3``), synchronous, and guarded by a process
wide ``threading.RLock`` because SQLite connections are not safe to share
across threads without care. Callers run on the event loop; the queries
here are tiny (index lookups on a local file) so the loop is not blocked
measurably. Writes go through WAL journaling so concurrent readers in
other agent subprocesses never see partial rows.

The store degrades gracefully: if the database cannot be opened (read-only
filesystem, corrupt file), every accessor turns into a no-op and cache
features report misses instead of crashing the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS counters (
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    n         INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS kv_updated ON kv (namespace, updated_at);
"""

DEFAULT_DB_SUBPATH = Path("cache") / "store.sqlite3"


def default_store_path() -> Path:
    """Return the default on-disk location of the persistent store."""
    from openharness.config.paths import get_data_dir

    return get_data_dir() / DEFAULT_DB_SUBPATH


class PersistentStore:
    """Small SQLite KV + counter store with graceful degradation."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_store_path()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._broken = False
        self._hit_count = 0
        self._miss_count = 0
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=5.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.warning("persistent store unavailable (%s): cache disabled", exc)
            self._conn = None
            self._broken = True

    # ------------------------------------------------------------------
    # Key/value JSON entries
    # ------------------------------------------------------------------

    def set(self, namespace: str, key: str, value: Any) -> bool:
        """Store ``value`` (JSON-serializable) under (namespace, key)."""
        if self._conn is None:
            return False
        try:
            payload = json.dumps(value, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return False
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv (namespace, key, value, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (namespace, key, payload, time.time()),
            )
        return True

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the stored value, or None on miss / unavailable store."""
        if self._conn is None:
            self._miss_count += 1
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM kv WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
            except sqlite3.Error:
                self._miss_count += 1
                return None
        if row is None:
            self._miss_count += 1
            return None
        try:
            self._hit_count += 1
            return json.loads(row[0])
        except (TypeError, ValueError):
            self._miss_count += 1
            return None

    def delete(self, namespace: str, key: str) -> bool:
        if self._conn is None:
            return False
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM kv WHERE namespace=? AND key=?", (namespace, key)
            )
        return cur.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Delete every entry in a namespace. Returns rows removed."""
        if self._conn is None:
            return 0
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM kv WHERE namespace=?", (namespace,))
        return cur.rowcount

    def namespace_entries(self, namespace: str, limit: int = 100) -> dict[str, Any]:
        """Return up to ``limit`` entries of a namespace (oldest first)."""
        if self._conn is None:
            return {}
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT key, value FROM kv WHERE namespace=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (namespace, int(limit)),
                ).fetchall()
            except sqlite3.Error:
                return {}
        out: dict[str, Any] = {}
        for key, payload in rows:
            try:
                out[key] = json.loads(payload)
            except (TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------
    # Counters (for failure statistics)
    # ------------------------------------------------------------------

    def bump(self, namespace: str, key: str, amount: int = 1) -> int:
        """Increment a counter and return its new value."""
        if self._conn is None:
            return 0
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO counters (namespace, key, n, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET n = n + excluded.n, "
                "updated_at = excluded.updated_at",
                (namespace, key, int(amount), time.time()),
            )
            row = self._conn.execute(
                "SELECT n FROM counters WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
        return int(row[0]) if row else 0

    def counter(self, namespace: str, key: str) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT n FROM counters WHERE namespace=? AND key=?", (namespace, key)
                ).fetchone()
            except sqlite3.Error:
                return 0
        return int(row[0]) if row else 0

    def counters_in_namespace(self, namespace: str) -> dict[str, int]:
        if self._conn is None:
            return {}
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT key, n FROM counters WHERE namespace=? ORDER BY n DESC",
                    (namespace,),
                ).fetchall()
            except sqlite3.Error:
                return {}
        return {key: int(n) for key, n in rows}

    # ------------------------------------------------------------------
    # Maintenance / stats
    # ------------------------------------------------------------------

    def prune_namespace(self, namespace: str, max_entries: int) -> int:
        """Keep only the newest ``max_entries`` rows in a namespace."""
        if self._conn is None or max_entries <= 0:
            return 0
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM kv WHERE namespace=? AND key NOT IN ("
                "  SELECT key FROM kv WHERE namespace=? "
                "  ORDER BY updated_at DESC LIMIT ?"
                ")",
                (namespace, namespace, int(max_entries)),
            )
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self._path),
            "available": self._conn is not None,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": (
                round(self._hit_count / (self._hit_count + self._miss_count), 3)
                if (self._hit_count + self._miss_count)
                else 0.0
            ),
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_STORE: PersistentStore | None = None
_STORE_ENV_VAR = "OPENHARNESS_CACHE_DB"


def get_persistent_store() -> PersistentStore:
    """Return the process-wide store (created lazily).

    The location honors ``OPENHARNESS_CACHE_DB`` so tests (and users with
    a custom data dir) can redirect it.
    """
    global _STORE
    if _STORE is None:
        custom = os.environ.get(_STORE_ENV_VAR, "").strip()
        _STORE = PersistentStore(custom or None)
    return _STORE


def set_persistent_store(store: PersistentStore | None) -> None:
    """Test/CLI helper: install a specific store instance (or reset)."""
    global _STORE
    _STORE = store


def fingerprint(*parts: Any) -> str:
    """Stable SHA-256 fingerprint over the given parts.

    Used by the render cache (scene code + assets + engine args) and the
    semantic cache (prompt text). Parts are serialized deterministically:
    ``repr``-style for scalars, JSON for collections.
    """
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            digest.update(part.encode("utf-8", errors="replace"))
        else:
            try:
                digest.update(
                    json.dumps(part, sort_keys=True, default=str, separators=(",", ":")).encode()
                )
            except (TypeError, ValueError):
                digest.update(repr(part).encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()
