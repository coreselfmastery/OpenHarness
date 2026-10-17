"""Semantic memory — recall past decisions by meaning, not exact text.

``MEMORY.md`` is plain text: you cannot search it by meaning. This
store keeps durable facts ("scene 12 uses cubic-bezier(0.4, 0, 0.2, 1)
for transitions") as *embeddings* in SQLite, so a later query ("what
easing did scene 12 use?") finds the closest stored memories by
cosine similarity — across sessions and across projects.

Embedding backends (auto-selected):

- **HTTP** — any OpenAI-compatible ``/embeddings`` endpoint (BGE-M3,
  MiniLM, nomic-embed...) configured via ``settings.embedding`` or
  ``OPENHARNESS_EMBEDDING_*`` env vars. Recommended for true semantic
  recall.
- **Local hashing fallback** — a deterministic character/word n-gram
  hashing embedder (256 dims, no network, no dependencies). It is
  lexical-approximate, not truly semantic, but keeps the store fully
  functional offline; the backend used is recorded per entry and
  surfaced in :meth:`SemanticMemory.stats`.

The database lives at ``<data_dir>/video/semantic_memory.sqlite3`` and
degrades gracefully (empty results) when unwritable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass
from itertools import pairwise as _pairwise
from pathlib import Path
from typing import Any

from openharness.services._sqlite_kit import SQLiteStore, video_store_dir

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project       TEXT NOT NULL DEFAULT 'default',
    kind          TEXT NOT NULL DEFAULT 'fact',
    text          TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    backend       TEXT NOT NULL DEFAULT 'hashing',
    dims          INTEGER NOT NULL,
    embedding     BLOB NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_project ON memories (project, created_at);
"""

_HASH_DIMS = 256


# ----------------------------------------------------------------------
# Embedding backends
# ----------------------------------------------------------------------
def _hashing_embed(text: str) -> list[float]:
    """Deterministic n-gram hashing embedding (256 dims, L2-normalized).

    Word unigrams/bigrams + character trigrams are hashed into a fixed
    vector — recall approximates lexical overlap. No network needed.
    """
    vector = [0.0] * _HASH_DIMS
    cleaned = " ".join(text.lower().split())
    if not cleaned:
        return vector
    words = cleaned.split(" ")
    tokens: list[str] = list(words)
    tokens.extend(f"{a}_{b}" for a, b in _pairwise(words))
    tokens.extend(
        cleaned[i : i + 3] for i in range(max(0, len(cleaned) - 2))
    )
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = struct.unpack("<I", digest[:4])[0] % _HASH_DIMS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _http_embed(
    base_url: str, model: str, api_key: str, texts: list[str], timeout: float
) -> list[list[float]] | None:
    """Call an OpenAI-compatible /embeddings endpoint; None on failure."""
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/embeddings",
                json={"model": model, "input": texts},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data") or []
        embedded = [list(map(float, item["embedding"])) for item in data]
        if len(embedded) != len(texts):
            return None
        return embedded
    except Exception as exc:  # noqa: BLE001 - fall back to hashing
        log.debug("http embedding failed, using hashing fallback: %s", exc)
        return None


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np

        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except ImportError:  # pragma: no cover - numpy ships with opencv
        # Correct fallback: ||a|| * ||b||. The old version used
        # sqrt(dot(a,b)) * sqrt(dot(b,b)) — cosine([1,1],[1,0]) returned
        # 1.0 instead of 0.707, and negative dots raised the value into
        # the complex plane, crashing the later sort with TypeError.
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        denom = norm_a * norm_b
        if denom == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / denom


@dataclass
class RecalledMemory:
    """One recalled memory with its similarity score."""

    id: int
    project: str
    kind: str
    text: str
    similarity: float
    metadata: dict[str, Any]
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "kind": self.kind,
            "text": self.text,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class SemanticMemory:
    """SQLite-backed semantic memory with pluggable embeddings."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._store = SQLiteStore(
            path or video_store_dir() / "semantic_memory.sqlite3", _SCHEMA
        )
        self._backend = "hashing"
        self._http: dict[str, Any] | None = None
        self._load_config()

    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        """Resolve the embedding backend from settings / environment."""
        base_url = ""
        model = "BAAI/bge-m3"
        api_key = "EMPTY"
        timeout = 30.0
        try:
            from openharness.config.settings import load_settings

            settings = load_settings()
            config = getattr(settings, "embedding", None)
            if config is not None:
                base_url = str(getattr(config, "base_url", "") or "").strip()
                model = str(getattr(config, "model", "") or model).strip()
                api_key = str(getattr(config, "api_key", "") or api_key).strip()
                timeout = float(getattr(config, "timeout_seconds", 30.0) or 30.0)
        except Exception:  # settings optional — env vars still apply below
            log.debug("semantic_memory: settings load failed, env-only config", exc_info=True)
        import os

        base_url = (
            os.environ.get("OPENHARNESS_EMBEDDING_BASE_URL", "").strip() or base_url
        )
        model = os.environ.get("OPENHARNESS_EMBEDDING_MODEL", "").strip() or model
        api_key = os.environ.get("OPENHARNESS_EMBEDDING_API_KEY", "").strip() or api_key
        if base_url:
            self._http = {
                "base_url": base_url,
                "model": model,
                "api_key": api_key or "EMPTY",
                "timeout": timeout,
            }

    # ------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> tuple[list[list[float]], str]:
        """Embed texts; returns (vectors, backend_name)."""
        if self._http is not None:
            embedded = _http_embed(
                self._http["base_url"],
                self._http["model"],
                self._http["api_key"],
                texts,
                float(self._http["timeout"]),
            )
            if embedded is not None:
                return embedded, f"http:{self._http['model']}"
        return [_hashing_embed(text) for text in texts], "hashing"

    # ------------------------------------------------------------------
    def remember(
        self,
        text: str,
        *,
        project: str = "default",
        kind: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Store one memory; returns its row id (None when disabled)."""
        text = text.strip()
        if not text:
            return None
        vector, backend = self._embed([text])
        cursor = self._store.execute(
            "INSERT INTO memories (project, kind, text, metadata_json, "
            "backend, dims, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project,
                kind,
                text,
                json.dumps(metadata or {}),
                backend,
                len(vector[0]),
                _pack(vector[0]),
                self._store.now(),
            ),
        )
        return int(cursor.lastrowid) if cursor is not None else None

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        project: str | None = None,
        min_similarity: float = 0.05,
    ) -> list[RecalledMemory]:
        """Return the top-K closest stored memories to ``query``."""
        query = query.strip()
        if not query:
            return []
        vector, _backend = self._embed([query])
        rows = self._store.query(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT 5000"
        )
        scored: list[RecalledMemory] = []
        for row in rows:
            if project is not None and row["project"] != project:
                continue
            if int(row["dims"]) != len(vector[0]):
                continue  # different embedding space — not comparable
            similarity = _cosine(vector[0], _unpack(row["embedding"]))
            if similarity < min_similarity:
                continue
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except ValueError:
                metadata = {}
            scored.append(
                RecalledMemory(
                    id=int(row["id"]),
                    project=row["project"],
                    kind=row["kind"],
                    text=row["text"],
                    similarity=similarity,
                    metadata=metadata,
                    created_at=float(row["created_at"]),
                )
            )
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[: max(0, int(top_k))]

    def forget(self, memory_id: int) -> bool:
        cursor = self._store.execute(
            "DELETE FROM memories WHERE id = ?", (int(memory_id),)
        )
        return cursor is not None and cursor.rowcount > 0

    def list_recent(
        self, *, project: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if project is not None:
            rows = self._store.query(
                "SELECT id, project, kind, text, backend, created_at FROM memories "
                "WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, int(limit)),
            )
        else:
            rows = self._store.query(
                "SELECT id, project, kind, text, backend, created_at FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            )
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        rows = self._store.query(
            "SELECT project, COUNT(*) AS n FROM memories GROUP BY project"
        )
        total = sum(int(row["n"]) for row in rows)
        backends = self._store.query(
            "SELECT backend, COUNT(*) AS n FROM memories GROUP BY backend"
        )
        return {
            "available": self._store.available,
            "path": str(self._store.path),
            "total": total,
            "by_project": {row["project"]: int(row["n"]) for row in rows},
            "by_backend": {row["backend"]: int(row["n"]) for row in backends},
            "active_backend": (
                f"http:{self._http['model']}" if self._http else "hashing"
            ),
        }

    def reset(self) -> None:
        """Drop every memory (test helper)."""
        self._store.execute("DELETE FROM memories")


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------
_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    global _memory
    if _memory is None:
        _memory = SemanticMemory()
    return _memory


def set_semantic_memory(memory: SemanticMemory | None) -> None:
    global _memory
    _memory = memory
