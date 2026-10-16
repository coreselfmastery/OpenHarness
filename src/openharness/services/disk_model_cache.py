"""Disk-based model cache — item 8.

Caches model responses to disk so repeated requests across sessions
don't re-call the LLM. 5-10 min saved per session on warm starts.

Different from the semantic cache (#7) which is in-memory: this one
persists to disk and survives process restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DiskModelCache:
    """Persistent disk cache for LLM responses.

    Caches by the hash of (model, messages, system_prompt, max_tokens).
    Survives process restarts. Useful for development iteration where
    the same prompts are sent repeatedly.

    Usage:
        cache = DiskModelCache(cache_dir=".openharness/model_cache")
        key = cache.make_key(model="gpt-4o", messages=msgs, system_prompt=sys)
        cached = cache.get(key)
        if cached:
            return cached  # skip LLM call
        response = await llm_call(...)
        cache.put(key, response)
    """

    def __init__(self, *, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        *,
        model: str,
        messages: list[Any],
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Build a stable cache key from the request parameters."""
        # Serialize messages deterministically
        try:
            msgs_str = json.dumps(messages, sort_keys=True, default=str)
        except (TypeError, ValueError):
            msgs_str = str(messages)
        payload = f"{model}|{system_prompt}|{max_tokens}|{msgs_str}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        """Return a cached response, or None if not cached."""
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return data.get("response", "")
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to read cache entry %s: %s", key, exc)
            self._misses += 1
            return None

    def put(self, key: str, response: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Store a response in the cache."""
        path = self._cache_dir / f"{key}.json"
        data = {
            "response": response,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("failed to write cache entry %s: %s", key, exc)

    def invalidate(self, key: str) -> None:
        """Remove a specific entry."""
        path = self._cache_dir / f"{key}.json"
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all cached entries."""
        for f in self._cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        entries = len(list(self._cache_dir.glob("*.json")))
        total = self._hits + self._misses
        return {
            "entries": entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "cache_dir": str(self._cache_dir),
        }


# Singleton
_DISK_CACHE: DiskModelCache | None = None


def get_disk_model_cache() -> DiskModelCache | None:
    """Return the global disk model cache, or None if not configured."""
    return _DISK_CACHE


def configure_disk_model_cache(*, cache_dir: str | Path) -> None:
    """Create / replace the global disk model cache."""
    global _DISK_CACHE
    _DISK_CACHE = DiskModelCache(cache_dir=cache_dir)


def reset_disk_model_cache() -> None:
    """Test helper: drop the global cache."""
    global _DISK_CACHE
    _DISK_CACHE = None
