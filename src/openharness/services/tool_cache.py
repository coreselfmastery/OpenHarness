"""Tool result cache.

Caches the output of read-only tool calls so the same ``read_file``,
``grep``, or ``glob`` invocation never runs twice within a session.

The cache is keyed by ``(tool_name, canonical_input)`` and invalidated
on any mutating tool call (``write_file``, ``edit_file``, ``notebook_edit``,
``bash`` with non-zero return code, or any tool marked non-read-only).

This implements optimization #7 from the top-10 list:
"Don't re-run identical tools. Cache results for a set time."

The cache is opt-in via ``settings.tool_cache_enabled = True`` (or
``OPENHARNESS_TOOL_CACHE=1``). When disabled, all methods are no-ops
and add zero overhead (a single bool check at entry).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)


# Tools whose results we will cache. Restricted to read-only tools whose
# output depends only on the filesystem state. ``web_fetch`` / ``web_search``
# are intentionally excluded because their results can change between calls.
_CACHEABLE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "grep",
    "glob",
    "list_mcp_resources",
})

# Tools that mutate state and therefore invalidate the cache when called.
# We invalidate the *entire* cache because the mutation could affect any
# cached path (e.g. ``write_file("a.py", ...)`` invalidates ``grep`` results
# that include ``a.py``).
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "notebook_edit",
    "bash",  # bash may mutate; we invalidate only on non-zero returncode
    "todo_write",
})


class ToolResultCache:
    """In-process LRU cache for read-only tool results.

    Thread-safe enough for asyncio single-thread use. If multiple
    coroutines call the same read-only tool concurrently, the second
    one will simply miss the cache, run the tool, and store its own
    result. This is the same behaviour as the previous uncached path.
    """

    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 600.0) -> None:
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    @staticmethod
    def is_cacheable(tool_name: str) -> bool:
        """Return True if ``tool_name`` is a read-only tool we cache."""
        return tool_name in _CACHEABLE_TOOLS

    @staticmethod
    def is_mutating(tool_name: str) -> bool:
        """Return True if ``tool_name`` mutates filesystem state."""
        return tool_name in _MUTATING_TOOLS

    @staticmethod
    def make_key(tool_name: str, tool_input: dict[str, Any]) -> str:
        """Build a stable cache key from the tool name + input kwargs.

        Sorts dict keys so identical inputs in different orders collide.
        Strips non-JSON-serializable values defensively.
        """
        try:
            payload = json.dumps(
                {"t": tool_name, "i": tool_input},
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            # If we can't serialize the input, fall back to a hash of the
            # repr so we at least get *some* collision resistance.
            payload = f"{tool_name}:{tool_input!r}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
        """Return the cached result for ``(tool_name, tool_input)`` or None.

        ``None`` means "not in cache OR expired". Callers must be prepared
        to execute the tool and call :meth:`put` with the result.

        When ``ttl_seconds`` is 0 (or negative), entries never expire —
        they only evict via LRU or mutation.
        """
        key = self.make_key(tool_name, tool_input)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self._ttl_seconds > 0 and (time.monotonic() - entry.created_at) > self._ttl_seconds:
            # Expired — evict and miss.
            self._entries.pop(key, None)
            self._misses += 1
            return None
        # Move to end (most-recently-used) for LRU eviction order.
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.result

    def put(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Store ``result`` for ``(tool_name, tool_input)``.

        ``result`` is the dict form of a ``ToolResult`` (output, is_error,
        metadata). We only cache successful results — errors are not cached
        so the next call retries.
        """
        if result.get("is_error"):
            return
        key = self.make_key(tool_name, tool_input)
        self._entries[key] = _CacheEntry(
            result=dict(result),
            created_at=time.monotonic(),
        )
        self._entries.move_to_end(key)
        # Evict oldest if over capacity.
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def invalidate_all(self) -> None:
        """Drop every cached entry. Called after any mutating tool call."""
        if self._entries:
            self._invalidations += len(self._entries)
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss/invalidation counters (for debugging)."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": self._invalidations,
            "size": len(self._entries),
            "max_size": self._max_entries,
        }

    def clear_stats(self) -> None:
        """Reset hit/miss counters without dropping the cache."""
        self._hits = 0
        self._misses = 0
        self._invalidations = 0


class _CacheEntry:
    """Single cache entry — holds the result and its creation time."""

    __slots__ = ("result", "created_at")

    def __init__(self, result: dict[str, Any], created_at: float) -> None:
        self.result = result
        self.created_at = created_at


# Singleton — one cache per process. Tools call :func:`get_tool_cache` to
# access it. The cache is created lazily on first use so we don't pay
# the memory cost in processes that don't enable it.
_TOOL_CACHE: ToolResultCache | None = None


def get_tool_cache() -> ToolResultCache | None:
    """Return the global tool cache, or ``None`` if caching is disabled.

    Callers should check the return value before calling ``get`` / ``put`` —
    a ``None`` return means "caching disabled, run the tool directly".
    """
    return _TOOL_CACHE


def enable_tool_cache(*, max_entries: int = 256, ttl_seconds: float = 600.0) -> None:
    """Create / replace the global tool cache with the given parameters."""
    global _TOOL_CACHE
    _TOOL_CACHE = ToolResultCache(max_entries=max_entries, ttl_seconds=ttl_seconds)


def disable_tool_cache() -> None:
    """Drop the global tool cache and disable caching."""
    global _TOOL_CACHE
    _TOOL_CACHE = None


def configure_tool_cache_from_settings(settings: Any) -> None:
    """Enable / disable the global cache based on Settings.

    Called once at engine startup. If ``settings.tool_cache_enabled`` is
    False (the default), the cache is disabled and ``get_tool_cache()``
    returns ``None``. When enabled, the TTL is read from
    ``settings.tool_cache_ttl_seconds`` (default 3600s) so stale
    filesystem reads eventually expire even without a mutating call.
    """
    if getattr(settings, "tool_cache_enabled", False):
        max_entries = int(getattr(settings, "tool_cache_max_entries", 256) or 256)
        ttl_seconds = float(getattr(settings, "tool_cache_ttl_seconds", 3600) or 3600)
        enable_tool_cache(max_entries=max_entries, ttl_seconds=ttl_seconds)
    else:
        disable_tool_cache()


def reset_tool_cache() -> None:
    """Test helper: drop the cache and reset all counters."""
    global _TOOL_CACHE
    if _TOOL_CACHE is not None:
        _TOOL_CACHE.invalidate_all()
        _TOOL_CACHE.clear_stats()
