"""Tool result cache.

Caches the output of read-only tool calls so the same ``read_file``,
``grep``, or ``glob`` invocation never runs twice within a session.

The cache is keyed by ``(tool_name, canonical_input)`` and invalidated
on any mutating tool call (``write_file``, ``edit_file``,
``notebook_edit``, a bash command that does not match the
obviously-read-only allowlist, or any tool marked non-read-only).

This implements optimization #7 from the top-10 list:
"Don't re-run identical tools. Cache results for a set time."

The cache is opt-in via ``settings.tool_cache_enabled = True`` (or
``OPENHARNESS_TOOL_CACHE=1``). When disabled, all methods are no-ops
and add zero overhead (a single bool check at entry).
"""
from __future__ import annotations

import asyncio
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
# that include ``a.py``). ``bash`` used to invalidate on *every* call,
# which collapsed hit rates in mutating sessions even when the model was
# just running ``ls``/``git status`` between edits — read-only bash calls
# are now recognized by ``_bash_command_mutates`` (conservative: anything
# not clearly read-only still invalidates).
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "notebook_edit",
    "todo_write",
})

# Bash first-words that cannot change filesystem state (no redirection,
# no subcommand that writes). ``git`` is special-cased by subcommand.
_BASH_READ_ONLY_WORDS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "pwd", "echo", "printf", "which",
    "whereis", "rg", "grep", "fd", "find", "wc", "stat", "file", "du",
    "df", "ps", "env", "printenv", "date", "uname", "whoami", "id",
    "hostname", "true", "false", "test", "[", "nproc", "free", "lscpu",
    "nvidia-smi", "python", "python3", "node", "npm", "pip", "cargo",
    "git", "diff", "cmp", "md5sum", "sha256sum", "tree", "w", "top",
    "uptime", "groups", "getconf", "type", "command", "help", "cd",
})

# ``git <subcommand>`` pairs that are read-only. Subcommands in the
# first set never mutate; the second set (branch/tag/remote) lists when
# invoked bare but CREATES when given a name argument, so they count as
# read-only only with no extra non-flag argument.
_GIT_ALWAYS_READ_ONLY: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "rev-parse", "blame",
    "shortlog", "describe", "ls-files", "ls-remote", "reflog",
    "name-rev", "annotate", "cat-file", "count-objects",
    "verify-commit", "verify-tag",
})
_GIT_LIST_ONLY_SUBS: frozenset[str] = frozenset({"branch", "tag", "remote", "worktree"})

# ``npm/pip/cargo/python <subcommand>`` pairs that are read-only-ish:
# pip list/show, npm ls, cargo --version, python --version. Anything
# else under these interpreters is assumed mutating (``python -c`` can
# write files, ``pip install`` mutates the environment).
_INTERPRETER_READ_ONLY_SUBS: dict[str, frozenset[str]] = {
    "pip": frozenset({"list", "show", "freeze", "index"}),
    "npm": frozenset({"ls", "list", "outdated", "view", "search", "config"}),
    "cargo": frozenset({"--version", "search"}),
    "python": frozenset({"--version", "-V", "-m", "pip"}),
    "python3": frozenset({"--version", "-V", "-m", "pip"}),
    "node": frozenset({"--version", "-v"}),
}

# Mutating markers anywhere in the command line. NOTE: plain substring
# matches — over-matching is SAFE here (it only costs a cache miss).
_BASH_MUTATING_MARKERS: tuple[str, ...] = (
    ">", ">>", "tee", "xargs", "truncate", "shred", "touch", "mkdir",
    "rmdir", "chmod", "chown", "ln -s", "sed -i", "perl -i",
    "gawk -i", "install -m", "sync", "mkfs", "dd ", "-delete",
    "-exec", "-ok", "-fprint", "unlink", "rsync", "git push",
    "git pull", "git commit", "git merge", "git rebase",
    "git checkout", "git reset", "git clean", "git apply", "git am",
    "git add", "git stash", "git worktree add", "git worktree remove",
    "git switch", "git restore", "pip install", "pip uninstall",
    "pip download", "npm install", "npm ci", "npm publish", "npm update",
    "npm uninstall", "cargo build", "cargo install", "cargo run",
    "make ", "cmake ", "tar ", "unzip", "gzip", "gunzip", "7z",
    "curl -o", "curl -O", "wget ", "scp ", "ssh ", "kill ", "pkill",
    "reboot", "shutdown", "apt ", "apt-get ", "yum ", "brew ",
)


def _split_shell_segments(command: str) -> list[str]:
    """Split a shell command on && ; | separators (quote-aware-lite)."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for ch in command:
        if quote:
            if ch == quote:
                quote = None
            current.append(ch)
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
            continue
        if ch in "&;|":
            if ch == "&" and current and current[-1] == "&":
                current.pop()
                segments.append("".join(current).strip())
                current = []
            elif ch in ";|":
                segments.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            continue
        current.append(ch)
    segments.append("".join(current).strip())
    return [s for s in segments if s]


def _segment_is_read_only(segment: str) -> bool:
    """One shell segment is read-only (conservatively judged)."""
    import re
    import shlex

    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False  # unparseable — assume mutating
    # Strip leading VAR=value assignments (FOO=bar ls ...).
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return True  # pure assignment is not a cache-relevant mutation
    head = tokens[0]
    # bare command name or a path to it ("./scripts/x.sh" → "x")
    base = head.rsplit("/", 1)[-1]
    if base == "sudo":
        tokens = tokens[1:]
        if not tokens:
            return False
        base = tokens[0].rsplit("/", 1)[-1]
    if base not in _BASH_READ_ONLY_WORDS:
        # Unknown command — assume mutating. rm/mv/cp/dd and friends all
        # land here: they are simply not in the read-only word list.
        return False
    if base == "git":
        rest = [t for t in tokens[1:]]
        sub = next((t for t in rest if not t.startswith("-")), "")
        if sub == "":
            # bare git / flags-only git (``git tag -l``) prints — safe
            return True
        if sub in _GIT_ALWAYS_READ_ONLY:
            return True
        if sub in _GIT_LIST_ONLY_SUBS:
            # bare ``git branch`` / ``git tag -l`` lists;
            # ``git branch feature`` / ``git tag v1`` creates.
            positional = [t for t in rest[1:] if not t.startswith("-")]
            return len(positional) == 0
        return False
    if base in _INTERPRETER_READ_ONLY_SUBS:
        sub = tokens[1] if len(tokens) > 1 else "--version"
        allowed = _INTERPRETER_READ_ONLY_SUBS[base]
        if sub in allowed:
            return True
        if base.startswith("python") and sub == "-m":
            mod = tokens[2] if len(tokens) > 2 else ""
            return mod == "pip"  # python -m pip → judge like pip
        return False
    # find with -delete/-exec is mutating — excluded via markers below.
    return True


def _bash_command_mutates(command: str) -> bool:
    """Conservative: True unless the WHOLE command line is obviously read-only.

    Any redirection, any mutating marker anywhere, any unknown command
    → invalidate (the safe default). A plain ``ls`` between two edits
    no longer wipes the cache.
    """
    if not command or not command.strip():
        return False
    if any(marker in command for marker in _BASH_MUTATING_MARKERS):
        return True
    try:
        segments = _split_shell_segments(command)
    except Exception:  # noqa: BLE001 - parser must never raise
        return True
    if not segments:
        return False
    return not all(_segment_is_read_only(seg) for seg in segments)


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
        # Monotonic generation counter. Bumped on every invalidate_all so
        # in-flight executions that started before a mutation can detect
        # that their result is stale and must not be cached.
        self._generation = 0
        # In-flight execution tracker: when two coroutines request the
        # same cache key concurrently, only the first executes the tool;
        # the second awaits the same future. Prevents thundering-herd
        # on cache misses for slow tools (large read_file, expensive grep).
        self._in_flight: dict[str, "asyncio.Future[dict[str, Any] | None]"] = {}

    @staticmethod
    def is_cacheable(tool_name: str) -> bool:
        """Return True if ``tool_name`` is a read-only tool we cache."""
        return tool_name in _CACHEABLE_TOOLS

    @staticmethod
    def is_mutating(tool_name: str, tool_input: dict[str, Any] | None = None) -> bool:
        """Return True if this call mutates filesystem state.

        For ``bash`` the decision is per-COMMAND via
        :func:`_bash_command_mutates` (read-only commands like ``ls`` or
        ``git status`` no longer wipe the whole cache); every other
        mutating tool still invalidates unconditionally.
        """
        if tool_name == "bash":
            command = str((tool_input or {}).get("command") or "")
            if not command.strip():
                return True  # no command visible — assume the worst
            return _bash_command_mutates(command)
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
        # Bump the generation so in-flight executions (started before the
        # mutation) know not to cache their now-stale results. The futures
        # themselves are left alone — their awaiters still get a result,
        # it just won't be persisted.
        self._generation += 1
        self._in_flight.clear()

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

    async def get_or_execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        executor: "Any",
    ) -> dict[str, Any] | None:
        """Return cached result, or execute ``executor()`` and cache it.

        This is the thundering-herd-safe entry point. When two coroutines
        call this with the same ``(tool_name, tool_input)`` concurrently:

        1. The first coroutine calls ``executor()`` and stores the result.
        2. The second coroutine awaits the same future — it does NOT
           call ``executor()`` again.

        Returns the cached result dict (``{output, is_error, metadata}``),
        or ``None`` if the executor returned an error (errors are not
        cached, so the next call retries).

        Args:
            tool_name: Tool name for the cache key.
            tool_input: Tool input dict for the cache key.
            executor: An async callable (zero-arg) returning a dict with
                ``output``, ``is_error``, ``metadata`` keys.
        """
        # Fast path: cache hit.
        cached = self.get(tool_name, tool_input)
        if cached is not None:
            return cached

        key = self.make_key(tool_name, tool_input)

        # Check if another coroutine is already executing this exact call.
        existing_future = self._in_flight.get(key)
        if existing_future is not None:
            # Another coroutine is executing. Await its result instead
            # of running the executor again.
            try:
                result = await existing_future
            except Exception:
                # The in-flight execution raised — fall through to
                # execute ourselves (rare; usually means the executor
                # had a bug). Don't cache the error.
                return None
            if result is None:
                # The other coroutine's executor returned an error —
                # don't reuse it; execute ourselves to retry.
                pass
            else:
                # Reuse the other coroutine's result.
                return result

        # We're the first coroutine. Create a future so concurrent
        # callers can await our result.
        future: "asyncio.Future[dict[str, Any] | None]" = asyncio.get_running_loop().create_future()
        self._in_flight[key] = future
        generation = self._generation
        try:
            try:
                result = await executor()
            except Exception:
                # Executor raised — don't cache, don't propagate to
                # waiters (let them retry on next call).
                future.set_result(None)
                self._in_flight.pop(key, None)
                raise
            if isinstance(result, dict) and not result.get("is_error"):
                # Only cache if no mutation invalidated the cache while the
                # executor was running — otherwise we'd persist a result
                # computed against pre-mutation filesystem state.
                if self._generation == generation:
                    self.put(tool_name, tool_input, result)
            future.set_result(result if isinstance(result, dict) else None)
            return result if isinstance(result, dict) else None
        finally:
            self._in_flight.pop(key, None)


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
        max_raw = getattr(settings, "tool_cache_max_entries", 256)
        # 0 is a legitimate value (cache nothing) — only a missing/None
        # field falls back to the default.
        max_entries = 256 if max_raw is None else max(0, int(max_raw))
        ttl_raw = getattr(settings, "tool_cache_ttl_seconds", 3600)
        # 0 means "never expire" (documented in MIGRATION.md: rely on
        # mutation-based invalidation only). The previous ``or 3600``
        # coerced 0 -> 3600, silently breaking the documented behaviour.
        ttl_seconds = 3600.0 if ttl_raw is None else max(0.0, float(ttl_raw))
        enable_tool_cache(max_entries=max_entries, ttl_seconds=ttl_seconds)
    else:
        disable_tool_cache()


def reset_tool_cache() -> None:
    """Test helper: drop the cache and reset all counters."""
    global _TOOL_CACHE
    if _TOOL_CACHE is not None:
        _TOOL_CACHE.invalidate_all()
        _TOOL_CACHE.clear_stats()
