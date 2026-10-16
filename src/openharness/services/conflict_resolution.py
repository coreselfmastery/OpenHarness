"""Cross-agent file conflict resolution — item #31.

When several agents edit the same project concurrently, the last
writer silently wins and the other agent's edit is LOST. This module
makes write/edit file operations lock-guarded with a two-level lock:

1. an in-process ``asyncio.Lock`` keyed by resolved path (two
   coroutines in this process cannot interleave), and
2. an atomic OS lockfile (``O_CREAT | O_EXCL``) so agents in OTHER
   processes also exclude each other. Stale locks (holder crashed
   without releasing) are stolen after ``stale_seconds``.

On timeout, :class:`FileConflictError` names the holder; the write and
edit tools surface it as a clear tool error so the model retries or
picks another file instead of clobbering a concurrent agent's work.

Lock files live under ``~/.openharness/locks/`` (never next to the
target file), keyed by a hash of the resolved path.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)


class FileConflictError(RuntimeError):
    """Another agent holds the lock on this path (#31)."""


def _lock_path_for(path: Path) -> Path:
    from openharness.config.paths import get_data_dir

    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in path.name)[:60]
    return get_data_dir() / "locks" / f"{safe_name}-{digest}.lock"


class _PathLockRegistry:
    """In-process asyncio locks keyed by resolved path."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._holders: dict[str, str] = {}

    def get(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock


_REGISTRY = _PathLockRegistry()
_DEFAULT_STALE_SECONDS = 600.0


def _pid_from_label(label: str) -> int | None:
    """Extract the trailing pid from a lock label like
    ``openharness edit_file (pid 1234)``. Returns None when absent."""
    match = re.search(r"pid\s+(\d+)", label)
    return int(match.group(1)) if match else None


def _process_alive(pid: int) -> bool:
    """Best-effort liveness probe for a pid (False = not alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else
    except OSError:
        return True
    return True


def _try_lockfile(lock_path: Path, label: str, *, stale_seconds: float) -> str:
    """Atomically create the cross-process lockfile. Returns "" on success.

    Returns the current holder's label when the lock is held by a live
    process. Stale locks are stolen when EITHER (a) the lockfile is older
    than ``stale_seconds``, or (b) the holder pid recorded in the file is
    no longer alive — the pid check recovers instantly from a killed or
    crashed holder instead of blocking edits on that file for the whole
    mtime window.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.debug("lock dir unavailable (%s); proceeding in-process only", exc)
        return ""
    # Steal stale locks (holder died without releasing).
    try:
        age = time.time() - lock_path.stat().st_mtime
        if age > stale_seconds:
            lock_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            current = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "another process"
        if not current:
            return "another process"
        holder_pid = _pid_from_label(current)
        if holder_pid is not None and holder_pid != os.getpid() and not _process_alive(holder_pid):
            # Holder process is gone — steal its lock immediately.
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return current
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(label)
            return ""
        return current or "another process"
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(label)
    return ""


def _release_lockfile(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


@asynccontextmanager
async def acquire_path_lock(
    path: Path,
    *,
    timeout_seconds: float = 30.0,
    holder_label: str = "",
    stale_seconds: float = _DEFAULT_STALE_SECONDS,
) -> AsyncIterator[None]:
    """Hold the cross-agent lock on ``path`` for the duration of the block.

    Raises :class:`FileConflictError` when the lock cannot be taken
    within ``timeout_seconds`` — callers surface that as a tool error
    rather than clobbering the concurrent writer.
    """
    resolved = str(Path(path).expanduser().resolve())
    lock_path = _lock_path_for(Path(resolved))
    # Always embed the pid so other processes can detect a dead holder
    # and steal the lock instantly instead of waiting out the stale age.
    label = holder_label or "agent"
    label = f"{label} (pid {os.getpid()})"
    asyncio_lock = _REGISTRY.get(resolved)

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    got_process_lock = False
    last_external_holder = ""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FileConflictError(
                    f"file is locked by {last_external_holder or _REGISTRY._holders.get(resolved) or 'another agent'}: {resolved}"
                )
            try:
                got_process_lock = await asyncio.wait_for(
                    asyncio_lock.acquire(), timeout=remaining
                )
            except asyncio.TimeoutError:
                raise FileConflictError(
                    f"file is locked by {last_external_holder or _REGISTRY._holders.get(resolved) or 'another agent'}: {resolved}"
                ) from None
            _REGISTRY._holders[resolved] = label
            holder = _try_lockfile(lock_path, label, stale_seconds=stale_seconds)
            if holder == "":
                break  # both levels acquired
            # Another PROCESS holds it: back off and retry until deadline.
            last_external_holder = holder
            _REGISTRY._holders.pop(resolved, None)
            asyncio_lock.release()
            got_process_lock = False
            if time.monotonic() >= deadline:
                raise FileConflictError(
                    f"file is locked by {holder}: {resolved}"
                )
            await asyncio.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
        try:
            yield
        finally:
            _release_lockfile(lock_path)
    finally:
        if got_process_lock:
            _REGISTRY._holders.pop(resolved, None)
            asyncio_lock.release()


def configure_file_locking_from_settings(settings: object) -> bool:
    """Report whether write/edit tools should lock (#31)."""
    control = getattr(settings, "control", None)
    return bool(getattr(control, "file_locking", True)) if control is not None else True


def lock_timeout_from_settings(settings: object) -> float:
    control = getattr(settings, "control", None)
    if control is None:
        return 30.0
    return float(getattr(control, "file_lock_timeout_seconds", 30) or 30)
