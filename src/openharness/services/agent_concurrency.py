"""Concurrency limiter for spawned agents.

Implements optimization #1 from the top-10 list:
"Increase wave limit to 15-20 concurrent agents".

The semaphore is held for the *entire lifetime* of the spawned agent
subprocess, not just around the spawn call. This is the critical fix:
``create_agent_task`` returns in milliseconds (it just forks the
subprocess), so without holding the permit until the subprocess exits,
the limit doesn't actually cap concurrent *running* agents.

We achieve this by registering a completion listener on the
``BackgroundTaskManager`` that releases the permit when the spawned
subprocess reaches a terminal state (completed / failed / killed).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# Singleton semaphore for the process. Created lazily on first use so
# we don't allocate it in processes that never spawn agents (e.g. CLI
# subcommands like ``oh doctor``).
_AGENT_SEMAPHORE: asyncio.Semaphore | None = None
_AGENT_LIMIT: int = 5


def configure_agent_concurrency(settings: Any) -> None:
    """Configure the semaphore from settings.

    Call once at engine startup. If called multiple times with a different
    limit, the semaphore is recreated (existing in-flight agents are
    unaffected because they already hold their permit).
    """
    global _AGENT_SEMAPHORE, _AGENT_LIMIT
    new_limit = max(1, int(getattr(settings, "max_concurrent_agents", 5) or 5))
    if _AGENT_SEMAPHORE is None or new_limit != _AGENT_LIMIT:
        _AGENT_LIMIT = new_limit
        _AGENT_SEMAPHORE = asyncio.Semaphore(new_limit)
        log.debug("agent concurrency limit set to %d", new_limit)


def get_agent_semaphore() -> asyncio.Semaphore:
    """Return the global agent semaphore, creating it if needed.

    If :func:`configure_agent_concurrency` was never called, the default
    limit of 5 is used. This fallback keeps existing tests working
    without modification.
    """
    global _AGENT_SEMAPHORE, _AGENT_LIMIT
    if _AGENT_SEMAPHORE is None:
        _AGENT_SEMAPHORE = asyncio.Semaphore(_AGENT_LIMIT)
    return _AGENT_SEMAPHORE


def get_agent_limit() -> int:
    """Return the configured agent concurrency limit."""
    return _AGENT_LIMIT


def reset_agent_concurrency() -> None:
    """Test helper: reset the semaphore so the next call reconfigures it."""
    global _AGENT_SEMAPHORE, _AGENT_LIMIT
    _AGENT_SEMAPHORE = None
    _AGENT_LIMIT = 5


async def with_agent_limit(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    """Acquire the agent semaphore, run ``coro_factory()``, release.

    ``coro_factory`` is a zero-arg callable that returns a coroutine.
    Used as a context manager around the actual spawn call so the
    semaphore is always released even on exception.

    NOTE: This only caps concurrent *spawn calls*, not concurrent
    *running* agents. For long-running subprocesses (the normal case),
    use :func:`acquire_permit_for_task` instead, which holds the permit
    until the subprocess completes.
    """
    sem = get_agent_semaphore()
    async with sem:
        return await coro_factory()


async def acquire_permit_for_task(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    task_manager: Any = None,
) -> Any:
    """Acquire a permit, spawn the task, hold the permit until the task completes.

    This is the correct API for capping concurrent *running* agents.
    The semaphore is acquired before the spawn, and released only when
    the spawned task reaches a terminal state (completed / failed /
    killed). If the spawn itself fails, the permit is released immediately.

    Args:
        coro_factory: Returns the coroutine that creates the task. The
            coroutine must return a TaskRecord (with an ``id`` field)
            from ``BackgroundTaskManager.create_agent_task``.
        task_manager: Optional ``BackgroundTaskManager`` instance. If
            provided, a completion listener is registered so the permit
            is released when the task exits. If not provided, the permit
            is released immediately after spawn (legacy behaviour —
            callers should pass the task_manager for correct capping).
    """
    sem = get_agent_semaphore()
    await sem.acquire()
    try:
        record = await coro_factory()
    except BaseException:
        # Spawn failed OR was cancelled — release immediately so the
        # next spawn can proceed. ``except Exception`` alone leaked the
        # permit on every CancelledError (kill switch / timeout): N
        # cancelled spawns permanently burned N permits and the pool
        # deadlocked (the next acquire blocked forever).
        sem.release()
        raise

    task_id = getattr(record, "id", None) or str(record)
    if task_manager is None:
        # No way to listen for completion — release now (legacy behaviour).
        # This means the cap is on spawn-rate, not running-count. Callers
        # that want true running-cap must pass task_manager.
        sem.release()
        return record

    # Register a completion listener that releases the permit when the
    # task reaches a terminal state. This is the critical fix: the permit
    # is held for the lifetime of the subprocess, not just the spawn call.
    def _release_on_completion(task_record: Any) -> None:
        if getattr(task_record, "id", None) != task_id:
            return  # Different task — ignore.
        try:
            sem.release()
        except ValueError:
            # Semaphore already at capacity (shouldn't happen, but be defensive).
            pass

    try:
        unregister = task_manager.register_completion_listener(_release_on_completion)
    except Exception:
        # Listener registration failed — release now to avoid leak.
        sem.release()
        return record

    # Edge case: the task may have already completed between spawn and
    # listener registration. Check and release immediately if so.
    existing = task_manager.get_task(task_id) if hasattr(task_manager, "get_task") else None
    if existing is not None and getattr(existing, "status", "") in {"completed", "failed", "killed"}:
        unregister()
        try:
            sem.release()
        except ValueError:
            pass
        return record

    # Stash the unregister callable on the record so callers can clean
    # up if they cancel the task externally. Not strictly required, but
    # makes the lifecycle observable for debugging.
    try:
        object.__setattr__(record, "_oh_permit_unregister", unregister)
    except (AttributeError, TypeError):
        pass  # Frozen record — skip.

    return record
