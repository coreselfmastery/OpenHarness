"""Concurrency limiter for spawned agents.

Implements optimization #1 from the top-10 list:
"Increase wave limit to 15-20 concurrent agents".

The historical default was 5 concurrent agents (implicit, via OS process
limits). This module provides a configurable ``asyncio.Semaphore`` that
the agent-spawn path acquires before starting a new subprocess. When the
limit is reached, additional spawn requests await an in-flight agent to
finish, instead of overwhelming the host.

The limit is read from ``settings.max_concurrent_agents`` (default 5).
Set it higher (15-20) for parallel-scene / batch-video workflows where
each agent is mostly I/O-bound waiting on its own LLM API calls.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

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


async def with_agent_limit(coro_factory):
    """Acquire the agent semaphore, run ``coro_factory()``, release.

    ``coro_factory`` is a zero-arg callable that returns a coroutine.
    Used as a context manager around the actual spawn call so the
    semaphore is always released even on exception.
    """
    sem = get_agent_semaphore()
    async with sem:
        return await coro_factory()
