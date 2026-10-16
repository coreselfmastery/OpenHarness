"""Shared test fixtures."""

from __future__ import annotations

import pytest
import pytest_asyncio

from openharness.tasks.manager import shutdown_task_manager


@pytest_asyncio.fixture(autouse=True)
async def _reset_background_task_manager():
    yield
    await shutdown_task_manager()


@pytest.fixture(autouse=True)
def _reset_perf_caches():
    """Clear performance caches that persist across tests.

    Several modules cache ``shutil.which`` lookups, env-var-derived
    constants, and settings to avoid repeated work on the hot path.
    When tests monkeypatch ``shutil.which`` or set env vars after import,
    those caches would serve stale values. This fixture clears them
    before each test so tests start from a clean state.
    """
    # Settings cache (load_settings caches by file mtime + env fingerprint)
    try:
        from openharness.config.settings import clear_settings_cache
        clear_settings_cache()
    except ImportError:
        pass

    # tool_outputs env-cache
    try:
        from openharness.services.tool_outputs import _reset_env_cache
        _reset_env_cache()
    except ImportError:
        pass

    # Cached binary lookups (shell.py uses lru_cache for shutil.which)
    try:
        from openharness.utils.shell import _which_cache_clear
        _which_cache_clear()
    except ImportError:
        pass
    # glob_tool's git-repo check cache
    try:
        from openharness.tools.glob_tool import _GIT_REPO_CACHE
        _GIT_REPO_CACHE.clear()
    except ImportError:
        pass
    # Optimization #1: reset agent concurrency semaphore between tests
    try:
        from openharness.services.agent_concurrency import reset_agent_concurrency
        reset_agent_concurrency()
    except ImportError:
        pass
    # Optimization #7: reset the tool result cache between tests
    try:
        from openharness.services.tool_cache import disable_tool_cache, reset_tool_cache
        reset_tool_cache()
        disable_tool_cache()
    except ImportError:
        pass
    # Speed wave 2: hermetic cost dashboard + batch executor. The
    # dashboard singleton would otherwise (a) persist aggregates across
    # tests and (b) lazily open the REAL ~/.openharness SQLite store when
    # a test wraps a client. Install a no-store dashboard per test;
    # tests that want their own dashboard call set_cost_dashboard
    # explicitly (overriding this one).
    try:
        from openharness.services.cost_dashboard import CostDashboard, set_cost_dashboard
        set_cost_dashboard(CostDashboard(store=None))
    except ImportError:
        pass
    # Batch executor: reset so a previous test's concurrency setting
    # cannot leak into the next test's fan-out behavior.
    try:
        from openharness.services.batch_executor import set_batch_executor
        set_batch_executor(None)
    except ImportError:
        pass

    yield

    # Also clear after the test so the next test starts clean even if
    # something was cached during the test.
    try:
        from openharness.config.settings import clear_settings_cache
        clear_settings_cache()
    except ImportError:
        pass
    try:
        from openharness.services.tool_outputs import _reset_env_cache
        _reset_env_cache()
    except ImportError:
        pass
    try:
        from openharness.utils.shell import _which_cache_clear
        _which_cache_clear()
    except ImportError:
        pass
    try:
        from openharness.tools.glob_tool import _GIT_REPO_CACHE
        _GIT_REPO_CACHE.clear()
    except ImportError:
        pass
    try:
        from openharness.services.agent_concurrency import reset_agent_concurrency
        reset_agent_concurrency()
    except ImportError:
        pass
    try:
        from openharness.services.tool_cache import disable_tool_cache, reset_tool_cache
        reset_tool_cache()
        disable_tool_cache()
    except ImportError:
        pass
    try:
        # Drop the per-test dashboard (flushing anything it collected to
        # nowhere — it has no store) and clear the executor singleton.
        from openharness.services.cost_dashboard import set_cost_dashboard
        set_cost_dashboard(None)
    except ImportError:
        pass
    try:
        from openharness.services.batch_executor import set_batch_executor
        set_batch_executor(None)
    except ImportError:
        pass
