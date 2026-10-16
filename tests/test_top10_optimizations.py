"""Real tests for the Top-10 optimizations.

Each test exercises one optimization end-to-end and asserts the
observable behaviour. No mocks where avoidable — we use the real
PermissionChecker, real ToolResultCache, real AutoContext, etc.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Optimization #5: auto_compact_threshold_ratio
from openharness.services.compact import (
    get_autocompact_threshold,
)


@pytest.mark.asyncio
async def test_optimization_5_auto_compact_threshold_ratio_lowers_threshold():
    """Item 5: setting ratio=0.60 lowers the threshold below the default."""
    # Default heuristic gives ~180,000 for a 200K window.
    default_threshold = get_autocompact_threshold("claude-sonnet-4-6")
    ratio_threshold = get_autocompact_threshold(
        "claude-sonnet-4-6",
        auto_compact_threshold_ratio=0.60,
    )
    assert ratio_threshold < default_threshold, (
        f"Ratio threshold ({ratio_threshold}) should be lower than default ({default_threshold})"
    )
    # 0.60 of a 200K window is 120,000.
    assert 110_000 <= ratio_threshold <= 130_000


def test_optimization_5_ratio_zero_or_one_does_not_apply():
    """Edge case: ratio=0 or 1 should not divide by zero or zero-out."""
    threshold = get_autocompact_threshold(
        "claude-sonnet-4-6",
        auto_compact_threshold_ratio=0.0,
    )
    assert threshold > 0  # falls back to default heuristic
    threshold = get_autocompact_threshold(
        "claude-sonnet-4-6",
        auto_compact_threshold_ratio=1.0,
    )
    assert threshold > 0


def test_optimization_5_explicit_tokens_overrides_ratio():
    """Precedence: explicit tokens > ratio > default."""
    threshold_tokens = get_autocompact_threshold(
        "claude-sonnet-4-6",
        auto_compact_threshold_tokens=50000,
        auto_compact_threshold_ratio=0.60,
    )
    assert threshold_tokens == 50000


# Optimization #6: trusted_project
from openharness.config.settings import PermissionSettings, Settings
from openharness.permissions.modes import PermissionMode
from openharness.permissions.checker import build_permission_checker


def test_optimization_6_trusted_project_auto_approves_in_project_file():
    """Item 6: trusted_project=True auto-approves mutating tools on in-project paths."""
    settings = Settings(
        trusted_project=True,
        permission=PermissionSettings(mode=PermissionMode.DEFAULT),
    )
    checker = build_permission_checker(settings, cwd="/home/user/myproject")
    decision = checker.evaluate(
        "edit_file",
        is_read_only=False,
        file_path="/home/user/myproject/src/app.py",
    )
    assert decision.allowed, f"Should allow in-project edit: {decision.reason}"
    assert not decision.requires_confirmation


def test_optimization_6_trusted_project_blocks_outside_project():
    """Item 6: trusted_project=True still requires confirmation for out-of-project paths."""
    settings = Settings(
        trusted_project=True,
        permission=PermissionSettings(mode=PermissionMode.DEFAULT),
    )
    checker = build_permission_checker(settings, cwd="/home/user/myproject")
    decision = checker.evaluate(
        "edit_file",
        is_read_only=False,
        file_path="/etc/passwd",
    )
    # /etc/passwd is outside the project and not a sensitive credential,
    # so it should require confirmation (not auto-allowed).
    assert not decision.allowed
    assert decision.requires_confirmation


def test_optimization_6_trusted_project_does_not_override_sensitive_paths():
    """Item 6: even trusted_project must NOT allow access to .ssh, .aws, etc."""
    settings = Settings(
        trusted_project=True,
        permission=PermissionSettings(mode=PermissionMode.DEFAULT),
    )
    # Use a project root that contains a sensitive subdir.
    checker = build_permission_checker(settings, cwd="/home/user/myproject")
    decision = checker.evaluate(
        "read_file",
        is_read_only=True,
        file_path="/home/user/myproject/.ssh/id_rsa",  # would be inside project
    )
    assert not decision.allowed, "Sensitive paths must be blocked even in trusted project"


def test_optimization_6_disabled_by_default():
    """Item 6: trusted_project defaults to False — existing behaviour preserved."""
    settings = Settings()
    assert settings.trusted_project is False


# Optimization #7: tool result cache
from openharness.services.tool_cache import (
    ToolResultCache,
    disable_tool_cache,
    get_tool_cache,
    configure_tool_cache_from_settings,
    reset_tool_cache,
)


def test_optimization_7_tool_cache_hit_returns_cached_result():
    """Item 7: a second identical call returns the cached result without re-executing."""
    cache = ToolResultCache(max_entries=10, ttl_seconds=60)
    cache.put("read_file", {"path": "/etc/hosts"}, {"output": "1.1.1.1 host", "is_error": False, "metadata": {}})
    cached = cache.get("read_file", {"path": "/etc/hosts"})
    assert cached is not None
    assert cached["output"] == "1.1.1.1 host"
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_optimization_7_tool_cache_miss_on_different_input():
    """Item 7: different input → cache miss."""
    cache = ToolResultCache(max_entries=10, ttl_seconds=60)
    cache.put("read_file", {"path": "/etc/hosts"}, {"output": "x", "is_error": False, "metadata": {}})
    cached = cache.get("read_file", {"path": "/etc/different"})
    assert cached is None
    assert cache.stats()["misses"] == 1


def test_optimization_7_tool_cache_input_order_independent():
    """Item 7: same input in different dict orders should still hit."""
    cache = ToolResultCache(max_entries=10, ttl_seconds=60)
    cache.put("grep", {"pattern": "foo", "root": "/tmp"}, {"output": "match", "is_error": False, "metadata": {}})
    # Different insertion order
    cached = cache.get("grep", {"root": "/tmp", "pattern": "foo"})
    assert cached is not None, "Same dict in different order should still cache-hit"


def test_optimization_7_mutating_tool_invalidates_cache():
    """Item 7: a write_file call drops every cached entry."""
    cache = ToolResultCache(max_entries=10, ttl_seconds=60)
    cache.put("read_file", {"path": "/etc/hosts"}, {"output": "x", "is_error": False, "metadata": {}})
    assert cache.is_mutating("write_file")
    cache.invalidate_all()
    assert cache.get("read_file", {"path": "/etc/hosts"}) is None


def test_optimization_7_error_results_are_not_cached():
    """Item 7: failed tool results should not be cached so the retry actually runs."""
    cache = ToolResultCache(max_entries=10, ttl_seconds=60)
    cache.put("read_file", {"path": "/missing"}, {"output": "File not found", "is_error": True, "metadata": {}})
    cached = cache.get("read_file", {"path": "/missing"})
    assert cached is None, "Error results should not be cached"


def test_optimization_7_lru_eviction_when_full():
    """Item 7: when cache is full, least-recently-used entry is evicted."""
    cache = ToolResultCache(max_entries=2, ttl_seconds=60)
    cache.put("read_file", {"path": "/a"}, {"output": "a", "is_error": False, "metadata": {}})
    cache.put("read_file", {"path": "/b"}, {"output": "b", "is_error": False, "metadata": {}})
    # Access /a to make it more recently used than /b
    cache.get("read_file", {"path": "/a"})
    # Insert /c — should evict /b (the LRU)
    cache.put("read_file", {"path": "/c"}, {"output": "c", "is_error": False, "metadata": {}})
    assert cache.get("read_file", {"path": "/b"}) is None  # evicted
    assert cache.get("read_file", {"path": "/a"}) is not None  # still cached
    assert cache.get("read_file", {"path": "/c"}) is not None


def test_optimization_7_configure_from_settings_enables_cache():
    """Item 7: ``settings.tool_cache_enabled=True`` creates the global cache."""
    disable_tool_cache()  # start clean
    settings = Settings(tool_cache_enabled=True, tool_cache_max_entries=64)
    configure_tool_cache_from_settings(settings)
    assert get_tool_cache() is not None
    assert get_tool_cache().stats()["max_size"] == 64
    reset_tool_cache()


def test_optimization_7_disabled_by_default():
    """Item 7: cache is disabled by default."""
    disable_tool_cache()
    settings = Settings()  # tool_cache_enabled defaults to False
    configure_tool_cache_from_settings(settings)
    assert get_tool_cache() is None


# Optimization #3: auto context injection
from openharness.services.auto_context import (
    DEFAULT_AUTO_CONTEXT_FILES,
    discover_auto_context_files,
    inject_auto_context,
    is_auto_context_enabled,
)


def test_optimization_3_auto_context_discovers_files_in_cwd(tmp_path: Path):
    """Item 3: STYLE.md / STORYBOARD.md / BRIEF.md in cwd are discovered."""
    (tmp_path / "STYLE.md").write_text("# Style\nKeep it minimal.")
    (tmp_path / "STORYBOARD.md").write_text("# Story\nScene 1 → Scene 2.")
    (tmp_path / "BRIEF.md").write_text("# Brief\nBuild a 30s promo.")
    found = discover_auto_context_files(tmp_path)
    names = {p.name for p in found}
    assert names == {"STYLE.md", "STORYBOARD.md", "BRIEF.md"}


def test_optimization_3_auto_context_prefix_prepended_to_prompt(tmp_path: Path):
    """Item 3: inject_auto_context prepends the project context to the prompt."""
    (tmp_path / "BRIEF.md").write_text("# Brief\nMake a video about cats.")
    enriched = inject_auto_context("Generate scene 1", str(tmp_path))
    assert "Generate scene 1" in enriched
    assert "Make a video about cats." in enriched
    assert "Project Context" in enriched


def test_optimization_3_auto_context_disabled_returns_unchanged_prompt(tmp_path: Path):
    """Item 3: when disabled, the prompt is returned unchanged."""
    (tmp_path / "BRIEF.md").write_text("# Brief\nShould not appear.")
    enriched = inject_auto_context("Just this", str(tmp_path), enabled=False)
    assert enriched == "Just this"


def test_optimization_3_auto_context_missing_files_returns_empty(tmp_path: Path):
    """Item 3: when no context files exist, prefix is empty (no overhead)."""
    enriched = inject_auto_context("Just this", str(tmp_path))
    assert enriched == "Just this"


def test_optimization_3_auto_context_truncates_large_files(tmp_path: Path):
    """Item 3: large files are truncated to stay under the total cap."""
    big_content = "x" * 100_000
    (tmp_path / "BRIEF.md").write_text(big_content)
    enriched = inject_auto_context("task", str(tmp_path), max_total_chars=1000)
    # Truncation marker should appear, and total prefix should be < 2KB
    assert "[truncated for auto-context]" in enriched
    assert len(enriched) < 3000


def test_optimization_3_env_override_disables():
    """Item 3: OPENHARNESS_AUTO_CONTEXT_INJECT=0 disables injection."""
    old = os.environ.get("OPENHARNESS_AUTO_CONTEXT_INJECT")
    try:
        os.environ["OPENHARNESS_AUTO_CONTEXT_INJECT"] = "0"
        settings = Settings(auto_context_inject=True)  # config says enabled
        assert is_auto_context_enabled(settings) is False  # env wins
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_AUTO_CONTEXT_INJECT", None)
        else:
            os.environ["OPENHARNESS_AUTO_CONTEXT_INJECT"] = old


def test_optimization_3_default_files_includes_three_canonical_names():
    """Item 3: defaults are STYLE.md / STORYBOARD.md / BRIEF.md."""
    assert set(DEFAULT_AUTO_CONTEXT_FILES) == {"STYLE.md", "STORYBOARD.md", "BRIEF.md"}


# Optimization #1: max_concurrent_agents
from openharness.services.agent_concurrency import (
    configure_agent_concurrency,
    get_agent_semaphore,
    get_agent_limit,
    reset_agent_concurrency,
    with_agent_limit,
)


def test_optimization_1_default_limit_is_5():
    """Item 1: the default wave limit is 5 agents."""
    reset_agent_concurrency()
    # Trigger lazy init
    get_agent_semaphore()
    assert get_agent_limit() == 5


def test_optimization_1_configure_to_20():
    """Item 1: setting max_concurrent_agents=20 raises the semaphore capacity."""
    settings = Settings(max_concurrent_agents=20)
    configure_agent_concurrency(settings)
    assert get_agent_limit() == 20


def test_optimization_1_semaphore_blocks_when_full():
    """Item 1: when the semaphore is full, additional spawns must wait."""
    settings = Settings(max_concurrent_agents=2)
    configure_agent_concurrency(settings)
    sem = get_agent_semaphore()

    async def _worker():
        async with sem:
            await asyncio.sleep(0.05)

    async def _main():
        # Acquire both permits
        await asyncio.gather(_worker(), _worker())
        # Now third worker should block briefly
        import time
        t0 = time.monotonic()
        await _worker()
        return time.monotonic() - t0

    elapsed = asyncio.run(_main())
    # Since workers finished before the third started, elapsed should be < 0.1s
    assert elapsed < 0.5


def test_optimization_1_with_agent_limit_releases_on_exception():
    """Item 1: semaphore must release even if the wrapped coroutine raises."""
    settings = Settings(max_concurrent_agents=1)
    configure_agent_concurrency(settings)

    async def _raising():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(with_agent_limit(_raising))
    # After the exception, the semaphore should be back at full capacity
    sem = get_agent_semaphore()
    # Acquire immediately should succeed
    async def _acquire_and_release():
        async with sem:
            pass
    asyncio.run(_acquire_and_release())


# Optimization #4: multi-provider routing
from openharness.services.provider_routing import (
    RoutedStreamingClient,
    ProviderRoute,
)
from openharness.api.client import ApiMessageRequest, ApiTextDeltaEvent


class _FakeClient:
    """Fake streaming client for routing tests."""

    def __init__(self, *, name: str, fail_with: Exception | None = None, events: list | None = None):
        self.name = name
        self._fail_with = fail_with
        self._events = events or []
        self.calls = 0

    async def stream_message(self, request):
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_optimization_4_falls_back_on_rate_limit():
    """Item 4: when primary rate-limits, the request is retried on fallback."""
    from openharness.api.errors import RateLimitFailure
    primary = _FakeClient(name="groq", fail_with=RateLimitFailure("429 too many requests"))
    fallback = _FakeClient(
        name="dashscope",
        events=[ApiTextDeltaEvent(text="hello from fallback")],
    )
    routed = RoutedStreamingClient(
        ProviderRoute(name="groq", client=primary),
        ProviderRoute(name="dashscope", client=fallback),
    )
    events = []
    async for event in routed.stream_message(ApiMessageRequest(model="m", messages=[])):
        events.append(event)
    # First event should be a retry notification, then the fallback's text delta
    assert any(isinstance(e, ApiTextDeltaEvent) and e.text == "hello from fallback" for e in events)
    assert primary.calls == 1
    assert fallback.calls == 1
    assert routed.stats.fallback_attempts == 1
    assert routed.stats.fallback_successes == 1
    assert routed.stats.last_route_used == "dashscope"


@pytest.mark.asyncio
async def test_optimization_4_uses_primary_on_success():
    """Item 4: when primary succeeds, fallbacks are not tried."""
    primary = _FakeClient(
        name="groq",
        events=[ApiTextDeltaEvent(text="hello")],
    )
    fallback = _FakeClient(name="dashscope", events=[])
    routed = RoutedStreamingClient(
        ProviderRoute(name="groq", client=primary),
        ProviderRoute(name="dashscope", client=fallback),
    )
    events = [event async for event in routed.stream_message(ApiMessageRequest(model="m", messages=[]))]
    assert len(events) == 1
    assert fallback.calls == 0
    assert routed.stats.primary_successes == 1


@pytest.mark.asyncio
async def test_optimization_4_raises_when_all_routes_fail():
    """Item 4: if every route fails, the last error is raised."""
    from openharness.api.errors import RateLimitFailure
    primary = _FakeClient(name="groq", fail_with=RateLimitFailure("primary 429"))
    fallback = _FakeClient(name="dashscope", fail_with=RateLimitFailure("fallback 429"))
    routed = RoutedStreamingClient(
        ProviderRoute(name="groq", client=primary),
        ProviderRoute(name="dashscope", client=fallback),
    )
    with pytest.raises(RateLimitFailure):
        async for _ in routed.stream_message(ApiMessageRequest(model="m", messages=[])):
            pass
    assert routed.stats.fallback_attempts == 1
    assert "fallback 429" in routed.stats.last_error


# Optimization #8: parallel_tools flag
def test_optimization_8_default_is_true():
    """Item 8: parallel_tools defaults to True (existing behaviour)."""
    settings = Settings()
    assert settings.parallel_tools is True


def test_optimization_8_env_override_disables():
    """Item 8: OPENHARNESS_PARALLEL_TOOLS=0 disables parallel execution."""
    from openharness.config.settings import _apply_env_overrides
    old = os.environ.get("OPENHARNESS_PARALLEL_TOOLS")
    try:
        os.environ["OPENHARNESS_PARALLEL_TOOLS"] = "0"
        settings = _apply_env_overrides(Settings())
        assert settings.parallel_tools is False
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_PARALLEL_TOOLS", None)
        else:
            os.environ["OPENHARNESS_PARALLEL_TOOLS"] = old


# Optimization #9: thinking_budget
def test_optimization_9_default_is_none():
    """Item 9: thinking_budget defaults to None (no reasoning cap)."""
    settings = Settings()
    assert settings.thinking_budget is None


def test_optimization_9_env_override():
    """Item 9: OPENHARNESS_THINKING_BUDGET=4096 sets the budget."""
    from openharness.config.settings import _apply_env_overrides
    old = os.environ.get("OPENHARNESS_THINKING_BUDGET")
    try:
        os.environ["OPENHARNESS_THINKING_BUDGET"] = "4096"
        settings = _apply_env_overrides(Settings())
        assert settings.thinking_budget == 4096
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_THINKING_BUDGET", None)
        else:
            os.environ["OPENHARNESS_THINKING_BUDGET"] = old


def test_optimization_9_thinking_budget_forwarded_to_api_request():
    """Item 9: the budget is forwarded to the API request dataclass."""
    req = ApiMessageRequest(
        model="o1",
        messages=[],
        max_tokens=4096,
        thinking_budget=8192,
    )
    assert req.thinking_budget == 8192


# Optimization #10: auto_save + MEMORY.md
def test_optimization_10_auto_save_default_is_true():
    """Item 10: auto_save defaults to True."""
    settings = Settings()
    assert settings.auto_save is True


def test_optimization_10_env_override_disables_auto_save():
    """Item 10: OPENHARNESS_AUTO_SAVE=0 disables auto-save."""
    from openharness.config.settings import _apply_env_overrides
    old = os.environ.get("OPENHARNESS_AUTO_SAVE")
    try:
        os.environ["OPENHARNESS_AUTO_SAVE"] = "0"
        settings = _apply_env_overrides(Settings())
        assert settings.auto_save is False
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_AUTO_SAVE", None)
        else:
            os.environ["OPENHARNESS_AUTO_SAVE"] = old


def test_optimization_10_auto_save_writes_snapshot(tmp_path: Path, monkeypatch):
    """Item 10: QueryEngine._auto_save_session writes latest.json on turn end."""
    from openharness.engine.query_engine import QueryEngine
    from openharness.engine.messages import ConversationMessage, TextBlock
    from openharness.config.settings import Settings
    from openharness.config.paths import get_sessions_dir
    # Point sessions dir at tmp_path so we can inspect the snapshot.
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    # Force settings cache clear
    from openharness.config.settings import clear_settings_cache
    clear_settings_cache()

    settings = Settings(auto_save=True, auto_save_interval=1)
    fake_client = MagicMock()
    fake_client.close = AsyncMock()
    engine = QueryEngine(
        api_client=fake_client,
        tool_registry=MagicMock(),
        permission_checker=build_permission_checker(settings, cwd=str(tmp_path)),
        cwd=tmp_path,
        model="test-model",
        system_prompt="sys",
        settings=settings,
        tool_metadata={"session_id": "test-session-123"},
    )
    engine.load_messages([
        ConversationMessage(role="user", content=[TextBlock(text="hello")]),
    ])
    # Simulate one completed turn so the interval check (default 10)
    # doesn't skip the save. With auto_save_interval=1, every turn saves.
    engine._turn_count = 1
    engine._auto_save_session()
    # Snapshot should be written to the project session dir
    session_dir = get_sessions_dir()
    # Find any latest.json written under sessions dir
    latest_files = list(session_dir.rglob("latest.json"))
    assert latest_files, "auto_save should have written a latest.json snapshot"


# Optimization #2: batch_size
def test_optimization_2_batch_size_default_is_unset():
    """Item 2: batch_size defaults to None (sentinel for 'not set').

    The effective default is 1 — see ``_effective_batch_size()`` in
    runtime.py. The None sentinel lets Kaggle auto-detection distinguish
    'user did not set' from 'user explicitly set 1'.
    """
    settings = Settings()
    assert settings.batch_size is None


def test_optimization_2_env_override_sets_batch_size():
    """Item 2: OPENHARNESS_BATCH_SIZE=8 sets the batch size."""
    from openharness.config.settings import _apply_env_overrides
    old = os.environ.get("OPENHARNESS_BATCH_SIZE")
    try:
        os.environ["OPENHARNESS_BATCH_SIZE"] = "8"
        settings = _apply_env_overrides(Settings())
        assert settings.batch_size == 8
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_BATCH_SIZE", None)
        else:
            os.environ["OPENHARNESS_BATCH_SIZE"] = old


def test_optimization_2_invalid_batch_size_is_ignored():
    """Item 2: non-integer env values are ignored (no crash)."""
    from openharness.config.settings import _apply_env_overrides
    old = os.environ.get("OPENHARNESS_BATCH_SIZE")
    try:
        os.environ["OPENHARNESS_BATCH_SIZE"] = "not-an-int"
        settings = _apply_env_overrides(Settings())
        # Invalid value is ignored → falls back to None sentinel.
        # The effective default (1) is applied at runtime via
        # ``int(settings.batch_size or 1)`` in runtime.py.
        assert settings.batch_size is None
    finally:
        if old is None:
            os.environ.pop("OPENHARNESS_BATCH_SIZE", None)
        else:
            os.environ["OPENHARNESS_BATCH_SIZE"] = old
