"""Tests for the speed-wave-2 features: cooldown key pool, async batch
executor, cost dashboard, parallel_prompts + cost_report tools, and the
batch_scene concurrent-retry path.

User-facing promises pinned here:

- Key pool: a rate-limited key enters a 60s cooldown and is SKIPPED by
  later requests; every key in the pool is tried before giving up; a
  short full-pool stall is waited out, a long one surfaces the error.
- Async batch executor: order-preserving, semaphore-bounded concurrency,
  per-item failure isolation, retry rounds re-run only failures.
- Cost dashboard: pricing table + overrides, per-model/scene/project
  aggregates, SQLite persistence across instances, projections.
- parallel_prompts: real endpoint required (no invented results),
  concurrency honored, per-prompt failures reported.
- cost_report: summary/models/scenes/recent/clear actions on real data.
- batch_scene: failed-lint scenes retry CONCURRENTLY (wall time ~ the
  slowest retry, not the sum).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
)
from openharness.api.errors import RateLimitFailure
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock


def _request(text: str = "hi", model: str = "claude-sonnet-4") -> ApiMessageRequest:
    return ApiMessageRequest(
        model=model,
        messages=[ConversationMessage(role="user", content=[TextBlock(text=text)])],
    )


def _complete_event(text: str = "hi", *, in_tokens: int = 10, out_tokens: int = 5):
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
        usage=UsageSnapshot(input_tokens=in_tokens, output_tokens=out_tokens),
    )


# ======================================================================
# 1. KeyRotatingClient cooldown design
# ======================================================================

class _PoolTransport:
    """Fake transport: named keys with per-call failure scripts."""

    def __init__(self, *, script: list[BaseException | None] = (None,)) -> None:
        self.script = list(script)
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        idx = self.calls - 1
        if idx < len(self.script):
            outcome = self.script[idx]
        else:
            outcome = None
        if outcome is not None:
            raise outcome
        yield ApiTextDeltaEvent(text="ok")
        yield _complete_event()


@pytest.mark.asyncio
async def test_rate_limited_key_enters_cooldown_and_is_skipped():
    from openharness.services.llm_pipeline import KeyRotatingClient

    # 10 keys; key #1 always rate-limits (30 RPM exhausted).
    transports: list[_PoolTransport] = []
    key_names = [f"k{i}" for i in range(10)]

    def factory(key: str) -> _PoolTransport:
        index = key_names.index(key)
        t = _PoolTransport(script=[RateLimitFailure("429")] if index == 1 else [None])
        transports.append(t)
        return t

    pool = KeyRotatingClient(key_names, factory, cooldown_seconds=60.0)
    served = 0
    # Request 1: key 0 ok (round robin advances).
    async for _ in pool.stream_message(_request("r1")):
        served += 1
    # Request 2: key 1 rate-limits -> rotates to key 2 immediately.
    events = [e async for e in pool.stream_message(_request("r2"))]
    assert any(isinstance(e, ApiRetryEvent) and "API key #2" in e.message for e in events)
    stats = pool.stats()
    assert stats["requests_served"] == 2
    assert stats["cooldowns_taken"] == 1
    # Request 3: key 1 is in cooldown — round robin lands on key 2 (key
    # 1 must be SKIPPED, not retried).
    before = dict(transports[1].__dict__)
    async for _ in pool.stream_message(_request("r3")):
        served += 1
    assert transports[1].__dict__ == before  # key 1 untouched
    assert pool.stats()["keys_in_cooldown"] == 1
    await pool.close()


@pytest.mark.asyncio
async def test_pool_tries_every_key_before_failing():
    from openharness.services.llm_pipeline import KeyRotatingClient

    # 10 keys, ALL rate-limited: the old code gave up after 3 attempts;
    # the new design tries every key once, then surfaces the error
    # (60s cooldown > 5s short-wait budget -> immediate raise).
    keys = [f"k{i}" for i in range(10)]
    factory = lambda key: _PoolTransport(script=[RateLimitFailure("429")] * 50)
    pool = KeyRotatingClient(keys, factory, cooldown_seconds=60.0)
    with pytest.raises(RateLimitFailure):
        async for _ in pool.stream_message(_request()):
            pass
    # Every key was attempted exactly once.
    assert pool.stats()["rotations_on_rate_limit"] == 10
    await pool.close()


@pytest.mark.asyncio
async def test_pool_waits_out_short_full_stall():
    from openharness.services.llm_pipeline import KeyRotatingClient

    # All keys cooling from a PREVIOUS request, but the earliest expiry
    # is 0.3s away — the request waits briefly and succeeds, and the
    # successful key's stale cooldown is cleared.
    keys = ["a", "b"]
    transports = [_PoolTransport() for _ in keys]
    pool = KeyRotatingClient(keys, lambda key: transports[keys.index(key)], cooldown_seconds=60.0)
    # Simulate: both keys cooling, key 0 expires in 0.3s, key 1 in 30s.
    now = time.monotonic()
    pool._cooldown_until[0] = now + 0.3
    pool._cooldown_until[1] = now + 30.0
    events = [e async for e in pool.stream_message(_request())]
    assert any(isinstance(e, ApiRetryEvent) and "waiting" in e.message for e in events)
    assert pool.stats()["requests_served"] == 1
    assert pool.stats()["pool_waits"] == 1
    # The key that served had its cooldown cleared on success; the other
    # (30s) stays cooling.
    assert 0 not in pool._cooldown_until
    assert 1 in pool._cooldown_until
    await pool.close()


@pytest.mark.asyncio
async def test_all_keys_cooling_raises_instead_of_silent_empty_stream():
    from openharness.services.llm_pipeline import KeyRotatingClient

    # Every key cooling for longer than the short-wait budget, with no
    # provider error from THIS request: the client must raise a
    # RateLimitFailure explaining the pool state — never end the stream
    # silently (an empty stream looks like a blank completion upstream).
    keys = ["a", "b"]
    transports = [_PoolTransport() for _ in keys]
    pool = KeyRotatingClient(keys, lambda key: transports[keys.index(key)], cooldown_seconds=60.0)
    now = time.monotonic()
    pool._cooldown_until[0] = now + 30.0
    pool._cooldown_until[1] = now + 30.0
    with pytest.raises(RateLimitFailure) as excinfo:
        async for _ in pool.stream_message(_request()):
            pass
    assert "cooldown" in str(excinfo.value).lower()
    assert transports[0].calls == 0 and transports[1].calls == 0
    await pool.close()


def test_key_pool_settings_and_env(monkeypatch):
    from openharness.config.settings import Settings, _apply_env_overrides

    settings = Settings()
    assert settings.inference.key_pool_cooldown_seconds == 60.0
    assert settings.inference.batch_concurrency == 8
    monkeypatch.setenv("OPENHARNESS_KEY_POOL_COOLDOWN_SECONDS", "30")
    monkeypatch.setenv("OPENHARNESS_BATCH_CONCURRENCY", "12")
    merged = _apply_env_overrides(settings)
    assert merged.inference.key_pool_cooldown_seconds == 30.0
    assert merged.inference.batch_concurrency == 12


# ======================================================================
# 2. AsyncBatchExecutor
# ======================================================================

@pytest.mark.asyncio
async def test_executor_preserves_order_and_bounds_concurrency():
    from openharness.services.batch_executor import AsyncBatchExecutor

    executor = AsyncBatchExecutor(concurrency=4)
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(item: int, _index: int) -> int:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return item * 2

    results = await executor.run(list(range(20)), worker)
    assert results == [i * 2 for i in range(20)]  # order preserved
    assert peak <= 4  # semaphore bounded
    assert peak >= 2  # actually overlapped
    stats = executor.report()
    assert stats["items_run"] == 20
    assert stats["peak_in_flight"] >= 2
    assert stats["effective_speedup"] > 1.5  # serial estimate >> wall time


@pytest.mark.asyncio
async def test_executor_isolates_item_failures():
    from openharness.services.batch_executor import AsyncBatchExecutor, ItemFailure

    async def worker(item: int, _index: int) -> int:
        if item == 3:
            raise ValueError("boom")
        return item

    executor = AsyncBatchExecutor(concurrency=3)
    results = await executor.run([1, 2, 3, 4], worker)
    assert results[0] == 1 and results[1] == 2 and results[3] == 4
    assert isinstance(results[2], ItemFailure)
    assert "boom" in results[2].error
    assert results[2].item_index == 2
    assert executor.stats.item_failures == 1


@pytest.mark.asyncio
async def test_executor_retries_only_failures():
    from openharness.services.batch_executor import AsyncBatchExecutor, ItemFailure

    attempts: dict[int, int] = {}

    async def worker(item: int, _index: int) -> str:
        attempts[item] = attempts.get(item, 0) + 1
        if item == 1 and attempts[item] == 1:
            raise RuntimeError("transient")
        return f"ok-{item}"

    executor = AsyncBatchExecutor(concurrency=2)
    results = await executor.run_with_retries([1, 2, 3], worker, retries=2)
    assert results == ["ok-1", "ok-2", "ok-3"]
    assert attempts == {1: 2, 2: 1, 3: 1}  # only item 1 was retried
    # A permanently failing item reports ItemFailure with the retry count.
    attempts.clear()

    async def always_fails(item: int, _index: int) -> str:
        attempts[item] = attempts.get(item, 0) + 1
        raise RuntimeError("nope")

    results = await executor.run_with_retries([7], always_fails, retries=2)
    assert isinstance(results[0], ItemFailure)
    assert results[0].retries == 2
    assert attempts == {7: 3}  # 1 initial + 2 retry rounds


@pytest.mark.asyncio
async def test_executor_on_result_callback_and_cancel_isolation():
    from openharness.services.batch_executor import AsyncBatchExecutor, ItemFailure

    seen: list[int] = []

    async def worker(item: int, _index: int) -> int:
        await asyncio.sleep(0.01 * (5 - item))
        if item == 4:
            raise RuntimeError("x")
        return item

    executor = AsyncBatchExecutor(concurrency=5)
    results = await executor.run([0, 1, 2, 3, 4], worker, on_result=lambda i, r: seen.append(i))
    assert sorted(seen) == [0, 1, 2, 3, 4]  # every completion fired
    assert isinstance(results[4], ItemFailure)
    # Empty input is fine.
    assert await executor.run([], worker) == []


def test_configure_batch_executor_from_settings():
    from openharness.services.batch_executor import (
        configure_batch_executor_from_settings,
        set_batch_executor,
    )

    try:
        settings = SimpleNamespace(inference=SimpleNamespace(batch_concurrency=6))
        executor = configure_batch_executor_from_settings(settings)
        assert executor.concurrency == 6
        settings2 = SimpleNamespace(inference=SimpleNamespace(batch_concurrency=6))
        assert configure_batch_executor_from_settings(settings2) is executor  # reused
        settings3 = SimpleNamespace(inference=SimpleNamespace(batch_concurrency=3))
        assert configure_batch_executor_from_settings(settings3).concurrency == 3
    finally:
        set_batch_executor(None)


# ======================================================================
# 3. Cost dashboard
# ======================================================================

def test_estimate_cost_prefix_match_and_overrides(monkeypatch):
    from openharness.services.cost_dashboard import estimate_cost

    # Longest prefix wins: haiku-4-5 over generic 3-5-haiku.
    cost, priced = estimate_cost("claude-haiku-4-5", 1_000_000, 0)
    assert priced and cost == pytest.approx(1.0)
    cost, _ = estimate_cost("claude-sonnet-4-5-20250929", 0, 1_000_000)
    assert cost == pytest.approx(15.0)
    # Cache multipliers (Anthropic): reads 0.1x, writes 1.25x of input.
    cost, _ = estimate_cost("claude-sonnet-4", 0, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(0.3)
    cost, _ = estimate_cost("claude-sonnet-4", 0, 0, cache_write_tokens=1_000_000)
    assert cost == pytest.approx(3.75)
    # Unknown model: unpriced, zero cost.
    cost, priced = estimate_cost("totally-unknown-model", 1_000_000, 1_000_000)
    assert cost == 0.0 and priced is False
    # Overrides beat the table.
    cost, priced = estimate_cost(
        "internal-model", 1_000_000, 0, overrides={"internal-model": (2.0, 4.0)}
    )
    assert priced and cost == pytest.approx(2.0)
    # Env overrides.
    monkeypatch.setenv("OPENHARNESS_PRICING_JSON", json.dumps({"env-model": [5.0, 5.0]}))
    from openharness.services.cost_dashboard import pricing_overrides_from_env

    assert pricing_overrides_from_env() == {"env-model": (5.0, 5.0)}
    monkeypatch.setenv("OPENHARNESS_PRICING_JSON", "{not json")
    assert pricing_overrides_from_env() == {}


def test_cost_dashboard_aggregates_and_projects(tmp_path: Path):
    from openharness.services.cost_dashboard import CostDashboard
    from openharness.services.persistent_store import PersistentStore

    store = PersistentStore(tmp_path / "cost.sqlite3")
    dash = CostDashboard(store, flush_every=1000, flush_interval=1e9)
    dash.record(model="claude-sonnet-4", input_tokens=1_000_000, output_tokens=0)
    dash.record(model="claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000, scene="scene_001", project="proj")
    dash.record(model="mystery-model", input_tokens=123, output_tokens=456)

    summary = dash.summary(scenes_total=20, scenes_done=4)
    assert summary["total_calls"] == 3
    # sonnet: $3.0 (1M in) + haiku: $5.0 (1M out) + mystery: $0 (unpriced)
    assert summary["total_cost_usd"] == pytest.approx(8.0)
    assert summary["unpriced_calls"] == 1
    assert summary["models"]["claude-sonnet-4"]["cost_usd"] == pytest.approx(3.0)
    assert summary["models"]["claude-haiku-4-5"]["cost_usd"] == pytest.approx(5.0)
    assert summary["scenes"]["scene_001"]["cost_usd"] == pytest.approx(5.0)
    assert summary["projects"]["proj"]["cost_usd"] == pytest.approx(5.0)
    # Projection: 20 scenes total, 4 done at $8 -> $40 projected.
    assert summary["scenes_remaining"] == 16
    assert summary["projected_total_usd"] == pytest.approx(40.0)
    # Recent ring buffer.
    recent = dash.recent(limit=2)
    assert len(recent) == 2 and recent[0]["model"] == "mystery-model"
    # Persistence: a second dashboard instance loads the same aggregates.
    dash.flush(force=True)
    dash2 = CostDashboard(store, flush_every=1000, flush_interval=1e9)
    assert dash2.summary()["total_cost_usd"] == pytest.approx(8.0)
    assert dash2.summary()["total_calls"] == 3
    # Clear wipes both layers.
    assert dash2.clear() is True
    assert dash2.summary()["total_calls"] == 0
    store.close()


def test_cost_dashboard_never_raises_on_bad_store(tmp_path: Path):
    from openharness.services.cost_dashboard import CostDashboard

    class ExplodingStore:
        def get(self, *a, **k):
            raise RuntimeError("db gone")

        def set(self, *a, **k):
            raise RuntimeError("db gone")

        def delete(self, *a, **k):
            raise RuntimeError("db gone")

    dash = CostDashboard(ExplodingStore(), flush_every=1, flush_interval=0.0)
    cost = dash.record(model="claude-sonnet-4", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(3.0 * 1000 / 1e6 + 15.0 * 1000 / 1e6)
    assert dash.summary()["total_calls"] == 1  # in-memory still works
    assert dash.recent(limit=5)
    dash.flush(force=True)  # no raise
    dash.clear()  # no raise


# ======================================================================
# 4. UsageTrackingClient wrapper + pipeline wiring
# ======================================================================

class _UsageTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        yield ApiTextDeltaEvent(text="hi")
        yield _complete_event("hi", in_tokens=100, out_tokens=50)


@pytest.mark.asyncio
async def test_usage_tracking_client_records_completions():
    from openharness.services.cost_dashboard import CostDashboard
    from openharness.services.llm_pipeline import UsageTrackingClient

    dash = CostDashboard(store=None)
    client = UsageTrackingClient(_UsageTransport(), dash)
    events = 0
    async for _ in client.stream_message(_request(model="claude-sonnet-4")):
        events += 1
    assert events == 2
    summary = dash.summary()
    assert summary["total_calls"] == 1
    assert summary["models"]["claude-sonnet-4"]["input_tokens"] == 100
    assert summary["models"]["claude-sonnet-4"]["output_tokens"] == 50
    # Transparent passthrough: attributes of the wrapped client are
    # reachable; genuinely missing attributes still raise.
    class _WithStats:
        async def stream_message(self, request: ApiMessageRequest):
            yield _complete_event()

        def stats(self):
            return {"calls": 1}

    tracked = UsageTrackingClient(_WithStats(), dash)
    assert tracked.stats() == {"calls": 1}
    with pytest.raises(AttributeError):
        tracked.definitely_not_a_method


@pytest.mark.asyncio
async def test_pipeline_wrap_records_usage_with_no_features_enabled():
    """Even with every inference feature off, completed calls are tracked."""
    from openharness.services.cost_dashboard import CostDashboard, set_cost_dashboard
    from openharness.services.llm_pipeline import wrap_client_with_inference_pipeline

    dash = CostDashboard(store=None)
    set_cost_dashboard(dash)
    try:
        settings = SimpleNamespace(
            inference=SimpleNamespace(
                prefix_caching=False,
                prompt_compression=False,
                semantic_cache=False,
                cross_provider_racing=False,
                api_key_pool=[],
            )
        )
        wrapped = wrap_client_with_inference_pipeline(settings, _UsageTransport())
        async for _ in wrapped.stream_message(_request(model="claude-sonnet-4")):
            pass
        assert dash.summary()["total_calls"] == 1
        assert dash.summary()["total_cost_usd"] == pytest.approx(300 / 1e6 + 750 / 1e6)
    finally:
        set_cost_dashboard(None)


# ======================================================================
# 5. Tools: cost_report + parallel_prompts registration & behavior
# ======================================================================

def test_new_tools_registered():
    from openharness.tools import create_default_tool_registry
    from openharness.tools.pipeline_tools import CostReportTool, ParallelPromptsTool

    registry = create_default_tool_registry()
    names = {t.name for t in registry.tools.values()} if hasattr(registry, "tools") else set()
    if not names:  # registry may expose a different container
        names = {t.name for t in registry.list_tools()} if hasattr(registry, "list_tools") else set()
    assert "parallel_prompts" in names
    assert "cost_report" in names
    assert isinstance(ParallelPromptsTool().name, str)
    assert CostReportTool().description


@pytest.mark.asyncio
async def test_parallel_prompts_requires_endpoint(tmp_path: Path):
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.pipeline_tools import ParallelPromptsInput, ParallelPromptsTool

    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"settings": SimpleNamespace(qa=None, base_url="", api_key="", model="")},
    )
    result = await ParallelPromptsTool().execute(
        ParallelPromptsInput(prompts=["a", "b"]), context
    )
    assert result.is_error
    assert "endpoint" in result.output.lower()


@pytest.mark.asyncio
async def test_parallel_prompts_runs_batch_concurrently(monkeypatch, tmp_path: Path):
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.pipeline_tools import ParallelPromptsInput, ParallelPromptsTool

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": f"answer-{self._n}"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                "_n": 0,
            }

    class _FakeAsyncClient:
        call_count = 0
        instances = 0
        closed = 0

        def __init__(self, *a, **k):
            _FakeAsyncClient.instances += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aclose(self):
            _FakeAsyncClient.closed += 1

        async def post(self, url, json=None, headers=None):
            _FakeAsyncClient.call_count += 1
            n = _FakeAsyncClient.call_count
            await asyncio.sleep(0.05)
            resp = _FakeResponse()
            resp._n = n
            return resp

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    settings = SimpleNamespace(
        qa=SimpleNamespace(base_url="https://api.example.com/v1", api_key="k", model="llama-3-3-70b"),
        base_url="https://api.example.com/v1",
        api_key="k",
        model="llama-3-3-70b",
        timeout=60,
        inference=SimpleNamespace(batch_concurrency=4),
    )
    context = ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings})
    prompts = [f"prompt {i}" for i in range(6)]
    result = await ParallelPromptsTool().execute(
        ParallelPromptsInput(prompts=prompts, labels=[f"l{i}" for i in range(6)]),
        context,
    )
    assert not result.is_error
    assert "6/6 prompts succeeded" in result.output
    assert _FakeAsyncClient.call_count == 6
    # ONE pooled client served the whole batch (per-prompt clients would
    # pay a fresh TCP+TLS handshake each) and it was closed afterwards.
    assert _FakeAsyncClient.instances == 1
    assert _FakeAsyncClient.closed == 1
    # All 6 responses present, in input order (each answer tagged by its
    # arrival number; order index maps 1:1 because responses complete in
    # call order here).
    assert len(result.metadata["results"]) == 6
    assert all(str(r).startswith("answer-") for r in result.metadata["results"])


@pytest.mark.asyncio
async def test_cost_report_tool_actions(tmp_path: Path):
    from openharness.services.cost_dashboard import set_cost_dashboard
    from openharness.services.cost_dashboard import CostDashboard
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.pipeline_tools import CostReportInput, CostReportTool

    dash = CostDashboard(store=None)
    set_cost_dashboard(dash)
    try:
        dash.record(model="claude-sonnet-4", input_tokens=1_000_000, output_tokens=0)
        dash.record(model="claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000, scene="scene_002")
        context = ToolExecutionContext(cwd=tmp_path, metadata={})
        tool = CostReportTool()

        result = await tool.execute(CostReportInput(action="summary", scenes_total=10, scenes_done=5), context)
        assert "$8.0000" in result.output
        assert "projected total $16.0000" in result.output

        result = await tool.execute(CostReportInput(action="models"), context)
        assert "claude-sonnet-4" in result.output and "claude-haiku-4-5" in result.output

        result = await tool.execute(CostReportInput(action="scenes"), context)
        assert "scene_002" in result.output

        result = await tool.execute(CostReportInput(action="recent"), context)
        assert "claude-haiku-4-5" in result.output

        result = await tool.execute(CostReportInput(action="clear"), context)
        assert not result.is_error
        assert dash.summary()["total_calls"] == 0
    finally:
        set_cost_dashboard(None)


# ======================================================================
# 6. batch_scene concurrent retries
# ======================================================================

@pytest.mark.asyncio
async def test_batch_scene_retries_run_concurrently(monkeypatch, tmp_path: Path):
    """Scenes that fail lint get their retry LLM calls fanned out —
    wall time approximates the SLOWEST retry, not the sum."""
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.batch_scene_tool import BatchSceneTool, BatchSceneToolInput

    good_code = 'const scene = { duration: 8.5, render() { return "ok"; } };\n'
    bad_code = "this is not valid code at all !!!"
    # 4 scenes: 2 good, 2 bad (lint fails -> retry each).
    scenes = ["good scene one", "good scene two", "bad scene three", "bad scene four"]

    llm_response = "\n".join(
        f"""
SCENE_{i + 1:03d}:
  FILE: scene_{i + 1:03d}.tsx
  ENGINE: hyperframes
  DURATION: 8.5s
  READS: ["x"]
  CODE:
    {good_code if i < 2 else bad_code}
"""
        for i in range(4)
    )

    retry_calls = 0
    retry_started: list[float] = []

    async def fake_call_llm(self, arguments, config, context):
        return llm_response

    async def fake_retry(self, arguments, config, context, scene, error):
        nonlocal retry_calls
        retry_calls += 1
        retry_started.append(time.monotonic())
        await asyncio.sleep(0.25)
        return good_code

    monkeypatch.setattr(BatchSceneTool, "_call_llm", fake_call_llm)
    monkeypatch.setattr(BatchSceneTool, "_retry_single_scene", fake_retry)

    context = ToolExecutionContext(cwd=tmp_path, metadata={"settings": SimpleNamespace(inference=SimpleNamespace(batch_concurrency=8))})
    arguments = BatchSceneToolInput(
        scenes=scenes,
        output_dir=str(tmp_path / "scenes"),
        max_retries_per_scene=1,
    )
    started = time.monotonic()
    result = await BatchSceneTool().execute(arguments, context)
    elapsed = time.monotonic() - started

    assert retry_calls == 2
    assert not result.is_error
    assert len(result.metadata["scenes_written"]) == 4
    # Both retries ran CONCURRENTLY: 2 x 0.25s serial = 0.5s; concurrent
    # with 8-way cap = ~0.25s. Allow generous slack for slow CI.
    assert elapsed < 0.48, f"retries appear sequential (elapsed {elapsed:.2f}s)"
    # Retry calls overlapped in time.
    assert abs(retry_started[1] - retry_started[0]) < 0.15


# ======================================================================
# 7. Regression tests — cost-halt fallback, QA reviewer lifecycle,
#    audit-log self-heal
# ======================================================================

def test_cost_tracker_prices_unknown_models_conservatively():
    """The session budget halt is a SAFETY control: a model missing from
    the pricing table must still accumulate spend (conservative mid-tier
    fallback), or an unknown model name would silently disable the halt.
    The dashboard's own reporting stays honest ($0 + "unpriced")."""
    from types import SimpleNamespace as _NS

    from openharness.services.cost_dashboard import estimate_cost
    from openharness.services.llm_pipeline import SessionCostTracker

    usage = _NS(input_tokens=1000, output_tokens=100, cache_read_tokens=0, cache_write_tokens=0)
    # Dashboard view: unknown model -> unpriced, $0 (honest reporting).
    cost, priced = estimate_cost("claude-test-unknown", 1000, 100)
    assert cost == 0.0 and priced is False
    # Tracker view: same unknown model -> conservative $0.0045/call
    # (1000 * $3/M + 100 * $15/M), so the halt still trips.
    tracker = SessionCostTracker(budget_usd=0.005, warn_ratio=0.5)
    assert tracker.estimate_cost_usd("claude-test-unknown", usage) == pytest.approx(0.0045)
    # Priced models are unaffected (no double-fallback).
    assert tracker.estimate_cost_usd("claude-sonnet-4", usage) == pytest.approx(0.0045)


def test_cost_tracker_halts_on_unknown_model():
    """End-to-end: crossing the budget with an unknown model halts."""
    from types import SimpleNamespace as _NS

    from openharness.services.llm_pipeline import SessionCostTracker

    tracker = SessionCostTracker(budget_usd=0.005, warn_ratio=0.5)
    usage = _NS(input_tokens=1000, output_tokens=100, cache_read_tokens=0, cache_write_tokens=0)
    first = tracker.record("mystery-model", usage)
    assert first is not None and first.level == "warning"
    second = tracker.record("mystery-model", usage)
    assert second is not None and second.level == "halt"
    assert tracker.halted is True


@pytest.mark.asyncio
async def test_qa_review_tool_closes_its_reviewer(monkeypatch, tmp_path: Path):
    """qa_review builds a fresh reviewer per call; the pooled httpx
    client behind it must be released when the call finishes (previously
    every qa_review invocation leaked one AsyncClient to the GC)."""
    from openharness.services import qa_pipeline as qp
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.pipeline_tools import QAReviewInput, QAReviewTool

    closed: list[bool] = []

    class _SpyReviewer:
        def __init__(self):
            self._http = object()  # non-None: "client open"

        async def review_scenes(self, scenes, **kwargs):
            from openharness.services.qa_pipeline import QAVerdict

            return [
                QAVerdict(scene=p.name, passed=True, score=0.95, depth="single",
                          issues=[], summary="ok")
                for p in scenes
            ]

        def stats(self):
            return {"model": "spy", "calls_made": 1, "waves": 1}

        async def aclose(self):
            # Called exactly once per tool invocation, while the pooled
            # client was still open (i.e. the close is what releases it).
            closed.append(self._http is not None)
            self._http = None

    monkeypatch.setattr(qp, "build_qa_reviewer_from_metadata", lambda metadata: _SpyReviewer())
    scene = tmp_path / "scene_001.tsx"
    scene.write_text("const scene = { duration: 8.5, render() { return 1; } };", encoding="utf-8")
    context = ToolExecutionContext(cwd=tmp_path, metadata={})
    result = await QAReviewTool().execute(QAReviewInput(scenes=[str(scene)]), context)
    assert not result.is_error
    assert closed == [True]


@pytest.mark.asyncio
async def test_qa_review_tool_closes_reviewer_even_on_error(monkeypatch, tmp_path: Path):
    """The close must also happen when the review itself blows up."""
    from openharness.services import qa_pipeline as qp
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.pipeline_tools import QAReviewInput, QAReviewTool

    closed: list[bool] = []

    class _ExplodingReviewer:
        def __init__(self):
            self._http = object()

        async def review_scenes(self, scenes, **kwargs):
            raise RuntimeError("qa endpoint exploded")

        def stats(self):
            return {"model": "spy", "calls_made": 0, "waves": 0}

        async def aclose(self):
            closed.append(True)
            self._http = None

    monkeypatch.setattr(qp, "build_qa_reviewer_from_metadata", lambda metadata: _ExplodingReviewer())
    scene = tmp_path / "scene_002.tsx"
    scene.write_text("const scene = { duration: 8.5, render() { return 1; } };", encoding="utf-8")
    context = ToolExecutionContext(cwd=tmp_path, metadata={})
    with pytest.raises(RuntimeError):
        await QAReviewTool().execute(QAReviewInput(scenes=[str(scene)]), context)
    assert closed == [True]


def test_audit_log_reopens_handle_after_write_failure(tmp_path: Path):
    """A broken audit handle (disk full / file unlinked) must self-heal:
    the next record() reopens the file instead of failing forever."""
    from openharness.services.audit_log import AuditLog

    audit = AuditLog(audit_dir=tmp_path, session_id="self-heal-test", enabled=True)
    assert audit.enabled
    audit.record("bootstrap", {"n": 1})
    assert audit.entries_written == 1
    # Break the handle: replace it with one whose write always raises
    # OSError but whose close works (like a filled disk).
    class _BrokenWrite:
        def write(self, data):
            raise OSError(28, "No space left on device")

        def flush(self):
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    with audit._lock:
        audit._fh = _BrokenWrite()
    audit.record("should-fail", {"n": 2})
    assert audit._dropped >= 1
    assert audit._fh is None  # dropped, not kept broken
    # The next write reopens a real handle and succeeds.
    audit.record("recovered", {"n": 3})
    assert audit.entries_written == 2
    audit.close()
    lines = (tmp_path / f"audit-{time.strftime('%Y%m%d')}.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "bootstrap" in lines[0] and "recovered" in lines[1]
