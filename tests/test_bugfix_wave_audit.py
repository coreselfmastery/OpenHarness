"""Regression tests for the second bug-fix wave (reported bugs 1-7).

Each test pins the exact failure mode reported in the audit:

1.  brand.yaml system-prompt section wiped on the first per-line rebuild
2.  Kaggle auto-detection running AFTER tool_metadata captured defaults
3.  tool cache: TTL=0 coerced to 3600 + cache hits skipping
    PostToolUse hooks / read_file carryover
4.  BrandKit safe-area regex matching opacity / zIndex / paddingTop /
    margin-right substrings
5.  thinking_budget breaking Anthropic multi-turn (thinking blocks
    dropped from history / stale thinking blocks sent when disabled)
6.  CrossProviderRacer sending the primary's model to fallback routes
7.  PromptCompressor not updating the engine's token estimate cache
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from openharness.config.settings import Settings
from openharness.engine.messages import (
    ConversationMessage,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    assistant_message_from_api,
    sanitize_conversation_messages,
)
from openharness.services.compact import estimate_message_tokens


# ======================================================================
# Fix 1 — brand anchors survive every prompt rebuild
# ======================================================================

def _write_brand_yaml(cwd: Path, name: str = "Test Channel") -> None:
    cwd.joinpath("brand.yaml").write_text(
        f"name: {name}\n"
        "colors:\n  primary: '#112233'\n"
        "fonts:\n  heading: Inter\n"
        "safe_area: 220\n",
        encoding="utf-8",
    )


def test_fix1_brand_anchors_in_every_prompt_rebuild(tmp_path: Path):
    """The brand section must come from build_runtime_system_prompt itself,
    so handle_line()'s per-line rebuild keeps it (previously wiped on
    Turn 1)."""
    from openharness.prompts.context import build_runtime_system_prompt

    _write_brand_yaml(tmp_path)
    settings = Settings()
    prompt = build_runtime_system_prompt(settings, cwd=tmp_path, latest_user_prompt="hi")
    assert "# Brand Anchors (from brand.yaml)" in prompt
    assert "Test Channel" in prompt
    # The per-line rebuild (what handle_line does) must still include it.
    prompt2 = build_runtime_system_prompt(settings, cwd=tmp_path, latest_user_prompt="go on")
    assert "# Brand Anchors (from brand.yaml)" in prompt2


def test_fix1_brand_cache_invalidates_on_change(tmp_path: Path):
    """The mtime/size cache must not serve a stale section after edits."""
    from openharness.prompts import context as ctx

    _write_brand_yaml(tmp_path)
    prompt_a = ctx.build_runtime_system_prompt(Settings(), cwd=tmp_path)
    assert "Test Channel" in prompt_a
    _write_brand_yaml(tmp_path, name="Renamed Channel")
    prompt_b = ctx.build_runtime_system_prompt(Settings(), cwd=tmp_path)
    assert "Renamed Channel" in prompt_b
    assert "Test Channel" not in prompt_b


def test_fix1_brand_section_absent_without_file(tmp_path: Path):
    from openharness.prompts.context import build_runtime_system_prompt

    prompt = build_runtime_system_prompt(Settings(), cwd=tmp_path)
    assert "Brand Anchors" not in prompt


# ======================================================================
# Fix 2 — Kaggle auto-detection order vs tool_metadata
# ======================================================================

class _NoopApiClient:
    async def stream_message(self, request):
        del request
        if False:
            yield None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_fix2_kaggle_defaults_reach_tool_metadata(tmp_path, monkeypatch):
    """tool_metadata snapshots concurrency/batch at engine construction —
    the Kagyle defaults must therefore be applied BEFORE that snapshot."""
    from openharness.ui.runtime import build_runtime, close_runtime

    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "openharness.services.kaggle_env.detect_kaggle_resources",
        lambda: {
            "cpu_count": 4,
            "gpu_count": 2,
            "gpu_type": "Tesla T4",
            "gpu_memory_total_mb": 32768,
            "ram_total_gb": 29.0,
            "is_kaggle": True,
            "kaggle_accelerator": "GPU_T4X2",
        },
    )
    bundle = await build_runtime(cwd=str(tmp_path), api_client=_NoopApiClient())
    try:
        assert bundle.engine.tool_metadata["max_concurrent_agents"] == 16
        assert bundle.engine.tool_metadata["batch_size"] == 4
    finally:
        await close_runtime(bundle)


# ======================================================================
# Fix 3 — tool cache TTL + hook/carryover on cache hits
# ======================================================================

def test_fix3_ttl_zero_means_never_expire():
    """TTL=0 is documented as 'never expire'; it must not become 3600."""
    from openharness.services.tool_cache import (
        configure_tool_cache_from_settings,
        get_tool_cache,
        reset_tool_cache,
    )

    class _Cfg:
        tool_cache_enabled = True
        tool_cache_max_entries = 16
        tool_cache_ttl_seconds = 0

    try:
        configure_tool_cache_from_settings(_Cfg())
        cache = get_tool_cache()
        assert cache is not None
        assert cache._ttl_seconds == 0.0
        cache.put("read_file", {"file_path": "a"}, {"output": "x", "is_error": False})
        # Even with an aged entry (frozen clock), a 0 TTL never expires.
        entry_key = cache.make_key("read_file", {"file_path": "a"})
        cache._entries[entry_key].created_at -= 999_999
        assert cache.get("read_file", {"file_path": "a"}) is not None
    finally:
        reset_tool_cache()


def test_fix3_max_entries_zero_caches_nothing():
    from openharness.services.tool_cache import (
        configure_tool_cache_from_settings,
        get_tool_cache,
        reset_tool_cache,
    )

    class _Cfg:
        tool_cache_enabled = True
        tool_cache_max_entries = 0
        tool_cache_ttl_seconds = 600

    try:
        configure_tool_cache_from_settings(_Cfg())
        cache = get_tool_cache()
        assert cache is not None
        assert cache._max_entries == 0
        cache.put("read_file", {"file_path": "a"}, {"output": "x", "is_error": False})
        assert cache.get("read_file", {"file_path": "a"}) is None
    finally:
        reset_tool_cache()


class _RecordingHookExecutor:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def execute(self, event, payload):
        from openharness.hooks.types import AggregatedHookResult

        if str(getattr(event, "value", event)) == "post_tool_use":
            self.events.append((payload.get("tool_name", ""), str(payload.get("tool_output", ""))))
        return AggregatedHookResult()


class _AllowAllPermissions:
    def evaluate(self, *args, **kwargs):
        from openharness.permissions.checker import PermissionDecision

        return PermissionDecision(allowed=True, requires_confirmation=False, reason="")


from pydantic import BaseModel

from openharness.tools.base import BaseTool, ToolRegistry, ToolResult
from openharness.engine.query import QueryContext


class _FakeReadInput(BaseModel):
    file_path: str = "a.txt"


class _FakeReadTool(BaseTool):
    name = "read_file"
    description = "fake read_file for cache tests"
    calls = 0
    input_model = _FakeReadInput

    def is_read_only(self, arguments) -> bool:
        return True

    async def execute(self, arguments, context) -> ToolResult:
        _FakeReadTool.calls += 1
        return ToolResult(output=f"content-v{_FakeReadTool.calls}")


@pytest.mark.asyncio
async def test_fix3_cache_hit_runs_post_tool_use_hook_and_carryover():
    """Cache hits must not skip PostToolUse hooks / read_file_state."""
    from openharness.engine.query import _execute_tool_call
    from openharness.services.tool_cache import (
        enable_tool_cache,
        reset_tool_cache,
    )

    enable_tool_cache(max_entries=16, ttl_seconds=600)
    registry = ToolRegistry()
    registry.register(_FakeReadTool())
    _FakeReadTool.calls = 0
    hooks = _RecordingHookExecutor()
    metadata: dict[str, Any] = {"read_file_state": []}
    context = QueryContext(
        api_client=_NoopApiClient(),
        tool_registry=registry,
        permission_checker=_AllowAllPermissions(),
        cwd=Path("."),
        model="test-model",
        system_prompt="sys",
        max_tokens=4096,
        tool_metadata=metadata,
        hook_executor=hooks,
    )
    try:
        first = await _execute_tool_call(context, "read_file", "tu_1", {"file_path": "a.txt"})
        assert "content-v1" in first.content
        second = await _execute_tool_call(context, "read_file", "tu_2", {"file_path": "a.txt"})
        assert "content-v1" in second.content  # served from cache
        assert _FakeReadTool.calls == 1
        # BOTH calls fired the PostToolUse hook (previously the cache hit
        # returned before the hook ran).
        assert len(hooks.events) == 2
        assert hooks.events[0][0] == "read_file"
        assert hooks.events[1][0] == "read_file"
        # read_file carryover ran on the cache hit too: the file path is
        # remembered as an active artifact.
        assert metadata["read_file_state"]
    finally:
        reset_tool_cache()


# ======================================================================
# Fix 4 — safe-area regex
# ======================================================================

def test_fix4_safe_area_ignores_unrelated_properties():
    from openharness.tools.brand_kit_tool import _find_safe_area_violations

    clean = (
        "opacity: 0.5; zIndex: 10; paddingTop: 12; paddingBottom: 16; "
        "marginRight: 8; margin-left: 8px; scrollLeft: 4; zIndex: 3;"
    )
    assert _find_safe_area_violations(clean, 220) == []


def test_fix4_safe_area_flags_real_positions():
    from openharness.tools.brand_kit_tool import _find_safe_area_violations

    bad = "left: 40; top: 30; x: 10; { right: 12 }; position.y = 5"
    violations = _find_safe_area_violations(bad, 220)
    assert len(violations) >= 5


# ======================================================================
# Fix 5 — thinking_budget + Anthropic multi-turn
# ======================================================================

class _RawThinkingBlock:
    type = "thinking"
    thinking = "internal reasoning text"
    signature = "sig-abc"


class _RawTextBlock:
    type = "text"
    text = "hello"


class _RawToolUseBlock:
    type = "tool_use"
    id = "toolu_1"
    name = "read_file"
    input = {"file_path": "a"}


class _RawRedactedBlock:
    type = "redacted_thinking"
    data = "encrypted-blob"


class _RawMessage:
    def __init__(self, blocks):
        self.content = blocks


def test_fix5_thinking_blocks_preserved_from_api():
    msg = assistant_message_from_api(
        _RawMessage([_RawThinkingBlock(), _RawToolUseBlock(), _RawTextBlock()])
    )
    kinds = [type(b) for b in msg.content]
    assert ThinkingBlock in kinds
    assert ToolUseBlock in kinds
    thinking = next(b for b in msg.content if isinstance(b, ThinkingBlock))
    assert thinking.signature == "sig-abc"
    # Round-trip through the wire format keeps the signature.
    from openharness.engine.messages import serialize_content_block

    wire = serialize_content_block(thinking)
    assert wire == {"type": "thinking", "thinking": "internal reasoning text", "signature": "sig-abc"}


def test_fix5_redacted_thinking_preserved():
    msg = assistant_message_from_api(_RawMessage([_RawRedactedBlock(), _RawToolUseBlock()]))
    assert any(isinstance(b, RedactedThinkingBlock) for b in msg.content)


def test_fix5_thinking_only_message_is_still_dropped_as_empty():
    msg = ConversationMessage(role="assistant", content=[ThinkingBlock(thinking="x", signature="y")])
    assert msg.is_effectively_empty() is True
    assert sanitize_conversation_messages([msg]) == []


def _anthropic_client_for_params():
    """Build an AnthropicApiClient with a fake SDK transport capturing params."""
    from openharness.api.client import AnthropicApiClient

    client = AnthropicApiClient(api_key="test-key")

    captured: dict[str, Any] = {}

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_message(self):
            return _RawMessage([_RawTextBlock()])

    class _FakeMessages:
        def stream(self, **params):
            captured.update(params)
            return _FakeStream()

    class _FakeSdkClient:
        messages = _FakeMessages()

    client._client = _FakeSdkClient()
    return client, captured


@pytest.mark.asyncio
async def test_fix5_thinking_enabled_keeps_history_blocks():
    """With thinking on, history thinking blocks must reach the provider."""
    from openharness.api.client import ApiMessageRequest

    client, captured = _anthropic_client_for_params()
    history = [
        ConversationMessage(role="user", content=[TextBlock(text="hi")]),
        ConversationMessage(
            role="assistant",
            content=[ThinkingBlock(thinking="r", signature="s"), ToolUseBlock(id="t1", name="read_file", input={})],
        ),
        ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
    ]
    request = ApiMessageRequest(
        model="claude-sonnet-4-20250514",
        messages=history,
        system_prompt="sys",
        max_tokens=8192,
        thinking_budget=2048,
    )
    events = [e async for e in client.stream_message(request)]
    assert any(getattr(e, "message", None) is not None for e in events)
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    sent = captured["messages"]
    kinds = [b["type"] for b in sent[1]["content"]]
    assert "thinking" in kinds
    assert sent[1]["content"][0]["signature"] == "s"


@pytest.mark.asyncio
async def test_fix5_thinking_disabled_strips_history_blocks():
    """With thinking off, stale thinking blocks must NOT reach the provider."""
    from openharness.api.client import ApiMessageRequest

    client, captured = _anthropic_client_for_params()
    history = [
        ConversationMessage(role="user", content=[TextBlock(text="hi")]),
        ConversationMessage(
            role="assistant",
            content=[ThinkingBlock(thinking="r", signature="s"), ToolUseBlock(id="t1", name="read_file", input={})],
        ),
        ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
    ]
    request = ApiMessageRequest(
        model="claude-sonnet-4-20250514",
        messages=history,
        system_prompt="sys",
        max_tokens=8192,
        thinking_budget=None,
    )
    _ = [e async for e in client.stream_message(request)]
    assert "thinking" not in captured
    for message in captured["messages"]:
        for block in message["content"]:
            assert block["type"] not in ("thinking", "redacted_thinking")


@pytest.mark.asyncio
async def test_fix5_unsupported_model_strips_thinking_from_history():
    from openharness.api.client import ApiMessageRequest

    client, captured = _anthropic_client_for_params()
    history = [
        ConversationMessage(role="user", content=[TextBlock(text="hi")]),
        ConversationMessage(
            role="assistant",
            content=[ThinkingBlock(thinking="r", signature="s"), TextBlock(text="old")],
        ),
    ]
    request = ApiMessageRequest(
        model="claude-3-haiku-20240307",  # no thinking support
        messages=history,
        system_prompt="sys",
        max_tokens=8192,
        thinking_budget=2048,
    )
    _ = [e async for e in client.stream_message(request)]
    assert "thinking" not in captured
    for message in captured["messages"]:
        for block in message["content"]:
            assert block["type"] != "thinking"


# ======================================================================
# Fix 6 — racing / routing must send each route its own model
# ======================================================================

def test_fix6_request_for_route_overrides_model():
    from openharness.api.client import ApiMessageRequest
    from openharness.services.provider_routing import request_for_route

    request = ApiMessageRequest(
        model="claude-sonnet-4",
        messages=[],
        effort="high",
        thinking_budget=4096,
    )
    adapted = request_for_route(request, "llama-3.3-70b-versatile")
    assert adapted.model == "llama-3.3-70b-versatile"
    assert adapted.effort is None
    assert adapted.thinking_budget is None
    # Same model (or no override) → same object, no copy.
    assert request_for_route(request, None) is request
    assert request_for_route(request, "claude-sonnet-4") is request


class _RecordingClient:
    def __init__(self, reply: str = "done") -> None:
        self.models: list[str] = []
        self._reply = reply

    async def stream_message(self, request):
        from openharness.api.client import ApiMessageCompleteEvent, ApiTextDeltaEvent
        from openharness.api.usage import UsageSnapshot

        self.models.append(request.model)
        yield ApiTextDeltaEvent(text=self._reply)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self._reply)]),
            usage=UsageSnapshot(),
        )


@pytest.mark.asyncio
async def test_fix6_racing_client_sends_each_route_its_model():
    from openharness.api.client import ApiMessageRequest
    from openharness.services.llm_pipeline import RacingClient

    primary = _RecordingClient("from-primary")
    fallback = _RecordingClient("from-fallback")
    racer = RacingClient([
        ("primary", primary, None),
        ("groq", fallback, "llama-3.3-70b-versatile"),
    ])
    request = ApiMessageRequest(model="claude-sonnet-4", messages=[], max_tokens=512)
    seen = []
    async for event in racer.stream_message(request):
        seen.append(type(event).__name__)
    assert seen  # streamed something
    # The fallback route must have been asked for ITS model, not the
    # primary's (previously it 404'd with "model not found").
    assert fallback.models == ["llama-3.3-70b-versatile"]


@pytest.mark.asyncio
async def test_fix6_routed_fallback_sends_route_model():
    """Sequential RoutedStreamingClient has the same requirement."""
    from openharness.api.client import ApiMessageRequest
    from openharness.api.errors import RequestFailure
    from openharness.services.provider_routing import ProviderRoute, RoutedStreamingClient

    class _FailingClient:
        async def stream_message(self, request):
            raise RequestFailure("primary down")
            yield  # pragma: no cover

    primary = _FailingClient()
    fallback = _RecordingClient("from-fallback")
    routed = RoutedStreamingClient(
        ProviderRoute(name="primary", client=primary),
        ProviderRoute(name="backup", client=fallback, model="qwen-plus"),
    )
    request = ApiMessageRequest(model="claude-sonnet-4", messages=[], max_tokens=512)
    collected = [e async for e in routed.stream_message(request)]
    assert collected
    assert fallback.models == ["qwen-plus"]


def test_fix6_build_fallback_route_reads_model():
    from openharness.services.provider_routing import build_fallback_route

    route = build_fallback_route(
        "backup",
        {
            "api_format": "openai",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-x",
            "model": "qwen-plus",
        },
    )
    assert route is not None
    assert route.model == "qwen-plus"


# ======================================================================
# Fix 7 — compressor publishes compressed token estimates
# ======================================================================

def _history_with_big_tool_result() -> list[ConversationMessage]:
    messages = [
        ConversationMessage(role="user", content=[TextBlock(text="start " + "x" * 200)]),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="read_file", input={"file_path": "huge"})],
        ),
        ConversationMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="y" * 6000)],
        ),
        ConversationMessage(role="user", content=[TextBlock(text="continue")]),
        ConversationMessage(role="user", content=[TextBlock(text="more")]),
    ]
    return messages


def test_fix7_compressor_updates_token_estimate_cache():
    from openharness.services.llm_pipeline import PromptCompressor

    messages = _history_with_big_tool_result()
    before = estimate_message_tokens(messages)
    compressor = PromptCompressor(min_chars=1000, keep_turns=2)
    compressed = compressor.compress(messages)
    assert compressed is not messages  # actually compressed
    after = estimate_message_tokens(messages)
    # The engine's estimate on the ORIGINAL list must now reflect the
    # compressed view (previously it kept the uncompressed count and
    # auto-compacted far too early).
    assert after < before


def test_fix7_compressor_never_mutates_input():
    from openharness.services.llm_pipeline import PromptCompressor

    messages = _history_with_big_tool_result()
    snapshot = json.dumps([m.model_dump() for m in messages], sort_keys=True)
    compressor = PromptCompressor(min_chars=1000, keep_turns=2)
    compressor.compress(messages)
    assert json.dumps([m.model_dump() for m in messages], sort_keys=True) == snapshot


# ======================================================================
# Deep audit — semantic cache stability + isolation
# ======================================================================

def test_audit_minhash_signature_is_process_salt_independent():
    """Signatures are persisted in SQLite and compared across processes;
    they must not depend on PYTHONHASHSEED."""
    import subprocess
    import sys

    code = (
        "from openharness.services.llm_pipeline import _minhash_signature;"
        "print(_minhash_signature('alpha beta gamma delta epsilon zeta eta theta'))"
    )
    out_a = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"},
        check=True,
    ).stdout.strip()
    out_b = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
        check=True,
    ).stdout.strip()
    assert out_a == out_b


def test_audit_semantic_cache_ignores_system_prompt_for_similarity():
    """Different user questions must not collide via a shared long system
    prompt (previously the system prompt dominated the shingles)."""
    from openharness.api.client import ApiMessageRequest
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage, TextBlock
    from openharness.services.llm_pipeline import ApiMessageCompleteEvent, SemanticCache

    class _Store:
        def __init__(self):
            self.data: dict[str, dict] = {}

        def namespace_entries(self, ns, limit):
            return dict(self.data.get(ns, {}))

        def get(self, ns, key):
            return self.data.get(ns, {}).get(key)

        def set(self, ns, key, payload):
            self.data.setdefault(ns, {})[key] = payload

        def prune_namespace(self, ns, limit):
            return 0

    store = _Store()
    cache = SemanticCache(store, threshold=0.88)
    long_system = "system preamble " * 400
    q1 = "please write a detailed summary of the quarterly revenue numbers"
    q2 = "please write a limerick about a cat named dog who barks"
    r1 = ApiMessageRequest(
        model="m", system_prompt=long_system,
        messages=[ConversationMessage(role="user", content=[TextBlock(text=q1)])],
    )
    r2 = ApiMessageRequest(
        model="m", system_prompt=long_system,
        messages=[ConversationMessage(role="user", content=[TextBlock(text=q2)])],
    )
    cache.store_response(
        r1,
        ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="revenue answer")]),
            usage=UsageSnapshot(),
        ),
    )
    hit = cache.lookup(r2)
    assert hit is None, "different question must not hit the cache"


def test_audit_semantic_cache_short_prompt_exact_only():
    from openharness.api.client import ApiMessageRequest
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage, TextBlock
    from openharness.services.llm_pipeline import ApiMessageCompleteEvent, SemanticCache

    class _Store:
        def __init__(self):
            self.data: dict[str, dict] = {}

        def namespace_entries(self, ns, limit):
            return dict(self.data.get(ns, {}))

        def get(self, ns, key):
            return self.data.get(ns, {}).get(key)

        def set(self, ns, key, payload):
            self.data.setdefault(ns, {})[key] = payload

        def prune_namespace(self, ns, limit):
            return 0

    store = _Store()
    cache = SemanticCache(store)
    req_ok = ApiMessageRequest(
        model="m",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="ok")])],
    )
    cache.store_response(
        req_ok,
        ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="sure")]),
            usage=UsageSnapshot(),
        ),
    )
    # Token-identical (punctuation-insensitive) short prompt hits.
    assert cache.lookup(req_ok) is not None
    # Any wording change on a short prompt must miss.
    req_other = ApiMessageRequest(
        model="m",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="go on")])],
    )
    assert cache.lookup(req_other) is None


def test_audit_semantic_cache_stable_key():
    from openharness.services.llm_pipeline import SemanticCache

    # The key is model-scoped: same model + same text -> same key;
    # different text OR different model -> different key.
    assert SemanticCache._stable_key("m1", "hello") == SemanticCache._stable_key("m1", "hello")
    assert SemanticCache._stable_key("m1", "hello") != SemanticCache._stable_key("m1", "hellp")
    assert SemanticCache._stable_key("m1", "hello") != SemanticCache._stable_key("m2", "hello")


# ======================================================================
# Deep audit — render pipeline
# ======================================================================

def test_audit_render_status_codec_never_crashes():
    from openharness.services.render_pipeline import RenderPipeline

    pipeline = RenderPipeline(workers=1, cwd=Path("."), nvenc_enabled=False)
    pipeline.nvenc.codec_args = lambda enabled=True: ""  # degenerate template
    status = pipeline.status()
    assert status["codec"] == "unknown"


def test_audit_render_worker_env_gpu_round_robin(monkeypatch):
    from openharness.services import render_pipeline as rp

    monkeypatch.setattr(rp, "_GPU_COUNT_CACHE", 2)
    envs = [rp._worker_env(i, 4) for i in range(4)]
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "0", "1"]
    # CPU-only hosts must not pin non-existent devices.
    monkeypatch.setattr(rp, "_GPU_COUNT_CACHE", 0)
    assert "CUDA_VISIBLE_DEVICES" not in rp._worker_env(0, 2)


def test_audit_render_jobs_for_scene_matches_by_path(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    scene_a = tmp_path / "a" / "scene.tsx"
    scene_b = tmp_path / "b" / "scene.tsx"  # same basename, different dir
    scene_a.parent.mkdir(exist_ok=True)
    scene_b.parent.mkdir(exist_ok=True)
    scene_a.write_text("// a", encoding="utf-8")
    scene_b.write_text("// b", encoding="utf-8")
    from openharness.services.render_pipeline import RenderJob

    pipeline._jobs["r00001"] = RenderJob(
        job_id="r00001", scene=scene_a, engine="hyperframes", tier="draft",
        output_dir=tmp_path, output_path=tmp_path / "a.mp4",
    )
    pipeline._jobs["r00002"] = RenderJob(
        job_id="r00002", scene=scene_b, engine="hyperframes", tier="draft",
        output_dir=tmp_path, output_path=tmp_path / "b.mp4",
    )
    matches = pipeline._jobs_for_scene(scene_a, tier="draft")
    assert len(matches) == 1
    assert matches[0].scene == scene_a


def test_audit_render_prune_jobs_bounds_registry(tmp_path: Path):
    from openharness.services.render_pipeline import RenderJob, RenderPipeline

    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    for i in range(50):
        pipeline._jobs[f"r{i:05d}"] = RenderJob(
            job_id=f"r{i:05d}", scene=tmp_path / f"s{i}.tsx", engine="hyperframes",
            tier="draft", output_dir=tmp_path, output_path=tmp_path / f"{i}.mp4",
        )
        pipeline._jobs[f"r{i:05d}"].status = "done"
    pipeline._prune_jobs(keep=10)
    assert len(pipeline._jobs) <= 10


# ======================================================================
# Deep audit — QA pipeline
# ======================================================================

def test_audit_qa_score_normalized_from_0_100_scale():
    from openharness.services.qa_pipeline import parse_verdict_payload

    payload = json.dumps({
        "verdicts": [
            {"scene": "scene_001", "passed": True, "score": 85, "summary": "fine"},
        ]
    })
    verdicts = parse_verdict_payload(payload, ["scene_001.tsx"])
    assert verdicts is not None
    assert verdicts[0].score == pytest.approx(0.85)


def test_audit_qa_score_clamped():
    from openharness.services.qa_pipeline import parse_verdict_payload

    payload = json.dumps({
        "verdicts": [
            {"scene": "scene_001", "passed": False, "score": 250, "summary": "bad"},
        ]
    })
    verdicts = parse_verdict_payload(payload, ["scene_001.tsx"])
    assert verdicts is not None
    assert verdicts[0].score == 1.0


@pytest.mark.asyncio
async def test_audit_qa_batching_without_smart_depth():
    """batching=True + smart_depth=False + auto mode → every scene is
    batched (previously all fell into the single-review bucket)."""
    from openharness.services.qa_pipeline import QAReviewer

    reviewer = QAReviewer(
        base_url="https://qa.example.com/v1",
        api_key="k",
        model="qa-model",
        wave_size=4,
        batching=True,
        batch_size=2,
        smart_depth=False,
    )
    calls: list[list[str]] = []

    async def fake_chat(prompt: str) -> str:
        import re as _re

        names = _re.findall(r"scene_\d+", prompt)
        calls.append(names)
        return json.dumps({
            "verdicts": [
                {"scene": n, "passed": True, "score": 0.9, "summary": "ok"}
                for n in dict.fromkeys(names)
            ]
        })

    reviewer._chat = fake_chat  # type: ignore[assignment]
    scenes = [Path(f"scene_{i:03d}.tsx") for i in range(1, 5)]
    verdicts = await reviewer.review_scenes(scenes, mode="auto")
    assert len(verdicts) == 4
    # 4 scenes / batch_size 2 → exactly 2 LLM calls, no single reviews.
    assert len(calls) == 2
    await reviewer.aclose()


@pytest.mark.asyncio
async def test_audit_qa_http_client_reused():
    from openharness.services.qa_pipeline import QAReviewer

    reviewer = QAReviewer(
        base_url="https://qa.example.com/v1", api_key="k", model="qa-model"
    )
    first = reviewer._http_client()
    second = reviewer._http_client()
    assert first is second
    await reviewer.aclose()
    assert reviewer._http is None


# ======================================================================
# Deep audit — control plane
# ======================================================================

@pytest.mark.asyncio
async def test_audit_control_bus_monitor_tasks_stopped_on_replace():
    from openharness.services.control_bus import ControlBus, set_control_bus

    bus = ControlBus(watchdog_timeout_seconds=60, deadlock_probe_seconds=30)
    bus.start_watchdog()
    bus.start_deadlock_monitor()
    set_control_bus(bus)
    watchdog_task = bus._watchdog_task
    deadlock_task = bus._deadlock_task
    assert watchdog_task is not None and deadlock_task is not None
    set_control_bus(None)  # replaces the singleton → must stop monitors
    await asyncio.sleep(0)
    assert watchdog_task.done()
    assert deadlock_task.done()


@pytest.mark.asyncio
async def test_audit_control_bus_shutdown_stops_monitors():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=60, deadlock_probe_seconds=30)
    bus.start_watchdog()
    bus.start_deadlock_monitor()
    await bus.begin_shutdown(reason="test")
    assert bus.shutdown_phase == "closed"
    await asyncio.sleep(0)
    assert bus._watchdog_task is None
    assert bus._deadlock_task is None


@pytest.mark.asyncio
async def test_audit_dead_lockfile_stolen_by_pid_check(tmp_path, monkeypatch):
    """A lockfile left by a dead process must be stolen instantly."""
    from openharness.services import conflict_resolution as cr

    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    lock_path = cr._lock_path_for(tmp_path / "victim.txt")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("openharness edit_file (pid 999999999)", encoding="utf-8")
    async with cr.acquire_path_lock(tmp_path / "victim.txt", timeout_seconds=1.0):
        pass  # must not raise FileConflictError
