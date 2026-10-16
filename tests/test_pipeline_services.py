"""Real tests for the waves 2-6 data-plane services (items #1-#25, #40-#50).

Covers the modules that carry work rather than control it:

1. Persistent store (#19)
2. Model cache (#20)
3. Render cache (#14/#17)
4. LLM inference pipeline (#2/#3/#4/#6/#18/#28 + pipeline client)
5. API client prefix caching (#2) + usage accounting
6. QA pipeline units (#7-#10)
7. Render pipeline units (#24/#25 + command building)
8. Server acceleration reporting (#41-#50)
9. Asset prefetch (#22/#23)
10. Tool registry surface + cache_stats / server_config tools
11. Runtime wiring helpers (key-override client builder, pipeline wrap)
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    AnthropicApiClient,
)
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import Settings
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.tools.base import ToolExecutionContext


# ======================================================================
# Isolation
# ======================================================================

@pytest.fixture(autouse=True)
def _isolated_services(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    from openharness.services import (
        audit_log as audit_mod,
        control_bus as bus_mod,
        persistent_store as store_mod,
        progress_tracker as tracker_mod,
    )
    from openharness.services import render_pipeline as render_mod
    from openharness.services.session_services import reset_session_services

    def _reset():
        audit_mod.set_audit_log(None)
        bus_mod.set_control_bus(None)
        store_mod.set_persistent_store(None)
        tracker_mod.set_progress_tracker(None)
        render_mod._PIPELINE = None
        reset_session_services()

    _reset()
    yield
    _reset()


# ======================================================================
# 1. Persistent store (#19)
# ======================================================================

def test_persistent_store_kv_roundtrip(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore

    store = PersistentStore(tmp_path / "store.sqlite3")
    assert store.stats()["available"]
    assert store.get("ns", "missing") is None
    assert store.set("ns", "k1", {"a": 1})
    assert store.get("ns", "k1") == {"a": 1}
    store.set("ns", "k1", {"a": 2})  # upsert
    assert store.get("ns", "k1") == {"a": 2}
    assert store.set("ns", "k2", "v2")
    entries = store.namespace_entries("ns")
    assert set(entries) == {"k1", "k2"}
    assert store.delete("ns", "k1") is True
    assert store.delete("ns", "k1") is False
    assert store.clear_namespace("ns") == 1
    assert store.namespace_entries("ns") == {}


def test_persistent_store_counters(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore

    store = PersistentStore(tmp_path / "store.sqlite3")
    assert store.bump("qa", "cat|x", 2) == 2
    assert store.bump("qa", "cat|x") == 3
    assert store.counter("qa", "cat|x") == 3
    assert store.counters_in_namespace("qa") == {"cat|x": 3}
    assert store.counter("qa", "other") == 0


def test_persistent_store_prune_keeps_newest(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore

    store = PersistentStore(tmp_path / "store.sqlite3")
    for index in range(5):
        store.set("bulk", f"key{index}", index)
        time.sleep(0.005)
    removed = store.prune_namespace("bulk", 2)
    assert removed == 3
    remaining = store.namespace_entries("bulk")
    assert len(remaining) == 2
    assert "key4" in remaining  # newest kept


def test_persistent_store_broken_path_degrades_to_noop(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore

    # Parent path is a FILE -> mkdir/open fails -> broken store.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    store = PersistentStore(blocker / "sub" / "store.sqlite3")
    assert not store.stats()["available"]
    assert store.set("ns", "k", 1) is False
    assert store.get("ns", "k") is None
    assert store.bump("ns", "k") == 0
    assert store.clear_namespace("ns") == 0
    store.close()  # must not raise


def test_persistent_store_close_then_reopen(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore

    path = tmp_path / "store.sqlite3"
    store = PersistentStore(path)
    store.set("ns", "persisted", [1, 2, 3])
    store.close()
    assert store.get("ns", "persisted") is None  # closed -> miss, no crash
    reopened = PersistentStore(path)
    assert reopened.get("ns", "persisted") == [1, 2, 3]  # WAL survived


def test_fingerprint_is_stable_and_sensitive():
    from openharness.services.persistent_store import fingerprint

    a = fingerprint("code", ["b", "a"], {"k": 1})
    b = fingerprint("code", ["b", "a"], {"k": 1})
    c = fingerprint("code2", ["b", "a"], {"k": 1})
    d = fingerprint("code", ["a", "b"], {"k": 1})  # order-sensitive collections
    assert a == b
    assert a != c
    assert a != d


def test_persistent_store_singleton_honors_env(tmp_path: Path, monkeypatch):
    from openharness.services import persistent_store as mod
    from openharness.services.persistent_store import get_persistent_store

    custom = tmp_path / "custom.sqlite3"
    monkeypatch.setenv("OPENHARNESS_CACHE_DB", str(custom))
    store = get_persistent_store()
    assert store.stats()["path"] == str(custom)
    mod.set_persistent_store(None)


# ======================================================================
# 2. Model cache (#20)
# ======================================================================

def _write_hf_snapshot(cache_dir: Path, model_id: str) -> None:
    safe = model_id.strip("/").replace("/", "--")
    snap = cache_dir / "hub" / f"models--{safe}" / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "weights.bin").write_bytes(b"\x00" * 16)


def test_model_cache_verify_cold_then_warm(tmp_path: Path):
    from openharness.services.model_cache import ModelCache

    cache = ModelCache(tmp_path / "hf", apply_env=False)
    cold = cache.verify("org/model-x")
    assert cold["cached"] is False and cold["files"] == 0
    _write_hf_snapshot(cache.cache_dir, "org/model-x")
    warm = cache.verify("org/model-x")
    assert warm["cached"] is True and warm["files"] >= 1
    assert cache.warm_verifies == 1 and cache.cold_verifies == 1
    summary = cache.verify_many(["org/model-x", "org/other"])
    assert summary["warm"] == ["org/model-x"]
    assert summary["cold"] == ["org/other"]
    assert summary["minutes_saved_estimate"] > 0


def test_model_cache_apply_env_redirects_and_respects_user(tmp_path: Path, monkeypatch):
    from openharness.services.model_cache import ModelCache

    cache = ModelCache(tmp_path / "hf-a")
    assert cache._env_applied is True
    import os

    assert os.environ["HF_HOME"] == str(tmp_path / "hf-a")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hf-a" / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(tmp_path / "hf-a" / "hub")

    # User-set HF_HOME always wins.
    monkeypatch.setenv("HF_HOME", "/user/chose/this")
    user_cache = ModelCache(tmp_path / "hf-b")
    assert user_cache.apply_env() is False
    import os

    assert os.environ["HF_HOME"] == "/user/chose/this"


def test_model_cache_persistence_notes_are_honest():
    from openharness.services.model_cache import ModelCache

    tmp_note = ModelCache("/tmp/openharness-cache", apply_env=False).persistence_note()
    assert "wiped on restart" in tmp_note
    kaggle_note = ModelCache(
        "/kaggle/working/cache", apply_env=False
    ).persistence_note()
    assert "Kaggle" in kaggle_note
    home_note = ModelCache(
        Path.home() / ".cache" / "oh-models", apply_env=False
    ).persistence_note()
    assert "persistent storage" in home_note


def test_model_cache_configure_from_settings(tmp_path: Path):
    from openharness.config.settings import ModelCacheSettings
    from openharness.services.model_cache import (
        ModelCache,
        configure_model_cache_from_settings,
    )

    disabled = Settings(model_cache=ModelCacheSettings(enabled=False))
    assert configure_model_cache_from_settings(disabled) is None
    enabled = Settings(
        model_cache=ModelCacheSettings(enabled=True, cache_dir=str(tmp_path / "mc"))
    )
    cache = configure_model_cache_from_settings(enabled)
    assert isinstance(cache, ModelCache)
    assert cache.cache_dir == tmp_path / "mc"


# ======================================================================
# 3. Render cache (#14/#17)
# ======================================================================

def test_find_asset_paths_filters_and_dedups(tmp_path: Path):
    from openharness.services.render_cache import find_asset_paths

    (tmp_path / "bg.png").write_bytes(b"img")
    (tmp_path / "voice.mp3").write_bytes(b"aud")
    code = (
        'load("bg.png"); load("bg.png"); logo("https://cdn.example.com/x.png"); '
        'audio("voice.mp3"); missing("gone.png")'
    )
    found = find_asset_paths(code, "hyperframes", cwd=tmp_path)
    names = [p.name for p in found]
    assert names == ["bg.png", "voice.mp3"]  # dedup + http skipped + missing skipped


def test_scene_fingerprint_stable_and_sensitive(tmp_path: Path):
    from openharness.services.render_cache import scene_fingerprint

    scene = tmp_path / "scene_001.py"
    scene.write_text("draw('bg.png')", encoding="utf-8")
    (tmp_path / "bg.png").write_bytes(b"asset-v1")
    kwargs = dict(engine="hyperframes", engine_args=["--fps", "30"], tier="draft",
                  codec="libx264", cwd=tmp_path)
    fp1 = scene_fingerprint(scene, **kwargs)
    fp2 = scene_fingerprint(scene, **kwargs)
    assert fp1 == fp2  # identical inputs -> identical fingerprint
    # Code change invalidates.
    scene.write_text("draw2('bg.png')", encoding="utf-8")
    assert scene_fingerprint(scene, **kwargs) != fp1
    # Asset content change invalidates.
    scene.write_text("draw('bg.png')", encoding="utf-8")
    (tmp_path / "bg.png").write_bytes(b"asset-v2")
    assert scene_fingerprint(scene, **kwargs) != fp1
    # Tier is part of the fingerprint.
    (tmp_path / "bg.png").write_bytes(b"asset-v1")
    other_tier = dict(kwargs, tier="final")
    assert scene_fingerprint(scene, **other_tier) != fp1
    # Missing scene -> nothing to cache.
    assert scene_fingerprint(tmp_path / "nope.py", **kwargs) is None


def test_render_cache_store_hit_promote_and_stale_miss(tmp_path: Path):
    from openharness.services.render_cache import RenderCache
    from openharness.services.persistent_store import PersistentStore

    store = PersistentStore(tmp_path / "store.sqlite3")
    cache = RenderCache(store)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"video")

    assert cache.lookup("fp-1") is None and cache.misses == 1
    cache.store("fp-1", output_path=str(output), scene="s1", engine="hyperframes",
                tier="draft", duration_seconds=2.5, quality_score=0.75)
    hit = cache.lookup("fp-1")
    assert hit is not None and hit["scene"] == "s1"
    assert cache.hits == 1

    cache.promote("fp-1", new_tier="final")
    assert cache.lookup("fp-1")["promoted_to"] == "final"
    assert cache.quality_score("fp-1") == 0.75
    cache.record_quality_score("fp-1", 0.9)
    assert cache.quality_score("fp-1") == 0.9

    # Deleted output file -> stale entry dropped, honest miss.
    output.unlink()
    assert cache.lookup("fp-1") is None
    assert cache.lookup("fp-1") is None  # entry was deleted, still missing
    assert cache.stats()["entries"] == 0


# ======================================================================
# 4. LLM inference pipeline (#2/#3/#4/#6/#18/#28)
# ======================================================================

def _convo(*texts: str) -> list[ConversationMessage]:
    return [ConversationMessage(role="user", content=[TextBlock(text=t)]) for t in texts]


def _tool_convo() -> list[ConversationMessage]:
    return [
        ConversationMessage(role="user", content=[TextBlock(text="render it")]),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="tu_1", name="render_scenes", input={})],
        ),
        ConversationMessage(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="tu_1", content="x" * 5000)  # huge body
            ],
        ),
    ] + _convo("thanks", "and bye")


def test_prompt_compressor_stubs_old_tool_results():
    from openharness.services.llm_pipeline import PromptCompressor

    compressor = PromptCompressor(min_chars=1, keep_turns=2, tool_result_cap=500)
    messages = _tool_convo()
    compressed = compressor.compress(messages)
    assert len(compressed) == len(messages)
    # The huge tool result (older than keep_turns) got stubbed...
    stub = [
        b for b in compressed[2].content
        if isinstance(b, ToolResultBlock)
    ][0]
    assert "elided by prompt compression" in stub.content
    assert len(stub.content) < 200
    # ...the tool_use assistant message is untouched (pairing preserved)...
    assert compressed[1].tool_uses[0].id == "tu_1"
    # ...and the input list was never mutated.
    original = [b for b in messages[2].content if isinstance(b, ToolResultBlock)][0]
    assert len(original.content) == 5000
    assert compressor.stats()["chars_saved"] > 4000


def test_prompt_compressor_noop_below_threshold():
    from openharness.services.llm_pipeline import PromptCompressor

    compressor = PromptCompressor(min_chars=10_000, keep_turns=3)
    messages = _convo("short", "conversation")
    assert compressor.compress(messages) is messages  # identity: below min_chars
    short = PromptCompressor(min_chars=1, keep_turns=10)
    assert short.compress(messages) is messages  # identity: fewer than keep_turns


def test_minhash_similarity_identical_vs_different():
    from openharness.services.llm_pipeline import (
        _minhash_signature,
        _signature_similarity,
    )

    text = "make a short video about the history of railways in europe"
    sig = _minhash_signature(text)
    same = _minhash_signature(text)
    assert _signature_similarity(sig, same) == 1.0
    other = _minhash_signature("completely unrelated prompt about cooking pasta")
    assert _signature_similarity(sig, other) < 0.5


def _request(text: str, model: str = "claude-test") -> ApiMessageRequest:
    return ApiMessageRequest(
        model=model,
        messages=_convo(text),
        system_prompt="You are a video agent.",
    )


def _complete_event(text: str, with_tools: bool = False) -> ApiMessageCompleteEvent:
    content = [ToolUseBlock(id="tu_x", name="bash", input={})] if with_tools else [TextBlock(text=text)]
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=content),
        usage=UsageSnapshot(input_tokens=10, output_tokens=5),
        stop_reason=None,
    )


def test_semantic_cache_store_and_lookup_hit(tmp_path: Path):
    from openharness.services.llm_pipeline import SemanticCache
    from openharness.services.persistent_store import PersistentStore

    cache = SemanticCache(PersistentStore(tmp_path / "s.sqlite3"), threshold=0.85)
    req = _request("write a scene about a red train crossing a mountain pass")
    assert cache.lookup(req) is None and cache.misses == 1
    cache.store_response(req, _complete_event("here is your scene"))
    # Slightly reworded prompt should still hit (that is the point).
    reworded = _request("write a scene about a red train crossing a mountain pass!")
    hit = cache.lookup(reworded)
    assert hit is not None
    assert hit.message.text == "here is your scene"
    assert cache.hits == 1
    # A topically different prompt misses.
    different = _request("debug this python stack trace about database connections")
    assert cache.lookup(different) is None


def test_semantic_cache_never_stores_tool_calls(tmp_path: Path):
    from openharness.services.llm_pipeline import SemanticCache
    from openharness.services.persistent_store import PersistentStore

    cache = SemanticCache(PersistentStore(tmp_path / "s.sqlite3"))
    req = _request("run the tool please")
    cache.store_response(req, _complete_event("calling", with_tools=True))
    assert cache.lookup(req) is None  # tool-call responses are never replayed


class _FakeTransport:
    """Minimal SupportsStreamingMessages for pipeline tests."""

    def __init__(self, *, delay: float = 0.0, fail_first_with=None) -> None:
        self.delay = delay
        self.fail_first_with = fail_first_with
        self.requests: list[ApiMessageRequest] = []
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        self.requests.append(request)
        if self.fail_first_with is not None and self.calls == 1:
            raise self.fail_first_with
        if self.delay:
            await asyncio.sleep(self.delay)
        yield ApiTextDeltaEvent(text="hi")
        yield _complete_event("hi")


@pytest.mark.asyncio
async def test_key_rotating_client_rotates_on_rate_limit():
    from openharness.api.errors import RateLimitFailure
    from openharness.services.llm_pipeline import KeyRotatingClient

    failing = _FakeTransport(fail_first_with=RateLimitFailure("429 too many requests"))
    working = _FakeTransport()
    clients = [failing, working]
    client = KeyRotatingClient(["key-1", "key-2"], lambda key: clients.pop(0))
    events = []
    async for event in client.stream_message(_request("hello")):
        events.append(event)
    # The first key rate-limited, the second served the request.
    assert failing.calls == 1 and working.calls == 1
    assert client.stats()["requests_served"] == 1
    assert client.stats()["rotations_on_rate_limit"] == 1
    assert any(getattr(e, "message", "").startswith("API key #1") for e in events)
    await client.close()


@pytest.mark.asyncio
async def test_key_rotating_client_raises_after_pool_exhausted():
    from openharness.api.errors import RateLimitFailure
    from openharness.services.llm_pipeline import KeyRotatingClient

    client = KeyRotatingClient(
        ["k1", "k2"], lambda key: _FakeTransport(fail_first_with=RateLimitFailure("429"))
    )
    with pytest.raises(Exception) as excinfo:
        async for _ in client.stream_message(_request("hello")):
            pass
    assert "429" in str(excinfo.value) or "rate" in str(excinfo.value).lower()
    await client.close()


@pytest.mark.asyncio
async def test_racing_client_first_content_wins():
    from openharness.services.llm_pipeline import RacingClient

    slow = _FakeTransport(delay=0.3)
    fast = _FakeTransport()
    race = RacingClient([("slow-route", slow), ("fast-route", fast)])
    text = []
    async for event in race.stream_message(_request("race!")):
        if isinstance(event, ApiTextDeltaEvent):
            text.append(event.text)
    assert text == ["hi"]
    assert race.races_won["fast-route"] == 1
    assert race.races_won["slow-route"] == 0
    await race.close()


def test_session_cost_tracker_pricing_and_thresholds():
    from openharness.services.llm_pipeline import SessionCostTracker

    tracker = SessionCostTracker(budget_usd=2.0, warn_ratio=0.3)
    # Sonnet pricing: $3/M in, $15/M out.
    assert tracker.estimate_cost_usd(
        "claude-sonnet-4", UsageSnapshot(input_tokens=1_000_000, output_tokens=0)
    ) == pytest.approx(3.0)
    # Cache-read tokens are 10x cheaper; cache-writes 25% pricier.
    assert tracker.estimate_cost_usd(
        "claude-sonnet-4", UsageSnapshot(input_tokens=0, cache_read_tokens=1_000_000)
    ) == pytest.approx(0.3)
    assert tracker.estimate_cost_usd(
        "claude-sonnet-4", UsageSnapshot(input_tokens=0, cache_write_tokens=1_000_000)
    ) == pytest.approx(3.75)

    alerts = []
    tracker.on_alert = alerts.append
    # Call 1: 200k in + 10k out = $0.75 -> crosses the 30% warn line ($0.60).
    warn = tracker.record(
        "claude-sonnet-4", UsageSnapshot(input_tokens=200_000, output_tokens=10_000)
    )
    assert warn is not None and warn.level == "warning"
    assert len(alerts) == 1
    # Call 2: +$0.30 = $1.05 -> still under the $2 budget, no repeat alerts.
    assert tracker.record("claude-sonnet-4", UsageSnapshot(input_tokens=100_000)) is None
    # Call 3: +$1.50 = $2.55 -> halt, exactly once.
    halt = tracker.record("claude-sonnet-4", UsageSnapshot(input_tokens=500_000))
    assert halt is not None and halt.level == "halt"
    assert tracker.halted
    assert tracker.record("claude-sonnet-4", UsageSnapshot(input_tokens=500_000)) is None
    assert len(alerts) == 2
    stats = tracker.stats()
    assert stats["calls"] == 4 and stats["halted"] is True


def test_session_cost_tracker_without_budget_never_alerts():
    from openharness.services.llm_pipeline import SessionCostTracker

    tracker = SessionCostTracker(budget_usd=None)
    assert tracker.record("claude-opus-4", UsageSnapshot(input_tokens=10_000_000)) is None
    assert not tracker.halted
    assert tracker.stats()["spent_usd"] > 0  # still counts spend


def test_speculative_capability_report_is_honest():
    from openharness.services.llm_pipeline import speculative_capability_report

    groq = speculative_capability_report(provider="groq")
    assert "active" in groq["speculative_decoding"]
    assert "No action" in groq["recommended_action"]
    local = speculative_capability_report(provider="vllm", base_url="http://localhost:8000/v1")
    assert "OFF" in local["speculative_decoding"]
    hosted = speculative_capability_report(provider="openai", base_url="https://api.openai.com/v1")
    assert "unknown" in hosted["speculative_decoding"]


@pytest.mark.asyncio
async def test_inference_pipeline_client_sets_prefix_caching_and_compresses():
    from openharness.services.llm_pipeline import (
        InferencePipelineClient,
        PromptCompressor,
    )

    transport = _FakeTransport()
    messages = _tool_convo()
    pipeline = InferencePipelineClient(
        transport,
        compressor=PromptCompressor(min_chars=1, keep_turns=2, tool_result_cap=500),
        prefix_caching=True,
    )
    request = ApiMessageRequest(model="claude-test", messages=messages)
    events = []
    async for event in pipeline.stream_message(request):
        events.append(event)
    # The transport saw the pipeline-adjusted request...
    seen = transport.requests[0]
    assert seen.prefix_caching is True  # #2 flag set by the pipeline
    assert len(seen.messages) == len(messages)
    stub = [b for b in seen.messages[2].content if isinstance(b, ToolResultBlock)][0]
    assert "elided" in stub.content  # #3 compression applied to the copy
    # ...and stats recorded the request.
    assert pipeline.stats.as_dict()["requests"] == 1
    await pipeline.close()


@pytest.mark.asyncio
async def test_inference_pipeline_semantic_cache_hit_skips_transport(tmp_path: Path):
    from openharness.services.llm_pipeline import (
        InferencePipelineClient,
        SemanticCache,
    )
    from openharness.services.persistent_store import PersistentStore

    cache = SemanticCache(PersistentStore(tmp_path / "s.sqlite3"), threshold=0.85)
    transport = _FakeTransport()
    pipeline = InferencePipelineClient(transport, semantic_cache=cache)
    request = _request("write a scene about a lighthouse in a storm")

    async for _ in pipeline.stream_message(request):
        pass
    assert transport.calls == 1
    assert pipeline.stats.semantic_cache_hits == 0

    # Second, near-identical request: served from the cache.
    reworded = _request("write a scene about a lighthouse in a storm!")
    text = []
    async for event in pipeline.stream_message(reworded):
        if isinstance(event, ApiTextDeltaEvent):
            text.append(event.text)
    assert transport.calls == 1  # never hit the network
    assert pipeline.stats.semantic_cache_hits == 1
    assert text == ["hi"]
    await pipeline.close()


def test_wrap_client_identity_without_inference_settings():
    from openharness.services.llm_pipeline import (
        UsageTrackingClient,
        wrap_client_with_inference_pipeline,
    )

    transport = _FakeTransport()
    settings = SimpleNamespace()  # no .inference attribute
    wrapped = wrap_client_with_inference_pipeline(settings, transport)
    # Cost tracking is always on (pure observability): the only wrapper
    # applied with no inference settings is the usage tracker, which is
    # a transparent passthrough to the original transport.
    if isinstance(wrapped, UsageTrackingClient):
        assert wrapped._inner is transport
    else:
        assert wrapped is transport


def test_wrap_client_applies_pipeline_when_features_on():
    from openharness.services.llm_pipeline import (
        InferencePipelineClient,
        UsageTrackingClient,
        wrap_client_with_inference_pipeline,
    )

    transport = _FakeTransport()
    settings = SimpleNamespace(inference=SimpleNamespace(prefix_caching=True))
    wrapped = wrap_client_with_inference_pipeline(settings, transport)
    assert isinstance(wrapped, InferencePipelineClient)
    # The usage tracker sits between the pipeline wrapper and the
    # transport (inside the semantic cache so free replays aren't billed).
    if isinstance(wrapped._inner, UsageTrackingClient):
        assert wrapped._inner._inner is transport
    else:
        assert wrapped._inner is transport


# ======================================================================
# 5. API client prefix caching (#2) + usage accounting
# ======================================================================

class _Captured(Exception):
    pass


# The retry wrapper in stream_message translates unknown exceptions into
# RequestFailure; both signal that params were captured successfully.
from openharness.api.errors import RequestFailure  # noqa: E402


@pytest.mark.asyncio
async def test_anthropic_prefix_caching_adds_cache_control_breakpoints():
    client = AnthropicApiClient(api_key="test-key")
    captured: dict = {}

    class _CM:
        async def __aenter__(self):
            raise _Captured()

        async def __aexit__(self, *args):
            return False

    def _capture(**params):
        captured.update(params)
        return _CM()

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_capture))
    request = ApiMessageRequest(
        model="claude-test",
        messages=_convo("hello"),
        system_prompt="You are helpful.",
        tools=[
            {"name": "bash", "description": "run", "input_schema": {}},
            {"name": "read_file", "description": "read", "input_schema": {}},
        ],
        prefix_caching=True,
    )
    with pytest.raises((RequestFailure, _Captured)):
        async for _ in client.stream_message(request):
            pass
    # System prompt became a block list with an ephemeral breakpoint.
    assert isinstance(captured["system"], list)
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Only the LAST tool definition carries the second breakpoint.
    tools = captured["tools"]
    assert len(tools) == 2
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[0]


@pytest.mark.asyncio
async def test_anthropic_without_prefix_caching_keeps_plain_params():
    client = AnthropicApiClient(api_key="test-key")
    captured: dict = {}

    class _CM:
        async def __aenter__(self):
            raise _Captured()

        async def __aexit__(self, *args):
            return False

    def _capture(**params):
        captured.update(params)
        return _CM()

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_capture))
    request = ApiMessageRequest(
        model="claude-test",
        messages=_convo("hello"),
        system_prompt="You are helpful.",
        tools=[{"name": "bash", "description": "run", "input_schema": {}}],
        prefix_caching=False,
    )
    with pytest.raises((RequestFailure, _Captured)):
        async for _ in client.stream_message(request):
            pass
    assert isinstance(captured["system"], str)
    assert all("cache_control" not in t for t in captured["tools"])


def test_usage_snapshot_cache_token_fields():
    usage = UsageSnapshot(
        input_tokens=100, output_tokens=50, cache_write_tokens=25, cache_read_tokens=200
    )
    assert usage.cache_write_tokens == 25
    assert usage.cache_read_tokens == 200
    # Defaults keep every existing constructor working.
    assert UsageSnapshot().total_tokens == 0
    assert UsageSnapshot(input_tokens=1, output_tokens=2).total_tokens == 3


# ======================================================================
# 6. QA pipeline units (#7-#10)
# ======================================================================

def test_classify_scene_simple_vs_complex(tmp_path: Path):
    from openharness.services.qa_pipeline import classify_scene

    simple = tmp_path / "simple.py"
    simple.write_text("draw('bg.png')\ntext('hello')\n")
    verdict = classify_scene(simple)
    assert verdict.is_simple
    assert verdict.level == "simple"

    heavy = tmp_path / "heavy.py"
    heavy.write_text(
        "\n".join(
            [
                "import numpy",
                "import math",
                "import json",
                "import os",
                *[f"keyframe({i})" for i in range(6)],   # animation keywords
                "mask(alpha=0.5)",
                "chroma(key='0x00FF00')",
                "parallax(depth=3)",
                "transition('wipe')",
                *[f"line_{i} = {i}" for i in range(90)],  # > 80 code lines
            ]
        )
    )
    verdict = classify_scene(heavy)
    assert verdict.level in ("standard", "complex")
    assert not verdict.is_simple
    assert verdict.code_lines > 80


def test_failure_predictor_learns_rates(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore
    from openharness.services.qa_pipeline import FailurePredictor

    predictor = FailurePredictor(PersistentStore(tmp_path / "qa.sqlite3"))
    # layout_overlap fails 3/4 on simple/hyperframes; text_too_small 1/4.
    for _ in range(3):
        predictor.record_failure(category="layout_overlap", complexity="simple", engine="hyperframes")
    predictor.record_success(complexity="simple", engine="hyperframes", category="layout_overlap")
    predictor.record_failure(category="text_too_small", complexity="simple", engine="hyperframes")
    for _ in range(3):
        predictor.record_success(complexity="simple", engine="hyperframes", category="text_too_small")

    predictions = predictor.predict(complexity="simple", engine="hyperframes", top_k=2)
    assert predictions[0]["category"] == "layout_overlap"
    assert predictions[0]["failure_rate"] == 0.75
    assert predictions[0]["samples"] == 4
    assert {p["category"] for p in predictions} == {"layout_overlap", "text_too_small"}
    # Never-seen profile -> no honest prediction.
    assert predictor.predict(complexity="complex", engine="blender") == []


def test_parse_verdict_payload_valid_and_invalid():
    from openharness.services.qa_pipeline import parse_verdict_payload

    import json as _json

    good = _json.dumps(
        {
            "verdicts": [
                {"scene": "scene_001", "passed": True, "score": 0.9, "issues": []},
                {"scene": "scene_002", "passed": False, "score": 0.4,
                 "issues": [{"category": "text_too_small", "detail": "6px font"}]},
            ]
        }
    )
    verdicts = parse_verdict_payload(good, ["scene_001", "scene_002"])
    assert verdicts is not None and len(verdicts) == 2
    assert verdicts[0].passed and verdicts[0].scene == "scene_001"
    assert not verdicts[1].passed and verdicts[1].issues[0]["category"] == "text_too_small"
    assert parse_verdict_payload("not json at all", ["scene_001"]) is None


def test_preflight_checks_find_nondeterminism_and_missing_assets(tmp_path: Path):
    from openharness.services.qa_pipeline import FailurePredictor
    from openharness.services.persistent_store import PersistentStore

    predictor = FailurePredictor(PersistentStore(tmp_path / "qa.sqlite3"))
    scene = tmp_path / "scene_001.py"
    (tmp_path / "real.png").write_bytes(b"img")
    scene.write_text(
        "const jitter = Math.random();\nbg('real.png')\nbg('ghost.png')\n",
        encoding="utf-8",
    )
    findings = predictor.preflight_checks(scene, engine="hyperframes")
    categories = {f["category"] for f in findings}
    assert "non_determinism" in categories
    assert "missing_asset" in categories
    nondet = next(f for f in findings if f["category"] == "non_determinism")
    assert "Math.random" in nondet["detail"] or "math.random" in nondet["detail"]
    missing = next(f for f in findings if f["category"] == "missing_asset")
    assert "ghost.png" in missing["detail"]


# ======================================================================
# 7. Render pipeline units (#24/#25 + command building)
# ======================================================================

def _write_scene(path: Path, code: str) -> Path:
    path.write_text(code, encoding="utf-8")
    return path


def test_dependency_graph_parallel_waves(tmp_path: Path):
    from openharness.services.render_pipeline import SceneDependencyGraph

    a = _write_scene(tmp_path / "scene_001.py", "// opener\n")
    b = _write_scene(tmp_path / "scene_002.py", "// depends_on: scene_001\n")
    c = _write_scene(tmp_path / "scene_003.py", "# depends_on: scene_002\n")
    d = _write_scene(tmp_path / "scene_004.py", "// independent parallel scene\n")
    graph = SceneDependencyGraph([a, b, c, d])
    waves = graph.render_waves()
    assert waves[0] == ["scene_001", "scene_004"]
    assert waves[1] == ["scene_002"]
    assert waves[2] == ["scene_003"]
    # Unknown deps (assets from earlier project stages) are ignored, not fatal.
    e = _write_scene(tmp_path / "scene_005.py", "// depends_on: precomputed_asset\n")
    graph2 = SceneDependencyGraph([e])
    assert graph2.render_waves() == [["scene_005"]]


def test_dependency_graph_cycle_raises(tmp_path: Path):
    from openharness.services.render_pipeline import SceneDependencyGraph

    a = _write_scene(tmp_path / "a.py", "// depends_on: b\n")
    b = _write_scene(tmp_path / "b.py", "// depends_on: a\n")
    with pytest.raises(ValueError, match="cycle"):
        SceneDependencyGraph([a, b]).render_waves()


@pytest.mark.asyncio
async def test_work_stealing_queue_fifo_and_steal():
    from openharness.services.render_pipeline import WorkStealingQueue

    queue = WorkStealingQueue(2, stealing=True)
    for item in ("j1", "j2", "j3"):
        await queue.put(item, worker_hint=0)
    assert queue.snapshot() == [3, 0]
    # Own deque: FIFO.
    assert await queue.get(0) == "j1"
    # Worker 1 steals from the BACK of worker 0's deque.
    assert await queue.get(1) == "j3"
    assert queue.steals == 1
    assert await queue.get(0) == "j2"
    # Empty + closed -> workers drain and get None.
    await queue.close()
    assert await queue.get(0) is None
    assert await queue.get(1) is None


@pytest.mark.asyncio
async def test_work_stealing_queue_put_goes_to_least_loaded():
    from openharness.services.render_pipeline import WorkStealingQueue

    queue = WorkStealingQueue(3, stealing=True)
    queue.put_nowait("a", worker_hint=1)
    await queue.put("b")  # least loaded -> worker 0 or 2
    assert queue.depth == 2
    assert sum(queue.snapshot()) == 2
    await queue.close()


def test_parse_resolution_and_fallback():
    from openharness.services.render_pipeline import parse_resolution

    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution("640X360") == (640, 360)
    assert parse_resolution("1280×720") == (1280, 720)
    assert parse_resolution("garbage") == (1280, 720)
    assert parse_resolution("") == (1280, 720)


def test_build_render_command_substitutes_everything(tmp_path: Path):
    from openharness.services.render_pipeline import build_render_command

    scene = _write_scene(tmp_path / "scene_001.py", "class Opener(Scene): pass")
    output = tmp_path / "renders" / "scene_001.mp4"
    argv = build_render_command(
        engine="hyperframes",
        scene_path=scene,
        output_path=output,
        tier="final",
        width=1920,
        height=1080,
        fps=30,
        codec_args="-c:v libx264 -crf 18",
        output_dir=tmp_path / "renders",
    )
    joined = " ".join(argv)
    assert str(scene) in joined
    assert str(output) in joined
    assert "1920" in joined and "1080" in joined
    assert "{scene}" not in joined and "{output}" not in joined  # all placeholders filled

    # A custom engine template exercising every placeholder (codec, fps,
    # quality, scene class, media dir) substitutes them all.
    custom = build_render_command(
        engine="custom",
        scene_path=scene,
        output_path=output,
        tier="review",
        width=1280,
        height=720,
        fps=24,
        codec_args="-c:v h264_nvenc",
        output_dir=tmp_path / "renders",
        templates={
            "custom": [
                "eng", "render", "{scene}", "{scene_class}",
                "--out", "{output}", "{width}x{height}", "--fps", "{fps}",
                "--codec", "{codec_args}", "--quality", "{quality}",
                "--media", "{media_dir}", "--input", "{input_args}",
                "--stem", "{output_stem}",
            ]
        },
    )
    custom_joined = " ".join(custom)
    assert "-c:v h264_nvenc" in custom_joined
    assert "1280x720" in custom_joined and "--fps 24" in custom_joined
    assert "Opener" in custom_joined
    assert "{codec_args}" not in custom_joined and "{input_args}" not in custom_joined


# ======================================================================
# 8. Server acceleration reporting (#41-#50)
# ======================================================================

def test_build_vllm_config_hopper_gpu():
    from openharness.services.server_accel import build_vllm_server_config

    config = build_vllm_server_config(
        model="Qwen/Qwen2.5-32B",
        gpu_count=2,
        gpu_memory_gb=80.0,
        gpu_name="NVIDIA H100 80GB",
        context_length=32768,
        enable_speculative=True,
        speculative_model="Qwen/Qwen2.5-0.5B",
    )
    argv = config["argv"]
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen2.5-32B"]
    assert "--tensor-parallel-size" in argv and argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--attention-backend" in argv and argv[argv.index("--attention-backend") + 1] == "FLASH_ATTN"
    assert "--kv-cache-dtype" in argv and argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--enable-chunked-prefill" in argv
    assert "--speculative-model" in argv
    items = " | ".join(e["item"] for e in config["explanations"])
    for wanted in ("#41", "#42", "#43", "#44", "#45", "#46", "#1"):
        assert wanted in items
    assert config["hardware"]["tensor_parallel"] == 2
    assert "vllm/vllm-openai" in config["docker_compose"]
    assert "localhost:8000/v1" in config["client_note"]


def test_build_vllm_config_pre_hopper_uses_xformers():
    from openharness.services.server_accel import build_vllm_server_config

    config = build_vllm_server_config(
        model="meta-llama/Llama-3-8B",
        gpu_count=1,
        gpu_memory_gb=15.0,
        gpu_name="NVIDIA T4",
    )
    argv = config["argv"]
    assert argv[argv.index("--attention-backend") + 1] == "XFORMERS"
    fa = next(e for e in config["explanations"] if "#43" in e["item"])
    assert "pre-Hopper" in fa["why"]


def test_write_server_config_writes_launch_files(tmp_path: Path):
    from openharness.services.server_accel import (
        build_vllm_server_config,
        write_server_config,
    )

    config = build_vllm_server_config(
        model="test/model", gpu_count=1, gpu_name="NVIDIA T4"
    )
    script = write_server_config(config, tmp_path)
    assert script.exists() and script.name == "run_vllm_server.sh"
    assert script.stat().st_mode & 0o111, "launch script must be executable"
    assert (tmp_path / "vllm-docker-compose.yml").exists()
    assert (tmp_path / "vllm-config.json").exists()
    assert "vllm serve test/model" in script.read_text(encoding="utf-8")


def test_detect_hardware_acceleration_is_measured_not_faked():
    from openharness.services.server_accel import detect_hardware_acceleration

    report = detect_hardware_acceleration()
    for family in ("nvlink", "rdma", "tpu", "zero_copy"):
        entry = report[family]
        assert isinstance(entry["available"], (bool, type(None)))
        assert entry["detail"], f"{family} needs an honest detail string"
    # zero-copy has no user-facing switch -> reported unknown, never 'enabled'.
    assert report["zero_copy"]["available"] is None
    assert "gpu_count" in report and "advisory" in report


# ======================================================================
# 9. Asset prefetch (#22/#23)
# ======================================================================

def test_extract_reads_parses_voiceover_blocks():
    from openharness.services.asset_prefetch import _extract_reads

    code = (
        "// scene\nREADS: [\"First line.\", \"Second line.\"]\n"
        "READS: []\nREADS: not-json\n"
    )
    assert _extract_reads(code) == ["First line.", "Second line."]
    assert _extract_reads("no reads here") == []


def test_audio_generator_requires_real_credentials():
    from openharness.services.asset_prefetch import AsyncAudioGenerator

    with pytest.raises(ValueError):
        AsyncAudioGenerator(base_url="", api_key="")
    with pytest.raises(ValueError):
        AsyncAudioGenerator(base_url="https://tts.example.com/v1", api_key="")


@pytest.mark.asyncio
async def test_prefetch_warms_tool_cache_for_assets(tmp_path: Path):
    from openharness.services.asset_prefetch import AssetPrefetcher
    from openharness.services.tool_cache import get_tool_cache, reset_tool_cache

    asset = tmp_path / "bg.png"
    asset.write_text("pretend-image-bytes", encoding="utf-8")
    scene = _write_scene(tmp_path / "scene_001.py", "draw('bg.png')\nREADS: [\"Hello.\"]\n")

    # Enable the tool cache so prefetch has somewhere to warm.
    from openharness.services.tool_cache import enable_tool_cache

    enable_tool_cache(max_entries=10, ttl_seconds=60)
    try:
        prefetcher = AssetPrefetcher()  # no audio generator -> audio skipped
        report = await prefetcher.prefetch_scenes([scene], cwd=tmp_path)
        assert report.files_read == [str(asset)]
        assert report.audio_generated == []
        cache = get_tool_cache()
        assert cache is not None
        cached = cache.get(
            "read_file", {"file_path": str(asset), "offset": 0, "limit": 200}
        )
        assert cached is not None
        assert "pretend-image-bytes" in cached["output"]
    finally:
        reset_tool_cache()


@pytest.mark.asyncio
async def test_prefetch_skips_unreadable_scenes(tmp_path: Path):
    from openharness.services.asset_prefetch import AssetPrefetcher

    prefetcher = AssetPrefetcher()
    report = await prefetcher.prefetch_scenes(
        [tmp_path / "missing_scene.py"], cwd=tmp_path
    )
    assert len(report.skipped) == 1
    assert "unreadable" in report.skipped[0]


# ======================================================================
# 10. Tool registry surface + cache_stats / server_config tools
# ======================================================================

def test_default_registry_contains_all_pipeline_tools():
    from openharness.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    names = set(registry._tools)
    expected = {
        "render_scenes", "render_queue_status", "qa_review", "oh_control",
        "checkpoint", "progress", "audio_gen", "server_config", "cache_stats",
    }
    missing = expected - names
    assert not missing, f"pipeline tools missing from registry: {missing}"


@pytest.mark.asyncio
async def test_cache_stats_tool_status_and_clear(tmp_path: Path):
    from openharness.services.persistent_store import get_persistent_store
    from openharness.services.render_cache import RenderCache
    from openharness.tools.pipeline_tools import CacheStatsInput, CacheStatsTool

    store = get_persistent_store()
    output = tmp_path / "out.mp4"
    output.write_bytes(b"v")
    cache = RenderCache(store)
    cache.store("fp", output_path=str(output), scene="s", engine="hyperframes",
                tier="draft", duration_seconds=1.0)
    tool = CacheStatsTool()
    ctx = ToolExecutionContext(cwd=tmp_path)

    status = await tool.execute(CacheStatsInput(action="status"), ctx)
    assert not status.is_error and "cache_stats" in status.output
    assert "hits" in status.output

    cleared = await tool.execute(CacheStatsInput(action="clear_render"), ctx)
    assert "Cleared 1" in cleared.output
    assert cache.lookup("fp") is None


@pytest.mark.asyncio
async def test_server_config_tool_writes_launch_script(tmp_path: Path):
    from openharness.tools.pipeline_tools import ServerConfigInput, ServerConfigTool

    tool = ServerConfigTool()
    result = await tool.execute(
        ServerConfigInput(
            model="Qwen/Qwen2.5-7B",
            output_dir=str(tmp_path / "server"),
            speculative_model="Qwen/Qwen2.5-0.5B",
        ),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert not result.is_error
    script = tmp_path / "server" / "run_vllm_server.sh"
    assert script.exists()
    assert "Qwen/Qwen2.5-7B" in result.output
    hardware = await tool.execute(
        ServerConfigInput(action="hardware"), ToolExecutionContext(cwd=tmp_path)
    )
    assert "nvlink" in hardware.output and "advisory" in hardware.output


# ======================================================================
# 11. Runtime wiring helpers
# ======================================================================

def test_build_primary_with_overrides_openai(tmp_path: Path):
    from openharness.api.openai_client import OpenAICompatibleClient
    from openharness.ui.runtime import _build_primary_with_overrides

    settings = Settings(
        api_key="original-key",
        api_format="openai",
        provider="openai",
        base_url="https://api.example.com/v1",
    )
    settings.active_profile = "openai-compatible"
    client = _build_primary_with_overrides(settings, "pool-key-2")
    assert isinstance(client, OpenAICompatibleClient)
    assert client._client.api_key == "pool-key-2"


def test_build_primary_with_overrides_anthropic():
    from openharness.ui.runtime import _build_primary_with_overrides

    settings = Settings(
        api_key="original",
        api_format="anthropic",
        provider="anthropic_claude",
        base_url="https://api.anthropic.com",
    )
    client = _build_primary_with_overrides(settings, "pool-key-9")
    assert isinstance(client, AnthropicApiClient)
    assert client._api_key == "pool-key-9"


def test_inference_settings_defaults_are_conservative():
    from openharness.config.settings import InferenceSettings

    inference = InferenceSettings()
    assert inference.prefix_caching is True  # pure win, on by default
    assert inference.prompt_compression is False  # lossy, opt-in
    assert inference.semantic_cache is False  # approximate, opt-in
    assert inference.cross_provider_racing is False  # costs tokens, opt-in
    assert inference.api_key_pool == []
