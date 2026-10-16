"""Real tests for the final wave of 23 optimizations.

Each test exercises one item and asserts observable behaviour.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest


# ============================================================
# #1 Prefix Cache
# ============================================================

def test_item_1_prefix_cache_creates_entry_on_first_call():
    from openharness.services.inference_optimizations import PrefixCache
    cache = PrefixCache()
    entry = cache.get_or_create("Hello world prefix")
    assert entry.prefix_text == "Hello world prefix"
    assert entry.use_count == 1
    assert entry.token_count > 0


def test_item_1_prefix_cache_reuses_entry_on_repeat():
    from openharness.services.inference_optimizations import PrefixCache
    cache = PrefixCache()
    cache.get_or_create("same prefix")
    entry = cache.get_or_create("same prefix")
    assert entry.use_count == 2


def test_item_1_prefix_cache_hit_rate():
    from openharness.services.inference_optimizations import PrefixCache
    cache = PrefixCache()
    cache.get_or_create("prefix A")
    cache.get_or_create("prefix A")  # hit
    cache.get_or_create("prefix B")  # miss
    assert cache.hit_rate > 0


# ============================================================
# #2 Multi-Token Prediction params
# ============================================================

def test_item_2_mtp_params_returns_dict():
    from openharness.services.inference_optimizations import get_mtp_params
    params = get_mtp_params("anthropic")
    assert isinstance(params, dict)


def test_item_2_mtp_params_unknown_provider_returns_empty():
    from openharness.services.inference_optimizations import get_mtp_params
    assert get_mtp_params("unknown") == {}


# ============================================================
# #3 Tool Call Batching
# ============================================================

def test_item_3_should_batch_read_only_tools():
    from openharness.services.inference_optimizations import should_batch_tool_calls
    calls = [{"name": "read_file"}, {"name": "grep"}, {"name": "glob"}]
    assert should_batch_tool_calls(calls) is True


def test_item_3_should_not_batch_mutating_tools():
    from openharness.services.inference_optimizations import should_batch_tool_calls
    calls = [{"name": "read_file"}, {"name": "write_file"}]
    assert should_batch_tool_calls(calls) is False


def test_item_3_should_not_batch_single_call():
    from openharness.services.inference_optimizations import should_batch_tool_calls
    assert should_batch_tool_calls([{"name": "read_file"}]) is False


# ============================================================
# #4 Streaming Tool Execution
# ============================================================

def test_item_4_can_stream_read_only_first_tool():
    from openharness.services.inference_optimizations import can_stream_tool_execution
    calls = [{"name": "read_file"}, {"name": "grep"}]
    assert can_stream_tool_execution(calls) is True


def test_item_4_cannot_stream_write_first():
    from openharness.services.inference_optimizations import can_stream_tool_execution
    calls = [{"name": "write_file"}, {"name": "read_file"}]
    assert can_stream_tool_execution(calls) is False


# ============================================================
# #5 Prompt Compression
# ============================================================

def test_item_5_compress_removes_stopwords():
    from openharness.services.inference_optimizations import compress_prompt
    original = "The quick brown fox jumps over the lazy dog and the cat"
    compressed = compress_prompt(original, target_ratio=0.5)
    assert len(compressed) < len(original)
    # "fox" and "dog" should survive (not stopwords)
    assert "fox" in compressed or "dog" in compressed


def test_item_5_compress_preserves_code():
    from openharness.services.inference_optimizations import compress_prompt
    original = "Use the file src/openharness/cli.py to run the command"
    compressed = compress_prompt(original, target_ratio=0.5)
    # File paths should survive compression
    assert "src/openharness/cli.py" in compressed or "cli.py" in compressed


def test_item_5_compress_preserves_numbers():
    from openharness.services.inference_optimizations import compress_prompt
    original = "Set the duration to 8.5 seconds and the size to 1024 pixels"
    compressed = compress_prompt(original, target_ratio=0.5)
    assert "8.5" in compressed
    assert "1024" in compressed


def test_item_5_compress_preserves_hex_colors():
    from openharness.services.inference_optimizations import compress_prompt
    original = "Use color #1A1A2E for the background"
    compressed = compress_prompt(original, target_ratio=0.5)
    assert "#1A1A2E" in compressed


# ============================================================
# #7 Semantic Cache
# ============================================================

def test_item_7_semantic_cache_hit_on_similar_prompt():
    from openharness.services.semantic_cache import SemanticCache
    cache = SemanticCache(similarity_threshold=0.5)
    cache.store("Generate a title card for the video", "code_for_title_card")
    # Similar prompt should hit
    result = cache.lookup("Generate a title card for the movie")
    assert result == "code_for_title_card"


def test_item_7_semantic_cache_miss_on_different_prompt():
    from openharness.services.semantic_cache import SemanticCache
    cache = SemanticCache(similarity_threshold=0.8)
    cache.store("Generate a title card", "title_code")
    result = cache.lookup("Render a 3D Blender animation of a rotating cube")
    assert result is None


def test_item_7_semantic_cache_stats():
    from openharness.services.semantic_cache import SemanticCache
    cache = SemanticCache()
    cache.store("prompt A", "response A")
    cache.lookup("prompt A")  # hit
    cache.lookup("prompt B")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


# ============================================================
# #8 Disk-Based Model Cache
# ============================================================

def test_item_8_disk_cache_persists_across_instances(tmp_path: Path):
    from openharness.services.disk_model_cache import DiskModelCache
    cache1 = DiskModelCache(cache_dir=tmp_path / "model_cache")
    key = DiskModelCache.make_key(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    cache1.put(key, "cached response")
    # New instance pointing at the same dir
    cache2 = DiskModelCache(cache_dir=tmp_path / "model_cache")
    result = cache2.get(key)
    assert result == "cached response"


def test_item_8_disk_cache_miss_on_different_key():
    from openharness.services.disk_model_cache import DiskModelCache
    cache = DiskModelCache(cache_dir=tmp_path if False else Path("/tmp/test_disk_cache_8"))
    key1 = DiskModelCache.make_key(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])
    key2 = DiskModelCache.make_key(model="gpt-4o", messages=[{"role": "user", "content": "world"}])
    cache.put(key1, "response1")
    assert cache.get(key1) == "response1"
    assert cache.get(key2) is None  # different key → miss


# ============================================================
# #9 Scene Similarity Detection
# ============================================================

def test_item_9_similar_scenes_detected():
    from openharness.services.semantic_cache import detect_scene_similarity
    result = detect_scene_similarity(
        "Stat card showing revenue data",
        "Stat card showing profit data",
    )
    assert result.is_similar is True
    assert result.similarity > 0.5


def test_item_9_different_scenes_not_similar():
    from openharness.services.semantic_cache import detect_scene_similarity
    result = detect_scene_similarity(
        "Title card with text",
        "3D Blender animation of rotating cube",
    )
    assert result.is_similar is False


# ============================================================
# #11 Draft-First Rendering
# ============================================================

def test_item_11_should_render_draft_on_early_iterations():
    from openharness.services.rendering_optimizations import should_render_draft
    assert should_render_draft(1) is True
    assert should_render_draft(2) is True


def test_item_11_should_not_render_draft_on_late_iterations():
    from openharness.services.rendering_optimizations import should_render_draft
    assert should_render_draft(5) is False


# ============================================================
# #12 Quality-Based Early Exit
# ============================================================

def test_item_12_early_exit_when_quality_high():
    from openharness.services.rendering_optimizations import check_quality_early_exit
    scores = [0.95] * 15  # all high quality
    result = check_quality_early_exit(15, 100, scores, threshold=0.9)
    assert result.should_stop is True


def test_item_12_no_early_exit_when_quality_low():
    from openharness.services.rendering_optimizations import check_quality_early_exit
    scores = [0.5] * 15  # low quality
    result = check_quality_early_exit(15, 100, scores, threshold=0.9)
    assert result.should_stop is False


def test_item_12_no_check_before_min_frames():
    from openharness.services.rendering_optimizations import check_quality_early_exit
    result = check_quality_early_exit(5, 100, [0.95] * 5, min_frames_checked=10)
    assert result.should_stop is False


# ============================================================
# #13 GPU Memory Pooling
# ============================================================

def test_item_13_gpu_pool_acquire_and_release():
    from openharness.services.rendering_optimizations import GPUMemoryPool
    pool = GPUMemoryPool(total_buffers=4, buffer_size_mb=512)
    buf = pool.acquire()
    assert buf is not None
    assert pool.available == 3
    pool.release(buf)
    assert pool.available == 4


def test_item_13_gpu_pool_exhausted_returns_none():
    from openharness.services.rendering_optimizations import GPUMemoryPool
    pool = GPUMemoryPool(total_buffers=2, buffer_size_mb=512)
    pool.acquire()
    pool.acquire()
    assert pool.acquire() is None  # exhausted


def test_item_13_recommend_pool_size():
    from openharness.services.rendering_optimizations import recommend_gpu_pool_size
    # 16GB GPU, 512MB per buffer → ~25 buffers
    size = recommend_gpu_pool_size(16_384, per_buffer_mb=512)
    assert size >= 20


# ============================================================
# #14 Async Audio Generation
# ============================================================

@pytest.mark.asyncio
async def test_item_14_async_audio_queue_processes_jobs():
    from openharness.services.async_optimizations import AsyncAudioQueue, AudioGenJob
    queue = AsyncAudioQueue(generate_fn=default_audio_fn_wrapper, max_concurrent=2)
    await queue.start()
    try:
        await queue.enqueue(AudioGenJob(scene_id="s1", text="Hello"))
        await queue.enqueue(AudioGenJob(scene_id="s2", text="World"))
        results = await queue.wait_all(timeout=5.0)
        assert results["s1"].state == "completed"
        assert results["s2"].state == "completed"
    finally:
        await queue.stop()


async def default_audio_fn_wrapper(job):
    await asyncio.sleep(0.01)
    return f"audio/{job.scene_id}.wav"


# ============================================================
# #15 Pre-Warmed Sessions
# ============================================================

@pytest.mark.asyncio
async def test_item_15_session_pool_warm_up():
    from openharness.services.async_optimizations import SessionPool, PreWarmedSession
    import time

    async def warm_fn():
        return PreWarmedSession(session_id="pre-warmed", created_at=time.time())

    pool = SessionPool(warm_fn=warm_fn, pool_size=2)
    await pool.warm_up()
    assert pool.available == 2
    session = await pool.acquire()
    assert session is not None
    assert session.in_use is True
    pool.release(session)
    assert pool.available == 2


# ============================================================
# #17 Review Batching
# ============================================================

def test_item_17_review_batches_split_correctly():
    from openharness.services.advanced_optimizations import create_review_batches
    ids = [f"scene_{i:03d}" for i in range(12)]
    codes = [f"// code {i}" for i in range(12)]
    batches = create_review_batches(ids, codes, batch_size=5)
    assert len(batches) == 3  # ceil(12/5) = 3
    assert len(batches[0].scene_ids) == 5
    assert len(batches[1].scene_ids) == 5
    assert len(batches[2].scene_ids) == 2


# ============================================================
# #18 Failure Prediction
# ============================================================

def test_item_18_failure_prediction_for_complex_scene():
    from openharness.services.advanced_optimizations import predict_failure_risk
    result = predict_failure_risk(
        "Complex 3D physics simulation with particles",
        historical_failure_rate={"3d_animation": 0.4},
    )
    assert result.failure_probability > 0.3
    assert result.recommended_strictness >= 2


def test_item_18_failure_prediction_for_simple_scene():
    from openharness.services.advanced_optimizations import predict_failure_risk
    result = predict_failure_risk("Simple title card with text")
    assert result.failure_probability < 0.3
    assert result.recommended_strictness == 1


# ============================================================
# #19 Compression-Aware Batching
# ============================================================

def test_item_19_high_compression_allows_larger_batch():
    from openharness.services.advanced_optimizations import compute_compression_aware_batch_size
    # 50% compression → should roughly double the batch
    size = compute_compression_aware_batch_size(5, 0.5)
    assert size > 5


def test_item_19_low_compression_keeps_small_batch():
    from openharness.services.advanced_optimizations import compute_compression_aware_batch_size
    # 90% ratio (barely compressed) → no increase
    size = compute_compression_aware_batch_size(5, 0.9)
    assert size <= 6  # minimal increase


def test_item_19_respects_max_size():
    from openharness.services.advanced_optimizations import compute_compression_aware_batch_size
    size = compute_compression_aware_batch_size(5, 0.3, max_size=10)
    assert size <= 10


# ============================================================
# #20 Cross-Provider Racing
# ============================================================

@pytest.mark.asyncio
async def test_item_20_race_returns_fastest_provider():
    from openharness.services.advanced_optimizations import race_providers

    async def fast_provider(req):
        await asyncio.sleep(0.01)
        return "fast response"

    async def slow_provider(req):
        await asyncio.sleep(1.0)
        return "slow response"

    result = await race_providers(
        "request",
        {"fast": fast_provider, "slow": slow_provider},
        timeout=5.0,
    )
    assert result.winner == "fast"
    assert result.response == "fast response"
    assert "slow" in result.losers


@pytest.mark.asyncio
async def test_item_20_race_timeout_when_all_fail():
    from openharness.services.advanced_optimizations import race_providers

    async def failing_provider(req):
        raise RuntimeError("failed")

    with pytest.raises(TimeoutError):
        await race_providers("req", {"p": failing_provider}, timeout=0.5)


# ============================================================
# #23 Cost Governor
# ============================================================

def test_item_23_cost_governor_tracks_spending():
    from openharness.tools.missing_tools import CostGovernor
    gov = CostGovernor(budget_usd=10.0)
    gov.record_spend(3.0, call_type="scene_gen")
    assert gov.spent == 3.0
    assert gov.remaining == 7.0
    assert gov.should_stop() is False


def test_item_23_cost_governor_halts_when_budget_exceeded():
    from openharness.tools.missing_tools import CostGovernor
    gov = CostGovernor(budget_usd=1.0)
    gov.record_spend(0.5)
    assert gov.should_stop() is False
    gov.record_spend(0.6)  # total 1.1 > 1.0
    assert gov.should_stop() is True


def test_item_23_cost_governor_stats():
    from openharness.tools.missing_tools import CostGovernor
    gov = CostGovernor(budget_usd=5.0)
    gov.record_spend(1.0)
    gov.record_spend(2.0)
    stats = gov.stats()
    assert stats["spent_usd"] == 3.0
    assert stats["budget_usd"] == 5.0
    assert stats["total_calls"] == 2
    assert stats["utilization"] == 0.6


@pytest.mark.asyncio
async def test_item_23_cost_governor_tool_check(tmp_path: Path):
    from openharness.tools.missing_tools import CostGovernorTool, CostGovernorToolInput
    from openharness.tools.base import ToolExecutionContext
    CostGovernorTool._governor = None  # reset
    tool = CostGovernorTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(CostGovernorToolInput(action="check"), ctx)
    assert not result.is_error
    assert "Budget OK" in result.output


# ============================================================
# #24 Audio QA Tool
# ============================================================

@pytest.mark.asyncio
async def test_item_24_audio_qa_checks_existing_file(tmp_path: Path):
    from openharness.tools.missing_tools import AudioQATool, AudioQAToolInput
    from openharness.tools.base import ToolExecutionContext
    audio_file = tmp_path / "narration.wav"
    audio_file.write_bytes(b"\x00" * 1000)
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(AudioQAToolInput(audio_path="narration.wav"), ctx)
    assert "Audio QA" in result.output
    assert "Overall:" in result.output


@pytest.mark.asyncio
async def test_item_24_audio_qa_missing_file(tmp_path: Path):
    from openharness.tools.missing_tools import AudioQATool, AudioQAToolInput
    from openharness.tools.base import ToolExecutionContext
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(AudioQAToolInput(audio_path="nonexistent.wav"), ctx)
    assert result.is_error
    assert "not found" in result.output


# ============================================================
# #25 Determinism Checker
# ============================================================

@pytest.mark.asyncio
async def test_item_25_determinism_checker_passes_for_consistent_code():
    from openharness.tools.missing_tools import DeterminismCheckerTool, DeterminismCheckerToolInput
    from openharness.tools.base import ToolExecutionContext
    tool = DeterminismCheckerTool()
    ctx = ToolExecutionContext(cwd=tmp_path if False else Path("/tmp"))
    result = await tool.execute(
        DeterminismCheckerToolInput(scene_code="function render() { return 42; }"),
        ctx,
    )
    assert not result.is_error
    assert "deterministic" in result.output.lower()


# ============================================================
# #26 Contact Sheet Generator
# ============================================================

@pytest.mark.asyncio
async def test_item_26_contact_sheet_generates_manifest(tmp_path: Path):
    from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput
    from openharness.tools.base import ToolExecutionContext
    # Create fake scene files
    for i in range(6):
        (tmp_path / f"scene_{i:03d}.mp4").write_bytes(b"\x00" * 100)
    tool = ContactSheetTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        ContactSheetToolInput(
            scene_files=[f"scene_{i:03d}.mp4" for i in range(6)],
            output_path="contact_sheet.png",
            columns=3,
        ),
        ctx,
    )
    assert not result.is_error
    assert "Contact Sheet" in result.output
    assert "6" in result.output  # 6 scenes
    manifest_path = tmp_path / "contact_sheet.json"
    assert manifest_path.is_file()


# ============================================================
# #27 Engine Router
# ============================================================

def test_item_27_engine_router_3d_to_blender():
    from openharness.tools.missing_tools import route_engine
    engine = route_engine("3D animation of a rotating cube", ["blender", "manim", "hyperframes"])
    assert engine == "blender"


def test_item_27_engine_router_math_to_manim():
    from openharness.tools.missing_tools import route_engine
    engine = route_engine("Math equation animation with graph", ["manim", "hyperframes"])
    assert engine == "manim"


def test_item_27_engine_router_default_to_hyperframes():
    from openharness.tools.missing_tools import route_engine
    engine = route_engine("Title card with text overlay", ["hyperframes", "manim"])
    assert engine == "hyperframes"


@pytest.mark.asyncio
async def test_item_27_engine_router_tool(tmp_path: Path):
    from openharness.tools.missing_tools import EngineRouterTool, EngineRouterToolInput
    from openharness.tools.base import ToolExecutionContext
    tool = EngineRouterTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        EngineRouterToolInput(scene_description="3D Blender animation"),
        ctx,
    )
    assert "blender" in result.output


# ============================================================
# Bonus: all new tools registered
# ============================================================

def test_bonus_all_5_new_tools_registered():
    from openharness.tools import create_default_tool_registry
    reg = create_default_tool_registry()
    for name in ["cost_governor", "audio_qa", "determinism_checker", "contact_sheet", "engine_router"]:
        assert reg.get(name) is not None, f"Tool '{name}' should be registered"


def test_bonus_all_new_services_import():
    from openharness.services import (
        inference_optimizations,
        semantic_cache,
        rendering_optimizations,
        async_optimizations,
        advanced_optimizations,
        disk_model_cache,
    )
    assert inference_optimizations is not None
    assert semantic_cache is not None
    assert rendering_optimizations is not None
    assert async_optimizations is not None
    assert advanced_optimizations is not None
    assert disk_model_cache is not None
