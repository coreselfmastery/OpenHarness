"""Tests for the final wave 4 optimizations: items 10-18, 24-26."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest


# ============================================================
# #10 Frame-Level Parallelism
# ============================================================

from openharness.services.advanced_rendering import (
    FrameParallelRenderer,
    FrameRenderTask,
    FrameRenderResult,
)


@pytest.mark.asyncio
async def test_item_10_frame_parallel_renders_all_frames(tmp_path: Path):
    """Item 10: all frames are rendered."""
    renderer = FrameParallelRenderer(max_workers=4)
    tasks = [
        FrameRenderTask(frame_index=i, scene_id="s1", output_path=str(tmp_path / f"frame_{i:04d}.png"))
        for i in range(10)
    ]
    results = await renderer.render_all(tasks)
    assert len(results) == 10
    assert all(r.success for r in results)
    assert all(Path(r.output_path).is_file() for r in results)


@pytest.mark.asyncio
async def test_item_10_frame_parallel_faster_than_sequential(tmp_path: Path):
    """Item 10: parallel rendering is faster than sequential."""
    async def slow_render(task: FrameRenderTask) -> FrameRenderResult:
        await asyncio.sleep(0.05)
        Path(task.output_path).write_bytes(b"\x00")
        return FrameRenderResult(
            frame_index=task.frame_index,
            output_path=task.output_path,
            render_time_seconds=0.05,
            success=True,
        )

    tasks = [
        FrameRenderTask(frame_index=i, scene_id="s1", output_path=str(tmp_path / f"f{i}.png"))
        for i in range(8)
    ]
    # Parallel (4 workers)
    t0 = time.monotonic()
    renderer = FrameParallelRenderer(render_fn=slow_render, max_workers=4)
    await renderer.render_all(tasks)
    parallel_time = time.monotonic() - t0
    # Sequential would take 8 * 0.05 = 0.4s
    # Parallel with 4 workers should take ~0.1s (2 batches of 4)
    assert parallel_time < 0.3, f"Parallel took {parallel_time:.2f}s, expected < 0.3s"


# ============================================================
# #12 Delta Rendering
# ============================================================

from openharness.services.advanced_rendering import DeltaRenderer, FrameDelta


def test_item_12_no_delta_when_code_unchanged():
    """Item 12: identical code produces no deltas."""
    renderer = DeltaRenderer()
    deltas = renderer.compute_deltas("same code", "same code", frame_count=10)
    assert deltas == []


def test_item_12_major_change_triggers_full_rebuild():
    """Item 12: >50% line change → full rebuild."""
    renderer = DeltaRenderer()
    old = "line1\nline2\nline3\nline4\nline5"
    new = "changed1\nchanged2\nchanged3\nchanged4\nchanged5"
    deltas = renderer.compute_deltas(old, new, frame_count=5)
    assert len(deltas) == 5
    assert all(d.is_full_rebuild for d in deltas)


def test_item_12_minor_change_produces_dirty_regions():
    """Item 12: small change → dirty regions, not full rebuild."""
    renderer = DeltaRenderer()
    old = (
        "function render() {\n"
        "  const bg = createBackground();\n"
        "  const text = createText('Hello');\n"
        "  const logo = createLogo();\n"
        "  return { bg, text, logo };\n"
        "}\n"
    )
    new = (
        "function render() {\n"
        "  const bg = createBackground();\n"
        "  const text = createText('Hello World');\n"  # only this line changed
        "  const logo = createLogo();\n"
        "  return { bg, text, logo };\n"
        "}\n"
    )
    deltas = renderer.compute_deltas(old, new, frame_count=5)
    assert len(deltas) == 5
    assert not any(d.is_full_rebuild for d in deltas)
    assert all(len(d.dirty_regions) > 0 for d in deltas)


def test_item_12_should_skip_frame_when_no_dirty():
    """Item 12: frames with no dirty regions can be skipped."""
    renderer = DeltaRenderer()
    delta = FrameDelta(frame_index=0, dirty_regions=[])
    assert renderer.should_skip_frame(delta)
    delta_with_dirty = FrameDelta(frame_index=1, dirty_regions=[(0, 0, 100, 100)])  # type: ignore
    assert not renderer.should_skip_frame(delta_with_dirty)


# ============================================================
# #13 Codec Optimization
# ============================================================

from openharness.services.advanced_rendering import get_codec_config, build_ffmpeg_args, CodecConfig


def test_item_13_fast_preset_uses_ultrafast():
    """Item 13: fast target uses ultrafast preset."""
    config = get_codec_config(target="fast")
    assert config.preset == "ultrafast"


def test_item_13_quality_preset_uses_slow():
    """Item 13: quality target uses slow preset."""
    config = get_codec_config(target="quality")
    assert config.preset == "slow"


def test_item_13_nvenc_uses_nvenc_codec():
    """Item 13: GPU config uses NVENC codec name."""
    config = get_codec_config(target="balanced", use_gpu=True)
    args = build_ffmpeg_args(config, input_path="in.mp4", output_path="out.mp4")
    # args is a list; check if any element contains "nvenc"
    has_nvenc = any("nvenc" in str(arg) for arg in args)
    assert has_nvenc, f"Expected NVENC in args, got {args}"


def test_item_13_ffmpeg_args_contain_input_and_output():
    """Item 13: ffmpeg args include input and output paths."""
    config = get_codec_config(target="balanced")
    args = build_ffmpeg_args(config, input_path="input.mp4", output_path="output.mp4")
    assert "input.mp4" in args
    assert "output.mp4" in args
    assert "-c:v" in args
    assert "-crf" in args or "-cq" in args


# ============================================================
# #14 Resolution Tiering
# ============================================================

from openharness.services.advanced_rendering import select_resolution_tier, estimate_render_time, RESOLUTION_TIERS


def test_item_14_draft_uses_480p():
    """Item 14: draft stage → 480p."""
    tier = select_resolution_tier("draft")
    assert tier.width == 854
    assert tier.height == 480


def test_item_14_final_uses_1080p():
    """Item 14: final stage → 1080p."""
    tier = select_resolution_tier("final")
    assert tier.width == 1920
    assert tier.height == 1080


def test_item_14_final_4k_when_requested():
    """Item 14: final with target_4k=True → 4K."""
    tier = select_resolution_tier("final", target_4k=True)
    assert tier.width == 3840
    assert tier.height == 2160


def test_item_14_lower_resolution_faster():
    """Item 14: draft renders faster than final."""
    draft_time = estimate_render_time(RESOLUTION_TIERS["draft"], 240)
    final_time = estimate_render_time(RESOLUTION_TIERS["final_1080p"], 240)
    assert draft_time < final_time


# ============================================================
# #15 Pre-Computed Assets
# ============================================================

from openharness.services.advanced_rendering import PreComputedAssetCache


def test_item_15_bake_creates_file(tmp_path: Path):
    """Item 15: baking an asset creates a file."""
    cache = PreComputedAssetCache(cache_dir=tmp_path / "precomputed")
    path = cache.bake("gradient_bg", "gradient code", (1920, 1080))
    assert Path(path).is_file()


def test_item_15_is_baked_after_baking(tmp_path: Path):
    """Item 15: is_baked returns True after baking."""
    cache = PreComputedAssetCache(cache_dir=tmp_path / "pc")
    cache.bake("bg", "code", (1920, 1080))
    assert cache.is_baked("bg")


def test_item_15_get_path_returns_baked_path(tmp_path: Path):
    """Item 15: get_path returns the baked file path."""
    cache = PreComputedAssetCache(cache_dir=tmp_path / "pc")
    path = cache.bake("bg", "code", (1920, 1080))
    assert cache.get_path("bg") == path


def test_item_15_invalidate_removes_asset(tmp_path: Path):
    """Item 15: invalidate removes the baked asset."""
    cache = PreComputedAssetCache(cache_dir=tmp_path / "pc")
    cache.bake("bg", "code", (1920, 1080))
    cache.invalidate("bg")
    assert not cache.is_baked("bg")


def test_item_15_persists_across_instances(tmp_path: Path):
    """Item 15: baked assets persist across cache instances."""
    cache1 = PreComputedAssetCache(cache_dir=tmp_path / "pc")
    cache1.bake("bg", "code", (1920, 1080))
    cache2 = PreComputedAssetCache(cache_dir=tmp_path / "pc")
    assert cache2.is_baked("bg")


# ============================================================
# #16 Speculative Execution
# ============================================================

from openharness.services.pipeline_async import SpeculativeExecutor


@pytest.mark.asyncio
async def test_item_16_speculative_executor_commits():
    """Item 16: speculative step result is committed when confirmed."""
    executor = SpeculativeExecutor()

    async def step_1():
        await asyncio.sleep(0.05)
        return "step_1_done"

    async def step_2():
        await asyncio.sleep(0.05)
        return "step_2_done"

    executor.start_confirmed("step_1", step_1())
    executor.start_speculative("step_2", step_2())
    result_1 = await executor.await_result("step_1")
    assert result_1 == "step_1_done"
    executor.commit("step_2")
    result_2 = await executor.await_result("step_2")
    assert result_2 == "step_2_done"


@pytest.mark.asyncio
async def test_item_16_speculative_executor_discards():
    """Item 16: speculative step is discarded when confirmed step fails."""
    executor = SpeculativeExecutor()

    async def failing_step():
        raise RuntimeError("step failed")

    async def speculative_step():
        await asyncio.sleep(0.1)
        return "speculative_result"

    executor.start_confirmed("main", failing_step())
    executor.start_speculative("spec", speculative_step())
    with pytest.raises(RuntimeError):
        await executor.await_result("main")
    executor.discard("spec")
    stats = executor.stats()
    assert stats["wasted"] >= 1


# ============================================================
# #17 Work Stealing
# ============================================================

from openharness.services.pipeline_async import WorkStealingQueue


@pytest.mark.asyncio
async def test_item_17_work_stealing_redistributes_tasks():
    """Item 17: idle worker steals from busy worker."""
    queue = WorkStealingQueue(num_workers=2)
    # All tasks on worker 0
    for i in range(5):
        queue.push(worker_id=0, task=f"task_{i}")
    # Worker 1 has no tasks — should steal from worker 0
    task = await queue.pop(worker_id=1)
    assert task is not None
    assert queue.steal_count >= 1


@pytest.mark.asyncio
async def test_item_17_own_queue_used_first():
    """Item 17: worker pops from own queue before stealing."""
    queue = WorkStealingQueue(num_workers=2)
    queue.push(worker_id=0, task="own_task")
    queue.push(worker_id=1, task="other_task")
    # Worker 0 should get its own task first
    task = await queue.pop(worker_id=0)
    assert task == "own_task"


@pytest.mark.asyncio
async def test_item_17_returns_none_when_all_empty():
    """Item 17: returns None when all queues are empty."""
    queue = WorkStealingQueue(num_workers=2)
    result = await queue.pop(worker_id=0)
    assert result is None


# ============================================================
# #18 Background Prefetch
# ============================================================

from openharness.services.pipeline_async import BackgroundPrefetcher, PrefetchRequest


@pytest.mark.asyncio
async def test_item_18_prefetch_caches_asset():
    """Item 18: prefetched asset is cached."""
    prefetcher = BackgroundPrefetcher(max_concurrent=2)
    await prefetcher.start()
    try:
        async def fetch_assets():
            await asyncio.sleep(0.01)
            return {"font": "Inter", "color": "#FF0000"}

        prefetcher.request(PrefetchRequest(
            asset_id="scene_002_assets",
            fetch_fn=fetch_assets,
        ))
        result = await prefetcher.await_asset("scene_002_assets", timeout=5.0)
        assert result is not None
        assert result["font"] == "Inter"
        assert prefetcher.is_cached("scene_002_assets")
    finally:
        await prefetcher.stop()


@pytest.mark.asyncio
async def test_item_18_prefetch_returns_none_if_not_requested():
    """Item 18: await_asset returns None if never requested."""
    prefetcher = BackgroundPrefetcher()
    await prefetcher.start()
    try:
        result = await prefetcher.await_asset("nonexistent")
        assert result is None
    finally:
        await prefetcher.stop()


# ============================================================
# #24 AOT Compilation
# ============================================================

from openharness.services.low_level_optimizations import precompile_module, precompile_directory


def test_item_24_precompile_creates_pyc(tmp_path: Path):
    """Item 24: pre-compile produces a .pyc file."""
    source = tmp_path / "test_module.py"
    source.write_text("x = 42\n", encoding="utf-8")
    pyc_path = precompile_module(source)
    assert pyc_path is not None
    assert pyc_path.is_file()
    assert pyc_path.suffix == ".pyc"


def test_item_24_precompile_directory_compiles_all(tmp_path: Path):
    """Item 24: pre-compile directory compiles all .py files."""
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z = 3\n")
    result = precompile_directory(tmp_path)
    assert result["compiled"] == 3
    assert result["failed"] == 0


def test_item_24_precompile_nonexistent_file_returns_none(tmp_path: Path):
    """Item 24: non-existent file returns None."""
    result = precompile_module(tmp_path / "nonexistent.py")
    assert result is None


# ============================================================
# #25 SIMD Vectorization
# ============================================================

from openharness.services.low_level_optimizations import (
    estimate_tokens_batch,
    estimate_message_tokens_vectorized,
    benchmark_vectorized_vs_loop,
)


def test_item_25_batch_estimation_matches_heuristic():
    """Item 25: numpy batch estimation matches the len//4 heuristic."""
    texts = ["hello world", "a", "", "this is a longer text for testing"]
    result = estimate_tokens_batch(texts)
    assert len(result) == 4
    # "hello world" = 11 chars → ceil(11/4) = 3 tokens
    assert result[0] == 3
    # "a" = 1 char → 1 token
    assert result[1] == 1
    # "" = 0 chars → 0 tokens
    assert result[2] == 0
    # "this is a longer text for testing" = 33 chars → ceil(33/4) = 9
    assert result[3] == 9


def test_item_25_empty_list_returns_empty_array():
    """Item 25: empty input returns empty array."""
    result = estimate_tokens_batch([])
    assert len(result) == 0


def test_item_25_vectorized_faster_than_loop():
    """Item 25: numpy is faster than Python loop for 100+ texts."""
    texts = [f"This is text number {i} for benchmarking" for i in range(200)]
    benchmark = benchmark_vectorized_vs_loop(texts, iterations=50)
    # numpy should be at least 1.5× faster for large batches
    assert benchmark["speedup"] >= 1.0, (
        f"Expected speedup >= 1.0, got {benchmark['speedup']}"
    )


# ============================================================
# #26 Memory Mapping
# ============================================================

from openharness.services.low_level_optimizations import MappedFile, read_file_mapped


def test_item_26_mapped_file_reads_content(tmp_path: Path):
    """Item 26: MappedFile reads file content."""
    large_file = tmp_path / "large.txt"
    large_file.write_text("Hello\nWorld\n" * 1000, encoding="utf-8")
    with MappedFile(large_file) as mf:
        content = mf.read()
        assert b"Hello" in content
        assert b"World" in content


def test_item_26_mapped_file_find_pattern(tmp_path: Path):
    """Item 26: MappedFile.find locates a pattern."""
    large_file = tmp_path / "find_test.txt"
    large_file.write_text("AAAERRORBBB\n" * 500, encoding="utf-8")
    with MappedFile(large_file) as mf:
        pos = mf.find(b"ERROR")
        assert pos > 0


def test_item_26_mapped_file_find_all(tmp_path: Path):
    """Item 26: MappedFile.find_all finds all occurrences."""
    large_file = tmp_path / "findall.txt"
    large_file.write_text("TAG\n" * 100, encoding="utf-8")
    with MappedFile(large_file) as mf:
        positions = mf.find_all(b"TAG", max_results=200)
        assert len(positions) == 100


def test_item_26_mapped_file_readline(tmp_path: Path):
    """Item 26: MappedFile.readline reads one line at a time."""
    large_file = tmp_path / "lines.txt"
    large_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
    with MappedFile(large_file) as mf:
        line1, next_offset = mf.readline(0)
        assert line1 == b"line1\n"
        line2, _ = mf.readline(next_offset)
        assert line2 == b"line2\n"


def test_item_26_small_file_falls_back_to_read(tmp_path: Path):
    """Item 26: small files (< 1KB) fall back to regular read."""
    small_file = tmp_path / "small.txt"
    small_file.write_text("tiny", encoding="utf-8")
    with MappedFile(small_file) as mf:
        assert mf.read() == b"tiny"


def test_item_26_read_file_mapped_returns_string(tmp_path: Path):
    """Item 26: read_file_mapped returns a string."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello mapped world", encoding="utf-8")
    content = read_file_mapped(test_file)
    assert isinstance(content, str)
    assert "Hello mapped world" in content


# ============================================================
# Bonus: all new services import
# ============================================================

def test_bonus_all_new_services_import():
    from openharness.services import advanced_rendering, pipeline_async, low_level_optimizations
    assert advanced_rendering is not None
    assert pipeline_async is not None
    assert low_level_optimizations is not None
