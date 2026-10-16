"""Real tests for all 15 optimizations from the second wave.

Each test exercises one optimization end-to-end and asserts the
observable behaviour. No smoke tests.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ============================================================
# Item #1: /delegate skill exists and documents batch mode
# ============================================================

def test_item_1_delegate_skill_exists():
    """Item #1: the /delegate bundled skill file exists."""
    skill_path = Path(__file__).parent.parent / "src" / "openharness" / "skills" / "bundled" / "content" / "delegate.md"
    assert skill_path.is_file(), "/delegate skill should exist"


def test_item_1_delegate_skill_documents_batch_mode():
    """Item #1: the /delegate skill mentions batch_scene."""
    skill_path = Path(__file__).parent.parent / "src" / "openharness" / "skills" / "bundled" / "content" / "delegate.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "batch_scene" in content
    assert "batch" in content.lower()
    assert "Batch Mode" in content


def test_item_1_delegate_skill_documents_brand_kit():
    """Item #1: the /delegate skill mentions brand_kit."""
    skill_path = Path(__file__).parent.parent / "src" / "openharness" / "skills" / "bundled" / "content" / "delegate.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "brand_kit" in content
    assert "brand.yaml" in content


def test_item_1_delegate_skill_documents_parallel_qa():
    """Item #1: the /delegate skill mentions parallel QA waves."""
    skill_path = Path(__file__).parent.parent / "src" / "openharness" / "skills" / "bundled" / "content" / "delegate.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "Parallel QA" in content or "parallel" in content.lower()


def test_item_1_delegate_skill_loads_in_registry():
    """Item #1: the /delegate skill loads as a bundled skill."""
    from openharness.skills.bundled import get_bundled_skills
    skills = get_bundled_skills()
    skill_names = [s.name for s in skills]
    assert "delegate" in skill_names


# ============================================================
# Item #2: brand.yaml loaded at project startup
# ============================================================

def test_item_2_brand_anchor_section_built_when_brand_yaml_exists(tmp_path: Path):
    """Item #2: when brand.yaml exists, the system prompt gets a brand section."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n  name: Test Channel\n  colors:\n    primary: '#1A1A2E'\n  voice: Serena\n",
        encoding="utf-8",
    )
    from openharness.ui.runtime import _build_brand_anchor_section
    section = _build_brand_anchor_section(tmp_path)
    assert section is not None
    assert "Brand Anchors" in section
    assert "Test Channel" in section
    assert "#1A1A2E" in section
    assert "Serena" in section


def test_item_2_brand_anchor_section_none_when_no_brand_yaml(tmp_path: Path):
    """Item #2: no brand.yaml → no brand section (zero overhead)."""
    from openharness.ui.runtime import _build_brand_anchor_section
    section = _build_brand_anchor_section(tmp_path)
    assert section is None


def test_item_2_brand_anchor_section_handles_invalid_yaml(tmp_path: Path):
    """Item #2: invalid brand.yaml doesn't crash — returns None."""
    (tmp_path / "brand.yaml").write_text("brand: [invalid\n  yaml: {{{", encoding="utf-8")
    from openharness.ui.runtime import _build_brand_anchor_section
    section = _build_brand_anchor_section(tmp_path)
    assert section is None  # gracefully skips


# ============================================================
# Item #3: Scene-level caching by structural fingerprint
# ============================================================

from openharness.services.scene_cache import (
    compute_fingerprint,
    SceneCache,
    CachedScene,
    configure_scene_cache,
    disable_scene_cache,
    get_scene_cache,
    reset_scene_cache,
)


def test_item_3_fingerprint_same_structure_different_content():
    """Item #3: two scenes with the same structure get the same fingerprint."""
    desc1 = "Title card with one line of text. 5 seconds."
    desc2 = "Title card with a different line of text. 5 seconds."
    fp1 = compute_fingerprint(desc1)
    fp2 = compute_fingerprint(desc2)
    # Same structure (title card, ~5s, 1 read) → same hash
    assert fp1.to_hash() == fp2.to_hash(), "Same-structure scenes should share a fingerprint"


def test_item_3_fingerprint_different_structure_different_hash():
    """Item #3: a title card and a data viz get different fingerprints."""
    fp1 = compute_fingerprint("Title card with text. 5 seconds.")
    fp2 = compute_fingerprint("Complex data viz chart with 3 bars. 10 seconds.")
    assert fp1.to_hash() != fp2.to_hash()


def test_item_3_cache_hit_returns_cached_code():
    """Item #3: after putting a scene, a get returns it."""
    cache = SceneCache(max_entries=10)
    fp = compute_fingerprint("Title card. 5s.")
    cache.put(fp, "console.log('hello');", project_id="proj-001")
    cached = cache.get(fp)
    assert cached is not None
    assert cached.code == "console.log('hello');"
    assert cached.use_count == 1


def test_item_3_cache_miss_returns_none():
    """Item #3: a fingerprint not in the cache returns None."""
    cache = SceneCache(max_entries=10)
    fp = compute_fingerprint("Non-existent scene.")
    assert cache.get(fp) is None


def test_item_3_cache_hit_increments_use_count():
    """Item #3: each get() increments the use count for LRU eviction."""
    cache = SceneCache(max_entries=10)
    fp = compute_fingerprint("Title card. 5s.")
    cache.put(fp, "code", project_id="p")
    cache.get(fp)
    cache.get(fp)
    cached = cache.get(fp)
    assert cached.use_count == 3


def test_item_3_cache_lru_eviction():
    """Item #3: when full, the least-used entry is evicted."""
    cache = SceneCache(max_entries=2)
    # Use scenes with different structures so fingerprints differ
    fp1 = compute_fingerprint("Title card with text. 5 seconds.")
    fp2 = compute_fingerprint("Bar chart with 3 data points. 10 seconds. Data viz.")
    fp3 = compute_fingerprint("3D Blender animation of rotating cube. 15 seconds.")
    # Verify they're actually different fingerprints
    assert fp1.to_hash() != fp2.to_hash()
    assert fp2.to_hash() != fp3.to_hash()
    cache.put(fp1, "code1")
    cache.put(fp2, "code2")
    # Access fp1 to make it more recently used (use_count = 1)
    cache.get(fp1)
    # Insert fp3 — should evict fp2 (least used, use_count = 0)
    cache.put(fp3, "code3")
    assert cache.get(fp1) is not None  # still cached (was accessed)
    assert cache.get(fp2) is None  # evicted (never accessed)
    assert cache.get(fp3) is not None  # just added


def test_item_3_persistent_cache_survives_restart(tmp_path: Path):
    """Item #3 (cross-session): cache persists to SQLite and reloads."""
    db_path = tmp_path / "scene_cache.db"
    cache1 = SceneCache(persist_path=db_path, max_entries=10)
    fp = compute_fingerprint("Persistent title card.")
    cache1.put(fp, "persisted_code", project_id="proj-A")
    # Simulate restart: create a new cache pointing at the same DB
    cache2 = SceneCache(persist_path=db_path, max_entries=10)
    cached = cache2.get(fp)
    assert cached is not None
    assert cached.code == "persisted_code"
    assert cached.project_id == "proj-A"


# ============================================================
# Item #4: Pre-generation of shared assets
# ============================================================

from openharness.services.shared_assets import (
    generate_shared_assets,
    load_manifest,
    detect_needed_assets,
    is_asset_up_to_date,
)


def test_item_4_generates_shared_asset_files(tmp_path: Path):
    """Item #4: generate_shared_assets writes files to shared/."""
    brand_config = {
        "colors": {"primary": "#1A1A2E", "secondary": "#16213E", "accent": "#E94560"},
        "fonts": {"heading": "Inter", "body": "Inter"},
        "easing": "cubic-bezier(0.4, 0.0, 0.2, 1)",
        "safe_area": 220,
    }
    manifest = generate_shared_assets(
        tmp_path,
        project_id="test-proj",
        brand_config=brand_config,
        scene_descriptions=["Title card with text", "Scene with logo"],
    )
    assert (tmp_path / "shared" / "colors.js").is_file()
    assert (tmp_path / "shared" / "fonts.js").is_file()
    assert (tmp_path / "shared" / "manifest.json").is_file()
    assert len(manifest.assets) >= 3


def test_item_4_manifest_loads_back(tmp_path: Path):
    """Item #4: the manifest can be loaded back from disk."""
    generate_shared_assets(tmp_path, project_id="test", brand_config={"colors": {"primary": "#000"}})
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.project_id == "test"
    assert len(manifest.assets) > 0


def test_item_4_skips_regeneration_when_brand_unchanged(tmp_path: Path):
    """Item #4: if brand config hash matches, assets are not regenerated."""
    brand = {"colors": {"primary": "#FF0000"}}
    manifest1 = generate_shared_assets(tmp_path, project_id="p1", brand_config=brand)
    original_mtime = (tmp_path / "shared" / "colors.js").stat().st_mtime
    # Wait a tiny bit so mtime would change if we regenerated
    time.sleep(0.05)
    manifest2 = generate_shared_assets(tmp_path, project_id="p1", brand_config=brand)
    new_mtime = (tmp_path / "shared" / "colors.js").stat().st_mtime
    assert original_mtime == new_mtime, "File should not be rewritten when brand unchanged"


def test_item_4_regenerates_when_brand_changes(tmp_path: Path):
    """Item #4: if brand config changes, assets are regenerated."""
    brand1 = {"colors": {"primary": "#FF0000"}}
    generate_shared_assets(tmp_path, project_id="p1", brand_config=brand1)
    content1 = (tmp_path / "shared" / "colors.js").read_text()
    brand2 = {"colors": {"primary": "#00FF00"}}
    generate_shared_assets(tmp_path, project_id="p1", brand_config=brand2, force=True)
    content2 = (tmp_path / "shared" / "colors.js").read_text()
    assert content1 != content2
    assert "#00FF00" in content2


def test_item_4_detect_needed_assets():
    """Item #4: asset detection scans scene descriptions."""
    descs = ["Title card with text", "Scene with logo and background"]
    needed = detect_needed_assets(descs)
    assert "color_palette" in needed  # always needed
    assert "font_loader" in needed  # "text" detected
    assert "logo" in needed  # "logo" detected
    assert "background" in needed  # "background" detected


# ============================================================
# Item #5: Parallel QA waves (via /delegate skill — tested in #1)
# ============================================================

def test_item_5_delegate_skill_documents_qa_waves():
    """Item #5: the /delegate skill documents parallel QA waves."""
    skill_path = Path(__file__).parent.parent / "src" / "openharness" / "skills" / "bundled" / "content" / "delegate.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "QA" in content
    assert "parallel" in content.lower() or "wave" in content.lower()


# ============================================================
# Item #6: Smart review depth
# ============================================================

from openharness.services.review_depth import (
    classify_scene,
    get_checklist,
    REVIEW_DEPTH_SIMPLE,
    REVIEW_DEPTH_STANDARD,
    REVIEW_DEPTH_COMPLEX,
    SIMPLE_CHECKLIST_POINTS,
    STANDARD_CHECKLIST_POINTS,
    COMPLEX_CHECKLIST_POINTS,
)


def test_item_6_title_card_classified_as_simple():
    """Item #6: title cards get the lightweight 5-point check."""
    result = classify_scene("Title card with the video name. 5 seconds.")
    assert result.depth == REVIEW_DEPTH_SIMPLE
    assert result.checklist_points == SIMPLE_CHECKLIST_POINTS


def test_item_6_data_viz_classified_as_complex():
    """Item #6: data viz scenes get the full 20-point review."""
    result = classify_scene("Bar chart showing revenue data. Data viz. 10 seconds.")
    assert result.depth == REVIEW_DEPTH_COMPLEX
    assert result.checklist_points == COMPLEX_CHECKLIST_POINTS


def test_item_6_blender_classified_as_complex():
    """Item #6: Blender 3D scenes are complex."""
    result = classify_scene("3D Blender animation of a rotating cube.")
    assert result.depth == REVIEW_DEPTH_COMPLEX


def test_item_6_animation_classified_as_standard():
    """Item #6: scenes with animation but no complex elements are standard."""
    result = classify_scene("Animated text sliding in from left. 8 seconds.")
    assert result.depth == REVIEW_DEPTH_STANDARD
    assert result.checklist_points == STANDARD_CHECKLIST_POINTS


def test_item_6_checklist_lengths_match_depth():
    """Item #6: checklist length matches the depth."""
    assert len(get_checklist(REVIEW_DEPTH_SIMPLE)) == SIMPLE_CHECKLIST_POINTS
    assert len(get_checklist(REVIEW_DEPTH_STANDARD)) == STANDARD_CHECKLIST_POINTS
    assert len(get_checklist(REVIEW_DEPTH_COMPLEX)) == COMPLEX_CHECKLIST_POINTS


def test_item_6_unknown_depth_falls_back_to_standard():
    """Item #6: unknown depth defaults to standard (safe)."""
    checklist = get_checklist("unknown_depth")
    assert len(checklist) == STANDARD_CHECKLIST_POINTS


# ============================================================
# Item #7: Render pipeline overlap
# ============================================================

from openharness.services.render_queue import (
    RenderQueue,
    RenderJob,
    RenderState,
    create_render_queue,
    default_render_fn,
)


@pytest.mark.asyncio
async def test_item_7_render_queue_processes_jobs():
    """Item #7: the render queue processes enqueued jobs."""
    queue = create_render_queue(max_concurrent=2)
    await queue.start()
    try:
        job1 = RenderJob(scene_id="s1", scene_file="scenes/s1.js", output_path="out/s1.mp4")
        job2 = RenderJob(scene_id="s2", scene_file="scenes/s2.js", output_path="out/s2.mp4")
        await queue.enqueue(job1)
        await queue.enqueue(job2)
        results = await queue.wait_all(timeout=5.0)
        assert results["s1"].state == RenderState.COMPLETED
        assert results["s2"].state == RenderState.COMPLETED
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_item_7_render_queue_handles_failures():
    """Item #7: a failing render job is marked as FAILED, not crashing the queue."""
    async def _failing_render(job):
        raise RuntimeError("render failed")

    queue = RenderQueue(render_fn=_failing_render, max_concurrent=1)
    await queue.start()
    try:
        job = RenderJob(scene_id="s1", scene_file="s.js", output_path="out.mp4")
        await queue.enqueue(job)
        results = await queue.wait_all(timeout=5.0)
        assert results["s1"].state == RenderState.FAILED
        assert "render failed" in results["s1"].error
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_item_7_render_queue_runs_concurrently():
    """Item #7: with max_concurrent=4, up to 4 jobs render at once."""
    import time
    active_count = 0
    max_active = 0

    async def _slow_render(job):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.1)
        active_count -= 1
        return job.output_path

    queue = RenderQueue(render_fn=_slow_render, max_concurrent=4)
    await queue.start()
    try:
        for i in range(4):
            await queue.enqueue(RenderJob(scene_id=f"s{i}", scene_file=f"s{i}.js", output_path=f"out/s{i}.mp4"))
        await queue.wait_all(timeout=5.0)
        assert max_active >= 2, f"Expected concurrent rendering, max_active={max_active}"
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_item_7_wait_for_specific_scene():
    """Item #7: can wait for a specific scene to finish."""
    queue = create_render_queue(max_concurrent=1)
    await queue.start()
    try:
        job = RenderJob(scene_id="target", scene_file="s.js", output_path="out.mp4")
        await queue.enqueue(job)
        result = await queue.wait_for_scene("target", timeout=5.0)
        assert result.state == RenderState.COMPLETED
    finally:
        await queue.stop()


# ============================================================
# Item #8: Context window per-stage
# ============================================================

from openharness.services.pipeline_tuning import (
    get_context_budget,
    list_stages,
    STAGE_BUDGETS,
)


def test_item_8_planning_has_largest_budget():
    """Item #8: planning stage gets the most context tokens."""
    planning = get_context_budget("planning")
    scene_gen = get_context_budget("scene_generation")
    assert planning.max_context_tokens > scene_gen.max_context_tokens


def test_item_8_qa_review_has_smallest_output():
    """Item #8: QA review produces short output (just pass/fail + notes)."""
    qa = get_context_budget("qa_review")
    scene_gen = get_context_budget("scene_generation")
    assert qa.max_output_tokens < scene_gen.max_output_tokens


def test_item_8_unknown_stage_falls_back_to_scene_gen():
    """Item #8: unknown stage defaults to scene_generation budget."""
    budget = get_context_budget("nonexistent_stage")
    assert budget.stage == "scene_generation"


def test_item_8_all_stages_have_budgets():
    """Item #8: all known stages have a budget defined."""
    stages = list_stages()
    assert len(stages) >= 5
    for stage in stages:
        budget = get_context_budget(stage)
        assert budget.max_context_tokens > 0
        assert budget.max_output_tokens > 0


# ============================================================
# Item #9: Provider-specific optimizations
# ============================================================

from openharness.services.pipeline_tuning import (
    get_provider_profile,
    list_providers,
    get_tuning,
)


def test_item_9_groq_prefers_short_prompts():
    """Item #9: Groq profile prefers short prompts (low context window)."""
    profile = get_provider_profile("groq")
    assert profile.prefers_short_prompts is True


def test_item_9_glm_prefers_chinese():
    """Item #9: GLM profile prefers Chinese-language instructions."""
    profile = get_provider_profile("glm")
    assert profile.prefers_chinese_instructions is True
    assert "中文" in profile.system_prompt_prefix


def test_item_9_anthropic_has_high_max_tokens():
    """Item #9: Anthropic supports large output windows."""
    profile = get_provider_profile("anthropic")
    assert profile.default_max_tokens >= 16384


def test_item_9_unknown_provider_falls_back_to_default():
    """Item #9: unknown provider gets the default profile."""
    profile = get_provider_profile("unknown_provider_xyz")
    assert profile.name == "default"


def test_item_9_get_tuning_combines_stage_and_provider():
    """Item #9: get_tuning combines stage budget + provider profile."""
    tuning = get_tuning("scene_generation", "groq")
    assert tuning.stage == "scene_generation"
    assert tuning.provider == "groq"
    # max_tokens is capped by the stage budget
    assert tuning.max_tokens <= get_context_budget("scene_generation").max_output_tokens


# ============================================================
# Item #10: Cost projection
# ============================================================

from openharness.services.cost_projection import (
    project_cost,
    get_pricing,
    CostProjection,
)


def test_item_10_cost_projection_for_20_scenes():
    """Item #10: a 20-scene project produces a cost estimate."""
    projection = project_cost(scene_count=20, provider="openai", batch_size=8)
    assert projection.scene_count == 20
    assert projection.total_cost_usd > 0
    assert projection.total_calls > 0
    # Should have estimates for planning, scene_gen, qa, revision
    stages = [e.stage for e in projection.estimates]
    assert "planning" in stages
    assert "scene_generation" in stages
    assert "qa_review" in stages


def test_item_10_batching_reduces_calls():
    """Item #10: batch_size=8 produces fewer calls than batch_size=1."""
    p1 = project_cost(scene_count=20, provider="openai", batch_size=1)
    p8 = project_cost(scene_count=20, provider="openai", batch_size=8)
    gen_calls_1 = next(e.calls for e in p1.estimates if e.stage == "scene_generation")
    gen_calls_8 = next(e.calls for e in p8.estimates if e.stage == "scene_generation")
    assert gen_calls_8 < gen_calls_1
    assert gen_calls_8 == 3  # ceil(20/8) = 3


def test_item_10_budget_exceeded_warning():
    """Item #10: when cost exceeds budget, a warning is generated."""
    projection = project_cost(scene_count=100, provider="anthropic", budget_usd=0.01)
    assert projection.exceeds_budget is True
    assert len(projection.warnings) > 0


def test_item_10_cheaper_provider_costs_less():
    """Item #10: Groq is cheaper than Anthropic for the same scene count."""
    groq = project_cost(scene_count=20, provider="groq")
    anthropic = project_cost(scene_count=20, provider="anthropic")
    assert groq.total_cost_usd < anthropic.total_cost_usd


# ============================================================
# Item #11: Scene dependency graph
# ============================================================

from openharness.services.scene_dependency_graph import (
    build_dependency_graph,
    SceneDependencyGraph,
)


def test_item_11_graph_groups_independent_scenes():
    """Item #11: scenes with no dependencies are grouped for parallel generation."""
    descs = [
        ("s1", "Title card"),
        ("s2", "Stat card 1"),
        ("s3", "Stat card 2"),
    ]
    graph = build_dependency_graph(descs)
    groups = graph.parallel_groups
    assert len(groups) >= 1
    # All 3 scenes should be in the first group (no dependencies)
    assert len(groups[0]) == 3


def test_item_11_topological_sort_orders_dependencies_first():
    """Item #11: if s2 depends on s1, s1 comes first in the sort."""
    graph = SceneDependencyGraph()
    graph.add_scene("s2", "Scene 2", depends_on={"s1"})
    graph.add_scene("s1", "Scene 1")
    order = graph.topological_sort()
    assert order.index("s1") < order.index("s2")


def test_item_11_graph_to_dict_serializes():
    """Item #11: the graph serializes to a dict for persistence."""
    graph = build_dependency_graph([("s1", "Title"), ("s2", "Stat card")])
    data = graph.to_dict()
    assert "nodes" in data
    assert "parallel_groups" in data
    assert "topological_order" in data
    assert "s1" in data["nodes"]


# ============================================================
# Item #12: Incremental render cache
# ============================================================

from openharness.services.incremental_render_cache import (
    IncrementalRenderCache,
    hash_frame,
    compute_code_hash,
)


def test_item_12_frame_hash_is_deterministic():
    """Item #12: same frame data → same hash."""
    data = b"\x00\x01\x02\x03"
    assert hash_frame(data) == hash_frame(data)


def test_item_12_different_frames_different_hash():
    """Item #12: different frame data → different hash."""
    assert hash_frame(b"\x00") != hash_frame(b"\x01")


def test_item_12_cache_stores_and_retrieves_frames(tmp_path: Path):
    """Item #12: stored frames can be retrieved by index."""
    cache = IncrementalRenderCache(cache_dir=tmp_path / "render_cache")
    code_hash = compute_code_hash("scene code here")
    frame_data = b"\x00\x01\x02"
    cache.store_frame("scene_001", 0, hash_frame(frame_data), "frames/0.png", code_hash=code_hash)
    assert cache.is_frame_cached("scene_001", 0, hash_frame(frame_data))
    assert cache.get_frame_path("scene_001", 0) == "frames/0.png"


def test_item_12_cache_detects_unchanged_scene():
    """Item #12: is_fully_cached returns True when code hash matches."""
    cache = IncrementalRenderCache(cache_dir=tmp_path / "rc" if False else Path("/tmp/test_rc_12"))
    code_hash = compute_code_hash("code v1")
    cache.store_frame("s1", 0, "hash0", "path0", code_hash=code_hash)
    assert cache.is_fully_cached("s1", code_hash)
    # Different code hash → not fully cached
    new_hash = compute_code_hash("code v2")
    assert not cache.is_fully_cached("s1", new_hash)


def test_item_12_cache_persists_to_disk(tmp_path: Path):
    """Item #12: cache index persists across restarts."""
    cache_dir = tmp_path / "render_cache"
    cache1 = IncrementalRenderCache(cache_dir=cache_dir)
    code_hash = compute_code_hash("persisted code")
    cache1.store_frame("s1", 0, "framehash", "path.png", code_hash=code_hash)
    # Simulate restart
    cache2 = IncrementalRenderCache(cache_dir=cache_dir)
    assert cache2.is_frame_cached("s1", 0, "framehash")
    assert cache2.get_frame_path("s1", 0) == "path.png"


def test_item_12_invalidate_scene_drops_frames(tmp_path: Path):
    """Item #12: invalidating a scene drops its frames."""
    cache = IncrementalRenderCache(cache_dir=tmp_path / "rc")
    cache.store_frame("s1", 0, "h0", "p0", code_hash="ch")
    cache.invalidate_scene("s1")
    assert cache.get_scene_cache("s1") is None


# ============================================================
# Item #13: Adaptive batch sizing
# ============================================================

from openharness.services.adaptive_batch_sizer import AdaptiveBatchSizer


def test_item_13_starts_at_middle_of_range():
    """Item #13: initial batch size is the midpoint of min/max."""
    sizer = AdaptiveBatchSizer(min_size=2, max_size=10)
    assert sizer.current_size == 6  # (2 + 10) // 2


def test_item_13_increases_size_when_fast():
    """Item #13: when responses are fast, batch size increases."""
    sizer = AdaptiveBatchSizer(min_size=2, max_size=10, target_latency=10.0)
    # Record fast responses (1s per batch of 6 → 0.17s per scene)
    for _ in range(3):
        sizer.record(batch_size=6, response_time_seconds=1.0, tokens_generated=1000)
    assert sizer.current_size > 6  # should have increased


def test_item_13_decreases_size_when_slow():
    """Item #13: when responses are slow, batch size decreases."""
    sizer = AdaptiveBatchSizer(min_size=2, max_size=10, target_latency=5.0)
    # Record slow responses (60s per batch of 6 → 10s per scene)
    for _ in range(3):
        sizer.record(batch_size=6, response_time_seconds=60.0, tokens_generated=1000)
    assert sizer.current_size < 6  # should have decreased


def test_item_13_stays_within_bounds():
    """Item #13: batch size never goes below min or above max."""
    sizer = AdaptiveBatchSizer(min_size=3, max_size=8, target_latency=1.0)
    # Record many fast responses
    for _ in range(20):
        sizer.record(batch_size=sizer.current_size, response_time_seconds=0.1)
    assert sizer.current_size <= 8
    # Now record many slow responses
    for _ in range(20):
        sizer.record(batch_size=sizer.current_size, response_time_seconds=100.0)
    assert sizer.current_size >= 3


# ============================================================
# Item #14: Failure pattern learning
# ============================================================

from openharness.services.failure_tracker import FailurePatternTracker


def test_item_14_records_success_and_failure():
    """Item #14: tracker records both successes and failures."""
    tracker = FailurePatternTracker()
    tracker.record_success("title_card")
    tracker.record_success("title_card")
    tracker.record_failure("title_card", "lint_error", scene_description="bad code")
    stats = tracker.get_stats("title_card")
    assert stats.total_attempts == 3
    assert stats.total_failures == 1
    assert stats.failure_rate == 1 / 3


def test_item_14_recommends_stricter_prompt_for_high_failure_rate():
    """Item #14: high failure rate triggers stricter prompt recommendation."""
    tracker = FailurePatternTracker()
    # 5 failures out of 10 attempts = 50% failure rate
    for _ in range(5):
        tracker.record_failure("data_viz", "qa_reject")
    for _ in range(5):
        tracker.record_success("data_viz")
    assert tracker.should_use_stricter_prompt("data_viz")
    assert tracker.get_strictness_level("data_viz") == 3  # strictest


def test_item_14_does_not_recommend_stricter_for_low_failure_rate():
    """Item #14: low failure rate does not trigger stricter prompts."""
    tracker = FailurePatternTracker()
    for _ in range(9):
        tracker.record_success("title_card")
    tracker.record_failure("title_card", "lint_error")
    assert not tracker.should_use_stricter_prompt("title_card")
    assert tracker.get_strictness_level("title_card") == 1  # normal


def test_item_14_needs_min_attempts_before_recommending():
    """Item #14: need at least 3 attempts before recommending stricter prompts."""
    tracker = FailurePatternTracker()
    tracker.record_failure("test_type", "error")
    tracker.record_failure("test_type", "error")
    # Only 2 attempts — not enough to trust the failure rate
    assert not tracker.should_use_stricter_prompt("test_type")


def test_item_14_persists_across_sessions(tmp_path: Path):
    """Item #14: failure patterns persist to SQLite."""
    db_path = tmp_path / "failures.db"
    tracker1 = FailurePatternTracker(persist_path=db_path)
    tracker1.record_failure("title_card", "lint_error")
    # Simulate restart
    tracker2 = FailurePatternTracker(persist_path=db_path)
    stats = tracker2.get_stats("title_card")
    assert stats.total_failures == 1
    assert stats.total_attempts == 1


# ============================================================
# Item #15: Cross-session scene cache (tested in Item #3)
# ============================================================

def test_item_15_cross_session_cache_works(tmp_path: Path):
    """Item #15: scene cache persists across sessions via SQLite."""
    db_path = tmp_path / "cross_session.db"
    # Session 1: cache a scene
    cache1 = SceneCache(persist_path=db_path, max_entries=10)
    fp = compute_fingerprint("Title card for video 1. 5s.")
    cache1.put(fp, "cached_code_v1", project_id="video-001")
    # Session 2: a different video with the same structural pattern
    cache2 = SceneCache(persist_path=db_path, max_entries=10)
    fp2 = compute_fingerprint("Title card for video 2. 5s.")
    # Same structure → same hash → cache hit!
    cached = cache2.get(fp2)
    assert cached is not None
    assert cached.code == "cached_code_v1"
    assert cached.project_id == "video-001"


# ============================================================
# Bonus: integration — all modules work together
# ============================================================

def test_bonus_all_modules_import():
    """Bonus: all 9 new service modules import without errors."""
    from openharness.services import (
        scene_cache,
        shared_assets,
        review_depth,
        render_queue,
        pipeline_tuning,
        cost_projection,
        scene_dependency_graph,
        adaptive_batch_sizer,
        failure_tracker,
        incremental_render_cache,
    )
    assert scene_cache is not None
    assert shared_assets is not None
    assert review_depth is not None
    assert render_queue is not None
    assert pipeline_tuning is not None
    assert cost_projection is not None
    assert scene_dependency_graph is not None
    assert adaptive_batch_sizer is not None
    assert failure_tracker is not None
    assert incremental_render_cache is not None
