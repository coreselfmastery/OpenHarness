"""Regression tests for the repo-wide audit fixes.

Each test pins a specific bug found in the deep audit:
1. Path traversal in trusted-project permission checks.
2. Stale token cache after in-place message mutation (microcompact).
3. Tool-cache generation race (in-flight result must not be cached
   after a mutation invalidates the cache mid-execution).
4. batch_scene filename traversal (LLM-controlled FILE field).
5. provider_routing has a single shared route builder.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from openharness.config.settings import PermissionSettings
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
)
from openharness.permissions.checker import (
    _path_is_inside,
    build_permission_checker,
)
from openharness.permissions.modes import PermissionMode
from openharness.services.compact import estimate_message_tokens
from openharness.services.tool_cache import ToolResultCache


# ============================================================
# Fix 1: path traversal in trusted-project mode
# ============================================================

class _Settings:
    """Minimal settings stand-in for build_permission_checker."""

    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self.permission = PermissionSettings(mode=mode)
        self.trusted_project = True
        self.trusted_paths: list[str] = []


def test_path_is_inside_blocks_traversal(tmp_path: Path):
    project = str(tmp_path / "project")
    outside = str(tmp_path / "etc-passwd")
    Path(project).mkdir(parents=True, exist_ok=True)
    assert _path_is_inside(f"{project}/../../etc/passwd", project) is False
    assert _path_is_inside(f"{project}/sub/../../{Path(project).name}/ok.txt", project) is True
    assert _path_is_inside(f"{project}/../../{Path(outside).name}", project) is False


def test_trusted_project_does_not_auto_approve_traversal(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    settings = _Settings()
    checker = build_permission_checker(settings, cwd=str(project))
    decision = checker.evaluate(
        "write_file",
        file_path=f"{project}/../../etc/passwd",
        is_read_only=False,
    )
    # Must NOT be auto-approved: the resolved target escapes the project.
    assert decision.allowed is False or decision.requires_confirmation


def test_trusted_project_auto_approves_in_project(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    settings = _Settings()
    checker = build_permission_checker(settings, cwd=str(project))
    decision = checker.evaluate(
        "write_file",
        file_path=str(project / "src" / "scene.js"),
        is_read_only=False,
    )
    assert decision.allowed is True
    assert decision.requires_confirmation is False


# ============================================================
# Fix 2: token cache invalidation on in-place mutation
# ============================================================

def _message_with_tool_result(content: str) -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="t1", content=content)],
    )


def test_token_cache_recomputes_after_in_place_content_replacement():
    msg = _message_with_tool_result("x" * 4000)
    before = estimate_message_tokens([msg])
    # Second call hits the cache — same value expected.
    assert estimate_message_tokens([msg]) == before
    # In-place mutation (what time-based microcompact does): replace the
    # content list with cleared blocks on the SAME message object.
    msg.content = [ToolResultBlock(tool_use_id="t1", content="[cleared]")]
    after = estimate_message_tokens([msg])
    assert after < before, "cleared tool result must shrink the estimate"


def test_token_cache_recomputes_after_block_swap_in_place():
    msg = ConversationMessage(
        role="user",
        content=[TextBlock(text="a" * 4000), TextBlock(text="b")],
    )
    before = estimate_message_tokens([msg])
    # In-place block swap (what the query loop does when collapsing a
    # tool result): replace one block within the SAME list object.
    msg.content[0] = TextBlock(text="short")
    after = estimate_message_tokens([msg])
    assert after < before


# ============================================================
# Fix 3: tool-cache generation race
# ============================================================

@pytest.mark.asyncio
async def test_tool_cache_does_not_persist_result_after_mutation_midflight():
    cache = ToolResultCache(max_entries=8, ttl_seconds=0)

    async def slow_read() -> dict:
        await asyncio.sleep(0.05)
        return {"output": "file-v1", "is_error": False, "metadata": {}}

    async def mutate_then_finish() -> None:
        await asyncio.sleep(0.01)
        cache.invalidate_all()  # a write_file lands mid-read

    results = await asyncio.gather(
        cache.get_or_execute("read_file", {"path": "a.txt"}, slow_read),
        mutate_then_finish(),
    )
    assert results[0] is not None
    assert results[0]["output"] == "file-v1"  # the caller still gets its result
    # But the stale result must NOT have been cached after the mutation.
    assert cache.get("read_file", {"path": "a.txt"}) is None


@pytest.mark.asyncio
async def test_tool_cache_caches_when_no_mutation_intervenes():
    cache = ToolResultCache(max_entries=8, ttl_seconds=0)

    async def read() -> dict:
        return {"output": "v1", "is_error": False, "metadata": {}}

    await cache.get_or_execute("read_file", {"path": "a.txt"}, read)
    assert cache.get("read_file", {"path": "a.txt"}) == {
        "output": "v1", "is_error": False, "metadata": {},
    }


# ============================================================
# Fix 4: batch_scene filename traversal
# ============================================================

@pytest.mark.asyncio
async def test_batch_scene_rejects_traversal_filenames(tmp_path: Path, monkeypatch):
    """A scene whose LLM-provided FILE field escapes output_dir must be
    rejected — the tool must not write outside the output directory."""
    from openharness.tools.batch_scene_tool import BatchSceneTool, BatchSceneToolInput
    from openharness.tools.base import ToolExecutionContext

    malicious_response = (
        "BATCH_ID: batch_x\n"
        "SCENES: 1\n\n"
        "SCENE_001:\n"
        "  FILE: ../../escaped.js\n"
        "  ENGINE: hyperframes\n"
        "  DURATION: 8s\n"
        '  READS: ["line"]\n'
        "  CODE:\n"
        "    function render() { return 1; }\n"
    )

    async def _fake_llm(self, arguments, config, context):
        return malicious_response

    monkeypatch.setattr(BatchSceneTool, "_call_llm", _fake_llm)

    tool = BatchSceneTool()
    ctx = ToolExecutionContext(cwd=tmp_path, metadata={"scene_gen_config": {}})
    result = await tool.execute(
        BatchSceneToolInput(scenes=["a scene"], output_dir="scenes"), ctx,
    )

    # The traversal scene must be reported as failed...
    assert result.metadata["scenes_failed"][0]["file"] == "../../escaped.js"
    assert "invalid filename" in result.metadata["scenes_failed"][0]["error"]
    # ...and nothing may be written outside the output directory.
    assert not (tmp_path / "escaped.js").exists()
    assert not (tmp_path.parent / "escaped.js").exists()


@pytest.mark.asyncio
async def test_batch_scene_writes_valid_filenames(tmp_path: Path, monkeypatch):
    """A well-formed FILE field writes inside output_dir (guard against
    over-blocking after the traversal fix)."""
    from openharness.tools.batch_scene_tool import BatchSceneTool, BatchSceneToolInput
    from openharness.tools.base import ToolExecutionContext

    good_response = (
        "BATCH_ID: batch_y\n"
        "SCENES: 1\n\n"
        "SCENE_001:\n"
        "  FILE: scene_001.js\n"
        "  ENGINE: hyperframes\n"
        "  DURATION: 8s\n"
        '  READS: ["line"]\n'
        "  CODE:\n"
        "    function render() { return 1; }\n"
    )

    async def _fake_llm(self, arguments, config, context):
        return good_response

    monkeypatch.setattr(BatchSceneTool, "_call_llm", _fake_llm)

    tool = BatchSceneTool()
    ctx = ToolExecutionContext(cwd=tmp_path, metadata={"scene_gen_config": {}})
    result = await tool.execute(
        BatchSceneToolInput(scenes=["a scene"], output_dir="scenes"), ctx,
    )
    assert result.metadata["scenes_written"] == [str(Path("scenes") / "scene_001.js")]
    assert (tmp_path / "scenes" / "scene_001.js").is_file()


# ============================================================
# Fix 5: provider_routing single route builder
# ============================================================

def test_build_fallback_route_shared_implementation():
    from openharness.services.provider_routing import ProviderRoute, build_fallback_route

    route = build_fallback_route("fb1", {
        "api_format": "openai",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "delay_seconds": 1.5,
    })
    assert isinstance(route, ProviderRoute)
    assert route.name == "fb1"
    assert route.delay_seconds == 1.5
    # No key -> skipped (None), not an exception.
    assert build_fallback_route("fb2", {"api_format": "openai"}) is None
    # Unsupported format -> skipped.
    assert build_fallback_route("fb3", {"api_format": "weird", "api_key": "x"}) is None


def test_provider_route_has_no_dead_fields():
    from openharness.services import provider_routing

    assert not hasattr(provider_routing.ProviderRoute(name="r", client=None), "min_attempt")
