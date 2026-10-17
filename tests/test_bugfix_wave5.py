"""Regression tests for the wave-5 full-depth audit fixes.

Every test pins the exact failure mode found in the audit:

A.  render_cache: ffmpeg asset regex was an invalid escape (\\i) — every
    ffmpeg-engine fingerprint/asset scan raised re.error.
B.  render_pipeline: a crashing job killed its worker task forever.
C.  render_pipeline: fingerprint ignored resolution/template changes.
D.  bash_tool: >64KB output deadlocked wait() and falsely reported
    "timed out".
E.  query_engine: tool results were lost when the turn ended after tool
    execution (max-turns / API error / empty assistant).
F.  control_bus: deadlock detector killed healthy 100%-busy sessions.
G.  agent_concurrency: CancelledError leaked a permit permanently.
H.  coordinator context: prefix check never matched, context accumulated.
I.  mcp client: server-side tool errors (isError=True) reported success.
J.  hooks executor: ApiRetryEvent crashed the LLM-hook (no .text attr).
K.  enter/exit_plan_mode: live permission checker never updated.
L.  config show: dumped every api_key into the conversation.
M.  checkpoint restore: id path traversal + empty-restore conversation wipe.
N.  todo_write: un-checking appended a duplicate line.
O.  file_edit: non-unique old_str silently edited the first occurrence.
P.  pixel_qa: corrupt video reported PASS; warn checks hard-failed scenes;
    frozen check false-flagged localized motion; unknown checks "passed".
Q.  frame_diff: formatter KeyError'd on per-video error entries;
    motion speed was in diluted analysis-units.
R.  style_consistency: inline easings never normalized (false deviation);
    same-stem scenes collided.
S.  semantic_memory: sync HTTP embed blocked the loop; cosine fallback
    math wrong; recall("" ) vs list("") disagreed.
T.  batch_executor: retry rounds passed sublist positions as indices.
U.  racing client: winning pump leaked on early consumer abort.
V.  semantic cache: replays re-billed the original usage.
W.  permissions: allowed_tools bypassed path deny rules; report_path was
    invisible to the permission system.
X.  tasks manager: stop/watch race double-fired completion listeners;
    agent api key sat on the command line.
Y.  minicpm/vlm tools: temp frame dirs never deleted.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from openharness.tools.base import ToolExecutionContext


# ======================================================================
# A — render_cache ffmpeg regex
# ======================================================================

def test_ffmpeg_asset_pattern_is_valid_regex():
    import re

    from openharness.services.render_cache import _ASSET_PATTERNS

    for engine, pattern in _ASSET_PATTERNS:
        re.compile(pattern)  # must not raise re.error (ffmpeg's was "\\i")
    ffmpeg_pattern = re.compile(dict(_ASSET_PATTERNS)["ffmpeg"])
    match = ffmpeg_pattern.search("ffmpeg -i intro.png out.mp4")
    assert match and match.group(1) == "intro.png"


def test_find_asset_paths_ffmpeg_does_not_raise(tmp_path: Path):
    from openharness.services.render_cache import find_asset_paths

    code = 'ffmpeg -i intro.png -i bg.mp3 -c:v libx264 out.mp4'
    paths = find_asset_paths(code, "ffmpeg", cwd=tmp_path)
    assert isinstance(paths, list)  # the call itself previously raised re.error


# ======================================================================
# B — worker survives a crashing job
# ======================================================================

async def test_worker_survives_crashing_job(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline, reset_render_pipeline

    reset_render_pipeline()
    boom = {"boom": True}
    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    pipeline._templates["hyperframes"] = ["{scene}", "{output}"]

    async def evil_run(job, worker_id):
        raise RuntimeError("worker-level crash")

    pipeline._run_job = evil_run  # force a crash inside the worker loop
    scene = tmp_path / "s.js"
    scene.write_text("x", encoding="utf-8")
    job = pipeline.submit(scene, engine="hyperframes", tier="draft")
    await pipeline.wait_all(timeout=10)
    assert job.status == "failed"
    assert "worker-level crash" in job.error
    # The worker must still be alive: a second job completes normally.
    pipeline._run_job = RenderPipeline._run_job.__get__(pipeline, RenderPipeline)
    pipeline._templates["hyperframes"] = ["bash", "-c", "echo rendered > {output}"]
    job2 = pipeline.submit(scene, engine="hyperframes", tier="draft")
    await pipeline.wait(job2, timeout=15)
    assert job2.status == "done", job2.error
    await pipeline.shutdown()


# ======================================================================
# C — fingerprint covers resolution
# ======================================================================

def test_fingerprint_changes_with_resolution(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    scene = tmp_path / "s.js"
    scene.write_text("same code", encoding="utf-8")
    low = RenderPipeline(
        workers=1, cwd=tmp_path, cache_enabled=False,
        resolution_tiers={"draft": "640x360"},
    )
    high = RenderPipeline(
        workers=1, cwd=tmp_path, cache_enabled=False,
        resolution_tiers={"draft": "960x540"},
    )
    job = type("J", (), {"engine": "hyperframes", "tier": "draft"})()
    assert low._engine_args_for(job) != high._engine_args_for(job)


# ======================================================================
# D — bash tool: big output must not deadlock
# ======================================================================

async def test_bash_tool_large_output_does_not_deadlock(tmp_path: Path):
    from openharness.tools.bash_tool import BashTool, BashToolInput

    result = await asyncio.wait_for(
        BashTool().execute(
            BashToolInput(
                command="python3 -c \"import sys; [sys.stdout.write('x'*1024) for _ in range(200)]\"",
                timeout_seconds=30,
            ),
            ToolExecutionContext(cwd=tmp_path),
        ),
        timeout=25,
    )
    # 200 KB of output — far beyond the ~64 KB pipe buffer. The old
    # implementation hung until the timeout and reported failure.
    assert not result.metadata.get("timed_out", False)
    assert len(result.output) >= 1000


# ======================================================================
# E — tool results survive abnormal loop exits
# ======================================================================

class _MaxTurnsApiClient:
    """Streams one assistant tool_use turn, then a final text turn.

    With max_turns=1 the loop exits right after the tool results are
    appended — the exact path that used to lose them.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):
        self.calls += 1
        from openharness.api.client import ApiMessageCompleteEvent
        from openharness.api.usage import UsageSnapshot

        if self.calls == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        TextBlock(text="Reading the file."),
                        ToolUseBlock(id="toolu_1", name="read_file", input={"path": "hello.txt"}),
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


async def test_tool_results_survive_max_turns_exit(tmp_path: Path):
    from openharness.engine.query_engine import QueryEngine
    from openharness.engine.query import MaxTurnsExceeded
    from openharness.permissions import PermissionChecker
    from openharness.permissions.checker import PermissionSettings
    from openharness.permissions.modes import PermissionMode
    from openharness.tools.base import ToolRegistry
    from openharness.tools.file_read_tool import FileReadTool

    (tmp_path / "hello.txt").write_text("hello world\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(FileReadTool())
    engine = QueryEngine(
        api_client=_MaxTurnsApiClient(),
        tool_registry=registry,
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        max_turns=1,
    )
    with pytest.raises(MaxTurnsExceeded):
        async for _ in engine.submit_message("read the file"):
            pass

    messages = engine.messages
    # The executed tool RESULT must be in the engine history (it used to
    # end at the dangling assistant tool_use with no results at all).
    assert any(
        isinstance(block, ToolResultBlock)
        for message in messages
        for block in message.content
    )
    assert engine.has_pending_continuation()


async def test_tool_results_survive_api_error_exit(tmp_path: Path):
    from openharness.engine.query_engine import QueryEngine
    from openharness.permissions import PermissionChecker
    from openharness.permissions.checker import PermissionSettings
    from openharness.permissions.modes import PermissionMode
    from openharness.tools.base import ToolRegistry
    from openharness.tools.file_read_tool import FileReadTool

    class _BrokenAfterFirstTurn:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_message(self, request):
            from openharness.api.client import ApiMessageCompleteEvent
            from openharness.api.usage import UsageSnapshot

            self.calls += 1
            if self.calls == 1:
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="Reading."),
                            ToolUseBlock(id="toolu_1", name="read_file", input={"path": "hello.txt"}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                )
                return
            raise RuntimeError("API exploded after tool execution")

    (tmp_path / "hello.txt").write_text("hello world\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(FileReadTool())
    engine = QueryEngine(
        api_client=_BrokenAfterFirstTurn(),
        tool_registry=registry,
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    events = [event async for event in engine.submit_message("read")]
    assert any(type(event).__name__ == "ErrorEvent" for event in events)
    assert any(
        isinstance(block, ToolResultBlock)
        for message in engine.messages
        for block in message.content
    )


# ======================================================================
# F — deadlock detector must not kill busy systems
# ======================================================================

def test_deadlock_detector_ignores_full_utilization_with_progress():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(deadlock_probe_seconds=0.05)
    bus.register_deadlock_probe(lambda: {"name": "agents", "blocked": 4, "running": 0})
    bus.register_deadlock_probe(lambda: {"name": "renders", "blocked": 2, "running": 0})
    for _ in range(5):
        bus.feed("llm")  # healthy session: constant observable progress
        assert bus.detect_deadlock() is None
        time.sleep(0.06)


def test_deadlock_detector_fires_when_no_progress_and_all_blocked():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(deadlock_probe_seconds=0.05)
    bus.register_deadlock_probe(lambda: {"name": "agents", "blocked": 4, "running": 0})
    time.sleep(0.1)  # no feed() at all — no progress
    first = bus.detect_deadlock()
    time.sleep(0.1)
    assert bus.detect_deadlock() is not None
    # The first probe primes the persistence window (spurious wakeup guard)
    _ = first


def test_deadlock_probes_count_busy_workers_as_running():
    from openharness.services.session_services import _agent_semaphore_probe

    report = _agent_semaphore_probe()
    # A held permit is RUNNING work, never "blocked".
    assert report["blocked"] == 0


# ======================================================================
# G — permit release on cancellation
# ======================================================================

async def test_permit_released_on_cancellation():
    from openharness.services import agent_concurrency as ac

    ac.reset_agent_concurrency()
    ac._AGENT_LIMIT = 1
    ac._AGENT_SEMAPHORE = asyncio.Semaphore(1)

    async def slow_spawn():
        await asyncio.sleep(30)
        return None

    task = asyncio.create_task(ac.acquire_permit_for_task(slow_spawn))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The permit must be free again: a new acquire completes immediately.
    acquired = asyncio.create_task(ac.acquire_permit_for_task(_quick_record))
    await asyncio.wait_for(acquired, timeout=2)
    ac.reset_agent_concurrency()


async def _quick_record():
    class _R:
        id = "t1"
    return _R()


# ======================================================================
# H — coordinator context no longer accumulates
# ======================================================================

async def test_coordinator_context_is_ephemeral(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_COORDINATOR_MODE", "1")
    from openharness.coordinator.coordinator_mode import get_coordinator_system_prompt
    from openharness.engine.query_engine import QueryEngine
    from openharness.permissions import PermissionChecker
    from openharness.permissions.checker import PermissionSettings
    from openharness.permissions.modes import PermissionMode
    from openharness.tools.base import ToolRegistry

    class _PlainClient:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_message(self, request):
            from openharness.api.client import ApiMessageCompleteEvent
            from openharness.api.usage import UsageSnapshot

            self.calls += 1
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )

    prompt = get_coordinator_system_prompt()
    assert "You are a **coordinator**." in prompt  # substring exists (old check was startswith)
    engine = QueryEngine(
        api_client=_PlainClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt=prompt,
    )
    async for _ in engine.submit_message("hi"):
        pass
    async for _ in engine.submit_message("again"):
        pass
    coordinator_messages = [
        message
        for message in engine.messages
        if message.role == "user"
        and message.text.startswith("# Coordinator User Context")
    ]
    # Ephemeral context never persists in the adopted history (one stale
    # copy used to accumulate per user turn).
    assert coordinator_messages == []


# ======================================================================
# I — MCP isError surfaces as failure
# ======================================================================

async def test_mcp_tool_error_reported_as_error():
    from openharness.mcp.client import McpToolExecutionError

    class _Result:
        isError = True
        content = []

        class _Text:
            type = "text"
            text = "boom: bad arguments"

        def __init__(self):
            self.content = [_Result._Text()]

    class _Session:
        async def call_tool(self, name, arguments):
            return _Result()

    from openharness.mcp.client import McpClientManager

    manager = McpClientManager({})
    manager._sessions["srv"] = _Session()
    with pytest.raises(McpToolExecutionError):
        await manager.call_tool("srv", "tool", {})


# ======================================================================
# J — hooks executor survives ApiRetryEvent
# ======================================================================

async def test_hook_llm_call_survives_retry_event():
    from openharness.api.client import (
        ApiMessageCompleteEvent,
        ApiRetryEvent,
        ApiTextDeltaEvent,
    )
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage
    from openharness.hooks.executor import HookExecutor
    from openharness.hooks.schemas import PromptHookDefinition

    class _RetryingClient:
        async def stream_message(self, request):
            yield ApiRetryEvent(message="429", attempt=0, max_attempts=3, delay_seconds=0.0)
            yield ApiTextDeltaEvent(text='{"ok": true}')
            yield ApiMessageCompleteEvent(
                message=ConversationMessage.from_user_text('{"ok": true}'),
                usage=UsageSnapshot(),
            )

    class _Context:
        api_client = _RetryingClient()
        default_model = "m"

    executor = HookExecutor.__new__(HookExecutor)
    executor._context = _Context()
    hook = PromptHookDefinition(name="p", type="prompt", prompt="say {event}")
    result = await executor._run_prompt_like_hook(
        hook, __import__("openharness.hooks.events", fromlist=["HookEvent"]).HookEvent.PRE_TOOL_USE,
        {"tool_name": "bash"},
        agent_mode=False,
    )
    assert result.success  # previously: AttributeError('ApiRetryEvent' has no 'text')


# ======================================================================
# K — plan mode tools update the live checker
# ======================================================================

async def test_enter_plan_mode_updates_live_checker(tmp_path: Path):
    from openharness.config.settings import Settings
    from openharness.permissions.checker import PermissionSettings
    from openharness.permissions.modes import PermissionMode
    from openharness.tools.enter_plan_mode_tool import EnterPlanModeTool, EnterPlanModeToolInput

    class _Engine:
        from openharness.permissions import build_permission_checker  # noqa: F401

        cwd = tmp_path

        def set_permission_checker(self, checker):
            self.checker = checker

    engine = _Engine()
    context = ToolExecutionContext(
        cwd=tmp_path, metadata={"engine_ref": engine, "settings": Settings()}
    )
    result = await EnterPlanModeTool().execute(EnterPlanModeToolInput(), context)
    assert "live session updated" in result.output
    assert engine.checker._settings.mode == PermissionMode.PLAN


# ======================================================================
# L — config show redacts secrets
# ======================================================================

async def test_config_show_redacts_api_keys(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENHARNESS_ANTHROPIC_API_KEY", "sk-super-secret-value")
    from openharness.config.settings import load_settings
    from openharness.tools.config_tool import ConfigTool, ConfigToolInput

    settings = load_settings()
    assert settings.api_key == "sk-super-secret-value"
    result = await ConfigTool().execute(
        ConfigToolInput(action="show"), ToolExecutionContext(cwd=tmp_path)
    )
    assert "sk-super-secret-value" not in result.output
    assert "redacted" in result.output


# ======================================================================
# M — checkpoint traversal + empty restore
# ======================================================================

def test_checkpoint_id_traversal_rejected(tmp_path: Path):
    from openharness.services.checkpoints import CheckpointManager

    manager = CheckpointManager("default")
    # Plant a file outside the checkpoint root that a traversal would hit.
    target = manager._root.parent.parent / "credentials.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"messages": []}', encoding="utf-8")
    try:
        assert manager.restore("../../credentials") is None
    finally:
        target.unlink(missing_ok=True)


async def test_checkpoint_empty_restore_aborts(tmp_path: Path):
    from openharness.tools.pipeline_tools import CheckpointTool, CheckpointInput

    class _Engine:
        live_messages = [ConversationMessage.from_user_text("real history")]

        def rollback_messages(self, messages):
            self.rolled_back = list(messages)

    engine = _Engine()
    context = ToolExecutionContext(cwd=tmp_path, metadata={"engine_ref": engine})
    # Create a checkpoint whose messages array is empty.
    from openharness.services.checkpoints import CheckpointManager

    manager = CheckpointManager("default")
    summary = manager.create([], "empty")
    result = await CheckpointTool().execute(
        CheckpointInput(action="restore", checkpoint_id=summary["id"]), context
    )
    assert result.is_error
    assert not getattr(engine, "rolled_back", None) == []
    assert engine.live_messages  # conversation NOT wiped


# ======================================================================
# N — todo_write un-check
# ======================================================================

async def test_todo_write_uncheck_replaces_checked_line(tmp_path: Path):
    from openharness.tools.todo_write_tool import TodoWriteTool, TodoWriteToolInput

    todo = tmp_path / "TODO.md"
    TodoWriteToolInput_ = TodoWriteToolInput
    tool = TodoWriteTool()
    await tool.execute(
        TodoWriteToolInput_(item="Buy milk"), ToolExecutionContext(cwd=tmp_path)
    )
    await tool.execute(
        TodoWriteToolInput_(item="Buy milk", checked=True),
        ToolExecutionContext(cwd=tmp_path),
    )
    await tool.execute(
        TodoWriteToolInput_(item="Buy milk", checked=False),
        ToolExecutionContext(cwd=tmp_path),
    )
    text = todo.read_text(encoding="utf-8")
    assert "- [x] Buy milk" not in text
    assert text.count("- [ ] Buy milk") == 1


# ======================================================================
# O — file_edit uniqueness
# ======================================================================

async def test_file_edit_rejects_ambiguous_old_str(tmp_path: Path):
    from openharness.tools.file_edit_tool import FileEditTool, FileEditToolInput

    target = tmp_path / "code.txt"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    result = await FileEditTool().execute(
        FileEditToolInput(path="code.txt", old_str="value = 1", new_str="value = 2"),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert result.is_error
    assert "not unique" in result.output
    assert target.read_text(encoding="utf-8") == "value = 1\nvalue = 1\n"


# ======================================================================
# P — pixel_qa honest verdicts
# ======================================================================

def _make_mp4(path: Path, frames: int = 24, drawer=None, size: tuple[int, int] = (320, 180)) -> None:
    import cv2
    import numpy as np

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        if drawer is not None:
            drawer(frame, i)
        writer.write(frame)
    writer.release()


def test_pixel_qa_corrupt_video_fails(tmp_path: Path):
    from openharness.services.pixel_qa import PixelQA

    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"this is not a video at all")
    report = PixelQA().inspect_video(bogus)
    assert report.passed is False
    assert report.check_summary["open"]["severity"] == "fail"


def test_pixel_qa_unknown_check_rejected(tmp_path: Path):
    from openharness.services.pixel_qa import PixelQA

    video = tmp_path / "ok.mp4"
    _make_mp4(video)
    with pytest.raises(ValueError):
        PixelQA().inspect_video(video, enabled_checks={"blackframe"})  # typo


def test_pixel_qa_warn_check_does_not_fail_scene(tmp_path: Path):
    import numpy as np

    from openharness.services.pixel_qa import PixelQA

    video = tmp_path / "small_text.mp4"

    def draw_small_box(frame, i):
        # 4-px-tall wide box on a 1080p canvas: below the 12 px minimum
        # -> text_size WARNING (not a scene failure).
        frame[400:404, 600:1400] = (255, 255, 255)

    _make_mp4(video, frames=12, drawer=draw_small_box, size=(1920, 1080))
    report = PixelQA().inspect_video(video, enabled_checks={"text_size"})
    assert report.check_summary["text_size"]["severity"] == "warn"
    assert report.passed is True  # warnings do not fail the scene


def test_pixel_qa_sliding_panel_not_frozen(tmp_path: Path):
    from openharness.services.pixel_qa import PixelQA

    video = tmp_path / "panel.mp4"

    def draw_panel(frame, i):
        x = 10 + i * 4  # panel slides right 4 px/frame -> real motion
        frame[40:140, x : x + 80] = (0, 200, 255)

    _make_mp4(video, drawer=draw_panel)
    report = PixelQA().inspect_video(video, enabled_checks={"frozen_frame"})
    assert report.check_summary["frozen_frame"]["frames_flagged"] == 0
    assert report.passed


def test_pixel_qa_truly_frozen_flagged(tmp_path: Path):
    from openharness.services.pixel_qa import PixelQA

    video = tmp_path / "frozen.mp4"

    def static_logo(frame, i):  # identical every frame
        frame[40:140, 100:200] = (0, 200, 255)

    _make_mp4(video, drawer=static_logo)
    report = PixelQA().inspect_video(video, enabled_checks={"frozen_frame"})
    assert report.check_summary["frozen_frame"]["frames_flagged"] > 0
    assert not report.passed


# ======================================================================
# Q — frame_diff formatter + motion speed semantics
# ======================================================================

def test_frame_diff_formatter_survives_error_entries(tmp_path: Path):
    from openharness.services.frame_diff import SceneMotionReport, format_report_text

    report = SceneMotionReport(scene="bad", video=str(tmp_path / "bad.mp4"))
    report.check_summary = {
        "open": {"severity": "fail", "detail": "cannot open video: bad.mp4"}
    }
    text = format_report_text([report])  # previously KeyError: 'count'
    assert "cannot open video" in text


def test_frame_diff_motion_speed_uses_source_scale(tmp_path: Path):
    from openharness.services.frame_diff import FrameDiffAnalyzer

    video = tmp_path / "move.mp4"

    def draw_moving_bar(frame, i):
        import numpy as np

        frame[:] = (30, 30, 30)
        x = 8 + i * 8  # 8 px/frame at 12 fps = 96 px/s source scale
        frame[60:120, x : x + 60] = (255, 255, 255)

    _make_mp4(video, frames=24, drawer=draw_moving_bar)
    report = FrameDiffAnalyzer().analyze_video(
        video, sample_fps=12.0, expected_flow_px_s=(40.0, 400.0)
    )
    motion_issues = [i for i in report.issues if i.check == "motion_speed"]
    assert motion_issues == []  # 96 px/s is inside the window


# ======================================================================
# R — style consistency
# ======================================================================

def test_style_consistency_inline_easing_spacing_equivalent(tmp_path: Path):
    from openharness.services.style_consistency import (
        StyleConsistencyChecker,
        load_reference,
    )

    scene = tmp_path / "s1.js"
    scene.write_text(
        "const a = 1; transition: cubic-bezier(0.4, 0, 0.2, 1); color: #F2F0EB;\n",
        encoding="utf-8",
    )
    reference = load_reference(easings=["cubic-bezier(0.4, 0, 0.2, 1)"])
    report = StyleConsistencyChecker(reference).check_files([scene])
    easings = [i for i in report.issues if i.kind == "easing_deviation"]
    assert easings == []  # same easing, different spacing — no deviation


def test_style_consistency_same_stem_scenes_both_counted(tmp_path: Path):
    from openharness.services.style_consistency import StyleConsistencyChecker

    d1 = tmp_path / "chapter1"
    d2 = tmp_path / "chapter2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "scene.js").write_text("color: #112233;\n", encoding="utf-8")
    (d2 / "scene.js").write_text("color: #445566;\n", encoding="utf-8")
    report = StyleConsistencyChecker({}).check_files([d1 / "scene.js", d2 / "scene.js"])
    assert report.scenes_checked == 2
    assert len(report.per_scene_tokens) == 2  # previously 1 (collision)


# ======================================================================
# S — semantic memory
# ======================================================================

def test_cosine_fallback_math(monkeypatch):
    from openharness.services import semantic_memory as sm

    monkeypatch.setitem(sys.modules, "numpy", None)  # forces ImportError branch
    # cosine([1,1],[1,0]) = 0.7071 — the old fallback returned 1.0.
    assert abs(sm._cosine([1.0, 1.0], [1.0, 0.0]) - 0.7071067811865475) < 1e-9
    assert sm._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


async def test_semantic_recall_empty_project_means_all(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    from openharness.tools.semantic_memory_tool import (
        SemanticMemoryTool,
        SemanticMemoryInput,
    )

    args = SemanticMemoryInput(action="remember", text="alpha fact", project="projA")
    tool = SemanticMemoryTool()
    result = await tool.execute(args, ToolExecutionContext(cwd=tmp_path))
    assert not result.is_error
    recall = await tool.execute(
        SemanticMemoryInput(action="recall", text="alpha", project=""),
        ToolExecutionContext(cwd=tmp_path),
    )
    # project="" must recall across projects (previously silent emptiness).
    assert "alpha fact" in recall.output


# ======================================================================
# T — batch executor retry indices
# ======================================================================

async def test_batch_executor_retry_passes_original_indices():
    from openharness.services.batch_executor import AsyncBatchExecutor, ItemFailure

    executor = AsyncBatchExecutor(concurrency=4)
    seen: list[int] = []

    async def worker(item, index):
        seen.append(index)
        if item == "bad":
            raise ValueError("nope")
        return item

    results = await executor.run_with_retries(
        ["a", "bad", "c"], worker, retries=1
    )
    # Round 1 sees 0,1,2. The retry round must see the ORIGINAL index 1
    # (previously it saw the sublist position 0).
    assert seen == [0, 1, 2, 1]
    assert results == ["a", "bad", "c"] or isinstance(results[1], ItemFailure)


# ======================================================================
# U/V — racing client abort + semantic cache billing
# ======================================================================

async def test_racing_client_cancels_pump_on_abort():
    from openharness.api.client import (
        ApiMessageCompleteEvent,
        ApiMessageRequest,
        ApiTextDeltaEvent,
    )
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage
    from openharness.services.llm_pipeline import RacingClient

    steps = {"n": 0}

    class _SlowClient:
        async def stream_message(self, request):
            for i in range(10):
                steps["n"] += 1
                yield ApiTextDeltaEvent(text="x")
                await asyncio.sleep(0.01)
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[TextBlock(text="x" * 10)]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )

    class _NeverYieldsClient:
        async def stream_message(self, request):
            await asyncio.sleep(30)

    racer = RacingClient([("fast", _SlowClient(), None), ("stuck", _NeverYieldsClient(), None)])
    request = ApiMessageRequest(model="m", messages=[], system_prompt="", max_tokens=10)
    stream = racer.stream_message(request)
    await stream.__anext__()  # consume one delta
    await stream.aclose()  # abort mid-stream
    await asyncio.sleep(0.2)
    assert steps["n"] <= 3  # the winning pump was cancelled, not drained to 10


async def test_semantic_cache_replay_is_free():
    from openharness.api.client import (
        ApiMessageCompleteEvent,
        ApiMessageRequest,
        ApiTextDeltaEvent,
    )
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.messages import ConversationMessage
    from openharness.services.llm_pipeline import InferencePipelineClient, SemanticCache

    class _Store:
        def __init__(self):
            self.data = {}

        def namespace_entries(self, ns, limit):
            return dict(self.data)

        def get(self, ns, key):
            return self.data.get(key)

        def set(self, ns, key, value):
            self.data[key] = value

        def prune_namespace(self, ns, limit):
            pass

    class _Transport:
        async def stream_message(self, request):
            yield ApiTextDeltaEvent(text="answer")
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[TextBlock(text="answer")]),
                usage=UsageSnapshot(input_tokens=100, output_tokens=50),
            )

    store = _Store()
    client = InferencePipelineClient(_Transport(), semantic_cache=SemanticCache(store))
    request = ApiMessageRequest(model="m", messages=[ConversationMessage.from_user_text("q")], system_prompt="", max_tokens=10)

    events = [event async for event in client.stream_message(request)]
    first_usage = events[-1].usage
    assert first_usage.input_tokens == 100

    replay = [event async for event in client.stream_message(request)]
    # The replay is FREE: usage zeroed so the session cost tracker cannot
    # re-bill the cached answer on every hit.
    assert replay[-1].usage.input_tokens == 0
    assert replay[-1].usage.output_tokens == 0
    assert replay[-1].message.text == "answer"


# ======================================================================
# W — permission precedence + report_path visibility
# ======================================================================

def test_deny_rule_beats_allowed_tools():
    from openharness.permissions.checker import (
        PathRule,
        PermissionChecker,
        PermissionSettings,
    )

    settings = PermissionSettings(allowed_tools=["write_file"])
    checker = PermissionChecker(settings)
    checker._path_rules = [PathRule(pattern="*/secrets/*", allow=False)]
    decision = checker.evaluate(
        "write_file", is_read_only=False, file_path="/home/user/project/secrets/key.pem"
    )
    assert decision.allowed is False  # previously allowed_tools bypassed the deny


def test_report_path_visible_to_permission_system(tmp_path: Path):
    from openharness.engine.query import _resolve_permission_file_path

    resolved = _resolve_permission_file_path(
        tmp_path,
        {"report_path": "out/report.json"},
        object(),
    )
    assert resolved is not None
    assert resolved.endswith("report.json")


# ======================================================================
# X — task manager races / argv secrets
# ======================================================================

async def test_agent_task_spawns_without_key_in_argv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-visible-on-ps")
    from openharness.tasks.manager import BackgroundTaskManager

    manager = BackgroundTaskManager()
    captured = {}

    async def fake_create_shell_task(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here — we only inspect the arguments")

    manager.create_shell_task = fake_create_shell_task
    with pytest.raises(RuntimeError):
        await manager.create_agent_task(
            prompt="hi", description="d", cwd=tmp_path, task_type="local_agent"
        )
    argv = captured.get("argv") or []
    assert "sk-visible-on-ps" not in argv
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-visible-on-ps"


async def test_stop_task_notifies_once(tmp_path: Path):
    from openharness.tasks.manager import BackgroundTaskManager

    manager = BackgroundTaskManager()
    record = await manager.create_shell_task(
        command="sleep 30", description="sleeper", cwd=tmp_path
    )
    notifications: list[str] = []

    def listener(task):
        notifications.append(task.status)

    manager.register_completion_listener(listener)
    await asyncio.sleep(0.3)
    stopped = await manager.stop_task(record.id)
    await asyncio.sleep(0.3)
    assert stopped.status == "killed"
    assert notifications.count("killed") == 1
    # No conflicting "failed" notification may sneak in from the watcher.
    assert set(notifications) == {"killed"}


# ======================================================================
# Y — temp frame cleanup
# ======================================================================

def test_extracted_frames_temp_dir_cleaned(tmp_path: Path):
    from openharness.tools.minicpm_vlm_tool import (
        _cleanup_extracted_frames,
        _extract_video_frames,
    )

    video = tmp_path / "v.mp4"
    _make_mp4(video, frames=6)
    frames = _extract_video_frames(video, 3)
    assert frames
    parent = frames[0].parent
    assert parent.name.startswith("minicpm-frames-")
    _cleanup_extracted_frames(frames)
    assert not parent.exists()


# ======================================================================
# Misc — read_task_output byte limit, notebook append, route_engine
# ======================================================================

def test_read_task_output_respects_byte_limit(tmp_path: Path):
    from openharness.tasks.manager import BackgroundTaskManager

    manager = BackgroundTaskManager()
    out_file = tmp_path / "log.txt"
    out_file.write_text("é" * 30000, encoding="utf-8")  # 60 KB multibyte

    class _Task:
        output_file = out_file

    manager._tasks["t1"] = _Task()
    text = manager.read_task_output("t1", max_bytes=12000)
    encoded = text.encode("utf-8")
    assert len(encoded) <= 12000


async def test_notebook_append_adds_newline(tmp_path: Path):
    import json

    from openharness.tools.notebook_edit_tool import (
        NotebookEditTool,
        NotebookEditToolInput,
    )

    nb = tmp_path / "nb.ipynb"
    nb.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "code", "source": "print(1)", "metadata": {}, "outputs": []}],
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    await NotebookEditTool().execute(
        NotebookEditToolInput(path="nb.ipynb", cell_index=0, mode="append", new_source="print(2)"),
        ToolExecutionContext(cwd=tmp_path),
    )
    source = json.loads(nb.read_text(encoding="utf-8"))["cells"][0]["source"]
    assert source == "print(1)\nprint(2)"


def test_route_engine_paragraph_not_manim():
    from openharness.tools.missing_tools import route_engine

    assert route_engine("Write a paragraph about forests", ["hyperframes", "manim"]) != "manim"
