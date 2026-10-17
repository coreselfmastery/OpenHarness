"""Wave-3 bugfix regression tests + the two new video tools.

Covers the six re-reported bugs (the previous "fixes" did not hold):

1. ParallelPromptsTool returns no output  — results now in ``output``.
2. SemanticCache ignores ToolResultBlock + model — both now scoped.
3. CheckpointTool restore still overwritten — rollback-aware adoption.
4. ProgressTracker eta_seconds() sums wrong stages — work stages only.
5. AssetPrefetcher cache key — ``path`` (not ``file_path``).
6. RenderPipeline ffmpeg argv + wait_all() wave race.

Plus the new tools:

- pixel_qa  (deterministic OpenCV frame inspection)
- minicpm_vlm (local MiniCPM-V 4.5 4-bit VLM endpoint)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import MiniCPMVLMConfig
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import AssistantTurnComplete
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.permissions.checker import PermissionSettings
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

# ======================================================================
# Bug 1 — ParallelPromptsTool must return the completions in output
# ======================================================================

class _MockTransport:
    """httpx-friendly transport stub: answers any /chat/completions."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[dict] = []

    def __call__(self, request):  # httpx.MockTransport protocol
        import httpx

        self.requests.append(json.loads(request.content.decode("utf-8")))
        prompt = self.requests[-1]["messages"][-1]["content"]
        content = self.answers.get(prompt, f"echo:{prompt}")
        payload = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }
        return httpx.Response(200, json=payload)


@pytest.mark.asyncio
async def test_parallel_prompts_output_contains_every_answer(monkeypatch):
    import httpx

    from openharness.tools.pipeline_tools import ParallelPromptsInput, ParallelPromptsTool

    transport = httpx.MockTransport(
        _MockTransport({"first question": "ANSWER-ONE", "second question": "ANSWER-TWO"})
    )
    original_client = httpx.AsyncClient

    class _PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PatchedClient)
    try:
        tool = ParallelPromptsTool()
        settings = SimpleNamespace(
            base_url="http://mock/v1",
            api_key="k",
            model="m",
            timeout=30.0,
            qa=None,
            scene_gen=None,
            inference=None,
        )
        result = await tool.execute(
            ParallelPromptsInput(prompts=["first question", "second question"], retries=0),
            ToolExecutionContext(cwd=Path("."), metadata={"settings": settings}),
        )
    finally:
        monkeypatch.undo()

    assert not result.is_error
    # THE BUG: answers used to live only in metadata["results"]; the
    # agent sees only ``output``. Both answers must appear there now.
    assert "ANSWER-ONE" in result.output
    assert "ANSWER-TWO" in result.output
    assert "2/2 prompts succeeded" in result.output
    # metadata remains for programmatic callers
    assert result.metadata["results"] == ["ANSWER-ONE", "ANSWER-TWO"]


@pytest.mark.asyncio
async def test_parallel_prompts_failures_visible_with_partial_success(monkeypatch):
    import httpx

    from openharness.tools.pipeline_tools import ParallelPromptsInput, ParallelPromptsTool

    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        payload = json.loads(request.content.decode("utf-8"))
        prompt = payload["messages"][-1]["content"]
        if prompt == "bad prompt":
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "OK"}}], "usage": {}}
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class _PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PatchedClient)
    try:
        tool = ParallelPromptsTool()
        settings = SimpleNamespace(
            base_url="http://mock/v1", api_key="k", model="m",
            timeout=30.0, qa=None, scene_gen=None, inference=None,
        )
        result = await tool.execute(
            ParallelPromptsInput(prompts=["good prompt", "bad prompt"], retries=0),
            ToolExecutionContext(cwd=Path("."), metadata={"settings": settings}),
        )
    finally:
        monkeypatch.undo()

    assert "OK" in result.output
    assert "FAILED" in result.output or "FAIL" in result.output
    assert result.metadata["executor"]["succeeded"] == 1
    assert result.metadata["executor"]["failed"] == 1


# ======================================================================
# Bug 2 — SemanticCache must fingerprint tool results and scope by model
# ======================================================================

class _FakeStore:
    """Minimal persistent-store stub (namespace_entries/get/set/prune)."""

    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, dict]] = {}

    def namespace_entries(self, namespace: str, limit: int) -> dict:
        entries = self.namespaces.get(namespace, {})
        items = sorted(entries.items(), key=lambda kv: -kv[1].get("created_at", 0))
        return dict(items[:limit])

    def get(self, namespace: str, key: str):
        return self.namespaces.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, payload: dict) -> None:
        self.namespaces.setdefault(namespace, {})[key] = payload

    def prune_namespace(self, namespace: str, keep: int) -> None:
        pass


def _msg(text: str, tool_results: list[tuple[str, str]] | None = None) -> ConversationMessage:
    content: list = [TextBlock(text=text)]
    for use_id, body in tool_results or []:
        content.append(ToolResultBlock(tool_use_id=use_id, content=body))
    return ConversationMessage(role="user", content=content)


def _request(model: str, messages: list[ConversationMessage]):
    from openharness.api.client import ApiMessageRequest

    return ApiMessageRequest(model=model, messages=messages, system_prompt="", max_tokens=64, tools=[])


def _event(text: str) -> ApiMessageCompleteEvent:
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
        usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        stop_reason=None,
    )


def _semantic_cache():
    from openharness.services.llm_pipeline import SemanticCache

    return SemanticCache(_FakeStore())


def test_semantic_cache_same_content_same_model_hits():
    cache = _semantic_cache()
    request = _request("model-a", [_msg("what is the frame count?", [("t1", "frames=240")])])
    cache.store_response(request, _event("there are 240 frames"))
    assert cache.lookup(request) is not None
    assert cache.lookup(request).message.text == "there are 240 frames"


def test_semantic_cache_different_tool_result_no_false_hit():
    cache = _semantic_cache()
    turn1 = _request("model-a", [_msg("what is the frame count?", [("t1", "frames=240")])])
    cache.store_response(turn1, _event("there are 240 frames"))
    # Same user text, DIFFERENT tool result (the agent loop's next turn):
    turn2 = _request("model-a", [_msg("what is the frame count?", [("t1", "frames=480")])])
    assert cache.lookup(turn2) is None, "different tool results must not replay"


def test_semantic_cache_different_model_no_replay():
    cache = _semantic_cache()
    request = _request("model-a", [_msg("explain the render pipeline")])
    cache.store_response(request, _event("answer from model A"))
    switched = _request("model-b", [_msg("explain the render pipeline")])
    assert cache.lookup(switched) is None, "switching models must not replay old answers"


# ======================================================================
# Bug 3 — checkpoint restore must survive the turn that issued it
# ======================================================================

class _RollbackApiClient:
    """Streams a tool_use that rolls the conversation back, then a final
    text answer — exactly the checkpoint-restore mid-turn flow."""

    def __init__(self, engine_holder: dict) -> None:
        self._engine_holder = engine_holder
        self._calls = 0

    async def stream_message(self, request):
        self._calls += 1
        if self._calls == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="toolu_rollback_1", name="rollback_now", input={})
                    ],
                ),
                usage=UsageSnapshot(input_tokens=2, output_tokens=2),
                stop_reason=None,
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text="Rolled back to the checkpoint.")]
            ),
            usage=UsageSnapshot(input_tokens=2, output_tokens=2),
            stop_reason=None,
        )


class _RollbackNowTool(BaseTool):
    """Test tool: restores the engine to a 1-message history mid-turn."""

    name = "rollback_now"
    description = "rollback hook for tests"

    class _Input(BaseModel):
        pass

    input_model = _Input

    def is_read_only(self, arguments) -> bool:
        return True

    async def execute(self, arguments, context) -> ToolResult:
        engine = (context.metadata or {}).get("engine_ref")
        engine.rollback_messages(
            [ConversationMessage(role="user", content=[TextBlock(text="restored base")])]
        )
        return ToolResult(output="rolled back")


@pytest.mark.asyncio
async def test_midturn_checkpoint_restore_survives_turn_completion(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    engine_holder: dict = {}
    registry = ToolRegistry()
    registry.register(_RollbackNowTool())
    engine = QueryEngine(
        api_client=_RollbackApiClient(engine_holder),
        tool_registry=registry,
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        tool_metadata={"engine_ref": None},
    )
    engine.tool_metadata["engine_ref"] = engine

    # Pre-turn history the restore should wipe.
    engine.load_messages(
        [
            ConversationMessage(role="user", content=[TextBlock(text="old turn 1")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="old answer 1")]),
            ConversationMessage(role="user", content=[TextBlock(text="old turn 2")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="old answer 2")]),
        ]
    )

    events = [event async for event in engine.submit_message("please roll back")]
    assert isinstance(events[-1], AssistantTurnComplete)

    texts = [message.text for message in engine.messages]
    # THE BUG: submit_message used to re-adopt the pre-rollback copy and
    # resurrect "old turn 1/2". The restored base must survive.
    assert "restored base" in texts, f"restore was overwritten: {texts}"
    assert "old answer 1" not in texts and "old answer 2" not in texts
    # The post-rollback tail (final answer the user saw streamed) is kept.
    assert "Rolled back to the checkpoint." in texts


@pytest.mark.asyncio
async def test_no_rollback_turn_adopts_normally(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)

    class _PlainClient:
        async def stream_message(self, request):
            del request
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="normal answer")]
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason=None,
            )

    engine = QueryEngine(
        api_client=_PlainClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    [event async for event in engine.submit_message("hi")]
    texts = [message.text for message in engine.messages]
    assert texts == ["hi", "normal answer"]


# ======================================================================
# Bug 4 — ProgressTracker ETA + pending -> failed transition
# ======================================================================

def test_progress_pending_to_failed_allowed():
    from openharness.services.progress_tracker import ProgressTracker

    tracker = ProgressTracker()
    assert tracker.mark("scene_1", "failed", detail="prompt rejected") is True
    assert tracker.scene_status("scene_1")["stage"] == "failed"


def test_progress_generated_and_queued_to_failed_allowed():
    from openharness.services.progress_tracker import ProgressTracker

    tracker = ProgressTracker()
    # Valid path: pending -> generating -> generated -> failed.
    assert tracker.mark("s1", "generating")
    assert tracker.mark("s1", "generated")
    assert tracker.mark("s1", "failed", detail="render spawn error") is True
    # Valid path: pending -> queued -> failed.
    assert tracker.mark("s2", "queued")
    assert tracker.mark("s2", "failed", detail="queue rejected") is True


def test_progress_eta_uses_work_stages_not_waypoints():
    from openharness.services.progress_tracker import ProgressTracker

    tracker = ProgressTracker()
    # Seed observed WORK durations directly (what real pipelines record
    # when scenes leave generating/rendering/qa).
    tracker._stage_durations["generating"].extend([10.0, 10.0])
    tracker._stage_durations["rendering"].extend([20.0, 20.0])
    tracker._stage_durations["qa"].extend([5.0, 5.0])
    # Passive waypoint durations stay empty — the old code summed
    # generated/rendered/approved and always got ~0/None.

    tracker.force("done_scene", "approved")
    tracker.force("render_scene", "rendering")  # in flight, active worker

    eta = tracker.eta_seconds()
    # render_scene owes the rest of rendering (20) + qa (5) = 25s,
    # parallelism 1 (one active scene) -> 25.0
    assert eta is not None and abs(eta - 25.0) < 0.2, f"eta was {eta}"


def test_progress_eta_pending_scene_owes_full_pipeline():
    from openharness.services.progress_tracker import ProgressTracker

    tracker = ProgressTracker()
    tracker._stage_durations["generating"].extend([10.0])
    tracker._stage_durations["rendering"].extend([20.0])
    tracker._stage_durations["qa"].extend([5.0])
    tracker.force("fresh", "pending")  # owes 10 + 20 + 5 = 35s
    tracker.force("active", "rendering")  # owes 20 + 5 = 25s; parallelism 1
    eta = tracker.eta_seconds()
    assert eta is not None and abs(eta - 60.0) < 0.2, f"eta was {eta}"


def test_progress_eta_none_without_work_observations():
    from openharness.services.progress_tracker import ProgressTracker

    tracker = ProgressTracker()
    tracker._stage_durations["generated"].extend([0.01])  # passive only
    tracker.force("s", "pending")
    assert tracker.eta_seconds() is None


# ======================================================================
# Bug 5 — AssetPrefetcher warms read_file under the `path` key
# ======================================================================

@pytest.mark.asyncio
async def test_prefetch_cache_hits_relative_path_call_shape(tmp_path: Path):
    from openharness.services.asset_prefetch import AssetPrefetcher
    from openharness.services.tool_cache import (
        enable_tool_cache,
        get_tool_cache,
        reset_tool_cache,
    )

    asset = tmp_path / "logo.svg"
    asset.write_text("<svg>brand mark</svg>", encoding="utf-8")
    scene = tmp_path / "scene_001.py"
    scene.write_text("draw('logo.svg')\n", encoding="utf-8")

    enable_tool_cache(max_entries=20, ttl_seconds=60)
    try:
        await AssetPrefetcher().prefetch_scenes([scene], cwd=tmp_path)
        cache = get_tool_cache()
        # The model usually sends the RELATIVE path — that shape must hit.
        cached = cache.get("read_file", {"path": "logo.svg"})
        assert cached is not None
        assert "<svg>brand mark</svg>" in cached["output"]
        # And the output is line-numbered like a real read_file call.
        assert cached["output"].splitlines()[0].startswith("     1\t")
    finally:
        reset_tool_cache()


# ======================================================================
# Bug 6 — RenderPipeline ffmpeg argv + wait_all dependency waves
# ======================================================================

def test_build_render_command_ffmpeg_splits_multi_flag_args(tmp_path: Path):
    from openharness.services.render_pipeline import build_render_command

    scene = tmp_path / "scene.filter"
    scene.write_text("[0:v]drawtext=text='hi'[v0]", encoding="utf-8")
    argv = build_render_command(
        engine="ffmpeg",
        scene_path=scene,
        output_path=tmp_path / "out.mp4",
        tier="draft",
        width=640,
        height=360,
        fps=30,
        codec_args="-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p",
        output_dir=tmp_path,
    )
    # input args must be separate argv tokens
    assert "-f" in argv and "lavfi" in argv and "-i" in argv
    assert "color=c=black:s=640x360:d=10:r=30" in argv
    # codec args must be separate argv tokens
    assert "-c:v" in argv and "libx264" in argv and "-preset" in argv
    assert "veryfast" in argv and "-crf" in argv and "20" in argv
    # the old bug produced ONE token "-c:v libx264 -preset veryfast ..."
    assert all(
        " " not in tok or tok.startswith("[") for tok in argv
    ), f"multi-flag string leaked as single argv token: {argv}"


def test_build_render_command_hyperframes_untouched(tmp_path: Path):
    from openharness.services.render_pipeline import build_render_command

    scene = tmp_path / "scene.js"
    scene.write_text("export default {}", encoding="utf-8")
    argv = build_render_command(
        engine="hyperframes",
        scene_path=scene,
        output_path=tmp_path / "out.mp4",
        tier="final",
        width=1920,
        height=1080,
        fps=30,
        codec_args="-c:v libx264",
        output_dir=tmp_path,
    )
    assert argv[:2] == ["hyperframes", "render"]
    assert "--width" in argv and "1920" in argv


@pytest.mark.asyncio
async def test_wait_all_blocks_until_all_dependency_waves_finish(tmp_path: Path):
    """3 chained waves: wait_all must not return during the submission
    gap between waves (the old race returned after wave 1)."""
    import sys

    from openharness.services.render_pipeline import RenderPipeline

    # A fake engine whose binary is python itself: sleeps, then writes
    # the output file so the job counts as done.
    render_engines = {
        "slowpy": [
            sys.executable, "-c",
            (
                "import sys, time; time.sleep(0.4); "
                "open(sys.argv[1], 'w').write('ok')"
            ),
            "{output}",
        ]
    }
    (tmp_path / "render_engines.json").write_text(json.dumps(render_engines))

    # Chain: a -> b -> c  (one scene per wave).
    (tmp_path / "a.py").write_text("# depends_on: nothing\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("# depends_on: a\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("# depends_on: b\n", encoding="utf-8")

    async with asyncio.timeout(30):
        pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
        pipeline._templates = render_engines  # already loaded from json, but be explicit
        first = pipeline.submit_batch(
            [tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"],
            engine="slowpy",
            tier="final",
        )
        assert len(first) == 1, "wave 1 = [a] only"
        jobs = await pipeline.wait_all(timeout=25)
        await pipeline.shutdown(drain=False)
        assert len(jobs) == 3, f"expected all 3 waves submitted, got {len(jobs)}"
        assert all(job.status in {"done", "failed", "timeout", "skipped"} for job in jobs)
        assert all(job.status == "done" for job in jobs), [
            (j.scene.name, j.status, j.error) for j in jobs
        ]


# ======================================================================
# New tool — pixel_qa
# ======================================================================

def _make_test_video(path: Path, palette_bgr: tuple[int, int, int]) -> None:
    """3 segments: animated brand-bg frames, frozen frames, black frames."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
    )
    for i in range(12):  # animated segment
        frame = np.full((180, 320, 3), palette_bgr, dtype=np.uint8)
        x = 20 + i * 8
        cv2.rectangle(frame, (x, 60), (x + 90, 120), (17, 17, 17), -1)
        writer.write(frame)
    frozen = np.full((180, 320, 3), palette_bgr, dtype=np.uint8)
    cv2.rectangle(frozen, (60, 60), (150, 120), (17, 17, 17), -1)
    for _ in range(8):  # frozen segment (identical frames)
        writer.write(frozen)
    for _ in range(8):  # black segment
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()


def test_pixel_qa_service_flags_black_frozen_and_passes_clean(tmp_path: Path):
    from openharness.services.pixel_qa import PixelQA

    video = tmp_path / "scene_main.mp4"
    _make_test_video(video, (235, 240, 242))  # #F2F0EB in BGR

    qa = PixelQA()
    report = qa.inspect_video(
        video,
        palette=[(235, 240, 242), (17, 17, 17)],
        sample_fps=10.0,  # sample every frame
    )
    checks = {issue.check for issue in report.issues}
    assert "black_frame" in checks, f"black frames not flagged: {checks}"
    assert "frozen_frame" in checks, f"frozen frames not flagged: {checks}"
    assert report.passed is False
    assert report.frames_sampled > 20
    assert report.score < 1.0

    # A clean animated video passes.
    clean = tmp_path / "clean.mp4"
    _make_test_video(clean, (235, 240, 242))
    import cv2
    import numpy as np

    # Trim the bad segments by writing a fresh 12-frame animated video.
    writer = cv2.VideoWriter(
        str(clean), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
    )
    for i in range(12):
        frame = np.full((180, 320, 3), (235, 240, 242), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i * 8, 60), (110 + i * 8, 120), (17, 17, 17), -1)
        writer.write(frame)
    writer.release()
    clean_report = qa.inspect_video(
        clean, palette=[(235, 240, 242), (17, 17, 17)], sample_fps=10.0
    )
    assert clean_report.passed is True, [
        (i.check, i.message) for i in clean_report.issues
    ]


def test_load_brand_palette_from_openvideoharness_tokens(tmp_path: Path):
    from openharness.services.pixel_qa import load_brand_palette

    tokens = {
        "slug": "swiss-grid-type",
        "palette": {
            "bg": "#F2F0EB",
            "fg": "#111111",
            "accent": "#E1251B",
            "extra": ["#CFCCC4", "#E2E0DA"],
        },
    }
    tokens_path = tmp_path / "tokens.json"
    tokens_path.write_text(json.dumps(tokens), encoding="utf-8")
    palette = load_brand_palette(tokens_path)
    assert (235, 240, 242) in palette  # #F2F0EB -> BGR
    assert (17, 17, 17) in palette  # #111111
    assert (27, 37, 225) in palette  # #E1251B -> BGR
    assert len(palette) == 5


@pytest.mark.asyncio
async def test_pixel_qa_tool_end_to_end(tmp_path: Path):
    from openharness.services.progress_tracker import (
        get_progress_tracker,
        set_progress_tracker,
    )
    from openharness.tools.pixel_qa_tool import PixelQAInput, PixelQATool

    video = tmp_path / "scene_hero.mp4"
    _make_test_video(video, (235, 240, 242))
    tokens = {
        "palette": {"bg": "#F2F0EB", "fg": "#111111", "accent": "#E1251B"}
    }
    tokens_path = tmp_path / "tokens.json"
    tokens_path.write_text(json.dumps(tokens), encoding="utf-8")

    tracker = get_progress_tracker()
    tracker.force("scene_hero", "rendered")  # simulate render completion
    try:
        tool = PixelQATool()
        result = await tool.execute(
            PixelQAInput(paths=[str(video)], palette_file=str(tokens_path), sample_fps=10.0),
            ToolExecutionContext(cwd=tmp_path),
        )
    finally:
        pass
    assert result.is_error  # hard failure (black + frozen frames)
    assert "scene_hero" in result.output
    assert "black_frame" in result.output
    assert "frozen_frame" in result.output
    # Progress tracker integration: marked qa, then failed.
    assert tracker.scene_status("scene_hero")["stage"] == "failed"
    # Report file option works.
    result2 = await PixelQATool().execute(
        PixelQAInput(
            paths=[str(video)],
            palette=["#F2F0EB", "#111111"],
            sample_fps=10.0,
            report_path=str(tmp_path / "report.json"),
        ),
        ToolExecutionContext(cwd=tmp_path),
    )
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved and saved[0]["scene"] == "scene_hero"
    assert result2.metadata["report_path"].endswith("report.json")
    set_progress_tracker(None)


# ======================================================================
# New tool — minicpm_vlm
# ======================================================================

@pytest.mark.asyncio
async def test_minicpm_vlm_analyzes_image_with_mock_endpoint(tmp_path, monkeypatch):
    from openharness.tools import minicpm_vlm_tool
    from openharness.tools.minicpm_vlm_tool import MiniCPMVLMInput, MiniCPMVLMTool

    image = tmp_path / "frame.png"
    import cv2
    import numpy as np

    cv2.imwrite(str(image), np.full((64, 64, 3), 200, dtype=np.uint8))

    async def fake_post(*, base_url, api_key, payload, timeout_seconds, transport=None):
        assert payload["model"] == "openbmb/MiniCPM-V-4_5"
        assert payload["messages"][0]["content"][0]["type"] == "text"
        assert payload["messages"][0]["content"][1]["type"] == "image_url"
        return {
            "choices": [{"message": {"content": "A plain gray frame."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 8},
        }

    monkeypatch.setattr(minicpm_vlm_tool, "_post_chat_completion", fake_post)
    tool = MiniCPMVLMTool()
    settings = SimpleNamespace(minicpm_v=MiniCPMVLMConfig())
    result = await tool.execute(
        MiniCPMVLMInput(prompt="describe", images=[str(image)]),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings}),
    )
    assert not result.is_error
    assert "A plain gray frame." in result.output
    assert "MiniCPM-V" in result.output
    assert result.metadata["model"] == "openbmb/MiniCPM-V-4_5"
    assert result.metadata["usage"]["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_minicpm_vlm_reports_unreachable_endpoint(tmp_path):
    from openharness.tools.minicpm_vlm_tool import MiniCPMVLMInput, MiniCPMVLMTool

    image = tmp_path / "frame.png"
    import cv2
    import numpy as np

    cv2.imwrite(str(image), np.full((32, 32, 3), 120, dtype=np.uint8))

    # Port 9 (discard) is closed in every sane environment.
    settings = SimpleNamespace(
        minicpm_v=MiniCPMVLMConfig(
            base_url="http://127.0.0.1:9/v1", timeout_seconds=0.5
        )
    )
    result = await MiniCPMVLMTool().execute(
        MiniCPMVLMInput(prompt="describe", images=[str(image)]),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings}),
    )
    assert result.is_error
    assert "unreachable" in result.output or "DOWN" in result.output


@pytest.mark.asyncio
async def test_minicpm_vlm_requires_input(tmp_path):
    from openharness.tools.minicpm_vlm_tool import MiniCPMVLMInput, MiniCPMVLMTool

    result = await MiniCPMVLMTool().execute(
        MiniCPMVLMInput(prompt="describe"),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": SimpleNamespace(minicpm_v=MiniCPMVLMConfig())}),
    )
    assert result.is_error
    assert "images" in result.output


def test_new_tools_registered():
    from openharness.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    assert registry.get("pixel_qa") is not None
    assert registry.get("minicpm_vlm") is not None


def test_settings_expose_minicpm_config():
    from openharness.config.settings import Settings

    settings = Settings()
    assert settings.minicpm_v.model == "openbmb/MiniCPM-V-4_5"
    assert settings.minicpm_v.quantization == "4-bit"
    assert settings.minicpm_v.is_configured
