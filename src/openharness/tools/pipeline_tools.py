"""Pipeline tools — the 40-item performance & safety feature surface.

Tools exposed to the agent (each wired to the real services):

- ``render_scenes`` (#11-#16, #21, #24, #25): parallel, tiered,
  cached, NVENC-aware scene rendering through the render pipeline.
- ``render_queue_status`` (#40): what is rendering, on which worker,
  with per-job timing.
- ``qa_review`` (#7-#10): parallel QA waves / batching / smart depth /
  failure-prediction-aware review, with optional draft promotion.
- ``oh_control`` (#26, #30, #34, #35, #36): kill switch, pause/resume,
  message injection, watchdog + deadlock status, graceful shutdown.
- ``checkpoint`` (#37): create/list/restore/diff conversation
  checkpoints (rollback more than one step).
- ``progress`` (#38, #39): whole-pipeline progress with ETA and
  per-scene status board.
- ``audio_gen`` (#22): real TTS via an OpenAI-compatible /audio/speech
  endpoint, foreground or background.
- ``server_config`` (#41-#46): generates a working vLLM launch config
  for the server-side optimizations + hardware report (#47-#50).
- ``cache_stats`` (#17-#19): what the caches did (hits/misses/sizes)
  and how to clear them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


def _settings_from(context: ToolExecutionContext) -> Any:
    return (context.metadata or {}).get("settings")


def _session_id_from(context: ToolExecutionContext) -> str:
    return str((context.metadata or {}).get("session_id") or "default")


# ======================================================================
# #11-#16, #21, #24, #25 — render_scenes
# ======================================================================

class RenderScenesInput(BaseModel):
    scenes: list[str] = Field(
        description=(
            "Scene file paths to render (or a directory — pass it as a single "
            "entry ending in '/')."
        )
    )
    engine: str = Field(
        default="hyperframes",
        description="Rendering engine: hyperframes | manim | blender | ffmpeg.",
    )
    tier: str = Field(
        default="final",
        description="Quality tier: draft (640x360) | review (720p) | final (1080p).",
    )
    mode: str = Field(
        default="wait",
        description="wait = block until rendered; background = return job ids immediately (pipeline overlap).",
    )
    output_dir: str | None = Field(
        default=None,
        description="Where rendered files go. Default: ./renders",
    )
    respect_dependencies: bool = Field(
        default=True,
        description="Honor '# depends_on:' scene comments; independent scenes render in parallel.",
    )


class RenderScenesTool(BaseTool):
    name = "render_scenes"
    description = (
        "Render scene files in PARALLEL across all GPUs/cores with caching. "
        "Skips scenes whose code+assets are unchanged (fingerprint cache), "
        "uses hardware NVENC when a real probe passes, honors '# depends_on:' "
        "ordering, and can run in background while you keep generating. "
        "Tiers: draft/review/final map to 360p/720p/1080p."
    )
    input_model = RenderScenesInput

    def is_read_only(self, arguments: RenderScenesInput) -> bool:
        return False

    async def execute(self, arguments: RenderScenesInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.progress_tracker import get_progress_tracker
        from openharness.services.render_pipeline import get_render_pipeline

        settings = _settings_from(context)
        pipeline = get_render_pipeline(context.cwd, settings)
        tracker = get_progress_tracker()

        scene_paths: list[Path] = []
        for entry in arguments.scenes:
            raw = Path(entry)
            if not raw.is_absolute():
                raw = context.cwd / raw
            if raw.is_dir():
                scene_paths.extend(sorted(raw.iterdir()))
            else:
                scene_paths.append(raw)
        scene_paths = [p for p in scene_paths if p.is_file()]
        if not scene_paths:
            return ToolResult(
                output="No scene files found to render. Pass scene file paths or a directory.",
                is_error=True,
            )

        for scene in scene_paths:
            tracker.mark(scene.stem, "queued", tier=arguments.tier)

        output_dir = (
            (context.cwd / arguments.output_dir) if arguments.output_dir else None
        )
        jobs = pipeline.submit_batch(
            scene_paths,
            engine=arguments.engine,
            tier=arguments.tier,
            output_dir=output_dir,
            respect_dependencies=arguments.respect_dependencies,
        )

        if arguments.mode == "background":
            lines = [
                f"[render_scenes] {len(scene_paths)} scene(s) submitted in background.",
                f"Workers: {pipeline.workers}; tier: {arguments.tier}; engine: {arguments.engine}.",
                "Track with render_queue_status. Job ids: "
                + ", ".join(j.job_id for j in jobs),
            ]
            return ToolResult(output="\n".join(lines), metadata={"job_ids": [j.job_id for j in jobs]})

        await pipeline.wait_all()
        jobs_sorted = sorted(pipeline.jobs_snapshot(), key=lambda j: j.job_id)
        lines = [f"[render_scenes] {len(scene_paths)} scene(s), tier={arguments.tier}, engine={arguments.engine}"]
        passed = failed = cached = 0
        for job in jobs_sorted:
            if job.status == "done":
                passed += 1
                tracker.mark(job.scene.stem, "rendered", tier=job.tier)
                lines.append(
                    f"  ok   {job.scene.name} -> {job.output_path.name} "
                    f"({job.duration_seconds:.1f}s)"
                )
            elif job.status == "skipped":
                cached += 1
                tracker.mark(job.scene.stem, "rendered", tier=job.tier, detail="cache hit")
                lines.append(
                    f"  skip {job.scene.name} (cache hit -> {job.output_path.name})"
                )
            else:
                failed += 1
                tracker.mark(job.scene.stem, "failed", detail=job.error[:120])
                lines.append(f"  FAIL {job.scene.name}: {job.error[:200]}")
        codec = pipeline.status().get("codec", "?")
        lines.append(
            f"Summary: {passed} rendered, {cached} cache-skipped, {failed} failed. Codec: {codec}."
        )
        return ToolResult(
            output="\n".join(lines),
            is_error=failed > 0 and passed + cached == 0,
            metadata={"rendered": passed, "cached": cached, "failed": failed},
        )


# ======================================================================
# #40 — render_queue_status
# ======================================================================

class RenderQueueStatusInput(BaseModel):
    recent: int = Field(default=15, ge=1, le=100, description="How many recent jobs to include.")


class RenderQueueStatusTool(BaseTool):
    name = "render_queue_status"
    description = (
        "Monitor the render queue: which workers are busy, what each is "
        "rendering, queue depth, cache hits, average render time, and the "
        "outcome of recent jobs."
    )
    input_model = RenderQueueStatusInput

    def is_read_only(self, arguments: RenderQueueStatusInput) -> bool:
        return True

    async def execute(self, arguments: RenderQueueStatusInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.render_pipeline import _PIPELINE, get_render_pipeline

        if _PIPELINE is None:
            get_render_pipeline(context.cwd, _settings_from(context))
        if _PIPELINE is None:  # pragma: no cover - defensive
            return ToolResult(output="Render pipeline not started yet; render something first.")
        status = _PIPELINE.status()
        lines = ["[render_queue_status]"]
        for worker_id, state in status["workers"].items():
            current = f" job={state['current_job']}" if state["current_job"] else ""
            lines.append(
                f"  worker {worker_id}: {'BUSY' if state['busy'] else 'idle'}{current} (done={state['completed']})"
            )
        lines.append(f"  queue depth: {status['queue_depth']} (per-worker {status['queue_per_worker']})")
        lines.append(f"  work steals: {status['work_steals']}")
        lines.append(f"  jobs: {json.dumps(status['jobs'])}")
        if status["avg_render_seconds"] is not None:
            lines.append(f"  avg render: {status['avg_render_seconds']}s  codec: {status['codec']}")
        recent = _PIPELINE.recent_jobs(arguments.recent)
        for job in recent:
            lines.append(f"  [{job['job_id']}] {job['scene']} {job['tier']} -> {job['status']} ({job['duration_seconds']}s)")
        return ToolResult(output="\n".join(lines), metadata=status)


# ======================================================================
# #7-#10 — qa_review
# ======================================================================

class QAReviewInput(BaseModel):
    scenes: list[str] = Field(description="Scene file paths to review.")
    engine: str = Field(default="hyperframes", description="Engine the scenes target.")
    tier: str = Field(default="draft", description="Tier being reviewed (draft/review/final).")
    mode: str = Field(
        default="auto",
        description="auto = batch simple scenes + parallel deep reviews; single = one call per scene; batch = pack scenes per call.",
    )
    promote: bool = Field(
        default=False,
        description="After review, promote passing scenes (early-exit a good draft or enqueue the final render).",
    )


class QAReviewTool(BaseTool):
    name = "qa_review"
    description = (
        "Review generated scenes with parallel QA: simple scenes get a "
        "lightweight checklist in batched calls, complex scenes get deep "
        "single reviews — all in parallel waves. Failure categories from "
        "project history are checked pre-emptively, and mechanical "
        "preflight findings (non-determinism, missing assets) are included. "
        "Optionally promotes passing drafts (quality-based early exit)."
    )
    input_model = QAReviewInput

    def is_read_only(self, arguments: QAReviewInput) -> bool:
        return True

    async def execute(self, arguments: QAReviewInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.progress_tracker import get_progress_tracker
        from openharness.services.qa_pipeline import build_qa_reviewer_from_metadata
        from openharness.services.render_pipeline import _PIPELINE

        scene_paths = []
        for entry in arguments.scenes:
            raw = Path(entry)
            if not raw.is_absolute():
                raw = context.cwd / raw
            scene_paths.append(raw)
        scene_paths = [p for p in scene_paths if p.is_file()]
        if not scene_paths:
            return ToolResult(output="No scene files found to review.", is_error=True)

        reviewer = build_qa_reviewer_from_metadata(context.metadata)
        if reviewer is None:
            return ToolResult(
                output=(
                    "QA reviewer is not configured: no LLM endpoint. Set "
                    "settings.qa (model/api_key/base_url) or scene_gen config "
                    "(OPENHARNESS_QA_MODEL / OPENHARNESS_QA_API_KEY / "
                    "OPENHARNESS_QA_BASE_URL env vars). No verdicts were invented."
                ),
                is_error=True,
            )

        tracker = get_progress_tracker()
        for scene in scene_paths:
            tracker.mark(scene.stem, "qa")

        settings = _settings_from(context)
        rendering = getattr(settings, "rendering", None)
        resolution = ""
        if rendering is not None:
            resolution = dict(getattr(rendering, "resolution_tiers", {}) or {}).get(
                arguments.tier, ""
            )
        try:
            verdicts = await reviewer.review_scenes(
                scene_paths,
                engine=arguments.engine,
                tier=arguments.tier,
                resolution=resolution or "640x360",
                mode=arguments.mode,
            )
        finally:
            # The reviewer pools one httpx.AsyncClient for its lifetime
            # (connection reuse across the review waves) — release it
            # here so every qa_review tool call closes its own client
            # instead of leaking it to the GC.
            await reviewer.aclose()

        lines = [f"[qa_review] {len(verdicts)} scene(s) reviewed (model: {reviewer.stats()['model']})"]
        promote_results: list[str] = []
        threshold = (
            float(getattr(rendering, "early_exit_score_threshold", 0.9) or 0.9)
            if rendering is not None
            else 0.9
        )
        for verdict in verdicts:
            status_word = "PASS" if verdict.passed else "FAIL"
            lines.append(
                f"  {status_word}  {verdict.scene}  score={verdict.score:.2f} depth={verdict.depth}"
            )
            if verdict.summary:
                lines.append(f"        {verdict.summary}")
            for issue in verdict.issues[:5]:
                lines.append(
                    f"        - [{issue.get('severity', '?')}] {issue.get('category')}: "
                    f"{issue.get('detail')} -> {issue.get('fix')}"
                )
            # Record the score with the render pipeline for early exit (#13).
            if _PIPELINE is not None:
                scene_path = next((p for p in scene_paths if p.name == verdict.scene or p.stem == verdict.scene), None)
                if scene_path is not None:
                    _PIPELINE.record_quality(scene_path, verdict.score, tier=arguments.tier)
                    if verdict.passed:
                        tracker.mark(scene_path.stem, "approved", detail=verdict.summary[:100])
                    else:
                        tracker.mark(scene_path.stem, "generated", detail=f"QA fail: {verdict.summary[:100]}")
            if arguments.promote and verdict.passed and _PIPELINE is not None:
                scene_path = next((p for p in scene_paths if p.name == verdict.scene or p.stem == verdict.scene), None)
                if scene_path is not None:
                    job = _PIPELINE.promote(
                        scene_path,
                        engine=arguments.engine,
                        early_exit_threshold=threshold,
                    )
                    if job.promoted:
                        promote_results.append(f"{verdict.scene}: early-exit (draft promoted)")
                    elif job.status in {"pending", "running"}:
                        promote_results.append(f"{verdict.scene}: final render enqueued")
        if promote_results:
            lines.append("[promotion]")
            lines.extend(f"  {line}" for line in promote_results)
        passed = sum(1 for v in verdicts if v.passed)
        lines.append(
            f"Summary: {passed}/{len(verdicts)} passed. LLM calls used: {reviewer.stats()['calls_made']}."
        )
        return ToolResult(
            output="\n".join(lines),
            metadata={
                "verdicts": [v.as_dict() for v in verdicts],
                "passed": passed,
                "total": len(verdicts),
            },
        )


# ======================================================================
# #26, #30, #34, #35, #36 — oh_control
# ======================================================================

class ControlInput(BaseModel):
    action: str = Field(
        description=(
            "status | pause | resume | kill | inject | deadlock_check | shutdown"
        )
    )
    message: str = Field(
        default="",
        description="For inject: the human decision/message to insert at the next turn. For kill/pause: the reason.",
    )


class ControlTool(BaseTool):
    name = "oh_control"
    description = (
        "Control the running agent pipeline: pause/resume for inspection, a "
        "kill switch that stops everything immediately, human message "
        "injection mid-task, watchdog/deadlock status, and graceful shutdown "
        "that finishes in-flight work first."
    )
    input_model = ControlInput

    def is_read_only(self, arguments: ControlInput) -> bool:
        return arguments.action in {"status", "deadlock_check"}

    async def execute(self, arguments: ControlInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.control_bus import get_control_bus

        bus = get_control_bus()
        action = arguments.action.strip().lower()
        if action == "status":
            return ToolResult(
                output="[oh_control] " + json.dumps(bus.status(), indent=2),
                metadata=bus.status(),
            )
        if action == "pause":
            bus.pause(arguments.message or "paused via oh_control tool")
            return ToolResult(output="Pipeline PAUSED. The query loop waits at the next turn boundary.")
        if action == "resume":
            bus.resume()
            return ToolResult(output="Pipeline RESUMED.")
        if action == "kill":
            bus.trip(arguments.message or "killed via oh_control tool")
            return ToolResult(
                output=(
                    "KILL SWITCH TRIPPED. In-flight tool calls finish, then the "
                    "session aborts at the next turn boundary."
                )
            )
        if action == "inject":
            if not arguments.message.strip():
                return ToolResult(output="inject requires a message.", is_error=True)
            pending = bus.inject_message(arguments.message)
            return ToolResult(
                output=(
                    f"Message queued; it will be appended as the next user turn "
                    f"({pending} pending). The running turn completes first."
                )
            )
        if action == "deadlock_check":
            report = bus.detect_deadlock()
            if report is None:
                return ToolResult(output="No deadlock detected. Subsystems have running work or idle capacity.")
            return ToolResult(
                output="DEADLOCK: " + json.dumps(report, indent=2),
                is_error=True,
                metadata=report,
            )
        if action == "shutdown":
            await bus.begin_shutdown(reason="oh_control tool")
            return ToolResult(output="Graceful shutdown complete: in-flight work drained, resources released.")
        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)


# ======================================================================
# #37 — checkpoint
# ======================================================================

class CheckpointInput(BaseModel):
    action: str = Field(description="create | list | restore | diff | prune")
    label: str = Field(default="", description="Label for create (e.g. 'before refactor').")
    checkpoint_id: str = Field(default="", description="Checkpoint id for restore/diff.")
    second_id: str = Field(default="", description="Other checkpoint id for diff.")
    keep: int = Field(default=40, ge=1, le=500, description="For prune: how many to keep.")


class CheckpointTool(BaseTool):
    name = "checkpoint"
    description = (
        "Conversation checkpoints with rollback: snapshot the session now, "
        "list snapshots, roll back to any earlier point (more than one "
        "step), or diff two snapshots. Auto-checkpoints also happen every "
        "N turns."
    )
    input_model = CheckpointInput

    def is_read_only(self, arguments: CheckpointInput) -> bool:
        return arguments.action in {"list", "diff"}

    async def execute(self, arguments: CheckpointInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.engine.messages import ConversationMessage
        from openharness.services.checkpoints import CheckpointManager

        metadata = context.metadata or {}
        engine = metadata.get("engine_ref")
        manager = CheckpointManager(_session_id_from(context))
        action = arguments.action.strip().lower()

        if action == "create":
            if engine is None:
                return ToolResult(output="No live conversation available for checkpointing.", is_error=True)
            summary = manager.create(
                engine.live_messages, arguments.label or "", tool_metadata=None
            )
            return ToolResult(
                output=(
                    f"[checkpoint] created {summary['id']} "
                    f"({summary['message_count']} messages, label={summary['label']!r})"
                ),
                metadata=summary,
            )
        if action == "list":
            items = manager.list()
            if not items:
                return ToolResult(output="No checkpoints yet. Create one with action=create.")
            lines = ["[checkpoint] available (newest first):"]
            for item in items:
                lines.append(
                    f"  {item['id']}  {item['created_at_iso']}  {item['message_count']:>3} msgs  "
                    f"{item['label']}  | {item['last_user_text']}"
                )
            return ToolResult(output="\n".join(lines), metadata={"checkpoints": items})
        if action == "restore":
            if engine is None:
                return ToolResult(output="No live conversation to roll back.", is_error=True)
            if not arguments.checkpoint_id:
                return ToolResult(output="restore requires checkpoint_id.", is_error=True)
            payload = manager.restore(arguments.checkpoint_id)
            if payload is None:
                return ToolResult(
                    output=f"Unknown checkpoint: {arguments.checkpoint_id}", is_error=True
                )
            restored = []
            for raw in payload.get("messages") or []:
                try:
                    restored.append(ConversationMessage.model_validate(raw))
                except Exception:
                    continue
            before = len(engine.live_messages)
            engine.rollback_messages(restored)
            return ToolResult(
                output=(
                    f"[checkpoint] rolled back to {arguments.checkpoint_id}: "
                    f"{before} -> {len(restored)} messages "
                    f"(label={payload.get('label')!r}). The conversation continues from there."
                )
            )
        if action == "diff":
            if not (arguments.checkpoint_id and arguments.second_id):
                return ToolResult(output="diff requires checkpoint_id and second_id.", is_error=True)
            diff = manager.diff(arguments.checkpoint_id, arguments.second_id)
            if diff is None:
                return ToolResult(output="One of the checkpoint ids is unknown.", is_error=True)
            return ToolResult(
                output=(
                    f"[checkpoint diff]\n  a: {diff['a']['id']} ({diff['a']['message_count']} msgs)\n"
                    f"  b: {diff['b']['id']} ({diff['b']['message_count']} msgs)\n"
                    f"  message delta: {diff['message_delta']:+d}\n"
                    f"  identical: {diff['different'] is False}"
                ),
                metadata=diff,
            )
        if action == "prune":
            removed = manager.prune(arguments.keep)
            return ToolResult(output=f"[checkpoint] pruned {removed} old checkpoint(s), keeping {arguments.keep}.")
        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)


# ======================================================================
# #38, #39 — progress
# ======================================================================

class ProgressInput(BaseModel):
    action: str = Field(default="summary", description="summary | scenes | scene")
    scene: str = Field(default="", description="Scene name for action=scene.")


class ProgressTool(BaseTool):
    name = "progress"
    description = (
        "Real-time pipeline progress: overall completion, per-stage counts, "
        "ETA from observed stage durations, and a per-scene status board "
        "(pending/generating/rendering/qa/approved/failed)."
    )
    input_model = ProgressInput

    def is_read_only(self, arguments: ProgressInput) -> bool:
        return True

    async def execute(self, arguments: ProgressInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.progress_tracker import get_progress_tracker

        tracker = get_progress_tracker()
        action = arguments.action.strip().lower()
        if action == "summary":
            summary = tracker.summary()
            if not summary["total_scenes"]:
                return ToolResult(
                    output="No scenes tracked yet. Scenes appear as batch_scene/render_scenes/qa_review run."
                )
            lines = [
                "[progress]",
                f"  scenes: {summary['total_scenes']}  approved: {summary['approved']}  "
                f"failed: {summary['failed']}  in-flight: {summary['in_flight']}",
                f"  completion: {summary['completion']}",
                f"  ETA: {summary['eta_seconds']}s" if summary["eta_seconds"] is not None else "  ETA: unknown (no completed cycle observed yet)",
                f"  stages: {json.dumps(summary['stage_counts'])}",
            ]
            for active in summary["active"][:12]:
                lines.append(f"  -> {active['scene']}: {active['stage']} {active['detail']}")
            return ToolResult(output="\n".join(lines), metadata=summary)
        if action == "scenes":
            scenes = tracker.all_scenes()
            if not scenes:
                return ToolResult(output="No scenes tracked yet.")
            lines = ["[progress scenes]"]
            for scene in scenes:
                lines.append(
                    f"  {scene['scene']:<28} {scene['stage']:<10} {scene['tier']:<6} {scene['detail']}"
                )
            return ToolResult(output="\n".join(lines), metadata={"scenes": scenes})
        if action == "scene":
            if not arguments.scene:
                return ToolResult(output="scene action requires a scene name.", is_error=True)
            status = tracker.scene_status(arguments.scene)
            if status is None:
                return ToolResult(output=f"Unknown scene: {arguments.scene}", is_error=True)
            return ToolResult(output="[progress scene] " + json.dumps(status, indent=2), metadata=status)
        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)


# ======================================================================
# #22 — audio_gen
# ======================================================================

class AudioGenInput(BaseModel):
    text: str = Field(description="Text to convert to speech.")
    output_path: str = Field(description="Output audio file path (e.g. audio/scene_001.mp3).")
    voice: str = Field(default="", description="Voice id (provider-specific, e.g. alloy).")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier.")
    background: bool = Field(
        default=False,
        description="Run in background and return immediately (async audio generation).",
    )


class AudioGenTool(BaseTool):
    name = "audio_gen"
    description = (
        "Generate speech audio (TTS) from text via the configured "
        "OpenAI-compatible /audio/speech endpoint. Can run in the "
        "background while scene code is being written."
    )
    input_model = AudioGenInput

    def is_read_only(self, arguments: AudioGenInput) -> bool:
        return False

    async def execute(self, arguments: AudioGenInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.asset_prefetch import AsyncAudioGenerator

        metadata = context.metadata or {}
        scene_gen = metadata.get("scene_gen_config") or {}
        base_url = str(scene_gen.get("base_url") or "").strip()
        api_key = str(scene_gen.get("api_key") or "").strip()
        if not (base_url and api_key):
            return ToolResult(
                output=(
                    "audio_gen needs a TTS-capable endpoint: configure scene_gen "
                    "base_url + api_key (OPENHARNESS_SCENE_GEN_BASE_URL / "
                    "OPENHARNESS_SCENE_GEN_API_KEY) pointing at an OpenAI-compatible "
                    "provider that serves /audio/speech. No audio was generated."
                ),
                is_error=True,
            )
        generator = AsyncAudioGenerator(base_url=base_url, api_key=api_key)
        output = Path(arguments.output_path)
        if not output.is_absolute():
            output = context.cwd / output
        if arguments.background:
            task = generator.run_in_background(
                arguments.text, output, voice=arguments.voice or None, speed=arguments.speed
            )
            return ToolResult(
                output=f"[audio_gen] background task {task.task_id} started -> {output}",
                metadata={"task_id": task.task_id},
            )
        task = await generator.generate(
            arguments.text, output, voice=arguments.voice or None, speed=arguments.speed
        )
        if task.status == "done":
            return ToolResult(
                output=f"[audio_gen] wrote {output} ({task.as_dict()['elapsed_seconds']}s)",
                metadata=task.as_dict(),
            )
        return ToolResult(output=f"[audio_gen] FAILED: {task.error}", is_error=True, metadata=task.as_dict())


# ======================================================================
# #41-#50 — server_config
# ======================================================================

class ServerConfigInput(BaseModel):
    action: str = Field(default="generate", description="generate | hardware")
    model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct",
        description="Model id the vLLM server should host.",
    )
    context_length: int = Field(default=32768, ge=2048, le=1_048_576)
    speculative_model: str = Field(
        default="",
        description="Draft model for speculative decoding (#1), e.g. meta-llama/Llama-3.2-1B-Instruct.",
    )
    enable_mtp: bool = Field(default=False, description="Document MTP wiring (#5).")
    output_dir: str = Field(default="server", description="Where to write run_vllm_server.sh + compose file.")


class ServerConfigTool(BaseTool):
    name = "server_config"
    description = (
        "Generate a WORKING vLLM server launch config (continuous batching, "
        "paged attention, Flash Attention, quantized KV cache, chunked "
        "prefill, speculative decoding) tuned to the detected hardware, or "
        "report which hardware accelerators (NVLink/RDMA/TPU) this machine "
        "actually has. Items #41-#50 are model-server/hardware features — "
        "this tool is their honest client-side surface."
    )
    input_model = ServerConfigInput

    def is_read_only(self, arguments: ServerConfigInput) -> bool:
        return arguments.action == "hardware"

    async def execute(self, arguments: ServerConfigInput, context: ToolExecutionContext) -> ToolResult:
        if arguments.action == "hardware":
            from openharness.services.server_accel import detect_hardware_acceleration

            report = detect_hardware_acceleration()
            lines = ["[hardware report]"]
            for key in ("gpu_count", "gpu_type", "is_kaggle"):
                lines.append(f"  {key}: {report[key]}")
            for family in ("nvlink", "rdma", "tpu", "zero_copy"):
                entry = report[family]
                lines.append(f"  {family}: {'yes' if entry['available'] else 'no'} — {entry['detail']}")
            lines.append(f"  advisory: {report['advisory']}")
            return ToolResult(output="\n".join(lines), metadata=report)

        from openharness.services.server_accel import build_vllm_server_config, write_server_config

        config = build_vllm_server_config(
            model=arguments.model,
            context_length=arguments.context_length,
            enable_speculative=bool(arguments.speculative_model),
            speculative_model=arguments.speculative_model,
            enable_mtp=arguments.enable_mtp,
        )
        out_dir = Path(arguments.output_dir)
        if not out_dir.is_absolute():
            out_dir = context.cwd / out_dir
        script = write_server_config(config, out_dir)
        lines = [
            f"[server_config] generated {script}",
            "Launch command:",
            "  " + " ".join(config["argv"]),
            "",
            "Where each requested optimization lands:",
        ]
        for explanation in config["explanations"]:
            lines.append(f"  {explanation['item']}: {explanation['flag']}")
            lines.append(f"      {explanation['why']}")
        lines.append("")
        lines.append(config["client_note"])
        return ToolResult(output="\n".join(lines), metadata={"script": str(script)})


# ======================================================================
# #17-#19 — cache_stats
# ======================================================================

class CacheStatsInput(BaseModel):
    action: str = Field(default="status", description="status | clear_render | clear_semantic | clear_qa | prune")
    keep: int = Field(default=256, ge=1, le=10_000, description="Entries to keep for prune.")


class CacheStatsTool(BaseTool):
    name = "cache_stats"
    description = (
        "Report what the caches actually did — render-cache hits/misses, "
        "semantic-cache hits, persistent store stats — and clear them when "
        "you want a clean run."
    )
    input_model = CacheStatsInput

    def is_read_only(self, arguments: CacheStatsInput) -> bool:
        return arguments.action == "status"

    async def execute(self, arguments: CacheStatsInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.persistent_store import get_persistent_store
        from openharness.services.render_cache import RenderCache

        store = get_persistent_store()
        action = arguments.action.strip().lower()
        if action == "status":
            render_cache = RenderCache(store)
            lines = [
                "[cache_stats]",
                f"  persistent store: {json.dumps(store.stats())}",
                f"  render cache: {json.dumps(render_cache.stats())}",
            ]
            return ToolResult(output="\n".join(lines), metadata={"store": store.stats()})
        if action == "clear_render":
            removed = store.clear_namespace("render_cache")
            return ToolResult(output=f"Cleared {removed} render-cache entries.")
        if action == "clear_semantic":
            removed = store.clear_namespace("llm_semantic")
            return ToolResult(output=f"Cleared {removed} semantic-cache entries.")
        if action == "clear_qa":
            removed = store.clear_namespace("qa_failures")
            return ToolResult(output=f"Cleared {removed} QA failure-stat entries.")
        if action == "prune":
            removed = store.prune_namespace("llm_semantic", arguments.keep)
            return ToolResult(output=f"Pruned semantic cache to {arguments.keep} entries (removed {removed}).")
        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)


# ======================================================================
# Speed feature #2 — parallel_prompts (async batch executor)
# ======================================================================

class ParallelPromptsInput(BaseModel):
    prompts: list[str] = Field(
        description=(
            "Independent prompts to run concurrently. Each gets its own "
            "response; results come back in the same order. Prompts that "
            "depend on each other's output do NOT belong here."
        )
    )
    system: str | None = Field(
        default=None,
        description="Optional system prompt applied to every request.",
    )
    max_tokens: int = Field(default=2048, ge=1, le=200_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    concurrency: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="Max requests in flight at once (default: settings.inference.batch_concurrency, 8).",
    )
    retries: int = Field(default=1, ge=0, le=5, description="Retry rounds for failed prompts.")
    labels: list[str] | None = Field(
        default=None,
        description="Optional labels per prompt (same length) used in the report.",
    )


class ParallelPromptsTool(BaseTool):
    name = "parallel_prompts"
    description = (
        "Run N independent LLM prompts CONCURRENTLY through the async batch "
        "executor (default 8 in flight, semaphore-bounded) instead of one at "
        "a time — an 8x throughput win for fan-out work like writing 50 "
        "title cards, naming scenes, or summarizing documents in bulk. "
        "Every response is recorded in the cost dashboard; failed prompts "
        "retry (configurable) and never lose the successful ones. Use this "
        "whenever the prompts don't depend on each other."
    )
    input_model = ParallelPromptsInput

    def is_read_only(self, arguments: ParallelPromptsInput) -> bool:
        return True

    async def execute(self, arguments: ParallelPromptsInput, context: ToolExecutionContext) -> ToolResult:

        from openharness.services.batch_executor import (
            AsyncBatchExecutor,
            ItemFailure,
        )
        from openharness.services.cost_dashboard import get_cost_dashboard

        if not arguments.prompts:
            return ToolResult(output="No prompts provided.", is_error=True)

        # Endpoint resolution — same precedence as the QA reviewer:
        # scene_gen config, then QA settings, then the primary model.
        metadata = context.metadata or {}
        settings = metadata.get("settings")
        scene_gen = metadata.get("scene_gen_config") or {}
        qa = getattr(settings, "qa", None) if settings is not None else None
        base_url = (
            (getattr(qa, "base_url", "") or "")
            or scene_gen.get("base_url")
            or (getattr(settings, "base_url", "") or "")
        ).strip()
        api_key = (
            (getattr(qa, "api_key", "") or "")
            or scene_gen.get("api_key")
            or (getattr(settings, "api_key", "") or "")
        ).strip()
        model = (
            (getattr(qa, "model", "") or "")
            or scene_gen.get("model")
            or (getattr(settings, "model", "") or "")
        ).strip()
        if not (base_url and api_key and model):
            return ToolResult(
                output=(
                    "parallel_prompts needs an LLM endpoint: configure "
                    "settings.scene_gen / settings.qa / the primary model "
                    "(model + api_key + base_url)."
                ),
                is_error=True,
            )
        if not base_url.rstrip("/").endswith(("v1", "api")):
            base_url = base_url.rstrip("/") + "/v1"

        concurrency = arguments.concurrency
        if concurrency is None:
            inference = getattr(settings, "inference", None) if settings is not None else None
            concurrency = int(getattr(inference, "batch_concurrency", 8) or 8)
        executor = AsyncBatchExecutor(concurrency)

        dashboard = get_cost_dashboard()
        labels = arguments.labels or []
        results: list[Any] = []

        # ONE pooled HTTP client for the whole batch (and its retries):
        # per-prompt clients would pay a fresh TCP+TLS handshake each —
        # exactly the overhead the batch executor exists to amortize.
        import httpx

        timeout = float(getattr(settings, "timeout", 120.0) or 120.0) if settings is not None else 120.0
        shared_client = httpx.AsyncClient(timeout=max(timeout, 60.0))

        async def _one(prompt: str, index: int) -> str:
            payload: dict[str, Any] = {
                "model": model,
                "messages": (
                    [{"role": "system", "content": arguments.system}, {"role": "user", "content": prompt}]
                    if arguments.system
                    else [{"role": "user", "content": prompt}]
                ),
                "max_tokens": arguments.max_tokens,
                "temperature": arguments.temperature,
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await shared_client.post(
                f"{base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"no choices in response: {str(data)[:200]}")
            content = str((choices[0].get("message") or {}).get("content") or "")
            usage = data.get("usage") or {}
            if usage:
                dashboard.record(
                    model=model,
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    source="parallel_prompts",
                )
            return content

        import time as _time

        started = _time.monotonic()
        try:
            results = await executor.run_with_retries(
                arguments.prompts,
                _one,
                retries=arguments.retries,
            )
        except Exception as exc:  # noqa: BLE001 - tool must always report
            return ToolResult(output=f"parallel_prompts failed: {exc}", is_error=True)
        finally:
            aclose = getattr(shared_client, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - cleanup must not mask results
                    pass
        elapsed = _time.monotonic() - started

        succeeded = [r for r in results if not isinstance(r, ItemFailure)]
        failed = [r for r in results if isinstance(r, ItemFailure)]
        lines = [
            f"[parallel_prompts] {len(succeeded)}/{len(arguments.prompts)} prompts "
            f"succeeded in {elapsed:.1f}s (concurrency {executor.concurrency}, "
            f"{executor.stats.items_run} calls incl. retries)."
        ]
        for index, result in enumerate(results):
            label = labels[index] if index < len(labels) else f"#{index + 1}"
            if isinstance(result, ItemFailure):
                lines.append(f"  FAIL  {label}: {result.error}")
        report = executor.report()
        report["elapsed_seconds"] = round(elapsed, 2)
        report["succeeded"] = len(succeeded)
        report["failed"] = len(failed)
        return ToolResult(
            output="\n".join(lines),
            metadata={
                "results": [
                    r.error if isinstance(r, ItemFailure) else str(r) for r in results
                ],
                "executor": report,
            },
        )


# ======================================================================
# Speed feature #6 — cost_report (cost dashboard)
# ======================================================================

class CostReportInput(BaseModel):
    action: str = Field(
        default="summary",
        description="summary | models | scenes | recent | clear",
    )
    project: str | None = Field(
        default=None,
        description="Filter/report by project name (uses the per-project aggregate).",
    )
    scenes_total: int | None = Field(
        default=None,
        ge=1,
        description="Total scenes in the current project (enables cost projection).",
    )
    scenes_done: int | None = Field(
        default=None,
        ge=0,
        description="Scenes completed so far (enables cost projection).",
    )
    limit: int = Field(default=15, ge=1, le=100, description="Entries for 'recent'.")


class CostReportTool(BaseTool):
    name = "cost_report"
    description = (
        "Real-time cost dashboard: every LLM call is priced from a "
        "per-model pricing table and aggregated in SQLite (survives "
        "restarts). See total spend, cost by model / by scene / by "
        "project, recent calls, and a projected total when you pass "
        "scenes_total + scenes_done. Prices are estimates and "
        "user-overridable (OPENHARNESS_PRICING_JSON)."
    )
    input_model = CostReportInput

    def is_read_only(self, arguments: CostReportInput) -> bool:
        return arguments.action != "clear"

    async def execute(self, arguments: CostReportInput, context: ToolExecutionContext) -> ToolResult:
        from openharness.services.cost_dashboard import get_cost_dashboard

        dashboard = get_cost_dashboard()
        action = arguments.action.strip().lower()
        if action == "clear":
            dashboard.clear()
            return ToolResult(output="Cost dashboard cleared (memory + persisted aggregates).")

        if action == "recent":
            rows = dashboard.recent(limit=arguments.limit)
            if not rows:
                return ToolResult(output="No calls recorded yet.", metadata={"recent": []})
            lines = ["[cost_report] recent calls (newest first)"]
            for row in rows:
                price_note = "" if row.get("priced") else "  (unpriced model)"
                scene_note = f"  scene={row['scene']}" if row.get("scene") else ""
                lines.append(
                    f"  {row['model']}: {row['input_tokens']} in / {row['output_tokens']} out "
                    f"-> ${row['cost_usd']:.4f}{price_note}{scene_note} [{row.get('source', 'llm')}]"
                )
            return ToolResult(output="\n".join(lines), metadata={"recent": rows})

        summary = dashboard.summary(
            project=arguments.project,
            scenes_total=arguments.scenes_total,
            scenes_done=arguments.scenes_done,
        )
        if action == "models":
            lines = [f"[cost_report] by model — total ${summary['total_cost_usd']:.4f} "
                     f"over {summary['total_calls']} calls"]
            for model, agg in summary["models"].items():
                lines.append(
                    f"  {model}: {agg['calls']} calls, {agg['input_tokens']} in / "
                    f"{agg['output_tokens']} out -> ${agg['cost_usd']:.4f}"
                )
            if summary["unpriced_calls"]:
                lines.append(
                    f"  ({summary['unpriced_calls']} calls used models with no pricing entry — "
                    f"add OPENHARNESS_PRICING_JSON to price them)"
                )
            return ToolResult(output="\n".join(lines), metadata={"models": summary["models"]})

        if action == "scenes":
            scenes = summary["scenes"]
            if not scenes:
                return ToolResult(output="No per-scene costs recorded yet "
                                         "(tools label records with scene=<name>).",
                                  metadata={"scenes": {}})
            lines = [f"[cost_report] by scene — {len(scenes)} scene(s)"]
            for scene, agg in list(scenes.items())[:50]:
                lines.append(f"  {scene}: {agg['calls']} calls -> ${agg['cost_usd']:.4f}")
            return ToolResult(output="\n".join(lines), metadata={"scenes": scenes})

        # summary (default)
        lines = [
            f"[cost_report] ${summary['total_cost_usd']:.4f} over {summary['total_calls']} LLM calls "
            f"({summary['total_input_tokens']} in / {summary['total_output_tokens']} out tokens)",
        ]
        if summary.get("project") is not None:
            lines.append(f"  project {summary['project']}: ${summary['project_cost_usd']:.4f}")
        for model, agg in list(summary["models"].items())[:5]:
            lines.append(
                f"  {model}: {agg['calls']} calls -> ${agg['cost_usd']:.4f}"
            )
        if summary["unpriced_calls"]:
            lines.append(
                f"  note: {summary['unpriced_calls']} calls on unpriced models "
                f"(not included in totals; set OPENHARNESS_PRICING_JSON)"
            )
        if "projected_total_usd" in summary:
            lines.append(
                f"  projection: {summary['scenes_done']}/{summary['scenes_total']} scenes done, "
                f"{summary['scenes_remaining']} remaining -> projected total "
                f"${summary['projected_total_usd']:.4f}"
            )
        return ToolResult(output="\n".join(lines), metadata={"summary": summary})
