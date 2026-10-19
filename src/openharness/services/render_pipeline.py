"""Render pipeline — items #11-#16, #21, #24, #25, #27 (render stage), #40.

A real, queue-based scene renderer:

- ``RenderQueue`` (#11): N async workers (auto-sized to the GPU count)
  render scenes in parallel subprocesses, each pinned to its own GPU
  via ``CUDA_VISIBLE_DEVICES`` on multi-GPU hosts (Kaggle T4 x2 ->
  2 workers).
- ``WorkStealingQueue`` (#24): idle workers steal queued jobs from the
  longest busy worker's queue.
- ``SceneDependencyGraph`` (#25): parses ``depends_on`` from scene code,
  detects cycles, and orders jobs so independent scenes render in
  parallel while dependent scenes wait.
- Resolution tiering + draft-first (#12/#16): ``draft`` (640x360) ->
  ``review`` (720p) -> ``final`` (1080p), with engine-specific quality
  flags.
- Quality-based early exit (#13): a draft that passes its quality gate
  is promoted instead of re-rendered at full res.
- Incremental/scene-level render cache (#14/#17): fingerprints skip
  byte-identical re-renders (see ``render_cache.py``).
- NVENC (#15): a REAL one-second probe encode verifies ``h264_nvenc``
  before it is preferred; otherwise libx264.
- Pipeline overlap (#21): ``submit()`` never blocks, so LLM scene
  generation and rendering proceed concurrently.
- Per-job timeout (#27 render stage) and queue monitoring (#40).
- Kill integration (#26): the pipeline installs itself as a control-bus
  watchee — ``trip()`` terminates live render subprocesses instead of
  letting them run to their own timeouts — and finishes feed the bus,
  so a healthy long render never looks like a hung session to the
  watchdog.
- Wait scoping (``wait_for_scenes``): wait for THIS call's scenes only
  (dependency waves included), never for foreign background batches.

Engine commands are templates (defaults for hyperframes/manim/blender/
ffmpeg) overridable per project via ``render_engines.json`` in the
project root. Renders run as explicit argv lists — never a shell string
— so scene code cannot inject shell syntax.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openharness.services.render_cache import RenderCache, scene_fingerprint

log = logging.getLogger(__name__)

TIERS = ("draft", "review", "final", "beast")
TIER_ORDER = {name: index for index, name in enumerate(TIERS)}
# "beast" is THE one tier (OpenVideoHarness playbook/01-pipeline.md): the film
# rendered ONCE at ship grade — 720p-class, NVENC p5/cq21 when a card answers —
# instead of walking the draft→review→final ladder. It sorts above final so
# tier comparisons treat it as the top of the ladder, but it is meant to be
# used alone: submit(tier="beast") and the {tier} placeholder routes it onto
# every engine's own beast policy (render.mjs --tier beast, blender --profile beast).


# ======================================================================
# Engine command templates
# ======================================================================

DEFAULT_ENGINE_TEMPLATES: dict[str, list[str]] = {
    # hyperframes: browser-based scene renderer.
    "hyperframes": [
        "hyperframes", "render", "{scene}",
        "--output", "{output}",
        "--width", "{width}", "--height", "{height}", "--fps", "{fps}",
    ],
    # manim: -ql 480p / -qm 720p / -qh 1080p quality flags.
    "manim": [
        "manim", "render", "{scene}", "{scene_class}",
        "-q{quality}", "--media_dir", "{media_dir}",
    ],
    # blender: headless background render of a .blend scene graph.
    "blender": [
        "blender", "-b", "{scene}", "-o", "{output_stem}", "-a",
    ],
    # ffmpeg: the scene file IS a filter_complex_script; codec args and
    # resolution come from tier settings.
    "ffmpeg": [
        "ffmpeg", "-y", "-hide_banner", "-nostdin",
        "{input_args}", "-filter_complex_script", "{scene}",
        "{codec_args}", "{output}",
    ],
}

# manim quality flag per tier.
_MANIM_QUALITY = {"draft": "l", "review": "m", "final": "h", "beast": "m"}

#: ffmpeg input args per tier — a black canvas the filter script draws on.
_FFMPEG_INPUT_ARGS = {
    "draft": "-f lavfi -i color=c=black:s=640x360:d=10:r=30",
    "review": "-f lavfi -i color=c=black:s=1280x720:d=10:r=30",
    "final": "-f lavfi -i color=c=black:s=1920x1080:d=10:r=30",
    "beast": "-f lavfi -i color=c=black:s=1280x720:d=10:r=30",
}

_NVENC_ARGS = "-c:v h264_nvenc -preset p4 -tune hq -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p"
_X264_ARGS = "-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p"


def load_engine_templates(cwd: Path) -> dict[str, list[str]]:
    """Load engine command templates, honoring ``render_engines.json``."""
    templates = {name: list(argv) for name, argv in DEFAULT_ENGINE_TEMPLATES.items()}
    override_path = cwd / "render_engines.json"
    if override_path.is_file():
        try:
            raw = json.loads(override_path.read_text(encoding="utf-8"))
            for name, argv in raw.items():
                if isinstance(argv, list) and all(isinstance(tok, str) for tok in argv):
                    templates[name] = argv
        except (OSError, ValueError) as exc:
            log.warning("ignoring invalid %s: %s", override_path, exc)
    return templates


def extract_scene_class(scene_path: Path) -> str:
    """Extract the manim scene class name from scene code (default ``Scene``)."""
    try:
        code = scene_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Scene"
    match = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?:Scene|.*Scene)\s*\)", code)
    return match.group(1) if match else "Scene"


# Placeholder keys whose replacement values are MULTI-FLAG argument
# strings ("-f lavfi -i color=..."). They must be shlex-split into
# separate argv tokens — there is no shell between us and the engine
# binary, so injecting them as one token makes ffmpeg fail on every
# single render.
_MULTI_FLAG_PLACEHOLDERS = frozenset({"input_args", "codec_args"})


def build_render_command(
    *,
    engine: str,
    scene_path: Path,
    output_path: Path,
    tier: str,
    width: int,
    height: int,
    fps: int,
    codec_args: str,
    templates: dict[str, list[str]] | None = None,
    output_dir: Path,
) -> list[str]:
    """Build the explicit argv for one render job (no shell involved).

    Tokens that are EXACTLY a multi-flag placeholder (``{input_args}``,
    ``{codec_args}``) expand into multiple argv entries via
    :func:`shlex.split`; placeholders embedded inside a larger token
    are substituted in place as before.

    ``{tier}`` substitutes the tier name itself (draft/review/final) — the
    hook ``render_engines.json`` uses to route OpenHarness tiers onto
    engine toolchains (e.g. OpenVideoHarness's Cycles profiles turbo/
    draft/final; see docs/VIDEO_HARNESS.md "The RenderQueue onto the
    OpenVideoHarness toolchain").
    """
    templates = templates or DEFAULT_ENGINE_TEMPLATES
    template = templates.get(engine) or templates["hyperframes"]
    replacements = {
        "scene": str(scene_path),
        "output": str(output_path),
        "output_stem": str(output_path.with_suffix("")),
        "width": str(width),
        "height": str(height),
        "fps": str(fps),
        "tier": tier,
        "codec_args": codec_args,
        "quality": _MANIM_QUALITY.get(tier, "m"),
        "scene_class": extract_scene_class(scene_path),
        "media_dir": str(output_dir / "manim_media"),
        "input_args": _FFMPEG_INPUT_ARGS.get(tier, _FFMPEG_INPUT_ARGS["review"]),
    }
    argv: list[str] = []
    for token in template:
        expanded = False
        for key in _MULTI_FLAG_PLACEHOLDERS:
            if token == "{" + key + "}":
                argv.extend(shlex.split(replacements[key]))
                expanded = True
                break
        if expanded:
            continue
        for key, value in replacements.items():
            token = token.replace("{" + key + "}", value)
        argv.append(token)
    return argv


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse ``WxH`` (``1280x720``) into ints with a safe fallback."""
    match = re.fullmatch(r"\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*", resolution or "")
    if not match:
        return 1280, 720
    return int(match.group(1)), int(match.group(2))


# ======================================================================
# #15 — NVENC codec detection (verified with a real probe encode)
# ======================================================================

class NVENCDetector:
    """Detect hardware-accelerated encoding with a REAL probe.

    Runs a genuine 0.2-second ``h264_nvenc`` test encode through the
    local ffmpeg. Only when the probe succeeds (exit 0 AND a non-empty
    output) is NVENC used for renders. Everything else falls back to
    libx264. The result is cached for the process lifetime.
    """

    def __init__(self) -> None:
        self._checked = False
        self._available = False
        self._reason = "not probed yet"

    def available(self, *, force_recheck: bool = False) -> tuple[bool, str]:
        if self._checked and not force_recheck:
            return self._available, self._reason
        self._checked = True
        self._available = False
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._reason = "ffmpeg not on PATH"
            return False, self._reason
        if not shutil.which("nvidia-smi"):
            self._reason = "no NVIDIA GPU (nvidia-smi missing)"
            return False, self._reason
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe_output = Path(tmp) / "probe.mp4"
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=256x144:d=0.2:r=30",
                "-c:v", "h264_nvenc", "-frames:v", "6", str(probe_output),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except (subprocess.TimeoutExpired, OSError) as exc:
                self._reason = f"probe failed: {exc}"
                return False, self._reason
            if result.returncode == 0 and probe_output.is_file() and probe_output.stat().st_size > 0:
                self._available = True
                self._reason = "h264_nvenc probe encode succeeded"
            else:
                self._reason = (
                    "h264_nvenc probe encode failed (driver or SDK mismatch): "
                    + (result.stderr.strip()[:160] or "no stderr")
                )
        return self._available, self._reason

    def codec_args(self, enabled: bool = True) -> str:
        """Return the codec argument string for the current machine."""
        if not enabled:
            return _X264_ARGS
        ok, reason = self.available()
        log.debug("nvenc detector: available=%s (%s)", ok, reason)
        return _NVENC_ARGS if ok else _X264_ARGS


# ======================================================================
# #25 — Scene dependency graph
# ======================================================================

_DEPENDS_RE = re.compile(r"^\s*(?:#|//)\s*depends_on:\s*([A-Za-z0-9_.\-]+)\s*$", re.MULTILINE)


@dataclass
class SceneNode:
    name: str
    path: Path
    depends_on: list[str] = field(default_factory=list)


class SceneDependencyGraph:
    """Parse scene dependencies and compute parallel render waves.

    Scenes declare dependencies with a comment line in their code::

        // depends_on: scene_001
        # depends_on: scene_001, scene_002   (comma-separated supported)

    ``render_waves()`` returns a list of lists: every scene in wave *i*
    depends only on scenes in earlier waves, so each wave renders fully
    in parallel. Cycles raise ``ValueError`` instead of hanging.
    """

    def __init__(self, scene_paths: list[Path]) -> None:
        self.nodes: dict[str, SceneNode] = {}
        for path in scene_paths:
            name = path.stem
            self.nodes[name] = SceneNode(name=name, path=path, depends_on=[])
        for node in self.nodes.values():
            code = self._read(node.path)
            for match in _DEPENDS_RE.finditer(code):
                for dep in match.group(1).split(","):
                    dep = dep.strip()
                    if dep:
                        node.depends_on.append(dep)

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def render_waves(self) -> list[list[str]]:
        """Topologically group scene names into parallel waves."""
        resolved: set[str] = set()
        waves: list[list[str]] = []
        remaining = set(self.nodes)
        # Local dependencies (files missing from the scene list) are
        # ignored — they refer to assets built earlier in the project.
        while remaining:
            ready = sorted(
                name for name in remaining
                if all(
                    dep in resolved or dep not in self.nodes
                    for dep in self.nodes[name].depends_on
                )
            )
            if not ready:
                raise ValueError(
                    "dependency cycle among scenes: "
                    + ", ".join(sorted(remaining))
                )
            waves.append(ready)
            resolved.update(ready)
            remaining.difference_update(ready)
        return waves

    def as_dict(self) -> dict[str, Any]:
        return {
            name: {"path": str(node.path), "depends_on": node.depends_on}
            for name, node in self.nodes.items()
        }


# ======================================================================
# #24 — Work-stealing job queue
# ======================================================================

class WorkStealingQueue:
    """Per-worker deques with steal-from-the-longest semantics.

    ``get(worker_id)`` pops the front of the worker's own deque; when
    empty, it steals the BACK of the currently-longest deque (FIFO for
    the owner, LIFO for the thief — minimizes contention on the hot
    end). Coroutines wait on a shared condition while every deque is
    empty and the queue is open.
    """

    def __init__(self, worker_count: int, *, stealing: bool = True) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        self._deques: list[deque[Any]] = [deque() for _ in range(worker_count)]
        self._stealing = stealing
        self._cond = asyncio.Condition()
        self._closed = False
        self.steals = 0
        # Waker task for put_nowait(): a synchronous enqueue cannot take
        # the asyncio lock to call cond.notify(), so it schedules this
        # task on the running loop instead. Without it, a job submitted
        # while every worker is idle-waiting in cond.wait() NEVER woke
        # the workers — dependency-wave renders (and any submit after
        # the workers went idle) hung as "pending" forever.
        self._waker: asyncio.Task[None] | None = None

    async def put(self, item: Any, *, worker_hint: int | None = None) -> None:
        """Enqueue ``item`` onto the least-loaded worker (or a hinted one)."""
        async with self._cond:
            if worker_hint is not None:
                index = worker_hint % len(self._deques)
            else:
                sizes = [len(d) for d in self._deques]
                index = sizes.index(min(sizes))
            self._deques[index].append(item)
            self._cond.notify()

    def put_nowait(self, item: Any, *, worker_hint: int | None = None) -> None:
        """Synchronous enqueue (callers on the event loop).

        Appends the item and schedules a wake for idle workers — see
        ``_schedule_wake``. Items enqueued before any worker waits are
        simply found by the first ``get`` pass.
        """
        if worker_hint is not None:
            self._deques[worker_hint % len(self._deques)].append(item)
        else:
            sizes = [len(d) for d in self._deques]
            self._deques[sizes.index(min(sizes))].append(item)
        self._schedule_wake()

    def _schedule_wake(self) -> None:
        """Wake idle workers after a synchronous enqueue.

        ``cond.notify()`` may only be called while holding the lock, and
        ``put_nowait`` is synchronous — so schedule an async waker task
        on the running loop instead. Safe when called off-loop (no-op):
        workers' ``get`` has a wait-timeout backstop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop in this thread; the get() backstop covers it
        if self._waker is None or self._waker.done():
            self._waker = loop.create_task(self._wake_waiters())

    async def _wake_waiters(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    def _next_item(self, worker_id: int) -> Any:
        own = self._deques[worker_id % len(self._deques)]
        if own:
            return own.popleft()
        if self._stealing:
            longest = max(self._deques, key=len)
            if longest:
                self.steals += 1
                return longest.pop()  # steal from the back
        return None

    async def get(self, worker_id: int) -> Any:
        """Block until a job is available for this worker (or queue closes).

        The wait is bounded (0.5s) as a belt-and-braces backstop: a
        notification missed for any reason costs at most half a second
        of latency instead of a dead worker.
        """
        while True:
            async with self._cond:
                item = self._next_item(worker_id)
                if item is not None:
                    return item
                if self._closed:
                    return None
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    async def close(self) -> None:
        """Wake every waiting worker; ``get`` then drains and returns None."""
        if self._waker is not None and not self._waker.done():
            try:
                await self._waker
            except Exception:  # noqa: BLE001 - closing must always proceed
                pass
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def depth(self) -> int:
        return sum(len(d) for d in self._deques)

    def snapshot(self) -> list[int]:
        return [len(d) for d in self._deques]


# ======================================================================
# Render jobs and the queue itself
# ======================================================================

@dataclass
class RenderJob:
    """One scene render through the pipeline."""

    scene: Path
    engine: str
    tier: str = "final"
    output_dir: Path | None = None
    status: str = "pending"  # pending|running|done|failed|timeout|skipped
    job_id: str = ""
    fingerprint: str = ""
    output_path: Path | None = None
    error: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0.0
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    worker: int | None = None
    cached: bool = False
    aborted: bool = False
    command: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    quality_score: float | None = None
    promoted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scene": str(self.scene),
            "engine": self.engine,
            "tier": self.tier,
            "status": self.status,
            "output_path": str(self.output_path) if self.output_path else None,
            "cached": self.cached,
            "aborted": self.aborted,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error[:300],
            "worker": self.worker,
            "quality_score": self.quality_score,
            "promoted": self.promoted,
            "command": self.command[:12] + (["..."] if len(self.command) > 12 else []),
        }


def _default_output_path(scene: Path, output_dir: Path, tier: str) -> Path:
    return output_dir / f"{scene.stem}.{tier}.mp4"


class RenderPipeline:
    """Parallel scene renderer with caching, tiering and monitoring.

    One instance per process. ``get_render_pipeline()`` returns the
    singleton used by tools and the batch_scene auto-enqueue hook.
    """

    def __init__(
        self,
        *,
        workers: int,
        cwd: Path,
        resolution_tiers: dict[str, str] | None = None,
        nvenc_enabled: bool = True,
        cache_enabled: bool = True,
        render_timeout_seconds: int = 1800,
        stealing: bool = True,
        audit_hook: Any = None,
        progress_hook: Any = None,
    ) -> None:
        self.cwd = cwd
        self.workers = max(1, workers)
        self.resolution_tiers = dict(resolution_tiers or {})
        self.nvenc_enabled = nvenc_enabled
        self.cache_enabled = cache_enabled
        self.render_timeout_seconds = render_timeout_seconds
        self._queue = WorkStealingQueue(self.workers, stealing=stealing)
        self.cache = RenderCache() if cache_enabled else None
        self.nvenc = NVENCDetector()
        self._templates = load_engine_templates(cwd)
        # mtime of render_engines.json at load time — the live re-bind
        # trigger in get_render_pipeline() (see G4).
        self._templates_stamp = _engine_templates_stamp(cwd)
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._jobs: dict[str, RenderJob] = {}
        self._job_seq = 0
        self._started = False
        self._closing = False
        # Scenes promised by an in-flight dependency-wave chain but not
        # yet submitted (submit_batch defers later waves behind earlier
        # ones). ``wait_all`` must wait for these too — otherwise it can
        # return in the gap between "wave N finished" and "wave N+1
        # submitted" (both the chain and wait_all poll on ~0.2s cadences,
        # so the race window is real and hit often).
        self._pending_submissions = 0
        # worker_id -> live info for monitoring (#40)
        self._worker_state: dict[int, dict[str, Any]] = {
            i: {"busy": False, "current_job": None, "completed": 0} for i in range(self.workers)
        }
        self.audit_hook = audit_hook
        self.progress_hook = progress_hook
        self.total_render_seconds = 0.0
        self.completed_renders = 0
        # Live render subprocesses by job id — the abort surface for the
        # control-bus watchee (#26). Registered on spawn, removed on exit.
        self._live_processes: dict[str, asyncio.subprocess.Process] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Best-effort: tell the control bus how to abort our subprocesses.
        self._install_kill_watchee()

    def _install_kill_watchee(self) -> None:
        """Register ``abort_all`` with the shared control bus (best-effort).

        The active bus is the process-wide singleton (the engine installs
        its configured bus there via ``set_control_bus``), so re-registering
        here after any bus swap is safe — registration is idempotent.
        """
        try:
            from openharness.services.control_bus import get_control_bus

            get_control_bus().install_watchee("render_pipeline", self.abort_all)
        except Exception:  # noqa: BLE001 - control integration is optional
            log.debug("render pipeline: control-bus watchee not installed", exc_info=True)

    async def abort_all(self, reason: str = "killed") -> int:
        """Terminate every live render subprocess (#26 watchee).

        Jobs are marked ``failed``/``aborted`` FIRST (so waiters and
        summaries see the abort instantly — racing the worker's own
        SIGTERM handling, which would otherwise record a generic
        exit-code failure first), then the OS processes are terminated
        (SIGTERM, 5s grace, SIGKILL). Returns the number of jobs aborted.
        """
        processes = list(self._live_processes.items())
        aborted = 0
        # 1. Mark first: status is terminal from this instant, and the
        #    ``aborted`` flag stops the worker from overwriting it with
        #    "done" if the engine exits 0 on SIGTERM.
        for job_id, _process in processes:
            job = self._jobs.get(job_id)
            if job is not None and job.status in {"pending", "running"}:
                job.status = "failed"
                job.error = f"aborted: {reason}"
                job.aborted = True
                job.finished_at = time.monotonic()
                self._finish(job)
                aborted += 1
            self._live_processes.pop(job_id, None)
        # 2. Terminate the OS processes.
        for _job_id, process in processes:
            try:
                process.terminate()
            except ProcessLookupError:
                pass  # already dead
            except Exception:  # noqa: BLE001 - abort must keep going
                pass
        if processes:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(p.wait() for _jid, p in processes if p.returncode is None),
                        return_exceptions=True,
                    ),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                for _job_id, process in processes:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    except Exception:  # noqa: BLE001
                        pass
        if aborted:
            log.warning("render pipeline aborted %d live job(s): %s", aborted, reason)
        return aborted

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker coroutines (idempotent)."""
        if self._started:
            return
        self._started = True
        for worker_id in range(self.workers):
            self._worker_tasks.append(
                asyncio.create_task(self._worker_loop(worker_id), name=f"render-worker-{worker_id}")
            )
        # Watchdog heartbeat (#36): running renders ARE observable
        # progress. Without this, one long Blender/ManimGX job with no
        # tool activity in between tripped the 900s watchdog and killed
        # healthy sessions mid-render.
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="render-heartbeat"
        )

    async def _heartbeat_loop(self) -> None:
        """Feed the control bus every 60s while renders are in flight."""
        try:
            from openharness.services.control_bus import get_control_bus
        except Exception:  # noqa: BLE001 - optional integration
            return
        while True:
            await asyncio.sleep(60.0)
            if self._has_inflight_jobs():
                try:
                    get_control_bus().feed("render-heartbeat")
                except Exception:  # noqa: BLE001 - never crash the heart
                    pass

    def _has_inflight_jobs(self) -> bool:
        return (
            self._pending_submissions > 0
            or any(j.status in {"pending", "running"} for j in self._jobs.values())
        )

    async def shutdown(self, *, drain: bool = True) -> None:
        """Stop accepting jobs; optionally drain queued ones first (#32)."""
        if self._closing:
            return
        self._closing = True
        if drain and self._queue.depth:
            deadline = time.monotonic() + self.render_timeout_seconds
            while self._queue.depth and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
        await self._queue.close()
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
        for task in self._worker_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._worker_tasks = []
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        self._started = False
        self._closing = False

    # ------------------------------------------------------------------
    # Submission (#11, #21 — non-blocking)
    # ------------------------------------------------------------------

    def submit(
        self,
        scene: Path,
        *,
        engine: str = "hyperframes",
        tier: str = "final",
        output_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> RenderJob:
        """Enqueue a render and return immediately (pipeline overlap #21)."""
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        self.start()
        self._job_seq += 1
        out_dir = Path(output_dir) if output_dir else self.cwd / "renders"
        job = RenderJob(
            job_id=f"r{self._job_seq:05d}",
            scene=Path(scene),
            engine=engine,
            tier=tier,
            output_dir=out_dir,
            output_path=_default_output_path(Path(scene), out_dir, tier),
            timeout_seconds=float(timeout_seconds or self.render_timeout_seconds),
        )
        self._jobs[job.job_id] = job
        self._prune_jobs()
        self._queue.put_nowait(job)
        self._audit("render_enqueued", job)
        self._progress(job.scene.name, f"queued:{tier}")
        return job

    def submit_batch(
        self,
        scenes: list[Path],
        *,
        engine: str = "hyperframes",
        tier: str = "final",
        output_dir: Path | None = None,
        respect_dependencies: bool = True,
    ) -> list[RenderJob]:
        """Submit many scenes honoring the dependency graph (#25).

        Dependent scenes are only enqueued after their dependencies
        finish, so waves of independent scenes render in parallel while
        preserving order where it matters.
        """
        if not respect_dependencies:
            return [self.submit(s, engine=engine, tier=tier, output_dir=output_dir) for s in scenes]

        graph = SceneDependencyGraph(scenes)
        jobs_by_scene: dict[str, RenderJob] = {}
        pending_waves = graph.render_waves()

        def submit_wave() -> list[RenderJob]:
            wave = pending_waves.pop(0) if pending_waves else []
            submitted: list[RenderJob] = []
            for name in wave:
                job = self.submit(
                    graph.nodes[name].path, engine=engine, tier=tier, output_dir=output_dir
                )
                jobs_by_scene[name] = job
                submitted.append(job)
            return submitted

        first_jobs = submit_wave()
        # Chain later waves onto the completion of earlier ones: each
        # next wave is submitted only when the previous one is finished.
        later_count = sum(len(wave) for wave in pending_waves)
        self._pending_submissions += later_count

        async def _chain() -> None:
            try:
                while pending_waves:
                    previous = list(jobs_by_scene.values())
                    # Bounded wait: a dead worker (or a stuck "pending"
                    # job after a shutdown) must not turn this task into
                    # an infinite poller.
                    stall_deadline = time.monotonic() + (
                        max(
                            (j.timeout_seconds or 1800.0 for j in previous),
                            default=1800.0,
                        )
                        + 60.0
                    )
                    while any(j.status in {"pending", "running"} for j in previous):
                        if time.monotonic() > stall_deadline:
                            log.warning(
                                "render wave chain: previous wave stalled; "
                                "submitting next wave anyway"
                            )
                            break
                        await asyncio.sleep(0.2)
                    wave = pending_waves.pop(0)
                    for name in wave:
                        job = self.submit(
                            graph.nodes[name].path, engine=engine, tier=tier, output_dir=output_dir
                        )
                        jobs_by_scene[name] = job
                        self._pending_submissions = max(0, self._pending_submissions - 1)
            except Exception:
                # The chain task is fire-and-forget; an unhandled error
                # would silently strand every later wave. Release the
                # un-submitted remainder so wait_all is not stuck on a
                # promise that will never be kept.
                remaining_waves = sum(len(wave) for wave in pending_waves)
                pending_waves.clear()
                self._pending_submissions = max(0, self._pending_submissions - remaining_waves)
                log.exception("render dependency-wave chaining failed")

        if pending_waves:
            asyncio.create_task(_chain())
        return first_jobs

    async def wait(self, job: RenderJob, *, timeout: float | None = None) -> RenderJob:
        """Wait for one job to reach a terminal state."""
        deadline = time.monotonic() + (timeout or job.timeout_seconds or 3600)
        while job.status in {"pending", "running"}:
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job.job_id} did not finish in time")
            await asyncio.sleep(0.1)
        return job

    async def wait_all(self, *, timeout: float | None = None) -> list[RenderJob]:
        """Wait for every known job to finish — including waves that a
        dependency chain has promised but not submitted yet."""
        expected = len(self._jobs) + self._pending_submissions
        deadline = time.monotonic() + (
            timeout or self.render_timeout_seconds * max(1, expected)
        )
        while (
            any(j.status in {"pending", "running"} for j in self._jobs.values())
            or self._pending_submissions > 0
        ):
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(0.2)
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Scoped waiting (#25 — dependency waves, G1 fix)
    # ------------------------------------------------------------------

    def jobs_for_scenes(
        self,
        scenes: list[Path],
        *,
        tier: str | None = None,
        since: float | None = None,
    ) -> dict[str, RenderJob]:
        """Latest job per scene — ALL dependency waves included.

        ``since`` (a ``time.monotonic()`` value) restricts the match to
        jobs enqueued at/after that instant, so an older background
        batch rendering the same scenes is never confused with this
        call's jobs. Scenes without a matching job are simply absent
        from the result.
        """
        wanted: dict[Path, str] = {}
        for scene in scenes:
            try:
                resolved = Path(scene).resolve()
            except OSError:
                resolved = Path(scene)
            wanted[resolved] = Path(scene).stem
        result: dict[str, RenderJob] = {}
        for job in self._jobs.values():
            if tier is not None and job.tier != tier:
                continue
            if since is not None and job.enqueued_at < since:
                continue
            try:
                resolved = Path(job.scene).resolve()
            except OSError:
                resolved = Path(job.scene)
            stem = wanted.get(resolved)
            if stem is None and not job.scene.exists() and job.scene.stem in set(wanted.values()):
                # moved/removed file: basename fallback (see _jobs_for_scene)
                stem = job.scene.stem
            if stem is None:
                continue
            current = result.get(stem)
            if current is None or job.enqueued_at >= current.enqueued_at:
                result[stem] = job
        return result

    async def wait_for_scenes(
        self,
        scenes: list[Path],
        *,
        tier: str | None = None,
        since: float | None = None,
        timeout: float | None = None,
    ) -> dict[str, RenderJob]:
        """Wait until every scene has a TERMINAL job — waves included.

        Unlike ``wait_all`` this is scoped to THIS call's scenes: foreign
        background batches submitted earlier never delay the return. The
        loop also exits when the machine is fully drained but some scene
        still has no job (dependency chain aborted) so a dead chain cannot
        hold the caller until the timeout — the summary then reports the
        missing scenes as never-submitted.
        """
        deadline = time.monotonic() + (
            timeout or self.render_timeout_seconds * max(1, len(scenes))
        )
        drained_polls = 0
        while time.monotonic() <= deadline:
            by_scene = self.jobs_for_scenes(scenes, tier=tier, since=since)
            terminal = {
                stem: job
                for stem, job in by_scene.items()
                if job.status not in {"pending", "running"}
            }
            if len(terminal) == len(scenes):
                return terminal
            # Drained = nothing pending/running anywhere, no promised
            # submissions, and two consecutive polls agreed — any scene
            # still without a job will NEVER get one (chain aborted).
            if not self._has_inflight_jobs():
                drained_polls += 1
                if drained_polls >= 2:
                    return terminal
            else:
                drained_polls = 0
            await asyncio.sleep(0.2)
        return self.jobs_for_scenes(scenes, tier=tier, since=since)

    # ------------------------------------------------------------------
    # Draft-first + early exit (#12, #13, #16)
    # ------------------------------------------------------------------

    def render_draft(self, scene: Path, *, engine: str, output_dir: Path | None = None) -> RenderJob:
        """Render a cheap draft for QA review (#12/#16)."""
        return self.submit(scene, engine=engine, tier="draft", output_dir=output_dir)

    def _jobs_for_scene(self, scene: Path, *, tier: str | None = None) -> list[RenderJob]:
        """Find jobs for a scene, matching the resolved path first.

        The basename fallback applies ONLY when the stored job path no
        longer exists on disk (file moved/removed since the job ran) —
        same-basename scenes in sibling directories must not collide.
        """
        try:
            resolved = Path(scene).resolve()
        except OSError:
            resolved = Path(scene)
        matches = []
        for job in self._jobs.values():
            if tier is not None and job.tier != tier:
                continue
            try:
                if Path(job.scene).resolve() == resolved:
                    matches.append(job)
                    continue
            except OSError:
                pass
            if not job.scene.exists() and job.scene.name == scene.name:
                matches.append(job)
        return matches

    def record_quality(self, scene: Path, score: float, *, tier: str = "draft") -> RenderJob | None:
        """Attach a QA score to the most recent job of this scene/tier."""
        candidates = self._jobs_for_scene(scene, tier=tier)
        if not candidates:
            return None
        job = max(candidates, key=lambda j: j.enqueued_at)
        job.quality_score = float(score)
        if self.cache is not None and job.fingerprint:
            self.cache.record_quality_score(job.fingerprint, float(score))
        return job

    def promote(
        self,
        scene: Path,
        *,
        engine: str,
        early_exit_threshold: float,
        output_dir: Path | None = None,
    ) -> RenderJob:
        """Promote a scene after QA: early-exit or full-quality render.

        - Draft quality >= threshold AND early exit allowed (#13):
          the draft IS the final (marked promoted, no re-render).
        - Otherwise: enqueue the full-resolution render (#12).
        """
        drafts = [
            j for j in self._jobs_for_scene(scene, tier="draft") if j.status == "done"
        ]
        draft = max(drafts, key=lambda j: j.enqueued_at) if drafts else None
        if draft is not None and draft.quality_score is not None and draft.quality_score >= early_exit_threshold:
            draft.promoted = True
            if self.cache is not None and draft.fingerprint:
                self.cache.promote(draft.fingerprint, new_tier="final")
            self._audit("render_early_exit", draft)
            self._progress(scene.name, "promoted:early-exit")
            return draft
        return self.submit(scene, engine=engine, tier="final", output_dir=output_dir)

    # ------------------------------------------------------------------
    # Monitoring (#40)
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        jobs = list(self._jobs.values())
        return {
            "workers": {
                str(worker_id): dict(state)
                for worker_id, state in self._worker_state.items()
            },
            "queue_depth": self._queue.depth,
            "queue_per_worker": self._queue.snapshot(),
            "work_steals": self._queue.steals,
            "jobs": {
                "total": len(jobs),
                "pending": sum(1 for j in jobs if j.status == "pending"),
                "running": sum(1 for j in jobs if j.status == "running"),
                "done": sum(1 for j in jobs if j.status == "done"),
                "skipped_cached": sum(1 for j in jobs if j.status == "skipped"),
                "failed": sum(1 for j in jobs if j.status in {"failed", "timeout"}),
            },
            "avg_render_seconds": (
                round(self.total_render_seconds / self.completed_renders, 2)
                if self.completed_renders
                else None
            ),
            "codec": self._active_codec_name(),
        }

    def _active_codec_name(self) -> str:
        """Codec short name for status displays (never raises)."""
        parts = self.nvenc.codec_args(self.nvenc_enabled).split()
        # Args look like "-c:v h264_nvenc -preset ..." — the codec name
        # follows -c:v. A defensive scan survives template edits that
        # change arg order; the previous ``.split()[1]`` raised IndexError
        # for any single-token codec string.
        for i, part in enumerate(parts):
            if part in ("-c:v", "-c") and i + 1 < len(parts):
                return parts[i + 1]
        return parts[0] if parts else "unknown"

    def recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return [j.as_dict() for j in list(self._jobs.values())[-limit:]]

    def jobs_snapshot(self) -> list[RenderJob]:
        """Public read-only view of every job ever submitted."""
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Worker internals
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job = await self._queue.get(worker_id)
            if job is None:
                return  # queue closed
            state = self._worker_state[worker_id]
            state["busy"] = True
            state["current_job"] = job.job_id
            try:
                await self._run_job(job, worker_id)
            except asyncio.CancelledError:
                # Worker shutdown: leave the job as-is (the pipeline is
                # closing) but still reset the worker state.
                state["busy"] = False
                state["current_job"] = None
                raise
            except BaseException as exc:  # noqa: BLE001 - a worker must survive any job
                # A crashing job must NEVER kill the worker task: a dead
                # worker silently shrinks the pool, the job stays
                # "running" forever and wait()/wait_all() hang for their
                # full multi-hour deadlines with no surfaced error.
                job.status = "failed"
                job.error = f"render worker crashed: {type(exc).__name__}: {exc}"
                job.finished_at = time.monotonic()
                try:
                    self._finish(job)
                except Exception:  # noqa: BLE001 - never mask the loop
                    log.exception("failed to finalize crashed job %s", job.job_id)
                log.exception("render worker %d: job %s crashed", worker_id, job.job_id)
            finally:
                state["busy"] = False
                state["current_job"] = None
                state["completed"] += 1

    async def _run_job(self, job: RenderJob, worker_id: int) -> None:
        job.status = "running"
        job.started_at = time.monotonic()
        job.worker = worker_id
        self._progress(job.scene.name, f"rendering:{job.tier}")

        # Render cache (#14/#17): skip byte-identical inputs.
        fingerprint_value = scene_fingerprint(
            job.scene,
            engine=job.engine,
            engine_args=self._engine_args_for(job),
            tier=job.tier,
            codec=self.nvenc.codec_args(self.nvenc_enabled),
            cwd=self.cwd,
        )
        job.fingerprint = fingerprint_value or ""
        if self.cache is not None and fingerprint_value:
            record = self.cache.lookup(fingerprint_value)
            if record is not None:
                job.status = "skipped"
                job.cached = True
                job.output_path = Path(record["output_path"])
                job.finished_at = time.monotonic()
                job.duration_seconds = 0.0
                self._audit("render_cache_hit", job)
                self._progress(job.scene.name, f"rendered:{job.tier}:cache-hit")
                return

        assert job.output_path is not None and job.output_dir is not None
        job.output_dir.mkdir(parents=True, exist_ok=True)
        width, height = parse_resolution(self.resolution_tiers.get(job.tier, "1280x720"))
        job.command = build_render_command(
            engine=job.engine,
            scene_path=job.scene,
            output_path=job.output_path,
            tier=job.tier,
            width=width,
            height=height,
            fps=30,
            codec_args=self.nvenc.codec_args(self.nvenc_enabled),
            templates=self._templates,
            output_dir=job.output_dir,
        )

        env = _worker_env(worker_id, self.workers)
        try:
            process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=str(self.cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            job.status = "failed"
            job.error = (
                f"engine binary not found: {job.command[0]}. Install it or override the "
                f"command template via render_engines.json ({exc})"
            )
            self._finish(job)
            return
        self._live_processes[job.job_id] = process
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=job.timeout_seconds)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            job.status = "timeout"
            job.error = f"render exceeded {job.timeout_seconds:.0f}s and was killed"
            job.exit_code = process.returncode
            self._finish(job)
            return
        finally:
            self._live_processes.pop(job.job_id, None)
        if job.aborted:
            # abort_all already recorded the outcome; never overwrite
            # the abort verdict (an engine exiting 0 on SIGTERM must not
            # resurrect itself as "done").
            return

        job.finished_at = time.monotonic()
        job.duration_seconds = job.finished_at - (job.started_at or job.finished_at)
        job.exit_code = process.returncode
        if process.returncode == 0 and job.output_path.is_file() and job.output_path.stat().st_size > 0:
            job.status = "done"
            self.completed_renders += 1
            self.total_render_seconds += job.duration_seconds
            if self.cache is not None and job.fingerprint:
                self.cache.store(
                    job.fingerprint,
                    output_path=str(job.output_path),
                    scene=job.scene.name,
                    engine=job.engine,
                    tier=job.tier,
                    duration_seconds=job.duration_seconds,
                )
            self._progress(job.scene.name, f"rendered:{job.tier}")
        else:
            job.status = "failed"
            tail = (stderr or b"").decode("utf-8", errors="replace").strip()[-400:]
            job.error = f"exit={process.returncode}: {tail or 'no stderr'}"
            self._progress(job.scene.name, f"render-failed:{job.tier}")
        self._finish(job)

    def _prune_jobs(self, *, keep: int = 1000) -> None:
        """Cap the job registry so long-running pipelines don't grow it
        without bound. Terminal jobs beyond the newest ``keep`` are
        dropped; pending/running jobs are always retained."""
        if len(self._jobs) <= keep:
            return
        terminal = sorted(
            (
                (job.enqueued_at, job_id)
                for job_id, job in self._jobs.items()
                if job.status not in {"pending", "running"}
            ),
            key=lambda item: item[0],
        )
        excess = len(self._jobs) - keep
        for _, job_id in terminal[:excess]:
            self._jobs.pop(job_id, None)

    def _engine_args_for(self, job: RenderJob) -> list[str]:
        """Fingerprint inputs that fully determine the rendered output.

        The previous version hashed only the first two template tokens +
        tier, so changing ``resolution_tiers`` (draft 640x360 ->
        960x540) or editing the template beyond its first two tokens
        produced the SAME fingerprint — the cache then served the stale
        wrong-resolution render as a hit. The resolved width/height/fps
        and the FULL effective template are included now.
        """
        template = self._templates.get(job.engine) or DEFAULT_ENGINE_TEMPLATES["hyperframes"]
        width, height = parse_resolution(self.resolution_tiers.get(job.tier, "1280x720"))
        return [*template, job.tier, f"{width}x{height}x30"]

    def _finish(self, job: RenderJob) -> None:
        if job.duration_seconds == 0.0 and job.started_at:
            job.duration_seconds = time.monotonic() - job.started_at
        if job.finished_at is None:
            job.finished_at = time.monotonic()
        self._audit(f"render_{job.status}", job)
        # A finished job is observable progress for the watchdog (#36):
        # without this feed, back-to-back long renders with no tool call
        # in between tripped the no-progress kill switch mid-session.
        try:
            from openharness.services.control_bus import get_control_bus

            get_control_bus().feed(f"render:{job.status}")
        except Exception:  # noqa: BLE001 - progress feed is best-effort
            pass

    def _audit(self, event: str, job: RenderJob) -> None:
        if self.audit_hook is not None:
            try:
                self.audit_hook(event, job.as_dict())
            except Exception:
                pass

    def _progress(self, scene: str, state: str) -> None:
        if self.progress_hook is not None:
            try:
                self.progress_hook(scene, state)
            except Exception:
                pass


def _visible_gpu_count() -> int:
    """Count of usable GPUs (0 on CPU-only hosts), re-probed at most
    once per minute.

    The old process-lifetime cache meant a GPU attached mid-session
    (or a Kaggle accelerator toggle) stayed invisible until restart.
    A 60s TTL bounds the nvidia-smi cost on the submit hot path while
    still picking up hardware changes within a minute.
    """
    global _GPU_COUNT_CACHE
    now = time.monotonic()
    if _GPU_COUNT_CACHE is not None and (now - _GPU_COUNT_CACHE[1]) < 60.0:
        return _GPU_COUNT_CACHE[0]
    try:
        from openharness.services.kaggle_env import detect_kaggle_resources

        count = int(detect_kaggle_resources().get("gpu_count", 0) or 0)
    except Exception:
        count = 0
    _GPU_COUNT_CACHE = (max(0, count), now)
    return _GPU_COUNT_CACHE[0]


def _worker_env(worker_id: int, worker_count: int) -> dict[str, str]:
    """Environment for a worker subprocess, pinning it to one GPU.

    On multi-GPU hosts each worker is pinned to one GPU via
    CUDA_VISIBLE_DEVICES, cycling through the available GPUs when there
    are more workers than GPUs (previously worker N was pinned to device
    N — workers beyond the GPU count got non-existent device ids and
    silently fell back to CPU rendering). On CPU-only hosts the variable
    is not set at all, so CUDA auto-detection behaves normally.
    """
    import os

    env = dict(os.environ)
    if worker_count > 1:
        gpu_count = _visible_gpu_count()
        if gpu_count > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(worker_id % gpu_count)
    return env


# ======================================================================
# Process-wide singleton + settings wiring
# ======================================================================

_PIPELINE: RenderPipeline | None = None
_GPU_COUNT_CACHE: tuple[int, float] | None = None


def recommended_worker_count(*, gpu_count: int | None = None) -> int:
    """Workers = GPU count when GPUs exist (min 1, cap 4)."""
    if gpu_count is None:
        from openharness.services.kaggle_env import detect_kaggle_resources

        try:
            gpu_count = int(detect_kaggle_resources().get("gpu_count", 0) or 0)
        except Exception:
            gpu_count = 0
    return max(1, min(4, gpu_count if gpu_count > 0 else 1))


def get_render_pipeline(cwd: Path | None = None, settings: Any = None) -> RenderPipeline:
    """Return (creating if needed) the shared render pipeline.

    Settings are only consumed on first creation; the singleton then
    persists for the process so tools and auto-enqueue hooks share one
    queue and one cache. Engine templates are the one thing re-bound
    live: editing ``render_engines.json`` takes effect on the next tool
    call WITHOUT a restart (a mid-session template edit used to be
    silently ignored until the process died).
    """
    global _PIPELINE
    if _PIPELINE is None:
        rendering = getattr(settings, "rendering", None) if settings is not None else None
        workers_setting = getattr(rendering, "render_workers", None) if rendering else None
        workers = (
            int(workers_setting)
            if workers_setting
            else recommended_worker_count()
        )
        _PIPELINE = RenderPipeline(
            workers=workers,
            cwd=Path(cwd) if cwd else Path.cwd(),
            resolution_tiers=getattr(rendering, "resolution_tiers", None) or None,
            nvenc_enabled=bool(getattr(rendering, "nvenc", True)),
            cache_enabled=bool(getattr(rendering, "render_cache", True)),
            render_timeout_seconds=int(getattr(rendering, "render_timeout_seconds", 1800) or 1800),
            stealing=bool(getattr(rendering, "work_stealing", True)),
        )
    else:
        # Re-bind engine templates when render_engines.json changed on disk
        # (same-cwd only — a different cwd means a different project).
        effective_cwd = Path(cwd) if cwd else _PIPELINE.cwd
        if effective_cwd == _PIPELINE.cwd:
            try:
                stamp = _engine_templates_stamp(_PIPELINE.cwd)
                if stamp is not None and stamp != _PIPELINE._templates_stamp:
                    _PIPELINE._templates = load_engine_templates(_PIPELINE.cwd)
                    _PIPELINE._templates_stamp = stamp
                    log.info("render_engines.json changed on disk; templates re-bound")
            except Exception:  # noqa: BLE001 - never break tool calls
                log.debug("engine template re-bind check failed", exc_info=True)
    return _PIPELINE


def _engine_templates_stamp(cwd: Path) -> float | None:
    """mtime of the project's render_engines.json (None when absent)."""
    try:
        path = cwd / "render_engines.json"
        return path.stat().st_mtime if path.is_file() else None
    except OSError:
        return None


def reset_render_pipeline() -> None:
    """Test helper: drop the singleton."""
    global _PIPELINE
    _PIPELINE = None
