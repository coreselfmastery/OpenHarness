"""Render pipeline overlap — render scenes while code is still being generated.

Item #7: Currently, rendering starts only after all scene code is written.
This module provides a render queue that starts rendering scene 1 as
soon as its code is ready, while the LLM is still writing scene 2.

The render queue runs as a background asyncio task. The main agent loop
enqueues scenes as they become ready; the queue processes them
concurrently (up to ``max_concurrent_renders``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Render states
# ---------------------------------------------------------------------------

class RenderState(str, Enum):
    QUEUED = "queued"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenderJob:
    """A single scene render job."""

    scene_id: str
    scene_file: str  # path to the scene code file
    output_path: str  # where to write the rendered video
    state: RenderState = RenderState.QUEUED
    enqueued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    render_time_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_file": self.scene_file,
            "output_path": self.output_path,
            "state": self.state.value,
            "enqueued_at": self.enqueued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "render_time_seconds": self.render_time_seconds,
            "metadata": self.metadata,
        }


# Type for the render function: takes a RenderJob, returns the output path.
RenderFunction = Callable[[RenderJob], Awaitable[str]]


# ---------------------------------------------------------------------------
# Render queue
# ---------------------------------------------------------------------------

class RenderQueue:
    """Async render queue that overlaps rendering with code generation.

    Usage:
        queue = RenderQueue(render_fn=my_render_function, max_concurrent=4)
        await queue.start()

        # Enqueue scenes as they become ready (while the LLM is still
        # generating the next scene's code):
        job = RenderJob(scene_id="scene_001", scene_file="scenes/scene_001.js", output_path="out/scene_001.mp4")
        await queue.enqueue(job)

        # Later, wait for all renders to complete:
        results = await queue.wait_all()
    """

    def __init__(
        self,
        *,
        render_fn: RenderFunction,
        max_concurrent: int = 4,
    ) -> None:
        self._render_fn = render_fn
        self._max_concurrent = max_concurrent
        self._queue: asyncio.Queue[RenderJob] = asyncio.Queue()
        self._jobs: dict[str, RenderJob] = {}  # scene_id → job
        self._worker_task: asyncio.Task | None = None
        self._stopping = False
        self._completed_count = 0
        self._failed_count = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background render workers."""
        if self._worker_task is not None:
            return  # already running
        self._stopping = False
        self._worker_task = asyncio.create_task(self._run_workers())

    async def stop(self) -> None:
        """Signal workers to stop after finishing current jobs."""
        self._stopping = True
        # Enqueue sentinel values to wake up idle workers
        for _ in range(self._max_concurrent):
            await self._queue.put(None)  # type: ignore[arg-type]
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def enqueue(self, job: RenderJob) -> None:
        """Add a render job to the queue."""
        async with self._lock:
            self._jobs[job.scene_id] = job
        await self._queue.put(job)

    async def get_job(self, scene_id: str) -> RenderJob | None:
        """Return the job for a scene_id, or None if not enqueued."""
        async with self._lock:
            return self._jobs.get(scene_id)

    async def wait_for_scene(self, scene_id: str, *, timeout: float | None = None) -> RenderJob:
        """Wait for a specific scene to finish rendering."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            job = await self.get_job(scene_id)
            if job is None:
                raise KeyError(f"Scene {scene_id} not in render queue")
            if job.state in (RenderState.COMPLETED, RenderState.FAILED, RenderState.CANCELLED):
                return job
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Scene {scene_id} did not complete within {timeout}s")
            await asyncio.sleep(0.05)

    async def wait_all(self, *, timeout: float | None = None) -> dict[str, RenderJob]:
        """Wait for all enqueued jobs to reach a terminal state."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            async with self._lock:
                pending = [
                    j for j in self._jobs.values()
                    if j.state in (RenderState.QUEUED, RenderState.RENDERING)
                ]
            if not pending:
                return dict(self._jobs)
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"{len(pending)} scenes did not complete within {timeout}s")
            await asyncio.sleep(0.1)

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self._jobs),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "pending": sum(
                1 for j in self._jobs.values()
                if j.state in (RenderState.QUEUED, RenderState.RENDERING)
            ),
            "max_concurrent": self._max_concurrent,
        }

    async def _run_workers(self) -> None:
        """Run N worker coroutines that pull from the queue."""
        workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self._max_concurrent)
        ]
        await asyncio.gather(*workers, return_exceptions=True)

    async def _worker_loop(self, worker_id: int) -> None:
        """One render worker — pulls jobs and renders them."""
        while not self._stopping:
            job = await self._queue.get()
            if job is None:  # sentinel → stop
                self._queue.task_done()
                return
            try:
                async with self._lock:
                    job.state = RenderState.RENDERING
                    job.started_at = time.time()
                log.debug("worker %d rendering %s", worker_id, job.scene_id)
                output = await self._render_fn(job)
                async with self._lock:
                    job.state = RenderState.COMPLETED
                    job.completed_at = time.time()
                    job.render_time_seconds = job.completed_at - (job.started_at or job.completed_at)
                    self._completed_count += 1
            except Exception as exc:
                log.warning("render failed for %s: %s", job.scene_id, exc)
                async with self._lock:
                    job.state = RenderState.FAILED
                    job.completed_at = time.time()
                    job.error = str(exc)
                    job.render_time_seconds = job.completed_at - (job.started_at or job.completed_at)
                    self._failed_count += 1
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Default render function (placeholder — real projects override this)
# ---------------------------------------------------------------------------

async def default_render_fn(job: RenderJob) -> str:
    """Default render function — a no-op that just returns the output path.

    Real projects should pass their own render_fn to RenderQueue. This
    default exists so tests can verify the queue mechanics without
    needing an actual renderer.
    """
    await asyncio.sleep(0.01)  # simulate render latency
    return job.output_path


def create_render_queue(
    *,
    render_fn: RenderFunction | None = None,
    max_concurrent: int = 4,
) -> RenderQueue:
    """Create a render queue with the given render function."""
    return RenderQueue(
        render_fn=render_fn or default_render_fn,
        max_concurrent=max_concurrent,
    )
