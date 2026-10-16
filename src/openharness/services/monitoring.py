"""Monitoring — items 13, 15.

#13 Progress Tracking: Real-time view of pipeline progress.
#15 Render Queue Monitoring: See what's rendering, what's queued.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ============================================================
# #13 Progress Tracking
# ============================================================

@dataclass
class ProgressStage:
    """A single stage in the pipeline progress tracker."""

    name: str
    total: int
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    started_at: float = field(default_factory=time.time)


@dataclass
class ProgressSnapshot:
    """A snapshot of pipeline progress at a point in time."""

    stages: dict[str, ProgressStage]
    overall_percent: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None = None


class ProgressTracker:
    """Real-time pipeline progress tracker.

    Tracks each pipeline stage (planning, scene_generation, qa, render,
    audio) with completion counts and estimated time remaining.

    Usage:
        tracker = ProgressTracker()
        tracker.add_stage("scene_generation", total=200)
        tracker.add_stage("qa_review", total=200)
        tracker.add_stage("render", total=200)

        # After generating 50 scenes:
        tracker.update("scene_generation", completed=50)

        # Get a snapshot:
        snapshot = tracker.snapshot()
        print(f"Overall: {snapshot.overall_percent:.0f}%")
    """

    def __init__(self) -> None:
        self._stages: dict[str, ProgressStage] = {}
        self._start_time = time.time()

    def add_stage(self, name: str, *, total: int) -> None:
        """Register a pipeline stage."""
        self._stages[name] = ProgressStage(name=name, total=total)

    def update(
        self,
        stage_name: str,
        *,
        completed: int | None = None,
        failed: int | None = None,
        in_progress: int | None = None,
        increment_completed: bool = False,
        increment_failed: bool = False,
    ) -> None:
        """Update the progress of a stage."""
        stage = self._stages.get(stage_name)
        if stage is None:
            return
        if increment_completed:
            stage.completed += 1
        elif completed is not None:
            stage.completed = completed
        if increment_failed:
            stage.failed += 1
        elif failed is not None:
            stage.failed = failed
        if in_progress is not None:
            stage.in_progress = in_progress

    def snapshot(self) -> ProgressSnapshot:
        """Return a snapshot of the current progress."""
        total_work = sum(s.total for s in self._stages.values())
        done_work = sum(s.completed + s.failed for s in self._stages.values())
        overall = (done_work / total_work * 100) if total_work > 0 else 0.0
        elapsed = time.time() - self._start_time

        # Estimate remaining time based on throughput
        remaining_work = total_work - done_work
        estimated_remaining = None
        if done_work > 0 and elapsed > 0:
            throughput = done_work / elapsed  # items per second
            if throughput > 0:
                estimated_remaining = remaining_work / throughput

        return ProgressSnapshot(
            stages=dict(self._stages),
            overall_percent=round(overall, 1),
            elapsed_seconds=round(elapsed, 1),
            estimated_remaining_seconds=round(estimated_remaining, 1) if estimated_remaining else None,
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for JSON output."""
        snap = self.snapshot()
        return {
            "overall_percent": snap.overall_percent,
            "elapsed_seconds": snap.elapsed_seconds,
            "estimated_remaining_seconds": snap.estimated_remaining_seconds,
            "stages": {
                name: {
                    "total": s.total,
                    "completed": s.completed,
                    "failed": s.failed,
                    "in_progress": s.in_progress,
                    "percent": round((s.completed + s.failed) / s.total * 100, 1) if s.total > 0 else 0.0,
                }
                for name, s in snap.stages.items()
            },
        }


# ============================================================
# #15 Render Queue Monitoring
# ============================================================

@dataclass
class RenderQueueStatus:
    """Real-time status of the render queue."""

    total_jobs: int
    queued: int
    rendering: int
    completed: int
    failed: int
    cancelled: int
    active_workers: int
    avg_render_time_seconds: float | None = None


class RenderQueueMonitor:
    """Monitor a render queue's status in real-time.

    Wraps the RenderQueue from render_queue.py and provides a
    status snapshot without modifying the queue itself.

    Usage:
        from openharness.services.render_queue import RenderQueue
        queue = RenderQueue(render_fn=my_fn, max_concurrent=4)
        monitor = RenderQueueMonitor(queue)
        await queue.start()

        # Check status:
        status = monitor.status()
        print(f"Rendering: {status.rendering}/{status.total_jobs}")
    """

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def status(self) -> RenderQueueStatus:
        """Return a snapshot of the render queue status."""
        jobs = getattr(self._queue, "_jobs", {})
        total = len(jobs)
        # Count by state
        from openharness.services.render_queue import RenderState
        queued = sum(1 for j in jobs.values() if getattr(j, "state", None) == RenderState.QUEUED)
        rendering = sum(1 for j in jobs.values() if getattr(j, "state", None) == RenderState.RENDERING)
        completed = sum(1 for j in jobs.values() if getattr(j, "state", None) == RenderState.COMPLETED)
        failed = sum(1 for j in jobs.values() if getattr(j, "state", None) == RenderState.FAILED)
        cancelled = sum(1 for j in jobs.values() if getattr(j, "state", None) == RenderState.CANCELLED)

        # Compute average render time for completed jobs
        render_times = [
            j.render_time_seconds for j in jobs.values()
            if getattr(j, "render_time_seconds", None) and j.render_time_seconds > 0
        ]
        avg_time = sum(render_times) / len(render_times) if render_times else None

        return RenderQueueStatus(
            total_jobs=total,
            queued=queued,
            rendering=rendering,
            completed=completed,
            failed=failed,
            cancelled=cancelled,
            active_workers=getattr(self._queue, "_max_concurrent", 0),
            avg_render_time_seconds=round(avg_time, 2) if avg_time else None,
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for JSON output."""
        s = self.status()
        return {
            "total_jobs": s.total_jobs,
            "queued": s.queued,
            "rendering": s.rendering,
            "completed": s.completed,
            "failed": s.failed,
            "cancelled": s.cancelled,
            "active_workers": s.active_workers,
            "avg_render_time_seconds": s.avg_render_time_seconds,
        }
