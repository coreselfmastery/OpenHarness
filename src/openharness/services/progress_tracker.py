"""Pipeline progress tracking — items #38 and #39.

#38 — whole-pipeline progress: every scene moves through the stage
     machine ``pending -> generating -> generated -> rendering -> qa ->
     approved | failed`` and the tracker reports overall completion
     plus a moving-average ETA from observed stage durations.

#39 — per-scene status: the same registry answers "which scenes are
     done, which are rendering, which failed" without re-reading the
     filesystem.

The tracker is fed by the live pipeline components: batch_scene marks
scenes generated, the render pipeline marks rendering/rendered, QA
marks qa/approved, and the control bus can abort everything. No
component is allowed to report a scene as done that it did not itself
advance — the stage machine rejects invalid transitions.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

STAGES = (
    "pending",       # not started
    "generating",    # scene code being written
    "generated",     # code written, not rendered
    "queued",        # render job enqueued
    "rendering",     # render subprocess running
    "rendered",      # output file exists (tier-aware)
    "qa",            # under QA review
    "approved",      # passed QA (terminal)
    "failed",        # failed terminally (terminal)
    "aborted",       # killed mid-flight (terminal)
)

TERMINAL = {"approved", "failed", "aborted"}

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"generating", "queued", "failed", "aborted"},
    "generating": {"generated", "failed", "aborted"},
    "generated": {"queued", "rendering", "qa", "failed", "aborted"},
    "queued": {"rendering", "rendered", "failed", "aborted"},   # cache-hit skips render
    "rendering": {"rendered", "failed", "aborted"},
    "rendered": {"qa", "queued", "failed", "aborted"},          # re-render allowed
    "qa": {"approved", "failed", "generated", "aborted"},
    "approved": {"queued", "rendering"},                  # re-render after approval
    "failed": {"queued", "rendering", "generating"},      # retry after failure
    "aborted": set(),
}

#: Stages where actual WORK happens — the only stages whose observed
#: durations feed the ETA model. The others (``generated``, ``rendered``,
#: ``approved``) are millisecond waypoint states between work stages
#: (and ``approved`` is terminal, so its clock never closes) — summing
#: them produced an ETA of ~zero or None for every real pipeline.
WORK_STAGES = ("generating", "rendering", "qa")


class SceneProgress:
    """One scene's tracked state (#39)."""

    __slots__ = ("scene", "stage", "updated_at", "detail", "history", "tier")

    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.stage = "pending"
        self.tier = ""
        self.updated_at = time.time()
        self.detail = ""
        self.history: list[tuple[float, str]] = [(self.updated_at, "pending")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "stage": self.stage,
            "tier": self.tier,
            "detail": self.detail,
            "updated_at_iso": time.strftime("%H:%M:%S", time.localtime(self.updated_at)),
        }


class ProgressTracker:
    """Registry of scene stages with overall progress + ETA (#38/#39)."""

    def __init__(self, *, eta_window: int = 20) -> None:
        self._scenes: dict[str, SceneProgress] = {}
        self._stage_durations: dict[str, deque[float]] = {
            stage: deque(maxlen=eta_window) for stage in STAGES
        }
        self._stage_clocks: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Stage machine
    # ------------------------------------------------------------------

    def mark(self, scene: str, stage: str, *, detail: str = "", tier: str = "") -> bool:
        """Advance a scene to ``stage``; returns False for invalid jumps.

        Invalid transitions are logged and ignored rather than raising —
        a progress tracker must never take down the pipeline — but they
        are visible in the log for debugging.
        """
        if stage not in STAGES:
            log.warning("progress: unknown stage %r for %s", stage, scene)
            return False
        record = self._scenes.setdefault(scene, SceneProgress(scene))
        if stage != record.stage and stage not in _VALID_TRANSITIONS[record.stage]:
            log.debug(
                "progress: %s cannot go %s -> %s (kept at %s)",
                scene, record.stage, stage, record.stage,
            )
            return False
        now = time.time()
        if stage != record.stage:
            self._close_stage_clock(scene, record.stage, now)
            record.stage = stage
            record.history.append((now, stage))
            self._stage_clocks[(scene, stage)] = now
        record.updated_at = now
        record.detail = detail or record.detail
        record.tier = tier or record.tier
        return True

    def force(self, scene: str, stage: str, *, detail: str = "") -> None:
        """Set a stage unconditionally (admin/restore path only)."""
        if stage not in STAGES:
            return
        record = self._scenes.setdefault(scene, SceneProgress(scene))
        record.stage = stage
        record.updated_at = time.time()
        record.detail = detail
        record.history.append((record.updated_at, stage))

    def _close_stage_clock(self, scene: str, stage: str, now: float) -> None:
        """Record how long ``scene`` spent in ``stage`` for the ETA model."""
        started = self._stage_clocks.pop((scene, stage), None)
        if started is not None and now > started:
            self._stage_durations[stage].append(now - started)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def scene_status(self, scene: str) -> dict[str, Any] | None:
        record = self._scenes.get(scene)
        return record.as_dict() if record else None

    def all_scenes(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self._scenes.values()]

    def by_stage(self, stage: str) -> list[str]:
        return [name for name, record in self._scenes.items() if record.stage == stage]

    def summary(self) -> dict[str, Any]:
        total = len(self._scenes)
        done = sum(1 for r in self._scenes.values() if r.stage in TERMINAL)
        approved = sum(1 for r in self._scenes.values() if r.stage == "approved")
        failed = sum(1 for r in self._scenes.values() if r.stage == "failed")
        active = [r.as_dict() for r in self._scenes.values() if r.stage not in TERMINAL]
        progress = (done / total) if total else None
        return {
            "total_scenes": total,
            "approved": approved,
            "failed": failed,
            "in_flight": total - done,
            "completion": round(progress, 3) if progress is not None else None,
            "eta_seconds": self.eta_seconds(),
            "active": active,
            "stage_counts": {
                stage: sum(1 for r in self._scenes.values() if r.stage == stage)
                for stage in STAGES
                if any(r.stage == stage for r in self._scenes.values())
            },
        }

    def eta_seconds(self) -> float | None:
        """ETA from the average remaining WORK time per in-flight scene.

        Honest model: for every scene still in flight, sum the observed
        average durations of the work stages it still owes (a scene in
        ``rendering`` owes the rest of rendering + QA; a ``pending``
        scene owes generating + rendering + QA), then divide by the
        number of scenes actively being worked right now (observed
        pipeline parallelism). Returns None until at least one work
        stage duration has been observed.
        """
        total = len(self._scenes)
        if total == 0:
            return None
        in_flight = [r for r in self._scenes.values() if r.stage not in TERMINAL]
        if not in_flight:
            return 0.0

        def _avg(stage: str) -> float:
            durations = self._stage_durations.get(stage)
            if not durations:
                return 0.0
            return sum(durations) / len(durations)

        if not any(_avg(stage) > 0 for stage in WORK_STAGES):
            return None  # no work-stage observation yet — refuse to guess

        order = {stage: index for index, stage in enumerate(STAGES)}
        remaining_work = 0.0
        for record in in_flight:
            # Stages at-or-after the scene's current position in the
            # machine: what this scene still owes. (``queued`` scenes sit
            # past ``generating`` in the ordering, so they correctly owe
            # only rendering + QA.)
            owed = [
                _avg(stage)
                for stage in WORK_STAGES
                if order[stage] >= order[record.stage]
            ]
            remaining_work += sum(owed)

        # Observed parallelism = scenes in a work stage RIGHT NOW. When
        # nothing is actively working (between waves, all queued), fall
        # back to a conservative estimate of up to 4 concurrent scenes.
        active_now = sum(1 for r in self._scenes.values() if r.stage in WORK_STAGES)
        parallelism = float(active_now) if active_now > 0 else float(min(len(in_flight), 4))
        parallelism = max(1.0, parallelism)
        return round(remaining_work / parallelism, 1)

    def reset(self) -> None:
        self._scenes.clear()
        self._stage_clocks.clear()
        for durations in self._stage_durations.values():
            durations.clear()


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_TRACKER: ProgressTracker | None = None


def get_progress_tracker() -> ProgressTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = ProgressTracker()
    return _TRACKER


def set_progress_tracker(tracker: ProgressTracker | None) -> None:
    """Test helper."""
    global _TRACKER
    _TRACKER = tracker
