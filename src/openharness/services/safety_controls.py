"""Safety controls — items 1, 2, 3, 6, 8, 9, 10.

#1 Kill Switch: Stop a runaway render or agent loop immediately.
#2 Time Limits Per Stage: Prevent a single scene/stage from running too long.
#3 Pause/Resume: Pause the agent mid-task, inspect state, then resume.
#6 Graceful Shutdown: Finish current work before Kaggle kills the session.
#8 Cost Alerts: Warn before budget is exceeded (not just after).
#9 Deadlock Detection: Detect if the agent is stuck in a loop.
#10 Manual Override: Human injects a decision mid-task.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# ============================================================
# #1 Kill Switch
# ============================================================

class KillSwitchState(str, Enum):
    RUNNING = "running"
    KILLED = "killed"


@dataclass
class KillSwitch:
    """Global kill switch to stop runaway renders or agent loops.

    Any long-running operation checks ``kill_switch.is_killed()``
    periodically. When killed, the operation should abort immediately.

    Usage:
        ks = KillSwitch()
        # In the render loop:
        for frame in frames:
            if ks.is_killed():
                break
            render(frame)

        # From another coroutine / signal handler:
        ks.kill("GPU overheating")
    """

    state: KillSwitchState = KillSwitchState.RUNNING
    reason: str = ""
    killed_at: float | None = None
    _listeners: list[Callable[[str], None]] = field(default_factory=list)

    def kill(self, reason: str = "") -> None:
        """Trigger the kill switch. All subsequent is_killed() checks return True."""
        self.state = KillSwitchState.KILLED
        self.reason = reason
        self.killed_at = time.time()
        log.warning("KILL SWITCH TRIGGERED: %s", reason)
        for listener in self._listeners:
            try:
                listener(reason)
            except Exception:
                pass

    def is_killed(self) -> bool:
        """Return True if the kill switch has been triggered."""
        return self.state == KillSwitchState.KILLED

    def reset(self) -> None:
        """Reset the kill switch for a new run."""
        self.state = KillSwitchState.RUNNING
        self.reason = ""
        self.killed_at = None

    def add_listener(self, fn: Callable[[str], None]) -> None:
        """Register a callback fired when the switch is triggered."""
        self._listeners.append(fn)

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "killed_at": self.killed_at,
        }


# Singleton kill switch for the process
_KILL_SWITCH: KillSwitch | None = None


def get_kill_switch() -> KillSwitch:
    """Return the global kill switch."""
    global _KILL_SWITCH
    if _KILL_SWITCH is None:
        _KILL_SWITCH = KillSwitch()
    return _KILL_SWITCH


def reset_kill_switch() -> None:
    """Test helper: reset the global kill switch."""
    global _KILL_SWITCH
    _KILL_SWITCH = None


# ============================================================
# #2 Time Limits Per Stage
# ============================================================

@dataclass
class StageTimeLimit:
    """Time limit for a single pipeline stage."""

    stage: str
    max_seconds: float
    started_at: float = field(default_factory=time.time)


class StageTimeLimits:
    """Enforce per-stage time limits to prevent runaway scenes.

    Each stage (planning, scene_generation, qa_review, render, etc.)
    gets its own time limit. If a stage exceeds its limit, a
    ``StageTimeoutError`` is raised.

    Usage:
        limits = StageTimeLimits()
        limits.set("scene_generation", max_seconds=120)
        limits.set("render", max_seconds=300)

        with limits.track("scene_generation"):
            generate_scenes()  # raises StageTimeoutError if > 120s
    """

    DEFAULT_LIMITS: dict[str, float] = {
        "planning": 300.0,  # 5 min
        "scene_generation": 120.0,  # 2 min per batch
        "batch_scene_generation": 300.0,  # 5 min per batch
        "qa_review": 60.0,  # 1 min per scene
        "revision": 120.0,  # 2 min per scene
        "render": 300.0,  # 5 min per scene
        "audio_generation": 60.0,  # 1 min per scene
    }

    def __init__(self) -> None:
        self._limits: dict[str, float] = dict(self.DEFAULT_LIMITS)
        self._active: dict[str, StageTimeLimit] = {}

    def set(self, stage: str, *, max_seconds: float) -> None:
        """Set a time limit for a stage."""
        self._limits[stage] = max_seconds

    def get(self, stage: str) -> float:
        """Return the time limit for a stage (or default 300s)."""
        return self._limits.get(stage, 300.0)

    def track(self, stage: str) -> "_StageTimer":
        """Return a context manager that enforces the time limit."""
        return _StageTimer(self, stage)

    def check(self, stage: str) -> bool:
        """Return True if the stage has exceeded its time limit."""
        active = self._active.get(stage)
        if active is None:
            return False
        elapsed = time.time() - active.started_at
        return elapsed > active.max_seconds

    def remaining(self, stage: str) -> float:
        """Return seconds remaining for a stage, or -1 if not tracked."""
        active = self._active.get(stage)
        if active is None:
            return -1.0
        elapsed = time.time() - active.started_at
        return max(0.0, active.max_seconds - elapsed)


class StageTimeoutError(TimeoutError):
    """Raised when a stage exceeds its time limit."""

    def __init__(self, stage: str, elapsed: float, limit: float) -> None:
        super().__init__(
            f"Stage '{stage}' timed out after {elapsed:.1f}s (limit: {limit:.1f}s)"
        )
        self.stage = stage
        self.elapsed = elapsed
        self.limit = limit


class _StageTimer:
    """Context manager for stage time limits."""

    def __init__(self, limits: StageTimeLimits, stage: str) -> None:
        self._limits = limits
        self._stage = stage
        self._limit_entry: StageTimeLimit | None = None

    def __enter__(self) -> "_StageTimer":
        max_seconds = self._limits.get(self._stage)
        self._limit_entry = StageTimeLimit(
            stage=self._stage,
            max_seconds=max_seconds,
            started_at=time.time(),
        )
        self._limits._active[self._stage] = self._limit_entry
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._limit_entry is not None:
            elapsed = time.time() - self._limit_entry.started_at
            if elapsed > self._limit_entry.max_seconds:
                raise StageTimeoutError(
                    self._stage, elapsed, self._limit_entry.max_seconds
                )
        self._limits._active.pop(self._stage, None)


# ============================================================
# #3 Pause/Resume
# ============================================================

class PauseResume:
    """Pause the agent mid-task, inspect state, then resume.

    The agent checks ``should_pause()`` periodically. When paused,
    it enters a wait loop until ``resume()`` is called.

    Usage:
        pr = PauseResume()
        # In the agent loop:
        for scene in scenes:
            await pr.check_pause()  # blocks if paused
            generate(scene)

        # From the user / API:
        pr.pause("User wants to inspect scene 5")
        # ... user inspects ...
        pr.resume()
    """

    def __init__(self) -> None:
        self._paused = False
        self._pause_reason = ""
        self._paused_at: float | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # not paused initially

    def pause(self, reason: str = "") -> None:
        """Pause the agent. The next ``check_pause()`` call will block."""
        self._paused = True
        self._pause_reason = reason
        self._paused_at = time.time()
        self._resume_event.clear()
        log.info("Agent paused: %s", reason)

    def resume(self) -> None:
        """Resume the agent after a pause."""
        self._paused = False
        self._pause_reason = ""
        self._paused_at = None
        self._resume_event.set()
        log.info("Agent resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def check_pause(self, *, timeout: float | None = None) -> None:
        """Block if paused. Returns immediately when not paused or after resume.

        Args:
            timeout: Maximum seconds to wait. None = wait forever.
        """
        if not self._paused:
            return
        try:
            await asyncio.wait_for(self._resume_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # timed out waiting for resume

    def stats(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "reason": self._pause_reason,
            "paused_at": self._paused_at,
        }


# ============================================================
# #6 Graceful Shutdown
# ============================================================

@dataclass
class GracefulShutdown:
    """Finish current work before Kaggle kills the session.

    Registers signal handlers for SIGTERM/SIGINT (Kaggle sends these
    when the session is about to end). When triggered, the shutdown
    handler sets a flag that the agent checks. The agent finishes the
    current scene/turn, saves state, then exits cleanly.

    Usage:
        gs = GracefulShutdown()
        gs.register()

        # In the agent loop:
        for scene in scenes:
            if gs.should_shutdown():
                gs.save_and_exit()  # saves state and exits
            generate(scene)
    """

    _shutdown_requested: bool = False
    _reason: str = ""
    _registered: bool = False
    _on_shutdown: list[Callable[[], None]] = field(default_factory=list)

    def register(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        if self._registered:
            return
        self._registered = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass  # can't register in non-main thread

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self._shutdown_requested = True
        self._reason = f"Signal {signum}"
        log.warning("Graceful shutdown requested: %s", self._reason)
        for callback in self._on_shutdown:
            try:
                callback()
            except Exception:
                pass

    def should_shutdown(self) -> bool:
        """Return True if a shutdown has been requested."""
        return self._shutdown_requested

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when shutdown is requested."""
        self._on_shutdown.append(callback)

    def save_and_exit(self, exit_code: int = 0) -> None:
        """Save state and exit the process."""
        log.info("Graceful shutdown: saving state and exiting")
        for callback in self._on_shutdown:
            try:
                callback()
            except Exception:
                pass
        import sys
        sys.exit(exit_code)

    def stats(self) -> dict[str, Any]:
        return {
            "shutdown_requested": self._shutdown_requested,
            "reason": self._reason,
            "registered": self._registered,
        }


# ============================================================
# #8 Cost Alerts (pre-budget warning)
# ============================================================

@dataclass
class CostAlert:
    """A cost alert when spending approaches the budget."""

    threshold: float  # fraction of budget (0.0 - 1.0)
    message: str
    triggered: bool = False


class CostAlertManager:
    """Warn before budget is exceeded, not just after.

    Fires alerts at configurable thresholds (e.g., 50%, 80%, 90%, 100%).

    Usage:
        alerts = CostAlertManager(budget_usd=10.0)
        alerts.set_threshold(0.5, "Half budget used")
        alerts.set_threshold(0.8, "80% budget used — consider stopping")
        alerts.set_threshold(0.9, "90% budget used — stop now")

        # After each LLM call:
        alerts.check(spent_usd=5.2)  # triggers the 50% alert
    """

    def __init__(self, *, budget_usd: float) -> None:
        self._budget = budget_usd
        self._alerts: list[CostAlert] = []
        self._callbacks: list[Callable[[CostAlert], None]] = []

    def set_threshold(self, threshold: float, message: str) -> None:
        """Set an alert at a fraction of the budget (0.5 = 50%)."""
        self._alerts.append(CostAlert(threshold=threshold, message=message))
        # Sort by threshold so we check in order
        self._alerts.sort(key=lambda a: a.threshold)

    def check(self, spent_usd: float) -> list[CostAlert]:
        """Check spending against all thresholds. Returns newly-triggered alerts."""
        triggered: list[CostAlert] = []
        fraction = spent_usd / self._budget if self._budget > 0 else 0.0
        for alert in self._alerts:
            if not alert.triggered and fraction >= alert.threshold:
                alert.triggered = True
                triggered.append(alert)
                log.warning("COST ALERT (%.0f%%): %s", alert.threshold * 100, alert.message)
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception:
                        pass
        return triggered

    def on_alert(self, callback: Callable[[CostAlert], None]) -> None:
        """Register a callback fired when an alert triggers."""
        self._callbacks.append(callback)

    def reset(self) -> None:
        """Reset all alerts (e.g., after a budget change)."""
        for alert in self._alerts:
            alert.triggered = False

    def stats(self) -> dict[str, Any]:
        return {
            "budget_usd": self._budget,
            "alerts": [
                {
                    "threshold": a.threshold,
                    "message": a.message,
                    "triggered": a.triggered,
                }
                for a in self._alerts
            ],
        }


# ============================================================
# #9 Deadlock Detection
# ============================================================

@dataclass
class DeadlockDetector:
    """Detect if the agent is stuck in a loop.

    Tracks the last N actions. If the same action repeats more than
    ``max_repeats`` times within ``window_seconds``, a deadlock is
    detected.

    Usage:
        detector = DeadlockDetector(max_repeats=5, window_seconds=60)
        # In the agent loop:
        detector.record(action="read_file", tool_input={"path": "scene_001.js"})
        if detector.is_deadlocked():
            log.error("Agent is stuck! Breaking out of loop.")
            break
    """

    max_repeats: int = 5
    window_seconds: float = 60.0
    _actions: list[tuple[str, str, float]] = field(default_factory=list)  # (action, input_hash, timestamp)

    def record(self, *, action: str, tool_input: dict[str, Any] | None = None) -> None:
        """Record an action for deadlock detection."""
        import hashlib
        import json
        input_hash = ""
        if tool_input:
            try:
                input_hash = hashlib.sha1(
                    json.dumps(tool_input, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:12]
            except (TypeError, ValueError):
                input_hash = str(tool_input)[:12]
        self._actions.append((action, input_hash, time.time()))
        # Prune old entries outside the window
        cutoff = time.time() - self.window_seconds
        self._actions = [(a, h, t) for a, h, t in self._actions if t > cutoff]

    def is_deadlocked(self) -> bool:
        """Return True if the same action+input repeats too many times."""
        if len(self._actions) < self.max_repeats:
            return False
        # Count the most recent N actions
        recent = self._actions[-self.max_repeats:]
        # Check if all are the same (action + input_hash)
        first = recent[0]
        return all(a == first[0] and h == first[1] for a, h, _ in recent)

    def reset(self) -> None:
        """Clear the action history."""
        self._actions.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "max_repeats": self.max_repeats,
            "window_seconds": self.window_seconds,
            "recent_actions": len(self._actions),
            "deadlocked": self.is_deadlocked(),
        }


# ============================================================
# #10 Manual Override
# ============================================================

@dataclass
class ManualOverride:
    """Human injects a decision mid-task.

    The agent checks ``has_override()`` periodically. When an override
    is queued, the agent reads it and incorporates the human's decision
    into its next action.

    Usage:
        mo = ManualOverride()
        # In the agent loop:
        if mo.has_override():
            decision = mo.get_override()
            if decision == "skip_scene":
                continue  # skip this scene
            elif decision == "use_red":
                # apply the override
                ...

        # From the user/API:
        mo.queue_override("skip_scene", reason="Scene 5 is broken")
    """

    _queue: list[tuple[str, str]] = field(default_factory=list)  # (decision, reason)

    def queue_override(self, decision: str, reason: str = "") -> None:
        """Queue a manual override for the agent to pick up."""
        self._queue.append((decision, reason))
        log.info("Manual override queued: %s (%s)", decision, reason)

    def has_override(self) -> bool:
        """Return True if there's a pending override."""
        return len(self._queue) > 0

    def get_override(self) -> tuple[str, str] | None:
        """Dequeue and return the next override, or None."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def clear(self) -> None:
        """Clear all pending overrides."""
        self._queue.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "pending_overrides": len(self._queue),
            "queue": [{"decision": d, "reason": r} for d, r in self._queue],
        }
