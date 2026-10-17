"""Control bus — items #26, #30, #32, #34, #35, #36.

One process-wide object that the query loop, tools and background
workers all share:

- #26 Kill switch: ``trip()`` arms an ``asyncio.Event``; the query loop
  (and render pipeline, via ``install_watchee``) observe it and abort
  promptly with ``KillSwitchTripped``.
- #30 Pause / resume: ``pause()`` closes an async gate; the query loop
  awaits it between turns; ``resume()`` reopens it. ``inject_message``
  queues a user message that the loop appends at the next turn
  boundary (#35 manual override).
- #36 Watchdog: every observable action calls ``feed()``; a monitor
  task trips the kill switch after ``watchdog_timeout_seconds``
  without progress.
- #34 Deadlock detection: probes the agent semaphore and registered
  long-running tasks; flags "everything blocked, nothing finishing".
- #32 Graceful shutdown: ``begin_shutdown()`` flips a phase flag,
  drains registered resources (render pipeline, audio tasks), and
  marks the bus closed — used by SIGTERM handling in the runtime.
- #27 helpers: ``call_with_limit`` wraps a coroutine with a stage
  timeout and converts ``asyncio.TimeoutError`` into a clear error.

The bus is deliberately independent of settings for its mechanics;
``configure_control_bus_from_settings`` sizes the timeouts and gates
whether the query loop actually consults it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from openharness.services.audit_log import get_audit_log

log = logging.getLogger(__name__)


class KillSwitchTripped(RuntimeError):
    """Raised inside loops that observe the control bus after a kill."""


class StageTimeoutExceeded(RuntimeError):
    """A stage exceeded its configured wall-clock limit (#27)."""


@dataclass
class ControlState:
    """Snapshot of the control bus for UIs and tools."""

    killed: bool = False
    kill_reason: str = ""
    paused: bool = False
    shutdown_phase: str = "running"  # running|draining|closed
    watchdog_timeout_seconds: float = 0.0
    last_progress_at: float = 0.0
    injected_messages: int = 0
    deadlock_alerts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "paused": self.paused,
            "shutdown_phase": self.shutdown_phase,
            "watchdog_timeout_seconds": self.watchdog_timeout_seconds,
            "seconds_since_progress": round(time.monotonic() - self.last_progress_at, 1)
            if self.last_progress_at
            else None,
            "injected_messages": self.injected_messages,
            "deadlock_alerts": self.deadlock_alerts,
        }


class ControlBus:
    """Kill switch, pause gate, watchdog, deadlock probe, overrides (#26/#30/#32/#34/#35/#36)."""

    def __init__(
        self,
        *,
        watchdog_timeout_seconds: float = 900.0,
        deadlock_probe_seconds: float = 180.0,
        deadlock_enabled: bool = True,
    ) -> None:
        self._kill_event = asyncio.Event()
        self._kill_reason = ""
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # open by default
        self._pause_requested = False
        self._shutdown_phase = "running"
        self._watchdog_timeout = max(0.0, watchdog_timeout_seconds)
        self._deadlock_probe_seconds = max(0.0, deadlock_probe_seconds)
        self._deadlock_enabled = deadlock_enabled
        self._injected: list[str] = []
        self._last_progress = time.monotonic()
        self._watchdog_task: asyncio.Task[None] | None = None
        self._deadlock_task: asyncio.Task[None] | None = None
        self._last_deadlock_check = time.monotonic()
        self.deadlock_alerts = 0
        self.state = ControlState(
            watchdog_timeout_seconds=self._watchdog_timeout,
            last_progress_at=self._last_progress,
        )
        # Registered probes for deadlock detection: callables returning a
        # dict {"blocked": int, "running": int, "name": str}.
        self._deadlock_probes: list[Callable[[], dict[str, Any]]] = []
        # Cleanup hooks executed during graceful shutdown (#32).
        self._shutdown_hooks: list[Callable[[], Awaitable[None]]] = []

    # ------------------------------------------------------------------
    # #26 — kill switch
    # ------------------------------------------------------------------

    def trip(self, reason: str = "manual kill") -> None:
        """Trip the kill switch. Idempotent; first reason wins."""
        if self._kill_event.is_set():
            return
        self._kill_reason = reason
        self._kill_event.set()
        self.state.killed = True
        self.state.kill_reason = reason
        get_audit_log().record_control("kill", {"reason": reason}, )
        log.warning("KILL SWITCH TRIPPED: %s", reason)

    @property
    def killed(self) -> bool:
        return self._kill_event.is_set()

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def check_killed(self) -> None:
        """Raise ``KillSwitchTripped`` when the switch is armed (#26)."""
        if self._kill_event.is_set():
            raise KillSwitchTripped(self._kill_reason or "kill switch tripped")

    async def wait_if_killed(self) -> None:
        await self._kill_event.wait()

    # ------------------------------------------------------------------
    # #30 — pause / resume
    # ------------------------------------------------------------------

    def pause(self, reason: str = "user pause") -> None:
        if self.killed:
            return
        self._pause_requested = True
        self._resume_event.clear()
        self.state.paused = True
        get_audit_log().record_control("pause", {"reason": reason})
        log.info("control bus paused (%s)", reason)

    def resume(self) -> None:
        self._pause_requested = False
        self._resume_event.set()
        self.state.paused = False
        self.feed("resume")
        get_audit_log().record_control("resume", {})

    @property
    def paused(self) -> bool:
        return self._pause_requested

    async def wait_if_paused(self) -> None:
        """Block while paused (#30). The query loop calls this between turns."""
        if self._pause_requested:
            await self._resume_event.wait()

    # ------------------------------------------------------------------
    # #35 — manual override (injected messages)
    # ------------------------------------------------------------------

    def inject_message(self, message: str) -> int:
        """Queue a human decision that the loop appends next turn (#35)."""
        text = message.strip()
        if not text:
            return 0
        self._injected.append(text)
        self.state.injected_messages = len(self._injected)
        get_audit_log().record_control("inject", {"chars": len(text), "head": text[:120]})
        self.feed("inject")
        return len(self._injected)

    def drain_injected_messages(self) -> list[str]:
        """Return and clear queued override messages."""
        messages = self._injected
        self._injected = []
        self.state.injected_messages = 0
        return messages

    # ------------------------------------------------------------------
    # #36 — watchdog
    # ------------------------------------------------------------------

    def feed(self, what: str = "") -> None:
        """Record observable progress; resets the watchdog clock."""
        self._last_progress = time.monotonic()
        self.state.last_progress_at = self._last_progress

    @property
    def seconds_since_progress(self) -> float:
        return time.monotonic() - self._last_progress

    def start_watchdog(self) -> None:
        """Start the watchdog monitor task (idempotent, no-op when timeout=0)."""
        if self._watchdog_task is not None or self._watchdog_timeout <= 0:
            return
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="control-watchdog"
        )

    async def _watchdog_loop(self) -> None:
        check_interval = min(5.0, max(0.5, self._watchdog_timeout / 20))
        while True:
            await asyncio.sleep(check_interval)
            if self.killed or self.paused:
                continue  # paused sessions make no progress by design
            if self.seconds_since_progress >= self._watchdog_timeout:
                self.trip(
                    f"watchdog: no progress for {self._watchdog_timeout:.0f}s"
                )
                return

    # ------------------------------------------------------------------
    # #34 — deadlock detection
    # ------------------------------------------------------------------

    def register_deadlock_probe(self, probe: Callable[[], dict[str, Any]]) -> None:
        """Register a probe describing a subsystem's blocked/running state."""
        self._deadlock_probes.append(probe)

    def detect_deadlock(self) -> dict[str, Any] | None:
        """Return a deadlock report, or None when the system looks alive.

        A deadlock is: every registered subsystem reports all of its
        workers/permits blocked, AND there has been NO OBSERVABLE
        PROGRESS for at least ``deadlock_probe_seconds`` (the docstring
        always required this — the previous implementation never
        checked it, so a healthy pipeline at 100% utilization, where
        the probes counted WORKING agents/workers as "blocked", was
        killed by the kill switch after ~3 minutes of full load).
        """
        if not self._deadlock_enabled or not self._deadlock_probes:
            return None
        # No-progress gate FIRST: while any observable progress happened
        # within the probe window, the system is alive by definition.
        if self.seconds_since_progress < self._deadlock_probe_seconds:
            self._last_deadlock_check = time.monotonic()
            return None
        reports = []
        all_blocked = True
        for probe in self._deadlock_probes:
            try:
                report = probe()
            except Exception:
                continue
            reports.append(report)
            blocked = int(report.get("blocked", 0) or 0)
            running = int(report.get("running", 0) or 0)
            if running > 0 or blocked == 0:
                all_blocked = False
        if not reports or not all_blocked:
            self._last_deadlock_check = time.monotonic()
            return None
        # Everything blocked — only count it as a deadlock when it
        # persists across consecutive probes (spurious wakeups excluded).
        persisted = (time.monotonic() - self._last_deadlock_check) >= self._deadlock_probe_seconds
        if not persisted:
            return None
        self.deadlock_alerts += 1
        self.state.deadlock_alerts = self.deadlock_alerts
        result = {
            "deadlocked": True,
            "subsystems": reports,
            "seconds_since_progress": round(self.seconds_since_progress, 1),
        }
        get_audit_log().record_control("deadlock", result)
        return result

    def start_deadlock_monitor(self) -> None:
        if self._deadlock_task is not None or not self._deadlock_enabled:
            return
        self._deadlock_task = asyncio.create_task(
            self._deadlock_loop(), name="control-deadlock"
        )

    async def _deadlock_loop(self) -> None:
        interval = max(5.0, self._deadlock_probe_seconds / 2)
        while True:
            await asyncio.sleep(interval)
            if self.killed:
                return
            report = self.detect_deadlock()
            if report is not None:
                log.error("deadlock detected: %s", report)
                self.trip("deadlock: all subsystems blocked with no progress")
                return

    # ------------------------------------------------------------------
    # #27 — stage time limits
    # ------------------------------------------------------------------

    async def call_with_limit(self, coro: Awaitable[Any], *, timeout_seconds: float, stage: str) -> Any:
        """Run ``coro`` under a stage timeout, raising ``StageTimeoutExceeded``."""
        if timeout_seconds <= 0:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            self.feed(f"{stage}:timeout")
            raise StageTimeoutExceeded(
                f"{stage} exceeded its time limit of {timeout_seconds:.0f}s"
            ) from exc

    # ------------------------------------------------------------------
    # #32 — graceful shutdown
    # ------------------------------------------------------------------

    def register_shutdown_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._shutdown_hooks.append(hook)

    @property
    def shutdown_phase(self) -> str:
        return self._shutdown_phase

    async def begin_shutdown(self, *, reason: str = "graceful shutdown") -> None:
        """Drain in-flight work, run shutdown hooks, close the bus (#32).

        Also stops the watchdog / deadlock monitor tasks — previously they
        kept looping forever after the bus was closed, leaking two tasks
        per session in long-lived processes and test suites.
        """
        if self._shutdown_phase != "running":
            return
        self._shutdown_phase = "draining"
        self.state.shutdown_phase = "draining"
        get_audit_log().record_control("shutdown.begin", {"reason": reason})
        log.info("graceful shutdown started (%s)", reason)
        for hook in self._shutdown_hooks:
            try:
                await asyncio.wait_for(hook(), timeout=60.0)
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                log.warning("shutdown hook failed: %s", exc)
        self._shutdown_phase = "closed"
        self.state.shutdown_phase = "closed"
        self._stop_monitor_tasks()
        get_audit_log().record_control("shutdown.complete", {})
        log.info("graceful shutdown complete")

    def _stop_monitor_tasks(self) -> None:
        """Cancel watchdog/deadlock monitor tasks (idempotent)."""
        for task in (self._watchdog_task, self._deadlock_task):
            if task is not None and not task.done():
                task.cancel()
        self._watchdog_task = None
        self._deadlock_task = None

    def status(self) -> dict[str, Any]:
        state = self.state.as_dict()
        state["injected_pending"] = len(self._injected)
        return state


# ----------------------------------------------------------------------
# Process-wide singleton + settings wiring
# ----------------------------------------------------------------------

_BUS: ControlBus | None = None


def get_control_bus() -> ControlBus:
    """Return the shared control bus, creating it on first use."""
    global _BUS
    if _BUS is None:
        _BUS = ControlBus()
    return _BUS


def set_control_bus(bus: ControlBus | None) -> None:
    """Install or reset the singleton.

    When replacing a live bus, its monitor tasks are stopped first —
    otherwise every session rebuild leaked a watchdog and a deadlock
    monitor task that kept referencing the orphaned bus forever.
    """
    global _BUS
    if _BUS is not None and _BUS is not bus:
        _BUS._stop_monitor_tasks()
    _BUS = bus


def configure_control_bus_from_settings(settings: Any) -> ControlBus | None:
    """Engine wiring: configure the shared bus from ``settings.control``.

    Returns None when control features are disabled — callers must then
    skip the kill/pause/watchdog checks entirely (identity behavior).
    """
    control = getattr(settings, "control", None)
    if control is None or not getattr(control, "enabled", False):
        return None
    bus = ControlBus(
        watchdog_timeout_seconds=float(
            getattr(control, "watchdog_timeout_seconds", 900) or 0
        ),
        deadlock_probe_seconds=float(
            getattr(control, "deadlock_probe_seconds", 180) or 0
        ),
        deadlock_enabled=bool(getattr(control, "deadlock_detection", True)),
    )
    return bus
