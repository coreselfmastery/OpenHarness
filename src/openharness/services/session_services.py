"""Session services bootstrap — wires items #20, #26-#40 into one session.

``bootstrap_session_services`` is called by the runtime when a session is
built (before the API client is resolved). It configures, in order:

1. the audit log (#29) — first, because the control bus records to it,
2. the control bus (#26 kill switch, #30 pause, #36 watchdog, #34
   deadlock monitor) from ``settings.control``,
3. the session cost tracker (#28) whose halt alert trips the kill switch,
4. the disk model cache (#20) before any HF import,
5. the render pipeline singleton (#11-#16) with audit + progress hooks
   attached, and deadlock probes for both the agent semaphore and the
   render workers (#34),
6. graceful-shutdown hooks (#32): drain the render queue, close the
   persistent store, flush a final audit entry — plus a SIGTERM handler
   that trips the kill switch so the query loop aborts at the next turn
   boundary instead of being shot mid-write.

``shutdown_session_services`` runs the shutdown hooks (draining in-flight
work) and is called from ``close_runtime``.

Every piece degrades to identity: with control features disabled, the
bootstrap returns an empty :class:`SessionServices` and the engine skips
all control checks.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openharness.services.audit_log import (
    AuditLog,
    configure_audit_log_from_settings,
    get_audit_log,
    set_audit_log,
)
from openharness.services.control_bus import (
    ControlBus,
    configure_control_bus_from_settings,
    set_control_bus,
)
from openharness.services.llm_pipeline import SessionCostTracker

log = logging.getLogger(__name__)


@dataclass
class SessionServices:
    """Everything the bootstrap configured for one session."""

    session_id: str = ""
    bus: ControlBus | None = None
    cost_tracker: SessionCostTracker | None = None
    audit: AuditLog | None = None
    signal_handlers_installed: bool = False
    _shutdown_done: bool = field(default=False, repr=False)


_SERVICES: SessionServices | None = None


def get_session_services() -> SessionServices:
    """Return the current session's services (empty record when absent)."""
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = SessionServices()
    return _SERVICES


def reset_session_services() -> None:
    """Test helper: drop the process-wide services record."""
    global _SERVICES
    _SERVICES = None


# ----------------------------------------------------------------------
# Deadlock probes (#34)
# ----------------------------------------------------------------------

def _agent_semaphore_probe() -> dict[str, Any]:
    """Running = permits currently held (agents actively working).

    A HELD permit is WORK IN PROGRESS, not a blocked resource: the old
    semantics (blocked=held, running=free) made a healthy 100%-utilized
    agent pool report ``running=0, blocked=N`` — which, combined with
    the deadlock detector, killed fully-busy sessions as "deadlocked".
    """
    from openharness.services.agent_concurrency import get_agent_limit, get_agent_semaphore

    limit = get_agent_limit()
    semaphore = get_agent_semaphore()
    held = limit - int(getattr(semaphore, "_value", limit))
    return {"name": "agent_semaphore", "blocked": 0, "running": held}


def _render_queue_probe() -> dict[str, Any]:
    """Running = busy render workers; blocked = queued jobs nobody works.

    A BUSY worker is WORK IN PROGRESS, not a blocked resource (old
    semantics made full utilization look deadlocked — see
    ``_agent_semaphore_probe``). A genuinely stuck queue — jobs waiting
    while EVERY worker is idle — still reports blocked>0/running=0.
    """
    from openharness.services.render_pipeline import _PIPELINE

    if _PIPELINE is None:
        return {"name": "render_queue", "blocked": 0, "running": 1}
    busy = sum(1 for state in _PIPELINE._worker_state.values() if state.get("busy"))
    idle = len(_PIPELINE._worker_state) - busy
    queued = _PIPELINE._queue.depth if _PIPELINE._queue is not None else 0
    stuck_pending = queued if (queued > 0 and busy == 0) else 0
    return {"name": "render_queue", "blocked": stuck_pending, "running": busy + idle}


# ----------------------------------------------------------------------
# Render pipeline progress hook (#38/#39 + watchdog feed)
# ----------------------------------------------------------------------

_RENDER_STAGE_MAP = {
    "queued": "queued",
    "rendering": "rendering",
    "rendered": "rendered",
    "render-failed": "failed",
    "promoted": "rendered",
}


def _render_progress_hook(scene: str, state: str) -> None:
    """Render pipeline -> progress tracker + control-bus watchdog feed."""
    from openharness.services.progress_tracker import get_progress_tracker

    try:
        stage = state.split(":", 1)[0]
        tracker_stage = _RENDER_STAGE_MAP.get(stage)
        if tracker_stage is not None:
            get_progress_tracker().mark(Path(scene).stem, tracker_stage)
    except Exception:  # progress must never break rendering
        pass
    services = get_session_services()
    if services.bus is not None:
        services.bus.feed(f"render:{scene}")


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------

def bootstrap_session_services(
    settings: Any,
    *,
    session_id: str,
    cwd: str | Path,
) -> SessionServices:
    """Configure every session-scoped service. Call before the engine runs.

    Must run inside the event loop that will drive the session (watchdog
    and deadlock monitors are asyncio tasks).
    """
    services = SessionServices(session_id=session_id)

    # 1. Audit log (#29) — first so every later action can be recorded.
    control = getattr(settings, "control", None)
    control_enabled = control is not None and bool(getattr(control, "enabled", False))
    audit_on = bool(getattr(control, "audit_log", False)) if control is not None else False
    if audit_on:
        services.audit = configure_audit_log_from_settings(settings, session_id=session_id)
    else:
        # Install a no-op audit log so control-bus/checkpoint code paths
        # do not silently write files the user disabled.
        set_audit_log(AuditLog(audit_dir=_null_audit_dir(), session_id=session_id, enabled=False))
    if services.audit is not None:
        set_audit_log(services.audit)

    # 2. Control bus (#26/#30/#34/#36).
    if control_enabled:
        bus = configure_control_bus_from_settings(settings)
        if bus is not None:
            set_control_bus(bus)
            services.bus = bus
            bus.register_deadlock_probe(_agent_semaphore_probe)
            bus.register_deadlock_probe(_render_queue_probe)
            if bool(getattr(control, "watchdog_timeout_seconds", 900)):
                bus.start_watchdog()
            if bool(getattr(control, "deadlock_detection", True)):
                bus.start_deadlock_monitor()
    else:
        # Control plane disabled for THIS session: clear any bus a
        # PREVIOUS session installed. Leaving it installed kept the old
        # monitor tasks looping and let get_control_bus() hand a dead
        # session's bus to the oh_control tool.
        set_control_bus(None)
        services.bus = None

    # 3. Cost tracker (#28) — halt trips the kill switch.
    if control_enabled and bool(getattr(control, "cost_alerts", True)):
        services.cost_tracker = SessionCostTracker(
            budget_usd=getattr(control, "cost_budget_usd", None),
            warn_ratio=float(getattr(control, "cost_warn_ratio", 0.8) or 0.8),
        )
        _tracker = services.cost_tracker
        _bus = services.bus

        def _on_cost_alert(alert: Any) -> None:
            get_audit_log().record_control(
                "cost_alert", {"level": alert.level, "message": alert.message}
            )
            if alert.level == "halt" and _bus is not None:
                _bus.trip(f"cost budget exceeded: {alert.message}")

        services.cost_tracker.on_alert = _on_cost_alert

    # 4. Model cache (#20) — must run before transformers/huggingface_hub
    #    imports so the redirected HF_HOME takes effect.
    try:
        from openharness.services.model_cache import configure_model_cache_from_settings

        configure_model_cache_from_settings(settings)
    except Exception as exc:  # never block the session on cache setup
        log.debug("model cache bootstrap skipped: %s", exc)

    # 5. Render pipeline singleton with hooks (#11-#16 + #38/#39 + #36 feed).
    try:
        from openharness.services.render_pipeline import get_render_pipeline

        get_render_pipeline(Path(cwd), settings)  # consumes settings on first create
        from openharness.services import render_pipeline as _rp

        if _rp._PIPELINE is not None and _rp._PIPELINE.progress_hook is None:
            _rp._PIPELINE.progress_hook = _render_progress_hook
        if _rp._PIPELINE is not None and _rp._PIPELINE.audit_hook is None and services.audit is not None:
            _rp._PIPELINE.audit_hook = (
                lambda event, detail: get_audit_log().record(f"render.{event}", detail)
            )
    except Exception as exc:
        log.debug("render pipeline bootstrap skipped: %s", exc)

    # 6a. Graceful shutdown hooks (#32).
    if services.bus is not None:
        services.bus.register_shutdown_hook(_shutdown_render_pipeline)
        services.bus.register_shutdown_hook(_close_persistent_store)
        services.bus.register_shutdown_hook(_final_audit_entry)
        # 6b. SIGTERM -> kill switch (abort at the next turn boundary).
        services.signal_handlers_installed = _install_sigterm_handler(services.bus)

    global _SERVICES
    _SERVICES = services
    log.info(
        "session services ready (control=%s audit=%s cost=%s)",
        services.bus is not None,
        services.audit is not None,
        services.cost_tracker is not None,
    )
    return services


def _null_audit_dir() -> Path:
    """A throwaway directory for the disabled audit log (never written)."""
    from openharness.config.paths import get_data_dir

    return get_data_dir() / "audit-disabled"


def _install_sigterm_handler(bus: ControlBus) -> bool:
    """Trip the kill switch on SIGTERM (#32 + #26). Best effort."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    try:
        loop.add_signal_handler(signal.SIGTERM, bus.trip, "SIGTERM received")
    except (NotImplementedError, RuntimeError, ValueError, OSError) as exc:
        # Windows or non-main-thread — the explicit shutdown path still works.
        log.debug("SIGTERM handler not installed: %s", exc)
        return False
    return True


async def _shutdown_render_pipeline() -> None:
    """Drain queued renders and stop the workers (#32)."""
    try:
        from openharness.services.render_pipeline import _PIPELINE

        if _PIPELINE is not None:
            await _PIPELINE.shutdown(drain=True)
    except Exception as exc:
        log.debug("render pipeline shutdown: %s", exc)


async def _close_persistent_store() -> None:
    """Close the SQLite store cleanly so WAL frames are checkpointed."""
    try:
        from openharness.services.persistent_store import _STORE

        if _STORE is not None:
            _STORE.close()
    except Exception as exc:
        log.debug("persistent store shutdown: %s", exc)


async def _final_audit_entry() -> None:
    audit = get_audit_log()
    audit.record("session.close", {"pid": __import__("os").getpid()})
    # Close the persistent append handle so the file is released cleanly.
    audit.close()
    # Flush the cost dashboard's in-memory aggregates to the persistent
    # store so a session kill loses at most the last flush interval.
    try:
        from openharness.services.cost_dashboard import _DASHBOARD

        if _DASHBOARD is not None:
            _DASHBOARD.flush(force=True)
    except Exception as exc:
        log.debug("cost dashboard flush: %s", exc)


async def shutdown_session_services() -> None:
    """Graceful shutdown (#32): drain in-flight work, release resources."""
    services = get_session_services()
    if services._shutdown_done:
        return
    services._shutdown_done = True
    bus = services.bus
    if bus is not None:
        await bus.begin_shutdown(reason="runtime close")
    else:
        # No control bus (features off) — still drain the render pipeline
        # and close the store so no background work leaks.
        await _shutdown_render_pipeline()
        await _close_persistent_store()
    reset_session_services()
