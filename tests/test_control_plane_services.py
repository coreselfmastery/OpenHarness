"""Real tests for the waves 2-6 control plane (items #26-#40).

Every test exercises the real modules — real ControlBus, real AuditLog,
real CheckpointManager, real file locks, a real QueryEngine driven by a
scripted streaming client — against an isolated data directory.

Sections:
1. Audit log (#29)
2. Checkpoints (#37)
3. Control bus (#26/#27/#30/#32/#34/#35/#36)
4. session_services bootstrap (the wiring hub)
5. Query-loop integration (kill switch, cost halt, pause, injection,
   tool timeouts, audit records)
6. State validator (#33)
7. Conflict resolution + file-tool locking (#31)
8. Progress tracker (#38/#39) + the oh_control / checkpoint / progress tools
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest, ApiTextDeltaEvent
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import ControlSettings, PermissionSettings, Settings
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import ErrorEvent, StatusEvent
from openharness.permissions.checker import build_permission_checker
from openharness.permissions.modes import PermissionMode
from openharness.tools.base import (
    BaseTool,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)


# ======================================================================
# Isolation: every test gets its own data dir + fresh singletons
# ======================================================================

@pytest.fixture(autouse=True)
def _isolated_services(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("HF_HOME", raising=False)
    from openharness.services import (
        audit_log as audit_mod,
        control_bus as bus_mod,
        persistent_store as store_mod,
        progress_tracker as tracker_mod,
    )
    from openharness.services import render_pipeline as render_mod
    from openharness.services.session_services import reset_session_services

    def _reset():
        audit_mod.set_audit_log(None)
        bus_mod.set_control_bus(None)
        store_mod.set_persistent_store(None)
        tracker_mod.set_progress_tracker(None)
        render_mod._PIPELINE = None
        reset_session_services()

    _reset()
    yield
    _reset()


def _settings(**control_overrides) -> Settings:
    kwargs: dict = {
        "permission": PermissionSettings(mode=PermissionMode.FULL_AUTO),
        "auto_save": False,
    }
    if control_overrides:
        kwargs["control"] = ControlSettings(**control_overrides)
    return Settings(**kwargs)


# ======================================================================
# 1. Audit log (#29)
# ======================================================================

def test_audit_log_writes_jsonl_and_tails_back(tmp_path: Path):
    from openharness.services.audit_log import AuditLog

    audit = AuditLog(tmp_path / "audit", session_id="sess-1")
    assert audit.available
    audit.record("tool_call", {"tool": "read_file", "is_error": False})
    audit.record("llm_call", {"model": "claude-test", "input_tokens": 10})
    assert audit.entries_written == 2
    entries = audit.tail(limit=10)
    assert len(entries) == 2
    assert entries[0]["kind"] == "tool_call"
    assert entries[0]["session"] == "sess-1"
    assert entries[1]["detail"]["model"] == "claude-test"
    # Each line is a single JSON object on disk.
    files = list((tmp_path / "audit").glob("audit-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "tool_call"


def test_audit_log_disabled_writes_nothing(tmp_path: Path):
    from openharness.services.audit_log import AuditLog

    audit = AuditLog(tmp_path / "audit", session_id="s", enabled=False)
    assert not audit.available
    assert not audit.enabled
    audit.record("tool_call", {"tool": "x"})
    audit.record_tool_call(tool="x", is_error=False, duration_seconds=0.1)
    assert audit.entries_written == 0
    assert not list((tmp_path / "audit").glob("*.jsonl"))


def test_audit_log_record_helpers_shape(tmp_path: Path):
    from openharness.services.audit_log import AuditLog

    audit = AuditLog(tmp_path / "audit", session_id="s")
    audit.record_tool_call(tool="write_file", is_error=True, duration_seconds=1.5)
    audit.record_llm_call(
        model="claude-sonnet",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=25,
        duration_seconds=0.4,
    )
    audit.record_control("kill", {"reason": "test"})
    tool, llm, control = audit.tail(limit=10)
    assert tool["kind"] == "tool_call" and tool["level"] == "warn"
    assert tool["detail"]["tool"] == "write_file" and tool["detail"]["is_error"] is True
    assert llm["detail"]["cache_read_tokens"] == 25
    assert control["kind"] == "control.kill"


def test_audit_log_bounds_huge_details(tmp_path: Path):
    from openharness.services.audit_log import AuditLog, _MAX_ENTRY_DETAIL_CHARS

    audit = AuditLog(tmp_path / "audit", session_id="s")
    huge = "x" * (_MAX_ENTRY_DETAIL_CHARS * 3)
    audit.record("tool_call", {"output": huge})
    (detail,) = audit.tail(limit=1)
    assert len(detail["detail"]["output"]) <= _MAX_ENTRY_DETAIL_CHARS


def test_audit_log_prune_removes_old_files(tmp_path: Path):
    from openharness.services.audit_log import AuditLog

    audit = AuditLog(tmp_path / "audit", session_id="s")
    audit.record("tool_call", {"tool": "x"})
    files = list((tmp_path / "audit").glob("audit-*.jsonl"))
    assert files
    # Age the file beyond retention.
    old = time.time() - 20 * 86_400
    os.utime(files[0], (old, old))
    assert audit.prune(retention_days=14) == 1
    assert not list((tmp_path / "audit").glob("audit-*.jsonl"))


def test_configure_audit_log_from_settings(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from openharness.services.audit_log import configure_audit_log_from_settings

    off = configure_audit_log_from_settings(Settings(control=ControlSettings(audit_log=False)))
    assert off is None
    on = configure_audit_log_from_settings(
        Settings(control=ControlSettings(audit_log=True)), session_id="s9"
    )
    assert on is not None and on.available
    entries = on.tail(limit=5)
    assert entries and entries[0]["kind"] == "audit.start"


# ======================================================================
# 2. Checkpoints (#37)
# ======================================================================

def _cp_messages(*texts: str) -> list[ConversationMessage]:
    return [ConversationMessage(role="user", content=[TextBlock(text=t)]) for t in texts]


def test_checkpoint_create_list_restore_diff(tmp_path: Path):
    from openharness.services.checkpoints import CheckpointManager

    manager = CheckpointManager("sess-cp")
    messages = _cp_messages("first", "second")
    summary = manager.create(messages, label="before refactor")
    assert summary["label"] == "before refactor"
    assert summary["message_count"] == 2
    assert summary["last_user_text"].startswith("second")

    listed = manager.list()
    assert len(listed) == 1 and listed[0]["id"] == summary["id"]

    payload = manager.restore(summary["id"])
    assert payload is not None
    assert payload["message_count"] == 2
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["content"][0]["text"] == "first"

    # Grow the conversation, checkpoint again, diff.
    time.sleep(0.01)  # distinct checkpoint ids (second-resolution + ms)
    more = _cp_messages("first", "second", "third")
    second = manager.create(more, label="after")
    diff = manager.diff(summary["id"], second["id"])
    assert diff is not None
    assert diff["message_delta"] == 1
    assert diff["different"] is True

    assert manager.restore("cp-does-not-exist") is None


def test_checkpoint_prune_keeps_newest(tmp_path: Path):
    from openharness.services.checkpoints import CheckpointManager

    manager = CheckpointManager("sess-prune")
    for index in range(5):
        manager.create(_cp_messages(f"m{index}"), label=f"cp{index}")
        time.sleep(0.005)  # distinct ids (ms suffix)
    removed = manager.prune(keep=2)
    assert removed == 3
    assert len(manager.list()) == 2


def test_checkpoint_metadata_drops_live_objects(tmp_path: Path):
    from openharness.services.checkpoints import _persistable_metadata

    filtered = _persistable_metadata(
        {"mcp_manager": object(), "settings": object(), "session_id": "sid", "n": 3}
    )
    assert "mcp_manager" not in filtered and "settings" not in filtered
    assert filtered == {"session_id": "sid", "n": 3}


# ======================================================================
# 3. Control bus (#26/#27/#30/#32/#34/#35/#36)
# ======================================================================

def test_kill_switch_trip_is_idempotent_and_first_reason_wins():
    from openharness.services.control_bus import ControlBus, KillSwitchTripped

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    assert not bus.killed
    bus.trip("watchdog fired")
    bus.trip("second reason is ignored")
    assert bus.killed
    assert bus.kill_reason == "watchdog fired"
    with pytest.raises(KillSwitchTripped):
        bus.check_killed()
    # The kill was audited.
    from openharness.services.audit_log import get_audit_log

    kills = get_audit_log().tail(limit=5, kind_prefix="control.kill")
    assert kills and kills[0]["detail"]["reason"] == "watchdog fired"


@pytest.mark.asyncio
async def test_pause_gate_blocks_until_resume():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    bus.pause("inspection")
    assert bus.paused

    async def _waiter():
        await bus.wait_if_paused()
        return "through"

    task = asyncio.create_task(_waiter())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.15)
    bus.resume()
    assert await asyncio.wait_for(task, timeout=1.0) == "through"
    assert not bus.paused


def test_inject_and_drain_messages():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    assert bus.inject_message("pivot to plan B") == 1
    assert bus.inject_message("  ") == 0  # blank ignored
    assert bus.inject_message("second note") == 2
    drained = bus.drain_injected_messages()
    assert drained == ["pivot to plan B", "second note"]
    assert bus.drain_injected_messages() == []
    assert bus.status()["injected_pending"] == 0


def test_feed_resets_watchdog_clock():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    before = bus.seconds_since_progress
    time.sleep(0.02)
    assert bus.seconds_since_progress >= before
    bus.feed("tool:read_file")
    assert bus.seconds_since_progress < 0.02


@pytest.mark.asyncio
async def test_call_with_limit_success_and_timeout():
    from openharness.services.control_bus import ControlBus, StageTimeoutExceeded

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    value = await bus.call_with_limit(asyncio.sleep(0, result=42), timeout_seconds=1, stage="t")
    assert value == 42
    with pytest.raises(StageTimeoutExceeded):
        await bus.call_with_limit(asyncio.sleep(5), timeout_seconds=0.1, stage="slow")
    # timeout<=0 passes through without a wrapper
    value = await bus.call_with_limit(asyncio.sleep(0, result="raw"), timeout_seconds=0, stage="x")
    assert value == "raw"


@pytest.mark.asyncio
async def test_watchdog_trips_after_timeout_without_progress():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0.4, deadlock_enabled=False)
    bus.start_watchdog()
    try:
        for _ in range(40):
            if bus.killed:
                break
            await asyncio.sleep(0.05)
        assert bus.killed
        assert "watchdog" in bus.kill_reason
    finally:
        # No public stop; the loop exits after tripping. Just assert done.
        pass


def test_deadlock_detection_requires_persistent_block():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_probe_seconds=0.0, deadlock_enabled=True)
    blocked_probe = lambda: {"name": "agents", "blocked": 2, "running": 0}
    bus.register_deadlock_probe(blocked_probe)
    # First check: everything blocked, but not yet "persisted" -> None
    # (probe_seconds=0 means it counts as persisted immediately).
    report = bus.detect_deadlock()
    assert report is not None
    assert report["deadlocked"] is True
    assert bus.deadlock_alerts == 1


def test_deadlock_detection_healthy_system_returns_none():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_probe_seconds=0, deadlock_enabled=True)
    bus.register_deadlock_probe(lambda: {"name": "agents", "blocked": 1, "running": 1})
    bus.register_deadlock_probe(lambda: {"name": "renders", "blocked": 0, "running": 2})
    assert bus.detect_deadlock() is None


def test_deadlock_detection_disabled_returns_none():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    bus.register_deadlock_probe(lambda: {"name": "agents", "blocked": 5, "running": 0})
    assert bus.detect_deadlock() is None


@pytest.mark.asyncio
async def test_begin_shutdown_runs_hooks_and_closes():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    order: list[str] = []

    async def _hook_a():
        order.append("a")

    async def _hook_b():
        order.append("b")

    bus.register_shutdown_hook(_hook_a)
    bus.register_shutdown_hook(_hook_b)
    assert bus.shutdown_phase == "running"
    await bus.begin_shutdown(reason="test")
    assert bus.shutdown_phase == "closed"
    assert order == ["a", "b"]
    # Idempotent: second call does not re-run hooks.
    await bus.begin_shutdown(reason="again")
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_shutdown_hook_failure_does_not_block_the_rest():
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    ran = []

    async def _boom():
        raise RuntimeError("hook exploded")

    async def _ok():
        ran.append("ok")

    bus.register_shutdown_hook(_boom)
    bus.register_shutdown_hook(_ok)
    await bus.begin_shutdown(reason="test")
    assert ran == ["ok"]
    assert bus.shutdown_phase == "closed"


def test_configure_control_bus_from_settings():
    from openharness.services.control_bus import configure_control_bus_from_settings

    assert configure_control_bus_from_settings(Settings(control=ControlSettings(enabled=False))) is None
    bus = configure_control_bus_from_settings(
        Settings(
            control=ControlSettings(
                enabled=True, watchdog_timeout_seconds=42, deadlock_probe_seconds=7
            )
        )
    )
    assert bus is not None
    assert bus.status()["watchdog_timeout_seconds"] == 42


# ======================================================================
# 4. session_services bootstrap (the wiring hub for #20, #26-#40)
# ======================================================================

@pytest.mark.asyncio
async def test_bootstrap_with_control_disabled_is_empty(tmp_path: Path):
    from openharness.services.session_services import (
        bootstrap_session_services,
        get_session_services,
    )

    # Control fully off: no bus, no cost tracker, no audit writes.
    settings = Settings(control=ControlSettings(enabled=False, audit_log=False))
    services = bootstrap_session_services(settings, session_id="s-off", cwd=tmp_path)
    assert services.bus is None
    assert services.audit is None
    assert services.cost_tracker is None
    # A disabled no-op audit log is installed so nothing writes files.
    from openharness.services.audit_log import get_audit_log

    assert get_audit_log().enabled is False
    # Identity: the shared record is the bootstrap result.
    assert get_session_services() is services


@pytest.mark.asyncio
async def test_bootstrap_audit_runs_even_with_control_disabled(tmp_path: Path):
    """ControlSettings docstring: ``enabled`` gates the invasive pieces
    only — audit logging is cheap and stays on when asked for."""
    from openharness.services.session_services import bootstrap_session_services

    settings = Settings(control=ControlSettings(enabled=False, audit_log=True))
    services = bootstrap_session_services(settings, session_id="s-audit", cwd=tmp_path)
    assert services.bus is None
    assert services.cost_tracker is None
    assert services.audit is not None and services.audit.available


@pytest.mark.asyncio
async def test_bootstrap_with_control_enabled_wires_everything(tmp_path: Path):
    from openharness.services import render_pipeline as render_mod
    from openharness.services.session_services import bootstrap_session_services

    settings = Settings(
        control=ControlSettings(
            enabled=True,
            audit_log=True,
            cost_alerts=True,
            cost_budget_usd=0.5,
            watchdog_timeout_seconds=0,  # no watchdog task in tests
            deadlock_detection=False,
        )
    )
    services = bootstrap_session_services(settings, session_id="s-on", cwd=tmp_path)
    assert services.bus is not None and not services.bus.killed
    assert services.audit is not None and services.audit.available
    assert services.cost_tracker is not None and services.cost_tracker.budget_usd == 0.5
    # Singletons installed for tools / query loop.
    from openharness.services.audit_log import get_audit_log
    from openharness.services.control_bus import get_control_bus

    assert get_control_bus() is services.bus
    assert get_audit_log() is services.audit
    # Deadlock probes registered (agent semaphore + render queue).
    assert len(services.bus._deadlock_probes) == 2
    # Render pipeline singleton created with progress + audit hooks.
    assert render_mod._PIPELINE is not None
    assert render_mod._PIPELINE.progress_hook is not None
    # Graceful shutdown drains hooks and closes the bus.
    from openharness.services.session_services import shutdown_session_services

    await shutdown_session_services()
    assert services.bus.shutdown_phase == "closed"


@pytest.mark.asyncio
async def test_bootstrap_cost_halt_trips_kill_switch(tmp_path: Path):
    from openharness.services.session_services import bootstrap_session_services

    settings = Settings(
        control=ControlSettings(
            enabled=True, audit_log=True, cost_alerts=True, cost_budget_usd=0.001,
            watchdog_timeout_seconds=0, deadlock_detection=False,
        )
    )
    services = bootstrap_session_services(settings, session_id="s-cost", cwd=tmp_path)
    tracker = services.cost_tracker
    # One big call: way past the halt threshold.
    alert = tracker.record("claude-sonnet-4", UsageSnapshot(input_tokens=1_000_000))
    assert alert is not None and alert.level == "halt"
    assert services.bus.killed
    assert "cost budget" in services.bus.kill_reason
    # The halt was audited as a cost_alert control record.
    from openharness.services.audit_log import get_audit_log

    alerts = get_audit_log().tail(limit=5, kind_prefix="control.cost_alert")
    assert alerts and alerts[0]["detail"]["level"] == "halt"


# ======================================================================
# 5. Query-loop integration (kill switch, cost halt, pause, injection,
#    tool timeouts, audit records)
# ======================================================================

class _ScriptedClient:
    """Streaming client that plays back fixed assistant messages."""

    def __init__(self, messages: list[ConversationMessage]) -> None:
        self._messages = list(messages)
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        if not self._messages:
            return
        msg = self._messages.pop(0)
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                yield ApiTextDeltaEvent(text=block.text)
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1000, output_tokens=100),
            stop_reason=None,
        )


class _SlowTool(BaseTool):
    name = "slow_tool"
    description = "Sleeps before answering (timeout test)."

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        # Tiny schema so the query loop can validate tool inputs.
        from pydantic import BaseModel

        class _SlowInput(BaseModel):
            seconds: float = 0.0

        self.input_model = _SlowInput

    def is_read_only(self, arguments):
        return True

    async def execute(self, arguments, context: ToolExecutionContext) -> ToolResult:
        await asyncio.sleep(self.seconds)
        return ToolResult(output="finally done")


def _make_engine(
    *,
    cwd: Path,
    client: _ScriptedClient,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    bus=None,
    cost_tracker=None,
) -> QueryEngine:
    return QueryEngine(
        api_client=client,
        tool_registry=registry or ToolRegistry(),
        permission_checker=build_permission_checker(
            settings or _settings(), cwd=str(cwd)
        ),
        cwd=cwd,
        model="claude-test",
        system_prompt="test",
        settings=settings,
        control_bus=bus,
        session_cost_tracker=cost_tracker,
        tool_metadata={"session_id": "engine-test"},
    )


async def _collect(engine: QueryEngine, prompt: str = "hello"):
    events = []
    async for event in engine.submit_message(prompt):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_kill_switch_aborts_query_loop_at_turn_boundary(tmp_path: Path):
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    bus.trip("user pressed ctrl-c")
    engine = _make_engine(
        cwd=tmp_path,
        client=_ScriptedClient(
            [ConversationMessage(role="assistant", content=[TextBlock(text="hi")])]
        ),
        bus=bus,
    )
    events = await _collect(engine)
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors and "kill switch" in errors[0].message
    # The scripted client was never called — the loop aborted first.
    assert engine.api_client.calls == 0


@pytest.mark.asyncio
async def test_cost_halt_aborts_query_loop(tmp_path: Path):
    from openharness.services.llm_pipeline import SessionCostTracker

    # Each scripted turn costs ~$0.0045 at default pricing (1k in / 100 out).
    # Budget $0.005: turn 1 (tool call) crosses the warn threshold, turn 2 halts.
    class _NoopTool(BaseTool):
        name = "noop_tool"
        description = "No-op."

        def __init__(self):
            from pydantic import BaseModel

            class _NoopInput(BaseModel):
                pass

            self.input_model = _NoopInput

        def is_read_only(self, arguments):
            return True

        async def execute(self, arguments, context):
            return ToolResult(output="ok")

    registry = ToolRegistry()
    registry.register(_NoopTool())
    tracker = SessionCostTracker(budget_usd=0.005, warn_ratio=0.5)
    engine = _make_engine(
        cwd=tmp_path,
        client=_ScriptedClient(
            [
                ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(id="tu_1", name="noop_tool", input={})],
                ),
                ConversationMessage(role="assistant", content=[TextBlock(text="part 2")]),
            ]
        ),
        registry=registry,
        cost_tracker=tracker,
    )
    events = await _collect(engine)
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors and "cost budget exceeded" in errors[0].message
    assert tracker.halted
    # The warning fired first (as a status event), then the halt.
    statuses = [e.message for e in events if isinstance(e, StatusEvent)]
    assert any("Cost warning" in m for m in statuses)
    assert tracker.stats()["calls"] == 2


@pytest.mark.asyncio
async def test_injected_message_appended_at_turn_boundary(tmp_path: Path):
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    bus.inject_message("pivot: use approach B instead")
    engine = _make_engine(
        cwd=tmp_path,
        client=_ScriptedClient(
            [ConversationMessage(role="assistant", content=[TextBlock(text="ok")])]
        ),
        bus=bus,
    )
    events = await _collect(engine)
    statuses = [e.message for e in events if isinstance(e, StatusEvent)]
    assert any("injected human message" in m for m in statuses)
    texts = [m.text for m in engine.messages]
    assert any("use approach B" in t for t in texts)


@pytest.mark.asyncio
async def test_pause_blocks_query_loop_until_resume(tmp_path: Path):
    from openharness.services.control_bus import ControlBus

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    engine = _make_engine(
        cwd=tmp_path,
        client=_ScriptedClient(
            [ConversationMessage(role="assistant", content=[TextBlock(text="done")])]
        ),
        bus=bus,
    )
    bus.pause("user wants to inspect")
    task = asyncio.create_task(_collect(engine))
    await asyncio.sleep(0.25)
    assert not task.done(), "query loop should be blocked at the pause gate"
    assert engine.api_client.calls == 0
    bus.resume()
    events = await asyncio.wait_for(task, timeout=5.0)
    assert engine.api_client.calls == 1
    assert not any(isinstance(e, ErrorEvent) for e in events)


@pytest.mark.asyncio
async def test_tool_timeout_turns_into_error_result(tmp_path: Path):
    from openharness.services.control_bus import ControlBus

    registry = ToolRegistry()
    registry.register(_SlowTool(seconds=3.0))
    settings = Settings(
        control=ControlSettings(
            enabled=True,
            stage_time_limits=True,
            tool_timeout_seconds=1,  # wait_for granularity: fires well before 3s
            checkpoints=False,
            audit_log=True,
        ),
        permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
        auto_save=False,
    )
    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    client = _ScriptedClient(
        [
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="slow_tool", input={})],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="all done")]),
        ]
    )
    engine = _make_engine(
        cwd=tmp_path, client=client, registry=registry, settings=settings, bus=bus
    )
    events = await _collect(engine)
    # The loop survived; the slow tool produced an ERROR RESULT (not a crash).
    assert not any(isinstance(e, ErrorEvent) for e in events)
    tool_results = [
        block
        for m in engine.messages
        if m.role == "user"
        for block in m.content
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_results and tool_results[0].is_error
    assert "aborted" in tool_results[0].content
    # The timeout fed the watchdog clock and was audited as a failed call.
    from openharness.services.audit_log import get_audit_log

    tool_calls = get_audit_log().tail(limit=20, kind_prefix="tool_call")
    slow = [e for e in tool_calls if e["detail"]["tool"] == "slow_tool"]
    assert slow and slow[0]["detail"]["is_error"] is True


@pytest.mark.asyncio
async def test_exempt_tool_is_not_killed_by_stage_timeout(tmp_path: Path):
    from openharness.engine.query import _TOOLS_WITH_OWN_TIME
    from openharness.services.control_bus import ControlBus

    assert "sleep" in _TOOLS_WITH_OWN_TIME

    class _SleepyTool(_SlowTool):
        name = "sleep"  # exempt: manages its own deadline

    registry = ToolRegistry()
    registry.register(_SleepyTool(seconds=1.2))
    settings = Settings(
        control=ControlSettings(
            enabled=True, stage_time_limits=True, tool_timeout_seconds=1,
            checkpoints=False, audit_log=False,
        ),
        permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
        auto_save=False,
    )
    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    client = _ScriptedClient(
        [
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="sleep", input={})],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
        ]
    )
    engine = _make_engine(
        cwd=tmp_path, client=client, registry=registry, settings=settings, bus=bus
    )
    events = await _collect(engine)
    assert not any(isinstance(e, ErrorEvent) for e in events)
    results = [
        block
        for m in engine.messages
        if m.role == "user"
        for block in m.content
        if isinstance(block, ToolResultBlock)
    ]
    assert results and not results[0].is_error
    assert results[0].content == "finally done"


@pytest.mark.asyncio
async def test_audit_log_records_llm_and_tool_calls_from_loop(tmp_path: Path):
    from openharness.services.audit_log import AuditLog, set_audit_log

    audit = AuditLog(enabled=True, session_id="audit-loop")
    set_audit_log(audit)
    registry = ToolRegistry()

    class _EchoTool(BaseTool):
        name = "echo_tool"
        description = "Echoes."

        def __init__(self):
            from pydantic import BaseModel

            class _EchoInput(BaseModel):
                text: str = ""

            self.input_model = _EchoInput

        def is_read_only(self, arguments):
            return True

        async def execute(self, arguments, context):
            return ToolResult(output="echo!")

    registry.register(_EchoTool())
    client = _ScriptedClient(
        [
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="echo_tool", input={})],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
        ]
    )
    engine = _make_engine(cwd=tmp_path, client=client, registry=registry)
    await _collect(engine)
    kinds = [e["kind"] for e in audit.tail(limit=30)]
    assert "llm_call" in kinds
    assert "tool_call" in kinds


@pytest.mark.asyncio
async def test_auto_checkpoint_snapshots_conversation(tmp_path: Path):
    from openharness.services.checkpoints import CheckpointManager

    settings = Settings(
        control=ControlSettings(
            enabled=True, stage_time_limits=False, checkpoints=True,
            checkpoint_interval_turns=1, audit_log=False,
        ),
        permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
        auto_save=False,
    )
    engine = _make_engine(
        cwd=tmp_path,
        client=_ScriptedClient(
            [ConversationMessage(role="assistant", content=[TextBlock(text="turn 1")])]
        ),
        settings=settings,
    )
    await _collect(engine)
    manager = CheckpointManager("engine-test")
    listed = manager.list()
    assert listed, "auto-checkpoint every turn should have created one"
    assert listed[0]["label"].startswith("auto turn")
    assert listed[0]["message_count"] >= 1


# ======================================================================
# 6. State validator (#33)
# ======================================================================

def _valid_conversation() -> list[ConversationMessage]:
    return [
        ConversationMessage(role="user", content=[TextBlock(text="make a video")]),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="tu_ok", name="write_file", input={})],
        ),
        ConversationMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id="tu_ok", content="wrote it")],
        ),
        ConversationMessage(role="assistant", content=[TextBlock(text="done!")]),
    ]


def test_validate_clean_conversation_is_noop():
    from openharness.services.state_validator import validate_and_repair

    messages = _valid_conversation()
    repaired, report = validate_and_repair(messages)
    assert not report.findings
    assert not report.repaired
    assert len(repaired) == len(messages)


def test_validate_drops_orphan_tool_results():
    from openharness.services.state_validator import validate_and_repair

    messages = _valid_conversation()
    messages.append(
        ConversationMessage(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="tu_ghost", content="result of a lost call"),
                TextBlock(text="and a crash later"),
            ],
        )
    )
    repaired, report = validate_and_repair(messages)
    assert report.repaired and report.dropped_orphan_results == 1
    assert any("orphan" in f for f in report.findings)
    # The non-tool text block survived; only tool plumbing was dropped.
    assert repaired[-1].text == "and a crash later"


def test_validate_drops_trailing_unanswered_tool_use():
    from openharness.services.state_validator import validate_and_repair

    messages = _valid_conversation()
    messages.append(
        ConversationMessage(
            role="assistant", content=[ToolUseBlock(id="tu_lost", name="bash", input={})]
        )
    )
    repaired, report = validate_and_repair(messages)
    assert report.repaired
    assert report.dropped_unanswered_tool_uses >= 1
    assert repaired[-1].role != "assistant" or not repaired[-1].tool_uses


def test_validate_killed_session_reports_but_keeps_messages():
    from openharness.services.state_validator import validate_and_repair

    messages = _valid_conversation()
    repaired, report = validate_and_repair(messages, session_was_killed=True)
    assert any("killed mid-task" in f for f in report.findings)
    assert len(repaired) == len(messages)  # nothing dropped, only reported


def test_validate_restored_payload_skips_corrupt_entries():
    from openharness.services.state_validator import validate_restored_payload

    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"totally": "not a message"},
            {"role": "user", "content": [{"type": "text", "text": "world"}]},
        ]
    }
    repaired, report = validate_restored_payload(payload)
    assert len(repaired) == 2
    assert any("failed to parse" in f for f in report.findings)
    assert report.repaired
    # Empty payload is explicit, not silent.
    empty, empty_report = validate_restored_payload(None)
    assert empty == [] and empty_report.findings == ["no session payload"]


def test_was_session_killed_reads_audit_log():
    from openharness.services.audit_log import AuditLog, set_audit_log
    from openharness.services.state_validator import was_session_killed

    audit = AuditLog(enabled=True, session_id="s-live")
    set_audit_log(audit)
    audit.record_control("kill", {"reason": "watchdog"})
    # tail() filters by kind prefix "control.kill" — write via record so
    # the session field matches the killed session.
    assert was_session_killed("s-live") is True
    assert was_session_killed("s-other") is False


# ======================================================================
# 7. Conflict resolution + file-tool locking (#31)
# ======================================================================

@pytest.mark.asyncio
async def test_in_process_lock_excludes_concurrent_writers(tmp_path: Path):
    from openharness.services.conflict_resolution import acquire_path_lock

    target = tmp_path / "scene_001.py"
    target.write_text("x = 1")
    active: list[str] = []
    violations: list[str] = []

    async def _writer(name: str) -> None:
        async with acquire_path_lock(target, timeout_seconds=5, holder_label=name):
            active.append(name)
            await asyncio.sleep(0.15)
            if len(active) != 1:
                violations.append(name)
            active.remove(name)

    await asyncio.gather(_writer("agent-a"), _writer("agent-b"))
    assert violations == [], "two writers held the lock simultaneously"


@pytest.mark.asyncio
async def test_external_lockfile_conflict_names_the_holder(tmp_path: Path):
    from openharness.services.conflict_resolution import (
        FileConflictError,
        _lock_path_for,
        acquire_path_lock,
    )

    target = tmp_path / "shared.py"
    target.write_text("x = 1")
    lock = _lock_path_for(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # A lock held by a LIVE process (our own pid) must be respected,
    # never stolen — the holder name is surfaced in the error.
    lock.write_text(f"agent (pid {os.getpid()})", encoding="utf-8")
    with pytest.raises(FileConflictError) as excinfo:
        async with acquire_path_lock(target, timeout_seconds=0.2, holder_label="me"):
            pass
    assert f"pid {os.getpid()}" in str(excinfo.value)
    # The foreign lock is respected, not stolen (live holder, fresh mtime).
    assert lock.exists()


@pytest.mark.asyncio
async def test_dead_holder_lockfile_is_stolen_instantly(tmp_path: Path):
    """A lockfile left by a crashed process is stolen without waiting out
    the stale-age window (pid liveness check)."""
    from openharness.services.conflict_resolution import _lock_path_for, acquire_path_lock

    target = tmp_path / "crashed.py"
    target.write_text("x = 1")
    lock = _lock_path_for(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Fresh mtime (age 0) but the holder pid is gone.
    lock.write_text("agent (pid 999999998)", encoding="utf-8")
    async with acquire_path_lock(target, timeout_seconds=1.0, stale_seconds=600.0):
        assert lock.exists()
        assert str(os.getpid()) in lock.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_stale_lockfile_is_stolen(tmp_path: Path):
    from openharness.services.conflict_resolution import _lock_path_for, acquire_path_lock

    target = tmp_path / "stale.py"
    target.write_text("x = 1")
    lock = _lock_path_for(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # No pid in the label (legacy format) — the mtime path must still steal.
    lock.write_text("legacy holder", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    async with acquire_path_lock(target, timeout_seconds=1, stale_seconds=600, holder_label="stealer"):
        # The stale lock was replaced by OUR lockfile (fresh label + pid).
        assert lock.exists()
        assert lock.read_text(encoding="utf-8").strip().startswith("stealer")


def test_configure_file_locking_from_settings():
    from openharness.services.conflict_resolution import (
        configure_file_locking_from_settings,
        lock_timeout_from_settings,
    )

    assert configure_file_locking_from_settings(None) is True  # default on
    off = Settings(control=ControlSettings(enabled=True, file_locking=False))
    assert configure_file_locking_from_settings(off) is False
    tuned = Settings(
        control=ControlSettings(enabled=True, file_locking=True, file_lock_timeout_seconds=5)
    )
    assert configure_file_locking_from_settings(tuned) is True
    assert lock_timeout_from_settings(tuned) == 5.0
    assert lock_timeout_from_settings(None) == 30.0


@pytest.mark.asyncio
async def test_write_file_tool_blocked_by_external_lock(tmp_path: Path):
    from openharness.services.conflict_resolution import _lock_path_for
    from openharness.tools.file_write_tool import FileWriteTool, FileWriteToolInput

    target = tmp_path / "protected.txt"
    target.write_text("original")
    lock = _lock_path_for(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("other agent", encoding="utf-8")
    settings = _settings()  # file_locking defaults on
    tool = FileWriteTool()
    result = await tool.execute(
        FileWriteToolInput(path=str(target), content="clobber attempt"),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings}),
    )
    assert result.is_error
    assert "Write blocked" in result.output
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_edit_file_tool_blocked_by_external_lock(tmp_path: Path):
    from openharness.services.conflict_resolution import _lock_path_for
    from openharness.tools.file_edit_tool import FileEditTool, FileEditToolInput

    target = tmp_path / "protected.py"
    target.write_text("alpha = 1\n")
    lock = _lock_path_for(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("other agent", encoding="utf-8")
    tool = FileEditTool()
    result = await tool.execute(
        FileEditToolInput(path=str(target), old_str="alpha", new_str="beta"),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": _settings()}),
    )
    assert result.is_error
    assert "Edit blocked" in result.output
    assert target.read_text(encoding="utf-8").startswith("alpha")


@pytest.mark.asyncio
async def test_write_and_edit_work_normally_with_locking_enabled(tmp_path: Path):
    from openharness.tools.file_edit_tool import FileEditTool, FileEditToolInput
    from openharness.tools.file_write_tool import FileWriteTool, FileWriteToolInput

    target = tmp_path / "free.txt"
    write = FileWriteTool()
    result = await write.execute(
        FileWriteToolInput(path=str(target), content="hello lock"),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": _settings()}),
    )
    assert not result.is_error and target.read_text(encoding="utf-8") == "hello lock"
    edit = FileEditTool()
    result = await edit.execute(
        FileEditToolInput(path=str(target), old_str="hello", new_str="hi"),
        ToolExecutionContext(cwd=tmp_path, metadata={"settings": _settings()}),
    )
    assert not result.is_error and target.read_text(encoding="utf-8") == "hi lock"
    # The lock file was released again.
    from openharness.services.conflict_resolution import _lock_path_for

    assert not _lock_path_for(target).exists()


# ======================================================================
# 8. Progress tracker (#38/#39) + control-plane tools
# ======================================================================

def test_progress_stage_machine_happy_path():
    from openharness.services.progress_tracker import get_progress_tracker

    tracker = get_progress_tracker()
    for stage in ("generating", "generated", "queued", "rendering", "rendered", "qa", "approved"):
        assert tracker.mark("scene_001", stage) is True
    assert tracker.scene_status("scene_001")["stage"] == "approved"


def test_progress_stage_machine_rejects_invalid_jumps():
    from openharness.services.progress_tracker import get_progress_tracker

    tracker = get_progress_tracker()
    assert tracker.mark("scene_x", "rendered") is False  # pending -> rendered invalid
    assert tracker.mark("scene_x", "generating") is True
    assert tracker.mark("scene_x", "pending") is False  # no going back
    assert tracker.mark("scene_x", "bogus_stage") is False  # unknown stage
    # Aborted is terminal.
    assert tracker.mark("scene_y", "aborted") is True
    assert tracker.mark("scene_y", "generating") is False
    # Retry after failure is allowed.
    assert tracker.mark("scene_z", "generating") is True
    assert tracker.mark("scene_z", "failed") is True
    assert tracker.mark("scene_z", "generating") is True


def test_progress_summary_and_by_stage():
    from openharness.services.progress_tracker import get_progress_tracker

    tracker = get_progress_tracker()
    tracker.mark("a", "generating")
    tracker.mark("b", "generating")
    tracker.mark("b", "generated")
    tracker.mark("c", "generating")
    tracker.mark("c", "generated")
    tracker.mark("c", "queued")
    tracker.mark("c", "rendering")
    tracker.mark("c", "rendered")
    tracker.mark("c", "qa")
    tracker.mark("c", "approved")
    summary = tracker.summary()
    assert summary["total_scenes"] == 3
    assert summary["approved"] == 1
    assert summary["completion"] == round(1 / 3, 3)
    assert tracker.by_stage("generated") == ["b"]
    assert summary["stage_counts"]["generating"] == 1


def test_progress_eta_needs_observed_durations():
    from openharness.services.progress_tracker import get_progress_tracker

    tracker = get_progress_tracker()
    tracker.mark("s", "generating")
    assert tracker.eta_seconds() is None  # nothing observed yet
    # Simulate one observed WORK cycle: the ETA model only reads the
    # work stages (generating/rendering/qa) — the passive waypoint
    # stages (generated/rendered/approved) are millisecond transitions
    # and must not produce an ETA by themselves.
    tracker._stage_durations["generated"].append(2.0)
    assert tracker.eta_seconds() is None  # waypoint durations don't count
    tracker._stage_durations["generating"].append(2.0)
    assert tracker.eta_seconds() is not None and tracker.eta_seconds() > 0
    # All-terminal -> zero.
    tracker.mark("s", "generated")
    tracker.mark("s", "queued")
    tracker.mark("s", "rendering")
    tracker.mark("s", "rendered")
    tracker.mark("s", "qa")
    tracker.mark("s", "approved")
    assert tracker.eta_seconds() == 0.0


def test_progress_force_sets_stage_unconditionally():
    from openharness.services.progress_tracker import get_progress_tracker

    tracker = get_progress_tracker()
    tracker.force("restored_scene", "rendered", detail="recovered from snapshot")
    assert tracker.scene_status("restored_scene")["stage"] == "rendered"


@pytest.mark.asyncio
async def test_control_tool_actions(tmp_path: Path):
    from openharness.services.control_bus import ControlBus, set_control_bus
    from openharness.tools.pipeline_tools import ControlInput, ControlTool

    bus = ControlBus(watchdog_timeout_seconds=0, deadlock_enabled=False)
    set_control_bus(bus)
    tool = ControlTool()
    ctx = ToolExecutionContext(cwd=tmp_path)

    result = await tool.execute(ControlInput(action="status"), ctx)
    assert not result.is_error and '"killed": false' in result.output

    result = await tool.execute(ControlInput(action="inject", message="human note"), ctx)
    assert not result.is_error
    assert bus.drain_injected_messages() == ["human note"]

    result = await tool.execute(ControlInput(action="kill", message="test kill"), ctx)
    assert not result.is_error
    assert bus.killed and bus.kill_reason == "test kill"

    result = await tool.execute(ControlInput(action="bogus"), ctx)
    assert result.is_error


@pytest.mark.asyncio
async def test_checkpoint_tool_create_list_restore(tmp_path: Path):
    from openharness.tools.pipeline_tools import (
        CheckpointInput,
        CheckpointTool,
    )

    class _FakeEngine:
        def __init__(self):
            self.messages = [
                ConversationMessage(role="user", content=[TextBlock(text="m1")]),
                ConversationMessage(role="user", content=[TextBlock(text="m2")]),
            ]
            self.rolled_back_to: list | None = None

        @property
        def live_messages(self):
            return self.messages

        def rollback_messages(self, messages):
            self.rolled_back_to = list(messages)

    engine = _FakeEngine()
    ctx = ToolExecutionContext(
        cwd=tmp_path, metadata={"session_id": "tool-cp", "engine_ref": engine}
    )
    tool = CheckpointTool()

    created = await tool.execute(CheckpointInput(action="create", label="before"), ctx)
    assert not created.is_error
    cp_id = created.metadata["id"]
    assert created.metadata["message_count"] == 2

    listed = await tool.execute(CheckpointInput(action="list"), ctx)
    assert cp_id in listed.output

    engine.messages.append(ConversationMessage(role="user", content=[TextBlock(text="m3")]))
    restored = await tool.execute(CheckpointInput(action="restore", checkpoint_id=cp_id), ctx)
    assert not restored.is_error
    assert "3 -> 2 messages" in restored.output
    assert engine.rolled_back_to is not None and len(engine.rolled_back_to) == 2

    diffed = await tool.execute(
        CheckpointInput(action="diff", checkpoint_id=cp_id, second_id=cp_id), ctx
    )
    assert not diffed.is_error and "identical: True" in diffed.output

    missing = await tool.execute(CheckpointInput(action="restore", checkpoint_id="nope"), ctx)
    assert missing.is_error


@pytest.mark.asyncio
async def test_progress_tool_reports_tracker_state(tmp_path: Path):
    from openharness.services.progress_tracker import get_progress_tracker
    from openharness.tools.pipeline_tools import ProgressInput, ProgressTool

    tracker = get_progress_tracker()
    tracker.mark("scene_001", "generating")
    tracker.mark("scene_002", "generating")
    tracker.mark("scene_002", "generated")
    tool = ProgressTool()
    ctx = ToolExecutionContext(cwd=tmp_path)

    summary = await tool.execute(ProgressInput(action="summary"), ctx)
    assert "scenes: 2" in summary.output
    assert summary.metadata["total_scenes"] == 2

    scene = await tool.execute(ProgressInput(action="scene", scene="scene_001"), ctx)
    assert '"stage": "generating"' in scene.output

    unknown = await tool.execute(ProgressInput(action="scene", scene="nope"), ctx)
    assert unknown.is_error
