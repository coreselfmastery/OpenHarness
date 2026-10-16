"""Tests for all 15 safety and operational control items."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ============================================================
# #1 Kill Switch
# ============================================================

from openharness.services.safety_controls import KillSwitch, get_kill_switch, reset_kill_switch


def test_item_1_kill_switch_starts_running():
    ks = KillSwitch()
    assert not ks.is_killed()
    assert ks.state.value == "running"


def test_item_1_kill_switch_triggers():
    ks = KillSwitch()
    ks.kill("GPU overheating")
    assert ks.is_killed()
    assert ks.reason == "GPU overheating"
    assert ks.killed_at is not None


def test_item_1_kill_switch_resets():
    ks = KillSwitch()
    ks.kill("test")
    assert ks.is_killed()
    ks.reset()
    assert not ks.is_killed()
    assert ks.reason == ""


def test_item_1_kill_switch_fires_listeners():
    ks = KillSwitch()
    fired = []
    ks.add_listener(lambda reason: fired.append(reason))
    ks.kill("overheating")
    assert fired == ["overheating"]


def test_item_1_singleton_kill_switch():
    reset_kill_switch()
    ks1 = get_kill_switch()
    ks2 = get_kill_switch()
    assert ks1 is ks2


# ============================================================
# #2 Time Limits Per Stage
# ============================================================

from openharness.services.safety_controls import StageTimeLimits, StageTimeoutError


def test_item_2_stage_time_limit_passes_when_fast():
    limits = StageTimeLimits()
    limits.set("test_stage", max_seconds=10.0)
    with limits.track("test_stage"):
        pass  # instant — should not raise


def test_item_2_stage_time_limit_raises_on_timeout():
    limits = StageTimeLimits()
    limits.set("test_stage", max_seconds=0.01)  # 10ms
    with pytest.raises(StageTimeoutError):
        with limits.track("test_stage"):
            time.sleep(0.05)  # 50ms > 10ms limit


def test_item_2_remaining_time_decreases():
    limits = StageTimeLimits()
    limits.set("test_stage", max_seconds=5.0)
    with limits.track("test_stage"):
        remaining = limits.remaining("test_stage")
        assert 0 < remaining <= 5.0


def test_item_2_default_limits_exist():
    limits = StageTimeLimits()
    assert limits.get("planning") > 0
    assert limits.get("scene_generation") > 0
    assert limits.get("render") > 0


# ============================================================
# #3 Pause/Resume
# ============================================================

from openharness.services.safety_controls import PauseResume


def test_item_3_pause_resume_starts_unpaused():
    pr = PauseResume()
    assert not pr.is_paused


@pytest.mark.asyncio
async def test_item_3_check_pause_returns_immediately_when_not_paused():
    pr = PauseResume()
    await pr.check_pause()  # should return instantly


@pytest.mark.asyncio
async def test_item_3_pause_blocks_check_resume_unblocks():
    pr = PauseResume()
    pr.pause("inspecting scene 5")
    assert pr.is_paused

    # check_pause should block — launch a task to resume after 50ms
    async def resume_after_delay():
        await asyncio.sleep(0.05)
        pr.resume()

    asyncio.create_task(resume_after_delay())
    await pr.check_pause(timeout=2.0)  # should unblock after ~50ms
    assert not pr.is_paused


# ============================================================
# #4 Audit Log (imported from state_management, not safety_controls)
# ============================================================

from openharness.services.state_management import AuditLog


def test_item_4_audit_log_records_action():
    from openharness.services.state_management import AuditLog
    audit = AuditLog()
    audit.log(action="scene_generated", scene_id="scene_001", result="success")
    entries = audit.query(scene_id="scene_001")
    assert len(entries) == 1
    assert entries[0].action == "scene_generated"


def test_item_4_audit_log_records_errors():
    from openharness.services.state_management import AuditLog
    audit = AuditLog()
    audit.log(action="tool_call", tool_name="bash", is_error=True, result="command failed")
    errors = audit.query(is_error=True)
    assert len(errors) == 1
    assert errors[0].is_error


def test_item_4_audit_log_queries_by_action():
    from openharness.services.state_management import AuditLog
    audit = AuditLog()
    audit.log(action="scene_generated", scene_id="s1")
    audit.log(action="qa_pass", scene_id="s1")
    audit.log(action="scene_generated", scene_id="s2")
    entries = audit.query(action="scene_generated")
    assert len(entries) == 2


def test_item_4_audit_log_persists_to_db(tmp_path: Path):
    from openharness.services.state_management import AuditLog
    db_path = tmp_path / "audit.db"
    audit1 = AuditLog(persist_path=db_path)
    audit1.log(action="test_action", scene_id="s1", result="ok")
    # New instance loads from DB
    audit2 = AuditLog(persist_path=db_path)
    entries = audit2.query(action="test_action")
    assert len(entries) == 1


# ============================================================
# #5 Conflict Resolution
# ============================================================

from openharness.services.state_management import FileConflictResolver, FileConflictError


def test_item_5_acquire_lock_succeeds():
    resolver = FileConflictResolver()
    assert resolver.acquire("scene_001.js", agent_id="agent_1")


def test_item_5_second_agent_blocked():
    resolver = FileConflictResolver()
    resolver.acquire("scene_001.js", agent_id="agent_1")
    with pytest.raises(FileConflictError):
        resolver.acquire("scene_001.js", agent_id="agent_2")


def test_item_5_same_agent_can_reacquire():
    resolver = FileConflictResolver()
    resolver.acquire("scene_001.js", agent_id="agent_1")
    # Same agent can reacquire
    assert resolver.acquire("scene_001.js", agent_id="agent_1")


def test_item_5_release_unlocks():
    resolver = FileConflictResolver()
    resolver.acquire("scene_001.js", agent_id="agent_1")
    resolver.release("scene_001.js")
    # Now another agent can acquire
    assert resolver.acquire("scene_001.js", agent_id="agent_2")


def test_item_5_is_locked_and_who_owns():
    resolver = FileConflictResolver()
    resolver.acquire("scene_001.js", agent_id="agent_1")
    assert resolver.is_locked("scene_001.js")
    assert resolver.who_owns("scene_001.js") == "agent_1"


def test_item_5_force_release():
    resolver = FileConflictResolver()
    resolver.acquire("scene_001.js", agent_id="agent_1")
    resolver.force_release("scene_001.js")
    assert not resolver.is_locked("scene_001.js")


# ============================================================
# #6 Graceful Shutdown
# ============================================================

from openharness.services.safety_controls import GracefulShutdown


def test_item_6_graceful_shutdown_starts_clean():
    gs = GracefulShutdown()
    assert not gs.should_shutdown()


def test_item_6_register_sets_flag():
    gs = GracefulShutdown()
    gs.register()
    assert gs._registered


def test_item_6_on_shutdown_callback():
    gs = GracefulShutdown()
    called = []
    gs.on_shutdown(lambda: called.append(True))
    gs._handle_signal(15, None)  # SIGTERM
    assert called == [True]
    assert gs.should_shutdown()


# ============================================================
# #7 State Validation
# ============================================================

from openharness.services.state_management import validate_project_state


def test_item_7_valid_state_passes(tmp_path: Path):
    (tmp_path / "project.json").write_text('{"scenes": []}')
    result = validate_project_state(tmp_path)
    assert result.valid
    assert len(result.errors) == 0


def test_item_7_invalid_json_fails(tmp_path: Path):
    (tmp_path / "project.json").write_text("{invalid json")
    result = validate_project_state(tmp_path)
    assert not result.valid
    assert any("invalid JSON" in e for e in result.errors)


def test_item_7_missing_manifest_warns(tmp_path: Path):
    result = validate_project_state(tmp_path)
    assert result.valid  # warnings don't fail validation
    assert any("project.json not found" in w for w in result.warnings)


# ============================================================
# #8 Cost Alerts
# ============================================================

from openharness.services.safety_controls import CostAlertManager


def test_item_8_alert_triggers_at_threshold():
    mgr = CostAlertManager(budget_usd=10.0)
    mgr.set_threshold(0.5, "Half budget used")
    triggered = mgr.check(spent_usd=5.5)  # 55% > 50%
    assert len(triggered) == 1
    assert triggered[0].threshold == 0.5


def test_item_8_alert_does_not_trigger_below_threshold():
    mgr = CostAlertManager(budget_usd=10.0)
    mgr.set_threshold(0.5, "Half budget used")
    triggered = mgr.check(spent_usd=3.0)  # 30% < 50%
    assert len(triggered) == 0


def test_item_8_alert_fires_only_once():
    mgr = CostAlertManager(budget_usd=10.0)
    mgr.set_threshold(0.5, "Half budget")
    mgr.check(spent_usd=6.0)  # triggers
    triggered = mgr.check(spent_usd=7.0)  # already triggered — no re-fire
    assert len(triggered) == 0


def test_item_8_multiple_thresholds():
    mgr = CostAlertManager(budget_usd=10.0)
    mgr.set_threshold(0.5, "50%")
    mgr.set_threshold(0.8, "80%")
    triggered = mgr.check(spent_usd=8.5)  # triggers both
    assert len(triggered) == 2


def test_item_8_callback_fires():
    mgr = CostAlertManager(budget_usd=10.0)
    mgr.set_threshold(0.5, "Half")
    fired = []
    mgr.on_alert(lambda alert: fired.append(alert.message))
    mgr.check(spent_usd=6.0)
    assert fired == ["Half"]


# ============================================================
# #9 Deadlock Detection
# ============================================================

from openharness.services.safety_controls import DeadlockDetector


def test_item_9_no_deadlock_with_varied_actions():
    detector = DeadlockDetector(max_repeats=3)
    detector.record(action="read_file", tool_input={"path": "a.js"})
    detector.record(action="grep", tool_input={"pattern": "foo"})
    detector.record(action="read_file", tool_input={"path": "b.js"})
    assert not detector.is_deadlocked()


def test_item_9_deadlock_detected_on_repeats():
    detector = DeadlockDetector(max_repeats=3)
    detector.record(action="read_file", tool_input={"path": "a.js"})
    detector.record(action="read_file", tool_input={"path": "a.js"})
    detector.record(action="read_file", tool_input={"path": "a.js"})
    assert detector.is_deadlocked()


def test_item_9_no_deadlock_with_different_inputs():
    detector = DeadlockDetector(max_repeats=3)
    detector.record(action="read_file", tool_input={"path": "a.js"})
    detector.record(action="read_file", tool_input={"path": "b.js"})
    detector.record(action="read_file", tool_input={"path": "c.js"})
    assert not detector.is_deadlocked()


# ============================================================
# #10 Manual Override
# ============================================================

from openharness.services.safety_controls import ManualOverride


def test_item_10_no_override_initially():
    mo = ManualOverride()
    assert not mo.has_override()


def test_item_10_queue_and_get_override():
    mo = ManualOverride()
    mo.queue_override("skip_scene", reason="Scene 5 is broken")
    assert mo.has_override()
    decision, reason = mo.get_override()
    assert decision == "skip_scene"
    assert reason == "Scene 5 is broken"


def test_item_10_get_returns_none_when_empty():
    mo = ManualOverride()
    assert mo.get_override() is None


def test_item_10_multiple_overrides_queued():
    mo = ManualOverride()
    mo.queue_override("skip_scene")
    mo.queue_override("use_red")
    assert mo.has_override()
    d1, _ = mo.get_override()
    d2, _ = mo.get_override()
    assert d1 == "skip_scene"
    assert d2 == "use_red"
    assert not mo.has_override()


# ============================================================
# #11 Agent Handoff
# ============================================================

from openharness.services.state_management import AgentHandoff


def test_item_11_create_handoff():
    ho = AgentHandoff()
    handoff = ho.create(
        from_agent="agent_1",
        to_agent="agent_2",
        task="Generate scenes 10-20",
        context={"done": ["s1", "s2"]},
        reason="Context window full",
    )
    assert handoff["from_agent"] == "agent_1"
    assert handoff["to_agent"] == "agent_2"
    assert handoff["task"] == "Generate scenes 10-20"


def test_item_11_accept_handoff():
    ho = AgentHandoff()
    ho.create(from_agent="a1", to_agent="a2", task="test task")
    accepted = ho.accept("a2")
    assert accepted is not None
    assert accepted["accepted"] is True
    assert accepted["task"] == "test task"


def test_item_11_pending_for_specific_agent():
    ho = AgentHandoff()
    ho.create(from_agent="a1", to_agent="a2", task="task1")
    ho.create(from_agent="a1", to_agent="a3", task="task2")
    pending_for_a2 = ho.pending_for("a2")
    assert len(pending_for_a2) == 1
    assert pending_for_a2[0]["task"] == "task1"


def test_item_11_accept_returns_none_for_wrong_agent():
    ho = AgentHandoff()
    ho.create(from_agent="a1", to_agent="a2", task="task")
    assert ho.accept("a3") is None  # wrong agent


# ============================================================
# #12 Rollback to Checkpoint
# ============================================================

from openharness.services.state_management import CheckpointManager


def test_item_12_save_and_rollback(tmp_path: Path):
    # Create a project with a file
    (tmp_path / "scene.js").write_text("original code", encoding="utf-8")
    mgr = CheckpointManager(project_root=tmp_path)
    # Save checkpoint
    cp = mgr.save(description="before revision")
    assert cp.checkpoint_id.startswith("cp_")
    # Modify the file
    (tmp_path / "scene.js").write_text("modified code", encoding="utf-8")
    # Roll back
    assert mgr.rollback(cp.checkpoint_id)
    # File should be restored
    assert (tmp_path / "scene.js").read_text() == "original code"


def test_item_12_list_checkpoints(tmp_path: Path):
    (tmp_path / "scene.js").write_text("code", encoding="utf-8")
    mgr = CheckpointManager(project_root=tmp_path)
    mgr.save(description="cp1")
    mgr.save(description="cp2")
    checkpoints = mgr.list_checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints[0].description == "cp1"
    assert checkpoints[1].description == "cp2"


def test_item_12_delete_checkpoint(tmp_path: Path):
    (tmp_path / "scene.js").write_text("code", encoding="utf-8")
    mgr = CheckpointManager(project_root=tmp_path)
    cp = mgr.save(description="to delete")
    mgr.delete(cp.checkpoint_id)
    assert cp.checkpoint_id not in mgr._checkpoints


def test_item_12_rollback_nonexistent_returns_false(tmp_path: Path):
    mgr = CheckpointManager(project_root=tmp_path)
    assert not mgr.rollback("nonexistent")


# ============================================================
# #13 Progress Tracking
# ============================================================

from openharness.services.monitoring import ProgressTracker


def test_item_13_progress_tracker_tracks_stages():
    tracker = ProgressTracker()
    tracker.add_stage("scene_gen", total=100)
    tracker.add_stage("qa", total=100)
    tracker.update("scene_gen", completed=50)
    snapshot = tracker.snapshot()
    assert snapshot.overall_percent == 25.0  # 50 of 200 total


def test_item_13_progress_tracker_increments():
    tracker = ProgressTracker()
    tracker.add_stage("render", total=10)
    tracker.update("render", increment_completed=True)
    tracker.update("render", increment_completed=True)
    snapshot = tracker.snapshot()
    assert snapshot.stages["render"].completed == 2


def test_item_13_progress_tracker_estimates_remaining():
    tracker = ProgressTracker()
    tracker.add_stage("gen", total=100)
    tracker.update("gen", completed=50)
    snapshot = tracker.snapshot()
    assert snapshot.estimated_remaining_seconds is not None


def test_item_13_progress_summary():
    tracker = ProgressTracker()
    tracker.add_stage("gen", total=10)
    tracker.update("gen", completed=5)
    summary = tracker.summary()
    assert "overall_percent" in summary
    assert "stages" in summary
    assert summary["stages"]["gen"]["completed"] == 5


# ============================================================
# #14 Scene Status Tracking
# ============================================================

from openharness.services.state_management import SceneStatusTracker, SceneStatus


def test_item_14_set_and_get_status(tmp_path: Path):
    tracker = SceneStatusTracker(project_root=tmp_path)
    tracker.set("scene_001", SceneStatus.GENERATED)
    record = tracker.get("scene_001")
    assert record is not None
    assert record.status == SceneStatus.GENERATED


def test_item_14_get_by_status(tmp_path: Path):
    tracker = SceneStatusTracker(project_root=tmp_path)
    tracker.set("s1", SceneStatus.GENERATED)
    tracker.set("s2", SceneStatus.FAILED)
    tracker.set("s3", SceneStatus.GENERATED)
    generated = tracker.get_by_status(SceneStatus.GENERATED)
    assert len(generated) == 2


def test_item_14_status_counts(tmp_path: Path):
    tracker = SceneStatusTracker(project_root=tmp_path)
    tracker.set("s1", SceneStatus.GENERATED)
    tracker.set("s2", SceneStatus.FAILED)
    tracker.set("s3", SceneStatus.RENDERED)
    counts = tracker.get_status_counts()
    assert counts.get("generated") == 1
    assert counts.get("failed") == 1
    assert counts.get("rendered") == 1


def test_item_14_persists_to_disk(tmp_path: Path):
    tracker1 = SceneStatusTracker(project_root=tmp_path)
    tracker1.set("s1", SceneStatus.GENERATED)
    # New instance loads from disk
    tracker2 = SceneStatusTracker(project_root=tmp_path)
    record = tracker2.get("s1")
    assert record is not None
    assert record.status == SceneStatus.GENERATED


def test_item_14_summary(tmp_path: Path):
    tracker = SceneStatusTracker(project_root=tmp_path)
    tracker.set("s1", SceneStatus.GENERATED)
    tracker.set("s2", SceneStatus.FAILED)
    tracker.set("s3", SceneStatus.RENDERED)
    summary = tracker.summary()
    assert summary["total"] == 3
    assert summary["completed"] == 1  # RENDERED only (QA_PASSED also counts)
    assert summary["failed"] == 1


# ============================================================
# #15 Render Queue Monitoring
# ============================================================

from openharness.services.monitoring import RenderQueueMonitor


@pytest.mark.asyncio
async def test_item_15_monitor_reports_queue_status():
    from openharness.services.render_queue import RenderQueue, RenderJob, RenderState, create_render_queue
    queue = create_render_queue(max_concurrent=2)
    await queue.start()
    try:
        job1 = RenderJob(scene_id="s1", scene_file="s1.js", output_path="o1.mp4")
        job2 = RenderJob(scene_id="s2", scene_file="s2.js", output_path="o2.mp4")
        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.wait_all(timeout=5.0)
        monitor = RenderQueueMonitor(queue)
        status = monitor.status()
        assert status.total_jobs == 2
        assert status.completed == 2
        assert status.queued == 0
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_item_15_monitor_summary_json():
    from openharness.services.render_queue import create_render_queue, RenderJob
    queue = create_render_queue(max_concurrent=1)
    await queue.start()
    try:
        await queue.enqueue(RenderJob(scene_id="s1", scene_file="s.js", output_path="o.mp4"))
        await queue.wait_all(timeout=5.0)
        monitor = RenderQueueMonitor(queue)
        summary = monitor.summary()
        assert "total_jobs" in summary
        assert "completed" in summary
        assert summary["total_jobs"] == 1
    finally:
        await queue.stop()
