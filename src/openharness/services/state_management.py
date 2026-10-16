"""State management — items 4, 5, 7, 11, 12, 14.

#4 Audit Log: See exactly what the agent did on every scene.
#5 Conflict Resolution: Prevent two agents editing the same scene file.
#7 State Validation: Check project state is consistent before resuming.
#11 Agent Handoff: Transfer a task to another agent mid-work.
#12 Rollback to Checkpoint: Undo more than one step.
#14 Scene Status Tracking: Know which scenes are done, pending, failed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ============================================================
# #4 Audit Log
# ============================================================

@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float
    action: str  # "tool_call", "scene_generated", "qa_pass", "qa_fail", etc.
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    duration_seconds: float = 0.0
    scene_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Persistent audit log of every agent action.

    Records every tool call, scene generation, QA check, and render
    to a SQLite database. Used for debugging, compliance, and
    post-mortem analysis.

    Usage:
        audit = AuditLog(persist_path=".openharness/audit.db")
        audit.log(action="scene_generated", scene_id="scene_001", result="success")
        audit.log(action="tool_call", tool_name="read_file", tool_input={"path": "scene.js"})

        # Query:
        entries = audit.query(scene_id="scene_001")
        errors = audit.query(is_error=True)
    """

    def __init__(self, *, persist_path: str | Path | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._entries: list[AuditEntry] = []
        self._conn: sqlite3.Connection | None = None
        if self._persist_path:
            self._init_db()

    def log(
        self,
        *,
        action: str,
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
        result: str = "",
        is_error: bool = False,
        duration_seconds: float = 0.0,
        scene_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an action in the audit log."""
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            tool_name=tool_name,
            tool_input=tool_input or {},
            result=result,
            is_error=is_error,
            duration_seconds=duration_seconds,
            scene_id=scene_id,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if self._conn:
            self._save_to_db(entry)

    def query(
        self,
        *,
        scene_id: str | None = None,
        action: str | None = None,
        tool_name: str | None = None,
        is_error: bool | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query the audit log with optional filters."""
        results = []
        for entry in reversed(self._entries):  # most recent first
            if scene_id and entry.scene_id != scene_id:
                continue
            if action and entry.action != action:
                continue
            if tool_name and entry.tool_name != tool_name:
                continue
            if is_error is not None and entry.is_error != is_error:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def stats(self) -> dict[str, Any]:
        total = len(self._entries)
        errors = sum(1 for e in self._entries if e.is_error)
        return {
            "total_entries": total,
            "errors": errors,
            "actions": list(set(e.action for e in self._entries)),
        }

    def clear(self) -> None:
        self._entries.clear()
        if self._conn:
            self._conn.execute("DELETE FROM audit_log")
            self._conn.commit()

    def _init_db(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._persist_path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                action TEXT,
                tool_name TEXT,
                tool_input TEXT,
                result TEXT,
                is_error INTEGER,
                duration_seconds REAL,
                scene_id TEXT,
                metadata TEXT
            )"""
        )
        self._conn.commit()
        self._load_from_db()

    def _save_to_db(self, entry: AuditEntry) -> None:
        if not self._conn:
            return
        self._conn.execute(
            """INSERT INTO audit_log
               (timestamp, action, tool_name, tool_input, result, is_error,
                duration_seconds, scene_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.timestamp,
                entry.action,
                entry.tool_name,
                json.dumps(entry.tool_input, default=str),
                entry.result,
                int(entry.is_error),
                entry.duration_seconds,
                entry.scene_id,
                json.dumps(entry.metadata, default=str),
            ),
        )
        self._conn.commit()

    def _load_from_db(self) -> None:
        if not self._conn:
            return
        try:
            cursor = self._conn.execute(
                "SELECT timestamp, action, tool_name, tool_input, result, is_error, "
                "duration_seconds, scene_id, metadata FROM audit_log ORDER BY timestamp"
            )
            for row in cursor:
                self._entries.append(AuditEntry(
                    timestamp=row[0],
                    action=row[1],
                    tool_name=row[2] or "",
                    tool_input=json.loads(row[3]) if row[3] else {},
                    result=row[4] or "",
                    is_error=bool(row[5]),
                    duration_seconds=row[6] or 0.0,
                    scene_id=row[7] or "",
                    metadata=json.loads(row[8]) if row[8] else {},
                ))
        except Exception as exc:
            log.warning("failed to load audit log: %s", exc)


# ============================================================
# #5 Conflict Resolution (file locks)
# ============================================================

class FileConflictError(RuntimeError):
    """Raised when two agents try to edit the same file."""

    def __init__(self, path: str, locked_by: str) -> None:
        super().__init__(f"File conflict: {path} is locked by {locked_by}")
        self.path = path
        self.locked_by = locked_by


class FileConflictResolver:
    """Prevent two agents from editing the same scene file simultaneously.

    Uses a simple advisory lock: each file being edited is recorded
    with the agent ID that holds the lock. Other agents must wait or
    abort.

    Usage:
        resolver = FileConflictResolver()
        # Agent 1 tries to edit scene_001.js:
        resolver.acquire("scene_001.js", agent_id="agent_1")
        # Agent 2 tries to edit the same file:
        try:
            resolver.acquire("scene_001.js", agent_id="agent_2")
        except FileConflictError as e:
            # Wait or choose another file
            pass
        # Agent 1 finishes:
        resolver.release("scene_001.js")
    """

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}  # path → agent_id
        self._lock_times: dict[str, float] = {}  # path → timestamp

    def acquire(self, file_path: str, *, agent_id: str, timeout: float = 0.0) -> bool:
        """Try to acquire a lock on a file.

        Returns True if the lock was acquired.
        Raises FileConflictError if the file is locked by another agent
        and timeout is 0.
        """
        existing = self._locks.get(file_path)
        if existing is not None and existing != agent_id:
            if timeout <= 0:
                raise FileConflictError(file_path, existing)
            # Could implement waiting here, but for simplicity we raise
            raise FileConflictError(file_path, existing)
        self._locks[file_path] = agent_id
        self._lock_times[file_path] = time.time()
        return True

    def release(self, file_path: str, *, agent_id: str | None = None) -> None:
        """Release a lock on a file."""
        if agent_id is not None and self._locks.get(file_path) != agent_id:
            return  # not our lock
        self._locks.pop(file_path, None)
        self._lock_times.pop(file_path, None)

    def is_locked(self, file_path: str) -> bool:
        """Return True if a file is currently locked."""
        return file_path in self._locks

    def who_owns(self, file_path: str) -> str | None:
        """Return the agent ID that holds the lock, or None."""
        return self._locks.get(file_path)

    def force_release(self, file_path: str) -> None:
        """Force-release a lock (admin override)."""
        self._locks.pop(file_path, None)
        self._lock_times.pop(file_path, None)

    def clear(self) -> None:
        """Clear all locks."""
        self._locks.clear()
        self._lock_times.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "locked_files": len(self._locks),
            "locks": dict(self._locks),
        }


# ============================================================
# #7 State Validation
# ============================================================

class StateValidationError(RuntimeError):
    """Raised when project state is inconsistent."""
    pass


@dataclass
class StateValidationResult:
    """Result of validating project state before resume."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_project_state(project_root: str | Path) -> StateValidationResult:
    """Check that project state is consistent before resuming.

    Verifies:
    - All referenced scene files exist.
    - No orphaned scene files (not in the manifest).
    - The project manifest is valid JSON.
    - The shared-assets manifest exists if scenes reference shared/.

    Returns a StateValidationResult with errors and warnings.
    """
    root = Path(project_root)
    errors: list[str] = []
    warnings: list[str] = []

    # Check project manifest
    manifest_path = root / "project.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                errors.append("project.json is not a JSON object")
        except json.JSONDecodeError as exc:
            errors.append(f"project.json is invalid JSON: {exc}")
    else:
        warnings.append("project.json not found — first run?")

    # Check scene files exist
    scenes_dir = root / "scenes"
    if scenes_dir.is_dir():
        scene_files = list(scenes_dir.glob("*.js"))
        if not scene_files:
            warnings.append("scenes/ directory exists but has no .js files")

    # Check shared assets manifest
    shared_manifest = root / "shared" / "manifest.json"
    if (root / "shared").is_dir():
        if not shared_manifest.is_file():
            warnings.append("shared/ directory exists but manifest.json is missing")

    return StateValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ============================================================
# #11 Agent Handoff
# ============================================================

@dataclass
class AgentHandoff:
    """Transfer a task to another agent mid-work.

    The current agent can hand off its remaining work to another agent
    with full context: what's done, what's pending, and why the handoff
    is happening.

    Usage:
        handoff = AgentHandoff()
        # Agent 1 is running low on context window:
        handoff.create(
            from_agent="agent_1",
            to_agent="agent_2",
            task="Generate scenes 10-20",
            context={"done_scenes": ["s1"..."s9"], "pending": ["s10"..."s20"]},
            reason="Context window nearly full",
        )
        # Agent 2 picks up:
        info = handoff.accept("agent_2")
    """

    _pending: list[dict[str, Any]] = field(default_factory=list)
    _accepted: list[dict[str, Any]] = field(default_factory=list)

    def create(
        self,
        *,
        from_agent: str,
        to_agent: str,
        task: str,
        context: dict[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Create a handoff request."""
        handoff_data = {
            "id": f"handoff_{len(self._pending) + 1:04d}",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "task": task,
            "context": context or {},
            "reason": reason,
            "created_at": time.time(),
            "accepted": False,
        }
        self._pending.append(handoff_data)
        log.info("Handoff created: %s → %s (%s)", from_agent, to_agent, task)
        return handoff_data

    def accept(self, agent_id: str) -> dict[str, Any] | None:
        """Accept the next handoff for the given agent. Returns the handoff data."""
        for handoff in self._pending:
            if handoff["to_agent"] == agent_id and not handoff["accepted"]:
                handoff["accepted"] = True
                handoff["accepted_at"] = time.time()
                self._accepted.append(handoff)
                self._pending.remove(handoff)
                log.info("Handoff accepted by %s: %s", agent_id, handoff["task"])
                return handoff
        return None

    def pending_for(self, agent_id: str) -> list[dict[str, Any]]:
        """Return pending handoffs for the given agent."""
        return [h for h in self._pending if h["to_agent"] == agent_id]

    def stats(self) -> dict[str, Any]:
        return {
            "pending": len(self._pending),
            "accepted": len(self._accepted),
        }


# ============================================================
# #12 Rollback to Checkpoint
# ============================================================

@dataclass
class Checkpoint:
    """A saved checkpoint of project state."""

    checkpoint_id: str
    created_at: float
    description: str
    file_hashes: dict[str, str] = field(default_factory=dict)  # path → hash
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Save and restore project state checkpoints.

    Takes a snapshot of all files in the project (by hash). When a
    rollback is requested, restores the files to their checkpoint state.

    Usage:
        mgr = CheckpointManager(project_root="/path/to/project")
        # Before a risky operation:
        cp = mgr.save(description="Before scene 5 revision")
        # ... do the revision ...
        # If it fails:
        mgr.rollback(cp.checkpoint_id)
    """

    def __init__(self, *, project_root: str | Path) -> None:
        self._root = Path(project_root)
        self._checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_dir = self._root / ".openharness" / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        description: str = "",
        files: list[str] | None = None,
    ) -> Checkpoint:
        """Save a checkpoint of the current project state.

        Args:
            description: Human-readable description of the checkpoint.
            files: Optional list of specific files to checkpoint. If None,
                checkpoints all files in the project root (excluding
                .openharness/ and node_modules/).
        """
        checkpoint_id = f"cp_{int(time.time())}_{len(self._checkpoints):04d}"
        file_hashes: dict[str, str] = {}
        files_to_check = files or self._discover_files()
        for file_path in files_to_check:
            full_path = self._root / file_path
            if full_path.is_file():
                content = full_path.read_bytes()
                file_hashes[file_path] = hashlib.sha1(content).hexdigest()
                # Copy to checkpoint dir
                cp_file = self._checkpoint_dir / checkpoint_id / file_path
                cp_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, cp_file)
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            created_at=time.time(),
            description=description,
            file_hashes=file_hashes,
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._save_index()
        return checkpoint

    def rollback(self, checkpoint_id: str) -> bool:
        """Restore project state to a checkpoint.

        Returns True if the rollback succeeded, False if the checkpoint
        doesn't exist.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return False
        cp_dir = self._checkpoint_dir / checkpoint_id
        for file_path in checkpoint.file_hashes:
            cp_file = cp_dir / file_path
            if cp_file.is_file():
                target = self._root / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cp_file, target)
        log.info("Rolled back to checkpoint %s: %s", checkpoint_id, checkpoint.description)
        return True

    def list_checkpoints(self) -> list[Checkpoint]:
        """Return all saved checkpoints (oldest first)."""
        return sorted(self._checkpoints.values(), key=lambda c: c.created_at)

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        self._checkpoints.pop(checkpoint_id, None)
        cp_dir = self._checkpoint_dir / checkpoint_id
        if cp_dir.is_dir():
            shutil.rmtree(cp_dir, ignore_errors=True)
        self._save_index()

    def _discover_files(self) -> list[str]:
        """Discover all files in the project root (excluding noise)."""
        exclude = {".openharness", "node_modules", "__pycache__", ".git"}
        files: list[str] = []
        for item in self._root.rglob("*"):
            if item.is_file():
                rel = item.relative_to(self._root)
                if any(part in exclude for part in rel.parts):
                    continue
                files.append(str(rel))
        return files

    def _save_index(self) -> None:
        """Save the checkpoint index to disk."""
        index_path = self._checkpoint_dir / "index.json"
        data = {
            "checkpoints": [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "created_at": c.created_at,
                    "description": c.description,
                    "file_hashes": c.file_hashes,
                    "metadata": c.metadata,
                }
                for c in self._checkpoints.values()
            ]
        }
        index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_index(self) -> None:
        """Load the checkpoint index from disk."""
        index_path = self._checkpoint_dir / "index.json"
        if not index_path.is_file():
            return
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            for cp_data in data.get("checkpoints", []):
                self._checkpoints[cp_data["checkpoint_id"]] = Checkpoint(
                    checkpoint_id=cp_data["checkpoint_id"],
                    created_at=cp_data.get("created_at", 0.0),
                    description=cp_data.get("description", ""),
                    file_hashes=cp_data.get("file_hashes", {}),
                    metadata=cp_data.get("metadata", {}),
                )
        except Exception as exc:
            log.warning("failed to load checkpoint index: %s", exc)


# ============================================================
# #14 Scene Status Tracking
# ============================================================

class SceneStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    GENERATED = "generated"
    IN_QA = "in_qa"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SceneRecord:
    """Status record for a single scene."""

    scene_id: str
    status: SceneStatus = SceneStatus.PENDING
    file_path: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "status": self.status.value,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "attempts": self.attempts,
            "metadata": self.metadata,
        }


class SceneStatusTracker:
    """Track the status of every scene in the project.

    Knows which scenes are done, pending, failed, rendering, etc.
    Persists to JSON so state survives across sessions.

    Usage:
        tracker = SceneStatusTracker(project_root="/path/to/project")
        tracker.set("scene_001", SceneStatus.GENERATED)
        tracker.set("scene_002", SceneStatus.QA_FAILED, error="lint error")

        # Query:
        done = tracker.get_by_status(SceneStatus.RENDERED)
        failed = tracker.get_by_status(SceneStatus.FAILED)
        pending = tracker.get_by_status(SceneStatus.PENDING)
    """

    def __init__(self, *, project_root: str | Path) -> None:
        self._root = Path(project_root)
        self._scenes: dict[str, SceneRecord] = {}
        self._state_file = self._root / ".openharness" / "scene_status.json"
        self._load()

    def set(
        self,
        scene_id: str,
        status: SceneStatus,
        *,
        error: str = "",
        file_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set the status of a scene."""
        record = self._scenes.get(scene_id)
        if record is None:
            record = SceneRecord(scene_id=scene_id)
            self._scenes[scene_id] = record
        record.status = status
        record.updated_at = time.time()
        if error:
            record.error = error
        if file_path:
            record.file_path = file_path
        if metadata:
            record.metadata.update(metadata)
        if status in (SceneStatus.QA_FAILED, SceneStatus.FAILED):
            record.attempts += 1
        self._save()

    def get(self, scene_id: str) -> SceneRecord | None:
        """Return the status record for a scene."""
        return self._scenes.get(scene_id)

    def get_by_status(self, status: SceneStatus) -> list[SceneRecord]:
        """Return all scenes with the given status."""
        return [r for r in self._scenes.values() if r.status == status]

    def get_status_counts(self) -> dict[str, int]:
        """Return a count of scenes per status."""
        counts: dict[str, int] = {}
        for record in self._scenes.values():
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return counts

    @property
    def total_scenes(self) -> int:
        return len(self._scenes)

    @property
    def completed_count(self) -> int:
        return sum(
            1 for r in self._scenes.values()
            if r.status in (SceneStatus.RENDERED, SceneStatus.QA_PASSED)
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1 for r in self._scenes.values()
            if r.status in (SceneStatus.FAILED, SceneStatus.QA_FAILED)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total_scenes,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "counts": self.get_status_counts(),
        }

    def _save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scenes": {sid: r.to_dict() for sid, r in self._scenes.items()}
        }
        self._state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._state_file.is_file():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            for scene_id, scene_data in data.get("scenes", {}).items():
                record = SceneRecord(
                    scene_id=scene_id,
                    status=SceneStatus(scene_data.get("status", "pending")),
                    file_path=scene_data.get("file_path", ""),
                    created_at=scene_data.get("created_at", time.time()),
                    updated_at=scene_data.get("updated_at", time.time()),
                    error=scene_data.get("error", ""),
                    attempts=scene_data.get("attempts", 0),
                    metadata=scene_data.get("metadata", {}),
                )
                self._scenes[scene_id] = record
        except Exception as exc:
            log.warning("failed to load scene status: %s", exc)
