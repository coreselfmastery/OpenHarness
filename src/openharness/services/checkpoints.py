"""Session checkpoints with rollback — item #37.

A checkpoint is a full, restorable snapshot of the conversation
(messages + engine tool-metadata) stored as JSON under
``~/.openharness/checkpoints/<session_id>/``. Unlike per-turn session
autosave, checkpoints are *named* and *listed* so the agent (or the
user via the ``checkpoint`` tool) can:

- create  — snapshot now ("before the risky refactor"),
- list    — see every checkpoint with timestamps and message counts,
- restore — roll the conversation back to any earlier checkpoint
            (more than one step back, which is the point of #37),
- diff    — compare two checkpoints (message count + last user text).

Restores are honest: restoring replaces the LIVE message list with the
snapshot's messages and returns what was rolled back.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from openharness.services.audit_log import get_audit_log

log = logging.getLogger(__name__)


def _checkpoint_root(session_id: str) -> Path:
    from openharness.config.paths import get_data_dir

    root = get_data_dir() / "checkpoints"
    if session_id:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)[:80]
        root = root / safe
    return root


class CheckpointManager:
    """Create / list / restore / diff conversation checkpoints (#37)."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._root = _checkpoint_root(session_id)
        self.auto_checkpoints = 0

    # ------------------------------------------------------------------

    def create(
        self,
        messages: list[Any],
        label: str = "",
        *,
        tool_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Snapshot ``messages`` (ConversationMessage models) now."""
        self._root.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"cp-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
        payload = {
            "id": checkpoint_id,
            "label": label.strip() or "unlabeled",
            "created_at": time.time(),
            "message_count": len(messages),
            "last_user_text": _last_user_text(messages),
            "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
            "tool_metadata": _persistable_metadata(tool_metadata),
        }
        path = self._root / f"{checkpoint_id}.json"
        path.write_text(
            json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8"
        )
        get_audit_log().record_control(
            "checkpoint.create", {"id": checkpoint_id, "label": label, "messages": len(messages)}
        )
        return self._summary(payload)

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        """List checkpoints newest-first."""
        if not self._root.is_dir():
            return []
        summaries: list[dict[str, Any]] = []
        for path in self._root.glob("cp-*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            summaries.append(self._summary(payload))
        summaries.sort(key=lambda s: s["created_at"], reverse=True)
        return summaries[:limit]

    def restore(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Load a checkpoint's messages/metadata for rollback.

        Returns the raw payload (caller rebuilds models and swaps them
        into the live conversation), or None when the id is unknown.
        """
        path = self._root / f"{checkpoint_id}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("checkpoint %s unreadable: %s", checkpoint_id, exc)
            return None
        get_audit_log().record_control(
            "checkpoint.restore", {"id": checkpoint_id, "messages": payload.get("message_count")}
        )
        return payload

    def diff(self, checkpoint_id_a: str, checkpoint_id_b: str) -> dict[str, Any] | None:
        """Compare two checkpoints (message counts + last user text)."""
        a = self.restore(checkpoint_id_a)
        b = self.restore(checkpoint_id_b)
        if a is None or b is None:
            return None
        return {
            "a": self._summary(a),
            "b": self._summary(b),
            "message_delta": int(b.get("message_count", 0)) - int(a.get("message_count", 0)),
            "different": a.get("messages") != b.get("messages"),
        }

    def prune(self, keep: int = 40) -> int:
        """Keep only the newest ``keep`` checkpoints. Returns removed count."""
        checkpoints = self.list(limit=10_000)
        removed = 0
        for summary in checkpoints[keep:]:
            try:
                (self._root / f"{summary['id']}.json").unlink()
                removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------

    @staticmethod
    def _summary(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": payload.get("id"),
            "label": payload.get("label"),
            "created_at": payload.get("created_at"),
            "created_at_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(float(payload.get("created_at") or 0))
            ),
            "message_count": payload.get("message_count"),
            "last_user_text": (payload.get("last_user_text") or "")[:120],
        }


def _last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role == "user":
            text = getattr(message, "text", None)
            if text is None and isinstance(message, dict):
                text = message.get("text")
            return str(text or "")
    return ""


def _persistable_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep checkpoint metadata small and JSON-safe (drop live objects)."""
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"mcp_manager", "bridge_manager", "settings", "tool_registry",
                   "ask_user_prompt", "edit_approval_prompt", "vision_model_config",
                   "engine_ref"}:
            continue  # live objects — rebuilt by the runtime, never stored
        try:
            json.dumps(value, default=str)
            out[key] = value
        except (TypeError, ValueError):
            continue
    return out
