"""Append-only JSONL audit log — item #29.

Every tool call, LLM call, render job, QA verdict and control action is
appended as one JSON line to a daily file under
``~/.openharness/audit/``. The log is:

- append-only (opened with "a", flushed per entry),
- process-safe (a threading lock; multiple agents each write their own
  process log and the JSON lines merge trivially),
- size-bounded (files rotate daily; ``prune`` drops files older than
  ``retention_days``),
- honest: entries record durations and outcomes, never invent data.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MAX_ENTRY_DETAIL_CHARS = 2000


def default_audit_dir() -> Path:
    from openharness.config.paths import get_data_dir

    return get_data_dir() / "audit"


class AuditLog:
    """Append-only JSONL audit trail (#29)."""

    def __init__(
        self,
        audit_dir: str | Path | None = None,
        *,
        session_id: str = "",
        enabled: bool = True,
    ) -> None:
        self._dir = Path(audit_dir) if audit_dir else default_audit_dir()
        self._session_id = session_id
        self._enabled = enabled
        self._lock = threading.Lock()
        self.entries_written = 0
        self._dropped = 0
        if not enabled:
            self._available = False
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._available = True
        except OSError as exc:
            log.warning("audit log disabled (cannot create %s): %s", self._dir, exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def enabled(self) -> bool:
        """False when the user turned audit logging off (no-op mode)."""
        return self._enabled

    def record(self, kind: str, detail: dict[str, Any] | None = None, *, level: str = "info") -> None:
        """Append one entry. Never raises — audit must not break the pipeline."""
        if not self._available:
            return
        entry = {
            "ts": round(time.time(), 3),
            "session": self._session_id,
            "kind": kind,
            "level": level,
            "detail": _bounded(detail or {}),
        }
        line = json.dumps(entry, separators=(",", ":"), default=str)
        path = self._file_for_today()
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self.entries_written += 1
        except OSError as exc:
            self._dropped += 1
            if self._dropped <= 3:
                log.warning("audit write failed: %s", exc)

    def record_tool_call(
        self,
        *,
        tool: str,
        is_error: bool,
        duration_seconds: float,
        summary: str = "",
    ) -> None:
        self.record(
            "tool_call",
            {
                "tool": tool,
                "is_error": is_error,
                "duration_seconds": round(duration_seconds, 3),
                "summary": summary[:200],
            },
            level="warn" if is_error else "info",
        )

    def record_llm_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        self.record(
            "llm_call",
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "duration_seconds": round(duration_seconds, 3),
            },
        )

    def record_control(self, action: str, detail: dict[str, Any] | None = None) -> None:
        self.record(f"control.{action}", detail or {})

    def _file_for_today(self) -> Path:
        return self._dir / f"audit-{time.strftime('%Y%m%d')}.jsonl"

    # ------------------------------------------------------------------

    def tail(self, limit: int = 50, *, kind_prefix: str = "") -> list[dict[str, Any]]:
        """Read the newest entries (newest last). Test/UI helper."""
        if not self._available:
            return []
        path = self._file_for_today()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        entries: list[dict[str, Any]] = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if kind_prefix and not str(entry.get("kind", "")).startswith(kind_prefix):
                continue
            entries.append(entry)
        return entries[-limit:]

    def prune(self, *, retention_days: int = 14) -> int:
        """Delete audit files older than ``retention_days``. Returns files removed."""
        if not self._available or retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86_400
        removed = 0
        for path in self._dir.glob("audit-*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    def stats(self) -> dict[str, Any]:
        return {
            "dir": str(self._dir),
            "available": self._available,
            "enabled": self._enabled,
            "entries_written": self.entries_written,
            "dropped": self._dropped,
        }


def _bounded(detail: dict[str, Any]) -> dict[str, Any]:
    """Keep entry payloads small so the log cannot balloon from one huge tool output."""
    out: dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, str):
            out[key] = value[:_MAX_ENTRY_DETAIL_CHARS]
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        else:
            try:
                serialized = json.dumps(value, default=str)
                out[key] = (
                    value
                    if len(serialized) <= _MAX_ENTRY_DETAIL_CHARS
                    else serialized[:_MAX_ENTRY_DETAIL_CHARS]
                )
            except (TypeError, ValueError):
                out[key] = repr(value)[:_MAX_ENTRY_DETAIL_CHARS]
    return out


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_AUDIT_LOG: AuditLog | None = None


def get_audit_log(session_id: str = "") -> AuditLog:
    """Return the shared audit log (lazily created, always non-None)."""
    global _AUDIT_LOG
    if _AUDIT_LOG is None:
        _AUDIT_LOG = AuditLog(session_id=session_id)
    return _AUDIT_LOG


def set_audit_log(log: AuditLog | None) -> None:
    """Test helper: install or reset the singleton."""
    global _AUDIT_LOG
    _AUDIT_LOG = log


def configure_audit_log_from_settings(settings: Any, *, session_id: str = "") -> AuditLog | None:
    """Engine wiring: build the audit log when ``settings.control.audit_log`` is on."""
    control = getattr(settings, "control", None)
    if control is None or not getattr(control, "audit_log", False):
        return None
    audit = AuditLog(session_id=session_id)
    audit.record("audit.start", {"session": session_id})
    return audit
