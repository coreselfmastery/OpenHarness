"""Session state validation — item #33.

Before a saved session is resumed, ``validate_and_repair`` checks the
conversation for the corruption patterns that actually occur after
crashes and kills:

- orphan ``tool_result`` blocks (their ``tool_use`` never made it to disk),
- assistant ``tool_use`` messages with no matching ``tool_result``,
- empty messages, duplicated messages, or reversed role ordering,
- sessions that were killed mid-render (detected via the audit log's
  ``control.kill`` entry for the same session).

``validate_and_repair`` returns a report listing every finding and the
repair performed, so resume is always explicit about what it changed.
The repair itself reuses the battle-tested
``sanitize_conversation_messages`` plus orphan-result pruning, and
never silently deletes a user message — only tool plumbing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openharness.engine.messages import (
    ConversationMessage,
    ToolResultBlock,
    sanitize_conversation_messages,
)

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """What the validator found and what it did."""

    findings: list[str] = field(default_factory=list)
    repaired: bool = False
    messages_before: int = 0
    messages_after: int = 0
    dropped_orphan_results: int = 0
    dropped_unanswered_tool_uses: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.findings,
            "findings": self.findings,
            "repaired": self.repaired,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "dropped_orphan_results": self.dropped_orphan_results,
            "dropped_unanswered_tool_uses": self.dropped_unanswered_tool_uses,
        }


def validate_and_repair(
    messages: list[ConversationMessage],
    *,
    session_was_killed: bool = False,
) -> tuple[list[ConversationMessage], ValidationReport]:
    """Validate + repair a message list before resume (#33).

    Order of operations:

    1. Drop orphan ``tool_result`` blocks (results whose ``tool_use_id``
       does not match any earlier assistant tool call).
    2. Run the standard sanitizer (drops unanswered trailing tool_use
       turns and empty messages, normalizes role order).
    3. Re-check; anything still invalid is reported (not hidden).

    Returns the (possibly repaired) message list and a report.
    """
    report = ValidationReport(messages_before=len(messages))
    if session_was_killed:
        report.findings.append(
            "session was killed mid-task (audit log shows control.kill); "
            "the last turn may be incomplete"
        )

    # 1. Orphan tool results.
    known_tool_use_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            known_tool_use_ids.update(block.id for block in message.tool_uses)
    repaired: list[ConversationMessage] = []
    orphan_dropped = 0
    for message in messages:
        if message.role == "user":
            results = [b for b in message.content if isinstance(b, ToolResultBlock)]
            orphans = [b for b in results if b.tool_use_id not in known_tool_use_ids]
            if orphans:
                orphan_dropped += len(orphans)
                kept = [b for b in message.content if not (
                    isinstance(b, ToolResultBlock) and b.tool_use_id not in known_tool_use_ids
                )]
                if kept:
                    repaired.append(ConversationMessage(role="user", content=kept))
                # If nothing else was in the message, drop it entirely.
                continue
        repaired.append(message)
    if orphan_dropped:
        report.dropped_orphan_results = orphan_dropped
        report.findings.append(f"dropped {orphan_dropped} orphan tool_result block(s)")
        report.repaired = True
        messages = repaired

    # 2. Standard sanitizer (trailing unanswered tool_use, empty msgs...).
    sanitized = sanitize_conversation_messages(messages)
    if len(sanitized) != len(messages):
        dropped = len(messages) - len(sanitized)
        report.dropped_unanswered_tool_uses = dropped
        report.findings.append(
            f"sanitizer dropped {dropped} malformed/unanswered tool_use message(s)"
        )
        report.repaired = True
    messages = sanitized

    # 3. Re-check: consecutive same-role user messages are legal for
    # Anthropic but a smell elsewhere; report only.
    previous_role = ""
    for index, message in enumerate(messages):
        if message.role == previous_role and message.role == "assistant":
            report.findings.append(
                f"consecutive assistant messages at index {index} (legal, but unusual)"
            )
        previous_role = message.role

    report.messages_after = len(messages)
    return messages, report


def was_session_killed(session_id: str) -> bool:
    """Check today's audit log for a kill of this session (#33 + #26 interplay)."""
    from openharness.services.audit_log import get_audit_log

    for entry in get_audit_log().tail(limit=200, kind_prefix="control.kill"):
        if str(entry.get("session", "")) == session_id:
            return True
    return False


def validate_restored_payload(
    payload: dict[str, Any] | None,
) -> tuple[list[ConversationMessage], ValidationReport]:
    """Validate a raw session-snapshot payload (JSON) end to end."""
    if not payload:
        return [], ValidationReport(findings=["no session payload"])
    raw_messages = payload.get("messages") or []
    messages: list[ConversationMessage] = []
    parse_errors = 0
    for raw in raw_messages:
        try:
            messages.append(ConversationMessage.model_validate(raw))
        except Exception:  # noqa: BLE001 - corrupt entries are reported, not fatal
            parse_errors += 1
    report_extra = ""
    if parse_errors:
        report_extra = f"{parse_errors} message(s) failed to parse and were skipped"
    repaired, report = validate_and_repair(messages)
    if report_extra:
        report.findings.append(report_extra)
        report.repaired = True
    return repaired, report
