"""String-based file editing tool."""

from __future__ import annotations

import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class FileEditToolInput(BaseModel):
    """Arguments for the file edit tool."""

    path: str = Field(description="Path of the file to edit")
    old_str: str = Field(description="Existing text to replace")
    new_str: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False)


class FileEditTool(BaseTool):
    """Replace text in an existing file."""

    name = "edit_file"
    description = "Edit an existing file by replacing a string."
    input_model = FileEditToolInput

    async def execute(
        self,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        path = _resolve_path(context.cwd, arguments.path)

        from openharness.sandbox.session import is_docker_sandbox_active

        if is_docker_sandbox_active():
            from openharness.sandbox.path_validator import validate_sandbox_path

            allowed, reason = validate_sandbox_path(path, context.cwd)
            if not allowed:
                return ToolResult(output=f"Sandbox: {reason}", is_error=True)

        if not path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        # Item #31: hold the cross-agent lock across the whole
        # read-modify-write so concurrent edits cannot interleave.
        from openharness.services.conflict_resolution import FileConflictError

        try:
            async with _edit_path_lock(context, path):
                return await self._execute_locked(arguments, context, path)
        except FileConflictError as exc:
            return ToolResult(output=f"Edit blocked: {exc}", is_error=True)

    async def _execute_locked(
        self,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
        path: Path,
    ) -> ToolResult:
        original = path.read_text(encoding="utf-8")
        if arguments.old_str not in original:
            return ToolResult(output="old_str was not found in the file", is_error=True)

        if arguments.replace_all:
            updated = original.replace(arguments.old_str, arguments.new_str)
        else:
            # A non-unique old_str is AMBIGUOUS: silently editing the
            # first occurrence edits the WRONG one whenever the model
            # meant the other — error out and ask for more context (the
            # same contract as Claude Code's Edit tool).
            occurrences = original.count(arguments.old_str)
            if occurrences > 1:
                return ToolResult(
                    output=(
                        f"old_str is not unique: found {occurrences} occurrences. "
                        "Include more surrounding context lines in old_str to make it "
                        "unique, or pass replace_all=true to replace every occurrence."
                    ),
                    is_error=True,
                )
            updated = original.replace(arguments.old_str, arguments.new_str, 1)

        approval_prompt = context.metadata.get("edit_approval_prompt") if context.metadata else None
        if approval_prompt is not None:
            diff_text, added, removed = _compute_diff(str(path), original, updated)
            reply = await approval_prompt(str(path), diff_text, added, removed)
            if reply == "reject":
                return ToolResult(output=f"Edit rejected by user: {path}", is_error=True)
            path.write_text(updated, encoding="utf-8")
            stats = f"  ({_ANSI_GREEN}+{added}{_ANSI_RESET} {_ANSI_RED}-{removed}{_ANSI_RESET})"
            return ToolResult(output=f"Updated {path}{stats}")

        path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {path}")


def _edit_path_lock(context: ToolExecutionContext, path: Path):
    """Return the cross-agent lock context for ``path`` when enabled (#31)."""
    from contextlib import nullcontext

    settings = (context.metadata or {}).get("settings")
    try:
        from openharness.services.conflict_resolution import (
            acquire_path_lock,
            configure_file_locking_from_settings,
            lock_timeout_from_settings,
        )

        if not configure_file_locking_from_settings(settings):
            return nullcontext()
        return acquire_path_lock(
            path,
            timeout_seconds=lock_timeout_from_settings(settings),
            holder_label="openharness edit_file",
        )
    except Exception:
        return nullcontext()


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _compute_diff(filename: str, original: str, updated: str) -> tuple[str, int, int]:
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=filename,
            tofile=filename,
            lineterm="",
        )
    )
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return "".join(diff_lines), added, removed


_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
