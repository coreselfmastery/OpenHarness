"""Auto-inject project context files into spawned agent prompts.

Implements optimization #3 from the top-10 list:
"Pass STYLE.md, STORYBOARD.md, and BRIEF.md to every subagent automatically."

When the user invokes the ``agent`` tool (the spawn path), we look for
these project-context files in the current working directory and prepend
them to the agent's prompt. This reduces re-work because every spawned
agent sees the same project style / story / brief without the user
having to re-attach them on every call.

The list of context files is configurable via ``settings.auto_context_files``
(default: ``["STYLE.md", "STORYBOARD.md", "BRIEF.md"]``). The feature is
on by default; disable with ``settings.auto_context_inject = False`` or
the env var ``OPENHARNESS_AUTO_CONTEXT_INJECT=0``.

We only inject files that exist — missing files are silently skipped.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# The default list of project-context files to look for. Order matters:
# BRIEF comes first (high-level context), then STORYBOARD (scene flow),
# then STYLE (visual / voice rules). Subagents need this order to ground
# themselves before reading the actual task prompt.
DEFAULT_AUTO_CONTEXT_FILES: tuple[str, ...] = (
    "BRIEF.md",
    "STORYBOARD.md",
    "STYLE.md",
)

# Hard cap on the total injected context size so we don't blow the spawned
# agent's context window before it even starts work. When the sum of
# file sizes exceeds this, we truncate the larger files.
MAX_TOTAL_INJECTED_CHARS = 20_000


def discover_auto_context_files(
    cwd: str | Path,
    *,
    candidates: Iterable[str] | None = None,
) -> list[Path]:
    """Return the project-context files that exist in or above ``cwd``.

    Walks up to 4 parent directories looking for each candidate. This
    matches the ``CLAUDE.md`` discovery pattern used by the prompt builder,
    so a single set of context files at the workspace root is picked up
    by subagents launched from any subdirectory.
    """
    if candidates is None:
        candidates = DEFAULT_AUTO_CONTEXT_FILES
    root = Path(cwd).resolve()
    found: list[Path] = []
    seen: set[Path] = set()
    # Walk up to 4 levels (workspace root is usually within 2-3 of cwd).
    directories = [root, *root.parents[:4]]
    for candidate_name in candidates:
        for directory in directories:
            candidate_path = directory / candidate_name
            if candidate_path.is_file() and candidate_path not in seen:
                found.append(candidate_path)
                seen.add(candidate_path)
                break  # Take the closest match per filename.
    return found


def build_auto_context_prefix(
    cwd: str | Path,
    *,
    candidates: Iterable[str] | None = None,
    enabled: bool = True,
    max_total_chars: int = MAX_TOTAL_INJECTED_CHARS,
) -> str:
    """Return the project-context prefix to prepend to a spawned agent prompt.

    Returns an empty string when disabled or when no candidate files exist.
    When the total size exceeds ``max_total_chars``, the larger files are
    truncated so the prefix stays under the cap.
    """
    if not enabled:
        return ""
    files = discover_auto_context_files(cwd, candidates=candidates)
    if not files:
        return ""
    sections: list[str] = []
    total = 0
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("auto-context: could not read %s: %s", path, exc)
            continue
        remaining = max_total_chars - total
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n...[truncated for auto-context]..."
        sections.append(f"## {path.name}\n\n```md\n{content.strip()}\n```")
        total += len(content)
    if not sections:
        return ""
    return "# Project Context (auto-injected)\n\n" + "\n\n".join(sections) + "\n\n"


def inject_auto_context(
    prompt: str,
    cwd: str | Path,
    *,
    candidates: Iterable[str] | None = None,
    enabled: bool = True,
    max_total_chars: int = MAX_TOTAL_INJECTED_CHARS,
) -> str:
    """Prepend the project-context prefix to ``prompt`` if enabled.

    This is the entry point called from the agent-spawn path. When the
    prefix is empty (no context files found, or feature disabled), the
    prompt is returned unchanged so we add zero overhead in the no-op case.
    """
    prefix = build_auto_context_prefix(
        cwd,
        candidates=candidates,
        enabled=enabled,
        max_total_chars=max_total_chars,
    )
    if not prefix:
        return prompt
    return f"{prefix}---\n\n# Task\n\n{prompt}"


def is_auto_context_enabled(settings: object | None) -> bool:
    """Read the ``auto_context_inject`` flag from settings, with env override."""
    if settings is not None:
        flag = getattr(settings, "auto_context_inject", True)
    else:
        flag = True
    env_override = os.environ.get("OPENHARNESS_AUTO_CONTEXT_INJECT")
    if env_override is not None:
        env_val = env_override.strip().lower()
        flag = env_val in {"1", "true", "yes", "on"}
    return bool(flag)


def get_auto_context_files(settings: object | None) -> tuple[str, ...]:
    """Read the list of context filenames from settings, falling back to default."""
    if settings is not None:
        files = getattr(settings, "auto_context_files", None)
        if files:
            return tuple(str(f) for f in files)
    return DEFAULT_AUTO_CONTEXT_FILES
