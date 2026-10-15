"""Tool-output context budget helpers."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_TOOL_OUTPUT_INLINE_CHARS = 16_000
DEFAULT_TOOL_OUTPUT_PREVIEW_CHARS = 3_000
DEFAULT_MICROCOMPACT_TOOL_RESULT_CHARS = 4_000


def _read_positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning("Ignoring invalid %s=%r", name, raw)
        return default


# Cache the parsed env values. These helpers are called on every tool result
# (multiple times per agent turn), and each call re-reads the environment
# and re-parses the integer. The values change only when the operator
# restarts the process with new env vars, so we read them once at module
# import. Tests that mutate the env should call ``_reset_env_cache()``.
_TOOL_OUTPUT_INLINE_CHARS: int | None = None
_TOOL_OUTPUT_PREVIEW_CHARS: int | None = None
_MICROCOMPACT_TOOL_RESULT_CHARS: int | None = None


def _reset_env_cache() -> None:
    """Clear the cached env-derived constants. Used by tests."""
    global _TOOL_OUTPUT_INLINE_CHARS, _TOOL_OUTPUT_PREVIEW_CHARS, _MICROCOMPACT_TOOL_RESULT_CHARS
    _TOOL_OUTPUT_INLINE_CHARS = None
    _TOOL_OUTPUT_PREVIEW_CHARS = None
    _MICROCOMPACT_TOOL_RESULT_CHARS = None


def tool_output_inline_chars() -> int:
    global _TOOL_OUTPUT_INLINE_CHARS
    if _TOOL_OUTPUT_INLINE_CHARS is None:
        _TOOL_OUTPUT_INLINE_CHARS = _read_positive_int_env(
            "OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS",
            DEFAULT_TOOL_OUTPUT_INLINE_CHARS,
            minimum=256,
        )
    return _TOOL_OUTPUT_INLINE_CHARS


def tool_output_preview_chars() -> int:
    global _TOOL_OUTPUT_PREVIEW_CHARS
    if _TOOL_OUTPUT_PREVIEW_CHARS is None:
        _TOOL_OUTPUT_PREVIEW_CHARS = _read_positive_int_env(
            "OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS",
            DEFAULT_TOOL_OUTPUT_PREVIEW_CHARS,
            minimum=128,
        )
    return _TOOL_OUTPUT_PREVIEW_CHARS


def microcompact_tool_result_chars() -> int:
    global _MICROCOMPACT_TOOL_RESULT_CHARS
    if _MICROCOMPACT_TOOL_RESULT_CHARS is None:
        _MICROCOMPACT_TOOL_RESULT_CHARS = _read_positive_int_env(
            "OPENHARNESS_MICROCOMPACT_TOOL_RESULT_CHARS",
            DEFAULT_MICROCOMPACT_TOOL_RESULT_CHARS,
            minimum=256,
        )
    return _MICROCOMPACT_TOOL_RESULT_CHARS


def is_microcompactable_tool_result(tool_name: str, content: str) -> bool:
    """Return True when a tool result should be eligible for old-result clearing."""
    normalized = tool_name.strip()
    if normalized.startswith("mcp__"):
        return True
    return len(content) >= microcompact_tool_result_chars()
