"""Higher-level system prompt assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openharness.config.paths import (
    get_project_active_repo_context_path,
    get_project_issue_file,
    get_project_pr_comments_file,
)
from openharness.config.settings import Settings
from openharness.coordinator.coordinator_mode import get_coordinator_system_prompt, is_coordinator_mode
from openharness.memory import load_memory_prompt
from openharness.memory.relevance import format_relevant_memories, select_relevant_memories
from openharness.memory.usage import mark_memory_used
from openharness.personalization.rules import load_local_rules
from openharness.permissions.modes import PermissionMode
from openharness.prompts.claudemd import load_claude_md_prompt
from openharness.prompts.system_prompt import build_system_prompt
from openharness.skills.loader import load_skill_registry


def _build_skills_section(
    cwd: str | Path,
    *,
    extra_skill_dirs: Iterable[str | Path] | None = None,
    extra_plugin_roots: Iterable[str | Path] | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Build a system prompt section listing available skills."""
    registry = load_skill_registry(
        cwd,
        extra_skill_dirs=extra_skill_dirs,
        extra_plugin_roots=extra_plugin_roots,
        settings=settings,
    )
    skills = [skill for skill in registry.list_skills() if not skill.disable_model_invocation]
    if not skills:
        return None
    lines = [
        "# Available Skills",
        "",
        "The following skills are available via the `skill` tool. "
        "When a user's request matches a skill, invoke it with `skill(name=\"<skill_name>\")` "
        "to load detailed instructions before proceeding. "
        "User-invocable skills can also be run directly by the user as `/<skill-name>`.",
        "",
    ]
    for skill in skills:
        command_name = skill.command_name or skill.name
        display = f" ({skill.display_name})" if skill.display_name else ""
        lines.append(f"- **{command_name}**{display}: {skill.description}")
    return "\n".join(lines)


def _build_delegation_section() -> str:
    """Build a concise section describing delegation and worker usage."""
    return "\n".join(
        [
            "# Delegation And Subagents",
            "",
            "OpenHarness can delegate background work with the `agent` tool.",
            "Use it when the user explicitly asks for a subagent, background worker, or parallel investigation, "
            "or when the task clearly benefits from splitting off a focused worker.",
            "",
            "Default pattern:",
            '- Spawn with `agent(description=..., prompt=..., subagent_type=\"worker\")`.',
            "- Inspect running or recorded workers with `/agents`.",
            "- Inspect one worker in detail with `/agents show TASK_ID`.",
            "- Send follow-up instructions with `send_message(task_id=..., message=...)`.",
            "- Read worker output with `task_output(task_id=...)`.",
            "",
            "Prefer a normal direct answer for simple tasks. Use subagents only when they materially help.",
        ]
    )


def _build_permission_mode_section(settings: Settings) -> str:
    """Build current permission-mode guidance for the model."""
    mode = settings.permission.mode
    if mode == PermissionMode.PLAN:
        guidance = (
            "Plan mode is enabled. Treat this session as read-only planning and analysis. "
            "Do not call mutating tools such as file writes, edits, package installs, "
            "state-changing shell commands, or task-spawning actions unless the user exits plan mode."
        )
    elif mode == PermissionMode.FULL_AUTO:
        guidance = (
            "Full-auto permission mode is enabled. You may use mutating tools when they are necessary "
            "for the user's request, while still keeping changes scoped and intentional."
        )
    else:
        guidance = (
            "Default permission mode is enabled. Read-only tools can run directly; mutating tools "
            "may require explicit user approval."
        )
    return f"# Current Permission Mode\n{guidance}"


# brand.yaml is re-read on EVERY system-prompt rebuild (once per user
# line). Cache the built section keyed by (mtime_ns, size) so unchanged
# files skip the YAML parse; any write invalidates immediately.
_BRAND_CACHE: dict[str, tuple[tuple[int, int], str | None]] = {}


def _build_brand_anchor_section(cwd: str | Path) -> str | None:
    """Load brand.yaml from the project root and build a system-prompt section.

    Returns None when brand.yaml is missing (no overhead). When present,
    returns a formatted section that lists the brand anchors (name,
    colors, fonts, voice, music style, min type size, safe area, easing)
    so the agent is brand-aware on every turn without needing to call
    the brand_kit tool explicitly.

    Defined here (not in ui/runtime.py) so that EVERY caller of
    :func:`build_runtime_system_prompt` — session build, per-line prompt
    refresh, command submit_prompt/continue_pending paths — includes the
    brand anchors. Previously only session build injected them, so the
    first per-line prompt rebuild silently wiped them.
    """
    brand_path = Path(cwd) / "brand.yaml"
    cache_key = str(brand_path)
    try:
        stat = brand_path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        _BRAND_CACHE.pop(cache_key, None)
        return None
    cached = _BRAND_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        # Lazy import: tools.brand_kit_tool must not be imported at module
        # load time (heavy deps; also avoids any import cycle risk).
        from openharness.tools.brand_kit_tool import BrandConfig

        brand = BrandConfig.from_yaml_file(brand_path)
    except Exception:
        _BRAND_CACHE[cache_key] = (stamp, None)
        return None  # Don't crash the session if brand.yaml is malformed.
    if not brand.name and not brand.colors.primary and not brand.fonts.heading:
        _BRAND_CACHE[cache_key] = (stamp, None)
        return None  # Empty brand config — skip.
    lines = ["# Brand Anchors (from brand.yaml)", ""]
    if brand.name:
        lines.append(f"- Channel: {brand.name}")
    if brand.colors.primary or brand.colors.secondary or brand.colors.accent:
        lines.append("- Colors:")
        if brand.colors.primary:
            lines.append(f"  - Primary: {brand.colors.primary}")
        if brand.colors.secondary:
            lines.append(f"  - Secondary: {brand.colors.secondary}")
        if brand.colors.accent:
            lines.append(f"  - Accent: {brand.colors.accent}")
    if brand.fonts.heading or brand.fonts.body:
        lines.append("- Fonts:")
        if brand.fonts.heading:
            lines.append(f"  - Heading: {brand.fonts.heading}")
        if brand.fonts.body:
            lines.append(f"  - Body: {brand.fonts.body}")
    if brand.voice:
        lines.append(f"- Voice: {brand.voice}")
    if brand.music_style:
        lines.append(f"- Music style: {brand.music_style}")
    if brand.min_type_size:
        lines.append(f"- Minimum type size: {brand.min_type_size}px")
    if brand.safe_area:
        lines.append(f"- Safe area: {brand.safe_area}px")
    if brand.easing:
        lines.append(f"- Easing: {brand.easing}")
    lines.append("")
    lines.append(
        "All generated scenes MUST comply with these brand anchors. "
        "Use the `brand_kit` tool with `action=drift` to get a per-project "
        "creative drift that stays within this envelope."
    )
    section = "\n".join(lines)
    _BRAND_CACHE[cache_key] = (stamp, section)
    return section


def build_runtime_system_prompt(
    settings: Settings,
    *,
    cwd: str | Path,
    latest_user_prompt: str | None = None,
    extra_skill_dirs: Iterable[str | Path] | None = None,
    extra_plugin_roots: Iterable[str | Path] | None = None,
    include_project_memory: bool = True,
) -> str:
    """Build the runtime system prompt with project instructions and memory."""
    if is_coordinator_mode():
        sections = [get_coordinator_system_prompt()]
    else:
        sections = [build_system_prompt(custom_prompt=settings.system_prompt, cwd=str(cwd))]

    if not is_coordinator_mode() and settings.system_prompt is None:
        sections[0] = build_system_prompt(cwd=str(cwd))

    sections.append(_build_permission_mode_section(settings))

    if settings.fast_mode:
        sections.append(
            "# Session Mode\nFast mode is enabled. Prefer concise replies, minimal tool use, and quicker progress over exhaustive exploration."
        )

    sections.append(
        "# Reasoning Settings\n"
        f"- Effort: {settings.effort}\n"
        f"- Passes: {settings.passes}\n"
        "Adjust depth and iteration count to match these settings while still completing the task."
    )

    skills_section = _build_skills_section(
        cwd,
        extra_skill_dirs=extra_skill_dirs,
        extra_plugin_roots=extra_plugin_roots,
        settings=settings,
    )
    if skills_section and not is_coordinator_mode():
        sections.append(skills_section)

    if not is_coordinator_mode():
        sections.append(_build_delegation_section())

    # Brand anchors (from brand.yaml, when present) — injected for every
    # mode and on EVERY prompt rebuild so per-turn refreshes never lose it.
    brand_section = _build_brand_anchor_section(cwd)
    if brand_section:
        sections.append(brand_section)

    claude_md = load_claude_md_prompt(cwd)
    if claude_md:
        sections.append(claude_md)

    local_rules = load_local_rules()
    if local_rules:
        sections.append(f"# Local Environment Rules\n\n{local_rules}")

    for title, path in (
        ("Issue Context", get_project_issue_file(cwd)),
        ("Pull Request Comments", get_project_pr_comments_file(cwd)),
        ("Active Repo Context", get_project_active_repo_context_path(cwd)),
    ):
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(f"# {title}\n\n```md\n{content[:12000]}\n```")

    if include_project_memory and settings.memory.enabled:
        memory_section = load_memory_prompt(
            cwd,
            max_entrypoint_lines=settings.memory.max_entrypoint_lines,
            max_entrypoint_bytes=settings.memory.max_entrypoint_bytes,
        )
        if memory_section:
            sections.append(memory_section)

        if latest_user_prompt:
            relevant = select_relevant_memories(
                latest_user_prompt,
                cwd,
                max_results=settings.memory.max_files,
            )
            if relevant:
                try:
                    headers = [item.header for item in relevant]
                    mark_memory_used(cwd, headers, memory_dir=headers[0].path.parent)
                except OSError:
                    pass
                sections.append(format_relevant_memories(relevant))

    return "\n\n".join(section for section in sections if section.strip())
