"""style_memory tool — cross-project style pattern suggestions.

When a project lands a high quality score, record its style DNA
(palette, fonts, motion curves, transitions). When a new project
starts, ask for suggestions and get concrete styles that WORKED,
ranked by measured quality — the harness improves with every
completed project.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class StyleMemoryInput(BaseModel):
    """Arguments for the style_memory tool."""

    action: str = Field(
        description=(
            "Operation: record (store a project's style on success), "
            "suggest (rank stored styles for a new project), projects "
            "(per-project summary), stats."
        ),
        pattern="^(record|suggest|projects|stats)$",
    )
    project: str = Field(
        default="",
        description="Project name (record) or filter (suggest/projects).",
    )
    style_name: str = Field(
        default="",
        description="Style slug for record (e.g. 'editorial-dark').",
    )
    content_type: str = Field(
        default="",
        description="Content category: 'knowledge shorts', 'promo', 'explainer' ...",
    )
    palette: list[str] | None = Field(
        default=None,
        description="Brand hex colors (#RRGGBB) for record.",
    )
    fonts: list[str] | None = Field(
        default=None,
        description="Font families for record (e.g. ['Inter', 'Playfair Display']).",
    )
    motion: list[str] | None = Field(
        default=None,
        description="Motion curves for record (e.g. ['cubic-bezier(0.4,0,0.2,1)']).",
    )
    transitions: list[str] | None = Field(
        default=None,
        description="Transition types for record (e.g. ['slide-in 0.6s']).",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final project quality score (1.0 = best).",
    )
    notes: str = Field(default="", description="Free-form notes for record.")
    tokens_file: str | None = Field(
        default=None,
        description=(
            "Optional tokens.json / brand.yaml path — palette and fonts are "
            "extracted automatically when palette/fonts are not given."
        ),
    )
    min_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Suggest: quality floor."
    )
    limit: int = Field(default=5, ge=1, le=25, description="Suggest result count.")


class StyleMemoryTool(BaseTool):
    """Cross-project memory of successful style patterns."""

    name = "style_memory"
    description = (
        "Learn styles across projects: `record` a finished project's style DNA "
        "(palette, fonts, motion curves, transitions + final quality score) — "
        "then `suggest` ranked styles that WORKED for a content type when "
        "starting a new project. The harness gets better with every completed "
        "project instead of starting from scratch."
    )
    input_model = StyleMemoryInput

    def is_read_only(self, arguments: StyleMemoryInput) -> bool:
        return arguments.action in {"suggest", "projects", "stats"}

    async def execute(
        self, arguments: StyleMemoryInput, context: ToolExecutionContext
    ) -> ToolResult:
        from pathlib import Path

        from openharness.services.style_memory import StyleMemory, StyleRecord, get_style_memory

        store = get_style_memory()

        if arguments.action == "record":
            if not (arguments.project and arguments.style_name):
                return ToolResult(
                    output="style_memory: `project` and `style_name` are required "
                    "for record.",
                    is_error=True,
                )
            palette = list(arguments.palette or [])
            fonts = list(arguments.fonts or [])
            if (not palette or not fonts) and arguments.tokens_file:
                tokens_path = Path(arguments.tokens_file)
                if not tokens_path.is_absolute():
                    tokens_path = Path(context.cwd) / tokens_path
                extracted = StyleMemory.extract_from_tokens(tokens_path)
                palette = palette or extracted.get("palette") or []
                fonts = fonts or extracted.get("fonts") or []
            record = StyleRecord(
                project=arguments.project,
                style_name=arguments.style_name,
                content_type=arguments.content_type,
                palette=palette,
                fonts=fonts,
                motion=list(arguments.motion or []),
                transitions=list(arguments.transitions or []),
                quality_score=arguments.quality_score,
                notes=arguments.notes,
            )
            if not store.record(record):
                return ToolResult(
                    output="style_memory: store unavailable (see stats).",
                    is_error=True,
                )
            return ToolResult(
                output=(
                    f"[style_memory] recorded {arguments.style_name!r} for "
                    f"{arguments.project!r} (quality {arguments.quality_score:g}): "
                    f"palette={len(record.palette)} colors, "
                    f"fonts={len(record.fonts)}, motion={len(record.motion)}, "
                    f"transitions={len(record.transitions)}."
                ),
                metadata={"recorded": record.as_dict()},
            )

        if arguments.action == "suggest":
            suggestions = store.suggest(
                content_type=arguments.content_type or None,
                min_quality=arguments.min_quality,
                limit=arguments.limit,
            )
            if not suggestions:
                scope = arguments.content_type or "any content type"
                return ToolResult(
                    output=(
                        f"[style_memory] no stored style for {scope} at quality >= "
                        f"{arguments.min_quality:g} — record successful projects "
                        "to build the library."
                    ),
                    metadata={"suggestions": []},
                )
            lines = [
                (
                    f"[style_memory] top {len(suggestions)} style(s) for "
                    f"{arguments.content_type or 'any content type'}:"
                )
            ]
            for item in suggestions:
                lines.append(
                    f"  - {item['style_name']} ({item['project']}, quality "
                    f"{item['quality_score']:.2f}): "
                    f"palette={','.join(item['palette'][:4]) or '-'}; "
                    f"fonts={','.join(item['fonts'][:3]) or '-'}; "
                    f"motion={','.join(item['motion'][:2]) or '-'}"
                )
                if item["notes"]:
                    lines.append(f"      note: {item['notes']}")
            return ToolResult(
                output="\n".join(lines), metadata={"suggestions": suggestions}
            )

        if arguments.action == "projects":
            projects = store.projects()
            if not projects:
                return ToolResult(
                    output="[style_memory] no projects recorded yet.",
                    metadata={"projects": []},
                )
            lines = ["[style_memory] recorded projects:"]
            for item in projects:
                lines.append(
                    f"  - {item['project']}: {item['n']} style(s), best quality "
                    f"{item['best']:.2f}"
                )
            return ToolResult(
                output="\n".join(lines), metadata={"projects": projects}
            )

        # stats
        stats = store.stats()
        return ToolResult(
            output=(
                "[style_memory] stats:\n"
                f"  total styles: {stats['total']}\n"
                f"  by content type: {json.dumps(stats['by_content_type'])}\n"
                f"  store: {stats['path']} (available: {stats['available']})"
            ),
            metadata=stats,
        )
