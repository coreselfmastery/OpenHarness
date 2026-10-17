"""style_consistency tool — one visual language across all scenes.

Extracts the style DNA (palette, fonts, font sizes, cubic-bezier
easings, transitions) from each scene's code, checks it against the
project reference (brand.yaml / tokens.json / STYLE.md or inline
values), and flags cross-scene drift — the slightly different blue,
the one scene with a different easing — with scene numbers and the
offending token, before it becomes visible on screen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class StyleConsistencyInput(BaseModel):
    """Arguments for the style_consistency tool."""

    paths: list[str] = Field(
        description=(
            "Scene code file(s) or directory(s) (.py/.js/.jsx/.ts/.tsx/.html/.css)."
        )
    )
    palette_file: str | None = Field(
        default=None,
        description="Brand palette file (brand.yaml/brand.json or styles/*/tokens.json).",
    )
    style_md: str | None = Field(
        default=None,
        description="Optional STYLE.md carrying palette/easing standards.",
    )
    palette: list[str] | None = Field(
        default=None,
        description="Inline reference palette hex colors (#RRGGBB).",
    )
    fonts: list[str] | None = Field(
        default=None,
        description="Inline reference font families.",
    )
    easings: list[str] | None = Field(
        default=None,
        description="Inline reference motion curves (cubic-bezier(...)).",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path to write the full JSON report.",
    )


class StyleConsistencyTool(BaseTool):
    """Verify all scenes share the same visual language."""

    name = "style_consistency"
    description = (
        "Check visual-language consistency across scene code: extracts palette, "
        "font families/sizes, cubic-bezier easings and transitions from every "
        "scene, compares them against the project reference (brand.yaml / "
        "tokens.json / STYLE.md or inline values) AND against each other "
        "(cross-scene drift). Flags the scene and the exact off-brand token "
        "('#2A6FF2 not in brand palette', 'easing differs from project "
        "standard') before viewers notice the drift."
    )
    input_model = StyleConsistencyInput

    def is_read_only(self, arguments: StyleConsistencyInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: StyleConsistencyInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.style_consistency import (
            StyleConsistencyChecker,
            find_scene_code_files,
            format_report_text,
            load_reference,
        )

        files: list[Path] = []
        for raw in arguments.paths:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(context.cwd) / path
            found = find_scene_code_files(path)
            if found:
                files.extend(found)
        if not files:
            return ToolResult(
                output=(
                    "style_consistency: no scene code files found (supported: "
                    ".py/.js/.jsx/.ts/.tsx/.html/.css)."
                ),
                is_error=True,
            )

        palette_file = None
        if arguments.palette_file:
            palette_file = Path(arguments.palette_file)
            if not palette_file.is_absolute():
                palette_file = Path(context.cwd) / palette_file
            if not palette_file.is_file():
                return ToolResult(
                    output=f"style_consistency: palette file not found: "
                    f"{arguments.palette_file}",
                    is_error=True,
                )
        style_md = None
        if arguments.style_md:
            style_md = Path(arguments.style_md)
            if not style_md.is_absolute():
                style_md = Path(context.cwd) / style_md
            if not style_md.is_file():
                return ToolResult(
                    output=f"style_consistency: STYLE.md not found: {arguments.style_md}",
                    is_error=True,
                )

        reference = load_reference(
            palette_file,
            palette=arguments.palette,
            fonts=arguments.fonts,
            easings=arguments.easings,
            style_md=style_md,
        )
        if not (reference["palette"] or reference["fonts"] or reference["easings"]):
            return ToolResult(
                output=(
                    "style_consistency: no reference style provided — pass "
                    "palette_file (brand.yaml/tokens.json), STYLE.md, or inline "
                    "palette/fonts/easings. Cross-scene drift checking still "
                    "needs at least a reference to be meaningful."
                ),
                is_error=True,
            )

        checker = StyleConsistencyChecker(reference)
        report = checker.check_files(files)

        text = format_report_text(report)
        if not report.passed:
            text += (
                "\n[style_consistency] deviations found — fix the flagged "
                "tokens so all scenes speak the same visual language."
            )
        else:
            text += "\n[style_consistency] project visual language is consistent."

        metadata: dict = {"report": report.as_dict()}
        if arguments.report_path:
            report_path = Path(arguments.report_path)
            if not report_path.is_absolute():
                report_path = Path(context.cwd) / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(metadata["report"], indent=2), encoding="utf-8"
            )
            text += f"\n[style_consistency] full report written to {report_path}"
            metadata["report_path"] = str(report_path)

        return ToolResult(output=text, is_error=not report.passed, metadata=metadata)
