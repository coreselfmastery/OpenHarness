"""scene_lint tool — structural analysis of scene code before rendering.

Agent-facing wrapper around
:mod:`openharness.services.static_scene_analyzer`. Parses generated
scene code (Python AST / calibrated JS regexes) and flags undeclared
asset references, off-safe-area positions, too-small fonts, unbranded
hardcoded colors, repeated magic constants, and non-deterministic
calls — all *before* any GPU time is spent rendering. 100x cheaper
than catching the same error after render.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SceneLintInput(BaseModel):
    """Arguments for the scene_lint tool."""

    paths: list[str] = Field(
        description=(
            "Scene code file(s) or directory(s) (.py/.js/.jsx/.ts/.tsx)."
        )
    )
    assets: list[str] | None = Field(
        default=None,
        description=(
            "Declared asset file names the scenes may reference (e.g. from "
            "the project asset manifest). References outside this list are "
            "flagged as undeclared."
        ),
    )
    assets_manifest: str | None = Field(
        default=None,
        description=(
            "Optional JSON file holding an asset list (array of strings, or "
            "an object with an 'assets' key)."
        ),
    )
    palette: list[str] | None = Field(
        default=None,
        description=(
            "Brand palette hex colors (#RRGGBB). When provided, hardcoded "
            "colors outside the palette are flagged."
        ),
    )
    palette_file: str | None = Field(
        default=None,
        description=(
            "Brand palette file (brand.yaml/brand.json or OpenVideoHarness "
            "styles/<slug>/tokens.json)."
        ),
    )
    frame_width: int = Field(default=1920, ge=16, le=16384)
    frame_height: int = Field(default=1080, ge=16, le=16384)
    min_font_px: int = Field(
        default=18,
        ge=1,
        le=200,
        description="Minimum font size in px at 1080p (scaled by frame height).",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path to write the full JSON report.",
    )


class SceneLintTool(BaseTool):
    """Statically analyze scene code before rendering."""

    name = "scene_lint"
    description = (
        "Code-level structural QA for scene files BEFORE rendering: parses "
        "Python scenes with the real AST and JS/TS scenes with calibrated "
        "regexes, then flags undeclared asset references (render will fail), "
        "off-safe-area positions, font sizes below the project minimum, "
        "hardcoded colors outside the brand palette, repeated magic "
        "constants (drift risk), and non-deterministic calls (Math.random, "
        "Date.now, random.random...) that break re-render determinism. "
        "Catching errors here is 100x cheaper than after rendering."
    )
    input_model = SceneLintInput

    def is_read_only(self, arguments: SceneLintInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: SceneLintInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.static_scene_analyzer import (
            StaticSceneAnalyzer,
            find_scene_files,
            format_report_text,
        )

        # ---- Resolve scene files ---------------------------------------
        files: list[Path] = []
        for raw in arguments.paths:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(context.cwd) / path
            found = find_scene_files(path)
            if found:
                files.extend(found)
            else:
                return ToolResult(
                    output=(
                        f"scene_lint: no scene files found at {raw} "
                        "(supported: .py/.js/.jsx/.ts/.tsx)."
                    ),
                    is_error=True,
                )
        if not files:
            return ToolResult(output="scene_lint: no scene files found.", is_error=True)

        # ---- Declared assets -------------------------------------------
        declared: list[str] = list(arguments.assets or [])
        if arguments.assets_manifest:
            manifest_path = Path(arguments.assets_manifest)
            if not manifest_path.is_absolute():
                manifest_path = Path(context.cwd) / manifest_path
            if not manifest_path.is_file():
                return ToolResult(
                    output=f"scene_lint: assets manifest not found: {arguments.assets_manifest}",
                    is_error=True,
                )
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return ToolResult(
                    output=f"scene_lint: cannot parse assets manifest: {exc}",
                    is_error=True,
                )
            if isinstance(payload, list):
                declared.extend(str(item) for item in payload)
            elif isinstance(payload, dict) and isinstance(payload.get("assets"), list):
                declared.extend(str(item) for item in payload["assets"])

        # ---- Brand palette ----------------------------------------------
        palette: list[str] = list(arguments.palette or [])
        if not palette and arguments.palette_file:
            from openharness.services.pixel_qa import load_brand_palette

            palette_path = Path(arguments.palette_file)
            if not palette_path.is_absolute():
                palette_path = Path(context.cwd) / palette_path
            palette = load_brand_palette(palette_path, as_hex=True) or []
            if not palette:
                return ToolResult(
                    output=(
                        f"scene_lint: no hex colors found in {arguments.palette_file}."
                    ),
                    is_error=True,
                )

        analyzer = StaticSceneAnalyzer(
            frame_width=arguments.frame_width,
            frame_height=arguments.frame_height,
            min_font_px=arguments.min_font_px,
        )
        reports = []
        for file in files:
            reports.append(
                analyzer.analyze_file(
                    file, declared_assets=declared, palette_hex=palette
                )
            )

        text = format_report_text(reports)
        failed = [r.scene for r in reports if not r.passed]
        if failed:
            text += (
                "\n[scene_lint] blocking issue(s) in: "
                + ", ".join(failed)
                + " — fix the code BEFORE rendering (undeclared assets and "
                "non-determinism are hard failures)."
            )
        else:
            text += "\n[scene_lint] all scenes passed the structural checks."

        metadata: dict = {"reports": [r.as_dict() for r in reports]}
        if arguments.report_path:
            report_path = Path(arguments.report_path)
            if not report_path.is_absolute():
                report_path = Path(context.cwd) / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(metadata["reports"], indent=2), encoding="utf-8"
            )
            text += f"\n[scene_lint] full report written to {report_path}"
            metadata["report_path"] = str(report_path)

        return ToolResult(output=text, is_error=bool(failed), metadata=metadata)
