"""Pixel QA tool — deterministic OpenCV inspection of rendered frames.

Agent-facing wrapper around :mod:`openharness.services.pixel_qa`.
Given rendered video(s) (files or a render directory) it runs the
frame-level battery — black/blank frames, frozen frames, brand-palette
color drift, text-size/off-screen risk, encoding artifacts — and
returns a per-scene pass/fail report with confidences.

Scene stage integration: each inspected video is marked ``qa`` in the
progress tracker; a hard pixel failure marks the scene ``failed``
(transition ``rendered -> failed``), while a pass leaves promotion to
the LLM reviewer (`qa_review`) — pixel QA is one input, not the whole
verdict.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class PixelQAInput(BaseModel):
    """Arguments for the pixel_qa tool."""

    paths: list[str] = Field(
        description=(
            "Video file path(s) or directory(s) of rendered scenes "
            "(.mp4/.mov/.mkv/.webm/.avi)."
        )
    )
    palette_file: str | None = Field(
        default=None,
        description=(
            "Brand palette file: brand.yaml/brand.yml/brand.json, or an "
            "OpenVideoHarness styles/<slug>/tokens.json (palette.bg/fg/"
            "accent/extra hex colors)."
        ),
    )
    palette: list[str] | None = Field(
        default=None,
        description="Inline brand palette as hex colors (#F2F0EB, #E1251B, ...).",
    )
    sample_fps: float = Field(
        default=4.0,
        ge=0.1,
        le=60.0,
        description="Frames per second to sample from each video.",
    )
    max_frames: int = Field(
        default=600,
        ge=1,
        le=10_000,
        description="Hard cap on sampled frames per video.",
    )
    checks: list[str] | None = Field(
        default=None,
        description=(
            "Subset of checks to run: black_frame, frozen_frame, "
            "color_drift, text_size, artifacts. Default: all."
        ),
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path to write the full JSON report.",
    )


class PixelQATool(BaseTool):
    """Inspect rendered frames with deterministic OpenCV checks."""

    name = "pixel_qa"
    description = (
        "Deterministic pixel-level QA over rendered video: black/blank frame "
        "detection, frozen-frame detection, brand-palette color drift "
        "(brand.yaml or OpenVideoHarness styles/*/tokens.json), text-too-small/"
        "off-screen risk via edge clustering, and encoding-artifact/noise "
        "analysis. Fast, offline, reproducible — every check returns pass/fail "
        "with a confidence, aggregated per scene. Run it on renders before the "
        "LLM qa_review to catch mechanical defects for free."
    )
    input_model = PixelQAInput

    def is_read_only(self, arguments: PixelQAInput) -> bool:
        del arguments
        return True  # reads videos; writes only an optional report file

    async def execute(
        self, arguments: PixelQAInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            import cv2  # noqa: F401
        except ImportError:
            return ToolResult(
                output=(
                    "pixel_qa: OpenCV is not installed. Install it with "
                    "`pip install opencv-python-headless` (or the "
                    "`openharness-ai[pixel-qa]` extra)."
                ),
                is_error=True,
            )

        from openharness.services.pixel_qa import (
            PixelQA,
            find_video_files,
            format_report_text,
            hex_to_bgr,
            load_brand_palette,
        )

        # ---- Resolve videos -------------------------------------------
        videos: list[Path] = []
        for raw in arguments.paths:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(context.cwd) / path
            found = find_video_files(path)
            if found:
                videos.extend(found)
            else:
                return ToolResult(
                    output=(
                        f"pixel_qa: no videos found at {raw} (supported: "
                        ".mp4/.mov/.mkv/.webm/.avi; directories are scanned "
                        "one level deep)."
                    ),
                    is_error=True,
                )
        if not videos:
            return ToolResult(output="pixel_qa: no videos found.", is_error=True)

        palette: list[tuple[int, int, int]] = []
        if arguments.palette:
            for hex_color in arguments.palette:
                try:
                    palette.append(hex_to_bgr(hex_color))
                except (ValueError, TypeError):
                    return ToolResult(
                        output=f"pixel_qa: invalid hex color {hex_color!r} "
                        "(expected #RRGGBB).",
                        is_error=True,
                    )
        elif arguments.palette_file:
            palette_path = Path(arguments.palette_file)
            if not palette_path.is_absolute():
                palette_path = Path(context.cwd) / palette_path
            palette = load_brand_palette(palette_path)
            if not palette:
                return ToolResult(
                    output=(
                        f"pixel_qa: no hex colors found in {arguments.palette_file} "
                        "(expected #RRGGBB values, e.g. an OpenVideoHarness "
                        "tokens.json palette section)."
                    ),
                    is_error=True,
                )

        enabled = set(arguments.checks) if arguments.checks else None
        qa = PixelQA()
        reports = []
        for video in videos:
            try:
                # OpenCV inspection (incl. Farneback optical flow over up
                # to 900 frames) is pure CPU — run it off the event loop
                # so minutes of analysis cannot freeze the whole agent.
                reports.append(
                    await asyncio.to_thread(
                        qa.inspect_video,
                        video,
                        palette=palette or None,
                        sample_fps=arguments.sample_fps,
                        max_frames=arguments.max_frames,
                        enabled_checks=enabled,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                from openharness.services.pixel_qa import ScenePixelReport

                failed = ScenePixelReport(scene=video.stem, video=str(video))
                failed.check_summary = {
                    "open": {"severity": "fail", "detail": f"{type(exc).__name__}: {exc}"}
                }
                reports.append(failed)

        # ---- Progress tracker integration --------------------------------
        try:
            from openharness.services.progress_tracker import get_progress_tracker

            tracker = get_progress_tracker()
            for report in reports:
                # Mark even when nothing was sampled: an unopenable video
                # reports passed=False (severity "fail" on the "open"
                # entry) and the scene must show up as failed, not stay
                # silently unmarked in the tracker.
                if not report.frames_sampled and report.passed:
                    continue
                tracker.mark(report.scene, "qa", detail=f"pixel-qa score {report.score}")
                if not report.passed:
                    tracker.mark(report.scene, "failed", detail="pixel-qa hard failure")
        except Exception:
            log.debug("pixel_qa: progress tracker update failed", exc_info=True)

        # ---- Output -------------------------------------------------------
        text = format_report_text(reports)
        failed_scenes = [r.scene for r in reports if not r.passed]
        if failed_scenes:
            text += (
                "\n[pixel_qa] hard-failed scene(s): "
                + ", ".join(failed_scenes)
                + " — fix and re-render before LLM review."
            )
        else:
            text += "\n[pixel_qa] all scenes passed the deterministic checks."

        metadata: dict = {"reports": [r.as_dict() for r in reports]}
        if arguments.report_path:
            report_path = Path(arguments.report_path)
            if not report_path.is_absolute():
                report_path = Path(context.cwd) / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(metadata["reports"], indent=2), encoding="utf-8"
            )
            text += f"\n[pixel_qa] full report written to {report_path}"
            metadata["report_path"] = str(report_path)

        return ToolResult(output=text, is_error=bool(failed_scenes), metadata=metadata)
