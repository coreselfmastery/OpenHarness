"""frame_diff tool — SSIM + optical-flow motion QA over rendered video.

Agent-facing wrapper around :mod:`openharness.services.frame_diff`.
A scene can look correct in a still frame yet stutter, jerk, or freeze
during playback — this tool compares consecutive sampled frames
(structural similarity + dense optical flow) and flags frozen motion,
mid-scene stalls, jump cuts/flashes, and animations that run at the
wrong speed. Deterministic, offline, reproducible; complements
``pixel_qa`` (per-frame defects) and ``vlm_qa`` (semantic judgment).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class FrameDiffInput(BaseModel):
    """Arguments for the frame_diff tool."""

    paths: list[str] = Field(
        description=(
            "Video file path(s) or directory(s) of rendered scenes "
            "(.mp4/.mov/.mkv/.webm/.avi)."
        )
    )
    sample_fps: float = Field(
        default=8.0,
        ge=0.1,
        le=60.0,
        description="Frames per second to sample for pair analysis.",
    )
    max_frames: int = Field(
        default=900,
        ge=2,
        le=10_000,
        description="Hard cap on sampled frames per video.",
    )
    expected_min_flow_px_s: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional: minimum intended motion speed in px/s. When set "
            "together with expected_max_flow_px_s, the observed optical-flow "
            "speed is verified against this window (did the slide-in "
            "actually happen, at the designed speed?)."
        ),
    )
    expected_max_flow_px_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional: maximum intended motion speed in px/s.",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional path to write the full JSON report.",
    )


class FrameDiffTool(BaseTool):
    """Motion-quality QA: SSIM + optical flow between adjacent frames."""

    name = "frame_diff"
    description = (
        "Analyze MOTION quality of rendered video: SSIM between adjacent "
        "frames catches frozen frames and stutter (runs of near-identical "
        "pairs); sharp SSIM drops catch jump cuts and flashes; Farneback "
        "optical flow verifies intended motion happens at the expected "
        "speed (pass expected_min/max_flow_px_s). Fast, offline, "
        "deterministic. Run after pixel_qa to catch playback-only defects "
        "a still frame cannot show."
    )
    input_model = FrameDiffInput

    def is_read_only(self, arguments: FrameDiffInput) -> bool:
        del arguments
        return True  # reads videos; writes only an optional report file

    async def execute(
        self, arguments: FrameDiffInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            import cv2  # noqa: F401
        except ImportError:
            return ToolResult(
                output=(
                    "frame_diff: OpenCV is not installed. Install it with "
                    "`pip install opencv-python-headless` (or the "
                    "`openharness-ai[pixel-qa]` extra)."
                ),
                is_error=True,
            )

        from openharness.services.frame_diff import (
            FrameDiffAnalyzer,
            SceneMotionReport,
            find_video_files,
            format_report_text,
        )

        if (arguments.expected_min_flow_px_s is None) != (
            arguments.expected_max_flow_px_s is None
        ):
            return ToolResult(
                output=(
                    "frame_diff: pass BOTH expected_min_flow_px_s and "
                    "expected_max_flow_px_s (or neither)."
                ),
                is_error=True,
            )

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
                        f"frame_diff: no videos found at {raw} (supported: "
                        ".mp4/.mov/.mkv/.webm/.avi; directories are scanned "
                        "one level deep)."
                    ),
                    is_error=True,
                )
        if not videos:
            return ToolResult(output="frame_diff: no videos found.", is_error=True)

        expected: tuple[float, float] | None = None
        if arguments.expected_min_flow_px_s is not None:
            expected = (
                arguments.expected_min_flow_px_s,
                float(arguments.expected_max_flow_px_s or 0.0),
            )
            if expected[0] > expected[1]:
                return ToolResult(
                    output=(
                        f"frame_diff: expected_min_flow_px_s ({expected[0]}) is "
                        f"greater than expected_max_flow_px_s ({expected[1]}) — "
                        "the expected-motion window is empty."
                    ),
                    is_error=True,
                )

        analyzer = FrameDiffAnalyzer()
        reports: list[SceneMotionReport] = []
        for video in videos:
            try:
                # SSIM + Farneback optical flow over hundreds of frames is
                # pure CPU — run it off the event loop.
                reports.append(
                    await asyncio.to_thread(
                        analyzer.analyze_video,
                        video,
                        sample_fps=arguments.sample_fps,
                        max_frames=arguments.max_frames,
                        expected_flow_px_s=expected,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed = SceneMotionReport(scene=video.stem, video=str(video))
                failed.check_summary = {
                    "open": {"severity": "fail", "detail": f"{type(exc).__name__}: {exc}"}
                }
                reports.append(failed)

        # Progress tracker: motion QA is part of the qa stage.
        try:
            from openharness.services.progress_tracker import get_progress_tracker

            tracker = get_progress_tracker()
            for report in reports:
                if not report.pairs_analyzed:
                    continue
                tracker.mark(
                    report.scene, "qa", detail=f"frame-diff score {report.score}"
                )
                if not report.passed:
                    tracker.mark(report.scene, "failed", detail="frame-diff hard failure")
        except Exception:
            log.debug("frame_diff: progress tracker update failed", exc_info=True)

        text = format_report_text(reports)
        failed_scenes = [r.scene for r in reports if not r.passed]
        if failed_scenes:
            text += (
                "\n[frame_diff] motion failure(s): "
                + ", ".join(failed_scenes)
                + " — frozen frames / jump cuts must be fixed before review."
            )
        else:
            text += "\n[frame_diff] all scenes passed the motion checks."

        metadata: dict = {"reports": [r.as_dict() for r in reports]}
        if arguments.report_path:
            report_path = Path(arguments.report_path)
            if not report_path.is_absolute():
                report_path = Path(context.cwd) / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(metadata["reports"], indent=2), encoding="utf-8"
            )
            text += f"\n[frame_diff] full report written to {report_path}"
            metadata["report_path"] = str(report_path)

        return ToolResult(output=text, is_error=bool(failed_scenes), metadata=metadata)
