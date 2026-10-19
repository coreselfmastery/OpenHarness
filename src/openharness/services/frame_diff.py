"""Frame Diff Analyzer — motion-quality QA for rendered video.

A scene can look correct in a still frame yet stutter, jerk, or freeze
during playback. This module compares *consecutive* sampled frames with
two complementary signals:

- **SSIM** (structural similarity, Wang et al. 2004) — computed on
  grayscale with an 11x11 Gaussian window, implemented directly on
  top of ``cv2.GaussianBlur`` (no SciPy required). SSIM ~ 1.0 means
  the two frames are structurally identical; a sharp drop means a
  hard visual discontinuity.
- **Optical flow magnitude** (Farneback dense flow, computed on a
  downscaled grayscale copy for speed) — measures how many pixels
  actually moved and how far, which distinguishes "identical frames"
  (frozen) from "a rectangle slid 8 px" (intended motion).

Checks (each pass/fail with a confidence in [0, 1]):

1. ``frozen_motion`` — adjacent SSIM above ``frozen_ssim`` AND mean
   flow below ``flow_floor``: the render is repeating a frame.
2. ``stutter`` — a *run* of near-identical pairs (SSIM above
   ``stutter_ssim``) inside an otherwise moving segment: the motion
   stalls mid-scene.
3. ``jump_cut`` — SSIM below ``jump_cut_ssim`` between adjacent
   frames: a flash, a hard cut, or a dropped segment.
4. ``motion_speed`` — observed mean flow outside the expected
   ``[min_flow, max_flow]`` px/s window when the caller declares one
   (verifies that an intended animation actually happens at the
   expected speed).

Results are aggregated per scene (one video = one scene) into a
:class:`SceneMotionReport` with a weighted score, mirroring
:mod:`openharness.services.pixel_qa`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

DEFAULTS = {
    # Adjacent-pair SSIM at/above which the pair is "structurally identical".
    "frozen_ssim": 0.999,
    # Mean Farneback flow magnitude (px on the analysis scale) below which
    # nothing is moving.
    "flow_floor": 0.05,
    # A *run* of >= stutter_run pairs this similar flags a stall.
    "stutter_ssim": 0.995,
    "stutter_run": 3,
    # Adjacent SSIM below this is a hard visual discontinuity.
    "jump_cut_ssim": 0.35,
    # Analysis scale: frames are downscaled so max width <= this.
    "analysis_width": 480,
}

_CRITICAL_CHECKS = ("frozen_motion", "jump_cut")


def find_video_files(path: Path) -> list[Path]:
    """Return video files at ``path`` (a file, or a dir scanned shallowly)."""
    if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in _VIDEO_EXTENSIONS
        )
    return []


# ----------------------------------------------------------------------
# SSIM
# ----------------------------------------------------------------------
def ssim(gray_a: Any, gray_b: Any) -> float:
    """Structural similarity between two same-shaped gray float32 images.

    Classic Wang et al. formulation with an 11x11 Gaussian window
    (sigma 1.5) and the standard stabilising constants. Returns a
    float in [-1, 1]; 1.0 = identical.
    """
    import cv2
    import numpy as np

    a = np.asarray(gray_a, dtype=np.float64)
    b = np.asarray(gray_b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return 0.0

    c1 = (0.03 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = (11, 11)
    sigma = 1.5

    mu_a = cv2.GaussianBlur(a, kernel, sigma)
    mu_b = cv2.GaussianBlur(b, kernel, sigma)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, kernel, sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, kernel, sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, kernel, sigma) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    if den.size == 0:
        return 0.0
    score_map = num / den
    # Guard against NaN/Inf from degenerate denominators.
    score_map = np.nan_to_num(score_map, nan=0.0, posinf=1.0, neginf=-1.0)
    return float(np.clip(score_map.mean(), -1.0, 1.0))


def mean_flow_magnitude(gray_a: Any, gray_b: Any) -> float:
    """Mean Farneback optical-flow magnitude between two gray frames."""
    import cv2
    import numpy as np

    a = np.asarray(gray_a, dtype=np.uint8)
    b = np.asarray(gray_b, dtype=np.uint8)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.linalg.norm(flow, axis=2)
    return float(magnitude.mean())


def percentile_flow_magnitude(gray_a: Any, gray_b: Any, percentile: float = 99.0) -> float:
    """High-percentile Farneback flow magnitude between two gray frames.

    The MEAN flow dilutes object motion across the whole (mostly static)
    frame — a 1920-px scene with an object moving 80 px/s measured a mean
    flow of ~0.2 px and the speed check reported "animation missing".
    The 99th percentile tracks the fastest-moving hundredth of the frame,
    which approximates the OBJECT velocity (what the expected-motion
    window actually describes) for any object covering more than ~1% of
    the frame, instead of the background-averaged speed. p99 (not p90)
    because moving objects routinely cover less than a tenth of a frame.
    """
    import cv2
    import numpy as np

    a = np.asarray(gray_a, dtype=np.uint8)
    b = np.asarray(gray_b, dtype=np.uint8)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.linalg.norm(flow, axis=2)
    if magnitude.size == 0:
        return 0.0
    return float(np.percentile(magnitude, percentile))


@dataclass
class MotionIssue:
    """One detected motion problem at one adjacent-frame pair."""

    check: str
    frame_index: int
    timestamp_s: float
    message: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 2),
            "message": self.message,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SceneMotionReport:
    """Aggregated frame-diff result for one video (scene)."""

    scene: str
    video: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    frames_sampled: int = 0
    pairs_analyzed: int = 0
    mean_ssim: float = 0.0
    mean_flow: float = 0.0
    issues: list[MotionIssue] = field(default_factory=list)
    check_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        if self.pairs_analyzed == 0:
            return False
        return not any(
            issue.check in _CRITICAL_CHECKS
            or self.check_summary.get(issue.check, {}).get("severity") == "fail"
            for issue in self.issues
        )

    @property
    def score(self) -> float:
        """1.0 = clean motion; each issue subtracts its confidence."""
        if self.pairs_analyzed == 0:
            return 0.0
        penalty = 0.0
        for issue in self.issues:
            weight = 1.0 if issue.check in _CRITICAL_CHECKS else 0.5
            penalty += weight * issue.confidence
        return float(max(0.0, 1.0 - penalty / max(1, self.pairs_analyzed)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "video": self.video,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "duration_s": round(self.duration_s, 2),
            "frames_sampled": self.frames_sampled,
            "pairs_analyzed": self.pairs_analyzed,
            "mean_ssim": round(self.mean_ssim, 4),
            "mean_flow": round(self.mean_flow, 4),
            "passed": self.passed,
            "score": round(self.score, 3),
            "issues": [issue.as_dict() for issue in self.issues],
            "check_summary": self.check_summary,
        }


class FrameDiffAnalyzer:
    """Run SSIM + optical-flow motion checks over sampled frame pairs."""

    def __init__(self, **overrides: Any) -> None:
        self.config: dict[str, Any] = {**DEFAULTS}
        for key, value in overrides.items():
            if key in self.config:
                self.config[key] = value

    # ------------------------------------------------------------------
    def analyze_video(
        self,
        path: Path,
        *,
        sample_fps: float = 8.0,
        max_frames: int = 900,
        expected_flow_px_s: tuple[float, float] | None = None,
    ) -> SceneMotionReport:
        """Analyze one video; raises when the file cannot be opened."""
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"cannot open video: {path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            report = SceneMotionReport(
                scene=path.stem,
                video=str(path),
                width=width,
                height=height,
                fps=fps,
                duration_s=(total / fps) if total and fps else 0.0,
            )

            step = max(1, round(fps / max(0.1, sample_fps)))
            grays: list[Any] = []
            indices: list[int] = []
            index = 0
            while len(grays) < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if index % step == 0:
                    grays.append(self._to_analysis_gray(frame, np))
                    indices.append(index)
                index += 1
            report.frames_sampled = len(grays)
            if len(grays) < 2:
                report.check_summary = {
                    "open": {
                        "severity": "warn",
                        "detail": (
                            "fewer than 2 frames sampled; motion analysis "
                            "not possible"
                        ),
                    }
                }
                return report
            report.pairs_analyzed = len(grays) - 1

            ssims: list[float] = []
            flows: list[float] = []
            strong_flows: list[float] = []
            pair_issues: list[MotionIssue] = []
            identical_run = 0
            sample_dt = step / fps  # seconds between analyzed pairs

            for i in range(len(grays) - 1):
                pair_ssim = ssim(grays[i], grays[i + 1])
                pair_flow = mean_flow_magnitude(grays[i], grays[i + 1])
                ssims.append(pair_ssim)
                flows.append(pair_flow)
                strong_flows.append(percentile_flow_magnitude(grays[i], grays[i + 1]))
                frame_index = indices[i + 1]
                timestamp = frame_index / fps

                # --- frozen pair --------------------------------------
                if (
                    pair_ssim >= self.config["frozen_ssim"]
                    and pair_flow < self.config["flow_floor"]
                ):
                    confidence = 0.5 + 0.5 * min(
                        1.0,
                        max(
                            0.0,
                            (pair_ssim - self.config["frozen_ssim"])
                            / max(1e-6, 1.0 - self.config["frozen_ssim"]),
                        ),
                    )
                    pair_issues.append(
                        MotionIssue(
                            check="frozen_motion",
                            frame_index=frame_index,
                            timestamp_s=timestamp,
                            message=(
                                f"adjacent frames structurally identical "
                                f"(SSIM {pair_ssim:.4f}, flow {pair_flow:.3f} px)"
                            ),
                            confidence=confidence,
                        )
                    )

                # --- stutter run ---------------------------------------
                if pair_ssim >= self.config["stutter_ssim"]:
                    identical_run += 1
                    if identical_run >= int(self.config["stutter_run"]):
                        pair_issues.append(
                            MotionIssue(
                                check="stutter",
                                frame_index=frame_index,
                                timestamp_s=timestamp,
                                message=(
                                    f"{identical_run} consecutive near-identical "
                                    f"pairs (SSIM >= {self.config['stutter_ssim']}) "
                                    "- motion stalled mid-scene"
                                ),
                                confidence=min(
                                    1.0, 0.5 + 0.1 * identical_run
                                ),
                            )
                        )
                else:
                    identical_run = 0

                # --- jump cut / flash -----------------------------------
                if pair_ssim < self.config["jump_cut_ssim"]:
                    distance = self.config["jump_cut_ssim"] - pair_ssim
                    pair_issues.append(
                        MotionIssue(
                            check="jump_cut",
                            frame_index=frame_index,
                            timestamp_s=timestamp,
                            message=(
                                f"sharp visual discontinuity "
                                f"(SSIM {pair_ssim:.3f} < "
                                f"{self.config['jump_cut_ssim']}) - jump cut, "
                                "flash, or dropped segment"
                            ),
                            confidence=min(
                                1.0, 0.5 + distance / self.config["jump_cut_ssim"]
                            ),
                        )
                    )

            # --- intended-motion speed check ---------------------------
            if expected_flow_px_s is not None:
                low, high = expected_flow_px_s
                # Observed speed uses the p99 (object-region) flow, the
                # MEDIAN across pairs (robust to one bad pair), and is
                # converted to SOURCE-scale pixels: the analysis grid is
                # 480-px wide, so raw flow magnitudes are in ANALYSIS
                # pixels. The old mean-flow figure was diluted by the
                # static background (~250x here) AND unrescaled, so a
                # correct 80 px/s animation was reported as 0.2 px/s
                # "below expected minimum" on every check.
                analysis_width = float(self.config["analysis_width"])
                scale_to_source = (
                    report.width / analysis_width
                    if report.width and analysis_width
                    else 1.0
                )
                observed = float(np.median(strong_flows)) / max(sample_dt, 1e-6)
                observed *= scale_to_source
                if observed < low:
                    pair_issues.append(
                        MotionIssue(
                            check="motion_speed",
                            frame_index=indices[-1],
                            timestamp_s=report.duration_s,
                            message=(
                                f"observed object-region motion {observed:.1f} px/s "
                                f"(source scale, p99 flow) below expected minimum "
                                f"{low:.1f} px/s - intended "
                                "animation may be missing"
                            ),
                            confidence=min(1.0, 0.5 + (low - observed) / max(low, 1e-6)),
                        )
                    )
                elif observed > high:
                    pair_issues.append(
                        MotionIssue(
                            check="motion_speed",
                            frame_index=indices[-1],
                            timestamp_s=report.duration_s,
                            message=(
                                f"observed object-region motion {observed:.1f} px/s "
                                f"(source scale, p99 flow) above expected maximum "
                                f"{high:.1f} px/s - animation "
                                "faster than designed"
                            ),
                            confidence=min(1.0, 0.5 + (observed - high) / max(high, 1e-6)),
                        )
                    )

            report.issues = pair_issues
            report.mean_ssim = float(np.mean(ssims)) if ssims else 0.0
            report.mean_flow = float(np.mean(flows)) if flows else 0.0
            report.check_summary = self._summarize(pair_issues, report.pairs_analyzed)
            return report
        finally:
            capture.release()

    # ------------------------------------------------------------------
    def _to_analysis_gray(self, frame: Any, np: Any) -> Any:
        """Downscale + grayscale a BGR frame for the analysis pass."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target = int(self.config["analysis_width"])
        height, width = gray.shape[:2]
        if width > target:
            new_height = max(1, int(height * target / width))
            gray = cv2.resize(gray, (target, new_height), interpolation=cv2.INTER_AREA)
        return gray

    @staticmethod
    def _summarize(
        issues: list[MotionIssue], pairs: int
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        by_check: dict[str, list[MotionIssue]] = {}
        for issue in issues:
            by_check.setdefault(issue.check, []).append(issue)
        for check, group in by_check.items():
            worst = max(issue.confidence for issue in group)
            summary[check] = {
                "severity": "fail" if check in _CRITICAL_CHECKS else "warn",
                "count": len(group),
                "worst_confidence": round(worst, 3),
                "first_frame": group[0].frame_index,
                "detail": group[0].message,
            }
        summary["pairs_analyzed"] = {"severity": "info", "count": pairs}
        return summary


def format_report_text(reports: list[SceneMotionReport]) -> str:
    """Render one or more motion reports as agent-readable text."""
    if not reports:
        return "frame_diff: nothing to analyze."
    lines: list[str] = []
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        lines.append(
            f"[frame_diff] {report.scene}: {status} "
            f"(score {report.score:.2f}, {report.frames_sampled} frames, "
            f"{report.pairs_analyzed} pairs, mean SSIM {report.mean_ssim:.3f}, "
            f"mean flow {report.mean_flow:.2f} px)"
        )
        for check, info in sorted(report.check_summary.items()):
            if info.get("severity") == "info":
                continue
            # ``.get`` (not ``[]``): "open" entries (unreadable video /
            # fewer-than-2-frames) carry only severity+detail — the old
            # indexing of 'count'/'first_frame' raised KeyError out of the
            # FORMATTER and aborted the whole tool call, losing the good
            # videos' results and the JSON report.
            lines.append(
                f"  - {check} [{info.get('severity', '?')}] x{info.get('count', 0)} "
                f"@frame {info.get('first_frame', 0)}: {info.get('detail', '')}"
            )
        if not report.issues:
            lines.append("  - motion checks clean: no freezes, stalls, or cuts.")
    return "\n".join(lines)
