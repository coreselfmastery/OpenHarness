"""Deterministic pixel-level QA for rendered video — the Pixel QA Tool.

Inspects every (sampled) rendered frame with fast, deterministic
OpenCV checks — no LLM involved, no network, reproducible on any host:

1. **Black / white / blank frame** — mean brightness and per-frame
   standard deviation catch dead renders and dropped fills.
2. **Frozen frame** — frame-to-frame absolute difference catches a
   stuck pipeline repeating one frame.
3. **Color drift** — per-frame pixel palette membership against the
   brand palette (``brand.yaml`` / ``brand.yml`` / ``brand.json``, or
   an OpenVideoHarness ``styles/<name>/tokens.json`` whose
   ``palette`` holds ``bg``/``fg``/``accent``/``extra`` hex colors).
4. **Text too small / off-screen** — Canny edges + morphological
   line-box clustering measure candidate text-box heights and detect
   boxes clipped at the frame border. (Heuristic — no OCR.)
5. **Noise / encoding artifacts** — 8-px macroblock-boundary gradient
   ratio catches blocky encodes; Laplacian variance catches
   over-smoothed or noisy frames.

Each check returns pass/fail with a confidence in [0, 1]; results are
aggregated per scene (one video = one scene) with a weighted score.
Hard failures (black/blank/frozen) fail the scene; the others are
reported as warnings unless they exceed their thresholds.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{6})$")

# Defaults tuned for 1080p-ish renders; scales with frame height.
DEFAULTS = {
    "black_mean": 8.0,          # gray mean below -> black frame
    "white_mean": 247.0,        # gray mean above -> white frame
    "blank_std": 3.0,           # gray std below -> uniform/blank frame
    "frozen_diff": 1.5,         # mean absdiff below -> frozen pair
    "min_in_palette": 0.60,     # fraction of pixels inside palette tolerance
    "palette_tolerance": 60.0,  # BGR euclidean distance to nearest palette color
    "min_text_height": 12,      # px at 1080p, scaled by height/1080
    "blockiness_ratio": 2.2,    # boundary/interior gradient ratio -> blocky
    "laplacian_low": 15.0,      # below -> smeared / over-smooth
}
_CRITICAL_CHECKS = ("black_frame", "frozen_frame")


@dataclass
class FrameIssue:
    """One detected problem at one (sampled) frame."""

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
class ScenePixelReport:
    """Aggregated pixel-QA result for one video (scene)."""

    scene: str
    video: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    frames_sampled: int = 0
    issues: list[FrameIssue] = field(default_factory=list)
    check_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(
            issue.check in _CRITICAL_CHECKS
            or self.check_summary.get(issue.check, {}).get("severity") == "fail"
            for issue in self.issues
        )

    @property
    def score(self) -> float:
        """1.0 = clean; each issue subtracts a confidence-weighted penalty."""
        if self.frames_sampled == 0:
            return 0.0
        penalty = sum(min(1.0, issue.confidence) * 0.05 for issue in self.issues)
        return round(max(0.0, 1.0 - penalty), 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "video": self.video,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "duration_s": round(self.duration_s, 2),
            "frames_sampled": self.frames_sampled,
            "passed": self.passed,
            "score": self.score,
            "checks": self.check_summary,
            "issues": [issue.as_dict() for issue in self.issues],
        }


# ----------------------------------------------------------------------
# Brand palette loading
# ----------------------------------------------------------------------

def load_brand_palette(
    source: str | Path, *, as_hex: bool = False
) -> list[tuple[int, int, int]] | list[str]:
    """Load hex colors from a brand file or an OpenVideoHarness tokens.json.

    Accepts ``brand.yaml``/``brand.yml``/``brand.json`` (any mapping whose
    nested values include ``#rrggbb`` strings) and the OpenVideoHarness
    ``styles/<slug>/tokens.json`` layout (``palette.bg/fg/accent/extra``).
    Returns BGR tuples (OpenCV order) by default, or ``#RRGGBB`` strings
    when ``as_hex=True``; empty list when nothing loads.
    """
    path = Path(source)
    if not path.is_file():
        log.warning("pixel-qa: palette file not found: %s", path)
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    data: Any = None
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # optional at runtime, in dependencies already

            data = yaml.safe_load(text)
        except ImportError:
            data = None
    else:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    if not isinstance(data, dict):
        return []
    return palette_from_mapping(data, as_hex=as_hex)


def palette_from_mapping(
    data: dict[str, Any], *, as_hex: bool = False
) -> list[tuple[int, int, int]] | list[str]:
    """Collect every ``#rrggbb`` string from a nested mapping.

    Returns BGR tuples (OpenCV order) by default, or the deduplicated
    ``#RRGGBB`` strings themselves when ``as_hex=True``.
    """
    hexes: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            candidate = node.strip()
            if _HEX_COLOR_RE.match(candidate):
                hexes.append(candidate)

    _walk(data)
    seen: set[str] = set()
    unique_hexes: list[str] = []
    for hex_color in hexes:
        if hex_color.lower() in seen:
            continue
        seen.add(hex_color.lower())
        unique_hexes.append(hex_color)
    if as_hex:
        return unique_hexes
    return [hex_to_bgr(hex_color) for hex_color in unique_hexes]


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = int(hex_color.lstrip("#"), 16)
    r = (value >> 16) & 0xFF
    g = (value >> 8) & 0xFF
    b = value & 0xFF
    return (b, g, r)


# ----------------------------------------------------------------------
# The inspector
# ----------------------------------------------------------------------

class PixelQA:
    """OpenCV-based deterministic frame inspection for rendered video."""

    def __init__(self, **thresholds: float) -> None:
        self.thresholds = {**DEFAULTS, **{k: float(v) for k, v in thresholds.items()}}

    # -- public entry ---------------------------------------------------

    def inspect_video(
        self,
        video_path: Path,
        *,
        palette: list[tuple[int, int, int]] | None = None,
        sample_fps: float = 4.0,
        max_frames: int = 600,
        enabled_checks: set[str] | None = None,
    ) -> ScenePixelReport:
        """Inspect one video; one report per scene."""
        import cv2
        import numpy as np

        report = ScenePixelReport(scene=video_path.stem, video=str(video_path))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            report.check_summary = {"open": {"severity": "fail", "detail": "cannot open video"}}
            return report
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            report.fps = float(fps)
            report.duration_s = (frame_count / fps) if fps > 0 else 0.0

            step = max(1.0, fps / max(0.1, sample_fps)) if fps > 0 else 1.0
            enabled = enabled_checks or {
                "black_frame", "frozen_frame", "color_drift", "text_size", "artifacts"
            }

            previous_gray: Any = None
            previous_index = -1
            sampled = 0
            counters: dict[str, int] = {name: 0 for name in enabled}
            index = 0
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if index % round(step) == 0 or step < 1.0:
                    self._inspect_frame(
                        frame,
                        index=index,
                        fps=fps,
                        palette=palette,
                        enabled=enabled,
                        report=report,
                        counters=counters,
                        previous_gray=previous_gray,
                        previous_index=previous_index,
                        np=np,
                        cv2=cv2,
                    )
                    previous_gray = cv2.cvtColor(
                        cv2.resize(frame, (320, max(1, int(320 * frame.shape[0] / frame.shape[1])))),
                        cv2.COLOR_BGR2GRAY,
                    )
                    previous_index = index
                    sampled += 1
                index += 1
            report.frames_sampled = sampled
            report.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            report.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            for name in enabled:
                if name == "color_drift" and not palette:
                    report.check_summary[name] = {
                        "severity": "skipped",
                        "detail": "no brand palette provided",
                    }
                    continue
                report.check_summary[name] = {
                    "severity": "fail" if counters.get(name) else "pass",
                    "frames_flagged": counters.get(name, 0),
                    "frames_sampled": sampled,
                }
            return report
        finally:
            capture.release()

    # -- per-frame checks --------------------------------------------------

    def _inspect_frame(
        self,
        frame: Any,
        *,
        index: int,
        fps: float,
        palette: list[tuple[int, int, int]] | None,
        enabled: set[str],
        report: ScenePixelReport,
        counters: dict[str, int],
        previous_gray: Any,
        previous_index: int,
        np: Any,
        cv2: Any,
    ) -> None:
        t = self.thresholds
        timestamp = (index / fps) if fps > 0 else float(index)
        height, width = frame.shape[:2]
        scale = max(0.4, height / 1080.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. black / white / blank ---------------------------------------
        if "black_frame" in enabled:
            mean = float(gray.mean())
            std = float(gray.std())
            if mean < t["black_mean"]:
                report.issues.append(FrameIssue(
                    "black_frame", index, timestamp,
                    f"black frame (mean brightness {mean:.1f} < {t['black_mean']})",
                    min(1.0, (t["black_mean"] - mean) / max(1.0, t["black_mean"]) + 0.5),
                ))
                counters["black_frame"] += 1
            elif mean > t["white_mean"]:
                report.issues.append(FrameIssue(
                    "black_frame", index, timestamp,
                    f"blown-out white frame (mean brightness {mean:.1f})",
                    0.9,
                ))
                counters["black_frame"] += 1
            elif std < t["blank_std"]:
                report.issues.append(FrameIssue(
                    "black_frame", index, timestamp,
                    f"blank uniform frame (std {std:.2f} < {t['blank_std']})",
                    min(1.0, (t["blank_std"] - std) / max(0.1, t["blank_std"]) + 0.5),
                ))
                counters["black_frame"] += 1

        # 2. frozen frame -------------------------------------------------
        if "frozen_frame" in enabled and previous_gray is not None:
            if previous_gray.shape != gray.shape:
                resized = cv2.resize(gray, (previous_gray.shape[1], previous_gray.shape[0]))
            else:
                resized = gray
            diff = float(cv2.absdiff(resized, previous_gray).mean())
            if diff < t["frozen_diff"]:
                report.issues.append(FrameIssue(
                    "frozen_frame", index, timestamp,
                    f"frozen frame (diff {diff:.2f} from frame {previous_index} "
                    f"< {t['frozen_diff']})",
                    min(1.0, (t["frozen_diff"] - diff) / max(0.1, t["frozen_diff"]) + 0.5),
                ))
                counters["frozen_frame"] += 1

        # 3. color drift vs brand palette ----------------------------------
        if "color_drift" in enabled and palette:
            small = cv2.resize(frame, (160, max(1, int(160 * height / width))))
            pixels = small.reshape(-1, 3).astype(np.int16)
            palette_array = np.array(palette, dtype=np.int16)
            distances = np.linalg.norm(
                pixels[:, None, :] - palette_array[None, :, :], axis=2
            )
            min_distance = distances.min(axis=1)
            in_palette = float((min_distance <= t["palette_tolerance"]).mean())
            if in_palette < t["min_in_palette"]:
                report.issues.append(FrameIssue(
                    "color_drift", index, timestamp,
                    f"color drift: only {in_palette:.0%} of pixels within brand "
                    f"palette tolerance (min {t['min_in_palette']:.0%})",
                    min(1.0, (t["min_in_palette"] - in_palette) / max(0.01, t["min_in_palette"]) + 0.4),
                ))
                counters["color_drift"] += 1

        # 4. text too small / off-screen (heuristic) -----------------------
        if "text_size" in enabled:
            min_height = t["min_text_height"] * scale
            edges = cv2.Canny(gray, 80, 200)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(9, int(width * 0.012)), 3)
            )
            merged = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(
                merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            small_boxes = 0
            clipped = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < width * height * 0.00004:
                    continue  # noise specks
                if h < min_height and w > h * 2:
                    small_boxes += 1
                margin = 2
                if (x <= margin or y <= margin
                        or x + w >= width - margin or y + h >= height - margin) \
                        and w * h > width * height * 0.0004:
                    clipped += 1
            if small_boxes:
                report.issues.append(FrameIssue(
                    "text_size", index, timestamp,
                    f"{small_boxes} text-like box(es) below {min_height:.0f}px "
                    "minimum height",
                    0.65,
                ))
                counters["text_size"] += 1
            if clipped:
                report.issues.append(FrameIssue(
                    "text_size", index, timestamp,
                    f"{clipped} content box(es) clipped at the frame border "
                    "(off-screen risk)",
                    0.6,
                ))
                counters["text_size"] += 1

        # 5. noise / encoding artifacts -------------------------------------
        if "artifacts" in enabled:
            blockiness = self._blockiness(gray, np, cv2)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blockiness > t["blockiness_ratio"]:
                report.issues.append(FrameIssue(
                    "artifacts", index, timestamp,
                    f"blocky encoding artifacts (boundary/interior gradient "
                    f"ratio {blockiness:.2f} > {t['blockiness_ratio']})",
                    min(1.0, (blockiness - t["blockiness_ratio"]) / t["blockiness_ratio"] + 0.5),
                ))
                counters["artifacts"] += 1
            elif lap_var < t["laplacian_low"]:
                report.issues.append(FrameIssue(
                    "artifacts", index, timestamp,
                    f"smeared/over-smoothed frame (Laplacian variance "
                    f"{lap_var:.1f} < {t['laplacian_low']})",
                    0.6,
                ))
                counters["artifacts"] += 1

    @staticmethod
    def _blockiness(gray: Any, np: Any, cv2: Any) -> float:
        """Macroblock-boundary sharpness ratio (high -> blocky encode)."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.abs(gx) + np.abs(gy)
        h, w = gray.shape
        col_boundary = gradient[:, 7::8].mean() if w > 8 else 0.0
        row_boundary = gradient[7::8, :].mean() if h > 8 else 0.0
        boundary = (col_boundary + row_boundary) / 2.0
        mask = np.ones_like(gradient, dtype=bool)
        mask[:, 7::8] = False
        mask[7::8, :] = False
        interior = gradient[mask].mean() if mask.any() else 1.0
        return float(boundary / max(1.0, interior))


# ----------------------------------------------------------------------
# Helpers for tools
# ----------------------------------------------------------------------

def find_video_files(path: Path) -> list[Path]:
    """Resolve a video path, a directory of videos, or a glob-ish path."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in _VIDEO_EXTENSIONS
        )
    return []


def format_report_text(reports: list[ScenePixelReport]) -> str:
    """Human-readable multi-scene summary."""
    if not reports:
        return "[pixel_qa] no videos inspected."
    lines = [f"[pixel_qa] {len(reports)} scene(s) inspected"]
    for report in reports:
        verdict = "PASS" if report.passed else "FAIL"
        lines.append(
            f"  {report.scene}: {verdict}  score={report.score:.2f}  "
            f"{report.width}x{report.height} @ {report.fps:.0f}fps  "
            f"{report.frames_sampled} frames sampled"
        )
        for name, summary in report.check_summary.items():
            severity = summary.get("severity", "?")
            if severity == "pass":
                lines.append(f"    ok    {name}")
            elif severity == "skipped":
                lines.append(f"    skip  {name} ({summary.get('detail', '')})")
            else:
                lines.append(
                    f"    FAIL  {name}: {summary.get('frames_flagged', 0)} "
                    f"frame(s) flagged of {summary.get('frames_sampled', 0)}"
                )
        for issue in report.issues[:12]:
            lines.append(
                f"      [{issue.timestamp_s:7.2f}s #{issue.frame_index}] "
                f"{issue.check}: {issue.message} (conf {issue.confidence:.2f})"
            )
        if len(report.issues) > 12:
            lines.append(f"      ... and {len(report.issues) - 12} more issue(s)")
    return "\n".join(lines)
