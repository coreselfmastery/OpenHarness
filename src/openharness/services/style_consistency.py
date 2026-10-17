"""Style consistency checker — one visual language across all scenes.

In a long video, scenes drift: scene 14 uses ``#2A6FF2`` while the
brand says ``#2F6BEE``; scene 7 eases with ``cubic-bezier(0.55,0,1,0.45)``
while every other scene uses ``cubic-bezier(0.4,0,0.2,1)``. Viewers
feel it even when they cannot name it.

This module extracts the style DNA from each scene's *code* — hex
colors, font families, font sizes, cubic-bezier easings, transition
durations — and checks it two ways:

1. **Against the reference** (brand.yaml / styles/*/tokens.json /
   STYLE.md, or inline values): any color/font/easing outside the
   reference is flagged with the scene and the offending token.
2. **Across scenes** (drift): tokens used by only some scenes while
   the project clearly standardizes on others — the early-warning
   signal before drift becomes visible on screen.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openharness.services.style_memory import StyleMemory

log = logging.getLogger(__name__)

_SCENE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".html", ".css"}

_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})\b")
_FONT_RE = re.compile(
    r"font(?:Family|\s*-\s*family)?\s*[:=]\s*['\"]([^'\"]{2,60})['\"]",
    re.IGNORECASE,
)
_EASING_RE = re.compile(r"cubic-bezier\(\s*([0-9.,\s-]+)\s*\)")
_FONT_SIZE_RE = re.compile(r"font(?:Size|\s*-\s*size)?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)")
_TRANSITION_RE = re.compile(
    r"(?:transition|animation)\s*[:=]\s*['\"]?([a-zA-Z-]+)\s+(\d+(?:\.\d+)?)s"
)


@dataclass
class StyleTokens:
    """Style DNA extracted from one scene's code."""

    colors: dict[str, int] = field(default_factory=dict)      # hex -> count
    fonts: dict[str, int] = field(default_factory=dict)       # family -> count
    font_sizes: dict[float, int] = field(default_factory=dict)
    easings: dict[str, int] = field(default_factory=dict)     # normalized -> count
    transitions: dict[str, int] = field(default_factory=dict)  # "type duration" -> count

    def as_dict(self) -> dict[str, Any]:
        return {
            "colors": dict(sorted(self.colors.items(), key=lambda kv: -kv[1])),
            "fonts": dict(sorted(self.fonts.items(), key=lambda kv: -kv[1])),
            "font_sizes": {str(k): v for k, v in sorted(self.font_sizes.items())},
            "easings": dict(sorted(self.easings.items(), key=lambda kv: -kv[1])),
            "transitions": dict(sorted(self.transitions.items(), key=lambda kv: -kv[1])),
        }


@dataclass
class ConsistencyIssue:
    """One style inconsistency found in one scene."""

    scene: str
    kind: str          # palette_deviation / font_deviation / easing_deviation / ...
    message: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "kind": self.kind,
            "message": self.message,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ConsistencyReport:
    """Aggregated style-consistency result for a set of scenes."""

    scenes_checked: int = 0
    reference: dict[str, Any] = field(default_factory=dict)
    per_scene_tokens: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[ConsistencyIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(
            issue.kind in {"palette_deviation", "font_deviation", "easing_deviation"}
            for issue in self.issues
        )

    @property
    def score(self) -> float:
        hard = sum(
            issue.confidence
            for issue in self.issues
            if issue.kind
            in {"palette_deviation", "font_deviation", "easing_deviation"}
        )
        soft = sum(
            issue.confidence
            for issue in self.issues
            if issue.kind not in {"palette_deviation", "font_deviation", "easing_deviation"}
        )
        return float(max(0.0, 1.0 - (hard * 0.25 + soft * 0.05)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenes_checked": self.scenes_checked,
            "reference": self.reference,
            "per_scene_tokens": self.per_scene_tokens,
            "issues": [issue.as_dict() for issue in self.issues],
            "passed": self.passed,
            "score": round(self.score, 3),
        }


def extract_tokens(source: str) -> StyleTokens:
    """Pull style tokens out of scene code (language-agnostic regexes)."""
    tokens = StyleTokens()
    for match in _HEX_RE.finditer(source):
        tokens.colors[match.group(1).upper()] = (
            tokens.colors.get(match.group(1).upper(), 0) + 1
        )
    for match in _FONT_RE.finditer(source):
        family = " ".join(match.group(1).split())
        if family:
            tokens.fonts[family] = tokens.fonts.get(family, 0) + 1
    for match in _FONT_SIZE_RE.finditer(source):
        try:
            size = float(match.group(1))
        except ValueError:
            continue
        if 1.0 <= size <= 400.0:  # ignore obviously-not-font-size numbers
            tokens.font_sizes[size] = tokens.font_sizes.get(size, 0) + 1
    for match in _EASING_RE.finditer(source):
        normalized = "cubic-bezier(" + ",".join(
            part.strip() for part in match.group(1).split(",")
        ) + ")"
        tokens.easings[normalized] = tokens.easings.get(normalized, 0) + 1
    for match in _TRANSITION_RE.finditer(source):
        key = f"{match.group(1)} {match.group(2)}s"
        tokens.transitions[key] = tokens.transitions.get(key, 0) + 1
    return tokens


def load_reference(
    palette_file: Path | None = None,
    *,
    palette: list[str] | None = None,
    fonts: list[str] | None = None,
    easings: list[str] | None = None,
    style_md: Path | None = None,
) -> dict[str, Any]:
    """Build the reference style from brand file(s) + inline values."""
    reference: dict[str, Any] = {
        "palette": [color.strip().lstrip("#").upper() for color in (palette or [])],
        "fonts": [family.strip() for family in (fonts or [])],
        "easings": [easing.strip() for easing in (easings or [])],
    }
    if palette_file is not None and palette_file.is_file():
        from openharness.services.pixel_qa import load_brand_palette

        hexes = load_brand_palette(palette_file, as_hex=True) or []
        reference["palette"].extend(
            color.strip().lstrip("#").upper() for color in hexes
        )
        if palette_file.suffix.lower() in {".json", ".yaml", ".yml"}:
            fonts_found = StyleMemory.extract_from_tokens(palette_file).get("fonts")
            reference["fonts"].extend(fonts_found or [])
    if style_md is not None and style_md.is_file():
        try:
            text = style_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        reference["palette"].extend(
            match.group(1).upper() for match in _HEX_RE.finditer(text)
        )
        reference["easings"].extend(
            match.group(0).replace(" ", "")
            .replace("cubic-bezier(", "cubic-bezier(")
            for match in _EASING_RE.finditer(text)
        )
        # Normalize easings extracted from markdown.
        reference["easings"] = [
            "cubic-bezier("
            + ",".join(part.strip() for part in easing.split("cubic-bezier(")[-1]
                       .rstrip(")").split(","))
            + ")"
            for easing in reference["easings"]
        ]
    # Deduplicate, preserving order.
    reference["palette"] = list(dict.fromkeys(reference["palette"]))
    reference["fonts"] = list(dict.fromkeys(reference["fonts"]))
    reference["easings"] = list(dict.fromkeys(reference["easings"]))
    return reference


class StyleConsistencyChecker:
    """Compare scene code tokens against the reference and each other."""

    def __init__(self, reference: dict[str, Any] | None = None) -> None:
        self.reference = reference or {"palette": [], "fonts": [], "easings": []}

    # ------------------------------------------------------------------
    def check_files(self, files: list[Path]) -> ConsistencyReport:
        report = ConsistencyReport(reference=self.reference)
        scene_tokens: dict[str, StyleTokens] = {}
        for file in files:
            try:
                source = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tokens = extract_tokens(source)
            scene_tokens[file.stem] = tokens
            report.per_scene_tokens[file.stem] = tokens.as_dict()
            report.scenes_checked += 1
            self._check_against_reference(file.stem, tokens, report)

        self._check_cross_scene_drift(scene_tokens, report)
        return report

    # ------------------------------------------------------------------
    def _check_against_reference(
        self, scene: str, tokens: StyleTokens, report: ConsistencyReport
    ) -> None:
        palette = set(self.reference.get("palette") or [])
        fonts = {family.lower() for family in (self.reference.get("fonts") or [])}
        easings = set(self.reference.get("easings") or [])

        if palette:
            for color, count in tokens.colors.items():
                if color not in palette:
                    report.issues.append(
                        ConsistencyIssue(
                            scene=scene,
                            kind="palette_deviation",
                            message=(
                                f"color #{color} (x{count}) is not in the brand "
                                "palette — scenes are drifting off-brand"
                            ),
                            confidence=0.8,
                        )
                    )
        if fonts:
            for family, count in tokens.fonts.items():
                if family.lower() not in fonts:
                    report.issues.append(
                        ConsistencyIssue(
                            scene=scene,
                            kind="font_deviation",
                            message=(
                                f"font family {family!r} (x{count}) is not in the "
                                "project font set (STYLE.md / tokens.json)"
                            ),
                            confidence=0.7,
                        )
                    )
        if easings:
            for easing, count in tokens.easings.items():
                if easing not in easings:
                    report.issues.append(
                        ConsistencyIssue(
                            scene=scene,
                            kind="easing_deviation",
                            message=(
                                f"motion curve {easing} (x{count}) differs from "
                                "the project's standard easings"
                            ),
                            confidence=0.7,
                        )
                    )

    # ------------------------------------------------------------------
    def _check_cross_scene_drift(
        self, scene_tokens: dict[str, StyleTokens], report: ConsistencyReport
    ) -> None:
        """Flag tokens that only *some* scenes standardized on (>= 3 scenes)."""
        if len(scene_tokens) < 3:
            return
        total = len(scene_tokens)

        def _coverage(token_map: dict[str, dict[str, int]], label: str) -> None:
            usage: dict[str, list[str]] = {}
            for scene, per_scene in token_map.items():
                for token in per_scene:
                    usage.setdefault(token, []).append(scene)
            for token, scenes in usage.items():
                if 1 < len(scenes) < total:
                    missing = [scene for scene in scene_tokens if scene not in scenes]
                    report.issues.append(
                        ConsistencyIssue(
                            scene=scenes[0],
                            kind="style_drift",
                            message=(
                                f"{label} {token!r} used in {len(scenes)}/{total} "
                                f"scenes but missing in: {', '.join(missing[:6])}"
                            ),
                            confidence=0.5,
                        )
                    )

        _coverage(
            {scene: tokens.colors for scene, tokens in scene_tokens.items()}, "color"
        )
        _coverage(
            {scene: tokens.fonts for scene, tokens in scene_tokens.items()}, "font"
        )
        _coverage(
            {scene: tokens.easings for scene, tokens in scene_tokens.items()}, "easing"
        )

        # Font-size outliers vs the project median.
        sizes: list[float] = []
        for tokens in scene_tokens.values():
            sizes.extend(tokens.font_sizes.keys())
        if len(sizes) >= 4:
            sizes_sorted = sorted(sizes)
            median = sizes_sorted[len(sizes_sorted) // 2]
            for scene, tokens in scene_tokens.items():
                for size in tokens.font_sizes:
                    if size < median * 0.4:
                        report.issues.append(
                            ConsistencyIssue(
                                scene=scene,
                                kind="font_size_outlier",
                                message=(
                                    f"font size {size:g} is far below the project "
                                    f"median {median:g} — readability risk"
                                ),
                                confidence=0.6,
                            )
                        )


def find_scene_code_files(path: Path) -> list[Path]:
    """Return scene-code files at ``path`` (file, or dir scanned shallowly)."""
    if path.is_file() and path.suffix.lower() in _SCENE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in _SCENE_EXTENSIONS
        )
    return []


def format_report_text(report: ConsistencyReport) -> str:
    """Render a consistency report as agent-readable text."""
    if report.scenes_checked == 0:
        return "style_consistency: no scene files found to check."
    status = "CONSISTENT" if report.passed else "INCONSISTENT"
    lines = [
        (
            f"[style_consistency] {status} across {report.scenes_checked} scene(s) "
            f"(score {report.score:.2f})"
        )
    ]
    if not report.issues:
        lines.append("  - every scene uses the same palette, fonts and easings.")
    for issue in report.issues:
        lines.append(
            f"  - [{issue.kind}] {issue.scene}: {issue.message} "
            f"(conf {issue.confidence:.2f})"
        )
    return "\n".join(lines)
