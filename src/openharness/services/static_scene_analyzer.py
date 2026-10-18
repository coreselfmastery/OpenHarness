"""Static Scene Analyzer — code-level scene linting before rendering.

Catching a structural error at the *code* level is ~100x cheaper than
catching it after rendering: no GPU time, no encode, no QA round-trip.
This module parses generated scene code and runs a battery of
structural checks, each returning pass/fail with a confidence:

1. ``undeclared_assets`` — file references (strings with asset-ish
   extensions: images, audio, fonts, video) that are not declared in
   the scene's own asset list / imports and not passed by the caller.
2. ``off_safe_area`` — element positions outside the frame's safe
   area (default 5% inset). Catches ``x: 1900`` on a 1920-wide frame
   and negative/off-screen placements.
3. ``text_too_small`` — font sizes below the project minimum
   (default 18 px at 1080p, scaled by frame height).
4. ``unbranded_colors`` — hardcoded hex colors that are not in the
   brand palette (``brand.yaml`` / ``styles/*/tokens.json``).
5. ``hardcoded_constants`` — position/dimension literals repeated
   across the scene that should be named variables (drift risk).
6. ``non_determinism`` — ``Math.random()`` / ``Date.now()`` /
   ``random.random()`` / ``time.time()`` / ``datetime.now()`` / ``uuid``
   calls that make a re-render differ from the QA'd render (fail).

Python scenes (.py) are parsed with the real :mod:`ast` (exact line
numbers); JavaScript/TypeScript scenes (.js/.jsx/.ts/.tsx) are scanned
with calibrated regexes (no parser dependency).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCENE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs"}

ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".bmp",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a",
    ".mp4", ".mov", ".webm",
    ".ttf", ".otf", ".woff", ".woff2",
}

DEFAULTS = {
    "frame_width": 1920,
    "frame_height": 1080,
    "safe_area_margin": 0.05,   # fraction of each dimension
    "min_font_px": 18,          # at 1080p, scaled by height/1080
    "max_constant_repeats": 3,  # literal used >= N times in position context
}

_CRITICAL_CHECKS = ("non_determinism", "undeclared_assets")

# ----------------------------------------------------------------------
# Pattern tables (JS side; Python uses the ast where possible)
# ----------------------------------------------------------------------
_JS_NON_DETERMINISTIC = [
    (r"\bMath\.random\s*\(", "Math.random()"),
    (r"\bDate\.now\s*\(", "Date.now()"),
    (r"\bnew\s+Date\s*\(", "new Date()"),
    (r"\bperformance\.now\s*\(", "performance.now()"),
]
_PY_NON_DETERMINISTIC_NAMES = {
    "random", "uniform", "randint", "choice", "shuffle", "sample", "randrange",
    "time", "monotonic", "now", "today", "utcnow", "uuid4", "uuid1", "urandom",
}
_PY_NON_DETERMINISTIC_MODULES = {"random", "time", "datetime", "uuid", "secrets"}

# Bare function name -> the non-deterministic module it (typically) comes
# from; used to decide whether a BARE call like ``choice(...)`` is really
# the library function (module imported) or a legitimate local helper.
_BARE_NAME_TO_MODULE: dict[str, str] = {
    "random": "random",
    "uniform": "random",
    "randint": "random",
    "choice": "random",
    "shuffle": "random",
    "sample": "random",
    "randrange": "random",
    "time": "time",
    "monotonic": "time",
    "now": "datetime",
    "today": "datetime",
    "utcnow": "datetime",
    "uuid4": "uuid",
    "uuid1": "uuid",
    "urandom": "os",
}

_JS_ASSET_RE = re.compile(r"['\"]([^'\"]+\.(?:png|jpe?g|webp|svg|gif|bmp|mp3|wav|ogg|flac|m4a|mp4|mov|webm|ttf|otf|woff2?))['\"]", re.IGNORECASE)
_JS_FONT_SIZE_RE = re.compile(r"font(?:Size|size)\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)")
_JS_HEX_COLOR_RE = re.compile(r"['\"]#([0-9a-fA-F]{6})['\"]")
# x/y/left/top/right/bottom assignments with numeric values (px or bare).
# NOTE: the negative lookbehind keeps CSS-ish compounds (margin-left,
# padding-top, scrollLeft, ...) from matching their tail word, and
# translateX/translateY are deliberately EXCLUDED — they are relative
# offsets (0 = no translation), not absolute frame positions, so
# checking them against safe-area bounds flagged every idle
# ``translateX: 0`` / ``translateY: 0`` as off_safe_area.
_JS_POSITION_RE = re.compile(
    r"(?<![\w$-])(x|y|left|top|right|bottom)\s*[:=]\s*"
    r"(-?\d+(?:\.\d+)?)(?:px|f)?\b"
)

_STRING_RE = re.compile(r"['\"]([^'\"]*)['\"]")
_HEX_ANY_RE = re.compile(r"#([0-9a-fA-F]{6})\b")


@dataclass
class LintIssue:
    """One structural problem found in scene code."""

    check: str
    line: int
    message: str
    confidence: float
    severity: str = "warn"  # or "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "line": self.line,
            "message": self.message,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
        }


@dataclass
class SceneLintReport:
    """Aggregated static-analysis result for one scene file."""

    scene: str
    path: str
    language: str = "unknown"
    lines_scanned: int = 0
    issues: list[LintIssue] = field(default_factory=list)
    check_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "fail" for issue in self.issues)

    @property
    def score(self) -> float:
        if self.lines_scanned == 0:
            return 0.0
        penalty = sum(
            (1.0 if issue.severity == "fail" else 0.4) * issue.confidence
            for issue in self.issues
        )
        return float(max(0.0, 1.0 - penalty / max(1.0, self.lines_scanned / 10)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "path": self.path,
            "language": self.language,
            "lines_scanned": self.lines_scanned,
            "passed": self.passed,
            "score": round(self.score, 3),
            "issues": [issue.as_dict() for issue in self.issues],
            "check_summary": self.check_summary,
        }


class StaticSceneAnalyzer:
    """Parse scene code and run the structural check battery."""

    def __init__(self, **overrides: Any) -> None:
        self.config: dict[str, Any] = {**DEFAULTS}
        for key, value in overrides.items():
            if key in self.config:
                self.config[key] = value

    # ------------------------------------------------------------------
    def analyze_file(
        self,
        path: Path,
        *,
        declared_assets: list[str] | None = None,
        palette_hex: list[str] | None = None,
    ) -> SceneLintReport:
        """Analyze one scene file. ``declared_assets`` are asset basenames
        the caller knows exist/are provided (manifest entries)."""
        source = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".py":
            report = self.analyze_python(source, scene=path.stem, path=str(path))
        elif path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
            report = self.analyze_javascript(source, scene=path.stem, path=str(path))
        else:
            report = SceneLintReport(scene=path.stem, path=str(path))
        report.lines_scanned = source.count("\n") + 1
        self._check_assets_common(source, report, declared_assets or [])
        self._check_colors_common(source, report, palette_hex or [])
        return report

    # ------------------------------------------------------------------
    def analyze_python(
        self, source: str, *, scene: str = "scene", path: str = "<memory>"
    ) -> SceneLintReport:
        report = SceneLintReport(scene=scene, path=path, language="python")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            report.issues.append(
                LintIssue(
                    check="syntax",
                    line=exc.lineno or 1,
                    message=f"scene does not parse: {exc.msg}",
                    confidence=1.0,
                    severity="fail",
                )
            )
            return report

        declared: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in getattr(node, "names", []):
                    declared.add(name.name)
        font_min = self._min_font()
        safe = self._safe_bounds()

        for node in ast.walk(tree):
            # ---- non-deterministic calls -------------------------------
            if isinstance(node, ast.Call):
                func_name = self._dotted_name(node.func)
                bare = func_name.rsplit(".", 1)[-1]
                if bare in _PY_NON_DETERMINISTIC_NAMES:
                    # Only flag calls whose non-deterministic source module
                    # is actually imported. The old ``"." not in func_name``
                    # escape hard-failed legitimate local helpers (a local
                    # ``def choice(items)`` + ``choice([...])`` was flagged
                    # and blocked the scene from rendering) while the
                    # import-derived ``declared`` set sat unused.
                    if "." in func_name:
                        root = func_name.split(".", 1)[0]
                        flagged = (
                            root in _PY_NON_DETERMINISTIC_MODULES
                            and root in declared
                        )
                    else:
                        # Bare call: either directly imported from the
                        # module (``from random import choice``) or the
                        # module itself is imported.
                        module = _BARE_NAME_TO_MODULE.get(bare)
                        flagged = bare in declared or (
                            module is not None and module in declared
                        )
                    if flagged:
                        report.issues.append(
                            LintIssue(
                                check="non_determinism",
                                line=getattr(node, "lineno", 1),
                                message=(
                                    f"non-deterministic call {func_name}() — a "
                                    "re-render will differ from the QA'd render"
                                ),
                                confidence=0.95,
                                severity="fail",
                            )
                        )
                # ---- font size ------------------------------------------
                for keyword in node.keywords:
                    if keyword.arg in {"font_size", "fontSize"} and isinstance(
                        keyword.value, (ast.Constant,)
                    ):
                        value = keyword.value.value
                        if isinstance(value, (int, float)) and 0 < value < font_min:
                            report.issues.append(
                                LintIssue(
                                    check="text_too_small",
                                    line=getattr(node, "lineno", 1),
                                    message=(
                                        f"font_size={value} below the project "
                                        f"minimum {font_min:g}"
                                    ),
                                    confidence=0.85,
                                )
                            )
            # ---- position arrays move_to([x, y, z]) ----------------------
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"move_to", "shift", "set_x", "set_y"}
            ):
                self._check_position_args(node, report, safe)

        self._check_hardcoded_constants_python(tree, report)
        return report

    # ------------------------------------------------------------------
    def analyze_javascript(
        self, source: str, *, scene: str = "scene", path: str = "<memory>"
    ) -> SceneLintReport:
        report = SceneLintReport(scene=scene, path=path, language="javascript")
        lines = source.split("\n")
        safe = self._safe_bounds()
        font_min = self._min_font()

        for line_no, line in enumerate(lines, start=1):
            for pattern, label in _JS_NON_DETERMINISTIC:
                if re.search(pattern, line):
                    report.issues.append(
                        LintIssue(
                            check="non_determinism",
                            line=line_no,
                            message=f"{label} — a re-render will differ from the QA'd render",
                            confidence=0.95,
                            severity="fail",
                        )
                    )
            for match in _JS_FONT_SIZE_RE.finditer(line):
                value = float(match.group(1))
                if 0 < value < font_min:
                    report.issues.append(
                        LintIssue(
                            check="text_too_small",
                            line=line_no,
                            message=f"fontSize {value:g} below the project minimum {font_min:g}",
                            confidence=0.85,
                        )
                    )
            for match in _JS_POSITION_RE.finditer(line):
                axis = "x" if match.group(1) in {"x", "left", "right"} else "y"
                value = float(match.group(2))
                lo, hi = safe["x"] if axis == "x" else safe["y"]
                if not (lo <= value <= hi):
                    report.issues.append(
                        LintIssue(
                            check="off_safe_area",
                            line=line_no,
                            message=(
                                f"{axis}-position {value:g} outside the safe "
                                f"area [{lo:g}, {hi:g}] (frame "
                                f"{self.config['frame_width']}x"
                                f"{self.config['frame_height']})"
                            ),
                            confidence=0.8,
                        )
                    )
        self._check_hardcoded_constants_js(source, report)
        return report

    # ------------------------------------------------------------------
    # Shared checks (language-agnostic, line-numbered via regex)
    # ------------------------------------------------------------------
    def _check_assets_common(
        self,
        source: str,
        report: SceneLintReport,
        declared_assets: list[str],
    ) -> None:
        declared = {Path(name).name.lower() for name in declared_assets}
        lines = source.split("\n")
        for line_no, line in enumerate(lines, start=1):
            # Skip import/require lines — those are module refs, not assets.
            if re.match(r"\s*(import|from|require\s*\()\s", line):
                continue
            for match in _STRING_RE.finditer(line):
                candidate = match.group(1).strip()
                suffix = Path(candidate).suffix.lower()
                # Flag any string with an asset-ish extension except URLs
                # ("https://cdn.../logo.png") — relative paths like
                # "assets/logo.png" or "./img/x.jpg" are exactly the
                # references that must be declared.
                if (
                    suffix in ASSET_EXTENSIONS
                    and not candidate.startswith(("http://", "https://"))
                    and Path(candidate).name.lower() not in declared
                ):
                    report.issues.append(
                        LintIssue(
                            check="undeclared_assets",
                            line=line_no,
                            message=(
                                f"asset reference {candidate!r} is not in the "
                                "declared asset list — the render will fail "
                                "or silently drop it"
                            ),
                            confidence=0.9,
                            severity="fail",
                        )
                    )

    def _check_colors_common(
        self, source: str, report: SceneLintReport, palette_hex: list[str]
    ) -> None:
        if not palette_hex:
            return  # no palette declared -> color check not applicable
        allowed = {color.strip().lstrip("#").upper() for color in palette_hex}
        lines = source.split("\n")
        for line_no, line in enumerate(lines, start=1):
            for match in _HEX_ANY_RE.finditer(line):
                hex_value = match.group(1).upper()
                if hex_value not in allowed:
                    report.issues.append(
                        LintIssue(
                            check="unbranded_colors",
                            line=line_no,
                            message=(
                                f"hardcoded color #{hex_value} is not in the "
                                "brand palette (brand.yaml / tokens.json)"
                            ),
                            confidence=0.7,
                        )
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _min_font(self) -> float:
        scale = self.config["frame_height"] / 1080.0
        return float(self.config["min_font_px"]) * max(scale, 0.05)

    def _safe_bounds(self) -> dict[str, tuple[float, float]]:
        margin_x = self.config["frame_width"] * self.config["safe_area_margin"]
        margin_y = self.config["frame_height"] * self.config["safe_area_margin"]
        return {
            "x": (margin_x, self.config["frame_width"] - margin_x),
            "y": (margin_y, self.config["frame_height"] - margin_y),
        }

    @staticmethod
    def _dotted_name(node: ast.expr) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    def _check_position_args(
        self, node: ast.Call, report: SceneLintReport, safe: dict[str, tuple[float, float]]
    ) -> None:
        """Check move_to([x, y, ...]) / set_x(1900) style manim positions.

        Manim uses scene units (frame 7.11 x 4 units by default) — we only
        flag values that look like *pixel* coordinates (abs > 8), which are
        the common copy-paste bug from design specs.
        """
        for arg in node.args:
            values: list[float] = []
            if isinstance(arg, (ast.List, ast.Tuple)):
                for item in arg.elts:
                    if isinstance(item, ast.Constant) and isinstance(
                        item.value, (int, float)
                    ):
                        values.append(float(item.value))
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                values.append(float(arg.value))
            if not values:
                continue
            frame_limit = 8.0  # manim scene units; beyond = off-screen
            for axis_index, value in enumerate(values[:2]):
                if abs(value) > frame_limit:
                    axis = "x" if axis_index == 0 else "y"
                    report.issues.append(
                        LintIssue(
                            check="off_safe_area",
                            line=getattr(node, "lineno", 1),
                            message=(
                                f"{axis}-position {value:g} looks like a pixel "
                                "coordinate in a scene-unit API (beyond the "
                                "visible frame)"
                            ),
                            confidence=0.6,
                        )
                    )

    def _check_hardcoded_constants_python(
        self, tree: ast.AST, report: SceneLintReport
    ) -> None:
        counts: dict[float, int] = {}
        first_line: dict[float, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                value = float(node.value)
                if value in {0, 1, 2, 3, -1} or abs(value) < 0.01:
                    continue
                counts[value] = counts.get(value, 0) + 1
                first_line.setdefault(value, getattr(node, "lineno", 1))
        self._flag_repeated_constants(counts, first_line, report)

    def _check_hardcoded_constants_js(
        self, source: str, report: SceneLintReport
    ) -> None:
        counts: dict[float, int] = {}
        first_line: dict[float, int] = {}
        for line_no, line in enumerate(source.split("\n"), start=1):
            if re.match(r"\s*(import|require|//|/\*|\*)", line):
                continue
            for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", line):
                value = float(match.group(1))
                if value in {0, 1, 2, 3} or abs(value) < 0.01:
                    continue
                counts[value] = counts.get(value, 0) + 1
                first_line.setdefault(value, line_no)
        self._flag_repeated_constants(counts, first_line, report)

    def _flag_repeated_constants(
        self,
        counts: dict[float, int],
        first_line: dict[float, int],
        report: SceneLintReport,
    ) -> None:
        threshold = int(self.config["max_constant_repeats"])
        for value, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            if count >= threshold and 4 <= abs(value):
                report.issues.append(
                    LintIssue(
                        check="hardcoded_constants",
                        line=first_line.get(value, 1),
                        message=(
                            f"literal {value:g} appears {count} times — extract "
                            "it to a named constant so scenes cannot drift apart"
                        ),
                        confidence=min(0.9, 0.4 + 0.15 * count),
                    )
                )


def find_scene_files(path: Path) -> list[Path]:
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


def format_report_text(reports: list[SceneLintReport]) -> str:
    """Render lint reports as agent-readable text."""
    if not reports:
        return "scene_lint: no scene files found to analyze."
    lines: list[str] = []
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        lines.append(
            f"[scene_lint] {report.scene} ({report.language}, "
            f"{report.lines_scanned} lines): {status} "
            f"(score {report.score:.2f})"
        )
        if not report.issues:
            lines.append("  - all structural checks passed.")
        for issue in report.issues:
            lines.append(
                f"  - L{issue.line} {issue.check} [{issue.severity}] "
                f"(conf {issue.confidence:.2f}): {issue.message}"
            )
    return "\n".join(lines)
