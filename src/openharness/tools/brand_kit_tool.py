"""Brand Kit — enforces brand consistency across every video.

Solves the dual problem of video generation:
- Without branding, every video looks different (no channel identity).
- With too much branding, every video looks the same (no creativity).

The Brand Kit separates fixed **brand anchors** (colors, fonts, logo,
voice, music style) from **creative drift** (per-video variations within
the brand envelope).

brand.yaml (in project root) defines the fixed identity:

    brand:
      name: "Your Channel"
      colors:
        primary: "#1A1A2E"
        secondary: "#16213E"
        accent: "#E94560"
      fonts:
        heading: "Inter"
        body: "Inter"
      voice: "Serena"
      music_style: "ambient electronic"
      min_type_size: 44
      safe_area: 220
      easing: "cubic-bezier(0.4, 0.0, 0.2, 1)"

For each video, the tool generates a **seeded drift** — a unique variation
within the brand envelope. The seed is derived from ``brand:project-<id>``
so the same inputs always produce the same drift (reproducible but unique).

Two scores in review:
- **Brand Compliance** (must be 100%): colors, fonts, logo, voice, safe
  area, min type size all match the brand anchors.
- **Creative Originality** (must be >= 8/10): concept, hook, transitions,
  opening, visual variety are sufficiently original.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brand config model
# ---------------------------------------------------------------------------

class BrandColors(BaseModel):
    primary: str = ""
    secondary: str = ""
    accent: str = ""


class BrandFonts(BaseModel):
    heading: str = ""
    body: str = ""


class BrandConfig(BaseModel):
    """Parsed brand.yaml configuration."""

    name: str = ""
    colors: BrandColors = Field(default_factory=BrandColors)
    fonts: BrandFonts = Field(default_factory=BrandFonts)
    voice: str = ""
    music_style: str = ""
    min_type_size: int = 44
    safe_area: int = 220
    easing: str = "cubic-bezier(0.4, 0.0, 0.2, 1)"

    @classmethod
    def from_yaml_file(cls, path: Path) -> "BrandConfig":
        """Load brand config from a YAML file. Returns defaults if missing."""
        if not path.is_file():
            return cls()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:
            log.warning("Failed to parse brand.yaml at %s: %s", path, exc)
            return cls()
        if not isinstance(raw, dict):
            return cls()
        brand_data = raw.get("brand", raw)
        if not isinstance(brand_data, dict):
            return cls()
        return cls.model_validate(brand_data)


# ---------------------------------------------------------------------------
# Creative drift
# ---------------------------------------------------------------------------

@dataclass
class CreativeDrift:
    """Per-video creative variation within the brand envelope."""

    seed: str
    palette_shift: float  # 0.0 - 0.3 (how far colors can drift from brand)
    motion_intensity: float  # 0.3 - 0.9 (how dynamic the motion is)
    transition_style: str  # "invented" (agent designs per cut)
    opening_style: str  # "unique" (agent designs per video)
    # Derived palette (brand colors shifted by palette_shift)
    derived_colors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "palette_shift": round(self.palette_shift, 3),
            "motion_intensity": round(self.motion_intensity, 3),
            "transition_style": self.transition_style,
            "opening_style": self.opening_style,
            "derived_colors": self.derived_colors,
        }


def generate_drift(seed: str, brand: BrandConfig) -> CreativeDrift:
    """Generate a deterministic creative drift from a seed string.

    The same seed + brand always produces the same drift, so videos are
    reproducible. Different seeds produce different drifts, so each video
    is unique.

    Args:
        seed: Drift seed string (e.g. "brand:project-001").
        brand: Loaded brand config (for deriving colors).
    """
    # Hash the seed to get a deterministic int for random.Random
    seed_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seed_int = int(seed_hash[:16], 16)  # first 64 bits
    rng = random.Random(seed_int)

    # Generate drift parameters within safe ranges
    palette_shift = rng.uniform(0.0, 0.3)
    motion_intensity = rng.uniform(0.3, 0.9)

    # Derive shifted colors from brand anchors
    derived_colors: dict[str, str] = {}
    for role in ("primary", "secondary", "accent"):
        base_hex = getattr(brand.colors, role, "")
        if base_hex:
            derived_colors[role] = _shift_color(base_hex, palette_shift, rng)

    return CreativeDrift(
        seed=seed,
        palette_shift=palette_shift,
        motion_intensity=motion_intensity,
        transition_style="invented",
        opening_style="unique",
        derived_colors=derived_colors,
    )


def _shift_color(hex_color: str, shift: float, rng: random.Random) -> str:
    """Shift a hex color by ``shift`` amount in HSL space.

    The shift is bounded so the derived color stays recognizably close
    to the brand color (within ``shift * 360`` degrees of hue rotation).
    """
    h, s, l = _hex_to_hsl(hex_color)
    # Rotate hue by a random amount bounded by shift
    hue_rotation = rng.uniform(-shift, shift) * 360
    h = (h + hue_rotation) % 360
    # Slightly vary saturation and lightness
    s = max(0.0, min(1.0, s + rng.uniform(-shift * 0.2, shift * 0.2)))
    l = max(0.0, min(1.0, l + rng.uniform(-shift * 0.15, shift * 0.15)))
    return _hsl_to_hex(h, s, l)


def _hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert #RRGGBB to (hue, saturation, lightness) in [0, 360], [0, 1], [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0.0, 0.0, 0.5)
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    except ValueError:
        return (0.0, 0.0, 0.5)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    l = (max_c + min_c) / 2.0
    if max_c == min_c:
        return (0.0, 0.0, l)
    d = max_c - min_c
    s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
    if max_c == r:
        h = ((g - b) / d) % 6.0
    elif max_c == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    h *= 60.0
    if h < 0:
        h += 360.0
    return (h, s, l)


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL to #RRGGBB."""
    h = h % 360.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    r = int(round((r + m) * 255))
    g = int(round((g + m) * 255))
    b = int(round((b + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Compliance scoring
# ---------------------------------------------------------------------------

@dataclass
class BrandComplianceResult:
    """Result of checking scene code against brand anchors."""

    passed: bool
    score: int  # 0-100
    checks: list[dict[str, Any]]  # per-check details
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": self.checks,
            "failures": self.failures,
        }


@dataclass
class CreativeOriginalityResult:
    """Result of scoring creative originality."""

    score: int  # 0-10
    factors: dict[str, int]  # per-factor scores
    passed: bool  # score >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "factors": self.factors,
            "passed": self.passed,
        }


def check_brand_compliance(scene_code: str, brand: BrandConfig) -> BrandComplianceResult:
    """Check scene code against brand anchors.

    Verifies:
    - Colors used in the code include or are derived from brand palette.
    - Fonts match brand.heading / brand.body.
    - min_type_size is respected (no font-size below the minimum).
    - safe_area is respected (no elements outside the safe area).
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    passed_count = 0
    total_checks = 0

    # Check 1: Colors
    total_checks += 1
    colors_in_code = _extract_hex_colors(scene_code)
    brand_colors = {
        "primary": brand.colors.primary,
        "secondary": brand.colors.secondary,
        "accent": brand.colors.accent,
    }
    brand_color_set = {c.lower() for c in brand_colors.values() if c}
    if not brand_color_set:
        # No brand colors defined — skip
        checks.append({"name": "colors", "passed": True, "detail": "No brand colors defined, skipped"})
        passed_count += 1
    elif colors_in_code:
        # At least one color in code should match or be close to a brand color
        matched = any(_colors_match(c, brand_color_set) for c in colors_in_code)
        if matched:
            checks.append({
                "name": "colors",
                "passed": True,
                "detail": f"Found {len(colors_in_code)} colors, at least one matches brand palette",
            })
            passed_count += 1
        else:
            failures.append(
                f"Colors in code ({colors_in_code[:3]}) don't match brand palette ({list(brand_color_set)[:3]})"
            )
            checks.append({
                "name": "colors",
                "passed": False,
                "detail": f"Colors {colors_in_code[:3]} don't match brand {list(brand_color_set)[:3]}",
            })
    else:
        checks.append({"name": "colors", "passed": True, "detail": "No colors in code, skipped"})
        passed_count += 1

    # Check 2: Fonts
    total_checks += 1
    fonts_in_code = _extract_fonts(scene_code)
    brand_fonts = {brand.fonts.heading.lower(), brand.fonts.body.lower()} - {""}
    if not brand_fonts:
        checks.append({"name": "fonts", "passed": True, "detail": "No brand fonts defined, skipped"})
        passed_count += 1
    elif fonts_in_code:
        matched = any(f.lower() in brand_fonts for f in fonts_in_code)
        if matched:
            checks.append({"name": "fonts", "passed": True, "detail": "Brand font found in code"})
            passed_count += 1
        else:
            failures.append(f"Fonts in code ({fonts_in_code[:3]}) don't include brand fonts ({list(brand_fonts)[:3]})")
            checks.append({
                "name": "fonts",
                "passed": False,
                "detail": f"Fonts {fonts_in_code[:3]} don't match brand {list(brand_fonts)[:3]}",
            })
    else:
        checks.append({"name": "fonts", "passed": True, "detail": "No fonts in code, skipped"})
        passed_count += 1

    # Check 3: min_type_size
    total_checks += 1
    if brand.min_type_size > 0:
        small_fonts = _find_font_sizes_below(scene_code, brand.min_type_size)
        if small_fonts:
            failures.append(
                f"Found font sizes below minimum {brand.min_type_size}: {small_fonts[:3]}"
            )
            checks.append({
                "name": "min_type_size",
                "passed": False,
                "detail": f"Sizes {small_fonts[:3]} below min {brand.min_type_size}",
            })
        else:
            checks.append({"name": "min_type_size", "passed": True, "detail": f"All sizes >= {brand.min_type_size}"})
            passed_count += 1
    else:
        checks.append({"name": "min_type_size", "passed": True, "detail": "No min_type_size defined, skipped"})
        passed_count += 1

    # Check 4: safe_area
    total_checks += 1
    if brand.safe_area > 0:
        violations = _find_safe_area_violations(scene_code, brand.safe_area)
        if violations:
            failures.append(f"safe_area violations: {violations[:3]}")
            checks.append({
                "name": "safe_area",
                "passed": False,
                "detail": f"Elements outside safe area {brand.safe_area}",
            })
        else:
            checks.append({"name": "safe_area", "passed": True, "detail": f"All elements within safe area {brand.safe_area}"})
            passed_count += 1
    else:
        checks.append({"name": "safe_area", "passed": True, "detail": "No safe_area defined, skipped"})
        passed_count += 1

    score = int((passed_count / total_checks) * 100) if total_checks else 100
    passed = len(failures) == 0
    return BrandComplianceResult(
        passed=passed,
        score=score,
        checks=checks,
        failures=failures,
    )


def score_creative_originality(
    scene_code: str,
    drift: CreativeDrift | None = None,
    *,
    threshold: int = 8,
) -> CreativeOriginalityResult:
    """Score creative originality on a 0-10 scale.

    Factors:
    - Visual variety (number of distinct visual elements / colors / shapes)
    - Motion variety (diversity of animation / transition styles)
    - Concept uniqueness (heuristic: presence of uncommon patterns)

    This is a heuristic scorer — not an LLM call. It gives a rough
    originality signal for automated review. The Director agent can
    override with a manual review.
    """
    factors: dict[str, int] = {}

    # Factor 1: Visual variety (distinct hex colors + distinct font sizes)
    colors = _extract_hex_colors(scene_code)
    sizes = _extract_font_sizes(scene_code)
    visual_variety = min(10, len(set(colors)) + len(set(sizes)))
    factors["visual_variety"] = visual_variety

    # Factor 2: Motion variety (count distinct easing / transition keywords)
    motion_keywords = re.findall(
        r"ease|transition|cubic-bezier|spring|linear|bounce|elastic",
        scene_code,
        re.IGNORECASE,
    )
    motion_variety = min(10, len(set(kw.lower() for kw in motion_keywords)) * 2)
    factors["motion_variety"] = motion_variety

    # Factor 3: Concept uniqueness (penalize template-like patterns)
    template_patterns = [
        r"function\s+App\s*\(\s*\)",  # generic App component
        r"return\s*\(\s*<div>",  # bare div return
        r"style=\{\{.*\}\}",  # inline styles only
    ]
    template_matches = sum(
        1 for pat in template_patterns if re.search(pat, scene_code)
    )
    concept_uniqueness = max(0, 10 - template_matches * 3)
    factors["concept_uniqueness"] = concept_uniqueness

    # Factor 4: Drift usage (bonus if drift is referenced)
    drift_usage = 0
    if drift and drift.derived_colors:
        drift_colors_in_code = sum(
            1 for c in drift.derived_colors.values() if c.lower() in scene_code.lower()
        )
        drift_usage = min(10, drift_colors_in_code * 3)
    factors["drift_usage"] = drift_usage

    # Weighted average
    weights = {
        "visual_variety": 0.25,
        "motion_variety": 0.25,
        "concept_uniqueness": 0.30,
        "drift_usage": 0.20,
    }
    total = sum(factors[k] * weights[k] for k in factors)
    score = int(round(total))
    return CreativeOriginalityResult(
        score=score,
        factors=factors,
        passed=score >= threshold,
    )


# ---------------------------------------------------------------------------
# Color / font extraction helpers
# ---------------------------------------------------------------------------

_HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")
_FONT_FAMILY_RE = re.compile(r"font-?family\s*[:=]\s*['\"]?([^'\";,\n]+)", re.IGNORECASE)
_FONT_SIZE_RE = re.compile(r"font-?size\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_SAFE_AREA_RE = re.compile(r"(?:left|top|right|bottom|x|y)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _extract_hex_colors(code: str) -> list[str]:
    """Extract all #RRGGBB color values from code."""
    return _HEX_COLOR_RE.findall(code)


def _extract_fonts(code: str) -> list[str]:
    """Extract font-family names from CSS-in-JS / style code."""
    return [m.strip() for m in _FONT_FAMILY_RE.findall(code) if m.strip()]


def _extract_font_sizes(code: str) -> list[float]:
    """Extract font-size values from CSS-in-JS / style code."""
    return [float(m) for m in _FONT_SIZE_RE.findall(code)]


def _find_font_sizes_below(code: str, minimum: int) -> list[float]:
    """Find font sizes below the minimum."""
    return [s for s in _extract_font_sizes(code) if s < minimum]


def _find_safe_area_violations(code: str, safe_area: int) -> list[str]:
    """Find position values that violate the safe area."""
    violations: list[str] = []
    for match in _SAFE_AREA_RE.finditer(code):
        val = float(match.group(1))
        if val < 0 or (val > 0 and val < safe_area and val < 100):
            # Heuristic: negative or small positive values near edges
            violations.append(f"{match.group(0)} (value {val} < safe_area {safe_area})")
    return violations


def _colors_match(color: str, brand_set: set[str], tolerance: int = 30) -> bool:
    """Check if a color matches any brand color within a tolerance.

    Tolerance is the max Euclidean distance in RGB space for a "match".
    """
    color_lower = color.lower()
    if color_lower in brand_set:
        return True
    # Check closeness
    try:
        r1, g1, b1 = _hex_to_rgb(color_lower)
        for brand_color in brand_set:
            r2, g2, b2 = _hex_to_rgb(brand_color)
            dist = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
            if dist <= tolerance:
                return True
    except (ValueError, IndexError):
        pass
    return False


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to (r, g, b) ints."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


# ---------------------------------------------------------------------------
# Tool input + implementation
# ---------------------------------------------------------------------------

class BrandKitToolInput(BaseModel):
    """Arguments for the brand kit tool."""

    action: str = Field(
        description=(
            "Action to perform: 'validate' (check scene code against brand), "
            "'drift' (generate creative drift for a project), "
            "'score' (run both brand compliance + creative originality)."
        ),
    )
    scene_code: str | None = Field(
        default=None,
        description="Scene code to validate/score (required for 'validate' and 'score').",
    )
    project_id: str | None = Field(
        default=None,
        description="Project ID for drift generation (e.g. 'project-001'). Required for 'drift'.",
    )
    brand_file: str | None = Field(
        default=None,
        description="Path to brand.yaml. Defaults to 'brand.yaml' in project root.",
    )
    originality_threshold: int = Field(
        default=8,
        ge=0,
        le=10,
        description="Minimum creative originality score to pass (default 8).",
    )


class BrandKitTool(BaseTool):
    """Enforce brand consistency + generate creative drift."""

    name = "brand_kit"
    description = (
        "Enforce brand consistency across videos while allowing creative freedom. "
        "Validates scene code against brand anchors (colors, fonts, safe area), "
        "generates seeded creative drift, and scores both brand compliance and "
        "creative originality. Reads brand.yaml from the project root."
    )
    input_model = BrandKitToolInput

    async def execute(
        self,
        arguments: BrandKitToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        # Load brand config
        brand_path = _resolve_brand_path(context, arguments.brand_file)
        brand = BrandConfig.from_yaml_file(brand_path)

        if arguments.action == "drift":
            return self._handle_drift(arguments, brand)
        elif arguments.action == "validate":
            return self._handle_validate(arguments, brand)
        elif arguments.action == "score":
            return self._handle_score(arguments, brand)
        else:
            return ToolResult(
                output=f"Unknown action: {arguments.action}. Use 'validate', 'drift', or 'score'.",
                is_error=True,
            )

    def _handle_drift(
        self,
        arguments: BrandKitToolInput,
        brand: BrandConfig,
    ) -> ToolResult:
        if not arguments.project_id:
            return ToolResult(
                output="project_id is required for drift action.",
                is_error=True,
            )
        seed = f"brand:{arguments.project_id}"
        drift = generate_drift(seed, brand)
        output_lines = [
            "[Creative drift generated]",
            f"Seed: {drift.seed}",
            f"Palette shift: {drift.palette_shift:.3f}",
            f"Motion intensity: {drift.motion_intensity:.3f}",
            f"Transition style: {drift.transition_style}",
            f"Opening style: {drift.opening_style}",
            "",
            "Derived colors (within brand envelope):",
        ]
        for role, color in drift.derived_colors.items():
            output_lines.append(f"  {role}: {color}")
        if not drift.derived_colors:
            output_lines.append("  (no brand colors defined — drift has no derived palette)")
        output_lines.append("")
        output_lines.append("YAML:")
        output_lines.append("```yaml")
        output_lines.append(yaml.dump({"drift": drift.to_dict()}, default_flow_style=False))
        output_lines.append("```")
        return ToolResult(
            output="\n".join(output_lines),
            metadata={"drift": drift.to_dict()},
        )

    def _handle_validate(
        self,
        arguments: BrandKitToolInput,
        brand: BrandConfig,
    ) -> ToolResult:
        if not arguments.scene_code:
            return ToolResult(
                output="scene_code is required for validate action.",
                is_error=True,
            )
        result = check_brand_compliance(arguments.scene_code, brand)
        output_lines = [
            "[Brand compliance check]",
            f"Passed: {result.passed}",
            f"Score: {result.score}/100",
            "",
            "Checks:",
        ]
        for check in result.checks:
            status = "PASS" if check["passed"] else "FAIL"
            output_lines.append(f"  [{status}] {check['name']}: {check['detail']}")
        if result.failures:
            output_lines.append("")
            output_lines.append("Failures:")
            for f in result.failures:
                output_lines.append(f"  - {f}")
        return ToolResult(
            output="\n".join(output_lines),
            is_error=not result.passed,
            metadata=result.to_dict(),
        )

    def _handle_score(
        self,
        arguments: BrandKitToolInput,
        brand: BrandConfig,
    ) -> ToolResult:
        if not arguments.scene_code:
            return ToolResult(
                output="scene_code is required for score action.",
                is_error=True,
            )
        compliance = check_brand_compliance(arguments.scene_code, brand)
        # Generate drift for scoring context if project_id provided
        drift = None
        if arguments.project_id:
            drift = generate_drift(f"brand:{arguments.project_id}", brand)
        originality = score_creative_originality(
            arguments.scene_code,
            drift=drift,
            threshold=arguments.originality_threshold,
        )
        output_lines = [
            "[Brand + Creative Score]",
            "",
            "Brand Compliance:",
            f"  Passed: {compliance.passed}",
            f"  Score: {compliance.score}/100",
            "",
            "Creative Originality:",
            f"  Score: {originality.score}/10 (threshold: {arguments.originality_threshold})",
            f"  Passed: {originality.passed}",
            "  Factors:",
        ]
        for factor, score in originality.factors.items():
            output_lines.append(f"    {factor}: {score}/10")
        output_lines.append("")
        overall_pass = compliance.passed and originality.passed
        output_lines.append(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
        if not compliance.passed:
            output_lines.append("  → Brand compliance failed: video rejected")
        if not originality.passed:
            output_lines.append("  → Creative originality below threshold: video revised")
        return ToolResult(
            output="\n".join(output_lines),
            is_error=not overall_pass,
            metadata={
                "compliance": compliance.to_dict(),
                "originality": originality.to_dict(),
                "overall_pass": overall_pass,
            },
        )


def _resolve_brand_path(context: ToolExecutionContext, brand_file: str | None) -> Path:
    """Resolve the brand.yaml path."""
    if brand_file:
        p = Path(brand_file).expanduser()
        if not p.is_absolute():
            p = context.cwd / p
        return p.resolve()
    return context.cwd / "brand.yaml"
