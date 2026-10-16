"""Smart review depth — classify scenes before QA.

Item #6: Not all scenes need the full 20-point checklist. Title cards,
transitions, and simple typography scenes can pass with a lightweight
check. Only complex scenes (Blender, Manim, data viz) get the full review.

This module classifies scenes into three tiers:
- Simple (5-point check): title cards, transitions, static typography.
- Standard (10-point review): most scenes with motion, narration sync.
- Complex (full 20-point review): Blender, Manim, data viz, physics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

REVIEW_DEPTH_SIMPLE = "simple"
REVIEW_DEPTH_STANDARD = "standard"
REVIEW_DEPTH_COMPLEX = "complex"

SIMPLE_CHECKLIST_POINTS = 5
STANDARD_CHECKLIST_POINTS = 10
COMPLEX_CHECKLIST_POINTS = 20


@dataclass
class SceneClassification:
    """Result of classifying a scene's complexity."""

    depth: str  # "simple" | "standard" | "complex"
    checklist_points: int
    reasons: list[str] = field(default_factory=list)
    detected_elements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "checklist_points": self.checklist_points,
            "reasons": self.reasons,
            "detected_elements": self.detected_elements,
        }


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

_COMPLEX_PATTERNS: list[tuple[str, str]] = [
    (r"\bblender\b", "Blender 3D"),
    (r"\bmanim\b", "Manim animation"),
    (r"\bthree\.?js\b", "Three.js 3D"),
    (r"\bwebgl\b", "WebGL"),
    (r"\bdata\s*viz\b", "Data visualization"),
    (r"\bchart\b", "Chart/graph"),
    (r"\bgraph\b", "Graph"),
    (r"\bphysics\b", "Physics simulation"),
    (r"\bparticle\b", "Particle system"),
    (r"\bsimulation\b", "Simulation"),
    (r"\b3d\b", "3D rendering"),
    (r"\bshader\b", "Shader"),
    (r"\bcomposit\b", "Multi-layer compositing"),
    (r"\bmask\b", "Masking"),
]

_SIMPLE_PATTERNS: list[tuple[str, str]] = [
    (r"\btitle\s*card\b", "Title card"),
    (r"\btransition\b", "Transition"),
    (r"\bstatic\b", "Static scene"),
    (r"\btypography\s*only\b", "Typography only"),
    (r"\btext\s*overlay\b", "Text overlay"),
    (r"\bcolor\s*only\b", "Color only"),
    (r"\bfade\s*(in|out)\b", "Simple fade"),
    (r"\blogo\s*reveal\b", "Logo reveal"),
    (r"\bend\s*card\b", "End card"),
    (r"\bsplash\b", "Splash screen"),
]

_STANDARD_PATTERNS: list[tuple[str, str]] = [
    (r"\banimat\b", "Animation"),
    (r"\bmotion\b", "Motion"),
    (r"\bslide\b", "Slide"),
    (r"\bnarrat\b", "Narration"),
    (r"\bvoice\s*over\b", "Voice over"),
    (r"\bcamera\b", "Camera movement"),
    (r"\bzoom\b", "Zoom"),
    (r"\bpan\b", "Pan"),
]


def classify_scene(description: str) -> SceneClassification:
    """Classify a scene's complexity to determine review depth.

    Rules (first match wins):
    1. If any complex pattern matches → complex (20-point review).
    2. Else if any standard pattern matches → standard (10-point review).
    3. Else if any simple pattern matches → simple (5-point check).
    4. Default → standard (safer to over-review than under-review).
    """
    lowered = description.lower()
    detected: list[str] = []
    reasons: list[str] = []

    # Check complex patterns first
    for pattern, label in _COMPLEX_PATTERNS:
        if re.search(pattern, lowered):
            detected.append(label)
    if detected:
        return SceneClassification(
            depth=REVIEW_DEPTH_COMPLEX,
            checklist_points=COMPLEX_CHECKLIST_POINTS,
            reasons=[f"Complex element detected: {label}" for label in detected],
            detected_elements=detected,
        )

    # Check standard patterns
    standard_detected: list[str] = []
    for pattern, label in _STANDARD_PATTERNS:
        if re.search(pattern, lowered):
            standard_detected.append(label)
    if standard_detected:
        detected.extend(standard_detected)
        return SceneClassification(
            depth=REVIEW_DEPTH_STANDARD,
            checklist_points=STANDARD_CHECKLIST_POINTS,
            reasons=[f"Standard element detected: {label}" for label in standard_detected],
            detected_elements=detected,
        )

    # Check simple patterns
    simple_detected: list[str] = []
    for pattern, label in _SIMPLE_PATTERNS:
        if re.search(pattern, lowered):
            simple_detected.append(label)
    if simple_detected:
        detected.extend(simple_detected)
        return SceneClassification(
            depth=REVIEW_DEPTH_SIMPLE,
            checklist_points=SIMPLE_CHECKLIST_POINTS,
            reasons=[f"Simple element detected: {label}" for label in simple_detected],
            detected_elements=detected,
        )

    # Default to standard (safer)
    return SceneClassification(
        depth=REVIEW_DEPTH_STANDARD,
        checklist_points=STANDARD_CHECKLIST_POINTS,
        reasons=["No complexity markers found; defaulting to standard review"],
        detected_elements=[],
    )


# ---------------------------------------------------------------------------
# Checklists
# ---------------------------------------------------------------------------

SIMPLE_CHECKLIST: list[str] = [
    "Code parses without syntax errors",
    "Scene renders without crashing",
    "Brand colors are used",
    "Duration matches specification",
    "No debug/console.log statements left in code",
]

STANDARD_CHECKLIST: list[str] = [
    "Code parses without syntax errors",
    "Scene renders without crashing",
    "Brand colors are used",
    "Brand fonts are used",
    "Duration matches specification",
    "Narration timing aligns with reads",
    "No debug/console.log statements left in code",
    "Safe area is respected",
    "Minimum type size is respected",
    "Animation easing uses brand curve",
]

COMPLEX_CHECKLIST: list[str] = [
    "Code parses without syntax errors",
    "Scene renders without crashing",
    "Brand colors are used",
    "Brand fonts are used",
    "Duration matches specification",
    "Narration timing aligns with reads",
    "No debug/console.log statements left in code",
    "Safe area is respected",
    "Minimum type size is respected",
    "Animation easing uses brand curve",
    "Determinism: same frame twice = same output",
    "No external network calls at runtime",
    "Memory usage is bounded (no leaks)",
    "Frame rate is stable (no dropped frames)",
    "Color space is correct (sRGB or specified)",
    "Anti-aliasing is enabled",
    "Z-ordering is correct for 3D scenes",
    "Lighting is consistent across frames",
    "Camera framing is intentional (not accidental)",
    "Export format matches project spec",
]


def get_checklist(depth: str) -> list[str]:
    """Return the QA checklist for the given review depth."""
    if depth == REVIEW_DEPTH_SIMPLE:
        return SIMPLE_CHECKLIST
    if depth == REVIEW_DEPTH_STANDARD:
        return STANDARD_CHECKLIST
    if depth == REVIEW_DEPTH_COMPLEX:
        return COMPLEX_CHECKLIST
    return STANDARD_CHECKLIST  # safe default
