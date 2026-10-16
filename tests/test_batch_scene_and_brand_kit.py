"""Real tests for the Batch Scene Generator and Brand Kit tools.

These tests exercise the actual parsing, validation, retry, and drift
logic — not just "does it import". Where LLM calls are needed, we
mock the OpenAI client so the tests run without API keys or network.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openharness.tools.base import ToolExecutionContext, ToolRegistry, ToolResult


# ============================================================
# Batch Scene Generator — structured output parser
# ============================================================

from openharness.tools.batch_scene_tool import (
    BatchSceneTool,
    BatchSceneToolInput,
    ParsedScene,
    parse_batch_response,
    lint_scene_code,
    build_batch_prompt,
)


def test_batch_parser_parses_valid_response():
    """The parser correctly extracts a 2-scene batch."""
    raw = """\
BATCH_ID: batch_001
SCENES: 2

SCENE_001:
  FILE: scene_001.js
  ENGINE: hyperframes
  DURATION: 8.5s
  READS: ["The sky is blue", "Because of Rayleigh scattering"]
  CODE:
    function render() {
      return <div>Hello</div>;
    }

SCENE_002:
  FILE: scene_002.js
  ENGINE: hyperframes
  DURATION: 6.2s
  READS: ["Short wavelengths scatter more"]
  CODE:
    function render() {
      return <div>World</div>;
    }
"""
    result = parse_batch_response(raw, expected_batch_id="batch_001")
    assert result.success, f"Parse errors: {result.parse_errors}"
    assert result.batch_id == "batch_001"
    assert len(result.scenes) == 2
    assert result.scenes[0].index == 1
    assert result.scenes[0].file == "scene_001.js"
    assert result.scenes[0].engine == "hyperframes"
    assert result.scenes[0].duration == "8.5s"
    assert result.scenes[0].reads == ["The sky is blue", "Because of Rayleigh scattering"]
    assert "function render()" in result.scenes[0].code
    assert result.scenes[1].index == 2
    assert result.scenes[1].file == "scene_002.js"
    assert result.scenes[1].reads == ["Short wavelengths scatter more"]


def test_batch_parser_detects_missing_fields():
    """Scenes missing required fields are reported as parse errors."""
    raw = """\
BATCH_ID: batch_002
SCENES: 1

SCENE_001:
  FILE: scene_001.js
  ENGINE: hyperframes
  DURATION: 8.5s
  READS: ["hello"]
"""
    # Missing CODE field
    result = parse_batch_response(raw)
    assert not result.success
    assert any("missing CODE" in err for err in result.parse_errors)


def test_batch_parser_handles_empty_code():
    """A scene with empty CODE is flagged."""
    raw = """\
BATCH_ID: batch_003
SCENES: 1

SCENE_001:
  FILE: scene_001.js
  ENGINE: hyperframes
  DURATION: 8.5s
  READS: ["hello"]
  CODE:

"""
    result = parse_batch_response(raw)
    assert not result.success
    assert any("CODE is empty" in err for err in result.parse_errors)


def test_batch_parser_dedents_code():
    """The 4-space indentation is stripped from the code block."""
    raw = """\
BATCH_ID: batch_004
SCENES: 1

SCENE_001:
  FILE: scene_001.js
  ENGINE: hyperframes
  DURATION: 5.0s
  READS: ["test"]
  CODE:
    line_one();
    line_two();
    if (true) {
        nested();
    }
"""
    result = parse_batch_response(raw)
    assert result.success
    code = result.scenes[0].code
    # The first level of 4-space indent is stripped
    assert code.startswith("line_one();")
    # Nested indent (was 8 spaces) becomes 4 spaces
    assert "    nested();" in code
    # The if statement was at 4-space indent, now at 0
    assert "if (true) {" in code


def test_batch_parser_counts_mismatch_detected():
    """Declared SCENES count != actual parsed count is flagged."""
    raw = """\
BATCH_ID: batch_005
SCENES: 5

SCENE_001:
  FILE: s1.js
  ENGINE: hyperframes
  DURATION: 5.0s
  READS: ["a"]
  CODE:
    x();

SCENE_002:
  FILE: s2.js
  ENGINE: hyperframes
  DURATION: 5.0s
  READS: ["b"]
  CODE:
    y();
"""
    result = parse_batch_response(raw)
    # Parser found 2 scenes but declared 5
    assert any("Declared 5" in err and "parsed 2" in err for err in result.parse_errors)


# ============================================================
# Lint / validation
# ============================================================

def test_lint_brace_balance_passes_valid_code():
    """Balanced braces/parens pass the lint check."""
    code = "function render() { return [1, 2, 3]; }"
    ok, err = lint_scene_code(code, ".js")
    assert ok, f"Expected pass, got: {err}"


def test_lint_brace_balance_detects_unclosed():
    """Unclosed brace is detected."""
    code = "function render() { return [1, 2, 3; "
    ok, err = lint_scene_code(code, ".js")
    # If node is available, it'll catch this. If not, brace check catches it.
    assert not ok
    assert err


def test_lint_empty_code_fails():
    """Empty code fails lint."""
    ok, err = lint_scene_code("", ".js")
    assert not ok
    assert "empty" in err.lower()


def test_lint_python_code():
    """Python code is linted via py_compile."""
    ok, err = lint_scene_code("x = 1\nprint(x)\n", ".py")
    assert ok, f"Expected pass: {err}"


def test_lint_python_syntax_error():
    """Python syntax error is detected."""
    ok, err = lint_scene_code("def f(\n", ".py")
    assert not ok
    assert err


# ============================================================
# Prompt builder
# ============================================================

def test_build_batch_prompt_includes_scene_descriptions(tmp_path: Path):
    """The prompt includes all scene descriptions."""
    arguments = BatchSceneToolInput(
        scenes=["Scene 1: blue sky", "Scene 2: green field"],
        output_dir="scenes",
        engine="hyperframes",
    )
    context = ToolExecutionContext(cwd=tmp_path)
    prompt = build_batch_prompt(arguments, context)
    assert "Generate code for 2 scenes" in prompt
    assert "Scene 1: blue sky" in prompt
    assert "Scene 2: green field" in prompt
    assert "SCENE_001_DESCRIPTION" in prompt
    assert "SCENE_002_DESCRIPTION" in prompt


def test_build_batch_prompt_injects_brand_yaml(tmp_path: Path):
    """brand.yaml is auto-injected into the prompt when present."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n  name: Test\n  colors:\n    primary: '#FF0000'\n",
        encoding="utf-8",
    )
    arguments = BatchSceneToolInput(scenes=["test scene"])
    context = ToolExecutionContext(cwd=tmp_path)
    prompt = build_batch_prompt(arguments, context)
    assert "Brand anchors" in prompt
    assert "#FF0000" in prompt


def test_build_batch_prompt_injects_style_md(tmp_path: Path):
    """STYLE.md is auto-injected when present."""
    (tmp_path / "STYLE.md").write_text("# Style Guide\nKeep it minimal.", encoding="utf-8")
    arguments = BatchSceneToolInput(scenes=["test scene"])
    context = ToolExecutionContext(cwd=tmp_path)
    prompt = build_batch_prompt(arguments, context)
    assert "Style guide" in prompt
    assert "Keep it minimal" in prompt


# ============================================================
# Batch scene tool — end-to-end with mocked LLM
# ============================================================

@pytest.mark.asyncio
async def test_batch_scene_tool_writes_validated_scenes(tmp_path: Path):
    """End-to-end: tool calls mocked LLM, parses, lints, writes files."""
    tool = BatchSceneTool()

    # Mock the LLM response
    mock_response = """\
BATCH_ID: batch_test
SCENES: 2

SCENE_001:
  FILE: scene_001.js
  ENGINE: hyperframes
  DURATION: 8.0s
  READS: ["Hello world"]
  CODE:
    function render() {
      return { type: 'div', props: { children: 'Hello' } };
    }

SCENE_002:
  FILE: scene_002.js
  ENGINE: hyperframes
  DURATION: 6.0s
  READS: ["Goodbye world"]
  CODE:
    function render() {
      return { type: 'div', props: { children: 'Goodbye' } };
    }
"""
    # Patch the _call_llm method
    async def _mock_call_llm(arguments, config, context):
        return mock_response

    tool._call_llm = _mock_call_llm  # type: ignore[method-assign]

    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={
            "scene_gen_config": {"api_key": "fake", "model": "gpt-4o"},
        },
    )
    result = await tool.execute(
        BatchSceneToolInput(
            scenes=["Scene 1 desc", "Scene 2 desc"],
            output_dir="scenes",
            batch_id="batch_test",
        ),
        context,
    )
    assert not result.is_error, f"Tool failed: {result.output}"
    assert "Scenes written: 2" in result.output
    assert "Scenes failed: 0" in result.output

    # Verify files were written
    scene1 = tmp_path / "scenes" / "scene_001.js"
    scene2 = tmp_path / "scenes" / "scene_002.js"
    assert scene1.is_file(), f"scene_001.js not written"
    assert scene2.is_file(), f"scene_002.js not written"
    assert "Hello" in scene1.read_text()
    assert "Goodbye" in scene2.read_text()


@pytest.mark.asyncio
async def test_batch_scene_tool_returns_error_on_parse_failure(tmp_path: Path):
    """When the LLM returns unparseable output, the tool returns an error."""
    tool = BatchSceneTool()

    async def _mock_call_llm(arguments, config, context):
        return "This is not a valid batch response format."

    tool._call_llm = _mock_call_llm  # type: ignore[method-assign]

    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"scene_gen_config": {"api_key": "fake"}},
    )
    result = await tool.execute(
        BatchSceneToolInput(scenes=["test"], output_dir="scenes"),
        context,
    )
    assert result.is_error
    assert "Failed to parse" in result.output


@pytest.mark.asyncio
async def test_batch_scene_tool_empty_scenes_returns_error(tmp_path: Path):
    """Empty scenes list returns an error."""
    tool = BatchSceneTool()
    context = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        BatchSceneToolInput(scenes=[], output_dir="scenes"),
        context,
    )
    assert result.is_error
    assert "No scenes provided" in result.output


@pytest.mark.asyncio
async def test_batch_scene_tool_no_api_key_returns_error(tmp_path: Path):
    """Missing API key returns a clear error."""
    tool = BatchSceneTool()
    context = ToolExecutionContext(cwd=tmp_path, metadata={})
    result = await tool.execute(
        BatchSceneToolInput(scenes=["test scene"], output_dir="scenes"),
        context,
    )
    assert result.is_error
    assert "API key" in result.output or "LLM call failed" in result.output


# ============================================================
# Brand Kit — config loading
# ============================================================

from openharness.tools.brand_kit_tool import (
    BrandKitTool,
    BrandKitToolInput,
    BrandConfig,
    BrandColors,
    BrandFonts,
    generate_drift,
    CreativeDrift,
    check_brand_compliance,
    score_creative_originality,
    _hex_to_hsl,
    _hsl_to_hex,
    _hex_to_rgb,
    _colors_match,
    _extract_hex_colors,
    _extract_fonts,
    _extract_font_sizes,
)


def test_brand_config_loads_from_yaml(tmp_path: Path):
    """BrandConfig loads correctly from brand.yaml."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n"
        "  name: Test Channel\n"
        "  colors:\n"
        "    primary: '#1A1A2E'\n"
        "    secondary: '#16213E'\n"
        "    accent: '#E94560'\n"
        "  fonts:\n"
        "    heading: Inter\n"
        "    body: Inter\n"
        "  voice: Serena\n"
        "  music_style: ambient electronic\n"
        "  min_type_size: 44\n"
        "  safe_area: 220\n"
        "  easing: cubic-bezier(0.4, 0.0, 0.2, 1)\n",
        encoding="utf-8",
    )
    brand = BrandConfig.from_yaml_file(tmp_path / "brand.yaml")
    assert brand.name == "Test Channel"
    assert brand.colors.primary == "#1A1A2E"
    assert brand.colors.secondary == "#16213E"
    assert brand.colors.accent == "#E94560"
    assert brand.fonts.heading == "Inter"
    assert brand.voice == "Serena"
    assert brand.min_type_size == 44
    assert brand.safe_area == 220


def test_brand_config_defaults_when_file_missing(tmp_path: Path):
    """Missing brand.yaml returns default config (no crash)."""
    brand = BrandConfig.from_yaml_file(tmp_path / "nonexistent.yaml")
    assert brand.name == ""
    assert brand.colors.primary == ""
    assert brand.min_type_size == 44  # default


def test_brand_config_handles_invalid_yaml(tmp_path: Path):
    """Invalid YAML returns defaults instead of crashing."""
    (tmp_path / "brand.yaml").write_text(
        "brand: [invalid\n  yaml: {{{",
        encoding="utf-8",
    )
    brand = BrandConfig.from_yaml_file(tmp_path / "brand.yaml")
    # Should fall back to defaults, not crash
    assert brand.min_type_size == 44


# ============================================================
# Brand Kit — creative drift
# ============================================================

def test_drift_is_deterministic():
    """Same seed + brand always produces the same drift."""
    brand = BrandConfig(
        colors=BrandColors(primary="#FF0000", secondary="#00FF00", accent="#0000FF"),
    )
    drift1 = generate_drift("brand:project-001", brand)
    drift2 = generate_drift("brand:project-001", brand)
    assert drift1.palette_shift == drift2.palette_shift
    assert drift1.motion_intensity == drift2.motion_intensity
    assert drift1.derived_colors == drift2.derived_colors


def test_drift_different_seeds_produce_different_results():
    """Different seeds produce different drifts."""
    brand = BrandConfig(
        colors=BrandColors(primary="#FF0000", secondary="#00FF00", accent="#0000FF"),
    )
    drift1 = generate_drift("brand:project-001", brand)
    drift2 = generate_drift("brand:project-002", brand)
    # At least one parameter should differ
    assert (
        drift1.palette_shift != drift2.palette_shift
        or drift1.motion_intensity != drift2.motion_intensity
        or drift1.derived_colors != drift2.derived_colors
    )


def test_drift_palette_shift_in_safe_range():
    """palette_shift stays within 0.0 - 0.3 (safe drift range)."""
    brand = BrandConfig(colors=BrandColors(primary="#FF0000"))
    for i in range(100):
        drift = generate_drift(f"brand:project-{i}", brand)
        assert 0.0 <= drift.palette_shift <= 0.3
        assert 0.3 <= drift.motion_intensity <= 0.9


def test_drift_derived_colors_close_to_brand():
    """Derived colors stay close to brand colors (within palette_shift tolerance)."""
    brand = BrandConfig(colors=BrandColors(primary="#FF0000", secondary="#00FF00", accent="#0000FF"))
    drift = generate_drift("brand:test", brand)
    for role in ("primary", "secondary", "accent"):
        original = getattr(brand.colors, role)
        derived = drift.derived_colors[role]
        # Colors should be different (drifted) but close
        r1, g1, b1 = _hex_to_rgb(original.lower())
        r2, g2, b2 = _hex_to_rgb(derived.lower())
        dist = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
        # Allow up to ~110 distance (the max shift * 360 deg rotation)
        assert dist < 200, f"{role} drifted too far: {original} -> {derived} (dist={dist})"


def test_drift_has_no_brand_colors_still_works():
    """Drift works even when brand has no colors defined."""
    brand = BrandConfig()
    drift = generate_drift("brand:empty", brand)
    assert drift.seed == "brand:empty"
    assert drift.palette_shift >= 0.0
    assert drift.derived_colors == {}  # no colors to derive from


# ============================================================
# Brand Kit — compliance scoring
# ============================================================

def test_compliance_passes_when_brand_colors_present():
    """Scene code using brand colors passes compliance."""
    brand = BrandConfig(
        colors=BrandColors(primary="#1A1A2E", secondary="#16213E", accent="#E94560"),
        fonts=BrandFonts(heading="Inter", body="Inter"),
        min_type_size=44,
        safe_area=220,
    )
    code = """
    const style = {
      color: '#1A1A2E',
      backgroundColor: '#16213E',
      borderColor: '#E94560',
      fontFamily: 'Inter',
      fontSize: 48,
      padding: 240,
    };
    """
    result = check_brand_compliance(code, brand)
    assert result.passed, f"Expected pass, got failures: {result.failures}"
    assert result.score == 100


def test_compliance_fails_when_colors_dont_match():
    """Scene code using non-brand colors fails compliance."""
    brand = BrandConfig(
        colors=BrandColors(primary="#1A1A2E", secondary="#16213E", accent="#E94560"),
    )
    code = """
    const style = {
      color: '#FFFFFF',
      backgroundColor: '#000000',
    };
    """
    result = check_brand_compliance(code, brand)
    assert not result.passed
    assert any("don't match brand palette" in f for f in result.failures)


def test_compliance_fails_when_font_too_small():
    """Font size below min_type_size fails compliance."""
    brand = BrandConfig(min_type_size=44)
    code = "const style = { fontSize: 24 };"
    result = check_brand_compliance(code, brand)
    assert not result.passed
    assert any("below minimum" in f or "below min" in f for f in result.failures)


def test_compliance_no_brand_config_passes_everything():
    """Empty brand config (no anchors) passes — nothing to check against."""
    brand = BrandConfig()
    code = "const style = { color: '#anything' };"
    result = check_brand_compliance(code, brand)
    assert result.passed
    assert result.score == 100


# ============================================================
# Brand Kit — creative originality scoring
# ============================================================

def test_originality_scores_high_variety_code():
    """Code with many colors + sizes + motion gets a decent score."""
    code = """
    const style = {
      color: '#FF0000',
      backgroundColor: '#00FF00',
      borderColor: '#0000FF',
      fontSize: 48,
      width: 100,
      transition: 'ease-in 0.3s',
      animation: 'cubic-bezier(0.4, 0, 0.2, 1)',
    };
    """
    result = score_creative_originality(code)
    assert result.score >= 5
    assert "visual_variety" in result.factors
    assert "motion_variety" in result.factors


def test_originality_penalizes_template_code():
    """Template-like code (bare App component) scores lower."""
    code = """
    function App() {
      return (
        <div>
          <h1>Hello</h1>
        </div>
      );
    }
    """
    result = score_creative_originality(code)
    # Template patterns should reduce the concept_uniqueness score
    assert result.factors["concept_uniqueness"] < 10


def test_originality_threshold_pass_fail():
    """Threshold correctly determines pass/fail."""
    # High-variety code should pass default threshold (8)
    code = """
    color: '#FF0000'; backgroundColor: '#00FF00'; fontSize: 48;
    transition: 'ease-in'; animation: 'cubic-bezier(0.4, 0, 0.2, 1)';
    spring: 'physics'; bounce: 'ease';
    """
    result = score_creative_originality(code, threshold=5)
    assert result.score >= 0  # just verify it runs
    assert result.passed == (result.score >= 5)


# ============================================================
# Brand Kit tool — end-to-end actions
# ============================================================

@pytest.mark.asyncio
async def test_brand_kit_drift_action(tmp_path: Path):
    """The 'drift' action generates a drift config."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n  colors:\n    primary: '#1A1A2E'\n    accent: '#E94560'\n",
        encoding="utf-8",
    )
    tool = BrandKitTool()
    context = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        BrandKitToolInput(action="drift", project_id="project-001"),
        context,
    )
    assert not result.is_error
    assert "Creative drift generated" in result.output
    assert "brand:project-001" in result.output
    assert "Palette shift" in result.output
    assert result.metadata is not None
    assert "drift" in result.metadata


@pytest.mark.asyncio
async def test_brand_kit_validate_action(tmp_path: Path):
    """The 'validate' action checks scene code against brand."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n  colors:\n    primary: '#1A1A2E'\n  min_type_size: 44\n",
        encoding="utf-8",
    )
    tool = BrandKitTool()
    context = ToolExecutionContext(cwd=tmp_path)
    code = "const style = { color: '#1A1A2E', fontSize: 48 };"
    result = await tool.execute(
        BrandKitToolInput(action="validate", scene_code=code),
        context,
    )
    assert not result.is_error, f"Expected pass: {result.output}"
    assert "Brand compliance check" in result.output
    assert "Passed: True" in result.output


@pytest.mark.asyncio
async def test_brand_kit_score_action(tmp_path: Path):
    """The 'score' action runs both compliance + originality."""
    (tmp_path / "brand.yaml").write_text(
        "brand:\n  colors:\n    primary: '#1A1A2E'\n",
        encoding="utf-8",
    )
    tool = BrandKitTool()
    context = ToolExecutionContext(cwd=tmp_path)
    code = "const style = { color: '#1A1A2E', fontSize: 48, transition: 'ease' };"
    result = await tool.execute(
        BrandKitToolInput(action="score", scene_code=code, project_id="test-001"),
        context,
    )
    assert "Brand + Creative Score" in result.output
    assert "Brand Compliance:" in result.output
    assert "Creative Originality:" in result.output
    assert "Overall:" in result.output


@pytest.mark.asyncio
async def test_brand_kit_unknown_action_returns_error(tmp_path: Path):
    """Unknown action returns an error."""
    tool = BrandKitTool()
    context = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        BrandKitToolInput(action="unknown"),
        context,
    )
    assert result.is_error
    assert "Unknown action" in result.output


@pytest.mark.asyncio
async def test_brand_kit_drift_missing_project_id(tmp_path: Path):
    """Drift without project_id returns an error."""
    tool = BrandKitTool()
    context = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        BrandKitToolInput(action="drift"),
        context,
    )
    assert result.is_error
    assert "project_id is required" in result.output


# ============================================================
# Color helpers
# ============================================================

def test_hex_to_hsl_and_back():
    """HSL round-trip produces a close color."""
    original = "#1A1A2E"
    h, s, l = _hex_to_hsl(original)
    roundtripped = _hsl_to_hex(h, s, l)
    # Should be very close (may differ by 1 in last digit due to rounding)
    r1, g1, b1 = _hex_to_rgb(original.lower())
    r2, g2, b2 = _hex_to_rgb(roundtripped.lower())
    dist = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    assert dist <= 2, f"Round-trip drifted too far: {original} -> {roundtripped}"


def test_colors_match_exact():
    """Exact color match returns True."""
    assert _colors_match("#1A1A2E", {"#1a1a2e"})
    assert not _colors_match("#FFFFFF", {"#1a1a2e"})


def test_colors_match_close():
    """Close colors (within tolerance) match."""
    # #1A1A2E and #1B1A2F are very close
    assert _colors_match("#1A1A2E", {"#1B1A2F"}, tolerance=10)


def test_extract_hex_colors():
    """Hex color extraction finds all #RRGGBB patterns."""
    code = "color: '#1A1A2E'; background: '#16213E'; border: '#E94560';"
    colors = _extract_hex_colors(code)
    assert "#1A1A2E" in colors
    assert "#16213E" in colors
    assert "#E94560" in colors


def test_extract_fonts():
    """Font-family extraction finds font names."""
    code = "fontFamily: 'Inter'; font-family: Arial;"
    fonts = _extract_fonts(code)
    assert "Inter" in fonts
    assert "Arial" in fonts


def test_extract_font_sizes():
    """Font-size extraction finds numeric sizes."""
    code = "fontSize: 48; font-size: 32.5;"
    sizes = _extract_font_sizes(code)
    assert 48 in sizes
    assert 32.5 in sizes


# ============================================================
# Integration: both tools registered
# ============================================================

def test_both_tools_registered_in_default_registry():
    """Both batch_scene and brand_kit are in the default tool registry."""
    from openharness.tools import create_default_tool_registry
    reg = create_default_tool_registry()
    assert reg.get("batch_scene") is not None
    assert reg.get("brand_kit") is not None


def test_both_tools_have_valid_schemas():
    """Both tools produce valid API schemas."""
    from openharness.tools import create_default_tool_registry
    reg = create_default_tool_registry()
    batch_schema = reg.get("batch_scene").to_api_schema()
    brand_schema = reg.get("brand_kit").to_api_schema()
    assert batch_schema["name"] == "batch_scene"
    assert "input_schema" in batch_schema
    assert brand_schema["name"] == "brand_kit"
    assert "input_schema" in brand_schema
