"""Tests for the media pipeline tools (items 23-27).

These tests exercise the REAL behaviour:
- AudioQA measures an ffmpeg-generated file (real LUFS / true-peak /
  silence numbers) and skips are reported honestly.
- DeterminismChecker statically flags non-deterministic constructs.
- ContactSheet composites a real PNG grid via ffmpeg.
- CostGovernor tracks budget.
- EngineRouter routes by keywords.

ffmpeg is available in the test environment; when it is not, the
ffmpeg-dependent tests skip rather than fake a pass.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from openharness.tools.base import ToolExecutionContext
from openharness.tools.missing_tools import (
    AudioQATool,
    AudioQAToolInput,
    ContactSheetTool,
    ContactSheetToolInput,
    CostGovernor,
    CostGovernorTool,
    CostGovernorToolInput,
    DeterminismCheckerTool,
    DeterminismCheckerToolInput,
    EngineRouterTool,
    EngineRouterToolInput,
    route_engine,
)

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _make_tone(path: Path, *, seconds: float = 3.0, freq: int = 440) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-ar", "44100", "-y", str(path)],
        check=True, capture_output=True, text=True,
    )


def _make_color_png(path: Path, color: str = "red") -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color={color}:size=640x360:duration=1",
         "-frames:v", "1", "-y", str(path)],
        check=True, capture_output=True, text=True,
    )


# ============================================================
# #23 Cost Governor
# ============================================================

def test_cost_governor_tracks_spending():
    gov = CostGovernor(budget_usd=10.0)
    gov.record_spend(3.0, call_type="scene_gen")
    assert gov.spent == 3.0
    assert gov.remaining == 7.0
    assert gov.should_stop() is False


def test_cost_governor_halts_when_budget_exceeded():
    gov = CostGovernor(budget_usd=1.0)
    gov.record_spend(0.5)
    assert gov.should_stop() is False
    gov.record_spend(0.6)  # total 1.1 > 1.0
    assert gov.should_stop() is True


def test_cost_governor_stats():
    gov = CostGovernor(budget_usd=5.0)
    gov.record_spend(1.0)
    gov.record_spend(2.0)
    stats = gov.stats()
    assert stats["spent_usd"] == 3.0
    assert stats["budget_usd"] == 5.0
    assert stats["total_calls"] == 2
    assert stats["utilization"] == 0.6


@pytest.mark.asyncio
async def test_cost_governor_tool_check(tmp_path: Path):
    CostGovernorTool._governor = None  # reset singleton
    tool = CostGovernorTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(CostGovernorToolInput(action="check"), ctx)
    assert not result.is_error
    assert "Budget OK" in result.output


# ============================================================
# #24 Audio QA (real measurements)
# ============================================================

@pytest.mark.asyncio
async def test_audio_qa_missing_file(tmp_path: Path):
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(AudioQAToolInput(audio_path="nonexistent.wav"), ctx)
    assert result.is_error
    assert "not found" in result.output


@requires_ffmpeg
@pytest.mark.asyncio
async def test_audio_qa_measures_real_tone(tmp_path: Path):
    _make_tone(tmp_path / "tone.wav")
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        AudioQAToolInput(audio_path="tone.wav", target_lufs=-22.0, lufs_tolerance=3.0),
        ctx,
    )
    assert not result.is_error, result.output
    assert "[Audio QA: tone.wav]" in result.output
    # Real measurements must be present in the details.
    assert "dBTP" in result.output       # clipping detail carries the true peak
    assert "LUFS" in result.output       # levels detail carries integrated LUFS
    # A continuous 3s tone has no >= 2s silence gap.
    assert "No silence gaps" in result.output


@requires_ffmpeg
@pytest.mark.asyncio
async def test_audio_qa_detects_real_silence_gap(tmp_path: Path):
    # tone - 1s of digital silence - tone
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-f", "lavfi", "-i", "anullsrc=duration=1:r=44100",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1",
         "-ar", "44100", "-y", str(tmp_path / "gap.wav")],
        check=True, capture_output=True, text=True,
    )
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        AudioQAToolInput(audio_path="gap.wav", min_silence_seconds=0.5),
        ctx,
    )
    assert result.is_error  # the 1s gap must be flagged
    assert "silence gap" in result.output


@requires_ffmpeg
@pytest.mark.asyncio
async def test_audio_qa_rejects_undecodable_file(tmp_path: Path):
    (tmp_path / "garbage.wav").write_bytes(b"\x00" * 1000)
    tool = AudioQATool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(AudioQAToolInput(audio_path="garbage.wav"), ctx)
    assert result.is_error
    assert "Could not decode" in result.output


# ============================================================
# #25 Determinism Checker (static analysis)
# ============================================================

@pytest.mark.asyncio
async def test_determinism_flags_random_and_time():
    tool = DeterminismCheckerTool()
    ctx = ToolExecutionContext(cwd=Path("/tmp"))
    result = await tool.execute(
        DeterminismCheckerToolInput(
            scene_code=(
                "const jitter = Math.random();\n"
                "const now = Date.now();\n"
                "function render(frame) { return frame + jitter; }\n"
            ),
            scene_file="scene_001.js",
        ),
        ctx,
    )
    assert result.is_error
    assert "line 1" in result.output and "Math.random()" in result.output
    assert "line 2" in result.output and "Date.now()" in result.output


@pytest.mark.asyncio
async def test_determinism_passes_clean_code():
    tool = DeterminismCheckerTool()
    ctx = ToolExecutionContext(cwd=Path("/tmp"))
    result = await tool.execute(
        DeterminismCheckerToolInput(scene_code="function render(frame) { return frame * 2; }"),
        ctx,
    )
    assert not result.is_error
    assert "PASS" in result.output


@pytest.mark.asyncio
async def test_determinism_flags_python_sources():
    tool = DeterminismCheckerTool()
    ctx = ToolExecutionContext(cwd=Path("/tmp"))
    result = await tool.execute(
        DeterminismCheckerToolInput(
            scene_code="import random\nimport uuid\nx = random.uniform(0, 1)\nu = uuid.uuid4()\n",
        ),
        ctx,
    )
    assert result.is_error
    assert "random.uniform" in result.output
    assert "uuid.uuid4" in result.output


# ============================================================
# #26 Contact Sheet (real compositing)
# ============================================================

@pytest.mark.asyncio
async def test_contact_sheet_no_files(tmp_path: Path):
    tool = ContactSheetTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(ContactSheetToolInput(scene_files=[]), ctx)
    assert result.is_error


@requires_ffmpeg
@pytest.mark.asyncio
async def test_contact_sheet_composites_real_grid(tmp_path: Path):
    for color in ("red", "blue", "green"):
        _make_color_png(tmp_path / f"{color}.png", color=color)
    tool = ContactSheetTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        ContactSheetToolInput(
            scene_files=["red.png", "blue.png", "green.png"],
            output_path="sheet.png",
            columns=2,
            thumbnail_width=320,
        ),
        ctx,
    )
    assert not result.is_error, result.output
    assert "[Contact Sheet Generated]" in result.output
    sheet = tmp_path / "sheet.png"
    assert sheet.is_file() and sheet.stat().st_size > 0
    # 2 columns x 2 rows of 320x180 cells = 640x360.
    probe = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(sheet)],
        capture_output=True, text=True, check=True,
    )
    assert "640,360" in probe.stdout
    # Manifest is written alongside.
    assert (tmp_path / "sheet.json").is_file()


@requires_ffmpeg
@pytest.mark.asyncio
async def test_contact_sheet_reports_undecodable_inputs(tmp_path: Path):
    (tmp_path / "fake.mp4").write_bytes(b"\x00" * 100)
    tool = ContactSheetTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        ContactSheetToolInput(scene_files=["fake.mp4"], output_path="sheet.png"),
        ctx,
    )
    assert result.is_error  # nothing compositable
    assert "NOT written" in result.output
    assert "Manifest" in result.output  # honest fallback is documented


@requires_ffmpeg
@pytest.mark.asyncio
async def test_contact_sheet_extracts_video_first_frames(tmp_path: Path):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=red:size=640x360:duration=0.5",
         "-pix_fmt", "yuv420p", "-y", str(tmp_path / "clip.mp4")],
        check=True, capture_output=True, text=True,
    )
    tool = ContactSheetTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        ContactSheetToolInput(scene_files=["clip.mp4"], output_path="sheet.png"),
        ctx,
    )
    assert not result.is_error, result.output
    assert (tmp_path / "sheet.png").is_file()


# ============================================================
# #27 Engine Router
# ============================================================

def test_engine_router_3d_to_blender():
    assert route_engine("3D animation of a rotating cube", ["blender", "manim", "hyperframes"]) == "blender"


def test_engine_router_math_to_manim():
    assert route_engine("Math equation animation with graph", ["manim", "hyperframes"]) == "manim"


def test_engine_router_video_to_ffmpeg():
    assert route_engine("overlay footage with clip compositing", ["ffmpeg", "hyperframes"]) == "ffmpeg"


def test_engine_router_defaults_to_hyperframes():
    assert route_engine("Kinetic typography title card", ["hyperframes"]) == "hyperframes"


@pytest.mark.asyncio
async def test_engine_router_tool_output():
    tool = EngineRouterTool()
    ctx = ToolExecutionContext(cwd=Path("/tmp"))
    result = await tool.execute(
        EngineRouterToolInput(scene_description="3D rotating cube"), ctx,
    )
    assert not result.is_error
    assert "blender" in result.output
    assert result.metadata["engine"] == "blender"
