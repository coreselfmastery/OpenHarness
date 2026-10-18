"""Wave-6 regression tests — the five remaining tool-level bugs on 5fdbaea.

1. style_memory   — StyleMemoryInput.project defaults to "" so the
                    read actions (suggest/projects/stats) validate
                    without a project argument.
2. contact_sheet  — mixed image+video inputs keep the caller's
                    final-cut order (images used to be hoisted before
                    all videos), and WITHOUT ffmpeg the tool still
                    writes a real PNG grid (OpenCV first frames + PIL
                    composite) instead of a manifest-only error.
3. scene_lint     — the JS position regex no longer false-positives on
                    CSS compounds (margin-left, padding-top,
                    scroll_left) or relative offsets (translateX: 0 /
                    translateY: 0), while absolute x/left/top values
                    are still checked against the safe area.
4. vlm_qa         — an empty `choices` list is a per-scene error (no
                    crash), and a present "verdict" can never bypass
                    the numeric pass_threshold.
5. prompt_tracker — record_result() auto-registers unknown prompts so
                    stats() shows them; best_version() skips retired
                    versions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.config.settings import MiniCPMVLMConfig
from openharness.tools.base import ToolExecutionContext

# Captured at import time, before any test hides ffmpeg.
_REAL_WHICH = shutil.which


def _no_ffmpeg_which(cmd, *args, **kwargs):
    """shutil.which replacement that pretends ffmpeg is not installed."""
    if str(cmd).lower() in {"ffmpeg", "ffprobe", "avconv"}:
        return None
    return _REAL_WHICH(cmd, *args, **kwargs)


def _make_video(path: Path, kind: str, frames: int = 12) -> Path:
    """Synthetic test video (same generator as test_video_qa_wave4)."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
    )
    if kind == "moving":
        for i in range(frames):
            frame = np.full((180, 320, 3), (235, 240, 242), dtype=np.uint8)
            cv2.rectangle(frame, (20 + i * 8, 60), (110 + i * 8, 120), (30, 30, 30), -1)
            writer.write(frame)
    elif kind == "frozen":
        frame = np.full((180, 320, 3), (100, 150, 200), dtype=np.uint8)
        cv2.rectangle(frame, (60, 60), (150, 120), (30, 30, 30), -1)
        for _ in range(frames):
            writer.write(frame)
    writer.release()
    return path


def _make_color_png(path: Path, color=(200, 40, 40), size=(320, 180)) -> Path:
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ======================================================================
# 1. style_memory — optional `project`
# ======================================================================

class TestStyleMemoryOptionalProject:
    @pytest.mark.asyncio
    async def test_read_actions_work_without_project(self, tmp_path: Path, ctx):
        """suggest/projects/stats used to raise ValidationError because
        `project` had no default."""
        from openharness.services.style_memory import (
            StyleMemory,
            get_style_memory,
            set_style_memory,
        )
        from openharness.tools.style_memory_tool import StyleMemoryInput, StyleMemoryTool

        saved = get_style_memory()
        set_style_memory(StyleMemory(tmp_path / "style.sqlite3"))
        try:
            for action in ("suggest", "projects", "stats"):
                arguments = StyleMemoryInput(action=action)  # no project kwarg
                result = await StyleMemoryTool().execute(arguments, ctx)
                assert not result.is_error, (action, result.output)
        finally:
            set_style_memory(saved)

    @pytest.mark.asyncio
    async def test_record_without_project_is_a_friendly_error(self, tmp_path: Path, ctx):
        from openharness.services.style_memory import (
            StyleMemory,
            get_style_memory,
            set_style_memory,
        )
        from openharness.tools.style_memory_tool import StyleMemoryInput, StyleMemoryTool

        saved = get_style_memory()
        set_style_memory(StyleMemory(tmp_path / "style.sqlite3"))
        try:
            result = await StyleMemoryTool().execute(
                StyleMemoryInput(action="record", style_name="editorial"), ctx
            )
            assert result.is_error
            assert "required" in result.output
        finally:
            set_style_memory(saved)


# ======================================================================
# 2. contact_sheet — order preservation + ffmpeg-free compositing
# ======================================================================

class TestContactSheetOrderAndNoFfmpeg:
    @pytest.mark.asyncio
    async def test_mixed_inputs_keep_final_cut_order(self, tmp_path: Path, ctx):
        """[video, image, video] stays in that order — images used to be
        hoisted before all videos, scrambling scene numbering and
        cumulative timestamps."""
        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        _make_video(tmp_path / "s1.mp4", "moving", frames=10)  # 1.0 s
        _make_color_png(tmp_path / "s2.png")                  # 0 s
        _make_video(tmp_path / "s3.mp4", "moving", frames=10)  # 1.0 s
        result = await ContactSheetTool().execute(
            ContactSheetToolInput(
                scene_files=["s1.mp4", "s2.png", "s3.mp4"],
                output_path="sheet.png",
                columns=3,
            ),
            ctx,
        )
        assert not result.is_error, result.output
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        order = [Path(f).name for f in manifest["included"]]
        assert order == ["s1.mp4", "s2.png", "s3.mp4"]
        assert [c["scene"] for c in manifest["cells"]] == [1, 2, 3]
        # videos contribute their duration, the image 0 — cumulative
        # timestamps follow the FINAL-CUT order, not a type grouping.
        ts = [c["timestamp_s"] for c in manifest["cells"]]
        assert ts[0] == pytest.approx(0.0)
        assert ts[1] == pytest.approx(1.0)
        assert ts[2] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_real_png_grid_without_ffmpeg(self, tmp_path: Path, ctx, monkeypatch):
        """Without ffmpeg the tool must still produce a real PNG contact
        sheet (OpenCV first frames + PIL grid) with populated manifest
        cells — it used to return an error with `included: []`."""
        from PIL import Image

        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        monkeypatch.setattr(shutil, "which", _no_ffmpeg_which)

        one = _make_video(tmp_path / "one.mp4", "moving", frames=10)  # 1.0 s
        two = _make_video(tmp_path / "two.mp4", "moving", frames=20)  # 2.0 s
        out = tmp_path / "sheet.png"
        result = await ContactSheetTool().execute(
            ContactSheetToolInput(
                scene_files=[str(one), str(two)], output_path=str(out), columns=2
            ),
            ctx,
        )
        assert not result.is_error, result.output
        assert out.is_file() and out.stat().st_size > 0
        with Image.open(out) as sheet:
            # 2 columns x 1 row of 320x180 cells.
            assert sheet.size == (640, 180)
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        assert manifest["image_composited"] is True
        assert [c["scene"] for c in manifest["cells"]] == [1, 2]
        assert manifest["cells"][0]["timestamp_s"] == pytest.approx(0.0)
        assert manifest["cells"][1]["timestamp_s"] == pytest.approx(1.0)
        assert manifest["cells"][1]["file"].endswith("two.mp4")

    @pytest.mark.asyncio
    async def test_mixed_order_also_preserved_without_ffmpeg(self, tmp_path: Path, ctx, monkeypatch):
        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        monkeypatch.setattr(shutil, "which", _no_ffmpeg_which)

        _make_video(tmp_path / "s1.mp4", "moving", frames=10)
        _make_color_png(tmp_path / "s2.png")
        _make_video(tmp_path / "s3.mp4", "moving", frames=10)
        result = await ContactSheetTool().execute(
            ContactSheetToolInput(
                scene_files=["s1.mp4", "s2.png", "s3.mp4"],
                output_path="sheet.png",
                columns=3,
            ),
            ctx,
        )
        assert not result.is_error, result.output
        assert (tmp_path / "sheet.png").is_file()
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        order = [Path(f).name for f in manifest["included"]]
        assert order == ["s1.mp4", "s2.png", "s3.mp4"]

    @pytest.mark.asyncio
    async def test_undecodable_video_still_reports_manifest_only(self, tmp_path: Path, ctx, monkeypatch):
        """A corrupt video with no ffmpeg and no OpenCV decoder is an
        honest manifest-only result, not a crash."""
        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        monkeypatch.setattr(shutil, "which", _no_ffmpeg_which)

        (tmp_path / "fake.mp4").write_bytes(b"\x00" * 100)
        result = await ContactSheetTool().execute(
            ContactSheetToolInput(scene_files=["fake.mp4"], output_path="sheet.png"),
            ctx,
        )
        assert result.is_error
        assert "NOT written" in result.output
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        assert manifest["cells"] == []


# ======================================================================
# 3. scene_lint — JS position regex false positives
# ======================================================================

class TestSceneLintPositionRegex:
    def test_css_compounds_and_relative_offsets_not_flagged(self):
        """margin-left / padding-top / scroll_left / translateX: 0 used
        to trip off_safe_area via word-boundary matches and absolute
        bounds applied to relative offsets."""
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        analyzer = StaticSceneAnalyzer()
        clean = (
            "const s = { margin-left: 10, padding-top: 12, scrollLeft: 4, "
            "scroll_left: 4, marginRight: 8, zIndex: 3, "
            "translateX: 0, translateY: 0 };"
        )
        report = analyzer.analyze_javascript(clean, scene="ok", path="<memory>")
        off_area = [i for i in report.issues if i.check == "off_safe_area"]
        assert off_area == [], [(i.line, i.message) for i in off_area]

    def test_absolute_positions_still_flagged(self):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        analyzer = StaticSceneAnalyzer()
        bad = "const t = { x: 1900, top: 5 }; t.left = 40;"
        report = analyzer.analyze_javascript(bad, scene="bad", path="<memory>")
        flagged = [i for i in report.issues if i.check == "off_safe_area"]
        axes = {i.message.split("-")[0] for i in flagged}
        assert {"x", "y"} <= axes
        assert len(flagged) == 3  # x: 1900, top: 5, left: 40

    def test_in_safe_area_position_not_flagged(self):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        analyzer = StaticSceneAnalyzer()
        report = analyzer.analyze_javascript(
            "const t = { x: 960, y: 540 };", scene="ok", path="<memory>"
        )
        assert not any(i.check == "off_safe_area" for i in report.issues)


# ======================================================================
# 4. vlm_qa — empty choices + verdict vs threshold
# ======================================================================

class TestVLMQAVerdictAndChoices:
    @staticmethod
    def _settings_ctx(tmp_path: Path) -> ToolExecutionContext:
        settings = SimpleNamespace(
            minicpm_v=MiniCPMVLMConfig(
                base_url="http://127.0.0.1:8001/v1", model="minicpm-test"
            )
        )
        return ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings})

    @pytest.mark.asyncio
    async def test_pass_verdict_cannot_bypass_threshold(self, tmp_path: Path, monkeypatch):
        """mean 2.5 < threshold 7.0 must FAIL even when the model emits
        verdict: "pass" (the rubric prompt always asks for one)."""
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_a.mp4", "moving")

        async def fake_post(**kwargs):
            return {
                "choices": [{"message": {"content": json.dumps({
                    "scores": {"composition": 2, "readability": 3},
                    "verdict": "pass",
                })}}],
                "usage": {},
            }

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)], frames_per_scene=2),
            self._settings_ctx(tmp_path),
        )
        entry = result.metadata["results"][0]
        assert entry["mean_score"] == pytest.approx(2.5)
        assert entry["passed"] is False
        assert "scene_a: FAIL" in result.output

    @pytest.mark.asyncio
    async def test_fail_verdict_blocks_pass_despite_high_scores(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_b.mp4", "moving")

        async def fake_post(**kwargs):
            return {
                "choices": [{"message": {"content": json.dumps({
                    "scores": {"composition": 9, "readability": 9},
                    "verdict": "fail",
                })}}],
                "usage": {},
            }

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)], frames_per_scene=2),
            self._settings_ctx(tmp_path),
        )
        entry = result.metadata["results"][0]
        assert entry["mean_score"] == pytest.approx(9.0)
        assert entry["passed"] is False

    @pytest.mark.asyncio
    async def test_high_scores_without_verdict_pass_on_threshold(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_c.mp4", "moving")

        async def fake_post(**kwargs):
            return {
                "choices": [{"message": {"content": json.dumps({
                    "scores": {"composition": 8, "readability": 8},
                })}}],
                "usage": {},
            }

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)], frames_per_scene=2),
            self._settings_ctx(tmp_path),
        )
        entry = result.metadata["results"][0]
        assert entry["passed"] is True

    @pytest.mark.asyncio
    async def test_empty_choices_is_per_scene_error_not_crash(self, tmp_path: Path, monkeypatch):
        """A 200 response with an empty choices list used to IndexError
        on ``choices[0]``."""
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_d.mp4", "moving")

        async def fake_post(**kwargs):
            return {"choices": [], "usage": {}}

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)], frames_per_scene=2),
            self._settings_ctx(tmp_path),
        )
        entry = result.metadata["results"][0]
        assert entry["error"] == "no choices in VLM response"
        assert entry["passed"] is False
        assert "no choices in VLM response" in result.output


# ======================================================================
# 5. prompt_tracker — orphan results + retired best_version
# ======================================================================

class TestPromptTrackerVisibility:
    def test_record_without_register_is_visible_in_stats(self, tmp_path: Path):
        """A result recorded without register_prompt() used to be
        invisible: stats() joins FROM prompts and found no row."""
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.record_result("orphan", passed=True, version=3)
        tracker.record_result("orphan", passed=False, version=3)
        stats = {s["prompt_id"]: s for s in tracker.stats()}
        assert "orphan" in stats
        assert stats["orphan"]["version"] == 3
        assert stats["orphan"]["uses"] == 2
        assert stats["orphan"]["passes"] == 1
        assert stats["orphan"]["success_rate"] == 0.5

    def test_best_version_skips_retired_versions(self, tmp_path: Path):
        """best_version() used to return retired versions (the retiring
        flag lived only in the prompts table it never joined)."""
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.register_prompt("hook", version=1)
        tracker.register_prompt("hook", version=2)
        for _ in range(4):
            tracker.record_result("hook", passed=True, version=1)  # rate 1.0
        for _ in range(4):
            tracker.record_result("hook", passed=False, version=2)  # rate 0.0
        assert tracker.best_version("hook")["version"] == 1
        tracker.retire_prompt("hook", version=1)
        # v1 retired -> the (worse) live v2 is the honest recommendation.
        assert tracker.best_version("hook")["version"] == 2

    def test_auto_register_keeps_existing_template(self, tmp_path: Path):
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.register_prompt("keep", version=1, template="ORIGINAL")
        tracker.record_result("keep", passed=True, version=1)
        stats = tracker.stats()
        assert stats[0]["template_preview"] == "ORIGINAL"
        assert stats[0]["retired"] is False
        assert stats[0]["uses"] == 1

    def test_best_version_none_when_all_retired(self, tmp_path: Path):
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.register_prompt("gone", version=1)
        tracker.record_result("gone", passed=True, version=1)
        tracker.retire_prompt("gone")
        assert tracker.best_version("gone") is None
