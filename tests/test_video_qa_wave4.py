"""Wave-4 video-intelligence tools: regression tests.

Covers the new QA stack and learning stores:

- frame_diff      — SSIM + optical-flow motion QA (service + tool)
- scene_lint      — static scene analysis before render (service + tool)
- vlm_qa          — selective MiniCPM-V rubric QA (parser + tool, mocked)
- contact_sheet   — upgraded: labels, timestamps, manifest cells
- pixel_qa        — regression: `checks` argument + palette as_hex API
- semantic_memory — SQLite + embeddings, remember/recall by meaning
- style_memory    — cross-project style suggestions
- failure_patterns— failure rates per scene type + strictness hints
- prompt_tracker  — versioned prompt success rates + retirement
- style_consistency — reference + cross-scene drift checks
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.config.settings import MiniCPMVLMConfig
from openharness.tools.base import ToolExecutionContext

# ======================================================================
# Helpers
# ======================================================================

def _make_video(path: Path, kind: str, frames: int = 12) -> Path:
    """Synthetic test video: moving | frozen | jumpcut."""
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
    elif kind == "jumpcut":
        for i in range(frames):
            shade = 30 if i < frames // 2 else 240
            writer.write(np.full((180, 320, 3), (shade, shade, shade), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ======================================================================
# 1. frame_diff — SSIM + optical flow
# ======================================================================

class TestFrameDiffService:
    def test_moving_video_passes(self, tmp_path: Path):
        from openharness.services.frame_diff import FrameDiffAnalyzer

        video = _make_video(tmp_path / "moving.mp4", "moving")
        report = FrameDiffAnalyzer().analyze_video(video, sample_fps=10.0)
        assert report.passed, [(i.check, i.message) for i in report.issues]
        assert report.frames_sampled == 12
        assert report.pairs_analyzed == 11
        assert 0.0 < report.mean_ssim < 1.0

    def test_frozen_video_flagged(self, tmp_path: Path):
        from openharness.services.frame_diff import FrameDiffAnalyzer

        video = _make_video(tmp_path / "frozen.mp4", "frozen")
        report = FrameDiffAnalyzer().analyze_video(video, sample_fps=10.0)
        checks = {i.check for i in report.issues}
        assert "frozen_motion" in checks or "stutter" in checks
        assert not report.passed
        assert all(i.confidence > 0 for i in report.issues)

    def test_jump_cut_flagged(self, tmp_path: Path):
        from openharness.services.frame_diff import FrameDiffAnalyzer

        video = _make_video(tmp_path / "cut.mp4", "jumpcut")
        report = FrameDiffAnalyzer().analyze_video(video, sample_fps=10.0)
        assert any(i.check == "jump_cut" for i in report.issues)
        assert not report.passed

    def test_expected_flow_window(self, tmp_path: Path):
        from openharness.services.frame_diff import FrameDiffAnalyzer

        analyzer = FrameDiffAnalyzer()
        video = _make_video(tmp_path / "moving.mp4", "moving")
        ok = analyzer.analyze_video(
            video, sample_fps=10.0, expected_flow_px_s=(5.0, 500.0)
        )
        assert ok.passed, [i.message for i in ok.issues]
        slow = analyzer.analyze_video(
            video, sample_fps=10.0, expected_flow_px_s=(500.0, 5000.0)
        )
        assert any(i.check == "motion_speed" for i in slow.issues)

    def test_ssim_identical_and_different(self):
        import numpy as np

        from openharness.services.frame_diff import ssim

        a = np.full((60, 80), 128.0, dtype=np.float32)
        assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)
        b = np.full((60, 80), 10.0, dtype=np.float32)
        # standard Wang et al. SSIM: flat 128 vs flat 10 keeps a small
        # luminance-term score (~0.16) — clearly dissimilar, not 1.0
        assert ssim(a, b) < 0.3
        assert ssim(a, b) < ssim(a, np.full((60, 80), 118.0, dtype=np.float32))

    def test_unreadable_video_raises_and_tool_reports(self, tmp_path: Path):
        from openharness.services.frame_diff import FrameDiffAnalyzer

        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"not a video")
        with pytest.raises((IOError, OSError)):
            FrameDiffAnalyzer().analyze_video(bogus)


class TestFrameDiffTool:
    @pytest.mark.asyncio
    async def test_tool_reports_failures_and_progress(self, tmp_path: Path, ctx):
        from openharness.services.progress_tracker import (
            ProgressTracker,
            get_progress_tracker,
            set_progress_tracker,
        )
        from openharness.tools.frame_diff_tool import FrameDiffInput, FrameDiffTool

        set_progress_tracker(ProgressTracker())
        _make_video(tmp_path / "frozen.mp4", "frozen")
        result = await FrameDiffTool().execute(
            FrameDiffInput(paths=[str(tmp_path)], sample_fps=10.0), ctx
        )
        assert result.is_error
        assert "frozen" in result.output
        assert "motion failure" in result.output
        tracker = get_progress_tracker()
        scene_stages = {
            name: record["stage"] if isinstance(record, dict) else record.stage
            for name, record in tracker.summary().get("scenes", {}).items()
        }
        if not scene_stages:  # summary shape differs — fall back to internals
            scene_stages = {
                name: record.stage for name, record in tracker._scenes.items()
            }
        assert scene_stages.get("frozen") == "failed"
        set_progress_tracker(None)

    @pytest.mark.asyncio
    async def test_tool_writes_json_report(self, tmp_path: Path, ctx):
        from openharness.tools.frame_diff_tool import FrameDiffInput, FrameDiffTool

        _make_video(tmp_path / "moving.mp4", "moving")
        report_path = tmp_path / "motion.json"
        result = await FrameDiffTool().execute(
            FrameDiffInput(paths=[str(tmp_path)], sample_fps=10.0,
                           report_path=str(report_path)), ctx
        )
        assert not result.is_error
        payload = json.loads(report_path.read_text())
        assert payload[0]["scene"] == "moving"
        assert payload[0]["passed"] is True

    @pytest.mark.asyncio
    async def test_tool_rejects_half_flow_window(self, tmp_path: Path, ctx):
        from openharness.tools.frame_diff_tool import FrameDiffInput, FrameDiffTool

        result = await FrameDiffTool().execute(
            FrameDiffInput(paths=[str(tmp_path)], expected_min_flow_px_s=5.0), ctx
        )
        assert result.is_error
        assert "BOTH" in result.output


# ======================================================================
# 2. scene_lint — static scene analysis
# ======================================================================

_BAD_PY = '''
import random
def build(scene):
    logo = scene.image("assets/logo.png")
    title = scene.text("Hello", font_size=12)
    title2 = scene.text("World", font_size=12)
    title3 = scene.text("!", font_size=12)
    box = scene.rect(color="#00FF00")
    scene.move_to([1200, 300])
    r = random.random()
    print(box, r)
'''

_BAD_JS = (
    "const t = { x: 1900, y: 100 };\n"
    "t.font = { fontSize: 9 };\n"
    "const jitter = Math.random();\n"
    "img.src = 'photo_1.png';\n"
)


class TestSceneLintService:
    def test_python_scene_full_battery(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "scene_bad.py"
        scene.write_text(_BAD_PY, encoding="utf-8")
        report = StaticSceneAnalyzer().analyze_file(
            scene, declared_assets=["other.png"], palette_hex=["#F2F0EB", "#111111"]
        )
        kinds = {i.check for i in report.issues}
        assert {"non_determinism", "undeclared_assets", "text_too_small",
                "unbranded_colors", "off_safe_area"} <= kinds
        assert not report.passed
        # every issue carries a line number and a confidence
        assert all(i.line >= 1 and 0 < i.confidence <= 1 for i in report.issues)

    def test_js_scene_full_battery(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "scene_bad.js"
        scene.write_text(_BAD_JS, encoding="utf-8")
        report = StaticSceneAnalyzer().analyze_file(scene, declared_assets=[])
        kinds = {i.check for i in report.issues}
        assert {"off_safe_area", "text_too_small", "non_determinism",
                "undeclared_assets"} <= kinds

    def test_declared_asset_not_flagged(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "scene_ok.py"
        scene.write_text(
            'def build(scene):\n    scene.image("assets/logo.png")\n', encoding="utf-8"
        )
        report = StaticSceneAnalyzer().analyze_file(
            scene, declared_assets=["logo.png"]
        )
        assert not any(i.check == "undeclared_assets" for i in report.issues)

    def test_clean_scene_passes(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "scene_good.py"
        scene.write_text(
            "def build(scene):\n    scene.text('OK', font_size=42)\n", encoding="utf-8"
        )
        report = StaticSceneAnalyzer().analyze_file(scene, palette_hex=["#F2F0EB"])
        assert report.passed, [i.message for i in report.issues]

    def test_syntax_error_reported(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "broken.py"
        scene.write_text("def build(:\n", encoding="utf-8")
        report = StaticSceneAnalyzer().analyze_file(scene)
        assert not report.passed
        assert any(i.check == "syntax" and i.severity == "fail" for i in report.issues)

    def test_url_assets_not_flagged(self, tmp_path: Path):
        from openharness.services.static_scene_analyzer import StaticSceneAnalyzer

        scene = tmp_path / "urls.py"
        scene.write_text(
            'def build(scene):\n    u = "https://cdn.example.com/logo.png"\n',
            encoding="utf-8",
        )
        report = StaticSceneAnalyzer().analyze_file(scene, declared_assets=[])
        assert not any(i.check == "undeclared_assets" for i in report.issues)


class TestSceneLintTool:
    @pytest.mark.asyncio
    async def test_tool_flags_and_reports(self, tmp_path: Path, ctx):
        from openharness.tools.scene_lint_tool import SceneLintInput, SceneLintTool

        (tmp_path / "scene_bad.py").write_text(_BAD_PY, encoding="utf-8")
        result = await SceneLintTool().execute(
            SceneLintInput(paths=[str(tmp_path / "scene_bad.py")],
                           assets=["other.png"], palette=["#F2F0EB"]), ctx
        )
        assert result.is_error
        assert "non_determinism" in result.output
        assert "BEFORE rendering" in result.output

    @pytest.mark.asyncio
    async def test_assets_manifest_json(self, tmp_path: Path, ctx):
        from openharness.tools.scene_lint_tool import SceneLintInput, SceneLintTool

        scene = tmp_path / "scene.py"
        scene.write_text(
            'def build(scene):\n    scene.image("assets/logo.png")\n', encoding="utf-8"
        )
        manifest = tmp_path / "assets.json"
        manifest.write_text(json.dumps({"assets": ["logo.png"]}), encoding="utf-8")
        result = await SceneLintTool().execute(
            SceneLintInput(paths=[str(scene)], assets_manifest=str(manifest)), ctx
        )
        assert not any(
            "logo.png" in line and "undeclared" in line
            for line in result.output.splitlines()
        )

    @pytest.mark.asyncio
    async def test_palette_file_via_tokens_json(self, tmp_path: Path, ctx):
        from openharness.tools.scene_lint_tool import SceneLintInput, SceneLintTool

        scene = tmp_path / "scene.py"
        scene.write_text('def build(scene):\n    scene.rect(color="#00FF00")\n',
                         encoding="utf-8")
        tokens = tmp_path / "tokens.json"
        tokens.write_text(
            json.dumps({"palette": {"bg": "#F2F0EB", "fg": "#111111"}}),
            encoding="utf-8",
        )
        result = await SceneLintTool().execute(
            SceneLintInput(paths=[str(scene)], palette_file=str(tokens)), ctx
        )
        assert "unbranded_colors" in result.output


# ======================================================================
# 3. vlm_qa — selective rubric review
# ======================================================================

class TestRubricParsing:
    def test_plain_json(self):
        from openharness.tools.vlm_qa_tool import parse_rubric_json

        payload = parse_rubric_json(
            '{"scores": {"composition": 8, "readability": 7}, "verdict": "pass"}'
        )
        assert payload["scores"]["composition"] == 8
        assert payload["verdict"] == "pass"

    def test_fenced_and_prose_wrapped(self):
        from openharness.tools.vlm_qa_tool import parse_rubric_json

        payload = parse_rubric_json(
            '```json\n{"scores": {"composition": 8}, "verdict": "pass"}\n```'
        )
        assert payload is not None
        payload = parse_rubric_json(
            'Sure!\n{"scores": {"composition": 3}, "verdict": "fail"} hope this helps'
        )
        assert payload["verdict"] == "fail"

    def test_garbage_returns_none(self):
        from openharness.tools.vlm_qa_tool import parse_rubric_json

        assert parse_rubric_json("") is None
        assert parse_rubric_json("no json at all") is None
        assert parse_rubric_json("{broken json}") is None


class TestVLMQATool:
    @staticmethod
    def _settings_ctx(tmp_path: Path) -> ToolExecutionContext:
        settings = SimpleNamespace(
            minicpm_v=MiniCPMVLMConfig(base_url="http://127.0.0.1:8001/v1",
                                       model="minicpm-test")
        )
        return ToolExecutionContext(cwd=tmp_path, metadata={"settings": settings})

    @pytest.mark.asyncio
    async def test_no_endpoint_configured(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        def fake_resolve(context):
            return MiniCPMVLMConfig(base_url="", model="")

        monkeypatch.setattr(vlm_module, "_resolve_config", fake_resolve)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(tmp_path / "x.mp4")]),
            ToolExecutionContext(cwd=tmp_path),
        )
        assert result.is_error
        assert "no MiniCPM-V endpoint" in result.output

    @pytest.mark.asyncio
    async def test_rubric_review_with_mocked_endpoint(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_a.mp4", "moving")
        calls: list[dict] = []

        async def fake_post(**kwargs):
            calls.append(kwargs["payload"])
            return {
                "choices": [{"message": {"content": json.dumps({
                    "scores": {"composition": 9, "readability": 8,
                               "brand_match": 7, "anomaly": 9},
                    "justification": "clean title card",
                    "verdict": "pass",
                })}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 60},
            }

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)], frames_per_scene=5),
            self._settings_ctx(tmp_path),
        )
        assert not result.is_error, result.output
        assert "scene_a: PASS" in result.output
        assert "composition 9" in result.output
        # multi-image request: text + 5 frames
        content = calls[0]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert sum(item["type"] == "image_url" for item in content) == 5
        assert "composition" in content[0]["text"]
        entry = result.metadata["results"][0]
        assert entry["mean_score"] == pytest.approx(8.25)
        assert entry["passed"] is True

    @pytest.mark.asyncio
    async def test_unparsable_model_response_isolated(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_b.mp4", "moving")

        async def fake_post(**kwargs):
            return {"choices": [{"message": {"content": "I cannot do that"}}],
                    "usage": {}}

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)]), self._settings_ctx(tmp_path)
        )
        assert result.is_error
        assert "did not return parsable" in result.output

    @pytest.mark.asyncio
    async def test_endpoint_down_per_scene(self, tmp_path: Path, monkeypatch):
        import openharness.tools.vlm_qa_tool as vlm_module
        from openharness.tools.vlm_qa_tool import VLMQAInput, VLMQATool

        video = _make_video(tmp_path / "scene_c.mp4", "moving")

        async def fake_post(**kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(vlm_module, "_post_chat_completion", fake_post)
        result = await VLMQATool().execute(
            VLMQAInput(videos=[str(video)]), self._settings_ctx(tmp_path)
        )
        assert result.is_error
        assert "endpoint unreachable" in result.output


# ======================================================================
# 4. contact_sheet — labels, timestamps, manifest
# ======================================================================

class TestContactSheetUpgrade:
    @pytest.mark.asyncio
    async def test_labels_timestamps_and_manifest(self, tmp_path: Path, ctx):
        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        one = _make_video(tmp_path / "one.mp4", "moving", frames=10)  # 1.0 s
        two = _make_video(tmp_path / "two.mp4", "moving", frames=20)  # 2.0 s
        out = tmp_path / "sheet.png"
        result = await ContactSheetTool().execute(
            ContactSheetToolInput(
                scene_files=[str(one), str(two)], output_path=str(out), columns=2
            ),
            ctx,
        )
        assert not result.is_error
        assert out.is_file() and out.stat().st_size > 0
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        cells = manifest["cells"]
        assert [c["scene"] for c in cells] == [1, 2]
        assert cells[0]["timestamp_s"] == pytest.approx(0.0)
        assert cells[1]["timestamp_s"] == pytest.approx(1.0)
        assert cells[1]["file"].endswith("two.mp4")

    @pytest.mark.asyncio
    async def test_explicit_timestamps_override(self, tmp_path: Path, ctx):
        from openharness.tools.missing_tools import ContactSheetTool, ContactSheetToolInput

        one = _make_video(tmp_path / "one.mp4", "moving")
        out = tmp_path / "sheet.png"
        await ContactSheetTool().execute(
            ContactSheetToolInput(
                scene_files=[str(one)], output_path=str(out),
                timestamps=[42.0],
            ),
            ctx,
        )
        manifest = json.loads((tmp_path / "sheet.json").read_text())
        assert manifest["cells"][0]["timestamp_s"] == 42.0

    def test_scene_labels_helper(self):
        from openharness.tools.missing_tools import _scene_labels

        assert _scene_labels([1.0, 2.0], None, 2) == [0.0, 1.0]
        assert _scene_labels([9.0, 9.0], [5.0, 6.0], 2) == [5.0, 6.0]
        assert _scene_labels([1.0], [5.0], 2) == [0.0, 1.0]  # short explicit -> cumulative


# ======================================================================
# 5. pixel_qa — regressions (checks arg + palette as_hex)
# ======================================================================

class TestPixelQARegressions:
    @pytest.mark.asyncio
    async def test_checks_subset_argument_works(self, tmp_path: Path, ctx):
        """Regression: the tool referenced ``arguments.checkes`` (typo) and
        crashed whenever the `checks` argument was passed."""
        from openharness.tools.pixel_qa_tool import PixelQAInput, PixelQATool

        video = _make_video(tmp_path / "v.mp4", "moving")
        result = await PixelQATool().execute(
            PixelQAInput(paths=[str(video)], checks=["black_frame"]), ctx
        )
        assert not result.is_error, result.output
        # only the requested check ran
        for report in result.metadata["reports"]:
            assert set(report["checks"]) <= {"black_frame"}

    def test_palette_as_hex(self, tmp_path: Path):
        from openharness.services.pixel_qa import load_brand_palette, palette_from_mapping

        tokens = tmp_path / "tokens.json"
        tokens.write_text(
            json.dumps({"palette": {"bg": "#F2F0EB", "fg": "#111111",
                                    "accent": "#E1251B"}}),
            encoding="utf-8",
        )
        hexes = load_brand_palette(tokens, as_hex=True)
        assert set(hexes) == {"#F2F0EB", "#111111", "#E1251B"}
        bgr = load_brand_palette(tokens)
        assert (235, 240, 242) in bgr  # #F2F0EB in BGR (OpenCV order)
        # mapping API keeps both shapes
        assert palette_from_mapping({"a": "#000000"}, as_hex=True) == ["#000000"]

    @pytest.mark.asyncio
    async def test_pixel_qa_still_passes_clean_video(self, tmp_path: Path, ctx):
        from openharness.tools.pixel_qa_tool import PixelQAInput, PixelQATool

        video = _make_video(tmp_path / "clean.mp4", "moving")
        result = await PixelQATool().execute(
            PixelQAInput(paths=[str(video)], palette=["#F2F0EB", "#1E1E1E"]), ctx
        )
        assert not result.is_error, result.output


# ======================================================================
# 6. semantic_memory
# ======================================================================

class TestSemanticMemory:
    def test_remember_recall_by_meaning(self, tmp_path: Path):
        from openharness.services.semantic_memory import SemanticMemory

        memory = SemanticMemory(tmp_path / "sem.sqlite3")
        memory.remember("scene 12 uses cubic-bezier(0.4, 0, 0.2, 1) for transitions",
                        project="proj")
        memory.remember("the brand accent color is #E1251B", project="proj")
        memory.remember("vllm renders math scenes fastest", project="other")
        hits = memory.recall("what easing did scene 12 use?", top_k=3, project="proj")
        assert hits and "cubic-bezier" in hits[0].text
        assert all(h.project == "proj" for h in hits)
        if len(hits) > 1:
            assert hits[0].similarity >= hits[1].similarity

    def test_forget_and_list(self, tmp_path: Path):
        from openharness.services.semantic_memory import SemanticMemory

        memory = SemanticMemory(tmp_path / "sem.sqlite3")
        memory_id = memory.remember("fact one", kind="lesson")
        memory.remember("fact two", kind="decision")
        entries = memory.list_recent(limit=10)
        assert len(entries) == 2
        assert memory.forget(memory_id)
        assert len(memory.list_recent(limit=10)) == 1
        assert not memory.forget(9999)

    def test_http_backend_and_dim_mismatch(self, tmp_path: Path, monkeypatch):
        import openharness.services.semantic_memory as module

        memory = module.SemanticMemory(tmp_path / "sem.sqlite3")

        # hashing entry first (256 dims), then an http entry (3 dims)
        memory.remember("hashing fact")
        memory._http = {"base_url": "http://x/v1", "model": "bge", "api_key": "E",
                        "timeout": 5.0}

        def fake_http_embed(base_url, model, api_key, texts, timeout):
            return [[0.1, 0.2, 0.3] for _ in texts]

        monkeypatch.setattr(module, "_http_embed", fake_http_embed)
        memory.remember("http-embedded fact")
        hits = memory.recall("http-embedded fact", top_k=5)
        assert any("http-embedded fact" == h.text for h in hits)
        # the 256-dim hashing entry is not comparable with the 3-dim query
        assert all(h.text != "hashing fact" for h in hits)
        stats = memory.stats()
        assert stats["by_backend"].get("http:bge") == 1

    def test_broken_store_degrades(self, tmp_path: Path):
        from openharness.services.semantic_memory import SemanticMemory

        broken = tmp_path / "sub" / "missing" / "sem.sqlite3"
        # a file where a directory is needed -> store disabled
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "missing").write_text("not a dir", encoding="utf-8")
        memory = SemanticMemory(broken)
        assert memory.remember("x") is None
        assert memory.recall("x") == []
        assert memory.stats()["available"] is False

    def test_singleton_setters(self, tmp_path: Path):
        from openharness.services.semantic_memory import (
            SemanticMemory,
            get_semantic_memory,
            set_semantic_memory,
        )

        original = get_semantic_memory()
        try:
            injected = SemanticMemory(tmp_path / "s2.sqlite3")
            set_semantic_memory(injected)
            assert get_semantic_memory() is injected
        finally:
            set_semantic_memory(original)


class TestSemanticMemoryTool:
    @pytest.mark.asyncio
    async def test_tool_remember_recall_flow(self, tmp_path: Path, ctx, monkeypatch):
        from openharness.services import semantic_memory as sm_module
        from openharness.services.semantic_memory import SemanticMemory, set_semantic_memory
        from openharness.tools.semantic_memory_tool import (
            SemanticMemoryInput,
            SemanticMemoryTool,
        )

        saved = sm_module._memory
        set_semantic_memory(SemanticMemory(tmp_path / "sem.sqlite3"))
        try:
            result = await SemanticMemoryTool().execute(
                SemanticMemoryInput(action="remember",
                                    text="use 24fps for knowledge shorts",
                                    project="proj", kind="lesson"), ctx
            )
            assert not result.is_error
            assert "remembered" in result.output
            result = await SemanticMemoryTool().execute(
                SemanticMemoryInput(action="recall", text="what fps for shorts?",
                                    project="proj"), ctx
            )
            assert "24fps" in result.output
            result = await SemanticMemoryTool().execute(
                SemanticMemoryInput(action="stats"), ctx
            )
            assert "entries: 1" in result.output
        finally:
            set_semantic_memory(saved)


# ======================================================================
# 7. style_memory
# ======================================================================

class TestStyleMemory:
    def test_record_suggest_ranking(self, tmp_path: Path):
        from openharness.services.style_memory import StyleMemory, StyleRecord

        store = StyleMemory(tmp_path / "style.sqlite3")
        store.record(StyleRecord("proj-a", "editorial-dark", "knowledge shorts",
                                 ["#F2F0EB"], ["Inter"], ["cubic-bezier(0.4,0,0.2,1)"],
                                 ["slide-in 0.6s"], 0.92))
        store.record(StyleRecord("proj-b", "bright", "promo", ["#FFF"], ["Carlito"],
                                 [], [], 0.5))
        suggestions = store.suggest(content_type="knowledge shorts")
        assert [s["style_name"] for s in suggestions] == ["editorial-dark"]
        all_styles = store.suggest(min_quality=0.0)
        assert all_styles[0]["quality_score"] >= all_styles[1]["quality_score"]
        projects = store.projects()
        assert {p["project"] for p in projects} == {"proj-a", "proj-b"}

    def test_record_upserts_same_project_style(self, tmp_path: Path):
        from openharness.services.style_memory import StyleMemory, StyleRecord

        store = StyleMemory(tmp_path / "style.sqlite3")
        store.record(StyleRecord("p", "s", quality_score=0.4))
        store.record(StyleRecord("p", "s", quality_score=0.9))
        suggestions = store.suggest()
        assert len(suggestions) == 1
        assert suggestions[0]["quality_score"] == 0.9

    def test_extract_from_tokens(self, tmp_path: Path):
        from openharness.services.style_memory import StyleMemory

        tokens = tmp_path / "tokens.json"
        tokens.write_text(json.dumps({
            "palette": {"bg": "#F2F0EB", "fg": "#111111"},
            "fonts": ["Inter", "Playfair Display"],
        }), encoding="utf-8")
        extracted = StyleMemory.extract_from_tokens(tokens)
        assert "#F2F0EB" in extracted["palette"]
        assert "Inter" in extracted["fonts"]


class TestStyleMemoryTool:
    @pytest.mark.asyncio
    async def test_record_and_suggest(self, tmp_path: Path, ctx):
        from openharness.services import style_memory as sm_module
        from openharness.services.style_memory import set_style_memory, StyleMemory
        from openharness.tools.style_memory_tool import StyleMemoryInput, StyleMemoryTool

        saved = sm_module._style_memory
        set_style_memory(StyleMemory(tmp_path / "style.sqlite3"))
        try:
            result = await StyleMemoryTool().execute(
                StyleMemoryInput(action="record", project="p1", style_name="warm-doc",
                                 content_type="knowledge shorts", quality_score=0.88,
                                 palette=["#F2F0EB"], fonts=["Inter"]), ctx
            )
            assert not result.is_error
            result = await StyleMemoryTool().execute(
                StyleMemoryInput(action="suggest", project="p1",
                                 content_type="knowledge shorts"), ctx
            )
            assert "warm-doc" in result.output
            result = await StyleMemoryTool().execute(
                StyleMemoryInput(action="projects", project="p1"), ctx
            )
            assert "p1" in result.output
        finally:
            set_style_memory(saved)


# ======================================================================
# 8. failure_patterns
# ======================================================================

class TestFailurePatterns:
    def test_rates_and_hint(self, tmp_path: Path):
        from openharness.services.failure_patterns import FailurePatterns

        store = FailurePatterns(tmp_path / "fail.sqlite3")
        for _ in range(4):
            store.record_failure("chart-overlay", reason="labels too small",
                                 prompt_id="chart-v1")
        for _ in range(2):
            store.record_success("chart-overlay", prompt_id="chart-v1")
        for _ in range(5):
            store.record_success("title-card", prompt_id="title-v1")

        rates = {r["scene_type"]: r for r in store.failure_rates()}
        assert rates["chart-overlay"]["failure_rate"] == pytest.approx(4 / 6, abs=0.01)
        assert rates["chart-overlay"]["top_reasons"][0]["reason"] == "labels too small"
        assert rates["title-card"]["failure_rate"] == 0.0

        hint = store.strictness_hint("chart-overlay")
        assert hint["high_risk"] is True
        assert "STRICTER" in hint["recommendation"].upper()
        assert "labels too small" in hint["recommendation"]

        hint = store.strictness_hint("title-card")
        assert hint["high_risk"] is False
        assert hint["samples"] == 5

    def test_hint_without_samples(self, tmp_path: Path):
        from openharness.services.failure_patterns import FailurePatterns

        store = FailurePatterns(tmp_path / "fail.sqlite3")
        hint = store.strictness_hint("never-seen")
        assert hint["samples"] == 0 and not hint["high_risk"]


class TestFailurePatternsTool:
    @pytest.mark.asyncio
    async def test_tool_flow(self, tmp_path: Path, ctx):
        from openharness.services import failure_patterns as fp_module
        from openharness.services.failure_patterns import FailurePatterns, set_failure_patterns
        from openharness.tools.failure_patterns_tool import (
            FailurePatternsInput,
            FailurePatternsTool,
        )
        saved = fp_module._failure_patterns
        set_failure_patterns(FailurePatterns(tmp_path / "fail.sqlite3"))
        try:
            result = await FailurePatternsTool().execute(
                FailurePatternsInput(action="record_failure", scene_type="chart",
                                     reason="tiny labels", prompt_id="p"), ctx
            )
            assert not result.is_error
            result = await FailurePatternsTool().execute(
                FailurePatternsInput(action="hint", scene_type="chart",
                                     min_samples=1), ctx
            )
            assert "HIGH RISK" in result.output
            assert "tiny labels" in result.output
            result = await FailurePatternsTool().execute(
                FailurePatternsInput(action="rates", min_samples=1), ctx
            )
            assert "chart" in result.output
        finally:
            set_failure_patterns(saved)

    @pytest.mark.asyncio
    async def test_record_failure_requires_reason(self, ctx):
        from openharness.tools.failure_patterns_tool import (
            FailurePatternsInput,
            FailurePatternsTool,
        )

        result = await FailurePatternsTool().execute(
            FailurePatternsInput(action="record_failure", scene_type="x"), ctx
        )
        assert result.is_error and "reason" in result.output


# ======================================================================
# 9. prompt_tracker
# ======================================================================

class TestPromptTracker:
    def test_version_success_rates_and_flags(self, tmp_path: Path):
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.register_prompt("title", version=1, template="t1")
        tracker.register_prompt("title", version=2, template="t2")
        for _ in range(6):
            tracker.record_result("title", passed=True, version=2)
        for _ in range(5):
            tracker.record_result("title", passed=False, version=1)

        stats = {s["version"]: s for s in tracker.stats()}
        assert stats[1]["success_rate"] == 0.0
        assert stats[2]["success_rate"] == 1.0
        assert stats[1]["flagged_low_performer"] is True
        assert stats[2]["flagged_low_performer"] is False

        assert tracker.best_version("title")["version"] == 2
        assert tracker.retire_prompt("title", version=1) == 1
        assert next(s for s in tracker.stats() if s["version"] == 1)["retired"] is True

    def test_register_keeps_template_on_empty_update(self, tmp_path: Path):
        from openharness.services.prompt_tracker import PromptTracker

        tracker = PromptTracker(tmp_path / "p.sqlite3")
        tracker.register_prompt("x", version=1, template="ORIGINAL")
        tracker.register_prompt("x", version=1, template="")
        stats = tracker.stats()
        assert stats[0]["template_preview"] == "ORIGINAL"


class TestPromptTrackerTool:
    @pytest.mark.asyncio
    async def test_tool_flow(self, tmp_path: Path, ctx):
        from openharness.services import prompt_tracker as pt_module
        from openharness.services.prompt_tracker import PromptTracker, set_prompt_tracker
        from openharness.tools.prompt_tracker_tool import (
            PromptTrackerInput,
            PromptTrackerTool,
        )
        saved = pt_module._prompt_tracker
        set_prompt_tracker(PromptTracker(tmp_path / "p.sqlite3"))
        try:
            await PromptTrackerTool().execute(
                PromptTrackerInput(action="register", prompt_id="p1", version=1,
                                   template="Make a card"), ctx
            )
            await PromptTrackerTool().execute(
                PromptTrackerInput(action="record", prompt_id="p1", passed=False), ctx
            )
            result = await PromptTrackerTool().execute(
                PromptTrackerInput(action="stats"), ctx
            )
            assert "p1 v1" in result.output
            result = await PromptTrackerTool().execute(
                PromptTrackerInput(action="best", prompt_id="p1"), ctx
            )
            assert "v1" in result.output
            result = await PromptTrackerTool().execute(
                PromptTrackerInput(action="retire", prompt_id="p1"), ctx
            )
            assert not result.is_error
        finally:
            set_prompt_tracker(saved)

    @pytest.mark.asyncio
    async def test_register_requires_template(self, ctx):
        from openharness.tools.prompt_tracker_tool import (
            PromptTrackerInput,
            PromptTrackerTool,
        )

        result = await PromptTrackerTool().execute(
            PromptTrackerInput(action="register", prompt_id="p1"), ctx
        )
        assert result.is_error and "template" in result.output


# ======================================================================
# 10. style_consistency
# ======================================================================

class TestStyleConsistency:
    @staticmethod
    def _scenes(tmp_path: Path):
        s1 = tmp_path / "s1.js"
        s1.write_text("fontFamily: 'Inter'; color: '#F2F0EB'; fontSize: 42;",
                      encoding="utf-8")
        s2 = tmp_path / "s2.js"
        s2.write_text("fontFamily: 'Inter'; color: '#F2F0EB';", encoding="utf-8")
        s3 = tmp_path / "s3.js"
        s3.write_text(
            "fontFamily: 'Comic Sans'; color: '#2A6FF2'; "
            "easing: cubic-bezier(0.55, 0, 1, 0.45);", encoding="utf-8")
        return s1, s2, s3

    def test_deviations_flagged_per_scene(self, tmp_path: Path):
        from openharness.services.style_consistency import (
            StyleConsistencyChecker,
            load_reference,
        )

        s1, s2, s3 = self._scenes(tmp_path)
        reference = load_reference(palette=["#F2F0EB"], fonts=["Inter"],
                                   easings=["cubic-bezier(0.4,0,0.2,1)"])
        report = StyleConsistencyChecker(reference).check_files([s1, s2, s3])
        kinds = {(i.kind, i.scene) for i in report.issues}
        assert ("palette_deviation", "s3") in kinds
        assert ("font_deviation", "s3") in kinds
        assert ("easing_deviation", "s3") in kinds
        assert not report.passed

    def test_consistent_project_passes(self, tmp_path: Path):
        from openharness.services.style_consistency import (
            StyleConsistencyChecker,
            load_reference,
        )

        s1, s2, _ = self._scenes(tmp_path)
        reference = load_reference(palette=["#F2F0EB"], fonts=["Inter"])
        report = StyleConsistencyChecker(reference).check_files([s1, s2, s1])
        assert report.passed, [(i.kind, i.message) for i in report.issues]

    def test_cross_scene_drift_detection(self, tmp_path: Path):
        from openharness.services.style_consistency import StyleConsistencyChecker

        scenes = []
        for index, color in enumerate(["#111111", "#111111", "#333333"]):
            scene = tmp_path / f"d{index}.js"
            scene.write_text(f"color: '{color}';", encoding="utf-8")
            scenes.append(scene)
        report = StyleConsistencyChecker(
            {"palette": [], "fonts": [], "easings": []}
        ).check_files(scenes)
        drift = [i for i in report.issues if i.kind == "style_drift"]
        assert drift and "d2" in drift[0].message  # d2 is the outlier scene


class TestStyleConsistencyTool:
    @pytest.mark.asyncio
    async def test_tool_requires_reference(self, tmp_path: Path, ctx):
        from openharness.tools.style_consistency_tool import (
            StyleConsistencyInput,
            StyleConsistencyTool,
        )

        scene = tmp_path / "a.js"
        scene.write_text("color: '#111111';", encoding="utf-8")
        result = await StyleConsistencyTool().execute(
            StyleConsistencyInput(paths=[str(scene)]), ctx
        )
        assert result.is_error
        assert "no reference style" in result.output

    @pytest.mark.asyncio
    async def test_tool_flags_with_inline_reference(self, tmp_path: Path, ctx):
        from openharness.tools.style_consistency_tool import (
            StyleConsistencyInput,
            StyleConsistencyTool,
        )

        bad = tmp_path / "bad.js"
        bad.write_text("color: '#2A6FF2'; fontFamily: 'Comic Sans';",
                       encoding="utf-8")
        result = await StyleConsistencyTool().execute(
            StyleConsistencyInput(paths=[str(bad)], palette=["#F2F0EB"],
                                  fonts=["Inter"]), ctx
        )
        assert result.is_error
        assert "palette_deviation" in result.output
        assert "font_deviation" in result.output


# ======================================================================
# 11. registry wiring
# ======================================================================

class TestRegistryWiring:
    def test_all_new_tools_registered(self):
        from openharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        names = {tool.name for tool in registry.list_tools()}
        expected = {
            "pixel_qa", "minicpm_vlm", "frame_diff", "scene_lint", "vlm_qa",
            "style_consistency", "semantic_memory", "style_memory",
            "failure_patterns", "prompt_tracker", "contact_sheet",
        }
        assert expected <= names, expected - names
        # 59 tools before this wave; every wave only adds.
        assert len(names) >= 67

    def test_read_only_classification(self):
        from openharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        write_actions = {
            "semantic_memory": {"action": "remember", "text": "x"},
            "style_memory": {"action": "record", "project": "p", "style_name": "s"},
            "failure_patterns": {"action": "record_failure", "scene_type": "t",
                                 "reason": "r"},
            "prompt_tracker": {"action": "register", "prompt_id": "p",
                               "template": "t"},
        }
        minimal_inputs = {
            "pixel_qa": {"paths": ["x"]},
            "frame_diff": {"paths": ["x"]},
            "scene_lint": {"paths": ["x"]},
            "vlm_qa": {"videos": ["x"]},
            "style_consistency": {"paths": ["x"]},
            "minicpm_vlm": {},
        }
        for tool in registry.list_tools():
            if tool.name in minimal_inputs:
                instance = tool.input_model(**minimal_inputs[tool.name])
                assert tool.is_read_only(instance), tool.name
            elif tool.name in write_actions:
                instance = tool.input_model(**write_actions[tool.name])
                assert not tool.is_read_only(instance), tool.name
