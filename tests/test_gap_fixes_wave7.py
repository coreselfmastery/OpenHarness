"""Wave 7 — regression tests for the G1-G12 gap fixes.

Covers:
- G1: render_scenes wait-mode sees ALL dependency waves (wait_for_scenes),
      foreign background batches never block the wait, and a dead
      dependency chain exits drained instead of hanging.
- G2: install_watchee aborts live render subprocesses on trip(); the
      watchdog default is >= 1.5x the tool timeout; finished jobs feed
      the control bus.
- G3: stable (SHA-256) TTS filenames; in-flight TTS dedupe; existing
      output files are reused without a paid API call.
- G4: render_engines.json edits re-bind without a process restart;
      the GPU-count cache re-probes after its TTL.
- G5: qa_error / general failure categories are surfaced by predict().
- G6: semantic-memory recall filters by project INSIDE the SQL.
- G7: the engine-binary stamp changes the fingerprint on upgrade.
- G8: read-only bash commands no longer invalidate the tool cache;
      anything write-looking still does.
- G9: cost_governor status/check fold in the real session tracker.
- G11: LLM turn duration lands in the audit record (not 0.0).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest


# ======================================================================
# G1 — wait_for_scenes covers every dependency wave
# ======================================================================

def _engine_project(tmp_path: Path, templates: dict) -> Path:
    import json

    (tmp_path / "render_engines.json").write_text(json.dumps(templates), encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_g1_wait_for_scenes_covers_all_dependency_waves(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    _engine_project(
        tmp_path,
        {
            "copier": ["cp", "{scene}", "{output}"],  # exit 0 + non-empty output
        },
    )
    pipeline = RenderPipeline(workers=2, cwd=tmp_path, cache_enabled=False)
    try:
        pipeline.start()
        wave1 = [tmp_path / "s1.tsx", tmp_path / "s2.tsx"]
        wave2 = [tmp_path / "s3.tsx"]  # depends on both wave-1 scenes
        for scene in [*wave1, *wave2]:
            scene.write_text("// scene", encoding="utf-8")
        # One dependency per comment line (the parser regex is single-dep).
        (tmp_path / "s3.tsx").write_text(
            "// depends_on: s1\n// depends_on: s2\n// scene", encoding="utf-8"
        )

        since = time.monotonic()
        first = pipeline.submit_batch(
            [*wave1, *wave2], engine="copier", tier="draft",
            output_dir=tmp_path / "renders",
        )
        assert len(first) == 2  # wave 1 only — the KNOWN limitation ...
        by_scene = await pipeline.wait_for_scenes(
            [*wave1, *wave2], tier="draft", since=since, timeout=20.0
        )
        # ... which wait_for_scenes fixes: every scene, wave 2 included.
        assert set(by_scene) == {"s1", "s2", "s3"}
        assert all(
            job.status in {"done", "failed", "timeout", "skipped"}
            for job in by_scene.values()
        )
        assert (tmp_path / "renders" / "s3.draft.mp4").is_file()
    finally:
        await pipeline.shutdown(drain=False)


@pytest.mark.asyncio
async def test_g1_wait_for_scenes_ignores_foreign_background_jobs(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    _engine_project(
        tmp_path,
        {
            "sleeper": ["sleep", "30"],
            "copier": ["cp", "{scene}", "{output}"],
        },
    )
    pipeline = RenderPipeline(workers=2, cwd=tmp_path, cache_enabled=False)
    try:
        pipeline.start()
        # Foreign long-running background job (e.g. an earlier batch).
        foreign = tmp_path / "foreign.tsx"
        foreign.write_text("// x", encoding="utf-8")
        pipeline.submit(foreign, engine="sleeper", tier="final",
                        output_dir=tmp_path / "renders")

        own = tmp_path / "own.tsx"
        own.write_text("// x", encoding="utf-8")
        since = time.monotonic()
        pipeline.submit(own, engine="copier", tier="draft",
                        output_dir=tmp_path / "renders")

        started = time.monotonic()
        by_scene = await pipeline.wait_for_scenes(
            [own], tier="draft", since=since, timeout=15.0
        )
        elapsed = time.monotonic() - started
        # Returned promptly while the 30s foreign job is still running —
        # wait_all() would have blocked on it.
        assert elapsed < 10.0
        assert set(by_scene) == {"own"}
        assert by_scene["own"].status == "done"
    finally:
        await pipeline.abort_all("test cleanup")
        await pipeline.shutdown(drain=False)


@pytest.mark.asyncio
async def test_g1_wait_for_scenes_exits_drained_when_no_job_ever_comes(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    try:
        pipeline.start()
        ghost = tmp_path / "never_submitted.tsx"
        started = time.monotonic()
        by_scene = await pipeline.wait_for_scenes(
            [ghost], tier="draft", since=time.monotonic(), timeout=30.0
        )
        # Drained exit, not a hang until the timeout.
        assert time.monotonic() - started < 5.0
        assert by_scene == {}
    finally:
        await pipeline.shutdown(drain=False)


# ======================================================================
# G2 — kill switch aborts render subprocesses; watchdog sanity
# ======================================================================

@pytest.mark.asyncio
async def test_g2_abort_all_kills_live_render_and_marks_jobs(tmp_path: Path):
    from openharness.services.render_pipeline import RenderPipeline

    _engine_project(tmp_path, {"sleeper": ["sleep", "60"]})
    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    try:
        pipeline.start()
        scene = tmp_path / "slow.tsx"
        scene.write_text("// x", encoding="utf-8")
        job = pipeline.submit(scene, engine="sleeper", tier="draft",
                               output_dir=tmp_path / "renders", timeout_seconds=120)
        for _ in range(100):
            if job.status == "running":
                break
            await asyncio.sleep(0.05)
        assert job.status == "running"
        assert pipeline._live_processes.get(job.job_id) is not None

        aborted = await pipeline.abort_all("unit test kill")
        assert aborted == 1
        assert job.status == "failed"
        assert "aborted: unit test kill" in job.error
        assert pipeline._live_processes.get(job.job_id) is None
    finally:
        await pipeline.shutdown(drain=False)


@pytest.mark.asyncio
async def test_g2_trip_invokes_watchees(tmp_path: Path):
    from openharness.services.control_bus import ControlBus, set_control_bus

    bus = ControlBus(watchdog_timeout_seconds=0)
    set_control_bus(bus)
    try:
        calls: list[str] = []

        async def abort(reason: str) -> None:
            calls.append(reason)

        def abort_sync(reason: str) -> None:
            calls.append("sync:" + reason)
            raise RuntimeError("boom")  # must not break the kill path

        bus.install_watchee("async_one", abort)
        bus.install_watchee("sync_broken", abort_sync)
        bus.install_watchee("async_one", abort)  # duplicate — ignored
        bus.trip("kill it")
        assert bus.killed
        # The fire-and-forget abort task needs a tick to run.
        for _ in range(50):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.05)
        assert calls == ["kill it", "sync:kill it"]
        assert bus.watchees() == ["async_one", "sync_broken"]
    finally:
        set_control_bus(None)


@pytest.mark.asyncio
async def test_g2_render_finish_feeds_control_bus(tmp_path: Path):
    from openharness.services.control_bus import ControlBus, set_control_bus
    from openharness.services.render_pipeline import RenderPipeline

    bus = ControlBus(watchdog_timeout_seconds=0)
    set_control_bus(bus)
    _engine_project(tmp_path, {"copier": ["cp", "{scene}", "{output}"]})
    pipeline = RenderPipeline(workers=1, cwd=tmp_path, cache_enabled=False)
    try:
        pipeline.start()
        scene = tmp_path / "quick.tsx"
        scene.write_text("// x", encoding="utf-8")
        job = pipeline.submit(scene, engine="copier", tier="draft",
                              output_dir=tmp_path / "renders", timeout_seconds=30)
        deadline = time.monotonic() + 15.0
        while job.status in {"pending", "running"} and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert job.status == "done"
        # The job finish fed the bus → progress was observed recently.
        assert bus.seconds_since_progress < 5.0
    finally:
        await pipeline.shutdown(drain=False)
        set_control_bus(None)


def test_g2_watchdog_default_at_least_1p5x_tool_timeout():
    from openharness.config.settings import ControlSettings

    control = ControlSettings()
    assert control.tool_timeout_seconds == 900
    assert control.watchdog_timeout_seconds == 1350

    raised = ControlSettings(tool_timeout_seconds=1000)
    assert raised.watchdog_timeout_seconds == 1500  # raised to 1.5 x 1000

    explicit = ControlSettings(watchdog_timeout_seconds=5000, tool_timeout_seconds=900)
    assert explicit.watchdog_timeout_seconds == 5000  # never lowered


def test_g2_render_timeout_default_fits_hero_shots():
    from openharness.config.settings import RenderSettings

    assert RenderSettings().render_timeout_seconds >= 7200


# ======================================================================
# G3 — stable TTS names + dedupe + reuse
# ======================================================================

def test_g3_voiceover_paths_are_stable_and_distinct(tmp_path: Path):
    from openharness.services.asset_prefetch import _voiceover_path

    a1 = _voiceover_path(tmp_path, "scene_001", "Hello world", "alloy")
    a2 = _voiceover_path(tmp_path, "scene_001", "Hello world", "alloy")
    assert a1 == a2  # same line + voice -> same file, across restarts
    assert a1.name.startswith("scene_001_") and a1.suffix == ".mp3"
    # A different voice over the same line is a different file.
    other_voice = _voiceover_path(tmp_path, "scene_001", "Hello world", "echo")
    assert other_voice != a1
    # A different line is a different file.
    other_line = _voiceover_path(tmp_path, "scene_001", "Bye world", "alloy")
    assert other_line != a1


class _FakeResponse:
    status_code = 200
    content = b"mp3-bytes"

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    posts = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        _FakeAsyncClient.posts += 1
        return _FakeResponse()


@pytest.mark.asyncio
async def test_g3_inflight_dedupe_and_file_reuse(tmp_path: Path, monkeypatch):
    import openharness.services.asset_prefetch as ap

    monkeypatch.setattr(ap.httpx, "AsyncClient", _FakeAsyncClient)
    gen = ap.AsyncAudioGenerator(base_url="http://tts", api_key="k")

    # In-flight dedupe: same line/voice/speed -> ONE task, ONE call.
    _FakeAsyncClient.posts = 0
    t1 = gen.run_in_background("Same line", tmp_path / "a.mp3")
    t2 = gen.run_in_background("Same line", tmp_path / "a.mp3")
    assert t1.task_id == t2.task_id
    for _ in range(100):
        if t1.status in {"done", "failed"}:
            break
        await asyncio.sleep(0.05)
    assert t1.status == "done"
    assert (tmp_path / "a.mp3").read_bytes() == b"mp3-bytes"
    assert _FakeAsyncClient.posts == 1

    # Existing-file reuse: no new HTTP call at all.
    _FakeAsyncClient.posts = 0
    t3 = gen.run_in_background("Same line", tmp_path / "a.mp3")
    assert t3.status == "done"  # immediately, no await needed
    assert t3.task_id != t1.task_id  # a fresh record ...
    await asyncio.sleep(0.1)
    assert _FakeAsyncClient.posts == 0  # ... but zero API spend


# ======================================================================
# G4 — live re-bind of engine templates
# ======================================================================

def test_g4_engine_templates_rebind_without_restart(tmp_path: Path):
    import openharness.services.render_pipeline as rp

    rp.reset_render_pipeline()
    try:
        _engine_project(tmp_path, {"custom": ["echo", "one"]})
        p1 = rp.get_render_pipeline(cwd=tmp_path)
        assert p1._templates["custom"] == ["echo", "one"]

        import json

        (tmp_path / "render_engines.json").write_text(
            json.dumps({"custom": ["echo", "two"]}), encoding="utf-8"
        )
        # Ensure a different mtime (some filesystems have 1s granularity).
        import os

        stat = os.stat(tmp_path / "render_engines.json")
        os.utime(tmp_path / "render_engines.json", (stat.st_atime, stat.st_mtime + 2))

        p2 = rp.get_render_pipeline(cwd=tmp_path)
        assert p2 is p1  # same singleton — no restart
        assert p2._templates["custom"] == ["echo", "two"]  # re-bound
    finally:
        rp.reset_render_pipeline()


# ======================================================================
# G5 — qa_error / general surface in predictions
# ======================================================================

def test_g5_catchall_categories_surface_in_predict(tmp_path: Path):
    from openharness.services.persistent_store import PersistentStore
    from openharness.services.qa_pipeline import FailurePredictor

    predictor = FailurePredictor(PersistentStore(tmp_path / "qa.sqlite3"))
    for _ in range(2):
        predictor.record_failure(category="qa_error", complexity="simple", engine="hyperframes")
    predictor.record_failure(category="general", complexity="simple", engine="hyperframes")
    predictor.record_success(complexity="simple", engine="hyperframes", category="qa_error")

    predicted = {p["category"]: p for p in predictor.predict(
        complexity="simple", engine="hyperframes", top_k=5)}
    assert "qa_error" in predicted
    assert predicted["qa_error"]["failure_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert "general" in predicted


# ======================================================================
# G6 — project filter inside the SQL
# ======================================================================

def test_g6_recall_project_filter_is_in_sql(tmp_path: Path, monkeypatch):
    from openharness.services.semantic_memory import SemanticMemory

    # Force the pure-hashing backend (no network).
    monkeypatch.delenv("OPENHARNESS_EMBED_BASE_URL", raising=False)
    memory = SemanticMemory(tmp_path / "sem.sqlite3")
    memory.remember("alpha project fact", project="alpha", kind="fact")
    memory.remember("beta project fact", project="beta", kind="fact")

    recalled_alpha = memory.recall("project fact", project="alpha", top_k=10)
    assert recalled_alpha, "alpha recall should find its own rows"
    assert all(r.project == "alpha" for r in recalled_alpha)
    assert not any("beta" in r.text for r in recalled_alpha)


# ======================================================================
# G7 — engine binary stamp invalidates on upgrade
# ======================================================================

def test_g7_engine_binary_stamp_changes_fingerprint(tmp_path: Path, monkeypatch):
    import openharness.services.render_cache as rc

    scene = tmp_path / "scene.tsx"
    scene.write_text("// scene", encoding="utf-8")
    binary = tmp_path / "fakeengine"
    binary.write_bytes(b"v1")
    monkeypatch.setattr(rc, "_ENGINE_BINARY_STAMPS", {})
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "fake" else None)

    fp1 = rc.scene_fingerprint(
        scene, engine="fake", engine_args=["x"], tier="draft", codec="libx264",
        cwd=tmp_path,
    )
    assert fp1 is not None

    # Upgrade the binary (different size/mtime) → different fingerprint.
    binary.write_bytes(b"version-2-much-longer")
    import os

    stat = os.stat(binary)
    os.utime(binary, (stat.st_atime, stat.st_mtime + 5))
    rc._ENGINE_BINARY_STAMPS.clear()

    fp2 = rc.scene_fingerprint(
        scene, engine="fake", engine_args=["x"], tier="draft", codec="libx264",
        cwd=tmp_path,
    )
    assert fp2 is not None and fp2 != fp1


# ======================================================================
# G8 — read-only bash no longer wipes the cache
# ======================================================================

@pytest.mark.parametrize(
    "command,expected_mutating",
    [
        ("ls -la", False),
        ("cat README.md", False),
        ("git status", False),
        ("git log --oneline -5", False),
        ("git diff HEAD~1", False),
        ("git branch", False),
        ("git tag", False),
        ("cd src && ls", False),
        ("ls | wc -l", False),
        ("FOO=1 rg 'pattern' src/", False),
        ("echo hello world", False),
        ("rm -rf /", True),
        ("mv a b", True),
        ("cp src dst", True),
        ("echo hi > file.txt", True),
        ("ls > out.txt", True),
        ("git push origin main", True),
        ("git add -A", True),
        ("git stash", True),
        ("git branch new-branch", True),
        ("git tag v1.0", True),
        ("python -c 'open(\"x\",\"w\")'", True),
        ("pip install requests", True),
        ("python -m pip install x", True),
        ("make all", True),
        ("curl -o x https://x", True),
        ("some-unknown-binary --flag", True),
        ("ls; rm x", True),
        ("cat a && echo hi >> b", True),
        ("find . -name x -delete", True),
        ("", False),
    ],
)
def test_g8_bash_mutation_heuristic(command: str, expected_mutating: bool):
    from openharness.services.tool_cache import _bash_command_mutates

    assert _bash_command_mutates(command) is expected_mutating, command


def test_g8_is_mutating_passes_bash_input():
    from openharness.services.tool_cache import ToolResultCache

    assert ToolResultCache.is_mutating("bash", {"command": "ls"}) is False
    assert ToolResultCache.is_mutating("bash", {"command": "rm x"}) is True
    assert ToolResultCache.is_mutating("bash") is True  # no input — safe default
    assert ToolResultCache.is_mutating("write_file", {}) is True
    assert ToolResultCache.is_mutating("read_file", {}) is False


# ======================================================================
# G9 — cost_governor folds in the real session tracker
# ======================================================================

@pytest.mark.asyncio
async def test_g9_cost_governor_reads_session_tracker(tmp_path: Path):
    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.missing_tools import (
        CostGovernorTool,
        CostGovernorToolInput,
    )

    class _Tracker:
        spent_usd = 1.234
        calls = 7
        budget_usd = 2.0

    tool = CostGovernorTool()
    context = ToolExecutionContext(cwd=tmp_path, metadata={"session_cost_tracker": _Tracker()})
    result = await tool.execute(CostGovernorToolInput(action="status"), context)
    assert "session-tracked spend $1.23" in result.output
    assert "7 LLM call(s)" in result.output
    assert result.metadata["session_spent_usd"] == 1.234
    assert result.metadata["effective_spent_usd"] == 1.234

    # check reports the tracker budget too
    check = await tool.execute(CostGovernorToolInput(action="check"), context)
    assert "1.23" in check.output
    assert not check.is_error  # 1.234 < 2.0


# ======================================================================
# G11 — LLM turn duration lands in the audit record
# ======================================================================

def test_g11_llm_usage_records_duration(tmp_path: Path, monkeypatch):
    import openharness.services.audit_log as audit_mod
    from openharness.api.usage import UsageSnapshot
    from openharness.engine.query import _record_llm_usage

    recorded: dict = {}

    class _Audit:
        def record_llm_call(self, **kwargs) -> None:
            recorded.update(kwargs)

    monkeypatch.setattr(audit_mod, "get_audit_log", lambda: _Audit())

    class _Ctx:  # minimal QueryContext stand-in
        model = "test-model"

    usage = UsageSnapshot(input_tokens=10, output_tokens=5)
    _record_llm_usage(_Ctx(), usage, duration_seconds=1.75)
    assert recorded["duration_seconds"] == 1.75
    assert recorded["input_tokens"] == 10
    assert recorded["output_tokens"] == 5
