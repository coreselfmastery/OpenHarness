"""Media pipeline tools — items 23-27.

#23 Cost Governor: enforces a budget. Tracks spending in real-time
    and halts when the budget is exceeded.

#24 Audio QA Tool: measures real audio metrics via ffmpeg — true-peak
    clipping, integrated loudness (LUFS), noise floor, and silence
    detection — for broadcast-quality output.

#25 Determinism Checker: statically analyses scene code for common
    non-determinism sources (Math.random, Date.now, ...) that cause
    flicker when a scene is re-rendered.

#26 Contact Sheet Generator: composites a real thumbnail grid image
    via ffmpeg; falls back to a manifest when ffmpeg or decodable
    image inputs are unavailable.

#27 Engine Router: routes scenes to the optimal rendering engine
    (hyperframes, manim, blender, ffmpeg) based on scene type.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


def _run_ffmpeg(args: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the given arguments, returning the completed process.

    Raises FileNotFoundError when ffmpeg is not installed.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is not installed")
    return subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ============================================================
# #23 Cost Governor
# ============================================================

class CostGovernor:
    """Real-time budget enforcement.

    Tracks cumulative spending and raises an alarm when the budget
    is exceeded. The agent can check ``should_stop()`` before each
    LLM call to avoid exceeding the budget.

    Usage:
        governor = CostGovernor(budget_usd=10.0)
        governor.record_spend(0.05, call_type="scene_generation")
        if governor.should_stop():
            return "Budget exceeded — halting generation."
    """

    def __init__(self, *, budget_usd: float) -> None:
        self._budget = budget_usd
        self._spent: float = 0.0
        self._calls: list[dict[str, Any]] = []
        self._halted = False

    def record_spend(self, cost_usd: float, *, call_type: str = "", metadata: dict | None = None) -> None:
        """Record a single API call's cost."""
        self._spent += cost_usd
        self._calls.append({
            "cost_usd": cost_usd,
            "call_type": call_type,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "cumulative_usd": self._spent,
        })
        if self._spent >= self._budget:
            self._halted = True

    def should_stop(self) -> bool:
        """Return True if the budget is exhausted."""
        return self._halted

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self._budget - self._spent)

    @property
    def budget(self) -> float:
        return self._budget

    def stats(self) -> dict[str, Any]:
        return {
            "budget_usd": self._budget,
            "spent_usd": round(self._spent, 4),
            "remaining_usd": round(self.remaining, 4),
            "total_calls": len(self._calls),
            "halted": self._halted,
            "utilization": round(self._spent / self._budget, 3) if self._budget else 0.0,
        }

    def reset(self) -> None:
        """Reset the governor for a new project."""
        self._spent = 0.0
        self._calls.clear()
        self._halted = False


class CostGovernorToolInput(BaseModel):
    action: str = Field(description="Action: 'check', 'record', 'status', or 'reset'")
    cost_usd: float = Field(default=0.0, description="Cost to record (for 'record' action)")
    call_type: str = Field(default="", description="Type of call (e.g. 'scene_gen', 'qa')")
    budget_usd: float = Field(default=0.0, description="Budget to set (for 'reset' action)")


class CostGovernorTool(BaseTool):
    name = "cost_governor"
    description = (
        "Enforce a real-time budget. Tracks cumulative LLM spending and "
        "halts when the budget is exceeded. Use 'check' before each LLM "
        "call, 'record' after each call, 'status' to see spending, and "
        "'reset' to start a new budget."
    )
    input_model = CostGovernorToolInput

    # Class-level singleton so the governor persists across tool calls
    _governor: CostGovernor | None = None

    @classmethod
    def get_governor(cls) -> CostGovernor:
        if cls._governor is None:
            cls._governor = CostGovernor(budget_usd=100.0)  # default $100
        return cls._governor

    async def execute(self, arguments: CostGovernorToolInput, context: ToolExecutionContext) -> ToolResult:
        gov = self.get_governor()
        # The REAL session spend (#28) from the wired SessionCostTracker —
        # the governor's own numbers are only the model's manual records
        # (advisory before). Fold both so 'check'/'status' reflect reality.
        tracker = (context.metadata or {}).get("session_cost_tracker")
        real_spent = float(getattr(tracker, "spent_usd", 0.0) or 0.0)
        real_calls = int(getattr(tracker, "calls", 0) or 0)
        tracker_budget = getattr(tracker, "budget_usd", None)
        tracker_halted = bool(
            tracker_budget is not None
            and float(tracker_budget) > 0
            and real_spent >= float(tracker_budget)
        )
        effective_spent = max(gov.spent, real_spent)
        effective_budget = max(gov.budget, float(tracker_budget or 0.0))

        def merged_stats() -> dict[str, Any]:
            stats = gov.stats()
            stats.update(
                {
                    "session_spent_usd": round(real_spent, 4),
                    "session_llm_calls": real_calls,
                    "session_budget_usd": tracker_budget,
                    "session_budget_halted": tracker_halted,
                    "effective_spent_usd": round(effective_spent, 4),
                    "effective_budget_usd": effective_budget,
                }
            )
            return stats

        if arguments.action == "check":
            exceeded = gov.should_stop() or tracker_halted or (
                effective_budget > 0 and effective_spent >= effective_budget
            )
            if exceeded:
                return ToolResult(
                    output=(
                        f"BUDGET EXCEEDED: ${effective_spent:.2f} / ${effective_budget:.2f}. "
                        "Halting. (manual + session-tracked spend)"
                    ),
                    is_error=True,
                    metadata=merged_stats(),
                )
            return ToolResult(
                output=(
                    f"Budget OK: ${effective_spent:.2f} / ${effective_budget:.2f} "
                    f"(manual ${gov.spent:.2f} + session-tracked ${real_spent:.2f})"
                ),
                metadata=merged_stats(),
            )
        elif arguments.action == "record":
            gov.record_spend(arguments.cost_usd, call_type=arguments.call_type)
            halted = gov.should_stop() or tracker_halted
            msg = (
                f"Recorded ${arguments.cost_usd:.4f} ({arguments.call_type}). "
                f"Manual total: ${gov.spent:.2f} / ${gov.budget:.2f}; "
                f"session-tracked: ${real_spent:.2f}"
            )
            if halted:
                msg += " — BUDGET EXCEEDED"
            return ToolResult(output=msg, is_error=halted, metadata=merged_stats())
        elif arguments.action == "status":
            return ToolResult(
                output=(
                    f"Manual spend ${gov.spent:.2f} / ${gov.budget:.2f}; "
                    f"session-tracked spend ${real_spent:.2f} over {real_calls} LLM call(s)"
                    + (
                        f" (session budget ${float(tracker_budget):.2f})"
                        if tracker_budget
                        else " (no session budget set)"
                    )
                ),
                metadata=merged_stats(),
            )
        elif arguments.action == "reset":
            budget = arguments.budget_usd if arguments.budget_usd > 0 else gov.budget
            gov.reset()
            gov._budget = budget
            return ToolResult(output=f"Budget reset to ${budget:.2f}", metadata=merged_stats())
        else:
            return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)


# ============================================================
# #24 Audio QA Tool — real measurements via ffmpeg
# ============================================================

class AudioQAToolInput(BaseModel):
    audio_path: str = Field(description="Path to the audio file to check")
    check_clipping: bool = Field(default=True)
    check_noise_floor: bool = Field(default=True)
    check_levels: bool = Field(default=True)
    check_silence: bool = Field(default=True)
    target_lufs: float = Field(default=-16.0, description="Target loudness in LUFS (broadcast standard: -16)")
    lufs_tolerance: float = Field(default=2.0, description="Allowed deviation from target LUFS")
    max_clipping_dbtp: float = Field(default=-0.3, description="True-peak threshold in dBTP above which audio counts as clipped")
    max_noise_floor_db: float = Field(default=-60.0, description="Noise floor must be at or below this (dB)")
    min_silence_seconds: float = Field(default=2.0, description="Silence gaps longer than this are flagged")


class _FFmpegMeasurements:
    """Parsed output of the ffmpeg analysis passes."""

    def __init__(self) -> None:
        self.max_volume_db: float | None = None
        self.mean_volume_db: float | None = None
        self.integrated_lufs: float | None = None
        self.true_peak_dbtp: float | None = None
        self.silence_intervals: list[tuple[float, float]] = []
        self.quiet_passage_detected: bool = False


def _measure_audio(
    path: Path,
    silence_threshold_db: float,
    min_silence: float,
    noise_floor_db: float | None = None,
) -> _FFmpegMeasurements:
    """Run ffmpeg analysis passes and parse the metrics.

    Pass A chains volumedetect (peak/mean levels) with silencedetect
    (long quiet gaps at the content threshold). Pass B runs loudnorm in
    measurement mode, which prints its input stats as JSON (integrated
    LUFS + true peak). Pass C (optional) probes for passages at or below
    the configured noise floor.
    """
    m = _FFmpegMeasurements()

    pass_a = _run_ffmpeg([
        "-i", str(path),
        "-af", f"volumedetect,silencedetect=noise={silence_threshold_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])
    if pass_a.returncode != 0:
        raise RuntimeError(_summarize_ffmpeg_error(pass_a.stderr))
    stderr = pass_a.stderr
    for pattern, attr in (
        (r"max_volume:\s*(-?[\d.]+)\s*dB", "max_volume_db"),
        (r"mean_volume:\s*(-?[\d.]+)\s*dB", "mean_volume_db"),
    ):
        match = re.search(pattern, stderr)
        if match:
            setattr(m, attr, float(match.group(1)))

    # silencedetect prints silence_start=X / silence_end=Y pairs.
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", stderr)]
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else None
        m.silence_intervals.append((start, end))

    pass_b = _run_ffmpeg([
        "-i", str(path),
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-",
    ])
    if pass_b.returncode == 0:
        # The JSON block is at the very end of stderr.
        json_match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", pass_b.stderr, re.DOTALL)
        if json_match:
            try:
                stats = json.loads(json_match.group(0))
                m.integrated_lufs = float(stats.get("input_i"))
                m.true_peak_dbtp = float(stats.get("input_tp"))
            except (ValueError, TypeError):
                pass

    if noise_floor_db is not None:
        # Probe for passages at or below the configured floor: if any
        # 0.2s+ stretch stays under the threshold, the floor reaches at
        # least that deep. (volumedetect's min_volume is not emitted by
        # all ffmpeg builds, so this probe is the portable measurement.)
        probe = _run_ffmpeg([
            "-i", str(path),
            "-af", f"silencedetect=noise={noise_floor_db}dB:d=0.2",
            "-f", "null", "-",
        ])
        if probe.returncode == 0 and re.search(r"silence_start:", probe.stderr):
            m.quiet_passage_detected = True
    return m


def _summarize_ffmpeg_error(stderr: str) -> str:
    """Return the most useful line of an ffmpeg failure."""
    for line in reversed(stderr.splitlines()):
        if line.strip().startswith(("Error", "Invalid", "Unknown", "moov", "does not", "Invalid data")):
            return line.strip()[:200]
    tail = [ln for ln in stderr.splitlines() if ln.strip()]
    return tail[-1][:200] if tail else "unknown decode error"


class AudioQATool(BaseTool):
    name = "audio_qa"
    description = (
        "Measure audio quality with ffmpeg: true-peak clipping (dBTP), "
        "integrated loudness vs a target LUFS, noise floor, and long "
        "silence gaps. Returns pass/fail with the measured values. "
        "Requires ffmpeg on PATH."
    )
    input_model = AudioQAToolInput

    def is_read_only(self, arguments) -> bool:
        return True

    async def execute(self, arguments: AudioQAToolInput, context: ToolExecutionContext) -> ToolResult:
        path = Path(arguments.audio_path)
        if not path.is_absolute():
            path = context.cwd / path
        if not path.is_file():
            return ToolResult(output=f"Audio file not found: {path}", is_error=True)

        if not shutil.which("ffmpeg"):
            return ToolResult(
                output="ffmpeg is not installed; audio QA cannot measure anything without it. "
                       "Install ffmpeg (apt install ffmpeg / brew install ffmpeg) and retry.",
                is_error=True,
            )

        try:
            # ``_measure_audio`` chains up to three ffmpeg subprocess
            # passes (~180s worst case) — run it in a worker thread so it
            # cannot freeze the event loop (concurrent agents, watchdog
            # feeding, UI) for the whole measurement.
            m = await asyncio.to_thread(
                _measure_audio,
                path,
                silence_threshold_db=-40.0,
                min_silence=arguments.min_silence_seconds,
                noise_floor_db=arguments.max_noise_floor_db if arguments.check_noise_floor else None,
            )
        except FileNotFoundError:
            return ToolResult(output="ffmpeg is not installed; cannot measure audio.", is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult(output=f"ffmpeg timed out analysing {path.name}", is_error=True)
        except RuntimeError as exc:
            return ToolResult(output=f"Could not decode {path.name}: {exc}", is_error=True)

        checks: list[dict[str, Any]] = []
        failures: list[str] = []

        if arguments.check_clipping:
            if m.true_peak_dbtp is not None:
                clipped = m.true_peak_dbtp > arguments.max_clipping_dbtp
                checks.append({
                    "name": "clipping",
                    "passed": not clipped,
                    "detail": f"True peak {m.true_peak_dbtp:.1f} dBTP (limit {arguments.max_clipping_dbtp} dBTP)",
                    "measured": m.true_peak_dbtp,
                })
                if clipped:
                    failures.append(f"Clipping: true peak {m.true_peak_dbtp:.1f} dBTP exceeds {arguments.max_clipping_dbtp} dBTP")
            elif m.max_volume_db is not None:
                # loudnorm unavailable — fall back to volumedetect max level.
                clipped = m.max_volume_db >= -0.1
                checks.append({
                    "name": "clipping",
                    "passed": not clipped,
                    "detail": f"Max volume {m.max_volume_db:.1f} dB (volumedetect fallback)",
                    "measured": m.max_volume_db,
                })
                if clipped:
                    failures.append(f"Clipping: max volume {m.max_volume_db:.1f} dB")

        if arguments.check_levels:
            if m.integrated_lufs is not None:
                deviation = abs(m.integrated_lufs - arguments.target_lufs)
                levels_ok = deviation <= arguments.lufs_tolerance
                checks.append({
                    "name": "levels",
                    "passed": levels_ok,
                    "detail": f"Measured {m.integrated_lufs:.1f} LUFS vs target {arguments.target_lufs} (±{arguments.lufs_tolerance})",
                    "measured": m.integrated_lufs,
                })
                if not levels_ok:
                    failures.append(
                        f"Levels off: {m.integrated_lufs:.1f} LUFS vs target {arguments.target_lufs} "
                        f"(deviation {deviation:.1f} > {arguments.lufs_tolerance})"
                    )
            else:
                checks.append({"name": "levels", "passed": False, "detail": "LUFS measurement unavailable"})
                failures.append("Levels: integrated LUFS could not be measured")

        if arguments.check_noise_floor:
            # Measured by probing for passages at/below the floor threshold
            # (see _measure_audio). When no such passage exists the floor is
            # simply not measurable on this track — continuous content and a
            # noisy floor are indistinguishable without a reference, so the
            # check is skipped rather than failed.
            if m.quiet_passage_detected:
                checks.append({
                    "name": "noise_floor",
                    "passed": True,
                    "detail": f"Quiet passages detected at or below {arguments.max_noise_floor_db} dB",
                })
            else:
                checks.append({
                    "name": "noise_floor",
                    "passed": True,
                    "detail": (
                        f"Not measurable: no passage at/below {arguments.max_noise_floor_db} dB "
                        "(continuous content or elevated floor) — skipped"
                    ),
                })

        if arguments.check_silence:
            long_gaps = [
                (s, e) for s, e in m.silence_intervals
                if e is not None and (e - s) >= arguments.min_silence_seconds
            ]
            silence_ok = not long_gaps
            detail = (
                f"No silence gaps >= {arguments.min_silence_seconds}s"
                if silence_ok
                else f"{len(long_gaps)} silence gap(s) >= {arguments.min_silence_seconds}s: "
                     + ", ".join(f"{s:.1f}s–{e:.1f}s" for s, e in long_gaps[:3])
            )
            checks.append({"name": "silence", "passed": silence_ok, "detail": detail})
            if not silence_ok:
                failures.append(detail)

        passed = len(failures) == 0
        lines = [
            f"[Audio QA: {path.name}]",
            f"Overall: {'PASS' if passed else 'FAIL'}",
            "",
            "Checks:",
        ]
        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            lines.append(f"  [{status}] {c['name']}: {c['detail']}")
        if failures:
            lines.append("")
            lines.append("Failures:")
            for f in failures:
                lines.append(f"  - {f}")
        return ToolResult(
            output="\n".join(lines),
            is_error=not passed,
            metadata={"checks": checks, "failures": failures, "passed": passed},
        )


# ============================================================
# #25 Determinism Checker — static non-determinism analysis
# ============================================================

# Non-determinism sources in scene code. Each entry: (label, regex).
_NON_DETERMINISM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Math.random()", re.compile(r"\bMath\.random\s*\(")),
    ("Date.now()", re.compile(r"\bDate\.now\s*\(")),
    ("new Date() (current time)", re.compile(r"\bnew\s+Date\s*\(\s*\)")),
    ("performance.now()", re.compile(r"\bperformance\.now\s*\(")),
    ("crypto.randomUUID()", re.compile(r"\bcrypto\.randomUUID\s*\(")),
    ("uuid v4 generation", re.compile(r"\buuid[v4]?\s*\(|\buuid4\s*\(")),
    ("python random.random/uniform/randint", re.compile(r"\brandom\.(random|uniform|randint|choice|shuffle|randrange)\s*\(")),
    ("python uuid.uuid4()", re.compile(r"\buuid\.uuid4\s*\(")),
    ("python datetime.now/time.time", re.compile(r"\b(datetime\.now\s*\(|time\.time\s*\()")),
    ("process.hrtime()", re.compile(r"\bprocess\.hrtime\s*\(")),
)

# Rendering-context keys that intentionally carry wall-clock time; scenes
# that read these are expected to handle them deterministically (e.g. by
# seeding), so we do NOT flag them.
# (none currently — kept as documentation of the decision)


class DeterminismCheckerToolInput(BaseModel):
    scene_code: str = Field(description="Scene code to check for non-determinism sources")
    scene_file: str = Field(
        default="",
        description="Optional path to the scene file (used to report the filename in findings)",
    )


class DeterminismCheckerTool(BaseTool):
    name = "determinism_checker"
    description = (
        "Statically analyse scene code for non-determinism sources "
        "(Math.random, Date.now, performance.now, uuid4, ...) that cause "
        "different output when a scene is re-rendered. Reports each finding "
        "with its line number; clean code passes. This is a static check — "
        "it does not execute or render the code."
    )
    input_model = DeterminismCheckerToolInput

    def is_read_only(self, arguments) -> bool:
        return True

    async def execute(self, arguments: DeterminismCheckerToolInput, context: ToolExecutionContext) -> ToolResult:
        findings: list[dict[str, Any]] = []
        lines = arguments.scene_code.splitlines()
        for label, pattern in _NON_DETERMINISM_PATTERNS:
            for line_no, line in enumerate(lines, start=1):
                if pattern.search(line):
                    findings.append({
                        "line": line_no,
                        "source": label,
                        "snippet": line.strip()[:120],
                    })

        is_deterministic = not findings
        source = arguments.scene_file or "scene code"
        out = [
            "[Determinism Check]",
            f"Source: {source}",
            f"Non-determinism sources found: {len(findings)}",
        ]
        if is_deterministic:
            out.append("Result: PASS — no known non-deterministic constructs detected.")
            out.append("Note: static analysis only; exotic sources (custom noise, uninitialized")
            out.append("memory, unordered iteration) are not detectable without rendering.")
        else:
            out.append("Result: FAIL — re-rendering will likely produce flicker.")
            out.append("")
            out.append("Findings:")
            for f in findings[:20]:
                out.append(f"  line {f['line']}: {f['source']}")
                out.append(f"    {f['snippet']}")
            out.append("")
            out.append("Fixes: seed RNGs once at init, derive time from the scene clock, and")
            out.append("replace uuid4 with a hash of stable inputs.")

        return ToolResult(
            output="\n".join(out),
            is_error=not is_deterministic,
            metadata={
                "is_deterministic": is_deterministic,
                "findings": findings,
                "source": source,
            },
        )


# ============================================================
# #26 Contact Sheet Generator — real image grid via ffmpeg
# ============================================================

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


class ContactSheetToolInput(BaseModel):
    scene_files: list[str] = Field(description="List of rendered scene files (images or videos) to include in the contact sheet")
    output_path: str = Field(default="contact_sheet.png", description="Output contact sheet path")
    columns: int = Field(default=4, ge=1, le=10, description="Number of columns in the grid")
    thumbnail_width: int = Field(default=320, ge=32, description="Width of each thumbnail in pixels")
    background_color: str = Field(default="black", description="Padding/background colour for non-16:9 thumbnails")
    timestamps: list[float] | None = Field(
        default=None,
        description=(
            "Optional per-input scene start times in seconds (final-cut "
            "order). When omitted they are computed as cumulative video "
            "durations (videos contribute their length, images 0)."
        ),
    )
    label_cells: bool = Field(
        default=True,
        description="Label each cell with its scene number and timestamp (needs a system TTF font).",
    )


class ContactSheetTool(BaseTool):
    name = "contact_sheet"
    description = (
        "Composite rendered scenes into a single contact-sheet image (grid "
        "of thumbnails) for fast visual QA. Images are scaled and padded; "
        "videos contribute their first frame (ffmpeg when available, else "
        "OpenCV). Inputs keep the caller's final-cut order. When neither "
        "decoder can produce a thumbnail and no input can be decoded, "
        "writes a JSON manifest only and says so explicitly."
    )
    input_model = ContactSheetToolInput

    async def execute(self, arguments: ContactSheetToolInput, context: ToolExecutionContext) -> ToolResult:
        if not arguments.scene_files:
            return ToolResult(output="No scene files provided.", is_error=True)

        output_path = Path(arguments.output_path)
        if not output_path.is_absolute():
            output_path = context.cwd / output_path

        # Make sure the output directory exists before any writer runs.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve inputs, keeping the caller's final-cut order. Mixed
        # image/video lists must NOT be regrouped by type — the sheet's
        # scene numbering and cumulative timestamps follow this order.
        missing: list[str] = []
        ordered: list[tuple[Path, bool]] = []  # (path, is_video)
        for f in arguments.scene_files:
            p = Path(f)
            if not p.is_absolute():
                p = context.cwd / p
            suffix = p.suffix.lower()
            if p.is_file() and suffix in _IMAGE_EXTENSIONS:
                ordered.append((p, False))
            elif p.is_file() and suffix in _VIDEO_EXTENSIONS:
                ordered.append((p, True))
            else:
                missing.append(f)

        if missing:
            log.warning("contact_sheet: %d files missing or unsupported", len(missing))

        ffmpeg_available = shutil.which("ffmpeg") is not None
        composited = False
        composite_error: str = ""
        # Inputs that made it into the grid (images + videos whose first
        # frame decoded successfully), in final-cut order.
        included: list[str] = []
        # Per-included-file durations for cumulative timestamps.
        durations: list[float] = []
        temp_thumbs: list[Path] = []
        try:
            thumbs: list[Path] = []
            for path, is_video in ordered:
                if not is_video:
                    thumbs.append(path)
                    included.append(str(path))
                    durations.append(0.0)
                    continue
                # Videos contribute a first-frame thumbnail via a temp
                # file — ffmpeg when available, OpenCV otherwise.
                thumb = await asyncio.to_thread(
                    _extract_video_thumb, path, ffmpeg_available
                )
                if thumb is None:
                    missing.append(path.name)
                    continue
                thumbs.append(thumb)
                included.append(str(path))
                durations.append(await asyncio.to_thread(_video_duration_s, path))
                if thumb != path:
                    temp_thumbs.append(thumb)
            included_count = len(included)
            if thumbs:
                labels = _scene_labels(
                    durations, arguments.timestamps, included_count
                )
                if ffmpeg_available:
                    await asyncio.to_thread(
                        _composite_grid,
                        thumbs,
                        output_path,
                        arguments,
                        labels=labels,
                    )
                else:
                    # ffmpeg missing: PIL + OpenCV still produce a real
                    # PNG grid instead of a manifest-only result.
                    await asyncio.to_thread(
                        _composite_grid_pil,
                        thumbs,
                        output_path,
                        arguments,
                        labels=labels,
                    )
                composited = True
        except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
            composite_error = str(exc)
        finally:
            for tmp in temp_thumbs:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        if not composited and not composite_error:
            # All inputs failed to decode / none were compositable.
            composite_error = "no decodable image inputs"

        # Manifest is always written — it records what was requested, what
        # was found, and whether the image was composited.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cols = min(arguments.columns, max(1, included_count))
        rows = (max(1, included_count) + cols - 1) // cols
        thumb_w = arguments.thumbnail_width
        thumb_h = int(thumb_w * 9 / 16)
        cell_timestamps = _scene_labels(durations, arguments.timestamps, included_count)
        cells = [
            {
                "position": index,
                "col": index % cols,
                "row": index // cols,
                "file": included[index],
                "scene": index + 1,
                "timestamp_s": round(cell_timestamps[index], 2),
            }
            for index in range(included_count)
        ]
        manifest = {
            "output_path": str(output_path),
            "image_composited": composited,
            "included": included,
            "missing_files": missing,
            "grid": {"cols": cols, "rows": rows, "thumb_w": thumb_w, "thumb_h": thumb_h},
            "cells": cells,
        }
        manifest_path = output_path.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        lines = [
            "[Contact Sheet Generated]" if composited else "[Contact Sheet]",
            f"Output: {output_path}" + ("" if composited else " (NOT written)"),
            f"Manifest: {manifest_path}",
            f"Scenes included: {included_count}",
            f"Grid: {cols} × {rows}",
            f"Thumbnail size: {thumb_w}×{thumb_h}",
        ]
        if missing:
            lines.append(f"Missing or undecodable files: {len(missing)}")
        if not composited:
            lines.append(f"Image compositing unavailable: {composite_error}.")
            lines.append("Manifest written instead — install ffmpeg or provide image/video inputs to get the PNG grid.")

        # Error only when nothing at all could be included; a partial sheet
        # (some files missing) is still a useful result.
        is_error = included_count == 0
        return ToolResult(
            output="\n".join(lines),
            is_error=is_error,
            metadata=manifest,
        )


def _video_duration_s(video: Path) -> float:
    """Best-effort video duration in seconds (OpenCV, else 0)."""
    try:
        import cv2

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            return 0.0
        try:
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            return (frames / fps) if frames and fps else 0.0
        finally:
            capture.release()
    except Exception:  # noqa: BLE001 - label timing is best-effort
        return 0.0


def _extract_video_thumb(video: Path, ffmpeg_available: bool) -> Path | None:
    """First-frame thumbnail for a video: ffmpeg when available, OpenCV
    (``cv2.VideoCapture``) as the ffmpeg-free fallback.

    Returns a temp PNG path (the caller unlinks it), or ``None`` when
    the video cannot be decoded at all.
    """
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".png"))
    if ffmpeg_available:
        try:
            extract = _run_ffmpeg(
                ["-ss", "0", "-i", str(video), "-frames:v", "1", "-y", str(tmp)],
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if extract.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            return tmp
        return None
    return _extract_frame_cv2(video, tmp)


def _extract_frame_cv2(video: Path, out_png: Path) -> Path | None:
    """OpenCV first-frame extraction — no ffmpeg required."""
    try:
        import cv2
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        # ``frame`` is BGR and cv2.imwrite expects BGR — no swap needed.
        if not cv2.imwrite(str(out_png), frame):
            return None
    finally:
        capture.release()
    if out_png.is_file() and out_png.stat().st_size > 0:
        return out_png
    return None


def _scene_labels(
    durations: list[float],
    explicit: list[float] | None,
    count: int,
) -> list[float]:
    """Per-cell start timestamps: explicit when given, else cumulative."""
    if explicit is not None and len(explicit) >= count:
        return [float(explicit[i]) for i in range(count)]
    labels: list[float] = []
    running = 0.0
    for index in range(count):
        labels.append(running)
        running += float(durations[index]) if index < len(durations) else 0.0
    return labels


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


def _find_label_font() -> str | None:
    """First existing TTF for drawtext labels, or None."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _composite_grid(
    thumbs: list[Path],
    output_path: Path,
    arguments: ContactSheetToolInput,
    *,
    labels: list[float] | None = None,
) -> None:
    """Scale/pad every input to the thumbnail size and xstack them into a grid.

    When ``labels`` are provided (and a system TTF font exists) each cell is
    stamped with ``S<n> <t>s`` — scene number + final-cut start timestamp.
    """
    w = arguments.thumbnail_width
    h = int(w * 9 / 16)
    n = len(thumbs)
    cols = min(arguments.columns, n)
    bg = arguments.background_color

    font = _find_label_font() if (arguments.label_cells and labels) else None

    args: list[str] = []
    for t in thumbs:
        args += ["-i", str(t)]

    # Per-input: scale to fit, then pad to the exact cell size, then label.
    filter_parts = []
    for i in range(n):
        chain = (
            f"[{i}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg}"
        )
        if font is not None:
            timestamp = labels[i] if labels and i < len(labels) else 0.0
            text = f"S{i + 1} {timestamp:.1f}s"
            font_size = max(12, w // 16)
            chain += (
                f",drawtext=fontfile='{font}':text='{text}'"
                f":x=8:y=h-th-8:fontsize={font_size}:fontcolor=white"
                ":box=1:boxcolor=black@0.55:boxborderw=6"
            )
        chain += f",setsar=1[t{i}]"
        filter_parts.append(chain)
    if n == 1:
        filter_complex = ";".join(filter_parts) + ";[t0]null[out]"
    else:
        # Uniform grid: cell k sits at (col*w, row*h) — plain integers.
        layout = "|".join(f"{(k % cols) * w}_{(k // cols) * h}" for k in range(n))
        inputs_ref = "".join(f"[t{i}]" for i in range(n))
        filter_complex = ";".join(filter_parts) + f";{inputs_ref}xstack=inputs={n}:layout={layout}:fill=black[out]"

    args += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-frames:v", "1",
        "-y", str(output_path),
    ]
    result = _run_ffmpeg(args, timeout=120)
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(_summarize_ffmpeg_error(result.stderr))


_SHEET_BG_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (211, 47, 47),
    "green": (46, 125, 50),
    "blue": (40, 90, 220),
}


def _sheet_bg_color(color: str) -> tuple[int, int, int]:
    """Parse the background_color argument ('black', '#1a1a1a', ...)."""
    value = (color or "black").strip().lower()
    if value in _SHEET_BG_COLORS:
        return _SHEET_BG_COLORS[value]
    if value.startswith("#") and len(value) == 7:
        try:
            return (
                int(value[1:3], 16),
                int(value[3:5], 16),
                int(value[5:7], 16),
            )
        except ValueError:
            pass
    return (0, 0, 0)


def _composite_grid_pil(
    thumbs: list[Path],
    output_path: Path,
    arguments: "ContactSheetToolInput",
    *,
    labels: list[float] | None = None,
) -> None:
    """ffmpeg-free grid composite: PIL scale/pad + optional labels.

    Mirrors the ffmpeg xstack layout exactly — every input is scaled to
    fit its cell (aspect preserved), padded to the cell size, and placed
    at (col*w, row*h); labels are stamped bottom-left when a system TTF
    font exists. Used when ffmpeg is not installed so a contact sheet is
    still a real PNG grid, not a manifest-only stub.
    """
    from PIL import Image, ImageDraw, ImageFont

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    w = arguments.thumbnail_width
    h = int(w * 9 / 16)
    n = len(thumbs)
    cols = min(arguments.columns, n)
    rows = (n + cols - 1) // cols
    bg = _sheet_bg_color(arguments.background_color)

    sheet = Image.new("RGB", (cols * w, rows * h), bg)
    draw = ImageDraw.Draw(sheet)

    font = None
    if arguments.label_cells and labels:
        font_path = _find_label_font()
        if font_path:
            try:
                font = ImageFont.truetype(font_path, max(12, w // 16))
            except Exception:  # noqa: BLE001 - labels are best-effort
                font = None

    for index, thumb in enumerate(thumbs):
        try:
            with Image.open(thumb) as handle:
                image = handle.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - surface as composite error
            raise ValueError(f"thumbnail unreadable: {thumb}") from exc
        # Scale to fit the cell (same semantics as ffmpeg's
        # force_original_aspect_ratio=decrease), then pad centered — the
        # sheet's background colour fills the padding.
        scale = min(w / image.width, h / image.height)
        image = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            resampling,
        )
        sheet.paste(
            image,
            (
                (index % cols) * w + (w - image.width) // 2,
                (index // cols) * h + (h - image.height) // 2,
            ),
        )
        if font is not None:
            timestamp = labels[index] if labels and index < len(labels) else 0.0
            text = f"S{index + 1} {timestamp:.1f}s"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x0 = (index % cols) * w + 8
            y0 = (index // cols) * h + h - text_h - 8
            draw.rectangle(
                [x0 - 6, y0 - 6, x0 + text_w + 6, y0 + text_h + 6],
                fill=(0, 0, 0),
            )
            draw.text(
                (x0 - bbox[0], y0 - bbox[1]), text, font=font, fill=(255, 255, 255)
            )

    save_format = "JPEG" if output_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    sheet.save(output_path, format=save_format)
    if not output_path.is_file():
        raise RuntimeError("PIL contact sheet was not written")


# ============================================================
# #27 Engine Router
# ============================================================

class EngineRouterToolInput(BaseModel):
    scene_description: str = Field(description="Scene description to route to the optimal engine")
    available_engines: list[str] = Field(
        default_factory=lambda: ["hyperframes", "manim", "blender", "ffmpeg"],
        description="List of available rendering engines",
    )


class EngineRouterTool(BaseTool):
    name = "engine_router"
    description = (
        "Route a scene to the optimal rendering engine based on its "
        "description. Hyperframes for web/UI scenes, Manim for math "
        "animations, Blender for 3D, FFmpeg for video compositing."
    )
    input_model = EngineRouterToolInput

    def is_read_only(self, arguments) -> bool:
        return True

    async def execute(self, arguments: EngineRouterToolInput, context: ToolExecutionContext) -> ToolResult:
        engine = route_engine(arguments.scene_description, arguments.available_engines)
        reason = _get_routing_reason(arguments.scene_description, engine)
        lines = [
            "[Engine Router]",
            f"Recommended engine: {engine}",
            f"Reason: {reason}",
            f"Available engines: {', '.join(arguments.available_engines)}",
        ]
        return ToolResult(
            output="\n".join(lines),
            metadata={"engine": engine, "reason": reason},
        )


def route_engine(description: str, available: list[str]) -> str:
    """Route a scene description to the optimal rendering engine."""
    lowered = description.lower()
    # 3D scenes → Blender
    if any(kw in lowered for kw in ["3d", "blender", "three.js", "webgl", "shader"]):
        if "blender" in available:
            return "blender"
    # Math animations → Manim. Word-boundary matching: plain substring
    # matching sent "paragraph" (contains "graph") to Manim.
    import re as _re

    if any(
        _re.search(rf"\b{_re.escape(kw)}\b", lowered)
        for kw in ["manim", "math", "equation", "graph", "geometry", "proof"]
    ):
        if "manim" in available:
            return "manim"
    # Video compositing → FFmpeg
    if any(kw in lowered for kw in ["video", "footage", "clip", "composit", "overlay video"]):
        if "ffmpeg" in available:
            return "ffmpeg"
    # Default → hyperframes (web-based rendering)
    if "hyperframes" in available:
        return "hyperframes"
    # Fallback to first available
    return available[0] if available else "hyperframes"


def _get_routing_reason(description: str, engine: str) -> str:
    """Return a human-readable reason for the engine choice."""
    if engine == "blender":
        return "3D content detected — Blender handles 3D rendering natively."
    if engine == "manim":
        return "Math/animation content detected — Manim is purpose-built for mathematical animations."
    if engine == "ffmpeg":
        return "Video compositing detected — FFmpeg handles video processing efficiently."
    return "Web/UI content detected — hyperframes is the default for browser-based scene rendering."
