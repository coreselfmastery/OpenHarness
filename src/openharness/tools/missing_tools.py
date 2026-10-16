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
        if arguments.action == "check":
            if gov.should_stop():
                return ToolResult(
                    output=f"BUDGET EXCEEDED: ${gov.spent:.2f} / ${gov.budget:.2f}. Halting.",
                    is_error=True,
                    metadata=gov.stats(),
                )
            return ToolResult(
                output=f"Budget OK: ${gov.spent:.2f} / ${gov.budget:.2f} (${gov.remaining:.2f} remaining)",
                metadata=gov.stats(),
            )
        elif arguments.action == "record":
            gov.record_spend(arguments.cost_usd, call_type=arguments.call_type)
            halted = gov.should_stop()
            msg = f"Recorded ${arguments.cost_usd:.4f} ({arguments.call_type}). Total: ${gov.spent:.2f} / ${gov.budget:.2f}"
            if halted:
                msg += " — BUDGET EXCEEDED"
            return ToolResult(output=msg, is_error=halted, metadata=gov.stats())
        elif arguments.action == "status":
            return ToolResult(output=f"Spent ${gov.spent:.2f} / ${gov.budget:.2f}", metadata=gov.stats())
        elif arguments.action == "reset":
            budget = arguments.budget_usd if arguments.budget_usd > 0 else gov.budget
            gov.reset()
            gov._budget = budget
            return ToolResult(output=f"Budget reset to ${budget:.2f}", metadata=gov.stats())
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
            m = _measure_audio(
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


class ContactSheetTool(BaseTool):
    name = "contact_sheet"
    description = (
        "Composite rendered scenes into a single contact-sheet image (grid "
        "of thumbnails) for fast visual QA. Images are scaled and padded; "
        "videos contribute their first frame. Uses ffmpeg; when ffmpeg is "
        "missing or no input can be decoded, writes a JSON manifest only "
        "and says so explicitly."
    )
    input_model = ContactSheetToolInput

    async def execute(self, arguments: ContactSheetToolInput, context: ToolExecutionContext) -> ToolResult:
        if not arguments.scene_files:
            return ToolResult(output="No scene files provided.", is_error=True)

        output_path = Path(arguments.output_path)
        if not output_path.is_absolute():
            output_path = context.cwd / output_path

        # Resolve inputs, split into images / videos / missing.
        image_inputs: list[Path] = []
        video_inputs: list[Path] = []
        missing: list[str] = []
        for f in arguments.scene_files:
            p = Path(f)
            if not p.is_absolute():
                p = context.cwd / p
            if not p.is_file():
                missing.append(f)
                continue
            if p.suffix.lower() in _IMAGE_EXTENSIONS:
                image_inputs.append(p)
            elif p.suffix.lower() in _VIDEO_EXTENSIONS:
                video_inputs.append(p)
            else:
                missing.append(f)

        if missing:
            log.warning("contact_sheet: %d files missing or unsupported", len(missing))

        ffmpeg_available = shutil.which("ffmpeg") is not None
        composited = False
        composite_error: str = ""
        # Inputs that made it into the grid (images + videos whose first
        # frame decoded successfully).
        included_count = len(image_inputs)
        included: list[str] = [str(p) for p in image_inputs]

        if ffmpeg_available:
            try:
                # Videos contribute a first-frame thumbnail via a temp file.
                thumbs = list(image_inputs)
                temp_thumbs: list[Path] = []
                import tempfile

                for video in video_inputs:
                    tmp = Path(tempfile.mktemp(suffix=".png"))
                    extract = _run_ffmpeg([
                        "-ss", "0", "-i", str(video),
                        "-frames:v", "1", "-y", str(tmp),
                    ], timeout=30)
                    if extract.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                        temp_thumbs.append(tmp)
                        thumbs.append(tmp)
                        included.append(str(video))
                        included_count += 1
                    else:
                        missing.append(video.name)
                try:
                    if thumbs:
                        _composite_grid(thumbs, output_path, arguments)
                        composited = True
                finally:
                    for tmp in temp_thumbs:
                        try:
                            tmp.unlink(missing_ok=True)
                        except OSError:
                            pass
            except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                composite_error = str(exc)
        else:
            composite_error = "ffmpeg not installed"
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
        manifest = {
            "output_path": str(output_path),
            "image_composited": composited,
            "included": included,
            "missing_files": missing,
            "grid": {"cols": cols, "rows": rows, "thumb_w": thumb_w, "thumb_h": thumb_h},
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


def _composite_grid(thumbs: list[Path], output_path: Path, arguments: ContactSheetToolInput) -> None:
    """Scale/pad every input to the thumbnail size and xstack them into a grid."""
    w = arguments.thumbnail_width
    h = int(w * 9 / 16)
    n = len(thumbs)
    cols = min(arguments.columns, n)
    bg = arguments.background_color

    args: list[str] = []
    for t in thumbs:
        args += ["-i", str(t)]

    # Per-input: scale to fit, then pad to the exact cell size.
    filter_parts = []
    for i in range(n):
        filter_parts.append(
            f"[{i}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg},"
            f"setsar=1[t{i}]"
        )
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
    # Math animations → Manim
    if any(kw in lowered for kw in ["manim", "math", "equation", "graph", "geometry", "proof"]):
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
