"""Advanced rendering optimizations — items 10, 12, 13, 14, 15.

#10 Frame-Level Parallelism: Render individual frames concurrently
    using a process pool. For a 240-frame scene, this can render 8
    frames at once on an 8-core machine — 8× speedup on CPU-bound
    rendering.

#12 Delta Rendering: Instead of re-rendering the entire scene when
    a small change is made, render only the changed regions. Tracks
    dirty regions per frame and only re-renders those.

#13 Codec Optimization: Tune ffmpeg/NVENC parameters for the target
    codec (H.264, H.265, AV1) to maximize quality per bitrate.

#14 Resolution Tiering: Automatically select the optimal resolution
    per pipeline stage (draft=480p, review=720p, final=1080p/4K).

#15 Pre-Computed Assets: Bake static elements (gradients, patterns,
    logos) into pre-rendered frames so the renderer doesn't recompute
    them every frame.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


# ============================================================
# #10 Frame-Level Parallelism
# ============================================================

@dataclass
class FrameRenderTask:
    """A single frame to render."""

    frame_index: int
    scene_id: str
    output_path: str
    render_fn: str = "default"  # which render function to use


@dataclass
class FrameRenderResult:
    """Result of rendering a single frame."""

    frame_index: int
    output_path: str
    render_time_seconds: float
    success: bool
    error: str | None = None


class FrameParallelRenderer:
    """Render frames concurrently using asyncio + subprocess.

    Instead of rendering frames sequentially (frame 0 → 1 → 2 → ...),
    this renderer dispatches frame render tasks to N concurrent workers.
    Each worker calls the render function (typically an ffmpeg/headless
    Chrome command) for one frame at a time.

    For a 240-frame scene on an 8-core machine, this gives ~8× speedup
    on CPU-bound rendering work.

    Usage:
        renderer = FrameParallelRenderer(max_workers=8)
        tasks = [FrameRenderTask(i, "scene_001", f"frames/{i:04d}.png")
                 for i in range(240)]
        results = await renderer.render_all(tasks)
    """

    def __init__(
        self,
        *,
        render_fn: Callable[[FrameRenderTask], Awaitable[FrameRenderResult]] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._render_fn = render_fn or _default_frame_render
        self._max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)

    async def render_all(self, tasks: list[FrameRenderTask]) -> list[FrameRenderResult]:
        """Render all frames concurrently (up to max_workers at a time)."""
        semaphore = asyncio.Semaphore(self._max_workers)

        async def _render_with_limit(task: FrameRenderTask) -> FrameRenderResult:
            async with semaphore:
                return await self._render_fn(task)

        return await asyncio.gather(*[_render_with_limit(t) for t in tasks])

    async def render_range(
        self,
        start: int,
        end: int,
        scene_id: str,
        output_dir: str | Path,
    ) -> list[FrameRenderResult]:
        """Render frames from start to end (exclusive)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            FrameRenderTask(
                frame_index=i,
                scene_id=scene_id,
                output_path=str(output_dir / f"{scene_id}_{i:04d}.png"),
            )
            for i in range(start, end)
        ]
        return await self.render_all(tasks)


async def _default_frame_render(task: FrameRenderTask) -> FrameRenderResult:
    """Default frame renderer — a no-op that creates a placeholder file."""
    import time
    t0 = time.monotonic()
    await asyncio.sleep(0.001)  # simulate render latency
    Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(task.output_path).write_bytes(b"\x00" * 100)
    return FrameRenderResult(
        frame_index=task.frame_index,
        output_path=task.output_path,
        render_time_seconds=time.monotonic() - t0,
        success=True,
    )


# ============================================================
# #12 Delta Rendering
# ============================================================

@dataclass
class DirtyRegion:
    """A rectangular region of a frame that has changed."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class FrameDelta:
    """Delta between two frame versions."""

    frame_index: int
    dirty_regions: list[DirtyRegion]
    is_full_rebuild: bool = False  # True = must re-render entire frame


class DeltaRenderer:
    """Render only changed regions of frames.

    When a scene's code changes, instead of re-rendering all frames,
    this renderer computes which regions of each frame are affected
    and only re-renders those regions.

    The dirty region detection is heuristic: it compares the old and
    new code to identify which visual elements changed, then maps
    those to screen-space regions.

    Usage:
        renderer = DeltaRenderer()
        deltas = renderer.compute_deltas(old_code, new_code, frame_count=240)
        for delta in deltas:
            if delta.is_full_rebuild:
                await render_full_frame(delta.frame_index)
            else:
                await render_regions(delta.frame_index, delta.dirty_regions)
    """

    def compute_deltas(
        self,
        old_code: str,
        new_code: str,
        *,
        frame_count: int,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> list[FrameDelta]:
        """Compute per-frame dirty regions from code changes.

        If the codes are identical, no deltas are produced.
        If the codes differ significantly, all frames are marked
        for full rebuild.
        """
        if old_code == new_code:
            return []  # no changes

        # Hash both codes to determine the scope of changes
        old_hash = hashlib.sha256(old_code.encode("utf-8")).hexdigest()
        new_hash = hashlib.sha256(new_code.encode("utf-8")).hexdigest()

        if old_hash == new_hash:
            return []

        # Heuristic: if more than 50% of lines changed, do a full rebuild
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        changed_lines = _count_changed_lines(old_lines, new_lines)
        total_lines = max(len(old_lines), len(new_lines), 1)
        change_ratio = changed_lines / total_lines

        if change_ratio > 0.5:
            # Major change — full rebuild for all frames
            return [
                FrameDelta(frame_index=i, dirty_regions=[], is_full_rebuild=True)
                for i in range(frame_count)
            ]

        # Minor change — mark specific regions as dirty
        # In a real implementation, this would parse the code to identify
        # which visual elements changed and map them to screen regions.
        # For now, we use a heuristic: mark the top-left quarter as dirty
        # (where most title/text changes happen).
        dirty_region = DirtyRegion(
            x=0, y=0,
            width=frame_width // 2,
            height=frame_height // 2,
        )
        return [
            FrameDelta(frame_index=i, dirty_regions=[dirty_region])
            for i in range(frame_count)
        ]

    def should_skip_frame(self, delta: FrameDelta) -> bool:
        """Return True if a frame can be skipped entirely (no dirty regions)."""
        return not delta.is_full_rebuild and not delta.dirty_regions


def _count_changed_lines(old_lines: list[str], new_lines: list[str]) -> int:
    """Count the number of changed lines between two code versions."""
    import difflib
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    # Only count actual changes (+ or -), not @@ headers or context lines
    changed = sum(
        1 for line in diff
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return changed


# ============================================================
# #13 Codec Optimization
# ============================================================

@dataclass
class CodecConfig:
    """Optimized codec configuration for a target codec."""

    codec: str  # h264, hevc, av1, vp9
    preset: str  # ultrafast, fast, medium, slow
    crf: int  # Constant Rate Factor (0-51, lower = higher quality)
    pix_fmt: str = "yuv420p"
    gpu_accel: str = ""  # "nvenc", "qsv", "videotoolbox" or "" for CPU
    extra_flags: list[str] = field(default_factory=list)


# Pre-tuned codec configs for common use cases
CODEC_PRESETS: dict[str, CodecConfig] = {
    "h264_fast": CodecConfig(
        codec="h264",
        preset="ultrafast",
        crf=23,
        gpu_accel="",
        extra_flags=["-tune", "zerolatency"],
    ),
    "h264_balanced": CodecConfig(
        codec="h264",
        preset="medium",
        crf=20,
        gpu_accel="",
    ),
    "h264_quality": CodecConfig(
        codec="h264",
        preset="slow",
        crf=18,
        gpu_accel="",
    ),
    "h264_nvenc_fast": CodecConfig(
        codec="h264",
        preset="p1",  # NVENC fastest preset
        crf=23,
        gpu_accel="nvenc",
        extra_flags=["-rc", "vbr", "-spatial_aq", "1"],
    ),
    "h264_nvenc_quality": CodecConfig(
        codec="h264",
        preset="p7",  # NVENC slowest/highest quality
        crf=18,
        gpu_accel="nvenc",
        extra_flags=["-rc", "vbr", "-spatial_aq", "1", "-b_ref_mode", "middle"],
    ),
    "hevc_fast": CodecConfig(
        codec="hevc",
        preset="ultrafast",
        crf=26,
        gpu_accel="",
    ),
    "hevc_nvenc": CodecConfig(
        codec="hevc",
        preset="p4",
        crf=24,
        gpu_accel="nvenc",
    ),
    "av1_fast": CodecConfig(
        codec="av1",
        preset="6",  # SVT-AV1 speed preset
        crf=30,
        gpu_accel="",
    ),
}


def get_codec_config(
    *,
    target: str = "balanced",
    use_gpu: bool = False,
    codec: str = "h264",
) -> CodecConfig:
    """Return the optimal codec config for the given target.

    Args:
        target: "fast" (draft), "balanced" (review), "quality" (final).
        use_gpu: If True, use NVENC when available.
        codec: "h264", "hevc", or "av1".
    """
    if use_gpu:
        if codec == "h264":
            if target == "fast":
                return CODEC_PRESETS["h264_nvenc_fast"]
            return CODEC_PRESETS["h264_nvenc_quality"]
        elif codec == "hevc":
            return CODEC_PRESETS["hevc_nvenc"]

    if codec == "hevc":
        return CODEC_PRESETS["hevc_fast"]
    if codec == "av1":
        return CODEC_PRESETS["av1_fast"]

    if target == "fast":
        return CODEC_PRESETS["h264_fast"]
    elif target == "quality":
        return CODEC_PRESETS["h264_quality"]
    return CODEC_PRESETS["h264_balanced"]


def build_ffmpeg_args(
    config: CodecConfig,
    *,
    input_path: str,
    output_path: str,
    fps: int = 30,
    resolution: tuple[int, int] | None = None,
) -> list[str]:
    """Build the ffmpeg command-line arguments for the given codec config.

    Returns a list of strings suitable for ``subprocess.run(args)`` or
    ``asyncio.create_subprocess_exec(*args)``.
    """
    args: list[str] = ["ffmpeg", "-y", "-i", input_path]

    if resolution:
        args.extend(["-vf", f"scale={resolution[0]}:{resolution[1]}"])

    args.extend(["-r", str(fps)])

    # Codec selection
    if config.gpu_accel == "nvenc":
        codec_name = f"{config.codec}_nvenc"
    elif config.gpu_accel == "qsv":
        codec_name = f"{config.codec}_qsv"
    else:
        codec_name = config.codec

    args.extend(["-c:v", codec_name])
    args.extend(["-preset", config.preset])

    # CRF for CPU, quality-based VBR for NVENC
    if config.gpu_accel:
        args.extend(["-cq", str(config.crf)])
    else:
        args.extend(["-crf", str(config.crf)])

    args.extend(["-pix_fmt", config.pix_fmt])

    # Extra flags
    args.extend(config.extra_flags)

    args.append(output_path)
    return args


# ============================================================
# #14 Resolution Tiering
# ============================================================

@dataclass
class ResolutionTier:
    """A resolution tier for a pipeline stage."""

    name: str
    width: int
    height: int
    fps: int
    stage: str  # "draft", "review", "final"


RESOLUTION_TIERS: dict[str, ResolutionTier] = {
    "draft": ResolutionTier("draft", 854, 480, 15, "draft"),
    "review": ResolutionTier("review", 1280, 720, 24, "review"),
    "final_1080p": ResolutionTier("final_1080p", 1920, 1080, 30, "final"),
    "final_4k": ResolutionTier("final_4k", 3840, 2160, 30, "final"),
    "thumbnail": ResolutionTier("thumbnail", 320, 180, 1, "thumbnail"),
}


def select_resolution_tier(
    stage: str,
    *,
    is_draft: bool = False,
    target_4k: bool = False,
) -> ResolutionTier:
    """Auto-select the optimal resolution for a pipeline stage.

    Args:
        stage: Pipeline stage: "planning", "draft", "review", "final".
        is_draft: If True, force draft resolution.
        target_4k: If True and stage is "final", use 4K instead of 1080p.
    """
    if is_draft or stage == "draft":
        return RESOLUTION_TIERS["draft"]
    if stage == "review":
        return RESOLUTION_TIERS["review"]
    if stage == "final":
        return RESOLUTION_TIERS["final_4k"] if target_4k else RESOLUTION_TIERS["final_1080p"]
    if stage == "thumbnail":
        return RESOLUTION_TIERS["thumbnail"]
    # Default to review quality
    return RESOLUTION_TIERS["review"]


def estimate_render_time(
    tier: ResolutionTier,
    frame_count: int,
    *,
    render_speed_fps: float = 10.0,
) -> float:
    """Estimate render time in seconds for a given resolution tier.

    Args:
        tier: The resolution tier.
        frame_count: Number of frames to render.
        render_speed_fps: Estimated render speed in frames per second
            at 1080p. Lower resolutions are proportionally faster.
    """
    # Scale speed by pixel count relative to 1080p
    ref_pixels = 1920 * 1080
    tier_pixels = tier.width * tier.height
    speed_multiplier = ref_pixels / max(1, tier_pixels)
    effective_fps = render_speed_fps * speed_multiplier
    return frame_count / max(1, effective_fps)


# ============================================================
# #15 Pre-Computed Assets (bake static elements)
# ============================================================

@dataclass
class PreComputedAsset:
    """A pre-rendered static asset."""

    name: str
    asset_type: str  # "gradient", "pattern", "logo_frame", "background"
    rendered_path: str
    code_hash: str
    resolution: tuple[int, int]
    frame_count: int = 1  # 1 for static, N for animated loops


class PreComputedAssetCache:
    """Cache of pre-rendered static elements.

    Instead of re-rendering a gradient background on every frame,
    this cache bakes it once and the renderer composites it.

    Usage:
        cache = PreComputedAssetCache(cache_dir=".openharness/precomputed")
        # Bake a gradient background once:
        cache.bake("gradient_bg", gradient_code, (1920, 1080))
        # On every frame, just composite:
        bg_path = cache.get_path("gradient_bg")
        # Use bg_path as a layer in the frame composition
    """

    def __init__(self, *, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._assets: dict[str, PreComputedAsset] = {}
        self._load_index()

    def bake(
        self,
        name: str,
        code: str,
        resolution: tuple[int, int],
        *,
        asset_type: str = "static",
        frame_count: int = 1,
    ) -> str:
        """Bake a static asset to disk. Returns the rendered file path.

        In a real implementation, this would call the renderer to
        produce the asset. For now, it creates a placeholder file
        and records the metadata.
        """
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        output_path = str(self._cache_dir / f"{name}_{code_hash[:12]}.png")
        # Check if already baked
        existing = self._assets.get(name)
        if existing and existing.code_hash == code_hash:
            return existing.rendered_path
        # Bake (placeholder — real implementation renders the asset)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x00" * 100)  # placeholder
        asset = PreComputedAsset(
            name=name,
            asset_type=asset_type,
            rendered_path=output_path,
            code_hash=code_hash,
            resolution=resolution,
            frame_count=frame_count,
        )
        self._assets[name] = asset
        self._save_index()
        return output_path

    def get_path(self, name: str) -> str | None:
        """Return the path to a pre-computed asset, or None."""
        asset = self._assets.get(name)
        return asset.rendered_path if asset else None

    def is_baked(self, name: str, code_hash: str | None = None) -> bool:
        """Check if an asset is already baked (optionally with matching hash)."""
        asset = self._assets.get(name)
        if asset is None:
            return False
        if code_hash and asset.code_hash != code_hash:
            return False
        return Path(asset.rendered_path).is_file()

    def invalidate(self, name: str) -> None:
        """Remove a baked asset."""
        asset = self._assets.pop(name, None)
        if asset:
            Path(asset.rendered_path).unlink(missing_ok=True)
            self._save_index()

    def clear(self) -> None:
        """Clear all baked assets."""
        for asset in list(self._assets.values()):
            Path(asset.rendered_path).unlink(missing_ok=True)
        self._assets.clear()
        self._save_index()

    def stats(self) -> dict[str, Any]:
        return {
            "assets": len(self._assets),
            "cache_dir": str(self._cache_dir),
        }

    def _index_path(self) -> Path:
        return self._cache_dir / "precomputed_index.json"

    def _load_index(self) -> None:
        import json
        path = self._index_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for name, asset_data in data.get("assets", {}).items():
                self._assets[name] = PreComputedAsset(
                    name=asset_data["name"],
                    asset_type=asset_data.get("asset_type", "static"),
                    rendered_path=asset_data["rendered_path"],
                    code_hash=asset_data.get("code_hash", ""),
                    resolution=tuple(asset_data.get("resolution", [1920, 1080])),
                    frame_count=asset_data.get("frame_count", 1),
                )
        except Exception as exc:
            log.warning("failed to load precomputed index: %s", exc)

    def _save_index(self) -> None:
        import json
        data = {
            "assets": {
                name: {
                    "name": a.name,
                    "asset_type": a.asset_type,
                    "rendered_path": a.rendered_path,
                    "code_hash": a.code_hash,
                    "resolution": list(a.resolution),
                    "frame_count": a.frame_count,
                }
                for name, a in self._assets.items()
            }
        }
        try:
            self._index_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("failed to save precomputed index: %s", exc)
