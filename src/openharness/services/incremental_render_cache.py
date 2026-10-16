"""Incremental render cache — Item #12.

When a scene is revised, only re-render the changed frames, not the
whole scene. Uses frame hashing to detect which frames changed.

This module provides:
- ``hash_frame(frame_data)`` — deterministic hash of a frame's pixels.
- ``IncrementalRenderCache`` — stores per-frame hashes and reuses
  unchanged frames from the previous render.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def hash_frame(frame_data: bytes) -> str:
    """Return a deterministic SHA-256 hash of a frame's pixel data."""
    return hashlib.sha256(frame_data).hexdigest()


@dataclass
class FrameHashEntry:
    """A single frame hash entry."""

    frame_index: int
    frame_hash: str
    rendered_path: str | None = None  # path to the cached rendered frame


@dataclass
class SceneRenderCache:
    """Cache of rendered frames for a single scene."""

    scene_id: str
    code_hash: str  # hash of the scene code that produced these frames
    frame_hashes: dict[int, str] = field(default_factory=dict)  # frame_index → hash
    frame_paths: dict[int, str] = field(default_factory=dict)  # frame_index → rendered path

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "code_hash": self.code_hash,
            "frame_hashes": dict(self.frame_hashes),
            "frame_paths": dict(self.frame_paths),
        }


class IncrementalRenderCache:
    """Per-scene frame cache for incremental rendering.

    When a scene is re-rendered after a code change, this cache lets us
    skip frames whose hash matches the previous render. Only changed
    frames are re-rendered.

    Usage:
        cache = IncrementalRenderCache(cache_dir=".openharness/render_cache")

        # Before rendering scene_001:
        scene_cache = cache.get_scene_cache("scene_001")

        # If the scene code hasn't changed, all frames are cached:
        if cache.is_fully_cached("scene_001", new_code_hash):
            return cache.get_all_frame_paths("scene_001")

        # Otherwise, render frame-by-frame, skipping cached frames:
        for frame_idx, frame_data in enumerate(frame_iterator):
            frame_hash = hash_frame(frame_data)
            if cache.is_frame_cached("scene_001", frame_idx, frame_hash):
                # Reuse the cached frame
                yield cache.get_frame_path("scene_001", frame_idx)
            else:
                # Render this frame and cache it
                rendered_path = await render_frame(frame_data)
                cache.store_frame("scene_001", frame_idx, frame_hash, rendered_path)
                yield rendered_path
    """

    def __init__(self, *, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._scenes: dict[str, SceneRenderCache] = {}
        self._load_index()

    def get_scene_cache(self, scene_id: str) -> SceneRenderCache | None:
        """Return the render cache for a scene, or None if not cached."""
        return self._scenes.get(scene_id)

    def is_frame_cached(
        self,
        scene_id: str,
        frame_index: int,
        frame_hash: str,
    ) -> bool:
        """Return True if a frame is cached and its hash matches."""
        scene = self._scenes.get(scene_id)
        if scene is None:
            return False
        cached_hash = scene.frame_hashes.get(frame_index)
        return cached_hash == frame_hash

    def is_fully_cached(self, scene_id: str, code_hash: str) -> bool:
        """Return True if a scene's code hash matches and all frames are cached."""
        scene = self._scenes.get(scene_id)
        if scene is None:
            return False
        return scene.code_hash == code_hash and bool(scene.frame_hashes)

    def get_frame_path(self, scene_id: str, frame_index: int) -> str | None:
        """Return the path to a cached frame, or None."""
        scene = self._scenes.get(scene_id)
        if scene is None:
            return None
        return scene.frame_paths.get(frame_index)

    def get_all_frame_paths(self, scene_id: str) -> list[str]:
        """Return paths to all cached frames for a scene, in order."""
        scene = self._scenes.get(scene_id)
        if scene is None:
            return []
        return [scene.frame_paths[i] for i in sorted(scene.frame_hashes.keys())]

    def store_frame(
        self,
        scene_id: str,
        frame_index: int,
        frame_hash: str,
        rendered_path: str,
        *,
        code_hash: str = "",
    ) -> None:
        """Store (or update) a frame in the cache."""
        scene = self._scenes.setdefault(
            scene_id,
            SceneRenderCache(scene_id=scene_id, code_hash=code_hash),
        )
        scene.frame_hashes[frame_index] = frame_hash
        scene.frame_paths[frame_index] = rendered_path
        self._save_index()

    def invalidate_scene(self, scene_id: str) -> None:
        """Drop all cached frames for a scene."""
        if scene_id in self._scenes:
            del self._scenes[scene_id]
            self._save_index()

    def clear_all(self) -> None:
        """Drop the entire cache."""
        self._scenes.clear()
        self._save_index()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        total_frames = sum(len(s.frame_hashes) for s in self._scenes.values())
        return {
            "cached_scenes": len(self._scenes),
            "total_cached_frames": total_frames,
        }

    def _index_path(self) -> Path:
        return self._cache_dir / "render_cache_index.json"

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        index_path = self._index_path()
        if not index_path.is_file():
            return
        try:
            import json
            data = json.loads(index_path.read_text(encoding="utf-8"))
            for scene_id, scene_data in data.get("scenes", {}).items():
                self._scenes[scene_id] = SceneRenderCache(
                    scene_id=scene_id,
                    code_hash=scene_data.get("code_hash", ""),
                    frame_hashes={int(k): v for k, v in scene_data.get("frame_hashes", {}).items()},
                    frame_paths={int(k): v for k, v in scene_data.get("frame_paths", {}).items()},
                )
        except Exception as exc:
            log.warning("failed to load render cache index: %s", exc)

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            import json
            data = {
                "scenes": {sid: s.to_dict() for sid, s in self._scenes.items()}
            }
            self._index_path().write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("failed to save render cache index: %s", exc)


def compute_code_hash(code: str) -> str:
    """Return a deterministic hash of scene code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()
