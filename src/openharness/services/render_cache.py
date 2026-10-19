"""Scene-level render cache — items #14 and #17.

Caches render outputs keyed by a *structural fingerprint* of everything
that determines the output — now including the engine binary's
(path, mtime, size), so upgrading the renderer invalidates stale hits:

    SHA-256( scene code
           + referenced asset contents
           + engine + engine args
           + resolution tier + codec settings
           + engine version (when detectable) )

A cache hit means the exact same render already exists on disk, so the
job is skipped entirely — not "probably similar", but byte-identical
inputs. Fingerprints persist in the shared SQLite store (#19), so the
cache survives restarts and is shared by every agent in the project.

Item #14 ("incremental render cache: only re-render changed frames") is
implemented at the same fidelity the engines allow: scene-level
fingerprints. A scene whose code changed re-renders; an unchanged scene
never does. Frame-level partial re-rendering would require engine
cooperation (frame hashing inside hyperframes/manim), which does not
exist, so claiming it here would be a lie.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openharness.services.persistent_store import fingerprint, get_persistent_store

log = logging.getLogger(__name__)

NAMESPACE = "render_cache"

# Asset reference patterns per engine. Matched against scene code to find
# files whose CONTENT affects the render (images, audio, fonts, meshes).
_ASSET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("hyperframes", r"""["']([^"']+\.(?:png|jpg|jpeg|webp|svg|gif|mp3|wav|ogg|mp4|webm|ttf|otf|woff2?))["']"""),
    ("manim", r"""["']([^"']+\.(?:png|jpg|jpeg|webp|svg|gif|mp3|wav|ogg|mp4|svg|ttf|otf))["']"""),
    ("blender", r"""["']([^"']+\.(?:blend|png|jpg|jpeg|hdr|exr|fbx|obj|stl|glb|gltf))["']"""),
    # NOTE: ``-\i`` below is the ffmpeg ``-i`` (input) flag — it MUST be
    # ``-i``, not ``\i``: a backslash makes this an invalid regex escape
    # (``re.error: bad escape \i``) which crashed every ffmpeg-engine
    # fingerprint/asset scan at submit time.
    ("ffmpeg", r"""["'\s](?:-i\s+)?([^\s"']+\.(?:png|jpg|jpeg|webp|svg|mp3|wav|ogg|mp4|webm|mov|srt|ass|ttf))["']?"""),
)
_DEFAULT_ASSET_PATTERN = _ASSET_PATTERNS[0][1]


def find_asset_paths(scene_code: str, engine: str, *, cwd: Path) -> list[Path]:
    """Extract referenced asset file paths from scene code (best effort)."""
    import re

    pattern = dict(_ASSET_PATTERNS).get(engine, _DEFAULT_ASSET_PATTERN)
    paths: list[Path] = []
    seen: set[str] = set()
    for match in re.finditer(pattern, scene_code):
        raw = match.group(1)
        if not raw or raw.startswith(("http://", "https://")):
            continue  # remote URLs are not hashed locally
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            paths.append(candidate)
    return paths


def scene_fingerprint(
    scene_path: Path,
    *,
    engine: str,
    engine_args: list[str],
    tier: str,
    codec: str,
    cwd: Path,
) -> str | None:
    """Compute the structural fingerprint for a render job.

    Returns None when the scene file does not exist (nothing to cache).
    """
    if not scene_path.is_file():
        return None
    try:
        code = scene_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    asset_digests: list[list[str]] = []
    for asset in find_asset_paths(code, engine, cwd=cwd):
        try:
            import hashlib

            digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:32]
        except OSError:
            digest = "unreadable"
        asset_digests.append([str(asset), digest])

    return fingerprint(
        {
            "scene": scene_path.name,
            "code": code,
            "assets": sorted(asset_digests),
            "engine": engine,
            "engine_args": list(engine_args),
            "tier": tier,
            "codec": codec,
            # Engine VERSION (when detectable): binary path + mtime +
            # size. Upgrading blender/manim/hyperframes used to serve
            # stale cache hits from the OLD binary — same inputs, but a
            # different renderer version can produce different output.
            "engine_binary": _engine_binary_stamp(engine),
        }
    )


_ENGINE_BINARY_STAMPS: dict[str, tuple[str, float, int] | None] = {}


def _engine_binary_stamp(engine: str) -> tuple[str, float, int] | None:
    """(path, mtime, size) of the engine binary — None when not found.

    Cached per engine name: the stat is only re-read when the cached
    mtime/size mismatches, so fingerprinting a batch of scenes costs
    one stat per engine, not per scene.
    """
    import shutil

    cached = _ENGINE_BINARY_STAMPS.get(engine)
    if cached is not None:
        try:
            stat = Path(cached[0]).stat()
            if stat.st_mtime == cached[1] and stat.st_size == cached[2]:
                return cached
        except OSError:
            pass  # binary moved — re-resolve below
    binary = shutil.which(engine)
    if binary is None:
        return None
    try:
        stat = Path(binary).stat()
        stamp = (binary, stat.st_mtime, stat.st_size)
    except OSError:
        return None
    _ENGINE_BINARY_STAMPS[engine] = stamp
    return stamp


class RenderCache:
    """Persistent scene-level render cache (#14/#17)."""

    def __init__(self, store: Any = None) -> None:
        self._store = store if store is not None else get_persistent_store()
        self.hits = 0
        self.misses = 0

    def lookup(self, fingerprint_value: str) -> dict[str, Any] | None:
        """Return the cached output record for this fingerprint, if valid.

        A record is only returned when its output file still exists on
        disk — a deleted file is a miss, not a stale hit.
        """
        if not fingerprint_value:
            self.misses += 1
            return None
        record = self._store.get(NAMESPACE, fingerprint_value)
        if not record:
            self.misses += 1
            return None
        output_path = record.get("output_path", "")
        if output_path and not Path(output_path).is_file():
            # Output vanished (e.g. cleanup) — drop the stale entry.
            self._store.delete(NAMESPACE, fingerprint_value)
            self.misses += 1
            return None
        self.hits += 1
        return record

    def store(
        self,
        fingerprint_value: str,
        *,
        output_path: str,
        scene: str,
        engine: str,
        tier: str,
        duration_seconds: float,
        quality_score: float | None = None,
    ) -> None:
        if not fingerprint_value:
            return
        self._store.set(
            NAMESPACE,
            fingerprint_value,
            {
                "output_path": output_path,
                "scene": scene,
                "engine": engine,
                "tier": tier,
                "duration_seconds": round(duration_seconds, 2),
                "quality_score": quality_score,
            },
        )

    def promote(self, fingerprint_value: str, *, new_tier: str) -> None:
        """Mark a draft output as promoted (early exit, item #13)."""
        record = self._store.get(NAMESPACE, fingerprint_value)
        if record:
            record["promoted_to"] = new_tier
            self._store.set(NAMESPACE, fingerprint_value, record)

    def quality_score(self, fingerprint_value: str) -> float | None:
        record = self._store.get(NAMESPACE, fingerprint_value)
        if not record:
            return None
        score = record.get("quality_score")
        return float(score) if score is not None else None

    def record_quality_score(self, fingerprint_value: str, score: float) -> None:
        record = self._store.get(NAMESPACE, fingerprint_value)
        if record:
            record["quality_score"] = float(score)
            self._store.set(NAMESPACE, fingerprint_value, record)

    def clear(self) -> int:
        return self._store.clear_namespace(NAMESPACE)

    def entries(self, limit: int = 50) -> dict[str, Any]:
        return self._store.namespace_entries(NAMESPACE, limit)

    def stats(self) -> dict[str, Any]:
        base = self._store.stats()
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries(limit=10_000)),
            "store_available": base.get("available", False),
        }
