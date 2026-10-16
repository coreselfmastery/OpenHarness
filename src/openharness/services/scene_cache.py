"""Scene-level caching by structural fingerprint.

Item #3: When two scenes share the same pattern (e.g., two stat cards,
two lower thirds), the agent generates the code twice. This cache stores
generated code keyed by a structural fingerprint, not the raw description.

Item #15: Cross-session persistence. The cache can be saved to a local
DB file so identical patterns across different videos don't get
regenerated.

The fingerprint is a hash of:
- Engine (hyperframes, manim, etc.)
- Duration bucket (rounded to nearest 0.5s)
- Read count (number of narration lines)
- Visual element types (text, shape, image, chart, etc.)
- Color count (palette size)
- Motion keyword count

Two scenes with the same fingerprint are structurally identical and
can reuse each other's code (with minor parameter substitution).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneFingerprint:
    """Structural fingerprint of a scene — used as the cache key.

    Two scenes with the same fingerprint are structurally identical
    (same engine, duration bucket, read count, visual element mix) and
    can reuse each other's generated code.
    """

    engine: str
    duration_bucket: float  # rounded to nearest 0.5s
    read_count: int
    visual_element_types: frozenset[str]
    color_count: int
    motion_keyword_count: int

    def to_hash(self) -> str:
        """Return a stable SHA-1 hash of the fingerprint."""
        payload = json.dumps(
            {
                "e": self.engine,
                "d": self.duration_bucket,
                "r": self.read_count,
                "v": sorted(self.visual_element_types),
                "c": self.color_count,
                "m": self.motion_keyword_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "duration_bucket": self.duration_bucket,
            "read_count": self.read_count,
            "visual_element_types": sorted(self.visual_element_types),
            "color_count": self.color_count,
            "motion_keyword_count": self.motion_keyword_count,
            "hash": self.to_hash(),
        }


def compute_fingerprint(
    description: str,
    *,
    engine: str = "hyperframes",
    duration: float | None = None,
    reads: list[str] | None = None,
) -> SceneFingerprint:
    """Compute a structural fingerprint from a scene description.

    The fingerprint captures the *shape* of the scene, not its content.
    Two scenes with different text but the same structure (e.g., both
    are "title card with one line of text, 5s, hyperframes") get the
    same fingerprint and can reuse each other's code.
    """
    read_count = len(reads) if reads else _estimate_read_count(description)
    duration_bucket = _bucket_duration(duration or _estimate_duration(description))
    visual_types = _extract_visual_element_types(description)
    color_count = _estimate_color_count(description)
    motion_count = _count_motion_keywords(description)
    return SceneFingerprint(
        engine=engine,
        duration_bucket=duration_bucket,
        read_count=read_count,
        visual_element_types=frozenset(visual_types),
        color_count=color_count,
        motion_keyword_count=motion_count,
    )


# ---------------------------------------------------------------------------
# Fingerprint extraction helpers
# ---------------------------------------------------------------------------

def _bucket_duration(duration: float) -> float:
    """Round duration to nearest 0.5s so 8.3s and 8.4s share a bucket."""
    return round(duration * 2) / 2


def _estimate_read_count(description: str) -> int:
    """Estimate the number of narration lines from the description.

    Heuristic: count sentences. A scene typically has 1-3 narration lines.
    """
    sentences = re.split(r"[.!?]+", description)
    sentences = [s.strip() for s in sentences if s.strip()]
    return max(1, min(len(sentences), 5))


def _estimate_duration(description: str) -> float:
    """Estimate scene duration from the description.

    Heuristic: longer descriptions → longer scenes. Base 5s + 0.5s per
    sentence, capped at 30s.
    """
    read_count = _estimate_read_count(description)
    return min(30.0, 5.0 + read_count * 0.5)


_VISUAL_TYPE_PATTERNS: dict[str, list[str]] = {
    "text": [r"\btext\b", r"\btitle\b", r"\bcaption\b", r"\bsubtitle\b", r"\btypography\b"],
    "shape": [r"\bshape\b", r"\brectangle\b", r"\bcircle\b", r"\btriangle\b", r"\bpolygon\b"],
    "image": [r"\bimage\b", r"\bphoto\b", r"\bpicture\b", r"\bbackground\b"],
    "chart": [r"\bchart\b", r"\bgraph\b", r"\bbar\b", r"\bpie\b", r"\bdata viz\b", r"\bstatistics?\b"],
    "animation": [r"\banimat\b", r"\bmotion\b", r"\btransition\b", r"\beasing\b"],
    "3d": [r"\b3d\b", r"\bblender\b", r"\bmanim\b", r"\bthree\.?js\b", r"\bwebgl\b"],
    "particle": [r"\bparticle\b", r"\bspark\b", r"\bsmoke\b", r"\bfire\b"],
    "video": [r"\bvideo\b", r"\bclip\b", r"\bfootage\b"],
}


def _extract_visual_element_types(description: str) -> set[str]:
    """Extract the set of visual element types from the description."""
    lowered = description.lower()
    types: set[str] = set()
    for vtype, patterns in _VISUAL_TYPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lowered):
                types.add(vtype)
                break
    if not types:
        types.add("generic")
    return types


def _estimate_color_count(description: str) -> int:
    """Estimate how many distinct colors the scene uses."""
    colors = re.findall(r"#[0-9A-Fa-f]{6}\b", description)
    color_words = re.findall(
        r"\b(red|blue|green|yellow|purple|orange|pink|cyan|magenta|white|black|gray|grey)\b",
        description,
        re.IGNORECASE,
    )
    return max(1, len(set(colors)) + len(set(c.lower() for c in color_words)))


_MOTION_KEYWORDS = re.compile(
    r"\b(ease|transition|cubic-bezier|spring|linear|bounce|elastic|fade|slide|zoom|pan|rotate|scale)\b",
    re.IGNORECASE,
)


def _count_motion_keywords(description: str) -> int:
    """Count distinct motion/animation keywords in the description."""
    matches = _MOTION_KEYWORDS.findall(description)
    return len(set(m.lower() for m in matches))


# ---------------------------------------------------------------------------
# Scene cache (in-memory + persistent)
# ---------------------------------------------------------------------------

@dataclass
class CachedScene:
    """A cached scene entry."""

    fingerprint_hash: str
    code: str
    engine: str
    created_at: float
    project_id: str  # which project generated this
    use_count: int = 0  # how many times this cache entry was reused
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_hash": self.fingerprint_hash,
            "code": self.code,
            "engine": self.engine,
            "created_at": self.created_at,
            "project_id": self.project_id,
            "use_count": self.use_count,
            "metadata": self.metadata,
        }


class SceneCache:
    """Scene-level cache with optional persistent storage.

    Item #3: in-memory LRU cache keyed by structural fingerprint.
    Item #15: optional SQLite persistence for cross-session reuse.

    Usage:
        cache = SceneCache(persist_path=".openharness/scene_cache.db")
        fingerprint = compute_fingerprint(description)
        cached = cache.get(fingerprint)
        if cached:
            return cached  # cache hit — skip LLM call
        code = await llm_generate(description)
        cache.put(fingerprint, code, project_id="my-project")
    """

    def __init__(
        self,
        *,
        persist_path: str | Path | None = None,
        max_entries: int = 500,
    ) -> None:
        self._entries: dict[str, CachedScene] = {}
        self._max_entries = max_entries
        self._persist_path = Path(persist_path) if persist_path else None
        self._hits = 0
        self._misses = 0
        self._saves = 0
        if self._persist_path:
            self._load_from_db()

    def get(self, fingerprint: SceneFingerprint) -> CachedScene | None:
        """Return the cached scene for this fingerprint, or None."""
        key = fingerprint.to_hash()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        entry.use_count += 1
        self._hits += 1
        return entry

    def put(
        self,
        fingerprint: SceneFingerprint,
        code: str,
        *,
        project_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a scene in the cache."""
        key = fingerprint.to_hash()
        entry = CachedScene(
            fingerprint_hash=key,
            code=code,
            engine=fingerprint.engine,
            created_at=time.time(),
            project_id=project_id,
            metadata=metadata or {},
        )
        self._entries[key] = entry
        self._saves += 1
        # LRU eviction
        while len(self._entries) > self._max_entries:
            # Evict the entry with the lowest use_count (LRU approximation)
            oldest_key = min(self._entries, key=lambda k: self._entries[k].use_count)
            self._entries.pop(oldest_key, None)
        if self._persist_path:
            self._save_to_db(entry)

    def invalidate(self, fingerprint: SceneFingerprint) -> None:
        """Remove a specific entry from the cache."""
        key = fingerprint.to_hash()
        self._entries.pop(key, None)
        if self._persist_path:
            self._delete_from_db(key)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        if self._persist_path and self._persist_path.exists():
            self._persist_path.unlink()

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "saves": self._saves,
            "size": len(self._entries),
            "max_size": self._max_entries,
        }

    # --- Persistence ---

    def _load_from_db(self) -> None:
        """Load cached scenes from the SQLite DB."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._persist_path))
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scene_cache (
                    fingerprint_hash TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    engine TEXT,
                    created_at REAL,
                    project_id TEXT,
                    use_count INTEGER DEFAULT 0,
                    metadata TEXT
                )"""
            )
            cursor = conn.execute(
                "SELECT fingerprint_hash, code, engine, created_at, project_id, use_count, metadata FROM scene_cache"
            )
            for row in cursor:
                key, code, engine, created_at, project_id, use_count, metadata_str = row
                metadata = json.loads(metadata_str) if metadata_str else {}
                self._entries[key] = CachedScene(
                    fingerprint_hash=key,
                    code=code,
                    engine=engine or "hyperframes",
                    created_at=created_at,
                    project_id=project_id or "",
                    use_count=use_count or 0,
                    metadata=metadata,
                )
            conn.close()
            log.debug("loaded %d scenes from %s", len(self._entries), self._persist_path)
        except Exception as exc:
            log.warning("failed to load scene cache from %s: %s", self._persist_path, exc)

    def _save_to_db(self, entry: CachedScene) -> None:
        """Save a single entry to the SQLite DB."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._persist_path))
            # Ensure the table exists (handles the case where the DB file
            # was created but the table hasn't been initialized yet).
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scene_cache (
                    fingerprint_hash TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    engine TEXT,
                    created_at REAL,
                    project_id TEXT,
                    use_count INTEGER DEFAULT 0,
                    metadata TEXT
                )"""
            )
            conn.execute(
                """INSERT OR REPLACE INTO scene_cache
                   (fingerprint_hash, code, engine, created_at, project_id, use_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.fingerprint_hash,
                    entry.code,
                    entry.engine,
                    entry.created_at,
                    entry.project_id,
                    entry.use_count,
                    json.dumps(entry.metadata),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("failed to save scene cache entry: %s", exc)

    def _delete_from_db(self, key: str) -> None:
        if not self._persist_path:
            return
        try:
            conn = sqlite3.connect(str(self._persist_path))
            conn.execute("DELETE FROM scene_cache WHERE fingerprint_hash = ?", (key,))
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("failed to delete scene cache entry: %s", exc)


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_SCENE_CACHE: SceneCache | None = None


def get_scene_cache() -> SceneCache | None:
    """Return the global scene cache, or None if not configured."""
    return _SCENE_CACHE


def configure_scene_cache(
    *,
    persist_path: str | Path | None = None,
    max_entries: int = 500,
) -> None:
    """Create / replace the global scene cache."""
    global _SCENE_CACHE
    _SCENE_CACHE = SceneCache(persist_path=persist_path, max_entries=max_entries)


def disable_scene_cache() -> None:
    """Drop the global scene cache."""
    global _SCENE_CACHE
    _SCENE_CACHE = None


def reset_scene_cache() -> None:
    """Test helper: drop the cache and reset stats."""
    global _SCENE_CACHE
    if _SCENE_CACHE is not None:
        _SCENE_CACHE.clear()
    _SCENE_CACHE = None
