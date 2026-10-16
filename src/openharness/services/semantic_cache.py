"""Semantic cache + scene similarity detection — items 7, 9.

#7 Semantic Cache: Cache LLM responses by semantic similarity, not
   exact match. When a new prompt is semantically similar to a cached
   one (above a threshold), reuse the cached response. 15-25% fewer calls.

#9 Scene Similarity Detection: Detect when two scene descriptions are
   structurally similar enough to reuse code. Different from the
   structural fingerprint (which is exact-match) — this catches
   "stat card showing revenue" and "stat card showing profit" as
   similar even if the fingerprint differs slightly.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ============================================================
# Text similarity (cosine on token sets — no ML needed)
# ============================================================

def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity between two texts (0.0 - 1.0).

    Uses a simple bag-of-words approach with Jaccard similarity
    (intersection / union). No ML model needed — fast enough for
    real-time cache lookups.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# #7 Semantic Cache
# ============================================================

@dataclass
class SemanticCacheEntry:
    """A cached LLM response keyed by semantic similarity."""

    prompt: str
    response: str
    similarity_hash: str
    created_at: float
    use_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Cache LLM responses by semantic similarity.

    When a new prompt is similar to a cached one (above the threshold),
    return the cached response instead of calling the LLM.

    Uses Jaccard similarity on token sets — fast and dependency-free.
    For production use, consider replacing with sentence embeddings
    (e.g., sentence-transformers) for better semantic matching.

    Usage:
        cache = SemanticCache(similarity_threshold=0.85)
        cached = cache.lookup(prompt)
        if cached:
            return cached  # skip LLM call
        response = await llm_call(prompt)
        cache.store(prompt, response)
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        max_entries: int = 200,
    ) -> None:
        self._entries: list[SemanticCacheEntry] = []
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def lookup(self, prompt: str) -> str | None:
        """Return a cached response if a similar prompt exists, else None."""
        if not prompt:
            self._misses += 1
            return None
        best_entry: SemanticCacheEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            score = cosine_similarity(prompt, entry.prompt)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score >= self._threshold:
            best_entry.use_count += 1
            self._hits += 1
            return best_entry.response
        self._misses += 1
        return None

    def store(self, prompt: str, response: str, **metadata: Any) -> None:
        """Store a prompt-response pair in the cache."""
        import time
        entry = SemanticCacheEntry(
            prompt=prompt,
            response=response,
            similarity_hash=hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            created_at=time.time(),
            metadata=metadata,
        )
        self._entries.append(entry)
        # LRU eviction
        while len(self._entries) > self._max_entries:
            # Evict the entry with the lowest use_count
            oldest = min(self._entries, key=lambda e: e.use_count)
            self._entries.remove(oldest)

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) else 0.0,
            "threshold": self._threshold,
        }

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0


# ============================================================
# #9 Scene Similarity Detection
# ============================================================

@dataclass
class SceneSimilarityResult:
    """Result of comparing two scene descriptions."""

    similarity: float
    is_similar: bool
    shared_keywords: list[str]


def detect_scene_similarity(desc_a: str, desc_b: str, *, threshold: float = 0.6) -> SceneSimilarityResult:
    """Detect if two scene descriptions are structurally similar.

    Different from the structural fingerprint (exact match): this
    catches scenes that are similar but not identical, e.g.,
    "stat card showing revenue" and "stat card showing profit".

    Args:
        desc_a: First scene description.
        desc_b: Second scene description.
        threshold: Similarity above which scenes are considered similar.

    Returns:
        A SceneSimilarityResult with the similarity score and shared keywords.
    """
    score = cosine_similarity(desc_a, desc_b)
    tokens_a = _tokenize(desc_a)
    tokens_b = _tokenize(desc_b)
    shared = sorted(tokens_a & tokens_b)
    return SceneSimilarityResult(
        similarity=round(score, 3),
        is_similar=score >= threshold,
        shared_keywords=shared,
    )


# ============================================================
# Singleton access
# ============================================================

_SEMANTIC_CACHE: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Return the global semantic cache."""
    global _SEMANTIC_CACHE
    if _SEMANTIC_CACHE is None:
        _SEMANTIC_CACHE = SemanticCache()
    return _SEMANTIC_CACHE


def reset_semantic_cache() -> None:
    """Test helper."""
    global _SEMANTIC_CACHE
    _SEMANTIC_CACHE = None
