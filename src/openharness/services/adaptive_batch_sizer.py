"""Adaptive batch sizing — Item #13.

Dynamically adjust batch size based on the model's current throughput.
If the model is fast, batch 10; if slow, batch 5. Uses a sliding
window of recent response times to estimate throughput.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class ThroughputSample:
    """A single throughput measurement."""

    timestamp: float
    batch_size: int
    response_time_seconds: float
    tokens_generated: int


class AdaptiveBatchSizer:
    """Adjusts batch_size based on observed model throughput.

    Tracks the last N render times and adjusts the batch size to
    target a specific response latency. If responses are fast, the
    batch size increases; if slow, it decreases.

    Usage:
        sizer = AdaptiveBatchSizer(min_size=2, max_size=10, target_latency=30.0)
        batch_size = sizer.recommend_batch_size()
        # ... call LLM with batch_size ...
        sizer.record(batch_size, response_time, tokens_generated)
        # Next call will use the adjusted size.
    """

    def __init__(
        self,
        *,
        min_size: int = 2,
        max_size: int = 10,
        target_latency: float = 30.0,
        window_size: int = 5,
    ) -> None:
        self._min_size = min_size
        self._max_size = max_size
        self._target_latency = target_latency
        self._window_size = window_size
        self._samples: Deque[ThroughputSample] = deque(maxlen=window_size)
        self._current_size = (min_size + max_size) // 2  # start in the middle

    @property
    def current_size(self) -> int:
        """The current recommended batch size."""
        return self._current_size

    @property
    def target_latency(self) -> float:
        return self._target_latency

    def record(
        self,
        batch_size: int,
        response_time_seconds: float,
        tokens_generated: int = 0,
    ) -> None:
        """Record a completed batch's performance."""
        self._samples.append(
            ThroughputSample(
                timestamp=time.time(),
                batch_size=batch_size,
                response_time_seconds=response_time_seconds,
                tokens_generated=tokens_generated,
            )
        )
        self._adjust_size()

    def recommend_batch_size(self) -> int:
        """Return the recommended batch size for the next call."""
        return self._current_size

    def _adjust_size(self) -> None:
        """Adjust the batch size based on recent performance."""
        if len(self._samples) < 2:
            return  # need at least 2 samples to detect a trend
        # Compute average latency per scene
        recent = list(self._samples)
        total_scenes = sum(s.batch_size for s in recent)
        total_time = sum(s.response_time_seconds for s in recent)
        if total_scenes == 0:
            return
        avg_latency_per_scene = total_time / total_scenes
        # Estimate: if we use current_size, latency ≈ current_size * avg_latency_per_scene
        estimated_latency = self._current_size * avg_latency_per_scene
        if estimated_latency < self._target_latency * 0.7:
            # Fast — increase batch size
            self._current_size = min(self._max_size, self._current_size + 1)
        elif estimated_latency > self._target_latency * 1.3:
            # Slow — decrease batch size
            self._current_size = max(self._min_size, self._current_size - 1)
        # Else: keep current size

    def stats(self) -> dict[str, float | int]:
        """Return throughput statistics."""
        if not self._samples:
            return {
                "current_size": self._current_size,
                "samples": 0,
                "avg_latency_per_scene": 0.0,
            }
        recent = list(self._samples)
        total_scenes = sum(s.batch_size for s in recent)
        total_time = sum(s.response_time_seconds for s in recent)
        return {
            "current_size": self._current_size,
            "samples": len(recent),
            "avg_latency_per_scene": total_time / total_scenes if total_scenes else 0.0,
            "target_latency": self._target_latency,
        }
