"""Advanced optimizations — items 17, 18, 19, 20.

#17 Review Batching: Instead of reviewing scenes one at a time, batch
    them into groups of N and review each batch in a single LLM call.

#18 Failure Prediction: Predict which scenes are likely to fail review
    before generating them, so stricter prompts can be applied preemptively.

#19 Compression-Aware Batching: Adjust batch size based on prompt
    compression ratio. More compressible → larger batches.

#20 Cross-Provider Racing: Send the same request to multiple providers
    simultaneously and use the first response. 30-50% lower latency.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# ============================================================
# #17 Review Batching
# ============================================================

@dataclass
class ReviewBatch:
    """A batch of scenes to review in a single LLM call."""

    batch_id: str
    scene_ids: list[str]
    scene_codes: list[str]
    review_depth: str = "standard"


def create_review_batches(
    scene_ids: list[str],
    scene_codes: list[str],
    *,
    batch_size: int = 5,
    review_depth: str = "standard",
) -> list[ReviewBatch]:
    """Split scenes into review batches.

    Instead of 1 LLM call per scene, this creates batches of N scenes
    so each batch is reviewed in a single call. For 50 scenes with
    batch_size=5, that's 10 review calls instead of 50 — a 5× reduction.

    Args:
        scene_ids: List of scene IDs.
        scene_codes: List of scene code strings (same order as scene_ids).
        batch_size: Number of scenes per review batch (default 5).
        review_depth: Review depth for all batches in this set.

    Returns:
        A list of ReviewBatch objects.
    """
    batches: list[ReviewBatch] = []
    for i in range(0, len(scene_ids), batch_size):
        batch_ids = scene_ids[i : i + batch_size]
        batch_codes = scene_codes[i : i + batch_size]
        batches.append(ReviewBatch(
            batch_id=f"review_batch_{i // batch_size + 1:03d}",
            scene_ids=batch_ids,
            scene_codes=batch_codes,
            review_depth=review_depth,
        ))
    return batches


# ============================================================
# #18 Failure Prediction
# ============================================================

@dataclass
class FailurePrediction:
    """Prediction of whether a scene will fail review."""

    scene_type: str
    failure_probability: float  # 0.0 - 1.0
    predicted_failures: list[str]  # likely failure reasons
    recommended_strictness: int  # 1-3


def predict_failure_risk(
    scene_description: str,
    *,
    historical_failure_rate: dict[str, float] | None = None,
) -> FailurePrediction:
    """Predict the risk of a scene failing review.

    Uses historical failure rates by scene type (from the failure
    tracker) + heuristics on the description itself.

    Args:
        scene_description: The scene's description text.
        historical_failure_rate: Dict mapping scene_type → failure_rate.
            If None, uses heuristics only.

    Returns:
        A FailurePrediction with the risk assessment.
    """
    historical = historical_failure_rate or {}
    # Detect scene type from description
    lowered = scene_description.lower()
    scene_type = "generic"
    if any(kw in lowered for kw in ["title", "card", "intro"]):
        scene_type = "title_card"
    elif any(kw in lowered for kw in ["chart", "graph", "data", "viz", "stat"]):
        scene_type = "data_viz"
    elif any(kw in lowered for kw in ["blender", "3d", "manim", "three"]):
        scene_type = "3d_animation"
    elif any(kw in lowered for kw in ["transition", "fade", "slide"]):
        scene_type = "transition"

    # Base failure rate from history (or heuristic default)
    base_rate = historical.get(scene_type, 0.15)  # default 15% failure rate

    # Adjust based on description complexity
    predicted_failures: list[str] = []
    if len(scene_description) > 500:
        base_rate *= 1.2
        predicted_failures.append("long_description_complexity")
    if any(kw in lowered for kw in ["physics", "particle", "simulation"]):
        base_rate *= 1.3
        predicted_failures.append("determinism_risk")
    if any(kw in lowered for kw in ["chart", "graph", "data"]):
        base_rate *= 1.15
        predicted_failures.append("data_accuracy_risk")

    probability = min(1.0, base_rate)

    if probability >= 0.5:
        strictness = 3
    elif probability >= 0.3:
        strictness = 2
    else:
        strictness = 1

    return FailurePrediction(
        scene_type=scene_type,
        failure_probability=round(probability, 3),
        predicted_failures=predicted_failures,
        recommended_strictness=strictness,
    )


# ============================================================
# #19 Compression-Aware Batching
# ============================================================

def compute_compression_aware_batch_size(
    base_batch_size: int,
    compression_ratio: float,
    *,
    min_size: int = 2,
    max_size: int = 10,
) -> int:
    """Adjust batch size based on prompt compression ratio.

    If the prompt compresses well (low ratio), we can fit more scenes
    into a single batch because the compressed prompt stays within the
    context window. If it doesn't compress, we use a smaller batch.

    Args:
        base_batch_size: The base batch size (e.g., from settings).
        compression_ratio: The ratio of compressed/original prompt size.
            Lower = more compressible.
        min_size: Minimum batch size.
        max_size: Maximum batch size.

    Returns:
        The adjusted batch size.
    """
    if compression_ratio <= 0:
        return min_size
    # If compression ratio is 0.5 (50% reduction), we can roughly double the batch
    adjustment_factor = 1.0 / max(0.3, compression_ratio)
    adjusted = int(base_batch_size * adjustment_factor)
    return max(min_size, min(max_size, adjusted))


# ============================================================
# #20 Cross-Provider Racing
# ============================================================

@dataclass
class RaceResult:
    """Result of a cross-provider race."""

    winner: str  # provider name
    response: Any
    latency_seconds: float
    losers: list[str] = field(default_factory=list)  # providers that were beaten


async def race_providers(
    request: Any,
    providers: dict[str, Callable[[Any], Awaitable[Any]]],
    *,
    timeout: float = 30.0,
) -> RaceResult:
    """Send the same request to multiple providers and use the first response.

    This is the "hedged request" pattern: fire the request at N providers
    simultaneously, return the first response, cancel the rest.

    Args:
        request: The request to send.
        providers: Dict mapping provider_name → async callable that takes
            the request and returns the response.
        timeout: Maximum time to wait for any provider.

    Returns:
        A RaceResult with the winning provider's response.

    Raises:
        TimeoutError: If no provider responds within the timeout.
    """
    if not providers:
        raise ValueError("At least one provider is required")

    import time
    start = time.monotonic()

    async def _try_provider(name: str, fn: Callable) -> tuple[str, Any]:
        try:
            result = await asyncio.wait_for(fn(request), timeout=timeout)
            return name, result
        except Exception as exc:
            log.debug("provider %s lost the race: %s", name, exc)
            raise

    tasks = {
        name: asyncio.create_task(_try_provider(name, fn))
        for name, fn in providers.items()
    }

    winner_name: str | None = None
    winner_response: Any = None
    losers: list[str] = []

    try:
        # Wait for the first task to complete
        done, pending = await asyncio.wait(
            tasks.values(),
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )
        for task in done:
            try:
                name, response = task.result()
                winner_name = name
                winner_response = response
                break
            except Exception:
                pass
        # Cancel all pending tasks
        for task in pending:
            task.cancel()
        # Collect losers
        for name, task in tasks.items():
            if name != winner_name and task.done():
                try:
                    task.result()
                except Exception:
                    losers.append(name)
            elif name != winner_name:
                losers.append(name)
    except asyncio.TimeoutError:
        raise TimeoutError(f"No provider responded within {timeout}s")

    if winner_name is None:
        raise TimeoutError(f"No provider responded within {timeout}s")

    latency = time.monotonic() - start
    return RaceResult(
        winner=winner_name,
        response=winner_response,
        latency_seconds=latency,
        losers=losers,
    )
