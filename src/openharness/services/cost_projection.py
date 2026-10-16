"""Cost projection before start — estimate total cost before generating.

Item #10: Before a project starts, this tool estimates the total cost
based on scene count, expected token usage, and the chosen provider.
It warns if the project will exceed a budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openharness.services.pipeline_tuning import get_tuning, get_context_budget

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, in USD) — approximate as of 2025.
# Update these when providers change their pricing.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderPricing:
    """Pricing per 1M tokens for a provider."""

    name: str
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    notes: str = ""


PRICING_TABLE: dict[str, ProviderPricing] = {
    "groq": ProviderPricing("groq", 0.59, 0.79, "Groq Llama 3.3 70B pricing"),
    "dashscope": ProviderPricing("dashscope", 2.50, 7.50, "Qwen Max pricing"),
    "glm": ProviderPricing("glm", 0.50, 1.50, "GLM-4 pricing"),
    "minimax": ProviderPricing("minimax", 1.00, 3.00, "MiniMax abab pricing"),
    "anthropic": ProviderPricing("anthropic", 3.00, 15.00, "Claude Sonnet 4 pricing"),
    "openai": ProviderPricing("openai", 2.50, 10.00, "GPT-4o pricing"),
    "together": ProviderPricing("together", 0.88, 1.20, "Llama 3.3 70B on Together"),
    "cerebras": ProviderPricing("cerebras", 0.85, 1.20, "Cerebras Llama pricing"),
    "default": ProviderPricing("default", 2.00, 8.00, "Default pricing estimate"),
}


def get_pricing(provider: str) -> ProviderPricing:
    """Return the pricing for a provider (falls back to default)."""
    normalized = (provider or "").strip().lower()
    if normalized in PRICING_TABLE:
        return PRICING_TABLE[normalized]
    for key, pricing in PRICING_TABLE.items():
        if normalized.startswith(key) or key.startswith(normalized):
            return pricing
    return PRICING_TABLE["default"]


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------

@dataclass
class CostEstimate:
    """Cost estimate for one stage of the pipeline."""

    stage: str
    calls: int
    input_tokens_per_call: int
    output_tokens_per_call: int
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float


@dataclass
class CostProjection:
    """Full cost projection for a video project."""

    provider: str
    scene_count: int
    batch_size: int
    estimates: list[CostEstimate] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    budget_usd: float | None = None
    exceeds_budget: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "scene_count": self.scene_count,
            "batch_size": self.batch_size,
            "estimates": [
                {
                    "stage": e.stage,
                    "calls": e.calls,
                    "input_tokens_per_call": e.input_tokens_per_call,
                    "output_tokens_per_call": e.output_tokens_per_call,
                    "total_input_tokens": e.total_input_tokens,
                    "total_output_tokens": e.total_output_tokens,
                    "cost_usd": round(e.cost_usd, 4),
                }
                for e in self.estimates
            ],
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "budget_usd": self.budget_usd,
            "exceeds_budget": self.exceeds_budget,
            "warnings": self.warnings,
        }


def project_cost(
    *,
    scene_count: int,
    provider: str = "openai",
    batch_size: int = 8,
    budget_usd: float | None = None,
    qa_pass_rate: float = 0.85,
    revision_rate: float = 0.15,
) -> CostProjection:
    """Estimate the total cost of a video project before starting.

    Args:
        scene_count: Total number of scenes to generate.
        provider: LLM provider name (groq, dashscope, anthropic, openai, etc.).
        batch_size: Scenes per batch call (default 8).
        budget_usd: Optional budget. If set, the projection warns when exceeded.
        qa_pass_rate: Fraction of scenes that pass QA on first try (default 0.85).
        revision_rate: Fraction of scenes that need revision (default 0.15).

    Returns:
        A CostProjection with per-stage estimates and totals.
    """
    pricing = get_pricing(provider)
    projection = CostProjection(
        provider=pricing.name,
        scene_count=scene_count,
        batch_size=batch_size,
        budget_usd=budget_usd,
    )

    # --- Stage 1: Planning ---
    planning_budget = get_context_budget("planning")
    planning_calls = 1
    planning_input = planning_budget.max_context_tokens
    planning_output = planning_budget.max_output_tokens
    planning_cost = _calc_cost(pricing, planning_input, planning_output, planning_calls)
    projection.estimates.append(CostEstimate(
        stage="planning",
        calls=planning_calls,
        input_tokens_per_call=planning_input,
        output_tokens_per_call=planning_output,
        total_input_tokens=planning_input * planning_calls,
        total_output_tokens=planning_output * planning_calls,
        cost_usd=planning_cost,
    ))

    # --- Stage 2: Scene generation (batched) ---
    gen_budget = get_context_budget("batch_scene_generation")
    # Number of batch calls = ceil(scene_count / batch_size)
    batch_calls = (scene_count + batch_size - 1) // batch_size
    gen_input = gen_budget.max_context_tokens
    gen_output = gen_budget.max_output_tokens
    gen_cost = _calc_cost(pricing, gen_input, gen_output, batch_calls)
    projection.estimates.append(CostEstimate(
        stage="scene_generation",
        calls=batch_calls,
        input_tokens_per_call=gen_input,
        output_tokens_per_call=gen_output,
        total_input_tokens=gen_input * batch_calls,
        total_output_tokens=gen_output * batch_calls,
        cost_usd=gen_cost,
    ))

    # --- Stage 3: QA review (one call per scene) ---
    qa_budget = get_context_budget("qa_review")
    # Scenes that pass on first try don't need a second QA call
    qa_calls = scene_count  # every scene gets at least one QA pass
    qa_input = qa_budget.max_context_tokens
    qa_output = qa_budget.max_output_tokens
    qa_cost = _calc_cost(pricing, qa_input, qa_output, qa_calls)
    projection.estimates.append(CostEstimate(
        stage="qa_review",
        calls=qa_calls,
        input_tokens_per_call=qa_input,
        output_tokens_per_call=qa_output,
        total_input_tokens=qa_input * qa_calls,
        total_output_tokens=qa_output * qa_calls,
        cost_usd=qa_cost,
    ))

    # --- Stage 4: Revisions (for scenes that fail QA) ---
    rev_budget = get_context_budget("revision")
    rev_calls = int(scene_count * revision_rate)
    rev_input = rev_budget.max_context_tokens
    rev_output = rev_budget.max_output_tokens
    rev_cost = _calc_cost(pricing, rev_input, rev_output, rev_calls)
    projection.estimates.append(CostEstimate(
        stage="revision",
        calls=rev_calls,
        input_tokens_per_call=rev_input,
        output_tokens_per_call=rev_output,
        total_input_tokens=rev_input * rev_calls,
        total_output_tokens=rev_output * rev_calls,
        cost_usd=rev_cost,
    ))

    # --- Totals ---
    projection.total_calls = sum(e.calls for e in projection.estimates)
    projection.total_input_tokens = sum(e.total_input_tokens for e in projection.estimates)
    projection.total_output_tokens = sum(e.total_output_tokens for e in projection.estimates)
    projection.total_cost_usd = sum(e.cost_usd for e in projection.estimates)

    # --- Budget warnings ---
    if budget_usd is not None and projection.total_cost_usd > budget_usd:
        projection.exceeds_budget = True
        projection.warnings.append(
            f"Projected cost ${projection.total_cost_usd:.2f} exceeds budget ${budget_usd:.2f}. "
            f"Consider: reducing scene count, using a cheaper provider, or increasing batch_size."
        )

    # --- Heuristic warnings ---
    if batch_size > 10:
        projection.warnings.append(
            f"batch_size={batch_size} is above the API maximum (10). Will be clamped."
        )
    if qa_pass_rate < 0.7:
        projection.warnings.append(
            f"QA pass rate {qa_pass_rate:.0%} is low — expect more revision calls."
        )
    if scene_count > 100 and provider in ("anthropic",):
        projection.warnings.append(
            f"{scene_count} scenes on {provider} may be expensive. "
            f"Consider a cheaper provider for scene generation."
        )

    return projection


def _calc_cost(pricing: ProviderPricing, input_tokens: int, output_tokens: int, calls: int) -> float:
    """Calculate the cost for ``calls`` LLM calls with the given token counts."""
    total_input = input_tokens * calls
    total_output = output_tokens * calls
    return (total_input / 1_000_000) * pricing.input_per_1m + (total_output / 1_000_000) * pricing.output_per_1m
