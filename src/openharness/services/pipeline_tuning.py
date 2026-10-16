"""Context window per-stage + provider-specific optimizations.

Item #8: Different pipeline stages need different amounts of context.
Planning needs the full brief; scene code needs only STYLE.md and the
storyboard entry. This module sets a per-stage context budget.

Item #9: Each provider (Groq, GLM, MiniMax, Anthropic) has different
strengths. This module tunes temperature, max tokens, and system prompt
format per provider.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context budgets per stage (Item #8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    """Token budget for a pipeline stage."""

    stage: str
    max_context_tokens: int
    max_output_tokens: int
    description: str


# Default budgets per stage. These are conservative — projects can override.
STAGE_BUDGETS: dict[str, ContextBudget] = {
    "planning": ContextBudget(
        stage="planning",
        max_context_tokens=100_000,
        max_output_tokens=8_000,
        description="Planning needs the full brief + storyboard + brand context.",
    ),
    "scene_generation": ContextBudget(
        stage="scene_generation",
        max_context_tokens=20_000,
        max_output_tokens=8_000,
        description="Scene code needs STYLE.md + storyboard entry + shared asset imports.",
    ),
    "batch_scene_generation": ContextBudget(
        stage="batch_scene_generation",
        max_context_tokens=40_000,
        max_output_tokens=16_000,
        description="Batch scene gen needs all scene descriptions + style + brand.",
    ),
    "qa_review": ContextBudget(
        stage="qa_review",
        max_context_tokens=15_000,
        max_output_tokens=2_000,
        description="QA review needs the scene code + checklist only.",
    ),
    "revision": ContextBudget(
        stage="revision",
        max_context_tokens=25_000,
        max_output_tokens=8_000,
        description="Revision needs the scene code + QA feedback.",
    ),
    "narration": ContextBudget(
        stage="narration",
        max_context_tokens=10_000,
        max_output_tokens=2_000,
        description="Narration needs the storyboard + voice profile only.",
    ),
}


def get_context_budget(stage: str) -> ContextBudget:
    """Return the token budget for a pipeline stage."""
    return STAGE_BUDGETS.get(stage, STAGE_BUDGETS["scene_generation"])


def list_stages() -> list[str]:
    """Return all known pipeline stage names."""
    return list(STAGE_BUDGETS.keys())


# ---------------------------------------------------------------------------
# Provider-specific tuning (Item #9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderProfile:
    """Tuning profile for a specific LLM provider."""

    name: str
    default_temperature: float
    default_max_tokens: int
    prefers_short_prompts: bool
    prefers_chinese_instructions: bool = False
    system_prompt_prefix: str = ""
    notes: str = ""


PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "groq": ProviderProfile(
        name="groq",
        default_temperature=0.5,
        default_max_tokens=4_096,
        prefers_short_prompts=True,
        notes="Groq has low TTFT but small context (128K). Keep prompts concise.",
    ),
    "dashscope": ProviderProfile(
        name="dashscope",
        default_temperature=0.7,
        default_max_tokens=8_192,
        prefers_short_prompts=False,
        notes="DashScope (Alibaba Qwen) handles long context well.",
    ),
    "glm": ProviderProfile(
        name="glm",
        default_temperature=0.6,
        default_max_tokens=8_192,
        prefers_short_prompts=False,
        prefers_chinese_instructions=True,
        system_prompt_prefix="请用中文回答。",
        notes="GLM (Zhipu) performs better with Chinese-language system prompts.",
    ),
    "minimax": ProviderProfile(
        name="minimax",
        default_temperature=0.7,
        default_max_tokens=8_192,
        prefers_short_prompts=False,
        notes="MiniMax handles long-form content well.",
    ),
    "anthropic": ProviderProfile(
        name="anthropic",
        default_temperature=0.7,
        default_max_tokens=16_384,
        prefers_short_prompts=False,
        notes="Anthropic Claude handles long context (200K+) and complex reasoning.",
    ),
    "openai": ProviderProfile(
        name="openai",
        default_temperature=0.7,
        default_max_tokens=16_384,
        prefers_short_prompts=False,
        notes="OpenAI GPT-4o handles long context well.",
    ),
    "together": ProviderProfile(
        name="together",
        default_temperature=0.7,
        default_max_tokens=8_192,
        prefers_short_prompts=False,
        notes="Together AI hosts many open-source models with varying context windows.",
    ),
    "cerebras": ProviderProfile(
        name="cerebras",
        default_temperature=0.5,
        default_max_tokens=4_096,
        prefers_short_prompts=True,
        notes="Cerebras is very fast but has a smaller context window.",
    ),
    "default": ProviderProfile(
        name="default",
        default_temperature=0.7,
        default_max_tokens=8_192,
        prefers_short_prompts=False,
        notes="Default profile for unknown providers.",
    ),
}


def get_provider_profile(provider: str) -> ProviderProfile:
    """Return the tuning profile for a provider.

    Falls back to the "default" profile when the provider is unknown.
    """
    normalized = (provider or "").strip().lower()
    # Try exact match first
    if normalized in PROVIDER_PROFILES:
        return PROVIDER_PROFILES[normalized]
    # Try prefix matching (e.g. "openai_compat" → "openai")
    for key, profile in PROVIDER_PROFILES.items():
        if normalized.startswith(key) or key.startswith(normalized):
            return profile
    return PROVIDER_PROFILES["default"]


def list_providers() -> list[str]:
    """Return all known provider names."""
    return [k for k in PROVIDER_PROFILES if k != "default"]


# ---------------------------------------------------------------------------
# Combined tuning helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageProviderTuning:
    """Combined tuning for a specific stage + provider combination."""

    stage: str
    provider: str
    temperature: float
    max_tokens: int
    context_budget: ContextBudget
    provider_profile: ProviderProfile
    system_prompt_prefix: str = ""


def get_tuning(stage: str, provider: str) -> StageProviderTuning:
    """Get the combined tuning for a stage + provider.

    The provider profile sets the default temperature and max_tokens.
    The stage budget caps max_tokens (scene generation shouldn't
    produce 16K tokens if the budget says 8K).
    """
    budget = get_context_budget(stage)
    profile = get_provider_profile(provider)
    # Cap max_tokens at the stage budget
    max_tokens = min(profile.default_max_tokens, budget.max_output_tokens)
    return StageProviderTuning(
        stage=stage,
        provider=profile.name,
        temperature=profile.default_temperature,
        max_tokens=max_tokens,
        context_budget=budget,
        provider_profile=profile,
        system_prompt_prefix=profile.system_prompt_prefix,
    )
