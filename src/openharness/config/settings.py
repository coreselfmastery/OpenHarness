"""Settings model and loading logic for OpenHarness.

Settings are resolved with the following precedence (highest first):
1. CLI arguments
2. Environment variables (ANTHROPIC_API_KEY, OPENHARNESS_MODEL, etc.)
3. Config file (~/.openharness/settings.json)
4. Defaults
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.hooks.schemas import HookDefinition
from openharness.mcp.types import McpServerConfig
from openharness.permissions.modes import PermissionMode
from openharness.utils.file_lock import exclusive_file_lock
from openharness.utils.fs import atomic_write_text

# ANSI escape sequence pattern
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi_escape_sequences(text: str) -> str:
    """Remove ANSI escape sequences from text.

    This is used to clean environment variables that may contain terminal
    formatting codes (e.g., '[1m' for bold) which can corrupt API requests.
    """
    if not text:
        return text
    return _ANSI_ESCAPE_PATTERN.sub("", text)


class PathRuleConfig(BaseModel):
    """A glob-pattern path permission rule."""

    pattern: str
    allow: bool = True


class PermissionSettings(BaseModel):
    """Permission mode configuration."""

    mode: PermissionMode = PermissionMode.DEFAULT
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    path_rules: list[PathRuleConfig] = Field(default_factory=list)
    denied_commands: list[str] = Field(default_factory=list)


class MemorySettings(BaseModel):
    """Memory system configuration."""

    enabled: bool = True
    max_files: int = 5
    max_entrypoint_lines: int = 200
    max_entrypoint_bytes: int = 25_000
    context_window_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None
    auto_extract_enabled: bool = False
    auto_extract_max_records: int = 3
    session_memory_enabled: bool = True
    auto_dream_enabled: bool = False
    auto_dream_min_hours: float = 24.0
    auto_dream_min_sessions: int = 5


class SandboxNetworkSettings(BaseModel):
    """OS-level network restrictions passed to sandbox-runtime."""

    allowed_domains: list[str] = Field(default_factory=list)
    denied_domains: list[str] = Field(default_factory=list)


class SandboxFilesystemSettings(BaseModel):
    """OS-level filesystem restrictions passed to sandbox-runtime."""

    allow_read: list[str] = Field(default_factory=list)
    deny_read: list[str] = Field(default_factory=list)
    allow_write: list[str] = Field(default_factory=lambda: ["."])
    deny_write: list[str] = Field(default_factory=list)


class DockerSandboxSettings(BaseModel):
    """Docker-specific sandbox configuration."""

    image: str = "openharness-sandbox:latest"
    auto_build_image: bool = True
    cpu_limit: float = 0.0
    memory_limit: str = ""
    extra_mounts: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)


class SandboxSettings(BaseModel):
    """Sandbox-runtime integration settings."""

    enabled: bool = False
    backend: str = "srt"
    fail_if_unavailable: bool = False
    enabled_platforms: list[str] = Field(default_factory=list)
    network: SandboxNetworkSettings = Field(default_factory=SandboxNetworkSettings)
    filesystem: SandboxFilesystemSettings = Field(default_factory=SandboxFilesystemSettings)
    docker: DockerSandboxSettings = Field(default_factory=DockerSandboxSettings)


class WebSettings(BaseModel):
    """Outbound web tool configuration."""

    proxy: str | None = None
    resolution_mode: str = "auto"
    synthetic_dns_cidrs: list[str] = Field(default_factory=list)


class ProviderProfile(BaseModel):
    """Named provider workflow configuration."""

    label: str
    provider: str
    api_format: str
    auth_source: str
    default_model: str
    base_url: str | None = None
    last_model: str | None = None
    credential_slot: str | None = None
    allowed_models: list[str] = Field(default_factory=list)
    context_window_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None

    @property
    def resolved_model(self) -> str:
        """Return the active model for this profile."""
        return resolve_model_setting(
            (self.last_model or "").strip() or self.default_model,
            self.provider,
            default_model=self.default_model,
        )


@dataclass(frozen=True)
class ResolvedAuth:
    """Normalized auth material used to construct API clients."""

    provider: str
    auth_kind: str
    value: str
    source: str
    state: str = "configured"


CLAUDE_MODEL_ALIAS_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("default", "Default", "Recommended model for this profile"),
    ("best", "Best", "Most capable available model"),
    ("sonnet", "Sonnet", "Latest Sonnet for everyday coding"),
    ("opus", "Opus", "Latest Opus for complex reasoning"),
    ("haiku", "Haiku", "Fastest Claude model"),
    ("sonnet[1m]", "Sonnet (1M context)", "Latest Sonnet with 1M context"),
    ("opus[1m]", "Opus (1M context)", "Latest Opus with 1M context"),
    ("opusplan", "Opus Plan Mode", "Use Opus in plan mode and Sonnet otherwise"),
)

_CLAUDE_ALIAS_TARGETS: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
    "sonnet[1m]": "claude-sonnet-4-6[1m]",
    "opus[1m]": "claude-opus-4-6[1m]",
}


def normalize_anthropic_model_name(model: str) -> str:
    """Normalize an Anthropic model name the same way Hermes does.

    - Strips the ``anthropic/`` prefix when present.
    - Converts dotted Claude version separators to Anthropic's hyphenated form.
    """
    normalized = model.strip()
    lower = normalized.lower()
    if lower.startswith("anthropic/"):
        normalized = normalized[len("anthropic/"):]
        lower = normalized.lower()
    if lower.startswith("claude-"):
        return normalized.replace(".", "-")
    return normalized


def default_provider_profiles() -> dict[str, ProviderProfile]:
    """Return the built-in provider workflow catalog."""
    return {
        "claude-api": ProviderProfile(
            label="Anthropic-Compatible API",
            provider="anthropic",
            api_format="anthropic",
            auth_source="anthropic_api_key",
            default_model="claude-sonnet-4-6",
        ),
        "claude-subscription": ProviderProfile(
            label="Claude Subscription",
            provider="anthropic_claude",
            api_format="anthropic",
            auth_source="claude_subscription",
            default_model="claude-sonnet-4-6",
        ),
        "openai-compatible": ProviderProfile(
            label="OpenAI-Compatible API",
            provider="openai",
            api_format="openai",
            auth_source="openai_api_key",
            default_model="gpt-5.4",
        ),
        "codex": ProviderProfile(
            label="Codex Subscription",
            provider="openai_codex",
            api_format="openai",
            auth_source="codex_subscription",
            default_model="gpt-5.4",
        ),
        "copilot": ProviderProfile(
            label="GitHub Copilot",
            provider="copilot",
            api_format="copilot",
            auth_source="copilot_oauth",
            default_model="gpt-5.4",
        ),
        "moonshot": ProviderProfile(
            label="Moonshot (Kimi)",
            provider="moonshot",
            api_format="openai",
            auth_source="moonshot_api_key",
            default_model="kimi-k2.5",
            base_url="https://api.moonshot.cn/v1",
        ),
        "gemini": ProviderProfile(
            label="Google Gemini",
            provider="gemini",
            api_format="openai",
            auth_source="gemini_api_key",
            default_model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        "minimax": ProviderProfile(
            label="MiniMax",
            provider="minimax",
            api_format="openai",
            auth_source="minimax_api_key",
            default_model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
        ),
        "nvidia": ProviderProfile(
            label="NVIDIA NIM",
            provider="nvidia",
            api_format="openai",
            auth_source="nvidia_api_key",
            default_model="openai/gpt-oss-120b",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
        "qwen": ProviderProfile(
            label="Qwen (DashScope)",
            provider="dashscope",
            api_format="openai",
            auth_source="dashscope_api_key",
            default_model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "modelscope": ProviderProfile(
            label="ModelScope",
            provider="modelscope",
            api_format="openai",
            auth_source="modelscope_api_key",
            default_model="deepseek-ai/DeepSeek-V4-Flash",
            base_url="https://api-inference.modelscope.cn/v1",
        ),
    }


def builtin_provider_profile_names() -> set[str]:
    """Return the names of built-in provider profiles."""
    return set(default_provider_profiles())


def display_label_for_profile(profile_name: str, profile: ProviderProfile) -> str:
    """Return the user-facing label for a profile.

    Built-in profiles always use the current built-in catalog label so old
    persisted settings don't keep stale wording in menus.
    """
    builtin = default_provider_profiles().get(profile_name)
    if builtin is not None:
        return builtin.label
    return profile.label


def is_claude_family_provider(provider: str) -> bool:
    """Return True when the provider is a Claude/Anthropic workflow."""
    return provider in {"anthropic", "anthropic_claude"}


def display_model_setting(profile: ProviderProfile) -> str:
    """Return the user-facing model setting for a profile."""
    configured = (profile.last_model or "").strip()
    if not configured and is_claude_family_provider(profile.provider):
        return "default"
    return configured or profile.default_model


def resolve_model_setting(
    model_setting: str,
    provider: str,
    *,
    default_model: str | None = None,
    permission_mode: str | None = None,
) -> str:
    """Resolve a user-facing model setting into the concrete runtime model ID."""
    configured = model_setting.strip()
    normalized = configured.lower()

    if not configured or normalized == "default":
        fallback = (default_model or "").strip()
        if fallback and fallback.lower() != "default":
            return resolve_model_setting(
                fallback,
                provider,
                default_model=None,
                permission_mode=permission_mode,
            )
        if is_claude_family_provider(provider):
            return _CLAUDE_ALIAS_TARGETS["sonnet"]
        return "gpt-5.4"

    if is_claude_family_provider(provider):
        if normalized == "best":
            return _CLAUDE_ALIAS_TARGETS["opus"]
        if normalized == "opusplan":
            if permission_mode == PermissionMode.PLAN.value:
                return _CLAUDE_ALIAS_TARGETS["opus"]
            return _CLAUDE_ALIAS_TARGETS["sonnet"]
        if normalized in _CLAUDE_ALIAS_TARGETS:
            return _CLAUDE_ALIAS_TARGETS[normalized]
        return normalize_anthropic_model_name(configured)

    if provider in {"openai", "openai_codex", "copilot"} and normalized in {"default", "best"}:
        return "gpt-5.4"

    return configured


def auth_source_provider_name(auth_source: str) -> str:
    """Map an auth source to the storage/runtime provider name."""
    mapping = {
        "anthropic_api_key": "anthropic",
        "openai_api_key": "openai",
        "codex_subscription": "openai_codex",
        "claude_subscription": "anthropic_claude",
        "copilot_oauth": "copilot",
        "dashscope_api_key": "dashscope",
        "bedrock_api_key": "bedrock",
        "vertex_api_key": "vertex",
        "moonshot_api_key": "moonshot",
        "gemini_api_key": "gemini",
        "minimax_api_key": "minimax",
        "nvidia_api_key": "nvidia",
        "modelscope_api_key": "modelscope",
    }
    return mapping.get(auth_source, auth_source)


def auth_source_uses_api_key(auth_source: str) -> bool:
    """Return True when the auth source is backed by a user-supplied API key."""
    return auth_source.endswith("_api_key")


def auth_source_env_var_candidates(auth_source: str) -> tuple[str, ...]:
    """Return env vars to probe for an auth source in precedence order."""
    mapping = {
        "anthropic_api_key": ("OPENHARNESS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        "openai_api_key": ("OPENHARNESS_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "dashscope_api_key": ("OPENHARNESS_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY"),
        "moonshot_api_key": ("OPENHARNESS_MOONSHOT_API_KEY", "MOONSHOT_API_KEY"),
        "gemini_api_key": ("OPENHARNESS_GEMINI_API_KEY", "GEMINI_API_KEY"),
        "minimax_api_key": ("OPENHARNESS_MINIMAX_API_KEY", "MINIMAX_API_KEY"),
        "nvidia_api_key": ("OPENHARNESS_NVIDIA_API_KEY", "NVIDIA_API_KEY"),
        "modelscope_api_key": ("OPENHARNESS_MODELSCOPE_API_KEY", "MODELSCOPE_API_KEY"),
    }
    return mapping.get(auth_source, ())


def resolve_auth_env_value(auth_source: str) -> tuple[str, str] | None:
    """Return the first configured env var/value pair for an auth source."""
    for env_var in auth_source_env_var_candidates(auth_source):
        env_value = os.environ.get(env_var, "")
        if env_value:
            return env_var, env_value
    return None


def credential_storage_provider_name(profile_name: str, profile: ProviderProfile) -> str:
    """Return the storage namespace used for this profile's credential.

    Built-in API-key flows continue to use provider-level storage by default.
    Custom compatible profiles can set ``credential_slot`` to bind their own key.
    """
    del profile_name
    if auth_source_uses_api_key(profile.auth_source) and profile.credential_slot:
        return f"profile:{profile.credential_slot}"
    return auth_source_provider_name(profile.auth_source)


def default_auth_source_for_provider(provider: str, api_format: str | None = None) -> str:
    """Infer the default auth source for a provider/backend."""
    if provider == "anthropic_claude":
        return "claude_subscription"
    if provider == "openai_codex":
        return "codex_subscription"
    if provider == "copilot":
        return "copilot_oauth"
    if provider == "dashscope":
        return "dashscope_api_key"
    if provider == "bedrock":
        return "bedrock_api_key"
    if provider == "vertex":
        return "vertex_api_key"
    if provider == "moonshot":
        return "moonshot_api_key"
    if provider == "gemini":
        return "gemini_api_key"
    if provider == "minimax":
        return "minimax_api_key"
    if provider == "nvidia":
        return "nvidia_api_key"
    if provider == "modelscope":
        return "modelscope_api_key"
    if provider == "openai" or api_format == "openai":
        return "openai_api_key"
    return "anthropic_api_key"


def _slugify_profile_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "custom"


def _infer_profile_name_from_flat_settings(settings: Settings) -> str:
    provider = (settings.provider or "").strip()
    if provider == "openai_codex":
        return "codex"
    if provider == "anthropic_claude":
        return "claude-subscription"
    if provider == "copilot" or settings.api_format == "copilot":
        return "copilot"
    if provider == "openai" and not settings.base_url:
        return "openai-compatible"
    if provider == "anthropic" and not settings.base_url:
        return "claude-api"
    if settings.base_url:
        return _slugify_profile_name(Path(settings.base_url).name or settings.base_url)
    if provider:
        return _slugify_profile_name(provider)
    return "claude-api"


def _profile_from_flat_settings(settings: Settings) -> tuple[str, ProviderProfile]:
    defaults = default_provider_profiles()
    name = _infer_profile_name_from_flat_settings(settings)
    existing = defaults.get(name)
    if existing is not None and (
        existing.provider == settings.provider or not settings.provider
    ) and (
        existing.api_format == settings.api_format
    ) and (
        existing.base_url == settings.base_url
    ):
        profile = existing.model_copy(
            update={
                "last_model": settings.model or existing.resolved_model,
            }
        )
        return name, profile

    provider = settings.provider or ("copilot" if settings.api_format == "copilot" else ("openai" if settings.api_format == "openai" else "anthropic"))
    profile = ProviderProfile(
        label=f"Imported {provider}",
        provider=provider,
        api_format=settings.api_format,
        auth_source=default_auth_source_for_provider(provider, settings.api_format),
        default_model=settings.model or defaults.get("claude-api", ProviderProfile(
            label="Claude API",
            provider="anthropic",
            api_format="anthropic",
            auth_source="anthropic_api_key",
            default_model="sonnet",
        )).default_model,
        last_model=settings.model or None,
        base_url=settings.base_url,
    )
    return name, profile


class ImageGenerationConfig(BaseModel):
    """Configuration for the image_generation tool."""

    provider: str = "auto"
    model: str = "gpt-image-2"
    api_key: str = ""
    base_url: str = ""
    codex_model: str = "gpt-5.4"
    codex_base_url: str = ""

    @classmethod
    def from_env(cls) -> ImageGenerationConfig:
        """Load image generation config from environment variables."""
        return cls(
            provider=os.environ.get("OPENHARNESS_IMAGE_GENERATION_PROVIDER", "auto").strip()
            or "auto",
            model=os.environ.get("OPENHARNESS_IMAGE_GENERATION_MODEL", "gpt-image-2").strip()
            or "gpt-image-2",
            api_key=os.environ.get("OPENHARNESS_IMAGE_GENERATION_API_KEY", "").strip(),
            base_url=os.environ.get("OPENHARNESS_IMAGE_GENERATION_BASE_URL", "").strip(),
            codex_model=os.environ.get("OPENHARNESS_IMAGE_GENERATION_CODEX_MODEL", "gpt-5.4").strip()
            or "gpt-5.4",
            codex_base_url=os.environ.get("OPENHARNESS_IMAGE_GENERATION_CODEX_BASE_URL", "").strip(),
        )

    @property
    def is_configured(self) -> bool:
        """Return True when either a key provider or Codex provider is selected."""
        return bool(self.api_key or self.provider in {"auto", "codex"})


class SceneGenConfig(BaseModel):
    """Configuration for the batch_scene tool (scene code generation).

    Separate from ImageGenerationConfig because scene generation uses
    text models (gpt-4o, claude-sonnet) not image models, and may use
    a different provider / endpoint for cost reasons.
    """

    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 16384
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> SceneGenConfig:
        """Load scene gen config from environment variables."""
        return cls(
            model=os.environ.get("OPENHARNESS_SCENE_GEN_MODEL", "gpt-4o").strip() or "gpt-4o",
            api_key=os.environ.get("OPENHARNESS_SCENE_GEN_API_KEY", "").strip(),
            base_url=os.environ.get("OPENHARNESS_SCENE_GEN_BASE_URL", "").strip(),
            max_tokens=int(os.environ.get("OPENHARNESS_SCENE_GEN_MAX_TOKENS", "16384") or 16384),
            temperature=float(os.environ.get("OPENHARNESS_SCENE_GEN_TEMPERATURE", "0.7") or 0.7),
        )

    @property
    def is_configured(self) -> bool:
        """Return True when an API key is available."""
        return bool(self.api_key)


class VisionModelConfig(BaseModel):
    """Configuration for the vision model used by the image_to_text tool.

    When the active model does not support multimodal input, the agent loop
    automatically falls back to this vision model to describe images.
    """

    model: str = ""
    api_key: str = ""
    base_url: str = ""

    @classmethod
    def from_env(cls) -> VisionModelConfig:
        """Load vision model config from environment variables."""
        return cls(
            model=os.environ.get("OPENHARNESS_VISION_MODEL", "").strip(),
            api_key=os.environ.get("OPENHARNESS_VISION_API_KEY", "").strip(),
            base_url=os.environ.get("OPENHARNESS_VISION_BASE_URL", "").strip(),
        )

    @property
    def is_configured(self) -> bool:
        """Return True when both model and api_key are set."""
        return bool(self.model and self.api_key)


class MiniCPMVLMConfig(BaseModel):
    """Local MiniCPM-V 4.5 vision-language model (8B, 4-bit quantized).

    MiniCPM-V 4.5 is an 8B-parameter multimodal model whose 4-bit
    quantization fits in 6-8 GB of VRAM — small enough for a single
    consumer GPU, a Kaggle T4, or a laptop — served through any
    OpenAI-compatible server (vLLM, llama.cpp's llama-server, Ollama).
    The ``minicpm_vlm`` tool uses this endpoint for frame/image
    understanding without touching the primary chat model's quota.

    Defaults point at a local vLLM instance; override via settings or
    the ``OPENHARNESS_MINICPM_*`` environment variables.
    """

    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "openbmb/MiniCPM-V-4_5"
    api_key: str = "EMPTY"  # local servers accept any non-empty key
    timeout_seconds: float = 180.0
    # Informational: the quantization the endpoint is expected to serve.
    quantization: str = "4-bit"
    # Approximate VRAM footprint for capacity planning / docs.
    vram_gb: str = "6-8"

    @classmethod
    def from_env(cls) -> MiniCPMVLMConfig:
        """Load MiniCPM-V config from environment variables, keeping the
        class defaults for anything unset."""
        return cls(
            base_url=os.environ.get("OPENHARNESS_MINICPM_BASE_URL", "").strip()
            or cls.model_fields["base_url"].default,
            model=os.environ.get("OPENHARNESS_MINICPM_MODEL", "").strip()
            or cls.model_fields["model"].default,
            api_key=os.environ.get("OPENHARNESS_MINICPM_API_KEY", "").strip() or "EMPTY",
            timeout_seconds=float(
                os.environ.get("OPENHARNESS_MINICPM_TIMEOUT", "0").strip() or 0
            )
            or cls.model_fields["timeout_seconds"].default,
        )

    @property
    def is_configured(self) -> bool:
        """Always True — the defaults target a local endpoint."""
        return bool(self.base_url and self.model)


class EmbeddingConfig(BaseModel):
    """Embedding endpoint for the semantic-memory store.

    Any OpenAI-compatible ``/embeddings`` server works — BGE-M3
    (``BAAI/bge-m3``, 768 dims), MiniLM
    (``sentence-transformers/all-MiniLM-L6-v2``, 384 dims), or an
    ``ollama serve`` gateway. When ``base_url`` is empty the semantic
    memory falls back to a deterministic local hashing embedder
    (lexical-approximate recall, zero dependencies) so the store
    always works.
    """

    base_url: str = ""  # empty -> local hashing fallback
    model: str = "BAAI/bge-m3"
    api_key: str = "EMPTY"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Load embedding config from environment variables, keeping the
        class defaults for anything unset."""
        return cls(
            base_url=os.environ.get("OPENHARNESS_EMBEDDING_BASE_URL", "").strip(),
            model=os.environ.get("OPENHARNESS_EMBEDDING_MODEL", "").strip()
            or cls.model_fields["model"].default,
            api_key=os.environ.get("OPENHARNESS_EMBEDDING_API_KEY", "").strip()
            or "EMPTY",
            timeout_seconds=float(
                os.environ.get("OPENHARNESS_EMBEDDING_TIMEOUT", "0").strip() or 0
            )
            or cls.model_fields["timeout_seconds"].default,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


class InferenceSettings(BaseModel):
    """Category 1: LLM inference speed features (items #1-#6, #18).

    All features are opt-in except prefix caching, which is a pure win
    when the provider supports it (Anthropic prompt caching) and a no-op
    elsewhere.
    """

    # #2 — add Anthropic ``cache_control`` breakpoints on the system
    # prompt + tool list so repeated prefill hits the provider cache
    # (30-50% cheaper/faster prefill on Anthropic; automatic and free on
    # OpenAI/Groq so the flag is harmless there).
    prefix_caching: bool = True
    # #3 — strip redundant content (old tool-output bodies, repeated
    # whitespace) from requests above ``min_chars``.
    prompt_compression: bool = False
    prompt_compression_min_chars: int = 6000
    prompt_compression_keep_turns: int = 3
    # #18 — semantic (MinHash/Jaccard) response cache with SQLite persistence.
    semantic_cache: bool = False
    semantic_cache_threshold: float = 0.88
    semantic_cache_max_entries: int = 512
    # #4 — race 2-3 providers concurrently, keep the first response.
    # Doubles token cost on raced calls, so it stays opt-in.
    cross_provider_racing: bool = False
    # #6 — rotate a pool of API keys on rate limits (env:
    # OPENHARNESS_API_KEY_POOL, comma-separated; Groq: OPENHARNESS_GROQ_API_KEYS).
    api_key_pool: list[str] = Field(default_factory=list)
    # Cooldown applied to a key after it hits a rate limit: the key is
    # skipped until the cooldown expires, then it is eligible again
    # (default 60s — matches typical provider RPM windows).
    key_pool_cooldown_seconds: float = 60.0
    # Bounded-concurrency fan-out for batch LLM work (the async batch
    # executor): how many requests may be in flight at once. 8 balances
    # throughput against typical provider concurrency limits.
    batch_concurrency: int = 8
    # #1/#5 — speculative decoding / multi-token prediction are model
    # server features (Groq does speculation natively; vLLM needs server
    # flags — see the ``server_config`` tool). When True, the client
    # probes the endpoint and reports capability honestly instead of
    # pretending to accelerate anything.
    speculative_reporting: bool = True


class RenderSettings(BaseModel):
    """Category 3+5: rendering speed and pipeline overlap (items #11-#16, #21, #24-#25, #40)."""

    # #11 — parallel scene rendering across N workers.
    parallel_rendering: bool = True
    # None = auto from GPU count (kaggle_env), 1 when no GPU is present.
    render_workers: int | None = None
    # #12/#16 — render a cheap draft first, QA it, only render the
    # full-quality version for approved scenes.
    draft_first: bool = True
    resolution_tiers: dict[str, str] = Field(
        default_factory=lambda: {"draft": "640x360", "review": "1280x720", "final": "1920x1080"}
    )
    # #13 — promote a passing draft (skip the full render) when its
    # quality gate score clears this threshold (0.0-1.0).
    quality_early_exit: bool = True
    early_exit_score_threshold: float = 0.9
    # #14/#17 — fingerprint-based render cache (scene code + assets +
    # engine args + tier). Persisted across sessions.
    render_cache: bool = True
    # #15 — prefer NVENC (h264_nvenc) when a supported GPU is present,
    # verified with a real probe encode; fall back to libx264.
    nvenc: bool = True
    # #21 — batch_scene auto-enqueues renders while generation continues.
    auto_enqueue_renders: bool = False
    # #25 — honor ``depends_on`` scene ordering; render independent
    # scenes in parallel waves.
    dependency_graph: bool = True
    # #24 — idle render workers steal queued work from busy workers.
    work_stealing: bool = True
    # Per-render-job hard timeout (#27 render stage).
    render_timeout_seconds: int = 1800


class QASettings(BaseModel):
    """Category 2: parallel QA (items #7-#10)."""

    # #7 — review scenes with parallel QA calls (wave size caps fan-out).
    parallel_waves: bool = True
    wave_size: int = 5
    # #8 — batch several scenes per QA call when they are simple.
    review_batching: bool = True
    review_batch_size: int = 5
    # #9 — lightweight checklist for simple scenes, deep review for complex.
    smart_depth: bool = True
    # #10 — predict likely failure categories from persisted stats and
    # pre-emptively check them.
    failure_prediction: bool = True
    # QA model config (falls back to scene_gen config when empty).
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 8192


class ControlSettings(BaseModel):
    """Category 6: control & safety (items #26-#40).

    ``enabled`` gates the invasive pieces (kill switch checks in the
    query loop, watchdog, pause gates). Audit logging and checkpoints
    are cheap and on by default.
    """

    enabled: bool = True
    # #26
    kill_switch: bool = True
    # #30
    pause_resume: bool = True
    # #36 — no progress for this many seconds -> trip the kill switch.
    watchdog_timeout_seconds: int = 900
    # #34 — detect stuck agents (semaphore fully held + no completions).
    deadlock_detection: bool = True
    deadlock_probe_seconds: int = 180
    # #29 — append-only JSONL audit log of every tool/LLM/render/QA action.
    audit_log: bool = True
    # #27 — per-stage wall-clock limits.
    stage_time_limits: bool = True
    llm_call_timeout_seconds: int = 600
    tool_timeout_seconds: int = 900
    # #28 — cost alerts at warning/halt ratios of the session budget.
    cost_alerts: bool = True
    cost_budget_usd: float | None = None
    cost_warn_ratio: float = 0.8
    # #37 — auto-checkpoint every N turns.
    checkpoints: bool = True
    checkpoint_interval_turns: int = 10
    # #31 — cross-agent file locking on write/edit.
    file_locking: bool = True
    file_lock_timeout_seconds: int = 30
    # #33 — validate + repair session state before resume.
    state_validation: bool = True
    # #32 — finish in-flight work before the process dies.
    graceful_shutdown: bool = True


class ModelCacheSettings(BaseModel):
    """#20 — disk-based model cache for HuggingFace downloads.

    Redirects HF_HOME / HF_HUB_CACHE to a persistent directory so a
    restarted session verifies existing weights instead of re-downloading
    them (5-10 minutes saved per session when the directory persists).
    """

    enabled: bool = False
    cache_dir: str = ""


class Settings(BaseModel):
    """Main settings model for OpenHarness."""

    # API configuration
    api_key: str = ""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 16384
    base_url: str | None = None
    timeout: float = 30.0
    context_window_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None
    api_format: str = "anthropic"  # "anthropic", "openai", or "copilot"
    provider: str = ""
    active_profile: str = "claude-api"
    profiles: dict[str, ProviderProfile] = Field(default_factory=default_provider_profiles)
    max_turns: int = 200

    # Behavior
    system_prompt: str | None = None
    permission: PermissionSettings = Field(default_factory=PermissionSettings)
    hooks: dict[str, list[HookDefinition]] = Field(default_factory=dict)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    enabled_plugins: dict[str, bool] = Field(default_factory=dict)
    allow_project_plugins: bool = False
    allow_project_skills: bool = True
    project_skill_dirs: list[str] = Field(
        default_factory=lambda: [".openharness/skills", ".agents/skills", ".claude/skills"]
    )
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)

    # UI
    theme: str = "default"
    output_style: str = "default"
    vim_mode: bool = False
    voice_mode: bool = False
    fast_mode: bool = False
    effort: str = "medium"
    passes: int = 1
    verbose: bool = False

    # --- Performance / scaling settings (top-10 optimizations) ---
    # Use ``None`` as the sentinel for "user did not set this" so Kaggle
    # auto-detection can distinguish explicit-5 from default-5. When the
    # runtime loads settings, it converts ``None`` to the documented
    # default; Kaggle auto-detection runs BEFORE that conversion so it
    # can fill in hardware-aware values without overriding explicit choices.
    max_concurrent_agents: int | None = None
    batch_size: int | None = None
    auto_compact_threshold_ratio: float | None = None
    trusted_project: bool = False
    # Optimization #6 + #8: extra paths auto-approved when trusted_project
    # is on. Use this to whitelist sibling project dirs, asset roots, etc.
    # Paths are matched as path-prefix (after rstrip "/"), so
    # "/data/assets" matches "/data/assets/foo.png" but not "/data/assets-extra".
    trusted_paths: list[str] = Field(default_factory=list)
    tool_cache_enabled: bool = False
    tool_cache_max_entries: int = 256
    # Optimization #7: TTL for tool-cache entries, in seconds. Default
    # 3600 (1 hour) so stale filesystem reads eventually expire. Set to
    # 0 to disable expiry (rely on mutation-based invalidation only).
    tool_cache_ttl_seconds: int = 3600
    parallel_tools: bool = True
    auto_save: bool = True
    # Optimization #7 + #10: save a session snapshot every N turns (default 10).
    # 1 = save after every turn (most resilient, slowest). 0 = same as 1
    # for backwards compat. Larger values reduce disk writes on long runs.
    auto_save_interval: int = 10
    thinking_budget: int | None = None
    auto_context_inject: bool = True
    auto_context_files: list[str] = Field(
        default_factory=lambda: ["BRIEF.md", "STORYBOARD.md", "STYLE.md"]
    )
    provider_routing: dict[str, Any] | None = None

    # --- Waves 2-6 performance/safety feature settings (items #1-#40) ---
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    rendering: RenderSettings = Field(default_factory=RenderSettings)
    qa: QASettings = Field(default_factory=QASettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    model_cache: ModelCacheSettings = Field(default_factory=ModelCacheSettings)

    # Vision model (image-to-text fallback)
    vision: VisionModelConfig = Field(default_factory=VisionModelConfig)

    # Local MiniCPM-V 4.5 VLM (minicpm_vlm tool)
    minicpm_v: MiniCPMVLMConfig = Field(default_factory=MiniCPMVLMConfig)
    # Embedding endpoint for the semantic-memory store
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Image generation model
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)

    # Scene generation model (for batch_scene tool)
    scene_gen: SceneGenConfig = Field(default_factory=SceneGenConfig)

    def merged_profiles(self) -> dict[str, ProviderProfile]:
        """Return the saved profiles merged over the built-in catalog."""
        merged = default_provider_profiles()
        for name, raw_profile in self.profiles.items():
            profile = (
                raw_profile.model_copy(deep=True)
                if isinstance(raw_profile, ProviderProfile)
                else ProviderProfile.model_validate(raw_profile)
            )
            builtin = merged.get(name)
            if builtin is not None and profile.base_url is None and builtin.base_url is not None:
                profile = profile.model_copy(update={"base_url": builtin.base_url})
            merged[name] = profile
        return merged

    def resolve_profile(self, name: str | None = None) -> tuple[str, ProviderProfile]:
        """Return the active provider profile."""
        profiles = self.merged_profiles()
        profile_name = (name or self.active_profile or os.environ.get("OPENHARNESS_PROFILE") or "").strip() or "claude-api"
        if profile_name not in profiles:
            fallback_name, fallback = _profile_from_flat_settings(self)
            profiles[fallback_name] = fallback
            profile_name = fallback_name
        return profile_name, profiles[profile_name].model_copy(deep=True)

    def materialize_active_profile(self) -> Settings:
        """Project the active profile back onto legacy flat settings fields."""
        profile_name, profile = self.resolve_profile()
        configured_model = (profile.last_model or "").strip() or profile.default_model
        return self.model_copy(
            update={
                "active_profile": profile_name,
                "profiles": self.merged_profiles(),
                "provider": profile.provider,
                "api_format": profile.api_format,
                "base_url": profile.base_url,
                "context_window_tokens": profile.context_window_tokens,
                "auto_compact_threshold_tokens": profile.auto_compact_threshold_tokens,
                "model": resolve_model_setting(
                    configured_model,
                    profile.provider,
                    default_model=profile.default_model,
                    permission_mode=self.permission.mode.value,
                ),
            }
        )

    def sync_active_profile_from_flat_fields(self) -> Settings:
        """Fold legacy flat provider fields back into the active profile.

        This preserves compatibility for callers that still construct `Settings`
        by setting top-level `provider` / `api_format` / `base_url` / `model`
        directly before the profile layer is used everywhere.
        """
        profile_name, profile = self.resolve_profile()
        profile_from_env = bool(os.environ.get("OPENHARNESS_PROFILE"))
        flat_profile_fields_match_profile = profile_from_env or (
            (self.provider or "").strip() == profile.provider
            and (self.api_format or "").strip() == profile.api_format
            and self.base_url == profile.base_url
        )
        next_provider = profile.provider if flat_profile_fields_match_profile else (self.provider or "").strip() or profile.provider
        next_api_format = profile.api_format if flat_profile_fields_match_profile else (self.api_format or "").strip() or profile.api_format
        next_base_url = profile.base_url if flat_profile_fields_match_profile else (self.base_url if self.base_url is not None else profile.base_url)
        next_context_window_tokens = (
            self.context_window_tokens
            if self.context_window_tokens is not None
            else profile.context_window_tokens
        )
        next_auto_compact_threshold_tokens = (
            self.auto_compact_threshold_tokens
            if self.auto_compact_threshold_tokens is not None
            else profile.auto_compact_threshold_tokens
        )
        flat_model = (self.model or "").strip()
        resolved_profile_model = resolve_model_setting(
            (profile.last_model or "").strip() or profile.default_model,
            profile.provider,
            default_model=profile.default_model,
            permission_mode=self.permission.mode.value,
        )
        if flat_model and flat_model != resolved_profile_model:
            next_model = flat_model
        else:
            next_model = profile.last_model
        current_default_auth = default_auth_source_for_provider(profile.provider, profile.api_format)
        next_auth_source = profile.auth_source
        if not next_auth_source or next_auth_source == current_default_auth:
            next_auth_source = default_auth_source_for_provider(next_provider, next_api_format)

        updated_profile = profile.model_copy(
            update={
                "provider": next_provider,
                "api_format": next_api_format,
                "base_url": next_base_url,
                "auth_source": next_auth_source,
                "last_model": next_model,
                "context_window_tokens": next_context_window_tokens,
                "auto_compact_threshold_tokens": next_auto_compact_threshold_tokens,
            }
        )
        profiles = self.merged_profiles()
        profiles[profile_name] = updated_profile
        return self.model_copy(
            update={
                "active_profile": profile_name,
                "profiles": profiles,
            }
        )

    def resolve_api_key(self) -> str:
        """Resolve API key with precedence: instance value > env var > empty.

        For ``copilot`` api_format the key is managed separately via
        ``oh auth copilot-login`` and this method is not called.

        Returns the API key string. Raises ValueError if no key is found.
        """
        profile_name, profile = self.resolve_profile()
        del profile_name
        if profile.provider == "openai_codex":
            return self.resolve_auth().value
        if profile.provider == "anthropic_claude":
            raise ValueError(
                "Current provider uses Anthropic auth tokens instead of API keys. "
                "Use resolve_auth() for runtime credential resolution."
            )
        # Copilot format manages its own auth; skip normal key resolution.
        if profile.api_format == "copilot":
            return "copilot-managed"

        if self.api_key:
            return self.api_key

        env_resolved = resolve_auth_env_value(profile.auth_source)
        if env_resolved:
            _, env_value = env_resolved
            return env_value

        raise ValueError(
            "No API key found. Set an OPENHARNESS_* provider API key "
            "(preferred) or the matching native provider environment variable, "
            "or configure api_key in ~/.openharness/settings.json"
        )

    def resolve_auth(self) -> ResolvedAuth:
        """Resolve auth for the current provider, including subscription bridges."""
        profile_name, profile = self.resolve_profile()
        provider = profile.provider.strip()
        auth_source = profile.auth_source.strip() or default_auth_source_for_provider(provider, profile.api_format)
        if auth_source in {"codex_subscription", "claude_subscription"}:
            env_auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
            if auth_source == "claude_subscription" and env_auth_token:
                return ResolvedAuth(
                    provider=provider,
                    auth_kind="oauth",
                    value=env_auth_token,
                    source="env:ANTHROPIC_AUTH_TOKEN",
                    state="configured",
                )
            from openharness.auth.external import (
                is_third_party_anthropic_endpoint,
                load_external_credential,
            )
            from openharness.auth.storage import load_external_binding

            if auth_source == "claude_subscription" and is_third_party_anthropic_endpoint(profile.base_url):
                raise ValueError(
                    "Claude subscription auth only supports direct Anthropic/Claude endpoints. "
                    "Use an API-key-backed Anthropic-compatible profile for third-party base URLs."
                )
            binding = load_external_binding(auth_source_provider_name(auth_source))
            if binding is None:
                raise ValueError(
                    f"No external auth binding found for {auth_source}. Run 'oh auth "
                    f"{'codex-login' if auth_source == 'codex_subscription' else 'claude-login'}' first."
                )
            credential = load_external_credential(
                binding,
                refresh_if_needed=(auth_source == "claude_subscription"),
            )
            return ResolvedAuth(
                provider=provider,
                auth_kind=credential.auth_kind,
                value=credential.value,
                source=f"external:{credential.source_path}",
                state="configured",
            )

        if auth_source == "copilot_oauth":
            return ResolvedAuth(
                provider="copilot",
                auth_kind="oauth_device",
                value="copilot-managed",
                source="copilot",
                state="configured",
            )

        storage_provider = auth_source_provider_name(auth_source)

        from openharness.auth.storage import load_credential

        if profile.credential_slot:
            scoped_storage_provider = f"profile:{profile.credential_slot}"
            scoped = load_credential(scoped_storage_provider, "api_key", use_keyring=False)
            if scoped is None:
                scoped = load_credential(scoped_storage_provider, "api_key")
            if scoped:
                return ResolvedAuth(
                    provider=provider or auth_source_provider_name(auth_source),
                    auth_kind="api_key",
                    value=scoped,
                    source=f"file:{scoped_storage_provider}",
                    state="configured",
                )

        storage_provider = credential_storage_provider_name(profile_name, profile)

        env_resolved = resolve_auth_env_value(auth_source)
        if env_resolved:
            env_var, env_value = env_resolved
            return ResolvedAuth(
                provider=provider or storage_provider,
                auth_kind="api_key",
                value=env_value,
                source=f"env:{env_var}",
                state="configured",
            )

        explicit_key = "" if profile.credential_slot else self.api_key
        if explicit_key:
            return ResolvedAuth(
                provider=provider or storage_provider,
                auth_kind="api_key",
                value=explicit_key,
                source="settings_or_env",
                state="configured",
            )

        stored = load_credential(storage_provider, "api_key")
        if stored:
            return ResolvedAuth(
                provider=provider or auth_source_provider_name(auth_source),
                auth_kind="api_key",
                value=stored,
                source=f"file:{storage_provider}",
                state="configured",
            )

        raise ValueError(
            f"No credentials found for auth source '{auth_source}'. "
            "Configure the matching provider or environment variable first."
        )

    def merge_cli_overrides(self, **overrides: Any) -> Settings:
        """Return a new Settings with CLI overrides applied (non-None values only)."""
        updates = {k: v for k, v in overrides.items() if v is not None}
        permission_mode = updates.pop("permission_mode", None)

        def apply_permission_mode(settings: Settings) -> Settings:
            if permission_mode is None:
                return settings
            return settings.model_copy(
                update={
                    "permission": settings.permission.model_copy(
                        update={"mode": PermissionMode(str(permission_mode))}
                    )
                }
            )

        # Strip ANSI escape sequences from model name if present
        if "model" in updates and isinstance(updates["model"], str):
            updates["model"] = strip_ansi_escape_sequences(updates["model"])
        if "effort" in updates and isinstance(updates["effort"], str):
            updates["effort"] = "xhigh" if updates["effort"].strip().lower() == "max" else updates["effort"].strip().lower()
        merged = apply_permission_mode(self.model_copy(update=updates))
        if not updates:
            return merged
        profile_keys = {
            "model",
            "base_url",
            "api_format",
            "provider",
            "api_key",
            "active_profile",
            "profiles",
            "context_window_tokens",
            "auto_compact_threshold_tokens",
        }
        profile_updates = profile_keys.intersection(updates)
        if not profile_updates:
            return merged
        if "active_profile" in profile_updates:
            switch_updates = {
                key: value
                for key, value in updates.items()
                if key not in profile_keys or key in {"active_profile", "profiles"}
            }
            switched = apply_permission_mode(self.model_copy(update=switch_updates)).materialize_active_profile()
            remaining_profile_updates = {
                key: value
                for key, value in updates.items()
                if key in profile_keys and key not in {"active_profile", "profiles"}
            }
            if not remaining_profile_updates:
                return switched
            return (
                switched.model_copy(update=remaining_profile_updates)
                .sync_active_profile_from_flat_fields()
                .materialize_active_profile()
            )
        return merged.sync_active_profile_from_flat_fields().materialize_active_profile()


class _CsvList:
    """Sentinel type marking comma-separated list env overrides."""


def _apply_env_overrides(settings: Settings) -> Settings:
    """Apply supported environment variable overrides over loaded settings.

    Provider-scoped env vars (``ANTHROPIC_BASE_URL``, ``ANTHROPIC_MODEL``,
    ``OPENAI_BASE_URL``) only apply when the active profile does *not*
    explicitly configure the corresponding field.  ``OPENHARNESS_*`` env vars
    always override (explicit user intent).
    """
    updates: dict[str, Any] = {}

    # Resolve the active profile to check for explicit settings.
    _, active_profile = settings.resolve_profile()
    profile_has_base_url = active_profile.base_url is not None
    profile_explicit_model = (active_profile.last_model or "").strip()
    profile_has_explicit_model = bool(profile_explicit_model) and profile_explicit_model.lower() not in {"", "default"}

    # --- model ---
    openharness_model = os.environ.get("OPENHARNESS_MODEL")
    if openharness_model:
        updates["model"] = strip_ansi_escape_sequences(openharness_model)
    elif not profile_has_explicit_model:
        anthropic_model = os.environ.get("ANTHROPIC_MODEL")
        if anthropic_model:
            updates["model"] = strip_ansi_escape_sequences(anthropic_model)

    # --- base_url ---
    openharness_base = os.environ.get("OPENHARNESS_BASE_URL")
    if openharness_base:
        updates["base_url"] = openharness_base
    elif not profile_has_base_url:
        generic_base = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if generic_base:
            updates["base_url"] = generic_base

    max_tokens = os.environ.get("OPENHARNESS_MAX_TOKENS")
    if max_tokens:
        updates["max_tokens"] = int(max_tokens)

    timeout = os.environ.get("OPENHARNESS_TIMEOUT")
    if timeout:
        updates["timeout"] = float(timeout)

    max_turns = os.environ.get("OPENHARNESS_MAX_TURNS")
    if max_turns:
        updates["max_turns"] = int(max_turns)

    context_window_tokens = os.environ.get("OPENHARNESS_CONTEXT_WINDOW_TOKENS")
    if context_window_tokens:
        updates["context_window_tokens"] = int(context_window_tokens)

    auto_compact_threshold_tokens = os.environ.get("OPENHARNESS_AUTO_COMPACT_THRESHOLD_TOKENS")
    if auto_compact_threshold_tokens:
        updates["auto_compact_threshold_tokens"] = int(auto_compact_threshold_tokens)

    provider = os.environ.get("OPENHARNESS_PROVIDER")
    api_format = os.environ.get("OPENHARNESS_API_FORMAT")
    env_auth_source = active_profile.auth_source
    if provider or api_format:
        env_auth_source = default_auth_source_for_provider(
            provider or active_profile.provider,
            api_format or active_profile.api_format,
        )

    env_resolved = resolve_auth_env_value(env_auth_source)
    if env_resolved:
        _, api_key = env_resolved
        updates["api_key"] = api_key

    if api_format:
        updates["api_format"] = api_format

    if provider:
        updates["provider"] = provider

    sandbox_enabled = os.environ.get("OPENHARNESS_SANDBOX_ENABLED")
    sandbox_fail = os.environ.get("OPENHARNESS_SANDBOX_FAIL_IF_UNAVAILABLE")
    sandbox_backend = os.environ.get("OPENHARNESS_SANDBOX_BACKEND")
    sandbox_docker_image = os.environ.get("OPENHARNESS_SANDBOX_DOCKER_IMAGE")
    sandbox_updates: dict[str, Any] = {}
    if sandbox_enabled is not None:
        sandbox_updates["enabled"] = _parse_bool_env(sandbox_enabled)
    if sandbox_fail is not None:
        sandbox_updates["fail_if_unavailable"] = _parse_bool_env(sandbox_fail)
    if sandbox_backend is not None:
        sandbox_updates["backend"] = sandbox_backend
    if sandbox_docker_image is not None:
        sandbox_updates["docker"] = settings.sandbox.docker.model_copy(
            update={"image": sandbox_docker_image}
        )
    if sandbox_updates:
        updates["sandbox"] = settings.sandbox.model_copy(update=sandbox_updates)

    web_updates: dict[str, Any] = {}
    web_proxy = os.environ.get("OPENHARNESS_WEB_PROXY")
    if web_proxy:
        web_updates["proxy"] = web_proxy
    web_resolution_mode = os.environ.get("OPENHARNESS_WEB_RESOLUTION_MODE")
    if web_resolution_mode:
        web_updates["resolution_mode"] = web_resolution_mode
    web_synthetic_dns_cidrs = os.environ.get("OPENHARNESS_WEB_SYNTHETIC_DNS_CIDRS")
    if web_synthetic_dns_cidrs:
        web_updates["synthetic_dns_cidrs"] = [
            entry.strip()
            for entry in web_synthetic_dns_cidrs.split(",")
            if entry.strip()
        ]
    if web_updates:
        updates["web"] = settings.web.model_copy(update=web_updates)

    # --- Top-10 performance / scaling overrides ---
    max_concurrent_agents = os.environ.get("OPENHARNESS_MAX_CONCURRENT_AGENTS")
    if max_concurrent_agents:
        try:
            updates["max_concurrent_agents"] = max(1, int(max_concurrent_agents))
        except ValueError:
            pass

    batch_size_env = os.environ.get("OPENHARNESS_BATCH_SIZE")
    if batch_size_env:
        try:
            updates["batch_size"] = max(1, int(batch_size_env))
        except ValueError:
            pass

    compact_ratio = os.environ.get("OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO")
    if compact_ratio:
        try:
            updates["auto_compact_threshold_ratio"] = float(compact_ratio)
        except ValueError:
            pass

    trusted_project_env = os.environ.get("OPENHARNESS_TRUSTED_PROJECT")
    if trusted_project_env is not None:
        updates["trusted_project"] = _parse_bool_env(trusted_project_env)

    tool_cache_env = os.environ.get("OPENHARNESS_TOOL_CACHE")
    if tool_cache_env is not None:
        updates["tool_cache_enabled"] = _parse_bool_env(tool_cache_env)

    tool_cache_ttl_env = os.environ.get("OPENHARNESS_TOOL_CACHE_TTL_SECONDS")
    if tool_cache_ttl_env:
        try:
            updates["tool_cache_ttl_seconds"] = max(0, int(tool_cache_ttl_env))
        except ValueError:
            pass

    trusted_paths_env = os.environ.get("OPENHARNESS_TRUSTED_PATHS")
    if trusted_paths_env:
        updates["trusted_paths"] = [
            p.strip().rstrip("/") for p in trusted_paths_env.split(":") if p.strip()
        ]

    auto_save_interval_env = os.environ.get("OPENHARNESS_AUTO_SAVE_INTERVAL")
    if auto_save_interval_env:
        try:
            updates["auto_save_interval"] = max(0, int(auto_save_interval_env))
        except ValueError:
            pass

    parallel_tools_env = os.environ.get("OPENHARNESS_PARALLEL_TOOLS")
    if parallel_tools_env is not None:
        updates["parallel_tools"] = _parse_bool_env(parallel_tools_env)

    auto_save_env = os.environ.get("OPENHARNESS_AUTO_SAVE")
    if auto_save_env is not None:
        updates["auto_save"] = _parse_bool_env(auto_save_env)

    thinking_budget_env = os.environ.get("OPENHARNESS_THINKING_BUDGET")
    if thinking_budget_env:
        try:
            updates["thinking_budget"] = max(0, int(thinking_budget_env))
        except ValueError:
            pass

    auto_context_env = os.environ.get("OPENHARNESS_AUTO_CONTEXT_INJECT")
    if auto_context_env is not None:
        updates["auto_context_inject"] = _parse_bool_env(auto_context_env)

    # --- Waves 2-6 feature overrides (items #1-#40) ---
    def _nested(namespace: str, model_cls, env_map: dict[str, tuple[str, Any]]) -> None:
        """Collect env overrides for a nested settings model.

        ``env_map`` maps ENV_VAR -> (field, parser). Only present env vars
        are applied, over a copy of the current nested model.
        """
        nested_updates: dict[str, Any] = {}
        for env_var, (field_name, parser) in env_map.items():
            raw = os.environ.get(env_var)
            if raw is None:
                continue
            if parser is bool:
                nested_updates[field_name] = _parse_bool_env(raw)
            elif parser is int:
                try:
                    nested_updates[field_name] = int(raw)
                except ValueError:
                    continue
            elif parser is float:
                try:
                    nested_updates[field_name] = float(raw)
                except ValueError:
                    continue
            elif isinstance(parser, _CsvList):
                nested_updates[field_name] = [p.strip() for p in raw.split(",") if p.strip()]
            else:
                nested_updates[field_name] = raw.strip()
        if nested_updates:
            current = getattr(settings, namespace, None)
            base = current.model_copy() if current is not None else model_cls()
            updates[namespace] = base.model_copy(update=nested_updates)

    _nested(
        "inference",
        InferenceSettings,
        {
            "OPENHARNESS_PREFIX_CACHING": ("prefix_caching", bool),
            "OPENHARNESS_PROMPT_COMPRESSION": ("prompt_compression", bool),
            "OPENHARNESS_PROMPT_COMPRESSION_MIN_CHARS": ("prompt_compression_min_chars", int),
            "OPENHARNESS_SEMANTIC_CACHE": ("semantic_cache", bool),
            "OPENHARNESS_SEMANTIC_CACHE_THRESHOLD": ("semantic_cache_threshold", float),
            "OPENHARNESS_CROSS_PROVIDER_RACING": ("cross_provider_racing", bool),
            "OPENHARNESS_API_KEY_POOL": ("api_key_pool", _CsvList()),
            "OPENHARNESS_KEY_POOL_COOLDOWN_SECONDS": ("key_pool_cooldown_seconds", float),
            "OPENHARNESS_BATCH_CONCURRENCY": ("batch_concurrency", int),
        },
    )
    _nested(
        "rendering",
        RenderSettings,
        {
            "OPENHARNESS_PARALLEL_RENDERING": ("parallel_rendering", bool),
            "OPENHARNESS_RENDER_WORKERS": ("render_workers", int),
            "OPENHARNESS_DRAFT_FIRST": ("draft_first", bool),
            "OPENHARNESS_RENDER_CACHE": ("render_cache", bool),
            "OPENHARNESS_NVENC": ("nvenc", bool),
            "OPENHARNESS_AUTO_ENQUEUE_RENDERS": ("auto_enqueue_renders", bool),
            "OPENHARNESS_RENDER_TIMEOUT_SECONDS": ("render_timeout_seconds", int),
        },
    )
    _nested(
        "qa",
        QASettings,
        {
            "OPENHARNESS_QA_PARALLEL": ("parallel_waves", bool),
            "OPENHARNESS_QA_WAVE_SIZE": ("wave_size", int),
            "OPENHARNESS_QA_BATCHING": ("review_batching", bool),
            "OPENHARNESS_QA_MODEL": ("model", str),
            "OPENHARNESS_QA_API_KEY": ("api_key", str),
            "OPENHARNESS_QA_BASE_URL": ("base_url", str),
        },
    )
    _nested(
        "control",
        ControlSettings,
        {
            "OPENHARNESS_CONTROL": ("enabled", bool),
            "OPENHARNESS_KILL_SWITCH": ("kill_switch", bool),
            "OPENHARNESS_WATCHDOG_TIMEOUT_SECONDS": ("watchdog_timeout_seconds", int),
            "OPENHARNESS_AUDIT_LOG": ("audit_log", bool),
            "OPENHARNESS_TOOL_TIMEOUT_SECONDS": ("tool_timeout_seconds", int),
            "OPENHARNESS_LLM_CALL_TIMEOUT_SECONDS": ("llm_call_timeout_seconds", int),
            "OPENHARNESS_COST_BUDGET_USD": ("cost_budget_usd", float),
            "OPENHARNESS_COST_ALERTS": ("cost_alerts", bool),
            "OPENHARNESS_CHECKPOINT_INTERVAL_TURNS": ("checkpoint_interval_turns", int),
            "OPENHARNESS_FILE_LOCKING": ("file_locking", bool),
            "OPENHARNESS_FILE_LOCK_TIMEOUT_SECONDS": ("file_lock_timeout_seconds", int),
        },
    )
    _nested(
        "model_cache",
        ModelCacheSettings,
        {
            "OPENHARNESS_MODEL_CACHE": ("enabled", bool),
            "OPENHARNESS_MODEL_CACHE_DIR": ("cache_dir", str),
        },
    )
    # Groq key pool: dedicated alias so users can keep provider-specific
    # pools separate from the generic one.
    groq_keys = os.environ.get("OPENHARNESS_GROQ_API_KEYS", "").strip()
    if groq_keys:
        pool = [k.strip() for k in groq_keys.split(",") if k.strip()]
        current = getattr(settings, "inference", None)
        base = current.model_copy() if current is not None else InferenceSettings()
        updates["inference"] = base.model_copy(update={"api_key_pool": pool})

    if not updates:
        return settings
    return settings.model_copy(update=updates)


def _parse_bool_env(value: str) -> bool:
    """Parse a boolean environment override."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config file, merging with defaults.

    Args:
        config_path: Path to settings.json. If None, uses the default location.

    Returns:
        Settings instance with file values merged over defaults.

    Performance note:
        ``load_settings`` re-reads the JSON config file, parses it, and
        re-validates with Pydantic on every call. On a Kaggle-style machine
        the median cost is ~0.5 ms, but it spikes to >100 ms when the file
        contains many profiles. The function is called from many hot paths
        (every bash/grep/glob call via ``create_shell_subprocess``, every
        command registry lookup, every network guard check, etc.), so we
        cache the result keyed by ``(config_path, mtime, size)`` and only
        re-read when the file actually changes. ``save_settings`` clears the
        cache so a write-then-read sequence always returns the fresh value.
    """
    from openharness.config.paths import get_config_file_path

    if config_path is None:
        config_path = get_config_file_path()

    cache_key = _settings_cache_key(config_path)
    cached = _SETTINGS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    settings = _load_settings_uncached(config_path)
    _SETTINGS_CACHE[cache_key] = settings
    return settings


def _settings_cache_key(config_path: Path) -> tuple[str, float, int, int]:
    """Build a cache key that invalidates when the file changes on disk.

    Also folds in a hash of all ``OPENHARNESS_*`` / ``ANTHROPIC_*`` /
    ``OPENAI_*`` environment variables, because ``_apply_env_overrides``
    re-reads them on every call and tests (and real users) mutate them
    between calls. Without this, setting ``OPENHARNESS_WEB_PROXY`` after
    a previous ``load_settings()`` would return the stale cached value.
    """
    try:
        stat = config_path.stat()
        path_part: tuple[str, float, int] = (str(config_path), stat.st_mtime, stat.st_size)
    except OSError:
        path_part = (str(config_path), -1.0, -1)
    env_hash = _env_fingerprint()
    return (*path_part, env_hash)


# Prefixes of env vars that influence the resolved Settings. Kept tight so the
# fingerprint is cheap to compute.
_SETTINGS_ENV_PREFIXES = ("OPENHARNESS_", "ANTHROPIC_", "OPENAI_", "CLAUDE_")


def _env_fingerprint() -> int:
    """Return a stable hash of all settings-relevant env vars.

    Uses ``frozenset`` of items so the hash is order-independent, and Python's
    built-in ``hash`` on a frozenset of (str, str) tuples. We slice the value
    to a small cap so a giant env var can't dominate the cost.
    """
    items: list[tuple[str, str]] = []
    for key, value in os.environ.items():
        if not key.startswith(_SETTINGS_ENV_PREFIXES):
            continue
        items.append((key, value[:256]))
    return hash(frozenset(items))


def _load_settings_uncached(config_path: Path) -> Settings:
    """Original implementation of ``load_settings`` without caching."""
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        settings = Settings.model_validate(raw)
        env_profile = os.environ.get("OPENHARNESS_PROFILE")
        if env_profile:
            settings = settings.model_copy(update={"active_profile": env_profile.strip()})
        if "profiles" not in raw or "active_profile" not in raw:
            profile_name, profile = _profile_from_flat_settings(settings)
            merged_profiles = settings.merged_profiles()
            merged_profiles[profile_name] = profile
            settings = settings.model_copy(
                update={
                    "active_profile": profile_name,
                    "profiles": merged_profiles,
                }
            )
        return _apply_env_overrides(settings.materialize_active_profile())

    settings = Settings()
    env_profile = os.environ.get("OPENHARNESS_PROFILE")
    if env_profile:
        settings = settings.model_copy(update={"active_profile": env_profile.strip()})
    return _apply_env_overrides(settings.materialize_active_profile())


# In-process cache of (config_path, mtime, size) -> Settings. A simple dict
# is fine here because settings writes are rare (CLI /config, /provider) and
# always go through ``save_settings`` which clears the cache.
_SETTINGS_CACHE: dict[tuple[str, float, int], Settings] = {}


def clear_settings_cache() -> None:
    """Clear the in-process settings cache.

    Used by ``save_settings`` and by tests that mutate the config file
    directly on disk.
    """
    _SETTINGS_CACHE.clear()


def save_settings(settings: Settings, config_path: Path | None = None) -> None:
    """Persist settings to the config file.

    Args:
        settings: Settings instance to save.
        config_path: Path to write. If None, uses the default location.
    """
    if config_path is None:
        from openharness.config.paths import get_config_file_path

        config_path = get_config_file_path()

    settings = settings.sync_active_profile_from_flat_fields().materialize_active_profile()
    lock_path = config_path.with_suffix(config_path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        atomic_write_text(
            config_path,
            settings.model_dump_json(indent=2) + "\n",
        )
    # Invalidate the in-process cache so the next load_settings() picks up
    # the new mtime/size and re-reads the file.
    _SETTINGS_CACHE.clear()
