"""Shared runtime assembly for headless and Textual UIs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from openharness.api.client import AnthropicApiClient, SupportsStreamingMessages
from openharness.api.codex_client import CodexApiClient
from openharness.api.copilot_client import CopilotClient
from openharness.api.openai_client import OpenAICompatibleClient
from openharness.api.provider import auth_status, detect_provider
from openharness.bridge import get_bridge_manager
from openharness.commands import (
    CommandContext,
    CommandResult,
    MemoryCommandBackend,
    create_default_command_registry,
    lookup_skill_slash_command,
)
from openharness.config import get_config_file_path, load_settings
from openharness.engine import QueryEngine
from openharness.engine.messages import (
    ConversationMessage,
    ToolResultBlock,
    ToolUseBlock,
    sanitize_conversation_messages,
)
from openharness.engine.query import MaxTurnsExceeded
from openharness.engine.stream_events import StreamEvent
from openharness.hooks import HookEvent, HookExecutionContext, HookExecutor, load_hook_registry
from openharness.hooks.hot_reload import HookReloader
from openharness.mcp.client import McpClientManager
from openharness.mcp.config import load_mcp_server_configs
from openharness.permissions import build_permission_checker
from openharness.plugins import load_plugins
from openharness.prompts import build_runtime_system_prompt
from openharness.state import AppState, AppStateStore
from openharness.services.session_backend import DEFAULT_SESSION_BACKEND, SessionBackend
from openharness.tools import ToolRegistry, create_default_tool_registry
from openharness.keybindings import load_keybindings

PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]
EditApprovalPrompt = Callable[[str, str, int, int], Awaitable[str]]
SystemPrinter = Callable[[str], Awaitable[None]]
StreamRenderer = Callable[[StreamEvent], Awaitable[None]]
ClearHandler = Callable[[], Awaitable[None]]


def _resolve_image_generation_config(settings) -> dict[str, str]:
    """Resolve image generation configuration from settings, environment, and Codex auth."""
    from openharness.config.settings import ImageGenerationConfig, ProviderProfile

    cfg = settings.image_generation
    env_cfg = ImageGenerationConfig.from_env()
    resolved = {
        "provider": cfg.provider or env_cfg.provider,
        "model": cfg.model or env_cfg.model,
        "api_key": cfg.api_key or env_cfg.api_key,
        "base_url": cfg.base_url or env_cfg.base_url,
        "codex_model": cfg.codex_model or env_cfg.codex_model,
        "codex_base_url": cfg.codex_base_url or env_cfg.codex_base_url,
    }

    try:
        codex_profile = settings.merged_profiles().get("codex") or ProviderProfile(
            label="Codex Subscription",
            provider="openai_codex",
            api_format="openai",
            auth_source="codex_subscription",
            default_model="gpt-5.4",
        )
        codex_settings = settings.model_copy(
            update={
                "active_profile": "codex",
                "profiles": {**settings.profiles, "codex": codex_profile},
            }
        ).materialize_active_profile()
        codex_auth = codex_settings.resolve_auth()
        resolved["codex_auth_token"] = codex_auth.value
        resolved["codex_base_url"] = resolved["codex_base_url"] or (codex_settings.base_url or "")
        resolved["codex_model"] = resolved["codex_model"] or codex_settings.model
    except Exception:
        pass

    return resolved


def _resolve_scene_gen_config(settings) -> dict[str, str]:
    """Resolve scene generation config from settings + env.

    Falls back to OPENAI_API_KEY if no scene-gen-specific key is set,
    so users don't need to configure a separate key unless they want
    a different provider for scene gen.
    """
    from openharness.config.settings import SceneGenConfig

    cfg = settings.scene_gen
    env_cfg = SceneGenConfig.from_env()
    api_key = cfg.api_key or env_cfg.api_key
    if not api_key:
        # Fall back to the main OPENAI_API_KEY so users don't have to
        # configure a separate key for scene gen.
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return {
        "model": cfg.model or env_cfg.model,
        "api_key": api_key,
        "base_url": cfg.base_url or env_cfg.base_url,
        "max_tokens": str(cfg.max_tokens or env_cfg.max_tokens),
        "temperature": str(cfg.temperature or env_cfg.temperature),
    }


def _build_brand_anchor_section(cwd: str | Path) -> str | None:
    """Load brand.yaml from the project root and build a system-prompt section.

    Returns None when brand.yaml is missing (no overhead). When present,
    returns a formatted section that lists the brand anchors (name,
    colors, fonts, voice, music style, min type size, safe area, easing)
    so the agent is brand-aware on every turn without needing to call
    the brand_kit tool explicitly.
    """
    from pathlib import Path
    brand_path = Path(cwd) / "brand.yaml"
    if not brand_path.is_file():
        return None
    try:
        from openharness.tools.brand_kit_tool import BrandConfig
        brand = BrandConfig.from_yaml_file(brand_path)
    except Exception:
        return None  # Don't crash the session if brand.yaml is malformed.
    if not brand.name and not brand.colors.primary and not brand.fonts.heading:
        return None  # Empty brand config — skip.
    lines = ["# Brand Anchors (from brand.yaml)", ""]
    if brand.name:
        lines.append(f"- Channel: {brand.name}")
    if brand.colors.primary or brand.colors.secondary or brand.colors.accent:
        lines.append("- Colors:")
        if brand.colors.primary:
            lines.append(f"  - Primary: {brand.colors.primary}")
        if brand.colors.secondary:
            lines.append(f"  - Secondary: {brand.colors.secondary}")
        if brand.colors.accent:
            lines.append(f"  - Accent: {brand.colors.accent}")
    if brand.fonts.heading or brand.fonts.body:
        lines.append("- Fonts:")
        if brand.fonts.heading:
            lines.append(f"  - Heading: {brand.fonts.heading}")
        if brand.fonts.body:
            lines.append(f"  - Body: {brand.fonts.body}")
    if brand.voice:
        lines.append(f"- Voice: {brand.voice}")
    if brand.music_style:
        lines.append(f"- Music style: {brand.music_style}")
    if brand.min_type_size:
        lines.append(f"- Minimum type size: {brand.min_type_size}px")
    if brand.safe_area:
        lines.append(f"- Safe area: {brand.safe_area}px")
    if brand.easing:
        lines.append(f"- Easing: {brand.easing}")
    lines.append("")
    lines.append(
        "All generated scenes MUST comply with these brand anchors. "
        "Use the `brand_kit` tool with `action=drift` to get a per-project "
        "creative drift that stays within this envelope."
    )
    return "\n".join(lines)


def _resolve_vision_config(settings) -> dict[str, str]:
    """Resolve the vision model configuration from settings or environment.

    Priority: settings.vision fields > environment variables > empty.
    """
    from openharness.config.settings import VisionModelConfig

    cfg = settings.vision
    if cfg.is_configured:
        return {
            "model": cfg.model,
            "api_key": cfg.api_key,
            "base_url": cfg.base_url,
        }

    # Fall back to environment variables
    env_cfg = VisionModelConfig.from_env()
    if env_cfg.is_configured:
        return {
            "model": env_cfg.model,
            "api_key": env_cfg.api_key,
            "base_url": env_cfg.base_url,
        }

    return {}


@dataclass
class RuntimeBundle:
    """Shared runtime objects for one interactive session."""

    api_client: SupportsStreamingMessages
    cwd: str
    mcp_manager: McpClientManager
    tool_registry: ToolRegistry
    app_state: AppStateStore
    hook_executor: HookExecutor
    engine: QueryEngine
    commands: object
    external_api_client: bool
    enforce_max_turns: bool = True
    session_id: str = ""
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    session_backend: SessionBackend = DEFAULT_SESSION_BACKEND
    extra_skill_dirs: tuple[str, ...] = ()
    extra_plugin_roots: tuple[str, ...] = ()
    memory_backend: MemoryCommandBackend | None = None
    include_project_memory: bool = True
    autodream_context: dict[str, object] | None = None

    def current_settings(self):
        """Return the effective settings for this session.

        We persist most settings to disk (``~/.openharness/settings.json``), but
        CLI options like ``--model``/``--api-format`` should remain in effect for
        the lifetime of the running process. Without this overlay, issuing any
        slash command (e.g. ``/fast``) would refresh UI state from disk and
        "snap back" the model/provider to whatever is stored in the config file.
        """
        return load_settings().merge_cli_overrides(**self.settings_overrides)

    def current_plugins(self):
        """Return currently visible plugins for the working tree."""
        return load_plugins(
            self.current_settings(),
            self.cwd,
            extra_roots=self.extra_plugin_roots,
        )

    def hook_summary(self) -> str:
        """Return the current hook summary."""
        return load_hook_registry(self.current_settings(), self.current_plugins()).summary()

    def plugin_summary(self) -> str:
        """Return the current plugin summary."""
        plugins = self.current_plugins()
        if not plugins:
            return "No plugins discovered."
        lines = ["Plugins:"]
        for plugin in plugins:
            state = "enabled" if plugin.enabled else "disabled"
            lines.append(f"- {plugin.manifest.name} [{state}] {plugin.manifest.description}")
        return "\n".join(lines)

    def mcp_summary(self) -> str:
        """Return the current MCP summary."""
        statuses = self.mcp_manager.list_statuses()
        if not statuses:
            return "No MCP servers configured."
        lines = ["MCP servers:"]
        for status in statuses:
            suffix = f" - {status.detail}" if status.detail else ""
            lines.append(f"- {status.name}: {status.state}{suffix}")
            if status.tools:
                lines.append(f"  tools: {', '.join(tool.name for tool in status.tools)}")
            if status.resources:
                lines.append(f"  resources: {', '.join(resource.uri for resource in status.resources)}")
        return "\n".join(lines)


def _resolve_api_client_from_settings(settings) -> SupportsStreamingMessages:
    """Build the appropriate API client for the resolved settings.

    If ``settings.provider_routing`` is configured (optimization #4),
    the returned client is a :class:`RoutedStreamingClient` that tries
    the primary provider first and falls back on retryable errors.
    """
    # Ensure profile fields (base_url, model, api_format) are projected to settings
    settings = settings.materialize_active_profile()

    def _safe_resolve_auth():
        try:
            return settings.resolve_auth()
        except Exception as exc:
            _print_auth_resolution_error(settings, exc)
            raise SystemExit(1)

    def _build_primary() -> SupportsStreamingMessages:
        if settings.api_format == "copilot":
            from openharness.api.copilot_client import COPILOT_DEFAULT_MODEL

            copilot_model = (
                COPILOT_DEFAULT_MODEL
                if settings.model in {"claude-sonnet-4-20250514", "claude-sonnet-4-6", "sonnet", "default"}
                else settings.model
            )
            return CopilotClient(model=copilot_model)
        if settings.provider == "openai_codex":
            auth = _safe_resolve_auth()
            return CodexApiClient(
                auth_token=auth.value,
                base_url=settings.base_url,
            )
        if settings.provider == "anthropic_claude":
            return AnthropicApiClient(
                auth_token=_safe_resolve_auth().value,
                base_url=settings.base_url,
                claude_oauth=True,
                auth_token_resolver=lambda: settings.resolve_auth().value,
            )
        if settings.api_format in ("openai", "openai_compat"):
            auth = _safe_resolve_auth()
            return OpenAICompatibleClient(
                api_key=auth.value,
                base_url=settings.base_url,
                timeout=settings.timeout,
            )
        auth = _safe_resolve_auth()
        return AnthropicApiClient(
            api_key=auth.value,
            base_url=settings.base_url,
        )

    primary = _build_primary()

    # Optimization #4: auto-construct a routed client when
    # ``settings.provider_routing`` is configured. The routed client
    # wraps ``primary`` and adds fallback providers; on a retryable
    # failure (rate limit, 5xx, network) it retries against the next
    # provider in the chain.
    routing_config = getattr(settings, "provider_routing", None)
    if routing_config and isinstance(routing_config, dict):
        try:
            from openharness.services.provider_routing import (
                ProviderRoute,
                RoutedStreamingClient,
            )

            fallback_routes: list[ProviderRoute] = []
            for fallback_cfg in routing_config.get("fallbacks", []):
                if not isinstance(fallback_cfg, dict):
                    continue
                name = fallback_cfg.get("name") or f"fallback_{len(fallback_routes) + 1}"
                # Shared route builder lives in provider_routing so there
                # is exactly one implementation of the config shape.
                from openharness.services.provider_routing import build_fallback_route
                route = build_fallback_route(name, fallback_cfg)
                if route is not None:
                    fallback_routes.append(route)
            if fallback_routes:
                # Wrap the primary client as the first route so the
                # routed client can replay against it on fallback too.
                primary_route = ProviderRoute(
                    name=routing_config.get("primary", {}).get("name") or "primary",
                    client=primary,
                    delay_seconds=0.0,
                )
                return RoutedStreamingClient(primary_route, *fallback_routes)
        except Exception:
            # Routing setup must never block the session — fall back to
            # the unwrapped primary client if anything goes wrong.
            pass

    return primary


def _print_auth_resolution_error(settings, exc: Exception) -> None:
    """Render auth failures without collapsing subscription errors into API-key advice."""
    try:
        profile_name, profile = settings.resolve_profile()
        auth_source = (getattr(profile, "auth_source", "") or "").strip()
    except Exception:
        profile_name = ""
        auth_source = ""

    message = str(exc).strip() or exc.__class__.__name__
    if auth_source in {"claude_subscription", "codex_subscription"}:
        login_command = "claude-login" if auth_source == "claude_subscription" else "codex-login"
        provider_name = profile_name or (
            "claude-subscription" if auth_source == "claude_subscription" else "codex"
        )
        print(
            f"Error: {message}\n"
            f"  This profile uses subscription auth, not an API key.\n"
            f"  Run `oh auth {login_command}` to bind the local CLI session, then\n"
            f"  run `oh provider use {provider_name}` to activate it.",
            file=sys.stderr,
        )
        return

    print(
        "Error: No API key configured.\n"
        f"  {message}\n"
        "  Run `oh auth login` to set up authentication, or set the\n"
        "  ANTHROPIC_API_KEY (or OPENAI_API_KEY) environment variable.",
        file=sys.stderr,
    )


async def build_runtime(
    *,
    prompt: str | None = None,
    cwd: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    effort: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    api_format: str | None = None,
    active_profile: str | None = None,
    api_client: SupportsStreamingMessages | None = None,
    permission_prompt: PermissionPrompt | None = None,
    ask_user_prompt: AskUserPrompt | None = None,
    edit_approval_prompt: EditApprovalPrompt | None = None,
    restore_messages: list[dict] | None = None,
    restore_tool_metadata: dict[str, object] | None = None,
    enforce_max_turns: bool = True,
    session_backend: SessionBackend | None = None,
    permission_mode: str | None = None,
    extra_skill_dirs: Iterable[str | Path] | None = None,
    extra_plugin_roots: Iterable[str | Path] | None = None,
    memory_backend: MemoryCommandBackend | None = None,
    include_project_memory: bool = True,
    autodream_context: dict[str, object] | None = None,
) -> RuntimeBundle:
    """Build the shared runtime for an OpenHarness session."""
    settings_overrides: dict[str, Any] = {
        "model": model,
        "max_turns": max_turns,
        "effort": effort,
        "base_url": base_url,
        "system_prompt": system_prompt,
        "api_key": api_key,
        "api_format": api_format,
        "active_profile": active_profile,
        "permission_mode": permission_mode,
    }
    settings = load_settings().merge_cli_overrides(**settings_overrides)
    cwd = str(Path(cwd).expanduser().resolve()) if cwd else str(Path.cwd())
    normalized_skill_dirs = tuple(str(Path(path).expanduser().resolve()) for path in (extra_skill_dirs or ()))
    normalized_plugin_roots = tuple(str(Path(path).expanduser().resolve()) for path in (extra_plugin_roots or ()))
    plugins = load_plugins(settings, cwd, extra_roots=normalized_plugin_roots)
    if api_client:
        resolved_api_client = api_client
    else:
        resolved_api_client = _resolve_api_client_from_settings(settings)
    mcp_manager = McpClientManager(load_mcp_server_configs(settings, plugins))
    await mcp_manager.connect_all()
    tool_registry = create_default_tool_registry(mcp_manager)
    # Register plugin-provided tools
    for plugin in plugins:
        if plugin.enabled and plugin.tools:
            for tool in plugin.tools:
                tool_registry.register(tool)
    provider = detect_provider(settings)
    bridge_manager = get_bridge_manager()
    app_state = AppStateStore(
        AppState(
            # Show the effective runtime model (after CLI/env/profile merges),
            # not profile.last_model which may be stale.
            model=settings.model,
            permission_mode=settings.permission.mode.value,
            theme=settings.theme,
            cwd=cwd,
            provider=provider.name,
            auth_status=auth_status(settings),
            base_url=settings.base_url or "",
            vim_enabled=settings.vim_mode,
            voice_enabled=settings.voice_mode,
            voice_available=provider.voice_supported,
            voice_reason=provider.voice_reason,
            fast_mode=settings.fast_mode,
            effort=settings.effort,
            passes=settings.passes,
            mcp_connected=sum(1 for status in mcp_manager.list_statuses() if status.state == "connected"),
            mcp_failed=sum(1 for status in mcp_manager.list_statuses() if status.state == "failed"),
            bridge_sessions=len(bridge_manager.list_sessions()),
            output_style=settings.output_style,
            keybindings=load_keybindings(),
        )
    )
    hook_reloader = HookReloader(get_config_file_path())
    hook_executor = HookExecutor(
        hook_reloader.current_registry() if api_client is None else load_hook_registry(settings, plugins),
        HookExecutionContext(
            cwd=Path(cwd).resolve(),
            api_client=resolved_api_client,
            default_model=settings.model,
        ),
    )
    engine_max_turns = settings.max_turns if (enforce_max_turns or max_turns is not None) else None
    system_prompt_text = build_runtime_system_prompt(
        settings,
        cwd=cwd,
        latest_user_prompt=prompt,
        extra_skill_dirs=normalized_skill_dirs,
        extra_plugin_roots=normalized_plugin_roots,
        include_project_memory=include_project_memory,
    )
    # Item #2 (Brand Kit wiring): inject brand.yaml anchors into the
    # system prompt so every turn sees the brand identity. This makes
    # the agent brand-aware without needing to call brand_kit on every
    # turn. The drift is generated lazily per-project, not here.
    brand_section = _build_brand_anchor_section(cwd)
    if brand_section:
        system_prompt_text = f"{system_prompt_text}\n\n{brand_section}"
    from uuid import uuid4

    session_id = uuid4().hex[:12]

    restored_metadata = {
        "permission_mode": settings.permission.mode.value,
        "read_file_state": [],
        "invoked_skills": [],
        "async_agent_state": [],
        "async_agent_tasks": [],
        "recent_work_log": [],
        "recent_verified_work": [],
        "task_focus_state": {
            "goal": "",
            "recent_goals": [],
            "active_artifacts": [],
            "verified_state": [],
            "next_step": "",
        },
        "compact_checkpoints": [],
    }
    if isinstance(restore_tool_metadata, dict):
        for key, value in restore_tool_metadata.items():
            restored_metadata[key] = value

    engine = QueryEngine(
        api_client=resolved_api_client,
        tool_registry=tool_registry,
        permission_checker=build_permission_checker(settings, cwd=str(cwd)),
        cwd=cwd,
        model=settings.model,
        system_prompt=system_prompt_text,
        max_tokens=settings.max_tokens,
        context_window_tokens=settings.context_window_tokens or settings.memory.context_window_tokens,
        auto_compact_threshold_tokens=(
            settings.auto_compact_threshold_tokens
            or settings.memory.auto_compact_threshold_tokens
        ),
        max_turns=engine_max_turns,
        permission_prompt=permission_prompt,
        ask_user_prompt=ask_user_prompt,
        hook_executor=hook_executor,
        settings=settings,
        tool_metadata={
            "mcp_manager": mcp_manager,
            "bridge_manager": bridge_manager,
            "extra_skill_dirs": normalized_skill_dirs,
            "extra_plugin_roots": normalized_plugin_roots,
            "session_id": session_id,
            "edit_approval_prompt": edit_approval_prompt,
            "vision_model_config": _resolve_vision_config(settings),
            "image_generation_config": _resolve_image_generation_config(settings),
            # Scene generation config (for batch_scene tool)
            "scene_gen_config": _resolve_scene_gen_config(settings),
            # Optimization #3: expose resolved settings to tools (agent_tool
            # reads auto_context_inject / auto_context_files from here).
            "settings": settings,
            # Optimization #8: forward parallel_tools flag to the query loop.
            "parallel_tools": bool(getattr(settings, "parallel_tools", True)),
            # Optimization #2: forward batch_size for tools that support it.
            "batch_size": int(getattr(settings, "batch_size", 1) or 1),
            # Optimization #1: forward max_concurrent_agents to spawned agents.
            "max_concurrent_agents": int(getattr(settings, "max_concurrent_agents", 5) or 5),
            **restored_metadata,
        },
    )
    if autodream_context is not None:
        engine.tool_metadata["autodream_context"] = autodream_context
    # Apply Kaggle-aware hardware defaults if running on Kaggle and the
    # user hasn't explicitly overridden them. This auto-raises
    # ``max_concurrent_agents`` and ``batch_size`` based on detected
    # CPU + GPU counts so Kaggle users get full hardware utilization
    # without manual config.
    try:
        from openharness.services.kaggle_env import apply_kaggle_defaults_to_settings
        apply_kaggle_defaults_to_settings(settings)
    except Exception:
        pass  # Off-Kaggle or detection failed — use defaults
    # Apply optimization settings: #1 agent concurrency, #7 tool cache.
    try:
        from openharness.services.agent_concurrency import configure_agent_concurrency
        configure_agent_concurrency(settings)
    except Exception:
        pass
    try:
        from openharness.services.tool_cache import configure_tool_cache_from_settings
        configure_tool_cache_from_settings(settings)
    except Exception:
        pass
    # Restore messages from a saved session if provided
    if restore_messages:
        restored = sanitize_conversation_messages(
            [ConversationMessage.model_validate(m) for m in restore_messages]
        )
        engine.load_messages(restored)

    # Start Docker sandbox if configured
    if settings.sandbox.enabled and settings.sandbox.backend == "docker":
        from openharness.sandbox.session import start_docker_sandbox

        await start_docker_sandbox(settings, session_id, Path(cwd))

    return RuntimeBundle(
        api_client=resolved_api_client,
        cwd=cwd,
        mcp_manager=mcp_manager,
        tool_registry=tool_registry,
        app_state=app_state,
        hook_executor=hook_executor,
        engine=engine,
        commands=create_default_command_registry(
            plugin_commands=[
                command
                for plugin in plugins
                if plugin.enabled
                for command in plugin.commands
            ]
        ),
        external_api_client=api_client is not None,
        enforce_max_turns=enforce_max_turns or max_turns is not None,
        session_id=session_id,
        settings_overrides=settings_overrides,
        session_backend=session_backend or DEFAULT_SESSION_BACKEND,
        extra_skill_dirs=normalized_skill_dirs,
        extra_plugin_roots=normalized_plugin_roots,
        memory_backend=memory_backend,
        include_project_memory=include_project_memory,
        autodream_context=autodream_context,
    )


async def start_runtime(bundle: RuntimeBundle) -> None:
    """Run session start hooks."""
    await bundle.hook_executor.execute(
        HookEvent.SESSION_START,
        {"cwd": bundle.cwd, "event": HookEvent.SESSION_START.value},
    )


async def close_runtime(bundle: RuntimeBundle) -> None:
    """Close runtime-owned resources."""
    from openharness.sandbox.session import stop_docker_sandbox

    await stop_docker_sandbox()
    # Extract local environment rules from session before closing
    try:
        from openharness.personalization.session_hook import update_rules_from_session
        update_rules_from_session(bundle.engine.messages)
    except Exception:
        pass  # personalization is best-effort, never block session end

    await bundle.mcp_manager.close()
    await bundle.hook_executor.execute(
        HookEvent.SESSION_END,
        {"cwd": bundle.cwd, "event": HookEvent.SESSION_END.value},
    )
    close_api_client = getattr(bundle.api_client, "close", None)
    if close_api_client is not None:
        await close_api_client()


def _last_user_text(messages: list[ConversationMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.text.strip():
            return msg.text.strip()
    return ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _format_pending_tool_results(messages: list[ConversationMessage]) -> str | None:
    """Render a compact summary when we stop after tool execution but before the follow-up model turn."""
    if not messages:
        return None

    last = messages[-1]
    if last.role != "user":
        return None
    tool_results = [block for block in last.content if isinstance(block, ToolResultBlock)]
    if not tool_results:
        return None

    tool_uses_by_id: dict[str, ToolUseBlock] = {}
    assistant_text = ""
    for msg in reversed(messages[:-1]):
        if msg.role != "assistant":
            continue
        if not msg.tool_uses:
            continue
        assistant_text = msg.text.strip()
        for tu in msg.tool_uses:
            tool_uses_by_id[tu.id] = tu
        break

    lines: list[str] = [
        "Pending continuation: tool results were produced, but the model did not get a chance to respond yet."
    ]
    if assistant_text:
        lines.append(f"Last assistant message: {_truncate(assistant_text, 400)}")

    max_results = 3
    for tr in tool_results[:max_results]:
        tu = tool_uses_by_id.get(tr.tool_use_id)
        if tu is not None:
            raw_input = json.dumps(tu.input, ensure_ascii=True, sort_keys=True)
            lines.append(
                f"- {tu.name} {_truncate(raw_input, 200)} -> {_truncate(tr.content.strip(), 400)}"
            )
        else:
            lines.append(
                f"- tool_result[{tr.tool_use_id}] -> {_truncate(tr.content.strip(), 400)}"
            )

    if len(tool_results) > max_results:
        lines.append(f"(+{len(tool_results) - max_results} more tool results)")

    lines.append("To continue from these results, run: /continue [COUNT].")
    return "\n".join(lines)


def sync_app_state(bundle: RuntimeBundle) -> None:
    """Refresh UI state from current settings and dynamic keybindings."""
    settings = bundle.current_settings()
    if bundle.enforce_max_turns:
        bundle.engine.set_max_turns(settings.max_turns)
    provider = detect_provider(settings)
    bundle.app_state.set(
        model=settings.model,
        permission_mode=settings.permission.mode.value,
        theme=settings.theme,
        cwd=bundle.cwd,
        provider=provider.name,
        auth_status=auth_status(settings),
        base_url=settings.base_url or "",
        vim_enabled=settings.vim_mode,
        voice_enabled=settings.voice_mode,
        voice_available=provider.voice_supported,
        voice_reason=provider.voice_reason,
        fast_mode=settings.fast_mode,
        effort=settings.effort,
        passes=settings.passes,
        mcp_connected=sum(1 for status in bundle.mcp_manager.list_statuses() if status.state == "connected"),
        mcp_failed=sum(1 for status in bundle.mcp_manager.list_statuses() if status.state == "failed"),
        bridge_sessions=len(get_bridge_manager().list_sessions()),
        output_style=settings.output_style,
        keybindings=load_keybindings(),
    )


def refresh_runtime_client(bundle: RuntimeBundle) -> None:
    """Refresh the active runtime client after provider/auth/profile changes."""
    settings = bundle.current_settings()
    if not bundle.external_api_client:
        bundle.api_client = _resolve_api_client_from_settings(settings)
        bundle.engine.set_api_client(bundle.api_client)
        bundle.hook_executor.update_context(
            api_client=bundle.api_client,
            default_model=settings.model,
        )
    bundle.engine.set_model(settings.model)
    bundle.engine.set_effort(settings.effort)
    bundle.engine.set_permission_checker(build_permission_checker(settings, cwd=str(bundle.engine.cwd)))
    system_prompt = build_runtime_system_prompt(
        settings,
        cwd=bundle.cwd,
        latest_user_prompt=_last_user_text(bundle.engine.messages),
        extra_skill_dirs=bundle.extra_skill_dirs,
        extra_plugin_roots=bundle.extra_plugin_roots,
        include_project_memory=bundle.include_project_memory,
    )
    bundle.engine.set_system_prompt(system_prompt)
    sync_app_state(bundle)


async def handle_line(
    bundle: RuntimeBundle,
    line: str,
    *,
    print_system: SystemPrinter,
    render_event: StreamRenderer,
    clear_output: ClearHandler,
    user_message: ConversationMessage | None = None,
) -> bool:
    """Handle one submitted line for either headless or TUI rendering."""
    if not bundle.external_api_client:
        bundle.hook_executor.update_registry(
            load_hook_registry(bundle.current_settings(), bundle.current_plugins())
        )

    command_context = CommandContext(
        engine=bundle.engine,
        hooks_summary=bundle.hook_summary(),
        mcp_summary=bundle.mcp_summary(),
        plugin_summary=bundle.plugin_summary(),
        cwd=bundle.cwd,
        tool_registry=bundle.tool_registry,
        app_state=bundle.app_state,
        session_backend=bundle.session_backend,
        session_id=bundle.session_id,
        extra_skill_dirs=bundle.extra_skill_dirs,
        extra_plugin_roots=bundle.extra_plugin_roots,
        memory_backend=bundle.memory_backend,
        include_project_memory=bundle.include_project_memory,
    )
    parsed = None if user_message is not None else (
        bundle.commands.lookup(line) or lookup_skill_slash_command(line, command_context)
    )
    if parsed is not None:
        command, args = parsed
        result = await command.handler(
            args,
            command_context,
        )
        if result.refresh_runtime:
            refresh_runtime_client(bundle)
        await _render_command_result(result, print_system, clear_output, render_event)
        if result.submit_prompt is not None:
            original_model = bundle.engine.model
            if result.submit_model:
                bundle.engine.set_model(result.submit_model)
            settings = bundle.current_settings()
            submit_prompt = result.submit_prompt
            system_prompt = build_runtime_system_prompt(
                settings,
                cwd=bundle.cwd,
                latest_user_prompt=submit_prompt,
                extra_skill_dirs=bundle.extra_skill_dirs,
                extra_plugin_roots=bundle.extra_plugin_roots,
                include_project_memory=bundle.include_project_memory,
            )
            bundle.engine.set_system_prompt(system_prompt)
            try:
                async for event in bundle.engine.submit_message(submit_prompt):
                    await render_event(event)
            except MaxTurnsExceeded as exc:
                await print_system(f"Stopped after {exc.max_turns} turns (max_turns).")
                pending = _format_pending_tool_results(bundle.engine.messages)
                if pending:
                    await print_system(pending)
            finally:
                if result.submit_model:
                    bundle.engine.set_model(original_model)
            bundle.session_backend.save_snapshot(
                cwd=bundle.cwd,
                model=bundle.engine.model,
                system_prompt=system_prompt,
                messages=bundle.engine.messages,
                usage=bundle.engine.total_usage,
                session_id=bundle.session_id,
                tool_metadata=bundle.engine.tool_metadata,
            )
        if result.continue_pending:
            settings = bundle.current_settings()
            if bundle.enforce_max_turns:
                bundle.engine.set_max_turns(settings.max_turns)
            system_prompt = build_runtime_system_prompt(
                settings,
                cwd=bundle.cwd,
                latest_user_prompt=_last_user_text(bundle.engine.messages),
                extra_skill_dirs=bundle.extra_skill_dirs,
                extra_plugin_roots=bundle.extra_plugin_roots,
                include_project_memory=bundle.include_project_memory,
            )
            bundle.engine.set_system_prompt(system_prompt)
            turns = result.continue_turns if result.continue_turns is not None else bundle.engine.max_turns
            try:
                async for event in bundle.engine.continue_pending(max_turns=turns):
                    await render_event(event)
            except MaxTurnsExceeded as exc:
                await print_system(f"Stopped after {exc.max_turns} turns (max_turns).")
                pending = _format_pending_tool_results(bundle.engine.messages)
                if pending:
                    await print_system(pending)
            bundle.session_backend.save_snapshot(
                cwd=bundle.cwd,
                model=settings.model,
                system_prompt=system_prompt,
                messages=bundle.engine.messages,
                usage=bundle.engine.total_usage,
                session_id=bundle.session_id,
                tool_metadata=bundle.engine.tool_metadata,
            )
        sync_app_state(bundle)
        return not result.should_exit

    settings = bundle.current_settings()
    if bundle.enforce_max_turns:
        bundle.engine.set_max_turns(settings.max_turns)
    latest_user_prompt = line or (user_message.text if user_message is not None else "")
    system_prompt = build_runtime_system_prompt(
        settings,
        cwd=bundle.cwd,
        latest_user_prompt=latest_user_prompt,
        extra_skill_dirs=bundle.extra_skill_dirs,
        extra_plugin_roots=bundle.extra_plugin_roots,
        include_project_memory=bundle.include_project_memory,
    )
    bundle.engine.set_system_prompt(system_prompt)
    try:
        async for event in bundle.engine.submit_message(user_message or line):
            await render_event(event)
    except MaxTurnsExceeded as exc:
        await print_system(f"Stopped after {exc.max_turns} turns (max_turns).")
        pending = _format_pending_tool_results(bundle.engine.messages)
        if pending:
            await print_system(pending)
        bundle.session_backend.save_snapshot(
            cwd=bundle.cwd,
            model=settings.model,
            system_prompt=system_prompt,
            messages=bundle.engine.messages,
            usage=bundle.engine.total_usage,
            session_id=bundle.session_id,
            tool_metadata=bundle.engine.tool_metadata,
        )
        sync_app_state(bundle)
        return True
    bundle.session_backend.save_snapshot(
        cwd=bundle.cwd,
        model=settings.model,
        system_prompt=system_prompt,
        messages=bundle.engine.messages,
        usage=bundle.engine.total_usage,
        session_id=bundle.session_id,
        tool_metadata=bundle.engine.tool_metadata,
    )
    sync_app_state(bundle)
    return True


async def _render_command_result(
    result: CommandResult,
    print_system: SystemPrinter,
    clear_output: ClearHandler,
    render_event: StreamRenderer | None = None,
) -> None:
    if result.clear_screen:
        await clear_output()
    if result.replay_messages and render_event is not None:
        # Replay restored conversation messages as transcript events
        from openharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete
        from openharness.api.usage import UsageSnapshot

        await clear_output()
        await print_system("Session restored:")
        for msg in result.replay_messages:
            if msg.role == "user":
                await print_system(f"> {msg.text}")
            elif msg.role == "assistant" and msg.text.strip():
                await render_event(AssistantTextDelta(text=msg.text))
                await render_event(AssistantTurnComplete(message=msg, usage=UsageSnapshot()))
    if result.message and not result.replay_messages:
        await print_system(result.message)
