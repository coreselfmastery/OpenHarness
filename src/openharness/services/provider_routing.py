"""Multi-provider routing with automatic fallback.

Implements optimization #4 from the top-10 list:
"Fast model for grunt work, powerful model for reasoning, automatic fallback."

When enabled, OpenHarness tries the configured primary provider first.
If the request fails with a retryable error (rate limit, 5xx, network
timeout) and a fallback provider is configured, the request is replayed
against the fallback. This is invisible to the rest of the engine —
the fallback client implements the same ``SupportsStreamingMessages``
protocol as the primary.

Typical usage (Kaggle video-generation pipeline):
- Primary: Groq (llama-3.3-70b-versatile) — fastest, lowest TTFT.
- Fallback: DashScope (qwen-max) — slower but higher availability.
- Fallback 2: Together (meta-llama/Llama-3.3-70B-Instruct-Turbo).

Configuration:
    settings.provider_routing = {
        "primary": {"provider": "openai_compat", "base_url": "...", "model": "..."},
        "fallbacks": [
            {"provider": "openai_compat", "base_url": "...", "model": "..."},
        ],
    }
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from openharness.api.errors import OpenHarnessApiError, RateLimitFailure, RequestFailure

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderRoute:
    """A single provider entry in the routing chain."""

    name: str
    client: SupportsStreamingMessages
    # When this route's request fails, wait this long before trying the
    # next fallback. Default 0 means immediate fallback (no delay).
    delay_seconds: float = 0.0
    # When True, only use this route for the Nth attempt onward. Used to
    # skip slow providers on the first try.
    min_attempt: int = 1


@dataclass
class RoutingStats:
    """Counters for routing decisions across the session."""

    primary_attempts: int = 0
    primary_successes: int = 0
    fallback_attempts: int = 0
    fallback_successes: int = 0
    last_route_used: str = ""
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_attempts": self.primary_attempts,
            "primary_successes": self.primary_successes,
            "fallback_attempts": self.fallback_attempts,
            "fallback_successes": self.fallback_successes,
            "last_route_used": self.last_route_used,
            "last_error": self.last_error,
        }


class RoutedStreamingClient(SupportsStreamingMessages):
    """Streaming client that routes to primary, then fallbacks on failure.

    The wrapped clients are constructed once (at engine startup) and
    reused across all turns so connection pools are kept warm. On a
    primary failure, we wait ``delay_seconds`` (configurable per route)
    and then replay the request against the next fallback in the chain.
    If all routes fail, the last error is raised.

    The caller sees the merged event stream as if a single provider
    served it — there is no API-visible difference between the primary
    and a successful fallback.
    """

    def __init__(self, primary: ProviderRoute, *fallbacks: ProviderRoute) -> None:
        if not primary.client:
            raise ValueError("primary client is required")
        self._routes: tuple[ProviderRoute, ...] = (primary, *fallbacks)
        self._stats = RoutingStats()

    @property
    def stats(self) -> RoutingStats:
        return self._stats

    @property
    def routes(self) -> tuple[ProviderRoute, ...]:
        return self._routes

    async def close(self) -> None:
        """Close all wrapped clients."""
        for route in self._routes:
            close = getattr(route.client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:
                    log.warning("error closing route %s: %s", route.name, exc)

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Stream the request through primary, falling back on retryable errors.

        Yields ``ApiRetryEvent`` events to the caller so the UI can show
        "falling back to <provider>" without losing the streaming
        semantics.
        """
        last_error: Exception | None = None
        for index, route in enumerate(self._routes):
            is_primary = index == 0
            if is_primary:
                self._stats.primary_attempts += 1
            else:
                self._stats.fallback_attempts += 1
                if route.delay_seconds > 0:
                    yield ApiRetryEvent(
                        message=f"Primary failed; falling back to {route.name}",
                        attempt=index,
                        max_attempts=len(self._routes),
                        delay_seconds=route.delay_seconds,
                    )
                    await asyncio.sleep(route.delay_seconds)
            try:
                async for event in route.client.stream_message(request):
                    yield event
                if is_primary:
                    self._stats.primary_successes += 1
                else:
                    self._stats.fallback_successes += 1
                self._stats.last_route_used = route.name
                self._stats.last_error = ""
                return  # Success — stop trying fallbacks.
            except OpenHarnessApiError as exc:
                last_error = exc
                self._stats.last_error = str(exc)
                if isinstance(exc, RateLimitFailure):
                    log.warning("route %s rate-limited; trying next", route.name)
                    continue
                if isinstance(exc, RequestFailure):
                    # Network / 5xx errors are retryable across providers.
                    log.warning("route %s failed (%s); trying next", route.name, exc)
                    continue
                # Authentication and other hard errors are not retryable.
                raise
            except Exception as exc:
                last_error = exc
                self._stats.last_error = str(exc)
                log.warning("route %s raised %s; trying next", route.name, type(exc).__name__)
                continue
        if last_error is not None:
            raise last_error


def build_routed_client_from_settings(settings: Any) -> RoutedStreamingClient | None:
    """Build a RoutedStreamingClient from ``settings.provider_routing``.

    Returns ``None`` when routing is not configured (single-provider
    mode). When routing is configured but a fallback client fails to
    construct, that fallback is skipped with a warning so the primary
    still works.
    """
    routing_config = getattr(settings, "provider_routing", None)
    if not routing_config:
        return None
    if not isinstance(routing_config, dict):
        log.warning("provider_routing must be a dict, got %r", type(routing_config))
        return None
    primary_cfg = routing_config.get("primary")
    if not primary_cfg:
        return None
    routes: list[ProviderRoute] = []
    primary = _build_route_from_config("primary", primary_cfg)
    if primary is None:
        return None
    routes.append(primary)
    for fallback_cfg in routing_config.get("fallbacks", []):
        if not isinstance(fallback_cfg, dict):
            continue
        name = fallback_cfg.get("name") or f"fallback_{len(routes)}"
        route = _build_route_from_config(name, fallback_cfg)
        if route is not None:
            routes.append(route)
    if len(routes) == 1:
        return None  # No fallbacks configured — no routing needed.
    return RoutedStreamingClient(*routes)


def _build_route_from_config(name: str, cfg: dict[str, Any]) -> ProviderRoute | None:
    """Construct a ProviderRoute from a config dict.

    The config shape mirrors ``ProviderProfile`` so users can copy a
    profile into the routing list:
        {"provider": "openai_compat", "api_format": "openai",
         "base_url": "...", "api_key": "...", "model": "..."}
    """
    from openharness.api.openai_client import OpenAICompatibleClient
    from openharness.api.client import AnthropicApiClient

    api_format = (cfg.get("api_format") or "openai").strip().lower()
    api_key = cfg.get("api_key") or cfg.get("auth_token") or ""
    base_url = cfg.get("base_url")
    model = cfg.get("model")  # Used for logging only — the actual model
                              # comes from the ApiMessageRequest.
    if not api_key:
        log.warning("route %s has no api_key; skipping", name)
        return None
    delay = float(cfg.get("delay_seconds", 0.0) or 0.0)
    if api_format in ("openai", "openai_compat"):
        client = OpenAICompatibleClient(api_key=api_key, base_url=base_url)
    elif api_format == "anthropic":
        client = AnthropicApiClient(api_key=api_key, base_url=base_url)
    else:
        log.warning("route %s has unsupported api_format %r; skipping", name, api_format)
        return None
    return ProviderRoute(name=name, client=client, delay_seconds=delay)
