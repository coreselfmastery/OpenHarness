"""Anthropic API client wrapper with retry logic."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from anthropic import APIError, APIStatusError, AsyncAnthropic

from openharness.api.errors import (
    AuthenticationFailure,
    OpenHarnessApiError,
    RateLimitFailure,
    RequestFailure,
)
from openharness.auth.external import (
    claude_attribution_header,
    claude_oauth_betas,
    claude_oauth_headers,
    get_claude_code_session_id,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, assistant_message_from_api

log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
OAUTH_BETA_HEADER = "oauth-2025-04-20"


@dataclass(frozen=True)
class ApiMessageRequest:
    """Input parameters for a model invocation."""

    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None = None
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    effort: str | None = None
    # Optimization #9: per-call reasoning budget for thinking models.
    thinking_budget: int | None = None
    # Item #2 (prefix caching): when True, Anthropic-format requests add
    # ``cache_control`` breakpoints on the system prompt and the tool
    # list so repeated prefills hit the provider KV cache (30-50% cheaper
    # and faster prefill). OpenAI-format clients ignore this flag — those
    # providers cache prefixes server-side automatically.
    prefix_caching: bool = False


@dataclass(frozen=True)
class ApiTextDeltaEvent:
    """Incremental text produced by the model."""

    text: str


@dataclass(frozen=True)
class ApiMessageCompleteEvent:
    """Terminal event containing the full assistant message."""

    message: ConversationMessage
    usage: UsageSnapshot
    stop_reason: str | None = None


@dataclass(frozen=True)
class ApiRetryEvent:
    """A recoverable upstream failure that will be retried automatically."""

    message: str
    attempt: int
    max_attempts: int
    delay_seconds: float


ApiStreamEvent = ApiTextDeltaEvent | ApiMessageCompleteEvent | ApiRetryEvent


class SupportsStreamingMessages(Protocol):
    """Protocol used by the query engine in tests and production."""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield streamed events for the request."""


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, APIError):
        return True  # Network errors are retryable
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _get_retry_delay(attempt: int, exc: Exception | None = None) -> float:
    """Calculate delay with exponential backoff and jitter."""
    import random

    # Check for Retry-After header
    if isinstance(exc, APIStatusError):
        retry_after = getattr(exc, "headers", {})
        if hasattr(retry_after, "get"):
            val = retry_after.get("retry-after")
            if val:
                try:
                    return min(float(val), MAX_DELAY)
                except (ValueError, TypeError):
                    pass

    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter


class AnthropicApiClient:
    """Thin wrapper around the Anthropic async SDK with retry logic."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        auth_token: str | None = None,
        base_url: str | None = None,
        claude_oauth: bool = False,
        auth_token_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._auth_token = auth_token
        self._base_url = base_url
        self._claude_oauth = claude_oauth
        self._auth_token_resolver = auth_token_resolver
        self._session_id = get_claude_code_session_id() if claude_oauth else ""
        self._client = self._create_client()

    def _create_client(self) -> AsyncAnthropic:
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._auth_token:
            kwargs["auth_token"] = self._auth_token
            kwargs["default_headers"] = (
                claude_oauth_headers()
                if self._claude_oauth
                else {"anthropic-beta": OAUTH_BETA_HEADER}
            )
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncAnthropic(**kwargs)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    def _refresh_client_auth(self) -> None:
        if not self._claude_oauth or self._auth_token_resolver is None:
            return
        next_token = self._auth_token_resolver()
        if next_token and next_token != self._auth_token:
            self._auth_token = next_token
            self._client = self._create_client()

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield text deltas and the final assistant message with retry on transient errors."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                self._refresh_client_auth()
                async for event in self._stream_once(request):
                    yield event
                return  # Success
            except OpenHarnessApiError:
                raise  # Auth errors are not retried
            except Exception as exc:
                last_error = exc
                if attempt >= MAX_RETRIES or not _is_retryable(exc):
                    if isinstance(exc, APIError):
                        raise _translate_api_error(exc) from exc
                    raise RequestFailure(str(exc)) from exc

                delay = _get_retry_delay(attempt, exc)
                status = getattr(exc, "status_code", "?")
                log.warning(
                    "API request failed (attempt %d/%d, status=%s), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, status, delay, exc,
                )
                yield ApiRetryEvent(
                    message=str(exc),
                    attempt=attempt + 1,
                    max_attempts=MAX_RETRIES + 1,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            if isinstance(last_error, APIError):
                raise _translate_api_error(last_error) from last_error
            raise RequestFailure(str(last_error)) from last_error

    async def _stream_once(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt at streaming a message."""
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_api_param() for message in request.messages],
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            params["system"] = request.system_prompt
        if self._claude_oauth:
            attribution = claude_attribution_header()
            params["system"] = (
                f"{attribution}\n{params['system']}"
                if params.get("system")
                else attribution
            )
        # Item #2 — Anthropic prompt caching. Convert the plain string
        # system prompt into a block list whose final block carries an
        # ``ephemeral`` cache breakpoint. The provider then serves cached
        # prefill tokens on subsequent turns (reported back through
        # ``usage.cache_read_input_tokens``).
        if request.prefix_caching and params.get("system"):
            params["system"] = [
                {
                    "type": "text",
                    "text": params["system"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if request.tools:
            params["tools"] = request.tools
            # Item #2 (cont.): the tool schema list is also stable across
            # turns, so the last tool definition gets the second (and,
            # per Anthropic limits, final) cache breakpoint. Tools are
            # emitted before messages in the request, so this caches the
            # full system + tools prefix.
            if request.prefix_caching:
                tools_copy = [dict(tool) for tool in params["tools"]]
                last = dict(tools_copy[-1])
                last["cache_control"] = {"type": "ephemeral"}
                tools_copy[-1] = last
                params["tools"] = tools_copy
        # Optimization #9: forward thinking_budget to Anthropic.
        # Claude 3.5+ supports extended thinking via a ``thinking`` field
        # carrying ``{"type": "enabled", "budget_tokens": N}``. We map
        # the integer ``thinking_budget`` to that field when it is set.
        # Older models that don't support thinking will reject the field,
        # so we only set it for claude-3-5-sonnet / claude-4-* / opus-4-*.
        if request.thinking_budget is not None and request.thinking_budget > 0:
            model_lower = (request.model or "").lower()
            if _model_supports_anthropic_thinking(model_lower):
                # Anthropic constraints:
                # - budget_tokens >= 1024 (minimum)
                # - budget_tokens < max_tokens
                # We clamp to satisfy both. If max_tokens < 2048, we
                # can't satisfy the minimum, so we skip thinking entirely
                # (better to run without thinking than to 400 the request).
                min_budget = 1024
                max_allowed = request.max_tokens - 1
                if max_allowed < min_budget:
                    # max_tokens too small for thinking — skip.
                    pass
                else:
                    budget = max(min_budget, min(request.thinking_budget, max_allowed))
                    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if self._claude_oauth:
            params["betas"] = claude_oauth_betas()
            params["metadata"] = {
                "user_id": json.dumps(
                    {
                        "device_id": "openharness",
                        "session_id": self._session_id,
                        "account_uuid": "",
                    },
                    separators=(",", ":"),
                )
            }
            params["extra_headers"] = {"x-client-request-id": str(uuid.uuid4())}

        try:
            stream_api = self._client.beta.messages if self._claude_oauth else self._client.messages
            async with stream_api.stream(**params) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    text = getattr(delta, "text", "")
                    if text:
                        yield ApiTextDeltaEvent(text=text)

                final_message = await stream.get_final_message()
        except APIError as exc:
            if isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES:
                raise  # Let retry logic handle it
            raise _translate_api_error(exc) from exc

        usage = getattr(final_message, "usage", None)
        yield ApiMessageCompleteEvent(
            message=assistant_message_from_api(final_message),
            usage=UsageSnapshot(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                cache_write_tokens=int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            ),
            stop_reason=getattr(final_message, "stop_reason", None),
        )


# Anthropic models that support extended thinking via the ``thinking`` field.
# Listing is conservative — Claude 3.5 Sonnet v2 + Claude 4 series. Older
# models reject the field with a 400, so we omit it for them.
_ANTHROPIC_THINKING_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "claude-3-7-sonnet",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-haiku-4",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
)


def _model_supports_anthropic_thinking(model_lower: str) -> bool:
    """Return True if the model accepts the ``thinking`` field."""
    # Strip provider prefix like "anthropic/claude-..."
    if "/" in model_lower:
        model_lower = model_lower.rsplit("/", 1)[-1]
    return any(model_lower.startswith(prefix) for prefix in _ANTHROPIC_THINKING_MODEL_PREFIXES)


def _translate_api_error(exc: APIError) -> OpenHarnessApiError:
    name = exc.__class__.__name__
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return AuthenticationFailure(str(exc))
    if name == "RateLimitError":
        return RateLimitFailure(str(exc))
    return RequestFailure(str(exc))
