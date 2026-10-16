"""LLM inference pipeline — items #1-#6 and #18 (inference speed).

This module wraps any ``SupportsStreamingMessages`` client with the
inference-speed features, composed in this order (outermost first):

1. ``InferencePipelineClient`` — semantic cache (#18), prompt
   compression (#3), prefix-caching flag (#2), usage/cost observation
   (#28), speculative/MTP capability reporting (#1/#5).
2. ``RacingClient`` (#4) — race 2-3 providers, keep the first response.
3. ``KeyRotatingClient`` (#6) — rotate a pool of API keys per request
   and on rate limits.
4. The real provider client.

Everything is opt-in via ``settings.inference`` (see
``openharness.config.settings.InferenceSettings``) except prefix caching,
which is on by default and only affects Anthropic-format requests.

Composition happens in :func:`wrap_client_with_inference_pipeline`,
called by the runtime when the engine's API client is built.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from openharness.api.errors import RateLimitFailure
from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock

log = logging.getLogger(__name__)


# ======================================================================
# #3 — Prompt compression
# ======================================================================

class PromptCompressor:
    """Strip redundant content from old conversation turns.

    Real, lossy-but-safe compression applied ONLY to messages older than
    ``keep_turns`` recent messages:

    - ``ToolResultBlock`` bodies longer than ``tool_result_cap`` chars are
      replaced by a one-line stub (first 100 chars + original length).
      Tool RESULTS are the dominant token sink in agent loops.
    - runs of 3+ blank lines in old text blocks collapse to one;
      trailing whitespace is stripped.

    The last ``keep_turns`` messages are passed through byte-for-byte, as
    is every assistant message containing tool calls (dropping those
    would corrupt tool_use/tool_result pairing). The compressor never
    mutates the input list — it builds a copy.
    """

    def __init__(
        self,
        *,
        min_chars: int = 6000,
        keep_turns: int = 3,
        tool_result_cap: int = 1200,
    ) -> None:
        self.min_chars = min_chars
        self.keep_turns = keep_turns
        self.tool_result_cap = tool_result_cap
        self.requests_compressed = 0
        self.chars_saved = 0

    def _estimate_chars(self, messages: list[ConversationMessage]) -> int:
        total = 0
        for message in messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    total += len(block.text)
                elif isinstance(block, ToolResultBlock):
                    total += len(block.content)
        return total

    def compress(self, messages: list[ConversationMessage]) -> list[ConversationMessage]:
        """Return a compressed copy of ``messages`` (input untouched)."""
        if len(messages) <= self.keep_turns:
            return messages
        if self._estimate_chars(messages) < self.min_chars:
            return messages

        protected = len(messages) - self.keep_turns
        out: list[ConversationMessage] = []
        saved = 0
        for index, message in enumerate(messages):
            if index >= protected:
                out.append(message)
                continue
            if message.role == "assistant" and message.tool_uses:
                out.append(message)  # keep pairing intact
                continue
            new_blocks: list[Any] = []
            changed = False
            for block in message.content:
                if isinstance(block, ToolResultBlock) and len(block.content) > self.tool_result_cap:
                    stub = (
                        f"[tool result elided by prompt compression: "
                        f"{block.content[:100].strip()!r}... "
                        f"({len(block.content)} chars total)]"
                    )
                    saved += len(block.content) - len(stub)
                    new_blocks.append(
                        ToolResultBlock(
                            tool_use_id=block.tool_use_id,
                            content=stub,
                            is_error=block.is_error,
                        )
                    )
                    changed = True
                elif isinstance(block, TextBlock):
                    compressed = re.sub(r"\n{3,}", "\n\n", block.text).rstrip()
                    if compressed != block.text:
                        saved += len(block.text) - len(compressed)
                        new_blocks.append(TextBlock(text=compressed))
                        changed = True
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            if changed:
                out.append(ConversationMessage(role=message.role, content=new_blocks))
            else:
                out.append(message)
        if saved > 0:
            self.requests_compressed += 1
            self.chars_saved += saved
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "requests_compressed": self.requests_compressed,
            "chars_saved": self.chars_saved,
        }


# ======================================================================
# #18 — Semantic cache (MinHash + Jaccard similarity, SQLite-backed)
# ======================================================================

_MERSENNE = (1 << 61) - 1


def _minhash_signature(text: str, *, hash_count: int = 64, seed: int = 0x9E3779B9) -> list[int]:
    """Deterministic MinHash signature over word 3-grams.

    A classic locality-sensitive sketch: 64 independent hash functions,
    each taking the minimum hash of every shingle. Similar texts produce
    similar signatures (estimated Jaccard = matching components / 64).
    Pure Python, no dependencies, ~microseconds for prompt-sized text.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < 3:
        # Too short to shingle meaningfully — fall back to word 1-grams
        # duplicated so similar short prompts still compare sanely.
        shingles = [" ".join([w]) for w in words] or [text.lower().strip()]
    else:
        shingles = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]

    signature: list[int] = []
    for func_index in range(hash_count):
        a = (seed * (func_index + 1) * 2 + 1) % _MERSENNE
        b = (seed + func_index * 1_114_111 + 7) % _MERSENNE
        minimum = _MERSENNE
        for shingle in shingles:
            h = (a * hash(shingle) + b) % _MERSENNE
            if h < minimum:
                minimum = h
        signature.append(minimum)
    return signature


def _signature_similarity(a: list[int], b: list[int]) -> float:
    """Estimated Jaccard similarity between two MinHash signatures."""
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


class SemanticCache:
    """Approximate-match response cache keyed by prompt meaning.

    - Lookup: exact-key fast path, then MinHash/Jaccard scan over stored
      signatures (linear over a bounded entry list — microseconds).
    - Only responses WITHOUT tool calls are cached and replayed; a tool
      call is context-sensitive and replaying it would be unsound.
    - Persisted in the shared SQLite store (#19), so hits survive
      session restarts.
    """

    NAMESPACE = "llm_semantic"

    def __init__(
        self,
        store: Any,
        *,
        threshold: float = 0.88,
        max_entries: int = 512,
    ) -> None:
        self._store = store
        self.threshold = threshold
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _request_text(request: ApiMessageRequest) -> str:
        parts: list[str] = []
        if request.system_prompt:
            parts.append(request.system_prompt)
        for message in request.messages:
            if message.role == "user":
                parts.append(message.text)
        return "\n".join(parts)

    def lookup(self, request: ApiMessageRequest) -> ApiMessageCompleteEvent | None:
        if self._store is None:
            return None
        text = self._request_text(request)
        if not text.strip():
            return None
        signature = _minhash_signature(text)
        best_key: str | None = None
        best_score = 0.0
        for key, entry in self._store.namespace_entries(self.NAMESPACE, self.max_entries).items():
            score = _signature_similarity(signature, entry.get("signature") or [])
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is not None and best_score >= self.threshold:
            entry = self._store.get(self.NAMESPACE, best_key) or {}
            payload = entry.get("response") or {}
            message_data = payload.get("message")
            usage_data = payload.get("usage") or {}
            if not message_data:
                return None
            try:
                from openharness.api.usage import UsageSnapshot

                message = ConversationMessage.model_validate(message_data)
                if message.tool_uses:
                    return None  # never replay tool calls
                self.hits += 1
                return ApiMessageCompleteEvent(
                    message=message,
                    usage=UsageSnapshot.model_validate(usage_data),
                    stop_reason=payload.get("stop_reason"),
                )
            except Exception:
                return None
        self.misses += 1
        return None

    def store_response(self, request: ApiMessageRequest, event: ApiMessageCompleteEvent) -> None:
        if self._store is None or event.message.tool_uses:
            return
        text = self._request_text(request)
        if not text.strip():
            return
        key = f"{hash(text) & 0xFFFFFFFF:08x}"
        payload = {
            "signature": _minhash_signature(text),
            "created_at": time.time(),
            "model": request.model,
            "response": {
                "message": event.message.model_dump(),
                "usage": event.usage.model_dump(),
                "stop_reason": event.stop_reason,
            },
        }
        self._store.set(self.NAMESPACE, key, payload)
        self._store.prune_namespace(self.NAMESPACE, self.max_entries)

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "threshold": self.threshold,
            "max_entries": self.max_entries,
        }


# ======================================================================
# #6 — API key rotation pool
# ======================================================================

class KeyRotatingClient(SupportsStreamingMessages):
    """Rotate across N API keys to multiply provider rate limits.

    Round-robins a key per request (spreading quota across accounts) and,
    when the active key gets rate-limited (``RateLimitFailure``),
    immediately rotates to the next key and replays the request — once
    per key before giving up. Rotation counters are exposed via
    ``stats`` for observability.

    ``client_factory(key)`` builds a fresh client per key; the factory
    is the runtime's own client builder, so every pool client has the
    exact same configuration apart from the credential.
    """

    def __init__(self, keys: list[str], client_factory: Callable[[str], SupportsStreamingMessages]) -> None:
        if not keys:
            raise ValueError("key pool must not be empty")
        self._keys = list(keys)
        self._factory = client_factory
        self._clients: dict[int, SupportsStreamingMessages] = {}
        self._index = 0
        self.rotations_on_rate_limit = 0
        self.requests_served = 0

    def _client_for(self, index: int) -> SupportsStreamingMessages:
        client = self._clients.get(index)
        if client is None:
            client = self._factory(self._keys[index])
            self._clients[index] = client
        return client

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        attempts = min(len(self._keys), 3)
        last_error: Exception | None = None
        for attempt in range(attempts):
            index = (self._index + attempt) % len(self._keys)
            client = self._client_for(index)
            try:
                async for event in client.stream_message(request):
                    yield event
                # Success: advance the round-robin past the used key.
                self._index = (index + 1) % len(self._keys)
                self.requests_served += 1
                return
            except RateLimitFailure as exc:
                last_error = exc
                self.rotations_on_rate_limit += 1
                log.warning(
                    "key pool: key #%d rate-limited; rotating (attempt %d/%d)",
                    index,
                    attempt + 1,
                    attempts,
                )
                yield ApiRetryEvent(
                    message=f"API key #{index + 1} rate-limited; rotating to next key",
                    attempt=attempt,
                    max_attempts=attempts,
                    delay_seconds=0.0,
                )
                continue
        if last_error is not None:
            raise last_error

    async def close(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    def stats(self) -> dict[str, Any]:
        return {
            "pool_size": len(self._keys),
            "requests_served": self.requests_served,
            "rotations_on_rate_limit": self.rotations_on_rate_limit,
        }


# ======================================================================
# #4 — Cross-provider racing
# ======================================================================

class RacingClient(SupportsStreamingMessages):
    """Send the request to 2-3 providers at once; keep the first stream.

    Winner selection: the first route whose stream yields a TEXT DELTA
    or a COMPLETE event — i.e. the first provider actually producing
    content. ``ApiRetryEvent`` events from non-winning routes are
    suppressed (they are internal retry chatter), and losing routes are
    cancelled as soon as a winner is chosen. If the winner fails
    mid-stream, the race restarts once without the failed route; if
    every route fails, the last error is raised.

    Honest cost note: racing multiplies token cost by the number of
    raced routes for requests where losers were already billed. It is
    opt-in (``settings.inference.cross_provider_racing``) for exactly
    that reason.
    """

    def __init__(self, routes: list[tuple[str, SupportsStreamingMessages]]) -> None:
        if len(routes) < 2:
            raise ValueError("racing needs at least two routes")
        self._routes = list(routes)
        self.races_won: dict[str, int] = {name: 0 for name, _ in routes}
        self.races_run = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        remaining = list(self._routes)
        last_error: Exception | None = None
        reraced = False
        while len(remaining) >= 1:
            self.races_run += 1
            winner = await self._race_once(remaining, request)
            if isinstance(winner, BaseException):
                last_error = winner
                failed_idx = winner._failed_route_index  # type: ignore[attr-defined]
                if len(remaining) > 1:
                    remaining = [r for i, r in enumerate(remaining) if i != failed_idx]
                    if not reraced:
                        reraced = True
                        continue
                break
            name, stream = winner
            self.races_won[name] = self.races_won.get(name, 0) + 1
            async for event in stream:
                yield event
            return
        if last_error is not None:
            raise last_error

    async def _race_once(
        self,
        routes: list[tuple[str, SupportsStreamingMessages]],
        request: ApiMessageRequest,
    ) -> tuple[str, AsyncIterator[ApiStreamEvent]] | Exception:
        """Run one race. Returns (winner name, its stream) or an exception.

        The returned stream is a live async generator that the caller
        continues to consume; all losing pump tasks are cancelled before
        returning. A failure marker exception carries the index of the
        route that failed (set as ``_failed_route_index``).
        """
        queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        pending: set[int] = set()
        tasks: list[asyncio.Task[None]] = []

        async def _pump(index: int, client: SupportsStreamingMessages) -> None:
            try:
                async for event in client.stream_message(request):
                    await queue.put((index, event))
                await queue.put((index, _RaceDone()))
            except Exception as exc:  # noqa: BLE001 - forwarded to the race loop
                setattr(exc, "_failed_route_index", index)
                await queue.put((index, exc))

        for index, (_name, client) in enumerate(routes):
            pending.add(index)
            tasks.append(asyncio.create_task(_pump(index, client)))

        streams: dict[int, AsyncIterator[ApiStreamEvent]] = {}
        error: Exception | None = None
        try:
            while pending:
                index, payload = await queue.get()
                if isinstance(payload, _RaceDone):
                    pending.discard(index)
                    continue
                if isinstance(payload, BaseException):
                    pending.discard(index)
                    if error is None:
                        error = payload
                    if not pending:
                        break
                    continue
                # A real event arrived.
                if index not in streams:
                    streams[index] = _QueuedStream(queue, index, first=payload)
                if isinstance(payload, (ApiTextDeltaEvent, ApiMessageCompleteEvent)):
                    # Winner chosen: hand the triggering event to the
                    # winner's stream directly (NOT re-queued — a fast
                    # producer may have already enqueued later events, and
                    # a re-queue would reorder behind them), then cancel
                    # the losers.
                    for other in list(pending):
                        if other != index:
                            pending.discard(other)
                    for task_index, task in enumerate(tasks):
                        if task_index != index and not task.done():
                            task.cancel()
                    return routes[index][0], streams[index]
                # Retry chatter from a non-winning route — keep waiting.
            if error is not None:
                return error
            return RuntimeError("all racing routes finished without producing content")
        finally:
            if error is not None:
                for task in tasks:
                    if not task.done():
                        task.cancel()

    async def close(self) -> None:
        for _name, client in self._routes:
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    def stats(self) -> dict[str, Any]:
        return {"races_run": self.races_run, "races_won": dict(self.races_won)}


class _RaceDone:
    """Internal marker: a racing route's stream finished cleanly."""


class _QueuedStream:
    """Adapter that turns the shared race queue into a stream for the winner.

    Events for the winning route index are consumed from the queue in
    order; the winning producer task keeps feeding it until it finishes.
    """

    def __init__(
        self, queue: asyncio.Queue[tuple[int, Any]], index: int, *, first: Any = None
    ) -> None:
        self._queue = queue
        self._index = index
        self._first = first  # the event that won the race, delivered first
        self._done = False

    def __aiter__(self) -> AsyncIterator[ApiStreamEvent]:
        return self

    async def __anext__(self) -> ApiStreamEvent:
        if self._done:
            raise StopAsyncIteration
        if self._first is not None:
            payload, self._first = self._first, None
            if isinstance(payload, _RaceDone):
                self._done = True
                raise StopAsyncIteration
            if isinstance(payload, BaseException):
                self._done = True
                raise payload
            return payload
        while True:
            index, payload = await self._queue.get()
            if index != self._index:
                continue  # Suppress residual events from cancelled losers.
            if isinstance(payload, _RaceDone):
                self._done = True
                raise StopAsyncIteration
            if isinstance(payload, BaseException):
                self._done = True
                raise payload
            return payload


# ======================================================================
# #1/#5 — Speculative decoding & MTP capability reporting (honest)
# ======================================================================

def speculative_capability_report(
    provider: str = "",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Report speculative-decoding / multi-token-prediction capability.

    These are MODEL-SERVER techniques — a client cannot accelerate token
    generation it does not run (items #1 and #5). This function reports
    the honest capability picture so agents and users can act on it:

    - Groq: speculative decoding is built into the serving stack —
      already active, nothing to configure.
    - Self-hosted vLLM / TensorRT-LLM: must be enabled server-side
      (``--speculative-model`` / MTP heads). The ``server_config`` tool
      generates those flags from detected hardware.
    - Hosted OpenAI-compatible endpoints: opaque; assume not active.

    It never claims to have accelerated anything — it only measures and
    reports, which is the only real client-side behavior possible.
    """
    provider_l = (provider or "").strip().lower()
    url = (base_url or "").strip().lower()
    if "groq" in provider_l or "groq" in url:
        speculative = "active (native to Groq's serving stack)"
        mtp = "provider-managed (not client-configurable)"
        action = "No action needed — requests already benefit."
    elif url and ("localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url or ":8000" in url):
        speculative = "available but OFF unless the server was launched with --speculative-model"
        mtp = "available but OFF unless the server enables MTP heads"
        action = "Run the server_config tool to generate vLLM launch flags."
    else:
        speculative = "unknown (hosted endpoint; server-side setting)"
        mtp = "unknown (hosted endpoint; server-side setting)"
        action = "For self-hosted inference, use the server_config tool."
    return {
        "speculative_decoding": speculative,
        "multi_token_prediction": mtp,
        "recommended_action": action,
        "note": (
            "Speculative decoding and MTP are inference-server features. "
            "A client can only choose providers that have them (Groq does) "
            "or configure its own server (see server_config tool)."
        ),
    }


# ======================================================================
# #28 — Session cost tracker (feeds cost alerts)
# ======================================================================

# Rough USD per 1M tokens (in, out). Editable, best-effort — used for
# ALERTING, not billing. Unknown models fall back to mid-tier pricing.
DEFAULT_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (15.0, 75.0),
    "claude-haiku": (0.8, 4.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-5": (3.0, 15.0),
    "llama": (0.05, 0.08),
    "qwen": (0.4, 1.2),
    "deepseek": (0.3, 1.1),
    "kimi": (0.6, 2.5),
    "gemini-flash": (0.1, 0.4),
}


@dataclass
class CostAlert:
    """One cost-threshold event, surfaced by the engine as a status event."""

    level: str  # "warning" | "halt"
    message: str
    spent_usd: float
    budget_usd: float


class SessionCostTracker:
    """Estimates session cost from usage snapshots and raises threshold alerts.

    Cache-read tokens are priced at 10% of full input price (Anthropic's
    prompt-cache discount) so enabling prefix caching shows up as a real
    saving. Alerts fire once per threshold crossing, never repeatedly.
    """

    def __init__(self, *, budget_usd: float | None, warn_ratio: float = 0.8) -> None:
        self.budget_usd = budget_usd
        self.warn_ratio = warn_ratio
        self.spent_usd = 0.0
        self.calls = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self._warned = False
        self._halted = False
        self.on_alert: Callable[[CostAlert], None] | None = None

    def estimate_cost_usd(self, model: str, usage: Any) -> float:
        model_l = (model or "").lower()
        price_in, price_out = 3.0, 15.0
        for prefix, prices in DEFAULT_PRICE_TABLE.items():
            if prefix in model_l:
                price_in, price_out = prices
                break
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
        cost = (
            input_tokens * price_in / 1_000_000
            + cache_read * price_in * 0.1 / 1_000_000
            + cache_write * price_in * 1.25 / 1_000_000
            + output_tokens * price_out / 1_000_000
        )
        return cost

    def record(self, model: str, usage: Any) -> CostAlert | None:
        """Record one LLM call's usage. Returns an alert if a threshold was crossed."""
        self.calls += 1
        self.spent_usd += self.estimate_cost_usd(model, usage)
        self.cache_read_tokens += int(getattr(usage, "cache_read_tokens", 0) or 0)
        self.cache_write_tokens += int(getattr(usage, "cache_write_tokens", 0) or 0)
        if self.budget_usd is None or self.budget_usd <= 0:
            return None
        if self.spent_usd >= self.budget_usd:
            if not self._halted:
                self._halted = True
                alert = CostAlert(
                    level="halt",
                    message=(
                        f"Session budget exceeded: ${self.spent_usd:.2f} of "
                        f"${self.budget_usd:.2f}. Halt generation and checkpoint."
                    ),
                    spent_usd=self.spent_usd,
                    budget_usd=self.budget_usd,
                )
                if self.on_alert:
                    try:
                        self.on_alert(alert)
                    except Exception:
                        pass
                return alert
            return None
        if self.spent_usd >= self.budget_usd * self.warn_ratio and not self._warned:
            self._warned = True
            alert = CostAlert(
                level="warning",
                message=(
                    f"Cost warning: ${self.spent_usd:.2f} spent of "
                    f"${self.budget_usd:.2f} budget ({int(self.warn_ratio * 100)}%)."
                ),
                spent_usd=self.spent_usd,
                budget_usd=self.budget_usd,
            )
            if self.on_alert:
                try:
                    self.on_alert(alert)
                except Exception:
                    pass
            return alert
        return None

    @property
    def halted(self) -> bool:
        return self._halted

    def stats(self) -> dict[str, Any]:
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 4),
            "calls": self.calls,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "halted": self._halted,
        }


# ======================================================================
# The outermost pipeline client
# ======================================================================

@dataclass
class InferenceStats:
    """Counters describing what the pipeline actually did."""

    requests: int = 0
    semantic_cache_hits: int = 0
    compressed_requests: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    prefix_cached_requests: int = 0
    inner_stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "semantic_cache_hits": self.semantic_cache_hits,
            "compressed_requests": self.compressed_requests,
            "prefix_cached_requests": self.prefix_cached_requests,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "inner": self.inner_stats,
        }


class InferencePipelineClient(SupportsStreamingMessages):
    """Applies semantic caching, compression and prefix caching around a client.

    Order per request:

    1. Semantic cache lookup (#18) — on a hit, replay the stored
       response as a stream and skip the network entirely.
    2. Prompt compression (#3) — compress a COPY of old messages.
    3. Prefix caching (#2) — set ``prefix_caching`` on the request so
       Anthropic-format clients emit ``cache_control`` breakpoints.
    4. Forward to the inner client (possibly racing / key-rotating).
    5. Observe the final usage (cache tokens, cost) for stats + alerts.
    """

    def __init__(
        self,
        inner: SupportsStreamingMessages,
        *,
        semantic_cache: SemanticCache | None = None,
        compressor: PromptCompressor | None = None,
        prefix_caching: bool = False,
        cost_tracker: SessionCostTracker | None = None,
        usage_callback: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._inner = inner
        self._semantic_cache = semantic_cache
        self._compressor = compressor
        self._prefix_caching = prefix_caching
        self._cost_tracker = cost_tracker
        self._usage_callback = usage_callback
        self.stats = InferenceStats()

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.stats.requests += 1

        if self._semantic_cache is not None:
            cached = self._semantic_cache.lookup(request)
            if cached is not None:
                self.stats.semantic_cache_hits += 1
                if cached.message.text:
                    yield ApiTextDeltaEvent(text=cached.message.text)
                yield cached
                return

        if self._compressor is not None:
            compressed = self._compressor.compress(request.messages)
            if compressed is not request.messages:
                self.stats.compressed_requests += 1
                request = _replace_request(request, messages=compressed)

        if self._prefix_caching and not request.prefix_caching:
            self.stats.prefix_cached_requests += 1
            request = _replace_request(request, prefix_caching=True)

        final_event: ApiMessageCompleteEvent | None = None
        async for event in self._inner.stream_message(request):
            if isinstance(event, ApiMessageCompleteEvent):
                final_event = event
            yield event

        if final_event is not None:
            self.stats.cache_read_tokens += final_event.usage.cache_read_tokens
            self.stats.cache_write_tokens += final_event.usage.cache_write_tokens
            if self._semantic_cache is not None:
                self._semantic_cache.store_response(request, final_event)
            if self._usage_callback is not None:
                try:
                    self._usage_callback(request.model, final_event.usage)
                except Exception:
                    pass

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def _replace_request(request: ApiMessageRequest, **updates: Any) -> ApiMessageRequest:
    """Return a copy of the frozen request dataclass with fields replaced."""
    from dataclasses import replace

    return replace(request, **updates)


# ======================================================================
# Composition entry point (used by the runtime)
# ======================================================================

class ClientFactoryProtocol(Protocol):
    """Runtime callback that rebuilds the primary client with a new API key."""

    def __call__(self, api_key: str) -> SupportsStreamingMessages: ...


def wrap_client_with_inference_pipeline(
    settings: Any,
    primary: SupportsStreamingMessages,
    *,
    fallback_routes: list[tuple[str, SupportsStreamingMessages]] | None = None,
    client_factory: ClientFactoryProtocol | None = None,
) -> SupportsStreamingMessages:
    """Wrap ``primary`` with every enabled inference feature.

    Called by the runtime right after the primary client is built. All
    wrappers degrade to identity: when no feature is enabled the
    original client is returned untouched.
    """
    inference = getattr(settings, "inference", None)
    if inference is None:
        return primary

    inner = primary

    # #6 — key rotation pool.
    key_pool = [k for k in (getattr(inference, "api_key_pool", None) or []) if k.strip()]
    if len(key_pool) >= 1 and client_factory is not None:
        try:
            inner = KeyRotatingClient(key_pool, client_factory)
            log.info("key rotation enabled with %d keys", len(key_pool))
        except Exception as exc:
            log.warning("key rotation setup failed: %s", exc)

    # #4 — cross-provider racing across primary + fallback routes.
    if (
        getattr(inference, "cross_provider_racing", False)
        and fallback_routes
        and len(fallback_routes) >= 1
    ):
        try:
            routes = [("primary", inner)] + list(fallback_routes)
            inner = RacingClient(routes[:3])
            log.info("cross-provider racing enabled across %d routes", len(routes[:3]))
        except Exception as exc:
            log.warning("racing setup failed: %s", exc)

    # #18/#3/#2 — pipeline wrapper (only when at least one is on).
    semantic_cache = None
    if getattr(inference, "semantic_cache", False):
        try:
            from openharness.services.persistent_store import get_persistent_store

            semantic_cache = SemanticCache(
                get_persistent_store(),
                threshold=float(getattr(inference, "semantic_cache_threshold", 0.88) or 0.88),
                max_entries=int(getattr(inference, "semantic_cache_max_entries", 512) or 512),
            )
        except Exception as exc:
            log.warning("semantic cache setup failed: %s", exc)

    compressor = None
    if getattr(inference, "prompt_compression", False):
        compressor = PromptCompressor(
            min_chars=int(getattr(inference, "prompt_compression_min_chars", 6000) or 6000),
            keep_turns=int(getattr(inference, "prompt_compression_keep_turns", 3) or 3),
        )

    prefix_caching = bool(getattr(inference, "prefix_caching", False))
    if semantic_cache is None and compressor is None and not prefix_caching:
        return inner

    return InferencePipelineClient(
        inner,
        semantic_cache=semantic_cache,
        compressor=compressor,
        prefix_caching=prefix_caching,
    )
