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
                new_message = ConversationMessage(role=message.role, content=new_blocks)
                out.append(new_message)
                self._propagate_compressed_estimate(message, new_message)
            else:
                out.append(message)
        if saved > 0:
            self.requests_compressed += 1
            self.chars_saved += saved
        return out

    @staticmethod
    def _propagate_compressed_estimate(
        original: ConversationMessage, compressed: ConversationMessage
    ) -> None:
        """Publish the compressed token count onto the ORIGINAL message.

        The engine's auto-compact decision walks the original message
        list with ``compact.estimate_message_tokens``, whose per-message
        cache (``_oh_token_cache``) previously still held the
        UNCOMPRESSED count after the compressor shrank the outbound copy.
        The engine therefore believed the conversation was larger than
        the request it actually sends and compacted earlier than
        necessary. Stashing the compressed count — validated against the
        original block identity, exactly like the normal cache path —
        keeps the estimate in sync with what goes on the wire. An
        in-place mutation of the original blocks invalidates it and the
        next estimate recomputes from scratch (safe degradation).
        """
        try:
            from openharness.services.compact import (
                _estimate_single_message_tokens,
                _vision_token_budget_per_image,
            )

            budget = _vision_token_budget_per_image()
            new_tokens = _estimate_single_message_tokens(compressed, budget)
            object.__setattr__(
                original,
                "_oh_token_cache",
                (budget, new_tokens, tuple(original.content)),
            )
        except Exception:
            pass  # cache maintenance must never break compression

    def stats(self) -> dict[str, Any]:
        return {
            "requests_compressed": self.requests_compressed,
            "chars_saved": self.chars_saved,
        }


# ======================================================================
# #18 — Semantic cache (MinHash + Jaccard similarity, SQLite-backed)
# ======================================================================

_MERSENNE = (1 << 61) - 1


def _stable_shingle_hash(shingle: str, memo: dict[str, int]) -> int:
    """Stable (cross-process) 64-bit hash of one shingle.

    Python's builtin ``hash()`` is salted per process (PYTHONHASHSEED),
    so signatures built with it cannot be compared across sessions —
    exactly what the SQLite-persisted semantic cache needs to do.
    blake2b is stable and fast; results are memoized per call because
    the same shingles recur within one prompt.
    """
    cached = memo.get(shingle)
    if cached is not None:
        return cached
    import hashlib

    value = int.from_bytes(
        hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
        "big",
        signed=False,
    )
    memo[shingle] = value
    return value


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
    memo: dict[str, int] = {}
    for func_index in range(hash_count):
        a = (seed * (func_index + 1) * 2 + 1) % _MERSENNE
        b = (seed + func_index * 1_114_111 + 7) % _MERSENNE
        minimum = _MERSENNE
        for shingle in shingles:
            h = (a * _stable_shingle_hash(shingle, memo) + b) % _MERSENNE
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

    # Cap on how much of each tool result body feeds the similarity
    # text: enough to distinguish different results, not so much that
    # one huge result dominates the shingle set.
    _TOOL_RESULT_SIMILARITY_CAP = 2000

    @classmethod
    def _request_text(cls, request: ApiMessageRequest) -> str:
        """Similarity text = the USER messages, tool results included.

        The system prompt is deliberately excluded: it is long, largely
        stable across a session (environment, skills, brand anchors,
        memory sections), and would dominate the shingle set. Two
        different short questions sharing one long system prompt scored
        > 0.88 similarity and replayed the WRONG cached response. The
        user turns are what actually distinguishes one request from
        another.

        ``ToolResultBlock`` bodies in user messages are ALSO included:
        in the agent loop, consecutive requests share the same user
        text but differ in what the tools returned. Ignoring tool
        results made Turn 1 and Turn 2 produce identical fingerprints —
        a guaranteed false cache hit that replayed the answer to the
        *previous* tool state.
        """
        parts: list[str] = []
        for message in request.messages:
            if message.role != "user":
                continue
            parts.append(message.text)
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    body = block.content[: cls._TOOL_RESULT_SIMILARITY_CAP]
                    parts.append(f"[tool {block.tool_use_id}] {body}")
        return "\n".join(parts)

    @staticmethod
    def _stable_key(model: str, text: str) -> str:
        """Stable cache key (sha256 prefix) — survives process restarts.

        The model id is part of the key: switching models must never
        replay an answer produced by a different model.
        """
        import hashlib

        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()[:16]

    def lookup(self, request: ApiMessageRequest) -> ApiMessageCompleteEvent | None:
        if self._store is None:
            return None
        text = self._request_text(request)
        if not text.strip():
            return None
        # Short prompts ("ok", "continue", one-line scene requests) only
        # match when token-identical (similarity exactly 1.0 — punctuation
        # and whitespace differences still hit, but any wording change
        # misses). Longer prompts use the configured approximate
        # threshold.
        word_count = len(text.split())
        effective_threshold = 1.0 if word_count < 12 else self.threshold
        signature = _minhash_signature(text)
        best_key: str | None = None
        best_score = 0.0
        for key, entry in self._store.namespace_entries(self.NAMESPACE, self.max_entries).items():
            # Model scoping: an entry cached for another model is not a
            # valid answer for this request, no matter how similar the
            # prompts are.
            if str(entry.get("model") or "") != request.model:
                continue
            score = _signature_similarity(signature, entry.get("signature") or [])
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is not None and best_score >= effective_threshold:
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
        key = self._stable_key(request.model, text)
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

DEFAULT_KEY_COOLDOWN_SECONDS = 60.0
# When EVERY key in the pool is cooling down mid-request, wait ONLY for
# short recoveries (a burst of 429s where a key frees up in seconds).
# Longer waits are the engine retry layer's job — surfacing the
# rate-limit error immediately is better than parking a request inside
# the client for a full cooldown window.
_MAX_POOL_WAIT_SECONDS = 5.0


class KeyRotatingClient(SupportsStreamingMessages):
    """Rotate across N API keys to multiply provider rate limits.

    Design (10x-throughput key pool):

    - Round-robins a key per request, spreading quota across accounts:
      request 1 -> key A, request 2 -> key B, ...
    - When the active key is rate-limited (``RateLimitFailure``), it
      enters a cooldown (default 60s) keyed by a monotonic timestamp and
      the request is instantly replayed on the next HEALTHY key — every
      key in the pool is tried, not just a fixed 3.
    - ``_next_available_key`` always returns the next key whose cooldown
      has expired, skipping cooling keys proactively (they fail fast
      with 429s otherwise and just burn latency).
    - A key that succeeds is immediately healthy again (cooldown
      cleared); when ALL keys are cooling, the request waits for the
      earliest expiry (bounded by ``_MAX_POOL_WAIT_SECONDS``) and then
      retries rather than failing outright.
    - Cooldown duration is configurable (``key_pool_cooldown_seconds``,
      env ``OPENHARNESS_KEY_POOL_COOLDOWN_SECONDS``).

    Rotation/cooldown counters are exposed via ``stats`` for
    observability. ``client_factory(key)`` builds a fresh client per key
    (the runtime's own client builder), so every pool client has the
    exact same configuration apart from the credential.
    """

    def __init__(
        self,
        keys: list[str],
        client_factory: Callable[[str], SupportsStreamingMessages],
        *,
        cooldown_seconds: float = DEFAULT_KEY_COOLDOWN_SECONDS,
    ) -> None:
        if not keys:
            raise ValueError("key pool must not be empty")
        self._keys = list(keys)
        self._factory = client_factory
        self._clients: dict[int, SupportsStreamingMessages] = {}
        self._index = 0
        # key index -> monotonic time when its cooldown expires.
        self._cooldown_until: dict[int, float] = {}
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.rotations_on_rate_limit = 0
        self.cooldowns_taken = 0
        self.requests_served = 0
        self.pool_waits = 0

    def _client_for(self, index: int) -> SupportsStreamingMessages:
        client = self._clients.get(index)
        if client is None:
            client = self._factory(self._keys[index])
            self._clients[index] = client
        return client

    def _next_available_key(self, exclude: set[int]) -> tuple[int | None, float]:
        """Pick the next healthy key.

        Scans the pool round-robin from ``self._index`` and returns the
        first key that (a) was not already tried for this request and
        (b) has no active cooldown. When no such key exists, returns
        ``(None, seconds_until_earliest_cooldown_expiry)`` so the caller
        can decide to wait.
        """
        n = len(self._keys)
        now = time.monotonic()
        for offset in range(n):
            index = (self._index + offset) % n
            if index in exclude:
                continue
            if now >= self._cooldown_until.get(index, 0.0):
                return index, 0.0
        # No untried healthy key. Earliest cooldown expiry across ALL
        # keys (tried or not — an untried cooling key may expire first).
        soonest = min(range(n), key=lambda i: self._cooldown_until.get(i, float("inf")))
        wait = max(0.0, self._cooldown_until.get(soonest, 0.0) - now)
        return None, wait

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        total_keys = len(self._keys)
        last_error: Exception | None = None
        tried: set[int] = set()
        # Bound the loop: one attempt per key, plus at most one wait
        # round when the whole pool was cooling — a permanently
        # throttled provider must surface its rate-limit error, not
        # spin here forever.
        attempts_left = 2 * total_keys
        while attempts_left > 0:
            attempts_left -= 1
            index, wait = self._next_available_key(tried)
            if index is None:
                if wait <= 0 or wait > _MAX_POOL_WAIT_SECONDS:
                    # Nothing recoverable within the wait budget. NEVER
                    # end the stream silently — an empty response would
                    # look like a successful blank completion to the
                    # engine. Surface the provider error, or a pool-
                    # cooldown error when no key was even attempted.
                    if last_error is not None:
                        raise last_error
                    raise RateLimitFailure(
                        f"key pool: all {total_keys} keys are in cooldown "
                        f"(earliest recovery in {max(0.0, wait):.0f}s)"
                    )
                self.pool_waits += 1
                yield ApiRetryEvent(
                    message=(
                        f"key pool: all {total_keys} keys rate-limited; "
                        f"waiting {wait:.0f}s for the earliest cooldown"
                    ),
                    attempt=total_keys - len([i for i in tried if i is not None]),
                    max_attempts=total_keys,
                    delay_seconds=wait,
                )
                await asyncio.sleep(wait + 0.05)
                tried.clear()  # every key is eligible again after the wait
                continue
            client = self._client_for(index)
            try:
                async for event in client.stream_message(request):
                    yield event
                # Success: this key is healthy again, and the
                # round-robin resumes on the NEXT key.
                self._cooldown_until.pop(index, None)
                self._index = (index + 1) % total_keys
                self.requests_served += 1
                return
            except RateLimitFailure as exc:
                last_error = exc
                tried.add(index)
                if self._cooldown_seconds > 0:
                    self._cooldown_until[index] = time.monotonic() + self._cooldown_seconds
                    self.cooldowns_taken += 1
                self.rotations_on_rate_limit += 1
                log.warning(
                    "key pool: key #%d rate-limited; cooling down %.0fs and rotating to next healthy key",
                    index,
                    self._cooldown_seconds,
                )
                yield ApiRetryEvent(
                    message=f"API key #{index + 1} rate-limited; rotating to next key",
                    attempt=len(tried),
                    max_attempts=total_keys,
                    delay_seconds=0.0,
                )
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
        now = time.monotonic()
        cooling = [
            index
            for index, deadline in self._cooldown_until.items()
            if deadline > now
        ]
        return {
            "pool_size": len(self._keys),
            "requests_served": self.requests_served,
            "rotations_on_rate_limit": self.rotations_on_rate_limit,
            "cooldowns_taken": self.cooldowns_taken,
            "keys_in_cooldown": len(cooling),
            "pool_waits": self.pool_waits,
            "cooldown_seconds": self._cooldown_seconds,
        }


# ======================================================================
# #6b — Cost dashboard usage tracking
# ======================================================================

class UsageTrackingClient(SupportsStreamingMessages):
    """Record every completed LLM response into the cost dashboard.

    Sits directly around the provider client stack, so it sees exactly
    what actually went on the wire — requests that reached a provider
    and completed. Free replays (semantic-cache hits) never reach this
    layer and are correctly NOT billed; rate-limited key-pool attempts
    produce no complete event and cost nothing. Cross-provider racing
    bills the losing routes at their own providers — those tokens are
    invisible here (the winner's stream is the only one surfaced), which
    the racing feature's own stats call out honestly.

    Recording is synchronous in-memory aggregation + occasional SQLite
    flush (see :mod:`openharness.services.cost_dashboard`); it never
    blocks the stream and never raises into it.
    """

    def __init__(self, inner: SupportsStreamingMessages, dashboard: Any | None = None) -> None:
        self._inner = inner
        self._dashboard = dashboard

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        dashboard = self._dashboard
        async for event in self._inner.stream_message(request):
            if (
                dashboard is not None
                and type(event) is ApiMessageCompleteEvent
                and (event.usage.input_tokens or event.usage.output_tokens)
            ):
                try:
                    dashboard.record(
                        model=request.model,
                        input_tokens=event.usage.input_tokens,
                        output_tokens=event.usage.output_tokens,
                        cache_write_tokens=event.usage.cache_write_tokens,
                        cache_read_tokens=event.usage.cache_read_tokens,
                        source="pipeline",
                    )
                except Exception:  # noqa: BLE001 - tracking must never break a stream
                    log.exception("cost dashboard recording failed")
            yield event

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        # Transparent passthrough (e.g. stats() on RacingClient /
        # KeyRotatingClient for the cache_stats tool) so wrapping never
        # hides observability surface. The _inner guard prevents infinite
        # recursion when the attribute is accessed before __init__ ran.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


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

    def __init__(self, routes: list[tuple]) -> None:
        if len(routes) < 2:
            raise ValueError("racing needs at least two routes")
        # Routes are (name, client) or (name, client, model) tuples — the
        # third element is the route's own model. When set, each request
        # is rewritten per route so a fallback endpoint is never asked
        # for the primary's model name (it would 404).
        self._routes: list[tuple[str, SupportsStreamingMessages, str | None]] = [
            (r[0], r[1], (r[2] if len(r) > 2 else None)) for r in routes
        ]
        self.races_won: dict[str, int] = {name: 0 for name, _, _ in self._routes}
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

        async def _pump(index: int, client: SupportsStreamingMessages, route_model: str | None) -> None:
            try:
                from openharness.services.provider_routing import request_for_route

                route_request = request_for_route(request, route_model)
                async for event in client.stream_message(route_request):
                    await queue.put((index, event))
                await queue.put((index, _RaceDone()))
            except Exception as exc:  # noqa: BLE001 - forwarded to the race loop
                setattr(exc, "_failed_route_index", index)
                await queue.put((index, exc))

        for index, (_name, client, route_model) in enumerate(routes):
            pending.add(index)
            tasks.append(asyncio.create_task(_pump(index, client, route_model)))

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
        for _name, client, _model in self._routes:
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
# NOTE: pricing lives in one place — openharness.services.cost_dashboard
# (longest-prefix table, OPENHARNESS_PRICING_JSON overrides). The session
# cost tracker delegates to it so the two can never disagree.


@dataclass
class CostAlert:
    """One cost-threshold event, surfaced by the engine as a status event."""

    level: str  # "warning" | "halt"
    message: str
    spent_usd: float
    budget_usd: float


class SessionCostTracker:
    """Estimates session cost from usage snapshots and raises threshold alerts.

    Pricing is delegated to the cost dashboard's table (longest-prefix
    match, env-overridable) so the session tracker and the persistent
    dashboard can never disagree. Models missing from the table are
    priced with a conservative mid-tier fallback ($3/$15 per 1M) — this
    is a SAFETY control, and an unknown model name must never silently
    disable the budget halt (the dashboard still reports such models
    honestly as $0/unpriced). Cache-read tokens are priced at 10% of
    full input price (Anthropic's prompt-cache discount) so enabling
    prefix caching shows up as a real saving. Alerts fire once per
    threshold crossing, never repeatedly.
    """

    # Conservative pricing for models with no table entry. Historically
    # the tracker priced EVERYTHING at (3.0, 15.0) when unknown; keeping
    # that as the explicit fallback preserves the halt's guarantee.
    UNPRICED_FALLBACK = (3.0, 15.0)

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
        # Single source of truth: the cost dashboard's pricing table.
        from openharness.services.cost_dashboard import estimate_cost

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
        cost, _priced = estimate_cost(
            model,
            input_tokens,
            output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            unpriced_fallback=self.UNPRICED_FALLBACK,
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


def _dashboard_if_available() -> Any | None:
    """Return the process-wide cost dashboard, or None (never raises).

    Lazy import: cost_dashboard pulls in the persistent store; keeping it
    out of module import time avoids any import-cycle / startup cost.
    """
    try:
        from openharness.services.cost_dashboard import get_cost_dashboard

        return get_cost_dashboard()
    except Exception:  # noqa: BLE001 - observability must never break setup
        return None


def wrap_client_with_inference_pipeline(
    settings: Any,
    primary: SupportsStreamingMessages,
    *,
    fallback_routes: list[tuple] | None = None,
    client_factory: ClientFactoryProtocol | None = None,
) -> SupportsStreamingMessages:
    """Wrap ``primary`` with every enabled inference feature.

    Called by the runtime right after the primary client is built. All
    wrappers degrade to identity: when no feature is enabled the
    original client is returned untouched.

    ``fallback_routes`` entries are ``(name, client)`` or
    ``(name, client, model)`` tuples — the optional third element is the
    route's own model name, used to rewrite raced requests per route.
    """
    inference = getattr(settings, "inference", None)
    if inference is None:
        # No inference settings at all (bare/test setups): still track
        # usage when a dashboard is available — cost visibility is not
        # an opt-in performance feature.
        dashboard = _dashboard_if_available()
        return UsageTrackingClient(primary, dashboard) if dashboard is not None else primary

    inner = primary

    # #6 — key rotation pool.
    key_pool = [k for k in (getattr(inference, "api_key_pool", None) or []) if k.strip()]
    if len(key_pool) >= 1 and client_factory is not None:
        try:
            cooldown = float(
                getattr(inference, "key_pool_cooldown_seconds", DEFAULT_KEY_COOLDOWN_SECONDS)
                or DEFAULT_KEY_COOLDOWN_SECONDS
            )
            inner = KeyRotatingClient(key_pool, client_factory, cooldown_seconds=cooldown)
            log.info("key rotation enabled with %d keys (cooldown %.0fs)", len(key_pool), cooldown)
        except Exception as exc:
            log.warning("key rotation setup failed: %s", exc)

    # #4 — cross-provider racing across primary + fallback routes.
    if (
        getattr(inference, "cross_provider_racing", False)
        and fallback_routes
        and len(fallback_routes) >= 1
    ):
        try:
            routes = [("primary", inner, None)] + [tuple(r) for r in fallback_routes]
            inner = RacingClient(routes[:3])
            log.info("cross-provider racing enabled across %d routes", len(routes[:3]))
        except Exception as exc:
            log.warning("racing setup failed: %s", exc)

    # Cost dashboard (#28/#6-speed): record every completed response.
    # Always on — pure observability, no request-path cost beyond an
    # in-memory counter update. Sits ABOVE the transport layers (key
    # pool, racing, routing) so whichever key/route served the request
    # is recorded once, and BELOW the semantic cache so free cache
    # replays are not billed.
    dashboard = _dashboard_if_available()
    if dashboard is not None:
        inner = UsageTrackingClient(inner, dashboard)

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
