"""Cost dashboard — real-time per-call LLM cost tracking (speed feature #6).

Every completed LLM response passes through the inference pipeline wrapper
(``UsageTrackingClient``), which records (model, input tokens, output
tokens, cache tokens) here. The dashboard:

1. prices the call from a per-model pricing table (USD per 1M tokens),
2. keeps running aggregates — per model, per scene, per project — plus a
   ring buffer of recent calls for live inspection,
3. persists aggregates to the SQLite persistent store so totals survive
   session restarts and Kaggle 12-hour kills,
4. projects totals ("$0.47 so far, 12 scenes remaining, projected total
   $0.89") from scenes-done vs. scenes-total.

Pricing notes (honesty first):

- Prices are **estimates** — provider pricing changes frequently. The
  built-in table covers common Anthropic / OpenAI / Google / Groq /
  DeepSeek / Moonshot / Qwen model families with prefix matching
  (longest prefix wins). Unknown models price at 0 and are reported as
  "unpriced" so totals never silently lie.
- The table is user-overridable: ``OPENHARNESS_PRICING_JSON`` env var or
  ``settings.cost.pricing_overrides`` maps a model prefix to
  ``[input_usd_per_mtok, output_usd_per_mtok]``.
- Anthropic prompt-cache tokens are priced with the documented
  multipliers (cache reads 0.1x input, cache writes 1.25x input); other
  providers report cache reads as ordinary input tokens or not at all.

Recording never blocks the request path measurably: records update
in-memory counters, and SQLite is touched only by ``flush()`` — every
``flush_every`` records, at most once per ``flush_interval`` seconds,
and on session shutdown.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Pricing table (USD per 1M tokens: input, output). Longest matching
# prefix wins; match is case-insensitive on the model id.
# ----------------------------------------------------------------------

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic (2025 public pricing)
    "claude-opus-4": (15.0, 75.0),
    "claude-4-opus": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-4-5-sonnet": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-4-sonnet": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-4-5-haiku": (1.0, 5.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI
    "gpt-5": (1.25, 10.0),
    "gpt-4-1": (2.0, 8.0),
    "gpt-4-1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (2.0, 8.0),
    # Google
    "gemini-2-5-pro": (1.25, 10.0),
    "gemini-2-5-flash": (0.30, 2.50),
    "gemini-2-0-flash": (0.10, 0.40),
    "gemini-1-5-pro": (1.25, 5.0),
    "gemini-1-5-flash": (0.075, 0.30),
    # Groq (hosted pricing, per-1M tokens)
    "llama-3-3-70b": (0.59, 0.79),
    "llama-4-maverick": (0.62, 2.10),
    "llama-4-scout": (0.11, 0.34),
    "llama-3-1-8b": (0.05, 0.08),
    "qwen-qwen3-32b": (0.29, 0.59),
    "moonshotai-kimi-k2": (1.00, 3.00),
    "gpt-oss-120b": (0.15, 0.75),
    "gpt-oss-20b": (0.10, 0.50),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Moonshot official
    "kimi-k2": (0.60, 2.50),
    # Qwen (DashScope / OpenRouter rough)
    "qwen3-235b": (0.20, 1.20),
    "qwen2-5-72b": (0.35, 0.40),
    # Local/self-hosted — free by definition.
    "local": (0.0, 0.0),
}

# Multipliers applied to the input price for provider cache accounting.
_CACHE_READ_MULTIPLIER = 0.10  # Anthropic cache reads: 10% of input price
_CACHE_WRITE_MULTIPLIER = 1.25  # Anthropic cache writes: 125% of input price


def pricing_overrides_from_env() -> dict[str, tuple[float, float]]:
    """Parse ``OPENHARNESS_PRICING_JSON`` into {prefix: (in, out)}.

    Malformed JSON or shapes log a warning and are ignored — pricing
    must never crash the pipeline.
    """
    raw = os.environ.get("OPENHARNESS_PRICING_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("OPENHARNESS_PRICING_JSON is not valid JSON; ignored")
        return {}
    out: dict[str, tuple[float, float]] = {}
    if isinstance(data, dict):
        for prefix, pair in data.items():
            try:
                price_in, price_out = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                log.warning("pricing override %r has invalid pair %r; ignored", prefix, pair)
                continue
            out[str(prefix).lower()] = (price_in, price_out)
    return out


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    overrides: dict[str, tuple[float, float]] | None = None,
    unpriced_fallback: tuple[float, float] | None = None,
) -> tuple[float, bool]:
    """Estimate the USD cost of one call. Returns (cost, priced).

    ``priced`` is False when the model matched no pricing entry. Without
    a fallback the cost is then 0.0 and reporting surfaces the model as
    "unpriced" rather than showing a misleading $0 total.

    ``unpriced_fallback`` is an optional ``(input_usd_per_mtok,
    output_usd_per_mtok)`` pair priced INSTEAD OF zero when the model is
    unknown — meant for safety controls (the session budget halt) that
    must keep functioning on models the table has never seen. The
    dashboard itself never passes it (honest $0 + "unpriced"), so
    reports and halts stay honest respectively.
    """
    model_key = (model or "").strip().lower()
    if not model_key:
        return 0.0, False
    # Hot path: this runs on EVERY completed LLM response. Skip the
    # table copy entirely when there are no overrides (the common case).
    table: dict[str, tuple[float, float]]
    if overrides:
        table = dict(_DEFAULT_PRICING)
        table.update({k.lower(): v for k, v in overrides.items()})
    else:
        table = _DEFAULT_PRICING
    # Longest matching prefix wins.
    best_prefix = ""
    best_price: tuple[float, float] | None = None
    for prefix, price in table.items():
        if model_key.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_price = price
    from_table = best_price is not None
    if best_price is None:
        if unpriced_fallback is None:
            return 0.0, False
        best_price = (float(unpriced_fallback[0]), float(unpriced_fallback[1]))
    price_in, price_out = best_price
    cost = (
        input_tokens * price_in / 1_000_000
        + output_tokens * price_out / 1_000_000
        + cache_read_tokens * price_in * _CACHE_READ_MULTIPLIER / 1_000_000
        + cache_write_tokens * price_in * _CACHE_WRITE_MULTIPLIER / 1_000_000
    )
    return round(cost, 6), from_table


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------


class CostDashboard:
    """Aggregates + persists per-call LLM cost across sessions."""

    NAMESPACE = "cost_dashboard"
    _AGG_KEY = "aggregate"

    def __init__(
        self,
        store: Any | None = None,
        *,
        flush_every: int = 25,
        flush_interval: float = 30.0,
        keep_recent: int = 400,
        pricing_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._flush_every = max(1, int(flush_every))
        self._flush_interval = max(0.0, float(flush_interval))
        self._recent: deque[dict[str, Any]] = deque(maxlen=max(10, int(keep_recent)))
        self._pricing_overrides = dict(pricing_overrides or {})
        # model -> {"calls", "input_tokens", "output_tokens",
        #           "cache_read_tokens", "cache_write_tokens", "cost_usd"}
        self._by_model: dict[str, dict[str, float]] = {}
        self._by_scene: dict[str, dict[str, float]] = {}
        self._by_project: dict[str, dict[str, float]] = {}
        self._total_calls = 0
        self._total_cost = 0.0
        self._unpriced_calls = 0
        self._loaded = False
        self._records_since_flush = 0
        self._last_flush = time.monotonic()
        self.flushed_count = 0

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        """Merge persisted aggregates into memory (once, best-effort)."""
        if self._loaded or self._store is None:
            self._loaded = True
            return
        self._loaded = True
        try:
            data = self._store.get(self.NAMESPACE, self._AGG_KEY) or {}
        except Exception:  # noqa: BLE001 - store must never break tracking
            return
        if not isinstance(data, dict):
            return
        for model, agg in (data.get("by_model") or {}).items():
            if isinstance(agg, dict):
                self._by_model[str(model)] = {
                    k: float(agg.get(k, 0) or 0)
                    for k in ("calls", "input_tokens", "output_tokens",
                              "cache_read_tokens", "cache_write_tokens", "cost_usd")
                }
        for scene, agg in (data.get("by_scene") or {}).items():
            if isinstance(agg, dict):
                self._by_scene[str(scene)] = {
                    k: float(agg.get(k, 0) or 0) for k in ("calls", "cost_usd")
                }
        for project, agg in (data.get("by_project") or {}).items():
            if isinstance(agg, dict):
                self._by_project[str(project)] = {
                    k: float(agg.get(k, 0) or 0) for k in ("calls", "cost_usd")
                }
        self._total_calls = int(data.get("total_calls") or 0)
        self._total_cost = float(data.get("total_cost_usd") or 0.0)
        self._unpriced_calls = int(data.get("unpriced_calls") or 0)

    def flush(self, *, force: bool = False) -> bool:
        """Persist aggregates to the store. Safe to call repeatedly."""
        with self._lock:
            if not force and self._records_since_flush == 0:
                return True
            if self._store is None:
                return False
            payload = {
                "by_model": self._by_model,
                "by_scene": dict(list(self._by_scene.items())[-500:]),
                "by_project": dict(list(self._by_project.items())[-200:]),
                "total_calls": self._total_calls,
                "total_cost_usd": round(self._total_cost, 6),
                "unpriced_calls": self._unpriced_calls,
            }
        try:
            ok = bool(self._store.set(self.NAMESPACE, self._AGG_KEY, payload))
            if ok:
                self.flushed_count += 1
                self._records_since_flush = 0
                self._last_flush = time.monotonic()
            return ok
        except Exception:  # noqa: BLE001
            return False

    def _maybe_flush(self) -> None:
        if self._store is None:
            return
        if (
            self._records_since_flush >= self._flush_every
            or time.monotonic() - self._last_flush >= self._flush_interval
        ):
            self.flush()

    # -- recording -----------------------------------------------------

    def record(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        scene: str | None = None,
        project: str | None = None,
        source: str = "llm",
    ) -> float:
        """Record one completed call; returns its estimated cost.

        Never raises — cost tracking must not break a live request.
        """
        try:
            self._load()
            cost, priced = estimate_cost(
                model,
                int(input_tokens or 0),
                int(output_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0),
                cache_read_tokens=int(cache_read_tokens or 0),
                overrides=self._pricing_overrides,
            )
            with self._lock:
                self._total_calls += 1
                self._total_cost += cost
                if not priced:
                    self._unpriced_calls += 1
                bucket = self._by_model.setdefault(
                    model or "unknown",
                    {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                     "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0},
                )
                bucket["calls"] += 1
                bucket["input_tokens"] += int(input_tokens or 0)
                bucket["output_tokens"] += int(output_tokens or 0)
                bucket["cache_read_tokens"] += int(cache_read_tokens or 0)
                bucket["cache_write_tokens"] += int(cache_write_tokens or 0)
                bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 6)
                if scene:
                    s = self._by_scene.setdefault(scene, {"calls": 0, "cost_usd": 0.0})
                    s["calls"] += 1
                    s["cost_usd"] = round(s["cost_usd"] + cost, 6)
                if project:
                    p = self._by_project.setdefault(project, {"calls": 0, "cost_usd": 0.0})
                    p["calls"] += 1
                    p["cost_usd"] = round(p["cost_usd"] + cost, 6)
                self._recent.append(
                    {
                        "model": model or "unknown",
                        "input_tokens": int(input_tokens or 0),
                        "output_tokens": int(output_tokens or 0),
                        "cost_usd": round(cost, 6),
                        "priced": priced,
                        "scene": scene or "",
                        "project": project or "",
                        "source": source,
                        "at": time.time(),
                    }
                )
                self._records_since_flush += 1
            self._maybe_flush()
            return cost
        except Exception:  # noqa: BLE001 - never break the request path
            log.exception("cost dashboard record failed")
            return 0.0

    # -- reporting -----------------------------------------------------

    def summary(
        self,
        *,
        project: str | None = None,
        scenes_total: int | None = None,
        scenes_done: int | None = None,
    ) -> dict[str, Any]:
        """Real-time summary: totals, per-model, per-scene, projection.

        ``scenes_total``/``scenes_done`` enable the projection: with both
        given, the projected total is ``spend * scenes_total / scenes_done``
        (requires scenes_done > 0).
        """
        self._load()
        with self._lock:
            by_model = {
                model: {
                    "calls": int(agg["calls"]),
                    "input_tokens": int(agg["input_tokens"]),
                    "output_tokens": int(agg["output_tokens"]),
                    "cost_usd": round(agg["cost_usd"], 6),
                }
                for model, agg in sorted(
                    self._by_model.items(), key=lambda kv: -kv[1]["cost_usd"]
                )
            }
            scenes_view = (
                {s: {"calls": int(a["calls"]), "cost_usd": round(a["cost_usd"], 6)}
                 for s, a in sorted(self._by_scene.items(), key=lambda kv: -kv[1]["cost_usd"])}
                if project is None else
                {s: {"calls": int(a["calls"]), "cost_usd": round(a["cost_usd"], 6)}
                 for s, a in self._by_scene.items()}
            )
            projects = {
                p: {"calls": int(a["calls"]), "cost_usd": round(a["cost_usd"], 6)}
                for p, a in sorted(self._by_project.items(), key=lambda kv: -kv[1]["cost_usd"])
            }
            total_cost = round(self._total_cost, 6)
            out = {
                "total_calls": self._total_calls,
                "total_cost_usd": total_cost,
                "total_input_tokens": sum(m["input_tokens"] for m in self._by_model.values()),
                "total_output_tokens": sum(m["output_tokens"] for m in self._by_model.values()),
                "unpriced_calls": self._unpriced_calls,
                "models": by_model,
                "scenes": scenes_view,
                "projects": projects,
            }
            if project is not None:
                out["project"] = project
                out["project_cost_usd"] = round(
                    self._by_project.get(project, {}).get("cost_usd", 0.0), 6
                )
        if scenes_total is not None and scenes_done and scenes_done > 0:
            remaining = max(0, int(scenes_total) - int(scenes_done))
            projected = out["total_cost_usd"] * int(scenes_total) / int(scenes_done)
            out["scenes_total"] = int(scenes_total)
            out["scenes_done"] = int(scenes_done)
            out["scenes_remaining"] = remaining
            out["projected_total_usd"] = round(projected, 6)
        return out

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent recorded calls (newest first)."""
        self._load()
        with self._lock:
            items = list(self._recent)[-max(1, int(limit)):]
        items.reverse()
        return items

    def clear(self) -> bool:
        """Reset in-memory aggregates and drop the persisted entry."""
        with self._lock:
            self._by_model.clear()
            self._by_scene.clear()
            self._by_project.clear()
            self._recent.clear()
            self._total_calls = 0
            self._total_cost = 0.0
            self._unpriced_calls = 0
        if self._store is None:
            return True
        try:
            self._store.delete(self.NAMESPACE, self._AGG_KEY)
            return True
        except Exception:  # noqa: BLE001
            return False


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_DASHBOARD: CostDashboard | None = None


def get_cost_dashboard() -> CostDashboard:
    """Return the process-wide dashboard (lazily built).

    Backed by the SQLite persistent store so aggregates survive restarts;
    pricing overrides come from ``OPENHARNESS_PRICING_JSON``.
    """
    global _DASHBOARD
    if _DASHBOARD is None:
        store = None
        try:
            from openharness.services.persistent_store import get_persistent_store

            store = get_persistent_store()
        except Exception:  # noqa: BLE001
            store = None
        _DASHBOARD = CostDashboard(
            store,
            pricing_overrides=pricing_overrides_from_env(),
        )
    return _DASHBOARD


def set_cost_dashboard(dashboard: CostDashboard | None) -> None:
    """Test helper: install or reset the singleton."""
    global _DASHBOARD
    if _DASHBOARD is not None:
        _DASHBOARD.flush()
    _DASHBOARD = dashboard
