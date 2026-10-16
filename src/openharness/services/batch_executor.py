"""Async batch executor — bounded-concurrency fan-out (speed feature #2).

The single biggest concurrency win for agent-side LLM work: instead of
firing requests one at a time and waiting for each full response before
starting the next, the executor keeps up to ``concurrency`` (default 8)
requests in flight at once behind a semaphore. As soon as one finishes,
the next one starts::

    executor = AsyncBatchExecutor(concurrency=8)
    results = await executor.run(
        prompts,
        lambda prompt, index: call_llm(prompt),
    )

Properties that matter for callers:

- **Order-preserving**: result ``i`` always corresponds to ``items[i]``,
  no matter which worker finished first.
- **Per-item failure isolation**: one failing item never cancels the
  others; its slot holds an :class:`ItemFailure` describing the error
  (callers can ``raise_if_any`` or handle selectively).
- **Retry rounds**: :meth:`run_with_retries` re-runs only the failed
  items (bounded rounds, fresh semaphore per round) so a single flaky
  prompt doesn't repeat the whole batch.
- **Progress callbacks**: ``on_result(index, result)`` fires as each
  item completes — ideal for progress trackers and live logs.
- **Honest stats**: per-batch and cumulative counters (items run,
  failures, peak in-flight, serial-estimate vs wall time) quantify the
  speedup actually achieved.

The executor is deliberately framework-free (stdlib asyncio only) and
never itself raises for worker errors — error policy belongs to the
caller.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

log = logging.getLogger(__name__)


@dataclass
class ItemFailure:
    """Marker placed in a result slot when that item's worker raised.

    ``exception`` carries the original error; ``item_index`` the position
    in the input list. ``retries`` says how many retry rounds the item
    went through before still failing (0 from plain :meth:`run`).
    """

    exception: BaseException
    item_index: int = -1
    retries: int = 0

    @property
    def error(self) -> str:
        return f"{type(self.exception).__name__}: {self.exception}"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ItemFailure({self.error!r}, index={self.item_index})"


@dataclass
class BatchStats:
    """Cumulative executor counters (also the speedup evidence)."""

    batches_run: int = 0
    items_run: int = 0
    item_failures: int = 0
    peak_in_flight: int = 0
    # Sum of per-item durations — how long the work would have taken if
    # run strictly sequentially (the counterfactual).
    serial_seconds: float = 0.0
    # Actual wall time spent inside run() calls.
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        speedup = (
            round(self.serial_seconds / self.wall_seconds, 2)
            if self.wall_seconds > 0 and self.serial_seconds > 0
            else 0.0
        )
        return {
            "batches_run": self.batches_run,
            "items_run": self.items_run,
            "item_failures": self.item_failures,
            "peak_in_flight": self.peak_in_flight,
            "serial_seconds": round(self.serial_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "effective_speedup": speedup,
        }


Worker = Callable[[Any, int], Any]


class AsyncBatchExecutor:
    """Run N awaitables with at most ``concurrency`` in flight at once."""

    def __init__(self, concurrency: int = 8) -> None:
        self.concurrency = max(1, int(concurrency))
        self.stats = BatchStats()

    async def run(
        self,
        items: Sequence[Any] | Iterable[Any],
        worker: Worker,
        *,
        on_result: Callable[[int, Any], None] | None = None,
        label: str = "",
    ) -> list[Any]:
        """Map ``worker(item, index)`` over ``items`` with bounded concurrency.

        Returns a list in input order. Failed items appear as
        :class:`ItemFailure` in their slot. ``worker`` results are awaited
        (coroutine functions / awaitables both work). ``on_result``, when
        given, is called synchronously with ``(index, result_or_ItemFailure)``
        as each item completes — it must not block and its exceptions are
        logged, never propagated.
        """
        items = list(items)
        if not items:
            return []

        # Fresh semaphore per run: asyncio primitives bind to the running
        # loop on first use, so sharing one across loops (tests, nested
        # event loops) would raise "bound to a different event loop".
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[Any] = [_PENDING] * len(items)
        in_flight = 0
        peak = 0
        start = time.monotonic()

        async def _run_one(index: int, item: Any) -> None:
            nonlocal in_flight, peak
            async with semaphore:
                in_flight += 1
                if in_flight > peak:
                    peak = in_flight
                t0 = time.monotonic()
                try:
                    results[index] = await _maybe_await(worker(item, index))
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001 - isolated per item
                    results[index] = ItemFailure(exception=exc, item_index=index)
                finally:
                    self.stats.items_run += 1
                    self.stats.serial_seconds += time.monotonic() - t0
                    in_flight -= 1
                    if on_result is not None:
                        try:
                            on_result(index, results[index])
                        except Exception:  # noqa: BLE001 - callback must not break the batch
                            log.exception("batch on_result callback failed (index %d)", index)

        try:
            await asyncio.gather(*(_run_one(i, item) for i, item in enumerate(items)))
        finally:
            elapsed = time.monotonic() - start
            self.stats.batches_run += 1
            self.stats.wall_seconds += elapsed
            if peak > self.stats.peak_in_flight:
                self.stats.peak_in_flight = peak
            failures = sum(1 for r in results if isinstance(r, ItemFailure))
            self.stats.item_failures += failures
            if label or failures:
                log.debug(
                    "batch %r: %d items in %.2fs (peak in-flight %d, %d failures)",
                    label or "<unnamed>",
                    len(items),
                    elapsed,
                    peak,
                    failures,
                )
        return results

    async def run_with_retries(
        self,
        items: Sequence[Any] | Iterable[Any],
        worker: Worker,
        *,
        retries: int = 1,
        backoff_seconds: float = 0.0,
        on_result: Callable[[int, Any], None] | None = None,
        label: str = "",
    ) -> list[Any]:
        """Run, then re-run only failed items up to ``retries`` extra rounds.

        Retry workers receive the original item and index again — they are
        free to take a different code path (e.g. a corrective prompt). The
        result slot of an item that still fails carries ``ItemFailure``
        with ``retries`` set to the number of retry rounds it used.
        """
        items = list(items)
        results = await self.run(items, worker, on_result=on_result, label=label)
        if retries <= 0:
            return results
        pending = {
            index: result
            for index, result in enumerate(results)
            if isinstance(result, ItemFailure)
        }
        round_index = 0
        while pending and round_index < retries:
            round_index += 1
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds * round_index)
            indexes = sorted(pending)
            round_results = await self.run(
                [items[i] for i in indexes],
                worker,
                on_result=on_result,
                label=f"{label}:retry{round_index}" if label else f"retry{round_index}",
            )
            for slot, result in zip(indexes, round_results):
                results[slot] = result
                if isinstance(result, ItemFailure):
                    result.retries = round_index
                    pending[slot] = result
                else:
                    pending.pop(slot, None)
        return results

    def report(self) -> dict[str, Any]:
        """Stats dict + configured concurrency (for tools / dashboards)."""
        data = self.stats.as_dict()
        data["concurrency"] = self.concurrency
        return data


_PENDING = object()


async def _maybe_await(value: Any) -> Any:
    """Await coroutine/awaitable results; pass plain values through."""
    if value is not None and hasattr(value, "__await__"):
        return await value
    return value


# ----------------------------------------------------------------------
# Process-wide default executor (concurrency from settings)
# ----------------------------------------------------------------------

_EXECUTOR: AsyncBatchExecutor | None = None


def get_batch_executor() -> AsyncBatchExecutor:
    """Return the process-wide executor (concurrency 8 by default)."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = AsyncBatchExecutor(8)
    return _EXECUTOR


def configure_batch_executor_from_settings(settings: Any) -> AsyncBatchExecutor:
    """(Re)build the process-wide executor from settings.

    ``settings.inference.batch_concurrency`` (default 8, env
    ``OPENHARNESS_BATCH_CONCURRENCY``) bounds how many LLM requests are
    in flight at once during fan-out work.
    """
    global _EXECUTOR
    inference = getattr(settings, "inference", None)
    raw = getattr(inference, "batch_concurrency", 8) if inference is not None else 8
    concurrency = max(1, int(raw or 8))
    if _EXECUTOR is not None and _EXECUTOR.concurrency == concurrency:
        return _EXECUTOR
    _EXECUTOR = AsyncBatchExecutor(concurrency)
    log.debug("batch executor concurrency set to %d", concurrency)
    return _EXECUTOR


def set_batch_executor(executor: AsyncBatchExecutor | None) -> None:
    """Test helper: install or reset the singleton."""
    global _EXECUTOR
    _EXECUTOR = executor
