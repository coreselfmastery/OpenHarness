"""Pipeline and async optimizations — items 16, 17, 18.

#16 Speculative Execution: Start the next pipeline step before the
    current one is confirmed. If the speculation is wrong, the work
    is discarded. If right, latency is reduced by one step.

#17 Work Stealing: When a worker finishes its queue and becomes idle,
    it steals tasks from the back of a busy worker's queue. Prevents
    stragglers from delaying the whole batch.

#18 Background Prefetch: While the agent is working on scene N,
    prefetch the assets needed for scene N+1 in the background. By the
    time the agent is ready for N+1, the assets are already loaded.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# ============================================================
# #16 Speculative Execution
# ============================================================

@dataclass
class SpeculativeResult:
    """Result of a speculative execution."""

    step: str
    result: Any
    committed: bool = False  # True if the result was used, False if discarded
    wasted: bool = False  # True if the speculation was wrong and discarded


class SpeculativeExecutor:
    """Start the next pipeline step before the current one is confirmed.

    Pattern:
        executor = SpeculativeExecutor()
        # Start step 1 (confirmed)
        executor.start_confirmed("step_1", step_1_coro)
        # Speculatively start step 2 (may be discarded if step 1 fails)
        executor.start_speculative("step_2", step_2_coro)
        # When step 1 completes:
        result_1 = await executor.await_result("step_1")
        if result_1.success:
            # Step 2 was already running — just await it (near-instant if done)
            result_2 = await executor.await_result("step_2")
            executor.commit("step_2")
        else:
            # Step 1 failed — discard step 2's speculative work
            executor.discard("step_2")
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, SpeculativeResult] = {}

    def start_confirmed(self, step: str, coro: Awaitable[Any]) -> None:
        """Start a step that will definitely be used (confirmed execution)."""
        self._tasks[step] = asyncio.create_task(self._run_step(step, coro))

    def start_speculative(self, step: str, coro: Awaitable[Any]) -> None:
        """Start a step speculatively — may be discarded later."""
        self._tasks[step] = asyncio.create_task(self._run_step(step, coro))

    async def _run_step(self, step: str, coro: Awaitable[Any]) -> Any:
        try:
            result = await coro
            return result
        except Exception as exc:
            log.warning("speculative step %s raised: %s", step, exc)
            raise

    async def await_result(self, step: str) -> Any:
        """Wait for a step to complete and return its result."""
        task = self._tasks.get(step)
        if task is None:
            raise KeyError(f"Step {step} not started")
        return await task

    def commit(self, step: str) -> None:
        """Mark a speculative step's result as committed (will be used)."""
        result = self._results.get(step)
        if result:
            result.committed = True
        # The task may not be done yet — the result will be committed
        # when it completes.
        task = self._tasks.get(step)
        if task and task.done() and not task.cancelled():
            try:
                result_val = task.result()
                self._results[step] = SpeculativeResult(
                    step=step, result=result_val, committed=True,
                )
            except Exception:
                pass

    def discard(self, step: str) -> None:
        """Discard a speculative step's result (cancel if still running)."""
        task = self._tasks.get(step)
        if task and not task.done():
            task.cancel()
        self._results[step] = SpeculativeResult(
            step=step, result=None, committed=False, wasted=True,
        )

    def stats(self) -> dict[str, Any]:
        committed = sum(1 for r in self._results.values() if r.committed)
        wasted = sum(1 for r in self._results.values() if r.wasted)
        return {
            "total_steps": len(self._tasks),
            "committed": committed,
            "wasted": wasted,
        }


# ============================================================
# #17 Work Stealing
# ============================================================

class WorkStealingQueue:
    """Distributed task queue with work stealing.

    Each worker has its own deque. When a worker's deque is empty, it
    steals tasks from the BACK of another worker's deque (LIFO stealing
    for cache locality).

    This prevents stragglers (one worker with 10 tasks while others
    are idle) from delaying the whole batch.

    Usage:
        queue = WorkStealingQueue(num_workers=4)
        for i in range(20):
            queue.push(worker_id=i % 4, task=f"task_{i}")
        # Workers pop from their own queue, steal from others when empty.
    """

    def __init__(self, *, num_workers: int) -> None:
        self._num_workers = num_workers
        self._queues: list[deque[Any]] = [deque() for _ in range(num_workers)]
        self._lock = asyncio.Lock()
        self._steals = 0

    def push(self, *, worker_id: int, task: Any) -> None:
        """Push a task to a specific worker's queue."""
        self._queues[worker_id % self._num_workers].append(task)

    async def pop(self, worker_id: int) -> Any | None:
        """Pop a task from the worker's queue, or steal from another.

        If the worker's own queue is empty, tries to steal from the
        back of every other worker's queue.
        """
        # Try own queue first
        own = self._queues[worker_id % self._num_workers]
        if own:
            return own.popleft()
        # Try stealing from other workers
        async with self._lock:
            for i in range(self._num_workers):
                if i == worker_id % self._num_workers:
                    continue
                other = self._queues[i]
                if other:
                    stolen = other.pop()  # steal from back (LIFO)
                    self._steals += 1
                    return stolen
        return None  # all queues empty

    @property
    def total_tasks(self) -> int:
        return sum(len(q) for q in self._queues)

    @property
    def steal_count(self) -> int:
        return self._steals

    def stats(self) -> dict[str, Any]:
        return {
            "num_workers": self._num_workers,
            "queue_sizes": [len(q) for q in self._queues],
            "total_tasks": self.total_tasks,
            "steals": self._steals,
        }


# ============================================================
# #18 Background Prefetch
# ============================================================

@dataclass
class PrefetchRequest:
    """A request to prefetch an asset in the background."""

    asset_id: str
    fetch_fn: Callable[[], Awaitable[Any]]
    priority: int = 0  # lower = higher priority


class BackgroundPrefetcher:
    """Prefetch assets for the next scene while the current one renders.

    While the agent is generating/rendering scene N, this prefetcher
    loads the assets needed for scene N+1 in the background. By the
    time the agent is ready for N+1, the assets are already in memory.

    Usage:
        prefetcher = BackgroundPrefetcher(max_concurrent=3)
        await prefetcher.start()
        # While scene N renders, prefetch N+1's assets:
        prefetcher.request(PrefetchRequest(
            asset_id="scene_002_assets",
            fetch_fn=lambda: load_assets("scene_002"),
        ))
        # When scene N is done and we start N+1:
        assets = await prefetcher.await_asset("scene_002_assets")
    """

    def __init__(self, *, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[str, Any] = {}
        self._futures: dict[str, asyncio.Future] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        """Start the prefetch worker loop."""
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop the prefetcher."""
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def request(self, req: PrefetchRequest) -> None:
        """Request a prefetch (non-blocking)."""
        if req.asset_id in self._cache:
            return  # already cached
        if req.asset_id in self._futures:
            return  # already requested
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._futures[req.asset_id] = future
        asyncio.create_task(self._fetch(req, future))

    async def _fetch(self, req: PrefetchRequest, future: asyncio.Future) -> None:
        try:
            async with self._semaphore:
                result = await req.fetch_fn()
            self._cache[req.asset_id] = result
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            self._futures.pop(req.asset_id, None)

    async def _process_queue(self) -> None:
        """Background loop (currently just keeps the event loop alive)."""
        while not self._stopping:
            await asyncio.sleep(0.1)

    async def await_asset(self, asset_id: str, *, timeout: float = 30.0) -> Any:
        """Wait for a prefetched asset. If not yet requested, returns None."""
        if asset_id in self._cache:
            return self._cache[asset_id]
        future = self._futures.get(asset_id)
        if future is None:
            return None  # not requested
        return await asyncio.wait_for(future, timeout=timeout)

    def is_cached(self, asset_id: str) -> bool:
        """Check if an asset is already prefetched."""
        return asset_id in self._cache

    def stats(self) -> dict[str, Any]:
        return {
            "cached_assets": len(self._cache),
            "in_flight": len(self._futures),
            "max_concurrent": self._max_concurrent,
        }

    def clear(self) -> None:
        """Clear the prefetch cache."""
        self._cache.clear()
