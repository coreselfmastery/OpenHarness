"""Async optimizations — items 14, 15.

#14 Async Audio Generation: Generate audio (TTS narration) in parallel
    with scene rendering instead of sequentially. 10-20% faster.

#15 Pre-Warmed Sessions: Pre-initialize API clients and tool registries
    so the first user prompt has no cold-start latency.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


# ============================================================
# #14 Async Audio Generation
# ============================================================

@dataclass
class AudioGenJob:
    """A single audio generation job (TTS narration for a scene)."""

    scene_id: str
    text: str
    voice: str = "default"
    output_path: str = ""
    state: str = "queued"  # "queued", "generating", "completed", "failed"
    error: str | None = None


class AsyncAudioQueue:
    """Generate audio in parallel with scene rendering.

    Instead of: render scene → generate audio → next scene,
    this queue generates audio for scene N while scene N+1 renders.

    Usage:
        queue = AsyncAudioQueue(generate_fn=my_tts_function, max_concurrent=2)
        await queue.start()
        await queue.enqueue(AudioGenJob(scene_id="s1", text="Hello world"))
        results = await queue.wait_all()
    """

    def __init__(
        self,
        *,
        generate_fn: Callable[[AudioGenJob], Awaitable[str]],
        max_concurrent: int = 2,
    ) -> None:
        self._generate_fn = generate_fn
        self._max_concurrent = max_concurrent
        self._queue: asyncio.Queue[AudioGenJob | None] = asyncio.Queue()
        self._jobs: dict[str, AudioGenJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._stopping = False
        self._worker_task = asyncio.create_task(self._run_workers())

    async def stop(self) -> None:
        self._stopping = True
        for _ in range(self._max_concurrent):
            await self._queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def enqueue(self, job: AudioGenJob) -> None:
        self._jobs[job.scene_id] = job
        await self._queue.put(job)

    async def wait_all(self, *, timeout: float | None = None) -> dict[str, AudioGenJob]:
        """Wait for all enqueued jobs to complete."""
        import time
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            pending = [j for j in self._jobs.values() if j.state in ("queued", "generating")]
            if not pending:
                return dict(self._jobs)
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"{len(pending)} audio jobs did not complete")
            await asyncio.sleep(0.05)

    async def _run_workers(self) -> None:
        workers = [asyncio.create_task(self._worker_loop()) for _ in range(self._max_concurrent)]
        await asyncio.gather(*workers, return_exceptions=True)

    async def _worker_loop(self) -> None:
        while not self._stopping:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                job.state = "generating"
                result = await self._generate_fn(job)
                job.output_path = result
                job.state = "completed"
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)
            finally:
                self._queue.task_done()


async def default_audio_fn(job: AudioGenJob) -> str:
    """Default audio generation — a no-op placeholder."""
    await asyncio.sleep(0.01)
    return f"audio/{job.scene_id}.wav"


# ============================================================
# #15 Pre-Warmed Sessions
# ============================================================

@dataclass
class PreWarmedSession:
    """A pre-initialized session ready to accept prompts."""

    session_id: str
    created_at: float
    api_client: Any = None
    tool_registry: Any = None
    system_prompt: str = ""
    in_use: bool = False


class SessionPool:
    """Pool of pre-warmed sessions to eliminate cold-start latency.

    Pre-initializes API clients, tool registries, and system prompts
    so the first user prompt is processed immediately.

    Usage:
        pool = SessionPool(warm_fn=create_session, pool_size=2)
        await pool.warm_up()
        session = await pool.acquire()  # instant — no cold start
        # ... use session ...
        pool.release(session)
    """

    def __init__(
        self,
        *,
        warm_fn: Callable[[], Awaitable[PreWarmedSession]] | None = None,
        pool_size: int = 2,
    ) -> None:
        self._warm_fn = warm_fn
        self._pool_size = pool_size
        self._pool: list[PreWarmedSession] = []
        self._lock = asyncio.Lock()

    async def warm_up(self) -> None:
        """Pre-create sessions to fill the pool."""
        if self._warm_fn is None:
            return
        import time
        tasks = [self._warm_fn() for _ in range(self._pool_size)]
        sessions = await asyncio.gather(*tasks, return_exceptions=True)
        for s in sessions:
            if isinstance(s, PreWarmedSession):
                self._pool.append(s)

    async def acquire(self) -> PreWarmedSession | None:
        """Acquire a pre-warmed session, or None if pool is empty."""
        async with self._lock:
            if not self._pool:
                return None
            session = self._pool.pop()
            session.in_use = True
            return session

    def release(self, session: PreWarmedSession) -> None:
        """Return a session to the pool for reuse."""
        session.in_use = False
        self._pool.append(session)

    @property
    def available(self) -> int:
        return len(self._pool)

    def stats(self) -> dict[str, int]:
        return {
            "pool_size": self._pool_size,
            "available": len(self._pool),
        }
