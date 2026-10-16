"""End-to-end integration tests for the Top-10 optimizations.

These tests run a REAL ``QueryEngine`` with REAL tools and verify that
the optimizations actually change observable behaviour. They complement
the unit tests in ``test_top10_optimizations.py`` by exercising the
full query loop, not just the individual modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
)
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import (
    PermissionSettings,
    Settings,
)
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock, ToolResultBlock
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import ToolExecutionCompleted
from openharness.permissions.checker import build_permission_checker
from openharness.permissions.modes import PermissionMode
from openharness.services.tool_cache import (
    disable_tool_cache,
    enable_tool_cache,
    reset_tool_cache,
    get_tool_cache,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from openharness.tools.file_read_tool import FileReadTool, FileReadToolInput
from pydantic import BaseModel, Field


class _CountingReadFileTool(BaseTool):
    """Read-file tool that counts executions so tests can verify caching."""

    name = "read_file"
    description = "Read a file (counting version for cache tests)."
    input_model = FileReadToolInput

    def __init__(self) -> None:
        self.call_count = 0

    def is_read_only(self, arguments):
        return True

    async def execute(self, arguments, context: ToolExecutionContext) -> ToolResult:
        self.call_count += 1
        path = Path(arguments.path).expanduser()
        if not path.is_absolute():
            path = context.cwd / path
        if not path.is_file():
            return ToolResult(output=f"File not found: {path}", is_error=True)
        return ToolResult(output=path.read_text(encoding="utf-8"))


class _ScriptedClient:
    """Streaming client that plays back a fixed list of assistant messages.

    Used to drive the query engine through specific tool-call sequences.
    """

    def __init__(self, messages: list[ConversationMessage]) -> None:
        self._messages = list(messages)
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest):
        self.calls += 1
        if not self._messages:
            return
        msg = self._messages.pop(0)
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                yield ApiTextDeltaEvent(text=block.text)
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


def _make_engine(
    *,
    cwd: Path,
    tool_registry: ToolRegistry,
    settings: Settings,
    api_messages: list[ConversationMessage],
) -> QueryEngine:
    return QueryEngine(
        api_client=_ScriptedClient(api_messages),
        tool_registry=tool_registry,
        permission_checker=build_permission_checker(settings, cwd=str(cwd)),
        cwd=cwd,
        model="claude-test",
        system_prompt="test",
        settings=settings,
        tool_metadata={"session_id": "test-session"},
    )


@pytest.mark.asyncio
async def test_tool_cache_prevents_reexecution_of_same_read_file(tmp_path: Path):
    """When tool_cache is enabled, calling read_file twice on the same path
    only runs the actual tool once. The second call returns the cached result."""
    # Setup: a file + a counting read_file tool
    target = tmp_path / "target.txt"
    target.write_text("hello world\n")
    counting_tool = _CountingReadFileTool()
    registry = ToolRegistry()
    registry.register(counting_tool)
    # Enable the cache
    enable_tool_cache(max_entries=10, ttl_seconds=60)
    try:
        settings = Settings(
            tool_cache_enabled=True,
            permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
            auto_save=False,  # don't write session snapshots during the test
        )
        # Two-turn conversation: each turn calls read_file on the same path.
        api_messages = [
            # Turn 1: assistant calls read_file("target.txt")
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="read_file", input={"path": "target.txt"})],
            ),
            # Turn 1 result: assistant says "ok"
            ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            # Turn 2: assistant calls read_file("target.txt") again
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_2", name="read_file", input={"path": "target.txt"})],
            ),
            # Turn 2 result: assistant done
            ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
        ]
        engine = _make_engine(
            cwd=tmp_path,
            tool_registry=registry,
            settings=settings,
            api_messages=api_messages,
        )
        # Drive turn 1
        events_1 = [event async for event in engine.submit_message("read target.txt")]
        # Drive turn 2 — model asks to read the same file again
        events_2 = [event async for event in engine.submit_message("read it again")]
        # Tool should have executed ONCE, not twice (cache hit on second call)
        assert counting_tool.call_count == 1, (
            f"Expected 1 actual execution (cache hit on 2nd), got {counting_tool.call_count}"
        )
        # Both turns should have produced ToolExecutionCompleted events
        completed_1 = [e for e in events_1 if isinstance(e, ToolExecutionCompleted)]
        completed_2 = [e for e in events_2 if isinstance(e, ToolExecutionCompleted)]
        assert len(completed_1) == 1
        assert len(completed_2) == 1
        # Cache stats should show 1 miss + 1 hit
        cache = get_tool_cache()
        stats = cache.stats()
        assert stats["hits"] == 1, f"Expected 1 cache hit, got {stats['hits']}"
        assert stats["misses"] == 1
    finally:
        reset_tool_cache()
        disable_tool_cache()


@pytest.mark.asyncio
async def test_tool_cache_disabled_runs_tool_every_time(tmp_path: Path):
    """When tool_cache is disabled (default), the tool runs on every call."""
    target = tmp_path / "target.txt"
    target.write_text("hello\n")
    counting_tool = _CountingReadFileTool()
    registry = ToolRegistry()
    registry.register(counting_tool)
    disable_tool_cache()
    settings = Settings(
        permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
        auto_save=False,
    )
    api_messages = [
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="tu_1", name="read_file", input={"path": "target.txt"})],
        ),
        ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="tu_2", name="read_file", input={"path": "target.txt"})],
        ),
        ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
    ]
    engine = _make_engine(
        cwd=tmp_path,
        tool_registry=registry,
        settings=settings,
        api_messages=api_messages,
    )
    [event async for event in engine.submit_message("read")]
    [event async for event in engine.submit_message("read again")]
    # With cache disabled, tool runs twice
    assert counting_tool.call_count == 2


@pytest.mark.asyncio
async def test_tool_cache_invalidated_after_write_file(tmp_path: Path):
    """A write_file call must invalidate the cache so the next read_file
    sees the updated content."""
    target = tmp_path / "data.txt"
    target.write_text("v1\n")
    counting_tool = _CountingReadFileTool()
    registry = ToolRegistry()
    registry.register(counting_tool)
    # Register a stub write_file tool so we can call it
    from openharness.tools.file_write_tool import FileWriteTool
    registry.register(FileWriteTool())
    enable_tool_cache(max_entries=10, ttl_seconds=60)
    try:
        settings = Settings(
            tool_cache_enabled=True,
            permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
            auto_save=False,
        )
        api_messages = [
            # Turn 1: read data.txt (caches v1)
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="read_file", input={"path": "data.txt"})],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            # Turn 2: write_file to data.txt (must invalidate cache)
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(
                    id="tu_2",
                    name="write_file",
                    input={"path": "data.txt", "content": "v2\n"},
                )],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="wrote")]),
            # Turn 3: read data.txt again (cache should have been invalidated, so we re-read)
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="tu_3", name="read_file", input={"path": "data.txt"})],
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
        ]
        engine = _make_engine(
            cwd=tmp_path,
            tool_registry=registry,
            settings=settings,
            api_messages=api_messages,
        )
        events_1 = [event async for event in engine.submit_message("read")]
        events_2 = [event async for event in engine.submit_message("write")]
        events_3 = [event async for event in engine.submit_message("read again")]
        # First read: 1 execution (cache miss)
        # Write: invalidates cache
        # Second read: 1 execution (cache miss because invalidated)
        assert counting_tool.call_count == 2, (
            f"Expected 2 executions (1st read + post-write read), got {counting_tool.call_count}"
        )
        # The third turn's read should show v2 (the written content), not v1
        completed_3 = [e for e in events_3 if isinstance(e, ToolExecutionCompleted)]
        assert len(completed_3) == 1
        assert "v2" in completed_3[0].output
    finally:
        reset_tool_cache()
        disable_tool_cache()


@pytest.mark.asyncio
async def test_parallel_tools_runs_independent_tools_concurrently(tmp_path: Path):
    """Item 8: when the model emits multiple tool calls in one turn,
    they run concurrently (asyncio.gather), not sequentially."""
    import asyncio
    import time

    class _SlowInput(BaseModel):
        seconds: float = Field(default=0.2)

    class _SlowTool(BaseTool):
        name = "slow"
        description = "Sleeps for a fixed duration."
        input_model = _SlowInput

        def __init__(self) -> None:
            self.start_times: list[float] = []
            self.end_times: list[float] = []

        def is_read_only(self, arguments):
            return True

        async def execute(self, arguments, context):
            self.start_times.append(time.monotonic())
            await asyncio.sleep(arguments.seconds)
            self.end_times.append(time.monotonic())
            return ToolResult(output=f"slept {arguments.seconds}s")

    slow_tool = _SlowTool()
    registry = ToolRegistry()
    registry.register(slow_tool)
    settings = Settings(
        permission=PermissionSettings(mode=PermissionMode.FULL_AUTO),
        auto_save=False,
        parallel_tools=True,
    )
    api_messages = [
        # Single turn with TWO concurrent tool calls
        ConversationMessage(
            role="assistant",
            content=[
                ToolUseBlock(id="tu_1", name="slow", input={"seconds": 0.2}),
                ToolUseBlock(id="tu_2", name="slow", input={"seconds": 0.2}),
            ],
        ),
        ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
    ]
    engine = _make_engine(
        cwd=tmp_path,
        tool_registry=registry,
        settings=settings,
        api_messages=api_messages,
    )
    t0 = time.monotonic()
    events = [event async for event in engine.submit_message("run two")]
    elapsed = time.monotonic() - t0
    # If parallel: 0.2s total (both run at once).
    # If sequential: 0.4s total.
    assert elapsed < 0.35, f"Expected parallel (~0.2s), got {elapsed:.2f}s"
    assert len(slow_tool.start_times) == 2
    # The two start times should be very close (within 50ms)
    assert abs(slow_tool.start_times[1] - slow_tool.start_times[0]) < 0.05
