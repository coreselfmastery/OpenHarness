"""Tool abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openharness.hooks.executor import HookExecutor


@dataclass
class ToolExecutionContext:
    """Shared execution context for tool invocations."""

    cwd: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    hook_executor: HookExecutor | None = None


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all OpenHarness tools."""

    name: str
    description: str
    input_model: type[BaseModel]

    # Cache for the JSON schema, since ``input_model.model_json_schema()`` is
    # expensive (Pydantic walks every field, resolves refs, and serializes
    # JSON Schema). The schema is immutable for a given class, so we compute
    # it once per tool instance and reuse it on every API call.
    _cached_api_schema: dict[str, Any] | None = None

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool."""

    def is_read_only(self, arguments: BaseModel) -> bool:
        """Return whether the invocation is read-only."""
        del arguments
        return False

    def to_api_schema(self) -> dict[str, Any]:
        """Return the tool schema expected by the Anthropic Messages API.

        The schema is computed once and cached on the instance, because the
        tool registry calls this on every API turn and the underlying
        ``model_json_schema()`` call is the dominant cost (~9 ms for the
        default 44-tool registry, with spikes to >1 s when Pydantic's ref
        cache is cold).
        """
        cached = self._cached_api_schema
        if cached is not None:
            # Return a shallow view — callers must not mutate it.
            return cached
        schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }
        self._cached_api_schema = schema
        return schema


class ToolRegistry:
    """Map tool names to implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        # Cache of the full schema list. Invalidated whenever the registry
        # mutates. Building the schema list for 44 tools costs ~10 ms median
        # (with 1.8 s p99 spikes when Pydantic's ref-cache is cold), and the
        # query loop calls this on every API turn.
        self._cached_schema_list: list[dict[str, Any]] | None = None

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        self._cached_schema_list = None

    def unregister(self, name: str) -> None:
        """Remove a registered tool, if present."""
        if self._tools.pop(name, None) is not None:
            self._cached_schema_list = None

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return all tool schemas in API format (cached)."""
        cached = self._cached_schema_list
        if cached is not None:
            return cached
        # Each tool's own schema is cached on the instance, so subsequent
        # rebuilds after invalidation are cheap (no Pydantic walks).
        cached = [tool.to_api_schema() for tool in self._tools.values()]
        self._cached_schema_list = cached
        return cached
