"""semantic_memory tool — recall past decisions by meaning.

MEMORY.md is plain text you cannot search by meaning. This tool wraps
:mod:`openharness.services.semantic_memory` (SQLite + embeddings):

- ``remember``: store a fact/decision ("scene 12 uses
  cubic-bezier(0.4, 0, 0.2, 1) for transitions") — embedded and kept
  across sessions and projects.
- ``recall``: query by meaning ("what easing did scene 12 use?") —
  top-K results by cosine similarity.

Embeddings come from the configured OpenAI-compatible endpoint
(settings.embedding / OPENHARNESS_EMBEDDING_* — BGE-M3, MiniLM...) or
a deterministic local hashing fallback that works offline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SemanticMemoryInput(BaseModel):
    """Arguments for the semantic_memory tool."""

    action: str = Field(
        description=(
            "Operation: remember (store a fact), recall (query top-K by "
            "meaning), list (recent entries), forget (delete by id), stats."
        ),
        pattern="^(remember|recall|list|forget|stats)$",
    )
    text: str | None = Field(
        default=None,
        description="The fact/decision to remember, or the recall query.",
    )
    project: str = Field(
        default="default",
        description="Project scope for remember/recall/list (default: 'default').",
    )
    kind: str = Field(
        default="fact",
        description="Entry kind tag: fact, decision, style, lesson, config ...",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured metadata attached to a memory.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Recall result count.")
    min_similarity: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Recall similarity floor (hashing backend scores are lower).",
    )
    memory_id: int | None = Field(default=None, description="id to forget.")
    limit: int = Field(default=20, ge=1, le=200, description="List limit.")


class SemanticMemoryTool(BaseTool):
    """Semantic memory: remember and recall decisions by meaning."""

    name = "semantic_memory"
    description = (
        "Long-term semantic memory over decisions and facts: `remember` a "
        "fact (embedded + stored in SQLite, durable across sessions and "
        "projects), then `recall` later by MEANING with cosine-similarity "
        "top-K (e.g. \"what easing did scene 12 use?\"). Also list/stats/"
        "forget. Uses the configured embedding endpoint (BGE-M3/MiniLM via "
        "settings.embedding) or a built-in offline hashing embedder."
    )
    input_model = SemanticMemoryInput

    def is_read_only(self, arguments: SemanticMemoryInput) -> bool:
        return arguments.action in {"recall", "list", "stats"}

    async def execute(
        self, arguments: SemanticMemoryInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.semantic_memory import get_semantic_memory

        memory = get_semantic_memory()

        if arguments.action == "remember":
            text = (arguments.text or "").strip()
            if not text:
                return ToolResult(
                    output="semantic_memory: `text` is required for remember.",
                    is_error=True,
                )
            memory_id = memory.remember(
                text,
                project=arguments.project,
                kind=arguments.kind,
                metadata=arguments.metadata,
            )
            if memory_id is None:
                return ToolResult(
                    output="semantic_memory: store unavailable (see stats).",
                    is_error=True,
                )
            return ToolResult(
                output=(
                    f"[semantic_memory] remembered #{memory_id} "
                    f"({arguments.kind} @ {arguments.project}): {text}"
                ),
                metadata={"memory_id": memory_id},
            )

        if arguments.action == "recall":
            query = (arguments.text or "").strip()
            if not query:
                return ToolResult(
                    output="semantic_memory: `text` (the query) is required for recall.",
                    is_error=True,
                )
            hits = memory.recall(
                query,
                top_k=arguments.top_k,
                project=arguments.project,
                min_similarity=arguments.min_similarity,
            )
            if not hits:
                return ToolResult(
                    output=(
                        f"[semantic_memory] no stored memory matches "
                        f"{query!r} above similarity "
                        f"{arguments.min_similarity:g}."
                    ),
                    metadata={"results": []},
                )
            lines = [f"[semantic_memory] top {len(hits)} match(es) for {query!r}:"]
            for hit in hits:
                lines.append(
                    f"  - {hit.similarity:.3f}  #{hit.id} [{hit.kind} @ "
                    f"{hit.project}] {hit.text}"
                )
            return ToolResult(
                output="\n".join(lines),
                metadata={"results": [hit.as_dict() for hit in hits]},
            )

        if arguments.action == "list":
            entries = memory.list_recent(
                project=arguments.project or None, limit=arguments.limit
            )
            if not entries:
                return ToolResult(output="[semantic_memory] (empty)", metadata={"entries": []})
            lines = ["[semantic_memory] recent entries:"]
            for entry in entries:
                lines.append(
                    f"  - #{entry['id']} [{entry['kind']} @ {entry['project']}] "
                    f"{entry['text'][:160]}"
                )
            return ToolResult(
                output="\n".join(lines), metadata={"entries": entries}
            )

        if arguments.action == "forget":
            if arguments.memory_id is None:
                return ToolResult(
                    output="semantic_memory: `memory_id` is required for forget.",
                    is_error=True,
                )
            if memory.forget(arguments.memory_id):
                return ToolResult(
                    output=f"[semantic_memory] forgot #{arguments.memory_id}."
                )
            return ToolResult(
                output=f"[semantic_memory] no memory #{arguments.memory_id}.",
                is_error=True,
            )

        # stats
        stats = memory.stats()
        return ToolResult(
            output=(
                "[semantic_memory] stats:\n"
                f"  entries: {stats['total']}\n"
                f"  by project: {json.dumps(stats['by_project'])}\n"
                f"  by backend: {json.dumps(stats['by_backend'])}\n"
                f"  active backend: {stats['active_backend']}\n"
                f"  store: {stats['path']} (available: {stats['available']})"
            ),
            metadata=stats,
        )
