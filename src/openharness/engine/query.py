"""Core tool-aware query loop."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from openharness.api.provider import is_model_multimodal
from openharness.api.usage import UsageSnapshot
from openharness.config.paths import get_data_dir
from openharness.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    sanitize_conversation_messages,
)
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTextReset,
    AssistantTurnComplete,
    CompactProgressEvent,
    ErrorEvent,
    StatusEvent,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.hooks import HookEvent, HookExecutor
from openharness.permissions.checker import PermissionChecker
from openharness.services.tool_outputs import tool_output_inline_chars, tool_output_preview_chars
from openharness.tools.base import ToolExecutionContext
from openharness.tools.base import ToolRegistry, ToolResult

AUTO_COMPACT_STATUS_MESSAGE = "Auto-compacting conversation memory to keep things fast and focused."
REACTIVE_COMPACT_STATUS_MESSAGE = "Prompt too long; compacting conversation memory and retrying."
MAX_SAFE_COMPLETION_TOKENS = 128_000

log = logging.getLogger(__name__)


PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]

MAX_TRACKED_READ_FILES = 6
MAX_TRACKED_SKILLS = 8
MAX_TRACKED_ASYNC_AGENT_EVENTS = 8
MAX_TRACKED_ASYNC_AGENT_TASKS = 12
MAX_TRACKED_WORK_LOG = 10
MAX_TRACKED_USER_GOALS = 5
MAX_TRACKED_ACTIVE_ARTIFACTS = 8
MAX_TRACKED_VERIFIED_WORK = 10


def _is_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "prompt too long",
            "context_length_exceeded",
            "context length",
            "maximum context",
            "context window",
            "input tokens exceed",
            "messages resulted in",
            "reduce the length of the messages",
            "configured limit",
            "too many tokens",
            "too large for the model",
            "maximum context length",
            "exceed_context",
            "exceeds the available context size",
            "available context size",
        )
    )


def _bounded_completion_tokens(max_tokens: int, context_window_tokens: int | None = None) -> int:
    """Return a conservative per-request output token cap.

    Some OpenAI-compatible providers reject very large ``max_tokens`` before
    the request reaches model-side context management.  Keep oversized user
    config from making every turn fail while preserving normal defaults.
    """
    limit = MAX_SAFE_COMPLETION_TOKENS
    if context_window_tokens is not None and context_window_tokens > 0:
        limit = min(limit, int(context_window_tokens))
    return max(1, min(int(max_tokens), limit))


def _extract_completion_token_limit(exc: Exception) -> int | None:
    """Parse provider errors such as "supports at most 128000 completion tokens"."""
    text = str(exc).lower().replace(",", "")
    patterns = (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"at most\s+(\d+)\s+completion tokens",
        r"max(?:imum)?(?:_completion)?[_\s-]tokens.*?(?:<=|less than or equal to|at most)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                return None
    return None


def _is_completion_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("max_tokens" in text or "max_completion_tokens" in text)
        and ("too large" in text or "at most" in text or "completion tokens" in text)
    )


class MaxTurnsExceeded(RuntimeError):
    """Raised when the agent exceeds the configured max_turns for one user prompt."""

    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Exceeded maximum turn limit ({max_turns})")
        self.max_turns = max_turns


@dataclass
class QueryContext:
    """Context shared across a query run."""

    api_client: SupportsStreamingMessages
    tool_registry: ToolRegistry
    permission_checker: PermissionChecker
    cwd: Path
    model: str
    system_prompt: str
    max_tokens: int
    effort: str | None = None
    context_window_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None
    # Optimization #5: compact at this fraction of the context window.
    auto_compact_threshold_ratio: float | None = None
    # Optimization #9: per-call reasoning budget for thinking models.
    thinking_budget: int | None = None
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    max_turns: int | None = 200
    hook_executor: HookExecutor | None = None
    tool_metadata: dict[str, object] | None = None
    # Waves 2-6 control plane (#26-#40). All optional; the loop behaves
    # exactly as before when they are None/0.
    control_bus: object | None = None
    session_cost_tracker: object | None = None
    tool_timeout_seconds: float = 0.0
    checkpoint_interval_turns: int = 0


def _append_capped_unique(bucket: list[Any], value: Any, *, limit: int) -> None:
    if value in bucket:
        bucket.remove(value)
    bucket.append(value)
    if len(bucket) > limit:
        del bucket[:-limit]


def _task_focus_state(tool_metadata: dict[str, object] | None) -> dict[str, object]:
    if tool_metadata is None:
        return {}
    value = tool_metadata.setdefault(
        "task_focus_state",
        {
            "goal": "",
            "recent_goals": [],
            "active_artifacts": [],
            "verified_state": [],
            "next_step": "",
        },
    )
    if isinstance(value, dict):
        value.setdefault("goal", "")
        value.setdefault("recent_goals", [])
        value.setdefault("active_artifacts", [])
        value.setdefault("verified_state", [])
        value.setdefault("next_step", "")
        return value
    replacement = {
        "goal": "",
        "recent_goals": [],
        "active_artifacts": [],
        "verified_state": [],
        "next_step": "",
    }
    tool_metadata["task_focus_state"] = replacement
    return replacement


def _summarize_focus_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    return normalized[:240]


def remember_user_goal(
    tool_metadata: dict[str, object] | None,
    prompt: str,
) -> None:
    state = _task_focus_state(tool_metadata)
    summary = _summarize_focus_text(prompt)
    if not summary:
        return
    recent_goals = state.setdefault("recent_goals", [])
    if isinstance(recent_goals, list):
        _append_capped_unique(recent_goals, summary, limit=MAX_TRACKED_USER_GOALS)
    state["goal"] = summary


def _remember_active_artifact(
    tool_metadata: dict[str, object] | None,
    artifact: str,
) -> None:
    normalized = artifact.strip()
    if not normalized:
        return
    state = _task_focus_state(tool_metadata)
    artifacts = state.setdefault("active_artifacts", [])
    if isinstance(artifacts, list):
        _append_capped_unique(artifacts, normalized[:240], limit=MAX_TRACKED_ACTIVE_ARTIFACTS)


def _remember_verified_work(
    tool_metadata: dict[str, object] | None,
    entry: str,
) -> None:
    normalized = entry.strip()
    if not normalized:
        return
    bucket = _tool_metadata_bucket(tool_metadata, "recent_verified_work")
    _append_capped_unique(bucket, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)
    state = _task_focus_state(tool_metadata)
    verified_state = state.setdefault("verified_state", [])
    if isinstance(verified_state, list):
        _append_capped_unique(verified_state, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)


def _tool_metadata_bucket(
    tool_metadata: dict[str, object] | None,
    key: str,
) -> list[Any]:
    if tool_metadata is None:
        return []
    value = tool_metadata.setdefault(key, [])
    if isinstance(value, list):
        return value
    replacement: list[Any] = []
    tool_metadata[key] = replacement
    return replacement


def _remember_read_file(
    tool_metadata: dict[str, object] | None,
    *,
    path: str,
    offset: int,
    limit: int,
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "read_file_state")
    preview_lines = [line.strip() for line in output.splitlines()[:6] if line.strip()]
    entry = {
        "path": path,
        "span": f"lines {offset + 1}-{offset + limit}",
        "preview": " | ".join(preview_lines)[:320],
        "timestamp": time.time(),
    }
    if isinstance(bucket, list):
        bucket[:] = [
            existing
            for existing in bucket
            if not isinstance(existing, dict) or str(existing.get("path") or "") != path
        ]
        bucket.append(entry)
        if len(bucket) > MAX_TRACKED_READ_FILES:
            del bucket[:-MAX_TRACKED_READ_FILES]


def _remember_skill_invocation(
    tool_metadata: dict[str, object] | None,
    *,
    skill_name: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "invoked_skills")
    normalized = skill_name.strip()
    if not normalized:
        return
    if normalized in bucket:
        bucket.remove(normalized)
    bucket.append(normalized)
    if len(bucket) > MAX_TRACKED_SKILLS:
        del bucket[:-MAX_TRACKED_SKILLS]


def _remember_async_agent_activity(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_state")
    if tool_name == "agent":
        description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
        summary = f"Spawned async agent. {description}".strip()
        if output.strip():
            summary = f"{summary} [{output.strip()[:180]}]".strip()
    elif tool_name == "send_message":
        target = str(tool_input.get("task_id") or "").strip()
        summary = f"Sent follow-up message to async agent {target}".strip()
    else:
        summary = output.strip()[:220] or f"Async agent activity via {tool_name}"
    bucket.append(summary)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_EVENTS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_EVENTS]


def _parse_spawned_agent_identity(
    output: str,
    metadata: dict[str, object] | None = None,
) -> tuple[str, str] | None:
    if isinstance(metadata, dict):
        agent_id = str(metadata.get("agent_id") or "").strip()
        task_id = str(metadata.get("task_id") or "").strip()
        if agent_id and task_id:
            return agent_id, task_id
    match = re.search(r"Spawned agent (.+?) \(task_id=(\S+?)(?:[,)]|$)", output.strip())
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _remember_async_agent_task(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
    result_metadata: dict[str, object] | None = None,
) -> None:
    if tool_name != "agent":
        return
    identity = _parse_spawned_agent_identity(output, result_metadata)
    if identity is None:
        return
    agent_id, task_id = identity
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_tasks")
    description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
    entry = {
        "agent_id": agent_id,
        "task_id": task_id,
        "description": description[:240],
        "status": "spawned",
        "notification_sent": False,
        "spawned_at": time.time(),
    }
    bucket[:] = [
        existing
        for existing in bucket
        if not isinstance(existing, dict) or str(existing.get("task_id") or "") != task_id
    ]
    bucket.append(entry)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_TASKS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_TASKS]


def _remember_work_log(
    tool_metadata: dict[str, object] | None,
    *,
    entry: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "recent_work_log")
    normalized = entry.strip()
    if not normalized:
        return
    bucket.append(normalized[:320])
    if len(bucket) > MAX_TRACKED_WORK_LOG:
        del bucket[:-MAX_TRACKED_WORK_LOG]


def _update_plan_mode(tool_metadata: dict[str, object] | None, mode: str) -> None:
    if tool_metadata is None:
        return
    tool_metadata["permission_mode"] = mode


def _record_tool_carryover(
    context: QueryContext,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    tool_output: str,
    tool_result_metadata: dict[str, object] | None,
    is_error: bool,
    resolved_file_path: str | None,
) -> None:
    if is_error:
        return
    if resolved_file_path is not None:
        _remember_active_artifact(context.tool_metadata, resolved_file_path)
    if tool_name == "read_file" and resolved_file_path is not None:
        offset = int(tool_input.get("offset") or 0)
        limit = int(tool_input.get("limit") or 200)
        _remember_read_file(
            context.tool_metadata,
            path=resolved_file_path,
            offset=offset,
            limit=limit,
            output=tool_output,
        )
        _remember_verified_work(
            context.tool_metadata,
            f"Inspected file {resolved_file_path} (lines {offset + 1}-{offset + limit})",
        )
    elif tool_name == "skill":
        _remember_skill_invocation(
            context.tool_metadata,
            skill_name=str(tool_input.get("name") or ""),
        )
        skill_name = str(tool_input.get("name") or "").strip()
        if skill_name:
            _remember_active_artifact(context.tool_metadata, f"skill:{skill_name}")
            _remember_verified_work(context.tool_metadata, f"Loaded skill {skill_name}")
    elif tool_name in {"agent", "send_message"}:
        _remember_async_agent_activity(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
        )
        _remember_async_agent_task(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
            result_metadata=tool_result_metadata,
        )
        description = str(tool_input.get("description") or tool_input.get("prompt") or tool_name).strip()
        _remember_verified_work(
            context.tool_metadata,
            f"Confirmed async-agent activity via {tool_name}: {description[:180]}",
        )
    elif tool_name == "enter_plan_mode":
        _update_plan_mode(context.tool_metadata, "plan")
    elif tool_name == "exit_plan_mode":
        _update_plan_mode(context.tool_metadata, "default")
    elif tool_name == "web_fetch":
        url = str(tool_input.get("url") or "").strip()
        if url:
            _remember_active_artifact(context.tool_metadata, url)
            _remember_verified_work(context.tool_metadata, f"Fetched remote content from {url}")
    elif tool_name == "web_search":
        query = str(tool_input.get("query") or "").strip()
        if query:
            _remember_verified_work(context.tool_metadata, f"Ran web search for {query[:180]}")
    elif tool_name == "glob":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Expanded glob pattern {pattern[:180]}")
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Checked repository matches for grep pattern {pattern[:180]}")
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_verified_work(
            context.tool_metadata,
            f"Ran bash command {command[:160]} [{summary[:120]}]",
        )
    if tool_name == "read_file" and resolved_file_path is not None:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Read file {resolved_file_path}",
        )
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_work_log(
            context.tool_metadata,
            entry=f"Ran bash: {command[:160]} [{summary[:120]}]",
        )
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        _remember_work_log(
            context.tool_metadata,
            entry=f"Searched with grep pattern={pattern[:160]}",
        )
    elif tool_name == "skill":
        _remember_work_log(
            context.tool_metadata,
            entry=f"Loaded skill {str(tool_input.get('name') or '').strip()}",
        )
    elif tool_name in {"agent", "send_message"}:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Async agent action via {tool_name}",
        )
    elif tool_name == "enter_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Entered plan mode")
    elif tool_name == "exit_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Exited plan mode")


def _tool_artifact_dir() -> Path:
    artifact_dir = get_data_dir() / "tool_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _safe_tool_artifact_name(tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name.strip())
    return (normalized or "tool")[:80]


def _offload_tool_output_if_needed(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
) -> tuple[str, Path | None]:
    inline_limit = tool_output_inline_chars()
    if len(output) <= inline_limit:
        return output, None

    artifact_path = (
        _tool_artifact_dir()
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{_safe_tool_artifact_name(tool_name)}-{uuid4().hex[:12]}.txt"
    )
    artifact_path.write_text(output, encoding="utf-8", errors="replace")
    preview = output[:tool_output_preview_chars()]
    omitted = max(0, len(output) - len(preview))
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Tool use id: {tool_use_id}\n"
        f"Original size: {len(output)} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_path


# ---------------------------------------------------------------------------
# Image preprocessing — convert ImageBlocks to text for non-multimodal models
# ---------------------------------------------------------------------------

_IMAGE_PREPROCESS_STATUS = "Converting image to text description via vision model…"


async def _preprocess_images_in_messages(
    messages: list[ConversationMessage],
    context: QueryContext,
) -> AsyncIterator[StreamEvent]:
    """Scan messages for ImageBlocks and convert them to text if the active
    model does not support multimodal input.

    Yields status events during conversion so the UI stays responsive.
    """
    if is_model_multimodal(context.model):
        return

    vision_config = (context.tool_metadata or {}).get("vision_model_config")
    if not vision_config:
        # No vision model configured — skip preprocessing.
        return

    # Collect all ImageBlocks with their parent message index and block index
    pending: list[tuple[int, int, ImageBlock]] = []
    for msg_idx, msg in enumerate(messages):
        if msg.role != "user":
            continue
        for blk_idx, block in enumerate(msg.content):
            if isinstance(block, ImageBlock):
                pending.append((msg_idx, blk_idx, block))

    if not pending:
        return

    yield StatusEvent(message=_IMAGE_PREPROCESS_STATUS)

    # Process images in parallel
    async def _describe(msg_idx: int, blk_idx: int, block: ImageBlock) -> tuple[int, int, str]:
        tool = context.tool_registry.get("image_to_text")
        if tool is None:
            return msg_idx, blk_idx, "[Image: could not describe — image_to_text tool not available]"

        # Build tool input
        tool_input_data: dict[str, object] = {
            "image_data": block.data,
            "media_type": block.media_type,
            "prompt": "Describe this image in detail, including any text, "
                      "UI elements, code, diagrams, or visual information present.",
        }

        try:
            parsed = tool.input_model.model_validate(tool_input_data)
        except Exception:
            return msg_idx, blk_idx, "[Image: could not parse image data]"

        exec_context = ToolExecutionContext(
            cwd=context.cwd,
            metadata={
                "vision_model_config": vision_config,
                **(context.tool_metadata or {}),
            },
        )
        result = await tool.execute(parsed, exec_context)
        if result.is_error:
            return msg_idx, blk_idx, f"[Image description failed: {result.output}]"
        return msg_idx, blk_idx, result.output

    results = await asyncio.gather(*[_describe(mi, bi, blk) for mi, bi, blk in pending])

    # Replace ImageBlocks with TextBlocks in-place
    for msg_idx, blk_idx, description in results:
        msg = messages[msg_idx]
        msg.content[blk_idx] = TextBlock(text=description)


async def run_query(
    context: QueryContext,
    messages: list[ConversationMessage],
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Auto-compaction is checked at the start of each turn.  When the
    estimated token count exceeds the model's auto-compact threshold,
    the engine first tries a cheap microcompact (clearing old tool result
    content) and, if that is not enough, performs a full LLM-based
    summarization of older messages.
    """
    from openharness.services.compact import (
        AutoCompactState,
        auto_compact_if_needed,
    )

    compact_state = AutoCompactState()
    reactive_compact_attempted = False
    last_compaction_result: tuple[list[ConversationMessage], bool] = (messages, False)
    effective_max_tokens = _bounded_completion_tokens(
        context.max_tokens,
        context.context_window_tokens,
    )
    reported_token_clamp = False

    async def _stream_compaction(
        *,
        trigger: str,
        force: bool = False,
    ) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
        nonlocal last_compaction_result
        progress_queue: asyncio.Queue[CompactProgressEvent] = asyncio.Queue()

        async def _progress(event: CompactProgressEvent) -> None:
            await progress_queue.put(event)

        task = asyncio.create_task(
            auto_compact_if_needed(
                messages,
                api_client=context.api_client,
                model=context.model,
                system_prompt=context.system_prompt,
                state=compact_state,
                progress_callback=_progress,
                force=force,
                trigger=trigger,
                hook_executor=context.hook_executor,
                carryover_metadata=context.tool_metadata,
                context_window_tokens=context.context_window_tokens,
                auto_compact_threshold_tokens=context.auto_compact_threshold_tokens,
                auto_compact_threshold_ratio=context.auto_compact_threshold_ratio,
            )
        )
        try:
            while True:
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
                    yield event, None
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue
            while not progress_queue.empty():
                yield progress_queue.get_nowait(), None
            last_compaction_result = await task
        except BaseException:
            # Consumer closed/cancelled the query mid-compaction: the
            # background summarization task would otherwise keep running
            # (a paid LLM call whose result is discarded) and mutate
            # shared message objects in the background. Cancel and reap.
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise
        return

    turn_count = 0
    while context.max_turns is None or turn_count < context.max_turns:
        turn_count += 1
        # --- control plane (#26/#30/#35/#36): turn-boundary checks -----
        bus = context.control_bus
        if bus is not None:
            if bus.killed:
                # In-flight work has finished; abort before the next LLM
                # call so the session ends cleanly instead of hanging.
                yield ErrorEvent(
                    message=f"Session aborted by kill switch: {bus.kill_reason or 'no reason recorded'}"
                ), None
                return
            await bus.wait_if_paused()  # #30: block here while paused
            injected = bus.drain_injected_messages()  # #35: human override
            if injected:
                messages.append(
                    ConversationMessage.from_user_text("\n\n---\n\n".join(injected))
                )
                yield StatusEvent(
                    message=(
                        f"{len(injected)} injected human message(s) appended "
                        "at the turn boundary."
                    )
                ), None
            bus.feed("turn")  # #36: watchdog progress signal
        # ----------------------------------------------------------------
        if effective_max_tokens != context.max_tokens and not reported_token_clamp:
            reported_token_clamp = True
            yield StatusEvent(
                message=(
                    "Requested max_tokens="
                    f"{context.max_tokens} exceeds the safe per-request output cap; "
                    f"using {effective_max_tokens}."
                )
            ), None
        # --- auto-compact check before calling the model ---------------
        async for event, usage in _stream_compaction(trigger="auto"):
            yield event, usage
        compacted_messages, was_compacted = last_compaction_result
        if compacted_messages is not messages:
            messages[:] = compacted_messages
        # ---------------------------------------------------------------

        # --- image preprocessing: convert ImageBlocks to text for non-vision models ---
        async for event in _preprocess_images_in_messages(messages, context):
            yield event, None
        # -----------------------------------------------------------------------------

        final_message: ConversationMessage | None = None
        # Wall-clock of THIS LLM turn (stream start → final message): fed
        # into the audit log below — the old stub logged 0.0 forever, so
        # slow-provider turns were invisible in the audit trail.
        llm_turn_started = time.monotonic()
        usage = UsageSnapshot()

        # Provider-safe history at request time: mid-turn rollbacks
        # (checkpoint restore, #37) and interrupted flows can leave orphan
        # tool_result blocks or a trailing unpaired tool_use; providers
        # reject both with a 400. The engine sanitizes at submit/continue
        # boundaries — this guards every outbound request as well.
        sanitized_view = sanitize_conversation_messages(messages)
        if len(sanitized_view) != len(messages) or any(
            a is not b for a, b in zip(sanitized_view, messages)
        ):
            messages[:] = sanitized_view

        try:
            async for event in context.api_client.stream_message(
                ApiMessageRequest(
                    model=context.model,
                    # Snapshot copy: the loop keeps mutating ``messages``
                    # after the response arrives (coordinator-context
                    # dance, tool-result appends). Handing the live list
                    # to the request let those mutations leak into the
                    # captured request object.
                    messages=list(messages),
                    system_prompt=context.system_prompt,
                    max_tokens=effective_max_tokens,
                    tools=context.tool_registry.to_api_schema(),
                    effort=context.effort,
                    thinking_budget=context.thinking_budget,
                )
            ):
                if isinstance(event, ApiTextDeltaEvent):
                    yield AssistantTextDelta(text=event.text), None
                    continue
                if isinstance(event, ApiRetryEvent):
                    # The retry restarts the provider stream from scratch;
                    # tell the UI to discard the partial assistant text it
                    # already received (appending the retried stream would
                    # duplicate the partial first attempt in the buffer).
                    yield AssistantTextReset(reason=event.message), None
                    yield StatusEvent(
                        message=(
                            f"Request failed; retrying in {event.delay_seconds:.1f}s "
                            f"(attempt {event.attempt + 1} of {event.max_attempts}): {event.message}"
                        )
                    ), None
                    continue

                if isinstance(event, ApiMessageCompleteEvent):
                    final_message = event.message
                    usage = event.usage
        except Exception as exc:
            error_msg = str(exc)
            if _is_completion_token_limit_error(exc):
                supported_limit = _extract_completion_token_limit(exc)
                if supported_limit is not None and effective_max_tokens > supported_limit:
                    previous_max_tokens = effective_max_tokens
                    effective_max_tokens = supported_limit
                    yield StatusEvent(
                        message=(
                            f"Model rejected max_tokens={previous_max_tokens}; "
                            f"retrying with provider limit {effective_max_tokens}."
                        )
                    ), None
                    turn_count = max(0, turn_count - 1)
                    continue
            if not reactive_compact_attempted and _is_prompt_too_long_error(exc):
                reactive_compact_attempted = True
                yield StatusEvent(message=REACTIVE_COMPACT_STATUS_MESSAGE), None
                async for event, usage in _stream_compaction(trigger="reactive", force=True):
                    yield event, usage
                compacted_messages, was_compacted = last_compaction_result
                if compacted_messages is not messages:
                    messages[:] = compacted_messages
                if was_compacted:
                    continue
            if "connect" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                yield ErrorEvent(message=f"Network error: {error_msg}. Check your internet connection and try again."), None
            else:
                yield ErrorEvent(message=f"API error: {error_msg}"), None
            return

        if final_message is None:
            raise RuntimeError("Model stream finished without a final message")

        coordinator_context_message: ConversationMessage | None = None
        # NOTE: substring, not startswith — the real coordinator system
        # prompt (``get_coordinator_system_prompt``) begins with "You are
        # Claude Code, an AI assistant that orchestrates …" and only
        # contains "You are a **coordinator**." a few lines in. The old
        # ``startswith`` check never matched, so the context message was
        # never moved out of the way and accumulated in history.
        if "You are a **coordinator**." in context.system_prompt:
            if messages and messages[-1].role == "user" and messages[-1].text.startswith("# Coordinator User Context"):
                coordinator_context_message = messages.pop()

        if final_message.role == "assistant" and final_message.is_effectively_empty():
            log.warning("dropping empty assistant message from provider response")
            yield ErrorEvent(
                message=(
                    "Model returned an empty assistant message. "
                    "The turn was ignored to keep the session healthy."
                )
            ), usage
            return

        messages.append(final_message)
        # --- cost tracking + audit (#28/#29) + auto-checkpoint (#37) ---
        _record_llm_usage(context, usage, duration_seconds=time.monotonic() - llm_turn_started)
        cost_alert = _record_cost_and_alert(context, usage)
        if cost_alert is not None:
            yield StatusEvent(message=cost_alert.message), None
            if cost_alert.level == "halt":
                yield ErrorEvent(message="Session halted: cost budget exceeded."), None
                return
        if context.control_bus is not None:
            context.control_bus.feed("llm")
        if (
            context.checkpoint_interval_turns > 0
            and turn_count % context.checkpoint_interval_turns == 0
        ):
            _auto_checkpoint(context, messages, turn_count)
        # ----------------------------------------------------------------
        yield AssistantTurnComplete(message=final_message, usage=usage), usage

        if not final_message.tool_uses:
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.STOP,
                    {
                        "event": HookEvent.STOP.value,
                        "stop_reason": "tool_uses_empty",
                    },
                )
            return

        tool_calls = final_message.tool_uses

        if len(tool_calls) == 1:
            # Single tool: sequential (stream events immediately)
            tc = tool_calls[0]
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None
            try:
                result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
            except Exception as exc:
                log.exception("tool execution raised: name=%s id=%s", tc.name, tc.id)
                result = ToolResultBlock(
                    tool_use_id=tc.id,
                    content=f"Tool {tc.name} failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
                metadata=result.result_metadata,
            ), None
            tool_results = [result]
        else:
            # Multiple tools: execute concurrently, emit events after.
            # Optimization #8: ``parallel_tools`` flag. When False, run
            # sequentially (deterministic, for debugging).
            parallel_tools = bool(
                (context.tool_metadata or {}).get("parallel_tools", True)
            )
            for tc in tool_calls:
                yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None

            async def _run(tc):
                return await _execute_tool_call(context, tc.name, tc.id, tc.input)

            if parallel_tools:
                raw_results = await asyncio.gather(
                    *[_run(tc) for tc in tool_calls], return_exceptions=True
                )
            else:
                raw_results = []
                for tc in tool_calls:
                    try:
                        raw_results.append(await _run(tc))
                    except Exception as exc:
                        raw_results.append(exc)
            tool_results = []
            for tc, result in zip(tool_calls, raw_results):
                if isinstance(result, BaseException):
                    log.exception(
                        "tool execution raised: name=%s id=%s",
                        tc.name,
                        tc.id,
                        exc_info=result,
                    )
                    result = ToolResultBlock(
                        tool_use_id=tc.id,
                        content=f"Tool {tc.name} failed: {type(result).__name__}: {result}",
                        is_error=True,
                    )
                tool_results.append(result)

            for tc, result in zip(tool_calls, tool_results):
                yield ToolExecutionCompleted(
                    tool_name=tc.name,
                    output=result.content,
                    is_error=result.is_error,
                    metadata=result.result_metadata,
                ), None

        messages.append(ConversationMessage(role="user", content=tool_results))
        # Re-append the coordinator context AFTER the tool results (the
        # old position — right after AssistantTurnComplete — inserted a
        # user message BETWEEN the assistant tool_use and its tool
        # results, which broke the tool pairing and made the sanitizer
        # delete the whole turn). Here it stays the LAST message, so the
        # next API call sees fresh coordinator context, and the next
        # iteration pops it again before appending the response.
        if coordinator_context_message is not None:
            messages.append(coordinator_context_message)

    if context.max_turns is not None:
        raise MaxTurnsExceeded(context.max_turns)
    raise RuntimeError("Query loop exited without a max_turns limit or final response")


def _record_llm_usage(
    context: QueryContext, usage: UsageSnapshot, *, duration_seconds: float = 0.0
) -> None:
    """Audit-log one completed LLM turn (#29). Never raises."""
    try:
        from openharness.services.audit_log import get_audit_log

        get_audit_log().record_llm_call(
            model=context.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0) or 0),
            duration_seconds=max(0.0, float(duration_seconds)),
        )
    except Exception:
        pass


def _record_cost_and_alert(context: QueryContext, usage: UsageSnapshot):
    """Feed the session cost tracker (#28); return an alert when crossed."""
    tracker = context.session_cost_tracker
    if tracker is None or usage is None:
        return None
    try:
        return tracker.record(context.model, usage)
    except Exception:
        return None


def _auto_checkpoint(context: QueryContext, messages: list[ConversationMessage], turn: int) -> None:
    """Snapshot the conversation every N turns (#37). Never raises."""
    try:
        from openharness.services.checkpoints import CheckpointManager

        session_id = str((context.tool_metadata or {}).get("session_id") or "default")
        manager = CheckpointManager(session_id)
        manager.create(list(messages), label=f"auto turn {turn}")
    except Exception:
        log.debug("auto-checkpoint failed", exc_info=True)


# Tools that legitimately block on humans or manage their own deadlines;
# the generic #27 stage timeout does not apply to them.
_TOOLS_WITH_OWN_TIME = {
    "ask_user_question",   # waits for a human reply
    "send_message",        # waits for IM delivery
    "agent",               # runs a full sub-agent with its own budget
    "task_output",         # streams a background task
    "sleep",               # intentional delay, bounded by its own input
    "render_scenes",       # per-job timeouts inside the render pipeline
    "qa_review",           # per-call timeouts inside the QA pipeline
    "audio_gen",           # endpoint timeout inside the generator
    "remote_trigger",      # external runner with its own lifecycle
}


async def _guarded_tool_execute(
    context: QueryContext,
    tool: Any,
    parsed_input: Any,
    exec_context: Any,
):
    """Run ``tool.execute`` with the #27 stage timeout, #29 audit and
    #36 watchdog feed. Timeouts surface as error results, never crashes."""
    from openharness.services.audit_log import get_audit_log

    t0 = time.monotonic()
    bus = context.control_bus
    timeout = (
        context.tool_timeout_seconds
        if context.tool_timeout_seconds > 0 and tool.name not in _TOOLS_WITH_OWN_TIME
        else 0.0
    )
    is_error = False
    try:
        if timeout > 0 and bus is not None:
            result = await bus.call_with_limit(
                tool.execute(parsed_input, exec_context),
                timeout_seconds=timeout,
                stage=f"tool:{tool.name}",
            )
        else:
            result = await tool.execute(parsed_input, exec_context)
        is_error = bool(result.is_error)
        return result
    except Exception as exc:
        # StageTimeoutExceeded (#27) or any tool-level failure becomes an
        # error result so the model can react instead of the loop dying.
        is_error = True
        return ToolResult(
            output=f"{tool.name} aborted: {type(exc).__name__}: {exc}",
            is_error=True,
        )
    finally:
        elapsed = time.monotonic() - t0
        try:
            get_audit_log().record_tool_call(
                tool=tool.name,
                is_error=is_error,
                duration_seconds=elapsed,
            )
        except Exception:
            pass
        if bus is not None:
            bus.feed(f"tool:{tool.name}")


async def _execute_tool_call(
    context: QueryContext,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, object],
) -> ToolResultBlock:
    if context.hook_executor is not None:
        pre_hooks = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE,
            {"tool_name": tool_name, "tool_input": tool_input, "event": HookEvent.PRE_TOOL_USE.value},
        )
        if pre_hooks.blocked:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre_hooks.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    log.debug("tool_call start: %s id=%s", tool_name, tool_use_id)

    tool = context.tool_registry.get(tool_name)
    if tool is None:
        log.warning("unknown tool: %s", tool_name)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        log.warning("invalid input for %s: %s", tool_name, exc)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Invalid input for {tool_name}: {exc}",
            is_error=True,
        )

    # Normalize common tool inputs before permission checks so path rules apply
    # consistently across built-in tools that use `file_path`, `path`, or
    # directory-scoped roots such as `glob`/`grep`.
    _file_path = _resolve_permission_file_path(context.cwd, tool_input, parsed_input)
    _command = _extract_permission_command(tool_input, parsed_input)
    log.debug("permission check: %s read_only=%s path=%s cmd=%s",
              tool_name, tool.is_read_only(parsed_input), _file_path, _command and _command[:80])
    decision = context.permission_checker.evaluate(
        tool_name,
        is_read_only=tool.is_read_only(parsed_input),
        file_path=_file_path,
        command=_command,
    )
    if not decision.allowed:
        if decision.requires_confirmation and context.permission_prompt is not None:
            log.debug("permission prompt for %s: %s", tool_name, decision.reason)
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.NOTIFICATION,
                    {
                        "event": HookEvent.NOTIFICATION.value,
                        "notification_type": "permission_prompt",
                        "tool_name": tool_name,
                        "reason": decision.reason,
                    },
                )
            confirmed = await context.permission_prompt(tool_name, decision.reason)
            if not confirmed:
                log.debug("permission denied by user for %s", tool_name)
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=decision.reason or f"Permission denied for {tool_name}",
                    is_error=True,
                )
        else:
            log.debug("permission blocked for %s: %s", tool_name, decision.reason)
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=decision.reason or f"Permission denied for {tool_name}",
                is_error=True,
            )

    log.debug("executing %s ...", tool_name)
    # Optimization #7: check the tool result cache before executing.
    # Use ``get_or_execute`` which is thundering-herd-safe: if two
    # coroutines call the same read-only tool concurrently, only the
    # first executes; the second awaits the first's result.
    from openharness.services.tool_cache import get_tool_cache
    tool_cache = get_tool_cache()
    use_cache = (
        tool_cache is not None
        and tool.is_read_only(parsed_input)
        and tool_cache.is_cacheable(tool_name)
    )
    if use_cache:
        async def _execute_and_format():
            t0 = time.monotonic()
            tool_exec_metadata_inner = dict(context.tool_metadata or {})
            tool_exec_metadata_inner.setdefault("tool_registry", context.tool_registry)
            tool_exec_metadata_inner.setdefault("ask_user_prompt", context.ask_user_prompt)
            tool_exec_metadata_inner.setdefault(
                "session_cost_tracker", getattr(context, "session_cost_tracker", None)
            )
            inner_result = await _guarded_tool_execute(
                context,
                tool,
                parsed_input,
                ToolExecutionContext(
                    cwd=context.cwd,
                    metadata=tool_exec_metadata_inner,
                    hook_executor=context.hook_executor,
                ),
            )
            elapsed_inner = time.monotonic() - t0
            log.debug("executed %s in %.2fs err=%s output_len=%d",
                      tool_name, elapsed_inner, inner_result.is_error, len(inner_result.output or ""))
            inner_inline, inner_artifact = _offload_tool_output_if_needed(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                output=inner_result.output,
            )
            if inner_artifact is not None:
                _remember_active_artifact(context.tool_metadata, str(inner_artifact))
            return {
                "output": inner_inline,
                "is_error": inner_result.is_error,
                "metadata": dict(inner_result.metadata or {}),
            }

        cached_result = await tool_cache.get_or_execute(
            tool_name, dict(tool_input), _execute_and_format
        )
        if cached_result is not None:
            log.debug("tool cache returned result for %s (err=%s)", tool_name, cached_result.get("is_error"))
            # ``get_or_execute`` returns the executor's result dict even
            # for errors (it just doesn't cache them). Return it here so
            # the user sees the error from the first execution, not a
            # confusing re-run.
            tool_result = ToolResultBlock(
                tool_use_id=tool_use_id,
                content=cached_result.get("output", ""),
                is_error=bool(cached_result.get("is_error", False)),
                result_metadata=dict(cached_result.get("metadata") or {}),
            )
            # Cache hits must not skip the post-execution bookkeeping:
            # read_file_state carryover (task focus / edit-approval
            # tracking) and PostToolUse hooks (audit, logging, security
            # observers). Previously a cache hit returned early and hooks
            # never saw the tool call at all.
            _record_tool_carryover(
                context,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_result.content,
                tool_result_metadata=tool_result.result_metadata,
                is_error=tool_result.is_error,
                resolved_file_path=_file_path,
            )
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.POST_TOOL_USE,
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_output": tool_result.content,
                        "tool_is_error": tool_result.is_error,
                        "from_cache": True,
                        "event": HookEvent.POST_TOOL_USE.value,
                    },
                )
            return tool_result
        # cached_result is None → executor returned None (shouldn't happen
        # for normal tools) or raised. Fall through to the non-cached path.

    t0 = time.monotonic()
    # Build the tool execution metadata, starting from the engine's
    # ``tool_metadata`` so we carry the mcp_manager, settings, etc.
    tool_exec_metadata = dict(context.tool_metadata or {})
    tool_exec_metadata.setdefault("tool_registry", context.tool_registry)
    tool_exec_metadata.setdefault("ask_user_prompt", context.ask_user_prompt)
    # The REAL tracked spend (#28) — cost_governor's status/check reads
    # this so the model sees actual session cost, not just its own
    # manual records (G9).
    tool_exec_metadata.setdefault("session_cost_tracker", getattr(context, "session_cost_tracker", None))
    result = await _guarded_tool_execute(
        context,
        tool,
        parsed_input,
        ToolExecutionContext(
            cwd=context.cwd,
            metadata=tool_exec_metadata,
            hook_executor=context.hook_executor,
        ),
    )
    elapsed = time.monotonic() - t0
    log.debug("executed %s in %.2fs err=%s output_len=%d",
              tool_name, elapsed, result.is_error, len(result.output or ""))
    inline_output, artifact_path = _offload_tool_output_if_needed(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        output=result.output,
    )
    if artifact_path is not None:
        _remember_active_artifact(context.tool_metadata, str(artifact_path))
    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=inline_output,
        is_error=result.is_error,
        result_metadata=dict(result.metadata or {}),
    )
    # Optimization #7: invalidate cache on mutating tools (successful or
    # not). bash is judged PER COMMAND now — ``ls``/``git status`` between
    # two edits no longer wipe every cached read.
    if tool_cache is not None and tool_cache.is_mutating(tool_name, tool_input):
        tool_cache.invalidate_all()
    _record_tool_carryover(
        context,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_result.content,
        tool_result_metadata=result.metadata,
        is_error=tool_result.is_error,
        resolved_file_path=_file_path,
    )
    if context.hook_executor is not None:
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )
    return tool_result


def _resolve_permission_file_path(
    cwd: Path,
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    # ``report_path`` is included: several read-only-classified QA tools
    # (pixel_qa, frame_diff, scene_lint, style_consistency, vlm_qa)
    # WRITE a JSON report to a caller-chosen path. Without this key the
    # permission system never saw that write target, so sensitive-path
    # and deny rules did not apply to it at all.
    for key in ("file_path", "path", "root", "report_path"):
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    for attr in ("file_path", "path", "root", "report_path"):
        value = getattr(parsed_input, attr, None)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    return None


def _extract_permission_command(
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    value = raw_input.get("command")
    if isinstance(value, str) and value.strip():
        return value

    value = getattr(parsed_input, "command", None)
    if isinstance(value, str) and value.strip():
        return value

    return None
