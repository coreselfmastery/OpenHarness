"""High-level conversation engine."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from openharness.api.client import SupportsStreamingMessages
from openharness.engine.cost_tracker import CostTracker
from openharness.coordinator.coordinator_mode import get_coordinator_user_context
from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, sanitize_conversation_messages
from openharness.engine.query import AskUserPrompt, PermissionPrompt, QueryContext, remember_user_goal, run_query
from openharness.engine.stream_events import AssistantTurnComplete, StreamEvent
from openharness.config.settings import Settings
from openharness.hooks import HookEvent, HookExecutor
from openharness.permissions.checker import PermissionChecker
from openharness.services.autodream.service import schedule_auto_dream
from openharness.tools.base import ToolRegistry


class QueryEngine:
    """Owns conversation history and the tool-aware model loop."""

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        tool_registry: ToolRegistry,
        permission_checker: PermissionChecker,
        cwd: str | Path,
        model: str,
        system_prompt: str,
        max_tokens: int = 4096,
        context_window_tokens: int | None = None,
        auto_compact_threshold_tokens: int | None = None,
        max_turns: int | None = 8,
        permission_prompt: PermissionPrompt | None = None,
        ask_user_prompt: AskUserPrompt | None = None,
        hook_executor: HookExecutor | None = None,
        tool_metadata: dict[str, object] | None = None,
        settings: Settings | None = None,
        control_bus: object | None = None,
        session_cost_tracker: object | None = None,
    ) -> None:
        self._api_client = api_client
        self._tool_registry = tool_registry
        self._permission_checker = permission_checker
        self._cwd = Path(cwd).resolve()
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._effort = settings.effort if settings is not None else None
        self._context_window_tokens = context_window_tokens
        self._auto_compact_threshold_tokens = auto_compact_threshold_tokens
        # Optimization #5 + #9: forward from settings if not explicitly given.
        if settings is not None:
            self._auto_compact_threshold_ratio = getattr(settings, "auto_compact_threshold_ratio", None)
            self._thinking_budget = getattr(settings, "thinking_budget", None)
        else:
            self._auto_compact_threshold_ratio = None
            self._thinking_budget = None
        self._max_turns = max_turns
        self._permission_prompt = permission_prompt
        self._ask_user_prompt = ask_user_prompt
        self._hook_executor = hook_executor
        self._tool_metadata = tool_metadata or {}
        self._settings = settings
        self._messages: list[ConversationMessage] = []
        # Rollback-aware turn accounting (#37). ``rollback_messages`` can
        # fire MID-TURN (the checkpoint tool runs inside the query loop,
        # which operates on a COPY of the history). Without this tracking,
        # ``submit_message`` would blindly re-adopt that pre-rollback copy
        # at AssistantTurnComplete and silently wipe the restore.
        self._messages_version = 0
        self._active_query_messages: list[ConversationMessage] | None = None
        self._rollback_tail_position = 0
        self._cost_tracker = CostTracker()
        # Waves 2-6 control plane (#26-#40): the shared control bus (kill
        # switch / pause gate / watchdog / injected messages) and the
        # session cost tracker (#28). Both optional — engine behavior is
        # identical to before when they are None.
        self._control_bus = control_bus
        self._session_cost_tracker = session_cost_tracker
        self._tool_timeout_seconds = 0.0
        self._checkpoint_interval_turns = 0
        if settings is not None:
            control = getattr(settings, "control", None)
            if control is not None and bool(getattr(control, "enabled", False)):
                if bool(getattr(control, "stage_time_limits", True)):
                    self._tool_timeout_seconds = float(
                        getattr(control, "tool_timeout_seconds", 900) or 0
                    )
                if bool(getattr(control, "checkpoints", True)):
                    self._checkpoint_interval_turns = max(
                        0, int(getattr(control, "checkpoint_interval_turns", 10) or 0)
                    )
        # Optimization #7 + #10: count user turns so we only auto-save
        # every ``auto_save_interval`` turns (default 10). Saving on
        # every turn is too aggressive for long video-generation runs.
        # Initial _last_save_turn is -interval so the FIRST turn always
        # saves (otherwise turns 1..N-1 have no snapshot, which is
        # dangerous on Kaggle where notebooks die without warning).
        self._turn_count = 0
        self._last_save_turn = -max(1, int(getattr(settings, "auto_save_interval", 10) or 10)) if settings is not None else -10

    @property
    def messages(self) -> list[ConversationMessage]:
        """Return the current conversation history."""
        return list(self._messages)

    @property
    def live_messages(self) -> list[ConversationMessage]:
        """Return the ACTUAL internal message list (not a copy).

        Used by the checkpoint tool (#37) to snapshot the conversation
        without copying on every read. Callers must not mutate; use
        :meth:`rollback_messages` for that.
        """
        return self._messages

    def rollback_messages(self, messages: list[ConversationMessage]) -> None:
        """Replace the conversation history in place (#37 rollback).

        The internal list object is mutated (not rebound) so any
        references held by the query loop stay valid.

        When a rollback lands MID-TURN, the in-flight query loop keeps
        running on its own copy of the (pre-rollback) history. The turn's
        adoption step (:meth:`_adopt_query_messages`) must therefore
        know where that copy stood when the rollback happened, so it can
        keep the restored history as the base and preserve only the
        post-rollback tail (the tool result + final answer the user
        already saw streamed) instead of resurrecting the abandoned
        branch.
        """
        self._messages[:] = list(messages)
        self._messages_version += 1
        active = self._active_query_messages
        self._rollback_tail_position = len(active) if active is not None else len(self._messages)

    def _adopt_query_messages(
        self,
        query_messages: list[ConversationMessage],
        *,
        version_at_start: int,
    ) -> None:
        """Adopt the turn's working copy back into the engine history.

        No mid-turn rollback: the copy IS the new history (unchanged
        behaviour). Mid-turn rollback: keep the restored history as the
        base and append only the messages generated after the rollback
        point, so the restore survives the turn. Orphaned tool results
        in that tail (their ``tool_use`` lived on the abandoned branch)
        are stripped by ``sanitize_conversation_messages``.
        """
        if self._messages_version == version_at_start:
            self._messages = list(query_messages)
            return
        tail = query_messages[self._rollback_tail_position :]
        self._messages = sanitize_conversation_messages(
            list(self._messages) + list(tail)
        )

    @property
    def cwd(self) -> Path:
        """Return the resolved working directory this engine operates in."""
        return self._cwd

    @property
    def max_turns(self) -> int | None:
        """Return the maximum number of agentic turns per user input, if capped."""
        return self._max_turns

    @property
    def api_client(self) -> SupportsStreamingMessages:
        """Return the active API client."""
        return self._api_client

    @property
    def model(self) -> str:
        """Return the active model identifier."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._system_prompt

    @property
    def tool_metadata(self) -> dict[str, object]:
        """Return the mutable tool metadata/carry-over state."""
        return self._tool_metadata

    @property
    def total_usage(self):
        """Return the total usage across all turns."""
        return self._cost_tracker.total

    def clear(self) -> None:
        """Clear the in-memory conversation history."""
        self._messages.clear()
        self._cost_tracker = CostTracker()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the active system prompt for future turns."""
        self._system_prompt = prompt

    def set_model(self, model: str) -> None:
        """Update the active model for future turns."""
        self._model = model

    def set_effort(self, effort: str | None) -> None:
        """Update the active reasoning effort for future turns."""
        self._effort = effort

    def set_api_client(self, api_client: SupportsStreamingMessages) -> None:
        """Update the active API client for future turns."""
        self._api_client = api_client

    def set_max_turns(self, max_turns: int | None) -> None:
        """Update the maximum number of agentic turns per user input."""
        self._max_turns = None if max_turns is None else max(1, int(max_turns))

    def set_permission_checker(self, checker: PermissionChecker) -> None:
        """Update the active permission checker for future turns."""
        self._permission_checker = checker

    def _build_coordinator_context_message(self) -> ConversationMessage | None:
        """Build a synthetic user message carrying coordinator runtime context."""
        context = get_coordinator_user_context()
        worker_tools_context = context.get("workerToolsContext")
        if not worker_tools_context:
            return None
        return ConversationMessage(
            role="user",
            content=[TextBlock(text=f"# Coordinator User Context\n\n{worker_tools_context}")],
        )

    def load_messages(self, messages: list[ConversationMessage]) -> None:
        """Replace the in-memory conversation history."""
        self._messages = list(messages)

    def _schedule_auto_dream(self) -> None:
        """Fire-and-forget background memory consolidation after a user turn."""
        if self._settings is None:
            return
        context = self._tool_metadata.get("autodream_context")
        kwargs = dict(context) if isinstance(context, dict) else {}
        schedule_auto_dream(
            cwd=self._cwd,
            settings=self._settings,
            model=self._model,
            current_session_id=str(self._tool_metadata.get("session_id") or ""),
            **kwargs,
        )

    def _prepare_session_memory(self) -> None:
        """Expose file-backed session memory to compaction when enabled."""

        if self._settings is None or not self._settings.memory.session_memory_enabled:
            return
        if not self._settings.memory.enabled:
            return
        from openharness.services.session_memory import prepare_session_memory_metadata

        prepare_session_memory_metadata(
            self._cwd,
            self._tool_metadata,
            session_id=str(self._tool_metadata.get("session_id") or "default"),
        )

    async def _update_session_memory(self) -> None:
        """Persist a session checkpoint after a user turn."""

        if self._settings is None or not self._settings.memory.session_memory_enabled:
            return
        if not self._settings.memory.enabled:
            return
        from openharness.services.session_memory import update_session_memory_file

        update_session_memory_file(
            self._cwd,
            list(self._messages),
            tool_metadata=self._tool_metadata,
            session_id=str(self._tool_metadata.get("session_id") or "default"),
        )

    async def _extract_durable_memories(self) -> None:
        """Run the optional durable memory extraction pass."""

        if self._settings is None or not self._settings.memory.auto_extract_enabled:
            return
        if not self._settings.memory.enabled:
            return
        from openharness.services.memory_extract import extract_memories_from_turn

        try:
            result = await extract_memories_from_turn(
                cwd=self._cwd,
                api_client=self._api_client,
                model=self._model,
                messages=list(self._messages),
                max_records=self._settings.memory.auto_extract_max_records,
            )
        except Exception as exc:
            self._tool_metadata["memory_extract_last_error"] = str(exc)
            return
        self._tool_metadata["memory_extract_last"] = {
            "skipped": result.skipped,
            "reason": result.reason,
            "written_paths": [str(path) for path in result.written_paths],
        }

    def has_pending_continuation(self) -> bool:
        """Return True when the conversation ends with tool results awaiting a follow-up model turn."""
        if not self._messages:
            return False
        last = self._messages[-1]
        if last.role != "user":
            return False
        if not any(isinstance(block, ToolResultBlock) for block in last.content):
            return False
        for msg in reversed(self._messages[:-1]):
            if msg.role != "assistant":
                continue
            return bool(msg.tool_uses)
        return False

    async def submit_message(self, prompt: str | ConversationMessage) -> AsyncIterator[StreamEvent]:
        """Append a user message and execute the query loop."""
        user_message = (
            prompt
            if isinstance(prompt, ConversationMessage)
            else ConversationMessage.from_user_text(prompt)
        )
        if user_message.text.strip() and not self._tool_metadata.pop("_suppress_next_user_goal", False):
            remember_user_goal(self._tool_metadata, user_message.text)
        self._prepare_session_memory()
        self._messages = sanitize_conversation_messages(self._messages)
        self._messages.append(user_message)
        if self._hook_executor is not None:
            await self._hook_executor.execute(
                HookEvent.USER_PROMPT_SUBMIT,
                {
                    "event": HookEvent.USER_PROMPT_SUBMIT.value,
                    "prompt": user_message.text,
                },
            )
        context = QueryContext(
            api_client=self._api_client,
            tool_registry=self._tool_registry,
            permission_checker=self._permission_checker,
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            effort=self._effort,
            context_window_tokens=self._context_window_tokens,
            auto_compact_threshold_tokens=self._auto_compact_threshold_tokens,
            auto_compact_threshold_ratio=self._auto_compact_threshold_ratio,
            thinking_budget=self._thinking_budget,
            max_turns=self._max_turns,
            permission_prompt=self._permission_prompt,
            ask_user_prompt=self._ask_user_prompt,
            hook_executor=self._hook_executor,
            tool_metadata=self._tool_metadata,
            control_bus=self._control_bus,
            session_cost_tracker=self._session_cost_tracker,
            tool_timeout_seconds=self._tool_timeout_seconds,
            checkpoint_interval_turns=self._checkpoint_interval_turns,
        )
        query_messages = list(self._messages)
        coordinator_context = self._build_coordinator_context_message()
        if coordinator_context is not None:
            query_messages.append(coordinator_context)
        # Rollback-aware adoption (#37): a checkpoint restore fired by a
        # tool DURING this query must not be clobbered when the turn
        # completes. Track the working copy + the history version it
        # started from.
        version_at_start = self._messages_version
        self._active_query_messages = query_messages
        try:
            async for event, usage in run_query(context, query_messages):
                if isinstance(event, AssistantTurnComplete):
                    self._adopt_query_messages(query_messages, version_at_start=version_at_start)
                if usage is not None:
                    self._cost_tracker.add(usage)
                yield event
        finally:
            self._active_query_messages = None
            await self._update_session_memory()
            await self._extract_durable_memories()
            self._schedule_auto_dream()
            self._turn_count += 1
            self._auto_save_session()

    def _auto_save_session(self) -> None:
        """Save a session snapshot when ``settings.auto_save`` is True.

        Implements optimization #10 + #7: saves a ``latest.json`` snapshot
        every ``auto_save_interval`` turns (default 10). Saving on every
        turn is too aggressive for long video-generation runs — disk
        writes on Kaggle are slow and the snapshot can be 1-10 MB.
        Failures are logged and swallowed so auto-save never crashes
        the agent loop.
        """
        if self._settings is None or not getattr(self._settings, "auto_save", True):
            return
        interval = max(1, int(getattr(self._settings, "auto_save_interval", 10) or 10))
        if self._turn_count - self._last_save_turn < interval:
            return
        try:
            from openharness.services.session_storage import save_session_snapshot
            save_session_snapshot(
                cwd=self._cwd,
                model=self._model,
                system_prompt=self._system_prompt,
                messages=list(self._messages),
                usage=self._cost_tracker.total,
                session_id=str(self._tool_metadata.get("session_id") or ""),
                tool_metadata=self._tool_metadata,
            )
            self._last_save_turn = self._turn_count
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "auto-save snapshot failed", exc_info=True
            )

    async def continue_pending(self, *, max_turns: int | None = None) -> AsyncIterator[StreamEvent]:
        """Continue an interrupted tool loop without appending a new user message."""
        self._prepare_session_memory()
        self._messages = sanitize_conversation_messages(self._messages)
        context = QueryContext(
            api_client=self._api_client,
            tool_registry=self._tool_registry,
            permission_checker=self._permission_checker,
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            effort=self._effort,
            context_window_tokens=self._context_window_tokens,
            auto_compact_threshold_tokens=self._auto_compact_threshold_tokens,
            auto_compact_threshold_ratio=self._auto_compact_threshold_ratio,
            thinking_budget=self._thinking_budget,
            max_turns=max_turns if max_turns is not None else self._max_turns,
            permission_prompt=self._permission_prompt,
            ask_user_prompt=self._ask_user_prompt,
            hook_executor=self._hook_executor,
            tool_metadata=self._tool_metadata,
            control_bus=self._control_bus,
            session_cost_tracker=self._session_cost_tracker,
            tool_timeout_seconds=self._tool_timeout_seconds,
            checkpoint_interval_turns=self._checkpoint_interval_turns,
        )
        # Same rollback-aware pattern as submit_message: run the query on
        # a COPY so a mid-turn checkpoint restore replaces self._messages
        # without yanking the list out from under the in-flight loop.
        query_messages = list(self._messages)
        version_at_start = self._messages_version
        self._active_query_messages = query_messages
        try:
            async for event, usage in run_query(context, query_messages):
                if isinstance(event, AssistantTurnComplete):
                    self._adopt_query_messages(query_messages, version_at_start=version_at_start)
                if usage is not None:
                    self._cost_tracker.add(usage)
                yield event
        finally:
            self._active_query_messages = None
            await self._update_session_memory()
            await self._extract_durable_memories()
