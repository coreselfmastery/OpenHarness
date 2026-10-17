"""failure_patterns tool — learn which scene types fail QA most.

Every QA failure is logged with the scene type, reason, and prompt
used. Over time the database builds a failure rate per scene type;
when a new scene of a high-failure type is about to be generated,
`hint` returns an explicit stricter-prompt recommendation with the
concrete historical failure reasons to constrain against.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class FailurePatternsInput(BaseModel):
    """Arguments for the failure_patterns tool."""

    action: str = Field(
        description=(
            "Operation: record_failure, record_success, rates (per scene type), "
            "hint (strictness guidance before generating), stats."
        ),
        pattern="^(record_failure|record_success|rates|hint|stats)$",
    )
    scene_type: str = Field(
        default="",
        description="Scene type slug: 'title-card', 'chart-overlay', 'math', ...",
    )
    reason: str = Field(
        default="",
        description="Failure reason for record_failure (be specific — it feeds hints).",
    )
    prompt_id: str = Field(
        default="",
        description="Prompt template id used for the scene (links to prompt_tracker).",
    )
    project: str = Field(default="", description="Project name for context.")
    min_samples: int = Field(
        default=3, ge=1, le=1000, description="rates/hint: minimum samples per type."
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="hint: failure rate at/above which a scene type is high-risk.",
    )


class FailurePatternsTool(BaseTool):
    """Failure-rate statistics per scene type; feeds stricter prompts."""

    name = "failure_patterns"
    description = (
        "Break the repeat-failure cycle: `record_failure`/`record_success` log "
        "QA outcomes per scene type (with reason + prompt id); `rates` shows "
        "failure rate per type with top reasons; `hint` returns strictness "
        "guidance BEFORE generating a scene of a given type — high-risk types "
        "get an explicit stricter-prompt recommendation built from the exact "
        "historical failure reasons."
    )
    input_model = FailurePatternsInput

    def is_read_only(self, arguments: FailurePatternsInput) -> bool:
        return arguments.action in {"rates", "hint", "stats"}

    async def execute(
        self, arguments: FailurePatternsInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.failure_patterns import get_failure_patterns

        store = get_failure_patterns()

        if arguments.action in {"record_failure", "record_success"}:
            if not arguments.scene_type:
                return ToolResult(
                    output="failure_patterns: `scene_type` is required.",
                    is_error=True,
                )
            if arguments.action == "record_failure":
                if not arguments.reason:
                    return ToolResult(
                        output=(
                            "failure_patterns: `reason` is required for "
                            "record_failure — vague reasons make useless hints."
                        ),
                        is_error=True,
                    )
                store.record_failure(
                    arguments.scene_type,
                    reason=arguments.reason,
                    prompt_id=arguments.prompt_id,
                    project=arguments.project,
                )
                return ToolResult(
                    output=(
                        f"[failure_patterns] logged FAILURE for "
                        f"{arguments.scene_type!r}: {arguments.reason}"
                    )
                )
            store.record_success(
                arguments.scene_type,
                prompt_id=arguments.prompt_id,
                project=arguments.project,
            )
            return ToolResult(
                output=(
                    f"[failure_patterns] logged SUCCESS for {arguments.scene_type!r}."
                )
            )

        if arguments.action == "rates":
            rates = store.failure_rates(min_samples=arguments.min_samples)
            if not rates:
                return ToolResult(
                    output=(
                        "[failure_patterns] not enough samples yet (min "
                        f"{arguments.min_samples} per type) — record QA outcomes "
                        "as scenes pass/fail."
                    ),
                    metadata={"rates": []},
                )
            lines = ["[failure_patterns] failure rate per scene type:"]
            for item in rates:
                lines.append(
                    f"  - {item['scene_type']}: {item['failure_rate']:.0%} "
                    f"({item['failures']}/{item['total']})"
                    + (
                        " — " + "; ".join(
                            f"{r['reason']} x{r['count']}" for r in item["top_reasons"]
                        )
                        if item["top_reasons"]
                        else ""
                    )
                )
            return ToolResult(output="\n".join(lines), metadata={"rates": rates})

        if arguments.action == "hint":
            if not arguments.scene_type:
                return ToolResult(
                    output="failure_patterns: `scene_type` is required for hint.",
                    is_error=True,
                )
            hint = store.strictness_hint(
                arguments.scene_type,
                threshold=arguments.threshold,
                min_samples=arguments.min_samples,
            )
            if hint["high_risk"]:
                return ToolResult(
                    output=(
                        f"[failure_patterns] HIGH RISK — {hint['scene_type']}: "
                        f"{hint['failure_rate']:.0%} failure rate over "
                        f"{hint['samples']} samples.\n  {hint['recommendation']}"
                    ),
                    metadata=hint,
                )
            return ToolResult(
                output=(
                    f"[failure_patterns] {hint['scene_type']}: "
                    f"{hint['failure_rate']:.0%} over {hint['samples']} samples — "
                    "standard prompt is fine."
                ),
                metadata=hint,
            )

        # stats
        stats = store.stats()
        return ToolResult(
            output=(
                "[failure_patterns] stats:\n"
                f"  events: {stats['total_events']} "
                f"({stats['total_failures']} failures)\n"
                f"  scene types tracked: {stats['scene_types']}\n"
                f"  store: {stats['path']} (available: {stats['available']})"
            ),
            metadata=stats,
        )
