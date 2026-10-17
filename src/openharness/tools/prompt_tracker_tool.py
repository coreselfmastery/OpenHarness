"""prompt_tracker tool — score prompt templates by real success rate.

Every prompt is versioned and registered here; every scene generated
from it records a pass/fail at QA. `stats` surfaces each version's
success rate and flags low performers for revision or retirement —
prompt engineering becomes measurable instead of anecdotal.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class PromptTrackerInput(BaseModel):
    """Arguments for the prompt_tracker tool."""

    action: str = Field(
        description=(
            "Operation: register (new/updated template version), record "
            "(pass/fail result for one use), stats (per-version success "
            "rates + flags), best (best version of one prompt), retire."
        ),
        pattern="^(register|record|stats|best|retire)$",
    )
    prompt_id: str = Field(
        default="",
        description="Prompt template id (e.g. 'scene-title-card'); required for "
        "register/record/best/retire, unused for stats.",
    )
    version: int = Field(default=1, ge=1, le=10_000, description="Template version.")
    template: str = Field(
        default="",
        description="Template text for register (kept private to the store).",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags for register (e.g. ['scene', 'title-card'])."
    )
    passed: bool = Field(
        default=True,
        description="record: whether the scene generated from this prompt passed QA.",
    )
    scene: str = Field(default="", description="record: scene name for traceability.")


class PromptTrackerTool(BaseTool):
    """Versioned prompt registry scored by QA outcomes."""

    name = "prompt_tracker"
    description = (
        "Make prompt engineering measurable: `register` versioned prompt "
        "templates, `record` each use's QA pass/fail, then `stats` ranks every "
        "version by real success rate and flags low performers (revision or "
        "retirement candidates). `best` picks the highest-performing version "
        "of a prompt before generating the next scene."
    )
    input_model = PromptTrackerInput

    def is_read_only(self, arguments: PromptTrackerInput) -> bool:
        return arguments.action in {"stats", "best"}

    async def execute(
        self, arguments: PromptTrackerInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.prompt_tracker import get_prompt_tracker

        tracker = get_prompt_tracker()

        if arguments.action == "register":
            if not arguments.prompt_id:
                return ToolResult(
                    output="prompt_tracker: `prompt_id` is required.", is_error=True
                )
            if not arguments.template:
                return ToolResult(
                    output=(
                        "prompt_tracker: `template` is required for register — "
                        "an unversioned registry is just names."
                    ),
                    is_error=True,
                )
            tracker.register_prompt(
                arguments.prompt_id,
                version=arguments.version,
                template=arguments.template,
                tags=arguments.tags,
            )
            return ToolResult(
                output=(
                    f"[prompt_tracker] registered {arguments.prompt_id} "
                    f"v{arguments.version} ({len(arguments.template)} chars)."
                )
            )

        if arguments.action == "record":
            if not arguments.prompt_id:
                return ToolResult(
                    output="prompt_tracker: `prompt_id` is required.", is_error=True
                )
            tracker.record_result(
                arguments.prompt_id,
                passed=arguments.passed,
                version=arguments.version,
                scene=arguments.scene,
            )
            outcome = "PASS" if arguments.passed else "FAIL"
            return ToolResult(
                output=(
                    f"[prompt_tracker] {outcome} recorded for "
                    f"{arguments.prompt_id} v{arguments.version}"
                    + (f" (scene {arguments.scene})" if arguments.scene else "")
                )
            )

        if arguments.action == "stats":
            stats = tracker.stats()
            summary = tracker.stats_summary()
            if not stats:
                return ToolResult(
                    output=(
                        "[prompt_tracker] no prompts registered yet — register "
                        "templates and record their QA outcomes."
                    ),
                    metadata={"stats": []},
                )
            lines = [f"[prompt_tracker] {len(stats)} prompt version(s):"]
            for item in stats:
                rate = (
                    f"{item['success_rate']:.0%}" if item["success_rate"] is not None
                    else "n/a"
                )
                flags = []
                if item["retired"]:
                    flags.append("RETIRED")
                if item["flagged_low_performer"]:
                    flags.append("LOW PERFORMER — revise or retire")
                lines.append(
                    f"  - {item['prompt_id']} v{item['version']}: {rate} over "
                    f"{item['uses']} use(s)"
                    + (f"  [{' | '.join(flags)}]" if flags else "")
                )
            return ToolResult(
                output="\n".join(lines),
                metadata={"stats": stats, "summary": summary},
            )

        if arguments.action == "best":
            best = tracker.best_version(arguments.prompt_id)
            if best is None:
                return ToolResult(
                    output=(
                        f"[prompt_tracker] no recorded results for "
                        f"{arguments.prompt_id!r} yet."
                    ),
                    metadata={"best": None},
                )
            return ToolResult(
                output=(
                    f"[prompt_tracker] best version of {arguments.prompt_id}: "
                    f"v{best['version']} ({best['success_rate']:.0%} over "
                    f"{best['uses']} uses)."
                ),
                metadata={"best": best},
            )

        # retire
        if not arguments.prompt_id:
            return ToolResult(
                output="prompt_tracker: `prompt_id` is required.", is_error=True
            )
        updated = tracker.retire_prompt(arguments.prompt_id, version=arguments.version)
        if not updated:
            return ToolResult(
                output="[prompt_tracker] no matching version to retire.",
                is_error=True,
            )
        return ToolResult(
            output=(
                f"[prompt_tracker] retired {updated} version(s) of "
                f"{arguments.prompt_id}."
            )
        )
