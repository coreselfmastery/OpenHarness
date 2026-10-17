"""Tool for leaving plan permission mode."""

from __future__ import annotations

from pydantic import BaseModel

from openharness.config.settings import load_settings, save_settings
from openharness.permissions import PermissionMode
from openharness.tools.enter_plan_mode_tool import _rebuild_live_permission_checker
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class ExitPlanModeToolInput(BaseModel):
    """No-op input model."""


class ExitPlanModeTool(BaseTool):
    """Switch settings permission mode back to default."""

    name = "exit_plan_mode"
    description = "Switch permission mode back to default."
    input_model = ExitPlanModeToolInput

    async def execute(self, arguments: ExitPlanModeToolInput, context: ToolExecutionContext) -> ToolResult:
        del arguments
        settings = load_settings()
        settings.permission.mode = PermissionMode.DEFAULT
        save_settings(settings)
        applied = _rebuild_live_permission_checker(context)
        note = " (live session updated)" if applied else " (live session: restart to apply)"
        return ToolResult(output=f"Permission mode set to default{note}")
