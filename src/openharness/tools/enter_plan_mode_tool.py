"""Tool for entering plan permission mode."""

from __future__ import annotations

from pydantic import BaseModel

from openharness.config.settings import load_settings, save_settings
from openharness.permissions import PermissionMode
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class EnterPlanModeToolInput(BaseModel):
    """No-op input model."""


def _rebuild_live_permission_checker(context: ToolExecutionContext) -> bool:
    """Rebuild the running engine's PermissionChecker from current settings.

    ``enter/exit_plan_mode`` used to only write the settings FILE: the
    engine's checker is built once at startup and keeps its own snapshot
    of the permission settings, so the live session kept enforcing the
    OLD mode (plan mode that didn't block anything, or a plan-mode
    session that could never exit) until a restart or an unrelated
    /permissions command happened to rebuild it.
    """
    engine = (context.metadata or {}).get("engine_ref") if context.metadata else None
    if engine is None or not hasattr(engine, "set_permission_checker"):
        return False
    try:
        from openharness.permissions import build_permission_checker

        cwd = str(getattr(engine, "cwd", context.cwd))
        engine.set_permission_checker(build_permission_checker(load_settings(), cwd=cwd))
        return True
    except Exception:  # noqa: BLE001 - mode change must still report
        return False


class EnterPlanModeTool(BaseTool):
    """Switch settings permission mode to plan."""

    name = "enter_plan_mode"
    description = "Switch permission mode to plan."
    input_model = EnterPlanModeToolInput

    async def execute(self, arguments: EnterPlanModeToolInput, context: ToolExecutionContext) -> ToolResult:
        del arguments
        settings = load_settings()
        settings.permission.mode = PermissionMode.PLAN
        save_settings(settings)
        applied = _rebuild_live_permission_checker(context)
        note = " (live session updated)" if applied else " (live session: restart to apply)"
        return ToolResult(output=f"Permission mode set to plan{note}")
