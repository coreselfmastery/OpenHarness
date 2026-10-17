"""Tool for reading and updating settings."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from openharness.config.settings import load_settings, save_settings
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class ConfigToolInput(BaseModel):
    """Arguments for config access."""

    action: str = Field(default="show", description="show or set")
    key: str | None = Field(default=None)
    value: str | None = Field(default=None)


# Field names whose values must NEVER be echoed into the conversation:
# `config show` output goes to the model, the transcript, and provider
# logs — dumping raw api_key/token/secret fields leaked every credential
# the process knows about on every show call.
_SECRET_FIELD_RE = re.compile(r"(api_key|token|secret|password|webhook_url)", re.IGNORECASE)


def _redact(value: Any) -> Any:
    """Recursively redact secret-looking fields from a settings dump."""
    if isinstance(value, dict):
        return {
            key: ("***redacted***" if _SECRET_FIELD_RE.search(str(key)) and val else _redact(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class ConfigTool(BaseTool):
    """Read or update OpenHarness settings."""

    name = "config"
    description = "Read or update OpenHarness settings."
    input_model = ConfigToolInput

    async def execute(self, arguments: ConfigToolInput, context: ToolExecutionContext) -> ToolResult:
        del context
        settings = load_settings()
        if arguments.action == "show":
            import json

            return ToolResult(output=json.dumps(_redact(settings.model_dump()), indent=2))
        if arguments.action == "set" and arguments.key and arguments.value is not None:
            if _SECRET_FIELD_RE.search(arguments.key.split(".")[-1]):
                return ToolResult(
                    output=(
                        f"Refusing to set {arguments.key}: credential fields are managed "
                        "via the settings file / environment variables, not the config tool."
                    ),
                    is_error=True,
                )
            target: Any = settings
            parts = arguments.key.split(".")
            for part in parts[:-1]:
                if not hasattr(target, part):
                    return ToolResult(output=f"Unknown config key: {arguments.key}", is_error=True)
                target = getattr(target, part)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                return ToolResult(output=f"Unknown config key: {arguments.key}", is_error=True)
            current = getattr(target, leaf)
            value: Any = arguments.value
            if isinstance(current, bool):
                value = arguments.value.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(arguments.value)
            elif isinstance(current, float):
                value = float(arguments.value)
            setattr(target, leaf, value)
            save_settings(settings)
            return ToolResult(output=f"Updated {arguments.key}")
        return ToolResult(output="Usage: action=show or action=set with key/value", is_error=True)
