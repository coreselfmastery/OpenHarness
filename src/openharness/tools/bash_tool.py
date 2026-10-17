"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from openharness.sandbox import SandboxUnavailableError
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.utils.shell import create_shell_subprocess


_READ_REMAINING_OUTPUT_TIMEOUT_SECONDS = 2.0


class BashToolInput(BaseModel):
    """Arguments for the bash tool."""

    command: str = Field(description="Shell command to execute")
    cwd: str | None = Field(default=None, description="Working directory override")
    timeout_seconds: int = Field(default=600, ge=1, le=600)


class BashTool(BaseTool):
    """Execute a shell command with stdout/stderr capture."""

    name = "bash"
    description = "Run a shell command in the local repository."
    input_model = BashToolInput

    async def execute(self, arguments: BashToolInput, context: ToolExecutionContext) -> ToolResult:
        cwd = Path(arguments.cwd).expanduser() if arguments.cwd else context.cwd
        preflight_error = _preflight_interactive_command(arguments.command)
        if preflight_error is not None:
            return ToolResult(
                output=preflight_error,
                is_error=True,
                metadata={"interactive_required": True},
            )
        process: asyncio.subprocess.Process | None = None
        try:
            process = await create_shell_subprocess(
                arguments.command,
                cwd=cwd,
                prefer_pty=True,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except SandboxUnavailableError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except asyncio.CancelledError:
            if process is not None:
                await _terminate_process(process, force=False)
            raise

        try:
            # communicate() reads stdout WHILE waiting for exit. The old
            # ``wait()``-only form deadlocked on any command producing
            # more than the ~64 KB pipe/pty buffer: the child blocked in
            # write(), wait() never returned, and the tool burned the
            # full timeout then falsely reported "timed out".
            # The task is kept (not wrapped in wait_for) so a TIMEOUT
            # cannot discard the output communicate already buffered:
            # the process is killed, the pipes close, communicate returns
            # the partial output it captured before the deadline.
            communicate_task: asyncio.Task = asyncio.create_task(process.communicate())
            done, _pending = await asyncio.wait(
                {communicate_task}, timeout=arguments.timeout_seconds
            )
            timed_out = not done
            if timed_out:
                await _terminate_process(process, force=True)
                try:
                    # Short grace: after the kill the pipes close and a
                    # REAL communicate() returns in milliseconds with the
                    # buffered partial output. (A long grace here would
                    # add dead latency to every timeout, and fake
                    # processes whose read never observes the kill are
                    # cancelled instead.)
                    await asyncio.wait_for(communicate_task, timeout=0.5)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - defensive
                    communicate_task.cancel()
        except asyncio.CancelledError:
            communicate_task.cancel()
            await _terminate_process(process, force=False)
            raise

        if timed_out:
            output_buffer = bytearray()
            if communicate_task.done() and not communicate_task.cancelled():
                try:
                    partial, _ = communicate_task.result()
                    output_buffer.extend(partial or b"")
                except Exception:  # noqa: BLE001 - partial output is best-effort
                    pass
            if not output_buffer:
                output_buffer.extend(await _drain_available_output(process.stdout))
                output_buffer.extend(await _read_remaining_output(process))
            return ToolResult(
                output=_format_timeout_output(
                    output_buffer,
                    command=arguments.command,
                    timeout_seconds=arguments.timeout_seconds,
                ),
                is_error=True,
                metadata={"returncode": process.returncode, "timed_out": True},
            )

        stdout, _ = await communicate_task
        output_buffer = bytearray(stdout or b"")
        text = _format_output(output_buffer)
        return ToolResult(
            output=text,
            is_error=process.returncode != 0,
            metadata={"returncode": process.returncode},
        )


async def _terminate_process(process: asyncio.subprocess.Process, *, force: bool) -> None:
    if process.returncode is not None:
        return
    if force:
        process.kill()
        await process.wait()
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _read_remaining_output(process: asyncio.subprocess.Process) -> bytearray:
    output_buffer = bytearray()
    if process.stdout is not None:
        try:
            remaining = await asyncio.wait_for(
                process.stdout.read(),
                timeout=_READ_REMAINING_OUTPUT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            remaining = b""
        output_buffer.extend(remaining)
    return output_buffer


async def _drain_available_output(
    stream: asyncio.StreamReader | None,
    *,
    read_timeout: float = 0.05,
) -> bytearray:
    output_buffer = bytearray()
    if stream is None:
        return output_buffer
    while True:
        try:
            chunk = await asyncio.wait_for(stream.read(65536), timeout=read_timeout)
        except asyncio.TimeoutError:
            return output_buffer
        if not chunk:
            return output_buffer
        output_buffer.extend(chunk)


def _format_output(output_buffer: bytearray) -> str:
    text = output_buffer.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
    if not text:
        return "(no output)"
    if len(text) > 12000:
        # Head + tail: for build/test logs the actionable part (errors,
        # summary) is at the END — a head-only truncation silently
        # dropped exactly the part the model needs to diagnose failures.
        head = text[:9000]
        tail = text[-2500:]
        return f"{head}\n...[{len(text) - 9000 - 2500} chars truncated]...\n{tail}"
    return text


def _format_timeout_output(output_buffer: bytearray, *, command: str, timeout_seconds: int) -> str:
    parts = [f"Command timed out after {timeout_seconds} seconds."]
    text = _format_output(output_buffer)
    if text != "(no output)":
        parts.extend(["", "Partial output:", text])
    hint = _interactive_command_hint(command=command, output=text)
    if hint:
        parts.extend(["", hint])
    return "\n".join(parts)


def _preflight_interactive_command(command: str) -> str | None:
    lowered_command = command.lower()
    if not _looks_like_interactive_scaffold(lowered_command):
        return None
    return (
        "This command appears to require interactive input before it can continue. "
        "The bash tool is non-interactive, so it cannot answer installer/scaffold prompts live. "
        "Prefer non-interactive flags (for example --yes, -y, --skip-install, --defaults, --non-interactive), "
        "or run the scaffolding step once in an external terminal before asking the agent to continue."
    )


def _interactive_command_hint(*, command: str, output: str) -> str | None:
    lowered_command = command.lower()
    if _looks_like_interactive_scaffold(lowered_command) or _looks_like_prompt(output):
        return (
            "This command appears to require interactive input. "
            "The bash tool is non-interactive, so prefer non-interactive flags "
            "(for example --yes, -y, --skip-install, or similar) or run the "
            "scaffolding step once in an external terminal before continuing."
        )
    return None


def _looks_like_interactive_scaffold(lowered_command: str) -> bool:
    scaffold_markers: tuple[str, ...] = (
        "create-next-app",
        "npm create ",
        "pnpm create ",
        "yarn create ",
        "bun create ",
        "pnpm dlx ",
        "npm init ",
        "pnpm init ",
        "yarn init ",
        "bunx create-",
        "npx create-",
    )
    non_interactive_markers: tuple[str, ...] = (
        "--yes",
        " -y",
        "--skip-install",
        "--defaults",
        "--non-interactive",
        "--ci",
    )
    return any(marker in lowered_command for marker in scaffold_markers) and not any(
        marker in lowered_command for marker in non_interactive_markers
    )


def _looks_like_prompt(output: str) -> bool:
    """Heuristic: detect when a command's output looks like an interactive prompt.

    Only fires on phrases that strongly indicate a real prompt — previous
    versions matched ``?`` and ``which`` anywhere in the output, which
    false-triggered on regex output, code snippets, and paths containing
    ``which``. The current markers require a leading prompt-like phrase
    (e.g. ``"would you like"``) or a standalone ``"? "`` at end-of-line.
    """
    if not output:
        return False
    prompt_markers: Iterable[str] = (
        "would you like",
        "ok to proceed",
        "select an option",
        "press enter to continue",
        "press any key",
        "[y/n]",
        "[yes/no]",
        "(y/n)",
        "do you want to",
        "are you sure",
        "confirm:",
    )
    lowered_output = output.lower()
    if any(marker in lowered_output for marker in prompt_markers):
        return True
    # A trailing "? " on its own line is a strong prompt signal; a "?"
    # buried inside other text (regex, code, JSON) is not.
    last_line = lowered_output.rstrip().rsplit("\n", 1)[-1].strip()
    if last_line.endswith("?") and len(last_line) < 80:
        return True
    return False
