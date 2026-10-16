"""Batch scene generator — produces code for N scenes in a single LLM call.

This is the single biggest throughput win for video-generation pipelines:
instead of 1 LLM call per scene (200 calls for a 200-scene video), the
agent calls this tool once per batch of 5-10 scenes, cutting total call
count by ~10×.

The tool:
1. Builds a batch prompt from the scene descriptions.
2. Calls the LLM (OpenAI-compatible endpoint).
3. Parses the structured response (BATCH_ID / SCENES / SCENE_NNN blocks).
4. Validates each scene (parse → lint → write).
5. Retries individual failed scenes instead of the whole batch.

Structured output format (what the LLM must return):

    BATCH_ID: batch_003
    SCENES: 5

    SCENE_001:
      FILE: scene_001.js
      ENGINE: hyperframes
      DURATION: 8.5s
      READS: ["The sky is blue", "Because of Rayleigh scattering"]
      CODE:
        <full code here>

    SCENE_002:
      ...

Retry logic:
- Parse failure (malformed batch) → reject whole batch, return error.
- Lint failure on 1 scene → retry only that scene with a focused prompt.
- 3 consecutive failures on the same scene → return partial results +
  flag the failed scenes for escalation to the Director agent.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from openharness.api.openai_client import _normalize_openai_base_url
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

class BatchSceneToolInput(BaseModel):
    """Arguments for the batch scene generator."""

    scenes: list[str] = Field(
        description=(
            "List of scene descriptions to generate code for. Each entry "
            "is a natural-language description of one scene (e.g. 'Open on "
            "a blue sky, camera tilts down to reveal a city'). The tool "
            "generates code for ALL scenes in a single LLM call."
        ),
    )
    output_dir: str = Field(
        default="scenes",
        description="Directory to write generated scene files to.",
    )
    engine: str = Field(
        default="hyperframes",
        description="Rendering engine name (included in scene metadata).",
    )
    batch_id: str | None = Field(
        default=None,
        description="Optional batch identifier. Auto-generated if not provided.",
    )
    file_prefix: str = Field(
        default="scene_",
        description="Filename prefix. Scenes are numbered scene_001.js, scene_002.js, etc.",
    )
    file_extension: str = Field(
        default=".js",
        description="File extension for generated scene files.",
    )
    extra_context: str | None = Field(
        default=None,
        description=(
            "Optional extra context to include in the batch prompt "
            "(e.g. style notes, brand anchors, reference links)."
        ),
    )
    max_retries_per_scene: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Max retry attempts for an individual failed scene before giving up.",
    )


# ---------------------------------------------------------------------------
# Parsed scene model
# ---------------------------------------------------------------------------

@dataclass
class ParsedScene:
    """One scene parsed from the LLM's structured response."""

    index: int  # 1-based
    file: str
    engine: str
    duration: str
    reads: list[str]
    code: str
    raw_block: str = ""  # original text block for debugging


@dataclass
class BatchParseResult:
    """Result of parsing the LLM's batch response."""

    batch_id: str
    scenes: list[ParsedScene]
    parse_errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.scenes) and not self.parse_errors


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class BatchSceneTool(BaseTool):
    """Generate code for multiple scenes in a single LLM call."""

    name = "batch_scene"
    description = (
        "Generate code for 5-10 scenes in a single LLM call instead of one "
        "call per scene. Parses structured output, validates each scene, and "
        "writes them to individual files. Retries individual failed scenes "
        "rather than the whole batch."
    )
    input_model = BatchSceneToolInput

    async def execute(
        self,
        arguments: BatchSceneToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        config = context.metadata.get("scene_gen_config", {}) if context.metadata else {}
        if not isinstance(config, dict):
            config = {}

        if not arguments.scenes:
            return ToolResult(output="No scenes provided. Pass a list of scene descriptions.", is_error=True)

        # Resolve output directory
        output_dir = Path(arguments.output_dir)
        if not output_dir.is_absolute():
            output_dir = context.cwd / output_dir
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_id = arguments.batch_id or f"batch_{int(time.time())}"

        # Step 1: Call the LLM with the batch prompt
        try:
            raw_response = await self._call_llm(arguments, config, context)
        except Exception as exc:
            log.exception("batch_scene LLM call failed")
            return ToolResult(
                output=f"LLM call failed: {exc}\nBatch ID: {batch_id}",
                is_error=True,
                metadata={"batch_id": batch_id, "error": str(exc)},
            )

        # Step 2: Parse the structured response
        parse_result = parse_batch_response(raw_response, batch_id)
        if not parse_result.success:
            return ToolResult(
                output=(
                    f"Failed to parse batch response.\n"
                    f"Batch ID: {batch_id}\n"
                    f"Parse errors: {'; '.join(parse_result.parse_errors)}\n"
                    f"Raw response (first 500 chars):\n{raw_response[:500]}"
                ),
                is_error=True,
                metadata={
                    "batch_id": batch_id,
                    "parse_errors": parse_result.parse_errors,
                },
            )

        # Step 3: Validate + write each scene
        written: list[str] = []
        failed: list[dict[str, Any]] = []
        for scene in parse_result.scenes:
            # Retry loop for individual scenes
            success = False
            last_error = ""
            for attempt in range(arguments.max_retries_per_scene + 1):
                # Determine the code to write (retry on lint failure)
                if attempt == 0:
                    code_to_write = scene.code
                else:
                    # Retry: call LLM for just this scene
                    try:
                        code_to_write = await self._retry_single_scene(
                            arguments, config, context, scene, last_error
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        log.warning("scene %s retry %d failed: %s", scene.index, attempt, exc)
                        continue

                # Lint check
                lint_ok, lint_error = lint_scene_code(code_to_write, arguments.file_extension)
                if lint_ok:
                    success = True
                    break
                last_error = lint_error
                log.warning(
                    "scene %s lint failed (attempt %d/%d): %s",
                    scene.index, attempt + 1, arguments.max_retries_per_scene + 1, lint_error,
                )

            if not success:
                failed.append({
                    "scene_index": scene.index,
                    "file": scene.file,
                    "error": last_error,
                })
                continue

            # Write the file. The filename comes from the LLM response, so
            # it is untrusted: reject anything that tries to escape
            # output_dir (path separators, "..", absolute paths, null
            # bytes) instead of relying on the model to behave.
            filename = scene.file.strip()
            if (
                not filename
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
                or "\x00" in filename
            ):
                failed.append({
                    "scene_index": scene.index,
                    "file": scene.file,
                    "error": "invalid filename (path separators / traversal are not allowed)",
                })
                continue
            file_path = output_dir / filename
            try:
                file_path.write_text(code_to_write, encoding="utf-8")
                try:
                    written.append(str(file_path.relative_to(context.cwd)))
                except ValueError:
                    # Output dir is outside the project cwd — report the
                    # absolute path instead of crashing.
                    written.append(str(file_path))
            except OSError as exc:
                failed.append({
                    "scene_index": scene.index,
                    "file": scene.file,
                    "error": f"write failed: {exc}",
                })

        # Step 4: Build summary
        summary_lines = [
            "[Batch scene generation complete]",
            f"Batch ID: {batch_id}",
            f"Scenes requested: {len(arguments.scenes)}",
            f"Scenes parsed: {len(parse_result.scenes)}",
            f"Scenes written: {len(written)}",
            f"Scenes failed: {len(failed)}",
            "",
        ]
        if written:
            summary_lines.append("Written files:")
            for path in written:
                summary_lines.append(f"  - {path}")
        if failed:
            summary_lines.append("")
            summary_lines.append("Failed scenes (escalate to Director):")
            for f in failed:
                summary_lines.append(f"  - scene {f['scene_index']} ({f['file']}): {f['error']}")

        is_error = bool(failed) and not written
        return ToolResult(
            output="\n".join(summary_lines),
            is_error=is_error,
            metadata={
                "batch_id": batch_id,
                "scenes_written": written,
                "scenes_failed": failed,
                "total_scenes": len(arguments.scenes),
            },
        )

    async def _call_llm(
        self,
        arguments: BatchSceneToolInput,
        config: dict[str, Any],
        context: ToolExecutionContext,
    ) -> str:
        """Call the LLM with the batch prompt and return the raw text response."""
        api_key = str(config.get("api_key") or "").strip()
        base_url = str(config.get("base_url") or "").strip()
        model = str(config.get("model") or "gpt-4o").strip()

        if not api_key:
            # Try to fall back to the main API key from env / settings
            api_key = (
                context.metadata.get("api_key", "") if context.metadata else ""
            ) or ""

        if not api_key:
            raise RuntimeError(
                "No API key configured for scene generation. Set scene_gen_config.api_key "
                "or OPENAI_API_KEY, or pass the scene code directly via parse_batch_response()."
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=_normalize_openai_base_url(base_url) or None,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

        prompt = build_batch_prompt(arguments, context)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=int(config.get("max_tokens", 16384) or 16384),
                temperature=float(config.get("temperature", 0.7) or 0.7),
            )
            return response.choices[0].message.content or ""
        finally:
            await client.close()

    async def _retry_single_scene(
        self,
        arguments: BatchSceneToolInput,
        config: dict[str, Any],
        context: ToolExecutionContext,
        scene: ParsedScene,
        error: str,
    ) -> str:
        """Retry generating code for a single failed scene."""
        api_key = str(config.get("api_key") or "").strip()
        base_url = str(config.get("base_url") or "").strip()
        model = str(config.get("model") or "gpt-4o").strip()
        if not api_key:
            api_key = (context.metadata.get("api_key", "") if context.metadata else "") or ""
        if not api_key:
            raise RuntimeError("No API key for retry")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=_normalize_openai_base_url(base_url) or None,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            retry_prompt = (
                f"The following scene code failed validation:\n\n"
                f"FILE: {scene.file}\n"
                f"ENGINE: {scene.engine}\n"
                f"DURATION: {scene.duration}\n"
                f"READS: {json.dumps(scene.reads)}\n\n"
                f"ERROR: {error}\n\n"
                f"Regenerate ONLY the CODE for this scene. Fix the error. "
                f"Return just the code, no markdown fences."
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a scene code generator. Return only valid code."},
                    {"role": "user", "content": retry_prompt},
                ],
                max_tokens=8192,
                temperature=0.3,  # lower temp for retries
            )
            code = response.choices[0].message.content or ""
            # Strip markdown code fences if present
            code = re.sub(r"^```[a-zA-Z]*\n", "", code)
            code = re.sub(r"\n```$", "", code)
            return code.strip()
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_BATCH_SYSTEM_PROMPT = """\
You are a batch scene code generator. You generate code for multiple scenes
in a single response. Your output MUST follow this exact format:

BATCH_ID: <batch_id>
SCENES: <count>

SCENE_001:
  FILE: <filename>
  ENGINE: <engine_name>
  DURATION: <duration>s
  READS: ["<narration line 1>", "<narration line 2>"]
  CODE:
<indented code here, each line indented by 4 spaces>

SCENE_002:
  FILE: <filename>
  ENGINE: <engine_name>
  DURATION: <duration>s
  READS: ["..."]
  CODE:
<indented code>

Rules:
- Generate exactly the number of scenes requested.
- Each scene MUST have FILE, ENGINE, DURATION, READS, and CODE fields.
- The CODE must be indented by exactly 4 spaces (so it's a child of SCENE_NNN).
- Do NOT wrap code in markdown fences.
- Do NOT add any text before BATCH_ID or after the last scene.
"""


def build_batch_prompt(arguments: BatchSceneToolInput, context: ToolExecutionContext) -> str:
    """Build the user prompt for the batch LLM call."""
    lines = [
        f"Generate code for {len(arguments.scenes)} scenes.",
        f"Batch ID: {arguments.batch_id or 'auto'}",
        f"Engine: {arguments.engine}",
        f"File prefix: {arguments.file_prefix}",
        f"File extension: {arguments.file_extension}",
        "",
        "Scene descriptions:",
        "",
    ]
    for i, desc in enumerate(arguments.scenes, start=1):
        lines.append(f"SCENE_{i:03d}_DESCRIPTION:")
        lines.append(f"  {desc}")
        lines.append("")

    if arguments.extra_context:
        lines.append("Extra context:")
        lines.append(arguments.extra_context)
        lines.append("")

    # Auto-inject brand.yaml if present (Brand Kit integration)
    brand_path = context.cwd / "brand.yaml"
    if brand_path.is_file():
        try:
            brand_content = brand_path.read_text(encoding="utf-8")
            lines.append("Brand anchors (from brand.yaml — all scenes MUST comply):")
            lines.append("```yaml")
            lines.append(brand_content)
            lines.append("```")
            lines.append("")
        except OSError:
            pass

    # Auto-inject STYLE.md if present (auto-context integration)
    style_path = context.cwd / "STYLE.md"
    if style_path.is_file():
        try:
            style_content = style_path.read_text(encoding="utf-8")[:4000]
            lines.append("Style guide (from STYLE.md):")
            lines.append("```md")
            lines.append(style_content)
            lines.append("```")
            lines.append("")
        except OSError:
            pass

    lines.append("Return the batch in the exact structured format specified.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured response parser
# ---------------------------------------------------------------------------

# Regex to find each SCENE_NNN block. Non-greedy match up to the next
# SCENE_ block or end of text.
_SCENE_BLOCK_RE = re.compile(
    r"^SCENE_(\d+):\s*\n(.*?)(?=^SCENE_\d+:\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Field extractors within a scene block.
_FILE_RE = re.compile(r"^\s*FILE:\s*(.+?)\s*$", re.MULTILINE)
_ENGINE_RE = re.compile(r"^\s*ENGINE:\s*(.+?)\s*$", re.MULTILINE)
_DURATION_RE = re.compile(r"^\s*DURATION:\s*(.+?)\s*$", re.MULTILINE)
_READS_RE = re.compile(r"^\s*READS:\s*(\[.*?\])\s*$", re.MULTILINE | re.DOTALL)
_CODE_RE = re.compile(r"^\s*CODE:\s*\n((?:    .*\n?|^\s*\n?)*)", re.MULTILINE)

_BATCH_ID_RE = re.compile(r"^BATCH_ID:\s*(.+?)\s*$", re.MULTILINE)
_SCENES_COUNT_RE = re.compile(r"^SCENES:\s*(\d+)\s*$", re.MULTILINE)


def parse_batch_response(raw: str, expected_batch_id: str | None = None) -> BatchParseResult:
    """Parse the LLM's structured batch response into a BatchParseResult.

    Tolerant of minor formatting issues (extra whitespace, missing indentation)
    but strict about required fields: each scene MUST have FILE, ENGINE,
    DURATION, READS, and CODE.
    """
    errors: list[str] = []

    # Extract batch ID
    batch_id_match = _BATCH_ID_RE.search(raw)
    batch_id = batch_id_match.group(1).strip() if batch_id_match else (expected_batch_id or "unknown")

    # Extract declared scene count (informational — we parse actual blocks)
    count_match = _SCENES_COUNT_RE.search(raw)
    declared_count = int(count_match.group(1)) if count_match else 0

    # Parse each SCENE_NNN block
    scenes: list[ParsedScene] = []
    for match in _SCENE_BLOCK_RE.finditer(raw):
        index_str = match.group(1)
        block = match.group(2)
        try:
            index = int(index_str)
        except ValueError:
            errors.append(f"Invalid scene index: {index_str}")
            continue

        scene, scene_errors = _parse_single_scene(index, block)
        if scene is not None:
            scenes.append(scene)
        else:
            errors.extend(scene_errors)

    if not scenes:
        errors.append("No valid scenes found in response")

    if declared_count and declared_count != len(scenes):
        errors.append(
            f"Declared {declared_count} scenes but parsed {len(scenes)}"
        )

    return BatchParseResult(batch_id=batch_id, scenes=scenes, parse_errors=errors)


def _parse_single_scene(index: int, block: str) -> tuple[ParsedScene | None, list[str]]:
    """Parse a single SCENE_NNN block. Returns (scene, errors)."""
    errors: list[str] = []

    file_match = _FILE_RE.search(block)
    engine_match = _ENGINE_RE.search(block)
    duration_match = _DURATION_RE.search(block)
    reads_match = _READS_RE.search(block)
    code_match = _CODE_RE.search(block)

    if not file_match:
        errors.append(f"Scene {index}: missing FILE field")
    if not engine_match:
        errors.append(f"Scene {index}: missing ENGINE field")
    if not duration_match:
        errors.append(f"Scene {index}: missing DURATION field")
    if not reads_match:
        errors.append(f"Scene {index}: missing READS field")
    if not code_match:
        errors.append(f"Scene {index}: missing CODE field")

    if errors:
        return None, errors

    # Parse READS as JSON array
    reads_str = reads_match.group(1)
    try:
        reads = json.loads(reads_str)
        if not isinstance(reads, list):
            errors.append(f"Scene {index}: READS is not a JSON array")
            reads = []
        reads = [str(r) for r in reads]
    except json.JSONDecodeError as exc:
        errors.append(f"Scene {index}: READS is not valid JSON: {exc}")
        reads = []

    # Extract code — dedent the 4-space indentation
    raw_code = code_match.group(1)
    code_lines = []
    for line in raw_code.splitlines():
        if line.startswith("    "):
            code_lines.append(line[4:])
        elif line.strip() == "":
            code_lines.append("")
        else:
            # Line doesn't have the expected 4-space indent — keep as-is
            code_lines.append(line)
    code = "\n".join(code_lines).rstrip()

    if not code.strip():
        errors.append(f"Scene {index}: CODE is empty")

    if errors:
        return None, errors

    scene = ParsedScene(
        index=index,
        file=file_match.group(1).strip(),
        engine=engine_match.group(1).strip(),
        duration=duration_match.group(1).strip(),
        reads=reads,
        code=code,
        raw_block=block,
    )
    return scene, []


# ---------------------------------------------------------------------------
# Lint / validation
# ---------------------------------------------------------------------------

def lint_scene_code(code: str, file_extension: str) -> tuple[bool, str]:
    """Run a syntax check on the scene code.

    For .js files, uses ``node --check`` if Node.js is available.
    For .py files, uses ``python -m py_compile``.
    For other extensions, does a basic brace-balance check.

    Returns ``(ok, error_message)``. ``error_message`` is empty when ok=True.
    """
    if not code.strip():
        return False, "Code is empty"

    ext = file_extension.lower()
    if ext in (".js", ".mjs", ".cjs", ".jsx"):
        return _lint_js(code)
    if ext == ".py":
        return _lint_python(code)
    # Fallback: basic brace/paren balance
    return _lint_brace_balance(code)


def _lint_js(code: str) -> tuple[bool, str]:
    """Lint JavaScript using ``node --check`` if available."""
    node = shutil.which("node")
    if not node:
        # No node — do a brace-balance check as a fallback.
        return _lint_brace_balance(code)
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name
        try:
            result = subprocess.run(
                [node, "--check", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        if result.returncode == 0:
            return True, ""
        # Extract the error line
        error_line = result.stderr.strip().splitlines()[0] if result.stderr else "unknown error"
        return False, f"node --check: {error_line}"
    except Exception:
        # Fallback to brace check if node invocation fails
        return _lint_brace_balance(code)


def _lint_python(code: str) -> tuple[bool, str]:
    """Lint Python using ``py_compile``."""
    try:
        import py_compile
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return True, ""
    except py_compile.PyCompileError as exc:
        return False, str(exc)
    except Exception:
        return _lint_brace_balance(code)


def _lint_brace_balance(code: str) -> tuple[bool, str]:
    """Basic brace/paren balance check — fallback when no linter is available."""
    stack: list[str] = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    closers = set(pairs.values())
    in_string: str | None = None
    escape = False
    for i, char in enumerate(code):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if in_string:
            if char == in_string:
                in_string = None
            continue
        if char in ('"', "'", "`"):
            in_string = char
            continue
        if char in pairs:
            stack.append(char)
        elif char in closers:
            if not stack:
                return False, f"Unmatched '{char}' at position {i}"
            opener = stack.pop()
            if pairs[opener] != char:
                return False, f"Mismatched '{opener}' and '{char}' at position {i}"
    if stack:
        return False, f"Unclosed: {''.join(stack)}"
    return True, ""
