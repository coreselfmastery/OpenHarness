"""Background asset prefetch and async audio generation — items #22, #23.

#22 — ``AsyncAudioGenerator``: real text-to-speech through any
OpenAI-compatible ``/audio/speech`` endpoint (the standard TTS API
shape used by OpenAI, Groq, and many self-hosted gateways). Generation
runs as a background task via :func:`run_in_background`, so scene code
and voiceover are produced concurrently.

#23 — ``AssetPrefetcher``: warms caches for the assets the next scenes
will reference — file reads go straight into the tool cache, and any
declared voiceover audio is generated in the background. Both effects
mean that when the agent finally asks for those assets, they are
already local.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


# ======================================================================
# #22 — Async audio generation (OpenAI-compatible /audio/speech)
# ======================================================================

@dataclass
class AudioTask:
    """One background TTS job."""

    task_id: str
    text: str
    voice: str
    output_path: Path
    status: str = "pending"  # pending|running|done|failed
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    http_status: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "voice": self.voice,
            "output_path": str(self.output_path),
            "error": self.error[:200],
            "http_status": self.http_status,
            "elapsed_seconds": round(
                (self.finished_at or time.monotonic()) - self.started_at, 2
            ),
        }


class AsyncAudioGenerator:
    """Real TTS against OpenAI-compatible endpoints, run in the background.

    Configuration (all required for real generation — without an
    endpoint configured the tool says so instead of pretending):

    - ``base_url``: e.g. ``https://api.openai.com/v1`` or a Groq/
      self-hosted gateway.
    - ``api_key``: bearer token.
    - ``model``: e.g. ``tts-1`` / ``tts-1-hd`` / a server-side alias.
    - ``voice``: default voice id (``alloy``, ...).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "tts-1",
        default_voice: str = "alloy",
        timeout_seconds: float = 120.0,
        audio_format: str = "mp3",
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("AsyncAudioGenerator needs a real base_url + api_key")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._default_voice = default_voice
        self._timeout = timeout_seconds
        self._format = audio_format
        self._tasks: dict[str, AudioTask] = {}
        self._seq = 0

    async def generate(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AudioTask:
        """Generate speech and wait for it (foreground variant)."""
        task = self._new_task(text, output_path, voice or self._default_voice)
        await self._run(task, text, voice or self._default_voice, speed)
        return task

    def run_in_background(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AudioTask:
        """Kick off generation without blocking (#22). Returns immediately."""
        task = self._new_task(text, output_path, voice or self._default_voice)
        asyncio.create_task(self._run(task, text, voice or self._default_voice, speed))
        return task

    def _new_task(self, text: str, output_path: Path, voice: str) -> AudioTask:
        self._seq += 1
        audio_task = AudioTask(
            task_id=f"audio{self._seq:04d}",
            text=text,
            voice=voice,
            output_path=Path(output_path),
        )
        self._tasks[audio_task.task_id] = audio_task
        return audio_task

    async def _run(self, task: AudioTask, text: str, voice: str, speed: float) -> None:
        task.status = "running"
        url = f"{self._base_url}/audio/speech"
        payload = {
            "model": self._model,
            "voice": voice,
            "input": text,
            "speed": max(0.25, min(4.0, float(speed))),
            "response_format": self._format,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
            task.http_status = response.status_code
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("empty audio response body")
            task.output_path.write_bytes(response.content)
            task.status = "done"
            log.info("audio %s written to %s (%d bytes)", task.task_id, task.output_path, len(response.content))
        except Exception as exc:  # noqa: BLE001 - surfaced in the task record
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            log.warning("audio task %s failed: %s", task.task_id, task.error)
        finally:
            task.finished_at = time.monotonic()

    def status(self, task_id: str | None = None) -> dict[str, Any]:
        if task_id is not None:
            task = self._tasks.get(task_id)
            return task.as_dict() if task else {"error": f"unknown task {task_id}"}
        tasks = [t.as_dict() for t in self._tasks.values()]
        return {
            "total": len(tasks),
            "running": sum(1 for t in tasks if t["status"] == "running"),
            "done": sum(1 for t in tasks if t["status"] == "done"),
            "failed": sum(1 for t in tasks if t["status"] == "failed"),
            "tasks": tasks[-20:],
        }


# ======================================================================
# #23 — Background asset prefetch
# ======================================================================

@dataclass
class PrefetchReport:
    """What the prefetcher actually did (honest accounting)."""

    files_read: list[str] = field(default_factory=list)
    audio_generated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_read": self.files_read,
            "audio_generated": self.audio_generated,
            "skipped": self.skipped,
        }


class AssetPrefetcher:
    """Warm the caches for what the next scenes will need (#23).

    Two real effects:

    1. Referenced asset files (found by the render-cache asset scanner)
       are read once now and pushed into the tool result cache, so the
       agent's later ``read_file`` calls on them are instant.
    2. Voiceover lines (a ``READS: [...]`` block in scene code) are
       queued to the AsyncAudioGenerator in the background, so audio
       and code production overlap.
    """

    def __init__(self, *, audio_generator: AsyncAudioGenerator | None = None) -> None:
        self._audio = audio_generator

    async def prefetch_scenes(
        self,
        scene_paths: list[Path],
        *,
        cwd: Path,
        engine: str = "hyperframes",
        audio_output_dir: Path | None = None,
    ) -> PrefetchReport:
        from openharness.services.render_cache import find_asset_paths
        from openharness.services.tool_cache import get_tool_cache

        report = PrefetchReport()
        tool_cache = get_tool_cache()
        for scene in scene_paths:
            try:
                code = Path(scene).read_text(encoding="utf-8", errors="replace")
            except OSError:
                report.skipped.append(f"{scene} (unreadable)")
                continue
            # 1. Warm the file-read cache for referenced assets.
            for asset in find_asset_paths(code, engine, cwd=cwd):
                try:
                    content = asset.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError):
                    report.skipped.append(f"{asset.name} (unreadable)")
                    continue
                if tool_cache is not None and tool_cache.is_cacheable("read_file"):
                    self._warm_read_file_cache(tool_cache, asset, content, cwd=cwd)
                    report.files_read.append(str(asset))
            # 2. Queue voiceover audio in the background.
            if self._audio is not None:
                for line in _extract_reads(code):
                    out = (audio_output_dir or cwd / "audio") / f"{Path(scene).stem}_{abs(hash(line)) % 10_000:04d}.mp3"
                    self._audio.run_in_background(line, out)
                    report.audio_generated.append(str(out))
        return report

    @staticmethod
    def _warm_read_file_cache(tool_cache: Any, asset: Path, content: str, *, cwd: Path) -> None:
        """Warm the tool cache so a later ``read_file`` hits.

        The cache key is built from the RAW tool-input dict the model
        sends, and ``FileReadToolInput`` spells the argument ``path``
        (not ``file_path``) — so the entry must be stored under
        ``{"path": ...}``. Both common call shapes are warmed (the
        minimal path-only call and the fully-defaulted one), and both
        path forms (absolute, and cwd-relative for assets under the
        project root).

        The cached value replicates the tool's ACTUAL output format
        (``{:>6}\t{line}`` numbered lines, first ``limit`` lines), so a
        cache hit is byte-identical to a real read.
        """
        lines = content.splitlines()
        selected = lines[0:200]
        output = "\n".join(f"{index + 1:>6}\t{line}" for index, line in enumerate(selected))
        if not output:
            return
        result = {
            "output": output,
            "is_error": False,
            "metadata": {"prefetched": True},
        }
        path_forms = [str(asset)]
        try:
            relative = asset.relative_to(cwd)
            if str(relative) != str(asset):
                path_forms.append(str(relative))
        except ValueError:
            pass  # asset outside cwd — only the absolute form applies
        for form in path_forms:
            tool_cache.put("read_file", {"path": form}, dict(result))
            tool_cache.put(
                "read_file",
                {"path": form, "offset": 0, "limit": 200},
                dict(result),
            )


_READS_RE = None


def _extract_reads(scene_code: str) -> list[str]:
    """Extract voiceover lines from a ``READS: [...]`` block (batch_scene format)."""
    import json
    import re

    global _READS_RE
    if _READS_RE is None:
        _READS_RE = re.compile(r"^\s*READS:\s*(\[.*?\])\s*$", re.MULTILINE)
    lines: list[str] = []
    for match in _READS_RE.finditer(scene_code):
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(parsed, list):
            lines.extend(str(item) for item in parsed if str(item).strip())
    return lines
