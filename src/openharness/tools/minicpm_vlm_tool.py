"""MiniCPM-V 4.5 vision-language tool — local frame & image understanding.

MiniCPM-V 4.5 is an 8B-parameter multimodal model; its 4-bit
quantization fits in 6-8 GB of VRAM, so a single consumer GPU (or a
Kaggle T4) can serve it locally through any OpenAI-compatible server:

- vLLM        (``vllm serve openbmb/MiniCPM-V-4_5 --quantization awq``)
- llama.cpp   (``llama-server -m MiniCPM-V-4_5-Q4_K_M.gguf --port 8001``)
- Ollama      (``ollama serve`` — any OpenAI-compatible gateway)

The tool sends one ``/chat/completions`` request with inline base64
images (the standard OpenAI vision wire format every one of those
servers speaks), so the agent gets real frame understanding —
"describe this rendered frame", "read the text on the title card",
"compare these two takes" — without spending primary-model quota.

Video input is supported: the tool extracts N evenly-spaced frames with
OpenCV (when available) and sends them as a multi-image message, which
MiniCPM-V 4.5 handles natively.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

#: How to launch a local MiniCPM-V 4.5 4-bit server (6-8 GB VRAM).
_LAUNCH_HINTS = (
    (
        "vLLM:      vllm serve openbmb/MiniCPM-V-4_5 --quantization awq "
        "--max-model-len 8192 --port 8001"
    ),
    (
        "llama.cpp: llama-server -m MiniCPM-V-4_5-Q4_K_M.gguf --host 127.0.0.1 "
        "--port 8001 -c 8192"
    ),
    (
        "Ollama:    ollama pull minicpm-v && ollama serve   (OpenAI-compatible "
        "endpoint on port 11434)"
    ),
)


def _resolve_config(context: ToolExecutionContext) -> Any:
    """Resolve MiniCPM-V endpoint config: settings, then env, then defaults."""
    from openharness.config.settings import MiniCPMVLMConfig

    settings = (context.metadata or {}).get("settings")
    config = getattr(settings, "minicpm_v", None) if settings is not None else None
    if config is not None and str(getattr(config, "base_url", "") or "").strip():
        return config
    return MiniCPMVLMConfig.from_env()


def _encode_image(path: Path) -> tuple[str, str] | None:
    """Return ``(data_uri, media_type)`` for an image file, or None."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{encoded}", media_type


def _extract_video_frames(video_path: Path, frame_count: int) -> list[Path]:
    """Extract up to ``frame_count`` evenly-spaced frames to a temp dir.

    Returns an empty list when OpenCV is unavailable or the video cannot
    be opened — the caller surfaces that honestly.
    """
    try:
        import tempfile

        import cv2
    except ImportError:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        count = max(1, min(frame_count, total))
        positions = [int(i * (total - 1) / max(1, count - 1)) for i in range(count)]
        out_dir = Path(tempfile.mkdtemp(prefix="minicpm-frames-"))
        written: list[Path] = []
        for index, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            out_path = out_dir / f"frame_{index:03d}.jpg"
            if cv2.imwrite(str(out_path), frame):
                written.append(out_path)
        return written
    finally:
        capture.release()


async def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    transport: Any = None,
) -> dict[str, Any]:
    """POST /chat/completions and return the parsed JSON body.

    ``transport`` is a test seam (httpx.MockTransport); production calls
    build a normal client.
    """
    import httpx

    client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
    if transport is not None:
        client_kwargs["transport"] = transport
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json()


class MiniCPMVLMInput(BaseModel):
    """Arguments for the MiniCPM-V 4.5 vision tool."""

    prompt: str = Field(
        default="Describe this image in detail: text content, layout, "
        "colors, and any visual defects.",
        description="What to ask the vision model about the image(s)/frames.",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Local image file paths to analyze (1 or more).",
    )
    video: str | None = Field(
        default=None,
        description="Optional video file path; evenly-spaced frames are "
        "extracted and analyzed together.",
    )
    frames: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of frames to extract from a video.",
    )
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    check_server: bool = Field(
        default=False,
        description="Only probe the endpoint (GET /models) and report "
        "status + launch instructions; no analysis.",
    )


class MiniCPMVLMTool(BaseTool):
    """Analyze images/video frames with a local MiniCPM-V 4.5 endpoint."""

    name = "minicpm_vlm"
    description = (
        "Analyze images or video frames with a LOCAL MiniCPM-V 4.5 "
        "vision-language model (8B, 4-bit, 6-8 GB VRAM) served through any "
        "OpenAI-compatible endpoint (vLLM / llama.cpp / Ollama). Use it to "
        "read text on rendered title cards, describe frames, spot visual "
        "defects, or compare takes — without spending primary-model quota. "
        "Video input extracts evenly-spaced frames automatically. Set "
        "check_server=true to verify the endpoint is up and get launch "
        "instructions."
    )
    input_model = MiniCPMVLMInput

    def is_read_only(self, arguments: MiniCPMVLMInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: MiniCPMVLMInput, context: ToolExecutionContext
    ) -> ToolResult:
        import httpx

        config = _resolve_config(context)
        base_url = str(getattr(config, "base_url", "") or "").strip()
        model = str(getattr(config, "model", "") or "").strip()
        api_key = str(getattr(config, "api_key", "") or "EMPTY")
        timeout = float(getattr(config, "timeout_seconds", 180.0) or 180.0)

        if not (base_url and model):
            return ToolResult(
                output=(
                    "minicpm_vlm: no endpoint configured. Set settings.minicpm_v "
                    "(base_url + model) or OPENHARNESS_MINICPM_BASE_URL / "
                    "OPENHARNESS_MINICPM_MODEL."
                ),
                is_error=True,
            )

        if arguments.check_server:
            return await self._check_server(base_url, api_key, model, timeout, config)

        # ---- Collect image sources -------------------------------------
        data_uris: list[str] = []
        sources: list[str] = []
        for raw in arguments.images:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(context.cwd) / path
            encoded = _encode_image(path)
            if encoded is None:
                return ToolResult(
                    output=f"minicpm_vlm: cannot read image: {raw}", is_error=True
                )
            data_uris.append(encoded[0])
            sources.append(str(path))

        if arguments.video:
            video_path = Path(arguments.video)
            if not video_path.is_absolute():
                video_path = Path(context.cwd) / video_path
            if not video_path.is_file():
                return ToolResult(
                    output=f"minicpm_vlm: video not found: {arguments.video}",
                    is_error=True,
                )
            frame_paths = _extract_video_frames(video_path, arguments.frames)
            if not frame_paths:
                return ToolResult(
                    output=(
                        "minicpm_vlm: could not extract frames (OpenCV missing or "
                        "video unreadable). Install opencv-python-headless or pass "
                        "individual images instead."
                    ),
                    is_error=True,
                )
            for frame_path in frame_paths:
                encoded = _encode_image(frame_path)
                if encoded is not None:
                    data_uris.append(encoded[0])
                    sources.append(str(frame_path))

        if not data_uris:
            return ToolResult(
                output="minicpm_vlm: provide `images` (paths) and/or `video`.",
                is_error=True,
            )

        # ---- One multi-image chat completion ---------------------------
        content: list[dict[str, Any]] = [{"type": "text", "text": arguments.prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": uri}} for uri in data_uris
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": arguments.max_tokens,
            "temperature": arguments.temperature,
        }
        try:
            data = await _post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout,
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=(
                    f"minicpm_vlm: server rejected the request "
                    f"({exc.response.status_code}): {exc.response.text[:300]}"
                ),
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 - report, never crash the loop
            return ToolResult(
                output=(
                    f"minicpm_vlm: endpoint unreachable at {base_url} ({exc}). "
                    "Start the server (see check_server=true for launch "
                    "commands) or adjust settings.minicpm_v."
                ),
                is_error=True,
            )

        choices = data.get("choices") or []
        if not choices:
            return ToolResult(
                output=f"minicpm_vlm: no choices in response: {str(data)[:300]}",
                is_error=True,
            )
        answer = str(((choices[0].get("message") or {}).get("content")) or "").strip()

        # Record usage in the cost dashboard when the server reports it.
        usage = data.get("usage") or {}
        if usage:
            try:
                from openharness.services.cost_dashboard import get_cost_dashboard

                get_cost_dashboard().record(
                    model=model,
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    source="minicpm_vlm",
                )
            except Exception:
                log.debug("minicpm_vlm: cost dashboard recording failed", exc_info=True)

        header = f"[minicpm_vlm | {model} | {len(data_uris)} image(s)]"
        return ToolResult(
            output=f"{header}\n\n{answer or '(empty response)'}",
            metadata={
                "model": model,
                "base_url": base_url,
                "images": sources,
                "usage": usage,
            },
        )

    async def _check_server(
        self, base_url: str, api_key: str, model: str, timeout: float, config: Any
    ) -> ToolResult:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=min(timeout, 15.0)) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                listed = [
                    str(item.get("id"))
                    for item in (response.json().get("data") or [])
                    if isinstance(item, dict)
                ]
        except Exception as exc:  # noqa: BLE001 - status tool reports, not raises
            return ToolResult(
                output=(
                    f"[minicpm_vlm] endpoint DOWN: {base_url} ({exc})\n\n"
                    "Launch a local MiniCPM-V 4.5 4-bit server (6-8 GB VRAM):\n  "
                    + "\n  ".join(_LAUNCH_HINTS)
                    + f"\n\nThen point settings.minicpm_v at it (model={model!r})."
                ),
                is_error=True,
            )
        serving = model in listed
        model_state = "served" if serving else "NOT in list — check settings.minicpm_v.model"
        quant = getattr(config, "quantization", "4-bit")
        vram = getattr(config, "vram_gb", "6-8")
        lines = [
            f"[minicpm_vlm] endpoint UP: {base_url}",
            f"  models served: {', '.join(listed[:8]) or '(none listed)'}",
            f"  configured model: {model} ({model_state})",
            f"  quantization: {quant} (~{vram} GB VRAM)",
        ]
        return ToolResult(output="\n".join(lines), metadata={"models": listed})
