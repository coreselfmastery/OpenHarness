"""Speech-to-text via Groq's OpenAI-compatible audio endpoint.

Telegram voice notes (and any other channel media) are transcribed with
``whisper-large-v3`` (configurable through ``GROQ_TRANSCRIPTION_MODEL``) by
posting the file to ``POST {base_url}/audio/transcriptions`` — the same
OpenAI-compatible shape the ``groq`` provider registry entry uses, so one
``GROQ_API_KEY`` serves both the chat model and transcription.

The client is deliberately tiny: no SDK dependency, only ``httpx`` (already a
core dependency). Every failure raises with a message that names the fix; the
caller decides whether that is fatal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

# Groq accepts these containers; Telegram voice is .ogg (Opus), audio
# attachments are commonly .mp3 / .m4a / .oga.
_MIME_BY_SUFFIX = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


class GroqTranscriptionProvider:
    """Transcribe an audio file with Groq Whisper.

    Args:
        api_key: Groq API key. Falls back to ``GROQ_API_KEY``.
        model: transcription model id. Falls back to ``GROQ_TRANSCRIPTION_MODEL``,
            else ``whisper-large-v3``.
        base_url: OpenAI-compatible base URL. Falls back to ``GROQ_BASE_URL``,
            else ``https://api.groq.com/openai/v1``.
        timeout: request timeout in seconds (uploads of long notes can be slow
            on weak uplinks, so the default is generous).
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "whisper-large-v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self.model = (model or os.environ.get("GROQ_TRANSCRIPTION_MODEL", "") or self.DEFAULT_MODEL).strip()
        self.base_url = (
            base_url or os.environ.get("GROQ_BASE_URL", "") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout

    async def transcribe(self, path: PathLike) -> str:
        """Return the transcript of ``path``, or ``""`` when it is silent.

        Raises:
            RuntimeError: no API key configured.
            FileNotFoundError: ``path`` does not exist.
            httpx.HTTPStatusError: the endpoint rejected the request.
        """
        if not self.api_key:
            raise RuntimeError(
                "Groq transcription needs an API key: pass api_key= or set GROQ_API_KEY"
            )
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"transcription source not found: {file_path}")

        import httpx  # core dependency; imported here so the module is cheap to import

        url = f"{self.base_url}/audio/transcriptions"
        mime = _MIME_BY_SUFFIX.get(file_path.suffix.lower(), "application/octet-stream")
        logger.debug("Transcribing %s (%s) with %s", file_path.name, mime, self.model)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(file_path, "rb") as fh:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (file_path.name, fh, mime)},
                    data={"model": self.model, "response_format": "json"},
                )
        response.raise_for_status()
        text = (response.json().get("text") or "").strip()
        logger.debug("Transcribed %s: %s chars", file_path.name, len(text))
        return text
