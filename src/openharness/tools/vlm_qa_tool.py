"""vlm_qa tool — selective vision-language QA on rendered scenes.

The expensive "human-like eye": runs **only** on scenes that already
passed the cheap gates (``scene_lint`` code analysis, ``pixel_qa``
deterministic frames, ``frame_diff`` motion) but deserve a deeper
aesthetic/semantic judgment — typically the 5-10% of scenes with
subtle composition, readability, or brand-match issues that pixel
checks cannot see.

Workflow per scene:

1. Sample ``frames_per_scene`` (5-10) evenly-spaced frames (OpenCV).
2. Send them in ONE multi-image request to the local MiniCPM-V 4.5
   endpoint (8B, 4-bit, 6-8 GB VRAM — see ``minicpm_vlm``) together
   with a structured rubric: composition, readability, brand match,
   anomaly presence — each scored 0-10 with a one-line justification.
3. Parse the JSON scores (robust against code fences and prose),
   aggregate per scene, and emit a pass/fail verdict against
   ``pass_threshold``.

MiniCPM-V 4.5's 96x token compression keeps a multi-frame rubric
request cheap, so the whole video project stays reviewable without
spending primary-model quota.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.tools.minicpm_vlm_tool import (
    _encode_image,
    _extract_video_frames,
    _post_chat_completion,
    _resolve_config,
)

log = logging.getLogger(__name__)

_DEFAULT_RUBRIC = (
    "composition: is the visual hierarchy clear, subject well placed, "
    "no awkward cropping?; "
    "readability: is all text legible at full size, high contrast, "
    "not clipped?; "
    "brand_match: do colors, typography and mood match a clean, "
    "consistent brand look?; "
    "anomaly: any artifacts, ghosting, mismatched elements, or "
    "visual defects?"
)

_RUBRIC_PROMPT = """You are a strict video QA reviewer. Inspect the {n} sampled \
frames of this scene in order (frame 1 .. frame {n}).

Score each rubric criterion from 0 (broken) to 10 (perfect):
{rubric}

Respond with ONLY a JSON object, no prose, no code fences:
{{
  "scores": {{
    "composition": <0-10>,
    "readability": <0-10>,
    "brand_match": <0-10>,
    "anomaly": <0-10>
  }},
  "justification": "<one line per criterion>",
  "verdict": "pass" | "fail"
}}"""


def parse_rubric_json(raw: str) -> dict[str, Any] | None:
    """Extract the rubric JSON object from a model response.

    Tolerates code fences, leading prose, and trailing commentary; the
    scores live in the first balanced ``{...}`` block containing a
    ``scores`` key.
    """
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except ValueError:
                        break
                    if isinstance(payload, dict) and "scores" in payload:
                        return payload
                    break
        start = text.find("{", start + 1)
    return None


class VLMQAInput(BaseModel):
    """Arguments for the vlm_qa tool."""

    videos: list[str] = Field(
        description="Rendered scene video file(s) (.mp4/.mov/.mkv/.webm/.avi)."
    )
    frames_per_scene: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Frames sampled per scene for the rubric review (5-10 recommended).",
    )
    rubric: str | None = Field(
        default=None,
        description=(
            "Optional rubric override as 'criterion: question; criterion: "
            "question; ...' — criteria names become the score keys. Default: "
            "composition / readability / brand_match / anomaly."
        ),
    )
    pass_threshold: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Mean score at/above which a scene passes.",
    )
    max_tokens: int = Field(default=1024, ge=128, le=8192)
    report_path: str | None = Field(
        default=None,
        description="Optional path to write the full JSON report.",
    )


class VLMQATool(BaseTool):
    """Selective deep QA: MiniCPM-V 4.5 rubric review of sampled frames."""

    name = "vlm_qa"
    description = (
        "Selective vision-language QA for scenes that passed the cheap gates "
        "(scene_lint + pixel_qa + frame_diff) but need human-like judgment: "
        "samples 5-10 frames per scene, sends them to the LOCAL MiniCPM-V 4.5 "
        "endpoint (8B, 4-bit, 6-8 GB VRAM) with a structured rubric "
        "(composition / readability / brand match / anomaly, 0-10 each), and "
        "returns parsed JSON scores with a per-scene pass/fail verdict. "
        "Configure the endpoint via settings.minicpm_v or "
        "OPENHARNESS_MINICPM_* (same server as the minicpm_vlm tool)."
    )
    input_model = VLMQAInput

    def is_read_only(self, arguments: VLMQAInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: VLMQAInput, context: ToolExecutionContext
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
                    "vlm_qa: no MiniCPM-V endpoint configured (shares config "
                    "with minicpm_vlm). Set settings.minicpm_v or "
                    "OPENHARNESS_MINICPM_BASE_URL / OPENHARNESS_MINICPM_MODEL."
                ),
                is_error=True,
            )

        # ---- Resolve videos ---------------------------------------------
        from openharness.services.frame_diff import find_video_files

        videos: list[Path] = []
        for raw in arguments.videos:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(context.cwd) / path
            found = find_video_files(path)
            if found:
                videos.extend(found)
            else:
                return ToolResult(
                    output=f"vlm_qa: no video found at {raw}", is_error=True
                )

        rubric = (arguments.rubric or _DEFAULT_RUBRIC).strip()
        # Criterion names = leading word (letters/underscore) before ':'.
        criteria = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", rubric) or [
            "composition",
            "readability",
            "brand_match",
            "anomaly",
        ]

        results: list[dict[str, Any]] = []
        for video in videos:
            entry: dict[str, Any] = {"scene": video.stem, "video": str(video)}
            frame_paths = _extract_video_frames(video, arguments.frames_per_scene)
            if not frame_paths:
                entry.update(
                    {
                        "error": "could not extract frames (OpenCV missing or video unreadable)",
                        "passed": False,
                    }
                )
                results.append(entry)
                continue

            data_uris: list[str] = []
            for frame_path in frame_paths:
                encoded = _encode_image(frame_path)
                if encoded is not None:
                    data_uris.append(encoded[0])
            if not data_uris:
                entry.update({"error": "frames unreadable", "passed": False})
                results.append(entry)
                continue

            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": _RUBRIC_PROMPT.format(
                        n=len(data_uris), rubric=rubric
                    ),
                }
            ]
            content.extend(
                {"type": "image_url", "image_url": {"url": uri}} for uri in data_uris
            )
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": arguments.max_tokens,
                "temperature": 0.1,
            }
            try:
                data = await _post_chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    payload=payload,
                    timeout_seconds=timeout,
                )
            except httpx.HTTPStatusError as exc:
                entry.update(
                    {
                        "error": (
                            f"server rejected request ({exc.response.status_code}): "
                            f"{exc.response.text[:200]}"
                        ),
                        "passed": False,
                    }
                )
                results.append(entry)
                continue
            except Exception as exc:  # noqa: BLE001 - per-scene isolation
                entry.update(
                    {
                        "error": f"endpoint unreachable: {exc}",
                        "passed": False,
                    }
                )
                results.append(entry)
                continue

            usage = data.get("usage") or {}
            choices = data.get("choices") or []
            answer = str(((choices[0].get("message") or {}).get("content")) or "")
            parsed = parse_rubric_json(answer)
            if parsed is None:
                entry.update(
                    {
                        "error": "model did not return parsable rubric JSON",
                        "raw_response": answer[:400],
                        "passed": False,
                        "usage": usage,
                    }
                )
                results.append(entry)
                continue

            scores_raw = parsed.get("scores") or {}
            scores = {
                str(key): float(value)
                for key, value in scores_raw.items()
                if isinstance(value, (int, float))
            }
            # Restrict aggregation to rubric criteria the prompt declared.
            scored = {
                key: value for key, value in scores.items() if key in criteria
            } or scores
            mean = sum(scored.values()) / len(scored) if scored else 0.0
            verdict = str(parsed.get("verdict") or "").lower()
            passed = (
                verdict == "pass" if verdict in {"pass", "fail"} else mean >= arguments.pass_threshold
            )
            entry.update(
                {
                    "frames_reviewed": len(data_uris),
                    "scores": {key: round(value, 2) for key, value in scored.items()},
                    "mean_score": round(mean, 2),
                    "justification": parsed.get("justification"),
                    "verdict": verdict or ("pass" if passed else "fail"),
                    "passed": passed,
                    "usage": usage,
                }
            )
            results.append(entry)

            if usage:
                try:
                    from openharness.services.cost_dashboard import get_cost_dashboard

                    get_cost_dashboard().record(
                        model=model,
                        input_tokens=int(usage.get("prompt_tokens") or 0),
                        output_tokens=int(usage.get("completion_tokens") or 0),
                        source="vlm_qa",
                    )
                except Exception:
                    log.debug("vlm_qa: cost dashboard recording failed", exc_info=True)

        # ---- Progress tracker -------------------------------------------
        try:
            from openharness.services.progress_tracker import get_progress_tracker

            tracker = get_progress_tracker()
            for entry in results:
                if "error" in entry and "frames_reviewed" not in entry:
                    continue
                tracker.mark(
                    entry["scene"], "qa", detail=f"vlm-qa mean {entry.get('mean_score', 0)}"
                )
        except Exception:
            log.debug("vlm_qa: progress tracker update failed", exc_info=True)

        # ---- Text report --------------------------------------------------
        lines = [f"[vlm_qa | {model} | {len(results)} scene(s)]"]
        for entry in results:
            if "error" in entry and "frames_reviewed" not in entry:
                lines.append(f"  {entry['scene']}: ERROR — {entry['error']}")
                continue
            status = "PASS" if entry["passed"] else "FAIL"
            scores_text = ", ".join(
                f"{key} {value:g}" for key, value in entry.get("scores", {}).items()
            )
            lines.append(
                f"  {entry['scene']}: {status} (mean {entry.get('mean_score', 0):.1f}, "
                f"{entry.get('frames_reviewed', 0)} frames — {scores_text})"
            )
            if entry.get("justification"):
                lines.append(f"    {entry['justification']}")
        failed = [entry["scene"] for entry in results if not entry["passed"]]
        if failed:
            lines.append(
                "[vlm_qa] scene(s) below threshold: " + ", ".join(failed) + "."
            )
        else:
            lines.append("[vlm_qa] all reviewed scenes passed the rubric.")

        metadata: dict[str, Any] = {
            "model": model,
            "pass_threshold": arguments.pass_threshold,
            "results": results,
        }
        if arguments.report_path:
            report_path = Path(arguments.report_path)
            if not report_path.is_absolute():
                report_path = Path(context.cwd) / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            lines.append(f"[vlm_qa] full report written to {report_path}")
            metadata["report_path"] = str(report_path)

        return ToolResult(
            output="\n".join(lines), is_error=bool(failed), metadata=metadata
        )
