"""Parallel QA pipeline — items #7-#10.

#7 — Parallel QA waves: several scenes reviewed at once, each by its
     own LLM call, bounded by a wave-sized semaphore.
#8 — Review batching: simple scenes packed N-per-call with a
     structured per-scene verdict response.
#9 — Smart review depth: a real complexity classifier (code size,
     dependency count, animation/effect keywords) picks a light
     checklist for simple scenes and a deep rubric for complex ones.
#10 — Failure prediction: persisted per-category failure counters
     (SQLite, #19) rank the failure modes most likely for a scene like
     this one, and those categories are injected into the review prompt
     for pre-emptive checking.

The reviewer needs a model endpoint; it reuses the same
OpenAI-compatible config the batch_scene tool uses (``scene_gen``
settings or ``qa`` overrides in settings). Without an endpoint
configured, review calls fail loudly and honestly — no fake verdicts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ======================================================================
# #9 — Scene complexity classification (real heuristics)
# ======================================================================

_COMPLEXITY_KEYWORDS = {
    "particle": 3, "physics": 2, "shader": 3, "webgl": 3, "three.js": 3,
    "mask": 1, "blur": 1, "chroma": 2, "transition": 1, "keyframe": 1,
    "gradient": 1, "easing": 1, "parallax": 2, "composite": 2, "overlay": 1,
    "blend": 1, "glow": 1, "shadow": 1, "tween": 1, "timeline": 1,
}

@dataclass
class SceneComplexity:
    """Measured complexity of one scene."""

    level: str  # "simple" | "standard" | "complex"
    score: int
    code_lines: int
    dependencies: int
    animation_keywords: int

    @property
    def is_simple(self) -> bool:
        return self.level == "simple"


def classify_scene(scene_path: Path) -> SceneComplexity:
    """Classify a scene file by measurable signals.

    - code lines (>80 = heavy)
    - imported dependencies (>3 = heavy)
    - animation/effect keywords (weighted)
    """
    try:
        code = scene_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        code = ""
    lines = [line for line in code.splitlines() if line.strip()]
    code_lines = len(lines)
    imports = len(re.findall(r"^\s*(?:import|require|from)\b", code, re.MULTILINE))
    keyword_score = 0
    keyword_hits = 0
    lowered = code.lower()
    for keyword, weight in _COMPLEXITY_KEYWORDS.items():
        occurrences = lowered.count(keyword)
        if occurrences:
            keyword_hits += 1
            keyword_score += weight * min(occurrences, 3)

    score = 0
    if code_lines > 80:
        score += 2
    elif code_lines > 40:
        score += 1
    if imports > 3:
        score += 2
    elif imports > 1:
        score += 1
    score += min(3, keyword_score // 3)

    if score <= 1:
        level = "simple"
    elif score <= 4:
        level = "standard"
    else:
        level = "complex"
    return SceneComplexity(
        level=level,
        score=score,
        code_lines=code_lines,
        dependencies=imports,
        animation_keywords=keyword_hits,
    )


# ======================================================================
# Failure categories + #10 — prediction from persisted stats
# ======================================================================

FAILURE_CATEGORIES: tuple[str, ...] = (
    "layout_overlap",      # elements collide / overflow the frame
    "text_too_small",      # unreadable at target resolution
    "timing_mismatch",     # animation duration vs scene duration
    "contrast_failure",    # text invisible against background
    "offbrand_color",      # violates brand palette
    "non_determinism",     # Math.random / Date.now flicker sources
    "missing_asset",       # referenced file does not exist
    "audio_desync",        # voiceover longer/shorter than scene
    "resolution_mismatch", # hardcoded size != tier size
)

# Known, mechanically-checkable signals per category — used to
# pre-emptively check before the LLM review (#10's "pre-emptively fix").
_CATEGORY_SIGNALS: dict[str, list[str]] = {
    "non_determinism": ["math.random", "date.now", "performance.now", "uuid"],
    "missing_asset": [],  # handled by the asset scanner (real fs check)
    "resolution_mismatch": ["1920", "1080", "1280", "720"],
}


class FailurePredictor:
    """Predict likely QA failures from persisted per-category stats (#10).

    Counters live in the shared SQLite store keyed by
    (category, complexity-level, engine), so the predictor sharpens as
    QA verdicts accumulate across sessions.
    """

    NAMESPACE = "qa_failures"

    def __init__(self, store: Any) -> None:
        self._store = store

    def record_failure(self, *, category: str, complexity: str, engine: str) -> None:
        """Record one QA failure for (category, complexity, engine)."""
        self._store.bump(self.NAMESPACE, f"{category}|{complexity}|{engine}")

    def record_success(self, *, complexity: str, engine: str, category: str | None = None) -> None:
        """Record QA passes so rates — not raw counts — drive prediction."""
        if category:
            self._store.bump(self.NAMESPACE, f"{category}|{complexity}|{engine}|ok")
        else:
            for cat in FAILURE_CATEGORIES:
                self._store.bump(self.NAMESPACE, f"{cat}|{complexity}|{engine}|ok")

    def predict(self, *, complexity: str, engine: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the most likely failure categories for this scene profile."""
        stats = self._store.counters_in_namespace(self.NAMESPACE)
        scored: list[dict[str, Any]] = []
        for category in FAILURE_CATEGORIES:
            fails = stats.get(f"{category}|{complexity}|{engine}", 0)
            oks = stats.get(f"{category}|{complexity}|{engine}|ok", 0)
            total = fails + oks
            if total == 0:
                continue  # never seen — no honest prediction
            scored.append(
                {
                    "category": category,
                    "failure_rate": round(fails / total, 3),
                    "samples": total,
                }
            )
        scored.sort(key=lambda item: (-item["failure_rate"], -item["samples"]))
        return scored[:top_k]

    def preflight_checks(self, scene_path: Path, *, engine: str) -> list[dict[str, str]]:
        """Mechanical pre-emptive checks for predicted categories (#10).

        Real static signals only: non-determinism sources and missing
        referenced assets. Findings are returned as fix suggestions the
        QA prompt (and the agent) can act on.
        """
        findings: list[dict[str, str]] = []
        try:
            code = scene_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return findings
        lowered = code.lower()
        for signal in _CATEGORY_SIGNALS.get("non_determinism", []):
            for match in re.finditer(re.escape(signal), lowered):
                line_no = lowered[: match.start()].count("\n") + 1
                findings.append(
                    {
                        "category": "non_determinism",
                        "detail": f"'{signal}' at line {line_no} causes flicker on re-render",
                        "fix": f"Seed the value or replace '{signal}' with a deterministic constant",
                    }
                )
                break  # one finding per signal is enough
        from openharness.services.render_cache import find_asset_paths

        referenced = find_asset_paths(code, engine, cwd=scene_path.parent)
        referenced_names = {p.name for p in referenced}
        for name in re.findall(r"""["']([^"']+\.(?:png|jpg|jpeg|webp|svg|mp3|wav|ogg|mp4))["']""", code):
            if Path(name).name not in referenced_names:
                findings.append(
                    {
                        "category": "missing_asset",
                        "detail": f"referenced asset not found on disk: {name}",
                        "fix": f"add {name} to the assets directory or fix the path",
                    }
                )
        return findings


# ======================================================================
# QA verdicts
# ======================================================================

@dataclass
class QAVerdict:
    """One scene's review outcome."""

    scene: str
    passed: bool
    score: float
    depth: str  # "light" | "deep"
    issues: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "passed": self.passed,
            "score": self.score,
            "depth": self.depth,
            "issues": self.issues,
            "summary": self.summary,
        }


def parse_verdict_payload(payload: str, scene_names: list[str]) -> list[QAVerdict] | None:
    """Parse the model's structured QA response (JSON, one verdict per scene).

    The prompt demands strict JSON of the form::

        {"verdicts": [{"scene": "...", "passed": true, "score": 0.0,
                        "issues": [{"category": "...", "detail": "...",
                                     "severity": "...", "fix": "..."}],
                        "summary": "..."}]}

    Returns None when parsing fails so callers can retry honestly.
    """
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    raw_verdicts = data.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return None
    # Accept the scene identified either by full filename ("scene_001.tsx")
    # or bare stem ("scene_001") — models echo both forms and the previous
    # one-way normalization silently failed the whole parse when the form
    # didn't match scene_names exactly.
    name_set = set(scene_names)
    stem_to_name = {Path(n).stem: n for n in scene_names}
    # Coverage is required over the CANONICAL names; stems are lookup
    # aliases only.
    expected = name_set
    got: set[str] = set()
    verdicts: list[QAVerdict] = []
    for raw in raw_verdicts:
        if not isinstance(raw, dict):
            continue
        scene = str(raw.get("scene") or "").strip()
        if not scene:
            continue
        if scene in name_set:
            name = scene
        elif Path(scene).stem in stem_to_name:
            name = stem_to_name[Path(scene).stem]
        elif scene in stem_to_name:
            name = stem_to_name[scene]
        else:
            continue  # verdict for a scene we were not asked about
        got.add(name)
        try:
            score = float(raw.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        # Models return either 0-1 or 0-100 scales. A raw "85" clamped
        # straight into [0, 1] would become a PERFECT 1.0 — and QA scores
        # feed the early-exit promotion threshold (0.9), so a mediocre
        # 85/100 scene would be auto-promoted. Normalize first.
        if score > 1.0:
            score = score / 100.0 if score <= 100.0 else 1.0
        score = max(0.0, min(1.0, score))
        issues = [
            {
                "category": str(i.get("category") or "general"),
                "detail": str(i.get("detail") or ""),
                "severity": str(i.get("severity") or "minor"),
                "fix": str(i.get("fix") or ""),
            }
            for i in (raw.get("issues") or [])
            if isinstance(i, dict)
        ]
        verdicts.append(
            QAVerdict(
                scene=name,
                passed=bool(raw.get("passed")),
                score=score,
                depth=str(raw.get("depth") or "deep"),
                issues=issues,
                summary=str(raw.get("summary") or "")[:400],
            )
        )
    if not verdicts or not expected.issubset(got):
        return None
    return verdicts


# ======================================================================
# Prompt builders (#9 depth + #10 predictions)
# ======================================================================

def build_qa_prompt(
    scene_path: Path,
    *,
    engine: str,
    tier: str,
    resolution: str,
    complexity: SceneComplexity,
    predictions: list[dict[str, Any]],
    preflight: list[dict[str, str]],
) -> str:
    """Build one scene's QA prompt with depth-appropriate rubric."""
    try:
        code = scene_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        code = ""
    name = scene_path.name
    if complexity.is_simple:
        depth_block = (
            "Review depth: LIGHTWEIGHT CHECK. This is a simple scene; check only:\n"
            "1. Syntax is valid for the engine.\n"
            "2. Text is readable at the target resolution.\n"
            "3. Timing fits the scene duration.\n"
        )
    else:
        depth_block = (
            "Review depth: DEEP REVIEW. This scene is complex; check:\n"
            "1. Syntax validity and engine API correctness.\n"
            "2. Layout collisions and safe margins at target resolution.\n"
            "3. Text legibility (size, contrast, weight).\n"
            "4. Timing: keyframe durations vs scene duration.\n"
            "5. Determinism (no random/time sources that flicker on re-render).\n"
            "6. Brand consistency (colors, fonts) when a brand kit exists.\n"
            "7. Performance hazards (unbounded particle counts, heavy shaders).\n"
        )
    prediction_lines = ""
    if predictions:
        ranked = ", ".join(
            f"{p['category']} ({int(p['failure_rate'] * 100)}% of {p['samples']} past scenes)"
            for p in predictions
        )
        prediction_lines = (
            f"\nHIGH-RISK CATEGORIES for scenes like this one (from project history): {ranked}\n"
            "Check these categories first and most carefully.\n"
        )
    preflight_lines = ""
    if preflight:
        preflight_lines = "\nPREFLIGHT FINDINGS (already detected mechanically — confirm and fix):\n" + "\n".join(
            f"- [{f['category']}] {f['detail']} → suggested fix: {f['fix']}" for f in preflight
        )
    return (
        f"You are a strict QA reviewer for generated video scenes.\n"
        f"Scene: {name}\n"
        f"Engine: {engine}\n"
        f"Render tier: {tier} (target resolution {resolution})\n"
        f"Complexity: {complexity.level} (score {complexity.score}, "
        f"{complexity.code_lines} lines, {complexity.dependencies} imports, "
        f"{complexity.animation_keywords} animation constructs)\n"
        f"{depth_block}{prediction_lines}{preflight_lines}\n"
        f"Scene code:\n```\n{code}\n```\n\n"
        "Respond with STRICT JSON only, no prose:\n"
        '{"verdicts": [{"scene": "<scene filename>", "passed": <bool>, '
        '"score": <0.0-1.0>, "issues": [{"category": "<one of: '
        + ", ".join(FAILURE_CATEGORIES)
        + ' or general>", "detail": "<what is wrong>", "severity": "<minor|major|blocker>", '
        '"fix": "<concrete fix>"}], "summary": "<one-line verdict>"}]}\n'
    )


def build_batch_qa_prompt(
    scene_paths: list[Path],
    *,
    engine: str,
    tier: str,
    resolution: str,
) -> str:
    """Build a multi-scene QA prompt (#8 — one call reviews N scenes)."""
    blocks: list[str] = []
    for path in scene_paths:
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            code = ""
        complexity = classify_scene(path)
        blocks.append(
            f"### SCENE: {path.name} (complexity: {complexity.level})\n```\n{code}\n```"
        )
    return (
        "You are a strict QA reviewer for generated video scenes. Review EACH scene below.\n"
        f"Engine: {engine}; render tier: {tier} (target resolution {resolution}).\n"
        "Check syntax, layout, text legibility at target resolution, timing, and determinism.\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with STRICT JSON only, no prose, ONE verdict object per scene:\n"
        '{"verdicts": [{"scene": "<scene filename>", "passed": <bool>, "score": <0.0-1.0>, '
        '"issues": [{"category": "...", "detail": "...", "severity": "<minor|major|blocker>", '
        '"fix": "..."}], "summary": "<one-line>"}]}\n'
    )


# ======================================================================
# The reviewer — #7 parallel waves, #8 batching, wiring it together
# ======================================================================

class QAReviewer:
    """Parallel scene QA against an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        wave_size: int = 5,
        batching: bool = True,
        batch_size: int = 5,
        smart_depth: bool = True,
        failure_prediction: bool = True,
        max_tokens: int = 8192,
        timeout_seconds: float = 180.0,
        store: Any = None,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("QAReviewer needs a real endpoint: base_url + api_key + model")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._wave_size = max(1, wave_size)
        self._batching = batching
        self._batch_size = max(1, batch_size)
        self._smart_depth = smart_depth
        self._failure_prediction = failure_prediction
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._predictor = FailurePredictor(store) if store is not None else None
        self.calls_made = 0
        self.review_waves = 0
        # One shared HTTP client for the reviewer's lifetime: connection
        # pooling means QA waves don't pay a fresh TCP+TLS handshake per
        # LLM call (5-10 calls per wave add up on hosted endpoints).
        self._http: Any = None

    def _http_client(self):
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def aclose(self) -> None:
        """Release the shared HTTP client (call when the session ends)."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    # -- single LLM call ------------------------------------------------

    async def _chat(self, prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": 0.1,
        }
        self.calls_made += 1
        client = self._http_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"QA endpoint returned no choices: {str(data)[:200]}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    # -- public API ------------------------------------------------

    async def review_scenes(
        self,
        scene_paths: list[Path],
        *,
        engine: str = "hyperframes",
        tier: str = "draft",
        resolution: str = "640x360",
        mode: str = "auto",  # auto | single | batch
    ) -> list[QAVerdict]:
        """Review scenes: batched (#8) and/or in parallel waves (#7)."""
        if not scene_paths:
            return []
        verdicts: list[QAVerdict] = []
        use_batch = self._batching and mode in {"auto", "batch"}
        if use_batch:
            # Batch simple scenes per call; complex scenes get single
            # deep reviews (their code is long and needs focus). With
            # smart_depth disabled, EVERY scene is batchable — previously
            # they all fell into the "complex" bucket and batching was
            # silently disabled even though batching=True.
            simple: list[Path] = []
            complex_scenes: list[Path] = []
            for path in scene_paths:
                is_simple = not self._smart_depth or classify_scene(path).is_simple
                (simple if is_simple else complex_scenes).append(path)
            groups = [simple[i : i + self._batch_size] for i in range(0, len(simple), self._batch_size)]
            single_jobs = complex_scenes
        elif mode == "batch":
            groups = [
                scene_paths[i : i + self._batch_size]
                for i in range(0, len(scene_paths), self._batch_size)
            ]
            single_jobs = []
        else:
            groups = []
            single_jobs = list(scene_paths)

        semaphore = asyncio.Semaphore(self._wave_size)
        start = time.monotonic()

        async def _review_single(path: Path) -> QAVerdict:
            async with semaphore:
                return await self._review_one(
                    path, engine=engine, tier=tier, resolution=resolution
                )

        async def _review_batch(group: list[Path]) -> list[QAVerdict]:
            async with semaphore:
                names = [p.name for p in group]
                prompt = build_batch_qa_prompt(
                    group, engine=engine, tier=tier, resolution=resolution
                )
                try:
                    payload = await self._chat(prompt)
                except Exception as exc:  # noqa: BLE001 - verdicts carry the error
                    error_msg = f"QA call failed: {type(exc).__name__}: {exc}"
                    return [
                        QAVerdict(scene=p.name, passed=False, score=0.0, depth="batch",
                                  issues=[{"category": "qa_error", "detail": error_msg,
                                           "severity": "blocker", "fix": "retry the review"}],
                                  summary=error_msg)
                        for p in group
                    ]
                parsed = parse_verdict_payload(payload, names)
                if parsed is None:
                    error_msg = "QA response was not valid per-scene JSON"
                    return [
                        QAVerdict(scene=p.name, passed=False, score=0.0, depth="batch",
                                  issues=[{"category": "qa_error", "detail": error_msg,
                                           "severity": "major", "fix": "retry the review"}],
                                  summary=error_msg)
                        for p in group
                    ]
                return parsed

        tasks = [asyncio.create_task(_review_single(p)) for p in single_jobs]
        tasks += [asyncio.create_task(_review_batch(g)) for g in groups if g]
        self.review_waves += 1
        results = await asyncio.gather(*tasks)
        for result in results:
            if isinstance(result, QAVerdict):
                verdicts.append(result)
            else:
                verdicts.extend(result)
        # Persist outcomes so the predictor sharpens (#10).
        if self._predictor is not None:
            for verdict in verdicts:
                complexity = classify_scene(_find_scene(scene_paths, verdict.scene))
                for issue in verdict.issues:
                    self._predictor.record_failure(
                        category=str(issue.get("category") or "general"),
                        complexity=complexity.level,
                        engine=engine,
                    )
                if verdict.passed:
                    self._predictor.record_success(
                        complexity=complexity.level, engine=engine
                    )
        log.info(
            "QA: %d scenes reviewed in %.1fs (%d LLM calls)",
            len(verdicts), time.monotonic() - start, self.calls_made,
        )
        return verdicts

    async def _review_one(
        self, path: Path, *, engine: str, tier: str, resolution: str
    ) -> QAVerdict:
        complexity = classify_scene(path)
        predictions: list[dict[str, Any]] = []
        preflight: list[dict[str, str]] = []
        if self._failure_prediction and self._predictor is not None:
            predictions = self._predictor.predict(
                complexity=complexity.level, engine=engine
            )
            preflight = self._predictor.preflight_checks(path, engine=engine)
        prompt = build_qa_prompt(
            path,
            engine=engine,
            tier=tier,
            resolution=resolution,
            complexity=complexity,
            predictions=predictions,
            preflight=preflight,
        )
        try:
            payload = await self._chat(prompt)
        except Exception as exc:  # noqa: BLE001 - verdict carries the error
            error_msg = f"QA call failed: {type(exc).__name__}: {exc}"
            return QAVerdict(
                scene=path.name, passed=False, score=0.0,
                depth="light" if complexity.is_simple else "deep",
                issues=[{"category": "qa_error", "detail": error_msg,
                         "severity": "blocker", "fix": "retry the review"}],
                summary=error_msg,
            )
        parsed = parse_verdict_payload(payload, [path.name])
        if parsed is None:
            error_msg = "QA response was not valid JSON"
            return QAVerdict(
                scene=path.name, passed=False, score=0.0,
                depth="light" if complexity.is_simple else "deep",
                issues=[{"category": "qa_error", "detail": error_msg,
                         "severity": "major", "fix": "retry the review"}],
                summary=error_msg,
            )
        verdict = parsed[0]
        verdict.depth = "light" if complexity.is_simple else "deep"
        return verdict

    def stats(self) -> dict[str, Any]:
        return {
            "calls_made": self.calls_made,
            "review_waves": self.review_waves,
            "wave_size": self._wave_size,
            "batching": self._batching,
            "model": self._model,
        }


def _find_scene(scene_paths: list[Path], name: str) -> Path:
    for path in scene_paths:
        if path.name == name or path.stem == name:
            return path
    return Path(name)


def build_qa_reviewer_from_metadata(metadata: dict[str, Any] | None) -> QAReviewer | None:
    """Construct a QAReviewer from tool metadata (settings-driven).

    Precedence: settings.qa overrides > scene_gen config. Returns None
    (with a logged reason) when no usable endpoint is configured —
    callers must surface that honestly instead of inventing verdicts.
    """
    metadata = metadata or {}
    settings = metadata.get("settings")
    qa_settings = getattr(settings, "qa", None) if settings is not None else None
    scene_gen = metadata.get("scene_gen_config") or {}
    base_url = (getattr(qa_settings, "base_url", "") or scene_gen.get("base_url") or "").strip()
    api_key = (getattr(qa_settings, "api_key", "") or scene_gen.get("api_key") or "").strip()
    model = (getattr(qa_settings, "model", "") or scene_gen.get("model") or "").strip()
    if not (base_url and api_key and model):
        return None
    from openharness.services.persistent_store import get_persistent_store

    return QAReviewer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        wave_size=int(getattr(qa_settings, "wave_size", 5) or 5),
        batching=bool(getattr(qa_settings, "review_batching", True)),
        batch_size=int(getattr(qa_settings, "review_batch_size", 5) or 5),
        smart_depth=bool(getattr(qa_settings, "smart_depth", True)),
        failure_prediction=bool(getattr(qa_settings, "failure_prediction", True)),
        max_tokens=int(getattr(qa_settings, "max_tokens", 8192) or 8192),
        store=get_persistent_store(),
    )
