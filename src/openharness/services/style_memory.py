"""Cross-project style memory — what worked, remembered across projects.

Every time a project completes with a high quality score, its style
DNA (palette, fonts, motion curves, transition types, drawn from
STYLE.md / brand tokens / scene code) can be recorded here. When a
new project starts, the agent asks "what styles worked well for
knowledge shorts?" and gets concrete, ranked suggestions instead of
starting from scratch — the harness gets *better over time*.

Storage: SQLite at ``<data_dir>/video/style_memory.sqlite3`` with one
row per (project, style) recording the extracted tokens plus the
project's final quality score. Suggestions rank by quality score and
recency, optionally filtered by content type ("knowledge shorts",
"promo", ...).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.services._sqlite_kit import SQLiteStore, video_store_dir

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS styles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT NOT NULL,
    style_name      TEXT NOT NULL,
    content_type    TEXT NOT NULL DEFAULT '',
    palette_json    TEXT NOT NULL DEFAULT '[]',
    fonts_json      TEXT NOT NULL DEFAULT '[]',
    motion_json     TEXT NOT NULL DEFAULT '[]',
    transitions_json TEXT NOT NULL DEFAULT '[]',
    quality_score   REAL NOT NULL DEFAULT 0.0,
    notes           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    UNIQUE (project, style_name)
);
CREATE INDEX IF NOT EXISTS styles_quality ON styles (content_type, quality_score DESC);
"""

_TOKEN_EXTS = {".json", ".yaml", ".yml", ".md"}


@dataclass
class StyleRecord:
    """One stored style pattern."""

    project: str
    style_name: str
    content_type: str = ""
    palette: list[str] = None  # type: ignore[assignment]
    fonts: list[str] = None  # type: ignore[assignment]
    motion: list[str] = None  # type: ignore[assignment]
    transitions: list[str] = None  # type: ignore[assignment]
    quality_score: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.palette = self.palette or []
        self.fonts = self.fonts or []
        self.motion = self.motion or []
        self.transitions = self.transitions or []

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "style_name": self.style_name,
            "content_type": self.content_type,
            "palette": self.palette,
            "fonts": self.fonts,
            "motion": self.motion,
            "transitions": self.transitions,
            "quality_score": self.quality_score,
            "notes": self.notes,
        }


class StyleMemory:
    """Persistent store of successful style patterns."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._store = SQLiteStore(
            path or video_store_dir() / "style_memory.sqlite3", _SCHEMA
        )

    # ------------------------------------------------------------------
    def record(
        self,
        record: StyleRecord,
    ) -> bool:
        """Insert or update a style record (keyed by project + style name)."""
        cursor = self._store.execute(
            """
            INSERT INTO styles (project, style_name, content_type, palette_json,
                                fonts_json, motion_json, transitions_json,
                                quality_score, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (project, style_name) DO UPDATE SET
                content_type     = excluded.content_type,
                palette_json     = excluded.palette_json,
                fonts_json       = excluded.fonts_json,
                motion_json      = excluded.motion_json,
                transitions_json = excluded.transitions_json,
                quality_score    = excluded.quality_score,
                notes            = excluded.notes,
                created_at       = excluded.created_at
            """,
            (
                record.project,
                record.style_name,
                record.content_type,
                json.dumps(record.palette),
                json.dumps(record.fonts),
                json.dumps(record.motion),
                json.dumps(record.transitions),
                float(record.quality_score),
                record.notes,
                self._store.now(),
            ),
        )
        return cursor is not None

    # ------------------------------------------------------------------
    def suggest(
        self,
        *,
        content_type: str | None = None,
        min_quality: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank stored styles by (quality, recency), optionally filtered."""
        if content_type:
            rows = self._store.query(
                "SELECT * FROM styles WHERE content_type = ? AND quality_score >= ? "
                "ORDER BY quality_score DESC, created_at DESC LIMIT ?",
                (content_type, float(min_quality), int(limit)),
            )
        else:
            rows = self._store.query(
                "SELECT * FROM styles WHERE quality_score >= ? "
                "ORDER BY quality_score DESC, created_at DESC LIMIT ?",
                (float(min_quality), int(limit)),
            )
        return [self._row_to_dict(row) for row in rows]

    def projects(self) -> list[dict[str, Any]]:
        """Per-project summary: styles stored, best quality score."""
        rows = self._store.query(
            "SELECT project, COUNT(*) AS n, MAX(quality_score) AS best, "
            "MAX(created_at) AS latest FROM styles GROUP BY project "
            "ORDER BY latest DESC"
        )
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        rows = self._store.query(
            "SELECT content_type, COUNT(*) AS n FROM styles GROUP BY content_type"
        )
        return {
            "available": self._store.available,
            "path": str(self._store.path),
            "total": sum(int(row["n"]) for row in rows),
            "by_content_type": {row["content_type"] or "(none)": int(row["n"]) for row in rows},
        }

    def reset(self) -> None:
        self._store.execute("DELETE FROM styles")

    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        def _load(raw: str) -> list[str]:
            try:
                value = json.loads(raw or "[]")
                return [str(item) for item in value] if isinstance(value, list) else []
            except ValueError:
                return []

        return {
            "project": row["project"],
            "style_name": row["style_name"],
            "content_type": row["content_type"],
            "palette": _load(row["palette_json"]),
            "fonts": _load(row["fonts_json"]),
            "motion": _load(row["motion_json"]),
            "transitions": _load(row["transitions_json"]),
            "quality_score": float(row["quality_score"]),
            "notes": row["notes"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def extract_from_tokens(path: Path) -> dict[str, list[str]]:
        """Extract palette/fonts tokens from a brand/tokens JSON or YAML file.

        Understands OpenVideoHarness ``styles/<slug>/tokens.json``
        (``palette`` hex colors, ``fonts`` families) and plain nested
        mappings containing hex colors or font-family strings.
        """
        import re

        result: dict[str, list[str]] = {"palette": [], "fonts": []}
        if not path.is_file() or path.suffix.lower() not in _TOKEN_EXTS:
            return result
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return result

        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            data: Any = None
            if path.suffix.lower() == ".json":
                try:
                    data = json.loads(text)
                except ValueError:
                    data = None
            else:
                try:
                    import yaml

                    data = yaml.safe_load(text)
                except ImportError:
                    data = None
            if isinstance(data, dict):
                result["palette"] = [
                    f"#{match.group(1)}"
                    for match in re.finditer(
                        r"#([0-9a-fA-F]{6})\b", json.dumps(data, default=str)
                    )
                ][:8]
                fonts = data.get("fonts")
                if isinstance(fonts, list):
                    result["fonts"] = [str(item) for item in fonts][:6]
                elif isinstance(fonts, dict):
                    result["fonts"] = [str(value) for value in fonts.values()][:6]
        else:  # .md — scan fenced blocks / lists for hex colors + font names
            result["palette"] = [
                f"#{match.group(1)}"
                for match in re.finditer(r"#([0-9a-fA-F]{6})\b", text)
            ][:8]
            font_matches = re.findall(
                r"font(?:s|family|\s*-\s*family)?\s*[:=]\s*['\"]?([A-Za-z0-9 _-]+)",
                text,
                re.IGNORECASE,
            )
            result["fonts"] = [match.strip() for match in font_matches][:6]
        return result


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------
_style_memory: StyleMemory | None = None


def get_style_memory() -> StyleMemory:
    global _style_memory
    if _style_memory is None:
        _style_memory = StyleMemory()
    return _style_memory


def set_style_memory(memory: StyleMemory | None) -> None:
    global _style_memory
    _style_memory = memory
