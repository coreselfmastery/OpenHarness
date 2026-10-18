"""Prompt effectiveness tracker — measure prompts by their success rate.

Prompt engineering is guesswork without data. Every prompt template is
versioned and tagged here; every scene generated from it records a
pass/fail result at QA time. Over time each prompt version has a real
success rate, and low performers are flagged for revision or
retirement — turning prompt quality from anecdote into measurement.

Storage: SQLite at ``<data_dir>/video/prompt_tracker.sqlite3`` with a
``prompts`` registry (template + version, upsertable) and a ``results``
event log.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openharness.services._sqlite_kit import SQLiteStore, video_store_dir

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id   TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    template    TEXT NOT NULL DEFAULT '',
    tags_json   TEXT NOT NULL DEFAULT '[]',
    retired     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE (prompt_id, version)
);
CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    scene       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS results_prompt ON results (prompt_id, version);
"""

#: Minimum uses before a prompt can be flagged as low-performing.
FLAG_MIN_USES = 5
#: Success rate below which a prompt is flagged for revision/retirement.
FLAG_RATE = 0.5


class PromptTracker:
    """Versioned prompt registry + per-use pass/fail scoring."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._store = SQLiteStore(
            path or video_store_dir() / "prompt_tracker.sqlite3", _SCHEMA
        )

    # ------------------------------------------------------------------
    def register_prompt(
        self,
        prompt_id: str,
        *,
        version: int = 1,
        template: str = "",
        tags: list[str] | None = None,
    ) -> bool:
        """Insert or update a prompt version (template stays when empty)."""
        import json

        cursor = self._store.execute(
            """
            INSERT INTO prompts (prompt_id, version, template, tags_json, retired,
                                 created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT (prompt_id, version) DO UPDATE SET
                template  = CASE WHEN excluded.template != '' THEN excluded.template
                                 ELSE prompts.template END,
                tags_json = excluded.tags_json,
                retired   = 0
            """,
            (
                prompt_id.strip(),
                int(version),
                template or "",
                json.dumps(tags or []),
                self._store.now(),
            ),
        )
        return cursor is not None

    def retire_prompt(self, prompt_id: str, *, version: int | None = None) -> int:
        """Mark matching prompt versions retired; returns rows updated."""
        if version is not None:
            cursor = self._store.execute(
                "UPDATE prompts SET retired = 1 WHERE prompt_id = ? AND version = ?",
                (prompt_id, int(version)),
            )
        else:
            cursor = self._store.execute(
                "UPDATE prompts SET retired = 1 WHERE prompt_id = ?", (prompt_id,)
            )
        return cursor.rowcount if cursor is not None else 0

    # ------------------------------------------------------------------
    def record_result(
        self, prompt_id: str, *, passed: bool, version: int = 1, scene: str = ""
    ) -> None:
        """Record one use of a prompt (pass or fail at QA)."""
        pid = prompt_id.strip()
        now = self._store.now()
        # Auto-register unseen (prompt_id, version) pairs: stats() joins
        # FROM prompts, so a result recorded without a prior
        # register_prompt() call would otherwise be invisible forever.
        self._store.execute(
            "INSERT OR IGNORE INTO prompts (prompt_id, version, created_at) "
            "VALUES (?, ?, ?)",
            (pid, int(version), now),
        )
        self._store.execute(
            "INSERT INTO results (prompt_id, version, passed, scene, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (pid, int(version), 1 if passed else 0, scene, now),
        )

    # ------------------------------------------------------------------
    def stats(self) -> list[dict[str, Any]]:
        """Per prompt version: uses, passes, success rate, flagged state."""
        rows = self._store.query(
            """
            SELECT p.prompt_id, p.version, p.retired, p.tags_json, p.template,
                   COUNT(r.id)          AS uses,
                   SUM(r.passed)        AS passes,
                   MAX(r.created_at)    AS last_used
            FROM prompts p
            LEFT JOIN results r
              ON r.prompt_id = p.prompt_id AND r.version = p.version
            GROUP BY p.prompt_id, p.version
            ORDER BY uses DESC, p.created_at DESC
            """
        )
        import json

        stats: list[dict[str, Any]] = []
        for row in rows:
            uses = int(row["uses"] or 0)
            passes = int(row["passes"] or 0)
            rate = passes / uses if uses else None
            try:
                tags = json.loads(row["tags_json"] or "[]")
            except ValueError:
                tags = []
            flagged = bool(
                uses >= FLAG_MIN_USES and rate is not None and rate < FLAG_RATE
            )
            stats.append(
                {
                    "prompt_id": row["prompt_id"],
                    "version": int(row["version"]),
                    "retired": bool(row["retired"]),
                    "tags": tags,
                    "uses": uses,
                    "passes": passes,
                    "success_rate": round(rate, 3) if rate is not None else None,
                    "flagged_low_performer": flagged,
                    "template_preview": (row["template"] or "")[:160],
                }
            )
        return stats

    def best_version(self, prompt_id: str) -> dict[str, Any] | None:
        """The highest-success NON-RETIRED version of a prompt with >= 1 use."""
        rows = self._store.query(
            """
            SELECT r.version, COUNT(*) AS uses, AVG(r.passed) AS rate
            FROM results r
            JOIN prompts p
              ON p.prompt_id = r.prompt_id AND p.version = r.version
            WHERE r.prompt_id = ? AND p.retired = 0
            GROUP BY r.version ORDER BY rate DESC, uses DESC LIMIT 1
            """,
            (prompt_id.strip(),),
        )
        if not rows:
            return None
        return {
            "prompt_id": prompt_id,
            "version": int(rows[0]["version"]),
            "uses": int(rows[0]["uses"]),
            "success_rate": round(float(rows[0]["rate"]), 3),
        }

    def stats_summary(self) -> dict[str, Any]:
        rows = self._store.query("SELECT COUNT(*) AS n FROM prompts")
        events = self._store.query("SELECT COUNT(*) AS n FROM results")
        stats = self.stats()
        return {
            "available": self._store.available,
            "path": str(self._store.path),
            "prompts_registered": int(rows[0]["n"]) if rows else 0,
            "results_recorded": int(events[0]["n"]) if events else 0,
            "flagged_low_performers": [
                f"{item['prompt_id']}@v{item['version']} "
                f"({item['success_rate']:.0%} over {item['uses']} uses)"
                for item in stats
                if item["flagged_low_performer"]
            ],
        }

    def reset(self) -> None:
        self._store.execute("DELETE FROM results")
        self._store.execute("DELETE FROM prompts")


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------
_prompt_tracker: PromptTracker | None = None


def get_prompt_tracker() -> PromptTracker:
    global _prompt_tracker
    if _prompt_tracker is None:
        _prompt_tracker = PromptTracker()
    return _prompt_tracker


def set_prompt_tracker(tracker: PromptTracker | None) -> None:
    global _prompt_tracker
    _prompt_tracker = tracker
