"""Failure pattern database — which scene types fail QA most often.

The same scene types fail repeatedly: dense math overlays, chart
scenes with small labels, photo-collage transitions. This store logs
every QA outcome with the scene type, the failure reason, and the
prompt used, and aggregates a **failure rate per scene type** over
time. When a new scene of a high-failure type is about to be
generated, the agent can call :meth:`FailurePatterns.strictness_hint`
and automatically apply a stricter prompt (more constraints, more
examples) — breaking the cycle instead of re-learning it every
project.

Storage: SQLite at ``<data_dir>/video/failure_patterns.sqlite3``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openharness.services._sqlite_kit import SQLiteStore, video_store_dir

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_type TEXT NOT NULL,
    outcome    TEXT NOT NULL CHECK (outcome IN ('pass', 'fail')),
    reason     TEXT NOT NULL DEFAULT '',
    prompt_id  TEXT NOT NULL DEFAULT '',
    project    TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_type ON events (scene_type, created_at);
"""


class FailurePatterns:
    """QA-failure statistics per scene type."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._store = SQLiteStore(
            path or video_store_dir() / "failure_patterns.sqlite3", _SCHEMA
        )

    # ------------------------------------------------------------------
    def record_failure(
        self,
        scene_type: str,
        *,
        reason: str = "",
        prompt_id: str = "",
        project: str = "",
    ) -> None:
        self._store.execute(
            "INSERT INTO events (scene_type, outcome, reason, prompt_id, project, "
            "created_at) VALUES (?, 'fail', ?, ?, ?, ?)",
            (scene_type.strip() or "unknown", reason.strip(), prompt_id, project,
             self._store.now()),
        )

    def record_success(
        self, scene_type: str, *, prompt_id: str = "", project: str = ""
    ) -> None:
        self._store.execute(
            "INSERT INTO events (scene_type, outcome, reason, prompt_id, project, "
            "created_at) VALUES (?, 'pass', '', ?, ?, ?)",
            (scene_type.strip() or "unknown", prompt_id, project, self._store.now()),
        )

    # ------------------------------------------------------------------
    def failure_rates(self, *, min_samples: int = 3) -> list[dict[str, Any]]:
        """Per scene type: total, failures, rate, top failure reasons."""
        rows = self._store.query(
            """
            SELECT scene_type,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome = 'fail' THEN 1 ELSE 0 END) AS failures
            FROM events GROUP BY scene_type
            HAVING COUNT(*) >= ?
            ORDER BY failures * 1.0 / COUNT(*) DESC, total DESC
            """,
            (int(min_samples),),
        )
        rates: list[dict[str, Any]] = []
        for row in rows:
            scene_type = row["scene_type"]
            top = self._store.query(
                "SELECT reason, COUNT(*) AS n FROM events "
                "WHERE scene_type = ? AND outcome = 'fail' AND reason != '' "
                "GROUP BY reason ORDER BY n DESC LIMIT 3",
                (scene_type,),
            )
            rates.append(
                {
                    "scene_type": scene_type,
                    "total": int(row["total"]),
                    "failures": int(row["failures"]),
                    "failure_rate": round(int(row["failures"]) / max(1, int(row["total"])), 3),
                    "top_reasons": [
                        {"reason": item["reason"], "count": int(item["n"])} for item in top
                    ],
                }
            )
        return rates

    # ------------------------------------------------------------------
    def strictness_hint(
        self, scene_type: str, *, threshold: float = 0.3, min_samples: int = 3
    ) -> dict[str, Any]:
        """Guidance for generating a scene of ``scene_type``.

        Returns the scene's observed stats and, when the failure rate
        exceeds ``threshold``, an explicit recommendation to use a
        stricter prompt together with the concrete failure reasons to
        constrain against.
        """
        rows = self._store.query(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN outcome = 'fail' THEN 1 ELSE 0 END) AS failures "
            "FROM events WHERE scene_type = ?",
            (scene_type.strip() or "unknown",),
        )
        total = int(rows[0]["total"]) if rows else 0
        failures = int(rows[0]["failures"] or 0) if rows else 0
        rate = failures / total if total else 0.0
        hint: dict[str, Any] = {
            "scene_type": scene_type,
            "samples": total,
            "failures": failures,
            "failure_rate": round(rate, 3),
            "high_risk": False,
            "recommendation": "",
        }
        if total >= min_samples and rate >= threshold:
            top = self._store.query(
                "SELECT reason, COUNT(*) AS n FROM events "
                "WHERE scene_type = ? AND outcome = 'fail' AND reason != '' "
                "GROUP BY reason ORDER BY n DESC LIMIT 3",
                (scene_type,),
            )
            reasons = [item["reason"] for item in top]
            hint["high_risk"] = True
            hint["top_reasons"] = reasons
            hint["recommendation"] = (
                f"High-risk scene type ({rate:.0%} failure rate over {total} "
                "samples): generate with a STRICTER prompt — add explicit "
                "constraints for: " + ("; ".join(reasons) if reasons else "the historically failing aspects")
                + ". Include worked examples and hard numeric limits."
            )
        return hint

    def stats(self) -> dict[str, Any]:
        rows = self._store.query(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN outcome = 'fail' THEN 1 ELSE 0 END) AS failures "
            "FROM events"
        )
        types = self._store.query(
            "SELECT COUNT(DISTINCT scene_type) AS n FROM events"
        )
        return {
            "available": self._store.available,
            "path": str(self._store.path),
            "total_events": int(rows[0]["total"]) if rows else 0,
            "total_failures": int(rows[0]["failures"] or 0) if rows else 0,
            "scene_types": int(types[0]["n"]) if types else 0,
        }

    def reset(self) -> None:
        self._store.execute("DELETE FROM events")


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------
_failure_patterns: FailurePatterns | None = None


def get_failure_patterns() -> FailurePatterns:
    global _failure_patterns
    if _failure_patterns is None:
        _failure_patterns = FailurePatterns()
    return _failure_patterns


def set_failure_patterns(store: FailurePatterns | None) -> None:
    global _failure_patterns
    _failure_patterns = store
