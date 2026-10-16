"""Failure pattern learning — Item #14.

Track which scene types fail review most often. Pre-emptively apply
stricter prompts for those types. Uses a persistent SQLite store so
failure patterns survive across sessions.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """A single scene failure record."""

    scene_type: str  # e.g. "title_card", "data_viz", "transition"
    failure_reason: str  # e.g. "lint_error", "qa_reject", "determinism_fail"
    timestamp: float
    scene_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureStats:
    """Aggregated failure stats for a scene type."""

    scene_type: str
    total_failures: int
    total_attempts: int
    failure_rate: float
    top_reasons: list[tuple[str, int]]  # (reason, count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "total_failures": self.total_failures,
            "total_attempts": self.total_attempts,
            "failure_rate": round(self.failure_rate, 3),
            "top_reasons": self.top_reasons,
        }


class FailurePatternTracker:
    """Tracks scene failures and recommends stricter prompts for failing types.

    Persists to SQLite so patterns survive across sessions. The tracker
    also tracks successes so we can compute failure rates.
    """

    def __init__(self, *, persist_path: str | Path | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._failures: dict[str, list[FailureRecord]] = defaultdict(list)
        self._attempts: dict[str, int] = defaultdict(int)
        self._conn: sqlite3.Connection | None = None
        if self._persist_path:
            self._init_db()
            self._load_from_db()

    def record_success(self, scene_type: str) -> None:
        """Record a successful scene generation."""
        self._attempts[scene_type] += 1
        if self._conn:
            self._conn.execute(
                "INSERT INTO attempts (scene_type, success, timestamp) VALUES (?, 1, ?)",
                (scene_type, time.time()),
            )
            self._conn.commit()

    def record_failure(
        self,
        scene_type: str,
        failure_reason: str,
        *,
        scene_description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a failed scene generation."""
        record = FailureRecord(
            scene_type=scene_type,
            failure_reason=failure_reason,
            timestamp=time.time(),
            scene_description=scene_description,
            metadata=metadata or {},
        )
        self._failures[scene_type].append(record)
        self._attempts[scene_type] += 1
        if self._conn:
            self._conn.execute(
                """INSERT INTO failures (scene_type, failure_reason, timestamp, scene_description, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    scene_type,
                    failure_reason,
                    record.timestamp,
                    scene_description,
                    json.dumps(record.metadata),
                ),
            )
            self._conn.execute(
                "INSERT INTO attempts (scene_type, success, timestamp) VALUES (?, 0, ?)",
                (scene_type, time.time()),
            )
            self._conn.commit()

    def get_stats(self, scene_type: str) -> FailureStats:
        """Return failure stats for a scene type."""
        failures = self._failures.get(scene_type, [])
        attempts = self._attempts.get(scene_type, 0)
        # Count reasons
        reason_counts: dict[str, int] = defaultdict(int)
        for f in failures:
            reason_counts[f.failure_reason] += 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
        return FailureStats(
            scene_type=scene_type,
            total_failures=len(failures),
            total_attempts=attempts,
            failure_rate=len(failures) / attempts if attempts else 0.0,
            top_reasons=top_reasons,
        )

    def get_all_stats(self) -> list[FailureStats]:
        """Return stats for all scene types with recorded attempts."""
        all_types = set(self._failures.keys()) | set(self._attempts.keys())
        return [self.get_stats(t) for t in sorted(all_types)]

    def should_use_stricter_prompt(self, scene_type: str, *, threshold: float = 0.3) -> bool:
        """Return True if this scene type has a high failure rate and should
        use a stricter generation prompt.

        Args:
            scene_type: The scene type to check.
            threshold: Failure rate above which to use stricter prompts (default 30%).
        """
        stats = self.get_stats(scene_type)
        # Need at least 3 attempts before we trust the failure rate
        if stats.total_attempts < 3:
            return False
        return stats.failure_rate >= threshold

    def get_strictness_level(self, scene_type: str) -> int:
        """Return a strictness level 1-3 for the scene type.

        1 = normal (low failure rate)
        2 = stricter (moderate failure rate)
        3 = strictest (high failure rate)
        """
        stats = self.get_stats(scene_type)
        if stats.total_attempts < 3:
            return 1
        if stats.failure_rate >= 0.5:
            return 3
        if stats.failure_rate >= 0.3:
            return 2
        return 1

    def clear(self) -> None:
        """Clear all recorded failures and attempts."""
        self._failures.clear()
        self._attempts.clear()
        if self._conn:
            self._conn.execute("DELETE FROM failures")
            self._conn.execute("DELETE FROM attempts")
            self._conn.commit()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._persist_path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_type TEXT,
                failure_reason TEXT,
                timestamp REAL,
                scene_description TEXT,
                metadata TEXT
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_type TEXT,
                success INTEGER,
                timestamp REAL
            )"""
        )
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load failures and attempts from the DB into memory."""
        if not self._conn:
            return
        try:
            cursor = self._conn.execute(
                "SELECT scene_type, failure_reason, timestamp, scene_description, metadata FROM failures"
            )
            for row in cursor:
                scene_type, failure_reason, timestamp, description, metadata_str = row
                metadata = json.loads(metadata_str) if metadata_str else {}
                self._failures[scene_type].append(
                    FailureRecord(
                        scene_type=scene_type,
                        failure_reason=failure_reason,
                        timestamp=timestamp,
                        scene_description=description or "",
                        metadata=metadata,
                    )
                )
            cursor = self._conn.execute(
                "SELECT scene_type, COUNT(*) FROM attempts GROUP BY scene_type"
            )
            for row in cursor:
                scene_type, count = row
                self._attempts[scene_type] = count
        except Exception as exc:
            log.warning("failed to load failure patterns: %s", exc)
