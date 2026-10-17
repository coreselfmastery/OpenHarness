"""Simple usage aggregation."""

from __future__ import annotations

from openharness.api.usage import UsageSnapshot


class CostTracker:
    """Accumulate usage over the lifetime of a session."""

    def __init__(self) -> None:
        self._usage = UsageSnapshot()

    def add(self, usage: UsageSnapshot) -> None:
        """Add a usage snapshot to the running total.

        Provider prompt-cache accounting is carried through: dropping
        cache_read/cache_write tokens under-reported total usage for
        prefix-cached Anthropic sessions (their input_tokens EXCLUDE
        cached tokens, so the cache fields are not redundant — they are
        the missing spend).
        """
        self._usage = UsageSnapshot(
            input_tokens=self._usage.input_tokens + usage.input_tokens,
            output_tokens=self._usage.output_tokens + usage.output_tokens,
            cache_read_tokens=self._usage.cache_read_tokens + getattr(
                usage, "cache_read_tokens", 0
            ),
            cache_write_tokens=self._usage.cache_write_tokens + getattr(
                usage, "cache_write_tokens", 0
            ),
        )

    @property
    def total(self) -> UsageSnapshot:
        """Return the aggregated usage."""
        return self._usage
