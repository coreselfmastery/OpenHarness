"""Simple token estimation utilities.

The heuristic ``len(text) // 4`` is intentionally rough — it's used by the
auto-compaction check and other budget guards, not for billing. We avoid
pulling in ``tiktoken`` or other accurate tokenizers because they add ~50 ms
of cold-start cost and several MB of memory, which matters on Kaggle.
"""


def estimate_tokens(text: str) -> int:
    """Estimate tokens from plain text using a rough character heuristic.

    The ``4/3`` padding applied at the conversation level handles the
    systematic underestimate from JSON serialization and structural
    overhead, so we keep the per-string math cheap here.
    """
    if not text:
        return 0
    # Fast path for the common case: short strings (most tool names,
    # status messages, file paths) fit in a single token bucket.
    length = len(text)
    if length <= 4:
        return 1
    return (length + 3) // 4


def estimate_message_tokens(messages: list[str]) -> int:
    """Estimate tokens for a collection of message strings."""
    return sum(estimate_tokens(message) for message in messages)
