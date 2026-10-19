"""Wave 8 — the no-timeout contract (speed pass: kill every premature cap).

Motivation (user-visible): "it's still too slow" / "don't timeout". Three
premature caps were found burning real wall-clock on Kaggle:

1. ``Settings.timeout`` defaulted to 30 s — the OpenAI-compat SDK request
   timeout. Big-prompt prefills (100k+ context) sit silent for 30-90 s
   before the first byte, so calls died mid-flight and the retry storm
   (4 attempts, each repaying connect+prefill) wasted minutes per call.
2. The tuning doc recommended ``OPENHARNESS_TIMEOUT=120`` — same failure
   mode, slightly later. Values below 120 s are now clamped with a warning.
3. ``COMPACT_TIMEOUT_SECONDS = 25`` — compaction summarizes a near-full
   context and can stream for minutes; 25 s forced the truncation
   fallback (lost context = creative damage + re-read files = slower).
   Now 300 s, env-tunable via ``OPENHARNESS_COMPACT_TIMEOUT_SECONDS``.
4. ``control.llm_call_timeout_seconds`` (600 s) existed but had no
   consumer — dead config. It is now wired as the floor for the
   OpenAI-compat client timeout everywhere the client is built.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from openharness.config.settings import Settings, _LLM_TIMEOUT_FLOOR_SECONDS
from openharness.ui.runtime import _llm_request_timeout


# --- 1. The default is a real LLM call budget ----------------------------


def test_default_llm_timeout_is_600_not_30() -> None:
    assert Settings().timeout == 600.0


def test_llm_timeout_floor_constant_is_120() -> None:
    assert _LLM_TIMEOUT_FLOOR_SECONDS == 120.0


def test_env_timeout_below_floor_is_clamped(monkeypatch, tmp_path) -> None:
    from openharness.config.settings import load_settings

    (tmp_path / "settings.json").write_text("{}")
    monkeypatch.setenv("OPENHARNESS_TIMEOUT", "42.5")
    s = load_settings(tmp_path / "settings.json")
    assert s.timeout == 120.0


def test_env_timeout_at_or_above_floor_passes_through(monkeypatch, tmp_path) -> None:
    from openharness.config.settings import load_settings

    (tmp_path / "settings.json").write_text("{}")
    monkeypatch.setenv("OPENHARNESS_TIMEOUT", "240")
    s = load_settings(tmp_path / "settings.json")
    assert s.timeout == 240.0


# --- 2. Compaction gets minutes, not seconds ----------------------------


def test_compact_timeout_default_is_300() -> None:
    from openharness.services.compact import COMPACT_TIMEOUT_SECONDS

    assert COMPACT_TIMEOUT_SECONDS >= 300.0


def test_compact_timeout_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENHARNESS_COMPACT_TIMEOUT_SECONDS", "457")
    module = importlib.reload(
        importlib.import_module("openharness.services.compact")
    )
    try:
        assert module.COMPACT_TIMEOUT_SECONDS == 457.0
    finally:
        monkeypatch.delenv("OPENHARNESS_COMPACT_TIMEOUT_SECONDS", raising=False)
        importlib.reload(
            importlib.import_module("openharness.services.compact")
        )


# --- 3. control.llm_call_timeout_seconds is wired, not dead --------------


class _Control:
    llm_call_timeout_seconds = 600


class _SettingsLike:
    control = _Control()


def test_llm_request_timeout_floors_low_values() -> None:
    assert _llm_request_timeout(_SettingsLike(), 30.0) == 600
    assert _llm_request_timeout(_SettingsLike(), None) == 600


def test_llm_request_timeout_passes_through_higher_values() -> None:
    assert _llm_request_timeout(_SettingsLike(), 900.0) == 900.0


def test_llm_request_timeout_no_control_is_noop() -> None:
    class _Bare:
        pass

    assert _llm_request_timeout(_Bare(), 30.0) == 30.0
    assert _llm_request_timeout(None, None) is None


def test_openai_compat_client_receives_floored_timeout(monkeypatch) -> None:
    """The runtime must hand the client at least the control floor."""
    from openharness.api.openai_client import OpenAICompatibleClient
    from openharness.ui import runtime

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "openharness.api.openai_client.AsyncOpenAI", _FakeClient
    )
    low = _SettingsLike()
    low.api_format = "openai"
    low.timeout = 30.0
    low.base_url = "https://example.test/v1"

    client = OpenAICompatibleClient(
        api_key="sk-test",
        base_url=low.base_url,
        timeout=runtime._llm_request_timeout(low, low.timeout),
    )
    assert captured["timeout"] == 600
    assert client is not None


# --- 4. Retry behavior stays sane (no new retry storm) -------------------


@pytest.mark.asyncio
async def test_openai_client_retry_keeps_exponential_backoff() -> None:
    """MAX_RETRIES/BASE_DELAY/MAX_DELAY unchanged: bounded, not a storm."""
    from openharness.api import openai_client

    assert openai_client.MAX_RETRIES == 3
    assert openai_client.BASE_DELAY == 1.0
    assert openai_client.MAX_DELAY == 30.0


def test_max_turns_default_supports_long_films() -> None:
    assert Settings().max_turns >= 200
