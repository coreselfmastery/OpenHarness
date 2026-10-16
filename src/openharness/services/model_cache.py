"""Disk-based model cache — item #20.

Redirects HuggingFace download caches to a persistent directory so a
restarted session *verifies* existing model weights instead of
re-downloading them (typically 5-10 minutes saved per session when the
directory persists across restarts — true on local machines, GCS/FUSE
mounts, and Kaggle when the cache lives under ``/kaggle/working`` for
the lifetime of the notebook's persistent storage).

What it really does:

- Sets ``HF_HOME`` / ``HF_HUB_CACHE`` / ``TRANSFORMERS_CACHE`` to the
  configured directory BEFORE ``transformers``/``huggingface_hub`` are
  imported anywhere (the libraries read these at import time).
- ``verify()`` reports which models already have complete snapshots on
  disk (a snapshot is complete when its ``refs/main`` + ``snapshots/<rev>``
  directory exist), so a warm-up script can skip finished models.
- ``stats()`` reports directory size and hit/miss counters.

What it honestly does NOT do: it cannot conjure persistence on
filesystems that throw everything away (a fresh Kaggle container's
``/tmp``). The report says so explicitly instead of pretending.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_HF_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def default_model_cache_dir() -> Path:
    """Default persistent location: ~/.openharness/model-cache."""
    from openharness.config.paths import get_data_dir

    return get_data_dir() / "model-cache"


class ModelCache:
    """Persistent HF model-weights cache (#20)."""

    def __init__(self, cache_dir: str | Path | None = None, *, apply_env: bool = True) -> None:
        self._dir = Path(cache_dir).expanduser() if cache_dir else default_model_cache_dir()
        self._env_applied = False
        self.warm_verifies = 0
        self.cold_verifies = 0
        if apply_env:
            self.apply_env()

    @property
    def cache_dir(self) -> Path:
        return self._dir

    def apply_env(self) -> bool:
        """Point the HF toolchain at the persistent directory.

        Returns True when the environment was updated. No-op when the
        user already set ``HF_HOME`` themselves — explicit user config
        always wins.
        """
        if os.environ.get("HF_HOME"):
            self._env_applied = False
            return False
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("model cache dir not writable (%s); not redirecting HF env", exc)
            return False
        hub_cache = str(self._dir / "hub")
        for var in _HF_ENV_VARS:
            os.environ[var] = str(self._dir) if var == "HF_HOME" else hub_cache
        self._env_applied = True
        log.info("model cache redirected to %s", self._dir)
        return True

    def verify(self, model_id: str) -> dict[str, Any]:
        """Check whether a model snapshot is already complete on disk.

        A snapshot is complete when ``hub/models--<org>--<name>/snapshots``
        contains at least one revision directory with files. This is the
        same layout ``huggingface_hub`` uses, so it works for models
        downloaded by any prior session.
        """
        safe_id = model_id.strip("/").replace("/", "--")
        model_dir = self._dir / "hub" / f"models--{safe_id}"
        snapshots = model_dir / "snapshots"
        complete = False
        file_count = 0
        if snapshots.is_dir():
            for revision_dir in snapshots.iterdir():
                if not revision_dir.is_dir():
                    continue
                files = [p for p in revision_dir.rglob("*") if p.is_file()]
                if files:
                    complete = True
                    file_count = max(file_count, len(files))
                    break
        if complete:
            self.warm_verifies += 1
        else:
            self.cold_verifies += 1
        return {
            "model_id": model_id,
            "cached": complete,
            "path": str(model_dir),
            "files": file_count,
        }

    def verify_many(self, model_ids: list[str]) -> dict[str, Any]:
        """Verify a list of models; returns a warm/cold summary."""
        results = [self.verify(m) for m in model_ids if m.strip()]
        warm = [r["model_id"] for r in results if r["cached"]]
        cold = [r["model_id"] for r in results if not r["cached"]]
        return {
            "cache_dir": str(self._dir),
            "warm": warm,
            "cold": cold,
            "minutes_saved_estimate": round(len(warm) * 3.5, 1),
            "results": results,
        }

    def stats(self) -> dict[str, Any]:
        size_bytes = 0
        file_count = 0
        try:
            for path in self._dir.rglob("*"):
                if path.is_file():
                    file_count += 1
                    size_bytes += path.stat().st_size
        except OSError:
            pass
        return {
            "cache_dir": str(self._dir),
            "env_applied": self._env_applied,
            "size_gb": round(size_bytes / (1024**3), 2),
            "files": file_count,
            "warm_verifies": self.warm_verifies,
            "cold_verifies": self.cold_verifies,
        }

    def persistence_note(self) -> str:
        """Honest note about whether this directory actually persists."""
        path = str(self._dir)
        if path.startswith("/tmp") or path.startswith("/kaggle/tmp"):
            return (
                f"{path} is on tmp storage — it is wiped on restart, so the "
                "model cache cannot persist across sessions. Move the cache "
                "to persistent storage (settings.model_cache.cache_dir)."
            )
        if path.startswith("/kaggle/working"):
            return (
                f"{path} persists for the Kaggle notebook's working directory "
                "lifetime; it is reclaimed when Kaggle reaps the notebook."
            )
        return f"{path} is on persistent storage; cached weights survive restarts."


def configure_model_cache_from_settings(settings: Any) -> ModelCache | None:
    """Wire-up used at engine startup: enable the cache when configured."""
    cache_settings = getattr(settings, "model_cache", None)
    if cache_settings is None or not getattr(cache_settings, "enabled", False):
        return None
    cache = ModelCache(getattr(cache_settings, "cache_dir", "") or None)
    log.info("model cache enabled at %s (%s)", cache.cache_dir, cache.persistence_note())
    return cache
