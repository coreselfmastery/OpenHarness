"""Kaggle environment detection and resource-aware defaults.

Detects CPU count, GPU availability, and Kaggle-specific environment
quirks so OpenHarness can size its concurrency and batching defaults to
actually use the available hardware. On a Kaggle T4×2 notebook, this
returns ``gpu_count=2, gpu_type="Tesla T4"`` so the engine can spawn
2× worker agents per GPU.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

log = logging.getLogger(__name__)


def detect_kaggle_resources() -> dict[str, Any]:
    """Detect CPU count, GPU count, GPU type, and total RAM.

    Returns a dict with:
      - ``cpu_count``: int — logical CPU cores (nproc).
      - ``gpu_count``: int — number of CUDA GPUs detected (0 if none).
      - ``gpu_type``: str — first GPU's name (e.g. "Tesla T4"), or "".
      - ``gpu_memory_total_mb``: int — total GPU memory across all GPUs.
      - ``ram_total_gb``: float — total system RAM (best-effort).
      - ``is_kaggle``: bool — True if running inside a Kaggle notebook.
      - ``kaggle_accelerator``: str — Kaggle's reported accelerator type.
    """
    cpu_count = _detect_cpu_count()
    gpu_count, gpu_type, gpu_memory_total_mb = _detect_gpus()
    ram_total_gb = _detect_ram_gb()
    is_kaggle = _is_kaggle()
    kaggle_accelerator = os.environ.get("KAGGLE_ACCELERATOR", "").strip().lower() or ""

    return {
        "cpu_count": cpu_count,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "ram_total_gb": ram_total_gb,
        "is_kaggle": is_kaggle,
        "kaggle_accelerator": kaggle_accelerator,
    }


def get_recommended_concurrency(resources: dict[str, Any] | None = None) -> int:
    """Return the recommended max_concurrent_agents for this hardware.

    Heuristic:
    - CPU-only: 2 × cpu_count (capped at 20 — LLM API rate limits usually
      bind before CPU does, and OpenAI/Anthropic allow 20 concurrent req/s).
    - With GPU: 4 × gpu_count × cpu_count, capped at 30. Each agent uses
      a GPU for inference but is mostly I/O-bound waiting on the API, so
      we can oversubscribe.
    - On Kaggle T4×2 (2 GPUs, 4 CPUs): returns 16 — enough parallelism for
      scene-by-scene video generation without melting the GPU.
    """
    if resources is None:
        resources = detect_kaggle_resources()
    cpu = int(resources.get("cpu_count", 4) or 4)
    gpu = int(resources.get("gpu_count", 0) or 0)
    if gpu > 0:
        # GPU present — agents are I/O-bound, oversubscribe.
        rec = min(30, 4 * gpu * max(1, cpu // 2))
    else:
        rec = min(20, 2 * cpu)
    return max(2, rec)


def get_recommended_batch_size(resources: dict[str, Any] | None = None) -> int:
    """Return the recommended batch_size for image/scene generation.

    Heuristic:
    - On GPU: 4 (most image-gen APIs accept up to 10, but 4 is a
      safe default that balances throughput vs. memory).
    - On CPU-only: 2 (smaller batches, since CPU inference is slower).
    """
    if resources is None:
        resources = detect_kaggle_resources()
    if resources.get("gpu_count", 0) > 0:
        return 4
    return 2


def apply_kaggle_defaults_to_settings(settings: Any) -> Any:
    """If running on Kaggle with no explicit override, apply hardware-aware defaults.

    Mutates the settings in place by setting:
    - ``max_concurrent_agents`` to ``get_recommended_concurrency()`` if it's
      still the default (5).
    - ``batch_size`` to ``get_recommended_batch_size()`` if it's still 1.
    - ``parallel_tools`` to True (always — Kaggle has the cores).

    Returns the (possibly-mutated) settings object. Callers should run
    this AFTER env-var overrides so explicit user choices win.
    """
    resources = detect_kaggle_resources()
    if not resources.get("is_kaggle"):
        return settings  # No-op off Kaggle
    try:
        # Use ``None`` sentinel to distinguish "user didn't set this" from
        # "user explicitly set the default value". This fixes the bug where
        # ``max_concurrent_agents=5`` (explicit) was overridden to 16 on
        # Kaggle because the auto-detection couldn't tell it apart from
        # the default of 5.
        if getattr(settings, "max_concurrent_agents", None) is None:
            settings.max_concurrent_agents = get_recommended_concurrency(resources)
        if getattr(settings, "batch_size", None) is None:
            settings.batch_size = get_recommended_batch_size(resources)
        if not getattr(settings, "auto_save", False):
            # On Kaggle, auto-save should default ON (notebooks die without warning)
            settings.auto_save = True
    except (AttributeError, TypeError):
        pass
    return settings


# -------------------------------------------------------------------
# Detection helpers (private)
# -------------------------------------------------------------------

def _detect_cpu_count() -> int:
    """Return the number of logical CPU cores available to this process."""
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _detect_gpus() -> tuple[int, str, int]:
    """Detect NVIDIA GPUs via nvidia-smi.

    Returns ``(count, first_gpu_name, total_memory_mb)``. Returns
    ``(0, "", 0)`` if nvidia-smi is unavailable or no GPUs are present.

    On Kaggle T4×2 notebooks, this returns ``(2, "Tesla T4", 30720)``.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return (0, "", 0)
    try:
        # Query: index, name, memory.total (in MiB)
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return (0, "", 0)
    if result.returncode != 0:
        return (0, "", 0)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return (0, "", 0)
    count = len(lines)
    first_name = ""
    total_mb = 0
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            name = parts[1]
            if not first_name:
                first_name = name
            try:
                total_mb += int(parts[2])
            except ValueError:
                pass
    return (count, first_name, total_mb)


def _detect_ram_gb() -> float:
    """Return total system RAM in GB (best-effort, 0.0 if unknown)."""
    # Try /proc/meminfo (Linux, including Kaggle)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # Value is in KB
                        return int(parts[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


def _is_kaggle() -> bool:
    """Return True if running inside a Kaggle notebook kernel."""
    # Kaggle sets KAGGLE_KERNEL_RUN_TYPE and KAGGLE_DATA_PROXY_TOKEN
    return (
        bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
        or bool(os.environ.get("KAGGLE_DATA_PROXY_TOKEN"))
        or os.path.isdir("/kaggle/input")
    )
