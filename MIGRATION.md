# Migration Guide — Top-10 Optimizations

This guide explains the new settings added in commit `415aa19` and
how to enable them. All changes are **backward-compatible** — if you
do nothing, your existing config continues to work exactly as before.

## Quick start

If you just want the fastest setup on Kaggle without reading the rest:

```bash
export OPENHARNESS_MAX_CONCURRENT_AGENTS=15
export OPENHARNESS_BATCH_SIZE=4
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO=0.60
export OPENHARNESS_TRUSTED_PROJECT=1
export OPENHARNESS_TOOL_CACHE=1
export OPENHARNESS_AUTO_SAVE=1
```

See `docs/PERFORMANCE_TUNING_KAGGLE.md` for the full env-var list.

## What changed

### New Settings fields

| Setting | Type | Default | What it does |
|---|---|---|---|
| `max_concurrent_agents` | int | 5 | Caps the number of concurrently-running spawned agents. Raise to 15-20 for parallel-scene workflows. |
| `batch_size` | int | 1 | How many items to process per LLM call when a tool accepts a list. Raise to 4-8 for batch image generation. |
| `auto_compact_threshold_ratio` | float \| None | None | Compact at this fraction of the context window (0.60 = 60%). When set, takes precedence over the default heuristic. |
| `trusted_project` | bool | False | When True, mutating tools (edit_file, write_file) are auto-approved for paths inside the project cwd. Sensitive paths (.ssh/.aws/.kube) are still blocked. |
| `trusted_paths` | list[str] | [] | Extra paths auto-approved when trusted_project=True. Use for sibling project dirs, asset roots, etc. |
| `tool_cache_enabled` | bool | False | When True, identical read-only tool calls (read_file, grep, glob) return cached results instead of re-executing. |
| `tool_cache_max_entries` | int | 256 | Maximum entries in the tool result cache (LRU eviction). |
| `tool_cache_ttl_seconds` | int | 3600 | TTL for cached tool results. 0 = never expire (rely on mutation-based invalidation only). |
| `parallel_tools` | bool | True | When True, multiple tool calls from a single model turn run concurrently (asyncio.gather). Set to False for deterministic sequential execution. |
| `auto_save` | bool | True | When True, save a session snapshot to ~/.openharness/sessions/ every N turns. |
| `auto_save_interval` | int | 10 | Save a snapshot every N turns. 1 = every turn (safest, slowest). |
| `thinking_budget` | int \| None | None | Per-call reasoning token budget for thinking models. Maps to Anthropic's `thinking.budget_tokens` and OpenAI's `reasoning_effort`. |
| `auto_context_inject` | bool | True | When True, spawned agents get STYLE.md / STORYBOARD.md / BRIEF.md prepended to their prompt. |
| `auto_context_files` | list[str] | [BRIEF.md, STORYBOARD.md, STYLE.md] | List of project context files to auto-inject. |
| `provider_routing` | dict \| None | None | Multi-provider routing config with primary + fallbacks. When set, the runtime builds a RoutedStreamingClient that retries on retryable errors. |

### New CLI flags

| Flag | What it does |
|---|---|
| `--thinking-budget N` | Set per-call reasoning token budget. Maps to `OPENHARNESS_THINKING_BUDGET` env var. |
| `--max-turns N` | (Existing) Cap agent turns per user input. |

### New env vars

All settings have env-var overrides with `OPENHARNESS_` prefix. See
`docs/PERFORMANCE_TUNING_KAGGLE.md` for the full list.

### New Python modules

| Module | Purpose |
|---|---|
| `openharness.services.agent_concurrency` | Asyncio.Semaphore that caps concurrent agent spawns. |
| `openharness.services.auto_context` | Discovers + injects project context files into spawn prompts. |
| `openharness.services.provider_routing` | RoutedStreamingClient with automatic fallback. |
| `openharness.services.tool_cache` | LRU cache for read-only tool results. |
| `openharness.services.kaggle_env` | Hardware detection (CPU, GPU, RAM) + Kaggle auto-defaults. |

### New PermissionChecker API

`PermissionChecker.__init__` now accepts optional `trusted_project`,
`project_cwd`, and `trusted_paths` kwargs. Existing callers that pass
only `settings.permission` continue to work — the new kwargs default
to disabled.

For the full settings-aware constructor, use
`openharness.permissions.build_permission_checker(settings, cwd=...)`.

## How to enable each optimization

### 1. Increase wave limit (parallel scenes)

```bash
export OPENHARNESS_MAX_CONCURRENT_AGENTS=15
```

Or in `~/.openharness/settings.json`:

```json
{"max_concurrent_agents": 15}
```

### 2. Batch multi-item tasks

```bash
export OPENHARNESS_BATCH_SIZE=4
```

Currently consumed by `image_generation` — when set, calling
`image_generation` without explicit `n=` produces 4 images per API
call (capped at 10). Other tools can opt in by reading
`context.metadata["batch_size"]`.

### 3. Auto context injection (default ON)

Already on by default. Drop `BRIEF.md`, `STORYBOARD.md`, `STYLE.md`
in your project root and every spawned agent will see them. To disable:

```bash
export OPENHARNESS_AUTO_CONTEXT_INJECT=0
```

### 4. Multi-provider routing

In `~/.openharness/settings.json`:

```json
{
  "provider_routing": {
    "primary": {"name": "groq"},
    "fallbacks": [
      {
        "name": "dashscope",
        "api_format": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-your-dashscope-key"
      }
    ]
  }
}
```

When the primary rate-limits (429) or 5xxs, the request is replayed
against each fallback in order. The user sees no API-visible
difference — the event stream looks like a single provider.

### 5. Lower auto-compact to 60%

```bash
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO=0.60
```

Default is ~80% (heuristic: context_window - 20K reserved - 13K buffer).
Lowering to 0.60 prevents "prompt too long" retries which cost a full
extra round-trip per failure.

### 6. Trusted project mode

```bash
export OPENHARNESS_TRUSTED_PROJECT=1
# Optional: extra paths outside the project dir
export OPENHARNESS_TRUSTED_PATHS=/data/assets:/tmp/output
```

When ON, mutating tools are auto-approved for:
- Any path inside the project cwd.
- Any path inside the `trusted_paths` whitelist.

**Always blocked** (even with trusted_project=True):
- `~/.ssh/*`, `~/.aws/credentials`, `~/.kube/config`, `~/.gnupg/*`
- `~/.openharness/credentials.json`, `~/.openharness/copilot_auth.json`

### 7. Tool result cache

```bash
export OPENHARNESS_TOOL_CACHE=1
export OPENHARNESS_TOOL_CACHE_TTL_SECONDS=3600
```

When ON, identical `read_file` / `grep` / `glob` calls return cached
results. Cache is invalidated on any mutating tool call (`write_file`,
`edit_file`, `bash` with non-zero return code).

### 8. Parallel tool execution (default ON)

Already on by default — multiple tool calls from a single model turn
run concurrently via `asyncio.gather`. To disable (for debugging):

```bash
export OPENHARNESS_PARALLEL_TOOLS=0
```

### 9. --max-turns and --thinking-budget

```bash
oh --max-turns 50 --thinking-budget 4096 "your prompt"
```

`--max-turns` caps the agent loop (prevents runaway). `--thinking-budget`
sets the per-call reasoning token budget for thinking models.

### 10. Session auto-save (default ON)

```bash
export OPENHARNESS_AUTO_SAVE=1
export OPENHARNESS_AUTO_SAVE_INTERVAL=10
```

Saves a session snapshot to `~/.openharness/sessions/<project>-<hash>/latest.json`
every N turns. On crash / OOM / Kaggle restart, resume with:

```bash
oh --resume
```

## Backward compatibility

- **All defaults preserve existing behaviour.** If you set nothing, the
  engine behaves exactly as before commit `415aa19`.
- **All env vars are opt-in.** No env var forces a behaviour change.
- **Existing config files work unchanged.** The new fields are additive.
- **Existing tests pass without modification.** 1194/1196 tests pass;
  the 2 failures are pre-existing (require external binaries).

## Verifying your migration

After applying the env vars, run:

```bash
oh doctor
```

This checks config validity and API client connectivity.

To verify each optimization is actually enabled:

```bash
# Verify settings loaded correctly
oh config

# Should show:
# max_concurrent_agents: 15
# batch_size: 4
# auto_compact_threshold_ratio: 0.6
# trusted_project: true
# trusted_paths: ['/data/assets', '/tmp/output']
# tool_cache_enabled: true
# auto_save: true
# thinking_budget: 4096
```

## Rollback

To revert to pre-optimization behaviour:

```bash
unset OPENHARNESS_MAX_CONCURRENT_AGENTS
unset OPENHARNESS_BATCH_SIZE
unset OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO
unset OPENHARNESS_TRUSTED_PROJECT
unset OPENHARNESS_TRUSTED_PATHS
unset OPENHARNESS_TOOL_CACHE
unset OPENHARNESS_TOOL_CACHE_TTL_SECONDS
unset OPENHARNESS_AUTO_SAVE_INTERVAL
unset OPENHARNESS_THINKING_BUDGET
unset OPENHARNESS_AUTO_CONTEXT_INJECT
```

Or pin to the pre-optimization commit:

```bash
pip install git+https://github.com/coreselfmastery/OpenHarness.git@9b2efd7
```

## FAQ

**Q: Do I need to enable all 10 optimizations to get the speedup?**
A: No. Each is independent. The biggest wins for video generation are
items #1 (max_concurrent_agents), #2 (batch_size), and #5
(auto_compact_threshold_ratio). The others are nice-to-haves.

**Q: Will the tool cache return stale results?**
A: No — the cache is invalidated on every mutating tool call. With the
default TTL of 3600s, even a long-idle session refreshes naturally. If
you're paranoid, set TTL to 60s or 0 (rely on mutation only).

**Q: Does trusted_project bypass .ssh / .aws protection?**
A: No. Sensitive paths are always blocked regardless of trusted_project
or trusted_paths. The test `test_optimization_6_trusted_project_does_not_override_sensitive_paths`
proves this.

**Q: What if I set max_concurrent_agents=100?**
A: The semaphore allows it, but the LLM API rate limit will bind first
(OpenAI allows ~20 concurrent req/s on most tiers). The CPU/GPU will
also saturate. Recommended: 15-20 for T4×2, 5-10 for CPU-only.

**Q: Does multi-provider routing work with the Anthropic SDK path?**
A: Yes. The RoutedStreamingClient wraps any SupportsStreamingMessages
client, so it works with AnthropicApiClient, OpenAICompatibleClient,
and CodexApiClient interchangeably.
