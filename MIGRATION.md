# Migration Guide — Top-10 Optimizations

This guide explains the new settings added in commit `415aa19` and
how to enable them. All changes are **backward-compatible** — if you
do nothing, your existing config continues to work exactly as before.

> **Wave 2-6 update (items #1-#50):** a second feature surface was
> added later — the inference / render / QA / control-plane pipelines.
> It has its own section at the bottom:
> [Waves 2-6: inference, render & control plane](#waves-2-6-inference-render--control-plane-items-1-50).
>
> **Speed wave 2 update:** the async batch executor + cost dashboard
> (and the seven-bug fix wave that shipped with them) are documented in
> [Speed wave 2: batch executor & cost dashboard](#speed-wave-2-batch-executor--cost-dashboard).

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
| `qa.parallel_waves` | bool | True | Review multiple scenes per QA round. Env: `OPENHARNESS_QA_PARALLEL`. |
| `qa.wave_size` | int | 5 | Scenes per parallel QA wave. Env: `OPENHARNESS_QA_WAVE_SIZE`. |
| `qa.review_batching` | bool | True | Batch several scenes into ONE QA LLM call (~10x fewer calls). Env: `OPENHARNESS_QA_BATCHING`. |
| `qa.model` / `qa.api_key` / `qa.base_url` | str | "" | Dedicated QA reviewer endpoint; falls back to the scene-gen config when empty. Env: `OPENHARNESS_QA_MODEL` / `OPENHARNESS_QA_API_KEY` / `OPENHARNESS_QA_BASE_URL`. |
| `rendering.render_timeout_seconds` | int | 7200 | Per-render-job hard timeout (Blender hero shots legitimately run 1-2h on a T4). Env: `OPENHARNESS_RENDER_TIMEOUT_SECONDS`. |
| `control.watchdog_timeout_seconds` | int | 1350 | No-progress kill switch window; auto-raised to >= 1.5x the tool timeout. Running renders feed the watchdog. Env: `OPENHARNESS_WATCHDOG_TIMEOUT_SECONDS`. |

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
| `openharness.services.provider_routing` | RoutedStreamingClient with automatic fallback + the shared `build_fallback_route()` config parser. |
| `openharness.services.tool_cache` | LRU cache for read-only tool results (thundering-herd safe, generation-guarded invalidation). |
| `openharness.services.kaggle_env` | Hardware detection (CPU, GPU, RAM) + Kaggle auto-defaults. |

### New tools

| Tool | What it really does |
|---|---|
| `batch_scene` | Generates code for 5-10 scenes in ONE LLM call (structured BATCH_ID/SCENES response, per-scene lint via `node --check` / `py_compile`, per-scene retry). Scene filenames are validated — no path separators or `..` — so LLM output cannot escape the output directory. |
| `brand_kit` | Loads `brand.yaml` from the project root; validates scene code against brand anchors (colors, fonts, min type size, safe area), generates a *seeded* creative drift (same project = same drift), and scores brand compliance + creative originality. Brand anchors are also auto-injected into the system prompt when `brand.yaml` exists. |
| `cost_governor` | In-memory budget tracker: `check` before each LLM call, `record` after, `status`, `reset`. |
| `audio_qa` | REAL measurements via ffmpeg: true-peak clipping (dBTP via `loudnorm`), integrated loudness (LUFS vs target), silence gaps (`silencedetect`), and a noise-floor probe. Requires ffmpeg on PATH; says so (and fails) when it can't measure. |
| `determinism_checker` | Static analysis of scene code for non-determinism sources (`Math.random`, `Date.now`, `performance.now`, `uuid4`, ...) with line numbers. Honest about being static — it does not render anything. |
| `contact_sheet` | Composites a REAL thumbnail grid PNG via ffmpeg (scale + pad + `xstack`); videos contribute their first frame. Writes a manifest alongside; when ffmpeg or decodable inputs are missing it writes the manifest only and says so. |
| `engine_router` | Keyword routing to the best rendering engine (hyperframes / manim / blender / ffmpeg). |

### New PermissionChecker API

`PermissionChecker.__init__` now accepts optional `trusted_project`,
`project_cwd`, and `trusted_paths` kwargs. Existing callers that pass
only `settings.permission` continue to work — the new kwargs default
to disabled.

For the full settings-aware constructor, use
`openharness.permissions.build_permission_checker(settings, cwd=...)`.

Trusted-path checks resolve symlinks and `..` segments first
(`os.path.realpath`), so a path like `<project>/../../etc/passwd` can
never pass as "inside the project".

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

---

# Waves 2-6: inference, render & control plane (items #1-#50)

The second feature surface is organized into five settings groups —
`inference`, `rendering`, `qa`, `control`, and `model_cache` — plus a
persistent SQLite store (`~/.openharness/cache/store.sqlite3`) shared by
the caches. **Everything except prefix caching is opt-in**, and every
feature degrades to identity when disabled, so an unchanged config
behaves exactly as before.

## Inference (`settings.inference`)

| Feature | Default | Env var | What it really does |
|---|---|---|---|
| `prefix_caching` | **on** | `OPENHARNESS_PREFIX_CACHING` | Adds Anthropic `cache_control` breakpoints on the system prompt and tool list so repeated prefills hit the provider KV cache (30-50% cheaper prefill on Anthropic; harmless no-op on OpenAI-format providers, which cache server-side automatically). Cache read/write tokens are reported in `UsageSnapshot`. |
| `prompt_compression` | off | `OPENHARNESS_PROMPT_COMPRESSION` | Strips redundant content (old tool-output bodies, repeated whitespace) from requests above `min_chars`, keeping the last `keep_turns` verbatim. Lossless for recent context. |
| `semantic_cache` | off | `OPENHARNESS_SEMANTIC_CACHE` | MinHash/Jaccard response cache with SQLite persistence — near-duplicate prompts above `threshold` (default 0.88) return the cached response. |
| `cross_provider_racing` | off | `OPENHARNESS_CROSS_PROVIDER_RACING` | Races 2-3 providers concurrently and keeps the first response. Doubles token cost on raced calls, hence opt-in. |
| `api_key_pool` | `[]` | `OPENHARNESS_API_KEY_POOL` (comma-separated) | Rotates a pool of API keys per request and on 429s. Groq alias: `OPENHARNESS_GROQ_API_KEYS`. |

Items #1/#5 (speculative decoding, multi-token prediction) are model
*server* features — the client only probes and reports capability
honestly (`server_config` tool); it does not pretend to accelerate
anything.

## Rendering (`settings.rendering`)

| Feature | Default | Env var | What it really does |
|---|---|---|---|
| `parallel_rendering` / `render_workers` | on / auto | `OPENHARNESS_PARALLEL_RENDERING` / `OPENHARNESS_RENDER_WORKERS` | Renders scenes across N workers (auto = GPU count, else 1). |
| `draft_first` + `resolution_tiers` | on | `OPENHARNESS_DRAFT_FIRST` | Renders a cheap 640x360 draft first, QA-reviews it, and only renders 1080p for approved scenes. |
| `quality_early_exit` | on | — | Promotes a passing draft straight to final (skips the full render) when its QA score clears `early_exit_score_threshold` (0.9). |
| `render_cache` | on | `OPENHARNESS_RENDER_CACHE` | Scene-level cache keyed by a SHA-256 fingerprint of scene code + assets + engine args + tier; persisted across sessions. |
| `nvenc` | on | `OPENHARNESS_NVENC` | Prefers `h264_nvenc` **only after a real probe encode succeeds**; otherwise falls back to libx264. |
| `auto_enqueue_renders` | off | `OPENHARNESS_AUTO_ENQUEUE_RENDERS` | `batch_scene` enqueues renders while generation continues (pipeline overlap). |
| `dependency_graph` | on | — | Honors `depends_on` scene ordering; independent scenes render in parallel waves with work stealing (#24). |

## Control plane (`settings.control`)

All off by default; `OPENHARNESS_CONTROL=1` enables the group.

| Item | Feature | What it really does |
|---|---|---|
| #26 | Kill switch | `control` tool or bus trip; the query loop aborts cleanly at the next turn boundary. |
| #27 | Stage timeouts | Per-tool deadline (`tool_timeout_seconds`, default 900s) enforced via `bus.call_with_limit`; tools that manage their own deadlines (ask_user_question, render_scenes, ...) are exempt. |
| #28 | Cost budget | Session cost tracker with warn/halt thresholds (`cost_budget_usd`). |
| #29 | Audit log | Append-only JSONL of every LLM/tool/render/QA/control event under `~/.openharness/audit/`. |
| #30 | Pause gate | `control pause` blocks the loop at the next turn boundary until resumed. |
| #31 | File locking | Cross-agent read-modify-write locks for `edit_file`/`write_file` (lock files under `~/.openharness/locks/`). |
| #33 | State repair | Saved sessions are validated and repaired on resume (truncated tool calls, orphaned tool_use blocks). |
| #35 | Injected messages | Human messages can be injected mid-run; they surface at the next turn boundary. |
| #36 | Watchdog | Stage-based progress signals; a stuck stage (default 600s) trips a diagnostic. |
| #37 | Checkpoints | Full conversation snapshots every N turns (`checkpoint_interval_turns`, default 10) with `checkpoint rollback`. |

## New tools

| Tool | Purpose |
|---|---|
| `render_scenes` | Submit scene files to the render pipeline (tier, workers, cache-aware). |
| `render_queue_status` | Live queue/worker/state view of the render pipeline. |
| `qa_review` | QA-gate scenes (per-scene verdicts; parallel waves, batching). |
| `control` | Kill / pause / resume / status of the control bus. |
| `checkpoint` | List / create / rollback conversation checkpoints. |
| `progress` | Whole-pipeline scene progress (pending → generating → … → approved/failed). |
| `audio_gen` | Real TTS through any OpenAI-compatible `/audio/speech` endpoint. |
| `server_config` | Reports model-server acceleration capabilities honestly (probes, doesn't fake). |
| `cache_stats` | Size/hit-rate status and clearing for the persistent caches. |

## Backward compatibility

- All new settings groups default to no-op except `inference.prefix_caching`
  (a pure win on Anthropic, invisible elsewhere).
- The query loop, tools and runtime behave identically when the control
  bus is absent (`None`) — verified by the 1423-test suite.
- The SQLite store and audit/checkpoint/lock files are created lazily
  under `~/.openharness/` only when their feature is used.

---

# Speed wave 2: batch executor & cost dashboard

Two new performance surfaces shipped after waves 2-6, plus a wave of
bug fixes (the seven reported bugs and a deeper audit — see CHANGELOG
for the full list). Everything here degrades to identity when unused.

## Async batch executor (`services.batch_executor`)

Bounded-concurrency fan-out for LLM work: up to
`inference.batch_concurrency` (default 8, env
`OPENHARNESS_BATCH_CONCURRENCY`) requests in flight at once behind a
semaphore. Results are order-preserving; one failing item never cancels
the others (its slot carries an `ItemFailure`); retry rounds re-run
only the failures. It powers:

- **`parallel_prompts` tool** — run N independent prompts concurrently
  through a shared pooled HTTP client. Requires a configured endpoint
  (scene_gen / QA / primary model settings); without one it refuses
  honestly instead of inventing results.
- **`batch_scene` concurrent retries** — scenes that fail their first
  lint get their retry LLM calls fanned out concurrently (wall time ≈
  the slowest retry, not the sum).

## Cost dashboard (`services.cost_dashboard`)

Every completed LLM response is priced and aggregated — always on, pure
observability (in-memory counters, SQLite flushed every 25 records /
30 s).

| Aspect | Detail |
|---|---|
| Pricing | Longest-prefix per-model table (USD per 1M tokens, in/out), Anthropic cache multipliers (read 0.1x, write 1.25x input). |
| Overrides | `OPENHARNESS_PRICING_JSON` — a JSON object mapping model prefix → `[input, output]`. |
| Unknown models | Priced at $0 and counted as "unpriced" in reports — totals never silently lie. The session budget halt (`cost_budget_usd`) prices them conservatively ($3/$15 per 1M) so the safety control still trips. |
| Persistence | Aggregates survive restarts in the shared SQLite store. |
| Querying | `cost_report` tool: `summary` / `models` / `scenes` / `recent` / `clear`, plus projection when you pass `scenes_total` + `scenes_done`. |

## Settings added

| Field | Default | Env var |
|---|---|---|
| `inference.batch_concurrency` | 8 | `OPENHARNESS_BATCH_CONCURRENCY` |
| `inference.key_pool_cooldown_seconds` | 60 | `OPENHARNESS_KEY_POOL_COOLDOWN_SECONDS` |

Key-pool behavior note: a rate-limited key now enters a cooldown and is
skipped until it expires; every key in the pool is tried before the
error surfaces, and a short full-pool stall (≤5 s to earliest recovery)
is waited out. Provider-routing fallbacks may declare their own `model`
in the `provider_routing` config so a fallback endpoint is never asked
for the primary's model name.
