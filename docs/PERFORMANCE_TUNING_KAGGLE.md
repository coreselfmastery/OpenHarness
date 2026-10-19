# OpenHarness Performance Tuning for Kaggle

This guide collects the environment-level and config-level tweaks that
make OpenHarness fastest on Kaggle's T4×2 / P100 notebooks. The
in-code optimizations (caching, connection reuse, etc.) are already
applied to the repo — this guide covers what you, the operator, can
still tune.

## 0. Auto-detection (no config needed)

On Kaggle, OpenHarness auto-detects the hardware and applies sensible
defaults for `max_concurrent_agents` and `batch_size` if you haven't
explicitly set them. On a T4×2 notebook (2 GPUs, 4 CPUs) this means:

- `max_concurrent_agents` → 16 (4 × 2 GPUs × 2 half-CPU groups)
- `batch_size` → 4 (one image-gen call produces 4 images)
- `auto_save` → True (notebooks die without warning)

You only need to set the env vars below if you want to override these
defaults or enable opt-in features.

## 1. Environment variables (set these in your Kaggle notebook)

### Top-10 optimization env vars

```bash
# Item 1: Max concurrent agents (default 5; raise to 15-20 for parallel scenes)
export OPENHARNESS_MAX_CONCURRENT_AGENTS=15

# Item 2: Batch size (default 1; raise to 4-8 for batch image/scene gen)
export OPENHARNESS_BATCH_SIZE=4

# Item 3: Auto-context injection (default ON; set to 0 to disable)
# When ON, every spawned agent gets STYLE.md / STORYBOARD.md / BRIEF.md
# prepended to its prompt automatically.
export OPENHARNESS_AUTO_CONTEXT_INJECT=1

# Item 5: Auto-compact at 60% of context window (default ~80%)
# Lowering prevents "prompt too long" retries that cost a full extra
# round-trip per failure.
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO=0.60

# Item 6: Trusted-project mode (default OFF)
# When ON, mutating tools (edit_file, write_file) are auto-approved for
# paths inside the project cwd. Sensitive paths (.ssh/.aws/.kube) are
# still blocked. Set to 0 for interactive confirmation prompts.
export OPENHARNESS_TRUSTED_PROJECT=1

# Item 7: Tool result cache (default OFF)
# When ON, identical read_file / grep / glob calls return cached results
# instead of re-executing. Cache is invalidated on any mutating tool.
export OPENHARNESS_TOOL_CACHE=1

# Item 7 cont: Tool cache TTL in seconds (default 3600 = 1 hour)
# Set to 0 to disable time-based expiry (rely on mutation only).
export OPENHARNESS_TOOL_CACHE_TTL_SECONDS=3600

# Item 8: Parallel tool execution (default ON)
# When ON, multiple tool calls from a single model turn run concurrently.
# Set to 0 for deterministic sequential execution (debugging).
export OPENHARNESS_PARALLEL_TOOLS=1

# Item 9: Thinking budget (default unset)
# Per-call reasoning token budget for thinking models. Maps to
# Anthropic's `thinking.budget_tokens` for Claude 3.5+ and to
# OpenAI's `reasoning_effort` (low/medium/high) for o1/o3/gpt-5.
export OPENHARNESS_THINKING_BUDGET=4096

# Item 10: Session auto-save (default ON)
# When ON, a session snapshot is written to ~/.openharness/sessions/
# every N turns. On crash / OOM / Kaggle restart, you can resume with
# `oh --resume`.
export OPENHARNESS_AUTO_SAVE=1

# Item 10 cont: Auto-save interval (default 10)
# Save every N turns. 1 = save every turn (safest, slowest).
# 10 = save every 10 turns (good balance for long runs).
export OPENHARNESS_AUTO_SAVE_INTERVAL=10
```

### Video QA + render pipeline (Kaggle-specific)

```bash
# --- QA funnel (items #7-#10) ------------------------------------------
# Parallel review waves. 0 = one scene at a time (debug only — slow).
export OPENHARNESS_QA_PARALLEL=1
# Scenes reviewed per wave. 5 is a good default on Kaggle; larger waves
# hit the LLM harder per call but finish the funnel in fewer rounds.
export OPENHARNESS_QA_WAVE_SIZE=5
# Batch several scenes into ONE QA LLM call (~10x fewer calls).
export OPENHARNESS_QA_BATCHING=1
# Dedicated QA model (defaults to the scene-gen model when empty).
export OPENHARNESS_QA_MODEL=
export OPENHARNESS_QA_API_KEY=
export OPENHARNESS_QA_BASE_URL=

# --- Render pipeline (items #11-#17) ------------------------------------
# Per-render-job hard timeout. Default 7200s: a Blender hero shot on a
# T4 (128spp, ~10-40 s/frame) legitimately runs 1-2h. Raise it for
# longer hero sequences; the engines' own per-frame caps still bound
# stuck frames.
export OPENHARNESS_RENDER_TIMEOUT_SECONDS=7200
# Kill switch watchdog: no observable progress for this many seconds
# trips the kill. Auto-raised to >= 1.5x the tool timeout; running
# renders also feed the watchdog (a healthy long render is progress).
export OPENHARNESS_WATCHDOG_TIMEOUT_SECONDS=1350
```

### Trusted paths whitelist (optional)

```bash
# Item 8: Extra paths auto-approved when trusted_project=1.
# Colon-separated list. Project cwd is always trusted; this adds
# sibling dirs like /data/assets or /tmp/output.
export OPENHARNESS_TRUSTED_PATHS=/data/assets:/tmp/output:/var/cache
```

### Speed: no premature timeouts (the biggest hidden time sink)

```bash
export OPENHARNESS_MAX_TOKENS=8192          # Larger output window per turn
# LLM request timeout. Default is 600 s — do NOT shrink it to "make things
# faster": big-prompt prefills (100k+ token context with STYLE.md /
# STORYBOARD.md injected) sit silent for 30-90 s before the first byte.
# A 30-120 s cap kills the call mid-flight and the auto-retry storm
# (4 attempts, each repaying connect + prefill) wastes minutes per scene.
# Values below 120 s are now clamped with a warning — for good reason.
export OPENHARNESS_TIMEOUT=600
# Long-film runs need headroom: 50 turns dies mid-pipeline on a 20-min
# video (it looks like a hang — nothing progresses, no error).
export OPENHARNESS_MAX_TURNS=200

# Compaction (context summarization) can stream for minutes on slow
# providers; the old 25 s cap forced truncation fallbacks.
export OPENHARNESS_COMPACT_TIMEOUT_SECONDS=300
```

### Speed: auto-compact absolute override

```bash
# Alternative to OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO: explicit token count.
# When set, takes precedence over the ratio.
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_TOKENS=120000
```

### Speed: tool output budgets

```bash
# Larger inline budget = fewer file offloads = fewer disk reads on the
# next turn. 32 KB is a good balance for T4 memory.
export OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS=32000
export OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS=4000
```

### Speed: disable features you don't need on Kaggle

```bash
# Memory extraction runs an extra LLM call after every turn — disable
# unless you specifically need cross-session memory.
export OPENHARNESS_MEMORY_ENABLED=1
export OPENHARNESS_MEMORY_AUTO_EXTRACT_ENABLED=0
export OPENHARNESS_MEMORY_SESSION_MEMORY_ENABLED=0
export OPENHARNESS_MEMORY_AUTO_DREAM_ENABLED=0

# Docker sandbox is not available on Kaggle — disable it so we don't
# waste time probing for Docker on every bash call.
export OPENHARNESS_SANDBOX_ENABLED=0
```

## 2. Use the OpenAI-compatible path when possible

If your model is served via an OpenAI-compatible endpoint (DashScope,
Together, Groq, Cerebras, etc.), set:

```bash
export OPENHARNESS_PROVIDER=openai_compat
export OPENHARNESS_API_FORMAT=openai
export OPENHARNESS_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENHARNESS_MODEL=qwen-max
```

The OpenAI-compatible client has lower per-turn overhead than the
Anthropic SDK path (smaller payload, no beta headers, no OAuth
attribution stitching).

## 3. Multi-provider routing (Item 4)

Configure in `~/.openharness/settings.json`:

```json
{
  "provider_routing": {
    "primary": {"name": "groq"},
    "fallbacks": [
      {
        "name": "dashscope",
        "api_format": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-your-dashscope-key",
        "delay_seconds": 0.0
      },
      {
        "name": "together",
        "api_format": "openai",
        "base_url": "https://api.together.xyz/v1",
        "api_key": "your-together-key"
      }
    ]
  }
}
```

When the primary rate-limits or 5xxs, the request is automatically
replayed against each fallback in order.

## 4. Use `fast` mode for iterative work

```bash
oh config set fast_mode true
```

Or invoke `oh --fast "your prompt"` for a single fast session. Fast
mode tells the model to prefer concise replies and minimal tool use,
which is what you want during exploratory Kaggle iterations.

## 5. CLI flags

```bash
# Cap agent turns (prevents runaway loops)
oh --max-turns 50 "your prompt"

# Set per-call reasoning budget for thinking models
oh --thinking-budget 4096 "your prompt"

# Both together
oh --max-turns 50 --thinking-budget 4096 -p "your prompt"
```

## 6. Provider-specific tips

### DashScope (Alibaba Qwen)
```bash
export OPENHARNESS_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=sk-your-dashscope-key
export OPENHARNESS_MODEL=qwen-max
```
DashScope has ~80 ms RTT from Kaggle's US-West machines — one of the
fastest providers available.

### Groq (Llama 3.3 70B)
```bash
export OPENHARNESS_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_API_KEY=gsk_your-groq-key
export OPENHARNESS_MODEL=llama-3.3-70b-versatile
```
Groq has the lowest latency of any provider (~50 ms TTFT) but a small
context window (128 K). Set `OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO=0.60`.

### Together AI
```bash
export OPENHARNESS_BASE_URL=https://api.together.xyz/v1
export OPENAI_API_KEY=your-together-key
export OPENHARNESS_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

## 7. Recommended Kaggle env-var block (copy-paste ready)

```bash
# === OpenHarness Kaggle optimizations ===
export OPENHARNESS_MAX_CONCURRENT_AGENTS=15
export OPENHARNESS_BATCH_SIZE=4
export OPENHARNESS_AUTO_CONTEXT_INJECT=1
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_RATIO=0.60
export OPENHARNESS_TRUSTED_PROJECT=1
export OPENHARNESS_TRUSTED_PATHS=/kaggle/working:/kaggle/input
export OPENHARNESS_TOOL_CACHE=1
export OPENHARNESS_TOOL_CACHE_TTL_SECONDS=3600
export OPENHARNESS_PARALLEL_TOOLS=1
export OPENHARNESS_AUTO_SAVE=1
export OPENHARNESS_AUTO_SAVE_INTERVAL=10
export OPENHARNESS_THINKING_BUDGET=4096
export OPENHARNESS_SANDBOX_ENABLED=0
export OPENHARNESS_MAX_TOKENS=8192
export OPENHARNESS_TIMEOUT=600
export OPENHARNESS_MAX_TURNS=200
export OPENHARNESS_COMPACT_TIMEOUT_SECONDS=300
```

## 8. What was optimized in the code (summary)

| Hot path | Before | After | Speedup |
|---|---:|---:|---:|
| `tool_registry.to_api_schema()` (per-turn) | 9.7 ms | 0.0002 ms | **48,500×** |
| `load_settings()` (per bash/grep call) | 0.5 ms / 123 ms p99 | 0.04 ms stable | **12.5× / 2,000× p99** |
| `estimate_message_tokens()` (per-turn) | 0.44 ms | 0.011 ms | **40×** |
| Codex HTTP per turn | new TLS handshake | keep-alive | **saves 100-300 ms/turn** |
| Tool cache (cache hit) | full re-execution | dict lookup | **variable, up to 100×** |
| Trusted project (edit_file) | user prompt | auto-approve | **saves 5-30 s/turn** |

## 9. What you should NOT do on Kaggle

- **Don't enable Docker sandbox.** Kaggle doesn't allow Docker.
- **Don't enable auto-dream.** It schedules background LLM calls.
- **Don't use `cron` features.** Kaggle notebooks don't persist.
- **Don't install `mcp` v2 servers.** OpenHarness is pinned to `mcp<2`.

## 10. Verifying your setup

After applying the env vars above, run:

```bash
oh doctor
```

To benchmark a single turn:

```bash
time oh --fast "list the files in the current directory"
```

The first turn (cold) will take ~2-4 s. Subsequent turns should be <1 s.
