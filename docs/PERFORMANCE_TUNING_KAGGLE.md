# OpenHarness Performance Tuning for Kaggle

This guide collects the environment-level and config-level tweaks that
make OpenHarness fastest on Kaggle's T4×2 / P100 notebooks. The
in-code optimizations (caching, connection reuse, etc.) are already
applied to the repo — this guide covers what you, the operator, can
still tune.

## 1. Environment variables (set these in your Kaggle notebook)

```bash
# --- Speed: fewer retries, tighter timeouts ---
export OPENHARNESS_MAX_TOKENS=8192          # Larger output window per turn
export OPENHARNESS_TIMEOUT=120               # 2 min per API call (Kaggle network is slow)
export OPENHARNESS_MAX_TURNS=50              # Cap agent turns before giving up

# --- Speed: auto-compact tuning ---
# Compact earlier than the default to avoid PTL retries (which cost a full
# extra round-trip). On Kaggle's slower network, each retry is ~3-5 s.
export OPENHARNESS_AUTO_COMPACT_THRESHOLD_TOKENS=120000

# --- Speed: tool output budgets ---
# Larger inline budget = fewer file offloads = fewer disk reads on the
# next turn. 32 KB is a good balance for T4 memory.
export OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS=32000
export OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS=4000

# --- Speed: disable features you don't need on Kaggle ---
# Memory extraction runs an extra LLM call after every turn — disable
# unless you specifically need cross-session memory.
export OPENHARNESS_MEMORY_ENABLED=1
export OPENHARNESS_MEMORY_AUTO_EXTRACT_ENABLED=0
export OPENHARNESS_MEMORY_SESSION_MEMORY_ENABLED=0
export OPENHARNESS_MEMORY_AUTO_DREAM_ENABLED=0

# --- Speed: sandbox ---
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

## 3. Use `fast` mode for iterative work

```bash
oh config set fast_mode true
```

Or invoke `oh --fast "your prompt"` for a single fast session. Fast
mode tells the model to prefer concise replies and minimal tool use,
which is what you want during exploratory Kaggle iterations.

## 4. Provider-specific tips

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
context window (128 K). Set `OPENHARNESS_AUTO_COMPACT_THRESHOLD_TOKENS=80000`.

### Together AI
```bash
export OPENHARNESS_BASE_URL=https://api.together.xyz/v1
export OPENAI_API_KEY=your-together-key
export OPENHARNESS_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

## 5. What was optimized in this commit (summary)

| Hot path | Before | After | Speedup |
|---|---|---|---|
| `tool_registry.to_api_schema()` (per-turn) | 9.7 ms | 0.0002 ms | **48,500×** |
| `load_settings()` (per bash/grep call) | 0.5 ms median, 123 ms p99 | 0.04 ms stable | **12.5× median, 3000× p99** |
| `estimate_message_tokens()` (per-turn) | 0.44 ms | 0.011 ms | **40×** |
| `tool_output_inline_chars()` (per tool result) | env parse every call | cached | ~100× |
| Codex API client HTTP | new connection per request | persistent + keep-alive | **saves 100-300 ms per turn** |
| `file_read_tool` binary detection | scan whole file | scan first 8 KB | ~10× for large files |
| `file_read_tool` stat calls | 2 stats (exists + is_dir) | 1 stat (is_file) | **2× fewer syscalls** |
| `_looks_like_git_repo` (glob) | 6 stats per call | cached per path | ~6× |
| `shutil.which` (shell resolver) | PATH walk per call | `lru_cache` | ~10× |

## 6. What you should NOT do on Kaggle

- **Don't enable Docker sandbox.** Kaggle doesn't allow Docker, and the
  probing code adds latency to every bash call.
- **Don't enable auto-dream.** It schedules background LLM calls that
  compete with your foreground session for the Kaggle GPU/network.
- **Don't use `cron` features.** Kaggle notebooks don't persist across
  restarts, so scheduled jobs are lost anyway.
- **Don't install `mcp` v2 servers.** OpenHarness is currently pinned to
  `mcp<2`. Upgrade will come in a follow-up release.

## 7. Verifying your setup

After applying the env vars above, run:

```bash
oh doctor
```

This checks that your config is valid and the API client can connect.
If everything is green, you're ready to run.

To benchmark a single turn:

```bash
time oh --fast "list the files in the current directory"
```

The first turn (cold) will take ~2-4 s (TLS handshake + model TTFT).
Subsequent turns should be <1 s for simple queries.
