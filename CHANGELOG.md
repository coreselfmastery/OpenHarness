# Changelog

All notable changes to OpenHarness should be recorded in this file.

The format is based on Keep a Changelog, and this project currently tracks changes in a lightweight, repository-oriented way.

## [Unreleased]

### Added

- **Video QA stack, wave 4 — a four-layer QA funnel plus learning stores** (8 new tools; 67 registered total):
  - **`scene_lint`** (`services.static_scene_analyzer`) — structural analysis of scene code BEFORE rendering: Python scenes parsed with the real `ast` (JS/TS via calibrated regexes); flags undeclared asset references (hard fail — the render would crash), off-safe-area positions, font sizes below the project minimum (scaled by frame height), hardcoded colors outside the brand palette, repeated magic constants (drift risk), and non-deterministic calls (`Math.random()`, `Date.now()`, `random.random()`, `time.time()`, ...) that break re-render determinism. Accepts an assets manifest (JSON list or `{"assets": [...]}`) and the same palette files as `pixel_qa`. Catching errors at the code level is ~100x cheaper than after rendering.
  - **`frame_diff`** (`services.frame_diff`) — motion-quality QA: SSIM (Wang et al., implemented on `cv2.GaussianBlur` — no SciPy) between consecutive sampled frames plus Farneback dense optical flow. Flags frozen motion (identical pair + zero flow), mid-scene stutter (runs of near-identical pairs), jump cuts/flashes (sharp SSIM drops), and animations running outside the expected speed window (`expected_min/max_flow_px_s`). Integrates with the progress tracker (`qa` stage, hard failures → `failed`).
  - **`vlm_qa`** (`tools.vlm_qa_tool`) — selective vision-language rubric review: samples 5-10 frames per scene, sends ONE multi-image request to the local MiniCPM-V 4.5 endpoint (shares `settings.minicpm_v`) with a structured rubric (composition / readability / brand_match / anomaly, 0-10 each), parses the JSON scores robustly (tolerates code fences and prose), aggregates per scene, and returns a pass/fail verdict against a configurable threshold. Records usage in the cost dashboard; per-scene error isolation (one unreachable endpoint fails the scene, not the run).
  - **`style_consistency`** (`services.style_consistency`) — extracts the style DNA from every scene's code (hex colors, font families/sizes, cubic-bezier easings, transition durations) and checks it against the project reference (brand.yaml / tokens.json / STYLE.md / inline values) AND across scenes: off-palette colors, foreign font families, non-standard motion curves, and cross-scene drift (token standardized in only some scenes) are flagged with the scene and the offending token before the drift becomes visible on screen.
  - **`contact_sheet` upgrade** — cells are now labeled `S<n> <t>s` (scene number + final-cut start timestamp) via ffmpeg `drawtext` (auto font discovery, graceful skip when no system TTF exists); timestamps default to cumulative video durations and can be given explicitly; the JSON manifest now maps every grid position to its file, scene number, and timestamp.
  - **`semantic_memory`** (`services.semantic_memory`) — semantic memory store: SQLite with an embedding column; `remember` facts/decisions, `recall` later by MEANING via cosine-similarity top-K (project-scoped). Embeddings from any OpenAI-compatible `/embeddings` endpoint (`settings.embedding`, `OPENHARNESS_EMBEDDING_*`; BGE-M3 / MiniLM recommended) with a deterministic offline hashing-embedder fallback (word/character n-gram vectors, 256 dims, zero dependencies). Backend and dimension recorded per entry; vectors from different embedding spaces are never compared.
  - **`style_memory`** (`services.style_memory`) — cross-project style-pattern store: record a finished project's style DNA (palette, fonts, motion curves, transitions + final quality score; palette/fonts auto-extractable from `tokens.json`), then `suggest` ranked styles that actually worked for a content type when the next project starts. Upsert keyed by (project, style).
  - **`failure_patterns`** (`services.failure_patterns`) — failure-rate database per scene type: every QA outcome is logged with scene type, reason, and prompt id; `rates` aggregates failure rate + top reasons; `hint` (call before generating) returns an explicit stricter-prompt recommendation for high-risk scene types, built from the exact historical failure reasons.
  - **`prompt_tracker`** (`services.prompt_tracker`) — prompt-effectiveness measurement: versioned prompt registry (template + tags, upsertable), per-use pass/fail results, per-version success rates, automatic flagging of low performers (< 50% over 5+ uses), `best` for the winning version, and retirement.
  - **`EmbeddingConfig` settings group** (+ `OPENHARNESS_EMBEDDING_*` env vars) for the semantic-memory embedding endpoint; `load_brand_palette()`/`palette_from_mapping()` gained `as_hex=True` for code-level color checks.
  - All learning stores are plain SQLite under `<data_dir>/video/*.sqlite3` (WAL, thread-locked, graceful degradation to empty results when unwritable) on a shared `services._sqlite_kit` base.
  - `docs/VIDEO_HARNESS.md` rewritten around the full QA funnel (code → pixel → motion → style → VLM → LLM → human contact sheet) and the learning-store loop.
- **Pixel QA tool** (`pixel_qa`) — deterministic, offline OpenCV inspection of rendered video: black/white/blank-frame detection, frozen-frame detection, brand-palette color drift (reads `brand.yaml`/`brand.json` or an OpenVideoHarness `styles/<slug>/tokens.json` palette), text-too-small/off-screen risk via Canny edge + bounding-box clustering, and encoding-artifact/noise analysis (macroblock-boundary ratio + Laplacian variance). Every check returns pass/fail with a confidence; results aggregate per scene, mark the progress tracker (`rendered -> qa`, hard failures `-> failed`), and can export a JSON report. Optional dependency: `openharness-ai[pixel-qa]` (`opencv-python-headless`).
- **MiniCPM-V 4.5 VLM tool** (`minicpm_vlm`) — local vision-language analysis through an OpenAI-compatible endpoint (vLLM / llama.cpp / Ollama). MiniCPM-V 4.5 is an 8B multimodal model whose 4-bit quantization fits in 6-8 GB VRAM — one consumer GPU or a Kaggle T4. Supports multiple images, video input (evenly-spaced frame extraction via OpenCV), a `check_server` probe that prints ready-to-run launch commands, usage recording into the cost dashboard, and configuration via `settings.minicpm_v` or `OPENHARNESS_MINICPM_*` env vars.
- `docs/VIDEO_HARNESS.md` — how OpenHarness's pipeline tools (render, pixel QA, LLM QA, VLM analysis, audio, progress, checkpoints) connect to the OpenVideoHarness workflow (brief/storyboard/gates, `styles/*/tokens.json` brand palettes, scene scripts, review loops).
- `opencv-python-headless` optional extra (`pixel-qa`).

### Fixed

- **`pixel_qa` crashed when the `checks` argument was passed** — the tool referenced `arguments.checkes` (typo) instead of `arguments.checks`, so selecting a subset of checks raised `AttributeError`. Regression test added (`test_checks_subset_argument_works`).
- Six re-reported bugs where the previous fixes did not hold, plus deeper issues found in the same audit:
  1. **`parallel_prompts` returned no usable output** — completions went into `metadata["results"]` while `output` only listed failures, so the agent never saw any answer. All results (successes with their labels and failures with errors) now appear in `output`; metadata stays for programmatic callers.
  2. **Semantic cache ignored tool results and the model** — `_request_text` read only `TextBlock`s, so consecutive agent-loop turns with identical user text but different tool results produced identical fingerprints (guaranteed false hits), and `lookup()` never compared `request.model` against the stored entry (switching models replayed another model's answers). Tool-result bodies now feed the fingerprint (capped per result) and both the stable key and the lookup scan are model-scoped.
  3. **Checkpoint restore was still overwritten** — `submit_message`'s `AssistantTurnComplete` handler re-adopted its pre-rollback working copy, wiping any mid-turn `rollback_messages()` restore. The engine now tracks a history version + the in-flight query copy; a mid-turn rollback survives turn completion (restored base + post-rollback tail, orphan tool results sanitized). `continue_pending` runs on a copy with the same adoption rules instead of mutating the live list mid-query.
  4. **`ProgressTracker.eta_seconds()` summed the wrong stages** — it averaged the passive waypoint stages (`generated`/`rendered`/`approved`); `approved` is terminal so its clock never closes and the ETA was ~always `None`. The ETA now sums observed `generating`/`rendering`/`qa` durations owed per in-flight scene and divides by observed pipeline parallelism. Missing transitions `pending -> failed`, `generated -> failed`, and `queued -> failed` were added to the stage machine.
  5. **Asset prefetcher cache key mismatch** — prefetch warmed `read_file` results under `{"file_path": ...}` while the tool's argument is `path` (100% miss), and stored raw content instead of the tool's numbered-line output format. It now warms both common call shapes (path-only and fully-defaulted) in both absolute and cwd-relative path forms, byte-identical to a real read.
  6. **Render pipeline ffmpeg argv + wave races** — `{input_args}`/`{codec_args}` multi-flag strings were injected as single argv tokens (no shell splits them; every ffmpeg render failed); exact-match placeholder tokens are now `shlex.split` into separate argv entries. `wait_all()` could return in the gap between "wave N finished" and "wave N+1 submitted" by the dependency chain; it now waits for promised-but-unsubmitted waves too, and the chain releases its promise on failure.
- Deeper render-queue bug found by the wave-race test: **`WorkStealingQueue.put_nowait` never woke idle workers** (it appended without notifying the condition), so any job submitted while all workers were idle-waiting — including every dependency-wave submission after the first — hung as `pending` forever. `put_nowait` now schedules a wake task on the running loop, and worker `get` uses a bounded wait as a belt-and-braces backstop.
- **MCP SDK 2.x compatibility** — `Tool.inputSchema` was renamed to `input_schema`, `streamable_http_client` reverted to a 2-tuple, and `FastMCP` became `MCPServer` (with `transport_security` moving from the constructor to `streamable_http_app()`); the client and test fixtures now support both SDK generations (`mcp>=1.0.0` as pinned). `structuredContent` is read via `getattr` (older SDKs lack it).

- Performance & safety feature surface (items #1-#50), all wired into the live runtime and opt-in via the new `inference`, `rendering`, `qa`, `control`, and `model_cache` settings groups (env overrides: `OPENHARNESS_PREFIX_CACHING`, `OPENHARNESS_SEMANTIC_CACHE`, `OPENHARNESS_CROSS_PROVIDER_RACING`, `OPENHARNESS_CONTROL`, `OPENHARNESS_RENDER_CACHE`, and more — see MIGRATION.md):
  - **Inference pipeline** (`services.llm_pipeline`, wrapped around every API client): Anthropic prompt-caching breakpoints on system + tools (on by default; cache tokens reported in `UsageSnapshot`), lossless prompt compression for old turns, MinHash semantic response cache with SQLite persistence, cross-provider racing, and API-key rotation pools on rate limits.
  - **Render pipeline** (`services.render_pipeline` + `services.render_cache`): parallel scene rendering across N workers, draft-first QA gating with quality early-exit, fingerprint-based render cache persisted across sessions, NVENC detection with real probe encode, dependency-graph scene ordering with work stealing.
  - **Control plane** (`services.control_bus`, `services.audit_log`, `services.checkpoints`, `services.progress_tracker`, `services.conflict_resolution`, `services.state_validator`, `services.session_services`): kill switch, pause gate, watchdog, per-stage timeouts, injected human messages at turn boundaries, JSONL audit log of every LLM/tool call, conversation checkpoints with rollback, whole-pipeline scene progress tracking, cross-agent file locking for `edit_file`/`write_file`, and saved-session repair on resume.
  - **Nine new tools**: `render_scenes`, `render_queue_status`, `qa_review`, `control`, `checkpoint`, `progress`, `audio_gen` (real TTS through OpenAI-compatible `/audio/speech` endpoints), `server_config`, `cache_stats`.
  - `UsageSnapshot` now carries `cache_write_tokens` / `cache_read_tokens` for provider prompt-cache accounting.
- Hooks now support a `priority` field (default `0`). Within an event, hooks run highest-priority first, and hooks sharing a priority keep their registration order. This lets users order, for example, a security-check hook ahead of a logging hook regardless of where each is declared in settings or contributed by plugins.
- `edit_file` and `write_file` in the React TUI now preview a unified diff before applying file changes, let users approve once or for the rest of the session, and skip the extra prompt automatically in `full_auto` mode.
- **Speed wave 2** — two new always-available performance features plus fan-out wiring:
  - **Async batch executor** (`services.batch_executor`): bounded-concurrency fan-out (semaphore, default 8 in flight via `inference.batch_concurrency` / `OPENHARNESS_BATCH_CONCURRENCY`) with order-preserving results, per-item failure isolation, and retry rounds that re-run only failures. Wired into the new `parallel_prompts` tool and into `batch_scene`'s concurrent scene-retry path (failed-lint scenes now retry in parallel instead of serially).
  - **Cost dashboard** (`services.cost_dashboard`): every completed LLM response is priced from a per-model pricing table (longest-prefix match, Anthropic cache multipliers, user-overridable via `OPENHARNESS_PRICING_JSON`), aggregated by model/scene/project, persisted to the SQLite store, and exposed through the new `cost_report` tool (summary/models/scenes/recent/clear plus a scenes-done → projected-total projection). Unknown models are reported honestly as unpriced; the session budget halt prices them conservatively so the safety control never silently disables.
  - Key pool cooldown design: a rate-limited key now enters a cooldown (default 60s, `inference.key_pool_cooldown_seconds` / `OPENHARNESS_KEY_POOL_COOLDOWN_SECONDS`) and is skipped until it expires; every key is tried before surfacing the error, and a short full-pool stall is waited out instead of failing.
  - QA reviewer now pools one HTTP client for its lifetime (connection reuse across review waves) and closes it per `qa_review` call; `parallel_prompts` shares one pooled client across the whole batch.
  - The audit log keeps one persistent append handle per day instead of one open/close syscall pair per entry, and self-heals (reopens) if a write ever fails.

### Fixed

- Seven reported bugs, plus deeper issues found in the same audit:
  1. `brand.yaml` anchors were injected once at session build and wiped by the first per-line system-prompt rebuild; the section is now built inside `build_runtime_system_prompt` (with an mtime/size cache) so every rebuild keeps it.
  2. Kaggle hardware auto-detection ran after the engine's `tool_metadata` snapshot, so auto-raised concurrency/batch values never reached the query loop; detection now runs before anything snapshots settings.
  3. Tool cache: `TTL=0` was coerced to 3600 (breaking the documented "never expire"), `max_entries=0` was coerced to 256, and cache hits skipped PostToolUse hooks and `read_file` carryover; all three fixed.
  4. BrandKit safe-area check matched substrings of unrelated properties (`opacity`, `zIndex`, `paddingTop`, `margin-right`, …); the regex now requires a fresh identifier boundary.
  5. `thinking_budget` broke Anthropic multi-turn: thinking blocks (with signatures) are now preserved in history when thinking is on, and stripped from outbound history when the current request runs without thinking (Anthropic rejects mixed histories with a 400).
  6. Cross-provider racing and sequential routing sent the primary's model name to fallback providers (usually a 404); each route can now declare its own `model`, and per-route requests drop model-specific knobs (`effort`, `thinking_budget`) that mean nothing on the fallback.
  7. `PromptCompressor` shrank the outbound request but left the engine's per-message token-estimate cache stale, so auto-compact fired far too early; the compressor now publishes the compressed estimate back onto the original messages.
- More fixes from the same deep audit:
  - MinHash signatures used salted `hash()` and could never match across processes (the SQLite semantic cache persisted them); they now use stable blake2b hashing.
  - The semantic cache included the (long, stable) system prompt in its similarity text, so two different short questions sharing a system prompt could replay the wrong cached response; similarity now considers user turns only, and short prompts (<12 words) only match token-identically.
  - The session cost tracker delegated to the new pricing table but priced unknown models at $0, silently disabling the budget halt on any model the table had never seen; it now falls back to conservative mid-tier pricing (the dashboard still reports such models honestly as unpriced).
  - QA: scene verdicts are accepted by filename or bare stem; 0-100 scores are normalized (a raw "85" no longer clamps to a perfect 1.0 and auto-promotes); `batching=True` with `smart_depth=False` now actually batches.
  - Render pipeline: workers beyond the GPU count were pinned to non-existent CUDA devices (now round-robin over visible GPUs, unset on CPU-only hosts); the dependency-wave chain task could poll forever on a stalled wave (bounded, and errors are logged instead of silently stranding later waves); the job registry is pruned at 1000 terminal jobs; `status()` no longer raises `IndexError` on degenerate codec templates; quality lookup matches jobs by resolved path instead of basename collision.
  - Control bus: watchdog and deadlock monitor tasks leaked on every session rebuild / shutdown (now stopped by `set_control_bus` replacement and `begin_shutdown`).
  - File locks: a lockfile left by a crashed process blocked edits for the whole stale-age window; holder pids are now embedded in lockfiles and dead holders are detected and stolen instantly.
  - Key-pool + provider-routing composition: a key pool silently dropped the fallback chain (each rotated key built a bare primary); each rotated key now rebuilds the full routed stack.
  - A rate-limited key pool could end its stream silently (looking like a blank completion); it now always surfaces the provider error or an explicit pool-cooldown error.
- Codex subscription requests now pass reasoning effort separately, enabling `gpt-5.5` with `xhigh` effort instead of treating `gpt-5.5 xhigh` as an unsupported model name.
- Telegram channel now delivers replies again under `ohmo init --no-interactive` and other configs that do not write a `reply_to_message` field. `TelegramConfig` declares `reply_to_message: bool = True` so the attribute access in `TelegramChannel.send` no longer raises `AttributeError` and outbound progress/tool-hint/final messages are sent as expected. See issue #243.

## [0.1.9] - 2026-05-07

### Added

- Added a bundled `skill-creator` skill for creating, improving, and verifying OpenHarness/ohmo skills.
- User-invocable skills can now be triggered directly as slash commands, with support for skill-specific arguments and model override metadata.

### Fixed

- `oh setup` can now update the API key for an already-configured API-key provider profile instead of only changing the model.
- `oh provider edit <profile> --api-key <key>` can now replace a saved profile API key, and `oh provider add ... --api-key <key>` can store one during profile creation.

## [0.1.8] - 2026-05-06

### Added

- Built-in `nvidia` provider profile so `oh setup` offers NVIDIA NIM as a first-class OpenAI-compatible provider choice, with `NVIDIA_API_KEY` auth source, `openai/gpt-oss-120b` as the default model, and the NVIDIA NIM endpoint.
- Built-in `qwen` provider profile so `oh setup` offers Qwen (DashScope) as a first-class provider choice, with `dashscope_api_key` auth source, `qwen-plus` as the default model, and the DashScope OpenAI-compatible endpoint.
- Plugin tool discovery: plugins can now provide `BaseTool` subclasses in a `<plugin>/tools/` directory and they are auto-discovered, instantiated, and registered in the tool registry at runtime. Add `tools_dir` to `plugin.json` (defaults to `"tools"`).
- `oh --dry-run` safe preview mode for inspecting resolved runtime settings, auth state, prompt assembly, commands, skills, tools, and configured MCP servers without executing the model or tools.
- Built-in `minimax` provider profile so `oh setup` offers MiniMax as a first-class provider choice, with `MINIMAX_API_KEY` auth source, `MiniMax-M2.7` as the default model, and `MiniMax-M2.7-highspeed` in the model picker.
- Docker as an alternative sandbox backend (`sandbox.backend = "docker"`) for stronger execution isolation with configurable resource limits, network isolation, and automatic image management.
- Built-in `gemini` provider profile so `oh setup` offers Google Gemini as a first-class provider choice, with `gemini_api_key` auth source and `gemini-2.5-flash` as the default model.
- `diagnose` skill: trace agent run failures and regressions using structured evidence from run artifacts.
- OpenAI-compatible API client (`--api-format openai`) supporting any provider that implements the OpenAI `/v1/chat/completions` format, including Alibaba DashScope, DeepSeek, GitHub Models, Groq, Together AI, Ollama, and more.
- `OPENHARNESS_API_FORMAT` environment variable for selecting the API format.
- `OPENAI_API_KEY` fallback when using OpenAI-format providers.
- GitHub Actions CI workflow for Python linting, tests, and frontend TypeScript checks.
- `CONTRIBUTING.md` with local setup, validation commands, and PR expectations.
- `docs/SHOWCASE.md` with concrete OpenHarness usage patterns and demo commands.
- GitHub issue templates and a pull request template.
- React TUI assistant messages now render structured Markdown blocks, including headings, lists, code fences, blockquotes, links, and tables.
- Built-in `codex` output style for compact, low-noise transcript rendering in React TUI.

### Fixed

- Subprocess teammate spawn (`agent` tool, `task_create`) now works on Windows under Git Bash. `subprocess_backend.spawn` builds a direct-exec `argv` list and passes it through new `argv=` and `env=` kwargs on `BackgroundTaskManager.create_agent_task` / `create_shell_task`; `_start_process` then runs the executable via `asyncio.create_subprocess_exec(*argv)` with no shell in between. Previously the spawn command was a single string interpreted by `bash -lc`, which on Windows could not reliably exec a Windows-pathed Python interpreter (e.g. `C:\Users\...\python.exe`) — Git Bash's escape parser consumed the backslashes from the embedded env-prefix and, even with proper quoting, bash launched via `asyncio.create_subprocess_exec` returned `command not found` for Windows-pathed binaries that worked perfectly when invoked interactively. Bypassing the shell sidesteps the entire class of cross-platform quoting and path-translation hazard. The legacy shell-evaluated `command=` path is preserved for callers (e.g. `BashTool`) that legitimately want shell semantics. See issue #230.
- Bundled skill loader now uses `yaml.safe_load` for SKILL.md frontmatter, matching the user-skill loader. The shared parser is extracted to `openharness.skills._frontmatter` so bundled and user skills handle YAML block scalars (`>`, `|`), quoted values, and other standard YAML constructs the same way.
- Compaction now detects llama.cpp/OpenAI-compatible context overflow errors, accounts for image blocks in auto-compact token estimates, and strips image payloads from summarizer-only compaction requests.
- Large tool results are now bounded in conversation history: oversized outputs are saved under `tool_artifacts`, old MCP results become microcompactable, and context collapse trims stale tool-result payloads.
- ohmo now keeps personal memory isolated from OpenHarness project memory: `/memory` in ohmo sessions targets the ohmo workspace memory store, and ohmo runtime prompt refreshes no longer inject project memory unless explicitly requested.
- Fixed `glob` and `grep` tools hanging indefinitely when the `rg` subprocess produced enough stderr output to fill the OS pipe buffer. `stderr` is now redirected to `DEVNULL` so it is discarded rather than blocking the child process.
- Fixed `bash_tool` hanging after a timed-out command when the subprocess stdout stream stayed open. `_read_remaining_output` now applies a 2-second `asyncio.wait_for` timeout so the tool always returns promptly.
- Fixed `session_runner` background task deadlock caused by an unread `stderr=PIPE` stream. The subprocess now uses `stderr=STDOUT` so all output merges into the single readable stdout pipe.
- React TUI prompt input now treats the raw DEL byte (`0x7f`) as backward delete while preserving true forward-delete escape sequences, fixing backspace failures seen in some macOS terminal environments.
- `todo_write` tool now updates an existing unchecked item in-place when `checked=True` instead of appending a duplicate `[x]` line.

- Built-in `Explore` and `claude-code-guide` agents no longer hard-code `model="haiku"`, which caused them to fail for users on non-Anthropic providers (OpenAI, Bedrock, custom base URLs, etc.). Both agents now use `model="inherit"` so they run with whatever model the parent session is using. `build_inherited_cli_flags` is also fixed to skip the `--model` flag entirely when the value is `"inherit"`, letting the subprocess correctly inherit the parent model via the `OPENHARNESS_MODEL` environment variable instead of receiving the literal string `"inherit"` as a model name.

- React TUI spinner now stays visible throughout the entire agent turn: `assistant_complete` no longer resets `busy` state prematurely, and `tool_started` explicitly sets `busy=true` so the status bar remains active even when tool calls follow an assistant message. `line_complete` is the sole signal that ends the turn and clears the spinner.
- Skill loader now uses `yaml.safe_load` to parse SKILL.md frontmatter, correctly handling YAML block scalars (`>`, `|`), quoted values, and other standard YAML constructs instead of naive line-by-line splitting.
- `BackendHostConfig` was missing the `cwd` field, causing `AttributeError: 'BackendHostConfig' object has no attribute 'cwd'` on startup when `oh` was run after the runtime refactor that added `cwd` support to `build_runtime`.
- Shell-escape `$ARGUMENTS` substitution in command hooks to prevent shell injection from payload values containing metacharacters like `$(...)` or backticks.
- Swarm `_READ_ONLY_TOOLS` now uses actual registered tool names (snake_case) instead of PascalCase, fixing read-only auto-approval in `handle_permission_request`.
- Memory scanner now parses YAML frontmatter (`name`, `description`, `type`) instead of returning raw `---` as description.
- Memory search matches against body content in addition to metadata, with metadata weighted higher for relevance.
- Memory search tokenizer handles Han characters for multilingual queries.
- Fixed duplicate response in React TUI caused by double Enter key submission in the input handler.
- Fixed concurrent permission modals overwriting each other in TUI default mode when the LLM returns multiple tool calls in one response; `_ask_permission` now serialises callers via an `asyncio.Lock` so each modal is shown and resolved before the next one is emitted.
- Fixed React TUI Markdown tables to size columns from rendered cell text so inline formatting like code spans and bold text no longer breaks alignment.
- Fixed grep tool crashing with `ValueError` / `LimitOverrunError` when ripgrep outputs a line longer than 64 KB (e.g. minified assets or lock files). The asyncio subprocess stream limit is now 8 MB and oversized lines are skipped rather than terminating the session.
- Fixed React TUI exit leaving the shell prompt concatenated with the last TUI line. The terminal cleanup handler now writes a trailing newline (`\n`) alongside the cursor-show escape sequence so the shell prompt always starts on a fresh line.
- Reduced React TUI redraw pressure when `output_style=codex` by avoiding token-level assistant buffer flushes during streaming.

### Changed

- ohmo Feishu group routing now supports managed group creation, gateway-scoped provider/model commands, and stricter group mention handling so group conversations only wake ohmo when explicitly addressed.
- Dry-run output now reports a `ready` / `warning` / `blocked` readiness verdict, concrete `next_actions`, likely matching skills/tools for normal prompts, and richer slash-command previews for read-only vs stateful command paths.
- React TUI now groups consecutive `tool` + `tool_result` transcript rows into a single compound row: success shows the result line count inline (e.g. `→ 24L`), errors show a red icon and up to 5 lines of error detail beneath the tool row. Standalone successful tool results are suppressed to reduce transcript noise; standalone errors are still surfaced.
- README now links to contribution docs, changelog, showcase material, and provider compatibility guidance.
- README quick start now includes a one-command demo and clearer provider compatibility notes.
- README provider compatibility section updated to include OpenAI-format providers.

## [0.1.7] - 2026-04-18

### Fixed

- Install script now links `oh`, `ohmo`, and `openharness` into `~/.local/bin` instead of prepending the virtualenv `bin` directory to `PATH`, which avoids overriding Conda-managed shells while preserving global command discovery.
- React TUI prompt now supports `Shift+Enter` for inserting a newline without submitting the current prompt.
- React TUI busy-state animation is less error-prone on Windows terminals: the extra pseudo-animation line was removed, Windows now uses conservative ASCII spinner frames, and the spinner interval was slightly slowed to reduce flashing.

## [0.1.0] - 2026-04-01

### Added

- Initial public release of OpenHarness.
- Core agent loop, tool registry, permission system, hooks, skills, plugins, MCP support, and terminal UI.
