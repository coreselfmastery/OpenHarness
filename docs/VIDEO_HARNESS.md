# OpenHarness × OpenVideoHarness

This document explains how OpenHarness's execution tools connect to the
[OpenVideoHarness](https://github.com/coreselfmastery/OpenVideoHarness)
video-production workflow — and how the video QA stack (`scene_lint`,
`pixel_qa`, `frame_diff`, `vlm_qa`, `contact_sheet`) and the learning
stores (`semantic_memory`, `style_memory`, `failure_patterns`,
`prompt_tracker`, `style_consistency`) slot into it.

## The two halves

**OpenVideoHarness** is the *method*: a playbook of gates and documents
(`BRIEF.md` → `STORYBOARD.md` → scene scripts → `REVIEW.md`), a style
library (`styles/<slug>/tokens.json`, `STYLE.md`, `events.json`,
`score.json`), recipes, and verification tools (`gatecheck.py`,
`speccheck.py`, `readcheck.py`, …). It tells an agent **what a good
video is and in what order to make it**.

**OpenHarness** is the *machine*: the agent runtime with the pipeline
tools that execute that method at speed — parallel rendering, caching,
progress tracking, checkpoints, cost control, a four-layer QA funnel
(code lint → pixel checks → motion checks → selective VLM rubric), and
learning stores that make every finished project improve the next one.

```
OpenVideoHarness playbook          OpenHarness pipeline tools
------------------------          --------------------------
BRIEF.md / gates (gatecheck)  ──►  batch_scene (parallel scene codegen)
styles/<slug>/tokens.json     ──►  brand_kit + pixel_qa (palette checks)
scene scripts (before render) ──►  scene_lint (AST structural checks)
scene scripts (+depends_on)   ──►  render_scenes (waves, tiers, NVENC)
scene code across scenes      ──►  style_consistency (drift detection)
draft renders                 ──►  pixel_qa (deterministic frame checks)
draft renders (playback)      ──►  frame_diff (SSIM + optical flow)
scenes that pass all gates    ──►  vlm_qa (MiniCPM-V rubric scores)
REVIEW.md loop                ──►  qa_review (LLM verdicts, promotion)
final review by a human       ──►  contact_sheet (labeled grid PNG)
QA outcomes per scene type    ──►  failure_patterns (stricter prompts)
prompt templates              ──►  prompt_tracker (success rates)
decisions & styles            ──►  semantic_memory / style_memory
voiceover lines (READS:)      ──►  audio_gen (TTS, background)
whole run                     ──►  progress / checkpoint / cost_report
```

## Suggested workflow

1. **Concept gate** — the OpenVideoHarness gate ① (`gatecheck.py`)
   happens before any rendering; OpenHarness adds nothing there except
   `checkpoint create` so the conversation can roll back.
2. **Scenes** — `batch_scene` writes scene scripts concurrently
   (style-anchored, `READS:` voiceover lines included). The render
   pipeline auto-enqueues drafts; `depends_on:` comments chain scenes
   into dependency waves that render in parallel.
3. **Lint the code BEFORE rendering — `scene_lint`** — structural
   errors cost nothing to fix here and a full GPU render to discover
   later:

   ```
   scene_lint(paths=["scenes/"], assets_manifest="assets.json",
              palette_file="styles/swiss-grid-type/tokens.json")
   ```

   It parses Python scenes with the real AST (JS via calibrated
   regexes) and flags undeclared asset references (hard fail — the
   render would crash), off-safe-area positions, fonts below the
   project minimum, hardcoded colors outside the brand palette,
   repeated magic constants, and non-deterministic calls
   (`Math.random()`, `random.random()`, `Date.now()` ...).

   Also run `style_consistency` on the whole scene set — it extracts
   the palette/fonts/easings from every scene's code and flags both
   off-reference values and cross-scene drift before it shows on
   screen.
4. **Deterministic QA first — `pixel_qa`** — run it on the render
   directory before spending any model tokens on review:

   ```
   pixel_qa(paths=["renders/"], palette_file="styles/swiss-grid-type/tokens.json")
   ```

   It flags black/blank frames, frozen frames, off-palette drift, tiny
   or clipped text boxes, and blocky encodes — offline, in seconds.
   Hard failures mark the scene `failed` in the progress tracker.
5. **Motion QA — `frame_diff`** — a scene can look right as a still
   and still stutter in playback. SSIM between adjacent frames plus
   Farneback optical flow catch frozen motion, stalled segments, jump
   cuts/flashes, and animations running at the wrong speed:

   ```
   frame_diff(paths=["renders/"], expected_min_flow_px_s=2.0,
              expected_max_flow_px_s=400.0)
   ```

6. **Selective VLM rubric — `vlm_qa`** — only for scenes that passed
   the cheap gates but deserve a human-like judgment (typically the
   5-10% with subtle composition/readability/brand questions). It
   samples 5-10 frames, sends ONE multi-image request to the local
   MiniCPM-V 4.5 endpoint with a structured rubric, and returns
   parsed JSON scores (0-10 per criterion) with a pass/fail verdict:

   ```
   vlm_qa(videos=["renders/s_title.draft.mp4"], frames_per_scene=8,
          pass_threshold=7.0)
   ```

   For ad-hoc questions ("transcribe this title card", "compare these
   two takes") use `minicpm_vlm` directly.
7. **LLM review — `qa_review`** — the reviewing model sees the scene
   code plus the pixel/motion/VLM findings and decides pass/fail/fix;
   passing drafts promote to final resolution.
8. **Human review in 30 seconds — `contact_sheet`** — one labeled PNG
   grid of all scenes (scene numbers + final-cut timestamps, plus a
   JSON manifest mapping cells to files):

   ```
   contact_sheet(scene_files=["renders/s01.mp4", "renders/s02.mp4", ...],
                 output_path="contact_sheet.png", columns=4)
   ```
9. **Learn from every outcome** — record QA results so the next
   project starts smarter (all four stores are plain SQLite under
   `<data_dir>/video/`):

   - `failure_patterns record_failure/record_success` per scene type →
     `failure_patterns hint` before generating a high-risk scene type
     returns a stricter-prompt recommendation built from the exact
     historical failure reasons.
   - `prompt_tracker register/record` per prompt version → `stats`
     flags low performers (< 50% over 5+ uses) for revision or
     retirement; `best` picks the winning version.
   - `semantic_memory remember/recall` — decisions recalled by
     *meaning* ("what easing did scene 12 use?"), durable across
     sessions and projects.
   - `style_memory record` a finished project's style DNA + final
     quality score → `suggest` ranked styles that *worked* when the
     next project starts.
10. **Ship** — `progress` shows the board; `cost_report` shows the
   damage; `checkpoint` protects the long run.

## The RenderQueue onto the OpenVideoHarness toolchain

`render_scenes` renders through **engine command templates**
(`DEFAULT_ENGINE_TEMPLATES` in `render_pipeline.py`) — plain argv
lists, overridable per project via `render_engines.json` in the
project root. Placeholders now include `{tier}` (the tier name
itself: draft/review/final) alongside `{scene}`/`{output}`/
`{output_stem}`/`{width}`/`{height}`/`{fps}`/`{codec_args}`/
`{quality}`/`{scene_class}`/`{media_dir}`/`{input_args}` — the hook
that routes this machine's tiers onto OpenVideoHarness's engine
toolchains.

For Blender scenes inside an OpenVideoHarness checkout, drop this
`render_engines.json` into the project root (this is the documented
form, engines/blender-perf.md "与 OpenHarness 的桥"):

```json
{
  "blender": [
    "tools/blender_run.sh", "--profile", "{tier}", "--",
    "blender", "-b", "{scene}", "-o", "{output_stem}", "-a"
  ]
}
```

- `--profile {tier}` lands on OpenVideoHarness's Cycles profiles:
  draft→draft, review→draft (the `review` alias), final→final —
  OIDN, the bounce budget and the determinism contract ride along.
- No `--gpu` needed: each RenderQueue worker already pins its
  subprocess with `CUDA_VISIBLE_DEVICES`, so each Blender process
  sees exactly one card — the one-card-per-frame contract.
- Long single sequences that want BOTH cards at once do not go
  through the queue: OpenVideoHarness's `tools/blender_split.sh`
  splits the frame range across the GPUs (one process per card).
- **The page-contract engines (canvas-film, ClaudeAnimationBase /
  p5.brush, any `window.film` page) bridge with `--tier`** — the same
  one-word policy on `tools/film/render.mjs` (run it as `bin/vh film
  render`, which finds the browser itself):

  ```json
  {
    "blender": [
      "tools/blender_run.sh", "--profile", "{tier}", "--",
      "blender", "-b", "{scene}", "-o", "{output_stem}", "-a"
    ],
    "page": [
      "bin/vh", "film", "render", "{scene}",
      "--tier", "{tier}", "--out", "{output}"
    ]
  }
  ```

  `--tier` maps the queue's tiers onto the whole quality policy:
  draft = 480p + h264_nvenc p4 cq28 when a card answers (and the
  chunk encoders spread round-robin over BOTH T4s), review = 720p +
  nvenc p5 cq23, final = the full-size x264 master (the
  byte-determinism contract). Explicit `--res`/`--dpr`/`--encoder`/
  `--crf`/`--preset` in the template override any single part.
- Web films (HyperFrames) are film-level renders with their own
  gates (`bin/vh`); the queue's `hyperframes` template covers
  per-scene render jobs only.
- While renders run, `tools/kaggle/monitor.sh --watch 5` (from the
  OpenVideoHarness side) shows both T4s, who holds each card, CPU,
  RAM and the `/kaggle/working` disk — the "no wasted machine"
  glance.

## MiniCPM-V 4.5: local vision that fits on one GPU

MiniCPM-V 4.5 is an 8B-parameter multimodal model; **4-bit quantized
it needs only 6–8 GB of VRAM** — a single consumer GPU, a Kaggle T4,
or a laptop. Serve it with any OpenAI-compatible server and point
OpenHarness at it:

```bash
# vLLM (recommended on GPU)
vllm serve openbmb/MiniCPM-V-4_5 --quantization awq --max-model-len 8192 --port 8001

# llama.cpp (GGUF, works on CPU too)
llama-server -m MiniCPM-V-4_5-Q4_K_M.gguf --host 127.0.0.1 --port 8001 -c 8192

# Ollama
ollama pull minicpm-v && ollama serve
```

Configuration (defaults target `http://127.0.0.1:8001/v1`):

```json
// settings.json
{
  "minicpm_v": {
    "base_url": "http://127.0.0.1:8001/v1",
    "model": "openbmb/MiniCPM-V-4_5",
    "api_key": "EMPTY",
    "timeout_seconds": 180
  }
}
```

or environment variables `OPENHARNESS_MINICPM_BASE_URL`,
`OPENHARNESS_MINICPM_MODEL`, `OPENHARNESS_MINICPM_API_KEY`,
`OPENHARNESS_MINICPM_TIMEOUT`.

Verify the endpoint from a session with
`minicpm_vlm(check_server=true)` — it probes `GET /models` and prints
the launch commands above when the server is down.

## pixel_qa details

- **Inputs**: video files or a render directory; palette from
  `brand.yaml`/`brand.yml`/`brand.json` (any `#rrggbb` values) or an
  OpenVideoHarness `styles/<slug>/tokens.json` (`palette.bg/fg/accent/extra`),
  or an inline hex list.
- **Checks** (each pass/fail + confidence, aggregated per scene):
  black/white/blank frame, frozen frame, color drift vs palette,
  text-too-small/off-screen (Canny edges + line-box clustering — a
  heuristic, no OCR), encoding artifacts/noise (macroblock boundary
  ratio, Laplacian variance).
- **Integration**: scenes move `rendered -> qa`; hard failures go to
  `failed` so `progress` reflects reality. `report_path=` writes the
  full JSON verdict for the review record.
- **Install**: `pip install openharness-ai[pixel-qa]`
  (opencv-python-headless). The tool fails loudly with the same hint
  when OpenCV is missing.

## Why deterministic QA before model QA

`scene_lint`, `pixel_qa` and `frame_diff` are free, instant, and
reproducible; the LLM reviewers are none of those. Catching structural
code errors before render and mechanical defects (dead frames, frozen
segments, palette drift, stutter, blockiness) with static analysis and
OpenCV means the expensive review calls spend their tokens on questions
that need judgment — composition, narrative, tone — and the VLM handles
reading text that pixels alone can't interpret. The layers complement
each other:

| Layer      | Tool          | Catches                         | Cost     |
|------------|---------------|---------------------------------|----------|
| Code       | `scene_lint`  | undeclared assets, off-safe-area, tiny fonts, unbranded colors, non-determinism | free |
| Pixel      | `pixel_qa`    | black/frozen frames, palette drift, tiny/clipped text, blocky encodes | free |
| Motion     | `frame_diff`  | frozen motion, stutter, jump cuts, wrong animation speed | free |
| Style      | `style_consistency` | off-brand tokens, cross-scene drift | free |
| Local VLM  | `vlm_qa` / `minicpm_vlm` | rubric scores, wrong text, misrendered charts, layout confusion | local GPU |
| Review LLM | `qa_review`   | narrative, composition, brand voice | API tokens |
| Human      | `contact_sheet` | everything else, 20 scenes in 30 seconds | 30 seconds |

## Semantic memory: embeddings for meaning-based recall

The `semantic_memory` store embeds every remembered fact so recall
works by *meaning*, not exact text. For true semantic recall point it
at any OpenAI-compatible `/embeddings` endpoint (BGE-M3 recommended,
MiniLM also fine — `ollama serve` works too):

```json
// settings.json
{
  "embedding": {
    "base_url": "http://127.0.0.1:8002/v1",
    "model": "BAAI/bge-m3",
    "api_key": "EMPTY",
    "timeout_seconds": 30
  }
}
```

or `OPENHARNESS_EMBEDDING_BASE_URL` / `OPENHARNESS_EMBEDDING_MODEL` /
`OPENHARNESS_EMBEDDING_API_KEY` env vars. With no endpoint configured
the store falls back to a built-in deterministic hashing embedder
(word/character n-gram vectors) — lexical-approximate recall, fully
offline, zero dependencies. Entries record which backend embedded
them; vectors from different embedding spaces are never compared
(dimension mismatch skips instead of scoring garbage).

The learning stores live in plain SQLite files under
`<data_dir>/video/` (`semantic_memory.sqlite3`, `style_memory.sqlite3`,
`failure_patterns.sqlite3`, `prompt_tracker.sqlite3`) — easy to back
up, inspect, or reset; each degrades gracefully to empty results when
the disk is unwritable.
