# OpenHarness × OpenVideoHarness

This document explains how OpenHarness's execution tools connect to the
[OpenVideoHarness](https://github.com/coreselfmastery/OpenVideoHarness)
video-production workflow — and how the two new video tools (`pixel_qa`
and `minicpm_vlm`) slot into it.

## The two halves

**OpenVideoHarness** is the *method*: a playbook of gates and documents
(`BRIEF.md` → `STORYBOARD.md` → scene scripts → `REVIEW.md`), a style
library (`styles/<slug>/tokens.json`, `STYLE.md`, `events.json`,
`score.json`), recipes, and verification tools (`gatecheck.py`,
`speccheck.py`, `readcheck.py`, …). It tells an agent **what a good
video is and in what order to make it**.

**OpenHarness** is the *machine*: the agent runtime with the pipeline
tools that execute that method at speed — parallel rendering, caching,
progress tracking, checkpoints, cost control, and now deterministic
pixel QA and local vision-language review.

```
OpenVideoHarness playbook          OpenHarness pipeline tools
------------------------          --------------------------
BRIEF.md / gates (gatecheck)  ──►  batch_scene (parallel scene codegen)
styles/<slug>/tokens.json     ──►  brand_kit + pixel_qa (palette checks)
scene scripts (+depends_on)   ──►  render_scenes (waves, tiers, NVENC)
draft renders                 ──►  pixel_qa (deterministic frame checks)
draft renders / title cards   ──►  minicpm_vlm (local VLM review)
REVIEW.md loop                ──►  qa_review (LLM verdicts, promotion)
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
3. **Deterministic QA first — `pixel_qa`** — run it on the render
   directory before spending any model tokens on review:

   ```
   pixel_qa(paths=["renders/"], palette_file="styles/swiss-grid-type/tokens.json")
   ```

   It flags black/blank frames, frozen frames, off-palette drift, tiny
   or clipped text boxes, and blocky encodes — offline, in seconds.
   Hard failures mark the scene `failed` in the progress tracker.
4. **Semantic QA — `minicpm_vlm`** — for the frames that need *reading*
   (title cards, charts, subtitles), ask the local MiniCPM-V 4.5 model:

   ```
   minicpm_vlm(video="renders/s_title.draft.mp4", frames=8,
               prompt="Transcribe every visible text and note any typos or layout problems.")
   ```

5. **LLM review — `qa_review`** — the reviewing model sees the scene
   code plus the pixel/VLM findings and decides pass/fail/fix;
   passing drafts promote to final resolution.
6. **Ship** — `progress` shows the board; `cost_report` shows the
   damage; `checkpoint` protects the long run.

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

`pixel_qa` is free, instant, and reproducible; the LLM reviewers are
none of those. Catching mechanical defects (dead frames, frozen
segments, palette drift, blockiness) with OpenCV means the expensive
review calls spend their tokens on questions that need judgment —
composition, narrative, tone — and the VLM handles reading text that
pixels alone can't interpret. The three layers complement each other:

| Layer      | Tool          | Catches                         | Cost     |
|------------|---------------|---------------------------------|----------|
| Pixel      | `pixel_qa`    | black/frozen frames, palette drift, tiny/clipped text, blocky encodes | free |
| Local VLM  | `minicpm_vlm` | wrong text, misrendered charts, layout confusion | local GPU |
| Review LLM | `qa_review`   | narrative, composition, brand voice | API tokens |
