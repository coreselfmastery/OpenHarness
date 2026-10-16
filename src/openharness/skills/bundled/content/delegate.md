---
name: delegate
description: >-
  Delegate work to subagents with batch mode, brand-aware generation, and
  parallel QA waves. Use this for any multi-scene, multi-step, or
  parallelizable workflow.
user-invocable: true
argument-hint: "[description of work to delegate]"
---

# delegate

Delegate work to subagents, with built-in batching, brand enforcement,
and parallel QA. Use this for any multi-scene, multi-step, or
parallelizable workflow.

## When to use

Use when the user asks you to:
- Generate multiple scenes for a video
- Review a batch of scenes in parallel
- Spawn background workers for independent tasks
- Run QA on a list of assets
- Produce a series of related code artifacts

## Workflow

### Step 1: Check for brand.yaml (Brand Kit wiring)

Before generating any scene code, check if `brand.yaml` exists in the
project root. If it does:

1. Call `brand_kit` with `action=drift` and `project_id=<project name>`
   to get a creative drift config.
2. The drift config gives you derived colors, motion intensity, and
   transition style — all within the brand envelope.
3. Pass these to every scene generation call so scenes stay on-brand
   but each is unique.

If `brand.yaml` is missing, proceed without brand enforcement — but
recommend the user create one for channel consistency.

### Step 2: Choose single-scene vs batch mode

**Batch Mode (default for ≥3 scenes):**

When you need to generate code for 3 or more scenes, use the
`batch_scene` tool instead of calling the LLM once per scene.

```
batch_scene(
  scenes=["Scene 1 description", "Scene 2 description", ...],
  output_dir="scenes",
  engine="hyperframes",
  batch_id="batch_001"
)
```

This generates code for all scenes in a single LLM call — a 10×
reduction in API calls vs one-scene-per-call.

**Single Mode (for 1-2 scenes or debugging):**

For 1-2 scenes, or when iterating on a single scene, generate the
code directly via the LLM. Use batch mode once the pattern is stable.

### Step 3: Determine batch size

Read `batch_size` from the tool metadata (set via
`OPENHARNESS_BATCH_SIZE` env var or `settings.batch_size`).

- Default: 1 (no batching)
- Recommended for video: 5-10 scenes per batch
- Maximum: 10 (API limit for most providers)

When the total scene count exceeds the batch size, split into multiple
batches. Example: 25 scenes with batch_size=8 → 4 batches (8+8+8+1).

### Step 4: Scene-level caching (optimization)

Before generating a scene, check if a structurally identical scene
already exists in the cache. The cache key is a hash of the scene's
structural fingerprint (engine + duration bucket + read count +
visual-element types), NOT the raw description text.

If a cache hit is found, reuse the cached code instead of calling the
LLM. This saves 20-30% of generation calls on videos with repeated
patterns (e.g., multiple stat cards, lower thirds, title cards).

### Step 5: Pre-generate shared assets (optimization)

Before generating any scenes, check if the project has shared assets
that every scene will import:

- Logo animation
- Background gradient
- Font loader
- Color palette constants
- Easing curves

If these don't exist yet, generate them ONCE at the start of the
project and store them in `shared/`. Every scene imports from `shared/`
instead of regenerating the same code.

### Step 6: Parallel QA waves (optimization)

After scenes are generated, run QA in parallel:

1. Split scenes into batches of 5.
2. Spawn 5 QA subagents in parallel, each reviewing one batch.
3. Each QA agent runs the appropriate review depth:
   - **Lightweight check** for title cards, transitions, simple typography
   - **Full 20-point review** for complex scenes (Blender, Manim, data viz)
4. Collect all QA results.
5. For failed scenes, retry only the failed ones (not the whole batch).

Use the `agent` tool with `subagent_type="qa"` to spawn QA workers.

### Step 7: Smart review depth

Not every scene needs the full 20-point checklist. Classify each scene
before review:

- **Simple (lightweight check, 5 points):** title cards, transitions,
  static typography, color-only scenes.
- **Standard (10-point review):** most scenes with motion, narration
  sync, basic animation.
- **Complex (full 20-point review):** Blender, Manim, data viz,
  physics simulation, multi-layer compositing.

This saves 20-30% of review time by skipping the full review on
simple scenes.

### Step 8: Render pipeline overlap (optimization)

When rendering scenes, don't wait for all scene code to be written
before starting renders. Instead:

1. As soon as scene 1's code is validated, add it to the render queue.
2. While the render queue processes scene 1, the LLM writes scene 2.
3. When scene 2 is ready, add it to the queue.
4. The render queue processes scenes in parallel with code generation.

This requires a render queue worker that runs independently of the
main agent loop.

### Step 9: Collect results

After all scenes are generated, QA'd, and rendered:

1. Summarize: scenes generated, scenes passed QA, scenes failed,
   render time, total LLM calls.
2. If any scenes failed, list them for escalation to the Director.
3. Save a project manifest to `project.json` with the final state.

## Rules

- Always use batch mode for ≥3 scenes.
- Always check for `brand.yaml` before generating.
- Always check the scene cache before calling the LLM.
- Always pre-generate shared assets before scene 1.
- Always run QA in parallel waves, not sequentially.
- Always classify scene complexity before choosing review depth.
- Never re-render a scene that hasn't changed (use incremental cache).
- Never regenerate a shared asset that already exists.
- Escalate to the Director after 3 consecutive failures on the same scene.

## Failure handling

| Failure | Action |
|---|---|
| Parse error in batch | Reject whole batch, retry with smaller batch size |
| Lint error in 1 scene | Retry only that scene (single-scene LLM call) |
| Determinism failure | Isolate scene, render alone, compare frames |
| Style drift | Retry batch with stronger STYLE.md emphasis |
| 3 failures on same scene | Escalate to Director agent |
| QA rejection | Retry scene with QA feedback incorporated |
| Render timeout | Split scene into sub-scenes, render separately |
