# Changelog

## v27.1.0 (2026-07-04)

Diagnostics: three new nodes that answer "is each artist actually working,
what did it change, and how strongly" — plus live GPU validation of the
v26.2.0 runtime fixes.

### New nodes
- `AnimaArtistTagCheck` (Anima Artist Tag Check (Encoder)): free encoder-level
  check straight from the pack — flags `[DUPLICATE]` entries (repeat/alias
  tags that encode the same style vector; pairwise cosine >= 0.999, validated
  live) and exact `[NO-OP]` entries. It deliberately does NOT claim to detect
  unknown tags: live calibration showed encoder shift cannot separate real
  artists from gibberish on Anima's LLM encoder (real 0.013-0.039 vs
  gibberish 0.015-0.035, overlapping, with a real artist as the top outlier).
- `AnimaArtistABVariants` (Anima Artist A/B Variants): emits a list of chain
  variants (off / full / solo_each / leave_one_out / cumulative) plus
  filename-safe labels; ComfyUI's list fan-out renders the whole same-seed
  comparison series in one queue. Weights, `@layers` and `%timing` routes
  stay attached to their artist, including comma layer routes.
- `AnimaArtistImpactMap` (Anima Artist Impact Map (A/B Diff)): compares two
  same-seed renders — triptych/overlay/heatmap visualization, impact score,
  changed-area %, composition(low-freq) vs texture(high-freq) and luminance
  splits, and a plain-language verdict.

### Validation
- Cleared the live-GPU validation debt from v26.2.0: 24/24 smoke matrix
  cases pass (RTX 3090, ComfyUI 0.26.2).
- Same-seed production A/B (1536x1024/32 steps) old-vs-new: `balanced` output
  is bit-identical (0.00% diff); `fast_preview` / `compatibility_safe` change
  by ~6%, concentrated on the styled figure — the intended effect of the
  v26.2.0 concat_with_base CFG fix, visually verified (no smearing/washout).
- New manual harnesses: `tests/live_diagnostics_check.py` (the three nodes
  end-to-end) and `tests/live_ab_capture.py` (fixed-seed A/B capture).
- `tests/live_comfy_smoke.py`: added the missing `stabilizer_end_percent` to
  `default_opts` — without it every direct-AnimaArtistOptions case failed
  ComfyUI submit validation (the matrix had not been re-run since that
  widget was added).

### Workflows
- New showcase `workflow/node_usage_showcase/07_diagnostics_tagcheck_ab_impact.json`:
  all three diagnostics nodes wired (variant fan-out + no-mixer baseline
  branch + impact map), counted by the showcase coverage guard.

### Review fixes (pre-release)
- ImpactMap auto_gain no longer crashes on inputs above torch.quantile's
  2^24-element cap (e.g. batch 4 x 2048x2048): the p99 estimate subsamples
  large magnitude tensors.
- ABVariants keeps variant labels aligned with their chains when an entry
  parses to no artist (e.g. a decorative bare `::`): such entries are now
  skipped with a warning instead of desyncing every later label.

## v27.0.0 (2026-07-04)

Forge fork point. Packaging identity only — runtime behavior is identical to
v26.2.0.

- Forked from [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)
  as **Anima-Artist-Mixer-Forge**; the v26 line was submitted upstream as
  [PR #4](https://github.com/An1X3R/Anima-Artist-Mixer/pull/4) and development
  (including new nodes) continues here.
- New ComfyUI registry id `anima-artist-mixer-forge` (publisher
  `peter119lee`); repository moved to
  https://github.com/peter119lee/Anima-Artist-Mixer-Forge.
- Node class names and display names are unchanged, so existing workflows
  load as-is. Do not install this pack alongside the original — they define
  the same node names.
- Bundled workflow metadata (`cnr_id`) now points at the forge pack id.
- New [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): how to add a node, the test
  guards, and the release/publish flow.

## v26.2.0 (2026-07-04)

Fixes from a full four-track review (parsing, runtime state lifecycle,
node integration, docs/packaging/CI). No node signatures changed; existing
workflows keep loading.

### Parsing / chain syntax
- Fixed phantom artists from comma layer routes inside weighted entries:
  `1.2::wlop@0,2,4::` and `wlop@0,2,4::1.2` now parse as one artist with
  route `{0,2,4}` instead of splitting off a bogus artist (previously
  `wlop@0,2,4::1.2` even produced a phantom artist named "1.2" at weight 4).
- A layer route that matches no blocks (e.g. `@30-40` on a 28-block model)
  now disables that artist's injection and reports a warning, instead of
  silently inverting to "inject into all layers".
- Timing routes accept bare percent values: `%0-45` reads as 0%-45%
  (values above 1 divide by 100, like layer routes); nonsense windows are
  rejected loudly instead of clamping to "always active".
- Full-width `，` is honored in base-prompt `1.5::tag::` expansion.
- `::name::` (decorative, no weight) strips its colons; double-weight
  entries like `1.5::wlop::0.8` keep the prefix weight and warn instead of
  encoding a `::0.8` remnant into the artist name.
- NaN weights are rejected as invalid input instead of clamping to 4.0.
- New chain lint surfaced in ChainPreview, Inspector, and the Pack log:
  leftover `::` markers, full-width route punctuation, swallowed
  route-shaped tails, and the `@layers` vs `%timing` confusion now warn.
- ChainPreview's `cleaned_chain` preserves explicit `1::name::` weights
  (an explicit 1.0 intentionally disables weight normalization) and no
  longer truncates weight precision.
- An explicit weight past the 32-artist truncation limit no longer leaks
  `has_explicit` into the surviving chain.

### Recipes
- Recipe JSON is now range-validated on load with the same bounds as the
  UI widgets (out-of-range values clamp with a warning; string booleans
  like `"false"` parse correctly instead of inverting).
- Recipes remember their source preset (format v2): a saved `drift_auto`
  recipe re-resolves its route against the real prompt at apply time
  instead of baking the empty-prompt fallback. v1 recipes load unchanged.

### Runtime state lifecycle
- A layer that hits an exception now falls back only for the rest of that
  run; a unified run-start reset clears disabled layers, EMA/static caches,
  and warning latches on the next queue. Previously one failure disabled
  the layer silently for as long as ComfyUI cached the patched model.
- Out-of-memory errors and interrupt exceptions propagate instead of being
  swallowed by the per-layer fallback.
- `artist_static_capture` correctly resets between runs; re-queueing with a
  new seed no longer reuses the previous generation's frozen artist outputs.
- `concat_with_base` no longer pads non-injected (uncond) rows with zero
  K/V tokens — uncond rows now get exactly the base output, keeping CFG
  intact on the `fast_preview` / `compatibility_safe` paths.
- EMA and static caches are keyed per forward fingerprint, so multiple
  positive conds or VRAM-split batches at the same sigma no longer
  cross-contaminate; the EMA cache also honors `low_vram_cache` now.
- Anchor pre-run pairs `t5xxl_ids`/`t5xxl_weights` with the same cond row
  as the context (uncond-first CFG batches were mispaired), skips forwards
  that carry no cond row, and disables itself with a warning when the
  wrappers captured nothing (instead of re-running the pre-pass every step).
- Static capture on the `combine=concat` path folds the effective
  weight*fade into its cache fingerprint, so freezing mid-fade cannot lock
  a stale weight.
- When another wrapper hides the sampling sigma, EMA/static capture skip
  with one warning instead of accumulating garbage.
- Chaining two mixer nodes over the same blocks now logs which blocks the
  later node overrides.

### Probe
- AnimaArtistProbe rejects non-Anima models with a clear error, restricts
  its delta measurement to cond rows under CFG, resets its statistics at
  run start (seed changes no longer mix trajectories), enforces its step
  budget even when the sigma is not visible, and reports per-artist sample
  counts.

### Nodes / UX
- AnimaArtistBasic caches its internal artist pack, so changing only
  preset/intensity no longer re-encodes every artist.
- The root package shim re-exports AnimaArtistPresetApply and
  AnimaArtistSimpleOptions; real import errors are no longer masked by the
  compatibility fallback.

### Docs / packaging / CI
- CI switched from `unittest discover` to pytest (+ pillow): the v26
  syntax tests were previously never collected in CI.
- Publishing now verifies pyproject version == top CHANGELOG entry.
- `sample workflow.json` no longer requires a third-party NVIDIA node or
  personal LoRA files, and uses the prefix weight syntax.
- Removed the broken legacy `workflow/Shift testing.before-basic-simplify.json`
  and a junk image; recompressed the hero image (2.7 MB → JPEG).
- Bundled workflows share one model filename (`anima-base-v1.0.safetensors`).
- New regression test validates every bundled workflow JSON against the
  current node definitions (node types, widget counts, link integrity).
- README documents AnimaArtistBasic; USAGE fixes broken relative links,
  documents `static_capture_mode`/`static_capture_blend_alpha`, corrects
  the `concat_with_base` strength description and the stale
  "replaces cross_attn" wording, and moves the full 12-preset table to the
  AnimaArtistPreset section (the Starter widget only offers 4 recipes).

## v26.1.0 (2026-06-21)

### Per-artist layer range control
- Layer routes now accept block percentages as well as block indices:
  `artist@0%-33%`, `artist@33%-67%`, or normalized decimals such as
  `artist@0.33-0.67`. Existing block routes like `artist@0-8` keep their
  previous behavior.
- Added `workflow/artist-layer-role-routing.json`, a bundled workflow that
  demonstrates separate artists across background/composition,
  character/body, and clothing/detail layer windows.

### v26 Syntax Enhancements (from upstream)
- **Prefix weight syntax** (recommended): `1.5::wlop` instead of `wlop::1.5`
  - More intuitive for users copying from NovelAI-style prompts
  - Postfix syntax (`wlop::1.5`, `::wlop::1.5`) remains supported for backward compatibility
  - Parsing order: tries prefix first, falls back to postfix
  - Works with parentheses: `0.8::(wlop:1.1)` for layered weighting

- **base_prompt weight syntax**: `1.5::masterpiece::, 1girl` expands to `(masterpiece:1.5), 1girl`
  - Allows weight syntax in base prompts for quality tags
  - Uses prefix-only syntax with trailing `::` boundary marker
  - Can span across commas: `1.3::detailed background, intricate::, 1girl`
  - Weights clamped to [0.0, 4.0] like artist weights

- **stabilizer_end_percent parameter** (advanced): Controls when stabilizers auto-disable
  - Default: `1.0` (full sampling, v24 behavior)
  - For FLS users: `0.5` recommended (stabilizer active first 50%, dynamic after)
  - Affects `static_capture`, `anchor_q`, and EMA; does not affect `lowrank_avg` or `sigma_range`
  - Enables compatibility with step-delta-based samplers (FLS, etc.)

### Bug Fixes
- **Multiple sampler workflows**: Fixed `AttributeError: 'Attention' object has no attribute 'original'`
  - Added defensive check for already-patched cross_attn.forward
  - Handles workflows where same model is used by multiple samplers
  - Tested and verified with user-reported workflow

### 2026-06-19 PR feedback fixes
- Patched cross-attention at the `cross_attn.forward` object-patch level
  instead of replacing the full attention module. This keeps state-dict
  paths stable and fixes multi-sampler workflows that compare patched and
  unpatched branches from the same loaded model.
- Kept `balanced` on the original `output_avg + interpolate` design for both
  single-artist and multi-artist chains. Use `compatibility_safe` explicitly
  when a workflow needs the `concat + concat_with_base` path.
- Restored original-like `balanced` defaults by leaving EMA and norm-lock
  disabled. `match_base_norm` remains available as an explicit stabilizer and
  in presets that intentionally use it, such as `face_lock`.
- Added `AnimaArtistSimpleOptions` for the common layer/timing/compatibility
  controls, leaving the full `AnimaArtistOptions` node as an expert/debug
  surface so existing workflows keep working without overwhelming new ones.
- Changed `AnimaArtistBasic` and the example workflow to default to
  `balanced` instead of `drift_auto`; automatic low-drift routing remains
  available, but the default product path is predictable artist mixing.

### 2026-06-16 workflow simplification
- Added `AnimaArtistBasic`, a minimal entry point that wraps
  `Pack + Preset + CrossAttn`.
- Added a complete example workflow at
  `workflow/Shift testing.before-basic-simplify.json` as a direct importable
  example. (This file was later removed from the package.)
- Kept `AnimaArtistOptions` as an advanced/debug node instead of removing it.

## v26.0.0 (2026-06-11)

### Restructure
- Split the 2,600-line `nodes.py` into the `anima_mixer` package
  (`constants`, `parsing`, `math_utils`, `options`, `chain_tools`,
  `patching`, `wrapper`, `anchor`, `recipe`, `nodes_core`, `nodes_ui`).
  `nodes.py` remains as a backward-compatibility shim.
- All log messages, error messages, and tooltips are now in English.
- Added `pyproject.toml` with Comfy Registry metadata and a GitHub Actions
  CI workflow (ruff + unittest on Python 3.10/3.12).

### Fixes
- **CFG mask (HIGH)**: when ComfyUI batches several latents per cond entry,
  `cond_or_uncond` markers are now expanded over the row chunks instead of
  falling back to injecting into every row (which silently styled the
  uncond pass and weakened CFG for batch sizes > 1).
- **Anchor cache key**: replaced `id(context)` with a content-based
  fingerprint (shape + dtype + value checksum). A freed tensor's id can be
  reused, which could silently re-hit a stale anchor cache.
- **Anchor pre-run no longer re-runs every step**: the cache check now fires
  only at the start of a sampling run (sigma jump). Previously the cache key
  contained the *current* step's sigma, so it missed on every step and the
  anchor pre-pass silently re-ran each step, making `anchor_q` far more
  expensive than documented.
- **Anchor pre-run hardening**: the private `dm._forward` API is now used
  only when present, with a public-call fallback.
- **`enabled=False` early return**: the patch node now returns the
  unpatched model immediately instead of installing wrappers that check a
  flag on every forward.
- **concat + static_capture**: the combined path now uses the same K-step
  temporal averaging as `output_avg` instead of a first-step-only snapshot.
- **Inspector**: accepts an optional MODEL input to read the real block
  count instead of assuming 28.

### Features
- **Timing fade** — `%start-end~fade` adds smoothstep ramps at the edges of
  per-artist timing windows, removing hard style pops at window boundaries:
  `wlop%0.0-0.45~0.1`.
- **Negative weights (style subtraction)** — `::artist::-0.5` pushes a
  style away instead of adding it. Weight range is now [-4, 4].
- **`match_base_norm` option (explicit stabilizer)** — rescales artist attention
  output to the base output's RMS energy before fusion. When enabled, v26 uses
  token-level, per-artist norm locking (`norm_lock_mode=token`,
  `norm_lock_scope=per_artist`) so each artist is calibrated before the
  weighted mix. This suppresses seed-specific high-energy artist spikes
  more aggressively than the legacy whole-row final-output lock while
  keeping `row` / `mixed` modes available for A/B comparisons.
- **Stable-seed preset retune** — `stable_seed` now uses the content-safer
  `output_avg + mixed_delta_cap + mixed_delta_cap_ratio=0.75 +
  match_base_norm=False` path, with strength 1.0 and `layer_filter=9-20`
  when `layer_mode=auto`. This keeps the real style-mixer path active while
  capping extreme mixed deltas, after multi-artist evidence showed static
  capture could over-constrain or wash out style.
- **Scene-tuned low-drift presets** — added `drift_soft`, `face_lock`, and
  `scene_lock` to turn measured manual A/B combinations into one-click
  choices. `drift_soft` uses light EMA and lower strength for portraits,
  `face_lock` adds token norm locking, `base_preserve`, and a mixed-delta cap
  for close-ups, and `scene_lock` uses light EMA plus a narrower
  `base_preserve` layer window for explicit wide / background-heavy prompt
  shapes instead of claiming a single universal drift fix.
- **`drift_auto` preset** — prompt- and artist-count-aware runtime routing that
  resolves from `AnimaArtistPack.base_prompt` and the active artist count.
  4+ artist wide / background-heavy scenes use `scene_lock`, 4+ artist simple
  fullbody prompts use `drift_soft`, 4+ artist close-ups use `stable_seed`
  plus `mixed_delta_cap_ratio=0.75`, and other 4+ artist portrait / street /
  broad-subject prompts stay on `drift_soft`. Smaller close-ups use
  `face_lock`, and simpler portrait / broad-subject prompts use `drift_soft`.
  Compatibility concat routes are now explicit user choices instead of
  automatic broad multi-artist routes, so artist-count changes stay visible.
  Inspector reports the resolved preset and reason so users can audit or
  manually override the decision. This is inference-time control, not training.
- **`anchor_lock` preset** — softened from the previous measured-strong
  4-anchor Q behavior to one anchor with `anchor_user_blend=0.35`,
  `layer_filter=9-15`, `anchor_deep_layer_threshold=12`, and strength 0.9
  after evidence showed the stronger lock could introduce extra limbs.
- **`identity_guard` preset retune** — moved from `lowrank_avg` to
  `output_avg + base_preserve + match_base_norm + mixed_delta_cap` after the
  10-artist evidence matrix showed the low-rank preset path was too slow for
  a one-click mode. `lowrank_avg` remains available as an expert combine mode.
- **`anchor_base_norm_ref` option** — optional A/B path for `anchor_q` +
  `match_base_norm` that uses the fixed anchor's base output as the norm
  reference instead of the current seed's base output.
- **`contribution_balance` option** — optional per-artist delta equalizer
  for dominance flips. It is available for A/B work but stays off by default;
  live multi-seed checks favored static capture as the safer opt-in route for
  low-drift presets.
- **`mixed_delta_cap` option** — optional inference-time limiter for the final
  mixed artist delta before `interpolate` / `base_preserve` fusion. It caps
  effective artist-delta RMS against base RMS after strength is considered,
  giving live A/B a direct way to test lower drift without training or changing
  the scene-tuned `drift_auto` routes.
- **`static_capture_mode=blend_perp`** — added an advanced experimental
  mode that reintroduces only base motion perpendicular to the frozen style
  delta. Live A/B showed a narrow scene win, but not a stable cross-scene
  win, so it remains off the default path.
- ~~`embed_avg` combine mode~~ — cut before release. Live A/B testing at
  real resolutions showed that averaging LLMAdapter embeddings re-creates
  the token-misalignment artifact that got the old `mean`/`weighted_sum`
  modes removed (artist tags shift the base prompt's token positions, so
  per-position averaging blends unrelated words). Recipes that reference it
  load with a warning and fall back to `output_avg`.
- **`max_batch_artists` option** — caps how many artists share one batched
  forward, bounding peak VRAM with many artists at high resolution.
- **`low_vram_cache` option** — stores static-capture and anchor caches in
  system RAM instead of VRAM.
- **Recipe nodes** — `AnimaArtistRecipeSave` / `AnimaArtistRecipeLoad`
  serialize a full mixer setup (chain + modes + options) to a shareable
  JSON string.
- **Layer probe** — `AnimaArtistProbe` + `AnimaArtistProbeReport` measure
  each artist's per-layer style influence during a sampling run and suggest
  `@layers` routes, replacing guesswork with measurement.

### Tests
- Real-torch test suite: low-rank determinism, perpendicular projection,
  fusion math, CFG mask expansion, timing fade factors, chunking, anchor
  fingerprints, recipe round-trips.
- Live ComfyUI smoke harness (`tests/live_comfy_smoke.py`): real sampling
  workflows against a running server + Anima model, including the low-drift
  preset paths.
- Live drift A/B harness (`tests/live_drift_ab.py`): foreground-weighted
  multi-seed checks for comparing stabilization presets by prompt type.

---

## v25.2
- Per-artist sampling timing (`%start-end`), `compatibility_safe` preset,
  Inspector block maps, runtime warnings for suspicious cross-attention /
  model-wrapper conflicts, and UX helper nodes (Starter, Chain Builder,
  Chain Preview) for building chains before CLIP encoding.

## v25.1
- Per-artist layer routing (`@layers`): different artists can inject into
  different DiT block ranges from the same chain.

## v25
- Fixed `::name::weight` explicit weights not actually disabling
  `normalize_weights` on the patch path.
- Added `AnimaArtistPreset` (balanced / strong_style / stable_seed /
  anchor_lock / fast_preview / identity_guard) and `AnimaArtistInspector`.
- `lowrank_avg` switched to a deterministic Gram eigendecomposition
  (no randomized SVD approximation).
- Anchor cache key gained the first timestep/sigma to reduce stale reuse
  after sampling-condition changes.

## v24
- New `::name::weight` chain syntax: a linear weight applied at the
  cross-attention injection layer (vs. the non-linear CLIP-side parentheses
  syntax). Any explicit `::weight` bypasses `normalize_weights`.

## v23
- `strength` upper bound raised from 1.0 to 4.0; values above 1.0 enter
  CFG-style extrapolation `out = base + strength * (artist - base)`.

## v22
- Anchor-Q tuning: `anchor_seeds_count` (multi-seed anchor averaging),
  `anchor_user_blend` (anchor/user Q blend), and
  `anchor_deep_layer_threshold` (shallow-anchor / deep-user split).

## v21
- Anchor-Q: replace the user-seed hidden state with a fixed-seed anchor's
  hidden state as the artist attention Q, decoupling style mixing from the
  user seed.

## v20
- `static_capture_k` made configurable (default 6, was hardcoded 3).

## v18-v19
- Static capture: freeze artist attention outputs after the first K steps
  (cross-seed stabilization + 30-50% speedup).

## v17
- Re-added `base_preserve` fusion mode (perpendicular-only artist deltas);
  EMA stabilizer (`artist_ema_alpha`); `lowrank_avg` combine mode.
