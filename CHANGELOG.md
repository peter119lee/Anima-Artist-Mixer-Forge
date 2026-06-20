# Changelog

## v26.1.0 (2026-06-21)

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

## v26.0.0 (2026-06-11)

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
  `workflow/Shift testing.before-basic-simplify.json` and documented it in
  README / USAGE as a direct importable example.
- Kept `AnimaArtistOptions` as an advanced/debug node instead of removing it.

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
  `output_avg + artist_static_capture + static_capture_k=4 +
  match_base_norm=False` path, with strength 1.0 and `layer_filter=9-20`
  when `layer_mode=auto`. Live foreground-weighted A/B across portrait,
  close-up, and street prompts reduced descriptor drift without the face,
  clothing, and full-layer smear failures seen in anchor-heavy runs.
- **Scene-tuned low-drift presets** — added `drift_soft`, `face_lock`, and
  `scene_lock` to turn measured manual A/B combinations into one-click
  choices. They keep the static-capture style lock: `drift_soft` lowers
  strength for portraits, `face_lock` adds token norm locking plus
  `base_preserve` for close-ups, and `scene_lock` uses a narrower
  `base_preserve` layer window for explicit wide / background-heavy prompt
  shapes instead of claiming a single universal drift fix.
- **`drift_auto` preset** — prompt- and artist-count-aware runtime routing that
  resolves from `AnimaArtistPack.base_prompt` and the active artist count.
  4+ artist wide / background-heavy scenes use `face_lock` after live A/B
  found it had the lowest average regret there, smaller explicit wide /
  background-heavy scenes use `scene_lock`, 4+ artist simple fullbody prompts
  use `drift_soft` after live A/B found it had the lowest average regret there,
  4+ artist close-ups use `stable_seed` plus `mixed_delta_cap_ratio=0.75`
  after live A/B lowered the worst seed-pair regret, 4+ artist street / urban
  prompts use full-layer `compatibility_safe`, and other 4+ artist portrait /
  broad-subject prompts use the internal `compatibility_safe_9_15` route after
  live A/B made foreground, full, center, and upper-center reductions all
  positive. Smaller close-ups use `face_lock`, and simpler portrait /
  broad-subject prompts use `drift_soft`.
  Inspector reports the resolved preset and reason so users can audit or
  manually override the decision. This is inference-time control, not training.
- **`anchor_lock` preset** — preserves the previous measured-strong
  4-anchor Q behavior (`anchor_seeds_count=4`, `layer_filter=9-25`,
  `anchor_deep_layer_threshold=16`, strength 1.2) for workflows that prefer
  the stronger fixed-anchor lock and accept its content-risk tradeoff.
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
