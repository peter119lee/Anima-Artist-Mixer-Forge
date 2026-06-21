# Anima-Artist-Mixer

## Introduction

This is a ComfyUI custom node that provides **multi-artist mixing** for the Anima model. It hooks into the cross-attention layers and combines multiple artist conditionings with controllable strategies, sidestepping the interference that LLM-based text encoders suffer from when multiple artist tags coexist in a single prompt.

The companion `AnimaArtistPack` node provides a one-shot experience: write your artist list in one text box (comma or newline separated) and your main prompt in another. The node automatically splits, encodes, and packages everything for downstream use.

This README documents the **v26 architecture**. The default `balanced` preset stays close to the original artist mixer, while `prompt_passthrough` uses the no-mixer/direct-prompt path and `drift_auto` provides an opt-in low-drift route. Older versions are still functionally a subset.

v25.1 also adds per-artist layer routing, matching the original repository's first public feature request: different artists can now be injected into different DiT block ranges from the same artist chain.

v25.2 adds per-artist sampling timing, a `compatibility_safe` preset, Inspector block maps, runtime warnings for suspicious cross-attention / model-wrapper conflicts, and UX helper nodes for building or previewing artist chains before CLIP encoding.

v26 adds prefix artist weights (`1.2::artist::`), base-prompt tag weights (`1.2::masterpiece::`), negative artist weights (style subtraction, `-0.5::artist::`), smoothstep timing fades (`%start-end~fade`), optional style-drift reduction (`match_base_norm` with token/per-artist norm lock), prompt-aware `drift_auto` plus scene-tuned low-drift presets (`drift_soft`, `face_lock`, `scene_lock`), VRAM controls (`max_batch_artists`, `low_vram_cache`), shareable JSON recipes (`AnimaArtistRecipeSave/Load`), a per-layer style probe (`AnimaArtistProbe` + `AnimaArtistProbeReport`), a CFG correctness fix for batch sizes > 1, and a package restructure with a real test suite and CI. Existing artist layer and timing routes remain supported. See [CHANGELOG.md](../CHANGELOG.md).

## What problem it solves

Anima uses an LLM as its text encoder (unlike SDXL's CLIP). LLM encoders are heavily **contextualized** — every token's embedding fuses semantics from surrounding tokens. This has a direct consequence:

- Single artist tag: the LLM produces a conditioning that captures that artist's style accurately. Works well.
- Multiple artist tags together: the artist tags' embeddings interfere with each other, and the resulting conditioning ends up looking like neither A nor B but a "squeezed-together" middle ground.

This node encodes each artist as a **separate** conditioning, bypassing the interference at the encoding stage, then mixes them inside the model at the cross-attention level using selectable strategies. Mixing happens in an already-stabilized feature space, where it's far more controllable than mixing at the prompt level.

The product goal is practical image generation: predictable artist mixing on top of the base model. The default path should preserve the prompt and expose controllable artist influence. Automatic low-drift routing and stabilizers are opt-in tools, not the default style source.

## How it works

### Anima's structure

Anima = MiniTrainDIT backbone + LLMAdapter text adapter. Text flows like this:

```
Prompt
  → LLM encoder (Qwen, etc.)
  → raw embedding (1, T, 1024)
  → LLMAdapter (6-layer transformer, adapts LLM output to DiT's expected distribution)
  → processed (1, 512, 1024), padded to fixed length 512
  → consumed as K/V by every DiT block's cross-attention
```

The DiT backbone has 28 blocks total, each with its own independent cross-attention layer. The same text conditioning is consumed 28 times across these layers.

### Injection mechanism

The node patches `diffusion_model.blocks[i].cross_attn.forward` using ComfyUI's `add_object_patch` API. The attention module itself stays in the original model tree, so parameter paths remain stable when one workflow compares patched and unpatched sampler branches.

Each artist conditioning is lazily run through the LLMAdapter on its first forward call (when the model is already on GPU), producing a `(1, 512, 1024)` processed embedding that's cached for reuse across sampling steps.

Each layer's injection is wrapped in exception isolation: if a single layer's injection fails, only that layer falls back to the original cross-attention; other layers continue working normally.

### CFG compatibility

ComfyUI batches cond and uncond into a single `batch=2` forward, with `transformer_options["cond_or_uncond"]` marking each row. This node injects only into the cond rows by default; uncond rows keep their original base context, so CFG guidance is preserved naturally. `apply_to_uncond` defaults to False and is not recommended to enable.

## The cross-seed instability problem (and how v26 addresses it)

In multi-artist setups, the same prompt with different seeds tends to produce **noticeably different style mixes** — sometimes wlop dominates, other times sakimichan does, even though their weights are equal. This is structural, not a bug:

- The cross-attn Q comes from base hidden state, which is seed-driven
- For each seed, attention picks slightly different artist token weights
- Across seeds, the "dominant artist" can flip

v26 keeps `balanced` close to the original mixer behavior by default, then layers optional stabilizers from light to heavy (configured in `AnimaArtistOptions` or selected through `AnimaArtistPreset`):

0. **match_base_norm + norm_lock_mode=token + norm_lock_scope=per_artist** — optional per-token RMS alignment applied to each artist before mixing, reducing seed-specific style-strength spikes
1. **artist_ema_alpha** — temporal EMA smoothing across sampling steps
2. **combine_mode = lowrank_avg + lowrank_k** — deterministic low-rank constraint on multi-artist deltas
3. **artist_static_capture + static_capture_k** — freeze artist attention after the first K steps
4. **contribution_balance** — optional experimental delta-strength equalizer for artist dominance flips; default off because static capture was more reliable in live A/B
5. **artist_anchor_q** — replace user-seed Q with a fixed-seed anchor's Q (most aggressive built-in stabilizer; not a complete image lock)

These can be combined freely. Enable `match_base_norm` only when you intentionally want v26 norm-lock stabilization instead of original-style behavior.

## Mathematical limits of artist mixing

Up front: **this node cannot achieve the near-lossless artist mixing that SDXL does.**

SDXL's CLIP encoder produces approximately linearly composable per-token features. Anima's LLM + LLMAdapter output is **strongly non-linear** — any mixing strategy introduces distortion. What this node does is make distortion as controllable as possible and avoid the worst failure modes, not eliminate it.

In practice:
- Style-similar artists tend to mix well
- Style-divergent artists may "regress to the mean", landing in a compromise that resembles neither A nor B. This is more pronounced with weight normalization on (the default), since features get averaged after being normalized to relative proportions
- Extreme weight ratios (e.g. `"1.0, 0.05"`) typically collapse back to the dominant artist's pure style
- v25's `lowrank_avg` (k=1) deliberately accepts more "regression to the mean" in exchange for cross-seed stability — good for production, less suited to experimental style exploration

## Requirements

- **Anima model only**. Depends on Anima's built-in `LLMAdapter` (`preprocess_text_embeds`); plain MiniTrainDIT or other DiTs won't work
- Must use the **CLIP loader compatible with Anima's text-encoding workflow** (i.e. one whose tokens carry `t5xxl_ids`). `AnimaArtistPack` calls `clip.encode_from_tokens_scheduled` internally
- Inference path only, no training support

## Installation

Clone or download into your ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/<this-plugin-folder>/
```

Restart ComfyUI. No extra dependencies.

## Quick start
![workflow](docs/images/workflow.png)

```
                          ┌──► artist_pack ──► AnimaArtistPresetApply ──► MODEL ──► KSampler
[Load CLIP] ─► CLIP ──────┤                                   │                          │
                          │                                   └──► base_prompt ──► (positive)
                          │
                          └──► CLIPTextEncode (Negative) ──► (negative)

[Load Anima Model] ──► MODEL ──► AnimaArtistPresetApply

(optional) AnimaArtistChainBuilder ──► artist_chain ──► AnimaArtistPack
(optional) AnimaArtistChainPreview ──► cleaned_chain / syntax report
(optional) AnimaArtistStarter ───────► artist_chain ──► AnimaArtistPack
                                  └──► preset / advanced_options ──► AnimaArtistPresetApply
(optional) AnimaArtistPreset  ──► preset ────────────► AnimaArtistPresetApply
(optional) AnimaArtistSimpleOptions ─► advanced_options ─► AnimaArtistPresetApply
(optional) AnimaArtistOptions (Expert) ─► advanced_options ─► AnimaArtistPresetApply
(optional) AnimaArtistInspector ◄── artist_pack / preset / advanced_options
```

Open [`../sample workflow.json`](<../sample workflow.json>) first. It uses the
current `AnimaArtistPack -> AnimaArtistPresetApply` route, so preset-owned
values are not mixed with ignored manual widgets.

Open [`../workflow/node_usage_showcase/README_zh.md`](../workflow/node_usage_showcase/README_zh.md)
when you want a Chinese guide and example workflow for every node in the pack.

Open [`../workflow/artist-layer-role-routing.json`](<../workflow/artist-layer-role-routing.json>)
for a focused character / clothing / background routing example. It maps
three separate artists to early, middle, and late layer windows from one
`AnimaArtistPack`.

Key points:
- Fastest guided builder: use `AnimaArtistStarter`, fill `artist_table`, select a recipe, then follow its in-UI wiring guide
- Use `AnimaArtistChainBuilder` for the fastest safe setup: enter a few artists in the shortcut rows or many artists in `artist_table`, pick a layout, then connect its `artist_chain` output into `AnimaArtistPack`
- Use `AnimaArtistChainPreview` when hand-writing chains; it catches syntax mistakes before CLIP encoding
- Write your artist chain in `AnimaArtistPack`'s top text box (comma or newline separated)
- Write your main prompt in the bottom text box
- Connect `AnimaArtistPresetApply`'s `base_prompt` output directly to KSampler's positive input
- Encode the negative prompt independently with `CLIPTextEncode`; it does not go through this plugin
- Start with `AnimaArtistPreset(preset=balanced)` unless you already know which advanced settings you want
- Use `AnimaArtistPreset(preset=compatibility_safe)` first when combining with regional prompts, Forge Couple-style routing, attention masks, or other cross-attention patch nodes
- Common layer/timing controls come via `AnimaArtistSimpleOptions`; stabilizers and debug controls stay in `AnimaArtistOptions (Expert)`
- Use `AnimaArtistInspector` to show the actual effective weights, block map, preset settings, and configuration warnings inside ComfyUI

## Parameters

### AnimaArtistStarter (recommended first node)

This is the lowest-friction entry point. It combines the common `ChainBuilder + Preset` setup into one helper node and outputs:

- `artist_chain` for `AnimaArtistPack.artist_chain`
- `preset` for `AnimaArtistPresetApply.preset`
- `advanced_options` for `AnimaArtistPresetApply.advanced_options` when you want the explicit option payload
- an in-UI `guide` with status, wiring steps, preset summary, chain preview, and warnings

Use `artist_table` as one artist per line:

```
artist | weight | layers | timing
@wlop | 1.2
krenz | 0.8
hiten
```

Only the artist column is required. Bad weights are not silently swallowed; the guide reports them and falls back to `1.0`.

| Recipe | Best use |
|---|---|
| `prompt_passthrough` | Direct prompt/no-mixer route while keeping positive artist weight syntax |
| `balanced` | Default first run |
| `strong_style` | Stronger visual style |
| `stable_seed` | Same prompt across many seeds |
| `drift_auto` | Explicit opt-in automatic low-drift routing from `base_prompt` |
| `drift_soft` | Portrait / broad-subject prompts with softer style lock |
| `face_lock` | Close-up face prompts |
| `scene_lock` | Explicit wide / background-heavy scene prompts |
| `anchor_lock` | Stronger fixed-anchor seed lock |
| `fast_preview` | Fast exploration |
| `identity_guard` | Preserve character/object identity |
| `compatibility_safe` | Regional prompts, Forge Couple-style routing, attention masks, or other cross-attention patch nodes |

Recommended start:

```
recipe = balanced
layout = layer_scheduled
```

If a workflow already uses regional prompting or another attention patcher, start with:

```
recipe = compatibility_safe
layout = layer_scheduled
```

### AnimaArtistChainBuilder (UX helper)

This is the easiest way to build a correct chain without memorizing syntax. It outputs a ready-to-connect `artist_chain` string plus a preview report. The three visible artist rows are shortcuts, not a hard limit; use `artist_table` for larger chains. `AnimaArtistPack` still applies the real upper limit, `MAX_ARTISTS = 32`.

| Layout | Behavior |
|---|---|
| `manual` | Uses the custom layer/timing fields exactly as entered |
| `even_layers` | Evenly splits available DiT blocks across the non-empty artists |
| `layer_scheduled` | Uses a three-stage recipe: early/mid/late layers plus early/mid/late sampling windows |

For `layer_scheduled`, the default rows are:

| Row | Blocks | Timing | Typical role |
|---|---|---|---|
| artist 1 | `0-8` | `0.0-0.45` | composition / global style |
| artist 2 | `9-18` | `0.35-0.85` | structure / character bias |
| artist 3 | `19-27` | `0.65-1.0` | detail / brushwork |

For four or more artists, `layer_scheduled` evenly splits blocks across the active rows and assigns overlapping sampling windows across `0.0-1.0`.

`artist_table` format:

```
artist | weight | layers | timing
@wlop | 1.2 | 0-8 | 0.0-0.45
krenz | 0.8 | 9-18 | 0.35-0.85
hiten | 1.0 | 19-27 | 0.65-1.0
```

Only the artist column is required. Blank weight defaults to `1.0`; blank layer/timing fields are filled by the selected layout.

### AnimaArtistChainPreview (syntax check)

Preview takes any hand-written `artist_chain` and reports:

- parsed artists and injection weights
- parsed `@layers` and `%timing`
- a block map
- warnings for invalid timing, risky explicit weights, or truncation

It does not need CLIP or a model. Use it before `AnimaArtistPack` when experimenting with complex chains.

### AnimaArtistPack (artist chain split + encode)

| Parameter | Type | Description |
|---|---|---|
| `clip` | CLIP | Anima-compatible CLIP |
| `artist_chain` | STRING (multiline) | Artist chain. Comma or newline separated. Supports CLIP weighting `(wlop:1.2)`, injection-layer weight `1.5::wlop::`, per-artist layer routing `@0-8` / `@33%-67%` / `@0.33-0.67`, and per-artist timing `%0.0-0.45` |
| `base_prompt` | STRING (multiline, optional) | Main prompt. Leave empty to encode artists alone |

Outputs `ANIMA_PACK`, an internal struct holding each artist's separately-encoded conditioning, the artist label list, the parsed per-artist weights, and a separately-encoded conditioning for the bare base prompt.

How it works internally: the node splits `artist_chain` into N artist names, parses any `weight::name::` injection weights (and old postfix weights for compatibility), strips those weights before CLIP encoding, and encodes each as `<artist_name>\n<base_prompt>` (Anima's recommended format: artist first, newline, then main prompt). It also encodes a clean copy of `base_prompt` alone for use as KSampler's positive conditioning.

### AnimaArtistPresetApply (preset node)

Use this node for preset workflows. It takes `model`, `artist_pack`, and a `preset` payload, then applies the preset-owned `combine_mode`, `fusion_mode`, and `strength` without exposing manual widgets that would be ignored.

| Parameter | Type | Description |
|---|---|---|
| `model` | MODEL | Anima model |
| `artist_pack` | ANIMA_PACK | Output from `AnimaArtistPack` |
| `preset` | ANIMA_PRESET | Output from `AnimaArtistPreset`, `AnimaArtistStarter`, or `AnimaArtistRecipeLoad` |
| `enabled` | BOOLEAN | Master switch |
| `apply_to_uncond` | BOOLEAN | Default False, **not recommended** (breaks CFG) |
| `advanced_options` | ANIMA_OPTS | Optional detailed override payload |

Outputs:
- `model`: model with the preset's artist mixing behavior applied. Connect to KSampler's `model` input
- `base_prompt`: the bare base-prompt conditioning from `artist_pack`. Connect to KSampler's positive input

### AnimaArtistCrossAttn (Manual/Advanced)

Use this node only when you want to set `combine_mode`, `fusion_mode`, and `strength` by hand. Its optional `preset` input remains available so old saved workflows continue to load, but new preset workflows should use `AnimaArtistPresetApply`.

| Parameter | Type | Description |
|---|---|---|
| `model` | MODEL | Anima model |
| `artist_pack` | ANIMA_PACK | Output from `AnimaArtistPack` |
| `combine_mode` | enum | How multiple artists are merged: `output_avg` (recommended) / `concat` / `lowrank_avg` (cross-seed-stable) |
| `fusion_mode` | enum | How merged artists act on the main prompt: `interpolate` (recommended) / `concat_with_base` / `base_preserve` |
| `strength` | FLOAT 0~4 | Overall artist contribution. 0~1 = interpolation, 1~4 = CFG-style extrapolation (style amplified) |
| `enabled` | BOOLEAN | Master switch |
| `apply_to_uncond` | BOOLEAN | Default False, **not recommended** (breaks CFG) |
| `advanced_options` | ANIMA_OPTS | Optional advanced controls |
| `preset` | ANIMA_PRESET | Compatibility input for old workflows. When connected, it overrides the visible manual fields; use `AnimaArtistPresetApply` for new preset workflows |

Outputs:
- `model`: model with artist mixing patched in. Connect to KSampler's `model` input
- `base_prompt`: the bare base-prompt conditioning from `artist_pack`. Connect to KSampler's positive input

### AnimaArtistPreset (one-click helper)

This is the one-click preset helper. Start with `balanced` for predictable artist mixing; choose `drift_auto` only when you explicitly want automatic low-drift routing. Use `prompt_passthrough` when you want the no-mixer/direct-prompt path, with positive artist weights converted into normal prompt weighting and no attention patch.

| Preset | What it does |
|---|---|
| `prompt_passthrough` | No mixer path. Builds a direct prompt from the artist chain and base prompt, converts positive artist weights to normal prompt weights like `(@artist:1.2)`, and returns the unpatched model |
| `balanced` | `output_avg + interpolate`, original-style default with EMA and norm-lock off |
| `strong_style` | Stronger style amplification with controlled extrapolation |
| `stable_seed` | `output_avg + mixed_delta_cap ratio 0.75 + strength 1.0 + auto layers 9-20`, content-safer cross-seed stability without static-capture freeze |
| `drift_auto` | Runtime route to `drift_soft`, `stable_seed`, `face_lock`, or `scene_lock` from `AnimaArtistPack.base_prompt` and artist count; Inspector reports the resolved preset and reason |
| `drift_soft` | `output_avg + light EMA + strength 0.85 + auto layers 9-20`, softer portrait / broad-subject drift control |
| `face_lock` | `output_avg + base_preserve + token/per-artist norm lock + mixed_delta_cap`, tuned for close-up faces |
| `scene_lock` | `output_avg + base_preserve + light EMA + strength 0.85 + auto layers 9-15`, tuned for explicit wide / background-heavy scenes |
| `anchor_lock` | Softer `output_avg + anchor_q + 1 seed + user blend 0.35 + strength 0.9 + auto layers 9-15` |
| `fast_preview` | `concat + concat_with_base`, fastest preview path, less precise mixing |
| `identity_guard` | `output_avg + base_preserve + norm lock + mixed_delta_cap`, protects prompt identity/composition without the very slow low-rank path |
| `compatibility_safe` | `concat + concat_with_base`, disables EMA/static/anchor paths, best first check when other nodes also patch attention |

`prompt_passthrough` supports positive artist weights only. Mixer-only controls such as negative style subtraction, per-artist layer routes, and per-artist timing routes require a real mixer preset such as `balanced`. `intensity` scales the preset's strength except for `fast_preview`, `compatibility_safe`, and `prompt_passthrough`, whose paths do not use mixer strength.

`layer_mode` gives fast layer targeting:

| layer_mode | Behavior |
|---|---|
| `auto` | Preset-specific default (`stable_seed`, `drift_soft`, and `face_lock` use `9-20`; `scene_lock` and `anchor_lock` use `9-15`) |
| `all_layers` | All layers |
| `style_core` | `0-18`, stronger global style control |
| `detail_layers` | `12-63`, more detail/brushwork focused |
| `custom` | Uses `custom_layer_filter` |

When both `preset` and `advanced_options` are connected to `AnimaArtistPresetApply`, the preset fills the base configuration and `advanced_options` overrides the detailed fields.

### AnimaArtistSimpleOptions (simple)

Use this node for common tweaks without opening the full expert panel.

| Parameter | Description |
|---|---|
| `normalize_weights` | Recommended on. Auto-bypassed when any artist uses `::weight` syntax |
| `layer_mode` | Global layer shortcut: auto / all layers / style core / detail layers / custom |
| `start_percent` / `end_percent` | Sampling progress window. Use this to reduce runtime or limit style timing |
| `custom_layer_filter` | Used only when `layer_mode=custom`. Example: `"0,3,5-10,-1"` |
| `compatibility_mode` | Forces the safer concat path when regional prompts or other attention patchers conflict |

It outputs the same `ANIMA_OPTS` type as the expert node, so wire it to `AnimaArtistPresetApply.advanced_options`.

### AnimaArtistInspector (UI report)

Connect `artist_pack`, and optionally the same `preset` / `advanced_options` used by `AnimaArtistPresetApply`. If you are not using presets, set Inspector's `combine_mode`, `fusion_mode`, and `strength` to match the manual CrossAttn node. It prints:

- parsed artist labels
- parsed linear `::weight` values
- per-artist layer routes
- per-artist timing routes
- block map showing which artists are active on which DiT blocks
- requested vs effective `normalize_weights`
- effective linear weight sum
- preset, fusion, combine, strength, layer filter, stabilizer settings
- `status`, warnings for risky or mutually-incompatible combinations, and non-blocking compatibility notes

Use this node whenever results look wrong. It catches common mistakes faster than reading console logs.

### AnimaArtistOptions (Expert)

Not connecting this node = default behavior. Prefer `AnimaArtistSimpleOptions` for normal workflows; use this expert node only for stabilizer A/B, VRAM controls, and debugging.

| Parameter | Description |
|---|---|
| `start_block` / `end_block` | Inject only on DiT blocks in `[start, end]`. `end_block = -1` means up to the last block |
| `start_percent` / `end_percent` | Inject only during sampling progress in `[start, end]`. `0.0` = sampling start, `1.0` = end |
| `normalize_weights` | True: weights are normalized to relative proportions. False: weights act as independent strength multipliers. Auto-bypassed when any artist uses `::weight` syntax |
| `layer_filter` | Advanced layer-selection string (overrides start_block/end_block). Example: `"0,3,5-10,-1"` |
| `artist_ema_alpha` | Temporal EMA on artist attention output across steps. 0 = off |
| `lowrank_k` | Low-rank truncation rank for `lowrank_avg`. 1 = most stable |
| `artist_static_capture` | Freeze artist attention after `static_capture_k` warmup steps |
| `static_capture_k` | Number of warmup steps before freezing. Default 6, range 1~12 |
| `artist_anchor_q` | Replace user-seed Q with a fixed-seed anchor's Q. The strongest cross-seed stabilizer |
| `anchor_seeds_count` | Number of anchor seeds to average. Default 1, range 1~4 |
| `anchor_user_blend` | Blend ratio between anchor Q and user Q. 0 = pure anchor, 1 = pure user |
| `anchor_deep_layer_threshold` | Use anchor for shallow layers `[0, N)`, user Q for deep layers `[N, end]`. -1 disables |
| `stabilizer_end_percent` | End point for cache-based stabilizers. `1.0` = whole sampling pass; `0.4-0.6` lets EMA/static/anchor yield during later dynamic steps |
| `anchor_base_norm_ref` (v26) | Optional A/B path for `anchor_q + match_base_norm`; uses the fixed anchor's base output as the norm reference instead of the current seed's base output. Off by default |
| `anchor_refresh_each_step` (v26) | Rebuild the fixed-seed anchor at every sampling step instead of only the first step. Slow A/B option, not a recommended default |
| `compatibility_mode` | Forces `concat + concat_with_base`, disables EMA/static/anchor stabilizers, and reduces conflict risk with regional/attention-patching nodes |
| `max_batch_artists` (v26) | Caps how many artists share one batched cross-attention forward. `0` = no cap. Set `2-8` to bound peak VRAM with many artists at high resolution instead of falling back to slow sequential mode |
| `low_vram_cache` (v26) | Stores static-capture and anchor caches in system RAM instead of VRAM. Saves hundreds of MB at high resolution for a small per-step transfer cost |
| `match_base_norm` (v26) | Enables inference-time norm locking. The artist attention output is rescaled against the base attention output so activation-energy mismatch does not compound across layers as seed-dependent style-strength swings. Default off for original-style behavior; enable for v26 stabilization A/B |
| `norm_lock_mode` (v26) | `token` (default) matches each image token's RMS to the base token, which is strongest for local style-strength stability. `row` keeps the legacy whole-row RMS behavior |
| `norm_lock_scope` (v26) | `per_artist` (default) normalizes each artist output before mixing, so one seed-specific artist spike cannot dominate the weighted average. `mixed` normalizes only the final mixed output. `both` applies both clamps and is the strongest but can make style blends more uniform |
| `contribution_balance` (v26) | Optional artist-delta equalizer. Default off; enable only when one artist repeatedly dominates after norm lock/static capture |
| `contribution_balance_alpha` (v26) | Strength for `contribution_balance`. `0` = no effect, `1` = full equalization before weights are applied |
| `mixed_delta_cap` (v26) | Optional final mixed-delta limiter. Default off; enable for A/B when the preset is mostly right but seed-to-seed composition/style swings remain too large |
| `mixed_delta_cap_ratio` (v26) | Maximum effective artist-delta RMS as a multiple of base RMS after strength is considered. `0.75-1.0` is the intended test range; lower values preserve base composition more strongly |

### AnimaArtistRecipeSave / AnimaArtistRecipeLoad (v26, sharing)

`AnimaArtistRecipeSave` merges the effective configuration (UI values + optional preset + optional advanced options) and packs it together with the artist chain into one JSON string. `AnimaArtistRecipeLoad` parses that JSON back into:

- `artist_chain` → wire to `AnimaArtistPack.artist_chain`
- `preset` (carries combine/fusion/strength/options) -> wire to `AnimaArtistPresetApply.preset`
- `advanced_options` → optional explicit payload

Unknown fields are ignored with warnings, so recipes stay loadable across versions. The JSON is paste-friendly — share exact mixes in a Discord message.

### AnimaArtistProbe + AnimaArtistProbeReport (v26, measurement)

Stop guessing `@layers` routes — measure them:

1. Wire your model through `AnimaArtistProbe` (instead of `AnimaArtistCrossAttn`) together with the same `artist_pack`.
2. Run one generation. The probe does **not** alter the image: every layer records `||artist_out − base_out|| / ||base_out||` per artist over the first `probe_steps` steps while the image generates from the base prompt only.
3. Wire `probe_id` into `AnimaArtistProbeReport` and connect any post-sampler output (e.g. the decoded IMAGE) to its `trigger` so it runs after sampling.

The report shows a per-layer bar chart per artist and suggests a concrete `artist@lo-hi` route. Artists with sharp peaks benefit most from layer routing; flat profiles mix well everywhere.

## Core concepts

### combine_mode: how multiple artists are merged

#### `output_avg` (recommended default)

Each artist runs through cross-attention **independently**, producing N outputs that are then weighted-averaged:

```
out = sum_i (w_i * cross_attn(x, K_i, V_i))
```

Each softmax is computed independently over its own K, V — artists don't compete for attention budget. Mathematically the cleanest mixing strategy. Cost: cross-attention forwards = number of artists.

#### `concat`

Concatenates artist conditionings along the token dimension:

```
K/V = [artist 1's 512 tokens, artist 2's 512 tokens, ...]
out = cross_attn(x, K_concat, V_concat)
```

Single cross-attention call, but all artists compete in the same softmax. The padding zero-vectors at the tail of LLMAdapter outputs are naturally suppressed by attention (no manual masking needed).

Pros: single forward, fast. Cons: attention is shared across artists, typically less expressive than output_avg.

#### `lowrank_avg` (v25 deterministic, cross-seed stable)

A stabilized variant of `output_avg`. Each artist's attention output minus the base output gives a delta tensor; the N delta tensors are stacked into a matrix `D ∈ ℝ^(N × M)`. v25 uses deterministic Gram eigendecomposition (`D @ D.T`) to reconstruct the top-k row-space directions before weighted-averaging:

```
delta_i = cross_attn(x, K_i, V_i) - base_out
D = stack(delta_i)
D_lowrank = topk_rowspace_project(D, k)
artist_total = base_out + sum_i (w_i * D_lowrank[i])
```

Why it stabilizes cross-seed: the seed-flipping "dominant artist" effect is largely high-rank noise in `D`. Truncating to top-k strips that noise. `k=1` (the default) projects all artists onto a single shared direction, giving maximum stability at the cost of artists looking more homogeneous. `k=2` or `3` keeps small per-artist differentiation. `k >= N` is mathematically equivalent to `output_avg` (no projection).

> Earlier versions had `mean` and `weighted_sum` modes (per-position weighted average over LLMAdapter outputs). They were removed: position-i in different artists carries different semantics, so element-wise averaging causes K/V semantic misalignment and inevitably produces broken images. A `replace` mode was also removed for discarding the main prompt's role entirely. An `embed_avg` mode briefly existed during v26 development and was cut before release after live A/B testing reproduced exactly that misalignment artifact (smeared, washed-out outputs at real resolutions).

### fusion_mode: how the merged artist acts on the main prompt

#### `interpolate` (recommended with output_avg)

Base and artist each run cross-attention once, then outputs are linearly interpolated by `strength`:

```
out = base_out * (1 - strength) + artist_out * strength
```

With `strength` in `[0, 1]`, this is strict interpolation (`strength=0` = pure base, `strength=1` = pure artist). With `strength` in `(1, 4]`, it becomes CFG-style extrapolation: `out = base_out + strength * (artist_out - base_out)` — the artist's deviation from base is amplified, producing a stronger style. Smooth transitions, minimal style drift. Cost: one extra base forward per layer.

#### `concat_with_base`

Cross-attention's K/V becomes `[base_tokens, artist_tokens]`, letting attention see both base and artist:

```
K = [K_base, K_artist]
V = [V_base, V_artist]
out = cross_attn(x, K, V)
```

The softmax decides per-pixel-position whether to attend to base or artist. With `strength < 1`, the result is mixed once more with a pure-base output.

Pros: base prompt stays in the attention computation, so prompt adherence is best preserved. Artist still dominates style, but with the lightest drift.

#### `base_preserve` (v17)

Decomposes the artist contribution into components parallel and perpendicular to the base output, and only injects the perpendicular component:

```
delta = artist_total - base_out
delta_perp = delta - proj_to_base(delta)
out = base_out + strength * delta_perp
```

The base direction is left untouched; the artist can only add a sideways offset. Mild style impact, good for keeping the main prompt in firm control. Useful when `interpolate` shifts composition more than desired.

## Cross-seed stabilizers in detail

Norm locking is off by default so `balanced` stays close to the original mixer. Enable stabilizers progressively from light to heavy.

`stabilizer_end_percent` limits only cache-based stabilizers: `artist_ema_alpha`, `artist_static_capture`, and `artist_anchor_q`. Keep it at `1.0` for the original full-pass behavior. Set it around `0.4-0.6` when a late-step sampler or post-processing workflow needs the artist attention to resume normal per-step motion after the early style direction is established.

### match_base_norm + norm lock (optional)

The weighted artist output can have a different activation energy from the base attention output expected by downstream blocks. That mismatch can compound across layers and show up as seed-dependent style-strength swings: one seed looks washed out, another overpowered, another balanced.

`match_base_norm` keeps the artist direction but rescales its RMS energy toward the base output. When enabled, the recommended v26 configuration is:

```text
match_base_norm = True
norm_lock_mode  = token
norm_lock_scope = per_artist
```

`token` mode matches each image token instead of one whole batch row. `per_artist` scope applies the lock before artists are mixed, which prevents a single artist's seed-specific high-energy response from dominating the average. The per-artist downscale has no 0.5 floor, so extreme spikes can be suppressed strongly; upscaling is still capped at 2.0x to avoid amplifying weak noise. Use `row + mixed` only when you need legacy v26.0 behavior for comparison.

### mixed_delta_cap (experimental A/B guard)

`mixed_delta_cap` limits the final mixed artist delta before `interpolate` or `base_preserve` fusion:

```text
effective_delta = artist_total - base_out
max_delta_rms   = base_rms * mixed_delta_cap_ratio / strength
```

For `base_preserve`, the limiter measures the perpendicular delta that will actually be injected. This is an inference-time control: it does not train, fine-tune, or add a LoRA. It is intended for the difficult case where static capture gives the right general look, but some seeds still push the face, pose, or background too far. Start with `mixed_delta_cap_ratio=1.0`, then try `0.75` if composition drift is still visible. Keep it off when you intentionally want strong artist override.

### artist_ema_alpha (lightest)

Applies a per-layer exponential moving average to the artist attention output across sampling steps:

```
artist_total_t = alpha * artist_total_{t-1} + (1 - alpha) * artist_total_t
```

Reasoning: cross-step jitter in artist attention often comes from the dominant artist flipping under shifting cross-attn QK match. EMA smooths this over time. Effective mainly on `interpolate` and `base_preserve` (not `concat_with_base`, where the artist isn't isolated).

Range 0~0.95. 0.3~0.5 is a light effect, 0.5~0.8 is medium-heavy. Higher values can lag the style behind the base content.

Cache resets when sigma jumps up (i.e. a new sampling run begins).

### lowrank_avg + lowrank_k (medium)

See `combine_mode = lowrank_avg` above. The low-rank constraint is permanent (every step), unlike EMA which only smooths over time. More aggressive but more uniform-feeling result.

### artist_static_capture + static_capture_k (heavy, also a perf win)

During the first K sampling steps, the artist attention output is computed and accumulated. After step K, the average is **frozen** and reused for all remaining steps. Subsequent steps skip the artist cross-attention entirely.

Why this works: after a few denoising steps, the user hidden state has stabilized enough that further per-step recomputation of artist attention only adds jitter without adding style. Freezing decouples style from later-step content fluctuations.

Side benefit: steps after K skip N artist cross-attention forwards each, giving 30~50% wall-clock speedup at typical K=6, N=5~7.

`K` ranges 1~12, default 6. Lower K = earlier freeze = more stable but earlier commitment to a possibly-suboptimal style estimate. Higher K = later freeze = closer to non-frozen behavior.

Cache resets on sigma jump. Mutually exclusive with EMA (which becomes a no-op once frozen).

### artist_anchor_q (heaviest cross-seed stabilizer)

One major source of cross-seed style drift is that the cross-attn Q comes from base hidden state, which is seed-driven. `artist_anchor_q` reduces that source by **replacing the artist-attention Q source with a fixed-seed anchor's hidden state**:

1. On first invocation, the plugin runs a single-step "anchor pass" using a fixed seed (default 42) with the user's prompt context
2. Each layer's pre-cross-attn hidden state is captured during this anchor pass and cached
3. During real sampling, when computing artist attention, Q is sourced from the anchor's cached hidden state instead of the user's current hidden state
4. The base attention still uses user Q (so base content adapts to user seed normally)

Result: the artist-attention Q path is fixed for the same prompt + resolution. Final images can still vary through the base branch and downstream nonlinear blocks, so this reduces style drift substantially but does not make all seeds identical.

Cache key is `(x.shape, context fingerprint, first_timestep)` — same prompt + same resolution + same initial sampling condition reuses the anchor for free across seeds. Different prompt, resolution, conditioning content, or initial timestep triggers a fresh anchor pass.

First-time cost: ~1 extra step worth of forward time for the anchor pass. After that, zero overhead per seed.

**Sub-options for finer control**:

- `anchor_seeds_count` (1~4, default 1): runs N anchor passes with different fixed seeds and averages their hidden states. Mitigates the small chance that a single fixed seed produces a systematically biased anchor. Cost scales linearly with N.
- `anchor_user_blend` (0~1, default 0): blends anchor Q with user Q. 0 = pure anchor (max stability), 1 = pure user (equivalent to disabling anchor). Useful if pure anchor produces brushwork that looks slightly disconnected from the actual content.
- `anchor_deep_layer_threshold` (-1~64, default -1 = disabled): when set to N, layers `[0, N)` use anchor Q (style stability) while layers `[N, end]` use user Q (content fidelity). Based on the principle that early DiT blocks set style and late blocks add detail.
- `anchor_base_norm_ref` (default off): if norm locking is enabled, match against the fixed anchor's base output instead of the current seed's base output. Useful for A/B, not a recommended default.
- `anchor_refresh_each_step` (default off): rebuilds the anchor for each sampling timestep. This is slower than the cached first-step anchor path, so keep it off unless you are testing.

Mutually exclusive with `artist_static_capture` (anchor takes priority, with a warn log).

### stable_seed (opt-in)

The current `stable_seed` preset uses `output_avg + mixed_delta_cap + mixed_delta_cap_ratio=0.75 + match_base_norm=False + strength=1.0 + layer_filter=9-20` when `layer_mode=auto`.
This keeps the real style-mixer path active while capping extreme mixed deltas. It avoids the static-capture freeze that can make multi-artist results look washed out or collapse artist differences.

`static_capture_mode` is an advanced A/B control, not a default tuning knob. `output` is the conservative default. `delta` preserves current base motion more directly, `blend` interpolates between the frozen output and delta path, and `blend_perp` only reintroduces base motion that is perpendicular to the frozen style delta. These modes are experimental and should be checked against the actual prompt and artist set.

### Scene-tuned low-drift presets

There is no single low-drift setting that wins across every prompt shape. Keep `balanced` as the default artist mixer; use `stable_seed`, `drift_auto`, and the scene-tuned variants only when you explicitly want lower drift over original-style behavior:

| Preset | Use when | Difference from `stable_seed` |
|---|---|---|
| `drift_auto` | You want the node to pick the common low-drift route from `base_prompt` and artist count | resolves at runtime to `drift_soft`, `stable_seed`, `face_lock`, or `scene_lock` and reports the reason in Inspector. 4+ artist broad prompts stay on mixer presets instead of compatibility concat, so 4-artist and 10-artist inputs remain style-controllable |
| `drift_soft` | Portrait or broad-subject prompts where full strength changes the look too much | lowers strength to `0.85` and adds light EMA without static capture |
| `face_lock` | Close-up faces where identity / facial detail shifts between seeds | turns token/per-artist `match_base_norm` on, uses `base_preserve`, and caps the mixed delta |
| `scene_lock` | Explicit wide-shot, small-figure, cityscape, landscape, or background-heavy prompts where composition should stay in charge | switches fusion to `base_preserve`, adds light EMA, and narrows auto layers to `9-15` |

These presets are inference-time controls, not training. They reduce measured descriptor drift in the prompt types they target, but they should still be checked with your actual artist set and prompt because artist tags can carry composition and subject bias as well as style.

## Recommended combinations

In v26, use `AnimaArtistStarter` for new workflows, or `AnimaArtistPreset` when you only need the preset payload:

| Goal | Preset |
|---|---|
| normal use | `balanced` |
| stronger visual style | `strong_style` |
| content-safer cross-seed work | `stable_seed` |
| lower drift without hand-picking the scene type | `drift_auto` |
| portrait / broad subject with lower drift | `drift_soft` |
| 4+ artist plain portrait / street with lower drift while preserving style control | `drift_auto` (`drift_soft` route) |
| close-up faces with lower drift | `face_lock` |
| explicit wide / background-heavy scenes with lower drift | `scene_lock` |
| softer fixed-anchor seed lock | `anchor_lock` |
| fast exploration | `fast_preview` |
| preserve character/object identity | `identity_guard` |
| regional prompts / other attention patchers | `compatibility_safe` |

Manual equivalents:

### Daily use, no stabilizers

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
```

### Light cross-seed improvement

```
combine_mode      = output_avg
fusion_mode       = interpolate
strength          = 1.0
artist_ema_alpha  = 0.4
```

### Strong cross-seed stability

```
combine_mode      = output_avg
fusion_mode       = interpolate
strength          = 1.0
artist_anchor_q   = True
anchor_seeds_count = 4
match_base_norm   = False
```

### Maximum cross-seed stability (production)

```
combine_mode      = output_avg
fusion_mode       = interpolate
strength          = 1.5
artist_anchor_q   = True
anchor_seeds_count = 1
anchor_user_blend = 0.0
```

To control individual artist strength within the chain, use either weight syntax inside `artist_chain`:

```
wlop, 1.2::sakimichan::, (krenz:0.7)
```

## Performance notes

### Computational cost

In `output_avg` and `lowrank_avg` modes, each layer runs `N + 1` cross-attention forwards (N artists + base). This is mathematical necessity:

```
sum_i (w_i * softmax(Q @ K_i^T / √d) @ V_i)
```

Each softmax must be computed independently over its own K, V. Merging into a single large attention would degrade the semantics to `concat` mode.

### Timing expectations

Single-run wall time depends on GPU, resolution, sampler, model cache state, VAE decode, and queue load. Treat one-off timings as rough signals, not benchmarks.

| Path | What to expect |
|---|---|
| no mixer / `prompt_passthrough` | close to the ordinary Anima prompt path |
| `balanced`, `drift_soft`, `scene_lock`, `face_lock`, `stable_seed` | slower as active artist count increases |
| `fast_preview` / `compatibility_safe` | usually faster concat path, but less precise as an artist mixer |
| layer routes / timing routes | reduce cost only for the layers or sampling steps where artists are inactive |
| `artist_static_capture` / `artist_anchor_q` | expert A/B controls; may help some repeated-seed workflows but can add overhead or constrain style |

### Recommended speed controls

After connecting `AnimaArtistSimpleOptions` or `AnimaArtistOptions`, reduce the amount of active mixer work first:

- **Layer range** (`start_block / end_block` or `layer_filter`): inject only on selected DiT blocks.
- **Sampling-step range** (`start_percent / end_percent`): inject only during part of sampling.
- **Per-artist routes** (`artist@layers%timing`): keep each artist active only where it is useful.

These controls reduce work only when they actually make artists inactive for some layers or steps. Quality impact is prompt- and artist-dependent, so compare images before treating a faster setting as equivalent.

## How to write the artist chain

### Recommended format: artist on top, main prompt separate

The two text boxes of `AnimaArtistPack` have distinct roles:

```
artist_chain (top box):
  wlop
  1.2::sakimichan::
  (krenz:0.7)

base_prompt (bottom box):
  masterpiece, 1girl, standing, in a forest, ...
```

Internally the node concatenates each as `<artist_name>\n<base_prompt>` before encoding — Anima's empirically most stable format. You don't need to repeat artist names in the main prompt.

### Two layers of weighting

There are **two independent** weighting points:

1. **CLIP weighting** (`(name:1.2)` syntax): scales token embeddings before they pass through the LLMAdapter (a non-linear 6-layer transformer). Outcome isn't strictly predictable but stays close to the LLM's natural output distribution. Same as SD/A1111 syntax.
2. **Injection-layer weighting** (`1.5::name::` syntax): scales the artist's contribution at the cross-attention output stage. Linear and predictable: `2.0::name::` makes that artist's relative contribution exactly twice as strong as a default-weight artist.

They can be **stacked**: `0.8::(wlop:1.1)::` applies CLIP weight 1.1 first, then injection weight 0.8.

When any artist uses `::weight` syntax, `normalize_weights` is automatically bypassed at runtime (the explicit weights are honored as-is).

Older postfix forms such as `::name::1.2`, `name::1.2`, and `::(name:1.1)::0.8` still load for backward compatibility. New workflows should use prefix syntax because it composes cleanly with routes:

```
1.2::@artist_a@0-8::
1.2::@artist_a@0-8%0.0-0.45~0.1::
```

Do not split the route outside the weighted target. Use `1.2::@artist_a@0-8::`, not `1.2::@artist_a::@0-8`.

#### Negative weights: style subtraction (v26)

Injection weights accept negative values in `[-4, 0)`:

```
wlop, -0.4::pixiv_generic::
```

A negative-weight artist's attention output is subtracted from the mix instead of added, pushing the result **away** from that style direction. Practical uses:

- de-bias a strong default look: mix your target artists positively and subtract a generic style tag lightly (`-0.2 ~ -0.5`)
- sharpen contrast between two artists by subtracting a third that sits "between" them

Notes: keep at least one positive artist in the chain; subtraction magnitudes above ~1.0 destabilize quickly; under `combine_mode = concat` negative weights scale raw conditioning tokens, which is much less meaningful — use `output_avg` or `lowrank_avg` for subtraction work. The Inspector and Chain Preview flag negative weights so you can confirm intent.

Global artist contribution is controlled by `AnimaArtistCrossAttn`'s `strength` (independent of per-artist weights).

### Per-artist layer routing

Add `@layer_filter` at the end of an artist entry to make that artist active only on selected DiT blocks:

```
wlop@0-8
1.2::sakimichan@9-18::
0.8::(krenz:1.1)@19-27::
```

Artist tags that already start with `@` still work. For example, `@wlop` is treated as the artist name, while `@wlop@0-8` means artist `@wlop` routed to blocks `0-8`. The parser only treats the final `@...` segment as a route when it contains layer-filter characters (`0-9`, comma, dash, decimal point, percent sign, spaces, or Chinese comma).

Layer filters use the same syntax as `AnimaArtistOptions.layer_filter`: comma-separated indices, ranges, and negative indices. They also accept layer-percent windows. Use `0%-33%` / `33%-67%` / `67%-100%` when you want model-independent windows, or normalized decimals like `0.33-0.67` when combining with sampling timing. Examples:

```
0-8
9,12,15
14-27,-1
0%-33%
0.33-0.67
```

Comma-separated layer routes are kept inside the artist entry, so `wlop@0,2,4, hiten` parses as two artists: `wlop` routed to blocks `0,2,4`, then `hiten`. Newlines always split artists and are the clearest format for complex chains.

This solves the "different artists mixed into different layers" workflow:

- early blocks (`0-8`): composition and global style
- middle blocks (`9-18`): character/body/shape bias
- late blocks (`19-27`): details, finish, brushwork

For the GitHub-requested character / clothing / background split, use one
artist per role:

```
@background_artist@0%-33%
@character_artist@0.33-0.67
@clothing_artist@67%-100%
```

If a layer has no matching artist after routing, that layer falls back to the original cross-attention. Global `layer_filter` still applies first: the node only patches the selected global layers, then per-artist routes decide which artists participate inside those patched layers.

### Per-artist sampling timing

Add `%start-end` after an artist entry to make that artist active only during a sampling-progress window:

```
wlop%0.0-0.45
krenz%0.45-0.85
hiten%0.75-1.0
```

Layer routing and timing can be combined. Put layer routing first, timing last:

```
wlop@0-8%0.0-0.45
1.2::krenz@9-18%0.35-0.85::
hiten@19-27%0.65-1.0
@background_artist@0.00-0.33%0.0-0.45
```

The timing range is normalized sampling progress:

- `0.0` = sampling start, highest noise
- `0.5` = middle of the denoising trajectory
- `1.0` = sampling end, final detail pass

This enables scheduled artist roles from one artist chain: one artist can shape early composition, another can dominate the middle structure, and another can add late brushwork. If the current layer and current sampling progress have no matching artist, that layer falls back to original cross-attention for that step.

Per-artist timing is independent from global `start_percent / end_percent`. Global timing still applies first; per-artist timing decides which artists participate inside the globally active window.

### Timing fade (v26)

A hard timing window switches an artist on/off between two adjacent steps, which can produce a visible style "pop". Append `~fade` to ramp the artist's weight smoothly (smoothstep) over a progress-wide edge on both sides of the window:

```
wlop%0.0-0.45~0.1
1.2::krenz@9-18%0.35-0.85~0.08::
```

`wlop%0.0-0.45~0.1` means: weight ramps 0→1 over progress `0.0-0.1`, stays at full weight until `0.35`, then ramps 1→0 over `0.35-0.45`. The fade is clamped to at most half the window. With `normalize_weights` on, a fading artist's share is smoothly redistributed to the other active artists — exactly the crossfade you want for scheduled chains. `~0` (or omitting `~fade`) reproduces the old hard-switch behavior.

### Important note on heavy weights

Do **not** push both `strength` and multiple `::weight` values high at once when you have many artists. Cross-attn output magnitude is roughly `strength * sum(::weight)`, which becomes unstable past ~6-8x baseline. Symptoms: oversaturation, noise patches, or fully broken output. Default values (`strength=1`, no `::weight`) are always safe.

### When `normalize_weights = False`

Default `normalize_weights = True`: in `output_avg`, N artists' weights are normalized to `1/N` each, so **total contribution always equals 1**.

With normalization off and no `::weight` used: each artist contributes at its raw weight (default 1.0). **Total contribution = N**, which exceeds the model's training distribution and produces pure noise.

The node intercepts dangerous configurations when `::weight` is not in use:

| Artist count + normalize=False | Behavior |
|---|---|
| 1 artist | Normal (equivalent to normalized) |
| 2~3 artists | Warning, but allowed (may overexpose) |
| 4+ artists | **Hard error**, with three suggested fixes |

When `::weight` is used, v25 judges risk by the **actual sum of absolute linear weights**, not by artist count. Four artists written as `0.25::artist::` each are valid because the total is still 1.0.

If you actually want "one artist weakened", the recommended approach is to use injection weighting `0.3::name::` to lower a specific artist:

```
wlop, 0.3::krenz::
```

This keeps wlop dominant with a krenz accent, without breaking total-contribution stability.

## Advanced layer/step controls

### Layer range (`start_block` / `end_block`)

DiT blocks at different depths correspond to different semantic levels: early blocks affect overall composition and style, later blocks affect detail and texture. For example:

- `0..13` (front half): artist dominates composition; details are filled in by the model
- `14..27` (back half): only inject into detail layers; composition follows the main prompt

### `layer_filter` (more flexible layer selection)

A string with **higher priority than `start_block / end_block`** (overrides them when set). Syntax:

- Comma-separated indices: `"0,3,7"`
- Ranges with hyphen: `"5-10"`
- Negative indices (counted from end): `"-1"` = last block
- Mix: `"0,3,5-10,-1"`

Useful for non-contiguous patterns like "early + last only" or interval injection experiments.

### Sampling-step range (`start_percent` / `end_percent`)

Different sampling stages determine different image content (high sigma = composition, low sigma = texture refinement). For example:

- `0.0..0.5`: inject only in the first half; artist sets the overall layout, then the model is free to refine details
- `0.3..1.0`: skip the very early steps to avoid the artist pushing composition too hard

Implementation detail: the node uses `set_model_unet_function_wrapper` to capture the current sigma at each `apply_model` call, then maps user-set percent ranges to sigma ranges via `model_sampling.percent_to_sigma()`.

## Known issues

### `model_function_wrapper` chain conflicts

When sampling-step range or any cross-seed stabilizer is enabled, this node uses `set_model_unet_function_wrapper` to capture per-step sigma. The implementation is chain-safe — it preserves and forwards calls to any pre-existing wrapper.

However, if another custom node connected **after** this one sets a wrapper without chain-safety (overwriting blindly), the sigma capture is lost, and step-range / EMA-reset / static-capture-reset all silently degrade.

Diagnosis: reset `start_percent / end_percent` to 0.0 / 1.0 and disable all stabilizers; if behavior recovers, another node is overwriting the wrapper.

### Cross-attention patch conflicts

This node replaces `diffusion_model.blocks[i].cross_attn` through ComfyUI object patches. Nodes that also rewrite the same cross-attention modules can interact in non-obvious ways:

- regional prompting / area composition nodes
- Forge Couple-style prompt routing ports
- attention replacement or attention masking nodes
- other custom nodes that patch `diffusion_model.blocks.*.cross_attn`

Symptoms:

- artist effect becomes weak or disappears
- batched artist path falls back to serial mode
- `output_avg` looks much weaker than `concat`
- cache-based stabilizers stop matching across passes
- per-artist timing or global step-range looks ignored

Practical fixes:

- try `AnimaArtistPreset(preset=compatibility_safe)` first; it is more tolerant when regional prompts dominate
- disable `artist_static_capture` and `artist_anchor_q` while debugging compatibility
- use `AnimaArtistInspector` to confirm the parsed artists, weights, per-artist layer routes, timing routes, and block map
- simplify the workflow until this node is the only cross-attention patcher, then add other nodes back one at a time

Runtime diagnostics:

- suspicious existing cross-attention wrappers are logged before this node patches the target blocks
- existing `model_function_wrapper` chains are logged when this node also needs sigma capture for step ranges, timing routes, EMA, static capture, or anchor-Q
- these warnings do not mean the workflow is broken; they mean you should test `compatibility_safe` before assuming the artist chain is bad

## Acknowledgements

Special thanks to **汐浮尘** for co-development, testing, and design contributions during the development of this node. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

MIT License. See [LICENSE](LICENSE) for the full text.
