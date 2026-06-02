# Anima-Artist-Mixer

## Introduction

This is a ComfyUI custom node that provides **multi-artist mixing** for the Anima model. It hooks into the cross-attention layers and combines multiple artist conditionings with controllable strategies, sidestepping the interference that LLM-based text encoders suffer from when multiple artist tags coexist in a single prompt.

The companion `AnimaArtistPack` node provides a one-shot experience: write your artist list in one text box (comma or newline separated) and your main prompt in another. The node automatically splits, encodes, and packages everything for downstream use.

This README documents the **v25 architecture**, which adds one-click presets, an in-UI inspector, deterministic low-rank mixing, safer explicit weights, layered cross-seed stabilization (EMA / low-rank / static-capture / anchor-Q), CFG-style strength extrapolation, and the linear injection-layer weight syntax `::name::weight`. Older versions are still functionally a subset.

v25.1 also adds per-artist layer routing, matching the original repository's first public feature request: different artists can now be injected into different DiT block ranges from the same artist chain.

v25.2 adds per-artist sampling timing, a `compatibility_safe` preset, Inspector block maps, runtime warnings for suspicious cross-attention / model-wrapper conflicts, and UX helper nodes for building or previewing artist chains before CLIP encoding.

## What problem it solves

Anima uses an LLM as its text encoder (unlike SDXL's CLIP). LLM encoders are heavily **contextualized** — every token's embedding fuses semantics from surrounding tokens. This has a direct consequence:

- Single artist tag: the LLM produces a conditioning that captures that artist's style accurately. Works well.
- Multiple artist tags together: the artist tags' embeddings interfere with each other, and the resulting conditioning ends up looking like neither A nor B but a "squeezed-together" middle ground.

This node encodes each artist as a **separate** conditioning, bypassing the interference at the encoding stage, then mixes them inside the model at the cross-attention level using selectable strategies. Mixing happens in an already-stabilized feature space, where it's far more controllable than mixing at the prompt level.

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

The node replaces `diffusion_model.blocks[i].cross_attn` with a wrapper using ComfyUI's `add_object_patch` API. This is clone-safe — it doesn't pollute the original model and is automatically undone when the workflow disconnects.

Each artist conditioning is lazily run through the LLMAdapter on its first forward call (when the model is already on GPU), producing a `(1, 512, 1024)` processed embedding that's cached for reuse across sampling steps.

Each layer's injection is wrapped in exception isolation: if a single layer's injection fails, only that layer falls back to the original cross-attention; other layers continue working normally.

### CFG compatibility

ComfyUI batches cond and uncond into a single `batch=2` forward, with `transformer_options["cond_or_uncond"]` marking each row. This node injects only into the cond rows by default; uncond rows keep their original base context, so CFG guidance is preserved naturally. `apply_to_uncond` defaults to False and is not recommended to enable.

## The cross-seed instability problem (and how v25 addresses it)

In multi-artist setups, the same prompt with different seeds tends to produce **noticeably different style mixes** — sometimes wlop dominates, other times sakimichan does, even though their weights are equal. This is structural, not a bug:

- The cross-attn Q comes from base hidden state, which is seed-driven
- For each seed, attention picks slightly different artist token weights
- Across seeds, the "dominant artist" can flip

v25 layers four optional stabilizers, ordered from light to heavy (configured in `AnimaArtistOptions` or selected through `AnimaArtistPreset`):

1. **artist_ema_alpha**  — temporal EMA smoothing across sampling steps
2. **combine_mode = lowrank_avg + lowrank_k**  — deterministic low-rank constraint on multi-artist deltas
3. **artist_static_capture + static_capture_k**  — freeze artist attention after the first K steps
4. **artist_anchor_q**  — replace user-seed Q with a fixed-seed anchor's Q (most aggressive, ~fully decouples cross-seed)

These can be combined freely. None of them are on by default — leaving them off gives the original v17-equivalent behavior.

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
                          ┌──► artist_pack ──► AnimaArtistCrossAttn ──► MODEL ──► KSampler
[Load CLIP] ─► CLIP ──────┤                              │                          │
                          │                              └──► base_prompt ──► (positive)
                          │
                          └──► CLIPTextEncode (Negative) ──► (negative)

[Load Anima Model] ──► MODEL ──► AnimaArtistCrossAttn

(optional) AnimaArtistChainBuilder ──► artist_chain ──► AnimaArtistPack
(optional) AnimaArtistChainPreview ──► cleaned_chain / syntax report
(optional) AnimaArtistPreset  ──► preset ────────────► AnimaArtistCrossAttn
(optional) AnimaArtistOptions ──► advanced_options ──► AnimaArtistCrossAttn
(optional) AnimaArtistInspector ◄── artist_pack / preset / advanced_options
```

Key points:
- Use `AnimaArtistChainBuilder` for the fastest safe setup: enter up to three artists, pick a layout, then connect its `artist_chain` output into `AnimaArtistPack`
- Use `AnimaArtistChainPreview` when hand-writing chains; it catches syntax mistakes before CLIP encoding
- Write your artist chain in `AnimaArtistPack`'s top text box (comma or newline separated)
- Write your main prompt in the bottom text box
- Connect `AnimaArtistCrossAttn`'s `base_prompt` output directly to KSampler's positive input
- Encode the negative prompt independently with `CLIPTextEncode`; it does not go through this plugin
- Start with `AnimaArtistPreset(preset=balanced)` unless you already know which advanced settings you want
- Use `AnimaArtistPreset(preset=compatibility_safe)` first when combining with regional prompts, Forge Couple-style routing, attention masks, or other cross-attention patch nodes
- Advanced controls (layer range, sampling-step range, stabilizers) come via the optional `AnimaArtistOptions` node
- Use `AnimaArtistInspector` to show the actual effective weights, block map, preset settings, and configuration warnings inside ComfyUI

## Parameters

### AnimaArtistChainBuilder (UX helper)

This is the easiest way to build a correct chain without memorizing syntax. It outputs a ready-to-connect `artist_chain` string plus a preview report.

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
| `artist_chain` | STRING (multiline) | Artist chain. Comma or newline separated. Supports CLIP weighting `(wlop:1.2)`, injection-layer weight `::wlop::1.5`, per-artist layer routing `@0-8`, and per-artist timing `%0.0-0.45` |
| `base_prompt` | STRING (multiline, optional) | Main prompt. Leave empty to encode artists alone |

Outputs `ANIMA_PACK`, an internal struct holding each artist's separately-encoded conditioning, the artist label list, the parsed per-artist weights, and a separately-encoded conditioning for the bare base prompt.

How it works internally: the node splits `artist_chain` into N artist names, parses any `::name::weight` syntax to extract per-artist injection weights (which are stripped before CLIP encoding), and encodes each as `<artist_name>\n<base_prompt>` (Anima's recommended format: artist first, newline, then main prompt). It also encodes a clean copy of `base_prompt` alone for use as KSampler's positive conditioning.

### AnimaArtistCrossAttn (main node)

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
| `preset` | ANIMA_PRESET | Optional one-click preset. When connected, it overrides `combine_mode`, `fusion_mode`, `strength`, then `advanced_options` can still override detailed options |

Outputs:
- `model`: model with artist mixing patched in. Connect to KSampler's `model` input
- `base_prompt`: the bare base-prompt conditioning from `artist_pack`. Connect to KSampler's positive input

### AnimaArtistPreset (one-click helper)

This is the recommended entry point for new workflows. It outputs both `ANIMA_PRESET` and `ANIMA_OPTS`.

| Preset | What it does |
|---|---|
| `balanced` | `output_avg + interpolate`, light EMA. Best default |
| `strong_style` | Stronger style amplification with controlled extrapolation |
| `stable_seed` | `lowrank_avg + static_capture`, prioritizes cross-seed consistency |
| `fast_preview` | `concat + concat_with_base`, fastest preview path, less precise mixing |
| `identity_guard` | `lowrank_avg + base_preserve`, protects prompt identity/composition |
| `compatibility_safe` | `concat + concat_with_base`, disables EMA/static/anchor paths, best first check when other nodes also patch attention |

`intensity` scales the preset's strength except for `fast_preview` and `compatibility_safe`, whose concat paths do not use strength.

`layer_mode` gives fast layer targeting:

| layer_mode | Behavior |
|---|---|
| `auto` / `all_layers` | All layers |
| `style_core` | `0-18`, stronger global style control |
| `detail_layers` | `12-63`, more detail/brushwork focused |
| `custom` | Uses `custom_layer_filter` |

When both `preset` and `advanced_options` are connected to `AnimaArtistCrossAttn`, the preset fills the base configuration and `advanced_options` overrides the detailed fields.

### AnimaArtistInspector (UI report)

Connect `artist_pack`, and optionally the same `preset` / `advanced_options` used by `AnimaArtistCrossAttn`. If you are not using presets, set Inspector's `combine_mode`, `fusion_mode`, and `strength` to match the CrossAttn node. It prints:

- parsed artist labels
- parsed linear `::weight` values
- per-artist layer routes
- per-artist timing routes
- block map showing which artists are active on which DiT blocks
- requested vs effective `normalize_weights`
- effective linear weight sum
- preset, fusion, combine, strength, layer filter, stabilizer settings
- warnings for risky or mutually-incompatible combinations

Use this node whenever results look wrong. It catches common mistakes faster than reading console logs.

### AnimaArtistOptions (advanced)

Not connecting this node = default behavior. Connecting it makes its settings take effect.

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
| `compatibility_mode` | Forces `concat + concat_with_base`, disables EMA/static/anchor stabilizers, and reduces conflict risk with regional/attention-patching nodes |

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

> Earlier versions had `mean` and `weighted_sum` modes (per-position weighted average over LLMAdapter outputs). They were removed: position-i in different artists carries different semantics, so element-wise averaging causes K/V semantic misalignment and inevitably produces broken images. A `replace` mode was also removed: it discards the main prompt's role in cross-attention entirely, severely degrading prompt adherence.

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

All four stabilizers are off by default. Enable progressively from light to heavy.

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

### artist_anchor_q (heaviest, true cross-seed decoupling)

Root cause of cross-seed style drift: the cross-attn Q comes from base hidden state, which is seed-driven. v23 fully addresses this by **replacing the Q's source with a fixed-seed anchor's hidden state**:

1. On first invocation, the plugin runs a single-step "anchor pass" using a fixed seed (default 42) with the user's prompt context
2. Each layer's pre-cross-attn hidden state is captured during this anchor pass and cached
3. During real sampling, when computing artist attention, Q is sourced from the anchor's cached hidden state instead of the user's current hidden state
4. The base attention still uses user Q (so base content adapts to user seed normally)

Result: artist attention is identical across all seeds for the same prompt + resolution. Cross-seed style drift drops to near-zero.

Cache key is `(x.shape, id(context), first_timestep)` — same prompt + same resolution + same initial sampling condition reuses the anchor for free across seeds. Different prompt, resolution, or initial timestep triggers a fresh anchor pass.

First-time cost: ~1 extra step worth of forward time for the anchor pass. After that, zero overhead per seed.

**Sub-options for finer control**:

- `anchor_seeds_count` (1~4, default 1): runs N anchor passes with different fixed seeds and averages their hidden states. Mitigates the small chance that a single fixed seed produces a systematically biased anchor. Cost scales linearly with N.
- `anchor_user_blend` (0~1, default 0): blends anchor Q with user Q. 0 = pure anchor (max stability), 1 = pure user (equivalent to disabling anchor). Useful if pure anchor produces brushwork that looks slightly disconnected from the actual content.
- `anchor_deep_layer_threshold` (-1~64, default -1 = disabled): when set to N, layers `[0, N)` use anchor Q (style stability) while layers `[N, end]` use user Q (content fidelity). Based on the principle that early DiT blocks set style and late blocks add detail.

Mutually exclusive with `artist_static_capture` (anchor takes priority, with a warn log).

## Recommended combinations

In v25, use `AnimaArtistPreset` first:

| Goal | Preset |
|---|---|
| normal use | `balanced` |
| stronger visual style | `strong_style` |
| same prompt across many seeds | `stable_seed` |
| fast exploration | `fast_preview` |
| preserve character/object identity | `identity_guard` |

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

### Strong cross-seed stability + style amplification

```
combine_mode      = lowrank_avg
lowrank_k         = 1
fusion_mode       = interpolate
strength          = 2.0
artist_static_capture = True
static_capture_k  = 6
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
wlop, ::sakimichan::1.2, (krenz:0.7)
```

## Performance notes

### Computational cost

In `output_avg` and `lowrank_avg` modes, each layer runs `N + 1` cross-attention forwards (N artists + base). This is mathematical necessity:

```
sum_i (w_i * softmax(Q @ K_i^T / √d) @ V_i)
```

Each softmax must be computed independently over its own K, V. Merging into a single large attention would degrade the semantics to `concat` mode.

### Approximate timing (30 steps, varies by GPU)

| Configuration | Relative time |
|---|---|
| 1 artist | 1.0x (baseline) |
| 4 artists | ~1.4x |
| 8 artists | ~1.7x |
| 5 artists + `artist_static_capture` (K=6) | ~1.1x |
| 5 artists + `artist_anchor_q` (cached, 2nd seed onward) | ~1.05x |

**More artists means more time** — but `artist_static_capture` and `artist_anchor_q` largely amortize this away after the warmup steps.

### Strongly recommended: use layer range and step range to reduce cost

After connecting `AnimaArtistOptions`, you can **dramatically cut generation time** with usually minimal quality loss:

- **Layer range** (`start_block / end_block` or `layer_filter`): inject only on specific DiT blocks. `0..13` (front half) cuts time roughly in half. Artist style is mostly determined by early blocks, so the loss is usually acceptable
- **Sampling-step range** (`start_percent / end_percent`): inject only during a portion of sampling. `0.0..0.5` (first half) similarly cuts time, since artist style is mostly absorbed during early sampling

Both can be **combined**: "front-half layers + front-half sampling" can bring 8-artist scenarios back to near-single-artist timing. This is the most effective optimization for multi-artist setups, and stacks with `artist_static_capture` / `artist_anchor_q`.

## How to write the artist chain

### Recommended format: artist on top, main prompt separate

The two text boxes of `AnimaArtistPack` have distinct roles:

```
artist_chain (top box):
  wlop
  ::sakimichan::1.2
  (krenz:0.7)

base_prompt (bottom box):
  masterpiece, 1girl, standing, in a forest, ...
```

Internally the node concatenates each as `<artist_name>\n<base_prompt>` before encoding — Anima's empirically most stable format. You don't need to repeat artist names in the main prompt.

### Two layers of weighting

There are **two independent** weighting points:

1. **CLIP weighting** (`(name:1.2)` syntax): scales token embeddings before they pass through the LLMAdapter (a non-linear 6-layer transformer). Outcome isn't strictly predictable but stays close to the LLM's natural output distribution. Same as SD/A1111 syntax.
2. **Injection-layer weighting** (`::name::1.5` syntax, v24): scales the artist's contribution at the cross-attention output stage. Linear and predictable: `::name::2.0` makes that artist's relative contribution exactly twice as strong as a default-weight artist.

They can be **stacked**: `::(wlop:1.1)::0.8` applies CLIP weight 1.1 first, then injection weight 0.8.

When any artist uses `::weight` syntax, `normalize_weights` is automatically bypassed at runtime (the explicit weights are honored as-is).

Global artist contribution is controlled by `AnimaArtistCrossAttn`'s `strength` (independent of per-artist weights).

### Per-artist layer routing

Add `@layer_filter` at the end of an artist entry to make that artist active only on selected DiT blocks:

```
wlop@0-8
::sakimichan::1.2@9-18
::(krenz:1.1)::0.8@19-27
```

Artist tags that already start with `@` still work. For example, `@wlop` is treated as the artist name, while `@wlop@0-8` means artist `@wlop` routed to blocks `0-8`. The parser only treats the final `@...` segment as a route when it contains layer-filter characters (`0-9`, comma, dash, spaces, or Chinese comma).

Layer filters use the same syntax as `AnimaArtistOptions.layer_filter`: comma-separated indices, ranges, and negative indices. Examples:

```
0-8
9,12,15
14-27,-1
```

Comma-separated layer routes are kept inside the artist entry, so `wlop@0,2,4, hiten` parses as two artists: `wlop` routed to blocks `0,2,4`, then `hiten`. Newlines always split artists and are the clearest format for complex chains.

This solves the "different artists mixed into different layers" workflow:

- early blocks (`0-8`): composition and global style
- middle blocks (`9-18`): character/body/shape bias
- late blocks (`19-27`): details, finish, brushwork

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
::krenz::1.2@9-18%0.35-0.85
hiten@19-27%0.65-1.0
```

The timing range is normalized sampling progress:

- `0.0` = sampling start, highest noise
- `0.5` = middle of the denoising trajectory
- `1.0` = sampling end, final detail pass

This enables scheduled artist roles from one artist chain: one artist can shape early composition, another can dominate the middle structure, and another can add late brushwork. If the current layer and current sampling progress have no matching artist, that layer falls back to original cross-attention for that step.

Per-artist timing is independent from global `start_percent / end_percent`. Global timing still applies first; per-artist timing decides which artists participate inside the globally active window.

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

When `::weight` is used, v25 judges risk by the **actual sum of absolute linear weights**, not by artist count. Four artists at `::0.25` each are valid because the total is still 1.0.

If you actually want "one artist weakened", the recommended approach is to use injection weighting `::name::0.3` to lower a specific artist:

```
wlop, ::krenz::0.3
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
