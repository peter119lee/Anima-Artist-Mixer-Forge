# Anima-Artist-Mixer

A ComfyUI custom node that enables **multi-artist mixing** for the Anima model by hooking into its cross-attention layers.

![surtr](docs/images/ComfyUI_01092_.png)
## What it does

Anima uses an LLM as its text encoder. When multiple artist tags are stacked in a single prompt, the LLM's contextualization causes them to interfere with each other, producing a conditioning that resembles neither artist clearly. This node encodes each artist independently and mixes them at the model's cross-attention layer, sidestepping the interference at the prompt-encoding stage.

The bundled `AnimaArtistPack` node provides a one-shot experience: write your artist list (separated by commas or newlines) in one text box, your main prompt in another, and the node handles splitting, encoding, and packaging automatically.

Product principle: the default path is predictable artist mixing on top of the base model. It should preserve the prompt and expose artist influence in a controllable way; automatic low-drift routing and stabilizers are opt-in tools, not the default style source.

The current release (v26) keeps the original controllable artist-mixer path, then makes the preset workflow clearer and safer. `balanced` stays close to the original mixer behavior; `prompt_passthrough` uses the no-mixer/direct-prompt path while preserving positive `1.2::tag::` weighting syntax; `drift_auto` and the scene presets are opt-in low-drift routes. v26 also supports prefix artist weights (`1.2::artist::`), base-prompt tag weights (`1.2::masterpiece::`), negative artist weights for style subtraction, timing fades (`%0.0-0.45~0.1`), recipes, the layer probe, VRAM controls, a CFG correctness fix for batch sizes > 1, sample workflow fixes, and tests/CI. Existing per-artist layer and timing routes remain supported. See [CHANGELOG.md](CHANGELOG.md).

## Quick links

- [Simple starter workflow](<sample workflow.json>) — safe preset route using `AnimaArtistPresetApply`
- [Node usage workflows](workflow/node_usage_showcase/README_zh.md) — Chinese guide covering all bundled nodes
- [Layer role workflow](workflow/artist-layer-role-routing.json) — character / clothing / background routing example
- [Full documentation](docs/USAGE.md) — usage, parameters, modes, stabilizers, performance tips
- [Changelog](CHANGELOG.md) — version history
- [Issues](../../issues) — bug reports, feature requests
- [Discussions](../../discussions) — usage questions, results sharing

## Installation

Clone or download into your ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/<this-plugin-folder>/
```

Restart ComfyUI. No extra dependencies.

## Requirements

- **Anima model only** — depends on Anima's built-in `LLMAdapter` (`preprocess_text_embeds`)
- Use the **same CLIP loader** that Anima's own text-encoding workflow uses (the one whose tokens carry `t5xxl_ids`)
- Inference only

## Quick start
![workflow](docs/images/workflow.png)

Open [`sample workflow.json`](<sample workflow.json>) first. It uses the current
`AnimaArtistPack -> AnimaArtistPresetApply` preset route and avoids the old
manual widgets that are ignored when a preset is connected.

Open [`workflow/node_usage_showcase/README_zh.md`](workflow/node_usage_showcase/README_zh.md)
when you want examples for every node in the pack.

Open [`workflow/artist-layer-role-routing.json`](<workflow/artist-layer-role-routing.json>)
for a bundled `AnimaArtistPreset -> AnimaArtistPresetApply` character /
clothing / background layer-role example using separate per-artist `@layers`
windows.

- Top text box of `AnimaArtistPack`: your artist chain (comma or newline separated)
- Fastest first run: use `AnimaArtistStarter`, fill `artist_table`, then follow its in-UI guide
- Use `AnimaArtistChainBuilder` when you do not want to hand-write `1.2::artist::`, `@layers`, and `%timing`
- Builder's three visible rows are only shortcuts; use its `artist_table` field for larger chains
- Use `AnimaArtistChainPreview` to validate a chain before paying the CLIP encoding cost
- Bottom text box: the main prompt (no need to repeat artist names here)
- Wire `AnimaArtistPresetApply`'s `base_prompt` output directly to KSampler's positive input
- For a sane first run, connect `AnimaArtistPreset` with `preset = balanced`, then wire it into `AnimaArtistPresetApply`
- For common layer/timing tweaks, use `AnimaArtistSimpleOptions`; keep `AnimaArtistOptions (Expert)` for stabilizer A/B and debugging
- If the workflow also uses regional prompting, Forge Couple-style routing, or other attention patchers, start with `preset = compatibility_safe`
- When a workflow behaves strangely, connect `AnimaArtistInspector` and read the effective weights / warnings directly in ComfyUI

For full parameter explanations and recommended combinations, see [docs/USAGE.md](docs/USAGE.md).

## Recommended defaults

For most users, start with:

```
AnimaArtistStarter:
recipe    = balanced
layout    = layer_scheduled

or:

AnimaArtistPreset:
preset    = balanced
intensity = 1.0

AnimaArtistPresetApply:
connect model + artist_pack + preset
```

Manual equivalent:

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
artist_ema_alpha = 0.0
match_base_norm  = False
```

To weight individual artists within the chain, use prefix injection weights:

```
wlop, 1.2::sakimichan::, (krenz:0.7), -0.4::pixiv_style::
```

- `(name:1.2)` — CLIP-side weighting (same as SD/A1111), non-linear, applied at text encoding
- `1.2::name::` — injection-side weighting, linear and predictable, applied at cross-attention output
- `-0.4::name::` — negative injection weight: subtracts that artist's style direction instead of adding it (style subtraction); range is [-4, 4]
- `1.2::masterpiece::` in the base prompt — tag weighting, expanded to normal prompt weight syntax before encoding
- Older postfix forms like `::name::1.2` still load for compatibility, but new examples use prefix syntax
- Any valid injection weight automatically disables normalization at runtime so explicit weights stay absolute
- Per-artist layer routing is supported with `@layers`: `wlop@0-8, krenz@33%-67%, hiten@0.67-1.0`
- Per-artist sampling timing is supported with `%start-end`: `wlop@0-8%0.0-0.45, krenz@9-18%0.45-0.85`
- Timing windows can fade in/out smoothly with `~fade` (v26+): `wlop%0.0-0.45~0.1` ramps the artist's weight with a smoothstep over a 0.1-progress-wide edge instead of switching on/off abruptly
- Anima artist tags that start with `@` are safe: `@wlop` remains the artist name; only a final numeric suffix like `@0-8`, `@33%-67%`, or `@0.33-0.67` is treated as layer routing
- When combining weight and routing, put the route inside the weighted target: `1.2::@artist_a@0-8::`, not `1.2::@artist_a::@0-8`

## Compatibility notes

This node wraps Anima cross-attention. Other nodes that also patch attention, regional prompts, Forge Couple-style routing, or model forward wrappers can change the same execution path. If the artist effect disappears or becomes very weak, use `AnimaArtistPreset(preset = compatibility_safe)` first. It forces the tolerant `concat + concat_with_base` path and disables cache-heavy stabilizers. Use `AnimaArtistInspector` to confirm parsed artists, weights, layer routes, timing routes, block map, and effective normalize state.

## Cross-seed stability

In multi-artist setups, the same prompt with different seeds tends to produce noticeably different style mixes — sometimes one artist dominates, other times another, even at equal weights. This is structural to how cross-attention interacts with seed-driven hidden state.

v26 keeps `balanced` close to the original mixer behavior by default. Common layer/timing controls live in `AnimaArtistSimpleOptions`; optional stabilizers live in `AnimaArtistOptions (Expert)`, ordered from light to heavy:

| Stabilizer | Strength | Notes |
|---|---|---|
| `match_base_norm` + `norm_lock_mode=token` + `norm_lock_scope=per_artist` | optional | Per-artist token RMS lock; reduces seed-specific style-strength spikes before artists are mixed |
| `artist_ema_alpha` | light | Temporal EMA across sampling steps |
| `combine_mode = lowrank_avg` + `lowrank_k` | medium | Deterministic low-rank constraint on multi-artist deltas |
| `artist_static_capture` + `static_capture_k` | heavy | Freeze artist attention after K warmup steps. This is an expert A/B control; current presets avoid it by default after multi-artist evidence showed it can over-constrain style. |
| `stabilizer_end_percent` | optional | Lets EMA/static/anchor stabilizers stop after an early sampling window; keep `1.0` for full-pass behavior, try `0.4-0.6` when late-step samplers need dynamic motion. |
| `contribution_balance` | optional | Delta-strength equalizer for artist dominance flips; default off until it has stronger live evidence |
| `mixed_delta_cap` | optional | Caps the final mixed artist delta against base attention energy before fusion; default off while it is evaluated as a live A/B candidate |
| `artist_anchor_q` | heaviest | Replace user-seed Q with a fixed-seed anchor's Q; `anchor_lock` now uses one anchor, user-Q blend, strength 0.9, and auto layers 9-15 to reduce pose artifacts |
| `anchor_base_norm_ref` | optional | Anchor the norm reference too when testing `anchor_q + match_base_norm`; off by default and mainly useful for A/B |

Recommended progression: start with `balanced` for original-style behavior, then use `stable_seed` or `drift_auto` for content-safer cross-seed work. Use `prompt_passthrough` only when you want the no-mixer/direct-prompt path while keeping positive artist weight syntax such as `1.2::@artist::`; it returns the unpatched model and does not support negative style subtraction, layer routes, or timing routes. For lower drift, `drift_auto` keeps broad 4+ artist prompts on the style-mixer path: wide / background-heavy scenes route to `scene_lock`, simple fullbody and broad portrait/street prompts route to `drift_soft`, 4+ artist close-ups route to `stable_seed` plus `mixed_delta_cap_ratio=0.75`, and smaller close-up face prompts route to `face_lock`. Use `compatibility_safe` explicitly for regional prompting or other attention patchers, not as the default multi-artist style path. See [docs/USAGE.md](docs/USAGE.md) for detailed mechanics and tuning.

## Style amplification

`strength` accepts values in `[0, 4]`:

- `0 ~ 1` — interpolation between base and artist (`strength=1` = pure artist replacement)
- `1 ~ 4` — CFG-style extrapolation: `out = base + strength * (artist - base)`, amplifying the artist's deviation from base for stronger style

`1.5 ~ 2.5` is a common range for "stronger style without breaking content"; pushing past `3` tends to oversaturate.

## Performance notes

Generation time scales with the number of active artists, active layers, and active sampling steps. In `output_avg` / `lowrank_avg`, each active layer computes the base attention plus one attention pass per active artist, so multi-artist `balanced` is expected to be slower than no mixer.

| Path | Speed expectation |
|---|---|
| no mixer / `prompt_passthrough` | close to the ordinary Anima prompt path |
| `balanced`, `drift_soft`, `scene_lock`, `face_lock`, `stable_seed` | slower as active artist count increases |
| `fast_preview` / `compatibility_safe` | usually faster concat path, but less precise as an artist mixer |
| layer routes / timing routes | reduce cost only for the layers or sampling steps where artists are inactive |
| `artist_static_capture` / `artist_anchor_q` | expert A/B controls; may help some repeated-seed workflows but can add overhead or constrain style, so they are not the default speed recommendation |

For speed, first limit the active artist count, layer range, or sampling window through `AnimaArtistSimpleOptions`. Quality impact is prompt- and artist-dependent, so check the result rather than treating a faster setting as automatically equivalent.

## Measuring where styles live (v26)

Instead of guessing `@layers` routes, wire `AnimaArtistProbe` between your model loader and the sampler, run one generation, and read `AnimaArtistProbeReport` (connect any post-sampler output as its trigger). The report shows each artist's per-layer style influence (`||artist_out − base_out|| / ||base_out||`) as a bar chart and suggests a concrete `artist@lo-hi` route per artist. The probe pass does not alter the generated image.

## Sharing recipes (v26)

`AnimaArtistRecipeSave` packs the artist chain plus the full effective configuration (combine/fusion/strength/advanced options) into one JSON string; `AnimaArtistRecipeLoad` turns it back into `artist_chain` + a `preset` payload you can wire straight into `AnimaArtistPresetApply`. Paste-friendly for sharing exact mixes with other users.

## Important caveat

This node **cannot achieve the near-lossless artist mixing that SDXL does**. Anima's text encoder is non-linear, so any mixing strategy introduces some distortion. What this node does is make that distortion controllable. Style-similar artists mix well; style-divergent artists may "regress to the mean" into a compromise look — `lowrank_avg` accepts more of this regression in exchange for cross-seed stability.

## Development

The implementation lives in the `anima_mixer/` package (`nodes.py` is a compatibility shim). Run the broad local test suite with:

```
python -m pytest -q
```

CI runs `ruff` plus `unittest` on Python 3.10/3.12 for every push and PR.

## Acknowledgements

Special thanks to **汐浮尘/utowo** for co-development, testing, and design contributions. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

MIT License. See [LICENSE](LICENSE) for the full text.
