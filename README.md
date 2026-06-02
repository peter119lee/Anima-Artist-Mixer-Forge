# Anima-Artist-Mixer

A ComfyUI custom node that enables **multi-artist mixing** for the Anima model by hooking into its cross-attention layers.

![surtr](docs/images/ComfyUI_01092_.png)
## What it does

Anima uses an LLM as its text encoder. When multiple artist tags are stacked in a single prompt, the LLM's contextualization causes them to interfere with each other, producing a conditioning that resembles neither artist clearly. This node encodes each artist independently and mixes them at the model's cross-attention layer, sidestepping the interference at the prompt-encoding stage.

The bundled `AnimaArtistPack` node provides a one-shot experience: write your artist list (separated by commas or newlines) in one text box, your main prompt in another, and the node handles splitting, encoding, and packaging automatically.

The current release (v25) adds one-click presets, an in-UI inspector, deterministic low-rank mixing, safer explicit weights, layered cross-seed stabilizers, CFG-style strength extrapolation, and the linear injection-layer weight syntax `::name::weight`.

## Quick links

- [Full documentation](docs/USAGE.md) — usage, parameters, modes, stabilizers, performance tips
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

```
                          ┌──► artist_pack ──► AnimaArtistCrossAttn ──► MODEL ──► KSampler
[Load CLIP] ─► CLIP ──────┤                              │                          │
                          │                              └──► base_prompt ──► (positive)
                          │
                          └──► CLIPTextEncode (Negative) ──► (negative)

[Load Anima Model] ──► MODEL ──► AnimaArtistCrossAttn

(optional) AnimaArtistPreset  ──► preset ────────────► AnimaArtistCrossAttn
(optional) AnimaArtistOptions ──► advanced_options ──► AnimaArtistCrossAttn
(optional) AnimaArtistInspector ◄── artist_pack / preset / advanced_options
```

- Top text box of `AnimaArtistPack`: your artist chain (comma or newline separated)
- Bottom text box: the main prompt (no need to repeat artist names here)
- Wire `AnimaArtistCrossAttn`'s `base_prompt` output directly to KSampler's positive input
- For a sane first run, connect `AnimaArtistPreset` with `preset = balanced`
- When a workflow behaves strangely, connect `AnimaArtistInspector` and read the effective weights / warnings directly in ComfyUI

For full parameter explanations and recommended combinations, see [docs/USAGE.md](docs/USAGE.md).

## Recommended defaults

For most users, start with:

```
AnimaArtistPreset:
preset    = balanced
intensity = 1.0
```

Manual equivalent:

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
artist_ema_alpha = 0.25
```

To weight individual artists within the chain, use either of two syntaxes (they can coexist and stack):

```
wlop, ::sakimichan::1.2, (krenz:0.7)
```

- `(name:1.2)` — CLIP-side weighting (same as SD/A1111), non-linear, applied at text encoding
- `::name::1.2` — injection-side weighting (v24+), linear and predictable, applied at cross-attention output
- In v25, any valid `::weight` automatically disables normalization at runtime so explicit weights stay absolute
- Per-artist layer routing is supported with `@layers`: `wlop@0-8, krenz@9-18, hiten@19-27`
- Anima artist tags that start with `@` are safe: `@wlop` remains the artist name; only a final numeric suffix like `@0-8` is treated as layer routing

## Compatibility notes

This node wraps Anima cross-attention. Other nodes that also patch attention, regional prompts, Forge Couple-style routing, or model forward wrappers can change the same execution path. If the artist effect disappears or becomes very weak, first try `combine_mode = concat`, disable cache-heavy stabilizers, or reduce other attention-patching nodes in the same workflow. Use `AnimaArtistInspector` to confirm the parsed artists, weights, layer routes, and effective normalize state.

## Cross-seed stability

In multi-artist setups, the same prompt with different seeds tends to produce noticeably different style mixes — sometimes one artist dominates, other times another, even at equal weights. This is structural to how cross-attention interacts with seed-driven hidden state.

v25 provides four optional stabilizers via `AnimaArtistOptions`, ordered from light to heavy:

| Stabilizer | Strength | Notes |
|---|---|---|
| `artist_ema_alpha` | light | Temporal EMA across sampling steps |
| `combine_mode = lowrank_avg` + `lowrank_k` | medium | Deterministic low-rank constraint on multi-artist deltas |
| `artist_static_capture` + `static_capture_k` | heavy | Freeze artist attention after K warmup steps (also a 30-50% speedup) |
| `artist_anchor_q` | heaviest | Replace user-seed Q with a fixed-seed anchor's Q (near-full cross-seed decoupling) |

All are off by default. Recommended progression: start with EMA, escalate as needed. See [docs/USAGE.md](docs/USAGE.md) for detailed mechanics and tuning.

## Style amplification

`strength` accepts values in `[0, 4]`:

- `0 ~ 1` — interpolation between base and artist (`strength=1` = pure artist replacement)
- `1 ~ 4` — CFG-style extrapolation: `out = base + strength * (artist - base)`, amplifying the artist's deviation from base for stronger style

`1.5 ~ 2.5` is a common range for "stronger style without breaking content"; pushing past `3` tends to oversaturate.

## Performance notes

Generation time scales with artist count. Per the math of `output_avg`, each layer runs `N + 1` cross-attention forwards (N artists + base). Approximate measured cost (varies by GPU):

| Configuration | Relative time |
|---|---|
| 1 artist | 1.0x |
| 4 artists | ~1.4x |
| 8 artists | ~1.7x |
| 5 artists + `artist_static_capture` (K=6) | ~1.1x |
| 5 artists + `artist_anchor_q` (cached) | ~1.05x |

**Strongly recommended**: connect `AnimaArtistOptions` and limit either the layer range (`start_block / end_block`) or the sampling-step range (`start_percent / end_percent`). Both can dramatically reduce generation time with minimal quality loss, and stack with the cache-based stabilizers above. See the docs for details.

## Important caveat

This node **cannot achieve the near-lossless artist mixing that SDXL does**. Anima's text encoder is non-linear, so any mixing strategy introduces some distortion. What this node does is make that distortion controllable. Style-similar artists mix well; style-divergent artists may "regress to the mean" into a compromise look — `lowrank_avg` accepts more of this regression in exchange for cross-seed stability.

## Acknowledgements

Special thanks to **汐浮尘/utowo** for co-development, testing, and design contributions. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

MIT License. See [LICENSE](LICENSE) for the full text.
