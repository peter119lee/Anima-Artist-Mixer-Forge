# Anima-Artist-Mixer

A ComfyUI custom node that enables **multi-artist mixing** for the Anima model by hooking into its cross-attention layers.

![surtr](ComfyUI_01092_.png)
## What it does

Anima uses an LLM as its text encoder. When multiple artist tags are stacked in a single prompt, the LLM's contextualization causes them to interfere with each other, producing a conditioning that resembles neither artist clearly. This node encodes each artist independently and mixes them at the model's cross-attention layer, sidestepping the interference at the prompt-encoding stage.

The bundled `AnimaArtistPack` node provides a one-shot experience: write your artist list (separated by commas or newlines) in one text box, your main prompt in another, and the node handles splitting, encoding, and packaging automatically.

## Quick links

- [Full documentation](docs/USAGE.md) — usage, parameters, modes, performance tips
- [Optimization notes](OPTIMIZATION_NOTES.md) — technical roadmap for contributors
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

(optional) AnimaArtistOptions ──► advanced_options ──► AnimaArtistCrossAttn
```

- Top text box of `AnimaArtistPack`: your artist chain (comma or newline separated)
- Bottom text box: the main prompt (no need to repeat artist names here)
- Wire `AnimaArtistCrossAttn`'s `base_prompt` output directly to KSampler's positive input

For full parameter explanations and recommended combinations, see [docs/USAGE.md](docs/USAGE.md).

## Recommended defaults

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 0.6 ~ 0.8
```

To weight individual artists, use CLIP weighting syntax inside the artist chain:

```
wlop, (sakimichan:1.2), (krenz:0.7)
```

## Performance notes

Generation time scales with artist count. Per the math of `output_avg`, each layer runs `N + 1` cross-attention forwards (N artists + base). Approximate measured cost (varies by GPU):

| Artist count | Relative time |
|---|---|
| 1 | 1.0x |
| 4 | ~1.4x |
| 8 | ~1.7x |

**Strongly recommended**: connect `AnimaArtistOptions` and limit either the layer range (`start_block / end_block`) or the sampling-step range (`start_percent / end_percent`). Both can dramatically reduce generation time with minimal quality loss. See the docs for details.

## Important caveat

This node **cannot achieve the near-lossless artist mixing that SDXL does**. Anima's text encoder is non-linear, so any mixing strategy introduces some distortion. What this node does is make that distortion controllable. Style-similar artists mix well; style-divergent artists may "regress to the mean" into a compromise look.

## Acknowledgements

Special thanks to **汐浮尘/utowo** for co-development, testing, and design contributions. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

MIT License. See [LICENSE](LICENSE) for the full text.
