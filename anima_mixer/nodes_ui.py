"""UI helper nodes: builder, preview, options, presets, starter, inspector, recipes."""

from .chain_tools import (
    build_artist_chain_from_rows,
    format_artist_block_map,
    format_artist_chain_preview,
    parse_builder_artist_table,
)
from .constants import (
    CHAIN_LAYOUT_CHOICES,
    CHAIN_LAYOUT_LAYER_SCHEDULED,
    COMBINE_CHOICES,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    DEFAULT_NUM_BLOCKS,
    FUSION_CHOICES,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    LAYER_MODE_AUTO,
    LAYER_MODE_CHOICES,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    MAX_ARTISTS,
    PRESET_BALANCED,
    PRESET_CHOICES,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .options import (
    build_preset_payload,
    format_bool,
    merge_runtime_options,
)
from .parsing import resolve_target_blocks_from_options, split_artist_chain
from .recipe import deserialize_recipe, serialize_recipe


class AnimaArtistChainBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        artist_input = {
            "multiline": False,
            "default": "",
            "tooltip": "Artist tag. Leave empty to skip this row.",
        }
        layer_input = {
            "multiline": False,
            "default": "",
            "tooltip": (
                "Optional layer route, e.g. 0-8 or 0,2,4. Used in manual "
                "layout; other layouts auto-fill empty routes."
            ),
        }
        timing_input = {
            "multiline": False,
            "default": "",
            "tooltip": (
                "Optional timing route, e.g. 0.0-0.45 or 0.0-0.45~0.1 "
                "(with fade). Used in manual layout; other layouts auto-fill "
                "empty routes."
            ),
        }
        return {
            "required": {
                "layout": (CHAIN_LAYOUT_CHOICES, {
                    "default": CHAIN_LAYOUT_LAYER_SCHEDULED,
                    "tooltip": (
                        "manual: use the per-row layer/timing values\n"
                        "even_layers: split DiT blocks evenly across artists\n"
                        "layer_scheduled: early/mid/late layers + early/mid/late "
                        "sampling windows in one click"
                    ),
                }),
                "artist_table": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Multi-artist table. One per line: artist | weight | layers | timing.\n"
                        "Example: @wlop | 1.2 | 0-8 | 0.0-0.45\n"
                        "Empty layers/timing auto-fill per layout. Lines starting "
                        "with # are ignored."
                    ),
                }),
                "artist_1": ("STRING", artist_input),
                "weight_1": ("FLOAT", {"default": 1.0, "min": WEIGHT_MIN, "max": WEIGHT_MAX, "step": 0.05}),
                "artist_2": ("STRING", artist_input),
                "weight_2": ("FLOAT", {"default": 1.0, "min": WEIGHT_MIN, "max": WEIGHT_MAX, "step": 0.05}),
                "artist_3": ("STRING", artist_input),
                "weight_3": ("FLOAT", {"default": 1.0, "min": WEIGHT_MIN, "max": WEIGHT_MAX, "step": 0.05}),
            },
            "optional": {
                "layer_route_1": ("STRING", layer_input),
                "timing_route_1": ("STRING", timing_input),
                "layer_route_2": ("STRING", layer_input),
                "timing_route_2": ("STRING", timing_input),
                "layer_route_3": ("STRING", layer_input),
                "timing_route_3": ("STRING", timing_input),
                "num_blocks": ("INT", {
                    "default": DEFAULT_NUM_BLOCKS, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Block count for the preview. Anima default is 28.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("artist_chain", "preview")
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def build(self, layout, artist_table, artist_1, weight_1, artist_2, weight_2, artist_3, weight_3,
              layer_route_1="", timing_route_1="", layer_route_2="", timing_route_2="",
              layer_route_3="", timing_route_3="", num_blocks=DEFAULT_NUM_BLOCKS):
        table_rows, table_warnings = parse_builder_artist_table(
            artist_table, return_warnings=True,
        )
        rows = [
            (artist_1, weight_1, layer_route_1, timing_route_1),
            (artist_2, weight_2, layer_route_2, timing_route_2),
            (artist_3, weight_3, layer_route_3, timing_route_3),
        ]
        rows.extend(table_rows)
        chain, report = build_artist_chain_from_rows(
            layout, rows, int(num_blocks), extra_warnings=table_warnings,
        )
        return {"ui": {"text": [report]}, "result": (chain, report)}


class AnimaArtistChainPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "artist_chain to validate. This node needs no CLIP/model; "
                        "use it to check ::weight, @layers, %timing and ~fade "
                        "before paying the CLIP encoding cost."
                    ),
                }),
            },
            "optional": {
                "num_blocks": ("INT", {
                    "default": DEFAULT_NUM_BLOCKS, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Block count for the preview. Anima default is 28.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cleaned_chain", "report")
    FUNCTION = "preview"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def preview(self, artist_chain, num_blocks=DEFAULT_NUM_BLOCKS):
        cleaned, report = format_artist_chain_preview(artist_chain, int(num_blocks))
        return {"ui": {"text": [report]}, "result": (cleaned, report)}


class AnimaArtistOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 63, "step": 1,
                    "tooltip": "First patched block (inclusive). 0 = first layer."
                }),
                "end_block": ("INT", {
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "Last patched block (inclusive). -1 = last layer."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling-progress start. 0.0 = beginning."
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling-progress end. 1.0 = end of sampling."
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True: weights become relative proportions. False: weights "
                        "act as independent strengths.\n\n"
                        "If the artist_chain uses ::weight syntax, this switch is "
                        "bypassed at runtime (explicit weights stay absolute)."
                    )
                }),
                "artist_ema_alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": (
                        "Cross-step EMA smoothing (fusion interpolate/base_preserve).\n"
                        "Smooths each layer's artist_total across steps to reduce\n"
                        "cross-seed dominant-artist flips.\n"
                        "0.0: off (default)\n"
                        "0.3-0.5: light smoothing\n"
                        "0.5-0.8: medium-heavy\n"
                        ">0.8: strong; style may lag behind base content\n"
                        "A new sampling run (sigma jump) resets the cache."
                    ),
                }),
                "lowrank_k": ("INT", {
                    "default": 1, "min": 1, "max": MAX_ARTISTS, "step": 1,
                    "tooltip": (
                        "LoRA-style low-rank dimension (combine_mode=lowrank_avg only).\n"
                        "Projects N artist deltas onto the top-k principal directions.\n"
                        "k=1: single consensus direction, most stable, most homogeneous\n"
                        "k=2-3: keeps main directions, allows some per-artist variety\n"
                        "k>=N: equivalent to output_avg (no projection)\n"
                        "Falls back to output_avg automatically when N=1."
                    ),
                }),
                "artist_static_capture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "H' temporal average: accumulate artist attention over the\n"
                        "first K steps, then freeze and reuse the average (also a\n"
                        "30-50% speedup). K set by static_capture_k.\n"
                        "Compatible: output_avg / lowrank_avg + interpolate / base_preserve.\n"
                        "Incompatible: fusion=concat_with_base (ignored automatically).\n"
                        "Mutually exclusive with EMA (EMA is ignored when on)."
                    ),
                }),
                "static_capture_k": ("INT", {
                    "default": STATIC_CAPTURE_K_DEFAULT,
                    "min": 1, "max": STATIC_CAPTURE_K_MAX, "step": 1,
                    "tooltip": (
                        "Steps accumulated before freezing (artist_static_capture only).\n"
                        "K=1: single-point cache (fastest, most cross-seed drift)\n"
                        "K=6: recommended default\n"
                        "K=8-12: stronger drift suppression, more warmup cost\n"
                        "If total steps < K it keeps averaging without freezing."
                    ),
                }),
                "artist_anchor_q": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Anchor-Q: artist cross-attention uses a fixed-seed anchor\n"
                        "hidden state as Q, decoupling style mixing from the user seed.\n"
                        "The strongest cross-seed stabilizer.\n\n"
                        "Cost: one extra full forward on the first generation; same\n"
                        "prompt across seeds then hits the cache for free.\n\n"
                        "Mutually exclusive with static_capture (anchor wins).\n"
                        "Incompatible with fusion=concat_with_base.\n\n"
                        "Risk: stroke placement may track the current image less\n"
                        "closely; raise anchor_user_blend if that bothers you."
                    ),
                }),
                "anchor_seeds_count": ("INT", {
                    "default": 1, "min": 1, "max": ANCHOR_SEEDS_MAX, "step": 1,
                    "tooltip": (
                        "Number of fixed seeds for the anchor pre-run (anchor_q only).\n"
                        "1: single seed\n"
                        "2-4: average several anchors to reduce single-seed bias.\n"
                        "Pre-run time scales with the count; cached afterwards."
                    ),
                }),
                "anchor_user_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Anchor / user-x blend (anchor_q only).\n"
                        "Q = blend * user_x + (1-blend) * anchor_x\n"
                        "0.0: pure anchor (most stable across seeds)\n"
                        "0.3-0.5: balanced stability vs stroke fit\n"
                        "1.0: pure user x (equivalent to anchor_q off)"
                    ),
                }),
                "anchor_deep_layer_threshold": ("INT", {
                    "default": ANCHOR_LAYER_THRESHOLD_DISABLED,
                    "min": ANCHOR_LAYER_THRESHOLD_DISABLED, "max": 64, "step": 1,
                    "tooltip": (
                        "Use the anchor only in shallow layers (anchor_q only).\n"
                        "-1: every layer uses the anchor\n"
                        "N>=0: layers < N use the anchor, layers >= N use user x.\n"
                        "Shallow layers set style direction; deep layers fit strokes.\n"
                        "Example: 28-block model with N=14 anchors the first half."
                    ),
                }),
            },
            "optional": {
                "layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Advanced layer selection. Comma-separated block indices, "
                        "ranges and negative indices supported.\n"
                        "Example: '0,3,5-10,-1'. Overrides start_block/end_block. "
                        "Empty = inactive."
                    ),
                }),
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Compatibility-safe mode. Forces concat + concat_with_base "
                        "and disables EMA / static_capture / anchor_q to minimize "
                        "conflicts with regional prompting, Forge Couple-style "
                        "routing, and other attention patch nodes."
                    ),
                }),
                "max_batch_artists": ("INT", {
                    "default": 0, "min": 0, "max": MAX_ARTISTS, "step": 1,
                    "tooltip": (
                        "Cap on how many artists run in one batched forward.\n"
                        "0 = no cap (fastest, highest peak VRAM).\n"
                        "Set 2-8 to bound VRAM with many artists at high "
                        "resolution instead of falling back to sequential mode."
                    ),
                }),
                "low_vram_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Store static-capture and anchor caches in system RAM "
                        "instead of VRAM. Saves hundreds of MB at high resolution "
                        "for a small per-step transfer cost."
                    ),
                }),
                "match_base_norm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Rescale the mixed artist attention output to the base "
                        "output's RMS energy (per batch row, clamped to 0.5-2.0x).\n"
                        "Keeps the style direction but stops activation-energy "
                        "mismatch from compounding across layers — the main "
                        "source of seed-to-seed style-strength swings (style "
                        "drift). Disable to reproduce pre-v26 behavior exactly."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, start_block, end_block, start_percent, end_percent, normalize_weights,
              artist_ema_alpha=0.0, lowrank_k=1, artist_static_capture=False,
              static_capture_k=STATIC_CAPTURE_K_DEFAULT, artist_anchor_q=False,
              anchor_seeds_count=1, anchor_user_blend=0.0,
              anchor_deep_layer_threshold=ANCHOR_LAYER_THRESHOLD_DISABLED,
              layer_filter="", compatibility_mode=False,
              max_batch_artists=0, low_vram_cache=False, match_base_norm=True):
        return ({
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "normalize_weights": bool(normalize_weights),
            "artist_ema_alpha": float(artist_ema_alpha),
            "lowrank_k": int(lowrank_k),
            "artist_static_capture": bool(artist_static_capture),
            "static_capture_k": int(static_capture_k),
            "artist_anchor_q": bool(artist_anchor_q),
            "anchor_seeds_count": int(anchor_seeds_count),
            "anchor_user_blend": float(anchor_user_blend),
            "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
            "layer_filter": str(layer_filter or ""),
            "compatibility_mode": bool(compatibility_mode),
            "max_batch_artists": int(max_batch_artists),
            "low_vram_cache": bool(low_vram_cache),
            "match_base_norm": bool(match_base_norm),
        },)


class AnimaArtistPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_CHOICES, {
                    "default": PRESET_BALANCED,
                    "tooltip": (
                        "One-knob working modes.\n"
                        "balanced: recommended default, light EMA\n"
                        "strong_style: stronger style, strength extrapolated to 1.65\n"
                        "stable_seed: lowrank + static capture, cross-seed stability first\n"
                        "fast_preview: concat path, speed first, good for hunting\n"
                        "identity_guard: base_preserve + lowrank, protects identity/composition\n"
                        "compatibility_safe: concat + concat_with_base, plays nice with "
                        "regional/Forge-style nodes"
                    ),
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "Preset strength multiplier. fast_preview and "
                        "compatibility_safe ignore it; other presets multiply it "
                        "into strength."
                    ),
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Default normalize_weights inside the preset. ::weight in "
                        "the chain still bypasses it at runtime."
                    ),
                }),
                "layer_mode": (LAYER_MODE_CHOICES, {
                    "default": LAYER_MODE_AUTO,
                    "tooltip": (
                        "Layer-range shortcut.\n"
                        "auto/all_layers: every layer\n"
                        "style_core: 0-18, overall style\n"
                        "detail_layers: 12-63, details and strokes\n"
                        "custom: use custom_layer_filter"
                    ),
                }),
                "custom_layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Active when layer_mode=custom. Example: 0,3,5-10,-1",
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("preset", "advanced_options", "summary")
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, preset, intensity, normalize_weights, layer_mode, custom_layer_filter):
        payload = build_preset_payload(
            preset, intensity, layer_mode, custom_layer_filter, normalize_weights,
        )
        adv = payload["advanced_options"]
        summary = "\n".join([
            f"Preset: {payload['preset']}",
            f"combine_mode: {payload['combine_mode']}",
            f"fusion_mode: {payload['fusion_mode']}",
            f"strength: {payload['strength']:.2f}",
            f"normalize_weights: {format_bool(adv.get('normalize_weights', True))}",
            f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
            f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
            f"static_capture: {format_bool(adv.get('artist_static_capture', False))}",
            f"static_capture_k: {int(adv.get('static_capture_k', STATIC_CAPTURE_K_DEFAULT))}",
            f"compatibility_mode: {format_bool(adv.get('compatibility_mode', False))}",
            f"layer_filter: {adv.get('layer_filter') or 'all'}",
        ])
        return {"ui": {"text": [summary]}, "result": (payload, adv, summary)}


class AnimaArtistStarter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": (PRESET_CHOICES, {
                    "default": PRESET_BALANCED,
                    "tooltip": (
                        "Pick the goal first.\n"
                        "balanced: safe default starting point\n"
                        "strong_style: stronger style\n"
                        "stable_seed: more stable across seeds\n"
                        "fast_preview: fast image hunting\n"
                        "identity_guard: protect subject/composition\n"
                        "compatibility_safe: when regional prompting or other "
                        "attention patch nodes are present"
                    ),
                }),
                "artist_table": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "One artist per line: artist | weight | layers | timing.\n"
                        "Just the artist name is fine; weight defaults to 1.0; "
                        "empty layers/timing auto-fill per layout."
                    ),
                }),
                "layout": (CHAIN_LAYOUT_CHOICES, {
                    "default": CHAIN_LAYOUT_LAYER_SCHEDULED,
                    "tooltip": (
                        "manual: use the table's layers/timing as-is\n"
                        "even_layers: split DiT blocks evenly across artists\n"
                        "layer_scheduled: auto layers + sampling windows, "
                        "recommended for new users"
                    ),
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "Recipe strength multiplier. compatibility_safe and "
                        "fast_preview do not scale strength."
                    ),
                }),
            },
            "optional": {
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Recommended on for multi-artist setups. ::weight in the "
                        "table bypasses it at runtime."
                    ),
                }),
                "layer_mode": (LAYER_MODE_CHOICES, {
                    "default": LAYER_MODE_AUTO,
                    "tooltip": "Global layer-range shortcut; usually keep auto.",
                }),
                "custom_layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Active when layer_mode=custom. Example: 0,3,5-10,-1",
                }),
                "num_blocks": ("INT", {
                    "default": DEFAULT_NUM_BLOCKS, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Block count for the preview. Anima default is 28.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("artist_chain", "preset", "advanced_options", "guide")
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def build(self, recipe, artist_table, layout, intensity,
              normalize_weights=True, layer_mode=LAYER_MODE_AUTO,
              custom_layer_filter="", num_blocks=DEFAULT_NUM_BLOCKS):
        rows, table_warnings = parse_builder_artist_table(
            artist_table, return_warnings=True,
        )
        chain, chain_report = build_artist_chain_from_rows(
            layout, rows, int(num_blocks), extra_warnings=table_warnings,
        )
        payload = build_preset_payload(
            recipe, intensity, layer_mode, custom_layer_filter, normalize_weights,
        )
        adv = payload["advanced_options"]
        status = "CHECK" if "status: CHECK" in chain_report else "OK"
        guide = "\n".join([
            "Anima Artist Starter",
            "",
            f"status: {status}",
            f"recipe: {payload['preset']}",
            f"layout: {layout}",
            f"artists: {len(split_artist_chain(chain))}",
            "",
            "wire:",
            "  - artist_chain -> AnimaArtistPack.artist_chain",
            "  - preset -> AnimaArtistCrossAttn.preset",
            "  - advanced_options -> AnimaArtistCrossAttn.advanced_options only if you want the explicit option payload",
            "  - AnimaArtistPack.artist_pack -> AnimaArtistCrossAttn.artist_pack",
            "",
            "preset summary:",
            f"  combine_mode: {payload['combine_mode']}",
            f"  fusion_mode: {payload['fusion_mode']}",
            f"  strength: {float(payload['strength']):.2f}",
            f"  compatibility_mode: {format_bool(adv.get('compatibility_mode', False))}",
            f"  layer_filter: {adv.get('layer_filter') or 'all'}",
            "",
            "chain report:",
            chain_report,
        ])
        return {"ui": {"text": [guide]}, "result": (chain, payload, adv, guide)}


class AnimaArtistInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_pack": ("ANIMA_PACK",),
            },
            "optional": {
                "combine_mode": (
                    COMBINE_CHOICES,
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    FUSION_CHOICES,
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET",),
                "model": ("MODEL", {
                    "tooltip": (
                        "Optional. Connect the Anima model to read the real block "
                        "count instead of assuming 28."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def inspect(self, artist_pack, combine_mode=COMBINE_OUTPUT_AVG,
                fusion_mode=FUSION_INTERPOLATE, strength=1.0,
                advanced_options=None, preset=None, model=None):
        if not isinstance(artist_pack, dict):
            report = "Anima Artist Inspector\nERROR: artist_pack is not a valid ANIMA_PACK."
            return {"ui": {"text": [report]}, "result": (report,)}

        labels = list(artist_pack.get("labels") or [])
        weights = artist_pack.get("weights")
        if not isinstance(weights, (list, tuple)) or len(weights) != len(labels):
            weights = [1.0] * len(labels)
        weights = [float(w) for w in weights]
        layer_routes = artist_pack.get("layer_routes")
        if not isinstance(layer_routes, (list, tuple)) or len(layer_routes) != len(labels):
            layer_routes = [""] * len(labels)
        timing_routes = artist_pack.get("timing_routes")
        if not isinstance(timing_routes, (list, tuple)) or len(timing_routes) != len(labels):
            timing_routes = [""] * len(labels)
        has_explicit = bool(artist_pack.get("has_explicit_weights", False))
        base_prompt = str(artist_pack.get("base_prompt", "") or "")

        combine_mode, fusion_mode, strength, adv, preset_name = merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
        )
        requested_normalize = bool(adv.get("normalize_weights", True))
        effective_normalize = requested_normalize and not has_explicit
        weight_sum = sum(abs(w) for w in weights)

        inspector_blocks = DEFAULT_NUM_BLOCKS
        blocks_source = f"assumes {DEFAULT_NUM_BLOCKS} Anima blocks"
        if model is not None:
            try:
                from .patching import validate_model
                try:
                    dm = model.get_model_object("diffusion_model")
                except Exception:
                    dm = model.model.diffusion_model
                ok, real_blocks, _, _ = validate_model(dm)
                if ok:
                    inspector_blocks = real_blocks
                    blocks_source = f"read {real_blocks} blocks from the model"
            except Exception:
                blocks_source = (
                    f"failed to read the model; assumes {DEFAULT_NUM_BLOCKS} blocks"
                )

        lines = [
            "Anima Artist Mixer Inspector",
            "",
            f"preset: {preset_name or '(none)'}",
            f"artists: {len(labels)}",
            f"base_prompt: {'yes' if base_prompt else 'empty'}",
            f"combine_mode: {combine_mode}",
            f"fusion_mode: {fusion_mode}",
            f"strength: {float(strength):.2f}",
            f"requested normalize_weights: {format_bool(requested_normalize)}",
            f"effective normalize_weights: {format_bool(effective_normalize)}",
            f"effective linear weight sum: {weight_sum:.3f}",
            f"layer_filter: {adv.get('layer_filter') or 'all'}",
            f"compatibility_mode: {format_bool(adv.get('compatibility_mode', False))}",
            f"sigma range percent: {float(adv.get('start_percent', 0.0)):.3f} - "
            f"{float(adv.get('end_percent', 1.0)):.3f}",
            f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
            f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
            f"static_capture: {format_bool(adv.get('artist_static_capture', False))} "
            f"(K={int(adv.get('static_capture_k', STATIC_CAPTURE_K_DEFAULT))})",
            f"anchor_q: {format_bool(adv.get('artist_anchor_q', False))}",
            f"max_batch_artists: {int(adv.get('max_batch_artists', 0) or 0) or 'unlimited'}",
            f"low_vram_cache: {format_bool(adv.get('low_vram_cache', False))}",
            f"match_base_norm: {format_bool(adv.get('match_base_norm', True))}",
            "",
            "artists:",
        ]

        if labels:
            for idx, (label, weight, route, timing) in enumerate(
                zip(labels, weights, layer_routes, timing_routes), start=1,
            ):
                route_text = f" @ {route}" if route else ""
                timing_text = f" % {timing}" if timing else ""
                lines.append(f"  {idx}. {label} :: {weight:.3g}{route_text}{timing_text}")
        else:
            lines.append("  (none)")

        target_blocks = resolve_target_blocks_from_options(adv, inspector_blocks)
        lines.append("")
        lines.append(f"block map ({blocks_source}):")
        lines.append(format_artist_block_map(
            labels, layer_routes, timing_routes, inspector_blocks, target_blocks,
        ))

        warnings = []
        notes = []
        if not labels:
            warnings.append("no artists; CrossAttn will return the base prompt untouched.")
        if has_explicit and requested_normalize:
            warnings.append(
                "::weight detected; normalize_weights is bypassed at runtime "
                "(this is the correct behavior)."
            )
        if not effective_normalize and weight_sum > 1.5:
            warnings.append(
                "linear weight sum > 1.5; the style may oversaturate or blow out."
            )
        if any(w < 0.0 for w in weights):
            notes.append(
                "negative weights present: those artists subtract their style "
                "direction (style subtraction); best combined with positive artists."
            )
        if (
            adv.get("artist_static_capture", False)
            and adv.get("artist_anchor_q", False)
        ):
            warnings.append(
                "static_capture and anchor_q are mutually exclusive; CrossAttn "
                "disables static_capture."
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_anchor_q", False):
            warnings.append(
                "concat_with_base does not support anchor_q; CrossAttn disables anchor_q."
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_static_capture", False):
            warnings.append(
                "concat_with_base does not support static_capture; the normal path is used."
            )
        if combine_mode == COMBINE_LOWRANK_AVG and len(labels) <= 1:
            warnings.append(
                "lowrank_avg is meaningless with one artist; output_avg is used instead."
            )
        if any(str(route or "").strip() for route in layer_routes):
            warnings.append(
                "per-artist layer routes active; layers with no matching artist "
                "fall back to the original cross-attention."
            )
        if any(str(timing or "").strip() for timing in timing_routes):
            warnings.append(
                "per-artist timing routes active; when no artist matches the "
                "current progress the layer falls back to the original cross-attention."
            )
        if adv.get("compatibility_mode", False):
            notes.append(
                "compatibility_mode is on: concat + concat_with_base is forced "
                "and heavy stabilizers are disabled."
            )
        notes.append(
            "compatibility reminder: regional prompting / Forge Couple / other "
            "cross-attn patch nodes may override or weaken this node; if the "
            "effect disappears, try the compatibility_safe preset first and use "
            "fewer attention patch nodes."
        )

        status = "CHECK" if warnings else "OK"
        lines.insert(2, f"status: {status}")
        lines.append("")
        lines.append("warnings:")
        if warnings:
            lines.extend(f"  - {w}" for w in warnings)
        else:
            lines.append("  - no obvious configuration risk")
        lines.append("")
        lines.append("notes:")
        if notes:
            lines.extend(f"  - {n}" for n in notes)
        else:
            lines.append("  - none")

        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report,)}


class AnimaArtistRecipeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The artist chain to embed in the recipe.",
                }),
                "combine_mode": (COMBINE_CHOICES, {"default": COMBINE_OUTPUT_AVG}),
                "fusion_mode": (FUSION_CHOICES, {"default": FUSION_INTERPOLATE}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET",),
                "notes": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Free-form notes stored inside the recipe.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recipe_json",)
    FUNCTION = "save"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def save(self, artist_chain, combine_mode, fusion_mode, strength,
             advanced_options=None, preset=None, notes=""):
        combine_mode, fusion_mode, strength, adv, _ = merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
        )
        recipe_json = serialize_recipe(
            artist_chain, combine_mode, fusion_mode, strength, adv, notes,
        )
        return {"ui": {"text": [recipe_json]}, "result": (recipe_json,)}


class AnimaArtistRecipeLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Paste a recipe produced by AnimaArtistRecipeSave. The "
                        "preset output carries combine/fusion/strength/options; "
                        "wire it to AnimaArtistCrossAttn.preset."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("artist_chain", "preset", "advanced_options", "summary")
    FUNCTION = "load"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def load(self, recipe_json):
        payload, warnings = deserialize_recipe(recipe_json)
        preset_payload = {
            "preset": "recipe",
            "combine_mode": payload["combine_mode"],
            "fusion_mode": payload["fusion_mode"],
            "strength": payload["strength"],
            "advanced_options": payload["advanced_options"],
        }
        lines = [
            "Anima Artist Recipe",
            "",
            f"status: {'CHECK' if warnings else 'OK'}",
            f"combine_mode: {payload['combine_mode']}",
            f"fusion_mode: {payload['fusion_mode']}",
            f"strength: {payload['strength']:.2f}",
            f"artists: {len(split_artist_chain(payload['artist_chain']))}",
        ]
        if payload["notes"]:
            lines.extend(["", "notes:", f"  {payload['notes']}"])
        lines.extend(["", "warnings:"])
        if warnings:
            lines.extend(f"  - {w}" for w in warnings)
        else:
            lines.append("  - none")
        lines.extend([
            "",
            "wire:",
            "  - artist_chain -> AnimaArtistPack.artist_chain",
            "  - preset -> AnimaArtistCrossAttn.preset",
        ])
        summary = "\n".join(lines)
        return {
            "ui": {"text": [summary]},
            "result": (payload["artist_chain"], preset_payload,
                       payload["advanced_options"], summary),
        }
