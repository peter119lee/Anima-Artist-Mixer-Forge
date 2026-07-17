"""UI helper nodes: builder, preview, options, presets, starter, inspector, recipes."""

from . import tag_vocab
from .chain_tools import (
    build_artist_chain_from_rows,
    chain_artist_names,
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
    CONTRIB_BALANCE_ALPHA_DEFAULT,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    NORM_LOCK_SCOPE_PER_ARTIST,
    NORM_LOCK_TOKEN,
    PRESET_BALANCED,
    PRESET_CHOICES,
    PRESET_DRIFT_AUTO,
    PRESET_RECOMMENDED_CHOICES,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_MODE_OUTPUT,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .options import (
    build_preset_payload,
    format_bool,
    merge_runtime_options,
)
from .parsing import resolve_target_blocks_from_options, split_artist_chain


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
                "Optional layer route, e.g. 0-8, 0,2,4, 0%-33%, "
                "or 0.33-0.67. Used in manual layout; other layouts "
                "auto-fill empty routes."
            ),
        }
        timing_input = {
            "multiline": False,
            "default": "",
            "tooltip": (
                "Optional timing route, e.g. 0.0-0.45 or 0.0-0.45~0.1 "
                "(with fade). Used in manual layout; other layouts auto-fill "
                "empty routes. Use % timing in artist_chain; @0.0-0.5 is a layer range."
            ),
        }
        return {
            "required": {
                "layout": (
                    CHAIN_LAYOUT_CHOICES,
                    {
                        "default": CHAIN_LAYOUT_LAYER_SCHEDULED,
                        "tooltip": (
                            "manual: use the per-row layer/timing values\n"
                            "even_layers: split DiT blocks evenly across artists\n"
                            "layer_scheduled: early/mid/late layers + early/mid/late "
                            "sampling windows in one click"
                        ),
                    },
                ),
                "artist_table": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Multi-artist table. One per line: artist | weight | layers | timing.\n"
                            "Example: @wlop | 1.2 | 0-8 | 0.0-0.45\n"
                            "Empty layers/timing auto-fill per layout. Lines starting "
                            "with # are ignored."
                        ),
                    },
                ),
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
                "num_blocks": (
                    "INT",
                    {
                        "default": DEFAULT_NUM_BLOCKS,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Block count for the preview. Anima default is 28.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("artist_chain", "preview")
    FUNCTION = "build"
    CATEGORY = "Anima/Setup"
    OUTPUT_NODE = True

    def build(
        self,
        layout,
        artist_table,
        artist_1,
        weight_1,
        artist_2,
        weight_2,
        artist_3,
        weight_3,
        layer_route_1="",
        timing_route_1="",
        layer_route_2="",
        timing_route_2="",
        layer_route_3="",
        timing_route_3="",
        num_blocks=DEFAULT_NUM_BLOCKS,
    ):
        table_rows, table_warnings = parse_builder_artist_table(
            artist_table,
            return_warnings=True,
        )
        rows = [
            (artist_1, weight_1, layer_route_1, timing_route_1),
            (artist_2, weight_2, layer_route_2, timing_route_2),
            (artist_3, weight_3, layer_route_3, timing_route_3),
        ]
        rows.extend(table_rows)
        chain, report = build_artist_chain_from_rows(
            layout,
            rows,
            int(num_blocks),
            extra_warnings=table_warnings,
        )
        return {"ui": {"text": [report]}, "result": (chain, report)}


class AnimaArtistChainPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_chain": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "artist_chain to validate. This node needs no CLIP/model; "
                            "use it to check ::weight, @layers, %timing and ~fade "
                            "before paying the CLIP encoding cost."
                        ),
                    },
                ),
            },
            "optional": {
                "num_blocks": (
                    "INT",
                    {
                        "default": DEFAULT_NUM_BLOCKS,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Block count for the preview. Anima default is 28.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cleaned_chain", "report")
    FUNCTION = "preview"
    CATEGORY = "Anima/Diagnostics"
    OUTPUT_NODE = True

    def preview(self, artist_chain, num_blocks=DEFAULT_NUM_BLOCKS):
        cleaned, report = format_artist_chain_preview(artist_chain, int(num_blocks))
        names = [n for n in chain_artist_names(split_artist_chain(artist_chain)) if n]
        if names:
            report = report + "\n\n" + "\n".join(tag_vocab.report_lines(names))
        return {"ui": {"text": [report]}, "result": (cleaned, report)}


class AnimaArtistPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    PRESET_CHOICES,
                    {
                        "default": PRESET_BALANCED,
                        "tooltip": (
                            "Advanced one-knob working modes.\n"
                            "prompt_passthrough: direct prompt/no mixer, preserves positive prompt weights\n"
                            "balanced: original-style output_avg + interpolate\n"
                            "strong_style: stronger style, strength extrapolated to 1.65\n"
                            "stable_seed: delta-capped output_avg, auto layers 9-20, content-safer seed stability\n"
                            "drift_auto: runtime route from base_prompt and artist count; "
                            "4+ broad prompts stay on drift_soft instead of compatibility concat\n"
                            "drift_soft: softer EMA output_avg for portrait / broad-subject prompts\n"
                            "face_lock: base_preserve + token norm lock for close-up faces\n"
                            "scene_lock: base_preserve + light EMA for wide / background-heavy scenes\n"
                            "anchor_lock: single-anchor Q with user blend, strength 0.9, auto layers 9-15\n"
                            "fast_preview: concat path, speed first, good for hunting\n"
                            "identity_guard: base_preserve + norm/delta guard, protects identity/composition\n"
                            "compatibility_safe: concat + concat_with_base, plays nice with "
                            "regional/Forge-style nodes"
                        ),
                    },
                ),
                "intensity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Preset strength multiplier. fast_preview and "
                            "compatibility_safe ignore it; other presets multiply it "
                            "into strength."
                        ),
                    },
                ),
                "normalize_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Default normalize_weights inside the preset. ::weight in "
                            "the chain still bypasses it at runtime."
                        ),
                    },
                ),
                "layer_mode": (
                    LAYER_MODE_CHOICES,
                    {
                        "default": LAYER_MODE_AUTO,
                        "tooltip": (
                            "Layer-range shortcut.\n"
                            "auto: preset-specific default\n"
                            "all_layers: every layer\n"
                            "style_core: 0-18, overall style\n"
                            "detail_layers: 12-63, details and strokes\n"
                            "custom: use custom_layer_filter"
                        ),
                    },
                ),
                "custom_layer_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Active when layer_mode=custom. Example: 0,3,5-10,-1",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("preset", "advanced_options", "summary")
    FUNCTION = "build"
    CATEGORY = "Anima/Setup"

    def build(self, preset, intensity, normalize_weights, layer_mode, custom_layer_filter):
        payload = build_preset_payload(
            preset,
            intensity,
            layer_mode,
            custom_layer_filter,
            normalize_weights,
        )
        adv = payload["advanced_options"]
        summary = "\n".join(
            [
                f"Preset: {payload['preset']}",
                f"combine_mode: {payload['combine_mode']}",
                f"fusion_mode: {payload['fusion_mode']}",
                f"strength: {payload['strength']:.2f}",
                f"normalize_weights: {format_bool(adv.get('normalize_weights', True))}",
                f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
                f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
                f"static_capture: {format_bool(adv.get('artist_static_capture', False))}",
                f"static_capture_k: {int(adv.get('static_capture_k', STATIC_CAPTURE_K_DEFAULT))}",
                f"static_capture_mode: {adv.get('static_capture_mode', STATIC_CAPTURE_MODE_OUTPUT)}",
                f"static_capture_blend_alpha: {float(adv.get('static_capture_blend_alpha', STATIC_CAPTURE_BLEND_ALPHA_DEFAULT)):.2f}",
                f"match_base_norm: {format_bool(adv.get('match_base_norm', False))}",
                f"norm_lock_mode: {adv.get('norm_lock_mode', NORM_LOCK_TOKEN)}",
                f"norm_lock_scope: {adv.get('norm_lock_scope', NORM_LOCK_SCOPE_PER_ARTIST)}",
                f"mixed_delta_cap: {format_bool(adv.get('mixed_delta_cap', False))}",
                f"mixed_delta_cap_ratio: {float(adv.get('mixed_delta_cap_ratio', MIXED_DELTA_CAP_RATIO_DEFAULT)):.2f}",
                f"compatibility_mode: {format_bool(adv.get('compatibility_mode', False))}",
                f"layer_filter: {adv.get('layer_filter') or 'all'}",
            ]
        )
        if payload["preset"] == PRESET_DRIFT_AUTO:
            summary += "\n" + "\n".join(
                [
                    "drift_auto: resolves at runtime from AnimaArtistPack.base_prompt and artist count",
                    "preview ignores base_prompt; use Inspector for the runtime route",
                    f"preview_resolved_preset: {adv.get('drift_auto_resolved_preset')}",
                    f"preview_reason: {adv.get('drift_auto_reason')}",
                ]
            )
        return {"ui": {"text": [summary]}, "result": (payload, adv, summary)}


class AnimaArtistStarter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": (
                    PRESET_RECOMMENDED_CHOICES,
                    {
                        "default": PRESET_BALANCED,
                        "tooltip": (
                            "Recommended modes only. Use Anima Artist Preset for "
                            "advanced / compatibility modes.\n"
                            "balanced: original-compatible default\n"
                            "strong_style: stronger artist style\n"
                            "drift_auto: automatic low-drift route\n"
                            "prompt_passthrough: direct prompt/no mixer"
                        ),
                    },
                ),
                "artist_table": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "One artist per line: artist | weight | layers | timing.\n"
                            "Just the artist name is fine; weight defaults to 1.0; "
                            "empty layers/timing auto-fill per layout."
                        ),
                    },
                ),
                "layout": (
                    CHAIN_LAYOUT_CHOICES,
                    {
                        "default": CHAIN_LAYOUT_LAYER_SCHEDULED,
                        "tooltip": (
                            "manual: use the table's layers/timing as-is\n"
                            "even_layers: split DiT blocks evenly across artists\n"
                            "layer_scheduled: auto layers + sampling windows, "
                            "recommended for new users"
                        ),
                    },
                ),
                "intensity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Recipe strength multiplier for balanced, strong_style, "
                            "and drift_auto. prompt_passthrough ignores it "
                            "(direct prompt, no mixer)."
                        ),
                    },
                ),
            },
            "optional": {
                "normalize_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Recommended on for multi-artist setups. ::weight in the "
                            "table bypasses it at runtime."
                        ),
                    },
                ),
                "layer_mode": (
                    LAYER_MODE_CHOICES,
                    {
                        "default": LAYER_MODE_AUTO,
                        "tooltip": "Global layer-range shortcut; usually keep auto.",
                    },
                ),
                "custom_layer_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Active when layer_mode=custom. Example: 0,3,5-10,-1",
                    },
                ),
                "num_blocks": (
                    "INT",
                    {
                        "default": DEFAULT_NUM_BLOCKS,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Block count for the preview. Anima default is 28.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("artist_chain", "preset", "advanced_options", "guide")
    FUNCTION = "build"
    CATEGORY = "Anima/Basic"
    OUTPUT_NODE = True

    def build(
        self,
        recipe,
        artist_table,
        layout,
        intensity,
        normalize_weights=True,
        layer_mode=LAYER_MODE_AUTO,
        custom_layer_filter="",
        num_blocks=DEFAULT_NUM_BLOCKS,
    ):
        rows, table_warnings = parse_builder_artist_table(
            artist_table,
            return_warnings=True,
        )
        chain, chain_report = build_artist_chain_from_rows(
            layout,
            rows,
            int(num_blocks),
            extra_warnings=table_warnings,
        )
        artist_count = len(split_artist_chain(chain))
        payload = build_preset_payload(
            recipe,
            intensity,
            layer_mode,
            custom_layer_filter,
            normalize_weights,
            artist_count=artist_count,
        )
        adv = payload["advanced_options"]
        status = "CHECK" if "status: CHECK" in chain_report else "OK"
        guide_lines = [
            "Anima Artist Starter",
            "",
            f"status: {status}",
            f"recipe: {payload['preset']}",
            f"layout: {layout}",
            f"artists: {artist_count}",
            "",
            "wire:",
            "  - artist_chain -> AnimaArtistPack.artist_chain",
            "  - model -> AnimaArtistPresetApply.model",
            "  - AnimaArtistPack.artist_pack -> AnimaArtistPresetApply.artist_pack",
            "  - preset -> AnimaArtistPresetApply.preset",
            "  - advanced_options -> AnimaArtistPresetApply.advanced_options only if you want the explicit option payload",
            "",
            "preset summary:",
            f"  combine_mode: {payload['combine_mode']}",
            f"  fusion_mode: {payload['fusion_mode']}",
            f"  strength: {float(payload['strength']):.2f}",
            f"  match_base_norm: {format_bool(adv.get('match_base_norm', False))}",
            f"  norm_lock_mode: {adv.get('norm_lock_mode', NORM_LOCK_TOKEN)}",
            f"  norm_lock_scope: {adv.get('norm_lock_scope', NORM_LOCK_SCOPE_PER_ARTIST)}",
            f"  mixed_delta_cap: {format_bool(adv.get('mixed_delta_cap', False))}",
            f"  mixed_delta_cap_ratio: {float(adv.get('mixed_delta_cap_ratio', MIXED_DELTA_CAP_RATIO_DEFAULT)):.2f}",
            f"  compatibility_mode: {format_bool(adv.get('compatibility_mode', False))}",
            f"  layer_filter: {adv.get('layer_filter') or 'all'}",
        ]
        if payload["preset"] == PRESET_DRIFT_AUTO:
            guide_lines.extend(
                [
                    "  drift_auto: resolves at runtime from AnimaArtistPack.base_prompt and artist count",
                    "  preview ignores base_prompt; use Inspector for the runtime route",
                    f"  preview_resolved_preset: {adv.get('drift_auto_resolved_preset')}",
                    f"  preview_reason: {adv.get('drift_auto_reason')}",
                ]
            )
        guide_lines.extend(
            [
                "",
                "chain report:",
                chain_report,
            ]
        )
        guide = "\n".join(guide_lines)
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
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                    },
                ),
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET",),
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Optional. Connect the Anima model to read the real block "
                            "count instead of assuming 28."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "Anima/Diagnostics"
    OUTPUT_NODE = True

    def inspect(
        self,
        artist_pack,
        combine_mode=COMBINE_OUTPUT_AVG,
        fusion_mode=FUSION_INTERPOLATE,
        strength=1.0,
        advanced_options=None,
        preset=None,
        model=None,
    ):
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
            combine_mode,
            fusion_mode,
            strength,
            advanced_options,
            preset,
            base_prompt=base_prompt,
            artist_count=len(labels),
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
                blocks_source = f"failed to read the model; assumes {DEFAULT_NUM_BLOCKS} blocks"

        lines = [
            "Anima Artist Mixer Inspector",
            "",
            f"preset: {preset_name or '(none)'}",
            f"resolved_preset: {adv.get('drift_auto_resolved_preset', preset_name or '(none)')}",
            f"drift_auto_reason: {adv.get('drift_auto_reason', '(not active)')}",
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
            f"stabilizer_end_percent: {float(adv.get('stabilizer_end_percent', 1.0)):.2f}",
            f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
            f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
            f"static_capture: {format_bool(adv.get('artist_static_capture', False))} "
            f"(K={int(adv.get('static_capture_k', STATIC_CAPTURE_K_DEFAULT))}, "
            f"mode={adv.get('static_capture_mode', STATIC_CAPTURE_MODE_OUTPUT)}, "
            f"blend_alpha={float(adv.get('static_capture_blend_alpha', STATIC_CAPTURE_BLEND_ALPHA_DEFAULT)):.2f})",
            f"anchor_q: {format_bool(adv.get('artist_anchor_q', False))}",
            f"max_batch_artists: {int(adv.get('max_batch_artists', 0) or 0) or 'unlimited'}",
            f"low_vram_cache: {format_bool(adv.get('low_vram_cache', False))}",
            f"match_base_norm: {format_bool(adv.get('match_base_norm', False))}",
            f"norm_lock_mode: {adv.get('norm_lock_mode', NORM_LOCK_TOKEN)}",
            f"norm_lock_scope: {adv.get('norm_lock_scope', NORM_LOCK_SCOPE_PER_ARTIST)}",
            f"contribution_balance: {format_bool(adv.get('contribution_balance', False))}",
            f"contribution_balance_alpha: {float(adv.get('contribution_balance_alpha', CONTRIB_BALANCE_ALPHA_DEFAULT)):.2f}",
            f"mixed_delta_cap: {format_bool(adv.get('mixed_delta_cap', False))}",
            f"mixed_delta_cap_ratio: {float(adv.get('mixed_delta_cap_ratio', MIXED_DELTA_CAP_RATIO_DEFAULT)):.2f}",
            "",
            "artists:",
        ]

        if labels:
            for idx, (label, weight, route, timing) in enumerate(
                zip(labels, weights, layer_routes, timing_routes),
                start=1,
            ):
                route_text = f" @ {route}" if route else ""
                timing_text = f" % {timing}" if timing else ""
                lines.append(f"  {idx}. {label} :: {weight:.3g}{route_text}{timing_text}")
        else:
            lines.append("  (none)")

        target_blocks = resolve_target_blocks_from_options(adv, inspector_blocks)
        lines.append("")
        lines.append(f"block map ({blocks_source}):")
        lines.append(
            format_artist_block_map(
                labels,
                layer_routes,
                timing_routes,
                inspector_blocks,
                target_blocks,
            )
        )

        warnings = []
        notes = []
        if not labels:
            warnings.append("no artists; CrossAttn will return the base prompt untouched.")
        if has_explicit and requested_normalize:
            warnings.append(
                "::weight detected; normalize_weights is bypassed at runtime (this is the correct behavior)."
            )
        if not effective_normalize and weight_sum > 1.5:
            warnings.append("linear weight sum > 1.5; the style may oversaturate or blow out.")
        if any(w < 0.0 for w in weights):
            notes.append(
                "negative weights present: those artists subtract their style "
                "direction (style subtraction); best combined with positive artists."
            )
        if adv.get("artist_static_capture", False) and adv.get("artist_anchor_q", False):
            warnings.append(
                "static_capture and anchor_q are mutually exclusive; CrossAttn disables static_capture."
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_anchor_q", False):
            warnings.append("concat_with_base does not support anchor_q; CrossAttn disables anchor_q.")
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_static_capture", False):
            warnings.append("concat_with_base does not support static_capture; the normal path is used.")
        if combine_mode == COMBINE_LOWRANK_AVG and len(labels) <= 1:
            warnings.append("lowrank_avg is meaningless with one artist; output_avg is used instead.")
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


# Backward-compatible re-exports (v27.4 module split).
from .nodes_options import AnimaArtistOptions  # noqa: E402,F401
from .nodes_options import AnimaArtistSimpleOptions  # noqa: E402,F401
from .nodes_options import AnimaArtistStyleBalance  # noqa: E402,F401
from .nodes_recipes import AnimaArtistRecipeLoad  # noqa: E402,F401
from .nodes_recipes import AnimaArtistRecipeSave  # noqa: E402,F401
