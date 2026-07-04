"""Recipe nodes: save/load shareable JSON mixing recipes."""


from .constants import (
    COMBINE_CHOICES,
    COMBINE_OUTPUT_AVG,
    FUSION_CHOICES,
    FUSION_INTERPOLATE,
    LAYER_MODE_AUTO,
    PRESET_DRIFT_AUTO,
)
from .options import (
    build_preset_payload,
    merge_runtime_options,
)
from .parsing import split_artist_chain
from .recipe import deserialize_recipe, serialize_recipe


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
    CATEGORY = "Anima/Recipes"
    OUTPUT_NODE = True

    def save(self, artist_chain, combine_mode, fusion_mode, strength,
             advanced_options=None, preset=None, notes=""):
        combine_mode, fusion_mode, strength, adv, _ = merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
        )
        recipe_json = serialize_recipe(
            artist_chain, combine_mode, fusion_mode, strength, adv, notes,
            source_preset=preset,
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
                        "wire it to AnimaArtistPresetApply.preset."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("artist_chain", "preset", "advanced_options", "summary")
    FUNCTION = "load"
    CATEGORY = "Anima/Recipes"
    OUTPUT_NODE = True

    def load(self, recipe_json):
        payload, warnings = deserialize_recipe(recipe_json)
        source_preset = payload.get("preset") or ""
        if source_preset == PRESET_DRIFT_AUTO:
            # Rebuild the deferred preset so AnimaArtistPresetApply re-resolves
            # drift_auto against the real base_prompt at patch time, instead of
            # replaying the empty-prompt route baked in at save time.
            preset_payload = build_preset_payload(
                PRESET_DRIFT_AUTO,
                payload.get("preset_intensity", 1.0),
                payload.get("preset_layer_mode") or LAYER_MODE_AUTO,
                payload.get("preset_custom_layer_filter", ""),
                payload["advanced_options"].get("normalize_weights", True),
            )
            advanced_options_out = preset_payload["advanced_options"]
        else:
            preset_payload = {
                "preset": "recipe",
                "combine_mode": payload["combine_mode"],
                "fusion_mode": payload["fusion_mode"],
                "strength": payload["strength"],
                "advanced_options": payload["advanced_options"],
            }
            advanced_options_out = payload["advanced_options"]
        lines = [
            "Anima Artist Recipe",
            "",
            f"status: {'CHECK' if warnings else 'OK'}",
            f"preset: {source_preset or 'recipe (baked options)'}",
            f"combine_mode: {payload['combine_mode']}",
            f"fusion_mode: {payload['fusion_mode']}",
            f"strength: {payload['strength']:.2f}",
            f"artists: {len(split_artist_chain(payload['artist_chain']))}",
        ]
        if source_preset == PRESET_DRIFT_AUTO:
            lines.append(
                "drift_auto: re-resolves at AnimaArtistPresetApply time from the "
                "real base_prompt (values above are the empty-prompt preview)"
            )
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
            "  - preset -> AnimaArtistPresetApply.preset",
            "  - advanced_options -> AnimaArtistPresetApply.advanced_options",
        ])
        summary = "\n".join(lines)
        return {
            "ui": {"text": [summary]},
            "result": (payload["artist_chain"], preset_payload,
                       advanced_options_out, summary),
        }
