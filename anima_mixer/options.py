"""Preset payloads and runtime option merging."""

from .constants import (
    COMBINE_CONCAT,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    LAYER_MODE_ALL,
    LAYER_MODE_AUTO,
    LAYER_MODE_CUSTOM,
    LAYER_MODE_DETAIL,
    LAYER_MODE_STYLE_CORE,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    PRESET_BALANCED,
    PRESET_CHOICES,
    PRESET_COMPATIBILITY_SAFE,
    PRESET_FAST_PREVIEW,
    PRESET_IDENTITY_GUARD,
    PRESET_STABLE_SEED,
    PRESET_STRONG_STYLE,
    STATIC_CAPTURE_K_DEFAULT,
)
from .parsing import clamp_float


def base_advanced_options():
    return {
        "start_block": 0,
        "end_block": -1,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "normalize_weights": True,
        "artist_ema_alpha": 0.0,
        "lowrank_k": 1,
        "artist_static_capture": False,
        "static_capture_k": STATIC_CAPTURE_K_DEFAULT,
        "artist_anchor_q": False,
        "anchor_seeds_count": 1,
        "anchor_user_blend": 0.0,
        "anchor_deep_layer_threshold": ANCHOR_LAYER_THRESHOLD_DISABLED,
        "layer_filter": "",
        "compatibility_mode": False,
        "max_batch_artists": 0,
        "low_vram_cache": False,
        "match_base_norm": True,
    }


def layer_filter_for_mode(layer_mode, custom_layer_filter):
    if layer_mode == LAYER_MODE_ALL:
        return ""
    if layer_mode == LAYER_MODE_STYLE_CORE:
        return "0-18"
    if layer_mode == LAYER_MODE_DETAIL:
        return "12-63"
    if layer_mode == LAYER_MODE_CUSTOM:
        return str(custom_layer_filter or "").strip()
    return ""


def build_preset_payload(preset_name, intensity=1.0, layer_mode=LAYER_MODE_AUTO,
                         custom_layer_filter="", normalize_weights=True):
    preset_name = preset_name if preset_name in PRESET_CHOICES else PRESET_BALANCED
    intensity = clamp_float(intensity, 0.0, 2.0)
    adv = base_advanced_options()
    adv["normalize_weights"] = bool(normalize_weights)
    adv["layer_filter"] = layer_filter_for_mode(layer_mode, custom_layer_filter)

    payload = {
        "preset": preset_name,
        "combine_mode": COMBINE_OUTPUT_AVG,
        "fusion_mode": FUSION_INTERPOLATE,
        "strength": 1.0,
        "advanced_options": adv,
    }

    if preset_name == PRESET_BALANCED:
        adv["artist_ema_alpha"] = 0.25
        payload["strength"] = 1.0
    elif preset_name == PRESET_STRONG_STYLE:
        adv["artist_ema_alpha"] = 0.20
        adv["end_percent"] = 0.92
        payload["strength"] = 1.65
    elif preset_name == PRESET_STABLE_SEED:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = True
        adv["static_capture_k"] = 6
        payload["combine_mode"] = COMBINE_LOWRANK_AVG
        payload["strength"] = 1.15
    elif preset_name == PRESET_FAST_PREVIEW:
        adv["end_percent"] = 0.82
        payload["combine_mode"] = COMBINE_CONCAT
        payload["fusion_mode"] = FUSION_CONCAT_WITH_BASE
        payload["strength"] = 1.0
    elif preset_name == PRESET_IDENTITY_GUARD:
        adv["artist_ema_alpha"] = 0.35
        adv["lowrank_k"] = 1
        payload["combine_mode"] = COMBINE_LOWRANK_AVG
        payload["fusion_mode"] = FUSION_BASE_PRESERVE
        payload["strength"] = 1.25
    elif preset_name == PRESET_COMPATIBILITY_SAFE:
        adv["compatibility_mode"] = True
        payload["combine_mode"] = COMBINE_CONCAT
        payload["fusion_mode"] = FUSION_CONCAT_WITH_BASE
        payload["strength"] = 1.0

    if preset_name not in (PRESET_FAST_PREVIEW, PRESET_COMPATIBILITY_SAFE):
        payload["strength"] = clamp_float(payload["strength"] * intensity, 0.0, 4.0)
    payload["intensity"] = intensity
    payload["layer_mode"] = layer_mode
    return payload


def apply_compatibility_mode(combine_mode, fusion_mode, strength, adv):
    if not bool(adv.get("compatibility_mode", False)):
        return combine_mode, fusion_mode, float(strength), adv
    adv = dict(adv)
    adv["compatibility_mode"] = True
    adv["artist_ema_alpha"] = 0.0
    adv["artist_static_capture"] = False
    adv["artist_anchor_q"] = False
    return COMBINE_CONCAT, FUSION_CONCAT_WITH_BASE, 1.0, adv


def merge_runtime_options(combine_mode, fusion_mode, strength,
                          advanced_options=None, preset=None):
    adv = {}
    preset_name = None
    if isinstance(preset, dict):
        preset_name = preset.get("preset")
        adv.update(preset.get("advanced_options") or {})
        combine_mode = preset.get("combine_mode", combine_mode)
        fusion_mode = preset.get("fusion_mode", fusion_mode)
        strength = preset.get("strength", strength)
    if isinstance(advanced_options, dict):
        adv.update(advanced_options)
    if preset_name == PRESET_COMPATIBILITY_SAFE:
        adv["compatibility_mode"] = True
    combine_mode, fusion_mode, strength, adv = apply_compatibility_mode(
        combine_mode, fusion_mode, strength, adv,
    )
    return combine_mode, fusion_mode, float(strength), adv, preset_name


def format_bool(value):
    return "on" if bool(value) else "off"
