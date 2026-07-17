"""Preset payloads and runtime option merging."""

from .constants import (
    COMBINE_CONCAT,
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
    CONTRIB_BALANCE_ALPHA_DEFAULT,
    DRIFT_AUTO_CLOSEUP_DELTA_CAP_RATIO,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    PRESET_ANCHOR_LOCK,
    PRESET_BALANCED,
    PRESET_CHOICES,
    PRESET_COMPATIBILITY_SAFE,
    PRESET_COMPATIBILITY_SAFE_9_15,
    PRESET_DRIFT_AUTO,
    PRESET_FAST_PREVIEW,
    PRESET_FACE_LOCK,
    PRESET_DRIFT_SOFT,
    PRESET_IDENTITY_GUARD,
    PRESET_PROMPT_PASSTHROUGH,
    PRESET_SCENE_LOCK,
    PRESET_STABLE_SEED,
    PRESET_STRONG_STYLE,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_MODE_OUTPUT,
    NORM_LOCK_TOKEN,
    NORM_LOCK_SCOPE_PER_ARTIST,
)
from .parsing import clamp_float


def _prompt_has_any(prompt, phrases):
    padded = f" {prompt} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _normalize_prompt_for_routing(base_prompt):
    text = str(base_prompt or "").lower()
    for ch in "\n\r\t,.;:()[]{}_/\\|+*=\"'":
        text = text.replace(ch, " ")
    text = text.replace("-", " ")
    return " ".join(text.split())


def resolve_drift_auto_preset(base_prompt, artist_count=0):
    """Choose the least-drifty preset family for a prompt, without training."""
    prompt = _normalize_prompt_for_routing(base_prompt)
    artist_count = max(0, int(artist_count or 0))

    face_score = 0
    if _prompt_has_any(prompt, (
        "close up", "closeup", "extreme close", "headshot", "face close",
        "portrait close", "close portrait", "bust shot",
    )):
        face_score += 3
    if _prompt_has_any(prompt, (
        "detailed eyes", "detailed eye", "face focus", "facial expression",
        "looking at viewer",
    )):
        face_score += 1

    wide_scene = _prompt_has_any(prompt, (
        "wide shot", "wide angle", "long shot", "establishing shot",
        "panoramic", "small figure", "distant view", "wide composition",
    ))
    background_heavy_scene = _prompt_has_any(prompt, (
        "cityscape", "landscape", "scenery", "environment",
        "background crowd", "crowded background", "background heavy",
        "background-heavy",
    ))
    simple_fullbody = _prompt_has_any(prompt, (
        "fullbody", "full body", "standing pose", "standing",
    )) and _prompt_has_any(prompt, (
        "simple background", "plain background", "studio background",
        "white background", "solid background",
    ))

    if artist_count >= 4 and (wide_scene or background_heavy_scene):
        return PRESET_SCENE_LOCK, "4+ artists wide / background-heavy drift guard"
    if wide_scene or background_heavy_scene:
        return PRESET_SCENE_LOCK, "wide or background-heavy scene prompt"
    if artist_count >= 4 and simple_fullbody:
        return PRESET_DRIFT_SOFT, "4+ artists simple fullbody drift guard"
    if artist_count >= 4 and face_score >= 3:
        return PRESET_STABLE_SEED, "4+ artists close-up delta-cap drift guard"
    if artist_count >= 4:
        return PRESET_DRIFT_SOFT, "4+ artists default portrait / broad-subject drift guard"
    if face_score >= 3:
        return PRESET_FACE_LOCK, "close-up or face-focused prompt"
    return PRESET_DRIFT_SOFT, "default portrait / broad-subject drift guard"


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
        "static_capture_mode": STATIC_CAPTURE_MODE_OUTPUT,
        "static_capture_blend_alpha": STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
        "artist_anchor_q": False,
        "anchor_seeds_count": 1,
        "anchor_user_blend": 0.0,
        "anchor_deep_layer_threshold": ANCHOR_LAYER_THRESHOLD_DISABLED,
        "anchor_refresh_each_step": False,
        "stabilizer_end_percent": 1.0,
        "layer_filter": "",
        "compatibility_mode": False,
        "max_batch_artists": 0,
        "artist_q_reuse": False,
        "low_vram_cache": False,
        "match_base_norm": False,
        "anchor_base_norm_ref": False,
        "norm_lock_mode": NORM_LOCK_TOKEN,
        "norm_lock_scope": NORM_LOCK_SCOPE_PER_ARTIST,
        "contribution_balance": False,
        "contribution_balance_alpha": CONTRIB_BALANCE_ALPHA_DEFAULT,
        "mixed_delta_cap": False,
        "mixed_delta_cap_ratio": MIXED_DELTA_CAP_RATIO_DEFAULT,
        "prompt_passthrough": False,
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
                         custom_layer_filter="", normalize_weights=True,
                         artist_count=0):
    internal_preset = preset_name == PRESET_COMPATIBILITY_SAFE_9_15
    preset_name = preset_name if preset_name in PRESET_CHOICES or internal_preset else PRESET_BALANCED
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
        adv["artist_ema_alpha"] = 0.0
        adv["match_base_norm"] = False
        payload["strength"] = 1.0
    elif preset_name == PRESET_PROMPT_PASSTHROUGH:
        adv["prompt_passthrough"] = True
        adv["normalize_weights"] = False
        payload["strength"] = 0.0
    elif preset_name == PRESET_STRONG_STYLE:
        adv["artist_ema_alpha"] = 0.20
        adv["end_percent"] = 0.92
        payload["strength"] = 1.65
    elif preset_name == PRESET_STABLE_SEED:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = False
        adv["static_capture_mode"] = STATIC_CAPTURE_MODE_OUTPUT
        adv["static_capture_blend_alpha"] = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
        adv["artist_anchor_q"] = False
        adv["anchor_seeds_count"] = 1
        adv["anchor_user_blend"] = 0.0
        adv["anchor_deep_layer_threshold"] = ANCHOR_LAYER_THRESHOLD_DISABLED
        adv["match_base_norm"] = False
        adv["anchor_base_norm_ref"] = False
        adv["contribution_balance"] = False
        adv["mixed_delta_cap"] = True
        adv["mixed_delta_cap_ratio"] = DRIFT_AUTO_CLOSEUP_DELTA_CAP_RATIO
        if layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-20"
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["strength"] = 1.0
    elif preset_name == PRESET_DRIFT_AUTO:
        resolved_name, reason = resolve_drift_auto_preset("", artist_count)
        resolved = build_preset_payload(
            resolved_name, 1.0, layer_mode, custom_layer_filter, normalize_weights,
            artist_count=artist_count,
        )
        adv = resolved["advanced_options"]
        adv["drift_auto_pending"] = True
        adv["drift_auto_resolved_preset"] = resolved_name
        adv["drift_auto_reason"] = reason
        payload["advanced_options"] = adv
        payload["combine_mode"] = resolved["combine_mode"]
        payload["fusion_mode"] = resolved["fusion_mode"]
        payload["strength"] = resolved["strength"]
    elif preset_name == PRESET_DRIFT_SOFT:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = False
        adv["artist_ema_alpha"] = 0.12
        adv["static_capture_mode"] = STATIC_CAPTURE_MODE_OUTPUT
        adv["static_capture_blend_alpha"] = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
        adv["artist_anchor_q"] = False
        adv["anchor_seeds_count"] = 1
        adv["anchor_user_blend"] = 0.0
        adv["anchor_deep_layer_threshold"] = ANCHOR_LAYER_THRESHOLD_DISABLED
        adv["match_base_norm"] = False
        adv["anchor_base_norm_ref"] = False
        adv["contribution_balance"] = False
        if layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-20"
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["strength"] = 0.85
    elif preset_name == PRESET_FACE_LOCK:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = False
        adv["static_capture_mode"] = STATIC_CAPTURE_MODE_OUTPUT
        adv["static_capture_blend_alpha"] = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
        adv["artist_anchor_q"] = False
        adv["anchor_seeds_count"] = 1
        adv["anchor_user_blend"] = 0.0
        adv["anchor_deep_layer_threshold"] = ANCHOR_LAYER_THRESHOLD_DISABLED
        adv["match_base_norm"] = True
        adv["anchor_base_norm_ref"] = False
        adv["norm_lock_mode"] = NORM_LOCK_TOKEN
        adv["norm_lock_scope"] = NORM_LOCK_SCOPE_PER_ARTIST
        adv["contribution_balance"] = False
        adv["mixed_delta_cap"] = True
        adv["mixed_delta_cap_ratio"] = 1.0
        if layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-20"
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["fusion_mode"] = FUSION_BASE_PRESERVE
        payload["strength"] = 0.9
    elif preset_name == PRESET_SCENE_LOCK:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = False
        adv["artist_ema_alpha"] = 0.10
        adv["static_capture_mode"] = STATIC_CAPTURE_MODE_OUTPUT
        adv["static_capture_blend_alpha"] = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
        adv["artist_anchor_q"] = False
        adv["anchor_seeds_count"] = 1
        adv["anchor_user_blend"] = 0.0
        adv["anchor_deep_layer_threshold"] = ANCHOR_LAYER_THRESHOLD_DISABLED
        adv["match_base_norm"] = False
        adv["anchor_base_norm_ref"] = False
        adv["contribution_balance"] = False
        if layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-15"
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["fusion_mode"] = FUSION_BASE_PRESERVE
        payload["strength"] = 0.85
    elif preset_name == PRESET_ANCHOR_LOCK:
        adv["lowrank_k"] = 1
        adv["artist_static_capture"] = False
        adv["static_capture_mode"] = STATIC_CAPTURE_MODE_OUTPUT
        adv["static_capture_blend_alpha"] = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
        adv["artist_anchor_q"] = True
        adv["anchor_seeds_count"] = 1
        adv["anchor_user_blend"] = 0.35
        adv["anchor_deep_layer_threshold"] = 12
        adv["match_base_norm"] = False
        adv["anchor_base_norm_ref"] = False
        adv["contribution_balance"] = False
        if layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-15"
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["strength"] = 0.9
    elif preset_name == PRESET_FAST_PREVIEW:
        adv["end_percent"] = 0.82
        payload["combine_mode"] = COMBINE_CONCAT
        payload["fusion_mode"] = FUSION_CONCAT_WITH_BASE
        payload["strength"] = 1.0
    elif preset_name == PRESET_IDENTITY_GUARD:
        adv["artist_ema_alpha"] = 0.12
        adv["lowrank_k"] = 1
        adv["match_base_norm"] = True
        adv["mixed_delta_cap"] = True
        adv["mixed_delta_cap_ratio"] = 0.9
        payload["combine_mode"] = COMBINE_OUTPUT_AVG
        payload["fusion_mode"] = FUSION_BASE_PRESERVE
        payload["strength"] = 0.85
    elif preset_name in (PRESET_COMPATIBILITY_SAFE, PRESET_COMPATIBILITY_SAFE_9_15):
        adv["compatibility_mode"] = True
        if preset_name == PRESET_COMPATIBILITY_SAFE_9_15 and layer_mode == LAYER_MODE_AUTO:
            adv["layer_filter"] = "9-15"
        payload["combine_mode"] = COMBINE_CONCAT
        payload["fusion_mode"] = FUSION_CONCAT_WITH_BASE
        payload["strength"] = 1.0

    if preset_name not in (
        PRESET_FAST_PREVIEW,
        PRESET_COMPATIBILITY_SAFE,
        PRESET_COMPATIBILITY_SAFE_9_15,
    ):
        payload["strength"] = clamp_float(payload["strength"] * intensity, 0.0, 4.0)
    payload["intensity"] = intensity
    payload["layer_mode"] = layer_mode
    payload["custom_layer_filter"] = str(custom_layer_filter or "")
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
                          advanced_options=None, preset=None, base_prompt="",
                          artist_count=0):
    adv = {}
    preset_name = None
    resolved_preset_name = None
    if isinstance(preset, dict):
        preset_name = preset.get("preset")
        if preset_name == PRESET_DRIFT_AUTO:
            resolved_preset_name, reason = resolve_drift_auto_preset(
                base_prompt, artist_count,
            )
            resolved = build_preset_payload(
                resolved_preset_name,
                preset.get("intensity", 1.0),
                preset.get("layer_mode", LAYER_MODE_AUTO),
                preset.get("custom_layer_filter", ""),
                (preset.get("advanced_options") or {}).get("normalize_weights", True),
                artist_count=artist_count,
            )
            adv.update(resolved.get("advanced_options") or {})
            if resolved_preset_name == PRESET_STABLE_SEED and "delta-cap" in reason:
                adv["mixed_delta_cap"] = True
                adv["mixed_delta_cap_ratio"] = DRIFT_AUTO_CLOSEUP_DELTA_CAP_RATIO
            adv["drift_auto_resolved_preset"] = resolved_preset_name
            adv["drift_auto_reason"] = reason
            combine_mode = resolved.get("combine_mode", combine_mode)
            fusion_mode = resolved.get("fusion_mode", fusion_mode)
            strength = resolved.get("strength", strength)
        else:
            adv.update(preset.get("advanced_options") or {})
            combine_mode = preset.get("combine_mode", combine_mode)
            fusion_mode = preset.get("fusion_mode", fusion_mode)
            strength = preset.get("strength", strength)
    if isinstance(advanced_options, dict):
        if not (preset_name == PRESET_DRIFT_AUTO and advanced_options.get("drift_auto_pending")):
            adv.update(advanced_options)
    if (
        preset_name in (PRESET_COMPATIBILITY_SAFE, PRESET_COMPATIBILITY_SAFE_9_15)
        or resolved_preset_name in (PRESET_COMPATIBILITY_SAFE, PRESET_COMPATIBILITY_SAFE_9_15)
    ):
        adv["compatibility_mode"] = True
    combine_mode, fusion_mode, strength, adv = apply_compatibility_mode(
        combine_mode, fusion_mode, strength, adv,
    )
    return combine_mode, fusion_mode, float(strength), adv, preset_name


def format_bool(value):
    return "on" if bool(value) else "off"
